"""
analytics/percentile_cache.py — Disk-persisted percentile pool cache for complete seasons.

Percentile pools for complete seasons (< CURRENT_SEASON) are expensive to build
(~5-15 s per season, full pitches table scan) but never change once a season ends.
This module persists them to parquet files so the server doesn't rebuild on every
restart or TTL expiry.

Storage layout:
    scores/percentile_pools/<season>/<player_type>_<split>_<pool_version>.parquet

where pool_version is an 8-character hex digest of:
    - analytics/percentiles.py content (code version)
    - qualification thresholds (SAVANT_PA_PER_TEAM_GAME_PITCHER/BATTER)
    - the season's pitch row count in the DB (data fingerprint)

Single-flight lock: concurrent first requests build the pool once and share the result.
Atomic writes: write to a temp file then rename so readers never see a partial file.
Never cache errors: if the build fails, nothing is written.
"""
import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# Lazy import CURRENT_SEASON to avoid import-time side effects.
_POOLS_DIR: Optional[Path] = None
_locks: dict = {}  # (season, player_type, split) -> threading.Lock
_locks_mu = threading.Lock()


def _pools_dir() -> Path:
    global _POOLS_DIR
    if _POOLS_DIR is None:
        # scores/ is next to this file's package root (project root / scores)
        here = Path(__file__).resolve().parent.parent
        _POOLS_DIR = here / "scores" / "percentile_pools"
        _POOLS_DIR.mkdir(parents=True, exist_ok=True)
    return _POOLS_DIR


def _pool_version(conn: sqlite3.Connection, season: int) -> str:
    """8-char hex: hash of code + thresholds + season row count."""
    from analytics.percentiles import SAVANT_PA_PER_TEAM_GAME_PITCHER, SAVANT_PA_PER_TEAM_GAME_BATTER
    code_path = Path(__file__).resolve().parent / "percentiles.py"
    code_bytes = code_path.read_bytes()
    row_count = conn.execute(
        "SELECT COUNT(*) FROM pitches WHERE season = ?", (season,)
    ).fetchone()[0]
    payload = (
        code_bytes
        + str(SAVANT_PA_PER_TEAM_GAME_PITCHER).encode()
        + str(SAVANT_PA_PER_TEAM_GAME_BATTER).encode()
        + str(row_count).encode()
    )
    return hashlib.sha256(payload).hexdigest()[:8]


def _pool_path(season: int, player_type: str, split: Optional[str], version: str) -> Path:
    split_str = split if split else "ALL"
    season_dir = _pools_dir() / str(season)
    season_dir.mkdir(exist_ok=True)
    return season_dir / f"{player_type}_{split_str}_{version}.parquet"


def _get_lock(season: int, player_type: str, split: Optional[str]) -> threading.Lock:
    key = (season, player_type, split)
    with _locks_mu:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def load_pool(conn: sqlite3.Connection, season: int, player_type: str,
              split: Optional[str]) -> Optional[pd.DataFrame]:
    """Load a persisted pool from disk if it exists and is current.

    Returns the DataFrame on cache hit, or None on miss/stale.
    player_type: 'pitcher' or 'batter'
    split: None/'ALL' for all batters, 'L' or 'R' for hand filter
    """
    from config.model_policy import CURRENT_SEASON
    if season >= CURRENT_SEASON:
        return None  # current season is always computed live

    version = _pool_version(conn, season)
    path = _pool_path(season, player_type, split, version)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"[pct_cache] load failed {path.name}: {e}")
            return None

    # Stale files: delete any old-version files for this (season, type, split)
    split_str = split if split else "ALL"
    season_dir = _pools_dir() / str(season)
    if season_dir.exists():
        for old in season_dir.glob(f"{player_type}_{split_str}_*.parquet"):
            if old != path:
                try:
                    old.unlink()
                    print(f"[pct_cache] removed stale {old.name}")
                except OSError:
                    pass
    return None


def save_pool(conn: sqlite3.Connection, season: int, player_type: str,
              split: Optional[str], df: pd.DataFrame) -> None:
    """Atomically persist a pool DataFrame to disk.

    Uses write-to-temp-then-rename so readers never see a partial file.
    Never called if df is empty or if the build raised an exception.
    """
    from config.model_policy import CURRENT_SEASON
    if season >= CURRENT_SEASON:
        return  # never persist the current season

    if df is None or df.empty:
        return

    version = _pool_version(conn, season)
    path = _pool_path(season, player_type, split, version)
    tmp = path.with_suffix(".tmp")
    try:
        df.to_parquet(tmp, index=True)
        tmp.replace(path)
        print(f"[pct_cache] saved {path.parent.name}/{path.name} ({len(df):,} rows)")
    except Exception as e:
        print(f"[pct_cache] save failed {path.name}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def get_or_build_pool(conn: sqlite3.Connection, season: int, player_type: str,
                      split: Optional[str], min_override: Optional[float] = None) -> pd.DataFrame:
    """Load from disk if available, otherwise build once (single-flight) and persist.

    This is the entry point called by the modified _pitcher_pool / _batter_pool wrappers
    in api/routers/players.py for complete (past) seasons.

    min_override: passed through to the percentile_metrics function if set
    (used for the threshold=0 unqualified pools which are NOT persisted).
    """
    from config.model_policy import CURRENT_SEASON
    from analytics.percentiles import pitcher_percentile_metrics, batter_percentile_metrics

    # Only persist qualified (default-threshold) pools for complete seasons.
    # Unqualified (min=0) pools are small and built on-demand; no disk cache.
    should_persist = (season < CURRENT_SEASON) and (min_override is None)

    if should_persist:
        cached = load_pool(conn, season, player_type, split)
        if cached is not None:
            return cached

    lock = _get_lock(season, player_type, split) if should_persist else None
    if lock is not None:
        lock.acquire()
        try:
            # Re-check after acquiring lock (another thread may have built it)
            cached = load_pool(conn, season, player_type, split)
            if cached is not None:
                return cached

            df = _build_pool(conn, season, player_type, split, min_override)
            if should_persist and df is not None and not df.empty:
                save_pool(conn, season, player_type, split, df)
            return df if df is not None else pd.DataFrame()
        finally:
            lock.release()
    else:
        return _build_pool(conn, season, player_type, split, min_override) or pd.DataFrame()


def _build_pool(conn, season, player_type, split, min_override):
    from analytics.percentiles import pitcher_percentile_metrics, batter_percentile_metrics
    t0 = time.monotonic()
    try:
        if player_type == "pitcher":
            df = pitcher_percentile_metrics(conn, season, min_bf=min_override, stand_filter=split)
        else:
            df = batter_percentile_metrics(conn, season, min_pa=min_override, throws_filter=split)
        elapsed = time.monotonic() - t0
        print(f"[pct_cache] built {player_type} season={season} split={split} "
              f"in {elapsed:.1f}s ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"[pct_cache] build failed {player_type} season={season} split={split}: {e}")
        return None


def preload_complete_seasons(conn: sqlite3.Connection) -> None:
    """Called at startup to warm complete-season pools from disk into the TTL cache.

    For each (season, player_type, split) where season < CURRENT_SEASON:
    - If a persisted pool exists: load it and inject into the players.py TTL cache
    - If missing: the first request will build it lazily and persist it
    """
    from config.model_policy import CURRENT_SEASON, EARLIEST_SEASON

    # We can't import _pitcher_pool/_batter_pool directly from players.py because
    # warmup.py already imports them — and warmup.py is called before routers are
    # loaded in some startup paths.  Instead, just ensure the pools are on disk;
    # the first real request will hit load_pool() and return the cached DataFrame
    # without rebuilding.  For seasons already on disk this is instant; for missing
    # seasons it triggers a lazy build on first access.
    print("[pct_cache] preload_complete_seasons: scanning disk for persisted pools...")
    from analytics.percentiles import SAVANT_PA_PER_TEAM_GAME_PITCHER, SAVANT_PA_PER_TEAM_GAME_BATTER
    splits_pitcher = [None, "L", "R"]
    splits_batter = [None, "L", "R"]

    built_count = 0
    for season in range(EARLIEST_SEASON, CURRENT_SEASON):
        for split in splits_pitcher:
            existing = load_pool(conn, season, "pitcher", split)
            if existing is None:
                print(f"[pct_cache] missing: pitcher season={season} split={split} — will build lazily")
            else:
                built_count += 1
        for split in splits_batter:
            existing = load_pool(conn, season, "batter", split)
            if existing is None:
                print(f"[pct_cache] missing: batter season={season} split={split} — will build lazily")
            else:
                built_count += 1

    total = (CURRENT_SEASON - EARLIEST_SEASON) * 6
    print(f"[pct_cache] {built_count}/{total} complete-season pools found on disk")
