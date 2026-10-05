"""
Offline rescore script — writes per-season raw-prediction parquets for Stuff+
v4, Tunneling+ v2 (v3 UI), and Location+ v3.  The server warmup reads these
instead of running LightGBM live, cutting cold-start time from ~90s to <5s.

Usage
-----
    python scripts/rescore_models.py                  # all models, all seasons
    python scripts/rescore_models.py --season 2026    # current season only
    python scripts/rescore_models.py --model stuff    # one model, all seasons
    python scripts/rescore_models.py --season 2026 --model tunneling

Past seasons are skipped when a parquet already exists for them (they're
static; no new pitches ever land in a closed season).  The current season is
ALWAYS rescored so weekly cron runs pick up the latest pitches.

Each parquet stores the raw per-pitch predictions in the EXACT same column
shape that the model's _raw_predictions() function returns, so the warmup
path can inject it directly into the model's module-level cache.  A
score_meta.json records scored_at, model_version, pitcher_count, and the
season-wide normalization params (norm_mean / norm_std) used to standardize
scores, so new mid-season debutants can be scored individually without
recomputing the full-season distribution.
"""
import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = ROOT / "scores"
SCORES_DIR.mkdir(exist_ok=True)
META_PATH = SCORES_DIR / "score_meta.json"

FIRST_SEASON = 2015
CURRENT_SEASON = 2026

sys.path.insert(0, str(ROOT))

from api.db import DB_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def _save_meta(meta: dict):
    META_PATH.write_text(json.dumps(meta, indent=2))


def _is_scored(meta: dict, model_key: str, season: int) -> bool:
    return str(season) in meta.get(model_key, {})


def _record_meta(meta: dict, model_key: str, season: int, model_version: str,
                 pitcher_count: int, norm_mean: float, norm_std: float):
    meta.setdefault(model_key, {})[str(season)] = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "pitcher_count": pitcher_count,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


# ---------------------------------------------------------------------------
# Stuff+ v4
# ---------------------------------------------------------------------------

def _stuff_parquet_path(season: int) -> Path:
    from analytics.stuff_model import STUFF_ACTIVE_VERSION
    return SCORES_DIR / f"stuff_plus_{STUFF_ACTIVE_VERSION}_{season}.parquet"


def rescore_stuff(conn, season: int, force: bool = False):
    from analytics.stuff_model import STUFF_ACTIVE_VERSION, _raw_predictions

    path = _stuff_parquet_path(season)
    meta = _load_meta()
    model_key = f"stuff_{STUFF_ACTIVE_VERSION}"

    if not force and season < CURRENT_SEASON and path.exists() and _is_scored(meta, model_key, season):
        print(f"  [stuff {STUFF_ACTIVE_VERSION}] {season}: already scored, skipping")
        return

    print(f"  [stuff {STUFF_ACTIVE_VERSION}] {season}: scoring...", end="", flush=True)
    t0 = time.monotonic()
    raw = _raw_predictions(conn, season)
    elapsed = time.monotonic() - t0
    print(f" {elapsed:.1f}s ({len(raw):,} rows)")

    if raw.empty:
        print(f"  [stuff {STUFF_ACTIVE_VERSION}] {season}: no data, skipping")
        return

    raw.to_parquet(path, index=False)
    norm_mean = float(raw["pred"].mean())
    norm_std = float(raw["pred"].std())
    pitcher_count = raw["pitcher_id"].nunique()
    _record_meta(meta, model_key, season, STUFF_ACTIVE_VERSION, pitcher_count, norm_mean, norm_std)
    _save_meta(meta)
    print(f"  [stuff {STUFF_ACTIVE_VERSION}] {season}: saved {path.name} ({pitcher_count} pitchers, "
          f"mean={norm_mean:.5f}, std={norm_std:.5f})")


# ---------------------------------------------------------------------------
# Tunneling+ v2 (served as v3 in the UI)
# ---------------------------------------------------------------------------

def _tunneling_parquet_path(season: int) -> Path:
    return SCORES_DIR / f"tunneling_plus_v3_{season}.parquet"


def rescore_tunneling(conn, season: int, force: bool = False):
    from analytics.tunneling_v2_model import _raw_predictions

    path = _tunneling_parquet_path(season)
    meta = _load_meta()
    model_key = "tunneling_v3"

    if not force and season < CURRENT_SEASON and path.exists() and _is_scored(meta, model_key, season):
        print(f"  [tunneling v3] {season}: already scored, skipping")
        return

    print(f"  [tunneling v3] {season}: scoring...", end="", flush=True)
    t0 = time.monotonic()
    raw = _raw_predictions(conn, season)
    elapsed = time.monotonic() - t0
    print(f" {elapsed:.1f}s ({len(raw):,} rows)")

    if raw.empty:
        print(f"  [tunneling v3] {season}: no data, skipping")
        return

    raw.to_parquet(path, index=False)

    g = raw.groupby("pitcher_id")["xrv"].mean()
    norm_mean = float(g.mean())
    norm_std = float(g.std()) if g.std() > 0 else 1.0
    pitcher_count = len(g)
    _record_meta(meta, model_key, season, "v3", pitcher_count, norm_mean, norm_std)
    _save_meta(meta)
    print(f"  [tunneling v3] {season}: saved {path.name} ({pitcher_count} pitchers, "
          f"mean={norm_mean:.5f}, std={norm_std:.5f})")


# ---------------------------------------------------------------------------
# Location+ v3
# ---------------------------------------------------------------------------

def _location_parquet_path(season: int) -> Path:
    return SCORES_DIR / f"location_plus_v3_{season}.parquet"


def rescore_location(conn, season: int, force: bool = False):
    from analytics.location_v3_model import _raw_predictions, LOCATION_ACTIVE_VERSION

    path = _location_parquet_path(season)
    meta = _load_meta()
    model_key = f"location_{LOCATION_ACTIVE_VERSION}"

    if not force and season < CURRENT_SEASON and path.exists() and _is_scored(meta, model_key, season):
        print(f"  [location {LOCATION_ACTIVE_VERSION}] {season}: already scored, skipping")
        return

    print(f"  [location {LOCATION_ACTIVE_VERSION}] {season}: scoring...", end="", flush=True)
    t0 = time.monotonic()
    raw = _raw_predictions(conn, season)
    elapsed = time.monotonic() - t0
    print(f" {elapsed:.1f}s ({len(raw):,} rows)")

    if raw.empty:
        print(f"  [location {LOCATION_ACTIVE_VERSION}] {season}: no data, skipping")
        return

    raw.to_parquet(path, index=False)

    g = raw.groupby("pitcher_id")["xrv"].mean()
    norm_mean = float(g.mean())
    norm_std = float(g.std()) if g.std() > 0 else 1.0
    pitcher_count = len(g)
    _record_meta(meta, model_key, season, LOCATION_ACTIVE_VERSION, pitcher_count, norm_mean, norm_std)
    _save_meta(meta)
    print(f"  [location {LOCATION_ACTIVE_VERSION}] {season}: saved {path.name} ({pitcher_count} pitchers, "
          f"mean={norm_mean:.5f}, std={norm_std:.5f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RESCORE_FNS = {
    "stuff": rescore_stuff,
    "tunneling": rescore_tunneling,
    "location": rescore_location,
}


def main():
    parser = argparse.ArgumentParser(description="Rescore Stuff+/Tunneling+/Location+ and write parquet caches.")
    parser.add_argument("--season", type=int, default=None,
                        help="Single season to score (default: all 2015-current).")
    parser.add_argument("--model", choices=["stuff", "tunneling", "location", "all"], default="all",
                        help="Which model to score (default: all).")
    parser.add_argument("--force", action="store_true",
                        help="Rescore even if past-season parquet already exists.")
    args = parser.parse_args()

    seasons = [args.season] if args.season else list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    models = list(RESCORE_FNS.keys()) if args.model == "all" else [args.model]

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        for season in seasons:
            force = args.force or (season == CURRENT_SEASON)
            print(f"\nSeason {season}:")
            for model in models:
                try:
                    RESCORE_FNS[model](conn, season, force=force)
                except Exception as e:
                    print(f"  [{model}] {season}: ERROR — {e}")
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
