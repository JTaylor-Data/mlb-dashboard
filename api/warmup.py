"""
Pre-warms the TTL-cached league-wide pool computations (percentile pools,
Stuff+, pitcher roles, league-average rows, standings) so the player page -
and the first request after a cold cache in general - doesn't have to eat a
~15s+ full-table scan live in the request path. Runs in a background thread
so the server starts accepting requests immediately; requests that land
before warmup finishes still work, they just recompute (and cache) on
demand like before.

Score-cache parquets
--------------------
Stuff+/Tunneling+/Location+ raw-prediction parquets (written by
scripts/rescore_models.py, refreshed weekly by the scheduler) are injected
into each model's module-level raw-predictions cache at warmup time.  This
means the @ttl_cache wrappers (_stuff_plus_table, _v3_raw_predictions_table,
_location_raw_predictions_table) run their normal aggregation logic but find
the expensive LightGBM step already done — cutting cold-start model scoring
from ~90s to effectively zero.  If a parquet is missing (first deploy, or
between a model-version bump and the next weekly rescore) the functions fall
back to live scoring exactly as before.
"""
import ctypes
import gc
import logging
import os
import sqlite3
import sys
import threading
import time as _time
from pathlib import Path

log = logging.getLogger("warmup")

_INSTRUMENT = os.environ.get("BOOT_INSTRUMENT") == "1"
_step_start = None


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1.0


def _step(label):
    """Flushed start/end markers for boot-condition instrumentation.
    No-op unless BOOT_INSTRUMENT=1 (see diagnostics/boot_test for the harness
    that reads these); never runs in normal boots."""
    global _step_start
    if not _INSTRUMENT:
        return
    now = _time.monotonic()
    if label.startswith("end:") and _step_start is not None:
        elapsed = now - _step_start
        print(f"BOOTSTEP {label} elapsed={elapsed:.1f}s rss={_rss_mb():.1f}MB", flush=True, file=sys.stderr)
    else:
        _step_start = now
        print(f"BOOTSTEP {label} rss={_rss_mb():.1f}MB", flush=True, file=sys.stderr)

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None  # not glibc/Linux (e.g. local Windows dev) - nothing to trim


def _trim_memory():
    """Return freed heap pages to the OS after a pandas-heavy step.

    gc.collect() alone isn't enough here: pulling a full season's pitches
    into a DataFrame and discarding it frees the memory to glibc's malloc,
    but glibc keeps those pages mapped in the process (arena fragmentation)
    rather than handing them back to the kernel - so RSS keeps climbing
    across successive warm-up steps even though each one's own data is
    short-lived. Confirmed via Docker memory tests this session: three
    back-to-back calls that each build and discard a ~700K-row DataFrame
    pushed RSS from ~290MB to 440MB+ with nothing live to show for it.
    malloc_trim(0) is glibc-specific and a no-op (never called) elsewhere.
    """
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass

from api.db import DB_PATH

from config.model_policy import CURRENT_SEASON  # single source of truth

SPLITS = [None, "L", "R"]
# Only the current season is pre-warmed — warming all backfilled seasons on
# every startup would multiply warmup cost. Complete seasons (< CURRENT_SEASON)
# are now served from disk-persisted percentile pools (see analytics/percentile_cache.py)
# so their first-request cost is near zero once built.

SCORES_DIR = Path(__file__).resolve().parent.parent / "parquets"
# Parquet-injected raw predictions are given a long TTL so they stay warm
# for the full day and aren't evicted before the API-layer TTL cache expires.
_PARQUET_TTL = 14 * 3600  # 14 hours — longer than api/cache.py's 12h TTL


def _inject_stuff_parquet(season: int):
    """Load Stuff+ raw predictions from parquet into stuff_model's cache so
    compute_stuff_plus/compute_stuff_plus_by_stand skip LightGBM scoring."""
    try:
        import pandas as pd
        from analytics.stuff_model import STUFF_ACTIVE_VERSION, _raw_predictions_cache
        path = SCORES_DIR / f"stuff_plus_{STUFF_ACTIVE_VERSION}_{season}.parquet"
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        _raw_predictions_cache[(STUFF_ACTIVE_VERSION, season)] = (df, _time.monotonic() + _PARQUET_TTL)
        log.info("stuff %s %s: loaded from parquet (%s rows)", STUFF_ACTIVE_VERSION, season, f"{len(df):,}")
        return True
    except Exception as e:
        log.warning("stuff parquet %s load failed: %s", season, e)
        return False


def _inject_tunneling_parquet(season: int):
    """Load Tunneling+ raw predictions from parquet into tunneling_v2_model's cache."""
    try:
        import pandas as pd
        from analytics import tunneling_v2_model
        path = SCORES_DIR / f"tunneling_plus_v3_{season}.parquet"
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        tunneling_v2_model._cache[f"raw_{season}"] = (df, _time.monotonic() + _PARQUET_TTL)
        log.info("tunneling v3 %s: loaded from parquet (%s rows)", season, f"{len(df):,}")
        return True
    except Exception as e:
        log.warning("tunneling parquet %s load failed: %s", season, e)
        return False


def _inject_location_parquet(season: int):
    """Load Location+ raw predictions from parquet into location_v3_model's cache."""
    try:
        import pandas as pd
        from analytics import location_v3_model
        path = SCORES_DIR / f"location_plus_v3_{season}.parquet"
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        location_v3_model._cache[f"raw_v3_{season}"] = (df, _time.monotonic() + _PARQUET_TTL)
        log.info("location v3 %s: loaded from parquet (%s rows)", season, f"{len(df):,}")
        return True
    except Exception as e:
        log.warning("location parquet %s load failed: %s", season, e)
        return False


def _inject_all_parquets(seasons):
    """Pre-load parquets for the given seasons before TTL-cache warmup runs.
    Returns (stuff_ok, tun_ok, loc_ok) booleans for each season in order, but
    in practice we just log and fall through — missing parquets are fine."""
    for season in seasons:
        _inject_stuff_parquet(season)
        _inject_tunneling_parquet(season)
        _inject_location_parquet(season)


def _warm(conn, full):
    """`full=True` (server startup - a one-time cost, worth spending fully)
    warms every split. `full=False` (after a "Refresh data" click, which a
    user can trigger repeatedly in one sitting) warms only the ALL split -
    the L/R percentile pools are a genuine ~25s chunk of warmup time that
    most refreshes don't need paid again immediately; a user who actually
    toggles to L/R shortly after a refresh just eats the same one-time cold
    cost this app already accepts for older seasons, rather than every
    refresh paying for splits that may not get viewed at all this session."""
    from api.routers.heatmaps import _league_command_pool
    from api.routers.players import (
        _all_players, _all_pitcher_arm_angles, _batter_pool, _league_avg_batter_row, _league_hr_fb,
        _pitcher_league_avg_row, _pitcher_pool, _pitcher_pool_unqualified, _pitcher_roles, _stuff_plus_by_stand_table, _stuff_plus_table,
    )
    from api.routers.deception_report import REPORT_SEASON as DECEPTION_REPORT_SEASON, _raw_predictions_table as _deception_raw_predictions_table
    from api.routers.tunneling_report import _v3_raw_predictions_table
    from api.routers.location_report import _raw_predictions_table as _location_raw_predictions_table
    from api.routers.similar_players import (
        _pitcher_arsenal_pool, _pitcher_hands, _similarity_pool_hitters, _similarity_pool_pitchers,
    )
    from api.routers.probable_pitchers import _stuff_dict, _tunneling_dict, _location_dict, _deception_dict
    from api.routers.standings import _standings
    from api.routers.stuff_model_report import _stuff_leaderboard_overall as _stuff_leaderboard
    from api.routers.team_card import _cached_rows_and_ranks, _cached_sos
    from api.routers.top_performers import _leaderboard

    season = CURRENT_SEASON

    if full:
        # Startup/scheduled refresh: full warmup — all model tables, all splits,
        # similar-players pool, career tables, and last 10 leaderboard dates.

        # Pre-check disk for complete-season percentile pools. This is a fast
        # read-only scan; missing pools will be built lazily on first request
        # and persisted so subsequent restarts don't rebuild them.
        _step("start:preload_complete_seasons")
        from analytics.percentile_cache import preload_complete_seasons
        preload_complete_seasons(conn)
        _step("end:preload_complete_seasons")

        # Only the CURRENT season's parquets are injected eagerly here -
        # it's needed immediately by the steps right below. Past seasons
        # used to be injected here too (6 seasons' worth, ~230MB retained
        # for the rest of the process's life, every boot) even though
        # nobody necessarily looks at any given past season this session.
        # Each model's _raw_predictions() now has its own lazy fallback to
        # read a season's parquet directly on a cache miss (see
        # analytics/score_parquet_cache.py), so past seasons load on
        # whichever request actually needs them instead of all at once.
        _step("start:inject_all_parquets")
        _inject_all_parquets([season])
        _step("end:inject_all_parquets")

        _step("start:stuff_plus_table")
        _stuff_plus_table(conn, season)
        _step("end:stuff_plus_table")
        _step("start:stuff_plus_by_stand_table")
        _stuff_plus_by_stand_table(conn, season)
        _step("end:stuff_plus_by_stand_table")
        _step("start:pitcher_roles")
        _pitcher_roles(conn, season)
        _step("end:pitcher_roles")
        _step("start:league_hr_fb")
        _league_hr_fb(conn, season)
        _step("end:league_hr_fb")
        _step("start:deception_raw_predictions")
        _deception_raw_predictions_table(conn, DECEPTION_REPORT_SEASON)
        _step("end:deception_raw_predictions")
        _step("start:v3_raw_predictions")
        _v3_raw_predictions_table(conn, season)
        _step("end:v3_raw_predictions")
        _step("start:location_raw_predictions")
        _location_raw_predictions_table(conn, season)
        _step("end:location_raw_predictions")
        _step("start:stuff_leaderboard")
        _stuff_leaderboard(conn, season)
        _step("end:stuff_leaderboard")
        _step("start:cached_rows_and_ranks")
        _cached_rows_and_ranks(conn, season)
        _step("end:cached_rows_and_ranks")
        _step("start:cached_sos")
        _cached_sos(conn, season)
        _step("end:cached_sos")
        for split in SPLITS:
            _step(f"start:pitcher_pool[{split}]")
            _pitcher_pool(conn, season, split)
            _step(f"end:pitcher_pool[{split}]")
            _step(f"start:batter_pool[{split}]")
            _batter_pool(conn, season, split)
            _step(f"end:batter_pool[{split}]")
            _step(f"start:league_avg_batter_row[{split}]")
            _league_avg_batter_row(conn, season, split)
            _step(f"end:league_avg_batter_row[{split}]")
            _step(f"start:pitcher_league_avg_row[{split}]")
            _pitcher_league_avg_row(conn, season, split)
            _step(f"end:pitcher_league_avg_row[{split}]")
            _step(f"start:pitcher_pool_unqualified[{split}]")
            _pitcher_pool_unqualified(conn, season, split)
            _step(f"end:pitcher_pool_unqualified[{split}]")
            _trim_memory()
        for split in ("L", "R"):
            _step(f"start:league_command_pool[{split}]")
            _league_command_pool(conn, season, split)
            _step(f"end:league_command_pool[{split}]")
            _trim_memory()
        _step("start:similarity_pools")
        _similarity_pool_hitters(conn)
        _similarity_pool_pitchers(conn)
        _pitcher_hands(conn)
        _pitcher_arsenal_pool(conn)
        _trim_memory()
        _step("end:similarity_pools")
        # Past-season career tables (Stuff+/Tunneling+/Location+ for every
        # season 2021-current) used to be pre-built for every past season
        # here, at every boot - the other half of the ~230MB this warmup
        # used to hold permanently (see the parquet-injection comment
        # above). They're not needed unless someone actually opens a
        # career page for that season, and when they do, _raw_predictions'
        # new lazy parquet fallback makes that first request fast (a file
        # read, not live LightGBM scoring) - so this no longer pre-builds
        # them eagerly; they load on demand instead.
        _step("start:career_dicts")
        _stuff_dict(conn, season)
        _tunneling_dict(conn, season)
        _location_dict(conn, season)
        _deception_dict(conn)
        _step("end:career_dicts")
        _step("start:standings")
        _standings(conn, season)
        _step("end:standings")
        _step("start:all_players")
        _all_players(conn, season)
        _step("end:all_players")
        _step("start:all_pitcher_arm_angles")
        _all_pitcher_arm_angles(conn, season)
        _step("end:all_pitcher_arm_angles")
        # Warm last 10 game dates for the "Top performer" badge on player pages.
        _step("start:leaderboard_10_dates")
        recent_dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT game_date FROM games ORDER BY game_date DESC LIMIT 10"
        ).fetchall()]
        for d in recent_dates:
            try:
                _leaderboard(conn, d)
            except Exception as e:
                log.warning("_leaderboard(%s) failed: %s", d, e)
            _trim_memory()
        _step("end:leaderboard_10_dates")
        if _INSTRUMENT:
            mods = {m: (m in sys.modules) for m in ("lightgbm", "sklearn", "shap")}
            print(f"BOOTSTEP sys_modules {mods} rss={_rss_mb():.1f}MB", flush=True, file=sys.stderr)
        _step("warmup_complete")
    else:
        # Manual "Refresh data" click: model scoring caches weren't cleared
        # (one day of new pitches doesn't meaningfully change Stuff+/Tunneling+/
        # Deception+ scores and recomputing them cold costs ~25-35s). Only
        # warm the genuinely daily-fresh data: standings, today's leaderboard,
        # and the light non-model pools that ARE cleared (player list, roles, etc).
        _pitcher_roles(conn, season)
        _standings(conn, season)
        _all_players(conn, season)
        _cached_rows_and_ranks(conn, season)
        _cached_sos(conn, season)
        # Warm only today and yesterday — enough for the Top Performers page and
        # the "Last 5 Games" badge for the most-recently-played dates, without
        # the 10-15s cost of warming a full 10-date window.
        recent_dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT game_date FROM games ORDER BY game_date DESC LIMIT 2"
        ).fetchall()]
        for d in recent_dates:
            try:
                _leaderboard(conn, d)
            except Exception as e:
                log.warning("_leaderboard(%s) failed: %s", d, e)


def _score_new_pitcher(conn, pitcher_id: int, season: int, model: str) -> float | None:
    """Score a single pitcher who didn't make it into the last weekly parquet
    (e.g. a mid-week debutant or a call-up after Sunday's rescore).

    Instead of recomputing the full-season distribution — which would require
    running LightGBM on hundreds of thousands of pitches — this function scores
    only the given pitcher's pitches and then normalizes using the stored
    norm_mean / norm_std from score_meta.json (the values computed when the
    parquet was last written).

    Returns the plus-scale score (100 = league mean, 10 pts = 1 SD), or None
    if the pitcher has no qualifying pitches or the meta is unavailable.

    model: one of "stuff", "tunneling", "location"
    """
    import json as _json
    meta_path = SCORES_DIR / "score_meta.json"
    if not meta_path.exists():
        return None

    meta = _json.loads(meta_path.read_text())
    season_str = str(season)

    if model == "stuff":
        from analytics.stuff_model import STUFF_ACTIVE_VERSION, _raw_predictions
        model_key = f"stuff_{STUFF_ACTIVE_VERSION}"
        season_meta = meta.get(model_key, {}).get(season_str)
        if season_meta is None:
            return None
        raw_all = _raw_predictions(conn, season)
        pitcher_rows = raw_all[raw_all["pitcher_id"] == pitcher_id]
        if pitcher_rows.empty:
            return None
        from analytics.stuff_model import _shrinkage_params, _shrunk_group_means, _to_stuff_plus
        k_by_type, fallback_k, type_means = _shrinkage_params(raw_all)
        g = _shrunk_group_means(pitcher_rows, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
        norm_mean = season_meta["norm_mean"]
        norm_std = season_meta["norm_std"]
        g["stuff_plus"] = _to_stuff_plus(g["shrunk_mean"], norm_mean, norm_std)
        import numpy as np
        return float(np.average(g["stuff_plus"], weights=g["n_pitches"]))

    if model == "tunneling":
        from analytics.tunneling_v2_model import _raw_predictions as tun_raw
        model_key = "tunneling_v3"
        season_meta = meta.get(model_key, {}).get(season_str)
        if season_meta is None:
            return None
        raw_all = tun_raw(conn, season)
        pitcher_rows = raw_all[raw_all["pitcher_id"] == pitcher_id]
        if pitcher_rows.empty:
            return None
        mean_xrv = float(pitcher_rows["xrv"].mean())
        norm_mean = season_meta["norm_mean"]
        norm_std = season_meta["norm_std"]
        return 100.0 - 10.0 * (mean_xrv - norm_mean) / norm_std if norm_std else None

    if model == "location":
        from analytics.location_v3_model import _raw_predictions as loc_raw, LOCATION_ACTIVE_VERSION
        model_key = f"location_{LOCATION_ACTIVE_VERSION}"
        season_meta = meta.get(model_key, {}).get(season_str)
        if season_meta is None:
            return None
        raw_all = loc_raw(conn, season)
        pitcher_rows = raw_all[raw_all["pitcher_id"] == pitcher_id]
        if pitcher_rows.empty:
            return None
        mean_xrv = float(pitcher_rows["xrv"].mean())
        norm_mean = season_meta["norm_mean"]
        norm_std = season_meta["norm_std"]
        return 100.0 - 10.0 * (mean_xrv - norm_mean) / norm_std if norm_std else None

    raise ValueError(f"Unknown model: {model!r}. Expected 'stuff', 'tunneling', or 'location'.")


def warm_cache_sync(full=True):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        _warm(conn, full)
    finally:
        conn.close()


def warm_cache_background():
    threading.Thread(target=warm_cache_sync, daemon=True).start()
