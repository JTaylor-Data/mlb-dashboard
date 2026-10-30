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
import logging
import sqlite3
import threading
import time as _time
from pathlib import Path

log = logging.getLogger("warmup")

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
        _pitcher_pool, _pitcher_roles, _stuff_plus_by_stand_table, _stuff_plus_table,
    )
    from api.routers.deception_report import REPORT_SEASON as DECEPTION_REPORT_SEASON, _raw_predictions_table as _deception_raw_predictions_table
    from api.routers.tunneling_report import _v3_raw_predictions_table
    from api.routers.location_report import _raw_predictions_table as _location_raw_predictions_table
    from api.routers.similar_players import _similarity_pool_hitters, _similarity_pool_pitchers
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
        from analytics.percentile_cache import preload_complete_seasons
        preload_complete_seasons(conn)

        # Inject parquets for current season and all career seasons up front so
        # every _raw_predictions() call below finds the module-level cache warm
        # and skips LightGBM entirely.  Missing parquets fall through silently.
        past_seasons_list = list(range(2021, season))
        _inject_all_parquets([season] + past_seasons_list)

        _stuff_plus_table(conn, season)
        _stuff_plus_by_stand_table(conn, season)
        _pitcher_roles(conn, season)
        _league_hr_fb(conn, season)
        _deception_raw_predictions_table(conn, DECEPTION_REPORT_SEASON)
        _v3_raw_predictions_table(conn, season)
        _location_raw_predictions_table(conn, season)
        _stuff_leaderboard(conn, season)
        _cached_rows_and_ranks(conn, season)
        _cached_sos(conn, season)
        for split in SPLITS:
            _pitcher_pool(conn, season, split)
            _batter_pool(conn, season, split)
            _league_avg_batter_row(conn, season, split)
        for split in ("L", "R"):
            _league_command_pool(conn, season, split)
        _similarity_pool_hitters(conn)
        _similarity_pool_pitchers(conn)
        from api.routers.stuff_model_report import _raw_predictions_table as _career_stuff_table
        from api.routers.tunneling_report import _v3_raw_predictions_table as _career_tun_table
        from api.routers.location_report import _raw_predictions_table as _career_loc_table
        past_seasons = [s for s in range(2021, CURRENT_SEASON)]
        for ps in past_seasons:
            try:
                _career_stuff_table(conn, ps)
            except Exception as e:
                log.warning("career stuff_table(%s) failed: %s", ps, e)
            try:
                _career_tun_table(conn, ps)
            except Exception as e:
                log.warning("career tun_table(%s) failed: %s", ps, e)
            try:
                _career_loc_table(conn, ps)
            except Exception as e:
                log.warning("career loc_table(%s) failed: %s", ps, e)
        _stuff_dict(conn, season)
        _tunneling_dict(conn, season)
        _location_dict(conn, season)
        _deception_dict(conn)
        _standings(conn, season)
        _all_players(conn, season)
        _all_pitcher_arm_angles(conn, season)
        # Warm last 10 game dates for the "Top performer" badge on player pages.
        recent_dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT game_date FROM games ORDER BY game_date DESC LIMIT 10"
        ).fetchall()]
        for d in recent_dates:
            try:
                _leaderboard(conn, d)
            except Exception as e:
                log.warning("_leaderboard(%s) failed: %s", d, e)
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
