"""
Pre-warms the TTL-cached league-wide pool computations (percentile pools,
Stuff+, pitcher roles, league-average rows, standings) so the player page -
and the first request after a cold cache in general - doesn't have to eat a
~15s+ full-table scan live in the request path. Runs in a background thread
so the server starts accepting requests immediately; requests that land
before warmup finishes still work, they just recompute (and cache) on
demand like before.
"""
import sqlite3
import threading

from api.db import DB_PATH

SPLITS = [None, "L", "R"]
# Only the current season is pre-warmed - warming all 6 backfilled seasons
# on every startup would multiply the warmup cost 6x for seasons that may
# never get clicked. Older seasons just pay the ~15s cost on first request
# instead, same as this app's normal cold-cache behavior everywhere else.
CURRENT_SEASON = 2026


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
    from api.routers.stuff_model_report import _stuff_leaderboard
    from api.routers.team_card import _cached_rows_and_ranks, _cached_sos
    from api.routers.top_performers import _leaderboard

    season = CURRENT_SEASON

    if full:
        # Startup/scheduled refresh: full warmup — all model tables, all splits,
        # similar-players pool, career tables, and last 10 leaderboard dates.
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
                print(f"[warmup] career stuff_table({ps}) failed: {e}")
            try:
                _career_tun_table(conn, ps)
            except Exception as e:
                print(f"[warmup] career tun_table({ps}) failed: {e}")
            try:
                _career_loc_table(conn, ps)
            except Exception as e:
                print(f"[warmup] career loc_table({ps}) failed: {e}")
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
                print(f"[warmup] _leaderboard({d}) failed: {e}")
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
                print(f"[warmup] _leaderboard({d}) failed: {e}")


def warm_cache_sync(full=True):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        _warm(conn, full)
    finally:
        conn.close()


def warm_cache_background():
    threading.Thread(target=warm_cache_sync, daemon=True).start()
