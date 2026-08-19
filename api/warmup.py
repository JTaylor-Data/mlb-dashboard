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
    from api.routers.players import (
        _batter_pool, _league_avg_batter_row, _league_hr_fb,
        _pitcher_pool, _pitcher_roles, _stuff_plus_by_stand_table, _stuff_plus_table,
    )
    from api.routers.standings import _standings

    season = CURRENT_SEASON
    _stuff_plus_table(conn, season)
    _stuff_plus_by_stand_table(conn, season)
    _pitcher_roles(conn, season)
    _league_hr_fb(conn, season)
    for split in (SPLITS if full else [None]):
        _pitcher_pool(conn, season, split)
        _batter_pool(conn, season, split)
        _league_avg_batter_row(conn, season, split)
    _standings(conn, season)


def warm_cache_sync(full=True):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        _warm(conn, full)
    finally:
        conn.close()


def warm_cache_background():
    threading.Thread(target=warm_cache_sync, daemon=True).start()
