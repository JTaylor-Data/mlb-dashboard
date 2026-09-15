"""
/api/player/{id}/similar - "similar players" comps, first pass. See
analytics/similarity.py for the actual nearest-neighbor logic; this file
just builds the cross-season pool (one percentile-ranked pool per season,
2021-current, reusing the SAME cached _pitcher_pool/_batter_pool the main
player page already builds - so a season that's already warm for the
player page is warm here too, no duplicate work) and exposes it as an
endpoint with names/teams resolved for display.
"""

from fastapi import APIRouter, Depends, Query


from analytics.similarity import (
    HITTER_SIMILARITY_METRICS, PITCHER_SIMILARITY_METRICS, SIMILARITY_METRIC_LABELS,
    build_pitcher_arsenal_pool, build_similarity_pool, find_comps,
)
from api.cache import ttl_cache
from api.db import get_conn
from api.routers.players import _batter_pool, _pitcher_pool

router = APIRouter(prefix="/api")


def _all_seasons(conn):
    return [r[0] for r in conn.execute("SELECT DISTINCT season FROM pitches ORDER BY season").fetchall()]


# Unsplit (ALL) pools only - a career trait profile (this is what's being
# compared, not a matchup) shouldn't move with the L/R toggle, same
# reasoning the pitcher side's movement plot already uses elsewhere on this
# page. Cached long-TTL like the other league-wide pools - see warmup.py.
# Only recomputes when the YEAR changes (season pool rebuild) - handedness
# toggling never touches this at all, by construction (nothing here reads
# split state).
@ttl_cache()
def _similarity_pool_hitters(conn):
    seasons = _all_seasons(conn)
    pools = {season: _batter_pool(conn, season, None) for season in seasons}
    return build_similarity_pool(pools, HITTER_SIMILARITY_METRICS)


@ttl_cache()
def _similarity_pool_pitchers(conn):
    seasons = _all_seasons(conn)
    pools = {season: _pitcher_pool(conn, season, None) for season in seasons}
    return build_similarity_pool(pools, PITCHER_SIMILARITY_METRICS)


# p_throws is a fixed per-pitcher fact, not season- or split-dependent -
# one small lookup, not part of the percentile pool itself. Used to hard-
# lock pitcher comps to the query pitcher's own throwing hand - see
# analytics/similarity.py's find_comps() docstring.
@ttl_cache()
def _pitcher_hands(conn):
    rows = conn.execute("SELECT player_id, throws FROM players WHERE throws IS NOT NULL").fetchall()
    return {pid: hand for pid, hand in rows}


# EXPLORATORY (see analytics/similarity.py's module docstring) - the
# arsenal-overlap pool feeding find_comps()'s first-attempt arsenal-
# similarity blend.
@ttl_cache()
def _pitcher_arsenal_pool(conn):
    seasons = _all_seasons(conn)
    return build_pitcher_arsenal_pool(conn, seasons)


def _resolve_player(conn, player_id):
    row = conn.execute("SELECT name, team_id FROM players WHERE player_id = ?", (player_id,)).fetchone()
    return {"name": row[0], "team_id": row[1]} if row else {"name": None, "team_id": None}


@router.get("/player/{player_id}/similar")
def get_similar_players(
    player_id: int,
    role: str = Query(..., pattern="^(pitcher|batter)$"),
    season: int = Query(2026),
    top_n: int = Query(3, ge=1, le=10),
    conn=Depends(get_conn),
):
    if role == "pitcher":
        pool = _similarity_pool_pitchers(conn)
        metrics = PITCHER_SIMILARITY_METRICS
        comps = find_comps(
            pool, player_id, season, metrics, top_n=top_n,
            p_throws_by_player=_pitcher_hands(conn), arsenal_pool=_pitcher_arsenal_pool(conn),
        )
    else:
        pool = _similarity_pool_hitters(conn)
        metrics = HITTER_SIMILARITY_METRICS
        comps = find_comps(pool, player_id, season, metrics, top_n=top_n)
    qualified = (player_id, season) in pool.index
    for c in comps:
        c.update(_resolve_player(conn, c["player_id"]))

    return {
        "player_id": player_id,
        "season": season,
        "role": role,
        "qualified": qualified,
        "metrics": [{"key": m, "label": SIMILARITY_METRIC_LABELS[m]} for m in metrics],
        "comps": comps,
    }
