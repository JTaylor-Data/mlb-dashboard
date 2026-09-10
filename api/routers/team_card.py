"""
/api/team/{team_id}/card - powers the click-any-logo team card popup
(frontend/js/team-card.js), event-delegated so EVERY team logo on EVERY
page opens through this same endpoint - hit constantly across a session,
including repeat clicks on the same team. See analytics/team_card.py for
the actual data assembly and pulls/pull_team_stats.py for where the
numbers come from.

_all_team_rows_and_ranks/remaining_sos each do a real season-wide
computation (a WAR groupby, a full `games` table scan for park factors, a
second independent full `games` scan for win/loss records) that analytics/
team_card.py's own module docstring called "cheap enough not to need"
caching - confirmed directly that it wasn't: both were being recomputed
from scratch on every single card open, all session, sitewide. Cached
here at the router layer (same layering convention as api/routers/
players.py's own wrappers around analytics/stuff_model.py etc. - the
analytics layer doesn't import api.cache itself) and warmed in
api/warmup.py.
"""
from fastapi import APIRouter, Depends, HTTPException

from analytics.strength_of_schedule import remaining_sos
from analytics.team_card import _all_team_rows_and_ranks, team_card, team_metric_rankings

from api.cache import ttl_cache
from api.db import get_conn

router = APIRouter(prefix="/api")


@ttl_cache()
def _cached_rows_and_ranks(conn, season):
    return _all_team_rows_and_ranks(conn, season)


@ttl_cache()
def _cached_sos(conn, season):
    return remaining_sos(conn, season)


@router.get("/team/{team_id}/card")
def get_team_card(team_id: int, season: int = 2026, conn=Depends(get_conn)):
    card = team_card(conn, team_id, season, rows_and_ranks=_cached_rows_and_ranks(conn, season), sos_by_team=_cached_sos(conn, season))
    if card is None:
        raise HTTPException(404, f"No team stats snapshot for team {team_id}/{season} - run pull_team_stats.py first")
    return card


@router.get("/team/metric-rankings/{metric}")
def get_metric_rankings(metric: str, season: int = 2026, conn=Depends(get_conn)):
    """Full 30-team ranked list for one team-card metric - powers the
    per-metric expand arrow."""
    teams = team_metric_rankings(conn, season, metric, rows_and_ranks=_cached_rows_and_ranks(conn, season))
    return {"metric": metric, "season": season, "teams": teams}
