"""/api/playoffs/* - wild card race, division leaders, projected bracket,
and projected playoff roster. See analytics/playoffs.py for the real
computations and the honest caveats on the bracket/roster projections."""
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analytics.playoffs import division_leaders, playoff_teams, projected_roster, unified_bracket, wildcard_race
from analytics.standings import compute_standings

from api.cache import ttl_cache
from api.db import get_conn

router = APIRouter(prefix="/api/playoffs")


@ttl_cache()
def _standings(conn, season):
    return compute_standings(conn, season)


@router.get("/division-leaders")
def get_division_leaders(season: int = Query(2026), conn=Depends(get_conn)):
    return {"season": season, "leaders": division_leaders(conn, season, _standings(conn, season))}


@router.get("/wildcard")
def get_wildcard_race(league: str = Query("AL", pattern="^(AL|NL)$"), season: int = Query(2026), conn=Depends(get_conn)):
    return {"season": season, "league": league, "race": wildcard_race(conn, season, _standings(conn, season), league)}


@ttl_cache()
def _bracket(conn, season):
    return unified_bracket(conn, _standings(conn, season), season)


@router.get("/bracket")
def get_bracket(season: int = Query(2026), conn=Depends(get_conn)):
    """Both leagues at once (no `league` param - the bracket page always
    shows both). Real results when they exist for this season, else the
    live standings projection - see analytics/playoffs.py's unified_bracket."""
    return _bracket(conn, season)


@router.get("/teams")
def get_playoff_teams(league: str = Query("AL", pattern="^(AL|NL)$"), season: int = Query(2026), conn=Depends(get_conn)):
    """Team picker options for the Rosters tab - the 6 teams currently in
    bracket position for `league`, with name/abbreviation resolved."""
    standings = _standings(conn, season)
    team_ids = set(playoff_teams(conn, season, standings, league))
    league_df = standings[standings["team_id"].astype(int).isin(team_ids)]
    teams = [
        {"team_id": int(r["team_id"]), "name": r["name"], "abbreviation": r["abbreviation"]}
        for _, r in league_df.sort_values("win_pct", ascending=False).iterrows()
    ]
    return {"season": season, "league": league, "teams": teams}


@ttl_cache()
def _roster(conn, season, team_id):
    standings = _standings(conn, season)
    row = standings[standings["team_id"].astype(int) == int(team_id)]
    games_played = int(row.iloc[0]["games_played"]) if not row.empty else None
    return projected_roster(conn, season, team_id, games_played_this_team=games_played)


@router.get("/roster")
def get_projected_roster(team_id: int = Query(...), season: int = Query(2026), conn=Depends(get_conn)):
    return _roster(conn, season, team_id)
