"""/api/standings - wraps standings.py, grouped by division."""
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from standings import best_record_team_id, compute_standings

from api.cache import ttl_cache
from api.db import get_conn

router = APIRouter(prefix="/api")


@ttl_cache()
def _standings(conn, season):
    return compute_standings(conn, season)


@router.get("/standings")
def get_standings(league: str = Query("AL", pattern="^(AL|NL)$"), season: int = Query(2026), conn=Depends(get_conn)):
    standings = _standings(conn, season)
    league_df = standings[standings["league"] == league]
    best_id = best_record_team_id(standings, league)

    divisions = []
    for division in sorted(league_df["division"].unique()):
        div_df = league_df[league_df["division"] == division]
        teams = []
        for _, row in div_df.iterrows():
            teams.append({
                "team_id": int(row["team_id"]),
                "name": row["name"],
                "abbreviation": row["abbreviation"],
                "wins": int(row["wins"]),
                "losses": int(row["losses"]),
                "win_pct": round(float(row["win_pct"]), 3),
                "run_diff": int(row["run_diff"]),
                "one_run_record": row["one_run_record"],
                "last_10": row["last_10"],
                "last_30": row["last_30"],
                "is_best_in_league": bool(int(row["team_id"]) == best_id),
            })
        divisions.append({"name": division, "teams": teams})

    return {"league": league, "best_team_id": int(best_id) if best_id is not None else None, "divisions": divisions}
