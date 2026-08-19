"""
/api/top-performers - wraps wpa.py + boxscore.py exactly as
pages/3_Top_Performers.py did. Cached per-date: a completed day's box
scores don't change, and even "today" changing mid-game is low-stakes for a
personal tool (the Refresh button / TTL expiry covers it).
"""
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from boxscore import batter_daily_stats, fetch_boxscore, pitcher_daily_stats
from wpa import is_walkoff_batter, top_performers

from api.cache import ttl_cache
from api.db import get_conn

router = APIRouter(prefix="/api")


def _opponent(conn, game_id, team_id):
    row = conn.execute("SELECT home_team, away_team FROM games WHERE game_id = ?", (game_id,)).fetchone()
    if row is None:
        return ""
    home, away = int(row[0]), int(row[1])
    is_home = team_id == home
    opp_id = away if is_home else home
    abbrev = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (opp_id,)).fetchone()
    if abbrev is None:
        return ""
    return f"{'vs' if is_home else '@'} {abbrev[0]}"


@ttl_cache()
def _top_10(conn, day_str):
    top = top_performers(conn, day_str, n=10)
    if top.empty:
        return []

    results = []
    boxscore_cache = {}
    for _, row in top.iterrows():
        game_id = int(row["game_id"])
        team_id = int(row["team_id"]) if row["team_id"] is not None else None
        if game_id not in boxscore_cache:
            boxscore_cache[game_id] = fetch_boxscore(game_id)
        bx = boxscore_cache[game_id]

        if row["role"] == "batter":
            stats = batter_daily_stats(bx, int(row["player_id"])) or {}
            walkoff = is_walkoff_batter(conn, int(row["player_id"]), game_id)
        else:
            stats = pitcher_daily_stats(bx, int(row["player_id"])) or {}
            walkoff = False

        results.append(dict(
            player_id=int(row["player_id"]), name=row["name"], role=row["role"],
            wpa=round(float(row["wpa"]), 3), team_id=team_id, stats=stats, walkoff=bool(walkoff),
            opponent=_opponent(conn, game_id, team_id) if team_id is not None else "",
        ))
    return results


@router.get("/top-performers")
def get_top_performers(date: str = Query(...), conn=Depends(get_conn)):
    entries = _top_10(conn, date)
    return {"date": date, "entries": entries}
