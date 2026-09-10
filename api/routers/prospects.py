"""
/api/prospects - MLB Pipeline rankings (overall Top 100 + each org's Top 30),
pulled by pull_prospects.py into the prospects table. Season stats on each
row (gp/k_pct/bb_pct/ops or era) come from the real MLB Stats API, not from
Pipeline itself (which has none) - see pull_prospects.py for how.

Not wired into the nav's "Refresh data" pull - rankings/stats don't change
day to day, so this is refreshed by running pull_prospects.py /
pull_prospect_scouting.py by hand.
"""
import json
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analytics.prospect_performance import top_prospect_performers

from api.cache import ttl_cache
from api.db import get_conn

router = APIRouter(prefix="/api")

_COLUMNS = (
    "p.mlb_id, p.name, p.position, p.is_pitcher, p.team_id, t.name, p.org_rank, p.top100_rank, "
    "p.level, p.eta, p.age, p.height_weight, p.bats, p.throws, p.headshot_url, "
    "p.gp, p.k_pct, p.bb_pct, p.ops, p.era"
)
_FROM = "prospects p JOIN teams t ON t.team_id = p.team_id"


def _row_to_dict(row):
    (mlb_id, name, position, is_pitcher, team_id, team_name, org_rank, top100_rank,
     level, eta, age, height_weight, bats, throws, headshot_url,
     gp, k_pct, bb_pct, ops, era) = row
    return {
        "mlb_id": mlb_id, "name": name, "position": position, "is_pitcher": bool(is_pitcher),
        "team_id": team_id, "team_name": team_name, "org_rank": org_rank, "top100_rank": top100_rank,
        "level": level, "eta": eta, "age": age, "height_weight": height_weight,
        "bats": bats, "throws": throws, "headshot_url": headshot_url,
        "gp": gp, "k_pct": k_pct, "bb_pct": bb_pct, "ops": ops, "era": era,
    }


@ttl_cache()
def _top100(conn, season):
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM {_FROM} WHERE p.season = ? AND p.top100_rank IS NOT NULL "
        "ORDER BY p.top100_rank",
        (season,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


@ttl_cache()
def _team(conn, season, team_id):
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM {_FROM} WHERE p.season = ? AND p.team_id = ? ORDER BY p.org_rank",
        (season, team_id),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


@ttl_cache()
def _teams_with_prospects(conn, season):
    rows = conn.execute(
        "SELECT DISTINCT t.team_id, t.name, t.abbreviation FROM prospects p "
        "JOIN teams t ON t.team_id = p.team_id WHERE p.season = ? ORDER BY t.name",
        (season,),
    ).fetchall()
    return [{"team_id": r[0], "name": r[1], "abbreviation": r[2]} for r in rows]


@ttl_cache()
def _farm_rankings(conn, season):
    """Not an official Pipeline list (Pipeline doesn't publish one) - a
    transparent, self-computed score: sum of (101 - rank) over each org's
    own Top 100 prospects, rewarding both having more blue-chippers and
    having them ranked higher. Ties broken by Top 100 count, then by best
    individual rank."""
    rows = conn.execute(
        "SELECT t.team_id, t.name, t.abbreviation, p.top100_rank, p.name "
        "FROM teams t LEFT JOIN prospects p ON p.team_id = t.team_id AND p.season = ? "
        "ORDER BY t.name",
        (season,),
    ).fetchall()

    by_team = {}
    for team_id, team_name, abbrev, top100_rank, prospect_name in rows:
        entry = by_team.setdefault(team_id, {
            "team_id": team_id, "team_name": team_name, "abbreviation": abbrev,
            "score": 0, "top100_count": 0, "top100_prospects": [],
        })
        if top100_rank is not None:
            entry["score"] += 101 - top100_rank
            entry["top100_count"] += 1
            entry["top100_prospects"].append({"name": prospect_name, "rank": top100_rank})

    teams = list(by_team.values())
    for t in teams:
        t["top100_prospects"].sort(key=lambda p: p["rank"])
        t["best_rank"] = t["top100_prospects"][0]["rank"] if t["top100_prospects"] else None
    teams.sort(key=lambda t: (-t["score"], -t["top100_count"], t["best_rank"] if t["best_rank"] is not None else 999))
    for i, t in enumerate(teams, start=1):
        t["farm_rank"] = i
    return teams


@ttl_cache()
def _scouting(conn, mlb_id):
    row = conn.execute(
        "SELECT report_label, grades_json, overall_grade, report_text, video_url "
        "FROM prospect_scouting WHERE mlb_id = ?",
        (mlb_id,),
    ).fetchone()
    if row is None:
        return None
    report_label, grades_json, overall_grade, report_text, video_url = row
    return {
        "report_label": report_label, "grades": json.loads(grades_json) if grades_json else {},
        "overall_grade": overall_grade, "report_text": report_text, "video_url": video_url,
    }


@ttl_cache()
def _prospect_top_performers(conn, date_str):
    return top_prospect_performers(conn, date_str, n=10)


@router.get("/prospects/top-performers")
def get_prospect_top_performers(date: str = Query(...), conn=Depends(get_conn)):
    return {"date": date, "entries": _prospect_top_performers(conn, date)}


@router.get("/prospects/farm-rankings")
def get_farm_rankings(season: int = Query(2026), conn=Depends(get_conn)):
    return {"season": season, "teams": _farm_rankings(conn, season)}


@router.get("/prospects/{mlb_id}/scouting")
def get_prospect_scouting(mlb_id: int, conn=Depends(get_conn)):
    scouting = _scouting(conn, mlb_id)
    if scouting is None:
        return {"mlb_id": mlb_id, "has_report": False}
    return {"mlb_id": mlb_id, "has_report": True, **scouting}


@router.get("/prospects/top100")
def get_top100(season: int = Query(2026), conn=Depends(get_conn)):
    return {"season": season, "prospects": _top100(conn, season)}


@router.get("/prospects/team")
def get_team_prospects(team_id: int = Query(...), season: int = Query(2026), conn=Depends(get_conn)):
    return {"season": season, "team_id": team_id, "prospects": _team(conn, season, team_id)}


@router.get("/prospects/teams")
def get_prospect_teams(season: int = Query(2026), conn=Depends(get_conn)):
    return {"season": season, "teams": _teams_with_prospects(conn, season)}
