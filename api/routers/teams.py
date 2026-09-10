"""
/api/teams and /api/teams/{team_id}/* - team list with aggregate model scores,
roster, depth chart, and team-level stats.

Aggregate model scores per team are computed by:
1. Building each model's per-pitcher score for the season (one league-wide pass)
2. Joining pitcher_id -> players.team_id
3. Averaging per team (mean across pitchers on the team's active roster)

This deliberately reuses the same _raw_predictions functions and plus-scale
transforms the player page already uses, so the numbers here match what
individual pitcher pages show.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analytics.stuff_model import compute_stuff_plus, overall_stuff_plus
from analytics.tunneling_v3_model import _raw_predictions as _tun_raw
from analytics.location_model import _raw_predictions as _loc_raw, official_location_plus
from analytics.deception_model import _raw_predictions as _dec_raw

from api.cache import ttl_cache
from api.db import get_conn

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Internal helpers: per-pitcher model scores, vectorised across the league
# ---------------------------------------------------------------------------

@ttl_cache()
def _all_stuff_plus(conn, season):
    """Dict: pitcher_id -> overall Stuff+ (usage-weighted across pitch types)."""
    tbl = compute_stuff_plus(conn, season)
    out = {}
    for pid in tbl.index.get_level_values("pitcher_id").unique():
        v = overall_stuff_plus(tbl, pid)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[int(pid)] = float(v)
    return out


@ttl_cache()
def _all_tunneling_plus(conn, season):
    """Dict: pitcher_id -> Tunneling+ score (same scale as player page badge)."""
    raw = _tun_raw(conn, season)
    if raw.empty or raw["pred"].std() == 0:
        return {}
    mean, std = raw["pred"].mean(), raw["pred"].std()
    g = (raw.groupby("pitcher_id")["pred"].mean()
           .reset_index().rename(columns={"pred": "mean_pred"}))
    g["score"] = 100 - 10 * (g["mean_pred"] - mean) / std
    return {int(r.pitcher_id): round(float(r.score), 1) for r in g.itertuples()}


@ttl_cache()
def _all_location_plus(conn, season):
    """Dict: pitcher_id -> Location+ score."""
    raw = _loc_raw(conn, season)
    if raw.empty:
        return {}
    scores = official_location_plus(raw)
    if scores.empty:
        return {}
    # filter min 100 pitches (matches the player page badge threshold)
    scores = scores[scores["total_pitches"] >= 100]
    return {int(r.pitcher_id): round(float(r.score), 1) for r in scores.itertuples()}


@ttl_cache()
def _all_deception_plus(conn, season):
    """Dict: pitcher_id -> Deception+ score.
    Deception+ is REPORT_SEASON-only (2025) - same constraint as the player
    page badge, which always uses REPORT_SEASON regardless of the page's
    season selector."""
    from api.routers.deception_report import REPORT_SEASON
    raw = _dec_raw(conn, REPORT_SEASON)
    if raw.empty or raw["pred"].std() == 0:
        return {}
    mean, std = raw["pred"].mean(), raw["pred"].std()
    g = (raw.groupby("pitcher_id")["pred"].mean()
           .reset_index().rename(columns={"pred": "mean_pred"}))
    g["score"] = 100 - 10 * (g["mean_pred"] - mean) / std
    return {int(r.pitcher_id): round(float(r.score), 1) for r in g.itertuples()}


def _team_model_aggregates(conn, season):
    """
    Returns a dict: team_id -> {stuff_plus, tunneling_plus, location_plus, deception_plus}
    Each is the mean across pitchers on that team who have a qualified score.
    Falls back to None per metric if no qualifying pitchers.
    """
    stuff = _all_stuff_plus(conn, season)
    tun = _all_tunneling_plus(conn, season)
    loc = _all_location_plus(conn, season)
    dec = _all_deception_plus(conn, season)

    # All pitcher_ids with at least one score
    all_pids = set(stuff) | set(tun) | set(loc) | set(dec)
    if not all_pids:
        return {}

    placeholders = ",".join("?" * len(all_pids))
    rows = conn.execute(
        f"SELECT player_id, team_id FROM players WHERE player_id IN ({placeholders})",
        list(all_pids),
    ).fetchall()

    # Accumulate per team
    from collections import defaultdict
    buckets = defaultdict(lambda: {"stuff": [], "tun": [], "loc": [], "dec": []})
    for pid, tid in rows:
        if not tid:
            continue
        team_id = int(tid) if not isinstance(tid, int) else tid
        if pid in stuff:
            buckets[team_id]["stuff"].append(stuff[pid])
        if pid in tun:
            buckets[team_id]["tun"].append(tun[pid])
        if pid in loc:
            buckets[team_id]["loc"].append(loc[pid])
        if pid in dec:
            buckets[team_id]["dec"].append(dec[pid])

    result = {}
    for tid, b in buckets.items():
        result[tid] = {
            "stuff_plus":      round(float(np.mean(b["stuff"])), 0) if b["stuff"] else None,
            "tunneling_plus":  round(float(np.mean(b["tun"])), 1) if b["tun"] else None,
            "location_plus":   round(float(np.mean(b["loc"])), 1) if b["loc"] else None,
            "deception_plus":  round(float(np.mean(b["dec"])), 1) if b["dec"] else None,
        }
    return result


# ---------------------------------------------------------------------------
# /api/teams
# ---------------------------------------------------------------------------

@router.get("/teams")
def list_teams(season: int = Query(2026), conn=Depends(get_conn)):
    """All 30 MLB teams with aggregate Stuff+/Tunneling+/Location+/Deception+."""
    teams = conn.execute(
        "SELECT team_id, name, abbreviation, league, division FROM teams ORDER BY division, name"
    ).fetchall()
    if not teams:
        return []

    try:
        agg = _team_model_aggregates(conn, season)
    except Exception:
        agg = {}

    result = []
    for team_id, name, abbreviation, league, division in teams:
        scores = agg.get(int(team_id), {})
        result.append({
            "team_id": team_id,
            "name": name,
            "abbreviation": abbreviation,
            "league": league,
            "division": division,
            "stuff_plus": scores.get("stuff_plus"),
            "tunneling_plus": scores.get("tunneling_plus"),
            "location_plus": scores.get("location_plus"),
            "deception_plus": scores.get("deception_plus"),
        })
    return result


# ---------------------------------------------------------------------------
# /api/teams/{team_id}/roster
# ---------------------------------------------------------------------------

@router.get("/teams/{team_id}/roster")
def get_team_roster(
    team_id: int,
    roster_type: str = Query("active", pattern="^(active|40Man|fullRoster|depthChart)$"),
    conn=Depends(get_conn),
):
    """Active (or 40-man) roster from the Part A cache (rosters table).
    Returns an empty list if the cache hasn't been populated yet — run
    pulls/refresh_mlb_stats.py to populate it."""
    rows = conn.execute(
        """SELECT player_id, player_name, position, jersey_number, status_code, pulled_at
           FROM rosters WHERE team_id = ? AND roster_type = ?
           ORDER BY position, player_name""",
        (team_id, roster_type),
    ).fetchall()

    if not rows:
        # Fall back to MLB Stats API live call rather than returning empty
        try:
            from data.mlb_stats_api import team_roster
            import os
            db = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "mlb_data.db")
            rows_dict = team_roster(team_id, roster_type, db_path=db)
            return rows_dict
        except Exception:
            return []

    player_ids = [r[0] for r in rows if r[0]]
    pitcher_stats = {}
    if player_ids:
        placeholders = ",".join("?" * len(player_ids))
        stat_rows = conn.execute(
            f"""
            SELECT pitcher_id,
                   COUNT(DISTINCT game_id) AS g,
                   SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
                   SUM(CASE WHEN events IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS bb,
                   SUM(CASE WHEN events IN ('field_out','force_out','grounded_into_double_play',
                       'double_play','strikeout','strikeout_double_play','sac_fly','sac_bunt',
                       'fielders_choice_out','caught_stealing_2b','caught_stealing_3b',
                       'caught_stealing_home','pickoff_caught_stealing_2b','pickoff_caught_stealing_3b',
                       'other_out','triple_play') THEN 1 ELSE 0 END) AS outs,
                   SUM(CASE WHEN post_bat_score > bat_score THEN post_bat_score - bat_score ELSE 0 END) AS ra
            FROM pitches
            WHERE pitcher_id IN ({placeholders}) AND season = 2026
            GROUP BY pitcher_id
            """,
            player_ids,
        ).fetchall()
        for sr in stat_rows:
            pid, g, k, bb, outs, ra = sr
            ip = round(outs / 3, 1) if outs else None
            era = round(ra / (outs / 3) * 9, 2) if outs else None
            pitcher_stats[pid] = {"g": g, "ip": ip, "era": era, "k": k, "bb": bb}

    return [
        {
            "player_id": r[0],
            "player_name": r[1],
            "position": r[2],
            "jersey_number": r[3],
            "status_code": r[4],
            "pulled_at": r[5],
            "season_stats": pitcher_stats.get(r[0]),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# /api/teams/{team_id}/depth_chart
# ---------------------------------------------------------------------------

@router.get("/teams/{team_id}/depth_chart")
def get_team_depth_chart(team_id: int, conn=Depends(get_conn)):
    """depthChart roster from cache. Falls back to live API call if cache is
    empty."""
    rows = conn.execute(
        """SELECT player_id, player_name, position, jersey_number, status_code, pulled_at
           FROM rosters WHERE team_id = ? AND roster_type = 'depthChart'
           ORDER BY position, player_name""",
        (team_id,),
    ).fetchall()

    if not rows:
        try:
            from data.mlb_stats_api import team_roster
            import os
            db = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "mlb_data.db")
            rows_dict = team_roster(team_id, "depthChart", db_path=db)
            return rows_dict
        except Exception:
            return []

    return [
        {
            "player_id": r[0],
            "player_name": r[1],
            "position": r[2],
            "jersey_number": r[3],
            "status_code": r[4],
            "pulled_at": r[5],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# /api/teams/{team_id}/stats
# ---------------------------------------------------------------------------

@router.get("/teams/{team_id}/stats")
def get_team_stats(team_id: int, season: int = Query(2026), conn=Depends(get_conn)):
    """
    Team-level pitching and batting stats derived from existing Statcast data
    plus the team_stats_snapshot table (OPS/ERA/FIP from the MLB Stats API
    pull — pulled by pulls/pull_team_stats.py).
    """
    team_row = conn.execute(
        "SELECT name, abbreviation, league, division FROM teams WHERE team_id = ?",
        (team_id,),
    ).fetchone()
    if not team_row:
        raise HTTPException(status_code=404, detail="Team not found")
    name, abbreviation, league, division = team_row

    # Official team stats from snapshot table
    snap = conn.execute(
        "SELECT ops, era, bullpen_era, fip, ops_30d, era_30d, fip_30d, season, pulled_at "
        "FROM team_stats_snapshot WHERE team_id = ?",
        (team_id,),
    ).fetchone()

    # Statcast-derived: pitching aggregates for this team's pitchers
    pitching = conn.execute(
        """
        SELECT
            COUNT(*) AS total_pitches,
            COUNT(DISTINCT pitcher_id) AS n_pitchers,
            AVG(release_speed) AS avg_velo,
            AVG(release_spin_rate) AS avg_spin,
            CAST(SUM(CASE WHEN description IN ('swinging_strike','swinging_strike_blocked') THEN 1 ELSE 0 END) AS REAL)
              / NULLIF(SUM(CASE WHEN description LIKE '%swing%' OR description LIKE '%hit%'
                               OR description = 'foul' OR description = 'foul_tip'
                               OR description = 'missed_bunt' THEN 1 ELSE 0 END), 0) AS whiff_rate
        FROM pitches p
        JOIN games g ON p.game_id = g.game_id
        WHERE CAST(g.home_team AS INTEGER) = ? AND g.inning_topbot IS NULL
           OR p.game_id IN (
               SELECT game_id FROM games
               WHERE (CAST(home_team AS INTEGER) = ? AND season = ?)
                  OR (CAST(away_team AS INTEGER) = ? AND season = ?)
           )
        """,
        (team_id, team_id, season, team_id, season),
    ).fetchone()

    # Simpler approach — get pitchers by team from players table
    pitching_stats = conn.execute(
        """
        SELECT
            COUNT(*) AS total_pitches,
            COUNT(DISTINCT p.pitcher_id) AS n_pitchers,
            ROUND(AVG(p.release_speed), 1) AS avg_velo,
            ROUND(AVG(p.release_spin_rate), 0) AS avg_spin,
            ROUND(CAST(SUM(CASE WHEN p.description IN ('swinging_strike','swinging_strike_blocked')
                              THEN 1 ELSE 0 END) AS REAL)
              / NULLIF(SUM(CASE WHEN p.description LIKE '%swing%'
                               OR p.description = 'foul' OR p.description = 'foul_tip'
                               OR p.description = 'hit_into_play' THEN 1 ELSE 0 END), 0), 3) AS whiff_rate
        FROM pitches p
        JOIN players pl ON p.pitcher_id = pl.player_id
        WHERE pl.team_id = ? AND p.season = ?
        """,
        (str(team_id), season),
    ).fetchone()

    result = {
        "team_id": team_id,
        "name": name,
        "abbreviation": abbreviation,
        "league": league,
        "division": division,
        "season": season,
        "official": None,
        "statcast_pitching": None,
    }

    if snap:
        result["official"] = {
            "ops": snap[0],
            "era": snap[1],
            "bullpen_era": snap[2],
            "fip": snap[3],
            "ops_30d": snap[4],
            "era_30d": snap[5],
            "fip_30d": snap[6],
            "snapshot_season": snap[7],
            "pulled_at": snap[8],
        }

    if pitching_stats and pitching_stats[0]:
        result["statcast_pitching"] = {
            "total_pitches": pitching_stats[0],
            "n_pitchers": pitching_stats[1],
            "avg_velo": pitching_stats[2],
            "avg_spin": pitching_stats[3],
            "whiff_rate": pitching_stats[4],
        }

    return result
