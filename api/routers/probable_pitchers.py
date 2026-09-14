"""
/api/probable-pitchers?date=YYYY-MM-DD

Returns today's (or a given date's) games from the schedule_games cache
(populated by data/mlb_stats_api.py), with probable starters and their
model scores joined from the live model scoring path.

Data freshness is bounded by when pulls/refresh_mlb_stats.py was last run
(daily cadence). The pulled_at timestamp is included in the response so the
frontend can display it visibly.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analytics.stuff_model import compute_stuff_plus, overall_stuff_plus
from analytics.tunneling_v3_model import _raw_predictions as _tun_raw
from analytics.location_model import _raw_predictions as _loc_raw, official_location_plus
from analytics.deception_model import _raw_predictions as _dec_raw

from api.cache import ttl_cache
from api.db import get_conn
from api.routers.deception_report import REPORT_SEASON as DECEPTION_REPORT_SEASON

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Reuse the same per-pitcher score dictionaries as the teams router
# ---------------------------------------------------------------------------

@ttl_cache()
def _stuff_dict(conn, season):
    tbl = compute_stuff_plus(conn, season)
    out = {}
    for pid in tbl.index.get_level_values("pitcher_id").unique():
        v = overall_stuff_plus(tbl, pid)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[int(pid)] = round(float(v), 0)
    return out


@ttl_cache()
def _tunneling_dict(conn, season):
    raw = _tun_raw(conn, season)
    if raw.empty or raw["pred"].std() == 0:
        return {}
    mean, std = raw["pred"].mean(), raw["pred"].std()
    g = raw.groupby("pitcher_id")["pred"].mean().reset_index()
    g["score"] = 100 - 10 * (g["pred"] - mean) / std
    return {int(r.pitcher_id): round(float(r.score), 1) for r in g.itertuples()}


@ttl_cache()
def _location_dict(conn, season):
    raw = _loc_raw(conn, season)
    if raw.empty:
        return {}
    scores = official_location_plus(raw)
    if scores.empty:
        return {}
    scores = scores[scores["total_pitches"] >= 100]
    return {int(r.pitcher_id): round(float(r.score), 1) for r in scores.itertuples()}


@ttl_cache()
def _deception_dict(conn):
    raw = _dec_raw(conn, DECEPTION_REPORT_SEASON)
    if raw.empty or raw["pred"].std() == 0:
        return {}
    mean, std = raw["pred"].mean(), raw["pred"].std()
    g = raw.groupby("pitcher_id")["pred"].mean().reset_index()
    g["score"] = 100 - 10 * (g["pred"] - mean) / std
    return {int(r.pitcher_id): round(float(r.score), 1) for r in g.itertuples()}


def _pitcher_scores(conn, season, pitcher_id):
    if pitcher_id is None:
        return None
    try:
        stuff = _stuff_dict(conn, season).get(pitcher_id)
    except Exception:
        stuff = None
    try:
        tun = _tunneling_dict(conn, season).get(pitcher_id)
    except Exception:
        tun = None
    try:
        loc = _location_dict(conn, season).get(pitcher_id)
    except Exception:
        loc = None
    try:
        dec = _deception_dict(conn).get(pitcher_id)
    except Exception:
        dec = None
    return {"stuff_plus": stuff, "tunneling_plus": tun, "location_plus": loc, "deception_plus": dec}


@ttl_cache()
def _recent_starts_cached(conn, season, pitcher_id, n=5):
    """Cached wrapper — GROUP BY on pitches per pitcher, not free."""
    return _recent_starts_inner(conn, season, pitcher_id, n)


def _recent_starts_inner(conn, season, pitcher_id, n=5):
    """Aggregated stats across last n starts for this pitcher."""
    rows = conn.execute(
        """
        SELECT g.game_id,
               g.game_date,
               COUNT(*) AS n_pitches,
               SUM(CASE WHEN p.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
               SUM(CASE WHEN p.events IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS bb,
               SUM(CASE WHEN p.events = 'home_run' THEN 1 ELSE 0 END) AS hr,
               SUM(CASE WHEN p.events IN ('field_out','force_out','grounded_into_double_play',
                   'double_play','strikeout','strikeout_double_play','sac_fly','sac_bunt',
                   'fielders_choice_out','caught_stealing_2b','caught_stealing_3b',
                   'caught_stealing_home','pickoff_caught_stealing_2b','pickoff_caught_stealing_3b',
                   'other_out','triple_play') THEN 1 ELSE 0 END) AS outs,
               SUM(CASE WHEN p.post_bat_score > p.bat_score
                        THEN p.post_bat_score - p.bat_score ELSE 0 END) AS runs_allowed
        FROM pitches p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.pitcher_id = ? AND p.season = ?
        GROUP BY g.game_id
        ORDER BY g.game_date DESC
        LIMIT ?
        """,
        (pitcher_id, season, n),
    ).fetchall()
    if not rows:
        return None
    total_k = sum(r[3] for r in rows)
    total_bb = sum(r[4] for r in rows)
    total_hr = sum(r[5] for r in rows)
    total_outs = sum(r[6] for r in rows)
    total_ra = sum(r[7] for r in rows)
    ip = total_outs / 3.0
    era = round(total_ra / ip * 9, 2) if ip > 0 else None
    return {
        "games": len(rows),
        "ip": round(ip, 1),
        "era": era,
        "k": total_k,
        "bb": total_bb,
        "hr": total_hr,
    }


def _team_name(conn, team_id):
    row = conn.execute(
        "SELECT name, abbreviation FROM teams WHERE team_id = ?", (team_id,)
    ).fetchone()
    return {"team_id": team_id, "name": row[0], "abbreviation": row[1]} if row else {"team_id": team_id, "name": str(team_id), "abbreviation": str(team_id)}


def _player_name(conn, player_id):
    row = conn.execute("SELECT name FROM players WHERE player_id = ?", (player_id,)).fetchone()
    return row[0] if row else None


@router.get("/probable-pitchers")
def get_probable_pitchers(
    date: str = Query(None, description="YYYY-MM-DD, defaults to today"),
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    """
    Games for a given date with probable starters and their model scores.

    If the schedule_games cache is empty for the requested date, falls back
    to a live MLB Stats API call (no caching to DB in the fallback path —
    the live data is still returned, just not stored).
    """
    game_date = date or str(__import__("datetime").date.today())

    rows = conn.execute(
        """SELECT game_pk, game_date, home_team_id, away_team_id,
                  home_probable_pitcher_id, away_probable_pitcher_id,
                  venue_name, game_status, pulled_at, venue_id,
                  COALESCE(game_time_et, '') as game_time_et
           FROM schedule_games WHERE game_date = ?
           ORDER BY game_pk""",
        (game_date,),
    ).fetchall()

    import os
    _db = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "mlb_data.db")

    # Fetch from live API if: no cached rows at all, OR rows exist but all
    # have empty game_time_et (means they were stored before the strftime fix)
    needs_fetch = not rows or all(not r[10] for r in rows)
    if needs_fetch:
        try:
            from data.mlb_stats_api import schedule as mlb_schedule
            mlb_schedule(game_date, game_date, db_path=_db)
            rows = conn.execute(
                """SELECT game_pk, game_date, home_team_id, away_team_id,
                          home_probable_pitcher_id, away_probable_pitcher_id,
                          venue_name, game_status, pulled_at, venue_id,
                          COALESCE(game_time_et, '') as game_time_et
                   FROM schedule_games WHERE game_date = ?
                   ORDER BY game_pk""",
                (game_date,),
            ).fetchall()
        except Exception:
            pass

    pulled_at = rows[0][8] if rows else None
    games = []
    for r in rows:
        game_pk, gdate, home_id, away_id, home_pp, away_pp, venue, status, _, venue_id, game_time_et = r
        home_team = _team_name(conn, home_id)
        away_team = _team_name(conn, away_id)

        def build_pitcher(pid):
            if not pid:
                return None
            name = _player_name(conn, pid)
            scores = _pitcher_scores(conn, season, pid)
            recent = _recent_starts_cached(conn, season, pid, 5)
            return {
                "player_id": pid,
                "name": name,
                "scores": scores,
                "recent_starts": recent,
            }

        games.append({
            "game_pk": game_pk,
            "game_date": gdate,
            "venue": venue,
            "venue_id": venue_id,
            "game_time_et": game_time_et,
            "status": status,
            "home_team": home_team,
            "away_team": away_team,
            "home_probable": build_pitcher(home_pp),
            "away_probable": build_pitcher(away_pp),
        })

    return {"date": game_date, "pulled_at": pulled_at, "games": games}
