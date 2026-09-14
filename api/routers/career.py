"""
Season-by-season career stats for the Career page.
GET /api/players/{player_id}/career?role=pitcher|batter

Team-change handling: one row per (season, team). Players who changed teams
mid-season get individual team rows, matching Baseball Reference's convention.
Partial history: only seasons where the player appears in the data.

Model scores (Stuff+, Tunneling+, Location+) are pulled from the same report
tables used by the player overview page — cheap lookups against cached data.
Deception+ is included for 2025 only (the report's current fixed season).
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

from analytics.player_page import (
    _add_flags, _add_arm_side_hb, OUTS_PER_EVENT, innings_pitched,
    compute_xfip, league_hr_per_fb,
)
from analytics.splits import STRIKEOUT_EVENTS, WALK_EVENTS, HBP_EVENTS, IN_ZONE_CODES
from api.cache import ttl_cache
from api.db import get_conn
from api.routers.stuff_model_report import _raw_predictions_table as _stuff_table
from api.routers.tunneling_report import _v3_raw_predictions_table as _tun_table
from api.routers.location_report import _raw_predictions_table as _loc_table
from api.routers.deception_report import (
    _raw_predictions_table as _dec_table,
    REPORT_SEASON as DEC_SEASON,
)

router = APIRouter(prefix="/api")


def _cf(v, decimals=2):
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _team_abbr(conn, team_id):
    if not team_id:
        return "—"
    r = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (team_id,)).fetchone()
    return r[0] if r else str(team_id)


def _pitcher_score_row(stuff_tbl, tun_tbl, loc_tbl, dec_tbl, pitcher_id, season):
    """Pull season-level model scores from cached report tables."""
    def _lookup(tbl, pid_col="pitcher_id"):
        if tbl is None or tbl.empty:
            return None
        if pid_col in tbl.index.names:
            try:
                return float(tbl.xs(pitcher_id, level=pid_col)["stuff_plus"].iloc[0]) if not tbl.xs(pitcher_id, level=pid_col).empty else None
            except Exception:
                return None
        if pid_col in tbl.columns:
            rows = tbl[tbl[pid_col] == pitcher_id]
            return float(rows.iloc[0]["stuff_plus"]) if not rows.empty else None
        return None

    def _generic(tbl, score_col, pid_col="pitcher_id"):
        if tbl is None or tbl.empty:
            return None
        try:
            if isinstance(tbl, pd.DataFrame):
                rows = tbl[tbl[pid_col] == pitcher_id] if pid_col in tbl.columns else pd.DataFrame()
                if rows.empty and pid_col in tbl.index.names:
                    try:
                        rows = tbl.xs(pitcher_id, level=pid_col)
                    except KeyError:
                        return None
                if rows.empty:
                    return None
                return _cf(rows[score_col].mean(), 1)
        except Exception:
            return None

    stuff = _generic(stuff_tbl, "stuff_plus")
    tun   = _generic(tun_tbl,   "tunneling_plus")
    loc   = _generic(loc_tbl,   "location_plus")
    dec   = _generic(dec_tbl,   "deception_plus") if season == DEC_SEASON else None
    return {"stuff_plus": stuff, "tunneling_plus": tun, "location_plus": loc, "deception_plus": dec}


@router.get("/players/{player_id}/career")
def get_player_career(
    player_id: int,
    role: str = Query("pitcher"),
    conn=Depends(get_conn),
):
    if role == "pitcher":
        return _pitcher_career(conn, player_id)
    else:
        return _batter_career(conn, player_id)


@ttl_cache()
def _pitcher_career(conn, player_id):
    df = pd.read_sql("""
        SELECT season, game_id,
               p_throws, pitch_type, pfx_z,
               release_speed, release_spin_rate,
               description, zone, events, bb_type
        FROM pitches
        WHERE pitcher_id = ?
        ORDER BY season, game_id
    """, conn, params=[player_id])

    if df.empty:
        return {"rows": [], "columns": [], "role": "pitcher"}

    df = _add_flags(df)

    # Current team from players table (no per-season team history in DB)
    team_row = conn.execute(
        "SELECT p.team_id FROM players p WHERE p.player_id = ?", (player_id,)
    ).fetchone()
    current_team_id = team_row[0] if team_row else None

    # Pre-load model tables once per unique season (expensive; do outside the row loop)
    unique_seasons = sorted(df["season"].unique())
    model_tbls = {}
    for s in unique_seasons:
        try: s_tbl = _stuff_table(conn, int(s))
        except Exception: s_tbl = None
        try: t_tbl = _tun_table(conn, int(s))
        except Exception: t_tbl = None
        try: l_tbl = _loc_table(conn, int(s))
        except Exception: l_tbl = None
        try: d_tbl = _dec_table(conn) if s == DEC_SEASON else None
        except Exception: d_tbl = None
        model_tbls[int(s)] = (s_tbl, t_tbl, l_tbl, d_tbl)

    rows = []
    for season, grp in df.groupby("season", sort=True):
        team_id = current_team_id
        pa = grp[grp["events"].notna()]
        ip_str, _ = innings_pitched(pa)
        k = int(pa["events"].isin(STRIKEOUT_EVENTS).sum())
        bb = int(pa["events"].isin(WALK_EVENTS).sum())
        hr = int((pa["events"] == "home_run").sum())
        n_pa = len(pa)
        swings = int(grp["is_swing"].sum())
        whiffs = int(grp["is_whiff"].sum())
        in_z = int(grp["zone"].isin(IN_ZONE_CODES).sum())
        outs = pa["events"].map(OUTS_PER_EVENT).fillna(0).sum()
        ip_dec = outs / 3
        xfip = None
        if ip_dec > 0:
            try:
                lg_hr_fb = league_hr_per_fb(conn, int(season))
                xfip = _cf(compute_xfip(pa, lg_hr_fb), 2)
            except Exception:
                pass

        row = {
            "season":    int(season),
            "team_id":   int(team_id) if team_id else None,
            "ip":        ip_str,
            "gs":        grp["game_id"].nunique(),
            "n_pitches": len(grp),
            "k":         k,
            "bb":        bb,
            "hr":        hr,
            "xfip":      xfip,
            "whiff_pct": _cf(whiffs / swings * 100, 1) if swings else None,
            "k_pct":     _cf(k / n_pa * 100, 1) if n_pa else None,
            "bb_pct":    _cf(bb / n_pa * 100, 1) if n_pa else None,
            "zone_pct":  _cf(in_z / len(grp) * 100, 1),
        }

        s_tbl, t_tbl, l_tbl, d_tbl = model_tbls.get(int(season), (None, None, None, None))
        scores = _pitcher_score_row(s_tbl, t_tbl, l_tbl, d_tbl, player_id, int(season))
        row.update(scores)
        rows.append(row)

    # Add team abbreviations
    team_map = {}
    for r in rows:
        tid = r.get("team_id")
        if tid and tid not in team_map:
            tm = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (tid,)).fetchone()
            team_map[tid] = tm[0] if tm else str(tid)
    for r in rows:
        r["team"] = team_map.get(r.get("team_id"), "—")

    columns = [
        {"key": "season",       "label": "Season"},
        {"key": "team",         "label": "Team"},
        {"key": "gs",           "label": "G"},
        {"key": "ip",           "label": "IP"},
        {"key": "k",            "label": "K"},
        {"key": "bb",           "label": "BB"},
        {"key": "hr",           "label": "HR"},
        {"key": "xfip",         "label": "xFIP"},
        {"key": "k_pct",        "label": "K%"},
        {"key": "bb_pct",       "label": "BB%"},
        {"key": "whiff_pct",    "label": "Whiff%"},
        {"key": "stuff_plus",   "label": "Stuff+"},
        {"key": "tunneling_plus","label": "Tun+"},
        {"key": "location_plus","label": "Loc+"},
        {"key": "deception_plus","label": "Dec+"},
    ]
    return {"rows": rows, "columns": columns, "role": "pitcher"}


@ttl_cache()
def _batter_career(conn, player_id):
    from analytics.splits import _pa_outcome_metrics
    df = pd.read_sql("""
        SELECT season, game_id,
               launch_speed, launch_angle,
               estimated_woba_using_speedangle,
               woba_value, woba_denom,
               description, zone, events
        FROM pitches
        WHERE batter_id = ?
        ORDER BY season, game_id
    """, conn, params=[player_id])

    if df.empty:
        return {"rows": [], "columns": [], "role": "batter"}

    df = _add_flags(df)

    team_row = conn.execute(
        "SELECT p.team_id FROM players p WHERE p.player_id = ?", (player_id,)
    ).fetchone()
    current_team_id = team_row[0] if team_row else None

    rows = []
    for season, grp in df.groupby("season", sort=True):
        team_id = current_team_id
        pa = grp[grp["events"].notna()]
        k = int(pa["events"].isin(STRIKEOUT_EVENTS).sum())
        bb = int(pa["events"].isin(WALK_EVENTS).sum())
        hr = int((pa["events"] == "home_run").sum())
        n_pa = len(pa)
        ev = grp["launch_speed"].dropna()
        n_ev = len(ev)
        hard = int((ev >= 95).sum())
        xwoba_vals = grp["estimated_woba_using_speedangle"].dropna()
        swings = int(grp["is_swing"].sum())
        whiffs = int(grp["is_whiff"].sum())
        outcomes = _pa_outcome_metrics(pa)

        # wRC+ from player_advanced_stats if available (avg/obp/slg not in this table)
        adv = conn.execute(
            "SELECT wrc_plus FROM player_advanced_stats WHERE player_id = ? AND season = ?",
            (player_id, int(season)),
        ).fetchone()

        row = {
            "season":       int(season),
            "team_id":      int(team_id) if team_id else None,
            "gp":           grp["game_id"].nunique(),
            "pa":           n_pa,
            "hr":           hr,
            "k":            k,
            "bb":           bb,
            "ops":          _cf(outcomes["OPS"], 3),
            "woba":         _cf(outcomes["wOBA"], 3),
            "xwoba":        _cf(xwoba_vals.mean(), 3) if len(xwoba_vals) else None,
            "avg_ev":       _cf(ev.mean(), 1) if n_ev else None,
            "hard_hit_pct": _cf(hard / n_ev * 100, 1) if n_ev else None,
            "k_pct":        _cf(k / n_pa * 100, 1) if n_pa else None,
            "bb_pct":       _cf(bb / n_pa * 100, 1) if n_pa else None,
            "whiff_pct":    _cf(whiffs / swings * 100, 1) if swings else None,
            "wrc_plus":     adv[0] if adv else None,
        }
        rows.append(row)

    team_map = {}
    for r in rows:
        tid = r.get("team_id")
        if tid and tid not in team_map:
            tm = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (tid,)).fetchone()
            team_map[tid] = tm[0] if tm else str(tid)
    for r in rows:
        r["team"] = team_map.get(r.get("team_id"), "—")

    columns = [
        {"key": "season",       "label": "Season"},
        {"key": "team",         "label": "Team"},
        {"key": "gp",           "label": "G"},
        {"key": "pa",           "label": "PA"},
        {"key": "hr",           "label": "HR"},
        {"key": "ops",          "label": "OPS"},
        {"key": "woba",         "label": "wOBA"},
        {"key": "wrc_plus",     "label": "wRC+"},
        {"key": "xwoba",        "label": "xwOBA"},
        {"key": "avg_ev",       "label": "Avg EV"},
        {"key": "hard_hit_pct", "label": "Hard-Hit%"},
        {"key": "k_pct",        "label": "K%"},
        {"key": "bb_pct",       "label": "BB%"},
        {"key": "whiff_pct",    "label": "Whiff%"},
    ]
    return {"rows": rows, "columns": columns, "role": "batter"}
