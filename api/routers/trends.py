"""
Per-outing/per-game trend data for the Trend Viewer page.
GET /api/players/{player_id}/trends?role=pitcher|batter&season=2026|all&pitch_type=FF
Physical metrics for pitchers use fastball-only (FF/SI/FA) via fb_ keys.
Pitch Type category uses avg_ keys computed on the pitch_type-filtered dataset.
"""
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

from analytics.player_page import _add_flags, _add_arm_side_hb
from analytics.splits import (
    IN_ZONE_CODES, OUT_ZONE_CODES, STRIKEOUT_EVENTS, WALK_EVENTS,
    SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS,
)
from api.cache import ttl_cache
from api.db import get_conn

router = APIRouter(prefix="/api")

FASTBALL_TYPES = frozenset({"FF", "SI", "FA"})

_PITCHER_SELECT = """
    SELECT game_id, game_date, season,
           p_throws, pitch_type,
           release_speed, release_spin_rate, release_extension,
           pfx_x, pfx_z, zone,
           description, events, bb_type
    FROM pitches
    WHERE pitcher_id = ?
"""

_BATTER_SELECT = """
    SELECT game_id, game_date, season,
           launch_speed, launch_angle,
           estimated_woba_using_speedangle,
           description, zone, events, bb_type
    FROM pitches
    WHERE batter_id = ?
"""

PITCHER_METRICS = [
    # Physical — fastball-only (fb_ prefix)
    {"key": "fb_avg_velo", "label": "FB Velocity",    "category": "Physical",   "unit": "mph", "decimals": 1},
    {"key": "fb_max_velo", "label": "FB Max Velo",    "category": "Physical",   "unit": "mph", "decimals": 1},
    {"key": "fb_avg_spin", "label": "FB Spin",        "category": "Physical",   "unit": "rpm", "decimals": 0},
    {"key": "fb_avg_ivb",  "label": "FB IVB",         "category": "Physical",   "unit": "in",  "decimals": 1},
    {"key": "fb_avg_hb",   "label": "FB Arm-Side HB", "category": "Physical",   "unit": "in",  "decimals": 1},
    {"key": "fb_avg_ext",  "label": "FB Extension",   "category": "Physical",   "unit": "ft",  "decimals": 1},
    # Results — all pitches
    {"key": "k_pct",       "label": "K%",    "category": "Results", "unit": "%", "decimals": 1},
    {"key": "bb_pct",      "label": "BB%",   "category": "Results", "unit": "%", "decimals": 1},
    {"key": "zone_pct",    "label": "Zone%", "category": "Results", "unit": "%", "decimals": 1},
    {"key": "csw_pct",     "label": "CSW%",  "category": "Results", "unit": "%", "decimals": 1},
    {"key": "gb_pct",      "label": "GB%",   "category": "Results", "unit": "%", "decimals": 1},
    # Process — all pitches
    {"key": "whiff_pct",        "label": "Whiff%",     "category": "Process", "unit": "%", "decimals": 1},
    {"key": "chase_pct",        "label": "Chase%",     "category": "Process", "unit": "%", "decimals": 1},
    {"key": "zone_contact_pct", "label": "Z-Contact%", "category": "Process", "unit": "%", "decimals": 1},
    # Pitch Type — per selected pitch type (avg_ keys, computed on filtered dataset)
    {"key": "avg_velo",  "label": "Avg Velocity",    "category": "Pitch Type", "unit": "mph", "decimals": 1},
    {"key": "avg_spin",  "label": "Avg Spin",        "category": "Pitch Type", "unit": "rpm", "decimals": 0},
    {"key": "avg_ivb",   "label": "IVB",             "category": "Pitch Type", "unit": "in",  "decimals": 1},
    {"key": "avg_hb",    "label": "Arm-Side HB",     "category": "Pitch Type", "unit": "in",  "decimals": 1},
    {"key": "avg_ext",   "label": "Extension",       "category": "Pitch Type", "unit": "ft",  "decimals": 1},
    {"key": "pt_whiff_pct", "label": "Whiff%", "category": "Pitch Type", "unit": "%", "decimals": 1},
    {"key": "pt_zone_pct",  "label": "Zone%",  "category": "Pitch Type", "unit": "%", "decimals": 1},
    {"key": "pt_csw_pct",   "label": "CSW%",   "category": "Pitch Type", "unit": "%", "decimals": 1},
    # Misc
    {"key": "n_pitches", "label": "Pitch Count", "category": "Misc", "unit": "", "decimals": 0},
]

BATTER_METRICS = [
    # Physical
    {"key": "avg_ev",         "label": "Avg Exit Velo",    "category": "Physical", "unit": "mph", "decimals": 1},
    {"key": "max_ev",         "label": "Max Exit Velo",    "category": "Physical", "unit": "mph", "decimals": 1},
    {"key": "ev_90th",        "label": "90th Pct EV",      "category": "Physical", "unit": "mph", "decimals": 1},
    {"key": "avg_la",         "label": "Avg Launch Angle", "category": "Physical", "unit": "°",   "decimals": 1},
    {"key": "sweet_spot_pct", "label": "Sweet Spot%",      "category": "Physical", "unit": "%",   "decimals": 1},
    # Results
    {"key": "k_pct",        "label": "K%",        "category": "Results", "unit": "%", "decimals": 1},
    {"key": "bb_pct",       "label": "BB%",       "category": "Results", "unit": "%", "decimals": 1},
    {"key": "hard_hit_pct", "label": "Hard Hit%", "category": "Results", "unit": "%", "decimals": 1},
    {"key": "barrel_pct",   "label": "Barrel%",   "category": "Results", "unit": "%", "decimals": 1},
    {"key": "gb_pct",       "label": "GB%",       "category": "Results", "unit": "%", "decimals": 1},
    # Process
    {"key": "xwoba",             "label": "xwOBA",      "category": "Process", "unit": "",  "decimals": 3},
    {"key": "whiff_pct",         "label": "Whiff%",     "category": "Process", "unit": "%", "decimals": 1},
    {"key": "chase_pct",         "label": "Chase%",     "category": "Process", "unit": "%", "decimals": 1},
    {"key": "zone_contact_pct",  "label": "Z-Contact%", "category": "Process", "unit": "%", "decimals": 1},
    {"key": "swing_pct",         "label": "Swing%",     "category": "Process", "unit": "%", "decimals": 1},
    # Misc
    {"key": "n_pitches", "label": "Pitches Seen", "category": "Misc", "unit": "", "decimals": 0},
]


def _cf(v, decimals=3):
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _pitcher_rows(conn, player_id, seasons, pitch_type=None):
    ph = ",".join("?" * len(seasons))
    extra = f" AND pitch_type = ?" if pitch_type else ""
    params = [player_id] + list(seasons) + ([pitch_type] if pitch_type else [])
    df = pd.read_sql(
        f"{_PITCHER_SELECT} AND season IN ({ph}){extra} ORDER BY game_date, game_id",
        conn, params=params,
    )
    if df.empty:
        return []
    df = _add_flags(df)
    df = _add_arm_side_hb(df)
    rows = []
    for (game_id, game_date, season), grp in df.groupby(
        ["game_id", "game_date", "season"], sort=False
    ):
        pa = grp[grp["events"].notna()]
        n = len(grp)
        swings = int(grp["is_swing"].sum())
        whiffs = int(grp["is_whiff"].sum())
        in_zone = grp["zone"].isin(IN_ZONE_CODES)
        out_zone = grp["zone"].isin(OUT_ZONE_CODES)
        in_z = int(in_zone.sum())
        k = int(pa["events"].isin(STRIKEOUT_EVENTS).sum())
        bb = int(pa["events"].isin(WALK_EVENTS).sum())
        n_pa = len(pa)
        called_s = int((grp["description"] == "called_strike").sum())
        csw = called_s + whiffs
        out_z_count = int(out_zone.sum())
        chase_sw = int((grp["is_swing"] & out_zone).sum())
        zone_sw = int((grp["is_swing"] & in_zone).sum())
        zone_con = int((grp["is_swing"] & in_zone & ~grp["is_whiff"]).sum())
        bip = grp[grp["bb_type"].notna()]
        gb = int((bip["bb_type"] == "ground_ball").sum())

        # Fastball subset for physical metrics (or full set if pitch_type filter active)
        fb = grp if pitch_type else grp[grp["pitch_type"].isin(FASTBALL_TYPES)]
        hb_col = "hb" if "hb" in fb.columns else None

        # Pitch-type-filtered metrics (avg_ prefix, same keys used for Pitch Type category)
        pt_swings = int(grp["is_swing"].sum())
        pt_whiffs = int(grp["is_whiff"].sum())
        pt_in_z = int(in_zone.sum())
        pt_called = int((grp["description"] == "called_strike").sum())
        pt_csw = pt_called + pt_whiffs

        rows.append({
            "game_id":   int(game_id),
            "game_date": str(game_date),
            "season":    int(season),
            "n_pitches": n,
            # Fastball-only physical
            "fb_avg_velo": _cf(fb["release_speed"].mean(), 1) if not fb.empty else None,
            "fb_max_velo": _cf(fb["release_speed"].max(), 1) if not fb.empty else None,
            "fb_avg_spin": _cf(fb["release_spin_rate"].mean(), 0) if not fb.empty else None,
            "fb_avg_ivb":  _cf(fb["pfx_z"].mean() * 12, 1) if not fb.empty else None,
            "fb_avg_hb":   _cf(fb[hb_col].mean(), 1) if hb_col and not fb.empty else None,
            "fb_avg_ext":  _cf(fb["release_extension"].mean(), 1) if not fb.empty else None,
            # All-pitches / pitch-type-filtered physical (Pitch Type category)
            "avg_velo": _cf(grp["release_speed"].mean(), 1),
            "avg_spin": _cf(grp["release_spin_rate"].mean(), 0),
            "avg_ivb":  _cf(grp["pfx_z"].mean() * 12, 1),
            "avg_hb":   _cf(grp["hb"].mean(), 1) if "hb" in grp.columns else None,
            "avg_ext":  _cf(grp["release_extension"].mean(), 1),
            # Results
            "k_pct":    _cf(k / n_pa * 100, 1) if n_pa else None,
            "bb_pct":   _cf(bb / n_pa * 100, 1) if n_pa else None,
            "zone_pct": _cf(in_z / n * 100, 1),
            "csw_pct":  _cf(csw / n * 100, 1),
            "gb_pct":   _cf(gb / len(bip) * 100, 1) if len(bip) else None,
            # Process
            "whiff_pct":        _cf(whiffs / swings * 100, 1) if swings else None,
            "chase_pct":        _cf(chase_sw / out_z_count * 100, 1) if out_z_count else None,
            "zone_contact_pct": _cf(zone_con / zone_sw * 100, 1) if zone_sw else None,
            # Pitch Type category results (same values, separate keys for lookup disambiguation)
            "pt_whiff_pct": _cf(pt_whiffs / pt_swings * 100, 1) if pt_swings else None,
            "pt_zone_pct":  _cf(pt_in_z / n * 100, 1),
            "pt_csw_pct":   _cf(pt_csw / n * 100, 1),
        })
    return rows


def _batter_rows(conn, player_id, seasons):
    ph = ",".join("?" * len(seasons))
    df = pd.read_sql(
        f"{_BATTER_SELECT} AND season IN ({ph}) ORDER BY game_date, game_id",
        conn, params=[player_id] + list(seasons),
    )
    if df.empty:
        return []
    df = _add_flags(df)
    rows = []
    for (game_id, game_date, season), grp in df.groupby(
        ["game_id", "game_date", "season"], sort=False
    ):
        pa = grp[grp["events"].notna()]
        ev = grp["launch_speed"].dropna()
        la = grp["launch_angle"].dropna()
        swings = int(grp["is_swing"].sum())
        whiffs = int(grp["is_whiff"].sum())
        in_zone = grp["zone"].isin(IN_ZONE_CODES)
        out_zone = grp["zone"].isin(OUT_ZONE_CODES)
        k = int(pa["events"].isin(STRIKEOUT_EVENTS).sum())
        bb = int(pa["events"].isin(WALK_EVENTS).sum())
        n_pa = len(pa)
        n_ev = len(ev)
        hard = int((ev >= 95).sum())
        # Barrel: EV >= 98 + LA 26-30° (simplified Statcast approximation)
        ev_aligned = grp.loc[grp["launch_speed"].notna() & grp["launch_angle"].notna()]
        barrel_mask = (ev_aligned["launch_speed"] >= 98) & (ev_aligned["launch_angle"].between(26, 30))
        barrel = int(barrel_mask.sum())
        xwoba_vals = grp["estimated_woba_using_speedangle"].dropna()
        sweet_spot = int(la.between(8, 32).sum()) if len(la) else 0
        out_z_count = int(out_zone.sum())
        chase_sw = int((grp["is_swing"] & out_zone).sum())
        zone_sw = int((grp["is_swing"] & in_zone).sum())
        zone_con = int((grp["is_swing"] & in_zone & ~grp["is_whiff"]).sum())
        bip = grp[grp["bb_type"].notna()]
        gb = int((bip["bb_type"] == "ground_ball").sum())
        rows.append({
            "game_id":        int(game_id),
            "game_date":      str(game_date),
            "season":         int(season),
            "n_pitches":      len(grp),
            "avg_ev":         _cf(ev.mean(), 1) if n_ev else None,
            "max_ev":         _cf(ev.max(), 1) if n_ev else None,
            "ev_90th":        _cf(ev.quantile(0.9), 1) if n_ev >= 5 else None,
            "avg_la":         _cf(la.mean(), 1) if len(la) else None,
            "sweet_spot_pct": _cf(sweet_spot / n_ev * 100, 1) if n_ev else None,
            "hard_hit_pct":   _cf(hard / n_ev * 100, 1) if n_ev else None,
            "barrel_pct":     _cf(barrel / n_ev * 100, 1) if n_ev else None,
            "xwoba":          _cf(xwoba_vals.mean(), 3) if len(xwoba_vals) else None,
            "whiff_pct":      _cf(whiffs / swings * 100, 1) if swings else None,
            "k_pct":          _cf(k / n_pa * 100, 1) if n_pa else None,
            "bb_pct":         _cf(bb / n_pa * 100, 1) if n_pa else None,
            "chase_pct":      _cf(chase_sw / out_z_count * 100, 1) if out_z_count else None,
            "zone_contact_pct": _cf(zone_con / zone_sw * 100, 1) if zone_sw else None,
            "swing_pct":      _cf(swings / len(grp) * 100, 1),
            "gb_pct":         _cf(gb / len(bip) * 100, 1) if len(bip) else None,
        })
    return rows


@router.get("/players/{player_id}/trends")
def get_player_trends(
    player_id: int,
    role: str = Query("pitcher"),
    season: str = Query("2026"),
    pitch_type: Optional[str] = Query(None),
    conn=Depends(get_conn),
):
    id_col = "pitcher_id" if role == "pitcher" else "batter_id"
    if season == "all":
        seasons = [
            r[0] for r in conn.execute(
                f"SELECT DISTINCT season FROM pitches WHERE {id_col} = ? ORDER BY season",
                (player_id,),
            ).fetchall()
        ]
    else:
        try:
            seasons = [int(season)]
        except ValueError:
            seasons = [2026]

    metrics = PITCHER_METRICS if role == "pitcher" else BATTER_METRICS
    if not seasons:
        return {"games": [], "metrics": metrics, "seasons": [], "pitch_types": []}

    # Distinct pitch types for this player/season(s)
    ph = ",".join("?" * len(seasons))
    pt_rows = conn.execute(
        f"SELECT DISTINCT pitch_type FROM pitches WHERE {id_col} = ? AND season IN ({ph})"
        " AND pitch_type IS NOT NULL ORDER BY pitch_type",
        [player_id] + seasons,
    ).fetchall()
    pitch_types = [r[0] for r in pt_rows if r[0]]

    if role == "pitcher":
        games = _pitcher_rows(conn, player_id, seasons, pitch_type=pitch_type)
    else:
        games = _batter_rows(conn, player_id, seasons)

    return {"games": games, "metrics": metrics, "seasons": seasons, "pitch_types": pitch_types}


@ttl_cache()
def _league_averages_cached(conn, role: str, season: int):
    if role == "pitcher":
        row = conn.execute("""
            SELECT AVG(release_speed), MAX(release_speed), AVG(release_spin_rate)
            FROM pitches WHERE season = ? AND release_speed IS NOT NULL
        """, (season,)).fetchone()
        fb_row = conn.execute("""
            SELECT AVG(release_speed), AVG(release_spin_rate)
            FROM pitches WHERE season = ? AND pitch_type IN ('FF','SI','FA')
            AND release_speed IS NOT NULL
        """, (season,)).fetchone()
        df = pd.read_sql(
            "SELECT description, zone, events, bb_type FROM pitches WHERE season = ?",
            conn, params=[season],
        )
        df = _add_flags(df)
        swings = int(df["is_swing"].sum())
        whiffs = int(df["is_whiff"].sum())
        pa = df[df["events"].notna()]
        n_pa = len(pa)
        n = len(df)
        k = int(pa["events"].isin(STRIKEOUT_EVENTS).sum())
        bb = int(pa["events"].isin(WALK_EVENTS).sum())
        in_z = int(df["zone"].isin(IN_ZONE_CODES).sum())
        out_z = int(df["zone"].isin(OUT_ZONE_CODES).sum())
        called_s = int((df["description"] == "called_strike").sum())
        csw = called_s + whiffs
        bip = df[df["bb_type"].notna()]
        gb = int((bip["bb_type"] == "ground_ball").sum())
        chase_sw = int((df["is_swing"] & df["zone"].isin(OUT_ZONE_CODES)).sum())
        zone_sw = int((df["is_swing"] & df["zone"].isin(IN_ZONE_CODES)).sum())
        zone_con = int((df["is_swing"] & df["zone"].isin(IN_ZONE_CODES) & ~df["is_whiff"]).sum())
        return {
            "fb_avg_velo": _cf(fb_row[0], 1) if fb_row else None,
            "fb_avg_spin": _cf(fb_row[1], 0) if fb_row else None,
            "avg_velo":    _cf(row[0], 1) if row else None,
            "k_pct":       _cf(k / n_pa * 100, 1) if n_pa else None,
            "bb_pct":      _cf(bb / n_pa * 100, 1) if n_pa else None,
            "zone_pct":    _cf(in_z / n * 100, 1) if n else None,
            "csw_pct":     _cf(csw / n * 100, 1) if n else None,
            "whiff_pct":   _cf(whiffs / swings * 100, 1) if swings else None,
            "chase_pct":   _cf(chase_sw / out_z * 100, 1) if out_z else None,
            "zone_contact_pct": _cf(zone_con / zone_sw * 100, 1) if zone_sw else None,
            "gb_pct":      _cf(gb / len(bip) * 100, 1) if len(bip) else None,
        }
    else:
        df = pd.read_sql(
            "SELECT launch_speed, launch_angle, description, zone, events, bb_type,"
            " estimated_woba_using_speedangle FROM pitches WHERE season = ?",
            conn, params=[season],
        )
        df = _add_flags(df)
        ev = df["launch_speed"].dropna()
        la = df["launch_angle"].dropna()
        hard = int((ev >= 95).sum())
        pa = df[df["events"].notna()]
        n_pa = len(pa)
        n = len(df)
        k = int(pa["events"].isin(STRIKEOUT_EVENTS).sum())
        bb = int(pa["events"].isin(WALK_EVENTS).sum())
        swings = int(df["is_swing"].sum())
        whiffs = int(df["is_whiff"].sum())
        xwoba_vals = df["estimated_woba_using_speedangle"].dropna()
        sweet_spot = int(la.between(8, 32).sum()) if len(la) else 0
        n_ev = len(ev)
        out_z = int(df["zone"].isin(OUT_ZONE_CODES).sum())
        zone_sw = int((df["is_swing"] & df["zone"].isin(IN_ZONE_CODES)).sum())
        zone_con = int((df["is_swing"] & df["zone"].isin(IN_ZONE_CODES) & ~df["is_whiff"]).sum())
        chase_sw = int((df["is_swing"] & df["zone"].isin(OUT_ZONE_CODES)).sum())
        bip = df[df["bb_type"].notna()]
        gb = int((bip["bb_type"] == "ground_ball").sum())
        return {
            "avg_ev":         _cf(ev.mean(), 1) if n_ev else None,
            "ev_90th":        _cf(ev.quantile(0.9), 1) if n_ev >= 5 else None,
            "avg_la":         _cf(la.mean(), 1) if len(la) else None,
            "sweet_spot_pct": _cf(sweet_spot / n_ev * 100, 1) if n_ev else None,
            "hard_hit_pct":   _cf(hard / n_ev * 100, 1) if n_ev else None,
            "xwoba":          _cf(xwoba_vals.mean(), 3) if len(xwoba_vals) else None,
            "whiff_pct":      _cf(whiffs / swings * 100, 1) if swings else None,
            "k_pct":          _cf(k / n_pa * 100, 1) if n_pa else None,
            "bb_pct":         _cf(bb / n_pa * 100, 1) if n_pa else None,
            "chase_pct":      _cf(chase_sw / out_z * 100, 1) if out_z else None,
            "zone_contact_pct": _cf(zone_con / zone_sw * 100, 1) if zone_sw else None,
            "swing_pct":      _cf(swings / n * 100, 1) if n else None,
            "gb_pct":         _cf(gb / len(bip) * 100, 1) if len(bip) else None,
        }


@router.get("/trends/league-averages")
def get_league_averages(
    role: str = Query("pitcher"),
    season: str = Query("2026"),
    conn=Depends(get_conn),
):
    try:
        s = int(season)
    except ValueError:
        s = 2026
    return _league_averages_cached(conn, role, s)
