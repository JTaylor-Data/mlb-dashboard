"""
/api/players and /api/player/{id} - the individual player page. Assembles
exactly what pages/1_Player_Page.py rendered, by calling the same analytics
modules it did, just packaged as JSON instead of Streamlit widgets/HTML.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from archetypes import ARCHETYPE_DESCRIPTIONS, classify_batter_archetype
from boxscore import season_era
from metrics import classify_pitcher_roles
from percentiles import (
    BATTER_POLARITY, PITCHER_POLARITY, batter_percentile_metrics,
    pitcher_percentile_metrics, player_percentiles,
)
from player_page import (
    batter_counting_stats, batter_pitch_type_table, batter_wrc_plus,
    field_zone_pcts, league_average_batter_row, league_average_swing_profile,
    league_hr_per_fb, movement_plot_data, pitch_usage_by_stand, pitcher_arm_angle,
    pitcher_counting_stats, pitcher_pitch_table, pitcher_total_row, swing_profile,
)
from stuff_model import compute_stuff_plus, compute_stuff_plus_by_stand, overall_stuff_plus

from api.cache import ttl_cache, ttl_cache_all_args
from api.db import get_conn
from api.presentation import (
    BATTER_FMT, BATTER_GROUPS, BATTER_LABELS, BATTER_TABLE_COLUMNS,
    PITCHER_FMT, PITCHER_GROUPS, PITCHER_LABELS, PITCHER_TABLE_COLUMNS,
    PITCHER_TABLE_FMT, format_row, format_value,
)

router = APIRouter(prefix="/api")


def _clean(v):
    """NaN/NaT -> None (and numpy scalars -> native python) so FastAPI's
    JSON encoder doesn't choke."""
    if v is None:
        return None
    if isinstance(v, (list, dict, str)):
        return v
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


@ttl_cache()
def _pitcher_pool(conn, season, stand_filter):
    return pitcher_percentile_metrics(conn, season, stand_filter=stand_filter)


@ttl_cache()
def _batter_pool(conn, season, throws_filter):
    return batter_percentile_metrics(conn, season, throws_filter=throws_filter)


@ttl_cache()
def _stuff_plus_table(conn, season):
    return compute_stuff_plus(conn, season)


@ttl_cache()
def _stuff_plus_by_stand_table(conn, season):
    return compute_stuff_plus_by_stand(conn, season)


def _stuff_breakdown(stuff_all, stuff_by_stand, overall_stuff, pitcher_id, pitch_types_ordered):
    """Grid data for the Stuff+ breakdown: columns = ALL + each pitch type
    (usage order), rows = ALL / vs RHH / vs LHH. Mirrors the layout of the
    per-pitch-type/handedness Stuff+ card the user referenced."""
    if pitcher_id not in stuff_all.index.get_level_values("pitcher_id"):
        return None

    all_rows = stuff_all.xs(pitcher_id, level="pitcher_id")
    stand_rows = None
    if pitcher_id in stuff_by_stand.index.get_level_values("pitcher_id"):
        stand_rows = stuff_by_stand.xs(pitcher_id, level="pitcher_id")

    columns = ["ALL"] + [pt for pt in pitch_types_ordered if pt in all_rows.index]

    def cell(col, row_stand):
        if col == "ALL":
            if row_stand == "ALL":
                return overall_stuff
            if stand_rows is None:
                return None
            sub = stand_rows[stand_rows.index.get_level_values("stand") == row_stand]
            if sub.empty:
                return None
            return float(np.average(sub["stuff_plus"], weights=sub["n_pitches"]))
        if row_stand == "ALL":
            return float(all_rows.loc[col, "stuff_plus"]) if col in all_rows.index else None
        if stand_rows is None or (col, row_stand) not in stand_rows.index:
            return None
        return float(stand_rows.loc[(col, row_stand), "stuff_plus"])

    rows = []
    for row_stand, row_label in [("ALL", "ALL"), ("R", "vs RHH"), ("L", "vs LHH")]:
        values = {}
        for col in columns:
            v = cell(col, row_stand)
            values[col] = round(v, 0) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
        rows.append({"label": row_label, "values": values})

    return {"columns": columns, "rows": rows}


@ttl_cache()
def _pitcher_roles(conn, season):
    return classify_pitcher_roles(conn, season)


@ttl_cache()
def _league_hr_fb(conn, season):
    return league_hr_per_fb(conn, season)


@ttl_cache()
def _league_avg_batter_row(conn, season, throws_filter):
    return league_average_batter_row(conn, season, throws_filter)


@ttl_cache()
def _league_swing_profile(conn, season):
    return league_average_swing_profile(conn, season)


@ttl_cache_all_args()
def _season_era(player_id, season):
    try:
        return season_era(player_id, season)
    except Exception:
        # Live external call - a network hiccup shouldn't 500 the whole
        # player page, just fall back to "-" for the ERA card entry.
        return None


def _norm_split(split):
    return None if split in (None, "ALL", "") else split


def _percentile_groups(pct_dict, labels, fmt_map, groups):
    out = []
    for title, metrics in groups:
        present = [m for m in metrics if m in pct_dict]
        if not present:
            continue
        rows = []
        for m in present:
            raw, p = pct_dict[m]
            rows.append({
                "key": m,
                "label": labels.get(m, m),
                "raw": _clean(raw),
                "pct": round(float(p), 1) if p is not None and not (isinstance(p, float) and math.isnan(p)) else None,
                "value_str": format_value(raw, fmt_map.get(m, "{:.1f}")),
            })
        out.append({"title": title, "metrics": rows})
    return out


def _table_to_rows(df, columns, fmt_map, league_avg_dict=None):
    rows = []
    for pitch_type, row in df.iterrows():
        raw = row.to_dict()
        formatted = format_row(raw, fmt_map)
        rows.append({
            "pitch_type": pitch_type,
            "cells": [{"key": k, "label": label, "value": formatted.get(k, "-")} for k, label in columns],
        })
    league_row = None
    if league_avg_dict is not None:
        formatted = format_row(league_avg_dict, fmt_map)
        league_row = [{"key": k, "label": label, "value": formatted.get(k, "-")} for k, label in columns]
    return rows, league_row


@router.get("/players")
def list_players(role: str = Query("pitcher", pattern="^(pitcher|batter)$"),
                  season: int = Query(2026), conn=Depends(get_conn)):
    id_col = "pitcher_id" if role == "pitcher" else "batter_id"
    df = pd.read_sql(f"""
        SELECT DISTINCT pl.player_id, pl.name, pl.team_id
        FROM players pl JOIN pitches p ON pl.player_id = p.{id_col}
        WHERE p.season = ?
        ORDER BY pl.name
    """, conn, params=(season,))
    return df.to_dict(orient="records")


@ttl_cache()
def _all_players(conn, season):
    # One row per (player, role) - a true two-way player (has both thrown
    # and batted) shows up twice, once per role, which is the right call for
    # a search box: picking "Ohtani (Pitcher)" vs "Ohtani (Hitter)" should
    # load two different views, not one ambiguous merged one.
    return pd.read_sql("""
        SELECT DISTINCT pl.player_id, pl.name, pl.team_id, 'pitcher' AS role
        FROM players pl JOIN pitches p ON pl.player_id = p.pitcher_id
        WHERE p.season = ?
        UNION
        SELECT DISTINCT pl.player_id, pl.name, pl.team_id, 'batter' AS role
        FROM players pl JOIN pitches p ON pl.player_id = p.batter_id
        WHERE p.season = ?
        ORDER BY name
    """, conn, params=(season, season))


@router.get("/players/all")
def list_all_players(season: int = Query(2026), conn=Depends(get_conn)):
    return _all_players(conn, season).to_dict(orient="records")


@router.get("/player/{player_id}")
def get_player(
    player_id: int,
    role: str = Query("pitcher", pattern="^(pitcher|batter)$"),
    split: str = Query("ALL", pattern="^(ALL|L|R)$"),
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    player_row = conn.execute(
        "SELECT name, team_id, bats, throws FROM players WHERE player_id = ?", (player_id,)
    ).fetchone()
    if player_row is None:
        raise HTTPException(status_code=404, detail="Player not found")
    name, team_id, bats, throws = player_row
    split_filter = _norm_split(split)

    base = {"player_id": player_id, "name": name, "team_id": team_id, "role": role, "split": split, "season": season}
    if role == "pitcher":
        base["handedness"] = f"{throws}HP" if throws else None
    else:
        base["handedness"] = f"Bats {bats}" if bats else None

    if role == "pitcher":
        pool = _pitcher_pool(conn, season, split_filter)
        stuff = _stuff_plus_table(conn, season)
        roles = _pitcher_roles(conn, season)
        gs = int(roles.loc[player_id, "games_started"]) if player_id in roles.index else None
        lg_hrfb = _league_hr_fb(conn, season)

        stats = pitcher_counting_stats(conn, season, player_id, split_filter, games_started=gs, league_hr_fb_rate=lg_hrfb)
        overall_stuff = overall_stuff_plus(stuff, player_id)
        era = _season_era(player_id, season)

        base["card"] = [
            {"label": "IP", "value": stats["ip"]},
            {"label": "K", "value": stats["k"]},
            {"label": "GS", "value": stats["gs"] if stats["gs"] is not None else "-"},
            {"label": "OPS against", "value": format_value(stats["OPS"], "{:.3f}")},
            # ERA has no official vs-LHB/vs-RHB split - the MLB Stats API's
            # own split-stats endpoint omits era/earnedRuns entirely for
            # sitCodes=vr/vl (unlike avg/obp/slg/whip, which are
            # attributable per plate appearance, earned-run status is an
            # inning-level judgment, not attributable to one batter side).
            # Label it explicitly as season-total rather than silently not
            # moving with the split toggle.
            {"label": "ERA (season)", "value": era if era is not None else "-"},
            {"label": "xFIP", "value": format_value(stats["xfip"], "{:.2f}")},
        ]
        base["badge"] = {
            "kind": "stuff_plus",
            "value": _clean(overall_stuff),
            "value_str": format_value(overall_stuff, "{:.0f}") if not (isinstance(overall_stuff, float) and math.isnan(overall_stuff)) else "N/A",
        }

        pct = player_percentiles(pool, player_id, PITCHER_POLARITY)
        base["percentiles"] = _percentile_groups(pct, PITCHER_LABELS, PITCHER_FMT, PITCHER_GROUPS)

        mv = movement_plot_data(conn, season, player_id, split_filter)
        arm_angle = pitcher_arm_angle(conn, season, player_id)
        base["pitch_usage_by_stand"] = pitch_usage_by_stand(conn, season, player_id)

        summary_rows = []
        filtered = mv
        if not mv.empty:
            closest_frames = []
            for pt, group in mv.groupby("pitch_type"):
                avg_hb, avg_ivb = group["HB"].mean(), group["IVB"].mean()
                summary_rows.append({
                    "pitch_type": pt, "hb": _clean(avg_hb), "ivb": _clean(avg_ivb), "n_pitches": len(group),
                })
                # Only plot the half of points closest to this pitch type's
                # own average shape - decluttering, the summary stats above
                # are still computed from every pitch, not just the kept half.
                dist = ((group["HB"] - avg_hb) ** 2 + (group["IVB"] - avg_ivb) ** 2) ** 0.5
                keep_n = max(1, len(group) // 2)
                closest_frames.append(group.loc[dist.sort_values().index[:keep_n]])
            filtered = pd.concat(closest_frames)
            summary_rows.sort(key=lambda r: -r["n_pitches"])

        base["movement_plot"] = {
            "arm_angle": _clean(arm_angle),
            "points": [
                {"pitch_type": r["pitch_type"], "hb": _clean(r["HB"]), "ivb": _clean(r["IVB"])}
                for _, r in filtered.iterrows()
            ] if not mv.empty else [],
            "summary": summary_rows,
        }

        table = pitcher_pitch_table(conn, season, player_id, split_filter, stuff)

        stuff_by_stand = _stuff_plus_by_stand_table(conn, season)
        pitch_types_ordered = list(table.index) if not table.empty else []
        base["stuff_breakdown"] = _stuff_breakdown(stuff, stuff_by_stand, overall_stuff, player_id, pitch_types_ordered)

        if not table.empty:
            table = table.sort_values("usage_pct", ascending=False)
            total_row = pitcher_total_row(conn, season, player_id, split_filter)
            total_row["usage_pct"] = None
            total_row["stuff_plus"] = overall_stuff
            rows, total_row_cells = _table_to_rows(table, PITCHER_TABLE_COLUMNS, PITCHER_TABLE_FMT, total_row)
        else:
            rows, total_row_cells = [], None
        base["pitch_table"] = {"columns": [{"key": k, "label": v} for k, v in PITCHER_TABLE_COLUMNS],
                                "rows": rows, "summary_row": total_row_cells, "summary_label": "Player Total"}

    else:
        pool = _batter_pool(conn, season, split_filter)
        stats = batter_counting_stats(conn, season, player_id, split_filter)
        wrc = batter_wrc_plus(conn, season, player_id, split_filter)

        base["card"] = [
            {"label": "GP", "value": stats["gp"]},
            {"label": "HR", "value": stats["hr"]},
            {"label": "OPS", "value": format_value(stats["OPS"], "{:.3f}")},
            {"label": "wRC+ (approx)", "value": format_value(wrc, "{:.0f}")},
        ]

        all_pool = _batter_pool(conn, season, None)
        archetype_pct = player_percentiles(all_pool, player_id, BATTER_POLARITY)
        archetype = classify_batter_archetype(archetype_pct) if archetype_pct else None
        base["badge"] = {
            "kind": "archetype",
            "value": archetype,
            "description": ARCHETYPE_DESCRIPTIONS.get(archetype, "") if archetype else "",
        } if archetype else None

        pct = player_percentiles(pool, player_id, BATTER_POLARITY)
        base["percentiles"] = _percentile_groups(pct, BATTER_LABELS, BATTER_FMT, BATTER_GROUPS)

        zones = field_zone_pcts(conn, season, player_id)
        base["swing_field"] = {
            "lf": _clean(zones["lf"]), "cf": _clean(zones["cf"]), "rf": _clean(zones["rf"]),
            "locations": zones["locations"],
        }

        profile = swing_profile(conn, season, player_id, split_filter)
        league_swing = _league_swing_profile(conn, season)
        base["swing_3d"] = {
            "bats": bats,
            "bat_speed": _clean(profile.get("bat_speed")),
            "attack_angle": _clean(profile.get("attack_angle")),
            "attack_direction": _clean(profile.get("attack_direction")),
            "swing_path_tilt": _clean(profile.get("swing_path_tilt")),
            "swing_length": _clean(profile.get("swing_length")),
            "league_avg": {
                "bat_speed": _clean(league_swing.get("bat_speed")),
                "attack_angle": _clean(league_swing.get("attack_angle")),
                "attack_direction": _clean(league_swing.get("attack_direction")),
                "swing_path_tilt": _clean(league_swing.get("swing_path_tilt")),
                "swing_length": _clean(league_swing.get("swing_length")),
            },
        }

        table = batter_pitch_type_table(conn, season, player_id, split_filter)
        if not table.empty:
            lg_row = _league_avg_batter_row(conn, season, split_filter)
            rows, league_row = _table_to_rows(table, BATTER_TABLE_COLUMNS, PITCHER_TABLE_FMT | {
                "OPS": "{:.3f}", "wOBA": "{:.3f}", "xwOBA": "{:.3f}", "K_pct": "{:.1%}",
                "whiff_pct": "{:.1%}", "barrel_pct": "{:.1%}",
            }, lg_row)
        else:
            rows, league_row = [], None
        base["pitch_type_table"] = {"columns": [{"key": k, "label": v} for k, v in BATTER_TABLE_COLUMNS],
                                     "rows": rows, "summary_row": league_row, "summary_label": "League Avg"}

    return base
