"""
/api/hitzones/{batter_id} - real xwOBA per 3x3 strike-zone cell, per pitch
type (plus an "ALL" combined view), for the batter page's Hit Zones card
(replaces the old percentile left-panel slot for batters - percentiles now
live in the shared full-width row below, matching pitchers).

plate_x/plate_z/sz_top/sz_bot already exist in `pitches` (same source
heatmaps.py's Pitch Locations chart uses). Zone assignment is computed
directly from those real coordinates rather than Statcast's own `zone`
integer code, to avoid trusting a second, independent numbering convention
on top of the one this page's Pitch Locations chart already has verified.
zone_idx is computed here from RAW, unmirrored plate_x (col 0 = most
negative plate_x). The frontend (buildHitZonesSVG, player.js) now mirrors
this to a Pitcher's View column-for-column at render time (matching the
Pitch Locations chart's own convention - was deliberately Catcher's View
here, switched per direction) rather than negating plate_x itself here, so
zone numbering stays a stable, unmirrored real-world quantity regardless of
which view the frontend chooses to draw.

xwOBA (not SLG - switched per direction) uses the same `woba_denom`/
`estimated_woba_using_speedangle` convention as the rest of this project's
xwOBA numbers (percentiles.py's pool, the player card) - mean expected wOBA
across qualifying plate appearances in that zone, not a sum/count ratio.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.cache import ttl_cache_all_args
from api.db import get_conn

router = APIRouter(prefix="/api")

PLATE_WIDTH_FT = 17 / 12
PITCH_TYPE_MERGE = {"KC": "CU"}


@ttl_cache_all_args()
def _hit_zone_summary(conn, season, batter_id, throws_filter=None):
    where, params = "WHERE season = ? AND batter_id = ? AND plate_x IS NOT NULL AND plate_z IS NOT NULL", (season, batter_id)
    if throws_filter and throws_filter != "ALL":
        where += " AND p_throws = ?"
        params += (throws_filter,)
    df = pd.read_sql(
        f"SELECT pitch_type, plate_x, plate_z, sz_top, sz_bot, woba_denom, estimated_woba_using_speedangle "
        f"FROM pitches {where}", conn, params=params,
    )
    if df.empty:
        return None
    df["pitch_type"] = df["pitch_type"].replace(PITCH_TYPE_MERGE)
    sz_top = float(df["sz_top"].mean()) if df["sz_top"].notna().any() else 3.5
    sz_bot = float(df["sz_bot"].mean()) if df["sz_bot"].notna().any() else 1.5

    # Zone assignment computed for EVERY real pitch (not just PA-ending ones)
    # so the displayed count is total pitches seen in that zone, not plate
    # appearances - xwOBA itself still only ever averages the PA-ending
    # subset (woba_denom==1), since xwOBA isn't a meaningful quantity for a
    # take/foul that didn't end the PA.
    col_w = PLATE_WIDTH_FT / 3
    row_h = (sz_top - sz_bot) / 3
    col = np.clip(((df["plate_x"] + PLATE_WIDTH_FT / 2) / col_w).astype(int), 0, 2)
    row = np.clip(((sz_top - df["plate_z"]) / row_h).astype(int), 0, 2)
    df["zone_idx"] = (row * 3 + col).astype(int)
    pa = df[(df["woba_denom"] == 1) & df["estimated_woba_using_speedangle"].notna()]

    def cells_for(all_group, pa_group):
        cells = []
        for z in range(9):
            zg_pa = pa_group[pa_group["zone_idx"] == z]
            n = int((all_group["zone_idx"] == z).sum())
            cells.append({"zone": z, "n": n, "xwoba": float(zg_pa["estimated_woba_using_speedangle"].mean()) if len(zg_pa) else None})
        return cells

    pitch_types = sorted(df["pitch_type"].dropna().unique().tolist())
    cells = {pt: cells_for(df[df["pitch_type"] == pt], pa[pa["pitch_type"] == pt]) for pt in pitch_types}
    cells["ALL"] = cells_for(df, pa)

    return {"pitch_types": pitch_types, "cells": cells, "sz_top": sz_top, "sz_bot": sz_bot, "plate_width_ft": PLATE_WIDTH_FT}


@router.get("/hitzones/{batter_id}")
def get_hit_zones(
    batter_id: int,
    season: int = Query(2026),
    split: str = Query("ALL", pattern="^(ALL|L|R)$"),
    conn=Depends(get_conn),
):
    summary = _hit_zone_summary(conn, season, batter_id, split)
    if summary is None:
        return {"batter_id": batter_id, "season": season, "has_data": False}
    return {"batter_id": batter_id, "season": season, "has_data": True, **summary}
