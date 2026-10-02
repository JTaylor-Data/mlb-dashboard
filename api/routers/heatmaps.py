"""
/api/heatmaps/{pitcher_id} - average pitch location + spread, v1 scope: one
combined circle across every pitch type (per-pitch-type breakdown is a
natural follow-up, not built yet). plate_x/plate_z/sz_top/sz_bot already
exist in `pitches` (schema.sql) and were previously unused by any analytics
function - no new pull needed.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query


from api.cache import ttl_cache_all_args
from api.db import get_conn

router = APIRouter(prefix="/api")

# Standard home-plate width, feet (17in) - plate_x's own natural scale.
PLATE_WIDTH_FT = 17 / 12


PITCH_TYPE_MERGE = {"KC": "CU"}   # same merge convention as analytics/player_page.py

# A pitcher needs at least this many real pitches of a given TYPE this
# split to get a command percentile for that type - below this, one wild
# pitch or one lucky cluster swings "spread" too much to mean anything.
# Same spirit as MIN_PITCHES_QUALIFY/MIN_PA_QUALIFY in analytics/
# percentiles.py, just a per-pitch-type threshold instead of a per-pitcher
# one (a pitcher can qualify overall while still being too thin on one
# specific secondary pitch to rank it).
MIN_PITCHES_FOR_COMMAND_PCT = 20


@ttl_cache_all_args()
def _league_command_pool(conn, season, stand_filter=None):
    """Real command percentile pool: for every (pitcher, pitch type) with
    at least MIN_PITCHES_FOR_COMMAND_PCT real pitches this season/split,
    the pitcher's own location spread (combined stddev of plate_x/plate_z -
    same "spread" _pitch_location_summary already computes for one pitcher,
    just done league-wide here) and where that spread RANKS against every
    other pitcher who throws the SAME pitch type (not across types - a
    splitter and a cutter don't have comparable typical spreads, so ranking
    has to stay within one pitch type). Lower spread = better command, so
    percentile is inverted (100 = the tightest, best-commanded version of
    that pitch in the league) - same polarity/formula convention as
    player_percentiles() in analytics/percentiles.py. Cached per (season,
    split) - a real league-wide scan, but narrow (4 columns) and fully
    vectorized (no per-pitcher Python loop), so it's much lighter than the
    full percentile pools; shared across every pitcher viewed this
    season+split rather than rebuilt per request."""
    where, params = "WHERE season = ? AND plate_x IS NOT NULL AND plate_z IS NOT NULL", (season,)
    if stand_filter and stand_filter != "ALL":
        where += " AND stand = ?"
        params += (stand_filter,)
    df = pd.read_sql(f"SELECT pitcher_id, pitch_type, plate_x, plate_z FROM pitches {where}", conn, params=params)
    if df.empty:
        return {}
    df["pitch_type"] = df["pitch_type"].replace(PITCH_TYPE_MERGE)

    g = df.groupby(["pitcher_id", "pitch_type"])
    agg = g.agg(
        std_x=("plate_x", lambda s: s.std(ddof=0)),
        std_z=("plate_z", lambda s: s.std(ddof=0)),
        n_pitches=("plate_x", "size"),
    ).reset_index()
    agg["spread"] = np.hypot(agg["std_x"], agg["std_z"])
    agg = agg[agg["n_pitches"] >= MIN_PITCHES_FOR_COMMAND_PCT].copy()
    if agg.empty:
        return {}

    # Percentile of spread WITHIN each pitch type, inverted (lower spread =
    # higher percentile = better command). transform() broadcasts each
    # pitch type's own rank back onto its rows in one vectorized pass.
    agg["command_pct"] = 100 - agg.groupby("pitch_type")["spread"].rank(pct=True) * 100

    pool = {}
    for row in agg.itertuples():
        pool.setdefault(row.pitcher_id, {})[row.pitch_type] = round(float(row.command_pct), 1)
    return pool


@ttl_cache_all_args()
def _pitch_location_summary(conn, season, pitcher_id, stand_filter=None):
    """One row per pitch TYPE: that type's own average (plate_x, plate_z)
    across the season, and its own spread (combined stddev of plate_x/
    plate_z) - a tight, well-commanded pitch gets a small dot, a scattered
    one gets a big dot. Sorted by spread descending so the largest (most
    scattered) dots are drawn FIRST and the tightest, most informative dots
    end up on top rather than buried underneath a big one. stand_filter
    ('L'/'R'/None): same opponent-bats split used everywhere else on the
    player page, against the `stand` column."""
    where, params = "WHERE season = ? AND pitcher_id = ? AND plate_x IS NOT NULL AND plate_z IS NOT NULL", (season, pitcher_id)
    if stand_filter and stand_filter != "ALL":
        where += " AND stand = ?"
        params += (stand_filter,)
    df = pd.read_sql(f"SELECT pitch_type, plate_x, plate_z, sz_top, sz_bot FROM pitches {where}", conn, params=params)
    if df.empty:
        return None
    df["pitch_type"] = df["pitch_type"].replace(PITCH_TYPE_MERGE)
    sz_top = float(df["sz_top"].mean()) if df["sz_top"].notna().any() else 3.5
    sz_bot = float(df["sz_bot"].mean()) if df["sz_bot"].notna().any() else 1.5

    command_pool = _league_command_pool(conn, season, stand_filter)
    pitcher_command = command_pool.get(pitcher_id, {})

    total_n = len(df)
    pitches = []
    for pt, group in df.groupby("pitch_type"):
        if len(group) / total_n * 100 < 1.0:
            continue
        std_x = float(group["plate_x"].std(ddof=0)) if len(group) > 1 else 0.0
        std_z = float(group["plate_z"].std(ddof=0)) if len(group) > 1 else 0.0
        pitches.append({
            "pitch_type": pt,
            "mean_x": float(group["plate_x"].mean()),
            "mean_z": float(group["plate_z"].mean()),
            "spread": float(np.hypot(std_x, std_z)),
            "n_pitches": int(len(group)),
            # None if this pitch type didn't meet MIN_PITCHES_FOR_COMMAND_PCT
            # this season/split - the frontend falls back to raw-spread
            # sizing for that one dot rather than guessing a percentile.
            "command_pct": pitcher_command.get(pt),
        })
    pitches.sort(key=lambda p: -p["spread"])
    return {"pitches": pitches, "sz_top": sz_top, "sz_bot": sz_bot, "n_pitches": int(len(df))}


@router.get("/heatmaps/{pitcher_id}")
def get_pitch_heatmap(
    pitcher_id: int,
    season: int = Query(2026),
    # No "ALL" - averaging pitch location across both batter hands together
    # is close to meaningless for a pitcher (his target shifts by opponent
    # hand), so this always requires a real split.
    split: str = Query("R", pattern="^(L|R)$"),
    conn=Depends(get_conn),
):
    summary = _pitch_location_summary(conn, season, pitcher_id, split)
    if summary is None:
        return {"pitcher_id": pitcher_id, "season": season, "has_data": False}
    return {
        "pitcher_id": pitcher_id, "season": season, "has_data": True,
        "plate_width_ft": PLATE_WIDTH_FT,
        "pitches": summary["pitches"],
        "sz_top": summary["sz_top"], "sz_bot": summary["sz_bot"],
        "n_pitches": summary["n_pitches"],
    }
