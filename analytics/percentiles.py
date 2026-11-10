"""
League-wide percentile rankings, Savant-style. Computed once across every
qualified player this season, then a specific player is located within that
distribution. Percentiles are simple rank-based (0-100), not shrinkage- or
park-adjusted.

Pull Air% uses the community-standard hc_x/hc_y -> spray angle formula (the
same one behind most public spray-chart tools); Statcast doesn't publish an
official closed-form version of this either, same situation as Barrel%.
It follows Savant's reading of the stat: "air" is every batted ball that is not
a grounder (fly ball, line drive AND popup), "pulled" is the pull third of the
field (more than 15 degrees toward the pull side of dead center), and the
denominator is all batted balls. For Bregman 2026 this gives 25.8% vs Savant's
25.6% (one batted ball apart); the earlier 0-degree pull cutoff with popups
excluded gave 30.9%.
"""
import gc

import numpy as np
import pandas as pd

from analytics.metrics import compute_vaa, compute_barrel
from analytics.splits import (
    SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS, STRIKE_DESCRIPTIONS,
    IN_ZONE_CODES, OUT_ZONE_CODES, STRIKEOUT_EVENTS, WALK_EVENTS, HBP_EVENTS,
)

# Duplicated from analytics/player_page.py deliberately, not imported - that
# module already imports fast_swing_bat_speed FROM this one, so importing
# back would be circular. Same values, same source of truth conceptually
# (this project already tolerates this exact kind of small constant
# duplication - FIP_CONSTANT_APPROX also appears in pulls/daily/pull_team_stats.py).
_OUTS_PER_EVENT = {
    "strikeout": 1, "strikeout_double_play": 2, "field_out": 1, "force_out": 1,
    "grounded_into_double_play": 2, "double_play": 2, "triple_play": 3,
    "sac_fly": 1, "sac_fly_double_play": 2, "sac_bunt": 1, "sac_bunt_double_play": 2,
    "fielders_choice_out": 1,
}
_FIP_CONSTANT_APPROX = 3.10


def _league_hr_per_fb(conn, season):
    row = conn.execute("""
        SELECT SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END),
               SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END)
        FROM pitches WHERE season = ?
    """, (season,)).fetchone()
    hr, fb = row
    return hr / fb if fb else np.nan


# Baseball Savant's own Percentile Rankings qualifiers (confirmed directly
# against baseballsavant.mlb.com/leaderboard/percentile-rankings, which
# states them explicitly): "2.1 PA per team game for batters, 1.25 PA per
# team game for pitchers" - rate-based against how far into the season the
# league actually is, not a flat season-total number. For pitchers "PA" in
# that sentence means PA AGAINST them (batters faced), which is exactly
# what `n_pa` already measures for the pitcher pool below (the shared
# _rate_metrics_pool() function makes no batter/pitcher distinction - PA
# is PA from whichever side of the plate this pool's WHERE clause is
# grouping by). Replaces the old flat MIN_PITCHES_QUALIFY=200/
# MIN_PA_QUALIFY=100 constants, which had no real source - see
# _team_games_played() below for how "how far into the season" is measured.
SAVANT_PA_PER_TEAM_GAME_BATTER = 2.1
SAVANT_PA_PER_TEAM_GAME_PITCHER = 1.25


def _team_games_played(conn, season):
    """Games played by whichever team has played the most this season - a
    single, league-wide "how far into the season are we" number, matching
    the spirit of Savant's own per-team-game qualifying rate (individual
    teams can differ by a game or two from raw schedule quirks - postponed/
    cancelled games, see the 2021/2024 cancelled-game cases found during
    the Spring Training cleanup - so MAX rather than any one specific team
    avoids under-counting qualifiers just because their particular team
    had a raincheck). For a completed season this settles at 162 (or 161
    for the two teams with a real, never-made-up cancellation)."""
    row = conn.execute("""
        SELECT MAX(n) FROM (
            SELECT COUNT(*) AS n FROM (
                SELECT home_team AS team_id FROM games WHERE season = ?
                UNION ALL
                SELECT away_team AS team_id FROM games WHERE season = ?
            ) GROUP BY team_id
        )
    """, (season, season)).fetchone()
    return row[0] or 0


def _split_fraction(conn, season, column, value):
    """Fraction of this season's pitches where `column` = `value` (e.g.
    column='p_throws', value='L') - used to scale a full-season qualifying
    threshold down for a hand-split view.

    The full-season SAVANT_PA_PER_TEAM_GAME_* rate assumes every PA counts
    toward the threshold; a split view (vs LHP, vs RHP) only ever accumulates
    the fraction of PAs that happened against that hand, so applying the
    unscaled full-season rate there is unreachable by construction - e.g. in
    a season where ~30% of pitches are thrown by LHP, no batter can rack up
    more than ~30% of a full season's PA total facing lefties, so a raw
    full-season threshold makes the L-split pool permanently empty (confirmed
    via live testing: 0 qualifiers despite 650+ batters having real L-split
    pitches that season). Approximated via pitch counts rather than true PA
    counts - close enough for a qualifying bar, and avoids a second
    PA-counting query here."""
    row = conn.execute(f"""
        SELECT SUM(CASE WHEN {column} = ? THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
        FROM pitches WHERE season = ? AND {column} IS NOT NULL
    """, (value, season)).fetchone()
    return row[0] if row and row[0] else 1.0

# Standard community constants for converting Statcast hc_x/hc_y into a spray
# angle in degrees (0 = straight up the middle, negative = left field side).
_SPRAY_X0, _SPRAY_Y0 = 125.42, 198.27
_SPRAY_SCALE = 0.75
# A ball is "pulled" (for Pull Air%) when it is more than 15 true degrees toward
# the pull side; compute_spray_angle() is scaled by _SPRAY_SCALE, so the cutoff
# in those units is 15 * 0.75.
PULL_ZONE_DEG = 15.0
_PULL_ZONE_SCALED = PULL_ZONE_DEG * _SPRAY_SCALE


def compute_spray_angle(hc_x, hc_y):
    hc_x, hc_y = np.asarray(hc_x, dtype=float), np.asarray(hc_y, dtype=float)
    return np.degrees(np.arctan2(hc_x - _SPRAY_X0, _SPRAY_Y0 - hc_y)) * _SPRAY_SCALE


def _downcast_pitch_pool(df):
    """float64 -> float32 for every numeric pitch column (precision far
    beyond what any of these measurements need or the DB stores - Statcast
    itself publishes to 1-3 decimal places), and the id column -> int32
    (MLB player ids fit comfortably). Halves this DataFrame's footprint
    before the groupby/aggregate passes that follow. Mutates and returns
    the same frame - caller's copy, not a fresh one.

    hc_x/hc_y are excluded: they feed compute_spray_angle's pull/oppo sign
    comparison (is_pulled), and float32 rounding was confirmed (before this
    exclusion) to occasionally flip a batted ball across that exact zero
    threshold, changing a handful of players' pull_air_pct by up to 0.5
    percentage points - a discrete classification change, not harmless
    float noise, so these two stay float64."""
    # One column at a time, not df[cols] = df[cols].astype(...) - the bulk
    # sub-frame form briefly holds the sub-frame's own float64 copy AND its
    # float32 result alongside the original, which measured WORSE than doing
    # nothing (1558MB vs the original 1316MB peak). Per-column assignment
    # lets each old float64 Series get freed as it's replaced.
    float_cols = df.select_dtypes(include=["float64"]).columns.difference(["hc_x", "hc_y"])
    for col in float_cols:
        df[col] = df[col].astype("float32")
    for id_col in ("pitcher_id", "batter_id"):
        if id_col in df.columns:
            df[id_col] = df[id_col].astype("int32")
    return df


def _add_shared_flags(df):
    """Mutates df in place (caller must not need the pre-call frame
    afterward - both call sites immediately reassign `df = _add_shared_flags(df)`,
    so there's nothing left pointing at the original to protect)."""
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["is_strike"] = df["description"].isin(STRIKE_DESCRIPTIONS)
    df["in_zone"] = df["zone"].isin(IN_ZONE_CODES)
    df["out_zone"] = df["zone"].isin(OUT_ZONE_CODES)
    df["spray_angle"] = compute_spray_angle(df["hc_x"], df["hc_y"])
    df["is_pulled"] = np.where(df["stand"] == "R", df["spray_angle"] < -_PULL_ZONE_SCALED,
                               df["spray_angle"] > _PULL_ZONE_SCALED)
    df["is_air"] = df["bb_type"].isin(["fly_ball", "line_drive", "popup"])
    df["is_batted_ball"] = df["bb_type"].notna()
    return df


def fast_swing_bat_speed(bat_speed_series, launch_speed_series=None):
    """Average bat speed over a hitter's 'competitive' swings, matching
    Baseball Savant's own Bat Speed leaderboard definition (confirmed via
    Savant's own methodology writeup): the fastest 90% of a player's swings,
    PLUS any swing >=60 mph that produced an exit velo >=90 mph (rescues a
    slower, well-executed swing that still barreled the ball) - i.e. drop
    only the bottom ~10% weakest swings (bunts, checked swings, defensive
    two-strike swings), not the inverse. This previously averaged the
    fastest QUARTER of swings instead (>=75th percentile) - the wrong end
    of the real definition, which systematically overstated bat speed
    (verified against a real player: Jazz Chisholm Jr. showed 77.8 mph here
    vs. Savant's own real ~73 mph for the same player/season)."""
    s = bat_speed_series.dropna()
    if s.empty:
        return np.nan
    threshold = s.quantile(0.10)
    keep = bat_speed_series >= threshold
    if launch_speed_series is not None:
        rescue = (bat_speed_series >= 60) & (launch_speed_series >= 90)
        keep = keep | rescue
    keep = keep & bat_speed_series.notna()
    return bat_speed_series[keep].mean()


def _safe_div(numer, denom):
    return numer / denom.replace(0, np.nan)


def _rate_metrics_pool(df, id_col):
    """Vectorized equivalent of calling the old per-group _rate_metrics()
    once per player in a Python loop - same exact formulas, computed as one
    groupby-aggregate pass across the whole pool instead of N separate
    pandas calls. Verified to reproduce the old loop's output exactly
    (float-tolerance) for both the batter and pitcher pools before this
    replaced it - a real, measured ~23-25x speedup (the old loop was ~4-5s
    for a full pool; this is ~0.2s), which mattered because this was the
    single biggest remaining per-request cost in the percentile/warmup path
    after the attack_direction vectorization fix."""
    d = df[[id_col, "is_swing", "is_whiff", "in_zone", "out_zone", "is_strike",
            "events", "is_batted_ball", "launch_speed", "launch_angle",
            "bb_type", "is_pulled", "is_air"]].copy()
    d["_swing_x_zone"] = d["is_swing"] & d["in_zone"]
    d["_whiff_x_zone"] = d["is_whiff"] & d["in_zone"]
    d["_swing_x_outzone"] = d["is_swing"] & d["out_zone"]
    is_pa = d["events"].notna()
    d["_is_pa"] = is_pa
    d["_is_k"] = is_pa & d["events"].isin(STRIKEOUT_EVENTS)
    d["_is_bb"] = is_pa & d["events"].isin(WALK_EVENTS)

    is_batted = d["is_batted_ball"]
    ls_bb = d["launch_speed"].where(is_batted)
    la_bb = d["launch_angle"].where(is_batted)
    is_barrel = compute_barrel(d["launch_speed"], d["launch_angle"])
    d["_barrel_bb"] = pd.Series(np.asarray(is_barrel, dtype=float), index=d.index).where(is_batted)
    d["_la_hardhit_bb"] = la_bb.where(ls_bb >= 95)
    d["_gb_bb"] = (d["bb_type"] == "ground_ball").astype(float).where(is_batted)
    d["_fb_bb"] = (d["bb_type"] == "fly_ball").astype(float).where(is_batted)
    d["_pull_air_bb"] = (d["is_pulled"] & d["is_air"]).astype(float).where(is_batted)
    d["_ls_bb"] = ls_bb
    d["_la_bb"] = la_bb

    g = d.groupby(id_col)
    out = pd.DataFrame(index=g.size().index)
    out["n_pitches"] = g.size()
    swings = g["is_swing"].sum()
    whiffs = g["is_whiff"].sum()
    out_zone_sum = g["out_zone"].sum()
    zone_swings = g["_swing_x_zone"].sum()
    zone_whiffs = g["_whiff_x_zone"].sum()
    swing_x_outzone = g["_swing_x_outzone"].sum()
    in_zone_n = g["in_zone"].sum()
    n_pa = g["_is_pa"].sum()
    k = g["_is_k"].sum()
    bb = g["_is_bb"].sum()

    out["whiff_pct"] = _safe_div(whiffs, swings)
    out["chase_pct"] = _safe_div(swing_x_outzone, out_zone_sum)
    out["zone_whiff_pct"] = _safe_div(zone_whiffs, zone_swings)
    out["zone_pct"] = g["in_zone"].mean()
    out["strike_pct"] = g["is_strike"].mean()
    out["swstr_pct"] = _safe_div(whiffs, out["n_pitches"])
    out["zone_swing_pct"] = _safe_div(zone_swings, in_zone_n)
    out["k_pct"] = _safe_div(k, n_pa)
    out["bb_pct"] = _safe_div(bb, n_pa)
    out["avg_exit_velo"] = g["_ls_bb"].mean()
    out["ev_90th"] = g["_ls_bb"].quantile(0.9)
    out["avg_la_95plus"] = g["_la_hardhit_bb"].mean()
    out["barrel_pct"] = g["_barrel_bb"].mean()
    out["gb_pct"] = g["_gb_bb"].mean()
    out["fb_pct"] = g["_fb_bb"].mean()
    out["pull_air_pct"] = g["_pull_air_bb"].mean()
    out["avg_launch_angle"] = g["_la_bb"].mean()
    out["n_pa"] = n_pa
    return out


# Players are processed in batches of this many at a time (one `WHERE id IN
# (...)` query per batch) so no single call to pitcher_percentile_metrics/
# batter_percentile_metrics ever materializes the full season's pitch-level
# table at once - that full-season read was the actual source of the
# multi-hundred-MB warmup spike (confirmed: narrowing columns/dtypes alone
# didn't move the peak, because pandas' read_sql on a raw sqlite3 connection
# fetches every matching row as Python tuples before any DataFrame exists,
# regardless of column count or dtype).
#
# 200, not ~100: measured under Docker --memory=512m against the current
# season (2026), None/L/R splits, pitcher+batter pools. 50 and 100 both
# peaked just OVER the 450MB target (452MB, 457MB); 200 stayed under it
# across two separate runs (445MB, 409MB) - fewer, larger batches means
# fewer separate read_sql round-trips, each with its own fixed overhead
# (connection/cursor setup, column/dtype inference) on top of the
# per-row cost, so going smaller doesn't reliably help once you're this
# far below the original ~1.3GB unbached peak.
_POOL_BATCH_SIZE = 200

_FB_TYPES = ("FF", "FA", "SI")


def _pitcher_vaa_height_regression(conn, where, params):
    """League-wide (every pitcher, not just one batch's ~100) regression of
    VAA against release height, fit separately per fastball-family pitch
    type - see pitcher_percentile_metrics' fb_vaa comment for why this
    adjustment exists. Fitting it from any single batch would give a
    different (wrong) slope than the real season-wide one, so this is a
    dedicated query over just the 5 columns the regression needs, run once
    before batching - not sourced from any per-player batch frame."""
    narrow = pd.read_sql(f"""
        SELECT pitch_type, plate_z, vy0, vz0, ay, az
        FROM pitches {where} AND pitch_type IN ('FF', 'FA', 'SI')
    """, conn, params=params)
    narrow["vaa"] = compute_vaa(narrow["vy0"], narrow["vz0"], narrow["ay"], narrow["az"])
    vaa_height_slope, type_mean_z = {}, {}
    for pt in _FB_TYPES:
        sub = narrow.loc[(narrow["pitch_type"] == pt) & narrow["vaa"].notna() & narrow["plate_z"].notna(), ["vaa", "plate_z"]]
        if len(sub) >= 50:
            vaa_height_slope[pt] = float(np.polyfit(sub["plate_z"], sub["vaa"], 1)[0])
            type_mean_z[pt] = float(sub["plate_z"].mean())
        else:
            vaa_height_slope[pt] = 0.0
            type_mean_z[pt] = float(sub["plate_z"].mean()) if len(sub) else np.nan
    return vaa_height_slope, type_mean_z


def _pitcher_pool_batch(conn, where, params, batch_ids, league_hr_fb_rate, vaa_height_slope, type_mean_z, advanced):
    """One batch of ~_POOL_BATCH_SIZE pitchers: same per-pitcher computation
    pitcher_percentile_metrics used to do over the whole season at once, just
    restricted to this batch's own pitches. Every metric here is a per-pitcher
    mean/sum - the one league-wide piece (the VAA/height regression) is
    passed in, already computed, not derived from this batch."""
    placeholders = ",".join("?" * len(batch_ids))
    batch_where = f"{where} AND pitcher_id IN ({placeholders})"
    batch_params = params + tuple(batch_ids)

    df = pd.read_sql(f"""
        SELECT pitcher_id, pitch_type, description, zone, events, stand, pitch_number,
               release_speed, effective_speed, release_extension, release_pos_z,
               release_pos_x, release_spin_rate, plate_z, arm_angle,
               launch_speed, launch_angle, bb_type, hc_x, hc_y,
               vy0, vz0, ay, az, woba_denom, estimated_woba_using_speedangle
        FROM pitches {batch_where}
    """, conn, params=batch_params)
    df = _downcast_pitch_pool(df)
    df = _add_shared_flags(df)
    df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
    is_fb = df["pitch_type"].isin(["FF", "FA", "SI"])

    rate_pool = _rate_metrics_pool(df, "pitcher_id")

    d = df
    d["_fb_speed"] = d["release_speed"].where(is_fb)
    d["_fb_eff_speed"] = d["effective_speed"].where(is_fb)
    d["_vaa_height_adj"] = d["vaa"] - d["pitch_type"].map(vaa_height_slope) * (
        d["plate_z"] - d["pitch_type"].map(type_mean_z)
    )

    # This pitcher's own most-thrown fastball-family type, by pitch count -
    # every other pitcher gets ranked on THEIR OWN primary type the same
    # way, so a sinkerballer's fb_vaa is built entirely from his sinkers,
    # never blended with an occasional four-seamer, and vice versa.
    fb_counts = d.loc[is_fb].groupby(["pitcher_id", "pitch_type"]).size().rename("n").reset_index()
    primary_fb_type = (
        fb_counts.sort_values("n", ascending=False)
        .drop_duplicates("pitcher_id")
        .set_index("pitcher_id")["pitch_type"]
    )
    d["_is_primary_fb_pitch"] = is_fb & (d["pitch_type"] == d["pitcher_id"].map(primary_fb_type))
    d["_fb_vaa"] = d["_vaa_height_adj"].where(d["_is_primary_fb_pitch"])

    qualifying = d["woba_denom"] == 1
    d["_qualifying"] = qualifying
    d["_xwoba_q"] = d["estimated_woba_using_speedangle"].where(qualifying)
    d["_outs"] = d["events"].map(_OUTS_PER_EVENT).fillna(0)
    d["_fb_batted"] = (d["bb_type"] == "fly_ball").astype(float)
    d["_bb_hbp"] = d["events"].isin(WALK_EVENTS | HBP_EVENTS).astype(float)
    d["_k"] = d["events"].isin(STRIKEOUT_EVENTS).astype(float)
    d["_fp_strike"] = d["is_strike"].where(d["pitch_number"] == 1)

    g = d.groupby("pitcher_id")
    extra = pd.DataFrame(index=g.size().index)
    extra["velo"] = g["_fb_speed"].mean()
    extra["perceived_velo"] = g["_fb_eff_speed"].mean()
    extra["fb_vaa"] = g["_fb_vaa"].mean()
    extra["fb_vaa_primary_type"] = primary_fb_type.reindex(extra.index)
    extra["extension"] = g["release_extension"].mean()
    extra["release_height"] = g["release_pos_z"].mean()
    extra["release_side"] = g["release_pos_x"].mean()
    extra["arm_angle"] = g["arm_angle"].mean()
    extra["spin_rate"] = g["release_spin_rate"].mean()
    extra["holding_runners"] = advanced["running_game_runs_prevented"].reindex(extra.index)
    xwoba_denom = g["_qualifying"].sum()
    extra["xwoba"] = _safe_div(g["_xwoba_q"].sum(), xwoba_denom)
    outs_sum = g["_outs"].sum()
    ip = outs_sum / 3
    fb_sum = g["_fb_batted"].sum()
    bb_hbp_sum = g["_bb_hbp"].sum()
    k_sum = g["_k"].sum()
    extra["xfip"] = _safe_div((13 * fb_sum * league_hr_fb_rate) + (3 * bb_hbp_sum) - (2 * k_sum), ip) + _FIP_CONSTANT_APPROX
    extra["first_pitch_strike_pct"] = g["_fp_strike"].mean()

    return rate_pool.join(extra)


def pitcher_percentile_metrics(conn, season, min_bf=None, stand_filter=None):
    """
    One row per qualified pitcher (>= min_bf batters faced WITHIN this
    filter - see SAVANT_PA_PER_TEAM_GAME_PITCHER/_team_games_played) with
    the raw metric values: whiff_pct, k_pct, chase_pct, zone_whiff_pct,
    perceived_velo, bb_pct, k_minus_bb_pct, avg_exit_velo, barrel_pct,
    gb_pct, extension, pull_air_pct, release_height, fb_vaa. All "against"
    (opponent's outcomes).

    stand_filter ('L'/'R'/None or 'ALL'): when set, the entire pool - both
    the qualifying threshold and the metrics themselves - is recomputed
    within pitches against that batter side only, not just re-ranked against
    the season-wide pool. The qualifying threshold itself is scaled down by
    that hand's real share of plate appearances (see _split_fraction) - the
    full-season rate alone is otherwise unreachable within a split.

    Processes pitchers in batches of _POOL_BATCH_SIZE (see that constant's
    comment) rather than loading every pitch of the season/split at once.
    """
    if min_bf is None:
        min_bf = SAVANT_PA_PER_TEAM_GAME_PITCHER * _team_games_played(conn, season)
        if stand_filter and stand_filter != "ALL":
            min_bf *= _split_fraction(conn, season, "stand", stand_filter)

    where = "WHERE season = ?"
    params = (season,)
    if stand_filter and stand_filter != "ALL":
        where += " AND stand = ?"
        params += (stand_filter,)

    ids = sorted(r[0] for r in conn.execute(f"SELECT DISTINCT pitcher_id FROM pitches {where}", params).fetchall())

    league_hr_fb_rate = _league_hr_per_fb(conn, season)

    # Real season-total xFIP/Barrel% (MLB Stats API sabermetrics / Baseball
    # Savant - see pulls/annual/pull_sabermetrics.py) have no opposing-batter-hand
    # split, so they only apply to the unsplit ALL view; a real L/R split
    # falls back to this project's own pitch-level approximation below.
    is_all = not stand_filter or stand_filter == "ALL"
    # Queried unconditionally (not gated by is_all like the xfip/barrel_pct
    # override below) because holding_runners (running_game_runs_prevented)
    # has no pitch-level fallback and no real L/R split source at all - same
    # reasoning as the batter side's sprint_speed/oaa, which also stay fixed
    # across that page's own split toggle.
    advanced = pd.read_sql(
        "SELECT player_id, xfip, barrel_pct_pitching, running_game_runs_prevented FROM player_advanced_stats WHERE season = ?",
        conn, params=(season,),
    ).set_index("player_id")

    # FB VAA height adjustment: VAA is measured at a fixed distance from the
    # plate (see compute_vaa's VAA_YF), not a fixed HEIGHT, so two pitches
    # with identical release/movement still read different VAA if one
    # crosses higher than the other. Fixed with a real per-pitch-type
    # regression adjustment, fit once across every pitcher (not per batch -
    # see _pitcher_vaa_height_regression).
    vaa_height_slope, type_mean_z = _pitcher_vaa_height_regression(conn, where, params)

    pool_parts = []
    for i in range(0, len(ids), _POOL_BATCH_SIZE):
        batch_pool = _pitcher_pool_batch(
            conn, where, params, ids[i:i + _POOL_BATCH_SIZE],
            league_hr_fb_rate, vaa_height_slope, type_mean_z, advanced,
        )
        pool_parts.append(batch_pool)
        del batch_pool
        gc.collect()

    pool = pd.concat(pool_parts)
    del pool_parts
    pool["k_minus_bb_pct"] = pool["k_pct"] - pool["bb_pct"]
    pool = pool[pool["n_pa"] >= min_bf].copy()

    if is_all:
        real_xfip = advanced["xfip"].reindex(pool.index)
        real_barrel = advanced["barrel_pct_pitching"].reindex(pool.index)
        pool["xfip"] = real_xfip.where(real_xfip.notna(), pool["xfip"])
        pool["barrel_pct"] = real_barrel.where(real_barrel.notna(), pool["barrel_pct"])

    pool = pool.sort_index()
    pool.index.name = "pitcher_id"
    return pool


_HIT_EVENTS = {"single", "double", "triple", "home_run"}


def _batter_pool_batch(conn, where, params, batch_ids, advanced):
    """One batch of ~_POOL_BATCH_SIZE batters: same per-batter computation
    batter_percentile_metrics used to do over the whole season at once, just
    restricted to this batch's own pitches. Every metric here is a
    per-batter mean/sum/quantile (attack_direction and the bat-speed-q10
    threshold are both grouped by batter_id) - unlike the pitcher side,
    there's no league-wide regression to pull out separately here."""
    placeholders = ",".join("?" * len(batch_ids))
    batch_where = f"{where} AND batter_id IN ({placeholders})"
    batch_params = params + tuple(batch_ids)

    df = pd.read_sql(f"""
        SELECT batter_id, description, zone, events, stand,
               launch_speed, launch_angle, bb_type, hc_x, hc_y, bat_speed, swing_length, attack_angle,
               swing_path_tilt, woba_denom, estimated_woba_using_speedangle, estimated_slg_using_speedangle,
               estimated_ba_using_speedangle
        FROM pitches {batch_where}
    """, conn, params=batch_params)
    df = _downcast_pitch_pool(df)
    df = _add_shared_flags(df)

    # Same hits-only, hand-adjusted landing-angle formula field_zone_pcts()
    # uses (negative=pull, positive=oppo for either batter hand) - computed
    # here too so it has a real percentile within the pool, not just a
    # standalone per-player number. Duplicated rather than imported since
    # player_page.py already imports FROM this module; keeping the direction
    # one-way avoids a circular import. Grouped by batter_id, so correct per
    # batch (every one of a given batter's rows lands in the same batch).
    hits_mask = df["events"].isin(_HIT_EVENTS) & df["hc_x"].notna() & df["hc_y"].notna()
    hits_df = df[hits_mask]
    hit_angle = compute_spray_angle(hits_df["hc_x"], hits_df["hc_y"])
    hit_hand_adj = np.where(hits_df["stand"] == "R", hit_angle, -hit_angle)
    attack_direction_by_batter = pd.Series(hit_hand_adj, index=hits_df.index).groupby(hits_df["batter_id"]).mean()

    rate_pool = _rate_metrics_pool(df, "batter_id")

    d = df
    qualifying = d["woba_denom"] == 1
    d["_qualifying"] = qualifying
    d["_xwoba_q"] = d["estimated_woba_using_speedangle"].where(qualifying)
    d["_xslg_q"] = d["estimated_slg_using_speedangle"].where(qualifying)
    d["_xba_q"] = d["estimated_ba_using_speedangle"].where(qualifying)
    # fast_swing_bat_speed's per-batter 10th-percentile threshold, computed
    # once via groupby.transform (pandas' own optimized per-group reduction)
    # instead of calling the function fresh per batter. Per-batter, so
    # correct within a batch same as attack_direction above.
    d["_bs_q10"] = d.groupby("batter_id")["bat_speed"].transform(lambda s: s.quantile(0.10))
    bs_keep = ((d["bat_speed"] >= d["_bs_q10"]) | ((d["bat_speed"] >= 60) & (d["launch_speed"] >= 90))) & d["bat_speed"].notna()
    d["_bs_kept"] = d["bat_speed"].where(bs_keep)

    g = d.groupby("batter_id")
    extra = pd.DataFrame(index=g.size().index)
    q_denom = g["_qualifying"].sum()
    extra["xwoba"] = _safe_div(g["_xwoba_q"].sum(), q_denom)
    extra["xslg"] = _safe_div(g["_xslg_q"].sum(), q_denom)
    extra["xba"] = _safe_div(g["_xba_q"].sum(), q_denom)
    extra["bat_speed"] = g["_bs_kept"].mean()
    extra["swing_length"] = g["swing_length"].mean()
    extra["attack_angle"] = g["attack_angle"].mean()
    extra["swing_path_tilt"] = g["swing_path_tilt"].mean()
    extra["attack_direction"] = attack_direction_by_batter.reindex(extra.index)
    extra["sprint_speed"] = advanced["sprint_speed"].reindex(extra.index)
    extra["oaa"] = advanced["oaa"].reindex(extra.index)
    extra["arm_strength"] = advanced["arm_strength"].reindex(extra.index)

    return rate_pool.join(extra)


def batter_percentile_metrics(conn, season, min_pa=None, throws_filter=None):
    """
    One row per qualified batter (>= min_pa plate appearances WITHIN this
    filter - see SAVANT_PA_PER_TEAM_GAME_BATTER/_team_games_played) with
    the raw metric values: xwoba, xslg, xba, avg_exit_velo, barrel_pct,
    avg_launch_angle, bat_speed, chase_pct, zone_whiff_pct, k_pct, bb_pct,
    pull_air_pct, gb_pct, sprint_speed, oaa.

    throws_filter ('L'/'R'/None or 'ALL'): when set, the entire pool - both
    the qualifying threshold and the metrics themselves - is recomputed
    within pitches from that pitcher hand only, not just re-ranked against
    the season-wide pool. sprint_speed/oaa are season-level (not pitch-level)
    so they don't change with this filter - same value regardless of split.
    The qualifying threshold itself is scaled down by that hand's real
    share of plate appearances (see _split_fraction) - e.g. ~30% of pitches
    league-wide come from LHP, so the unscaled full-season rate is
    unreachable within the L split and left the pool permanently empty
    (confirmed: 0 qualifiers for 2026's L split despite 650+ batters with
    real L-split pitches that season, max achievable PA ~280 vs. an
    unscaled threshold of ~340).

    Processes batters in batches of _POOL_BATCH_SIZE rather than loading
    every pitch of the season/split at once - see that constant's comment.
    """
    if min_pa is None:
        min_pa = SAVANT_PA_PER_TEAM_GAME_BATTER * _team_games_played(conn, season)
        if throws_filter and throws_filter != "ALL":
            min_pa *= _split_fraction(conn, season, "p_throws", throws_filter)

    where = "WHERE season = ?"
    params = (season,)
    if throws_filter and throws_filter != "ALL":
        where += " AND p_throws = ?"
        params += (throws_filter,)

    ids = sorted(r[0] for r in conn.execute(f"SELECT DISTINCT batter_id FROM pitches {where}", params).fetchall())

    advanced = pd.read_sql(
        "SELECT player_id, sprint_speed, oaa, barrel_pct_batting, arm_strength FROM player_advanced_stats WHERE season = ?",
        conn, params=(season,),
    ).set_index("player_id")
    # Real Barrel% (Baseball Savant, see pulls/annual/pull_sabermetrics.py) has no
    # opposing-pitcher-hand split; only applies to the unsplit ALL view, same
    # reasoning as the pitcher side's xfip/barrel_pct override above.
    is_all = not throws_filter or throws_filter == "ALL"

    pool_parts = []
    for i in range(0, len(ids), _POOL_BATCH_SIZE):
        batch_pool = _batter_pool_batch(conn, where, params, ids[i:i + _POOL_BATCH_SIZE], advanced)
        pool_parts.append(batch_pool)
        del batch_pool
        gc.collect()

    pool = pd.concat(pool_parts)
    del pool_parts
    pool = pool[pool["n_pa"] >= min_pa].copy()

    if is_all:
        real_barrel = advanced["barrel_pct_batting"].reindex(pool.index)
        pool["barrel_pct"] = real_barrel.where(real_barrel.notna(), pool["barrel_pct"])

    pool = pool.sort_index()
    pool.index.name = "batter_id"
    return pool


# Metric polarity: True = higher raw value is better (percentile follows raw
# rank), False = lower raw value is better (percentile is inverted), None =
# no inherent "better" direction, purely descriptive positioning.
PITCHER_POLARITY = {
    "whiff_pct": True, "k_pct": True, "chase_pct": True, "zone_whiff_pct": True,
    "velo": True, "perceived_velo": True, "bb_pct": False, "k_minus_bb_pct": True,
    "avg_exit_velo": False, "barrel_pct": False, "gb_pct": True,
    "extension": True, "pull_air_pct": False, "release_height": False, "fb_vaa": True,
    "zone_pct": True, "strike_pct": True, "swstr_pct": True, "zone_swing_pct": None,
    "ev_90th": False, "avg_la_95plus": True, "xwoba": False,
    "release_side": None, "spin_rate": None,
    "xfip": False, "first_pitch_strike_pct": True,
    "holding_runners": True,  # higher runs-prevented-via-running-game = better; replaces fb_pct here per direction, see pulls/annual/pull_pitcher_running_game.py
}
BATTER_POLARITY = {
    "xwoba": True, "xslg": True, "xba": True, "avg_exit_velo": True, "barrel_pct": True,
    "avg_launch_angle": None, "bat_speed": True, "chase_pct": False,
    "zone_whiff_pct": False, "k_pct": False, "bb_pct": True,
    "pull_air_pct": True, "gb_pct": False, "sprint_speed": True, "oaa": True,
    "swstr_pct": False, "zone_swing_pct": True, "ev_90th": True, "avg_la_95plus": True,
    "fb_pct": True, "zone_pct": None, "strike_pct": None, "whiff_pct": False,
    "swing_length": None, "attack_angle": None, "arm_strength": True,
    "swing_path_tilt": None, "attack_direction": None,
}


def player_percentiles(pool_df, player_id, polarity_map):
    """
    Given a league-wide metrics table (pool_df, indexed by player id) and one
    player_id in it, returns {metric: (raw_value, percentile_0_to_100)} for
    every metric in polarity_map that's present in pool_df.
    """
    if player_id not in pool_df.index:
        return {}
    out = {}
    for metric, higher_is_better in polarity_map.items():
        if metric not in pool_df.columns:
            continue
        series = pool_df[metric].dropna()
        if player_id not in series.index:
            continue
        raw_value = series.loc[player_id]
        pct = (series < raw_value).mean() * 100
        if higher_is_better is False:
            pct = 100 - pct
        out[metric] = (raw_value, pct)
    return out


def player_percentiles_allow_unqualified(qualified_pool, player_id, polarity_map, unqualified_pool=None):
    """
    Same as player_percentiles(), but for a player who DIDN'T clear the
    qualifying threshold (so isn't a row in `qualified_pool` at all):
    looks up their raw value from `unqualified_pool` (a separate,
    THRESHOLD-FREE pool - e.g. pitcher_percentile_metrics(conn, season,
    min_bf=0, ...) - built by the caller, not this function, since building
    it is a real league-wide query the caller may want to skip entirely for
    the common case of a qualified player), but ranks that raw value
    against `qualified_pool`'s population ONLY. Matches Baseball Savant's
    own convention: an unqualified player still gets a real percentile
    number (against the real qualified population), they're just excluded
    from DEFINING that population.

    THE BUG THIS GUARDS AGAINST, explicitly: naively concatenating the
    unqualified player's own row onto qualified_pool before ranking would
    make them count toward their own percentile's denominator - a small
    but real, and easy to introduce, self-reference. This function never
    constructs that concatenation; `series = qualified_pool[metric]` below
    is read directly off the untouched qualified-only pool, and the
    unqualified player's raw value is only ever the thing being LOCATED
    within it, never added to it.

    Returns ({metric: (raw_value, pct)}, is_qualified: bool). If player_id
    IS in qualified_pool, delegates straight to player_percentiles() and
    reports qualified=True (the normal path, unaffected by any of the
    above). If player_id is in neither pool, returns ({}, False).
    """
    if player_id in qualified_pool.index:
        return player_percentiles(qualified_pool, player_id, polarity_map), True
    if unqualified_pool is None or player_id not in unqualified_pool.index:
        return {}, False
    out = {}
    for metric, higher_is_better in polarity_map.items():
        if metric not in qualified_pool.columns or metric not in unqualified_pool.columns:
            continue
        raw_value = unqualified_pool[metric].get(player_id)
        if pd.isna(raw_value):
            continue
        series = qualified_pool[metric].dropna()  # the QUALIFIED population only - never touched by the unqualified player's own row
        pct = (series < raw_value).mean() * 100
        if higher_is_better is False:
            pct = 100 - pct
        out[metric] = (raw_value, pct)
    return out, False
