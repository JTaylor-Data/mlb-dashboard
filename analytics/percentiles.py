"""
League-wide percentile rankings, Savant-style. Computed once across every
qualified player this season, then a specific player is located within that
distribution. Percentiles are simple rank-based (0-100), not shrinkage- or
park-adjusted.

Pull Air% uses the community-standard hc_x/hc_y -> spray angle formula (the
same one behind most public spray-chart tools); Statcast doesn't publish an
official closed-form version of this either, same situation as Barrel%.
"""
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
# duplication - FIP_CONSTANT_APPROX also appears in pulls/pull_team_stats.py).
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

# Standard community constants for converting Statcast hc_x/hc_y into a spray
# angle in degrees (0 = straight up the middle, negative = left field side).
_SPRAY_X0, _SPRAY_Y0 = 125.42, 198.27


def compute_spray_angle(hc_x, hc_y):
    hc_x, hc_y = np.asarray(hc_x, dtype=float), np.asarray(hc_y, dtype=float)
    return np.degrees(np.arctan2(hc_x - _SPRAY_X0, _SPRAY_Y0 - hc_y)) * 0.75


def _add_shared_flags(df):
    df = df.copy()
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["is_strike"] = df["description"].isin(STRIKE_DESCRIPTIONS)
    df["in_zone"] = df["zone"].isin(IN_ZONE_CODES)
    df["out_zone"] = df["zone"].isin(OUT_ZONE_CODES)
    df["spray_angle"] = compute_spray_angle(df["hc_x"], df["hc_y"])
    df["is_pulled"] = np.where(df["stand"] == "R", df["spray_angle"] < 0, df["spray_angle"] > 0)
    df["is_air"] = df["bb_type"].isin(["fly_ball", "line_drive"])
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
    the season-wide pool.
    """
    if min_bf is None:
        min_bf = SAVANT_PA_PER_TEAM_GAME_PITCHER * _team_games_played(conn, season)

    where = "WHERE season = ?"
    params = (season,)
    if stand_filter and stand_filter != "ALL":
        where += " AND stand = ?"
        params += (stand_filter,)

    df = pd.read_sql(f"""
        SELECT pitcher_id, pitch_type, description, zone, events, stand, pitch_number,
               release_speed, effective_speed, release_extension, release_pos_z,
               release_pos_x, release_spin_rate, plate_z, arm_angle,
               launch_speed, launch_angle, bb_type, hc_x, hc_y,
               vy0, vz0, ay, az, woba_denom, estimated_woba_using_speedangle
        FROM pitches {where}
    """, conn, params=params)
    df = _add_shared_flags(df)
    df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
    is_fb = df["pitch_type"].isin(["FF", "FA", "SI"])
    league_hr_fb_rate = _league_hr_per_fb(conn, season)

    # Real season-total xFIP/Barrel% (MLB Stats API sabermetrics / Baseball
    # Savant - see pulls/pull_sabermetrics.py) have no opposing-batter-hand
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

    rate_pool = _rate_metrics_pool(df, "pitcher_id")

    # Vectorized replacement for the per-pitcher Python loop that used to
    # live here (velo/extension/xwoba/xfip/etc, one groupby-mean call per
    # pitcher via `for pid, group in df.groupby(...)`) - verified to
    # reproduce that loop's output exactly (float-tolerance, zero
    # mismatches across ALL/L/R splits) before replacing it. Same pattern
    # as _rate_metrics_pool above: precompute masked columns once, then one
    # groupby-aggregate pass instead of N per-pitcher pandas calls.
    d = df.copy()
    d["_fb_speed"] = d["release_speed"].where(is_fb)
    d["_fb_eff_speed"] = d["effective_speed"].where(is_fb)

    # FB VAA: (1) PRIMARY fastball only, not FF+SI blended together - a
    # pitcher who mixes both had his VAA averaged across two pitch shapes
    # that mean opposite things (flat vs steep is desirable for opposite
    # reasons), which muddies the number into something that isn't really
    # anyone's actual pitch. (2) Height-adjusted - VAA is measured at a
    # fixed distance from the plate (see compute_vaa's VAA_YF), not a fixed
    # HEIGHT, so two pitches with identical release/movement still read
    # different VAA if one crosses higher than the other (a ball still
    # rising toward its target has less time to have fallen) - a pitcher
    # who happens to have located that day's fastballs up in the zone reads
    # artificially flat, and vice versa. That's an EXECUTION effect (command
    # that day), not the delivery/movement TRAIT this metric is meant to
    # isolate for projecting future shape rather than grading past results.
    # Fixed with a real per-pitch-type regression adjustment: fit VAA's
    # actual league-wide slope against plate_z separately for each fastball
    # type (FF and SI have different typical shapes, so a shared slope
    # would blur them the same way blending the pitch types themselves
    # would), then shift every pitch's VAA to what it would have been at
    # that type's own league-average height before averaging per pitcher.
    _FB_TYPES = ("FF", "FA", "SI")
    vaa_height_slope = {}
    type_mean_z = {}
    for pt in _FB_TYPES:
        sub = d.loc[(d["pitch_type"] == pt) & d["vaa"].notna() & d["plate_z"].notna(), ["vaa", "plate_z"]]
        if len(sub) >= 50:
            vaa_height_slope[pt] = float(np.polyfit(sub["plate_z"], sub["vaa"], 1)[0])
            type_mean_z[pt] = float(sub["plate_z"].mean())
        else:
            vaa_height_slope[pt] = 0.0
            type_mean_z[pt] = float(sub["plate_z"].mean()) if len(sub) else np.nan
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
    # Side-channel, not a percentile metric itself (not in PITCHER_POLARITY,
    # so player_percentiles() ignores it) - lets the API layer show a
    # sinkerballer-specific note next to fb_vaa's percentile (steeper is
    # GOOD for a sinker; the percentile number itself stays a neutral,
    # un-flipped "how flat is this fastball vs. the league" rank either way).
    extra["fb_vaa_primary_type"] = primary_fb_type.reindex(extra.index)
    extra["extension"] = g["release_extension"].mean()
    extra["release_height"] = g["release_pos_z"].mean()
    extra["release_side"] = g["release_pos_x"].mean()
    extra["arm_angle"] = g["arm_angle"].mean()
    extra["spin_rate"] = g["release_spin_rate"].mean()
    # Real runs-prevented-via-running-game value (Baseball Savant, see
    # pulls/pull_pitcher_running_game.py) - season-total only, no pitch-level
    # fallback exists, so unlike xfip/barrel_pct below this is never
    # overridden by anything computed from `d` - it's just reindexed
    # straight from `advanced`, same as sprint_speed/oaa on the batter side.
    extra["holding_runners"] = advanced["running_game_runs_prevented"].reindex(extra.index)
    # sum(qualifying)/count(ALL qualifying rows), NOT .mean() - woba_denom==1
    # also covers walks/HBP/strikeouts (a real xwOBA credit under Statcast's
    # own convention) which have no estimated_woba_using_speedangle value,
    # so count(non-null) would undercount the denominator vs. the original
    # formula this replaces.
    xwoba_denom = g["_qualifying"].sum()
    extra["xwoba"] = _safe_div(g["_xwoba_q"].sum(), xwoba_denom)
    outs_sum = g["_outs"].sum()
    ip = outs_sum / 3
    fb_sum = g["_fb_batted"].sum()
    bb_hbp_sum = g["_bb_hbp"].sum()
    k_sum = g["_k"].sum()
    extra["xfip"] = _safe_div((13 * fb_sum * league_hr_fb_rate) + (3 * bb_hbp_sum) - (2 * k_sum), ip) + _FIP_CONSTANT_APPROX
    extra["first_pitch_strike_pct"] = g["_fp_strike"].mean()

    pool = rate_pool.join(extra)
    pool["k_minus_bb_pct"] = pool["k_pct"] - pool["bb_pct"]
    pool = pool[pool["n_pa"] >= min_bf].copy()

    if is_all:
        real_xfip = advanced["xfip"].reindex(pool.index)
        real_barrel = advanced["barrel_pct_pitching"].reindex(pool.index)
        pool["xfip"] = real_xfip.where(real_xfip.notna(), pool["xfip"])
        pool["barrel_pct"] = real_barrel.where(real_barrel.notna(), pool["barrel_pct"])

    pool.index.name = "pitcher_id"
    return pool


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
    """
    if min_pa is None:
        min_pa = SAVANT_PA_PER_TEAM_GAME_BATTER * _team_games_played(conn, season)

    where = "WHERE season = ?"
    params = (season,)
    if throws_filter and throws_filter != "ALL":
        where += " AND p_throws = ?"
        params += (throws_filter,)

    df = pd.read_sql(f"""
        SELECT batter_id, description, zone, events, stand,
               launch_speed, launch_angle, bb_type, hc_x, hc_y, bat_speed, swing_length, attack_angle,
               swing_path_tilt, woba_denom, estimated_woba_using_speedangle, estimated_slg_using_speedangle,
               estimated_ba_using_speedangle
        FROM pitches {where}
    """, conn, params=params)
    df = _add_shared_flags(df)

    advanced = pd.read_sql(
        "SELECT player_id, sprint_speed, oaa, barrel_pct_batting, arm_strength FROM player_advanced_stats WHERE season = ?",
        conn, params=(season,),
    ).set_index("player_id")
    # Real Barrel% (Baseball Savant, see pulls/pull_sabermetrics.py) has no
    # opposing-pitcher-hand split; only applies to the unsplit ALL view, same
    # reasoning as the pitcher side's xfip/barrel_pct override above.
    is_all = not throws_filter or throws_filter == "ALL"

    # Same hits-only, hand-adjusted landing-angle formula field_zone_pcts()
    # uses (negative=pull, positive=oppo for either batter hand) - computed
    # here too so it has a real percentile within the pool, not just a
    # standalone per-player number. Duplicated rather than imported since
    # player_page.py already imports FROM this module; keeping the direction
    # one-way avoids a circular import.
    #
    # Vectorized ONCE across the whole pool (not recomputed per batter
    # inside the groupby loop below) - a real, measured slowdown: the
    # earlier version called compute_spray_angle/np.where separately for
    # each of 600+ batters inside the loop, which meaningfully dragged out
    # every cold-cache pool build (server startup warmup AND "Refresh
    # data" both call this function for every split). Same math, just done
    # as one pandas groupby instead of N Python-level calls.
    _HIT_EVENTS = {"single", "double", "triple", "home_run"}
    hits_mask = df["events"].isin(_HIT_EVENTS) & df["hc_x"].notna() & df["hc_y"].notna()
    hits_df = df[hits_mask]
    hit_angle = compute_spray_angle(hits_df["hc_x"], hits_df["hc_y"])
    hit_hand_adj = np.where(hits_df["stand"] == "R", hit_angle, -hit_angle)
    attack_direction_by_batter = pd.Series(hit_hand_adj, index=hits_df.index).groupby(hits_df["batter_id"]).mean()

    rate_pool = _rate_metrics_pool(df, "batter_id")

    # Vectorized replacement for the per-batter Python loop that used to
    # live here - verified to reproduce that loop's output exactly
    # (float-tolerance, zero mismatches across ALL/L/R splits) before
    # replacing it. Same pattern as the pitcher side above.
    d = df.copy()
    qualifying = d["woba_denom"] == 1
    d["_qualifying"] = qualifying
    d["_xwoba_q"] = d["estimated_woba_using_speedangle"].where(qualifying)
    d["_xslg_q"] = d["estimated_slg_using_speedangle"].where(qualifying)
    d["_xba_q"] = d["estimated_ba_using_speedangle"].where(qualifying)
    # fast_swing_bat_speed's per-batter 10th-percentile threshold, computed
    # once via groupby.transform (pandas' own optimized per-group reduction)
    # instead of calling the function fresh per batter.
    d["_bs_q10"] = d.groupby("batter_id")["bat_speed"].transform(lambda s: s.quantile(0.10))
    bs_keep = ((d["bat_speed"] >= d["_bs_q10"]) | ((d["bat_speed"] >= 60) & (d["launch_speed"] >= 90))) & d["bat_speed"].notna()
    d["_bs_kept"] = d["bat_speed"].where(bs_keep)

    g = d.groupby("batter_id")
    extra = pd.DataFrame(index=g.size().index)
    # sum(qualifying)/count(ALL qualifying rows) - see the pitcher side's
    # xwoba comment for why this isn't .mean() (walks/HBP/Ks count toward
    # woba_denom but have no estimated_slg/ba_using_speedangle value at all,
    # only estimated_woba does).
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

    pool = rate_pool.join(extra)
    pool = pool[pool["n_pa"] >= min_pa].copy()

    if is_all:
        real_barrel = advanced["barrel_pct_batting"].reindex(pool.index)
        pool["barrel_pct"] = real_barrel.where(real_barrel.notna(), pool["barrel_pct"])

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
    "holding_runners": True,  # higher runs-prevented-via-running-game = better; replaces fb_pct here per direction, see pulls/pull_pitcher_running_game.py
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
