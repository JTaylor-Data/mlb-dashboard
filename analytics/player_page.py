"""
Data-assembly helpers for the individual player page (pitcher + batter
views). Pure functions, no Streamlit - kept separate from the page itself so
this logic is testable without running the app.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from analytics.boxscore import batter_daily_stats, fetch_boxscore, pitcher_daily_stats
from analytics.metrics import classify_pitcher_roles, compute_wrc_plus, league_average_woba
from analytics.percentiles import fast_swing_bat_speed
from analytics.splits import (
    SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS, IN_ZONE_CODES, OUT_ZONE_CODES,
    STRIKEOUT_EVENTS, WALK_EVENTS, HBP_EVENTS, _pa_outcome_metrics, _pitch_rate_metrics, _barrel_rate,
)
from analytics.stuff_features import EXCLUDED_PITCH_TYPES

# Outs recorded per PA-ending event. Statcast has no direct "outs on this
# play" field, so this is a manual mapping - events not listed (hits, walks,
# HBP, fielder's choice with no out, catcher's interference) record 0 outs.
OUTS_PER_EVENT = {
    "strikeout": 1, "strikeout_double_play": 2, "field_out": 1, "force_out": 1,
    "grounded_into_double_play": 2, "double_play": 2, "triple_play": 3,
    "sac_fly": 1, "sac_fly_double_play": 2, "sac_bunt": 1, "sac_bunt_double_play": 2,
    "fielders_choice_out": 1,
}

# FIP needs a constant normally calibrated so league FIP matches league ERA -
# we have no earned-run data at all (Statcast pitch data doesn't carry official
# scorer rulings), so this is a fixed historical approximation, not derived
# from this dataset. Recent MLB seasons have run close to this value.
FIP_CONSTANT_APPROX = 3.10


def innings_pitched(pa_df):
    """Returns (ip_display_string, total_outs) from a PA-ending slice of
    pitches (events IS NOT NULL). Display string uses baseball's .1/.2
    notation (thirds of an inning), not true decimal."""
    outs = int(pa_df["events"].map(OUTS_PER_EVENT).fillna(0).sum())
    full_innings, remainder = divmod(outs, 3)
    return f"{full_innings}.{remainder}", outs


def league_hr_per_fb(conn, season):
    row = conn.execute("""
        SELECT
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END),
            SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END)
        FROM pitches
        WHERE season = ?
    """, (season,)).fetchone()
    hr, fb = row
    return hr / fb if fb else np.nan


def compute_xfip(pa_df, league_hr_fb_rate):
    outs = pa_df["events"].map(OUTS_PER_EVENT).fillna(0).sum()
    ip = outs / 3
    if ip == 0:
        return np.nan
    fb = (pa_df["bb_type"] == "fly_ball").sum() if "bb_type" in pa_df.columns else 0
    bb_hbp = pa_df["events"].isin(WALK_EVENTS | HBP_EVENTS).sum()
    k = pa_df["events"].isin(STRIKEOUT_EVENTS).sum()
    return ((13 * fb * league_hr_fb_rate) + (3 * bb_hbp) - (2 * k)) / ip + FIP_CONSTANT_APPROX


def _add_flags(df):
    df = df.copy()
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["in_zone"] = df["zone"].isin(IN_ZONE_CODES)
    df["out_zone"] = df["zone"].isin(OUT_ZONE_CODES)
    return df


def _add_arm_side_hb(df):
    """Horizontal break in inches, sign-flipped for RHP so positive = arm
    side for every pitcher regardless of hand - matches Baseball Savant's
    documented convention ('positive = toward the pitcher's arm side').
    Statcast's raw pfx_x is a fixed field-relative coordinate, not
    handedness-adjusted, so without this a RHP's arm-side run and a LHP's
    arm-side run show with opposite signs even though they mean the same
    thing."""
    df = df.copy()
    sign = np.where(df["p_throws"] == "R", -1, 1)
    df["hb"] = df["pfx_x"] * 12 * sign
    return df


LEAGUE_AVG_BATTER_ROW_COLUMNS = [
    "description", "zone", "events", "launch_speed", "launch_angle",
    "woba_denom", "woba_value", "estimated_woba_using_speedangle",
]


def league_average_batter_row(conn, season, throws_filter=None):
    """League-wide average of the batter pitch-type table's metrics
    (blended across all pitch types), respecting the same opponent-throws
    filter."""
    where, params = "WHERE season = ?", (season,)
    if throws_filter and throws_filter != "ALL":
        where, params = "WHERE season = ? AND p_throws = ?", (season, throws_filter)
    # `SELECT *` was pulling every one of pitches' ~45 columns (a season is
    # ~700K rows) just to compute 8 columns' worth of metrics - confirmed
    # the real, single biggest cost in a "Refresh data" click (~19s of a
    # ~61s warmup) via direct profiling, not assumed. Only what _add_flags/
    # _pa_outcome_metrics/_pitch_rate_metrics/_barrel_rate actually read.
    cols = ", ".join(LEAGUE_AVG_BATTER_ROW_COLUMNS + (["p_throws"] if "p_throws" in where else []))
    df = pd.read_sql(f"SELECT {cols} FROM pitches {where}", conn, params=params)
    df = _add_flags(df)
    pa_group = df[df["events"].notna()]
    row = _pa_outcome_metrics(pa_group)
    _, whiff_pct, _ = _pitch_rate_metrics(df)
    row["whiff_pct"] = whiff_pct
    row["barrel_pct"] = _barrel_rate(pa_group)
    del row["n_pa"]
    return row


# Knuckle curve is treated as the same pitch as a regular curveball everywhere
# in this page - not worth the distinction for display purposes.
PITCH_TYPE_MERGE = {"KC": "CU"}


def _load(conn, season, id_col, player_id, extra_filter_col=None, extra_filter_val=None, columns=None):
    where = ["season = ?", f"{id_col} = ?"]
    params = [season, player_id]
    if extra_filter_col and extra_filter_val and extra_filter_val != "ALL":
        where.append(f"{extra_filter_col} = ?")
        params.append(extra_filter_val)
    cols = ", ".join(columns) if columns else "*"
    df = pd.read_sql(f"SELECT {cols} FROM pitches WHERE {' AND '.join(where)}", conn, params=params)
    if "pitch_type" in df.columns:
        df["pitch_type"] = df["pitch_type"].replace(PITCH_TYPE_MERGE)
    return df


# ---------------------------------------------------------------- Pitcher --

_PITCHER_PT_COLS = [
    "pitch_type", "description", "zone", "events",
    "woba_denom", "woba_value", "estimated_woba_using_speedangle",
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "p_throws",
]


def pitcher_pitch_table(conn, season, pitcher_id, stand_filter=None, stuff_plus_table=None):
    """One row per pitch type: usage%, OPS against, velo, spin, IVB, HB,
    zone%, chase%, whiff%, K%, stuff_plus (if a precomputed
    stuff_plus_table is passed - see stuff_model.compute_stuff_plus).
    Extension lives in the percentiles panel now, not here."""
    df = _load(conn, season, "pitcher_id", pitcher_id, "stand", stand_filter, columns=_PITCHER_PT_COLS)
    if df.empty:
        return pd.DataFrame()
    # Excludes the same non-competitive pitch types Stuff+ itself excludes
    # (EXCLUDED_PITCH_TYPES) - this table is the one that carries stuff_plus
    # per row, so a pitchout/eephus/unclassified row would show real
    # velo/spin/etc stats next to a stuff_plus that no longer exists for
    # that type (it's not in stuff_plus_table's index anymore) rather than
    # just not appearing at all. Deliberately NOT applied to _load() itself
    # - pitcher_counting_stats also calls _load() and real counting stats
    # (K/BB/HR/IP) must still include every real pitch, eephus included (an
    # eephus strikeout is still a real strikeout).
    df = df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)]
    if df.empty:
        return pd.DataFrame()
    df = _add_flags(df)
    df = _add_arm_side_hb(df)
    total = len(df)

    rows = []
    for pt, group in df.groupby("pitch_type"):
        usage_pct = len(group) / total * 100
        if usage_pct < 1.0:
            continue
        swings = group["is_swing"].sum()
        out_zone_n = group["out_zone"].sum()
        pa = group[group["events"].notna()]
        k = pa["events"].isin(STRIKEOUT_EVENTS).sum()
        rows.append(dict(
            pitch_type=pt,
            usage_pct=usage_pct,
            n_pitches=len(group),
            ops=_pa_outcome_metrics(pa)["OPS"],
            velo=group["release_speed"].mean(),
            spin=group["release_spin_rate"].mean(),
            ivb=group["pfx_z"].mean() * 12,
            hb=group["hb"].mean(),
            zone_pct=group["in_zone"].mean() * 100,
            chase_pct=(group["is_swing"] & group["out_zone"]).sum() / out_zone_n * 100 if out_zone_n else np.nan,
            whiff_pct=group["is_whiff"].sum() / swings * 100 if swings else np.nan,
            k_pct=k / len(pa) * 100 if len(pa) else np.nan,
        ))
    out = pd.DataFrame(rows).set_index("pitch_type")

    if stuff_plus_table is not None and pitcher_id in stuff_plus_table.index.get_level_values("pitcher_id"):
        s = stuff_plus_table.xs(pitcher_id, level="pitcher_id")[["stuff_plus"]]
        out = out.join(s)

    return out.sort_values("usage_pct", ascending=False)


def pitcher_league_avg_row(conn, season, stand_filter=None):
    """League-wide (every qualifying pitch that season/split, not just this
    pitcher's) zone%/chase%/whiff%/K% - the summary row at the bottom of the
    pitch table. Deliberately NOT velo/spin/IVB/HB - those are shape
    properties that only make sense compared pitch-type-to-pitch-type (a
    league-average curveball spin rate isn't a meaningful bar for a
    fastball's own row to clear), so those four columns are simply absent
    here and render as "-" for this row rather than a misleading blend.
    (A prior version of this row instead used the pitcher's OWN blended
    average - reverted per direct instruction back to a real league-wide
    reference point.)"""
    where, params = "WHERE season = ?", (season,)
    if stand_filter and stand_filter != "ALL":
        where, params = "WHERE season = ? AND stand = ?", (season, stand_filter)
    # Excludes the same non-competitive pitch types Stuff+ itself excludes
    # (analytics/stuff_features.py's EXCLUDED_PITCH_TYPES) - this is the
    # league-average REFERENCE row for the by-pitch-type table Stuff+
    # scores live in, so it should reflect the same population, not a
    # wider one that still includes pitchouts/eephus/unclassified pitches.
    excl_placeholders = ",".join("?" * len(EXCLUDED_PITCH_TYPES))
    where += f" AND pitch_type NOT IN ({excl_placeholders})"
    params = params + tuple(EXCLUDED_PITCH_TYPES)
    # Same SELECT * fix as league_average_batter_row above - this is the
    # pitcher-side equivalent, same full-season/every-column pull, same
    # real cost (confirmed live: this exact query was the stall behind the
    # player page's loading bar getting stuck on "Building pitch table").
    # Only description/zone/events feed _add_flags + the metrics below.
    cols = ", ".join(["description", "zone", "events"] + (["stand"] if "stand" in where else []))
    df = pd.read_sql(f"SELECT {cols} FROM pitches {where}", conn, params=params)
    if df.empty:
        return {}
    df = _add_flags(df)
    swings = df["is_swing"].sum()
    out_zone_n = df["out_zone"].sum()
    pa = df[df["events"].notna()]
    k = pa["events"].isin(STRIKEOUT_EVENTS).sum()
    return dict(
        zone_pct=df["in_zone"].mean() * 100,
        chase_pct=(df["is_swing"] & df["out_zone"]).sum() / out_zone_n * 100 if out_zone_n else np.nan,
        whiff_pct=df["is_whiff"].sum() / swings * 100 if swings else np.nan,
        k_pct=k / len(pa) * 100 if len(pa) else np.nan,
    )


def movement_plot_data(conn, season, pitcher_id, stand_filter=None):
    """Arm-side-adjusted horizontal break (positive = arm side regardless of
    pitcher hand) - matches Baseball Savant's convention and the main pitch
    table. A prior version plotted raw, unadjusted field-relative break
    instead (a direct request at the time, so a LHP's chart would mirror a
    same-shaped RHP's) - reverted after that turned out to make a RHP's
    chart look backwards compared to Savant's own real chart for the same
    pitcher (raw pfx_x needs the sign flip for RHP specifically; LHP was
    fine either way, which is why only righties looked wrong)."""
    df = _load(conn, season, "pitcher_id", pitcher_id, "stand", stand_filter,
               columns=["pitch_type", "pfx_x", "pfx_z", "p_throws"])
    # Same non-competitive pitch types excluded from the pitch table
    # (EXCLUDED_PITCH_TYPES = pitchouts/eephus/unclassified) - these aren't
    # real pitch shapes worth plotting a movement point for.
    df = df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)]
    df = df[df["pfx_x"].notna() & df["pfx_z"].notna()].copy()
    if df.empty:
        return df[["pitch_type"]].iloc[:0]
    total_n = len(df)
    keep = {pt for pt, g in df.groupby("pitch_type") if len(g) / total_n * 100 >= 1.0}
    df = df[df["pitch_type"].isin(keep)]
    df = _add_arm_side_hb(df)
    df["HB"] = df["hb"]
    df["IVB"] = df["pfx_z"] * 12
    return df[["pitch_type", "HB", "IVB"]]


def pitch_usage_by_stand(conn, season, pitcher_id):
    """Usage% of each pitch type against LHH and against RHH separately
    (e.g. '46% of pitches to lefties were the fastball') - independent of
    the page's opponent-bats toggle, since showing both sides side by side
    is the whole point of this widget. Ordered by overall usage, matching
    the pitch table's own row order."""
    df = _load(conn, season, "pitcher_id", pitcher_id)
    if df.empty:
        return []

    overall_order = df["pitch_type"].value_counts().index.tolist()
    totals = df.groupby("stand").size()
    counts = df.groupby(["pitch_type", "stand"]).size()

    out = []
    for pt in overall_order:
        row = {"pitch_type": pt, "vs_l": 0.0, "vs_r": 0.0}
        for stand in ("L", "R"):
            total = totals.get(stand, 0)
            if total and (pt, stand) in counts.index:
                pct = counts.loc[(pt, stand)] / total * 100
                row["vs_l" if stand == "L" else "vs_r"] = float(pct)
        out.append(row)
    return out


def pitcher_rubber_position(conn, season, pitcher_id):
    """Average RAW (not arm-side-adjusted) release_pos_x, in feet - this
    answers "which physical side of the rubber does he stand on," a fixed
    field-relative fact everyone shares regardless of throwing hand, unlike
    the arm-side-adjusted release_pos_x_as used elsewhere on this page for
    movement/deception purposes (that answers a different question: how far
    toward HIS OWN arm side his release point is)."""
    row = conn.execute(
        "SELECT AVG(release_pos_x) FROM pitches WHERE season = ? AND pitcher_id = ? AND release_pos_x IS NOT NULL",
        (season, pitcher_id),
    ).fetchone()
    return row[0]


def pitcher_arm_angle(conn, season, pitcher_id):
    """Season-wide average arm angle (degrees), independent of the L/R
    toggle - used as a fixed reference line on the movement plot, not a
    percentile (there's no inherent 'better' direction for arm slot)."""
    row = conn.execute(
        "SELECT AVG(arm_angle) FROM pitches WHERE season = ? AND pitcher_id = ? AND arm_angle IS NOT NULL",
        (season, pitcher_id),
    ).fetchone()
    return row[0]


ARM_ANGLE_PEER_TOLERANCE_DEG = 3

def all_pitcher_arm_angles(conn, season):
    """Every pitcher's own season-average arm angle - pulled out of
    similar_arm_angle_movement (below) into its own function specifically
    so it can be cached ONCE PER SEASON at the router level, not recomputed
    on every distinct pitcher's first page view. This query doesn't depend
    on which pitcher is asking, but used to run fresh inside a function
    keyed by (season, pitcher_id) - a real, confirmed cost (~20s live,
    scanning the whole season's pitches to GROUP BY pitcher_id) paid again
    for every NEW pitcher a user looked at, even though the answer for
    "every pitcher's arm angle this season" never changes within that
    season."""
    return pd.read_sql("""
        SELECT pitcher_id, AVG(arm_angle) AS arm_angle
        FROM pitches WHERE season = ? AND arm_angle IS NOT NULL
        GROUP BY pitcher_id
    """, conn, params=(season,))


def similar_arm_angle_movement(conn, season, pitcher_id, arm_angles=None):
    """Real league-wide average movement (HB/IVB), per pitch type, among
    OTHER pitchers whose own season-average arm angle is within +/-3deg of
    this pitcher's - the movement plot's "average movement" reference
    circle used to be this pitcher's OWN average, which is close to
    tautological (a pitcher's own points are, by definition, centered near
    their own mean); this instead answers "how does this pitch move
    compared to other pitchers who release it from a similar slot," a real
    delivery-comparable peer group rather than a self-reference. Same
    arm-side-adjusted HB convention as movement_plot_data (positive = arm
    side for either hand).

    `arm_angles` (all_pitcher_arm_angles' output) can be passed in by a
    caller that already has it cached season-wide; computed fresh here
    if omitted, so this function still works standalone."""
    my_arm_angle = pitcher_arm_angle(conn, season, pitcher_id)
    if my_arm_angle is None:
        return {}

    if arm_angles is None:
        arm_angles = all_pitcher_arm_angles(conn, season)
    peers = arm_angles.loc[
        (arm_angles["pitcher_id"] != pitcher_id)
        & ((arm_angles["arm_angle"] - my_arm_angle).abs() <= ARM_ANGLE_PEER_TOLERANCE_DEG),
        "pitcher_id",
    ]
    if peers.empty:
        return {}

    placeholders = ",".join("?" * len(peers))
    df = pd.read_sql(f"""
        SELECT pitcher_id, pitch_type, pfx_x, pfx_z, p_throws
        FROM pitches
        WHERE season = ? AND pitcher_id IN ({placeholders}) AND pfx_x IS NOT NULL AND pfx_z IS NOT NULL
    """, conn, params=(season, *peers.tolist()))
    if df.empty:
        return {}
    df["pitch_type"] = df["pitch_type"].replace(PITCH_TYPE_MERGE)
    df = _add_arm_side_hb(df)
    df["IVB"] = df["pfx_z"] * 12

    g = df.groupby("pitch_type")
    out = pd.DataFrame({
        "hb": g["hb"].mean(),
        "ivb": g["IVB"].mean(),
        "n_pitchers": g["pitcher_id"].nunique(),
        "n_pitches": g.size(),
    })
    return out.to_dict("index")


_COUNTING_COLS = ["events", "woba_denom", "woba_value", "estimated_woba_using_speedangle", "game_id"]


def pitcher_counting_stats(conn, season, pitcher_id, stand_filter=None, games_started=None, league_hr_fb_rate=None):
    df = _load(conn, season, "pitcher_id", pitcher_id, "stand", stand_filter, columns=_COUNTING_COLS)
    pa = df[df["events"].notna()]
    outcomes = _pa_outcome_metrics(pa)
    ip_display, outs = innings_pitched(pa)
    xfip = compute_xfip(pa, league_hr_fb_rate) if league_hr_fb_rate is not None else np.nan
    return dict(
        n_pitches=len(df),
        n_pa=len(pa),
        k=int(pa["events"].isin(STRIKEOUT_EVENTS).sum()),
        bb=int(pa["events"].isin(WALK_EVENTS).sum()),
        hr=int((pa["events"] == "home_run").sum()),
        ip=ip_display,
        gs=games_started,
        xfip=xfip,
        OPS=outcomes["OPS"],
        wOBA=outcomes["wOBA"],
        xwOBA=outcomes["xwOBA"],
        babip=outcomes["babip"],
    )


# ----------------------------------------------------------------- Batter --

_BATTER_PT_COLS = [
    "pitch_type", "description", "zone", "events",
    "woba_denom", "woba_value", "estimated_woba_using_speedangle",
    "launch_speed", "launch_angle",
]


def batter_pitch_type_table(conn, season, batter_id, throws_filter=None):
    """Batter performance vs. each pitch type, same metric shape as the
    splits tab: OPS, wOBA, xwOBA, K%, whiff%, Barrel%."""
    df = _load(conn, season, "batter_id", batter_id, "p_throws", throws_filter, columns=_BATTER_PT_COLS)
    if df.empty:
        return pd.DataFrame()
    df = _add_flags(df)

    total_pitches = len(df)
    rows = []
    for pt, group in df.groupby("pitch_type"):
        if len(group) / total_pitches * 100 < 1.0:
            continue
        pa_group = group[group["events"].notna()]
        row = _pa_outcome_metrics(pa_group)
        _, whiff_pct, _ = _pitch_rate_metrics(group)
        row.update(pitch_type=pt, n_pitches=len(group),
                    whiff_pct=whiff_pct, barrel_pct=_barrel_rate(pa_group))
        del row["n_pa"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).set_index("pitch_type").sort_values("n_pitches", ascending=False)
    total = result["n_pitches"].sum()
    result["pct_faced"] = result["n_pitches"] / total if total > 0 else 0.0
    return result


HIT_EVENTS = {"single", "double", "triple", "home_run"}
PULL_OPPO_THRESHOLD_DEG = 5   # anything closer to 0 than this counts as "Spray Hitter"


def field_zone_pcts(conn, season, batter_id, throws_filter=None):
    """HITS-ONLY (outs are deliberately not shown or counted - this chart is
    about where his hits go): % landing in LF/CF/RF (field-absolute spray
    angle), each individual hit's real plotted position + whether it was a
    home run, and a single HAND-ADJUSTED pull/oppo angle (negative=pull,
    positive=oppo, for either batter hand - flips the raw field-absolute
    angle the same way `is_pulled` already does in percentiles.py, just kept
    continuous instead of collapsed to a boolean). That angle is the one
    number the Attack Direction gauge's dot/label AND this chart's own
    highlighted wedge both read from now, so they can't visually disagree
    with each other the way independently-computed metrics used to.

    `locations[i]` is `{x, y, is_hr}`: x/y are hc_x/hc_y converted to
    field-relative feet with home plate at the origin (same transform
    compute_spray_angle already uses)."""
    from analytics.percentiles import _SPRAY_X0, _SPRAY_Y0, compute_spray_angle

    where, params = "WHERE season = ? AND batter_id = ? AND hc_x IS NOT NULL AND hc_y IS NOT NULL", (season, batter_id)
    if throws_filter and throws_filter != "ALL":
        where += " AND p_throws = ?"
        params += (throws_filter,)

    df = pd.read_sql(f"SELECT hc_x, hc_y, events, stand FROM pitches {where}", conn, params=params)
    df = df[df["events"].isin(HIT_EVENTS)]
    if df.empty:
        return dict(lf=33.3, cf=33.3, rf=33.3, locations=[], pull_angle=None,
                    highlight_side=None, highlight_opacity=0.0)

    angle = compute_spray_angle(df["hc_x"], df["hc_y"])
    n = len(angle)
    lf = (angle < -15).sum() / n * 100
    rf = (angle > 15).sum() / n * 100

    hand_adj = np.where(df["stand"] == "R", angle, -angle)
    pull_angle = float(np.mean(hand_adj))
    bat_hand = df["stand"].mode().iloc[0] if not df["stand"].mode().empty else "R"
    if abs(pull_angle) <= PULL_OPPO_THRESHOLD_DEG:
        highlight_side, highlight_opacity = None, 0.0
    elif pull_angle < 0:   # pull side
        highlight_side = "lf" if bat_hand == "R" else "rf"
        highlight_opacity = min(1.0, abs(pull_angle) / 45)
    else:                  # oppo side
        highlight_side = "rf" if bat_hand == "R" else "lf"
        highlight_opacity = min(1.0, abs(pull_angle) / 45)

    x = df["hc_x"] - _SPRAY_X0
    y = _SPRAY_Y0 - df["hc_y"]
    locations = [
        {"x": round(float(xi), 1), "y": round(float(yi), 1), "is_hr": ev == "home_run", "event": ev}
        for xi, yi, ev in zip(x, y, df["events"])
    ]
    return dict(lf=lf, cf=100 - lf - rf, rf=rf, locations=locations, pull_angle=pull_angle,
                highlight_side=highlight_side, highlight_opacity=highlight_opacity)


def _game_opponent(conn, game_id, team_id):
    """Same pattern as top_performers.py's own _opponent() - not imported
    directly since that module isn't meant to be a shared library, this is a
    small, cheap query worth just repeating."""
    row = conn.execute("SELECT home_team, away_team FROM games WHERE game_id = ?", (game_id,)).fetchone()
    if row is None:
        return ""
    home, away = int(row[0]), int(row[1])
    is_home = team_id == home
    opp_id = away if is_home else home
    abbrev = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (opp_id,)).fetchone()
    return (f"{'vs' if is_home else '@'} {abbrev[0]}") if abbrev else ""


def player_last_n_games(conn, season, player_id, role, n=5):
    """Last n games with real, live box-score stats (RBI/SB/ER aren't
    derivable from Statcast alone - same reasoning as boxscore.py's own
    module docstring), most recent first.

    Starters: STARTS only, not every relief appearance - a pitcher started a
    game if their own earliest at_bat_number in it is 1 (the literal first
    plate appearance of the game), true for every real start and false for
    a reliever's own outing that same game.
    Relievers: every appearance, no start filter - a reliever's "last 5
    outings" naturally means his last 5 times pitching, not his last 5
    starts (he may have none).
    Batters: any game with a real plate appearance, no filter."""
    team_row = conn.execute("SELECT team_id FROM players WHERE player_id = ?", (player_id,)).fetchone()
    team_id = int(team_row[0]) if team_row and team_row[0] is not None else None

    if role == "pitcher":
        roles = classify_pitcher_roles(conn, season)
        is_starter = player_id in roles.index and roles.loc[player_id, "role"] == "SP"
        if is_starter:
            rows = conn.execute("""
                SELECT game_id, game_date, MIN(at_bat_number) AS first_ab
                FROM pitches WHERE season = ? AND pitcher_id = ?
                GROUP BY game_id HAVING first_ab = 1
                ORDER BY game_date DESC LIMIT ?
            """, (season, player_id, n)).fetchall()
        else:
            rows = conn.execute("""
                SELECT DISTINCT game_id, game_date FROM pitches
                WHERE season = ? AND pitcher_id = ?
                ORDER BY game_date DESC LIMIT ?
            """, (season, player_id, n)).fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT game_id, game_date FROM pitches
            WHERE season = ? AND batter_id = ?
            ORDER BY game_date DESC LIMIT ?
        """, (season, player_id, n)).fetchall()

    # fetch_boxscore is a blocking network call (no `conn` involved at all -
    # see its own module docstring/cache) and was previously run once per
    # game, strictly sequentially - confirmed directly as the single
    # largest remaining per-player cost on the player page (~9s for 5
    # games, since nothing here can be pre-warmed for an arbitrary future
    # player). Fetching all N games concurrently (ThreadPoolExecutor - pure
    # I/O, the GIL releases during each requests.get) cuts that to roughly
    # the slowest single request instead of the sum of all of them. The
    # DB-touching part of the loop (_game_opponent, which uses `conn`)
    # stays sequential on the main thread afterward - sqlite3 connections
    # aren't safely shared across threads, and that part was never the
    # slow part anyway.
    game_ids = [game_id for game_id, *_ in rows]
    boxscores = {}
    if game_ids:
        with ThreadPoolExecutor(max_workers=min(len(game_ids), 5)) as pool:
            futures = {pool.submit(fetch_boxscore, gid): gid for gid in game_ids}
            for future in futures:
                gid = futures[future]
                try:
                    boxscores[gid] = future.result()
                except Exception:
                    boxscores[gid] = None  # a network hiccup on one game shouldn't blank the whole card

    out = []
    for game_id, game_date, *_ in rows:
        bx = boxscores.get(game_id)
        if bx is None:
            continue
        stats = (pitcher_daily_stats(bx, player_id) if role == "pitcher" else batter_daily_stats(bx, player_id))
        if not stats:
            continue
        out.append(dict(
            game_id=game_id, date=game_date,
            opponent=_game_opponent(conn, game_id, team_id) if team_id is not None else "",
            stats=stats,
        ))
    return out


_SWING_COLS = ["bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt", "launch_speed"]


def swing_profile(conn, season, batter_id, throws_filter=None):
    df = _load(conn, season, "batter_id", batter_id, "p_throws", throws_filter, columns=_SWING_COLS)
    df = df[df["bat_speed"].notna()]
    cols = ["bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt"]
    result = df[cols].mean().to_dict()
    result["bat_speed"] = fast_swing_bat_speed(df["bat_speed"], df["launch_speed"])
    return result


def league_average_swing_profile(conn, season):
    """Same shape as swing_profile() but league-wide (every swing with a
    bat_speed reading, no player filter) - the comparison point for the
    swing gauges on the batter page, same idea as league_average_batter_row
    for the pitch-type table."""
    df = pd.read_sql("SELECT bat_speed, swing_length, attack_angle, attack_direction, "
                      "swing_path_tilt, launch_speed FROM pitches WHERE season = ? AND bat_speed IS NOT NULL",
                      conn, params=(season,))
    cols = ["swing_length", "attack_angle", "attack_direction", "swing_path_tilt"]
    result = df[cols].mean().to_dict()
    result["bat_speed"] = fast_swing_bat_speed(df["bat_speed"], df["launch_speed"])
    return result


def batter_wrc_plus(conn, season, batter_id, throws_filter=None, league_woba=None):
    if league_woba is None:
        league_woba = league_average_woba(conn, season)
    df = _load(conn, season, "batter_id", batter_id, "p_throws", throws_filter)
    pa = df[df["events"].notna()]
    outcomes = _pa_outcome_metrics(pa)
    return compute_wrc_plus(outcomes["wOBA"], league_woba)


def batter_counting_stats(conn, season, batter_id, throws_filter=None):
    df = _load(conn, season, "batter_id", batter_id, "p_throws", throws_filter, columns=_COUNTING_COLS)
    pa = df[df["events"].notna()]
    outcomes = _pa_outcome_metrics(pa)
    return dict(
        n_pitches=len(df),
        n_pa=len(pa),
        gp=df["game_id"].nunique(),
        k=int(pa["events"].isin(STRIKEOUT_EVENTS).sum()),
        bb=int(pa["events"].isin(WALK_EVENTS).sum()),
        hr=int((pa["events"] == "home_run").sum()),
        OPS=outcomes["OPS"],
        wOBA=outcomes["wOBA"],
        xwOBA=outcomes["xwOBA"],
    )


def batter_rolling_xwoba(conn, season, batter_id, windows=(50, 100, 250)):
    """Rolling xwOBA for a batter over the given PA-count windows.

    Returns a list of {pa_index, date, <window>: xwoba_value} dicts, one per
    qualifying PA (woba_denom == 1), ordered chronologically. Each window's
    value is NaN until enough PAs have accumulated.
    """
    df = pd.read_sql(
        """
        SELECT game_date, at_bat_number, pitch_number,
               events, woba_denom, estimated_woba_using_speedangle
        FROM pitches
        WHERE season = ? AND batter_id = ?
        ORDER BY game_date, at_bat_number, pitch_number
        """,
        conn,
        params=(season, batter_id),
    )
    if df.empty:
        return []

    # One row per PA (last pitch of the AB)
    pa = (
        df[df["events"].notna()]
        .groupby(["game_date", "at_bat_number"], sort=False)
        .last()
        .reset_index()
    )
    pa = pa.sort_values(["game_date", "at_bat_number"]).reset_index(drop=True)

    # Qualifying PAs for xwOBA
    qualifying_mask = pa["woba_denom"] == 1
    qpa = pa[qualifying_mask].copy()
    qpa["_xw"] = qpa["estimated_woba_using_speedangle"].astype(float)
    qpa = qpa.reset_index(drop=True)

    if qpa.empty:
        return []

    # Build one point per qualifying PA with rolling means for each window
    points = []
    xw_vals = qpa["_xw"].values
    dates = qpa["game_date"].values
    total = len(qpa)

    for i in range(total):
        pt = {"pa_index": i + 1, "date": str(dates[i])}
        for w in windows:
            if i + 1 >= w:
                pt[str(w)] = float(np.mean(xw_vals[i + 1 - w: i + 1]))
            else:
                pt[str(w)] = None
        points.append(pt)

    return points
