"""
Data-assembly helpers for the individual player page (pitcher + batter
views). Pure functions, no Streamlit - kept separate from the page itself so
this logic is testable without running the app.
"""
import numpy as np
import pandas as pd

from metrics import compute_wrc_plus, league_average_woba
from percentiles import fast_swing_bat_speed
from splits import (
    SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS, IN_ZONE_CODES, OUT_ZONE_CODES,
    STRIKEOUT_EVENTS, WALK_EVENTS, HBP_EVENTS, _pa_outcome_metrics, _pitch_rate_metrics, _barrel_rate,
)

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
    fb = (pa_df["bb_type"] == "fly_ball").sum()
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


def league_average_pitcher_row(conn, season, stand_filter=None):
    """League-wide average of the pitch table's metrics (blended across all
    pitch types, not broken out), respecting the same opponent-bats filter
    so it's a fair comparison point. Stuff+ is 100 by construction (it's a
    z-score scale centered on the league mean)."""
    where, params = "WHERE season = ?", (season,)
    if stand_filter and stand_filter != "ALL":
        where, params = "WHERE season = ? AND stand = ?", (season, stand_filter)
    df = pd.read_sql(f"SELECT * FROM pitches {where}", conn, params=params)
    df = _add_flags(df)
    df = _add_arm_side_hb(df)
    swings = df["is_swing"].sum()
    out_zone_n = df["out_zone"].sum()
    pa = df[df["events"].notna()]
    k = pa["events"].isin(STRIKEOUT_EVENTS).sum()
    return dict(
        velo=df["release_speed"].mean(), spin=df["release_spin_rate"].mean(),
        ivb=df["pfx_z"].mean() * 12, hb=df["hb"].mean(),
        zone_pct=df["in_zone"].mean() * 100,
        chase_pct=(df["is_swing"] & df["out_zone"]).sum() / out_zone_n * 100 if out_zone_n else np.nan,
        whiff_pct=df["is_whiff"].sum() / swings * 100 if swings else np.nan,
        k_pct=k / len(pa) * 100 if len(pa) else np.nan,
        stuff_plus=100.0,
    )


def league_average_batter_row(conn, season, throws_filter=None):
    """League-wide average of the batter pitch-type table's metrics
    (blended across all pitch types), respecting the same opponent-throws
    filter."""
    where, params = "WHERE season = ?", (season,)
    if throws_filter and throws_filter != "ALL":
        where, params = "WHERE season = ? AND p_throws = ?", (season, throws_filter)
    df = pd.read_sql(f"SELECT * FROM pitches {where}", conn, params=params)
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


def _load(conn, season, id_col, player_id, extra_filter_col=None, extra_filter_val=None):
    where = ["season = ?", f"{id_col} = ?"]
    params = [season, player_id]
    if extra_filter_col and extra_filter_val and extra_filter_val != "ALL":
        where.append(f"{extra_filter_col} = ?")
        params.append(extra_filter_val)
    df = pd.read_sql(f"SELECT * FROM pitches WHERE {' AND '.join(where)}", conn, params=params)
    if "pitch_type" in df.columns:
        df["pitch_type"] = df["pitch_type"].replace(PITCH_TYPE_MERGE)
    return df


# ---------------------------------------------------------------- Pitcher --

def pitcher_pitch_table(conn, season, pitcher_id, stand_filter=None, stuff_plus_table=None):
    """One row per pitch type: usage%, velo, spin, IVB, HB, zone%, chase%,
    whiff%, K%, stuff_plus (if a precomputed stuff_plus_table is passed -
    see stuff_model.compute_stuff_plus). Extension lives in the percentiles
    panel now, not here."""
    df = _load(conn, season, "pitcher_id", pitcher_id, "stand", stand_filter)
    if df.empty:
        return pd.DataFrame()
    df = _add_flags(df)
    df = _add_arm_side_hb(df)
    total = len(df)

    rows = []
    for pt, group in df.groupby("pitch_type"):
        swings = group["is_swing"].sum()
        out_zone_n = group["out_zone"].sum()
        pa = group[group["events"].notna()]
        k = pa["events"].isin(STRIKEOUT_EVENTS).sum()
        rows.append(dict(
            pitch_type=pt,
            usage_pct=len(group) / total * 100,
            n_pitches=len(group),
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


def pitcher_total_row(conn, season, pitcher_id, stand_filter=None):
    """This pitcher's own blended average across every pitch type they
    throw - same metric shape as one row of pitcher_pitch_table(), just not
    grouped by pitch_type. Used as the summary row at the bottom of that
    table (labeled "Player Total"), replacing what used to be a league-wide
    average there - comparing a pitcher's slider against *their own*
    overall repertoire is the more useful reference point than the league's."""
    df = _load(conn, season, "pitcher_id", pitcher_id, "stand", stand_filter)
    if df.empty:
        return {}
    df = _add_flags(df)
    df = _add_arm_side_hb(df)
    swings = df["is_swing"].sum()
    out_zone_n = df["out_zone"].sum()
    pa = df[df["events"].notna()]
    k = pa["events"].isin(STRIKEOUT_EVENTS).sum()
    return dict(
        velo=df["release_speed"].mean(), spin=df["release_spin_rate"].mean(),
        ivb=df["pfx_z"].mean() * 12, hb=df["hb"].mean(),
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
    df = _load(conn, season, "pitcher_id", pitcher_id, "stand", stand_filter)
    df = df[df["pfx_x"].notna() & df["pfx_z"].notna()].copy()
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


def pitcher_arm_angle(conn, season, pitcher_id):
    """Season-wide average arm angle (degrees), independent of the L/R
    toggle - used as a fixed reference line on the movement plot, not a
    percentile (there's no inherent 'better' direction for arm slot)."""
    row = conn.execute(
        "SELECT AVG(arm_angle) FROM pitches WHERE season = ? AND pitcher_id = ? AND arm_angle IS NOT NULL",
        (season, pitcher_id),
    ).fetchone()
    return row[0]


def pitcher_counting_stats(conn, season, pitcher_id, stand_filter=None, games_started=None, league_hr_fb_rate=None):
    df = _load(conn, season, "pitcher_id", pitcher_id, "stand", stand_filter)
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
    )


# ----------------------------------------------------------------- Batter --

def batter_pitch_type_table(conn, season, batter_id, throws_filter=None):
    """Batter performance vs. each pitch type, same metric shape as the
    splits tab: OPS, wOBA, xwOBA, K%, whiff%, Barrel%."""
    df = _load(conn, season, "batter_id", batter_id, "p_throws", throws_filter)
    if df.empty:
        return pd.DataFrame()
    df = _add_flags(df)

    rows = []
    for pt, group in df.groupby("pitch_type"):
        if len(group) <= 10:
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
    return pd.DataFrame(rows).set_index("pitch_type").sort_values("n_pitches", ascending=False)


HIT_EVENTS = {"single", "double", "triple", "home_run"}


def field_zone_pcts(conn, season, batter_id):
    """% of this batter's batted balls landing in LF/CF/RF (field-absolute
    spray angle, not handedness-relative pull/oppo), plus each individual
    batted ball's real plotted position and outcome - a real spray chart
    (like Baseball Savant's), not just three aggregate percentages.

    `locations[i]` is `{x, y, hit}`: x/y are hc_x/hc_y converted to
    field-relative feet with home plate at the origin and y increasing away
    from home plate (the same standard community transform - a straight
    unscaled version of the constants compute_spray_angle already uses,
    kept as (x, y) here instead of collapsed into one angle so individual
    balls can be plotted, not just summarized). `hit` is True for a single/
    double/triple/home run, False otherwise (out, error, fielder's choice)."""
    from percentiles import _SPRAY_X0, _SPRAY_Y0, compute_spray_angle

    df = pd.read_sql(
        "SELECT hc_x, hc_y, events FROM pitches "
        "WHERE season = ? AND batter_id = ? AND hc_x IS NOT NULL AND hc_y IS NOT NULL",
        conn, params=(season, batter_id),
    )
    if df.empty:
        return dict(lf=33.3, cf=33.3, rf=33.3, locations=[])
    angle = compute_spray_angle(df["hc_x"], df["hc_y"])
    n = len(angle)
    lf = (angle < -15).sum() / n * 100
    rf = (angle > 15).sum() / n * 100

    x = df["hc_x"] - _SPRAY_X0
    y = _SPRAY_Y0 - df["hc_y"]
    locations = [
        {"x": round(float(xi), 1), "y": round(float(yi), 1), "hit": ev in HIT_EVENTS}
        for xi, yi, ev in zip(x, y, df["events"].fillna(""))
    ]
    return dict(lf=lf, cf=100 - lf - rf, rf=rf, locations=locations)


def swing_profile(conn, season, batter_id, throws_filter=None):
    df = _load(conn, season, "batter_id", batter_id, "p_throws", throws_filter)
    df = df[df["bat_speed"].notna()]
    cols = ["bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt"]
    result = df[cols].mean().to_dict()
    result["bat_speed"] = fast_swing_bat_speed(df["bat_speed"])
    return result


def league_average_swing_profile(conn, season):
    """Same shape as swing_profile() but league-wide (every swing with a
    bat_speed reading, no player filter) - the comparison point for the
    swing gauges on the batter page, same idea as league_average_batter_row
    for the pitch-type table."""
    df = pd.read_sql("SELECT bat_speed, swing_length, attack_angle, attack_direction, "
                      "swing_path_tilt FROM pitches WHERE season = ? AND bat_speed IS NOT NULL",
                      conn, params=(season,))
    cols = ["swing_length", "attack_angle", "attack_direction", "swing_path_tilt"]
    result = df[cols].mean().to_dict()
    result["bat_speed"] = fast_swing_bat_speed(df["bat_speed"])
    return result


def batter_wrc_plus(conn, season, batter_id, throws_filter=None, league_woba=None):
    if league_woba is None:
        league_woba = league_average_woba(conn, season)
    df = _load(conn, season, "batter_id", batter_id, "p_throws", throws_filter)
    pa = df[df["events"].notna()]
    outcomes = _pa_outcome_metrics(pa)
    return compute_wrc_plus(outcomes["wOBA"], league_woba)


def batter_counting_stats(conn, season, batter_id, throws_filter=None):
    df = _load(conn, season, "batter_id", batter_id, "p_throws", throws_filter)
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
