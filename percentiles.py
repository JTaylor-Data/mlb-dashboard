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

from metrics import compute_vaa, compute_barrel
from splits import (
    SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS, STRIKE_DESCRIPTIONS,
    IN_ZONE_CODES, OUT_ZONE_CODES, STRIKEOUT_EVENTS, WALK_EVENTS,
)

MIN_PITCHES_QUALIFY = 200
MIN_PA_QUALIFY = 100

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


def fast_swing_bat_speed(bat_speed_series):
    """Average bat speed of a hitter's own fastest quarter of swings (at or
    above their individual 75th percentile) - matches Baseball Savant's bat
    speed leaderboard methodology. A flat mean over every swing gets dragged
    down by bunts/check-swings/defensive two-strike swings; anchoring the
    threshold to each hitter's own distribution (rather than a league-wide
    cutoff) keeps it fair across hitters with very different swing profiles."""
    s = bat_speed_series.dropna()
    if s.empty:
        return np.nan
    threshold = s.quantile(0.75)
    return s[s >= threshold].mean()


def _rate_metrics(group):
    n = len(group)
    swings = group["is_swing"].sum()
    whiffs = group["is_whiff"].sum()
    out_zone = group["out_zone"]
    in_zone = group["in_zone"]
    zone_swings = (group["is_swing"] & in_zone).sum()
    zone_whiffs = (group["is_whiff"] & in_zone).sum()

    batted = group[group["is_batted_ball"]]
    n_batted = len(batted)

    pa = group[group["events"].notna()]
    n_pa = len(pa)
    k = pa["events"].isin(STRIKEOUT_EVENTS).sum()
    bb = pa["events"].isin(WALK_EVENTS).sum()

    in_zone_n = in_zone.sum()
    hard_hit = batted[batted["launch_speed"] >= 95] if n_batted else batted

    return dict(
        n_pitches=n,
        whiff_pct=whiffs / swings if swings else np.nan,
        chase_pct=(group["is_swing"] & out_zone).sum() / out_zone.sum() if out_zone.sum() else np.nan,
        zone_whiff_pct=zone_whiffs / zone_swings if zone_swings else np.nan,
        zone_pct=in_zone.mean() if n else np.nan,
        strike_pct=group["is_strike"].mean() if n else np.nan,
        swstr_pct=whiffs / n if n else np.nan,
        zone_swing_pct=(group["is_swing"] & in_zone).sum() / in_zone_n if in_zone_n else np.nan,
        k_pct=k / n_pa if n_pa else np.nan,
        bb_pct=bb / n_pa if n_pa else np.nan,
        avg_exit_velo=batted["launch_speed"].mean() if n_batted else np.nan,
        ev_90th=batted["launch_speed"].quantile(0.9) if n_batted else np.nan,
        avg_la_95plus=hard_hit["launch_angle"].mean() if len(hard_hit) else np.nan,
        barrel_pct=compute_barrel(batted["launch_speed"], batted["launch_angle"]).mean() if n_batted else np.nan,
        gb_pct=(batted["bb_type"] == "ground_ball").mean() if n_batted else np.nan,
        fb_pct=(batted["bb_type"] == "fly_ball").mean() if n_batted else np.nan,
        pull_air_pct=(batted["is_pulled"] & batted["is_air"]).sum() / n_batted if n_batted else np.nan,
        avg_launch_angle=batted["launch_angle"].mean() if n_batted else np.nan,
        n_pa=n_pa,
    )


def pitcher_percentile_metrics(conn, season, min_pitches=MIN_PITCHES_QUALIFY, stand_filter=None):
    """
    One row per qualified pitcher (>= min_pitches total pitches WITHIN this
    filter) with the raw metric values: whiff_pct, k_pct, chase_pct,
    zone_whiff_pct, perceived_velo, bb_pct, k_minus_bb_pct, avg_exit_velo,
    barrel_pct, gb_pct, extension, pull_air_pct, release_height, fb_vaa. All
    "against" (opponent's outcomes).

    stand_filter ('L'/'R'/None or 'ALL'): when set, the entire pool - both
    the qualifying threshold and the metrics themselves - is recomputed
    within pitches against that batter side only, not just re-ranked against
    the season-wide pool.
    """
    where = "WHERE season = ?"
    params = (season,)
    if stand_filter and stand_filter != "ALL":
        where += " AND stand = ?"
        params += (stand_filter,)

    df = pd.read_sql(f"""
        SELECT pitcher_id, pitch_type, description, zone, events, stand,
               release_speed, effective_speed, release_extension, release_pos_z,
               release_pos_x, release_spin_rate,
               launch_speed, launch_angle, bb_type, hc_x, hc_y,
               vy0, vz0, ay, az, woba_denom, estimated_woba_using_speedangle
        FROM pitches {where}
    """, conn, params=params)
    df = _add_shared_flags(df)
    df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
    is_fb = df["pitch_type"].isin(["FF", "FA", "SI"])

    rows = []
    for pid, group in df.groupby("pitcher_id"):
        if len(group) < min_pitches:
            continue
        row = _rate_metrics(group)
        fb = group[is_fb.loc[group.index]]
        qualifying = group["woba_denom"] == 1
        denom = qualifying.sum()
        row.update(
            pitcher_id=pid,
            velo=fb["release_speed"].mean() if len(fb) else np.nan,
            perceived_velo=fb["effective_speed"].mean() if len(fb) else np.nan,
            fb_vaa=fb["vaa"].mean() if len(fb) else np.nan,
            k_minus_bb_pct=row["k_pct"] - row["bb_pct"],
            extension=group["release_extension"].mean(),
            release_height=group["release_pos_z"].mean(),
            release_side=group["release_pos_x"].mean(),
            spin_rate=group["release_spin_rate"].mean(),
            xwoba=group.loc[qualifying, "estimated_woba_using_speedangle"].sum() / denom if denom else np.nan,
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("pitcher_id")


def batter_percentile_metrics(conn, season, min_pa=MIN_PA_QUALIFY, throws_filter=None):
    """
    One row per qualified batter (>= min_pa plate appearances WITHIN this
    filter) with the raw metric values: xwoba, xslg, xba, avg_exit_velo,
    barrel_pct, avg_launch_angle, bat_speed, chase_pct, zone_whiff_pct,
    k_pct, bb_pct, pull_air_pct, gb_pct, sprint_speed, oaa.

    throws_filter ('L'/'R'/None or 'ALL'): when set, the entire pool - both
    the qualifying threshold and the metrics themselves - is recomputed
    within pitches from that pitcher hand only, not just re-ranked against
    the season-wide pool. sprint_speed/oaa are season-level (not pitch-level)
    so they don't change with this filter - same value regardless of split.
    """
    where = "WHERE season = ?"
    params = (season,)
    if throws_filter and throws_filter != "ALL":
        where += " AND p_throws = ?"
        params += (throws_filter,)

    df = pd.read_sql(f"""
        SELECT batter_id, description, zone, events, stand,
               launch_speed, launch_angle, bb_type, hc_x, hc_y, bat_speed,
               woba_denom, estimated_woba_using_speedangle, estimated_slg_using_speedangle,
               estimated_ba_using_speedangle
        FROM pitches {where}
    """, conn, params=params)
    df = _add_shared_flags(df)

    advanced = pd.read_sql(
        "SELECT player_id, sprint_speed, oaa FROM player_advanced_stats WHERE season = ?",
        conn, params=(season,),
    ).set_index("player_id")

    rows = []
    for bid, group in df.groupby("batter_id"):
        row = _rate_metrics(group)
        if row["n_pa"] < min_pa:
            continue
        qualifying = group["woba_denom"] == 1
        denom = qualifying.sum()
        row.update(
            batter_id=bid,
            xwoba=group.loc[qualifying, "estimated_woba_using_speedangle"].sum() / denom if denom else np.nan,
            xslg=group.loc[qualifying, "estimated_slg_using_speedangle"].sum() / denom if denom else np.nan,
            xba=group.loc[qualifying, "estimated_ba_using_speedangle"].sum() / denom if denom else np.nan,
            bat_speed=fast_swing_bat_speed(group["bat_speed"]),
            sprint_speed=advanced["sprint_speed"].get(bid, np.nan),
            oaa=advanced["oaa"].get(bid, np.nan),
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("batter_id")


# Metric polarity: True = higher raw value is better (percentile follows raw
# rank), False = lower raw value is better (percentile is inverted), None =
# no inherent "better" direction, purely descriptive positioning.
PITCHER_POLARITY = {
    "whiff_pct": True, "k_pct": True, "chase_pct": True, "zone_whiff_pct": True,
    "velo": True, "perceived_velo": True, "bb_pct": False, "k_minus_bb_pct": True,
    "avg_exit_velo": False, "barrel_pct": False, "gb_pct": True,
    "extension": True, "pull_air_pct": False, "release_height": False, "fb_vaa": True,
    "zone_pct": True, "strike_pct": True, "swstr_pct": True, "zone_swing_pct": None,
    "ev_90th": False, "avg_la_95plus": True, "fb_pct": False, "xwoba": False,
    "release_side": None, "spin_rate": None,
}
BATTER_POLARITY = {
    "xwoba": True, "xslg": True, "xba": True, "avg_exit_velo": True, "barrel_pct": True,
    "avg_launch_angle": None, "bat_speed": True, "chase_pct": False,
    "zone_whiff_pct": False, "k_pct": False, "bb_pct": True,
    "pull_air_pct": True, "gb_pct": False, "sprint_speed": True, "oaa": True,
    "swstr_pct": False, "zone_swing_pct": True, "ev_90th": True, "avg_la_95plus": True,
    "fb_pct": True, "zone_pct": None, "strike_pct": None, "whiff_pct": False,
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
