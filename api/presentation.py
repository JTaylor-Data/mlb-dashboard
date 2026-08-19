"""
Presentation constants for the player page endpoint - ported verbatim from
pages/1_Player_Page.py's module-level dicts. This is where "how do we label
and format this metric" lives now instead of inside a Streamlit page, so the
API can return {label, value_str} pairs the frontend just renders.
"""
import math

import pandas as pd

PITCHER_LABELS = {
    "whiff_pct": "Whiff%", "k_pct": "K%", "chase_pct": "Chase%",
    "zone_whiff_pct": "ZnWhiff%", "velo": "FB Velo", "perceived_velo": "Perceived Velo",
    "bb_pct": "BB%", "k_minus_bb_pct": "K-BB%", "avg_exit_velo": "Avg EV",
    "barrel_pct": "Barrel%", "gb_pct": "GB%", "extension": "Extension",
    "pull_air_pct": "Pull Air%", "release_height": "Rel Ht", "fb_vaa": "FB VAA",
    "zone_pct": "Zone%", "strike_pct": "Strike%", "swstr_pct": "Miss%",
    "zone_swing_pct": "Zone Swing%", "ev_90th": "EV 90th", "avg_la_95plus": "Avg LA 95+",
    "fb_pct": "FB%", "xwoba": "xwOBA", "release_side": "Rel Side", "spin_rate": "Spin Cap.",
}
PITCHER_FMT = {
    "whiff_pct": "{:.1%}", "k_pct": "{:.1%}", "chase_pct": "{:.1%}",
    "zone_whiff_pct": "{:.1%}", "velo": "{:.1f}", "perceived_velo": "{:.1f}",
    "bb_pct": "{:.1%}", "k_minus_bb_pct": "{:.1%}", "avg_exit_velo": "{:.1f}",
    "barrel_pct": "{:.1%}", "gb_pct": "{:.1%}", "extension": "{:.1f} ft",
    "pull_air_pct": "{:.1%}", "release_height": "{:.1f} ft", "fb_vaa": "{:.1f}°",
    "zone_pct": "{:.1%}", "strike_pct": "{:.1%}", "swstr_pct": "{:.1%}",
    "zone_swing_pct": "{:.1%}", "ev_90th": "{:.1f}", "avg_la_95plus": "{:.1f}°",
    "fb_pct": "{:.1%}", "xwoba": "{:.3f}", "release_side": "{:.1f} ft", "spin_rate": "{:.0f}",
}
BATTER_LABELS = {
    "xwoba": "xwOBA", "xslg": "xSLG", "xba": "xBA", "avg_exit_velo": "Avg EV",
    "barrel_pct": "Barrel%", "avg_launch_angle": "Avg LA", "bat_speed": "Bat Speed",
    "chase_pct": "Chase%", "zone_whiff_pct": "Zone Whiff%", "k_pct": "K%",
    "bb_pct": "BB%", "pull_air_pct": "Pull Air%", "gb_pct": "GB%",
    "sprint_speed": "Sprint Speed", "oaa": "Outs Above Avg",
    "swstr_pct": "Swinging Strike%", "zone_swing_pct": "Zone Swing%",
    "ev_90th": "90th Pct EV", "avg_la_95plus": "Hard Hit Launch", "fb_pct": "FB%",
    "whiff_pct": "Whiff%", "zone_contact_pct": "Zone Contact",
}
BATTER_FMT = {
    "xwoba": "{:.3f}", "xslg": "{:.3f}", "xba": "{:.3f}", "avg_exit_velo": "{:.1f}",
    "barrel_pct": "{:.1%}", "avg_launch_angle": "{:.1f}°", "bat_speed": "{:.1f} mph",
    "chase_pct": "{:.1%}", "zone_whiff_pct": "{:.1%}", "k_pct": "{:.1%}",
    "bb_pct": "{:.1%}", "pull_air_pct": "{:.1%}", "gb_pct": "{:.1%}",
    "sprint_speed": "{:.1f} ft/s", "oaa": "{:.0f}",
    "swstr_pct": "{:.1%}", "zone_swing_pct": "{:.1%}", "ev_90th": "{:.1f}",
    "avg_la_95plus": "{:.1f}°", "fb_pct": "{:.1%}", "whiff_pct": "{:.1%}",
    "zone_contact_pct": "{:.1%}",
}

# Grouped by what the metric is actually about, not sorted by percentile
# value, so related stats sit next to each other (e.g. whiff% by K%).
# Pitcher groups mirror a reference "Results / Traits" boxed-bar layout the
# user pointed at directly - replaces the earlier Command/Swing&Miss/Stuff/
# Contact-Against grouping entirely, not additively.
PITCHER_GROUPS = [
    ("Results", ["strike_pct", "zone_pct", "swstr_pct", "chase_pct", "xwoba", "gb_pct"]),
    ("Traits", ["velo", "release_height", "extension", "release_side", "fb_vaa", "spin_rate"]),
]
BATTER_GROUPS = [
    ("Results", ["xwoba", "xslg", "xba"]),
    ("Contact Quality", ["avg_exit_velo", "barrel_pct", "avg_launch_angle", "pull_air_pct", "gb_pct"]),
    ("Approach", ["chase_pct", "zone_whiff_pct", "k_pct", "bb_pct"]),
    ("Physical & Defense", ["bat_speed", "sprint_speed", "oaa"]),
]

# Curated 6-axis set for the Radar tab - same underlying metrics for both
# roles (xwoba is against-batter for pitchers, own-production for hitters;
# polarity per-metric already encoded in percentiles.PITCHER_POLARITY /
# BATTER_POLARITY so "toward the edge" always means "good" regardless of role).
RADAR_METRICS = [
    ("avg_la_95plus", "Avg LA 95+"), ("ev_90th", "EV 90th"), ("xwoba", "xwOBA"),
    ("zone_whiff_pct", "Zone Whiff%"), ("whiff_pct", "SwM%"), ("chase_pct", "Chase%"),
]

# Compare tab: swing decisions + batted ball outcomes, matching a reference
# layout the user pointed at. Batter-only (that reference is batter-specific).
# Four genuinely distinct swing-decision angles, not overlapping framings of
# the same stat: Swinging Strike% (misses / all pitches seen - the "how often
# do they just miss" rate), Whiff% (misses / swings only - bat-to-ball skill
# once they've decided to swing), Zone Whiff% (misses / in-zone swings only -
# bat-to-ball skill specifically on strikes), Chase% (O-zone swing rate -
# discipline), Zone Swing% (in-zone swing rate - aggression on takeable
# strikes).
COMPARE_SWING_METRICS = [
    ("swstr_pct", "Swinging Strike%"), ("whiff_pct", "Whiff%"), ("zone_whiff_pct", "Zone Whiff%"),
    ("chase_pct", "Chase%"), ("zone_swing_pct", "Zone Swing%"),
]
COMPARE_BATTED_BALL_METRICS = [
    ("ev_90th", "90th Pct EV"), ("barrel_pct", "Barrel Rate"), ("avg_la_95plus", "Hard Hit Launch"),
    ("pull_air_pct", "Pulled Air"), ("gb_pct", "GB%"), ("fb_pct", "FB%"),
]

PITCH_COLORS = {
    "FF": "#d62728", "CU": "#1f77b4", "FC": "#8c564b", "SI": "#ff7f0e",
    "CH": "#2ca02c", "FS": "#40E0D0", "ST": "#b35900", "SL": "#e6c200",
}
DEFAULT_PITCH_COLOR = "#999999"

PITCHER_TABLE_FMT = {
    "usage_pct": "{:.1f}%", "n_pitches": "{:.0f}", "velo": "{:.1f}", "spin": "{:.0f}",
    "ivb": "{:.1f}", "hb": "{:.1f}",
    "zone_pct": "{:.1f}%", "chase_pct": "{:.1f}%",
    "whiff_pct": "{:.1f}%", "k_pct": "{:.1f}%", "stuff_plus": "{:.0f}",
}
PITCHER_TABLE_COLUMNS = [
    ("usage_pct", "Usage%"), ("n_pitches", "Pitches"), ("velo", "Velo"), ("spin", "Spin"),
    ("ivb", "IVB"), ("hb", "HB"), ("zone_pct", "Zone%"), ("chase_pct", "Chase%"),
    ("whiff_pct", "Whiff%"), ("k_pct", "K%"), ("stuff_plus", "Stuff+"),
]
BATTER_TABLE_COLUMNS = [
    ("n_pitches", "Pitches"), ("OPS", "OPS"), ("wOBA", "wOBA"), ("xwOBA", "xwOBA"),
    ("K_pct", "K%"), ("whiff_pct", "Whiff%"), ("barrel_pct", "Barrel%"),
]


def format_value(raw, fmt):
    if raw is None or (isinstance(raw, float) and (pd.isna(raw) or math.isnan(raw))):
        return "-"
    return fmt.format(raw)


def format_row(row_dict, fmt_map, default_fmt="{:.3f}"):
    """row_dict: plain dict of raw values (as returned by pandas .to_dict()).
    Returns a dict of the same keys with formatted display strings."""
    out = {}
    for key, raw in row_dict.items():
        fmt = fmt_map.get(key, default_fmt)
        out[key] = format_value(raw, fmt)
    return out
