"""
Individual player page - pitcher and batter views. Savant-style layout with
colored percentile bars, a horizontal player card, and a full-width pitch
table with league-average comparison row.

Streamlit auto-discovers this from the pages/ folder when you run
`streamlit run dashboard.py` and adds it to the sidebar nav.
"""
import math
import sqlite3

import altair as alt
import pandas as pd
import streamlit as st

from percentiles import (
    pitcher_percentile_metrics, batter_percentile_metrics, player_percentiles,
    PITCHER_POLARITY, BATTER_POLARITY,
)
from player_page import (
    pitcher_pitch_table, movement_plot_data, pitcher_counting_stats, pitcher_arm_angle,
    league_hr_per_fb, league_average_pitcher_row, league_average_batter_row,
    batter_pitch_type_table, swing_profile, batter_wrc_plus, batter_counting_stats,
    field_zone_pcts,
)
from metrics import classify_pitcher_roles
from stuff_model import compute_stuff_plus, overall_stuff_plus
from archetypes import classify_batter_archetype
from ui_helpers import (
    percentile_bar_html, percentile_group_header_html, player_card_html,
    stuff_plus_badge_html, archetype_badge_html, bat_swing_svg, row_tint_styler,
)

DB_PATH = "mlb_data.db"
# Legacy fallback page - never got a season selector when the dashboard was
# rearchitected to FastAPI (that's where the year toggle lives now). Pinned
# to 2026 so it keeps working rather than crashing now that every analytics
# function below requires a season argument.
SEASON = 2026

st.set_page_config(page_title="Player Page", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 16px; }
h2, h3 { font-size: 17px !important; margin-top: 2px !important; margin-bottom: 2px !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 100%; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def load_pitcher_pool(stand_filter):
    return pitcher_percentile_metrics(get_conn(), SEASON, stand_filter=stand_filter)


@st.cache_data
def load_batter_pool(throws_filter):
    return batter_percentile_metrics(get_conn(), SEASON, throws_filter=throws_filter)


@st.cache_data
def load_stuff_plus():
    return compute_stuff_plus(get_conn(), SEASON)


@st.cache_data
def load_pitcher_roles():
    return classify_pitcher_roles(get_conn(), SEASON)


@st.cache_data
def load_league_hr_fb():
    return league_hr_per_fb(get_conn(), SEASON)


conn = get_conn()
st.title("Player Page")

mode = st.radio("Mode", ["Pitcher", "Batter"], horizontal=True)
id_col = "pitcher_id" if mode == "Pitcher" else "batter_id"

players = pd.read_sql(f"""
    SELECT DISTINCT pl.player_id, pl.name, pl.team_id
    FROM players pl JOIN pitches p ON pl.player_id = p.{id_col}
    ORDER BY pl.name
""", conn)
c_player, c_toggle = st.columns([2, 2])
with c_player:
    player_name = st.selectbox("Player", players["name"])
player_row = players.loc[players["name"] == player_name].iloc[0]
player_id, team_id = int(player_row["player_id"]), player_row["team_id"]

PITCHER_LABELS = {
    "whiff_pct": "Whiff%", "k_pct": "K%", "chase_pct": "OZ Swing%",
    "zone_whiff_pct": "ZnWhiff%", "perceived_velo": "Perceived Velo", "bb_pct": "BB%",
    "k_minus_bb_pct": "K-BB%", "avg_exit_velo": "Avg EV", "barrel_pct": "Barrel%",
    "gb_pct": "GB%", "extension": "Extension", "pull_air_pct": "Pull Air%",
    "release_height": "Release Height", "fb_vaa": "FB VAA",
}
PITCHER_FMT = {
    "whiff_pct": "{:.1%}", "k_pct": "{:.1%}", "chase_pct": "{:.1%}",
    "zone_whiff_pct": "{:.1%}", "perceived_velo": "{:.1f}", "bb_pct": "{:.1%}",
    "k_minus_bb_pct": "{:.1%}", "avg_exit_velo": "{:.1f}", "barrel_pct": "{:.1%}",
    "gb_pct": "{:.1%}", "extension": "{:.1f} ft", "pull_air_pct": "{:.1%}",
    "release_height": "{:.1f} ft", "fb_vaa": "{:.1f}°",
}
BATTER_LABELS = {
    "xwoba": "xwOBA", "xslg": "xSLG", "xba": "xBA", "avg_exit_velo": "Avg EV",
    "barrel_pct": "Barrel%", "avg_launch_angle": "Avg LA", "bat_speed": "Bat Speed",
    "chase_pct": "OZ Swing%", "zone_whiff_pct": "ZnMiss%", "k_pct": "K%",
    "bb_pct": "BB%", "pull_air_pct": "Pull Air%", "gb_pct": "GB%",
    "sprint_speed": "Sprint Speed", "oaa": "Outs Above Avg",
}
BATTER_FMT = {
    "xwoba": "{:.3f}", "xslg": "{:.3f}", "xba": "{:.3f}", "avg_exit_velo": "{:.1f}",
    "barrel_pct": "{:.1%}", "avg_launch_angle": "{:.1f}°", "bat_speed": "{:.1f} mph",
    "chase_pct": "{:.1%}", "zone_whiff_pct": "{:.1%}", "k_pct": "{:.1%}",
    "bb_pct": "{:.1%}", "pull_air_pct": "{:.1%}", "gb_pct": "{:.1%}",
    "sprint_speed": "{:.1f} ft/s", "oaa": "{:.0f}",
}

PITCH_COLORS = {
    "FF": "#d62728", "CU": "#1f77b4", "FC": "#8c564b", "SI": "#ff7f0e",
    "CH": "#2ca02c", "FS": "#40E0D0", "ST": "#b35900", "SL": "#e6c200",
}
DEFAULT_PITCH_COLOR = "#999999"

# Grouped by what the metric is actually about, not sorted by percentile
# value, so related stats sit next to each other (e.g. whiff% by K%).
PITCHER_GROUPS = [
    ("Command", ["k_pct", "bb_pct", "k_minus_bb_pct"]),
    ("Swing & Miss", ["whiff_pct", "zone_whiff_pct", "chase_pct"]),
    ("Stuff", ["perceived_velo", "extension", "release_height", "fb_vaa"]),
    ("Contact Against", ["avg_exit_velo", "barrel_pct", "gb_pct", "pull_air_pct"]),
]
BATTER_GROUPS = [
    ("Results", ["xwoba", "xslg", "xba"]),
    ("Contact Quality", ["avg_exit_velo", "barrel_pct", "avg_launch_angle", "pull_air_pct", "gb_pct"]),
    ("Approach", ["chase_pct", "zone_whiff_pct", "k_pct", "bb_pct"]),
    ("Physical & Defense", ["bat_speed", "sprint_speed", "oaa"]),
]


def render_percentiles(pct_dict, labels, fmt_map, groups):
    html = ""
    for title, metrics in groups:
        present = [m for m in metrics if m in pct_dict]
        if not present:
            continue
        html += percentile_group_header_html(title)
        for m in present:
            raw, p = pct_dict[m]
            html += percentile_bar_html(labels.get(m, m), raw, p, fmt_map.get(m, "{:.1f}"))
    st.markdown(html, unsafe_allow_html=True)


def styled_table(df, fmt):
    flat = df.reset_index().rename(columns={"index": "pitch_type"})
    styler = flat.style.apply(
        row_tint_styler(PITCH_COLORS, DEFAULT_PITCH_COLOR), axis=1
    ).format(fmt, na_rep="-").hide(axis="index")

    def _pitch_type_cell(val):
        color = PITCH_COLORS.get(val, DEFAULT_PITCH_COLOR)
        return f"background-color:{color}; color:white; font-weight:bold;"

    return styler.map(_pitch_type_cell, subset=["pitch_type"])


if mode == "Pitcher":
    with c_toggle:
        stand_filter = st.radio("Opponent bats", ["ALL", "L", "R"], horizontal=True)

    pool = load_pitcher_pool(stand_filter)
    stuff = load_stuff_plus()
    roles = load_pitcher_roles()
    gs = int(roles.loc[player_id, "games_started"]) if player_id in roles.index else None
    lg_hrfb = load_league_hr_fb()

    stats = pitcher_counting_stats(conn, SEASON, player_id, stand_filter, games_started=gs, league_hr_fb_rate=lg_hrfb)
    overall_stuff = overall_stuff_plus(stuff, player_id)
    card_rows = [
        ("IP", stats["ip"]), ("K", stats["k"]), ("GS", stats["gs"] if stats["gs"] is not None else "-"),
        ("OPS against", f"{stats['OPS']:.3f}" if pd.notna(stats["OPS"]) else "-"),
        ("xFIP", f"{stats['xfip']:.2f}" if pd.notna(stats["xfip"]) else "-"),
    ]
    st.markdown(player_card_html(player_name, team_id, card_rows, stuff_plus_badge_html(overall_stuff)),
                unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1, 1.6])
    with left:
        st.subheader("Percentiles")
        pct = player_percentiles(pool, player_id, PITCHER_POLARITY)
        if pct:
            render_percentiles(pct, PITCHER_LABELS, PITCHER_FMT, PITCHER_GROUPS)
        else:
            st.write("Not enough pitches to qualify in this split.")

    with right:
        st.subheader("Movement Plot")
        mv = movement_plot_data(conn, SEASON, player_id, stand_filter)
        if not mv.empty:
            arm_angle = pitcher_arm_angle(conn, SEASON, player_id)
            zero_h = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#999", strokeWidth=1).encode(y="y")
            zero_v = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#999", strokeWidth=1).encode(x="x")

            layers = [zero_h, zero_v]
            if pd.notna(arm_angle):
                r = 25
                ax, ay = r * math.cos(math.radians(arm_angle)), r * math.sin(math.radians(arm_angle))
                arm_df = pd.DataFrame({"HB": [-ax, ax], "IVB": [-ay, ay]})
                layers.append(
                    alt.Chart(arm_df).mark_line(color="lightgrey", strokeWidth=3, strokeDash=[4, 2])
                    .encode(x="HB", y="IVB")
                )

            present = mv["pitch_type"].unique().tolist()
            domain = [t for t in PITCH_COLORS if t in present] + [t for t in present if t not in PITCH_COLORS]
            color_range = [PITCH_COLORS.get(t, DEFAULT_PITCH_COLOR) for t in domain]

            scatter = alt.Chart(mv).mark_circle(size=35, opacity=0.55).encode(
                x=alt.X("HB", title="Horizontal Break (in)", scale=alt.Scale(domain=[-25, 25])),
                y=alt.Y("IVB", title="Induced Vertical Break (in)", scale=alt.Scale(domain=[-25, 25])),
                color=alt.Color("pitch_type", title="Pitch", scale=alt.Scale(domain=domain, range=color_range)),
                tooltip=["pitch_type"],
            )
            layers.append(scatter)
            chart = (alt.layer(*layers).properties(height=620, background="transparent")
                     .configure_view(strokeWidth=0)
                     .configure_axis(labelFontSize=13, titleFontSize=14)
                     .configure_legend(labelFontSize=13, titleFontSize=14, symbolSize=140))
            st.altair_chart(chart, use_container_width=True)
            st.caption("Grey dashed line = this pitcher's average arm angle direction.")
        else:
            st.write("No pitch data for this selection.")

    st.subheader("Pitch Table")
    table = pitcher_pitch_table(conn, SEASON, player_id, stand_filter, stuff)
    if not table.empty:
        lg_row = league_average_pitcher_row(conn, SEASON, stand_filter)
        lg_row["usage_pct"] = None
        table = pd.concat([table, pd.DataFrame([lg_row], index=["League Avg"])])
        fmt = {
            "usage_pct": "{:.1f}%", "n_pitches": "{:.0f}", "velo": "{:.1f}", "spin": "{:.0f}",
            "ivb": "{:.1f}", "hb": "{:.1f}",
            "zone_pct": "{:.1f}%", "chase_pct": "{:.1f}%",
            "whiff_pct": "{:.1f}%", "k_pct": "{:.1f}%", "stuff_plus": "{:.0f}",
        }
        st.dataframe(styled_table(table, fmt), use_container_width=True)

else:
    with c_toggle:
        throws_filter = st.radio("Opponent pitcher throws", ["ALL", "L", "R"], horizontal=True)

    pool = load_batter_pool(throws_filter)
    stats = batter_counting_stats(conn, SEASON, player_id, throws_filter)
    wrc = batter_wrc_plus(conn, SEASON, player_id, throws_filter)
    card_rows = [
        ("GP", stats["gp"]), ("HR", stats["hr"]),
        ("OPS", f"{stats['OPS']:.3f}" if pd.notna(stats["OPS"]) else "-"),
        ("wRC+ (approx)", f"{wrc:.0f}" if pd.notna(wrc) else "-"),
    ]
    # Archetype is season-overall (always the ALL-handedness pool), not
    # re-derived per L/R toggle - it's a descriptive label, not a split stat.
    all_pool = load_batter_pool("ALL")
    archetype_pct = player_percentiles(all_pool, player_id, BATTER_POLARITY)
    archetype = classify_batter_archetype(archetype_pct) if archetype_pct else None
    badge = archetype_badge_html(archetype) if archetype else ""
    st.markdown(player_card_html(player_name, team_id, card_rows, badge), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1, 1.6])
    with left:
        st.subheader("Percentiles")
        pct = player_percentiles(pool, player_id, BATTER_POLARITY)
        if pct:
            render_percentiles(pct, BATTER_LABELS, BATTER_FMT, BATTER_GROUPS)
        else:
            st.write("Not enough plate appearances to qualify in this split.")
        st.caption("Sprint speed / OAA are season-level (2026), not affected by the L/R toggle.")

    with right:
        st.subheader("Swing & Field")
        zones = field_zone_pcts(conn, SEASON, player_id)
        profile = swing_profile(conn, SEASON, player_id, throws_filter)
        svg = bat_swing_svg(zones["lf"], zones["cf"], zones["rf"],
                             profile.get("bat_speed"), profile.get("swing_path_tilt"),
                             profile.get("attack_angle"))
        st.markdown(svg, unsafe_allow_html=True)
        st.caption("Field-level view from directly behind the batter. Zones colored by this batter's "
                   "actual share of batted balls in that direction. Bat arc is illustrative.")

    st.subheader("Performance vs. Pitch Type")
    st.caption("Pitch types with 10 or fewer pitches seen are excluded.")
    table = batter_pitch_type_table(conn, SEASON, player_id, throws_filter)
    if not table.empty:
        lg_row = league_average_batter_row(conn, SEASON, throws_filter)
        table = pd.concat([table, pd.DataFrame([lg_row], index=["League Avg"])])
        pct_cols = [c for c in table.columns if c != "n_pitches"]
        fmt = {c: "{:.3f}" for c in pct_cols}
        fmt["n_pitches"] = "{:.0f}"
        st.dataframe(styled_table(table, fmt), use_container_width=True)
    else:
        st.write("No pitch type has more than 10 pitches seen for this selection.")
