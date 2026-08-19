"""
Standings tab - AL/NL toggle, broken out by division, with team logos,
record, run differential, one-run record, last 10, last 30. The team with
the best overall record in the selected league is highlighted.

Computed from our own `games` table (see standings.py), not the MLB Stats
API's standings endpoint.
"""
import sqlite3

import streamlit as st

from standings import compute_standings, best_record_team_id
from ui_helpers import team_logo_url, TEXT_DARK, TEXT_MUTED

DB_PATH = "mlb_data.db"
SEASON = 2026  # legacy fallback page, never got a season selector

st.set_page_config(page_title="Standings", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 16px; }
h2, h3 { font-size: 17px !important; margin-top: 4px !important; margin-bottom: 4px !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 100%; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def load_standings():
    return compute_standings(get_conn(), SEASON)


def _flatten(html):
    return " ".join(line.strip() for line in html.strip().splitlines())


def division_table_html(div_df, best_team_id):
    header = _flatten("""
    <tr style="border-bottom:2px solid #ccc; font-size:11px; color:#666; text-transform:uppercase;">
      <th style="text-align:left; padding:4px 8px;"></th>
      <th style="text-align:left; padding:4px 8px;">Team</th>
      <th style="text-align:center; padding:4px 8px;">W</th>
      <th style="text-align:center; padding:4px 8px;">L</th>
      <th style="text-align:center; padding:4px 8px;">Win%</th>
      <th style="text-align:center; padding:4px 8px;">Run Diff</th>
      <th style="text-align:center; padding:4px 8px;">1-Run</th>
      <th style="text-align:center; padding:4px 8px;">Last 10</th>
      <th style="text-align:center; padding:4px 8px;">Last 30</th>
    </tr>
    """)

    rows_html = ""
    for _, row in div_df.iterrows():
        is_best = row["team_id"] == best_team_id
        row_style = ("background:#fff3cd; border-left:4px solid #d4a017;"
                     if is_best else "border-left:4px solid transparent;")
        crown = " &#128081;" if is_best else ""
        rd = row["run_diff"]
        rd_str = f"+{rd}" if rd > 0 else str(rd)
        rd_color = "#1a7f37" if rd > 0 else ("#c9302c" if rd < 0 else TEXT_MUTED)

        rows_html += _flatten(f"""
        <tr style="{row_style} border-bottom:1px solid #eee;">
          <td style="padding:6px 8px;"><img src="{team_logo_url(row['team_id'])}" width="28" height="28" onerror="this.style.display='none'"/></td>
          <td style="padding:6px 8px; font-weight:600; color:{TEXT_DARK};">{row['name']}{crown}</td>
          <td style="text-align:center; padding:6px 8px; color:{TEXT_DARK};">{row['wins']}</td>
          <td style="text-align:center; padding:6px 8px; color:{TEXT_DARK};">{row['losses']}</td>
          <td style="text-align:center; padding:6px 8px; color:{TEXT_DARK};">{row['win_pct']:.3f}</td>
          <td style="text-align:center; padding:6px 8px; color:{rd_color}; font-weight:600;">{rd_str}</td>
          <td style="text-align:center; padding:6px 8px; color:{TEXT_DARK};">{row['one_run_record']}</td>
          <td style="text-align:center; padding:6px 8px; color:{TEXT_DARK};">{row['last_10']}</td>
          <td style="text-align:center; padding:6px 8px; color:{TEXT_DARK};">{row['last_30']}</td>
        </tr>
        """)

    return _flatten(f"""
    <table style="width:100%; border-collapse:collapse; margin-bottom:18px;">
      {header}
      {rows_html}
    </table>
    """)


conn = get_conn()
st.title("Standings")

league = st.radio("League", ["AL", "NL"], horizontal=True)

standings = load_standings()
league_df = standings[standings["league"] == league]
best_id = best_record_team_id(standings, league)

st.caption("Best overall record in the league highlighted in gold. Standings computed from full-season "
           "game results, not a live API feed - refresh by re-running the ingestion scripts.")

for division in sorted(league_df["division"].unique()):
    st.subheader(division)
    div_df = league_df[league_df["division"] == division]
    st.markdown(division_table_html(div_df, best_id), unsafe_allow_html=True)
