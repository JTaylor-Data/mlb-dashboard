"""
Top Performers - top 10 players leaguewide by WPA (Win Probability Added)
for a given day. Batters and pitchers show different stat lines since their
box scores don't share columns, so this is a ranked list of rows rather
than one shared table - simplest way to handle a mixed leaderboard cleanly.
Every row uses the same CSS Grid column template (fixed pixel/fraction
widths, not flex-content-sized) so columns land in the same position
regardless of how many stats a given row type has - a pitcher row just
leaves its unused trailing slot(s) blank.

WPA comes straight from Statcast's own delta_home_win_exp (see wpa.py).
Daily box score stats and season OPS/ERA come live from the MLB Stats API
boxscore endpoint (boxscore.py) - official, scorer-verified numbers, not
reconstructed from Statcast.
"""
import sqlite3
from datetime import datetime

import streamlit as st

from wpa import top_performers, is_walkoff_batter
from boxscore import fetch_boxscore, batter_daily_stats, pitcher_daily_stats
from ui_helpers import team_logo_url, TEXT_DARK, TEXT_MUTED

DB_PATH = "mlb_data.db"

# 6 stat slots is the max either row type needs (batter uses all 6, pitcher
# uses 5 and leaves the last blank) so every row shares this exact template.
GRID_TEMPLATE = "30px 46px 230px repeat(6, 1fr) 80px"

# Rank 1 darkest, fading out by rank 4+ (which fall back to plain alternating
# stripes).
RANK_BACKGROUNDS = {1: "#f5cf6b", 2: "#f7dd94", 3: "#fbebc0"}

st.set_page_config(page_title="Top Performers", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 100%; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _opponent(conn, game_id, team_id):
    row = conn.execute("SELECT home_team, away_team FROM games WHERE game_id = ?", (game_id,)).fetchone()
    if row is None:
        return ""
    home, away = int(row[0]), int(row[1])
    is_home = team_id == home
    opp_id = away if is_home else home
    abbrev = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (opp_id,)).fetchone()
    if abbrev is None:
        return ""
    return f"{'vs' if is_home else '@'} {abbrev[0]}"


@st.cache_data
def load_top_10(day_str):
    conn = get_conn()
    top = top_performers(conn, day_str, n=10)
    if top.empty:
        return []

    results = []
    boxscore_cache = {}
    for _, row in top.iterrows():
        game_id = int(row["game_id"])
        team_id = int(row["team_id"]) if row["team_id"] is not None else None
        if game_id not in boxscore_cache:
            boxscore_cache[game_id] = fetch_boxscore(game_id)
        bx = boxscore_cache[game_id]

        if row["role"] == "batter":
            stats = batter_daily_stats(bx, int(row["player_id"]))
            walkoff = is_walkoff_batter(conn, int(row["player_id"]), game_id)
        else:
            stats = pitcher_daily_stats(bx, int(row["player_id"]))
            walkoff = False

        results.append(dict(
            name=row["name"], role=row["role"], wpa=row["wpa"],
            team_id=team_id, stats=stats, walkoff=walkoff,
            opponent=_opponent(conn, game_id, team_id) if team_id is not None else "",
        ))
    return results


def _flatten(html):
    return " ".join(line.strip() for line in html.strip().splitlines())


def stat_cell_html(label, value):
    return _flatten(f"""
    <div style="text-align:center;">
      <div style="font-size:18px; font-weight:700; color:{TEXT_DARK};">{value}</div>
      <div style="font-size:10px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
    </div>
    """)


def performer_row_html(rank, entry):
    logo = team_logo_url(entry["team_id"])
    stats = entry["stats"] or {}
    walkoff_badge = (' <span style="background:#d4a017; color:white; font-size:10px; '
                      'padding:2px 6px; border-radius:4px;">WALK-OFF</span>') if entry["walkoff"] else ""

    if entry["role"] == "batter":
        cells = [
            stat_cell_html("AB", stats.get("ab", "-")),
            stat_cell_html("H", stats.get("h", "-")),
            stat_cell_html("HR", stats.get("hr", 0)),
            stat_cell_html("RBI", stats.get("rbi", 0)),
            stat_cell_html("SB", stats.get("sb", 0)),
            stat_cell_html("OPS", stats.get("season_ops", "-")),
        ]
    else:
        cells = [
            stat_cell_html("IP", stats.get("ip", "-")),
            stat_cell_html("H", stats.get("h", 0)),
            stat_cell_html("K", stats.get("k", 0)),
            stat_cell_html("R", stats.get("r", 0)),
            stat_cell_html("ER", stats.get("er", 0)),
            stat_cell_html("ERA", stats.get("season_era", "-")),
        ]
    cells_html = "".join(cells)
    role_tag = "PITCHER" if entry["role"] == "pitcher" else "BATTER"

    bg = RANK_BACKGROUNDS.get(rank, "#fff" if rank % 2 else "#fafafa")

    return _flatten(f"""
    <div style="display:grid; grid-template-columns:{GRID_TEMPLATE}; align-items:center; gap:10px;
                padding:14px 20px; border-bottom:1px solid #eee; background:{bg};">
      <div style="font-size:20px; font-weight:800; color:{TEXT_MUTED};">{rank}</div>
      <img src="{logo}" width="42" height="42" onerror="this.style.display='none'"/>
      <div>
        <div style="font-size:20px; font-weight:800; color:{TEXT_DARK}; line-height:1.25;">{entry['name']}</div>
        <div style="font-size:10px; color:{TEXT_MUTED}; font-weight:600;">
          {role_tag} &nbsp;&#183;&nbsp; {entry['opponent']}{walkoff_badge}
        </div>
      </div>
      {cells_html}
      <div style="text-align:right;">
        <div style="font-size:10px; color:{TEXT_MUTED};">WPA</div>
        <div style="font-size:20px; font-weight:800; color:{TEXT_DARK};">{entry['wpa']:.3f}</div>
      </div>
    </div>
    """)


st.title("Top Performers")

# Defaults to whatever the most recent date actually in the database is,
# rather than a hardcoded date that goes stale every time new data is
# pulled in.
latest_date_str = get_conn().execute("SELECT MAX(game_date) FROM games").fetchone()[0]
default_date = datetime.strptime(latest_date_str, "%Y-%m-%d").date()

picked_date = st.date_input("Date", value=default_date)
day_str = picked_date.strftime("%Y-%m-%d")

st.caption("Ranked by WPA (Win Probability Added), straight from Statcast's own per-pitch win "
           "expectancy - not modeled by us. Box score stats and season OPS/ERA are live, official "
           "numbers from the MLB Stats API, not reconstructed from Statcast.")

entries = load_top_10(day_str)
if not entries:
    st.write(f"No games found for {day_str}.")
else:
    html = "".join(performer_row_html(i + 1, e) for i, e in enumerate(entries))
    st.markdown(f'<div style="border:1px solid #ddd; border-radius:10px; overflow:hidden;">{html}</div>',
                unsafe_allow_html=True)
