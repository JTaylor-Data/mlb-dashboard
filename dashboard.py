"""
Rudimentary splits explorer - search a player, pick a split, look at the
table. Just a way to poke at splits.py interactively; not a real UI.

Run with:
    streamlit run dashboard.py
"""
import sqlite3

import pandas as pd
import streamlit as st

from splits import compute_pitcher_splits, compute_batter_splits

DB_PATH = "mlb_data.db"
SEASON = 2026  # legacy/prototype page, never got a season selector

st.set_page_config(page_title="MLB Splits Explorer", layout="wide")
st.title("MLB Splits Explorer")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)

mode = st.radio("Mode", ["Pitcher", "Batter"], horizontal=True)
id_col = "pitcher_id" if mode == "Pitcher" else "batter_id"

players = pd.read_sql(f"""
    SELECT DISTINCT pl.player_id, pl.name
    FROM players pl JOIN pitches p ON pl.player_id = p.{id_col}
    ORDER BY pl.name
""", conn)
player_name = st.selectbox("Player", players["name"])
player_id = int(players.loc[players["name"] == player_name, "player_id"].iloc[0])

pitch_types = pd.read_sql(
    f"SELECT DISTINCT pitch_type FROM pitches "
    f"WHERE {id_col} = ? AND pitch_type IS NOT NULL ORDER BY pitch_type",
    conn, params=(player_id,),
)["pitch_type"].tolist()
pitch_type = st.selectbox("Pitch type", ["All"] + pitch_types)
pitch_type_arg = None if pitch_type == "All" else pitch_type

if mode == "Pitcher":
    split_by = st.selectbox("Split by", ["home_away", "month", "half", "stand"])
    df = compute_pitcher_splits(conn, SEASON, split_by, pitcher_id=player_id, pitch_type=pitch_type_arg)
else:
    split_by = st.selectbox("Split by", ["home_away", "month", "half", "p_throws", "archetype"])
    df = compute_batter_splits(conn, SEASON, split_by, batter_id=player_id, pitch_type=pitch_type_arg)

st.subheader(f"{player_name} — split by {split_by}")

if df.empty:
    st.write("No data for this selection.")
else:
    pct_cols = [c for c in df.columns if c not in ("n_pa", "n_pitches")]
    st.dataframe(df.style.format("{:.3f}", subset=pct_cols), use_container_width=True)
