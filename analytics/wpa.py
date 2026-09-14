"""
Daily WPA ranking. WPA (Win Probability Added) comes directly from
Statcast's own delta_home_win_exp (change in the home team's win probability
on each pitch) - not derived or modeled by us, same trust level as
delta_run_exp elsewhere in this project.

A player's own WPA is the sum of that change across every pitch they were
involved in that day, sign-adjusted for whether their team is home/away and
whether they're batting or pitching (a good outcome for the batting side is
a bad outcome for the pitching side, by definition, same run of logic as
delta_run_exp's sign flip for Stuff+).
"""
import numpy as np
import pandas as pd


def compute_daily_wpa(conn, date):
    """One row per (player, role, game) with their summed WPA for that date.
    role is 'batter' or 'pitcher'."""
    df = pd.read_sql("""
        SELECT game_id, pitcher_id, batter_id, inning_topbot, delta_home_win_exp
        FROM pitches WHERE game_date = ?
    """, conn, params=(date,))
    if df.empty:
        return pd.DataFrame(columns=["player_id", "game_id", "wpa", "role", "name", "team_id"])

    # Batting team's WPA: if they're the home team (bottom of inning), it's
    # delta_home_win_exp directly; if they're away, flip the sign.
    df["batting_side_wpa"] = np.where(
        df["inning_topbot"] == "Bot", df["delta_home_win_exp"], -df["delta_home_win_exp"]
    )
    df["pitching_side_wpa"] = -df["batting_side_wpa"]

    batters = df.groupby(["batter_id", "game_id"], as_index=False)["batting_side_wpa"].sum()
    batters.columns = ["player_id", "game_id", "wpa"]
    batters["role"] = "batter"

    pitchers = df.groupby(["pitcher_id", "game_id"], as_index=False)["pitching_side_wpa"].sum()
    pitchers.columns = ["player_id", "game_id", "wpa"]
    pitchers["role"] = "pitcher"

    combined = pd.concat([batters, pitchers], ignore_index=True)
    player_ids = [int(x) for x in pd.concat([batters["player_id"], pitchers["player_id"]]).unique()]
    ph = ",".join("?" * len(player_ids))
    players = pd.read_sql(
        f"SELECT player_id, name, team_id FROM players WHERE player_id IN ({ph})",
        conn, params=player_ids,
    ).set_index("player_id")
    combined["name"] = combined["player_id"].map(players["name"])
    combined["team_id"] = combined["player_id"].map(players["team_id"])
    return combined


def top_performers(conn, date, n=10):
    """Top n (player, role, game) entries by WPA for the date. A player who
    both batted and pitched that day (rare) is deduped to their higher-WPA
    role rather than appearing twice."""
    combined = compute_daily_wpa(conn, date)
    if combined.empty:
        return combined
    idx = combined.groupby("player_id")["wpa"].idxmax()
    deduped = combined.loc[idx]
    return deduped.sort_values("wpa", ascending=False).head(n)


def all_daily_participants(conn, date):
    """Every (player, role, game) that appeared on this date - the full
    candidate pool for the box-score-weighted leaderboard, which (unlike WPA)
    can't be pre-filtered to a top-n before its ranking stat is known. Same
    dedup rule as top_performers(): a player who both batted and pitched, or
    played a doubleheader, keeps only their higher-WPA row so they don't
    appear twice."""
    combined = compute_daily_wpa(conn, date)
    if combined.empty:
        return combined
    idx = combined.groupby("player_id")["wpa"].idxmax()
    return combined.loc[idx]


def find_walkoff_batter(conn, game_id):
    """The batter_id who ended this game with a walkoff (bottom 9th or
    later, the literal last play of the game, home team winning), or None
    if this game had no walkoff."""
    # order by at_bat_number/pitch_number (true in-game chronological order),
    # not our own pitch_id (insertion order - not guaranteed to match game
    # order once multiple games get bulk-inserted together)
    row = conn.execute("""
        SELECT inning, inning_topbot, batter_id
        FROM pitches WHERE game_id = ?
        ORDER BY at_bat_number DESC, pitch_number DESC LIMIT 1
    """, (game_id,)).fetchone()
    if row is None:
        return None
    inning, inning_topbot, last_batter_id = row
    if inning_topbot != "Bot" or inning < 9:
        return None

    final_score = conn.execute("SELECT final_score FROM games WHERE game_id = ?", (game_id,)).fetchone()
    if final_score is None:
        return None
    home_score, away_score = (int(x) for x in final_score[0].split("-"))
    if home_score <= away_score:
        return None
    return last_batter_id
