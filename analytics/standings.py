"""
Standings computed directly from our own `games` table (final scores we
already pulled), not from the MLB Stats API's standings endpoint - keeps a
single source of truth and gives us "last 30," which isn't a standard split
the API exposes.
"""
import pandas as pd


def _team_game_log(games_df):
    """One row per (team, game) - the same game appears twice, once from
    each team's perspective."""
    g = games_df.copy()
    if g.empty:
        return pd.DataFrame(columns=["team_id", "game_date", "team_score", "opp_score", "won", "margin"])
    split = g["final_score"].str.split("-", expand=True)
    if split.shape[1] < 2:
        return pd.DataFrame(columns=["team_id", "game_date", "team_score", "opp_score", "won", "margin"])
    g[["home_score", "away_score"]] = split.astype(int)
    g["home_team"] = g["home_team"].astype(int)
    g["away_team"] = g["away_team"].astype(int)

    home = g.rename(columns={"home_team": "team_id", "away_team": "opp_id",
                              "home_score": "team_score", "away_score": "opp_score"})
    away = g.rename(columns={"away_team": "team_id", "home_team": "opp_id",
                              "away_score": "team_score", "home_score": "opp_score"})
    log = pd.concat([home, away])[["team_id", "game_date", "team_score", "opp_score"]]
    log["won"] = log["team_score"] > log["opp_score"]
    log["margin"] = log["team_score"] - log["opp_score"]
    return log.sort_values("game_date")


def _record_str(sub):
    w = int(sub["won"].sum())
    l = len(sub) - w
    return f"{w}-{l}"


def compute_standings(conn, season):
    """Returns a DataFrame, one row per team: name, abbreviation, league,
    division, wins, losses, win_pct, run_diff, one_run_record, last_10,
    last_30. Sorted by league/division/win_pct."""
    games_df = pd.read_sql(
        "SELECT game_date, home_team, away_team, final_score FROM games WHERE season = ?",
        conn, params=(season,),
    )
    teams_df = pd.read_sql("SELECT team_id, name, abbreviation, league, division FROM teams", conn)

    log = _team_game_log(games_df)

    rows = []
    for team_id, team_log in log.groupby("team_id"):
        wins = int(team_log["won"].sum())
        losses = len(team_log) - wins
        one_run = team_log[team_log["margin"].abs() == 1]
        rows.append(dict(
            team_id=team_id,
            wins=wins,
            losses=losses,
            win_pct=wins / len(team_log) if len(team_log) else 0.0,
            run_diff=int(team_log["team_score"].sum() - team_log["opp_score"].sum()),
            one_run_record=_record_str(one_run) if len(one_run) else "0-0",
            last_10=_record_str(team_log.tail(10)),
            last_30=_record_str(team_log.tail(30)),
            games_played=len(team_log),
        ))

    if not rows:
        return pd.DataFrame(columns=["team_id", "name", "abbreviation", "league", "division",
                                     "wins", "losses", "win_pct", "run_diff",
                                     "one_run_record", "last_10", "last_30", "games_played"])
    standings = pd.DataFrame(rows).merge(teams_df, on="team_id")
    return standings.sort_values(["league", "division", "win_pct"], ascending=[True, True, False])


def best_record_team_id(standings_df, league):
    sub = standings_df[standings_df["league"] == league]
    if sub.empty:
        return None
    return sub.sort_values(["win_pct", "run_diff"], ascending=False).iloc[0]["team_id"]
