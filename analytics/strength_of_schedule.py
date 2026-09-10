"""
Remaining Strength of Schedule - for each team, the average CURRENT win% of
every opponent left on their schedule (remaining_schedule, pulled by
pull_remaining_schedule.py), ranked 1-30 where 30 = hardest (highest average
opponent win%). Win% itself is computed live from `games` (completed games
only) every call - not cached at pull time - so it moves as results come in
even between schedule pulls, matching "should update as team winning
percentages do."
"""


def team_win_pct(conn, season):
    """{team_id: win_pct} from completed games this season. final_score is
    stored "home-away"; a team's own win/loss is derived by comparing its
    own side's score to the other, for every game it appears in as either
    home or away."""
    rows = conn.execute("""
        SELECT home_team, away_team, final_score FROM games WHERE season = ? AND final_score IS NOT NULL
    """, (season,)).fetchall()

    wins, losses = {}, {}
    for home, away, score in rows:
        try:
            home_score, away_score = (int(x) for x in score.split("-"))
        except (ValueError, AttributeError):
            continue
        home_id, away_id = int(home), int(away)
        if home_score == away_score:
            continue  # shouldn't happen in real MLB data, skip rather than misclassify
        winner, loser = (home_id, away_id) if home_score > away_score else (away_id, home_id)
        wins[winner] = wins.get(winner, 0) + 1
        losses[loser] = losses.get(loser, 0) + 1

    team_ids = set(wins) | set(losses)
    out = {}
    for tid in team_ids:
        w, l = wins.get(tid, 0), losses.get(tid, 0)
        out[tid] = w / (w + l) if (w + l) else None
    return out


def remaining_sos(conn, season):
    """{team_id: {avg_opponent_win_pct, rank, n_games_remaining}} - rank is
    1-30, 30 = hardest remaining schedule (highest average opponent win%).
    A team with no remaining games (or no completed opponent data yet) is
    simply absent from the result, not zero-filled - the caller decides how
    to display "no data" (this project's convention is a dash, not a fake 0)."""
    win_pct = team_win_pct(conn, season)

    sched_rows = conn.execute("""
        SELECT home_team_id, away_team_id FROM remaining_schedule WHERE season = ?
    """, (season,)).fetchall()

    opponents = {}  # team_id -> list of opponent win_pct
    for home_id, away_id in sched_rows:
        for team_id, opp_id in ((home_id, away_id), (away_id, home_id)):
            opp_wp = win_pct.get(opp_id)
            if opp_wp is None:
                continue
            opponents.setdefault(team_id, []).append(opp_wp)

    avg_by_team = {
        tid: dict(avg_opponent_win_pct=sum(vals) / len(vals), n_games_remaining=len(vals))
        for tid, vals in opponents.items() if vals
    }
    ordered = sorted(avg_by_team.items(), key=lambda kv: kv[1]["avg_opponent_win_pct"])
    for rank, (tid, d) in enumerate(ordered, start=1):
        d["rank"] = rank
    return avg_by_team
