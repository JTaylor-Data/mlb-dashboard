"""
Daily "Top Performers" for prospects - the MLB-level Top Performers page
ranks by WPA (Statcast's own win-probability delta), which doesn't exist for
these players: MiLB games aren't Statcast-tracked at the level this project
ingests, so there's no delta_home_win_exp to sum. Instead this ranks by a
points system - DraftKings' actual MLB/MiLB Classic fantasy scoring (a real,
already-calibrated "low score for a single, high score for a homer" system),
not something invented from scratch:

  Batting:  1B +3, 2B +5, 3B +8, HR +10, BB/HBP +2, SB +5, R +2, RBI +2
  Pitching: out recorded +0.75 (so 3 outs/inning = +2.25), K +2, ER -2,
            BB against -0.6, H against -0.6, HBP against -0.6, Win +4

Points are used for RANKING only, not shown to the user - a raw point total
with no context reads as an arbitrary made-up number even though the system
behind it isn't. The card shows the day's real box-score line plus the
prospect's actual season OPS/ERA (from the `prospects` table) instead.

Fetched on demand per date, not bulk-ingested (same reasoning as
boxscore.py): pulls the day's schedule across every MiLB sportId, fetches
each game's boxscore, and keeps only players who are in our own prospects
table - most of a day's ~90 MiLB games involve zero of our 900 tracked
prospects, we just don't know which until we look.
"""
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from analytics.boxscore import fetch_boxscore

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MILB_SPORT_IDS = "11,12,13,14,16"  # AAA, AA, A+, A, Rookie (see api/v1/sports)
MAX_WORKERS = 10

BATTING_POINTS = {
    "singles": 3, "doubles": 5, "triples": 8, "homeRuns": 10,
    "baseOnBalls": 2, "hitByPitch": 2, "stolenBases": 5, "runs": 2, "rbi": 2,
}
PITCHING_PER_OUT = 0.75
PITCHING_POINTS = {"strikeOuts": 2, "baseOnBalls": -0.6, "hits": -0.6, "hitBatsmen": -0.6, "wins": 4}
PITCHING_ER_POINTS = -2


def fetch_milb_games(date_str):
    """Returns list of {game_id, team names, team ids} for every MiLB game
    (AAA down through Rookie) on the given date."""
    resp = requests.get(
        SCHEDULE_URL, params={"sportId": MILB_SPORT_IDS, "date": date_str, "gameType": "R"}, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            teams = g.get("teams", {})
            games.append({
                "game_id": g["gamePk"],
                "home_team_id": teams.get("home", {}).get("team", {}).get("id"),
                "home_team_name": teams.get("home", {}).get("team", {}).get("name"),
                "away_team_id": teams.get("away", {}).get("team", {}).get("id"),
                "away_team_name": teams.get("away", {}).get("team", {}).get("name"),
            })
    return games


def _innings_to_outs(ip_str):
    """'6.1' -> 19 outs (6 innings + 1), '6.2' -> 20, '6.0' -> 18."""
    if not ip_str:
        return 0
    m = re.match(r"(\d+)(?:\.(\d))?", str(ip_str))
    if not m:
        return 0
    full, partial = int(m.group(1)), int(m.group(2) or 0)
    return full * 3 + partial


def batting_points(stat):
    hits = stat.get("hits", 0) or 0
    doubles = stat.get("doubles", 0) or 0
    triples = stat.get("triples", 0) or 0
    hr = stat.get("homeRuns", 0) or 0
    singles = max(0, hits - doubles - triples - hr)

    pts = singles * BATTING_POINTS["singles"]
    pts += doubles * BATTING_POINTS["doubles"]
    pts += triples * BATTING_POINTS["triples"]
    pts += hr * BATTING_POINTS["homeRuns"]
    pts += (stat.get("baseOnBalls", 0) or 0) * BATTING_POINTS["baseOnBalls"]
    pts += (stat.get("hitByPitch", 0) or 0) * BATTING_POINTS["hitByPitch"]
    pts += (stat.get("stolenBases", 0) or 0) * BATTING_POINTS["stolenBases"]
    pts += (stat.get("runs", 0) or 0) * BATTING_POINTS["runs"]
    pts += (stat.get("rbi", 0) or 0) * BATTING_POINTS["rbi"]
    return round(pts, 2), singles


def pitching_points(stat):
    outs = _innings_to_outs(stat.get("inningsPitched"))
    pts = outs * PITCHING_PER_OUT
    pts += (stat.get("strikeOuts", 0) or 0) * PITCHING_POINTS["strikeOuts"]
    pts += (stat.get("baseOnBalls", 0) or 0) * PITCHING_POINTS["baseOnBalls"]
    pts += (stat.get("hits", 0) or 0) * PITCHING_POINTS["hits"]
    pts += (stat.get("hitBatsmen", 0) or 0) * PITCHING_POINTS["hitBatsmen"]
    pts += (stat.get("wins", 0) or 0) * PITCHING_POINTS["wins"]
    pts += (stat.get("earnedRuns", 0) or 0) * PITCHING_ER_POINTS
    return round(pts, 2)


def _team_label(game, is_home):
    opp_name = game["away_team_name"] if is_home else game["home_team_name"]
    return f"{'vs' if is_home else '@'} {opp_name}"


def top_prospect_performers(conn, date_str, n=10):
    # Restricted to the overall Top 100 only, per direct request - the full
    # prospects table also includes every org's Top 30 (900+ names), most
    # of them unfamiliar; this leaderboard should only ever surface names
    # you'd actually recognize.
    prospect_rows = conn.execute(
        "SELECT mlb_id, name, is_pitcher, team_id, headshot_url, level, ops, era "
        "FROM prospects WHERE season = 2026 AND top100_rank IS NOT NULL"
    ).fetchall()
    prospects_by_id = {
        r[0]: {
            "name": r[1], "is_pitcher": bool(r[2]), "team_id": r[3], "headshot_url": r[4],
            "level": r[5], "season_ops": r[6], "season_era": r[7],
        }
        for r in prospect_rows
    }
    if not prospects_by_id:
        return []

    games = fetch_milb_games(date_str)
    if not games:
        return []

    results = []

    def process_game(game):
        try:
            box = fetch_boxscore(game["game_id"])
        except Exception:
            return []
        found = []
        for side, is_home in (("home", True), ("away", False)):
            players = box.get("teams", {}).get(side, {}).get("players", {})
            for key, p in players.items():
                pid = p.get("person", {}).get("id")
                info = prospects_by_id.get(pid)
                if info is None:
                    continue
                if info["is_pitcher"]:
                    stat = p.get("stats", {}).get("pitching", {})
                    if not stat or _innings_to_outs(stat.get("inningsPitched")) == 0:
                        continue
                    pts = pitching_points(stat)
                    line = {
                        "ip": stat.get("inningsPitched", "0.0"), "k": stat.get("strikeOuts", 0),
                        "h": stat.get("hits", 0), "er": stat.get("earnedRuns", 0),
                        "bb": stat.get("baseOnBalls", 0), "w": stat.get("wins", 0),
                    }
                else:
                    stat = p.get("stats", {}).get("batting", {})
                    if not stat or not stat.get("atBats"):
                        continue
                    pts, singles = batting_points(stat)
                    line = {
                        "ab": stat.get("atBats", 0), "h": stat.get("hits", 0), "hr": stat.get("homeRuns", 0),
                        "rbi": stat.get("rbi", 0), "sb": stat.get("stolenBases", 0), "r": stat.get("runs", 0),
                    }
                found.append({
                    "mlb_id": pid, "name": info["name"], "team_id": info["team_id"],
                    "headshot_url": info["headshot_url"], "is_pitcher": info["is_pitcher"],
                    "level": info["level"], "season_ops": info["season_ops"], "season_era": info["season_era"],
                    "points": pts, "line": line, "opponent": _team_label(game, is_home),
                })
        return found

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process_game, g) for g in games]
        for future in as_completed(futures):
            results.extend(future.result())

    results.sort(key=lambda r: r["points"], reverse=True)
    return results[:n]


if __name__ == "__main__":
    import sys
    date_str = sys.argv[1] if len(sys.argv) > 1 else "2026-08-15"
    conn = sqlite3.connect("mlb_data.db")
    for r in top_prospect_performers(conn, date_str, n=10):
        print(f"{r['points']:>6.2f}  {r['name']:25s} {r['opponent']:15s} {r['line']}")
