"""
Pulls the REMAINING (not-yet-played) regular-season schedule for every team,
so the team card's Remaining Strength of Schedule metric has something to
average opponents' win% over. Separate from `games` (which only ever holds
COMPLETED games, populated from pulled pitch data) - this is forward-looking
schedule data the site otherwise has no source for.

Overwrites remaining_schedule fully on each run (today's "remaining" set is
correct, yesterday's is stale) - not an upsert/merge, a full replace.

Usage:
    python pull_remaining_schedule.py [SEASON]   # defaults to 2026
"""
import sqlite3
import sys
from datetime import date
from pathlib import Path

import requests

DB_PATH = Path(__file__).resolve().parent.parent / "mlb_data.db"
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_SEASON = 2026


def run(season: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS remaining_schedule (
            game_id      INTEGER PRIMARY KEY,
            season       INTEGER,
            game_date    TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER
        )
    """)

    today = date.today().isoformat()
    season_end = f"{season}-11-30"  # comfortably past any real regular season end
    print(f"Pulling remaining schedule for {season} ({today} to {season_end})...", flush=True)

    resp = requests.get(f"{STATSAPI_BASE}/schedule", params={
        "sportId": 1, "season": season, "startDate": today, "endDate": season_end,
        "gameType": "R",  # regular season only - postseason isn't known yet this early
    }, timeout=30)
    resp.raise_for_status()
    dates = resp.json().get("dates", [])

    rows = []
    for d in dates:
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                continue  # a "remaining" window can still include a game that finished since the call started
            rows.append(dict(
                game_id=g["gamePk"], season=season, game_date=d["date"],
                home_team_id=g["teams"]["home"]["team"]["id"], away_team_id=g["teams"]["away"]["team"]["id"],
            ))

    conn.execute("DELETE FROM remaining_schedule WHERE season = ?", (season,))
    conn.executemany(
        "INSERT OR REPLACE INTO remaining_schedule (game_id, season, game_date, home_team_id, away_team_id) "
        "VALUES (:game_id, :season, :game_date, :home_team_id, :away_team_id)",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"Upserted {len(rows)} remaining games into remaining_schedule.")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
