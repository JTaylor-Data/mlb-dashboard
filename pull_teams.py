"""
Pulls static team reference data (name, abbreviation, league, division) from
the MLB Stats API into the `teams` table. Static data, so this rarely needs
re-running - just once, or after an offseason realignment.

Usage:
    python pull_teams.py [SEASON]   # defaults to 2026
"""
import sqlite3
import sys

import requests

DB_PATH = "mlb_data.db"
DEFAULT_SEASON = 2026


def run(season: int):
    resp = requests.get(
        f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={season}", timeout=30
    )
    resp.raise_for_status()
    teams = resp.json()["teams"]

    rows = [
        (
            t["id"],
            t["name"],
            t["abbreviation"],
            "AL" if t["league"]["id"] == 103 else "NL",
            t["division"]["name"].replace("American League", "AL").replace("National League", "NL"),
        )
        for t in teams
    ]

    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "INSERT INTO teams (team_id, name, abbreviation, league, division) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(team_id) DO UPDATE SET name=excluded.name, abbreviation=excluded.abbreviation, "
        "league=excluded.league, division=excluded.division",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"Upserted {len(rows)} teams.")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
