"""
Pulls major MLB awards (All-Star, MVP, Cy Young, ROY, Silver Slugger, Gold Glove)
from the MLB Stats API into the player_awards table.

    python pulls/pull_awards.py            # 2015-2026
    python pulls/pull_awards.py 2024       # single season

Uses MLBAM player IDs directly — no crosswalk needed.
"""
import sqlite3
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = "mlb_data.db"
FIRST_SEASON = 2015
CURRENT_SEASON = 2026

# Awards to pull with their display abbreviation
AWARD_IDS = {
    # All-Star
    "ALAS": "AS",
    "NLAS": "AS",
    # MVP
    "ALMVP": "MVP",
    "NLMVP": "MVP",
    # Cy Young
    "ALCY": "CY",
    "NLCY": "CY",
    # Rookie of the Year
    "ALROY": "ROY",
    "NLROY": "ROY",
    # Silver Slugger
    "ALSS": "SS",
    "NLSS": "SS",
    # Gold Glove
    "ALGG": "GG",
    "NLGG": "GG",
}

BASE_URL = "https://statsapi.mlb.com/api/v1/awards/{award_id}/recipients"


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_awards (
            player_id INTEGER NOT NULL,
            season    INTEGER NOT NULL,
            award     TEXT    NOT NULL,
            PRIMARY KEY (player_id, season, award)
        )
    """)
    conn.commit()


def pull_season(season: int, conn: sqlite3.Connection):
    print(f"\n=== Season {season} ===")
    rows = []
    for award_id, abbr in AWARD_IDS.items():
        try:
            resp = requests.get(
                BASE_URL.format(award_id=award_id),
                params={"season": season},
                timeout=15,
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            recipients = resp.json().get("awards", [])
            for r in recipients:
                pid = r.get("player", {}).get("id")
                if pid:
                    rows.append((int(pid), season, abbr))
            print(f"  {award_id}: {len(recipients)}", end="  ")
        except Exception as e:
            print(f"  {award_id}: FAILED ({e})", end="  ")
        time.sleep(0.1)

    print()
    if rows:
        conn.executemany("""
            INSERT INTO player_awards (player_id, season, award)
            VALUES (?, ?, ?) ON CONFLICT DO NOTHING
        """, rows)
        conn.commit()
        print(f"  → {len(rows)} award records stored")


def run(seasons):
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    for season in seasons:
        pull_season(season, conn)
    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].isdigit():
        seasons = [int(args[0])]
    else:
        seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    run(seasons)
