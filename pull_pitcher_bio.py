"""
One-time (well, re-runnable) pull of pitcher biographical data (height,
weight) from the MLB Stats API `/people/{id}` endpoint, into a new
`pitcher_bio` table - height/weight aren't in Statcast's pitch-level data
at all, and the existing `players` table (built for team/position/bats/
throws) never pulled them either. Needed as features for the delivery-only
Deception+ model (deception_model.py) - not wired into "Refresh data",
same category as pull_prospects.py: run by hand when needed.

    python pull_pitcher_bio.py

Height comes back as a string like `6' 3"` - parsed to total inches.
Weight comes back as an integer, already in pounds. This is CURRENT
listed bio data only (not season-specific) - same "current-snapshot"
limitation this project already accepts for `players.team_id` elsewhere;
a pitcher's weight in 2021 may differ slightly from what's pulled here.
"""
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DB_PATH = Path(__file__).parent / "mlb_data.db"


def _parse_height(height_str):
    """'6' 3\"' -> 75 (total inches). None if unparseable."""
    if not height_str:
        return None
    m = re.match(r"(\d+)'\s*(\d+)", height_str)
    if not m:
        return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return feet * 12 + inches


def _fetch_one(player_id):
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}", timeout=10)
        r.raise_for_status()
        people = r.json().get("people", [])
        if not people:
            return player_id, None, None
        person = people[0]
        return player_id, _parse_height(person.get("height")), person.get("weight")
    except Exception:
        return player_id, None, None


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_bio (
            player_id INTEGER PRIMARY KEY,
            height_in INTEGER,
            weight_lb INTEGER,
            pulled_at TEXT
        )
    """)
    conn.commit()

    pitcher_ids = [r[0] for r in conn.execute("SELECT DISTINCT pitcher_id FROM pitches")]
    print(f"Pulling bio data for {len(pitcher_ids)} pitchers...", flush=True)

    start = time.time()
    rows = []
    with ThreadPoolExecutor(10) as pool:
        futures = {pool.submit(_fetch_one, pid): pid for pid in pitcher_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, height_in, weight_lb = fut.result()
            rows.append((pid, height_in, weight_lb, time.strftime("%Y-%m-%d %H:%M:%S")))
            if i % 250 == 0:
                print(f"  {i}/{len(pitcher_ids)}...", flush=True)

    conn.executemany(
        "INSERT OR REPLACE INTO pitcher_bio (player_id, height_in, weight_lb, pulled_at) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    n_resolved = sum(1 for r in rows if r[1] is not None and r[2] is not None)
    print(f"\nDone in {time.time()-start:.0f}s. {n_resolved}/{len(rows)} pitchers resolved with both height and weight.")


if __name__ == "__main__":
    main()
