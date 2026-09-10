"""
Pulls season-level Savant leaderboard stats (Sprint Speed, Outs Above
Average) into player_advanced_stats. These aren't in the pitch-by-pitch
Statcast pull - separate leaderboards, genuinely external season aggregates.

Usage:
    python pull_advanced_stats.py [SEASON]   # defaults to 2026

Upserts, so re-running just refreshes the numbers.
"""
import sqlite3
import sys

from pybaseball import statcast_sprint_speed, statcast_outs_above_average

DB_PATH = "mlb_data.db"
DEFAULT_SEASON = 2026


def run(season: int):
    conn = sqlite3.connect(DB_PATH)

    print(f"Pulling sprint speed for {season}...", end=" ", flush=True)
    speed_df = statcast_sprint_speed(season, min_opp=1)
    print(f"done, {len(speed_df)} players")

    print(f"Pulling outs above average for {season}...", end=" ", flush=True)
    oaa_df = statcast_outs_above_average(season, "all", min_att=1)
    print(f"done, {len(oaa_df)} players")

    speed_map = dict(zip(speed_df["player_id"], speed_df["sprint_speed"]))
    oaa_map = dict(zip(oaa_df["player_id"], oaa_df["outs_above_average"]))

    all_ids = set(speed_map) | set(oaa_map)
    rows = [(pid, season, speed_map.get(pid), oaa_map.get(pid)) for pid in all_ids]

    conn.executemany(
        "INSERT INTO player_advanced_stats (player_id, season, sprint_speed, oaa) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(player_id, season) DO UPDATE SET "
        "sprint_speed=excluded.sprint_speed, oaa=excluded.oaa",
        rows,
    )
    conn.commit()

    have_both = sum(1 for pid in all_ids if pid in speed_map and pid in oaa_map)
    print(f"\nUpserted {len(rows)} players ({have_both} with both sprint speed and OAA).")
    conn.close()


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
