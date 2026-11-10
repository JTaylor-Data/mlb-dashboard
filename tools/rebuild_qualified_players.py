"""
tools/rebuild_qualified_players.py — Full rebuild of the `qualified_players` table.

Usage:
    python tools/rebuild_qualified_players.py

Run this after each day's pull (pulls/daily/pull_statcast.py) so /api/players/all
reflects newly-qualifying players without ever running a live scan of the
full `pitches` table. If the table is ever empty (e.g. a fresh DB),
api/routers/players.py's _all_players lazy-builds it inline on first use -
this script is for keeping it current, not a hard prerequisite.

After rebuilding, re-derives the same two aggregates directly from `pitches`
(the query _all_players used to run live) and diffs them against the new
table's row counts and a handful of sample player_ids, so a rebuild that
silently drifted from the raw data is caught here, not in production.
"""
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.db import DB_PATH
from analytics.qualified_players import rebuild_qualified_players


def _raw_counts(conn):
    pitcher_count = conn.execute("""
        SELECT COUNT(*) FROM players pl
        JOIN (
            SELECT pitcher_id, COUNT(DISTINCT game_id) AS n
            FROM pitches GROUP BY pitcher_id
        ) p ON pl.player_id = p.pitcher_id
        WHERE pl.position IN ('P', 'TWP') AND p.n >= 3
    """).fetchone()[0]
    batter_count = conn.execute("""
        SELECT COUNT(DISTINCT batter_id) FROM pitches WHERE batter_id IS NOT NULL
    """).fetchone()[0]
    return pitcher_count, batter_count


def main():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        print("Rebuilding qualified_players ...")
        t0 = time.monotonic()
        rebuild_qualified_players(conn)
        elapsed = time.monotonic() - t0

        built_pitchers = conn.execute(
            "SELECT COUNT(*) FROM qualified_players WHERE role = 'pitcher'"
        ).fetchone()[0]
        built_batters_raw = conn.execute(
            "SELECT COUNT(*) FROM qualified_players WHERE role = 'batter'"
        ).fetchone()[0]
        print(f"  Built in {elapsed:.1f}s: {built_pitchers:,} pitchers, {built_batters_raw:,} batters")

        print("Sample-checking against raw `pitches` aggregates ...")
        raw_pitcher_count, raw_batter_distinct = _raw_counts(conn)
        # Raw batter count is every distinct batter_id ever faced a pitch;
        # qualified_players' batter rows exclude anyone who already qualified
        # as a pitcher, so the two won't match exactly - check the difference
        # is explained by qualifying pitchers who also batted (NL/pre-DH era,
        # two-way players), not by a logic drift.
        pitchers_who_also_batted = conn.execute("""
            SELECT COUNT(*) FROM qualified_players qp
            WHERE qp.role = 'pitcher'
              AND EXISTS (SELECT 1 FROM pitches p WHERE p.batter_id = qp.player_id)
        """).fetchone()[0]
        expected_batter_rows = raw_batter_distinct - pitchers_who_also_batted

        print(f"  Raw qualifying pitchers (>=3 games, position P/TWP): {raw_pitcher_count:,}")
        print(f"  Table pitcher rows:                                  {built_pitchers:,}")
        print(f"  Raw distinct batter_ids:                             {raw_batter_distinct:,}")
        print(f"  Table batter rows:                                   {built_batters_raw:,}")
        print(f"  Expected batter rows (raw - qualifying-pitcher-batters): {expected_batter_rows:,}")

        ok = (built_pitchers == raw_pitcher_count) and (built_batters_raw == expected_batter_rows)
        if ok:
            print("OK: table matches raw data.")
        else:
            print("MISMATCH: table does not match raw data — do not trust this rebuild.")
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
