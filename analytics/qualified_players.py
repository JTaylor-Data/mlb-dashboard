"""
analytics/qualified_players.py — Build the `qualified_players` table.

One row per (player_id, role) that qualifies for /api/players/all: a pitcher
with >=3 career games pitched (position P/TWP), or any batter who has a
batted-ball season on record and isn't already counted as a qualifying
pitcher. This is the exact same logic _all_players in api/routers/players.py
used to run live, as a full GROUP BY scan over `pitches`, on every cache
miss - moved here so it runs once (via tools/rebuild_qualified_players.py,
or lazily on first use) instead of on every boot/TTL expiry.
"""
import sqlite3


def rebuild_qualified_players(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM qualified_players")
    conn.execute("""
        INSERT INTO qualified_players (player_id, role, latest_season)
        SELECT pl.player_id, 'pitcher', p.latest_season
        FROM players pl
        JOIN (
            SELECT pitcher_id, MAX(season) AS latest_season
            FROM pitches
            GROUP BY pitcher_id
            HAVING COUNT(DISTINCT game_id) >= 3
        ) p ON pl.player_id = p.pitcher_id
        WHERE pl.position IN ('P', 'TWP')
    """)
    conn.execute("""
        INSERT OR IGNORE INTO qualified_players (player_id, role, latest_season)
        SELECT pl.player_id, 'batter', b.latest_season
        FROM players pl
        JOIN (SELECT batter_id, MAX(season) AS latest_season FROM pitches GROUP BY batter_id) b
          ON pl.player_id = b.batter_id
        WHERE pl.player_id NOT IN (SELECT player_id FROM qualified_players WHERE role = 'pitcher')
    """)
    conn.commit()
