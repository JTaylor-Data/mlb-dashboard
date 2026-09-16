"""
Pulls FanGraphs fWAR (batting + pitching) into player_advanced_stats via
pybaseball's batting_stats()/pitching_stats(), crosswalked to MLBAM IDs
via the Chadwick Bureau register (chadwick_register()).

FanGraphs direct page access is Cloudflare-blocked (see pull_sabermetrics.py),
but pybaseball's stat-export endpoints use a different FanGraphs URL that
bypasses the block. This script tests whether that route is live; if not,
it exits cleanly with a non-zero status so the caller knows fWAR wasn't pulled.

The fWAR columns (fwar_batting, fwar_pitching) are stored separately from
the existing bWAR columns (war_batting, war_pitching, which come from the
MLB Stats API / Baseball Reference) so both can be displayed and labeled
correctly in the UI.

Usage:
    python pull_fangraphs_stats.py [SEASON]   # defaults to 2026

Re-running refreshes. Players not in the FanGraphs leaderboard (too few
PA/BF) keep NULL.
"""
import sqlite3
import sys

DB_PATH = "mlb_data.db"
DEFAULT_SEASON = 2026


def _ensure_columns(conn):
    """Add fwar columns if they don't exist yet."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(player_advanced_stats)")}
    if "fwar_batting" not in cols:
        conn.execute("ALTER TABLE player_advanced_stats ADD COLUMN fwar_batting REAL")
    if "fwar_pitching" not in cols:
        conn.execute("ALTER TABLE player_advanced_stats ADD COLUMN fwar_pitching REAL")
    conn.commit()


def _build_crosswalk():
    """Returns {fg_id -> mlbam_id} from the Chadwick Bureau register."""
    from pybaseball import chadwick_register
    print("Fetching Chadwick crosswalk...", end=" ", flush=True)
    reg = chadwick_register()
    sub = reg[["key_mlbam", "key_fangraphs"]].dropna()
    sub = sub[sub["key_mlbam"] > 0]
    crosswalk = {int(row.key_fangraphs): int(row.key_mlbam) for row in sub.itertuples()}
    print(f"done, {len(crosswalk):,} entries")
    return crosswalk


def run(season: int):
    try:
        from pybaseball import batting_stats, pitching_stats
    except ImportError:
        print("pybaseball not installed — skipping FanGraphs pull.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    _ensure_columns(conn)

    crosswalk = _build_crosswalk()

    print(f"Pulling FanGraphs batting stats for {season}...", end=" ", flush=True)
    try:
        bat = batting_stats(season, qual=0)
    except Exception as e:
        print(f"FAILED ({e}) — FanGraphs may still be blocked. Exiting.")
        conn.close()
        sys.exit(1)
    print(f"done, {len(bat)} rows")

    print(f"Pulling FanGraphs pitching stats for {season}...", end=" ", flush=True)
    try:
        pit = pitching_stats(season, qual=0)
    except Exception as e:
        print(f"FAILED ({e})")
        conn.close()
        sys.exit(1)
    print(f"done, {len(pit)} rows")

    # FanGraphs uses "IDfg" as the player id column, "WAR" as fWAR
    bat_war = {}
    for _, row in bat.iterrows():
        fg_id = row.get("IDfg")
        war = row.get("WAR")
        if fg_id is not None and war is not None:
            mlb_id = crosswalk.get(int(fg_id))
            if mlb_id:
                bat_war[mlb_id] = float(war)

    pit_war = {}
    for _, row in pit.iterrows():
        fg_id = row.get("IDfg")
        war = row.get("WAR")
        if fg_id is not None and war is not None:
            mlb_id = crosswalk.get(int(fg_id))
            if mlb_id:
                pit_war[mlb_id] = float(war)

    all_ids = set(bat_war) | set(pit_war)
    rows = [
        (bat_war.get(pid), pit_war.get(pid), pid, season)
        for pid in all_ids
    ]

    # Upsert: ensure a row exists first, then update fwar columns
    conn.executemany("""
        INSERT INTO player_advanced_stats (player_id, season)
        VALUES (?, ?)
        ON CONFLICT(player_id, season) DO NOTHING
    """, [(pid, season) for pid in all_ids])

    conn.executemany("""
        UPDATE player_advanced_stats
        SET fwar_batting = ?, fwar_pitching = ?
        WHERE player_id = ? AND season = ?
    """, rows)
    conn.commit()
    conn.close()

    print(f"\nUpserted fWAR for {len(all_ids)} players "
          f"({len(bat_war)} batting, {len(pit_war)} pitching) for {season}.")
    print("Crosswalk miss rate: "
          f"{len(bat) + len(pit) - len(all_ids)}/{len(bat) + len(pit)} rows had no MLBAM match.")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
