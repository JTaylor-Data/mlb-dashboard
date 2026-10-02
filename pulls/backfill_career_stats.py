"""
Backfills ERA and WAR (batting + pitching) into player_advanced_stats for
seasons 2015-present using the MLB Stats API. Also adds primary_position.

ERA comes from the standard pitching stat group (earned runs / innings pitched).
WAR comes from the sabermetrics group (same source as pull_sabermetrics.py but
applied to all historical seasons, not just the current one).

Run once to populate all seasons:
    python pulls/backfill_career_stats.py            # 2015-present
    python pulls/backfill_career_stats.py 2023       # single season

Uses MLBAM player IDs directly — no crosswalk needed.
"""
import sqlite3
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = "mlb_data.db"
STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
FIRST_SEASON = 2015
CURRENT_SEASON = 2026


def _ensure_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(player_advanced_stats)")}
    for col, typ in [("era", "REAL"), ("fwar_batting", "REAL"), ("fwar_pitching", "REAL"), ("primary_position", "TEXT")]:
        if col not in cols:
            conn.execute(f"ALTER TABLE player_advanced_stats ADD COLUMN {col} {typ}")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_awards (
            player_id INTEGER NOT NULL,
            season    INTEGER NOT NULL,
            award     TEXT    NOT NULL,
            PRIMARY KEY (player_id, season, award)
        )
    """)
    conn.commit()


def _fetch(group, season, stat_keys, *, stat_type="season"):
    """Fetch MLB Stats API stats for a group+season. Returns {player_id: {key: val}}."""
    try:
        resp = requests.get(STATS_URL, params={
            "stats": stat_type, "group": group, "season": season,
            "sportId": 1, "limit": 2000, "playerPool": "all",
        }, timeout=30)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        out = {}
        for s in splits:
            pid = s["player"]["id"]
            vals = {}
            for k in stat_keys:
                v = s["stat"].get(k)
                if v is not None and str(v) not in ("", "-.--", "-"):
                    vals[k] = v
            if vals:
                out[pid] = vals
        return out
    except Exception as e:
        print(f"    FAILED: {e}")
        return {}


def _fetch_era(season):
    """ERA from standard pitching stats."""
    return _fetch("pitching", season, ["era"])


def _fetch_war(season):
    """WAR from sabermetrics stat group."""
    pit = _fetch("pitching", season, ["war"], stat_type="sabermetrics")
    bat = _fetch("hitting",  season, ["war"], stat_type="sabermetrics")
    return pit, bat


def _fetch_positions(season):
    """Primary position from standard fielding stats."""
    try:
        resp = requests.get(STATS_URL, params={
            "stats": "season", "group": "fielding", "season": season,
            "sportId": 1, "limit": 5000, "playerPool": "all",
        }, timeout=30)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        # For each player, find position(s) with most games
        from collections import defaultdict
        pos_games = defaultdict(lambda: defaultdict(int))
        for s in splits:
            pid = s["player"]["id"]
            pos = s.get("position", {}).get("abbreviation", "")
            gp  = s["stat"].get("gamesStarted") or s["stat"].get("games") or 0
            if pos and pos != "P":
                pos_games[pid][pos] += int(gp)
        result = {}
        for pid, pg in pos_games.items():
            total = sum(pg.values())
            if total == 0:
                continue
            sorted_pos = sorted(pg.items(), key=lambda x: -x[1])
            top_pos, top_g = sorted_pos[0]
            if top_g / total >= 0.70 or len(sorted_pos) == 1:
                result[pid] = top_pos
            else:
                result[pid] = f"{top_pos}/{sorted_pos[1][0]}"
        return result
    except Exception as e:
        print(f"    Positions FAILED: {e}")
        return {}


def run_season(season: int, conn: sqlite3.Connection):
    print(f"\n=== Season {season} ===")

    print(f"  ERA...", end=" ", flush=True)
    era_map = _fetch_era(season)
    print(f"{len(era_map)} pitchers")

    print(f"  WAR...", end=" ", flush=True)
    war_pit, war_bat = _fetch_war(season)
    print(f"{len(war_pit)} pitching, {len(war_bat)} batting")

    print(f"  Positions...", end=" ", flush=True)
    pos_map = _fetch_positions(season)
    print(f"{len(pos_map)} players")

    all_ids = set(era_map) | set(war_pit) | set(war_bat) | set(pos_map)

    # Ensure rows exist
    conn.executemany("""
        INSERT INTO player_advanced_stats (player_id, season)
        VALUES (?, ?) ON CONFLICT DO NOTHING
    """, [(pid, season) for pid in all_ids])

    for pid in all_ids:
        try:
            era = float(era_map[pid]["era"]) if pid in era_map else None
        except (ValueError, TypeError):
            era = None
        wp    = float(war_pit[pid]["war"]) if pid in war_pit else None
        wb    = float(war_bat[pid]["war"]) if pid in war_bat else None
        pos   = pos_map.get(pid)
        conn.execute("""
            UPDATE player_advanced_stats
            SET era = COALESCE(?, era),
                war_pitching = COALESCE(?, war_pitching),
                war_batting  = COALESCE(?, war_batting),
                primary_position = COALESCE(?, primary_position)
            WHERE player_id = ? AND season = ?
        """, (era, wp, wb, pos, pid, season))

    conn.commit()
    print(f"  → {len(all_ids)} players updated")


def run(seasons):
    conn = sqlite3.connect(DB_PATH)
    _ensure_columns(conn)
    for season in seasons:
        run_season(season, conn)
        time.sleep(0.3)
    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].isdigit():
        seasons = [int(args[0])]
    else:
        seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    run(seasons)
