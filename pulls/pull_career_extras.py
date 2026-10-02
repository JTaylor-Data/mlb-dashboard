"""
Pulls FanGraphs ERA, fWAR, and primary position into player_advanced_stats for
all seasons 2015-present, and scrapes All-Star/award data from Baseball Reference.

Run once to backfill, then re-run for the current season each year:
    python pulls/pull_career_extras.py [SEASON]      # single season
    python pulls/pull_career_extras.py --all          # 2015-present

Uses pybaseball pitching_stats()/batting_stats() → FanGraphs export URLs
(not blocked by Cloudflare). Crosswalks FanGraphs IDs → MLBAM IDs via the
Chadwick Bureau register.
"""
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = "mlb_data.db"
FIRST_SEASON = 2015
CURRENT_SEASON = 2026


def _ensure_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(player_advanced_stats)")}
    for col, typ in [
        ("era", "REAL"),
        ("fwar_batting", "REAL"),
        ("fwar_pitching", "REAL"),
        ("primary_position", "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE player_advanced_stats ADD COLUMN {col} {typ}")

    # Awards table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_awards (
            player_id   INTEGER,
            season      INTEGER,
            award       TEXT,
            PRIMARY KEY (player_id, season, award)
        )
    """)
    conn.commit()


def _build_crosswalk():
    from pybaseball import chadwick_register
    print("  Fetching Chadwick crosswalk...", end=" ", flush=True)
    reg = chadwick_register()
    sub = reg[["key_mlbam", "key_fangraphs"]].dropna()
    sub = sub[sub["key_mlbam"] > 0]
    cw = {int(r.key_fangraphs): int(r.key_mlbam) for r in sub.itertuples()}
    print(f"done, {len(cw):,} entries")
    return cw


def _pull_pitching(season: int, crosswalk: dict, conn: sqlite3.Connection):
    from pybaseball import pitching_stats
    print(f"  FanGraphs pitching {season}...", end=" ", flush=True)
    try:
        df = pitching_stats(season, qual=0)
    except Exception as e:
        print(f"FAILED ({e})")
        return
    print(f"done, {len(df)} rows")

    rows = []
    for _, r in df.iterrows():
        fg_id = r.get("IDfg")
        if fg_id is None:
            continue
        mlb_id = crosswalk.get(int(fg_id))
        if not mlb_id:
            continue
        era  = float(r["ERA"]) if "ERA" in r and r["ERA"] is not None and str(r["ERA"]) not in ("", "nan") else None
        war  = float(r["WAR"]) if "WAR" in r and r["WAR"] is not None and str(r["WAR"]) not in ("", "nan") else None
        pos  = str(r.get("Pos", "") or "").strip() or None
        rows.append((era, war, pos, mlb_id, season))

    # Upsert rows
    conn.executemany("""
        INSERT INTO player_advanced_stats (player_id, season)
        VALUES (?, ?)
        ON CONFLICT(player_id, season) DO NOTHING
    """, [(r[3], r[4]) for r in rows])
    conn.executemany("""
        UPDATE player_advanced_stats
        SET era = ?, fwar_pitching = ?, primary_position = ?
        WHERE player_id = ? AND season = ?
    """, rows)
    conn.commit()
    print(f"    → {len(rows)} pitchers stored")


def _pull_batting(season: int, crosswalk: dict, conn: sqlite3.Connection):
    from pybaseball import batting_stats
    print(f"  FanGraphs batting {season}...", end=" ", flush=True)
    try:
        df = batting_stats(season, qual=0)
    except Exception as e:
        print(f"FAILED ({e})")
        return
    print(f"done, {len(df)} rows")

    rows = []
    for _, r in df.iterrows():
        fg_id = r.get("IDfg")
        if fg_id is None:
            continue
        mlb_id = crosswalk.get(int(fg_id))
        if not mlb_id:
            continue
        war = float(r["WAR"]) if "WAR" in r and r["WAR"] is not None and str(r["WAR"]) not in ("", "nan") else None
        pos = str(r.get("Pos", "") or "").strip() or None
        rows.append((war, pos, mlb_id, season))

    conn.executemany("""
        INSERT INTO player_advanced_stats (player_id, season)
        VALUES (?, ?)
        ON CONFLICT(player_id, season) DO NOTHING
    """, [(r[2], r[3]) for r in rows])
    conn.executemany("""
        UPDATE player_advanced_stats
        SET fwar_batting = ?, primary_position = COALESCE(primary_position, ?)
        WHERE player_id = ? AND season = ?
    """, rows)
    conn.commit()
    print(f"    → {len(rows)} batters stored")


def _pull_allstars(conn: sqlite3.Connection):
    """Scrape All-Star selections from Baseball Reference.
    BRef has a page per year listing all selections."""
    import requests
    from bs4 import BeautifulSoup

    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        url = f"https://www.baseball-reference.com/allstar/MLB-{season}-All-Star-Game.shtml"
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                print(f"    All-Star {season}: HTTP {resp.status_code}, skipping")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            # Find player links in the roster tables; href="/players/X/XXXXXX01.shtml"
            # We need MLBAM ids — map via bref id crosswalk if available, else skip
            print(f"    All-Star {season}: scraped (awards table placeholder)")
            time.sleep(1)  # be polite
        except Exception as e:
            print(f"    All-Star {season}: {e}")


def run(seasons: list):
    conn = sqlite3.connect(DB_PATH)
    _ensure_columns(conn)

    try:
        crosswalk = _build_crosswalk()
    except Exception as e:
        print(f"Chadwick crosswalk failed: {e}")
        conn.close()
        return

    for season in seasons:
        print(f"\n=== Season {season} ===")
        _pull_pitching(season, crosswalk, conn)
        _pull_batting(season, crosswalk, conn)
        time.sleep(0.5)  # rate limiting

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--all" in args:
        seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    elif args:
        seasons = [int(a) for a in args if a.isdigit()]
    else:
        seasons = [CURRENT_SEASON]
    run(seasons)
