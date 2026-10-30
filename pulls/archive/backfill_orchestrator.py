"""
Resumable, checkpointed Statcast backfill for seasons 2015-2020.

Usage:
    python -m pulls.backfill_orchestrator              # run all pending seasons
    python -m pulls.backfill_orchestrator --status     # print checkpoint table and exit
    python -m pulls.backfill_orchestrator --seasons 2020 2019
    python -m pulls.backfill_orchestrator --reset 2018 # mark one season pending again

Designed to run unattended for hours. Every season is pulled, validated and
checkpointed independently, so an interruption at any point (sleep, crash,
network drop) loses at most the season in flight - a restart resumes at the
first season not marked `done`, and never re-pulls a completed one.

Season date bounds come from the MLB schedule API rather than being
hardcoded, so the COVID-shortened 2020 season and the various international
openers are handled correctly instead of being guessed at. Combined with
pull_statcast's game_type filter, this is what keeps spring training and
postseason out of the table.

This script pulls and validates only. It deliberately does NOT retrain any
model - that is a separate, deliberate decision to be made after reviewing
the completeness audit this produces.
"""
import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulls import pull_statcast

DB_PATH = Path(__file__).resolve().parent.parent / "mlb_data.db"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH = LOG_DIR / "backfill_2015_2020.log"

# Newest first: 2020 sits right against the existing 2021 boundary, so if
# something about the ingest is subtly wrong it shows up against known-good
# neighbouring data immediately rather than six seasons deep.
TARGET_SEASONS = [2020, 2019, 2018, 2017, 2016, 2015]

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def log(msg):
    """Timestamped, line-buffered, to both console and the persistent log -
    this job runs unattended for hours and gets checked remotely, so console
    output alone is not enough."""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_checkpoint_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backfill_checkpoint (
            season        INTEGER PRIMARY KEY,
            status        TEXT NOT NULL,   -- pending | in_progress | done | failed
            rows_pulled   INTEGER,
            games_pulled  INTEGER,
            expected_games INTEGER,
            start_date    TEXT,
            end_date      TEXT,
            started_at    TEXT,
            completed_at  TEXT,
            notes         TEXT
        )
    """)
    conn.commit()


def get_status(conn, season):
    row = conn.execute(
        "SELECT status FROM backfill_checkpoint WHERE season = ?", (season,)
    ).fetchone()
    return row[0] if row else "pending"


def set_status(conn, season, status, **fields):
    ensure_checkpoint_table(conn)
    existing = conn.execute(
        "SELECT season FROM backfill_checkpoint WHERE season = ?", (season,)
    ).fetchone()
    if existing is None:
        conn.execute("INSERT INTO backfill_checkpoint (season, status) VALUES (?, ?)",
                     (season, status))
    sets = ["status = ?"]
    params = [status]
    for k, v in fields.items():
        sets.append(f"{k} = ?")
        params.append(v)
    params.append(season)
    conn.execute(f"UPDATE backfill_checkpoint SET {', '.join(sets)} WHERE season = ?", params)
    conn.commit()


def season_bounds(season):
    """Real first/last regular-season game dates and the authoritative game
    count, straight from MLB's own schedule. Hardcoding these is how you
    silently truncate 2020 (COVID: July 23 start, 60 games) or miss an
    international opener."""
    r = requests.get(SCHEDULE_URL, params={
        "sportId": 1,
        "startDate": f"{season}-01-01",
        "endDate": f"{season}-12-31",
        "gameType": "R",
    }, timeout=120)
    r.raise_for_status()
    data = r.json()

    dates, game_ids = [], set()
    for d in data.get("dates", []):
        for g in d.get("games", []):
            # Cancelled games never generate pitches - counting them would
            # make every validation look permanently one game short.
            if g["status"]["detailedState"] in ("Cancelled", "Postponed"):
                continue
            dates.append(d["date"])
            game_ids.add(g["gamePk"])
    if not dates:
        raise RuntimeError(f"MLB schedule API returned no regular-season games for {season}")
    return min(dates), max(dates), len(game_ids)


def validate_season(conn, season, expected_games):
    """A season is only checkpointed `done` if it actually looks like a
    season: right game count, right date window, no foreign game types."""
    row = conn.execute("""
        SELECT COUNT(*), COUNT(DISTINCT game_id), MIN(game_date), MAX(game_date)
        FROM pitches WHERE season = ?
    """, (season,)).fetchone()
    rows, games, min_d, max_d = row

    bad_types = conn.execute("""
        SELECT game_type, COUNT(*) FROM pitches
        WHERE season = ? AND game_type IS NOT NULL AND game_type != 'R'
        GROUP BY game_type
    """, (season,)).fetchall()

    problems = []
    if rows == 0:
        problems.append("no rows ingested")
    if bad_types:
        problems.append(f"non-regular-season rows present: {dict(bad_types)}")
    # Statcast coverage for a played game is essentially total; anything
    # below ~98% means real games are missing, not just incidental gaps.
    if expected_games and games < expected_games * 0.98:
        problems.append(f"only {games}/{expected_games} games ({games / expected_games:.1%})")

    return dict(rows=rows, games=games, min_date=min_d, max_date=max_d,
                expected_games=expected_games, problems=problems)


def run_season(conn, season):
    status = get_status(conn, season)
    if status == "done":
        log(f"{season}: already done, skipping")
        return True

    log(f"{season}: === STARTING === (previous status: {status})")
    try:
        start, end, expected_games = season_bounds(season)
    except Exception as e:
        log(f"{season}: FAILED to fetch schedule bounds: {e}")
        set_status(conn, season, "failed", notes=f"schedule lookup failed: {e}")
        return False

    log(f"{season}: regular season runs {start} to {end}, {expected_games} games scheduled")
    set_status(conn, season, "in_progress", start_date=start, end_date=end,
               expected_games=expected_games,
               started_at=datetime.now().isoformat(timespec="seconds"), notes=None)

    t0 = time.time()
    try:
        result = pull_statcast.run(start, end)
    except Exception as e:
        log(f"{season}: FAILED during pull: {e}")
        set_status(conn, season, "failed", notes=f"pull raised: {e}")
        return False

    elapsed = time.time() - t0
    log(f"{season}: pull finished in {elapsed / 60:.1f} min, "
        f"{result['rows_inserted']:,} rows inserted, {result['failures']} chunk failure(s)")

    v = validate_season(conn, season, expected_games)
    log(f"{season}: validation - {v['rows']:,} rows, {v['games']}/{v['expected_games']} games, "
        f"dates {v['min_date']} to {v['max_date']}")

    if result["failures"]:
        v["problems"].append(f"{result['failures']} chunk(s) failed during pull")

    if v["problems"]:
        note = "; ".join(v["problems"])
        log(f"{season}: !! VALIDATION FAILED: {note}")
        set_status(conn, season, "failed", rows_pulled=v["rows"], games_pulled=v["games"],
                   notes=note)
        return False

    set_status(conn, season, "done", rows_pulled=v["rows"], games_pulled=v["games"],
               completed_at=datetime.now().isoformat(timespec="seconds"), notes=None)
    log(f"{season}: === DONE === checkpointed")
    return True


def print_status(conn):
    ensure_checkpoint_table(conn)
    rows = conn.execute("""
        SELECT season, status, rows_pulled, games_pulled, expected_games,
               start_date, end_date, completed_at, notes
        FROM backfill_checkpoint ORDER BY season DESC
    """).fetchall()
    if not rows:
        print("No checkpoint rows yet - nothing has been attempted.")
        return
    print(f"{'season':>6}  {'status':<12} {'rows':>10} {'games':>12}  {'window':<25} notes")
    for s, st, rp, gp, eg, sd, ed, ca, note in rows:
        games = f"{gp or 0}/{eg or 0}"
        window = f"{sd or '?'} to {ed or '?'}"
        print(f"{s:>6}  {st:<12} {(rp or 0):>10,} {games:>12}  {window:<25} {note or ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="print checkpoint state and exit")
    ap.add_argument("--seasons", nargs="+", type=int, help="override which seasons to run")
    ap.add_argument("--reset", nargs="+", type=int, help="mark seasons pending again, then exit")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_checkpoint_table(conn)

    if args.status:
        print_status(conn)
        return
    if args.reset:
        for s in args.reset:
            set_status(conn, s, "pending", notes="manually reset")
            print(f"{s}: reset to pending")
        return

    seasons = args.seasons or TARGET_SEASONS
    log(f"BACKFILL RUN STARTED - seasons {seasons}")
    done, failed = [], []
    for season in seasons:
        ok = run_season(conn, season)
        (done if ok else failed).append(season)

    log(f"BACKFILL RUN FINISHED - succeeded: {done or 'none'}, failed: {failed or 'none'}")
    if failed:
        log(f"Re-run this script to retry the {len(failed)} failed season(s); "
            f"completed seasons will be skipped.")
    log("NOTE: no model has been retrained. Review the completeness audit "
        "(python -m analytics.completeness_audit) before any retrain decision.")
    print()
    print_status(conn)
    conn.close()


if __name__ == "__main__":
    main()
