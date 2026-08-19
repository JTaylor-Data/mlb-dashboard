"""
Incremental Statcast pitch-level ingestion into mlb_data.db.

Usage:
    python pull_statcast.py START_DATE END_DATE
    python pull_statcast.py               # uses DEFAULT_START / DEFAULT_END below

Pulls in weekly chunks, inserting after each chunk completes. Dates already
present in the `ingested_dates` tracking table are skipped UNLESS they fall
within the last RECENT_WINDOW_DAYS calendar days (see below - a day touched
while games were still in progress needs revisiting once the rest of that
day's games finish, and `ingested_dates` alone can't tell "fully done" apart
from "touched once"). Row-level duplicates are prevented by a real
UNIQUE(game_id, at_bat_number, pitch_number) index on `pitches` plus
INSERT OR IGNORE - re-pulling an already-partially-present day is safe by
construction, not just by convention.
"""
import sys
import sqlite3
from datetime import date, timedelta

import pandas as pd
from pybaseball import statcast

DB_PATH = "mlb_data.db"
CHUNK_DAYS = 7

# A date gets marked "ingested" in `ingested_dates` the first time ANY row for
# it lands - which used to mean a day touched mid-slate (games still in
# progress) got marked done and never revisited, permanently missing
# whatever finished later that day. Fix: always re-pull the most recent
# few days on every run regardless of `ingested_dates`, no matter how long
# it's been since the last refresh. This is only safe because of the real
# UNIQUE(game_id, at_bat_number, pitch_number) index on `pitches` (added
# alongside this fix) - re-pulling a day that's already partially present
# now safely no-ops on rows that already exist via INSERT OR IGNORE,
# instead of relying on a constraint that (before this fix) didn't
# actually exist despite this file's own docstring claiming it did.
RECENT_WINDOW_DAYS = 3

# 2026 Opening Night was 3/25, Opening Day 3/26. Default test range covers
# the first week of the season.
DEFAULT_START = "2026-03-25"
DEFAULT_END = "2026-04-01"

# raw pybaseball/Statcast column -> our pitches column
COLUMN_MAP = {
    "game_pk": "game_id",
    "game_date": "game_date",
    "game_year": "season",
    "at_bat_number": "at_bat_number",
    "pitch_number": "pitch_number",
    "pitcher": "pitcher_id",
    "batter": "batter_id",
    "pitch_type": "pitch_type",
    "release_speed": "release_speed",
    "release_spin_rate": "release_spin_rate",
    "release_pos_x": "release_pos_x",
    "release_pos_y": "release_pos_y",
    "release_pos_z": "release_pos_z",
    "pfx_x": "pfx_x",
    "pfx_z": "pfx_z",
    "plate_x": "plate_x",
    "plate_z": "plate_z",
    "zone": "zone",
    "balls": "balls",
    "strikes": "strikes",
    "outs_when_up": "outs_when_up",
    "inning": "inning",
    "stand": "stand",
    "p_throws": "p_throws",
    "description": "description",
    "events": "events",
    "launch_speed": "launch_speed",
    "launch_angle": "launch_angle",
    "delta_run_exp": "delta_run_exp",
    "woba_value": "woba_value",
    "estimated_woba_using_speedangle": "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle": "estimated_slg_using_speedangle",
    "estimated_ba_using_speedangle": "estimated_ba_using_speedangle",
    "vx0": "vx0",
    "vy0": "vy0",
    "vz0": "vz0",
    "ax": "ax",
    "ay": "ay",
    "az": "az",
    "release_extension": "release_extension",
    "spin_axis": "spin_axis",
    "effective_speed": "effective_speed",
    "arm_angle": "arm_angle",
    "sz_top": "sz_top",
    "sz_bot": "sz_bot",
    "inning_topbot": "inning_topbot",
    "woba_denom": "woba_denom",
    "bb_type": "bb_type",
    "hc_x": "hc_x",
    "hc_y": "hc_y",
    "bat_speed": "bat_speed",
    "swing_length": "swing_length",
    "attack_angle": "attack_angle",
    "attack_direction": "attack_direction",
    "swing_path_tilt": "swing_path_tilt",
    "delta_home_win_exp": "delta_home_win_exp",
    "bat_score": "bat_score",
    "post_bat_score": "post_bat_score",
    "post_home_score": "post_home_score",
    "post_away_score": "post_away_score",
}
TARGET_COLUMNS = list(COLUMN_MAP.values())


def daterange_chunks(start, end, days=CHUNK_DAYS):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def get_uningested_dates(conn, start, end):
    already = {
        row[0]
        for row in conn.execute("SELECT game_date FROM ingested_dates").fetchall()
    }
    all_dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    recent_cutoff = date.today() - timedelta(days=RECENT_WINDOW_DAYS)
    return [d for d in all_dates if d.isoformat() not in already or d >= recent_cutoff]


def mark_dates_ingested(conn, dates, rows_by_date):
    now = pd.Timestamp.now().isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO ingested_dates (game_date, rows_inserted, ingested_at) "
        "VALUES (?, ?, ?)",
        [(d.isoformat(), rows_by_date.get(d.isoformat(), 0), now) for d in dates],
    )
    conn.commit()


def validate_columns(df, label):
    df_cols = set(df.columns)
    expected = set(COLUMN_MAP.keys())
    missing = expected - df_cols
    extra = df_cols - expected
    if missing:
        print(f"  WARNING [{label}]: Statcast response is missing expected columns "
              f"{sorted(missing)} - these will be stored as NULL")
    if extra:
        sample = sorted(extra)[:5]
        print(f"  NOTE [{label}]: Statcast returned {len(extra)} column(s) not in our "
              f"schema (e.g. {sample}) - ignored")


def insert_chunk(conn, df, label):
    validate_columns(df, label)

    for raw_col in COLUMN_MAP:
        if raw_col not in df.columns:
            df[raw_col] = None

    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

    before = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    df.to_sql("_staging_pitches", conn, if_exists="replace", index=False)
    conn.execute(
        f"INSERT OR IGNORE INTO pitches ({', '.join(TARGET_COLUMNS)}) "
        f"SELECT {', '.join(TARGET_COLUMNS)} FROM _staging_pitches"
    )
    conn.execute("DROP TABLE _staging_pitches")
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]

    rows_by_date = df.groupby("game_date").size().to_dict()
    return after - before, rows_by_date


def run(start_date: str, end_date: str):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    conn = sqlite3.connect(DB_PATH)

    total_inserted = 0
    failures = []

    for chunk_start, chunk_end in daterange_chunks(start, end):
        label = f"{chunk_start} to {chunk_end}"
        missing = get_uningested_dates(conn, chunk_start, chunk_end)

        if not missing:
            print(f"Skipping {label} (already ingested)")
            continue

        pull_start, pull_end = min(missing), max(missing)
        print(f"Pulling {pull_start} to {pull_end}...", end=" ", flush=True)

        try:
            df = statcast(start_dt=pull_start.isoformat(), end_dt=pull_end.isoformat())
        except Exception as e:
            print("FAILED")
            print(f"  ERROR: pull failed for {pull_start} to {pull_end}: {e}")
            failures.append((str(pull_start), str(pull_end), str(e)))
            continue

        if df is None or df.empty:
            print("done, 0 rows (no games in range)")
            mark_dates_ingested(conn, missing, {})
            continue

        try:
            inserted, rows_by_date = insert_chunk(conn, df, label)
        except Exception as e:
            print("FAILED")
            print(f"  ERROR: insert failed for {pull_start} to {pull_end}: {e}")
            failures.append((str(pull_start), str(pull_end), str(e)))
            continue

        mark_dates_ingested(conn, missing, rows_by_date)
        total_inserted += inserted
        print(f"done, {inserted:,} rows inserted")

    conn.close()

    print(f"\nTotal rows inserted this run: {total_inserted:,}")
    if failures:
        print(f"{len(failures)} chunk(s) FAILED - re-run the script to retry them "
              f"(succeeded chunks will be skipped):")
        for s, e, err in failures:
            print(f"  {s} to {e}: {err}")
    else:
        print("All chunks succeeded.")

    return dict(rows_inserted=total_inserted, failures=len(failures))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        run(DEFAULT_START, DEFAULT_END)
    else:
        print("Usage: python pull_statcast.py [START_DATE END_DATE]")
        sys.exit(1)
