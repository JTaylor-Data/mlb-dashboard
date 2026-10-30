"""
Precomputes Stuff+/Tunneling+/Location+ per-pitcher-season scores and stores
them in pitcher_model_scores. Run once to populate all historical seasons;
re-run for the current season after each weekly model retrain.

    python pulls/precompute_model_scores.py              # 2021-current
    python pulls/precompute_model_scores.py 2024         # single season
    python pulls/precompute_model_scores.py --all        # 2021-current (same)

Career page reads from this table instead of re-running the model on every request.
"""
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = "mlb_data.db"
FIRST_SEASON = 2015   # Statcast tracking began 2015
CURRENT_SEASON = 2026


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_model_scores (
            pitcher_id      INTEGER NOT NULL,
            season          INTEGER NOT NULL,
            stuff_plus      REAL,
            tunneling_plus  REAL,
            location_plus   REAL,
            computed_at     TEXT,
            PRIMARY KEY (pitcher_id, season)
        )
    """)
    conn.commit()


def score_season(conn, season: int):
    print(f"  Scoring season {season}...")

    from analytics.stuff_model import compute_stuff_plus, overall_stuff_plus
    from analytics.tunneling_v2_model import _raw_predictions as tun_raw
    from analytics.location_model import _raw_predictions as loc_raw
    import numpy as np
    import pandas as pd

    t0 = time.monotonic()

    # Stuff+
    try:
        stuff_tbl = compute_stuff_plus(conn, season)
        print(f"    Stuff+: {len(stuff_tbl.index.get_level_values('pitcher_id').unique())} pitchers", end="")
    except Exception as e:
        print(f"    Stuff+ FAILED: {e}")
        stuff_tbl = None
    print(f" ({time.monotonic()-t0:.1f}s)")

    # Tunneling+
    t1 = time.monotonic()
    try:
        tun_df = tun_raw(conn, season)
        if not tun_df.empty and tun_df["pred"].std() > 0:
            mean, std = tun_df["pred"].mean(), tun_df["pred"].std()
            tun_scores = (tun_df.groupby("pitcher_id")["pred"].mean() - mean) / std * 15 + 100
        else:
            tun_scores = None
        print(f"    Tunneling+: {len(tun_scores) if tun_scores is not None else 0} pitchers ({time.monotonic()-t1:.1f}s)")
    except Exception as e:
        print(f"    Tunneling+ FAILED: {e}")
        tun_scores = None

    # Location+
    t2 = time.monotonic()
    try:
        loc_df = loc_raw(conn, season)
        if not loc_df.empty and loc_df["pred"].std() > 0:
            mean, std = loc_df["pred"].mean(), loc_df["pred"].std()
            loc_scores = (loc_df.groupby("pitcher_id")["pred"].mean() - mean) / std * 15 + 100
        else:
            loc_scores = None
        print(f"    Location+: {len(loc_scores) if loc_scores is not None else 0} pitchers ({time.monotonic()-t2:.1f}s)")
    except Exception as e:
        print(f"    Location+ FAILED: {e}")
        loc_scores = None

    # Collect all pitcher IDs seen in any model
    all_pids = set()
    if stuff_tbl is not None:
        all_pids.update(stuff_tbl.index.get_level_values("pitcher_id").unique().tolist())
    if tun_scores is not None:
        all_pids.update(tun_scores.index.tolist())
    if loc_scores is not None:
        all_pids.update(loc_scores.index.tolist())

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for pid in all_pids:
        # Stuff+: usage-weighted average across pitch types
        sp = None
        if stuff_tbl is not None and pid in stuff_tbl.index.get_level_values("pitcher_id"):
            try:
                sp = round(float(overall_stuff_plus(stuff_tbl, pid)), 1)
            except Exception:
                pass

        tp = round(float(tun_scores[pid]), 1) if (tun_scores is not None and pid in tun_scores.index) else None
        lp = round(float(loc_scores[pid]), 1) if (loc_scores is not None and pid in loc_scores.index) else None

        rows.append((int(pid), season, sp, tp, lp, now))

    conn.executemany("""
        INSERT INTO pitcher_model_scores (pitcher_id, season, stuff_plus, tunneling_plus, location_plus, computed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(pitcher_id, season) DO UPDATE SET
            stuff_plus = excluded.stuff_plus,
            tunneling_plus = excluded.tunneling_plus,
            location_plus = excluded.location_plus,
            computed_at = excluded.computed_at
    """, rows)
    conn.commit()
    print(f"    → {len(rows)} pitchers stored for {season} (total {time.monotonic()-t0:.1f}s)")


def run(seasons: list):
    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)
    for season in seasons:
        score_season(conn, season)
    conn.close()
    print("\nAll seasons done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].isdigit():
        seasons = [int(args[0])]
    else:
        seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    run(seasons)
