"""
Tunneling+ v3 - dataset construction (Parts A, B, C). Real within-at-bat
lag pairing (up to 6 pitches back, same-type prior pitches included - see
analytics/tunneling_features.py's sequential_lag_features() docstring for
the full logic and the worked-example verification already run against
real data), using v2's 200ms commit_point_distance table (the best-
performing commit-point definition from the v2 hand-built grid - reused
here, not re-derived).

Single qualifying threshold (Part C): a pitcher-season needs >= 100 total
pitches that season (post PO/UN/EP exclusion) to be included AT ALL - not
a per-pitch-type count, not a pair count. v1/v2's MIN_PITCHES_PER_TYPE
does not apply to v3.

Usage:
    python training/build_tunneling_v3_dataset.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analytics.tunneling_features import (
    lag_feature_columns, load_pitches, sequential_lag_features,
)

DB_PATH = "mlb_data.db"
ARTIFACTS_DIR = Path("training/tunneling_v3_artifacts")
V2_COMMIT_TABLE_PATH = Path("training/tunneling_v2_artifacts/200ms/pitcher_season_commit_point.csv")
ALL_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
MIN_SEASON_PITCHES = 100  # Part C - total pitches that pitcher-season, post-exclusion. Not per-type, not per-pair.


def main():
    import sqlite3
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    print(f"Loading pitches for seasons {ALL_SEASONS} (PO/UN/EP excluded at load time, KN kept)...", flush=True)
    t0 = time.time()
    frames = [load_pitches(conn, s) for s in ALL_SEASONS]
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  {len(all_df):,} raw rows in {time.time()-t0:.0f}s", flush=True)

    print(f"\nReusing v2's 200ms commit-point table: {V2_COMMIT_TABLE_PATH}", flush=True)
    commit_table = pd.read_csv(V2_COMMIT_TABLE_PATH)
    print(f"  {len(commit_table):,} pitcher-seasons")

    # ---------------------------------------------------------- Part C
    print(f"\n=== Part C: qualifying threshold (>= {MIN_SEASON_PITCHES} total pitches/season, post-exclusion) ===", flush=True)
    season_totals = all_df.groupby(["pitcher_id", "season"]).size().rename("season_total_pitches").reset_index()
    qualifying = season_totals[season_totals["season_total_pitches"] >= MIN_SEASON_PITCHES]
    print(f"  {len(season_totals):,} total pitcher-seasons -> {len(qualifying):,} qualify (>= {MIN_SEASON_PITCHES} pitches)")
    qualifying_keys = set(map(tuple, qualifying[["pitcher_id", "season"]].to_numpy()))

    idx = all_df.set_index(["pitcher_id", "season"]).index
    all_df = all_df[[k in qualifying_keys for k in idx]].copy()
    print(f"  {len(all_df):,} rows remain after the qualifying-pitcher-season filter")

    # ------------------------------------------------------- Parts A/B
    print(f"\n=== Parts A/B: real within-at-bat lag pairing (max_lag=6) ===", flush=True)
    t0 = time.time()
    feat = sequential_lag_features(all_df, commit_table)
    print(f"  {len(feat):,} training rows (one per real 2nd-or-later pitch in its at-bat) in {time.time()-t0:.0f}s", flush=True)

    lag_cols = lag_feature_columns()
    other_cols = ["pitcher_id", "season", "pitch_type", "p_throws", "stand", "delta_run_exp"]
    print(f"\n=== Schema confirmation ===")
    print(f"  {len(lag_cols)} lag-derived columns (6 slots x 4 columns): {lag_cols}")
    print(f"  + current-pitch columns: {other_cols}")
    print(f"  Full training-relevant column count: {len(lag_cols) + len(other_cols)}")
    print(f"  All columns actually in the saved table: {len(feat.columns)} -> {sorted(feat.columns.tolist())}")

    print(f"\n=== Null-rate by lag slot (expected: increasing with lag - fewer at-bats reach deeper lags) ===")
    for lag in range(1, 7):
        null_pct = feat[f"lag{lag}_pitch_type"].isna().mean() * 100
        print(f"  lag{lag}: {null_pct:.1f}% null")

    out_path = ARTIFACTS_DIR / "tunneling_v3_features.parquet"
    feat.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path} ({len(feat):,} rows, {len(feat.columns)} columns)")


if __name__ == "__main__":
    main()
