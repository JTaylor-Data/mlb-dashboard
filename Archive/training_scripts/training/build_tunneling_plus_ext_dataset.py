"""
Tunneling+ v3 EXT - dataset construction (Parts A, B, C), same logic as
build_tunneling_v3_dataset.py, extended to the 2015-2026 backfilled range.

IMPORTANT deviation from the live v3 build, flagged explicitly: the live
build reuses a precomputed v2 commit-point table that only covers
2021-2026 pitcher-seasons (training/tunneling_v2_artifacts/200ms/
pitcher_season_commit_point.csv). That table cannot simply be "reused" for
2015-2020 - those pitcher-seasons aren't in it at all, which would silently
drop every 2015-2020 pitcher from the lag features (sequential_lag_features
merges on pitcher_id/season). Rather than assume coverage, this script
regenerates the SAME commit-point table using analytics.tunneling_features.
build_commit_point_table() - the identical function/methodology (150ms
default offset; called here at 0.200s to match the "200ms" table actually
in production use) that produced the original 2021-2026 table - across the
full ALL_SEASONS range. Same formula, same offset, just computed over more
seasons instead of reusing a table that doesn't cover them.

Usage:
    python training/build_tunneling_plus_ext_dataset.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analytics.tunneling_features import (
    build_commit_point_table, lag_feature_columns, load_pitches, sequential_lag_features,
)

DB_PATH = "mlb_data.db"
ARTIFACTS_DIR = Path("training/tunneling_plus_ext_artifacts")
COMMIT_TIME_OFFSET_S = 0.200  # matches the "200ms" table used in production (see module docstring)
ALL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]  # EXT: full backfilled range, no structural gaps for this model's feature set
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

    print(f"\nRegenerating the 200ms commit-point table across all {len(ALL_SEASONS)} seasons "
          f"(same build_commit_point_table() function/offset as the live v2 table, which only "
          f"covers 2021-2026 and would silently exclude every 2015-2020 pitcher otherwise)...", flush=True)
    commit_table = build_commit_point_table(all_df, time_offset_s=COMMIT_TIME_OFFSET_S)
    commit_table.to_csv(ARTIFACTS_DIR / "pitcher_season_commit_point_ext.csv", index=False)
    print(f"  {len(commit_table):,} pitcher-seasons "
          f"(seasons present: {sorted(commit_table['season'].unique().tolist())})")

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
