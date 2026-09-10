"""
Tunneling+ v3 - 150ms commit-point EXPERIMENT (dataset construction).

Identical to build_tunneling_v3_dataset.py in every respect except which
commit-point table is used - v2's 150ms table instead of v2's 200ms table
(the one the live, promoted v3 model actually uses). Built to test a direct
hypothesis: would a commit point closer to the plate produce a meaningfully
wider Tunneling+ score spread? v2's own hand-built comparison already found
200ms beat 150ms on the sequential same-at-bat test (z=9.98 vs 9.28) - this
answers the separate, not-yet-tested question of what a closer point does
to v3's trained-model score spread specifically.

This is an EXPERIMENT ONLY - output goes to training/tunneling_v3_150ms_
artifacts/, never to models/, and nothing here touches the live-serving
tunneling_v3.txt/meta.json or any file analytics/tunneling_v3_model.py reads.

Usage:
    python training/build_tunneling_v3_150ms_dataset.py
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
ARTIFACTS_DIR = Path("training/tunneling_v3_150ms_artifacts")
V2_COMMIT_TABLE_PATH = Path("training/tunneling_v2_artifacts/150ms/pitcher_season_commit_point.csv")  # only line that differs from the 200ms build script, besides ARTIFACTS_DIR
ALL_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
MIN_SEASON_PITCHES = 100  # same single qualifying threshold as the live version - not part of this experiment


def main():
    import sqlite3
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    print(f"Loading pitches for seasons {ALL_SEASONS} (PO/UN/EP excluded at load time, KN kept)...", flush=True)
    t0 = time.time()
    frames = [load_pitches(conn, s) for s in ALL_SEASONS]
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  {len(all_df):,} raw rows in {time.time()-t0:.0f}s", flush=True)

    print(f"\nReusing v2's 150ms commit-point table: {V2_COMMIT_TABLE_PATH}", flush=True)
    commit_table = pd.read_csv(V2_COMMIT_TABLE_PATH)
    print(f"  {len(commit_table):,} pitcher-seasons, mean commit_point_distance={commit_table['commit_point_distance'].mean():.2f} ft", flush=True)

    print(f"\n=== Qualifying threshold (>= {MIN_SEASON_PITCHES} total pitches/season, post-exclusion) ===", flush=True)
    season_totals = all_df.groupby(["pitcher_id", "season"]).size().rename("season_total_pitches").reset_index()
    qualifying = season_totals[season_totals["season_total_pitches"] >= MIN_SEASON_PITCHES]
    print(f"  {len(season_totals):,} total pitcher-seasons -> {len(qualifying):,} qualify (>= {MIN_SEASON_PITCHES} pitches)")
    qualifying_keys = set(map(tuple, qualifying[["pitcher_id", "season"]].to_numpy()))

    idx = all_df.set_index(["pitcher_id", "season"]).index
    all_df = all_df[[k in qualifying_keys for k in idx]].copy()
    print(f"  {len(all_df):,} rows remain after the qualifying-pitcher-season filter")

    print(f"\n=== Real within-at-bat lag pairing (max_lag=6) ===", flush=True)
    t0 = time.time()
    feat = sequential_lag_features(all_df, commit_table)
    print(f"  {len(feat):,} training rows (one per real 2nd-or-later pitch in its at-bat) in {time.time()-t0:.0f}s", flush=True)

    lag_cols = lag_feature_columns()
    print(f"\n=== Null-rate by lag slot (expected: increasing with lag) ===")
    for lag in range(1, 7):
        null_pct = feat[f"lag{lag}_pitch_type"].isna().mean() * 100
        print(f"  lag{lag}: {null_pct:.1f}% null")

    out_path = ARTIFACTS_DIR / "tunneling_v3_150ms_features.parquet"
    feat.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path} ({len(feat):,} rows, {len(feat.columns)} columns)")


if __name__ == "__main__":
    main()
