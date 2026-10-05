"""
Stuff+ v3 - dataset construction + sanity checks. DOES NOT TRAIN A MODEL -
stops once the final feature table is built and validated, per direct
scope for this session. See analytics/stuff_features_v3.py for the actual
feature-engineering logic; this script is the runner: pulls raw pitches,
builds the fastball-reference classification (Part A), builds the final
per-pitch feature table (Parts B+C), and prints every sanity check asked
for before anything gets used to train.

Usage:
    python training/build_stuff_v3_dataset.py

Outputs (written to training/stuff_v3_artifacts/, gitignored - regenerable
research artifacts, not committed source):
    pitcher_season_fb_reference.csv   - Part A's audit table (small, human-
                                          readable, spot-checkable directly)
    stuff_v3_features.parquet         - the final per-pitch feature table
    calibrations.json                 - fitted VAA + spin-axis coefficients,
                                          so a later training run reuses the
                                          EXACT same transform, not a refit
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, CORE_NUMERIC_FEATURES, OPTIONAL_NUMERIC_FEATURES,
    RAW_QUERY_COLUMNS, SEPARATION_FEATURES,
    add_movement_columns, build_fb_reference_table, engineer_features,
    fb_reference_audit_columns, fit_v3_calibrations, merge_pitch_type,
)

DB_PATH = "mlb_data.db"
ALL_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]  # EXT: 2015 excluded, spin_axis 0% populated that season
OUT_DIR = Path("training/pitching_plus_ext_artifacts/stuff")

SPOT_CHECK_PITCHERS = [
    (801139, "Payton Tolle"),
    (694819, "Jacob Misiorowski"),
    (605280, "Clay Holmes"),
]


def load_all_pitches(conn):
    print(f"Loading pitches for seasons {ALL_SEASONS} (delta_run_exp IS NOT NULL, matching v2's row scope)...", flush=True)
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)}
        FROM pitches
        WHERE delta_run_exp IS NOT NULL AND season IN ({','.join('?' * len(ALL_SEASONS))})
    """, conn, params=ALL_SEASONS)
    print(f"  {len(df):,} raw rows loaded")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    raw = load_all_pitches(conn)
    raw = merge_pitch_type(raw)
    raw = add_movement_columns(raw)

    # ---------------------------------------------------------------- Part A
    print("\n=== Part A: fastball reference classification ===")
    fb_ref = build_fb_reference_table(raw)
    print(f"{len(fb_ref):,} pitcher-seasons classified")

    has_ref = fb_ref["primary_fb_type"].notna()
    print("\nPrimary-reference coverage:")
    print(f"  has a primary reference: {has_ref.sum():,} pitcher-seasons ({has_ref.mean()*100:.1f}%)")
    print(f"  no FF/SI/FC thrown at all: {(~has_ref).sum():,} pitcher-seasons ({(~has_ref).mean()*100:.1f}%)")
    print("\nPrimary type distribution:")
    print(fb_ref["primary_fb_type"].value_counts(dropna=False))

    print("\nSpot checks (named pitchers):")
    for pid, name in SPOT_CHECK_PITCHERS:
        rows = fb_ref[fb_ref["pitcher_id"] == pid]
        if rows.empty:
            print(f"  {name} ({pid}): not found in data")
            continue
        for _, r in rows.sort_values("season").iterrows():
            print(f"  {name} {int(r['season'])}: primary_fb = {r['primary_fb_type']}")

    audit_table = fb_reference_audit_columns(fb_ref)
    audit_path = OUT_DIR / "pitcher_season_fb_reference.csv"
    audit_table.to_csv(audit_path, index=False)
    print(f"\nSaved {audit_path} ({len(audit_table):,} rows)")

    # -------------------------------------------------------- calibrations
    print("\n=== Fitting calibrations (VAA expectation + spin-axis) ===")
    calibrations = fit_v3_calibrations(raw)
    cal = calibrations["spin_calibration"]
    print(f"Spin-axis calibration: flip={cal['flip']}, offset={cal['offset_deg']:.1f}deg, "
          f"mean_abs_resid={cal['mean_abs_resid_deg']:.1f}deg (fit on {cal['n_calibration_pitches']:,} FF pitches)")
    vaa_coefs_serializable = (
        calibrations["vaa_coefs"]
        if isinstance(calibrations["vaa_coefs"], dict)
        else calibrations["vaa_coefs"].tolist()
    )
    with open(OUT_DIR / "calibrations.json", "w") as f:
        json.dump({
            "vaa_coefs": vaa_coefs_serializable,
            "spin_calibration": {k: v for k, v in cal.items() if k != "both_candidates"},
        }, f, indent=2)

    # ------------------------------------------------------- Parts B and C
    print("\n=== Building final v3 feature table (Parts B + C) ===")
    feat = engineer_features(raw, calibrations, fb_ref)
    print(f"{len(feat):,} rows in final table (from {len(raw):,} raw rows - "
          f"{len(raw) - len(feat):,} dropped for missing a CORE feature)")

    print("\nNull rates, new/optional features (expected: primary_fb separation features "
          "null whenever a pitch IS the primary reference type itself or the pitcher-season "
          "has no FF/SI/FC at all - a substantial fraction, not near-zero, since primary is "
          "now the only slot; axis_differential should be low/near-zero):")
    for col in OPTIONAL_NUMERIC_FEATURES:
        null_pct = feat[col].isna().mean() * 100
        print(f"  {col}: {null_pct:.1f}% null")

    print("\nNull rates, core features (should all be 0.0% - these are the dropna columns):")
    for col in CORE_NUMERIC_FEATURES:
        null_pct = feat[col].isna().mean() * 100
        print(f"  {col}: {null_pct:.1f}% null")

    # ---------------------------------------------------- self-comparison check
    print("\n=== Self-comparison exclusion check ===")
    check = feat.merge(fb_ref[["pitcher_id", "season", "primary_fb_type"]], on=["pitcher_id", "season"], how="left", suffixes=("", "_ref"))
    self_rows = check[check["pitch_type"] == check["primary_fb_type"]]
    bad = self_rows["velo_sep_vs_primary_fb"].notna().sum()
    print(f"  primary_fb: {len(self_rows):,} self-comparison rows, {bad} incorrectly non-NaN (should be 0)")

    # --------------------------------------------------------- axis_differential
    print("\n=== axis_differential sanity checks ===")
    print(feat["axis_differential"].describe())
    print("\nBy handedness:")
    print(feat.groupby("p_throws")["axis_differential"].agg(["mean", "median", "count"]))
    print("\nBy pitch type (n>=1000):")
    g = feat.groupby("pitch_type")["axis_differential"].agg(["mean", "median", "count"])
    print(g[g["count"] >= 1000].sort_values("mean", ascending=False))

    feat_path = OUT_DIR / "stuff_v3_features.parquet"
    feat.to_parquet(feat_path, index=False)
    print(f"\nSaved {feat_path} ({len(feat):,} rows, {len(ALL_FEATURES)} features + id columns)")
    print("\nDone. Model NOT trained - dataset construction only, per session scope.")


if __name__ == "__main__":
    main()
