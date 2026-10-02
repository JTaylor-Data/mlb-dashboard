"""
Part A — Corrected Tunneling+ v2 sequential same-at-bat test.

The prior Part 4 test (z = 180.1) bucketed pitches by the FULL blended xRV,
which already includes the xWhiff% classifier's output as a direct input.
Checking whether whiff rate differs by bucket is therefore close to circular —
the model's own whiff prediction is doing most of the work, not necessarily the
tunneling-specific lag-geometry features.

This script isolates the question: do the lag-geometry features specifically
(`lagN_commit_distance`, `lagN_plate_distance`, `lagN_shape_break`) predict real
whiff differences, independent of pitch-type information?

Procedure:
  1. Load xWhiff% classifier (tunneling_plus_v2_xwhiff.txt) — the single
     sub-model where tunneling geometry most directly matters.
  2. Compute per-pitch SHAP values using shap.TreeExplainer on a sample of rows.
  3. For each pitch, compute three isolated scores:
       a. lag1_geom_shap: sum of SHAP contributions from lag1_commit_distance,
          lag1_plate_distance, lag1_shape_break ONLY (primary result).
       b. alllag_geom_shap: sum across all 6 lag slots' geometry columns (secondary).
       c. pt_shap: sum of SHAP contributions from ALL lagN_pitch_type columns
          (confound control — how much effect is really pitch-type-only?).
  4. Bucket each score into terciles. For each tercile assignment, compute the
     real whiff rate and run z-test (high_score - low_score whiff diff / pooled SE).
     Note sign: higher geometry SHAP = model says geometry is pushing xWhiff% up
     = harder to hit = should correspond to more real whiffs.
  5. Report all three z-scores side by side.

Run from mlb_dashboard root:
    python training/part_a_tunneling_shap_geometry.py
"""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from analytics.tunneling_features import (
    lag_feature_columns, load_pitches, sequential_lag_features, MAX_LAG,
)

MODELS_DIR = ROOT / "models"
META_PATH = MODELS_DIR / "tunneling_plus_v2_meta.json"
COMMIT_TABLE_PATH = (
    ROOT / "project_artifacts" / "training_artifacts"
    / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv"
)
OUT_PATH = ROOT / "training" / "v2_artifacts" / "part_a_tunneling_shap_geometry.json"
LOG_PATH = ROOT / "training" / "part_a_tunneling_shap_geometry.log"
DB_PATH = "mlb_data.db"

# Seasons covered by commit table
COMMIT_SEASONS = list(range(2021, 2026))

# SHAP is expensive — sample this many rows for the analysis.
# 200K gives very tight SE on tercile whiff rates (each tercile ~66K rows).
SHAP_SAMPLE_N = 200_000

WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "bunt_foul_tip", "missed_bunt", "swinging_pitchout",
}

LAG_COLS = lag_feature_columns()
NUMERIC_LAG_COLS = [c for c in LAG_COLS if not c.endswith("_pitch_type")]
CATEGORICAL_LAG_COLS = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CAT = ["pitch_type", "p_throws", "stand", "season"]
ALL_CAT = CATEGORICAL_LAG_COLS + CURRENT_CAT

# Feature group definitions
LAG1_GEOM_COLS = ["lag1_commit_distance", "lag1_plate_distance", "lag1_shape_break"]
ALL_LAG_GEOM_COLS = [f"lag{n}_{m}" for n in range(1, MAX_LAG + 1)
                     for m in ("commit_distance", "plate_distance", "shape_break")]
PITCH_TYPE_LAG_COLS = CATEGORICAL_LAG_COLS  # lag1_pitch_type ... lag6_pitch_type

_log_fh = None


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def stringify_keep_na(series):
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)


def z_test(p_high, n_high, p_low, n_low):
    """Two-proportion z-test: high_score whiff - low_score whiff / pooled SE.
    Positive z = higher geometry SHAP → more real whiffs (what we'd expect if
    the geometry signal is real)."""
    se = np.sqrt(p_high * (1 - p_high) / n_high + p_low * (1 - p_low) / n_low)
    if se == 0:
        return 0.0
    return float((p_high - p_low) / se)


def tercile_whiff_z(df, score_col, label):
    """Bucket `score_col` into terciles, compute whiff rate per tercile, z-test."""
    df2 = df.dropna(subset=[score_col, "is_whiff"]).copy()
    df2["tercile"] = pd.qcut(df2[score_col], 3,
                             labels=["low", "mid", "high"], duplicates="drop")
    by_t = df2.groupby("tercile", observed=True)["is_whiff"].agg(["mean", "count"])

    low_r  = by_t.loc["low",  "mean"]
    low_n  = by_t.loc["low",  "count"]
    high_r = by_t.loc["high", "mean"]
    high_n = by_t.loc["high", "count"]

    z = z_test(high_r, high_n, low_r, low_n)

    log(f"\n  [{label}]")
    for t in ["low", "mid", "high"]:
        log(f"    {t:4s}: whiff={by_t.loc[t,'mean']:.4f}  n={by_t.loc[t,'count']:,}")
    log(f"    z (high - low) = {z:.3f}")

    return {
        "label": label,
        "low_tercile_whiff":  float(low_r),
        "mid_tercile_whiff":  float(by_t.loc["mid", "mean"]),
        "high_tercile_whiff": float(high_r),
        "low_n":  int(low_n),
        "mid_n":  int(by_t.loc["mid", "count"]),
        "high_n": int(high_n),
        "z_high_minus_low": float(z),
    }


def main():
    global _log_fh
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")

    log("Part A — Corrected Tunneling+ v2 sequential same-at-bat test (SHAP geometry isolation)")
    log(f"SHAP sample N = {SHAP_SAMPLE_N:,}")

    commit_table = pd.read_csv(COMMIT_TABLE_PATH)
    log(f"Commit table: {len(commit_table):,} rows")

    xwhiff_booster = lgb.Booster(model_file=str(MODELS_DIR / "tunneling_plus_v2_xwhiff.txt"))
    meta = json.loads(META_PATH.read_text())
    feat_cols = meta["feat_cols"]
    log(f"xWhiff% booster loaded. Feature count: {len(feat_cols)}")

    # Derive category levels
    conn = sqlite3.connect(DB_PATH)
    log("Deriving category levels...")
    raw_cats = pd.read_sql("""
        SELECT DISTINCT pitch_type, p_throws, stand, season FROM pitches
        WHERE season BETWEEN 2015 AND 2026
    """, conn)
    raw_cats["pitch_type"] = raw_cats["pitch_type"].replace({"KC": "CU"})
    raw_cats = raw_cats[~raw_cats["pitch_type"].isin({"PO", "UN", "EP"})]
    raw_cats["season"] = raw_cats["season"].astype(str)
    pt_levels = sorted(raw_cats["pitch_type"].dropna().unique().tolist())
    cat_levels = {
        "pitch_type": pt_levels,
        "p_throws": sorted(raw_cats["p_throws"].dropna().unique().tolist()),
        "stand": sorted(raw_cats["stand"].dropna().unique().tolist()),
        "season": sorted(raw_cats["season"].dropna().unique().tolist()),
    }
    for col in CATEGORICAL_LAG_COLS:
        cat_levels[col] = pt_levels

    # Build full feature matrix + descriptions across all commit seasons
    log(f"\nBuilding lag features for seasons {COMMIT_SEASONS}...")
    all_chunks = []
    t_total = time.time()

    for season in COMMIT_SEASONS:
        t0 = time.time()
        log(f"  Season {season}...")
        df_s = load_pitches(conn, season)
        if len(df_s) == 0:
            log(f"    No pitches, skipping"); continue

        feat = sequential_lag_features(df_s, commit_table, max_lag=MAX_LAG)
        feat = feat.dropna(subset=["lag1_commit_distance", "lag1_plate_distance"])
        if len(feat) == 0:
            log(f"    No lag rows, skipping"); continue

        feat["season"] = str(season)
        for col in ALL_CAT:
            feat[col] = stringify_keep_na(feat[col])
            levels = cat_levels.get(col)
            if levels:
                feat[col] = pd.Categorical(feat[col], categories=levels)
            else:
                feat[col] = pd.Categorical(feat[col])

        keep_cols = feat_cols + ["description"]
        all_chunks.append(feat[[c for c in keep_cols if c in feat.columns]].copy())
        log(f"    {len(feat):,} lag rows ({time.time()-t0:.0f}s)")

    conn.close()
    log(f"Load time: {(time.time()-t_total)/60:.1f} min")

    df_full = pd.concat(all_chunks, ignore_index=True)
    log(f"Total rows: {len(df_full):,}")

    df_full["is_whiff"] = df_full["description"].isin(WHIFF_DESCRIPTIONS).astype(float)
    total_whiff_rate = df_full["is_whiff"].mean()
    log(f"Overall whiff rate: {total_whiff_rate:.4f}")

    # Sample for SHAP (stratified by whiff to preserve whiff rate in sample)
    rng = np.random.default_rng(42)
    whiff_idx  = df_full.index[df_full["is_whiff"] == 1.0].tolist()
    no_whiff_idx = df_full.index[df_full["is_whiff"] == 0.0].tolist()
    n_whiff_target = int(SHAP_SAMPLE_N * total_whiff_rate)
    n_no_whiff_target = SHAP_SAMPLE_N - n_whiff_target
    sampled_idx = (
        rng.choice(whiff_idx, size=min(n_whiff_target, len(whiff_idx)), replace=False).tolist() +
        rng.choice(no_whiff_idx, size=min(n_no_whiff_target, len(no_whiff_idx)), replace=False).tolist()
    )
    df_samp = df_full.loc[sampled_idx].copy()
    log(f"\nSHAP sample: {len(df_samp):,} rows  (whiff rate: {df_samp['is_whiff'].mean():.4f})")

    X_samp = df_samp[feat_cols]

    log("Computing SHAP values (TreeExplainer)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(xwhiff_booster)
    shap_vals = explainer.shap_values(X_samp)  # shape (n_rows, n_features)
    log(f"SHAP done in {time.time()-t0:.0f}s. Shape: {shap_vals.shape}")

    # Map feature names to column indices
    feat_idx = {f: i for i, f in enumerate(feat_cols)}

    # --- Isolated SHAP score 1: lag-1 geometry only ---
    lag1_geom_idx = [feat_idx[c] for c in LAG1_GEOM_COLS if c in feat_idx]
    df_samp["lag1_geom_shap"] = shap_vals[:, lag1_geom_idx].sum(axis=1)
    log(f"\nLag-1 geometry SHAP columns ({len(lag1_geom_idx)}): {LAG1_GEOM_COLS[:3]}")

    # --- Isolated SHAP score 2: all-lag geometry ---
    alllag_geom_idx = [feat_idx[c] for c in ALL_LAG_GEOM_COLS if c in feat_idx]
    df_samp["alllag_geom_shap"] = shap_vals[:, alllag_geom_idx].sum(axis=1)
    log(f"All-lag geometry SHAP columns ({len(alllag_geom_idx)}): {ALL_LAG_GEOM_COLS[:3]}...")

    # --- Isolated SHAP score 3: pitch-type lag columns (confound control) ---
    pt_lag_idx = [feat_idx[c] for c in PITCH_TYPE_LAG_COLS if c in feat_idx]
    df_samp["pt_lag_shap"] = shap_vals[:, pt_lag_idx].sum(axis=1)
    log(f"Pitch-type lag SHAP columns ({len(pt_lag_idx)}): {PITCH_TYPE_LAG_COLS}")

    # Distribution summary
    for col, label in [
        ("lag1_geom_shap", "Lag-1 geom SHAP"),
        ("alllag_geom_shap", "All-lag geom SHAP"),
        ("pt_lag_shap", "Pitch-type lag SHAP"),
    ]:
        arr = df_samp[col].values
        log(f"\n  {label}: mean={arr.mean():.5f}  std={arr.std():.5f}  "
            f"p10={np.percentile(arr,10):.5f}  p90={np.percentile(arr,90):.5f}")

    # --- Tercile z-tests ---
    log("\n=== TERCILE WHIFF-RATE Z-TESTS ===")
    result_lag1   = tercile_whiff_z(df_samp, "lag1_geom_shap",   "Lag-1 geometry SHAP (primary)")
    result_alllag = tercile_whiff_z(df_samp, "alllag_geom_shap", "All-lag geometry SHAP")
    result_pt     = tercile_whiff_z(df_samp, "pt_lag_shap",      "Pitch-type lag SHAP (confound control)")

    log("\n=== SUMMARY ===")
    log(f"  Lag-1 geometry SHAP z   = {result_lag1['z_high_minus_low']:.3f}")
    log(f"  All-lag geometry SHAP z = {result_alllag['z_high_minus_low']:.3f}")
    log(f"  Pitch-type lag SHAP z   = {result_pt['z_high_minus_low']:.3f}")
    log(f"  (Prior Part 4 full-xRV z = 180.1 for reference)")

    output = {
        "description": (
            "Corrected Tunneling+ v2 sequential test. SHAP values from xWhiff% classifier only. "
            "Scores = sum of specific feature groups' SHAP contributions. "
            "Higher SHAP = model says features are pushing xWhiff% up = should see more real whiffs."
        ),
        "sample_n": len(df_samp),
        "overall_whiff_rate": float(total_whiff_rate),
        "lag1_geometry_primary": result_lag1,
        "alllag_geometry_secondary": result_alllag,
        "pitchtype_lag_confound_control": result_pt,
        "prior_part4_full_xrv_z": 180.1,
        "feature_groups": {
            "lag1_geom": LAG1_GEOM_COLS,
            "alllag_geom": ALL_LAG_GEOM_COLS,
            "pt_lag": PITCH_TYPE_LAG_COLS,
        },
        "run_date": datetime.now().strftime("%Y-%m-%d"),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {OUT_PATH}")
    _log_fh.close()


if __name__ == "__main__":
    main()
