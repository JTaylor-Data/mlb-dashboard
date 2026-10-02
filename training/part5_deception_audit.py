"""
Part 5 — Deception+ v1 feature-set audit. Diagnostic only, no changes.

1. Collinearity matrix for all numeric features — confirm arm_angle/release_pos_z r=0.79
2. SHAP mean |SHAP| grouped by feature type: raw delivery, _std/consistency, bio (height/weight)
3. height_in/weight_lb individual SHAP contributions
4. Season-level vs. rolling window note for _std features (theoretical, no code change)

Run from mlb_dashboard root:
    python training/part5_deception_audit.py
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from analytics.deception_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    RAW_QUERY_COLUMNS, engineer_features, load_bio,
)

MODELS_DIR = ROOT / "models"
BOOSTER_PATH = MODELS_DIR / "deception_plus_v1.txt"
OUT_PATH = ROOT / "training" / "v2_artifacts" / "deception_audit.json"
DB_PATH = "mlb_data.db"

AUDIT_SEASON = 2025  # one representative season for SHAP sampling
SAMPLE_N = 50_000    # SHAP sample size (full season is ~700k rows)

RAW_DELIVERY_FEATURES = ["arm_angle", "release_extension", "release_pos_x_as", "release_pos_z"]
VARIABILITY_FEATURES  = ["arm_angle_std", "release_extension_std", "release_consistency"]
BIO_FEATURES          = ["height_in", "weight_lb"]


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Part 5 — Deception+ v1 feature-set audit")
    conn = sqlite3.connect(DB_PATH)
    bio = load_bio(conn)

    # Load full 2015–2025 for collinearity (need representative sample)
    log("Loading pitches 2021–2025 for feature matrix...")
    df_raw = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season BETWEEN 2021 AND 2025
          AND delta_run_exp IS NOT NULL
          AND game_type = 'R'
    """, conn)
    conn.close()
    log(f"  Loaded {len(df_raw):,} rows")

    feat = engineer_features(df_raw, bio)
    log(f"  After engineer_features: {len(feat):,} rows")

    # ── 1. Collinearity matrix ───────────────────────────────────────────────
    log("\n=== 1. Collinearity matrix (numeric features) ===")
    num_df = feat[NUMERIC_FEATURES].copy()
    corr = num_df.corr(method="pearson")

    log("Pearson correlation matrix:")
    for i, col_a in enumerate(NUMERIC_FEATURES):
        for j, col_b in enumerate(NUMERIC_FEATURES):
            if j <= i:
                continue
            r = corr.loc[col_a, col_b]
            if abs(r) >= 0.3:
                log(f"  {col_a:30s} x {col_b:30s}: r = {r:+.3f}")

    key_pair = float(corr.loc["arm_angle", "release_pos_z"])
    log(f"\nKey pair: arm_angle x release_pos_z: r = {key_pair:+.3f}")

    # ── 2. SHAP feature importance by group ──────────────────────────────────
    log(f"\n=== 2. SHAP mean |SHAP| — sampling {SAMPLE_N:,} rows from season {AUDIT_SEASON} ===")
    season_feat = feat[feat["season"].astype(str) == str(AUDIT_SEASON)].copy()
    log(f"  Season {AUDIT_SEASON} rows: {len(season_feat):,}")

    if len(season_feat) > SAMPLE_N:
        season_feat = season_feat.sample(SAMPLE_N, random_state=42)
    log(f"  Sample size: {len(season_feat):,}")

    for col in CATEGORICAL_FEATURES:
        cats = sorted(feat[col].astype(str).unique())
        season_feat[col] = pd.Categorical(season_feat[col].astype(str), categories=cats)

    model = lgb.Booster(model_file=str(BOOSTER_PATH))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(season_feat[ALL_FEATURES])

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=ALL_FEATURES,
    )

    log("\nMean |SHAP| per feature:")
    for feat_name, val in mean_abs_shap.sort_values(ascending=False).items():
        log(f"  {feat_name:30s}: {val:.5f}")

    # Group totals
    group_totals = {
        "raw_delivery": float(mean_abs_shap[RAW_DELIVERY_FEATURES].sum()),
        "variability":  float(mean_abs_shap[VARIABILITY_FEATURES].sum()),
        "bio":          float(mean_abs_shap[BIO_FEATURES].sum()),
        "season":       float(mean_abs_shap["season"]) if "season" in mean_abs_shap.index else 0.0,
    }
    log("\nGroup totals (sum of mean |SHAP|):")
    total = sum(group_totals.values())
    for group, val in group_totals.items():
        log(f"  {group:20s}: {val:.5f}  ({100*val/total:.1f}%)")

    # ── 3. Bio feature individual contributions ──────────────────────────────
    log("\n=== 3. Bio feature (height_in / weight_lb) SHAP summary ===")
    for col in BIO_FEATURES:
        idx = ALL_FEATURES.index(col)
        col_shap = shap_values[:, idx]
        log(f"  {col}: mean |SHAP| = {np.abs(col_shap).mean():.5f}  "
            f"mean SHAP = {col_shap.mean():+.5f}  "
            f"std = {col_shap.std():.5f}")

    # ── 4. Rolling vs. season-level _std note ────────────────────────────────
    log("\n=== 4. Season-level vs. rolling window note ===")
    log("  Current implementation: arm_angle_std / release_extension_std are computed")
    log("  as the season-long standard deviation per (pitcher_id, season) pair.")
    log("  This means the _std value for every pitch a pitcher throws in a season")
    log("  is identical — it's a season-level aggregate, not a pitch-level running stat.")
    log("  Rolling window alternative: compute std over last N pitches.")
    log("  Advantage: captures mid-season delivery changes (e.g., mechanical adjustment).")
    log("  Disadvantage: introduces look-ahead bias unless carefully implemented.")
    log("  Recommendation: defer. Season-level aggregation is the honest, simple choice")
    log("  and aligns with the 'year-over-year' framing of Deception+ as a whole.")

    # ── Save results ─────────────────────────────────────────────────────────
    result = {
        "audit_season": AUDIT_SEASON,
        "sample_n": len(season_feat),
        "collinearity_key_pairs": {
            f"{a}_x_{b}": float(corr.loc[a, b])
            for a in NUMERIC_FEATURES for b in NUMERIC_FEATURES
            if a < b and abs(corr.loc[a, b]) >= 0.3
        },
        "arm_angle_x_release_pos_z": key_pair,
        "mean_abs_shap": {k: float(v) for k, v in mean_abs_shap.items()},
        "group_shap_totals": group_totals,
        "group_shap_pct": {k: round(100 * v / total, 1) for k, v in group_totals.items()},
        "bio_shap_detail": {
            col: {
                "mean_abs": float(np.abs(shap_values[:, ALL_FEATURES.index(col)]).mean()),
                "mean": float(shap_values[:, ALL_FEATURES.index(col)].mean()),
                "std": float(shap_values[:, ALL_FEATURES.index(col)].std()),
            }
            for col in BIO_FEATURES
        },
        "rolling_window_recommendation": "defer — season-level aggregation is the correct default",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    log(f"\nAudit results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
