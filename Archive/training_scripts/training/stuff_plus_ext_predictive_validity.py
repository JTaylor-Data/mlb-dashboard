"""
Stuff+ v3 - out-of-sample predictive validity. Replaces
training/validate_predictive_power.py (removed, along with the v1/v2
legacy model code it depended on) - that script's output
(models/predictive_validity_report.json) was still being served on the
live Stuff+ model page as a v1-vs-v2 comparison, a content bug: v3 has
been the live model for a while, so the page meant to build trust in the
CURRENT model was showing validity numbers for two retired versions
instead. This script generates the same style of check, but for v3 alone.

Trains a one-off model on every season except 2024 (refitting
vaa_residual from only that training split - same per-fold discipline
train_stuff_v3_lgbm.py's own refit_vaa_per_fold uses; the fastball-
reference classification and spin-axis calibration are reused as-is from
the precomputed feature table, the same already-documented deviation that
script makes, not a new one introduced here), scores 2024, correlates
against REAL 2025 outcomes - same methodology and same five outcomes as
Tunneling+ v3's and Location+'s own out-of-sample checks, for direct
comparability across all the "+" models.

Usage:
    python training/stuff_v3_predictive_validity.py
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from analytics.metrics import compute_vaa
from analytics.percentiles import pitcher_percentile_metrics
from analytics.stuff_features import ALL_FEATURES, CATEGORICAL_FEATURES, fit_vaa_expectation
from training.train_stuff_plus_ext_lgbm import MODELS_DIR, _fit_one_model, load_v3_features

DB_PATH = "mlb_data.db"
REPORT_PATH = MODELS_DIR / "stuff_plus_v3_ext_predictive_validity.json"
SCORE_SEASON = 2024
OUTCOME_SEASON = 2025
OUTCOMES = ["xwoba", "whiff_pct", "swstr_pct", "k_minus_bb_pct", "chase_pct"]


def _to_plus_scale(values, mean, std):
    return 100 - 10 * (values - mean) / std


def main():
    conn = sqlite3.connect(DB_PATH)
    full_df = load_v3_features()
    train_df = full_df[full_df["season"] != str(SCORE_SEASON)].copy()
    score_df = full_df[full_df["season"] == str(SCORE_SEASON)].copy()

    train_df["vaa"] = compute_vaa(train_df["vy0"], train_df["vz0"], train_df["ay"], train_df["az"])
    vaa_coefs = fit_vaa_expectation(train_df)
    for df in (train_df, score_df):
        df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
        z = df["plate_z"].to_numpy(dtype=float)
        v = df["release_speed"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(df)), z, z**2, v])
        expected_vaa = design @ vaa_coefs
        df["vaa_residual"] = (df["vaa"] - expected_vaa).astype("float32")

    for col in CATEGORICAL_FEATURES:
        cats = sorted(set(train_df[col].astype(str)) | set(score_df[col].astype(str)))
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
        score_df[col] = pd.Categorical(score_df[col].astype(str), categories=cats)

    print(f"Training on {len(train_df):,} rows (excludes {SCORE_SEASON})...", flush=True)
    model = _fit_one_model(train_df, ALL_FEATURES)
    preds = model.predict(score_df[ALL_FEATURES])
    out = score_df[["pitcher_id"]].copy()
    out["pred"] = preds

    g = out.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n_2024_pitches=("pred", "size")).reset_index()
    g["score"] = _to_plus_scale(g["mean_pred"], out["pred"].mean(), out["pred"].std())

    print("Loading real 2025 outcomes...", flush=True)
    pool = pitcher_percentile_metrics(conn, OUTCOME_SEASON, stand_filter=None).reset_index()
    pool = pool.rename(columns={pool.columns[0]: "pitcher_id"})
    merged = g.merge(pool, on="pitcher_id", how="inner")

    results_by_min_pitches = {}
    for min_n in (100, 200, 300):
        sub = merged[merged["n_2024_pitches"] >= min_n]
        correlations = {o: round(float(sub["score"].corr(sub[o])), 3) for o in OUTCOMES}
        results_by_min_pitches[str(min_n)] = dict(n_qualified=int(len(sub)), correlations=correlations)
        print(f"  min {min_n} pitches (n={len(sub)}): {correlations}", flush=True)

    report = dict(
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        score_season=SCORE_SEASON, outcome_season=OUTCOME_SEASON,
        description=(
            f"Does each pitcher's {SCORE_SEASON} Stuff+ v3 predict their real {OUTCOME_SEASON} "
            f"results? Scored by a model trained on every season EXCEPT {SCORE_SEASON} (honest "
            f"out-of-sample - the live site's model, unlike this one, is trained on all seasons "
            f"including this one)."
        ),
        results_by_min_pitches=results_by_min_pitches,
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
