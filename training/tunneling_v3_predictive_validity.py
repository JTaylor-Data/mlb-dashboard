"""
Tunneling+ v3 - Part E.5: out-of-sample predictive validity. Trains a
one-off model on every season except 2024, scores 2024, correlates
against REAL 2025 outcomes (a model that never saw 2024 or 2025) - same
methodology and same five outcome metrics as Stuff+'s and v2 Tunneling+'s
own out-of-sample checks, for direct comparability.

Usage:
    python training/tunneling_v3_predictive_validity.py
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analytics.percentiles import pitcher_percentile_metrics
from training.train_tunneling_v3_model import (
    ALL_FEATURES, CATEGORICAL_FEATURES, ARTIFACTS_DIR, _fit_one_model, _to_plus_scale, load_feat,
)

SCORE_SEASON = 2024
OUTCOME_SEASON = 2025
OUTCOMES = ["xwoba", "whiff_pct", "swstr_pct", "k_minus_bb_pct", "chase_pct"]


def main():
    feat = load_feat()
    train_df = feat[feat["season"] != str(SCORE_SEASON)].copy()
    score_df = feat[feat["season"] == str(SCORE_SEASON)].copy()
    for col in CATEGORICAL_FEATURES:
        train_vals = train_df[col][train_df[col].notna()]
        score_vals = score_df[col][score_df[col].notna()]
        cats = sorted(set(train_vals) | set(score_vals))
        train_df[col] = pd.Categorical(train_df[col], categories=cats)
        score_df[col] = pd.Categorical(score_df[col], categories=cats)

    print(f"Training on {len(train_df):,} rows (excludes {SCORE_SEASON})...", flush=True)
    model = _fit_one_model(train_df, ALL_FEATURES)
    preds = model.predict(score_df[ALL_FEATURES])
    out = score_df[["pitcher_id"]].copy()
    out["pred"] = preds

    g = out.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n_2024_pitches=("pred", "size")).reset_index()
    g["score"] = _to_plus_scale(g["mean_pred"], out["pred"].mean(), out["pred"].std())

    print("Loading real 2025 outcomes...", flush=True)
    conn = sqlite3.connect("mlb_data.db")
    pool = pitcher_percentile_metrics(conn, OUTCOME_SEASON, stand_filter=None).reset_index()
    pool = pool.rename(columns={pool.columns[0]: "pitcher_id"})
    merged = g.merge(pool, on="pitcher_id", how="inner")

    results_by_min_pitches = {}
    for min_n in (100, 200, 300):
        sub = merged[merged["n_2024_pitches"] >= min_n]
        correlations = {o: round(float(sub["score"].corr(sub[o])), 3) for o in OUTCOMES}
        results_by_min_pitches[str(min_n)] = dict(n_qualified=int(len(sub)), correlations=correlations)
        print(f"  min {min_n} pitches (n={len(sub)}): {correlations}", flush=True)

    report = dict(score_season=SCORE_SEASON, outcome_season=OUTCOME_SEASON, results_by_min_pitches=results_by_min_pitches)
    out_path = ARTIFACTS_DIR / "tunneling_v3_predictive_validity.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
