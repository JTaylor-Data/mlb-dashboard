"""
Location+ v1 - Part C: trains the model. Loads pitches directly (no
separate dataset-build script - unlike Tunneling+ v3/Stuff+ v3, Location+'s
feature engineering is cheap, no expensive pairwise/lag precomputation to
cache separately, matching Deception+'s single-script pattern instead).

Trains LightGBM (Huber loss, leave-one-season-out CV, assert_no_identifiers
guard) against delta_run_exp - same conventions, same target, as every
other model in this project, so all four "+" models stay commensurable for
a future Pitching+ stack. See analytics/location_features.py's module
docstring for the full design rationale (what's included/excluded, the
deliberate stand-inclusion deviation from FanGraphs' public methodology,
and the accepted location-vs-intent limitation).

Offline build only this session - does NOT create a live-serving module,
does NOT touch models/stuff_plus*, models/deception_plus*, models/
tunneling_v3*, or any API/frontend file.

Usage:
    python training/train_location_model.py
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from analytics.location_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_QUERY_COLUMNS,
    engineer_features, shrinkage_params, to_plus_scale,
)

DB_PATH = "mlb_data.db"
ARTIFACTS_DIR = Path("training/location_v1_artifacts")
MODELS_DIR = Path("models")
COMPLETE_SEASONS = [2021, 2022, 2023, 2024, 2025]
ALL_SEASONS = COMPLETE_SEASONS + [2026]  # 2026 partial - train-only, never a CV holdout, same convention as every other model here

BANNED_IDENTIFIER_COLS = {"pitcher_id", "game_id", "pitch_id", "player_id", "batter_id", "at_bat_number"}


def assert_no_identifiers(feature_cols, context):
    leaked = BANNED_IDENTIFIER_COLS & set(feature_cols)
    if leaked:
        raise ValueError(f"Identifier leakage in {context}: {leaked} found in feature columns. Refusing to train.")


LGBM_PARAMS = dict(
    objective="huber",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,  # small feature set (7 total) - a deep tree (Stuff+'s 63) would just overfit noise on this few dimensions, same reasoning Deception+'s own num_leaves=31 used
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)


def load_pitches(conn, seasons):
    print(f"Loading pitches for seasons {seasons} (PO/UN/EP excluded, KN kept)...", flush=True)
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE delta_run_exp IS NOT NULL AND season IN ({','.join('?' * len(seasons))})
    """, conn, params=seasons)
    return df


def load_feat(conn):
    raw = load_pitches(conn, ALL_SEASONS)
    feat = engineer_features(raw)
    assert_no_identifiers(ALL_FEATURES, "load_feat")
    return feat


def _fit_one_model(train_df, feature_cols):
    assert_no_identifiers(feature_cols, "_fit_one_model")
    fit_rows, stop_rows = train_test_split(train_df, test_size=0.08, random_state=42)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        fit_rows[feature_cols], fit_rows["delta_run_exp"],
        eval_set=[(stop_rows[feature_cols], stop_rows["delta_run_exp"])],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _coerce_categoricals(train_df, test_df):
    for col in CATEGORICAL_FEATURES:
        train_vals = train_df[col][train_df[col].notna()]
        test_vals = test_df[col][test_df[col].notna()]
        cats = sorted(set(train_vals) | set(test_vals))
        train_df[col] = pd.Categorical(train_df[col], categories=cats)
        test_df[col] = pd.Categorical(test_df[col], categories=cats)
    return train_df, test_df


def run_cross_validation(feat):
    fold_reports = []
    for holdout in COMPLETE_SEASONS:
        train_df = feat[feat["season"] != str(holdout)].copy()
        test_df = feat[feat["season"] == str(holdout)].copy()
        train_df, test_df = _coerce_categoricals(train_df, test_df)
        print(f"  fitting on {len(train_df):,} rows, evaluating on {len(test_df):,}...", flush=True)
        model = _fit_one_model(train_df, ALL_FEATURES)
        preds = model.predict(test_df[ALL_FEATURES])
        r2 = r2_score(test_df["delta_run_exp"], preds)
        mae = mean_absolute_error(test_df["delta_run_exp"], preds)
        print(f"  holdout {holdout}: R^2={r2:.4f} MAE={mae:.4f} n={len(test_df):,} best_iter={model.best_iteration_}", flush=True)
        fold_reports.append(dict(holdout_season=holdout, r2=float(r2), mae=float(mae), n=int(len(test_df))))
    return fold_reports


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    t0 = time.time()

    print("Loading and engineering Location+ feature table...", flush=True)
    feat = load_feat(conn)
    print(f"  {len(feat):,} rows, {len(ALL_FEATURES)} features ({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)", flush=True)

    print("\n--- Cross-validation ---", flush=True)
    fold_reports = run_cross_validation(feat)
    cv_r2_mean = float(np.mean([f["r2"] for f in fold_reports]))
    cv_mae_mean = float(np.mean([f["mae"] for f in fold_reports]))
    print(f"CV mean R^2={cv_r2_mean:.4f}, mean MAE={cv_mae_mean:.4f}", flush=True)

    print("\n--- Training final model on all seasons ---", flush=True)
    final_feat = feat.copy()
    for col in CATEGORICAL_FEATURES:
        cats = sorted(set(final_feat[col][final_feat[col].notna()]))
        final_feat[col] = pd.Categorical(final_feat[col], categories=cats)
    model = _fit_one_model(final_feat, ALL_FEATURES)
    print(f"  final model best_iteration={model.best_iteration_}", flush=True)

    print("\n--- SHAP summary (30k sample) ---", flush=True)
    sample = final_feat.sample(n=min(30_000, len(final_feat)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[ALL_FEATURES])
    mean_abs_shap = dict(zip(ALL_FEATURES, np.abs(shap_values).mean(axis=0).tolist()))
    print("\nPer-feature (sorted):")
    for k, v in sorted(mean_abs_shap.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.5f}")

    print("\n--- Scoring every pitch with the final model ---", flush=True)
    preds_final = model.predict(final_feat[ALL_FEATURES])
    raw = final_feat[["pitcher_id", "season", "pitch_type"]].copy()
    raw["pred"] = preds_final

    print("\n--- Estimating Location+-specific empirical-Bayes shrinkage K's ---", flush=True)
    k_by_type, fallback_k, type_means = shrinkage_params(raw)
    print(f"  fallback_k={fallback_k:.2f} (Stuff+'s own fallback is 1.5 - expect this to be much larger)")
    for pt, k in sorted(k_by_type.items(), key=lambda x: x[1]):
        print(f"  {pt}: K={k:.2f}")

    print("\n--- Per-pitcher-season scores (unshrunk, for reference) ---", flush=True)
    per_season = []
    for season, group in raw.groupby("season"):
        g = group.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n_pitches=("pred", "size")).reset_index()
        g["score_unshrunk"] = to_plus_scale(g["mean_pred"], group["pred"].mean(), group["pred"].std())
        g["season"] = season
        per_season.append(g)
    scores = pd.concat(per_season, ignore_index=True)
    scores["total_pitches"] = scores["n_pitches"]

    model.booster_.save_model(str(MODELS_DIR / "location_plus_v1.txt"))
    meta = dict(
        all_features=ALL_FEATURES, numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES,
        category_levels={c: final_feat[c].cat.categories.tolist() for c in CATEGORICAL_FEATURES},
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"), lgbm_params=LGBM_PARAMS,
        seasons_trained_on=ALL_SEASONS,
        shrinkage_k_by_pitch_type=k_by_type, shrinkage_fallback_k=fallback_k,
    )
    (MODELS_DIR / "location_plus_v1_meta.json").write_text(json.dumps(meta, indent=2))
    scores.to_parquet(ARTIFACTS_DIR / "location_v1_scores.parquet", index=False)
    raw.to_parquet(ARTIFACTS_DIR / "location_v1_per_pitch_preds.parquet", index=False)

    report = dict(
        n_training_rows=int(len(feat)), all_features=ALL_FEATURES,
        cv_folds=fold_reports, cv_r2_mean=cv_r2_mean, cv_mae_mean=cv_mae_mean,
        mean_abs_shap=mean_abs_shap,
        shrinkage_k_by_pitch_type=k_by_type, shrinkage_fallback_k=fallback_k,
    )
    (MODELS_DIR / "location_plus_v1_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nDone in {time.time()-t0:.0f}s. Saved models/location_plus_v1.txt (+meta/report), "
          f"{ARTIFACTS_DIR}/location_v1_scores.parquet")


if __name__ == "__main__":
    main()
