"""
Tunneling+ v3 - 150ms commit-point EXPERIMENT (training).

Identical to train_tunneling_v3_model.py in every respect - same features,
same LGBM_PARAMS, same leave-one-season-out CV, same SHAP/lag-slot summary -
except it reads the 150ms feature table built by
build_tunneling_v3_150ms_dataset.py instead of the live 200ms one, and
saves everything to training/tunneling_v3_150ms_artifacts/ instead of
models/. Nothing here writes to models/ or touches the live-serving
tunneling_v3.txt/meta.json - this is a side-by-side comparison run only,
same pattern v2 used for its own 150ms/200ms/fixed25ft/fixed35ft grid.

Usage:
    python training/train_tunneling_v3_150ms_model.py
"""
import json
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

from analytics.tunneling_features import lag_feature_columns, MAX_LAG

ARTIFACTS_DIR = Path("training/tunneling_v3_150ms_artifacts")
FEATURES_PATH = ARTIFACTS_DIR / "tunneling_v3_150ms_features.parquet"
COMPLETE_SEASONS = [2021, 2022, 2023, 2024, 2025]
ALL_SEASONS = COMPLETE_SEASONS + [2026]

BANNED_IDENTIFIER_COLS = {"pitcher_id", "game_id", "pitch_id", "player_id", "batter_id", "at_bat_number"}


def assert_no_identifiers(feature_cols, context):
    leaked = BANNED_IDENTIFIER_COLS & set(feature_cols)
    if leaked:
        raise ValueError(f"Identifier leakage in {context}: {leaked} found in feature columns. Refusing to train.")


LGBM_PARAMS = dict(
    objective="huber",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)

LAG_COLS = lag_feature_columns()
NUMERIC_LAG_COLS = [c for c in LAG_COLS if not c.endswith("_pitch_type")]
CATEGORICAL_LAG_COLS = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CATEGORICAL = ["pitch_type", "p_throws", "stand", "season"]
CATEGORICAL_FEATURES = CATEGORICAL_LAG_COLS + CURRENT_CATEGORICAL
ALL_FEATURES = NUMERIC_LAG_COLS + CATEGORICAL_FEATURES


def _stringify_keep_na(series):
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)


def load_feat():
    feat = pd.read_parquet(FEATURES_PATH)
    feat = feat.dropna(subset=["delta_run_exp"])
    feat["season"] = feat["season"].astype(str)
    for col in CATEGORICAL_FEATURES:
        feat[col] = _stringify_keep_na(feat[col])
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


def _to_plus_scale(values, mean, std):
    return 100 - 10 * (values - mean) / std


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading v3 (150ms) feature table...", flush=True)
    feat = load_feat()
    print(f"  {len(feat):,} rows, {len(ALL_FEATURES)} features", flush=True)

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

    print("\n--- Lag-slot SHAP breakdown ---")
    lag_slot_summary = {}
    for lag in range(1, MAX_LAG + 1):
        commit_shap = mean_abs_shap[f"lag{lag}_commit_distance"]
        plate_shap = mean_abs_shap[f"lag{lag}_plate_distance"]
        shape_shap = mean_abs_shap[f"lag{lag}_shape_break"]
        type_shap = mean_abs_shap[f"lag{lag}_pitch_type"]
        distance_group_total = commit_shap + plate_shap + shape_shap
        lag_slot_summary[f"lag{lag}"] = dict(
            commit_distance=commit_shap, plate_distance=plate_shap, shape_break=shape_shap,
            distance_group_total=distance_group_total, pitch_type=type_shap,
        )
        print(f"  lag{lag}: distance-group total={distance_group_total:.5f} "
              f"(commit={commit_shap:.5f}, plate={plate_shap:.5f}, shape_break={shape_shap:.5f}) "
              f"pitch_type={type_shap:.5f}", flush=True)

    preds_final = model.predict(final_feat[ALL_FEATURES])
    out = final_feat[["pitcher_id", "season"]].copy()
    out["pred"] = preds_final
    per_season = []
    for season, group in out.groupby("season"):
        g = group.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n_pitches=("pred", "size")).reset_index()
        g["score"] = _to_plus_scale(g["mean_pred"], group["pred"].mean(), group["pred"].std())
        g["season"] = season
        per_season.append(g)
    scores = pd.concat(per_season, ignore_index=True)
    scores["total_pitches"] = scores["n_pitches"]

    model.booster_.save_model(str(ARTIFACTS_DIR / "tunneling_v3_150ms.txt"))
    meta = dict(
        all_features=ALL_FEATURES, numeric_features=NUMERIC_LAG_COLS, categorical_features=CATEGORICAL_FEATURES,
        category_levels={c: final_feat[c].cat.categories.tolist() for c in CATEGORICAL_FEATURES},
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"), max_lag=MAX_LAG,
        commit_point_variant="150ms",
    )
    (ARTIFACTS_DIR / "tunneling_v3_150ms_meta.json").write_text(json.dumps(meta, indent=2))
    scores.to_parquet(ARTIFACTS_DIR / "tunneling_v3_150ms_scores.parquet", index=False)
    out.to_parquet(ARTIFACTS_DIR / "tunneling_v3_150ms_per_pitch_preds.parquet", index=False)

    report = dict(
        n_training_rows=int(len(feat)), all_features=ALL_FEATURES,
        cv_folds=fold_reports, cv_r2_mean=cv_r2_mean, cv_mae_mean=cv_mae_mean,
        mean_abs_shap=mean_abs_shap, lag_slot_shap_summary=lag_slot_summary,
    )
    (ARTIFACTS_DIR / "tunneling_v3_150ms_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nDone in {time.time()-t0:.0f}s. Saved {ARTIFACTS_DIR}/tunneling_v3_150ms.txt (+meta/report/scores)")


if __name__ == "__main__":
    main()
