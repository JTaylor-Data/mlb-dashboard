"""
Stuff+ v3 - Part A: LightGBM training + validation battery. Mirrors
train_stuff_model.py (v2) as closely as possible - same target, same Huber
objective, same LGBM_PARAMS, same z-score plus-scale formula, same
empirical-Bayes shrinkage (reused directly from analytics/stuff_model.py,
model-agnostic), same leave-one-season-out CV across the five complete
seasons. Only the feature set changes (see analytics/stuff_features_v3.py).

Does NOT touch models/stuff_plus_v2.txt, api/routers/, or any live-serving
code - offline comparison only, per session scope.

## One deliberate, flagged deviation from v2's CV methodology

v2 refits its VAA-expectation regression PER FOLD, from only that fold's
training data, so nothing about a held-out season leaks into feature
construction. v3 does the same for vaa_residual (see `refit_vaa_per_fold`
below - cheap, a plain least-squares fit on already-loaded data). It does
NOT refit the fastball-reference classification or the spin-axis
calibration per fold - those are reused as-is from the precomputed
training/stuff_v3_artifacts/stuff_v3_features.parquet (built on all 6
seasons combined), per this session's explicit instruction not to rebuild
that table. This is a real, if minor, difference from strict apples-to-
apples CV discipline - flagged here and in the final write-up rather than
silently accepted. The leakage risk is judged low: fb-reference
classification is a structural fact about a pitcher's arsenal composition
(which pitch type IS his fastball), and the spin-axis calibration is a
broad physical/geometric relationship fit on 1.3M+ FF pitches - neither is
a parameter fit TO the run-value target the way VAA's regression is, so
neither should meaningfully preview a held-out season's actual outcomes.
"""
import gc
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

import analytics.stuff_model as sm_runtime
from analytics.metrics import compute_vaa
from analytics.stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, OPTIONAL_NUMERIC_FEATURES,
    fit_vaa_expectation, fit_vaa_expectations_per_type,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "stuff_v3_artifacts"
FEATURES_PATH = ARTIFACTS_DIR / "stuff_v3_features.parquet"

# Same guard as ablate_stuff_v3_slots.py - checked once by inspection isn't
# enough per the user's explicit instruction; this asserts it at every fit.
BANNED_IDENTIFIER_COLS = {"pitcher_id", "game_id", "pitch_id", "player_id", "batter_id"}


def assert_no_identifiers(feature_cols, context):
    leaked = BANNED_IDENTIFIER_COLS & set(feature_cols)
    if leaked:
        raise ValueError(f"Identifier leakage in {context}: {leaked} found in feature columns. Refusing to train.")

COMPLETE_SEASONS = [2021, 2022, 2023, 2024, 2025]
ALL_SEASONS = COMPLETE_SEASONS + [2026]

# Identical to v2's LGBM_PARAMS - only the feature set changes this round.
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


def load_v3_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found - this script does not rebuild the v3 feature "
            "table (see module docstring). Run training/build_stuff_v3_dataset.py first."
        )
    print(f"Loading precomputed v3 feature table from {FEATURES_PATH}...", flush=True)
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(df):,} rows loaded", flush=True)
    return df


def refit_vaa_per_fold(train_df, test_df):
    """Refits vaa_residual (per-type where possible) from ONLY train_df's raw
    columns, matching v2's per-fold VAA-leakage discipline. Returns (train_df,
    test_df) with vaa/vaa_residual overwritten using the fold-specific fit."""
    train_df["vaa"] = compute_vaa(train_df["vy0"], train_df["vz0"], train_df["ay"], train_df["az"])
    pooled_coefs = fit_vaa_expectation(train_df)
    per_type_coefs = fit_vaa_expectations_per_type(train_df)
    vaa_coefs = {"pooled": pooled_coefs.tolist(), **per_type_coefs}

    for df in (train_df, test_df):
        df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
        z_arr = df["plate_z"].to_numpy(dtype=float)
        v_arr = df["release_speed"].to_numpy(dtype=float)
        expected_vaa = np.empty(len(df), dtype=float)
        pooled = np.array(vaa_coefs["pooled"])
        pt_map = {k: np.array(v) for k, v in vaa_coefs.items() if k != "pooled"}
        pt_series = df["pitch_type"].astype(str)
        for pt, idx in df.groupby(pt_series).groups.items():
            coefs = pt_map.get(pt, pooled)
            iloc_idx = df.index.get_indexer(idx)
            sub_z = z_arr[iloc_idx]
            sub_v = v_arr[iloc_idx]
            design = np.column_stack([np.ones(len(idx)), sub_z, sub_z**2, sub_v])
            expected_vaa[iloc_idx] = design @ coefs
        df["vaa_residual"] = (df["vaa"].to_numpy(dtype=float) - expected_vaa).astype("float32")
    return train_df, test_df, vaa_coefs


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


def run_cross_validation(full_df):
    fold_reports = []
    for holdout_season in COMPLETE_SEASONS:
        print(f"\n--- CV fold: holding out {holdout_season} ---", flush=True)
        train_df = full_df[full_df["season"] != str(holdout_season)].copy()
        test_df = full_df[full_df["season"] == str(holdout_season)].copy()

        train_df, test_df, _ = refit_vaa_per_fold(train_df, test_df)

        for col in CATEGORICAL_FEATURES:
            cats = sorted(set(train_df[col].astype(str)) | set(test_df[col].astype(str)))
            train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
            test_df[col] = pd.Categorical(test_df[col].astype(str), categories=cats)

        print(f"  fitting on {len(train_df):,} rows, evaluating on {len(test_df):,}...", flush=True)
        model = _fit_one_model(train_df, ALL_FEATURES)
        preds = model.predict(test_df[ALL_FEATURES])
        actual = test_df["delta_run_exp"]

        r2 = r2_score(actual, preds)
        mae = mean_absolute_error(actual, preds)
        print(f"  holdout {holdout_season}: R^2={r2:.4f}  MAE={mae:.4f}  n={len(test_df):,}  best_iter={model.best_iteration_}", flush=True)

        fold_reports.append(dict(
            holdout_season=holdout_season, r2=float(r2), mae=float(mae), n_pitches=int(len(test_df)),
            best_iteration=int(model.best_iteration_),
            feature_importance_gain=dict(zip(ALL_FEATURES, [float(x) for x in model.booster_.feature_importance("gain")])),
        ))
        del train_df, test_df, model
        gc.collect()
    return fold_reports


def train_final_model(full_df):
    print("\n--- Training final model on all seasons ---", flush=True)
    feat = full_df.copy()
    # Final model uses vaa_residual as already computed in the precomputed
    # table (fit on all 6 seasons combined) - matches v2's train_final_model,
    # which also fits VAA on ALL_SEASONS combined for the shipped model.
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype(str).astype("category")
    print(f"  fitting final model on {len(feat):,} rows...", flush=True)
    model = _fit_one_model(feat, ALL_FEATURES)
    print(f"  final model best_iteration={model.best_iteration_}", flush=True)
    return model, feat


def compute_shap_summary(model, feat, sample_size=30_000):
    print(f"\n--- Computing SHAP summary on a sample of {sample_size:,} pitches ---", flush=True)
    sample = feat.sample(n=min(sample_size, len(feat)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[ALL_FEATURES])
    mean_abs_shap = dict(zip(ALL_FEATURES, np.abs(shap_values).mean(axis=0).tolist()))
    return mean_abs_shap, sample, shap_values, explainer


def worked_examples(model, feat):
    preds = model.predict(feat[ALL_FEATURES])
    feat = feat.assign(pred=preds)
    cols = ["pitcher_id", "pitch_type", "season", "release_speed", "hb", "ivb", "axis_differential", "pred"]
    best = feat.nsmallest(5, "pred")[cols]
    worst = feat.nlargest(5, "pred")[cols]
    return dict(best_predicted_pitches=best.to_dict(orient="records"), worst_predicted_pitches=worst.to_dict(orient="records"))


def shrinkage_report(final_feat, preds):
    raw = final_feat[["pitcher_id", "pitch_type"]].copy()
    raw["pred"] = preds
    k_by_type, fallback_k, _ = sm_runtime._shrinkage_params(raw)
    return dict(
        k_by_pitch_type={str(k): float(v) for k, v in k_by_type.items()},
        fallback_k=float(fallback_k),
    )


def collinearity_report(feat):
    """Pairwise correlation among the new v3 features, as its own explicit
    check per direction - not skipped, not folded silently into the SHAP
    summary. Cross-slot separation correlation (the OTHER collinearity
    finding from the original 3-slot build) no longer applies now that
    there's only one slot - see train_stuff_v3_ablate_slots.py's saved
    results for that historical finding (it's what led to collapsing to
    primary-only in the first place)."""
    print("\n--- Collinearity check ---", flush=True)
    report = {}

    axis_group = ["axis_differential", "hb", "ivb", "spin_axis"]
    axis_corr = feat[axis_group].corr()
    report["axis_differential_vs_movement_spin"] = axis_corr.round(4).to_dict()
    print("axis_differential vs hb/ivb/spin_axis:")
    print(axis_corr.round(3))

    sep_group = [f"{metric}_sep_vs_primary_fb" for metric in ("velo", "hb", "ivb")]
    sep_corr = feat[sep_group + ["hb", "ivb", "release_speed", "effective_speed"]].corr()
    report["separation_vs_raw_features"] = sep_corr.round(4).to_dict()
    print("\nvelo/hb/ivb_sep_vs_primary_fb vs. their own raw counterparts:")
    print(sep_corr.round(3))

    return report


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    start = time.time()

    full_df = load_v3_features()
    collinearity = collinearity_report(full_df)

    fold_reports = run_cross_validation(full_df)
    model, final_feat = train_final_model(full_df)
    shap_summary, shap_sample, shap_values, explainer = compute_shap_summary(model, final_feat)
    examples = worked_examples(model, final_feat)
    preds_final = model.predict(final_feat[ALL_FEATURES])
    shrinkage = shrinkage_report(final_feat, preds_final)

    model.booster_.save_model(str(MODELS_DIR / "stuff_plus_v3_lgbm.txt"))

    # analytics/stuff_model.py (live scoring) needs vaa_coefs + spin_calibration
    # in this same meta.json to reproduce the exact training-time transform -
    # they're not fit by this script (see cv_methodology_note below), they're
    # fit once by build_stuff_v3_dataset.py and saved to calibrations.json.
    # Merged in here, not as a separate manual patch step, after a real
    # promotion bug: an earlier retrain overwrote this file WITHOUT these
    # keys (this script never used to write them), which would have broken
    # live scoring the moment the process cache reloaded - caught before it
    # shipped, but only by hand-checking, not because anything would have
    # errored loudly at retrain time.
    calibrations_path = ARTIFACTS_DIR / "calibrations.json"
    calibrations = json.loads(calibrations_path.read_text())

    meta = dict(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        all_features=ALL_FEATURES,
        category_levels={col: final_feat[col].cat.categories.tolist() for col in CATEGORICAL_FEATURES},
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        seasons_trained_on=ALL_SEASONS,
        lgbm_params=LGBM_PARAMS,
        cv_methodology_note=(
            "vaa_residual refit per CV fold (matches v2). Fastball-reference "
            "classification and spin-axis calibration are NOT refit per fold - "
            "reused from the precomputed full-dataset build, per this session's "
            "explicit instruction not to rebuild that table. See module docstring."
        ),
        vaa_coefs=calibrations["vaa_coefs"],
        spin_calibration=calibrations["spin_calibration"],
    )
    (MODELS_DIR / "stuff_plus_v3_lgbm_meta.json").write_text(json.dumps(meta, indent=2))

    report = dict(
        trained_at=meta["trained_at"],
        target="delta_run_exp (same as v2)",
        cv_folds=fold_reports,
        cv_r2_mean=float(np.mean([f["r2"] for f in fold_reports])),
        cv_mae_mean=float(np.mean([f["mae"] for f in fold_reports])),
        mean_abs_shap=shap_summary,
        gain_importance_final_model=dict(zip(ALL_FEATURES, [float(x) for x in model.booster_.feature_importance("gain")])),
        worked_examples=examples,
        shrinkage=shrinkage,
        collinearity=collinearity,
    )
    (MODELS_DIR / "stuff_plus_v3_lgbm_report.json").write_text(json.dumps(report, indent=2))

    # Also persist the raw SHAP sample+values for the spot-check script to
    # reuse without recomputing (TreeExplainer over 30k rows is not free).
    shap_sample.reset_index(drop=True).to_parquet(ARTIFACTS_DIR / "shap_sample.parquet")
    np.save(ARTIFACTS_DIR / "shap_values.npy", shap_values)
    final_feat.reset_index(drop=True).to_parquet(ARTIFACTS_DIR / "final_feat_with_preds.parquet") if False else None
    # (final_feat is the same as the precomputed table plus vaa_residual
    # already baked in - not re-saved to avoid a second 300MB+ file; preds
    # can be recomputed cheaply from the saved booster if needed.)

    print(f"\nDone in {time.time() - start:.0f}s. CV mean R^2={report['cv_r2_mean']:.4f}, mean MAE={report['cv_mae_mean']:.4f}")
    print(f"Saved: {MODELS_DIR / 'stuff_plus_v3_lgbm.txt'}")
    print(f"Saved: {MODELS_DIR / 'stuff_plus_v3_lgbm_meta.json'}")
    print(f"Saved: {MODELS_DIR / 'stuff_plus_v3_lgbm_report.json'}")


if __name__ == "__main__":
    main()
