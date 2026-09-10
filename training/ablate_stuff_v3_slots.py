"""
Stuff+ v3 - fastball-reference slot count ablation: 1-slot (primary only)
vs. 2-slot (primary+secondary) vs. 3-slot (primary+secondary+tertiary,
the full v3 feature set already trained). Deferred item from the original
dataset-construction session, now run with the real validation harness
built since (train_stuff_v3_lgbm.py's CV + validate_stuff_v3.py's
year-over-year scoring), reused here rather than reimplemented.

Does NOT touch stuff_plus_v2.txt, stuff_plus_v3_lgbm.txt, or any live-
serving code. Three new offline artifacts only:
    models/stuff_plus_v3_ablation_1slot.txt
    models/stuff_plus_v3_ablation_2slot.txt
    (3slot = the already-trained stuff_plus_v3_lgbm.txt, reused, not retrained)

## Identifier-leakage guard

Per direct instruction, every feature list used for training is checked
against a banned-identifier set before any .fit()/.predict() call - not
just asserted once by inspection. `assert_no_identifiers()` below raises
immediately if pitcher_id (or any other identifier) ever ends up in a
feature list, rather than relying on it having been correct historically.
pitcher_id IS used elsewhere in this script (grouping pitches into
per-pitcher-season reference profiles, aggregating per-pitcher scores for
the year-over-year check) - that's data engineering / evaluation
bookkeeping, not a training feature, and is a different, legitimate use
that the guard deliberately does not block.
"""
import gc
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from analytics.metrics import compute_vaa
from analytics.stuff_features import (
    CATEGORICAL_FEATURES, CORE_NUMERIC_FEATURES, fit_vaa_expectation,
)
from analytics import stuff_model as sm_runtime

MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("training/stuff_v3_artifacts")
FEATURES_PATH = ARTIFACTS_DIR / "stuff_v3_features.parquet"

COMPLETE_SEASONS = [2021, 2022, 2023, 2024, 2025]
ALL_SEASONS_INCL_PARTIAL = COMPLETE_SEASONS + [2026]
MIN_PITCHES_QUALIFY = 200

LGBM_PARAMS = dict(
    objective="huber", n_estimators=2000, learning_rate=0.03, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42,
)

BANNED_IDENTIFIER_COLS = {"pitcher_id", "game_id", "pitch_id", "player_id", "batter_id"}

SLOTS_BY_VARIANT = {
    "1slot": ["primary_fb"],
    "2slot": ["primary_fb", "secondary_fb"],
    "3slot": ["primary_fb", "secondary_fb", "tertiary_fb"],
}


def assert_no_identifiers(feature_cols, context):
    leaked = BANNED_IDENTIFIER_COLS & set(feature_cols)
    if leaked:
        raise ValueError(f"Identifier leakage in {context}: {leaked} found in feature columns. Refusing to train.")


def numeric_features_for(variant):
    slots = SLOTS_BY_VARIANT[variant]
    sep = [f"{metric}_sep_vs_{slot}" for slot in slots for metric in ("velo", "hb", "ivb")]
    return CORE_NUMERIC_FEATURES + ["axis_differential"] + sep


def refit_vaa_per_fold(train_df, test_df):
    train_df["vaa"] = compute_vaa(train_df["vy0"], train_df["vz0"], train_df["ay"], train_df["az"])
    vaa_coefs = fit_vaa_expectation(train_df)
    for df in (train_df, test_df):
        df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
        z = df["plate_z"].to_numpy(dtype=float)
        v = df["release_speed"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(df)), z, z**2, v])
        expected_vaa = design @ vaa_coefs
        df["vaa_residual"] = (df["vaa"] - expected_vaa).astype("float32")
    return train_df, test_df


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


def run_cv_for_variant(full_df, variant):
    numeric_features = numeric_features_for(variant)
    all_features = numeric_features + CATEGORICAL_FEATURES
    assert_no_identifiers(all_features, f"run_cv_for_variant({variant})")
    print(f"\n########## Variant: {variant} ({len(numeric_features)} numeric features) ##########", flush=True)

    fold_reports = []
    for holdout_season in COMPLETE_SEASONS:
        print(f"--- {variant} CV fold: holding out {holdout_season} ---", flush=True)
        train_df = full_df[full_df["season"] != str(holdout_season)].copy()
        test_df = full_df[full_df["season"] == str(holdout_season)].copy()
        train_df, test_df = refit_vaa_per_fold(train_df, test_df)

        for col in CATEGORICAL_FEATURES:
            cats = sorted(set(train_df[col].astype(str)) | set(test_df[col].astype(str)))
            train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
            test_df[col] = pd.Categorical(test_df[col].astype(str), categories=cats)

        model = _fit_one_model(train_df, all_features)
        preds = model.predict(test_df[all_features])
        actual = test_df["delta_run_exp"]
        r2 = r2_score(actual, preds)
        mae = mean_absolute_error(actual, preds)
        print(f"  holdout {holdout_season}: R^2={r2:.4f}  MAE={mae:.4f}  best_iter={model.best_iteration_}", flush=True)
        fold_reports.append(dict(holdout_season=holdout_season, r2=float(r2), mae=float(mae), best_iteration=int(model.best_iteration_)))
        del train_df, test_df, model
        gc.collect()
    return fold_reports, numeric_features, all_features


def train_final_and_score(full_df, variant, numeric_features, all_features):
    """Trains the final (all-seasons) model for this variant and computes
    per-pitcher-season Stuff+ scores for the year-over-year check - same
    shrinkage/z-score approach as the primary v3 pipeline, reused
    directly (model-agnostic)."""
    feat = full_df.copy()
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype(str).astype("category")
    print(f"\n--- {variant}: training final model on {len(feat):,} rows ---", flush=True)
    model = _fit_one_model(feat, all_features)

    preds = model.predict(feat[all_features])
    raw = feat[["pitcher_id", "pitch_type", "season"]].copy()
    raw["pred"] = preds

    per_season = []
    for season, group in raw.groupby("season"):
        k_by_type, fallback_k, type_means = sm_runtime._shrinkage_params(group)
        g = sm_runtime._shrunk_group_means(group, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
        g["stuff_plus"] = sm_runtime._to_stuff_plus(g["shrunk_mean"], group["pred"].mean(), group["pred"].std())
        g["season"] = season
        per_season.append(g)
    by_type = pd.concat(per_season, ignore_index=True)

    totals = by_type.groupby(["pitcher_id", "season"])["n_pitches"].sum().rename("n_pitches")
    weighted = by_type.groupby(["pitcher_id", "season"]).apply(
        lambda d: np.average(d["stuff_plus"], weights=d["n_pitches"]), include_groups=False
    )
    overall = pd.DataFrame({"stuff_plus": weighted}).join(totals).reset_index()
    overall = overall[overall["n_pitches"] >= MIN_PITCHES_QUALIFY]

    gain = dict(zip(all_features, [float(x) for x in model.booster_.feature_importance("gain")]))
    return model, overall, gain


def year_over_year(overall):
    seasons = sorted(overall["season"].unique())
    results = []
    for i in range(len(seasons) - 1):
        t, t1 = seasons[i], seasons[i + 1]
        dt = overall[overall["season"] == t][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t"})
        dt1 = overall[overall["season"] == t1][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t1"})
        merged = dt.merge(dt1, on="pitcher_id", how="inner")
        if len(merged) < 5:
            continue
        slope, intercept, r, p, se = stats.linregress(merged["t"], merged["t1"])
        results.append({"season_t": t, "season_t1": t1, "n": len(merged), "r2": round(float(r**2), 3)})
    return results


def main():
    start = time.time()
    print(f"Loading precomputed v3 feature table from {FEATURES_PATH}...", flush=True)
    full_df = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(full_df):,} rows loaded", flush=True)

    results = {}
    for variant in ("1slot", "2slot"):
        fold_reports, numeric_features, all_features = run_cv_for_variant(full_df, variant)
        model, overall, gain = train_final_and_score(full_df, variant, numeric_features, all_features)
        yoy = year_over_year(overall)

        model.booster_.save_model(str(MODELS_DIR / f"stuff_plus_v3_ablation_{variant}.txt"))

        cv_r2_mean = float(np.mean([f["r2"] for f in fold_reports]))
        cv_mae_mean = float(np.mean([f["mae"] for f in fold_reports]))
        yoy_r2_mean = float(np.mean([r["r2"] for r in yoy]))
        print(f"\n{variant} SUMMARY: CV R2={cv_r2_mean:.5f}, CV MAE={cv_mae_mean:.5f}, YoY r2 mean={yoy_r2_mean:.3f}", flush=True)

        sep_gain = {k: v for k, v in gain.items() if "_sep_vs_" in k}
        print(f"{variant} separation-feature gain importance: {sep_gain}", flush=True)

        results[variant] = dict(
            numeric_features=numeric_features, n_numeric_features=len(numeric_features),
            cv_folds=fold_reports, cv_r2_mean=cv_r2_mean, cv_mae_mean=cv_mae_mean,
            year_over_year=yoy, yoy_r2_mean=yoy_r2_mean,
            gain_importance=gain, separation_feature_gain=sep_gain,
            n_qualified_pitcher_seasons=len(overall),
        )
        del model, overall
        gc.collect()

    with open(ARTIFACTS_DIR / "slot_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {ARTIFACTS_DIR / 'slot_ablation_results.json'}")
    print(f"Done in {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
