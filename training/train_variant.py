"""
Train a Stuff+ variant for comparison (diagnostic, not promoted to production).
Usage: python training/train_variant.py <variant>
  Variants:
    b1_no_vaa           — drop vaa_residual entirely
    b2_raw_vaa          — replace vaa_residual with raw vaa
    b2_both_vaa         — keep vaa_residual AND add raw vaa
    b3_interaction      — keep vaa_residual AND add vaa_residual*below_zone_mid interaction
    b3_si_only          — SI-restricted interaction (zero for non-SI)
    vaa_directional     — sign-flipped feature: for SI rows multiply vaa_residual by -1

Each variant saves model + meta + report to models/variants/<variant>/.
Cross-validation uses the same 5-fold leave-one-season-out as the main model.
"""
import gc, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from analytics.metrics import compute_vaa
from analytics.stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    fit_vaa_expectation, fit_vaa_expectations_per_type,
)
import training.train_stuff_v3_lgbm as base

ARTIFACTS_DIR = Path(__file__).resolve().parent / "stuff_v3_artifacts"
FEATURES_PATH = ARTIFACTS_DIR / "stuff_v3_features.parquet"
COMPLETE_SEASONS = base.COMPLETE_SEASONS
LGBM_PARAMS = base.LGBM_PARAMS
ZONE_MID = 2.5  # ft — same as diagnostic


def build_variant_features(df, variant):
    """Return (feature_cols, df_with_new_cols) for the given variant."""
    df = df.copy()
    base_numeric = [f for f in NUMERIC_FEATURES if f != "vaa_residual"]

    if variant == "b1_no_vaa":
        numeric = base_numeric
    elif variant == "b2_raw_vaa":
        # Replace vaa_residual with raw vaa
        numeric = base_numeric + ["vaa"]
    elif variant == "b2_both_vaa":
        numeric = base_numeric + ["vaa_residual", "vaa"]
    elif variant == "b3_interaction":
        # Keep vaa_residual, add vaa_residual * (plate_z < ZONE_MID) — all pitch types
        below = (df["plate_z"].fillna(ZONE_MID) < ZONE_MID).astype(float)
        df["vaa_res_x_below"] = (df["vaa_residual"] * below).astype("float32")
        numeric = NUMERIC_FEATURES + ["vaa_res_x_below"]  # includes vaa_residual
    elif variant == "b3_si_only":
        # SI-restricted: interaction is non-zero only for SI rows — zero for all other types
        below = (df["plate_z"].fillna(ZONE_MID) < ZONE_MID).astype(float)
        is_si = (df["pitch_type"].astype(str) == "SI").astype(float)
        df["vaa_res_x_below_si"] = (df["vaa_residual"] * below * is_si).astype("float32")
        numeric = NUMERIC_FEATURES + ["vaa_res_x_below_si"]  # includes vaa_residual
    elif variant == "vaa_directional":
        # Sign-flip for SI: for SI rows, favorable VAA is steep (negative residual),
        # so multiply by -1 so that higher directional value = more favorable for ALL types.
        # For FF/CU/CH/SL/FS/FC the sign is unchanged — flat (positive) residual stays favorable.
        sign = np.where(df["pitch_type"].astype(str) == "SI", -1.0, 1.0)
        df["vaa_residual_directional"] = (df["vaa_residual"] * sign).astype("float32")
        numeric = base_numeric + ["vaa_residual_directional"]  # replaces vaa_residual
    else:
        raise ValueError(f"Unknown variant: {variant}")

    feature_cols = numeric + CATEGORICAL_FEATURES
    # Drop rows missing required numeric features
    req = [f for f in numeric if f in df.columns]
    df = df.dropna(subset=req)
    return feature_cols, df


def refit_vaa_per_fold_variant(train_df, test_df, variant):
    """Refit vaa-based features per fold for variants that need it."""
    if variant == "b1_no_vaa":
        return train_df, test_df  # vaa not used
    # For all other variants, refit vaa_residual (and compute vaa)
    train_df["vaa"] = compute_vaa(train_df["vy0"], train_df["vz0"], train_df["ay"], train_df["az"])
    pooled_coefs = fit_vaa_expectation(train_df)
    per_type_coefs = fit_vaa_expectations_per_type(train_df)
    vaa_coefs = {"pooled": pooled_coefs.tolist(), **per_type_coefs}

    for df in (train_df, test_df):
        df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
        z_arr = df["plate_z"].to_numpy(dtype=float)
        v_arr = df["release_speed"].to_numpy(dtype=float)
        expected = np.empty(len(df), dtype=float)
        pooled = np.array(vaa_coefs["pooled"])
        pt_map = {k: np.array(v) for k, v in vaa_coefs.items() if k != "pooled"}
        for pt, idx in df.groupby(df["pitch_type"].astype(str)).groups.items():
            coefs = pt_map.get(pt, pooled)
            iloc_idx = df.index.get_indexer(idx)
            design = np.column_stack([np.ones(len(idx)),
                                      z_arr[iloc_idx], z_arr[iloc_idx]**2,
                                      v_arr[iloc_idx]])
            expected[iloc_idx] = design @ coefs
        df["vaa_residual"] = (df["vaa"].to_numpy(dtype=float) - expected).astype("float32")
    return train_df, test_df


def run_cv(full_df, variant):
    fold_reports = []
    for holdout in COMPLETE_SEASONS:
        print(f"  fold holdout={holdout}...", flush=True)
        train_df = full_df[full_df["season"] != str(holdout)].copy()
        test_df  = full_df[full_df["season"] == str(holdout)].copy()
        train_df, test_df = refit_vaa_per_fold_variant(train_df, test_df, variant)
        feature_cols, train_df = build_variant_features(train_df, variant)
        _, test_df = build_variant_features(test_df, variant)

        for col in CATEGORICAL_FEATURES:
            cats = sorted(set(train_df[col].astype(str)) | set(test_df[col].astype(str)))
            train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
            test_df[col]  = pd.Categorical(test_df[col].astype(str),  categories=cats)

        fit_rows, stop_rows = train_test_split(train_df, test_size=0.08, random_state=42)
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(fit_rows[feature_cols], fit_rows["delta_run_exp"],
                  eval_set=[(stop_rows[feature_cols], stop_rows["delta_run_exp"])],
                  categorical_feature=CATEGORICAL_FEATURES,
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        preds  = model.predict(test_df[feature_cols])
        actual = test_df["delta_run_exp"]
        r2 = r2_score(actual, preds)
        mae = mean_absolute_error(actual, preds)
        print(f"    R²={r2:.4f}  MAE={mae:.4f}  n={len(test_df):,}  iter={model.best_iteration_}", flush=True)
        fold_reports.append(dict(holdout_season=holdout, r2=float(r2), mae=float(mae),
                                 n_pitches=int(len(test_df)),
                                 best_iteration=int(model.best_iteration_)))
        del train_df, test_df, model
        gc.collect()
    return fold_reports


def train_final(full_df, variant):
    print("  training final model...", flush=True)
    feat = full_df.copy()
    feature_cols, feat = build_variant_features(feat, variant)
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype(str).astype("category")
    fit_rows, stop_rows = train_test_split(feat, test_size=0.08, random_state=42)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(fit_rows[feature_cols], fit_rows["delta_run_exp"],
              eval_set=[(stop_rows[feature_cols], stop_rows["delta_run_exp"])],
              categorical_feature=CATEGORICAL_FEATURES,
              callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"  final best_iteration={model.best_iteration_}", flush=True)
    return model, feat, feature_cols


def main():
    if len(sys.argv) < 2:
        print("Usage: python training/train_variant.py <variant>")
        sys.exit(1)
    variant = sys.argv[1]
    print(f"\n=== Training variant: {variant} ===", flush=True)
    start = time.time()

    out_dir = Path(__file__).resolve().parent.parent / "models" / "variants" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded {len(full_df):,} rows", flush=True)

    fold_reports = run_cv(full_df, variant)
    model, final_feat, feature_cols = train_final(full_df, variant)

    model_path = out_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    calibrations = json.loads((ARTIFACTS_DIR / "calibrations.json").read_text())
    meta = dict(
        variant=variant,
        numeric_features=[f for f in feature_cols if f not in CATEGORICAL_FEATURES],
        categorical_features=CATEGORICAL_FEATURES,
        all_features=feature_cols,
        category_levels={col: final_feat[col].cat.categories.tolist()
                         for col in CATEGORICAL_FEATURES},
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        vaa_coefs=calibrations["vaa_coefs"],
        spin_calibration=calibrations["spin_calibration"],
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    report = dict(
        variant=variant,
        cv_folds=fold_reports,
        cv_r2_mean=float(np.mean([f["r2"] for f in fold_reports])),
        cv_mae_mean=float(np.mean([f["mae"] for f in fold_reports])),
        gain_importance={f: float(v) for f, v in
                         zip(feature_cols, model.booster_.feature_importance("gain"))},
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\nDone in {time.time()-start:.0f}s  CV R²={report['cv_r2_mean']:.4f}  MAE={report['cv_mae_mean']:.4f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
