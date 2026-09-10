"""
Train a standalone LightGBM sub-model on sinker (SI) rows only.
This is a diagnostic variant — NOT promoted to production.
Saves to models/variants/si_submodel/.

Design:
  - Feature set: same NUMERIC_FEATURES minus vaa_residual? No — we keep vaa_residual
    because within SI-only training, the model can learn the correct direction
    from SI's own empirical signal without being swamped by other pitch types.
  - pitch_type is dropped (constant = "SI").
  - Target: delta_run_exp (same as main model).
  - CV: 5-fold leave-one-season-out.
  - Z-scoring: raw predictions are z-scored against the POOLED league-wide
    distribution (all-pitch-type mean/std from the main model's predictions)
    so SI scores remain comparable to other pitch types' scores on the same page.
    This is stored in meta.json so the serving layer can apply it.
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
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    fit_vaa_expectation, fit_vaa_expectations_per_type,
)
import training.train_stuff_v3_lgbm as base

ARTIFACTS_DIR = Path(__file__).resolve().parent / "stuff_v3_artifacts"
FEATURES_PATH = ARTIFACTS_DIR / "stuff_v3_features.parquet"
COMPLETE_SEASONS = base.COMPLETE_SEASONS
LGBM_PARAMS = base.LGBM_PARAMS

# Features: drop pitch_type from categoricals (constant for SI); keep all numerics.
SI_NUMERIC   = NUMERIC_FEATURES          # includes vaa_residual
SI_CATEGORIC = [c for c in CATEGORICAL_FEATURES if c != "pitch_type"]
SI_FEATURES  = SI_NUMERIC + SI_CATEGORIC


def refit_vaa_for_fold(train_df, test_df):
    """Refit per-type VAA calibration inside each fold (same as main model)."""
    for df in (train_df, test_df):
        df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
    pooled_coefs = fit_vaa_expectation(train_df)
    per_type_coefs = fit_vaa_expectations_per_type(train_df)
    vaa_coefs = {"pooled": pooled_coefs.tolist(), **per_type_coefs}

    for df in (train_df, test_df):
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
    return train_df, test_df, vaa_coefs


def align_cats(train_df, test_df):
    for col in SI_CATEGORIC:
        cats = sorted(set(train_df[col].astype(str)) | set(test_df[col].astype(str)))
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
        test_df[col]  = pd.Categorical(test_df[col].astype(str),  categories=cats)
    return train_df, test_df


def run_cv(si_df):
    fold_reports = []
    for holdout in COMPLETE_SEASONS:
        print(f"  fold holdout={holdout}...", flush=True)
        train_df = si_df[si_df["season"] != str(holdout)].copy()
        test_df  = si_df[si_df["season"] == str(holdout)].copy()
        train_df, test_df, _ = refit_vaa_for_fold(train_df, test_df)
        train_df = train_df.dropna(subset=SI_NUMERIC)
        test_df  = test_df.dropna(subset=SI_NUMERIC)
        train_df, test_df = align_cats(train_df, test_df)

        fit_rows, stop_rows = train_test_split(train_df, test_size=0.08, random_state=42)
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(fit_rows[SI_FEATURES], fit_rows["delta_run_exp"],
                  eval_set=[(stop_rows[SI_FEATURES], stop_rows["delta_run_exp"])],
                  categorical_feature=SI_CATEGORIC,
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        preds  = model.predict(test_df[SI_FEATURES])
        actual = test_df["delta_run_exp"]
        r2  = r2_score(actual, preds)
        mae = mean_absolute_error(actual, preds)
        print(f"    R²={r2:.4f}  MAE={mae:.4f}  n={len(test_df):,}  iter={model.best_iteration_}", flush=True)
        fold_reports.append(dict(holdout_season=holdout, r2=float(r2), mae=float(mae),
                                 n_pitches=int(len(test_df)),
                                 best_iteration=int(model.best_iteration_)))
        del train_df, test_df, model
        gc.collect()
    return fold_reports


def train_final(si_df):
    print("  training final model...", flush=True)
    feat = si_df.dropna(subset=SI_NUMERIC).copy()
    for col in SI_CATEGORIC:
        feat[col] = feat[col].astype(str).astype("category")
    fit_rows, stop_rows = train_test_split(feat, test_size=0.08, random_state=42)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(fit_rows[SI_FEATURES], fit_rows["delta_run_exp"],
              eval_set=[(stop_rows[SI_FEATURES], stop_rows["delta_run_exp"])],
              categorical_feature=SI_CATEGORIC,
              callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"  final best_iteration={model.best_iteration_}", flush=True)
    return model, feat


def main():
    variant = "si_submodel"
    print(f"\n=== Training variant: {variant} ===", flush=True)
    start = time.time()

    out_dir = Path(__file__).resolve().parent.parent / "models" / "variants" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded {len(full_df):,} total rows", flush=True)
    si_df = full_df[full_df["pitch_type"] == "SI"].copy()
    print(f"SI rows: {len(si_df):,}", flush=True)

    fold_reports = run_cv(si_df)
    model, final_feat = train_final(si_df)

    model_path = out_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    calibrations = json.loads((ARTIFACTS_DIR / "calibrations.json").read_text())
    meta = dict(
        variant=variant,
        numeric_features=SI_NUMERIC,
        categorical_features=SI_CATEGORIC,
        all_features=SI_FEATURES,
        category_levels={col: final_feat[col].astype("category").cat.categories.tolist()
                         for col in SI_CATEGORIC},
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        vaa_coefs=calibrations["vaa_coefs"],
        spin_calibration=calibrations["spin_calibration"],
        # Z-scoring note: SI sub-model raw predictions are z-scored against the
        # pooled league-wide distribution from the main model, NOT the SI-only
        # distribution, so SI Stuff+ scores remain cross-type comparable.
        # The serving layer applies: 100 - 10*(pred - main_norm_mean)/main_norm_std
        # where main_norm_mean/std are loaded from the main model's artifacts.
        scoring_note="z-score against pooled league-wide distribution (main model norm params)",
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    report = dict(
        variant=variant,
        cv_folds=fold_reports,
        cv_r2_mean=float(np.mean([f["r2"] for f in fold_reports])),
        cv_mae_mean=float(np.mean([f["mae"] for f in fold_reports])),
        n_si_rows=int(len(si_df)),
        gain_importance={f: float(v) for f, v in
                         zip(SI_FEATURES, model.booster_.feature_importance("gain"))},
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\nDone in {time.time()-start:.0f}s  CV R²={report['cv_r2_mean']:.4f}  MAE={report['cv_mae_mean']:.4f}")
    print(f"n_si_rows={report['n_si_rows']:,}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
