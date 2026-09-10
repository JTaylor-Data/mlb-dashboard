"""
Location+ v1 - Part E.5: stand-inclusion ablation. Trains a second variant
with `stand` dropped from the feature set (matching FanGraphs' own public
methodology exactly) and compares CV R²/MAE (accuracy) and year-over-year
pooled r² (stability) against the real, promoted-for-this-report model
(which keeps `stand`) - same two metrics Stuff+ v3's own primary-fastball-
slot ablation used to judge whether a feature/design choice earns its keep,
not just accuracy alone. Same CV scheme, same LGBM_PARAMS, same everything
else - `stand` is the only thing that changes.

Usage:
    python training/location_stand_ablation.py
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from training.train_location_model import (
    ARTIFACTS_DIR, CATEGORICAL_FEATURES, DB_PATH, LGBM_PARAMS, NUMERIC_FEATURES,
    load_feat,
)
from analytics.location_features import to_plus_scale
from sklearn.metrics import mean_absolute_error, r2_score

NO_STAND_CATEGORICAL = [c for c in CATEGORICAL_FEATURES if c != "stand"]
NO_STAND_FEATURES = NUMERIC_FEATURES + NO_STAND_CATEGORICAL
COMPLETE_SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_PITCHES_QUALIFY = 100


def run_cv_no_stand(feat):
    fold_reports = []
    for holdout in COMPLETE_SEASONS:
        train_df = feat[feat["season"] != str(holdout)].copy()
        test_df = feat[feat["season"] == str(holdout)].copy()
        for col in NO_STAND_CATEGORICAL:
            cats = sorted(set(train_df[col][train_df[col].notna()]) | set(test_df[col][test_df[col].notna()]))
            train_df[col] = pd.Categorical(train_df[col], categories=cats)
            test_df[col] = pd.Categorical(test_df[col], categories=cats)
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        fit_rows, stop_rows = train_test_split(train_df, test_size=0.08, random_state=42)
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            fit_rows[NO_STAND_FEATURES], fit_rows["delta_run_exp"],
            eval_set=[(stop_rows[NO_STAND_FEATURES], stop_rows["delta_run_exp"])],
            categorical_feature=NO_STAND_CATEGORICAL,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(test_df[NO_STAND_FEATURES])
        r2 = r2_score(test_df["delta_run_exp"], preds)
        mae = mean_absolute_error(test_df["delta_run_exp"], preds)
        print(f"  holdout {holdout}: R^2={r2:.4f} MAE={mae:.4f} n={len(test_df):,}", flush=True)
        fold_reports.append(dict(holdout_season=holdout, r2=float(r2), mae=float(mae), n=int(len(test_df))))
    return fold_reports


def year_over_year_no_stand(feat):
    """Trains the final no-stand model on all seasons, scores every pitch,
    computes per-pitcher-season unshrunk scores, and the same pooled YoY r²
    as the main validation script - to see if dropping stand meaningfully
    changes season-to-season stability, not just raw CV accuracy."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    final_feat = feat.copy()
    for col in NO_STAND_CATEGORICAL:
        cats = sorted(set(final_feat[col][final_feat[col].notna()]))
        final_feat[col] = pd.Categorical(final_feat[col], categories=cats)
    fit_rows, stop_rows = train_test_split(final_feat, test_size=0.08, random_state=42)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        fit_rows[NO_STAND_FEATURES], fit_rows["delta_run_exp"],
        eval_set=[(stop_rows[NO_STAND_FEATURES], stop_rows["delta_run_exp"])],
        categorical_feature=NO_STAND_CATEGORICAL,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    preds = model.predict(final_feat[NO_STAND_FEATURES])
    raw = final_feat[["pitcher_id", "season"]].copy()
    raw["pred"] = preds

    per_season = []
    for season, group in raw.groupby("season"):
        g = group.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n_pitches=("pred", "size")).reset_index()
        g["score"] = to_plus_scale(g["mean_pred"], group["pred"].mean(), group["pred"].std())
        g["season"] = season
        per_season.append(g)
    scores = pd.concat(per_season, ignore_index=True)
    scores["total_pitches"] = scores["n_pitches"]

    q = scores[scores["total_pitches"] >= MIN_PITCHES_QUALIFY]
    seasons = sorted(q["season"].unique())
    pooled_t, pooled_t1 = [], []
    for i in range(len(seasons) - 1):
        t, t1 = seasons[i], seasons[i + 1]
        dt = q[q["season"] == t][["pitcher_id", "score"]].rename(columns={"score": "t"})
        dt1 = q[q["season"] == t1][["pitcher_id", "score"]].rename(columns={"score": "t1"})
        merged = dt.merge(dt1, on="pitcher_id", how="inner")
        pooled_t.extend(merged["t"].tolist())
        pooled_t1.extend(merged["t1"].tolist())
    r = np.corrcoef(pooled_t, pooled_t1)[0, 1] if len(pooled_t) >= 5 else None
    return round(float(r ** 2), 3) if r is not None else None, len(pooled_t)


def main():
    conn = sqlite3.connect(DB_PATH)
    t0 = time.time()
    print("Loading feature table (same as the main model)...", flush=True)
    feat = load_feat(conn)

    print("\n--- CV, no-stand variant ---", flush=True)
    fold_reports = run_cv_no_stand(feat)
    cv_r2_mean = float(np.mean([f["r2"] for f in fold_reports]))
    cv_mae_mean = float(np.mean([f["mae"] for f in fold_reports]))
    print(f"No-stand CV mean R^2={cv_r2_mean:.4f}, mean MAE={cv_mae_mean:.4f}", flush=True)

    print("\n--- Year-over-year, no-stand variant ---", flush=True)
    yoy_r2, n_pooled = year_over_year_no_stand(feat)
    print(f"No-stand pooled YoY r2={yoy_r2} (n={n_pooled})")

    main_report = json.loads((Path("models") / "location_plus_v1_report.json").read_text())
    main_validation = json.loads((ARTIFACTS_DIR / "location_v1_validation.json").read_text())

    print("\n=== Comparison: WITH stand (promoted) vs. WITHOUT stand (FanGraphs-exact) ===")
    print(f"  CV R^2:        with={main_report['cv_r2_mean']:.4f}   without={cv_r2_mean:.4f}   delta={main_report['cv_r2_mean']-cv_r2_mean:+.4f}")
    print(f"  CV MAE:        with={main_report['cv_mae_mean']:.4f}   without={cv_mae_mean:.4f}   delta={main_report['cv_mae_mean']-cv_mae_mean:+.4f}")
    print(f"  YoY pooled r2: with={main_validation['year_over_year']['pooled_r2']}   without={yoy_r2}")

    out = dict(
        with_stand=dict(cv_r2_mean=main_report["cv_r2_mean"], cv_mae_mean=main_report["cv_mae_mean"],
                         yoy_pooled_r2=main_validation["year_over_year"]["pooled_r2"]),
        without_stand=dict(cv_r2_mean=cv_r2_mean, cv_mae_mean=cv_mae_mean, yoy_pooled_r2=yoy_r2, cv_folds=fold_reports),
    )
    out_path = ARTIFACTS_DIR / "location_v1_stand_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nDone in {time.time()-t0:.0f}s. Saved {out_path}")


if __name__ == "__main__":
    main()
