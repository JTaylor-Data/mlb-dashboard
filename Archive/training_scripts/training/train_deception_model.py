"""
Deception+ v1 - offline training + validation script.

    python training/train_deception_model.py

Expanded from the original 2021-2026 training window to 2015-2026 now that
all those seasons are loaded in mlb_data.db with delta_run_exp available.
REPORT_SEASON updated to 2026. CV scheme stays leave-one-season-out on
2021-2025 (modern era, same 5-fold set as before) to keep runtime reasonable;
the final production model trains on the full 2015-2026 corpus.

All other modeling choices (Huber loss, LightGBM, feature set) are unchanged.
See project_artifacts/training_scripts/_restored_training/train_deception_model.py
for the original version and full design rationale.
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
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import analytics.stuff_model as sm
from analytics.deception_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    RAW_QUERY_COLUMNS, engineer_features, load_bio,
)
from analytics.percentiles import pitcher_percentile_metrics

DB_PATH = Path(__file__).resolve().parent.parent / "mlb_data.db"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Full training corpus — all seasons with delta_run_exp in the DB
COMPLETE_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
ALL_SEASONS = COMPLETE_SEASONS + [2026]
# CV only on the modern era to keep runtime manageable (5 folds)
CV_SEASONS = [2021, 2022, 2023, 2024, 2025]
REPORT_SEASON = 2026
MIN_PITCHES_QUALIFY = 300

LGBM_PARAMS = dict(
    objective="huber",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)


def load_pitches(conn, seasons):
    print(f"Loading pitches for seasons {seasons}...", flush=True)
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)}
        FROM pitches
        WHERE delta_run_exp IS NOT NULL
          AND game_type = 'R'
          AND season IN ({','.join('?' * len(seasons))})
    """, conn, params=seasons)
    for col in df.columns:
        if col == "p_throws":
            df[col] = df[col].astype("category")
        elif col not in ("pitcher_id", "season"):
            df[col] = df[col].astype("float32")
    print(f"  {len(df):,} rows loaded", flush=True)
    return df


def _fit_one_model(train_df, feature_cols):
    fit_rows, stop_rows = train_test_split(train_df, test_size=0.08, random_state=42)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        fit_rows[feature_cols], fit_rows["delta_run_exp"],
        eval_set=[(stop_rows[feature_cols], stop_rows["delta_run_exp"])],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def run_cross_validation(conn, bio):
    fold_reports = []
    for holdout_season in CV_SEASONS:
        print(f"\n--- CV fold: holding out {holdout_season} ---", flush=True)
        train_seasons = [s for s in ALL_SEASONS if s != holdout_season]
        train_raw = load_pitches(conn, train_seasons)
        test_raw = load_pitches(conn, [holdout_season])

        train_feat = engineer_features(train_raw, bio)
        test_feat = engineer_features(test_raw, bio)
        del train_raw, test_raw

        for col in CATEGORICAL_FEATURES:
            cats = sorted(set(train_feat[col].astype(str)) | set(test_feat[col].astype(str)))
            train_feat[col] = pd.Categorical(train_feat[col].astype(str), categories=cats)
            test_feat[col] = pd.Categorical(test_feat[col].astype(str), categories=cats)

        print(f"  fitting on {len(train_feat):,} rows, evaluating on {len(test_feat):,}...", flush=True)
        model = _fit_one_model(train_feat, ALL_FEATURES)
        preds = model.predict(test_feat[ALL_FEATURES])
        actual = test_feat["delta_run_exp"]

        r2 = r2_score(actual, preds)
        mae = mean_absolute_error(actual, preds)
        print(f"  holdout {holdout_season}: R^2={r2:.5f}  MAE={mae:.4f}  n={len(test_feat):,}", flush=True)

        fold_reports.append(dict(
            holdout_season=holdout_season, r2=r2, mae=mae, n_pitches=len(test_feat),
            best_iteration=int(model.best_iteration_),
        ))
        del train_feat, test_feat, model
        gc.collect()
    return fold_reports


def train_final_model(conn, bio):
    print("\n--- Training final model on all seasons (2015-2026) ---", flush=True)
    raw_df = load_pitches(conn, ALL_SEASONS)
    feat = engineer_features(raw_df, bio)
    del raw_df
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype(str).astype("category")
    print(f"  fitting on {len(feat):,} rows...", flush=True)
    model = _fit_one_model(feat, ALL_FEATURES)
    print(f"  final model best_iteration={model.best_iteration_}", flush=True)
    return model, feat


def compute_shap_summary(model, feat, sample_size=30_000):
    sample = feat.sample(n=min(sample_size, len(feat)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[ALL_FEATURES])
    return dict(zip(ALL_FEATURES, np.abs(shap_values).mean(axis=0).tolist()))


def _to_plus_scale(values, mean, std):
    return 100 - 10 * (values - mean) / std


def pitcher_deception_scores(model, feat, season):
    season_feat = feat[feat["season"].astype(str) == str(season)].copy()
    season_feat["pred"] = model.predict(season_feat[ALL_FEATURES])
    mean_, std_ = season_feat["pred"].mean(), season_feat["pred"].std()
    g = season_feat.groupby("pitcher_id").agg(deception_pred=("pred", "mean"), total_pitches=("pred", "size"))
    g["deception_plus"] = _to_plus_scale(g["deception_pred"], mean_, std_)
    return g[["deception_plus", "total_pitches"]]


def split_half_reliability(conn, model, bio, season):
    print("Running split-half reliability check...", flush=True)
    raw = load_pitches(conn, [season])
    raw = raw.sample(frac=1.0, random_state=42).reset_index(drop=True)
    raw["half"] = raw.index % 2

    half_feats = []
    for half in [0, 1]:
        hf = engineer_features(raw[raw["half"] == half].drop(columns=["half"]), bio)
        hf["half"] = half
        half_feats.append(hf)
    full_feat = pd.concat(half_feats, ignore_index=True)
    for col in CATEGORICAL_FEATURES:
        full_feat[col] = full_feat[col].astype(str).astype("category")
    full_feat["pred"] = model.predict(full_feat[ALL_FEATURES])
    mean_, std_ = full_feat["pred"].mean(), full_feat["pred"].std()

    halves = {}
    for half in [0, 1]:
        sub = full_feat[full_feat["half"] == half]
        g = sub.groupby("pitcher_id").agg(deception_pred=("pred", "mean"), n=("pred", "size"))
        g["deception_plus"] = _to_plus_scale(g["deception_pred"], mean_, std_)
        halves[half] = g[["deception_plus", "n"]]
    merged = halves[0].join(halves[1], lsuffix="_0", rsuffix="_1", how="inner")
    results = {}
    for min_pitches in [50, 100, 150]:
        sub = merged[(merged.n_0 >= min_pitches) & (merged.n_1 >= min_pitches)]
        results[str(min_pitches)] = dict(
            n=int(len(sub)),
            correlation=float(sub["deception_plus_0"].corr(sub["deception_plus_1"])) if len(sub) > 2 else None,
        )
    return results


def outcome_validity(conn, season, scores):
    print("Running outcome-correlation checks...", flush=True)
    stuff = sm.compute_stuff_plus(conn, season).reset_index()
    overall_stuff = stuff.groupby("pitcher_id").apply(
        lambda g: pd.Series({"overall_stuff": np.average(g["stuff_plus"], weights=g["n_pitches"])})
    ).reset_index()

    pool = pitcher_percentile_metrics(conn, season, stand_filter=None).reset_index()
    pool = pool.rename(columns={pool.columns[0]: "pitcher_id"})

    merged = scores.reset_index().merge(overall_stuff, on="pitcher_id").merge(pool, on="pitcher_id")
    merged = merged[merged["total_pitches"] >= MIN_PITCHES_QUALIFY].copy()

    def residualize(y, x):
        b, a = np.polyfit(x, y, 1)
        return y - (a + b * x)

    outcomes = ["whiff_pct", "chase_pct", "swstr_pct", "k_minus_bb_pct", "xwoba"]
    rows = {}
    resid_dec = residualize(merged["deception_plus"].values, merged["overall_stuff"].values)
    for o in outcomes:
        raw_r = float(merged["deception_plus"].corr(merged[o]))
        resid_outcome = residualize(merged[o].values, merged["overall_stuff"].values)
        partial_r = float(np.corrcoef(resid_outcome, resid_dec)[0, 1])
        rows[o] = dict(raw_correlation=raw_r, partial_correlation_stuff_removed=partial_r)

    return dict(
        n_qualified=int(len(merged)),
        min_pitches_qualify=MIN_PITCHES_QUALIFY,
        correlation_with_overall_stuff=float(merged["deception_plus"].corr(merged["overall_stuff"])),
        outcomes=rows,
    )


def worked_examples(conn, scores, top_n=10):
    names = dict(conn.execute("SELECT player_id, name FROM players").fetchall())
    s = scores.reset_index().copy()
    s["name"] = s["pitcher_id"].map(names)
    s = s[s["total_pitches"] >= MIN_PITCHES_QUALIFY]
    return dict(
        highest_deception=s.sort_values("deception_plus", ascending=False).head(top_n)[["pitcher_id", "name", "deception_plus", "total_pitches"]].to_dict(orient="records"),
        lowest_deception=s.sort_values("deception_plus").head(top_n)[["pitcher_id", "name", "deception_plus", "total_pitches"]].to_dict(orient="records"),
    )


def verdict(cv_r2_mean, reliability, validity):
    rel = reliability.get("100", {}).get("correlation")
    partials = [v["partial_correlation_stuff_removed"] for v in validity["outcomes"].values()]
    meaningful_partials = sum(1 for p in partials if abs(p) > 0.10)
    is_reliable = rel is not None and rel > 0.5
    has_signal = meaningful_partials >= 2
    if is_reliable and has_signal:
        headline = "Useful, with real caveats. The feature set is thin, but it's measuring something stable and predictive beyond Stuff+."
    elif is_reliable and not has_signal:
        headline = "Reliable but not obviously useful yet. The score is a stable measurement, but it doesn't clearly predict outcomes beyond what Stuff+ already explains."
    elif not is_reliable:
        headline = "Not yet useful. Split-half reliability is weak — the feature set as-is is likely too thin without the CV timing/pose features."
    else:
        headline = "Inconclusive."
    return dict(
        split_half_correlation_100plus=rel,
        n_outcomes_with_meaningful_partial_correlation=meaningful_partials,
        headline=headline,
    )


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    bio = load_bio(conn)
    print(f"Loaded bio data for {len(bio)} pitchers", flush=True)
    start = time.time()

    fold_reports = run_cross_validation(conn, bio)
    model, final_feat = train_final_model(conn, bio)
    shap_summary = compute_shap_summary(model, final_feat)

    scores = pitcher_deception_scores(model, final_feat, REPORT_SEASON)
    reliability = split_half_reliability(conn, model, bio, REPORT_SEASON)
    validity = outcome_validity(conn, REPORT_SEASON, scores)
    examples = worked_examples(conn, scores)
    verdict_result = verdict(np.mean([f["r2"] for f in fold_reports]), reliability, validity)

    model.booster_.save_model(str(MODELS_DIR / "deception_plus_v1.txt"))

    meta = dict(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        all_features=ALL_FEATURES,
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        seasons_trained_on=ALL_SEASONS,
        cv_seasons=CV_SEASONS,
        report_season=REPORT_SEASON,
        lgbm_params=LGBM_PARAMS,
    )
    (MODELS_DIR / "deception_plus_v1_meta.json").write_text(json.dumps(meta, indent=2))

    report = dict(
        trained_at=meta["trained_at"],
        report_season=REPORT_SEASON,
        seasons_trained_on=ALL_SEASONS,
        cv_seasons=CV_SEASONS,
        target="delta_run_exp (linear-weights run value per pitch)",
        excluded_features_note=(
            "Not yet included (awaiting CV pose pipeline): leg lift timing, "
            "time held in set, timing variance/quick pitches."
        ),
        cv_folds=fold_reports,
        cv_r2_mean=float(np.mean([f["r2"] for f in fold_reports])),
        cv_mae_mean=float(np.mean([f["mae"] for f in fold_reports])),
        mean_abs_shap=shap_summary,
        gain_importance_final_model=dict(zip(ALL_FEATURES, [float(x) for x in model.booster_.feature_importance("gain")])),
        split_half_reliability=reliability,
        outcome_validity=validity,
        worked_examples=examples,
        verdict=verdict_result,
    )
    (MODELS_DIR / f"deception_plus_v1_report_{REPORT_SEASON}.json").write_text(json.dumps(report, indent=2))

    print(f"\nDone in {time.time()-start:.0f}s.")
    print(f"CV mean R^2={report['cv_r2_mean']:.5f}, mean MAE={report['cv_mae_mean']:.4f}")
    print(f"VERDICT: {verdict_result['headline']}")
    print(f"Saved: {MODELS_DIR / 'deception_plus_v1.txt'}")
    print(f"Saved: {MODELS_DIR / 'deception_plus_v1_meta.json'}")
    print(f"Saved: {MODELS_DIR / f'deception_plus_v1_report_{REPORT_SEASON}.json'}")
    conn.close()


if __name__ == "__main__":
    main()
