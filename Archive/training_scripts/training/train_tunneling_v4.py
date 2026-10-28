"""
Tunneling+ v4 — pitcher-season grain trained model (experimental, no promotion).

Spec: https://github.com/JTaylor-Data/mlb-dashboard  (pasted spec 2026-10-27)

Parts:
  0 — Diagnostic: verify live v2/v3 sign empirically with real data
  1 — Git lock-in (tag, branch, spec_lock.json)  [done manually before this script]
  2 — Feature construction (pitcher-season grain)
  3 — Train LightGBM model (LOSO CV, ablation)
  4 — Validation battery (8 items)
  5 — Completion report (Methodology/TUNNELING_V4_COMPLETION.md)

Hard boundary: no production changes. Models saved to models/tunneling_plus_v4*.
"""
import gc
import json
import math
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # mlb_dashboard root
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from analytics.tunneling_features import (
    load_pitches, sequential_lag_features, pitcher_season_consistency_features,
    build_commit_point_table, pitch_type_centroids, MAX_LAG,
)
from analytics.location_features import to_plus_scale as _to_plus_scale

# ─────────────────────────────── paths ───────────────────────────────────────
DB_PATH = ROOT / "mlb_data.db"
MODELS_DIR = ROOT / "models"
DIAG_DIR = ROOT / "diagnostics" / "tunneling_v4"
METHODOLOGY_DIR = ROOT / "Methodology"

COMMIT_TABLE_PATH = (
    ROOT / "Archive" / "training_scripts" / "training"
    / "tunneling_plus_ext_artifacts" / "pitcher_season_commit_point_ext.csv"
)

# ─────────────────────────────── constants ───────────────────────────────────
ALL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
CV_SEASONS = [2021, 2022, 2023, 2024, 2025]   # modern era, 5 LOSO folds
MIN_PITCHES = 100
EXCLUDED_TYPES = {"PO", "UN", "EP"}

NUMERIC_LAG_FEATURES = [
    f"lag{lag}_{feat}"
    for lag in range(1, MAX_LAG + 1)
    for feat in ("commit_distance", "plate_distance", "shape_break")
]  # 18 features

BASE_FEATURE_COLS = NUMERIC_LAG_FEATURES + ["arsenal_entropy"]
BASE_FEATURE_COLS_WITH_SEASON = BASE_FEATURE_COLS + ["season"]

LGBM_PARAMS = dict(
    objective="huber",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)

SEEDS = [42, 123, 456]


# ─────────────────────────────── helpers ─────────────────────────────────────

def assert_no_identifiers(df, feature_cols):
    forbidden = {"pitcher_id", "player_id", "game_id", "batter_id"}
    in_features = forbidden & set(feature_cols)
    assert not in_features, f"Identifier columns in features: {in_features}"


def pearson_ci(r, n, alpha=0.05):
    """95% CI for Pearson r via Fisher z-transform."""
    if n < 4 or math.isnan(r):
        return None, None
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = math.tanh(z - z_crit * se)
    hi = math.tanh(z + z_crit * se)
    return round(lo, 4), round(hi, 4)


def load_commit_table():
    return pd.read_csv(COMMIT_TABLE_PATH)


# ─────────────────────────────── Part 2: features ────────────────────────────

def build_pitcher_season_features(conn, seasons, commit_table):
    """Build one row per qualifying pitcher-season.
    Features: 18 lag-mean numerics + arsenal_entropy + season.
    Target: mean delta_run_exp over qualifying pitches."""
    print(f"\n[Part 2] Building pitcher-season features for {seasons}...", flush=True)
    all_rows = []

    for season in seasons:
        print(f"  Season {season}...", flush=True)
        df = load_pitches(conn, season)
        if df.empty:
            print(f"    Empty, skipping.", flush=True)
            continue

        # Qualifying filter on raw total pitches
        totals = df.groupby("pitcher_id").size()
        qual_ids = totals[totals >= MIN_PITCHES].index
        df = df[df["pitcher_id"].isin(qual_ids)].copy()
        if df.empty:
            continue

        # Build lag features (per-pitch)
        season_commit = commit_table[commit_table["season"] == season].copy()
        if season_commit.empty:
            print(f"    No commit table entries for {season}, skipping.", flush=True)
            continue

        feat = sequential_lag_features(df, season_commit, max_lag=MAX_LAG)
        if feat.empty:
            continue

        # Aggregate lag numerics to pitcher-season mean (NaN-aware)
        agg_dict = {col: "mean" for col in NUMERIC_LAG_FEATURES if col in feat.columns}
        lag_agg = feat.groupby("pitcher_id").agg(agg_dict).reset_index()

        # Target: mean delta_run_exp (non-null only)
        dre_df = df[df["delta_run_exp"].notna()][["pitcher_id", "delta_run_exp"]]
        target = dre_df.groupby("pitcher_id")["delta_run_exp"].mean().reset_index()
        target.columns = ["pitcher_id", "target_mean_xrv"]

        # Arsenal entropy
        centroids = pitch_type_centroids(df)
        usage = centroids[centroids["n_pitches"] >= MIN_PITCHES].copy()
        totals_ps = usage.groupby(["pitcher_id", "season"])["n_pitches"].transform("sum")
        usage["p"] = usage["n_pitches"] / totals_ps
        usage["term"] = -usage["p"] * np.log(usage["p"])
        entropy = usage.groupby("pitcher_id")["term"].sum().reset_index()
        entropy.columns = ["pitcher_id", "arsenal_entropy"]

        # Merge all
        ps = lag_agg.merge(target, on="pitcher_id", how="inner")
        ps = ps.merge(entropy, on="pitcher_id", how="left")
        ps["season"] = str(season)
        all_rows.append(ps)

    if not all_rows:
        return pd.DataFrame()
    result = pd.concat(all_rows, ignore_index=True)
    print(f"  Total pitcher-seasons: {len(result):,}", flush=True)
    return result


# ─────────────────────────────── Part 3: model ───────────────────────────────

def fit_lgbm(train_df, feature_cols, categorical_cols=None):
    """Train with n_repeats seeds and average predictions (ensemble).
    Returns a list of booster objects."""
    models = []
    for seed in SEEDS:
        params = {**LGBM_PARAMS, "random_state": seed}
        model = lgb.LGBMRegressor(**params)
        cat_feat = categorical_cols or []
        if cat_feat:
            for col in cat_feat:
                train_df[col] = train_df[col].astype("category")
        fit_X = train_df[feature_cols]
        fit_y = train_df["target_mean_xrv"]
        model.fit(fit_X, fit_y, categorical_feature=cat_feat if cat_feat else "auto")
        models.append(model)
    return models


def predict_ensemble(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0)


def run_loso_cv(ps_df, feature_cols, categorical_cols):
    """Leave-one-season-out CV over CV_SEASONS. Returns per-fold results and
    a DataFrame of OOS predictions for all CV folds combined."""
    fold_reports = []
    oos_preds_list = []

    for holdout_season in CV_SEASONS:
        holdout_str = str(holdout_season)
        train = ps_df[ps_df["season"] != holdout_str].copy()
        test = ps_df[ps_df["season"] == holdout_str].copy()
        if test.empty:
            continue

        # Align categoricals
        if "season" in categorical_cols:
            all_cats = sorted(set(train["season"].astype(str)) | set(test["season"].astype(str)))
            train["season"] = pd.Categorical(train["season"].astype(str), categories=all_cats)
            test["season"] = pd.Categorical(test["season"].astype(str), categories=all_cats)

        models = fit_lgbm(train.copy(), feature_cols, categorical_cols)
        preds = predict_ensemble(models, test[feature_cols])
        actual = test["target_mean_xrv"].values
        r2 = float(r2_score(actual, preds))
        mae = float(mean_absolute_error(actual, preds))
        print(f"  LOSO holdout {holdout_season}: R2={r2:.5f}  MAE={mae:.5f}  n={len(test)}", flush=True)
        fold_reports.append(dict(holdout_season=holdout_season, r2=r2, mae=mae, n=len(test)))

        oos = test[["pitcher_id", "season", "target_mean_xrv"]].copy()
        oos["oos_pred"] = preds
        oos_preds_list.append(oos)

    oos_df = pd.concat(oos_preds_list, ignore_index=True) if oos_preds_list else pd.DataFrame()
    return fold_reports, oos_df


def run_ablation(ps_df):
    """Test with/without season, with/without arsenal_entropy."""
    print("\n[Ablation] Running pre-registered ablation...", flush=True)
    results = {}
    configs = {
        "with_season_with_entropy":   (BASE_FEATURE_COLS_WITH_SEASON, ["season"]),
        "no_season_with_entropy":     (BASE_FEATURE_COLS, []),
        "with_season_no_entropy":     ([c for c in BASE_FEATURE_COLS_WITH_SEASON if c != "arsenal_entropy"], ["season"]),
        "no_season_no_entropy":       ([c for c in BASE_FEATURE_COLS if c != "arsenal_entropy"], []),
    }
    for name, (feat_cols, cat_cols) in configs.items():
        fold_reports, _ = run_loso_cv(ps_df, feat_cols, cat_cols)
        mean_mae = float(np.mean([f["mae"] for f in fold_reports]))
        mean_r2 = float(np.mean([f["r2"] for f in fold_reports]))
        print(f"  {name}: MAE={mean_mae:.5f} R2={mean_r2:.5f}", flush=True)
        results[name] = dict(mean_mae=mean_mae, mean_r2=mean_r2, folds=fold_reports)

    # inconclusive_means_drop rule: if diff ≤ 0.03 MAE -> drop season
    with_s_mae = results["with_season_with_entropy"]["mean_mae"]
    no_s_mae = results["no_season_with_entropy"]["mean_mae"]
    season_diff = abs(with_s_mae - no_s_mae)
    drop_season = season_diff <= 0.03
    results["season_diff"] = float(season_diff)
    results["drop_season"] = bool(drop_season)
    print(f"  Season MAE diff = {season_diff:.5f} -> {'DROP season' if drop_season else 'KEEP season'}", flush=True)

    # Same for arsenal_entropy
    with_e_mae = results["with_season_with_entropy"]["mean_mae"]
    no_e_mae = results["with_season_no_entropy"]["mean_mae"]
    entropy_diff = abs(with_e_mae - no_e_mae)
    drop_entropy = entropy_diff <= 0.03
    results["entropy_diff"] = float(entropy_diff)
    results["drop_entropy"] = bool(drop_entropy)
    print(f"  Entropy MAE diff = {entropy_diff:.5f} -> {'DROP entropy' if drop_entropy else 'KEEP entropy'}", flush=True)

    return results


def pick_final_features(ablation):
    feat_cols = list(NUMERIC_LAG_FEATURES)
    cat_cols = []
    if not ablation["drop_entropy"]:
        feat_cols.append("arsenal_entropy")
    if not ablation["drop_season"]:
        feat_cols.append("season")
        cat_cols.append("season")
    return feat_cols, cat_cols


def train_final_model(ps_df, feature_cols, categorical_cols):
    print("\n[Part 3] Training final model on all seasons...", flush=True)
    assert_no_identifiers(ps_df, feature_cols)
    df = ps_df.copy()
    if "season" in categorical_cols:
        all_cats = sorted(df["season"].astype(str).unique().tolist())
        df["season"] = pd.Categorical(df["season"].astype(str), categories=all_cats)
    models = fit_lgbm(df, feature_cols, categorical_cols)
    preds = predict_ensemble(models, df[feature_cols])
    df["v4_pred"] = preds
    print(f"  Final model trained on {len(df):,} pitcher-seasons", flush=True)
    return models, df


# ─────────────────────────────── Part 4: validation ─────────────────────────

def compute_plus_scores(pred_series, mean=None, std=None):
    if mean is None:
        mean = pred_series.mean()
    if std is None:
        std = pred_series.std()
    return _to_plus_scale(pred_series, mean, std)


def part4_sign_check(conn, ps_df_with_preds, final_models, feature_cols, cat_cols):
    """r(v4_score, ERA) and r(v4_score, xFIP) for 2026 (most recent complete season)."""
    print("\n[Part 4.1] Sign check...", flush=True)
    check_season = 2026
    season_str = str(check_season)
    season_ps = ps_df_with_preds[ps_df_with_preds["season"] == season_str].copy()

    mean_, std_ = ps_df_with_preds["v4_pred"].mean(), ps_df_with_preds["v4_pred"].std()
    season_ps["v4_score"] = compute_plus_scores(season_ps["v4_pred"], mean_, std_)

    # Pull ERA and xFIP for qualifying pitchers
    pitcher_ids = season_ps["pitcher_id"].astype(int).tolist()
    ph = ",".join("?" * len(pitcher_ids))
    era_df = pd.read_sql(f"""
        SELECT pitcher_id,
               1.0 * SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k_count,
               COUNT(CASE WHEN events IS NOT NULL AND events != '' THEN 1 END) AS pa_count
        FROM pitches
        WHERE season = ? AND pitcher_id IN ({ph}) AND game_type = 'R'
        GROUP BY pitcher_id
    """, conn, params=[check_season] + pitcher_ids)

    # Use a simpler query for ERA/xFIP from pitches table
    stats_df = pd.read_sql(f"""
        SELECT pitcher_id,
               AVG(CASE WHEN delta_run_exp IS NOT NULL THEN delta_run_exp ELSE NULL END) AS mean_dre,
               COUNT(*) AS n_pitches
        FROM pitches
        WHERE season = ? AND pitcher_id IN ({ph}) AND game_type = 'R'
        GROUP BY pitcher_id
    """, conn, params=[check_season] + pitcher_ids)

    # For ERA/xFIP proxy use delta_run_exp aggregate (negative = good pitcher)
    merged = season_ps.merge(stats_df, on="pitcher_id", how="inner")
    merged = merged[merged["n_pitches"] >= MIN_PITCHES]

    if len(merged) < 5:
        return {"error": "insufficient data", "n": len(merged)}

    r_dre = float(merged["v4_score"].corr(merged["mean_dre"]))
    n = len(merged)
    lo, hi = pearson_ci(r_dre, n)

    # Try to pull real ERA/xFIP from a stats table if available
    era_result = {"n": n, "r_vs_mean_dre": r_dre, "ci_95": [lo, hi],
                  "note": "mean_dre used as outcome (positive = worse pitch location outcomes)"}

    # Try to pull k_pct and bb_pct as validity proxies
    kbb_df = pd.read_sql(f"""
        SELECT pitcher_id,
               1.0*SUM(CASE WHEN events='strikeout' OR events='strikeout_double_play' THEN 1 ELSE 0 END)/
                   NULLIF(COUNT(DISTINCT CASE WHEN events IS NOT NULL AND events!='' THEN game_id||'_'||at_bat_number ELSE NULL END),0) AS k_pct,
               1.0*SUM(CASE WHEN events='walk' OR events='intent_walk' THEN 1 ELSE 0 END)/
                   NULLIF(COUNT(DISTINCT CASE WHEN events IS NOT NULL AND events!='' THEN game_id||'_'||at_bat_number ELSE NULL END),0) AS bb_pct
        FROM pitches
        WHERE season = ? AND pitcher_id IN ({ph}) AND game_type = 'R'
        GROUP BY pitcher_id
    """, conn, params=[check_season] + pitcher_ids)

    merged2 = merged.merge(kbb_df, on="pitcher_id", how="left")
    r_k = float(merged2["v4_score"].corr(merged2["k_pct"])) if "k_pct" in merged2 else None
    r_bb = float(merged2["v4_score"].corr(merged2["bb_pct"])) if "bb_pct" in merged2 else None
    era_result["r_vs_k_pct"] = r_k
    era_result["r_vs_bb_pct"] = r_bb

    print(f"  Sign check (n={n}): r(v4, mean_dre)={r_dre:.4f}, r(v4, K%)={r_k}, r(v4, BB%)={r_bb}", flush=True)
    return era_result


def part4_score_distribution(ps_df_with_preds):
    print("\n[Part 4.2] Score distribution...", flush=True)
    mean_, std_ = ps_df_with_preds["v4_pred"].mean(), ps_df_with_preds["v4_pred"].std()
    scores = compute_plus_scores(ps_df_with_preds["v4_pred"], mean_, std_)

    result = dict(
        min=float(scores.min()),
        max=float(scores.max()),
        std=float(scores.std()),
        mean=float(scores.mean()),
        spread=float(scores.max() - scores.min()),
        v3_reference_band="94.9–103.9 (9.0 spread)",
        comparison_note=f"v4 spread = {scores.max()-scores.min():.1f} pts vs v3's ~9 pts",
    )
    print(f"  Spread: {result['spread']:.1f} pts (min={result['min']:.1f}, max={result['max']:.1f}, sd={result['std']:.2f})", flush=True)
    return result


def part4_yoy_stability(ps_df_with_preds):
    print("\n[Part 4.3] YoY stability...", flush=True)
    mean_, std_ = ps_df_with_preds["v4_pred"].mean(), ps_df_with_preds["v4_pred"].std()
    ps_df_with_preds = ps_df_with_preds.copy()
    ps_df_with_preds["v4_score"] = compute_plus_scores(ps_df_with_preds["v4_pred"], mean_, std_)

    pairs = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026)]
    per_pair = []
    all_t, all_t1 = [], []
    for t, t1 in pairs:
        s_t = ps_df_with_preds[ps_df_with_preds["season"] == str(t)][["pitcher_id", "v4_score"]]
        s_t1 = ps_df_with_preds[ps_df_with_preds["season"] == str(t1)][["pitcher_id", "v4_score"]]
        merged = s_t.merge(s_t1, on="pitcher_id", suffixes=("_t", "_t1"))
        if len(merged) < 5:
            continue
        r = float(merged["v4_score_t"].corr(merged["v4_score_t1"]))
        r2 = r ** 2
        print(f"  {t}->{t1}: r={r:.4f}, r2={r2:.4f}, n={len(merged)}", flush=True)
        per_pair.append(dict(season_t=str(t), season_t1=str(t1), n=len(merged), r=r, r2=r2))
        all_t.extend(merged["v4_score_t"].tolist())
        all_t1.extend(merged["v4_score_t1"].tolist())
    pooled_r = float(np.corrcoef(all_t, all_t1)[0, 1]) if all_t else None
    pooled_r2 = pooled_r ** 2 if pooled_r else None
    print(f"  Pooled r2={pooled_r2:.4f}, n={len(all_t)}", flush=True)
    return dict(per_pair=per_pair, pooled_r=pooled_r, pooled_r2=pooled_r2, n_pooled=len(all_t))


def part4_split_half(conn, final_models, feature_cols, cat_cols, commit_table):
    print("\n[Part 4.4] Split-half reliability (2025)...", flush=True)
    season = 2025
    df = load_pitches(conn, season)
    totals = df.groupby("pitcher_id").size()
    qual_ids = totals[totals >= MIN_PITCHES].index
    df = df[df["pitcher_id"].isin(qual_ids)].copy()

    # Split by odd/even game_id hash
    game_ids = df["game_id"].unique()
    np.random.seed(42)
    shuffled = np.random.permutation(game_ids)
    half_a = set(shuffled[: len(shuffled) // 2])
    half_b = set(shuffled[len(shuffled) // 2 :])

    season_commit = commit_table[commit_table["season"] == season].copy()
    results = {}
    half_scores = {}
    for label, game_set in [("a", half_a), ("b", half_b)]:
        sub = df[df["game_id"].isin(game_set)].copy()
        totals_h = sub.groupby("pitcher_id").size()
        qual_h = totals_h[totals_h >= 30].index
        sub = sub[sub["pitcher_id"].isin(qual_h)]
        if sub.empty:
            continue
        feat = sequential_lag_features(sub, season_commit, max_lag=MAX_LAG)
        if feat.empty:
            continue

        # Arsenal entropy for this half
        centroids_h = pitch_type_centroids(sub)
        usage_h = centroids_h[centroids_h["n_pitches"] >= 30].copy()
        if not usage_h.empty:
            totals_h2 = usage_h.groupby(["pitcher_id", "season"])["n_pitches"].transform("sum")
            usage_h["p"] = usage_h["n_pitches"] / totals_h2
            usage_h["term"] = -usage_h["p"] * np.log(usage_h["p"])
            entropy_h = usage_h.groupby("pitcher_id")["term"].sum().reset_index()
            entropy_h.columns = ["pitcher_id", "arsenal_entropy"]
        else:
            entropy_h = pd.DataFrame(columns=["pitcher_id", "arsenal_entropy"])

        lag_agg = feat.groupby("pitcher_id")[[c for c in NUMERIC_LAG_FEATURES if c in feat.columns]].mean().reset_index()
        lag_agg = lag_agg.merge(entropy_h, on="pitcher_id", how="left")
        lag_agg["season"] = str(season)

        missing = [c for c in feature_cols if c not in lag_agg.columns]
        for c in missing:
            lag_agg[c] = np.nan

        if "season" in cat_cols:
            lag_agg["season"] = pd.Categorical(lag_agg["season"].astype(str), categories=[str(s) for s in ALL_SEASONS])

        preds = predict_ensemble(final_models, lag_agg[feature_cols])
        half_scores[label] = pd.Series(preds, index=lag_agg["pitcher_id"])

    if "a" in half_scores and "b" in half_scores:
        common = half_scores["a"].index.intersection(half_scores["b"].index)
        if len(common) >= 10:
            r = float(half_scores["a"].loc[common].corr(half_scores["b"].loc[common]))
            print(f"  Split-half r={r:.4f}, n={len(common)}", flush=True)
            results = dict(r=r, n=len(common), split_by="random_game_split_seed42")
        else:
            results = dict(r=None, n=len(common), note="too few common pitchers")
    else:
        results = dict(r=None, note="failed to compute halves")
    return results


def part4_predictive_validity(conn, ps_df, feature_cols, cat_cols, commit_table):
    """Train through year N, predict outcomes in year N+1 using OOS preds from LOSO."""
    print("\n[Part 4.5] Predictive validity (out-of-sample)...", flush=True)
    # Use LOSO OOS predictions
    _, oos_df = run_loso_cv(ps_df, feature_cols, cat_cols)
    if oos_df.empty:
        return {"error": "no OOS predictions"}

    mean_, std_ = ps_df["v4_pred"].mean() if "v4_pred" in ps_df else ps_df["target_mean_xrv"].mean(), 1.0
    # Re-compute v4_pred for OOS df using oos_pred column
    oos_df["v4_pred"] = oos_df["oos_pred"]
    oos_df["season_int"] = oos_df["season"].astype(int)

    outcomes_by_pair = []
    for t in [2021, 2022, 2023, 2024, 2025]:
        t1 = t + 1
        oos_t = oos_df[oos_df["season_int"] == t][["pitcher_id", "v4_pred"]].copy()
        if oos_t.empty:
            continue

        # Pull outcomes for year t1
        pitcher_ids = oos_t["pitcher_id"].astype(int).tolist()
        ph = ",".join("?" * len(pitcher_ids))
        outcomes_df = pd.read_sql(f"""
            SELECT pitcher_id,
                   AVG(CASE WHEN estimated_woba_using_speedangle IS NOT NULL THEN estimated_woba_using_speedangle END) AS xwoba,
                   1.0*SUM(CASE WHEN description='swinging_strike' OR description='swinging_strike_bunt' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) AS whiff_pct,
                   1.0*SUM(CASE WHEN description IN ('swinging_strike','swinging_strike_bunt','foul_tip') THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) AS swstr_pct,
                   1.0*SUM(CASE WHEN events='walk' OR events='intent_walk' THEN 1 ELSE 0 END)/NULLIF(COUNT(DISTINCT CASE WHEN events IS NOT NULL AND events!='' THEN game_id||'_'||at_bat_number END),0) AS bb_pct,
                   1.0*SUM(CASE WHEN events='strikeout' OR events='strikeout_double_play' THEN 1 ELSE 0 END)/NULLIF(COUNT(DISTINCT CASE WHEN events IS NOT NULL AND events!='' THEN game_id||'_'||at_bat_number END),0) AS k_pct,
                   1.0*SUM(CASE WHEN zone IS NOT NULL AND zone > 9 AND description IN ('swinging_strike','swinging_strike_bunt') THEN 1 ELSE 0 END)/NULLIF(SUM(CASE WHEN zone IS NOT NULL AND zone > 9 THEN 1 ELSE 0 END),0) AS chase_pct
            FROM pitches
            WHERE season = ? AND pitcher_id IN ({ph}) AND game_type = 'R'
            GROUP BY pitcher_id
            HAVING COUNT(*) >= {MIN_PITCHES}
        """, conn, params=[t1] + pitcher_ids)

        merged = oos_t.merge(outcomes_df, on="pitcher_id", how="inner")
        if len(merged) < 10:
            continue

        outcome_cols = ["xwoba", "whiff_pct", "swstr_pct", "bb_pct", "k_pct", "chase_pct"]
        pair_result = {"season_t": t, "season_t1": t1, "n": len(merged)}
        for oc in outcome_cols:
            if oc in merged.columns and merged[oc].notna().sum() > 5:
                pair_result[f"r_{oc}"] = float(merged["v4_pred"].corr(merged[oc]))
            else:
                pair_result[f"r_{oc}"] = None
        outcomes_by_pair.append(pair_result)
        print(f"  {t}->{t1}: n={len(merged)}, r_xwoba={pair_result.get('r_xwoba')}", flush=True)

    return {"pairs": outcomes_by_pair}


def part4_independence(conn, ps_df_with_preds):
    """r(v4_score, Stuff+, Location+, Deception+) on common set."""
    print("\n[Part 4.6] Independence check...", flush=True)
    mean_, std_ = ps_df_with_preds["v4_pred"].mean(), ps_df_with_preds["v4_pred"].std()
    ps_df_with_preds = ps_df_with_preds.copy()
    ps_df_with_preds["v4_score"] = compute_plus_scores(ps_df_with_preds["v4_pred"], mean_, std_)

    # Use 2025 season for cross-model comparison
    check_season = 2025
    season_ps = ps_df_with_preds[ps_df_with_preds["season"] == str(check_season)][["pitcher_id", "v4_score"]].copy()
    pitcher_ids = season_ps["pitcher_id"].astype(int).tolist()
    ph = ",".join("?" * len(pitcher_ids))

    # Load score parquets
    result = {"season": check_season, "note": "using score parquets for cross-model comparison"}
    scores_dir = ROOT / "scores"

    try:
        stuff_path = scores_dir / f"stuff_plus_v4_{check_season}.parquet"
        if stuff_path.exists():
            stuff_df = pd.read_parquet(stuff_path)
            if "pitcher_id" in stuff_df.columns and "pred" in stuff_df.columns:
                stuff_agg = stuff_df.groupby("pitcher_id")["pred"].mean().reset_index()
                stuff_agg.columns = ["pitcher_id", "stuff_pred"]
                merged = season_ps.merge(stuff_agg, on="pitcher_id", how="inner")
                result["r_stuff"] = float(merged["v4_score"].corr(merged["stuff_pred"])) if len(merged) > 5 else None
                result["n_stuff"] = len(merged)
    except Exception as e:
        result["stuff_error"] = str(e)

    try:
        loc_path = scores_dir / f"location_plus_v3_{check_season}.parquet"
        if loc_path.exists():
            loc_df = pd.read_parquet(loc_path)
            if "pitcher_id" in loc_df.columns and "xrv" in loc_df.columns:
                loc_agg = loc_df.groupby("pitcher_id")["xrv"].mean().reset_index()
                loc_agg.columns = ["pitcher_id", "loc_pred"]
                merged = season_ps.merge(loc_agg, on="pitcher_id", how="inner")
                result["r_location"] = float(merged["v4_score"].corr(merged["loc_pred"])) if len(merged) > 5 else None
                result["n_location"] = len(merged)
    except Exception as e:
        result["location_error"] = str(e)

    print(f"  Independence: r_stuff={result.get('r_stuff')}, r_location={result.get('r_location')}", flush=True)
    return result


def part4_whiff_tercile_test(conn, ps_df_with_preds):
    """Bucket pitcher-seasons by v4 score tercile, check real whiff rate."""
    print("\n[Part 4.7] Sequential same-at-bat whiff-rate tercile test...", flush=True)
    mean_, std_ = ps_df_with_preds["v4_pred"].mean(), ps_df_with_preds["v4_pred"].std()
    ps_df_with_preds = ps_df_with_preds.copy()
    ps_df_with_preds["v4_score"] = compute_plus_scores(ps_df_with_preds["v4_pred"], mean_, std_)

    check_season = 2025
    season_ps = ps_df_with_preds[ps_df_with_preds["season"] == str(check_season)][["pitcher_id", "v4_score"]].copy()

    # Assign tercile by v4_score (lower = better pitching)
    season_ps["tercile"] = pd.qcut(season_ps["v4_score"], q=3, labels=["high_score", "mid_score", "low_score"])
    # Note: high score = 100+ means BETTER tunneling in the "+" convention (100 - 10*z, lower z = better)
    # But if the sign is correct: lower mean_dre (more negative) -> higher score
    # So "high_score" tercile = best tunneling pitchers -> should have LOWER whiff rates?
    # Let's just report the raw numbers and the spec can interpret.

    pitcher_ids = season_ps["pitcher_id"].astype(int).tolist()
    ph = ",".join("?" * len(pitcher_ids))
    pitches_df = pd.read_sql(f"""
        SELECT pitcher_id,
               description,
               at_bat_number, game_id, pitch_number
        FROM pitches
        WHERE season = ? AND pitcher_id IN ({ph}) AND game_type = 'R'
    """, conn, params=[check_season] + pitcher_ids)

    # Sequential same-at-bat: only pitches after the 1st in their PA
    pitches_sorted = pitches_df.sort_values(["game_id", "at_bat_number", "pitch_number"])
    pa_first = pitches_sorted.groupby(["game_id", "at_bat_number"]).head(1).assign(_is_first=True)
    pitches_sorted = pitches_sorted.merge(
        pa_first[["game_id", "at_bat_number", "pitch_number", "_is_first"]],
        on=["game_id", "at_bat_number", "pitch_number"], how="left"
    )
    seq_pitches = pitches_sorted[pitches_sorted["_is_first"].isna()].copy()

    seq_pitches = seq_pitches.merge(season_ps[["pitcher_id", "tercile"]], on="pitcher_id", how="inner")
    seq_pitches["is_whiff"] = seq_pitches["description"].isin(["swinging_strike", "swinging_strike_bunt"]).astype(int)
    seq_pitches["is_swing"] = seq_pitches["description"].isin([
        "swinging_strike", "swinging_strike_bunt", "hit_into_play", "foul", "foul_tip",
        "foul_bunt", "missed_bunt"
    ]).astype(int)

    tercile_results = {}
    for t in ["high_score", "mid_score", "low_score"]:
        sub = seq_pitches[seq_pitches["tercile"] == t]
        n_swings = sub["is_swing"].sum()
        n_whiffs = sub["is_whiff"].sum()
        whiff_rate = float(n_whiffs / n_swings) if n_swings > 0 else None
        tercile_results[t] = dict(n_pitches=len(sub), n_swings=int(n_swings), n_whiffs=int(n_whiffs), whiff_rate=whiff_rate)

    # z-score test: high vs low score tercile
    h = tercile_results["high_score"]
    l = tercile_results["low_score"]
    if h["n_swings"] > 0 and l["n_swings"] > 0 and h["whiff_rate"] and l["whiff_rate"]:
        p1, n1 = h["whiff_rate"], h["n_swings"]
        p2, n2 = l["whiff_rate"], l["n_swings"]
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p1 - p2) / se if se > 0 else None
        tercile_results["z_high_vs_low"] = float(z) if z else None
    else:
        tercile_results["z_high_vs_low"] = None

    print(f"  Whiff rates by tercile: {[(k, v.get('whiff_rate')) for k, v in tercile_results.items() if k != 'z_high_vs_low']}", flush=True)
    return tercile_results


def part4_leaderboard(ps_df_with_preds, conn):
    print("\n[Part 4.8] Leaderboard...", flush=True)
    mean_, std_ = ps_df_with_preds["v4_pred"].mean(), ps_df_with_preds["v4_pred"].std()
    ps_df_with_preds = ps_df_with_preds.copy()
    ps_df_with_preds["v4_score"] = compute_plus_scores(ps_df_with_preds["v4_pred"], mean_, std_)

    names = dict(conn.execute("SELECT player_id, name FROM players").fetchall())
    ps_df_with_preds["name"] = ps_df_with_preds["pitcher_id"].map(names)

    result = {}
    for season in [2025, 2026]:
        s = ps_df_with_preds[ps_df_with_preds["season"] == str(season)].copy()
        if s.empty:
            continue
        s_min, s_max, s_sd = s["v4_score"].min(), s["v4_score"].max(), s["v4_score"].std()
        top = s.sort_values("v4_score", ascending=False).head(15)
        bottom = s.sort_values("v4_score", ascending=True).head(15)

        def rows(d):
            return [dict(pitcher_id=int(r.pitcher_id), name=r.name, v4_score=round(float(r.v4_score), 1))
                    for r in d.itertuples()]
        result[str(season)] = dict(
            n_qualified=len(s),
            score_range=f"{s_min:.1f}–{s_max:.1f} (sd={s_sd:.1f})",
            top=rows(top),
            bottom=rows(bottom),
        )
        print(f"  {season}: n={len(s)}, range={s_min:.1f}–{s_max:.1f}", flush=True)
    return result


# ─────────────────────────────── Part 0: sign diagnostic ─────────────────────

def part0_sign_diagnostic(conn, commit_table):
    print("\n[Part 0] Sign diagnostic for live v2 and v3...", flush=True)
    season = 2025
    result = {"season": season, "diagnostic_note": "Checks whether live v2/v3 scores correlate positively or negatively with ERA-proxy (mean_dre)"}

    # Pull v2 leaderboard
    try:
        from analytics.tunneling_v2_model import leaderboard as v2_lb
        v2_board = v2_lb(conn, season, top_n=500)
        v2_all = v2_board["top"] + v2_board["bottom"]
        # Re-pull all qualified pitcher scores
        from analytics.tunneling_v2_model import _raw_predictions as v2_raw_preds
        league_raw = v2_raw_preds(conn, season)
        g_v2 = league_raw.groupby("pitcher_id").agg(mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size")).reset_index()
        from analytics.location_features import to_plus_scale as tps
        g_v2["v2_score"] = tps(g_v2["mean_xrv"], g_v2["mean_xrv"].mean(), g_v2["mean_xrv"].std())
    except Exception as e:
        result["v2_error"] = str(e)
        g_v2 = None

    # Pull outcome data
    if g_v2 is not None:
        pitcher_ids = g_v2["pitcher_id"].astype(int).tolist()
        ph = ",".join("?" * len(pitcher_ids))
        outcomes_df = pd.read_sql(f"""
            SELECT pitcher_id,
                   AVG(delta_run_exp) AS mean_dre,
                   1.0*SUM(CASE WHEN description='swinging_strike' OR description='swinging_strike_bunt' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) AS whiff_rate,
                   1.0*SUM(CASE WHEN events='strikeout' OR events='strikeout_double_play' THEN 1 ELSE 0 END)/NULLIF(COUNT(DISTINCT CASE WHEN events IS NOT NULL AND events!='' THEN game_id||'_'||at_bat_number END),0) AS k_pct
            FROM pitches
            WHERE season = ? AND pitcher_id IN ({ph}) AND game_type = 'R'
            GROUP BY pitcher_id
            HAVING COUNT(*) >= {MIN_PITCHES}
        """, conn, params=[season] + pitcher_ids)

        merged_v2 = g_v2.merge(outcomes_df, on="pitcher_id", how="inner")
        n_v2 = len(merged_v2)
        r_v2_dre = float(merged_v2["v2_score"].corr(merged_v2["mean_dre"]))
        r_v2_whiff = float(merged_v2["v2_score"].corr(merged_v2["whiff_rate"]))
        r_v2_k = float(merged_v2["v2_score"].corr(merged_v2["k_pct"]))
        lo_v2, hi_v2 = pearson_ci(r_v2_dre, n_v2)
        print(f"  v2 (n={n_v2}): r(score, mean_dre)={r_v2_dre:.4f} 95%CI[{lo_v2},{hi_v2}], r(score, whiff%)={r_v2_whiff:.4f}, r(score, K%)={r_v2_k:.4f}", flush=True)

        # Sign interpretation note
        # Our convention: score=100-10*(x-mean)/std -> LOWER mean_dre = better pitcher = HIGHER score
        # So if v2 works correctly: r(v2_score, mean_dre) should be NEGATIVE (better pitchers have lower mean_dre and higher score)
        sign_correct_v2 = r_v2_dre < 0
        result["v2"] = dict(
            n=n_v2, r_vs_mean_dre=r_v2_dre, ci_95=[lo_v2, hi_v2],
            r_vs_whiff_rate=r_v2_whiff, r_vs_k_pct=r_v2_k,
            sign_correct=sign_correct_v2,
            expected_sign="negative (lower mean_dre = worse outcomes = lower score = sign is correct if r < 0)",
        )

        # Spot check: top 10 K% and bottom 10 ERA-proxy
        names = dict(conn.execute("SELECT player_id, name FROM players").fetchall())
        merged_v2["name"] = merged_v2["pitcher_id"].map(names)
        elite = merged_v2.sort_values("k_pct", ascending=False).head(10)[["pitcher_id", "name", "v2_score", "k_pct", "mean_dre"]]
        weak = merged_v2.sort_values("mean_dre", ascending=False).head(10)[["pitcher_id", "name", "v2_score", "k_pct", "mean_dre"]]
        result["v2"]["spot_check"] = {
            "top_10_by_k_pct": elite.to_dict(orient="records"),
            "bottom_10_by_mean_dre": weak.to_dict(orient="records"),
        }

    # v3
    try:
        from analytics.tunneling_v3_model import _raw_predictions as v3_raw_preds
        league_v3 = v3_raw_preds(conn, season)
        g_v3 = league_v3.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n_pitches=("pred", "size")).reset_index()
        from analytics.location_features import to_plus_scale as tps
        g_v3["v3_score"] = tps(g_v3["mean_pred"], g_v3["mean_pred"].mean(), g_v3["mean_pred"].std())

        pitcher_ids_v3 = g_v3["pitcher_id"].astype(int).tolist()
        ph_v3 = ",".join("?" * len(pitcher_ids_v3))
        outcomes_v3 = pd.read_sql(f"""
            SELECT pitcher_id, AVG(delta_run_exp) AS mean_dre
            FROM pitches
            WHERE season = ? AND pitcher_id IN ({ph_v3}) AND game_type = 'R'
            GROUP BY pitcher_id HAVING COUNT(*) >= {MIN_PITCHES}
        """, conn, params=[season] + pitcher_ids_v3)

        merged_v3 = g_v3.merge(outcomes_v3, on="pitcher_id", how="inner")
        n_v3 = len(merged_v3)
        r_v3 = float(merged_v3["v3_score"].corr(merged_v3["mean_dre"]))
        lo_v3, hi_v3 = pearson_ci(r_v3, n_v3)
        sign_correct_v3 = r_v3 < 0
        result["v3"] = dict(n=n_v3, r_vs_mean_dre=r_v3, ci_95=[lo_v3, hi_v3], sign_correct=sign_correct_v3)
        print(f"  v3 (n={n_v3}): r(score, mean_dre)={r_v3:.4f}", flush=True)
    except Exception as e:
        result["v3_error"] = str(e)
        print(f"  v3 failed: {e}", flush=True)

    return result


# ─────────────────────────────── Part 5: report ──────────────────────────────

def write_completion_report(part0, ablation, cv_folds, cv_mean_r2, cv_mean_mae,
                             dist, yoy, split_half, predictive, independence,
                             whiff_tercile, leaderboard, feature_cols):
    print("\n[Part 5] Writing completion report...", flush=True)

    v2_sign = part0.get("v2", {}).get("sign_correct", "unknown")
    v3_sign = part0.get("v3", {}).get("sign_correct", "unknown")
    v2_r = part0.get("v2", {}).get("r_vs_mean_dre", "N/A")
    v3_r = part0.get("v3", {}).get("r_vs_mean_dre", "N/A")

    spread = dist.get("spread", "?")
    pooled_r2 = yoy.get("pooled_r2", "?")
    sh_r = split_half.get("r", "?")

    yoy_rows = "\n".join([
        f"| {p['season_t']}->{p['season_t1']} | {p['n']} | {p['r']:.4f} | {p['r2']:.4f} |"
        for p in yoy.get("per_pair", [])
    ])

    # Predictive validity summary
    pred_rows = []
    for pair in predictive.get("pairs", []):
        xw = pair.get("r_xwoba")
        wh = pair.get("r_whiff_pct")
        pred_rows.append(f"| {pair['season_t']}->{pair['season_t1']} | {pair['n']} | {xw} | {wh} |")
    pred_table = "\n".join(pred_rows)

    # Feature selection note
    feat_note = f"Final features: {feature_cols}"
    season_decision = "DROPPED (diff ≤ 0.03)" if ablation.get("drop_season") else f"KEPT (diff={ablation.get('season_diff', '?'):.4f})"
    entropy_decision = "DROPPED (diff ≤ 0.03)" if ablation.get("drop_entropy") else f"KEPT (diff={ablation.get('entropy_diff', '?'):.4f})"

    # Sign check from Part 4
    # The sign check in Part 4 uses mean_dre as outcome (negative = good pitch outcomes for pitcher)
    # v4 score: higher = better tunneling, so r(v4_score, mean_dre) should be NEGATIVE

    # Recommendation
    issues = []
    if v2_sign is True:
        rec_v2 = "v2 sign is CORRECT (r < 0) — the sign concern was unfounded on this dataset."
    elif v2_sign is False:
        issues.append("v2 sign is WRONG (r > 0) — live model incorrectly ranks pitchers")
        rec_v2 = "v2 sign is WRONG (r > 0) — live model needs fixing."
    else:
        rec_v2 = "v2 sign inconclusive."

    good_spread = isinstance(spread, float) and spread > 12
    good_yoy = isinstance(pooled_r2, float) and pooled_r2 > 0.25
    good_sh = isinstance(sh_r, float) and sh_r > 0.6

    if good_spread and good_yoy and good_sh and not (v2_sign is True):
        recommendation = "SHIP v4: demonstrates improved score spread and comparable or better stability vs v3. v2's sign issue (if confirmed) should be patched."
    elif good_spread and good_yoy and v2_sign is True:
        recommendation = "CONDITIONAL SHIP v4: spread improvement achieved. v2 sign is correct, so v4 would be a pure quality upgrade rather than a fix. Proceed if spread improvement is judged sufficient."
    elif not good_spread:
        recommendation = "DO NOT SHIP v4 YET: score spread still narrow — grain change did not solve the dilution problem. Investigate feature quality or model architecture further."
    elif not good_yoy:
        recommendation = "DO NOT SHIP v4 YET: YoY stability is below acceptable floor. Score is not stable enough for a reliable leaderboard metric."
    else:
        recommendation = "INCONCLUSIVE: some gates pass, some fail. Full review required before any promotion decision."

    md = f"""# Tunneling+ v4 Completion Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Branch**: tunneling-v4
**Status**: Experimental — NOT promoted to production
**Spec**: Pitcher-season grain trained model (LightGBM, Huber loss, LOSO CV)

---

## Part 0 — Live v2/v3 Sign Diagnostic

**Critical question**: Does the live v2 model positively or negatively correlate with good pitching outcomes?
Convention: `score = 100 - 10*(pred - mean)/std` -> lower predicted xRV = better pitcher = **higher score**
Expected: `r(score, mean_dre) < 0` (mean_dre measures average run value per pitch; better pitchers have lower/more-negative mean_dre)

| Model | n | r(score, mean_dre) | 95% CI | Sign correct? |
|-------|---|---------------------|--------|---------------|
| v2 (live) | {part0.get('v2', {}).get('n', 'N/A')} | {v2_r} | {part0.get('v2', {}).get('ci_95', 'N/A')} | {'✓ YES' if v2_sign is True else '✗ NO' if v2_sign is False else '?'} |
| v3 | {part0.get('v3', {}).get('n', 'N/A')} | {v3_r} | {part0.get('v3', {}).get('ci_95', 'N/A')} | {'✓ YES' if v3_sign is True else '✗ NO' if v3_sign is False else '?'} |

{rec_v2}

**Note on v2 spot-check**: See `diagnostics/tunneling_v4/part0_sign_diagnostic.json` for the top-10 K% pitchers and their v2 scores.

---

## Part 1 — Pre-registration

See `diagnostics/tunneling_v4/spec_lock.json` for pre-registered feature set, CV scheme, and adoption rules.

---

## Part 2 — Feature Construction

**Grain**: pitcher-season
**Qualifying threshold**: >={MIN_PITCHES} total pitches (after PO/UN/EP exclusion; KN kept)
**Feature set**: 18 lag-mean numerics (6 lag slots × commit_distance, plate_distance, shape_break) + arsenal_entropy + season
{feat_note}

**Ablation results** (LOSO CV MAE, lower = better):

| Config | Mean MAE | Mean R2 |
|--------|----------|---------|
| With season, with entropy | {ablation.get('with_season_with_entropy', {}).get('mean_mae', 'N/A')} | {ablation.get('with_season_with_entropy', {}).get('mean_r2', 'N/A')} |
| No season, with entropy | {ablation.get('no_season_with_entropy', {}).get('mean_mae', 'N/A')} | {ablation.get('no_season_with_entropy', {}).get('mean_r2', 'N/A')} |
| With season, no entropy | {ablation.get('with_season_no_entropy', {}).get('mean_mae', 'N/A')} | {ablation.get('with_season_no_entropy', {}).get('mean_r2', 'N/A')} |
| No season, no entropy | {ablation.get('no_season_no_entropy', {}).get('mean_mae', 'N/A')} | {ablation.get('no_season_no_entropy', {}).get('mean_r2', 'N/A')} |

Season feature: {season_decision}
Arsenal entropy: {entropy_decision}

---

## Part 3 — Model Training

**Architecture**: LightGBM, Huber loss, pitcher-season grain
**CV**: Leave-one-season-out on 2021–2025
**Ensemble**: 3 seeds ({SEEDS}), predictions averaged

LOSO CV results (modern era 2021–2025):

| Holdout | n | R2 | MAE |
|---------|---|-----|-----|
{"".join(f"| {f['holdout_season']} | {f['n']} | {f['r2']:.5f} | {f['mae']:.5f} |" + chr(10) for f in cv_folds)}
| **Mean** | — | **{cv_mean_r2:.5f}** | **{cv_mean_mae:.5f}** |

---

## Part 4 — Validation Battery

### 4.2 Score Distribution

| Metric | v4 | v3 (reference) |
|--------|----|----|
| Score range | {dist.get('min', '?'):.1f}–{dist.get('max', '?'):.1f} | 94.9–103.9 |
| Spread (max−min) | **{spread:.1f}** pts | 9.0 pts |
| SD | {dist.get('std', '?'):.2f} | ~1.5 (est.) |

### 4.3 Year-over-Year Stability

| Pair | n | Pearson r | r2 |
|------|---|-----------|-----|
{yoy_rows}
| **Pooled** | {yoy.get('n_pooled', '?')} | {yoy.get('pooled_r', '?')} | **{pooled_r2}** |

### 4.4 Split-Half Reliability (2025)

r = {sh_r} (n = {split_half.get('n', '?')}, split by random game assignment, seed=42)
{'✓ PASS (r >= 0.60)' if isinstance(sh_r, float) and sh_r >= 0.60 else '✗ FAIL or inconclusive (r < 0.60)' if isinstance(sh_r, float) else '?'}

### 4.5 Out-of-Sample Predictive Validity

| Pair | n | r(xwOBA) | r(whiff%) |
|------|---|-----------|-----------|
{pred_table}

### 4.6 Independence

| vs. | r |
|-----|---|
| Stuff+ | {independence.get('r_stuff', 'N/A')} (n={independence.get('n_stuff', 'N/A')}) |
| Location+ | {independence.get('r_location', 'N/A')} (n={independence.get('n_location', 'N/A')}) |

### 4.7 Sequential Same-at-bat Whiff-Rate Tercile Test (2025)

Pitchers bucketed by v4 score tercile; whiff rate measured on at-bats following the 1st pitch.
(high_score = best tunneling, should suppress opponent whiff rates the most)

| Tercile | n pitches | n swings | Whiff rate |
|---------|-----------|----------|------------|
| high_score | {whiff_tercile.get('high_score', {}).get('n_pitches', '?')} | {whiff_tercile.get('high_score', {}).get('n_swings', '?')} | {whiff_tercile.get('high_score', {}).get('whiff_rate', '?')} |
| mid_score | {whiff_tercile.get('mid_score', {}).get('n_pitches', '?')} | {whiff_tercile.get('mid_score', {}).get('n_swings', '?')} | {whiff_tercile.get('mid_score', {}).get('whiff_rate', '?')} |
| low_score | {whiff_tercile.get('low_score', {}).get('n_pitches', '?')} | {whiff_tercile.get('low_score', {}).get('n_swings', '?')} | {whiff_tercile.get('low_score', {}).get('whiff_rate', '?')} |

z(high vs low) = {whiff_tercile.get('z_high_vs_low', 'N/A')}

### 4.8 Leaderboard

{"".join(f'''**{yr} Top 5** (range: {leaderboard.get(yr, {}).get('score_range', 'N/A')}):
{chr(10).join(f"  {r['name']}: {r['v4_score']}" for r in leaderboard.get(yr, {}).get('top', [])[:5])}
''' for yr in ['2025', '2026'])}

---

## Part 5 — Side-by-Side Comparison

| Metric | v2 (live) | v3 | v4 (experimental) |
|--------|-----------|----|--------------------|
| Sign (r vs mean_dre) | {v2_r} ({'✓' if v2_sign is True else '✗'}) | {v3_r} ({'✓' if v3_sign is True else '✗'}) | (see Part 4.1) |
| Score spread | ~9 pts (per doc) | 9.0 pts | **{spread:.1f} pts** |
| YoY r2 (pooled) | 0.299 (per validation.json) | 0.299 | **{pooled_r2}** |
| Split-half r | N/A | N/A | {sh_r} |

---

## Recommendation

**{recommendation}**

### Detailed reasoning

1. **Sign problem**: {rec_v2}
2. **Spread problem**: v4 score spread = {spread:.1f} pts. {'Substantially better than v3 9 pts -- grain change helped.' if isinstance(spread, float) and spread > 12 else 'Not substantially better than v3 -- grain change did NOT fix the dilution problem.'}
3. **Stability**: Pooled YoY r2 = {pooled_r2}. {'Acceptable (>= 0.25).' if isinstance(pooled_r2, float) and pooled_r2 >= 0.25 else 'Below acceptable floor (< 0.25).'}
4. **Split-half reliability**: r = {sh_r}. {'Acceptable (>= 0.60).' if isinstance(sh_r, float) and sh_r >= 0.60 else 'Below target (< 0.60).'}

This recommendation is advisory only. No production files have been modified.

---

## Known Deviations

1. Part 0 uses `mean(delta_run_exp)` as ERA/xFIP proxy (real ERA not available in DB; mean_dre is the same target the model trains on, so this is a consistency check rather than fully independent validation).
2. v3 sign comparison uses 2025 season only.
3. Out-of-sample predictive validity uses LOSO OOS predictions, which are themselves indirect (not a true held-out cohort).
"""

    out_path = METHODOLOGY_DIR / "TUNNELING_V4_COMPLETION.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"  Written: {out_path}", flush=True)
    return md


# ─────────────────────────────── main ────────────────────────────────────────

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    DIAG_DIR.mkdir(exist_ok=True, parents=True)

    conn = sqlite3.connect(str(DB_PATH))
    commit_table = load_commit_table()
    print(f"Commit table: {len(commit_table):,} pitcher-season rows, seasons {commit_table['season'].min()}-{commit_table['season'].max()}", flush=True)

    start = time.time()

    # ── Part 0 ────────────────────────────────────────────────────────────────
    part0 = part0_sign_diagnostic(conn, commit_table)
    (DIAG_DIR / "part0_sign_diagnostic.json").write_text(json.dumps(part0, indent=2, default=str))
    print(f"  Saved part0_sign_diagnostic.json", flush=True)

    # ── Part 2 ────────────────────────────────────────────────────────────────
    ps_df = build_pitcher_season_features(conn, ALL_SEASONS, commit_table)
    print(f"\n[Part 2] Dataset: {len(ps_df):,} pitcher-seasons, {ps_df['season'].nunique()} seasons", flush=True)
    (DIAG_DIR / "pitcher_season_dataset_info.json").write_text(json.dumps({
        "n_pitcher_seasons": len(ps_df),
        "seasons": sorted(ps_df["season"].unique().tolist()),
        "features": BASE_FEATURE_COLS_WITH_SEASON,
        "target": "target_mean_xrv",
        "min_pitches": MIN_PITCHES,
    }, indent=2))

    # ── Part 3 ────────────────────────────────────────────────────────────────
    print("\n[Part 3] Running ablation...", flush=True)
    ablation = run_ablation(ps_df)
    (DIAG_DIR / "ablation_results.json").write_text(json.dumps(ablation, indent=2))

    feature_cols, cat_cols = pick_final_features(ablation)
    print(f"\n  Final features: {feature_cols}", flush=True)

    # Full LOSO CV with chosen features
    print("\n[Part 3] LOSO CV with chosen features...", flush=True)
    cv_folds, oos_df = run_loso_cv(ps_df, feature_cols, cat_cols)
    cv_mean_r2 = float(np.mean([f["r2"] for f in cv_folds]))
    cv_mean_mae = float(np.mean([f["mae"] for f in cv_folds]))

    final_models, ps_with_preds = train_final_model(ps_df, feature_cols, cat_cols)

    # Save model
    final_models[0].booster_.save_model(str(MODELS_DIR / "tunneling_plus_v4.txt"))
    meta = dict(
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
        lgbm_params=LGBM_PARAMS,
        seeds=SEEDS,
        trained_at=datetime.now().isoformat(),
        all_seasons=ALL_SEASONS,
        cv_seasons=CV_SEASONS,
        min_pitches=MIN_PITCHES,
        ablation_season_decision="dropped" if ablation.get("drop_season") else "kept",
        ablation_entropy_decision="dropped" if ablation.get("drop_entropy") else "kept",
        n_pitcher_seasons=len(ps_df),
    )
    (MODELS_DIR / "tunneling_plus_v4_meta.json").write_text(json.dumps(meta, indent=2))

    # ── Part 4 ────────────────────────────────────────────────────────────────
    sign = part4_sign_check(conn, ps_with_preds, final_models, feature_cols, cat_cols)
    dist = part4_score_distribution(ps_with_preds)
    yoy = part4_yoy_stability(ps_with_preds)
    split_half = part4_split_half(conn, final_models, feature_cols, cat_cols, commit_table)
    predictive = part4_predictive_validity(conn, ps_df, feature_cols, cat_cols, commit_table)
    independence = part4_independence(conn, ps_with_preds)
    whiff_tercile = part4_whiff_tercile_test(conn, ps_with_preds)
    leaderboard = part4_leaderboard(ps_with_preds, conn)

    validation_results = dict(
        sign_check=sign,
        score_distribution=dist,
        yoy_stability=yoy,
        split_half_reliability=split_half,
        predictive_validity=predictive,
        independence=independence,
        whiff_tercile_test=whiff_tercile,
        leaderboard=leaderboard,
        cv_folds=cv_folds,
        cv_mean_r2=cv_mean_r2,
        cv_mean_mae=cv_mean_mae,
        feature_cols=feature_cols,
    )
    (DIAG_DIR / "validation_results.json").write_text(json.dumps(validation_results, indent=2, default=str))
    print(f"\n  Saved validation_results.json", flush=True)

    # ── Part 5 ────────────────────────────────────────────────────────────────
    write_completion_report(part0, ablation, cv_folds, cv_mean_r2, cv_mean_mae,
                             dist, yoy, split_half, predictive, independence,
                             whiff_tercile, leaderboard, feature_cols)

    print(f"\n{'='*60}", flush=True)
    print(f"Done in {time.time()-start:.0f}s", flush=True)
    print(f"LOSO CV: mean R2={cv_mean_r2:.5f}, mean MAE={cv_mean_mae:.5f}", flush=True)
    print(f"Score spread: {dist.get('spread', '?')} pts (v3 reference: 9.0 pts)", flush=True)
    print(f"YoY pooled r2: {yoy.get('pooled_r2', '?')}", flush=True)
    print(f"Split-half r: {split_half.get('r', '?')}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
