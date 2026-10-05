"""
Stuff+ Pitcher Identity Memorization Diagnostic
================================================
Tests whether Stuff+ v3's high year-over-year stability (r≈0.83–0.88) reflects
genuine generalization or identity memorization via near-unique feature combinations,
without any literal pitcher_id column.

Reusable: set FEATURES_PATH, ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
and LGBM_PARAMS to adapt for Location+ or future Tunneling+ redesign.

Sections
--------
II.A  Leave-one-pitcher-out retrain (5-fold pitcher-grouped CV)
II.B  Distinctiveness subgroup breakdown
II.C  Direct identifiability probe (multiclass: feature vector → pitcher_id)
II.D  Regularization counter-test (if II.B shows real memorization)

Output: diagnostics/stuff_memorization/memorization_report.json
        diagnostics/stuff_memorization/<section>_*.csv  (detail tables)

Usage
-----
    python diagnostics/stuff_memorization/memorization_diagnostic.py

Run from project root (C:/Users/joshs/Desktop/mlb_dashboard).
"""
import gc
import json
import sys
import time
from pathlib import Path

# Make analytics importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

import analytics.stuff_model as sm_runtime
from analytics.metrics import compute_vaa
from analytics.stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    fit_vaa_expectation, fit_vaa_expectations_per_type,
)

# ---------------------------------------------------------------------------
# Configuration — swap these to adapt for Location+ / Tunneling+
# ---------------------------------------------------------------------------
FEATURES_PATH = Path(
    "project_artifacts/training_artifacts/pitching_plus_ext_artifacts/stuff/stuff_v3_features.parquet"
)
OUTPUT_DIR = Path("diagnostics/stuff_memorization")

COMPLETE_SEASONS = [str(s) for s in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]]

# Exact hyperparameters from train_stuff_plus_ext_lgbm.py
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

# Hyperparameter variants for II.D regularization counter-test
_base = {k: v for k, v in LGBM_PARAMS.items() if k not in ("min_child_samples", "num_leaves")}
LGBM_PARAMS_2X = dict(**_base, min_child_samples=400, num_leaves=31, max_depth=6)
LGBM_PARAMS_4X = dict(**_base, min_child_samples=800, num_leaves=15, max_depth=5)

MIN_PITCHES_QUALIFY = 200   # per-season pitcher qualification gate (same as validate script)
MIN_PITCHES_IDENTIFIABILITY = 500  # gate for II.C classifier
N_FOLDS = 5
BOOTSTRAP_N = 1000
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

BANNED_IDENTIFIER_COLS = {"pitcher_id", "game_id", "pitch_id", "player_id", "batter_id"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def assert_no_identifiers(cols):
    leaked = BANNED_IDENTIFIER_COLS & set(cols)
    if leaked:
        raise ValueError(f"Identifier leakage: {leaked}")


def refit_vaa_per_fold(train_df, test_df):
    """Refit vaa_residual from train_df only — matches v3 CV discipline."""
    for df in (train_df, test_df):
        df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])

    pooled = fit_vaa_expectation(train_df)
    per_type = fit_vaa_expectations_per_type(train_df)
    vaa_coefs = {"pooled": pooled.tolist(), **per_type}
    pooled_arr = np.array(vaa_coefs["pooled"])
    pt_map = {k: np.array(v) for k, v in vaa_coefs.items() if k != "pooled"}

    for df in (train_df, test_df):
        z = df["plate_z"].to_numpy(float)
        v = df["release_speed"].to_numpy(float)
        expected = np.empty(len(df), float)
        for pt, idx in df.groupby(df["pitch_type"].astype(str)).groups.items():
            coefs = pt_map.get(pt, pooled_arr)
            iloc = df.index.get_indexer(idx)
            design = np.column_stack([np.ones(len(idx)), z[iloc], z[iloc] ** 2, v[iloc]])
            expected[iloc] = design @ coefs
        df["vaa_residual"] = (df["vaa"].to_numpy(float) - expected).astype("float32")

    return train_df, test_df


def set_categoricals(train_df, test_df, cols):
    for col in cols:
        cats = sorted(set(train_df[col].astype(str)) | set(test_df[col].astype(str)))
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
        test_df[col] = pd.Categorical(test_df[col].astype(str), categories=cats)
    return train_df, test_df


def fit_model(train_df, params=None):
    if params is None:
        params = LGBM_PARAMS
    assert_no_identifiers(ALL_FEATURES)
    fit_rows, stop_rows = train_test_split(train_df, test_size=0.08, random_state=RANDOM_STATE)
    model = lgb.LGBMRegressor(**params)
    model.fit(
        fit_rows[ALL_FEATURES], fit_rows["delta_run_exp"],
        eval_set=[(stop_rows[ALL_FEATURES], stop_rows["delta_run_exp"])],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def pitcher_season_scores(test_df, preds, min_pitches=MIN_PITCHES_QUALIFY):
    """Per-(pitcher, season) shrunk Stuff+ scores, mirroring the production pipeline."""
    raw = test_df[["pitcher_id", "pitch_type", "season"]].copy()
    raw["pred"] = preds
    per_season = []
    for season, grp in raw.groupby("season", observed=True):
        if len(grp) < 10:
            continue
        k_by_type, fallback_k, type_means = sm_runtime._shrinkage_params(grp)
        g = sm_runtime._shrunk_group_means(
            grp, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means
        )
        g["season"] = season
        per_season.append(g)
    if not per_season:
        return pd.DataFrame()
    by_type = pd.concat(per_season, ignore_index=True)

    totals = by_type.groupby(["pitcher_id", "season"])["n_pitches"].sum()
    weighted = by_type.groupby(["pitcher_id", "season"]).apply(
        lambda d: np.average(
            sm_runtime._to_stuff_plus(d["shrunk_mean"], d["shrunk_mean"].mean(), d["shrunk_mean"].std() + 1e-9),
            weights=d["n_pitches"]
        ),
        include_groups=False
    )
    overall = pd.DataFrame({"stuff_plus": weighted}).join(totals).reset_index()
    return overall[overall["n_pitches"] >= min_pitches]


def yoy_r(scores_df):
    """Year-over-year Pearson r across all consecutive season pairs."""
    seasons = sorted(scores_df["season"].unique())
    pairs = []
    for i in range(len(seasons) - 1):
        t, t1 = seasons[i], seasons[i + 1]
        dt  = scores_df[scores_df["season"] == t][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t"})
        dt1 = scores_df[scores_df["season"] == t1][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t1"})
        m = dt.merge(dt1, on="pitcher_id", how="inner")
        if len(m) < 10:
            continue
        r, _ = stats.pearsonr(m["t"], m["t1"])
        pairs.append({"t": t, "t1": t1, "n": len(m), "r": float(r)})
    if not pairs:
        return float("nan"), []
    mean_r = float(np.mean([p["r"] for p in pairs]))
    return mean_r, pairs


def bootstrap_ci(values, stat_fn, n=BOOTSTRAP_N, ci=0.95):
    """Pitcher-clustered bootstrap CI for a scalar statistic."""
    bs = [stat_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    lo = float(np.percentile(bs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(bs, (1 + ci) / 2 * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# II.A  Pitcher-grouped CV
# ---------------------------------------------------------------------------

def build_pitcher_folds(df, n_folds=N_FOLDS):
    """
    Assign each pitcher to one of n_folds groups, stratified so that each
    fold contains a roughly equal number of pitches. Shuffle pitchers first
    so the assignment is not correlated with any incidental ordering.
    """
    pitcher_sizes = df.groupby("pitcher_id")["delta_run_exp"].count().reset_index()
    pitcher_sizes.columns = ["pitcher_id", "n_pitches"]
    pitcher_sizes = pitcher_sizes.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Assign folds by sorting on cumulative pitch count (round-robin binning)
    pitcher_sizes = pitcher_sizes.sort_values("n_pitches", ascending=False).reset_index(drop=True)
    pitcher_sizes["fold"] = pitcher_sizes.index % n_folds
    return pitcher_sizes[["pitcher_id", "fold"]]


def run_lopo_cv(df):
    """Leave-one-pitcher-out (pitcher-grouped) 5-fold CV."""
    print("\n=== II.A  Leave-one-pitcher-out CV ===", flush=True)
    fold_map = build_pitcher_folds(df)
    df = df.merge(fold_map, on="pitcher_id", how="left")

    fold_reports = []
    all_test_preds = []

    for fold_id in range(N_FOLDS):
        print(f"\n  Fold {fold_id + 1}/{N_FOLDS}...", flush=True)
        train_df = df[df["fold"] != fold_id].drop(columns=["fold"]).copy()
        test_df  = df[df["fold"] == fold_id].drop(columns=["fold"]).copy()

        # Refit vaa_residual from training data only
        train_df, test_df = refit_vaa_per_fold(train_df, test_df)
        train_df, test_df = set_categoricals(train_df, test_df, CATEGORICAL_FEATURES)

        n_train_pitchers = train_df["pitcher_id"].nunique()
        n_test_pitchers  = test_df["pitcher_id"].nunique()
        print(f"    Train: {len(train_df):,} pitches, {n_train_pitchers} pitchers", flush=True)
        print(f"    Test:  {len(test_df):,} pitches,  {n_test_pitchers} pitchers", flush=True)

        model = fit_model(train_df)
        preds = model.predict(test_df[ALL_FEATURES])

        r2  = float(r2_score(test_df["delta_run_exp"], preds))
        mae = float(mean_absolute_error(test_df["delta_run_exp"], preds))
        print(f"    R²={r2:.5f}  MAE={mae:.5f}  best_iter={model.best_iteration_}", flush=True)

        # Collect predictions for YoY analysis
        fold_test = test_df[["pitcher_id", "pitch_type", "season"]].copy()
        fold_test["pred"] = preds
        all_test_preds.append(fold_test)

        fold_reports.append(dict(
            fold=fold_id, r2=r2, mae=mae,
            n_pitches=len(test_df), n_pitchers=n_test_pitchers,
            best_iteration=int(model.best_iteration_),
        ))
        del train_df, test_df, model
        gc.collect()

    # YoY analysis on OOS predictions (all folds pooled)
    oos_preds = pd.concat(all_test_preds, ignore_index=True)
    oos_scores = pitcher_season_scores(oos_preds, oos_preds["pred"].values)
    lopo_yoy_mean, lopo_yoy_pairs = yoy_r(oos_scores)

    # Bootstrap CI for YoY r
    # Build paired data: for each consecutive pair merge YoY and stack
    def _yoy_r_from_indices(indices, paired_df):
        sub = paired_df.iloc[indices]
        if len(sub) < 5:
            return float("nan")
        r, _ = stats.pearsonr(sub["t"], sub["t1"])
        return r

    all_pairs = []
    for p in lopo_yoy_pairs:
        sub_t  = oos_scores[oos_scores["season"] == p["t"]][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t"})
        sub_t1 = oos_scores[oos_scores["season"] == p["t1"]][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t1"})
        all_pairs.append(sub_t.merge(sub_t1, on="pitcher_id", how="inner"))
    paired_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=["t", "t1"])

    if len(paired_df) >= 10:
        yoy_ci_lo, yoy_ci_hi = bootstrap_ci(
            np.arange(len(paired_df)),
            lambda idx: _yoy_r_from_indices(idx, paired_df),
        )
    else:
        yoy_ci_lo, yoy_ci_hi = float("nan"), float("nan")

    mean_r2  = float(np.mean([f["r2"]  for f in fold_reports]))
    mean_mae = float(np.mean([f["mae"] for f in fold_reports]))

    print(f"\n  LOPO CV summary: mean R²={mean_r2:.5f}  mean MAE={mean_mae:.5f}", flush=True)
    print(f"  LOPO YoY r: {lopo_yoy_mean:.4f}  (95% CI: {yoy_ci_lo:.4f}–{yoy_ci_hi:.4f})", flush=True)

    return dict(
        fold_reports=fold_reports,
        mean_r2=mean_r2,
        mean_mae=mean_mae,
        yoy_mean_r=lopo_yoy_mean,
        yoy_pairs=lopo_yoy_pairs,
        yoy_ci_lo=yoy_ci_lo,
        yoy_ci_hi=yoy_ci_hi,
        oos_scores=oos_scores,
    ), fold_map


# ---------------------------------------------------------------------------
# II.B  Distinctiveness subgroup breakdown
# ---------------------------------------------------------------------------

DISTINCTIVENESS_FEATURES = [
    "release_speed", "effective_speed", "release_spin_rate", "spin_axis",
    "hb", "ivb", "vaa_residual", "axis_differential",
]


def compute_distinctiveness(df):
    """
    Per-(pitcher, season, pitch_type) Mahalanobis distance of mean feature
    vector from that pitch_type × season league mean. A pitcher-season-type
    with z a unique spin_axis or unusual movement profile will score high.

    Returns a DataFrame with (pitcher_id, season, pitch_type, distinctiveness,
    n_pitches) plus a pitcher-level distinctiveness score (max over pitch types).
    """
    print("\n  Computing distinctiveness scores...", flush=True)
    feats = [f for f in DISTINCTIVENESS_FEATURES if f in df.columns]
    sub = df[["pitcher_id", "season", "pitch_type"] + feats].copy()

    # Standardize within (season, pitch_type) to handle scale differences
    records = []
    for (season, ptype), grp in sub.groupby(["season", "pitch_type"], observed=True):
        if len(grp) < 50:
            continue
        fmat = grp[feats].to_numpy(float)
        league_mean = np.nanmean(fmat, axis=0)
        league_std  = np.nanstd(fmat, axis=0) + 1e-9

        by_pitcher = grp.groupby("pitcher_id")[feats].mean().reset_index()
        pitcher_mat = by_pitcher[feats].to_numpy(float)
        standardized = (pitcher_mat - league_mean) / league_std

        # Euclidean distance in standardized space (robust to NaN via nansum)
        sq = np.nansum(standardized ** 2, axis=1)
        n_valid = np.sum(~np.isnan(standardized), axis=1).clip(min=1)
        dist = np.sqrt(sq / n_valid * len(feats))  # normalize by n_feats

        n_pitches = grp.groupby("pitcher_id")["release_speed"].count().reset_index(name="n_pitches")
        n_pitches = n_pitches.rename(columns={"release_speed": "n_pitches"}) if "release_speed" in n_pitches.columns else n_pitches

        rows = pd.DataFrame({
            "pitcher_id": by_pitcher["pitcher_id"].values,
            "season": season,
            "pitch_type": ptype,
            "distinctiveness": dist.astype(float),
        })
        rows = rows.merge(
            grp.groupby("pitcher_id")["release_speed"].count().reset_index().rename(columns={"release_speed": "n_pitches"}),
            on="pitcher_id", how="left"
        )
        records.append(rows)

    dist_df = pd.concat(records, ignore_index=True)

    # Pitcher-season level: usage-weighted max distinctiveness
    pitcher_dist = (
        dist_df.groupby(["pitcher_id", "season"])
        .apply(lambda d: np.average(d["distinctiveness"], weights=d["n_pitches"].clip(lower=1)), include_groups=False)
        .reset_index(name="distinctiveness")
    )

    # Tercile labels: typical / unusual / extreme
    q33, q67 = pitcher_dist["distinctiveness"].quantile([1/3, 2/3]).values
    pitcher_dist["tercile"] = pd.cut(
        pitcher_dist["distinctiveness"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["typical", "unusual", "extreme"],
    )

    print(f"  Distinctiveness computed for {len(pitcher_dist):,} pitcher-seasons", flush=True)
    print(f"  Tercile thresholds: q33={q33:.3f}, q67={q67:.3f}", flush=True)

    return dist_df, pitcher_dist, (q33, q67)


def subgroup_yoy_by_tercile(oos_scores, pitcher_dist):
    """
    Compute YoY r separately for each distinctiveness tercile.
    oos_scores: DataFrame with (pitcher_id, season, stuff_plus) from LOPO CV.
    """
    merged = oos_scores.merge(pitcher_dist[["pitcher_id", "season", "tercile"]], on=["pitcher_id", "season"], how="left")
    results = {}
    for tercile in ["typical", "unusual", "extreme"]:
        sub = merged[merged["tercile"] == tercile]
        mean_r, pairs = yoy_r(sub)
        n_pitchers = sub["pitcher_id"].nunique()
        results[tercile] = dict(
            yoy_mean_r=mean_r,
            yoy_pairs=pairs,
            n_pitcher_seasons=len(sub),
            n_pitchers_unique=int(n_pitchers),
        )
        print(f"  Tercile '{tercile}': YoY r={mean_r:.4f}  n_pitcher_seasons={len(sub)}", flush=True)
    return results


def extreme_pitcher_deltas(lopo_scores, loso_yoy_pairs_ref, pitcher_dist, top_n=10):
    """
    For the most extreme pitcher-seasons (by distinctiveness), compute
    their individual LOSO-model YoY r vs LOPO YoY r delta. Since we don't
    have per-pitcher LOSO scores from the training script, we proxy this by
    comparing each extreme pitcher's LOPO OOS score to their average score
    across adjacent seasons (a within-LOPO regression-to-mean check).

    Returns top_n most extreme pitcher-seasons with their distinctiveness
    and YoY delta (LOPO vs adjacent-season baseline).
    """
    extreme = pitcher_dist.nlargest(top_n * 2, "distinctiveness")

    deltas = []
    for _, row in extreme.iterrows():
        pid = row["pitcher_id"]
        season = row["season"]
        # Get all LOPO scores for this pitcher across seasons
        pitcher_scores = lopo_scores[lopo_scores["pitcher_id"] == pid].sort_values("season")
        if len(pitcher_scores) < 2:
            continue
        # Season we're examining
        this_score_rows = pitcher_scores[pitcher_scores["season"] == season]
        if this_score_rows.empty:
            continue
        this_score = this_score_rows["stuff_plus"].iloc[0]
        # Other seasons' mean
        other_mean = pitcher_scores[pitcher_scores["season"] != season]["stuff_plus"].mean()
        delta = abs(this_score - other_mean)  # deviation from own-pitcher mean
        deltas.append(dict(
            pitcher_id=int(pid),
            season=season,
            distinctiveness=float(row["distinctiveness"]),
            tercile=str(row["tercile"]),
            lopo_stuff_plus=float(this_score),
            other_seasons_mean=float(other_mean),
            deviation_from_own_mean=float(delta),
            n_seasons=int(len(pitcher_scores)),
        ))
        if len(deltas) >= top_n:
            break

    return sorted(deltas, key=lambda x: -x["distinctiveness"])


# ---------------------------------------------------------------------------
# II.C  Direct identifiability probe
# ---------------------------------------------------------------------------

def run_identifiability_probe(df, pitcher_dist):
    """
    Train a LightGBM multiclass classifier to predict pitcher_id from the
    Stuff+ feature vector (no pitcher_id column). Restricted to pitchers
    with ≥ MIN_PITCHES_IDENTIFIABILITY pitches (across all seasons).

    Reports top-1 and top-5 accuracy vs naive baseline, overall and by
    distinctiveness tercile.
    """
    print("\n=== II.C  Direct identifiability probe ===", flush=True)

    # Restrict to qualifying pitchers
    pitcher_counts = df.groupby("pitcher_id")["delta_run_exp"].count()
    qualifying = pitcher_counts[pitcher_counts >= MIN_PITCHES_IDENTIFIABILITY].index
    sub = df[df["pitcher_id"].isin(qualifying)].copy()
    n_pitchers = sub["pitcher_id"].nunique()
    print(f"  Qualifying pitchers (≥{MIN_PITCHES_IDENTIFIABILITY} pitches): {n_pitchers}", flush=True)
    print(f"  Total pitches for probe: {len(sub):,}", flush=True)

    # Encode pitcher_id as integer class
    pitcher_ids = sorted(sub["pitcher_id"].unique())
    pid_to_cls = {pid: i for i, pid in enumerate(pitcher_ids)}
    sub["target_cls"] = sub["pitcher_id"].map(pid_to_cls)

    # Sample to keep runtime tractable (cap at 500k pitches)
    MAX_SAMPLE = 500_000
    if len(sub) > MAX_SAMPLE:
        sub = sub.sample(n=MAX_SAMPLE, random_state=RANDOM_STATE)
        print(f"  Sampled to {MAX_SAMPLE:,} for tractability", flush=True)

    # Stratified train/test split by pitcher
    pitchers_array = sub["pitcher_id"].values
    unique_pitchers = np.unique(pitchers_array)
    rng2 = np.random.default_rng(RANDOM_STATE)
    test_pitchers = set(rng2.choice(unique_pitchers, size=int(len(unique_pitchers) * 0.2), replace=False))
    train_mask = ~sub["pitcher_id"].isin(test_pitchers)
    train_sub = sub[train_mask].copy()
    test_sub  = sub[~train_mask].copy()
    print(f"  Train: {len(train_sub):,} pitches, Test: {len(test_sub):,} pitches", flush=True)

    # Set categoricals for classifier
    cat_feats = CATEGORICAL_FEATURES
    for col in cat_feats:
        cats = sorted(set(train_sub[col].astype(str)) | set(test_sub[col].astype(str)))
        train_sub[col] = pd.Categorical(train_sub[col].astype(str), categories=cats)
        test_sub[col]  = pd.Categorical(test_sub[col].astype(str), categories=cats)

    # Multiclass LightGBM — lighter params for a classifier
    clf_params = dict(
        objective="multiclass",
        num_class=n_pitchers,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    assert_no_identifiers(ALL_FEATURES)

    print("  Fitting classifier...", flush=True)
    clf_fit, clf_stop = train_test_split(train_sub, test_size=0.1, random_state=RANDOM_STATE)
    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(
        clf_fit[ALL_FEATURES], clf_fit["target_cls"],
        eval_set=[(clf_stop[ALL_FEATURES], clf_stop["target_cls"])],
        categorical_feature=cat_feats,
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    print("  Predicting probabilities...", flush=True)
    proba = clf.predict_proba(test_sub[ALL_FEATURES])

    true_cls = test_sub["target_cls"].values
    top1_preds = np.argmax(proba, axis=1)
    top1_acc = float(np.mean(top1_preds == true_cls))

    # Top-5 accuracy
    top5_idx = np.argsort(proba, axis=1)[:, -5:]
    top5_acc = float(np.mean([true_cls[i] in top5_idx[i] for i in range(len(true_cls))]))

    naive_top1 = 1.0 / n_pitchers
    naive_top5 = 5.0 / n_pitchers

    print(f"\n  Top-1 accuracy: {top1_acc:.4f}  (naive baseline: {naive_top1:.4f})", flush=True)
    print(f"  Top-5 accuracy: {top5_acc:.4f}  (naive baseline: {naive_top5:.4f})", flush=True)
    print(f"  Top-1 lift over naive: {top1_acc / naive_top1:.1f}x", flush=True)

    # Breakdown by distinctiveness tercile
    pid_tercile = pitcher_dist.groupby("pitcher_id")["distinctiveness"].mean()
    pid_tercile_labels = pitcher_dist.groupby("pitcher_id")["tercile"].apply(
        lambda s: s.mode().iat[0] if len(s) > 0 else "unknown"
    )
    test_sub_copy = test_sub.copy()
    test_sub_copy["tercile"] = test_sub_copy["pitcher_id"].map(pid_tercile_labels)
    test_sub_copy["correct_top1"] = (top1_preds == true_cls)
    test_sub_copy["correct_top5"] = [true_cls[i] in top5_idx[i] for i in range(len(true_cls))]

    by_tercile = {}
    for tercile in ["typical", "unusual", "extreme"]:
        mask = test_sub_copy["tercile"] == tercile
        if mask.sum() < 10:
            continue
        t1 = float(test_sub_copy.loc[mask, "correct_top1"].mean())
        t5 = float(test_sub_copy.loc[mask, "correct_top5"].mean())
        n_p = int(test_sub_copy.loc[mask, "pitcher_id"].nunique())
        by_tercile[tercile] = dict(top1_acc=t1, top5_acc=t5, n_pitchers=n_p, lift_over_naive=t1 / naive_top1)
        print(f"  Tercile '{tercile}': top-1={t1:.4f}  top-5={t5:.4f}  n_pitchers={n_p}", flush=True)

    return dict(
        n_pitchers=n_pitchers,
        top1_acc=top1_acc,
        top5_acc=top5_acc,
        naive_top1=naive_top1,
        naive_top5=naive_top5,
        lift_top1=top1_acc / naive_top1,
        by_tercile=by_tercile,
    )


# ---------------------------------------------------------------------------
# II.D  Regularization counter-test
# ---------------------------------------------------------------------------

def run_regularization_counter_test(df, fold_map, pitcher_dist):
    """
    Retrain under pitcher-grouped CV with 2x and 4x stronger regularization.
    Report whether the extreme-tercile performance gap narrows and whether
    overall performance degrades.
    """
    print("\n=== II.D  Regularization counter-test ===", flush=True)
    df = df.merge(fold_map, on="pitcher_id", how="left")

    results = {}
    for label, params in [("2x_reg", LGBM_PARAMS_2X), ("4x_reg", LGBM_PARAMS_4X)]:
        print(f"\n  Variant: {label}", flush=True)
        all_test_preds = []
        fold_r2s = []
        for fold_id in range(N_FOLDS):
            train_df = df[df["fold"] != fold_id].drop(columns=["fold"]).copy()
            test_df  = df[df["fold"] == fold_id].drop(columns=["fold"]).copy()
            train_df, test_df = refit_vaa_per_fold(train_df, test_df)
            train_df, test_df = set_categoricals(train_df, test_df, CATEGORICAL_FEATURES)
            model = fit_model(train_df, params=params)
            preds = model.predict(test_df[ALL_FEATURES])
            fold_r2s.append(float(r2_score(test_df["delta_run_exp"], preds)))
            fold_preds = test_df[["pitcher_id", "pitch_type", "season"]].copy()
            fold_preds["pred"] = preds
            all_test_preds.append(fold_preds)
            del train_df, test_df, model
            gc.collect()

        oos_preds = pd.concat(all_test_preds, ignore_index=True)
        oos_scores = pitcher_season_scores(oos_preds, oos_preds["pred"].values)
        mean_r, _ = yoy_r(oos_scores)

        # YoY by tercile
        merged = oos_scores.merge(pitcher_dist[["pitcher_id", "season", "tercile"]], on=["pitcher_id", "season"], how="left")
        by_tercile = {}
        for tercile in ["typical", "unusual", "extreme"]:
            sub = merged[merged["tercile"] == tercile]
            r, _ = yoy_r(sub)
            by_tercile[tercile] = float(r)

        results[label] = dict(
            mean_r2=float(np.mean(fold_r2s)),
            yoy_mean_r=float(mean_r),
            by_tercile=by_tercile,
        )
        print(f"  {label}: mean R²={np.mean(fold_r2s):.5f}  YoY r={mean_r:.4f}", flush=True)
        for t, r in by_tercile.items():
            print(f"    tercile '{t}': YoY r={r:.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Interpret and summarize
# ---------------------------------------------------------------------------

def interpret(section_a, section_b, section_c, section_d):
    """
    Plain-language interpretation of all evidence.
    Returns a dict with a top-line verdict and supporting rationale.
    """
    lopo_yoy  = section_a["yoy_mean_r"]
    loso_yoy  = 0.845  # observed mean from v3_validation_battery.json (2016-2025 avg)
    yoy_drop  = loso_yoy - lopo_yoy

    typical_r = section_b.get("typical", {}).get("yoy_mean_r", float("nan"))
    extreme_r = section_b.get("extreme", {}).get("yoy_mean_r", float("nan"))
    tercile_gap = typical_r - extreme_r if not (np.isnan(typical_r) or np.isnan(extreme_r)) else float("nan")

    top1_lift   = section_c["lift_top1"]
    extreme_t1  = section_c.get("by_tercile", {}).get("extreme", {}).get("lift_over_naive", float("nan"))
    typical_t1  = section_c.get("by_tercile", {}).get("typical", {}).get("lift_over_naive", float("nan"))

    # Criteria for memorization conclusion
    memo_signals = []
    genuine_signals = []

    if yoy_drop > 0.05:
        memo_signals.append(f"LOPO YoY r ({lopo_yoy:.3f}) drops >0.05 from LOSO ({loso_yoy:.3f}), consistent with memorization.")
    else:
        genuine_signals.append(f"LOPO vs LOSO YoY drop is small ({yoy_drop:.3f}), consistent with genuine generalization.")

    if not np.isnan(tercile_gap) and tercile_gap > 0.05:
        memo_signals.append(f"Extreme tercile YoY r ({extreme_r:.3f}) is notably lower than typical ({typical_r:.3f}), gap={tercile_gap:.3f}.")
    elif not np.isnan(tercile_gap):
        genuine_signals.append(f"Tercile gap in YoY r is small ({tercile_gap:.3f}), no unusual concentration of drop at extreme.")

    if top1_lift > 20:
        memo_signals.append(f"Identifiability probe: top-1 lift {top1_lift:.1f}x over naive — feature vector encodes meaningful pitcher identity.")
    elif top1_lift > 5:
        memo_signals.append(f"Moderate identifiability: top-1 lift {top1_lift:.1f}x over naive.")
    else:
        genuine_signals.append(f"Feature vector is only weakly identifiable (top-1 lift {top1_lift:.1f}x).")

    if not np.isnan(extreme_t1) and not np.isnan(typical_t1) and extreme_t1 > typical_t1 * 1.5:
        memo_signals.append(f"Extreme tercile identifiability ({extreme_t1:.1f}x) is substantially higher than typical ({typical_t1:.1f}x).")
    elif not np.isnan(extreme_t1) and not np.isnan(typical_t1):
        genuine_signals.append(f"Extreme vs typical identifiability lift is not dramatic ({extreme_t1:.1f}x vs {typical_t1:.1f}x).")

    n_memo = len(memo_signals)
    n_genuine = len(genuine_signals)

    if n_memo >= 3:
        verdict = "MEMORIZATION"
        summary = "Multiple independent signals point to identity memorization in the extreme-distinctiveness population."
    elif n_memo >= 2 and n_genuine < 2:
        verdict = "MIXED — LEAN MEMORIZATION"
        summary = "Evidence is mixed but tilts toward memorization for unusual/extreme pitchers."
    elif n_memo >= 1 and n_genuine >= 2:
        verdict = "MIXED — LEAN GENUINE"
        summary = "Some memorization signal exists but the weight of evidence supports genuine generalization."
    else:
        verdict = "GENUINE GENERALIZATION"
        summary = "Evidence supports genuine generalization across all distinctiveness groups."

    return dict(
        verdict=verdict,
        summary=summary,
        lopo_yoy=lopo_yoy,
        loso_yoy=loso_yoy,
        yoy_drop=yoy_drop,
        typical_yoy=typical_r,
        extreme_yoy=extreme_r,
        tercile_gap=tercile_gap,
        top1_lift=top1_lift,
        extreme_top1_lift=extreme_t1,
        typical_top1_lift=typical_t1,
        memorization_signals=memo_signals,
        genuine_signals=genuine_signals,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print("Loading features...", flush=True)
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["season"].astype(str).isin(COMPLETE_SEASONS)].copy()
    df["season"] = df["season"].astype(str)
    print(f"  {len(df):,} pitches across {len(COMPLETE_SEASONS)} seasons, {df['pitcher_id'].nunique()} pitchers", flush=True)

    # Ensure delta_run_exp present (required for training)
    df = df.dropna(subset=["delta_run_exp"])

    # Ensure categoricals are plain strings before fold processing
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    # ---------- II.A ----------
    iia_start = time.time()
    section_a_raw, fold_map = run_lopo_cv(df)
    oos_scores = section_a_raw.pop("oos_scores")
    print(f"  II.A done in {time.time() - iia_start:.0f}s", flush=True)

    # LOSO reference YoY from existing validation battery
    loso_yoy_ref = [
        {"t": "2016", "t1": "2017", "n": 435, "r": 0.820},
        {"t": "2017", "t1": "2018", "n": 427, "r": 0.818},
        {"t": "2018", "t1": "2019", "n": 450, "r": 0.830},
        {"t": "2019", "t1": "2020", "n": 350, "r": 0.751},
        {"t": "2020", "t1": "2021", "n": 374, "r": 0.824},
        {"t": "2021", "t1": "2022", "n": 444, "r": 0.854},
        {"t": "2022", "t1": "2023", "n": 450, "r": 0.864},
        {"t": "2023", "t1": "2024", "n": 436, "r": 0.844},
        {"t": "2024", "t1": "2025", "n": 430, "r": 0.878},
    ]
    loso_yoy_mean = float(np.mean([p["r"] for p in loso_yoy_ref]))
    section_a_raw["loso_yoy_reference"] = loso_yoy_ref
    section_a_raw["loso_yoy_mean"] = loso_yoy_mean

    # ---------- II.B ----------
    print("\n=== II.B  Distinctiveness subgroup breakdown ===", flush=True)
    iib_start = time.time()
    dist_df, pitcher_dist, (q33, q67) = compute_distinctiveness(df)
    by_tercile = subgroup_yoy_by_tercile(oos_scores, pitcher_dist)
    top10_deltas = extreme_pitcher_deltas(oos_scores, loso_yoy_ref, pitcher_dist, top_n=10)
    section_b = dict(
        tercile_thresholds=dict(q33=float(q33), q67=float(q67)),
        by_tercile=by_tercile,
        top10_extreme_pitcher_seasons=top10_deltas,
    )
    print(f"  II.B done in {time.time() - iib_start:.0f}s", flush=True)

    # Save detail tables
    pitcher_dist.to_csv(OUTPUT_DIR / "pitcher_distinctiveness.csv", index=False)
    pd.DataFrame(top10_deltas).to_csv(OUTPUT_DIR / "top10_extreme_pitchers.csv", index=False)

    # ---------- II.C ----------
    iic_start = time.time()
    section_c = run_identifiability_probe(df, pitcher_dist)
    print(f"  II.C done in {time.time() - iic_start:.0f}s", flush=True)

    # ---------- II.D (conditional) ----------
    # Determine if memorization is real: need II.B tercile gap > 0.05 AND II.C extreme lift > typical
    typical_r = by_tercile.get("typical", {}).get("yoy_mean_r", float("nan"))
    extreme_r = by_tercile.get("extreme", {}).get("yoy_mean_r", float("nan"))
    extreme_t1_lift = section_c.get("by_tercile", {}).get("extreme", {}).get("lift_over_naive", 0)
    typical_t1_lift = section_c.get("by_tercile", {}).get("typical", {}).get("lift_over_naive", 0)
    tercile_gap = typical_r - extreme_r if not (np.isnan(typical_r) or np.isnan(extreme_r)) else 0.0

    run_reg_test = (tercile_gap > 0.05) or (extreme_t1_lift > typical_t1_lift * 1.5)
    section_d = None
    if run_reg_test:
        print("\n  Memorization signal detected — running regularization counter-test.", flush=True)
        iid_start = time.time()
        section_d = run_regularization_counter_test(df, fold_map, pitcher_dist)
        print(f"  II.D done in {time.time() - iid_start:.0f}s", flush=True)
    else:
        print("\n  No strong memorization signal — skipping II.D regularization test.", flush=True)

    # ---------- Interpretation ----------
    interp = interpret(section_a_raw, by_tercile, section_c, section_d)

    # ---------- Final report ----------
    report = dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_runtime_seconds=round(time.time() - total_start, 1),
        section_A=section_a_raw,
        section_B=section_b,
        section_C=section_c,
        section_D=section_d,
        interpretation=interp,
        reusability_note=(
            "Adapt for Location+ / Tunneling+: update FEATURES_PATH, ALL_FEATURES, "
            "CATEGORICAL_FEATURES, NUMERIC_FEATURES, LGBM_PARAMS at the top of this script. "
            "The DISTINCTIVENESS_FEATURES list may also need updating to reflect those models' "
            "most identity-informative numeric features."
        ),
    )

    report_path = OUTPUT_DIR / "memorization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"VERDICT: {interp['verdict']}", flush=True)
    print(f"Summary: {interp['summary']}", flush=True)
    print(f"\nLOSO YoY r (reference): {interp['loso_yoy']:.4f}", flush=True)
    print(f"LOPO YoY r (this run):  {interp['lopo_yoy']:.4f}", flush=True)
    print(f"YoY drop:               {interp['yoy_drop']:.4f}", flush=True)
    print(f"\nIdentifiability top-1 lift: {interp['top1_lift']:.1f}x over naive baseline", flush=True)
    print(f"\nMemoization signals:  {len(interp['memorization_signals'])}", flush=True)
    for s in interp["memorization_signals"]:
        print(f"  [!] {s}", flush=True)
    print(f"Genuine signals:      {len(interp['genuine_signals'])}", flush=True)
    for s in interp["genuine_signals"]:
        print(f"  [+] {s}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
