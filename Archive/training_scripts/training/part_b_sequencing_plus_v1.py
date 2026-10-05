"""
Part B — Sequencing-only diagnostic model (placeholder name: sequencing_plus_v1).

Tests whether dropping the 18 lag-geometry features (commit_distance, plate_distance,
shape_break across 6 lags) and leaning fully into pitch-type sequencing and arsenal-
composition features produces a model that holds up or improves vs Tunneling+ v2.

Feature set (actual, not proposed):
  KEPT from v2:
    - lagN_pitch_type for N=1-6 (6 categoricals)
    - pitch_type, p_throws, stand, season (4 categoricals)
  DROPPED:
    - lagN_commit_distance, lagN_plate_distance, lagN_shape_break for N=1-6 (18 numerics)
  ADDED:
    - same_as_lag1 (bool → int): is current pitch_type == lag1_pitch_type
    - pitch_n_in_pa (int): position in at-bat (= count of non-null lag slots + 1)
    - n_distinct_prior_types (int): count of unique pitch types seen in prior lag slots
    - arsenal_entropy (float): Shannon entropy of pitcher-season pitch-type distribution

  DELIBERATELY EXCLUDED (construct-separation, matching standing direction):
    - count (balls/strikes), plate location (plate_x/plate_z)
    Note: these exclusions keep this model's construct cleanly separated from Location+.

Architecture: same decomposed 5-classifier chain as Tunneling+ v2 (xSwing%, xCS%,
xWhiff%, xFoul%, BIP type×EV) + same fixed run-value table → per-pitch xRV → pitcher-
season plus-scale. Directly apples-to-apples comparable.

Naming: ALL files use the `sequencing_plus_v1` prefix. This is provisional; final
name undecided.

Validation battery:
  1. Score spread (std, percentiles)
  2. YoY stability (Pearson r, pooled)
  3. Split-half reliability
  4. Out-of-sample predictive validity (whiff%, swing%, chase%)
  5. ERA / xFIP correlation + direction
  6. Classifier AUCs vs Tunneling+ v2 reference AUCs

No production changes. Not wired into api/. Tunneling+ v2 stays live.

Run from mlb_dashboard root:
    python training/part_b_sequencing_plus_v1.py
"""
import json
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss

from analytics.tunneling_features import (
    lag_feature_columns, load_pitches, sequential_lag_features, MAX_LAG,
)
from analytics.location_features import to_plus_scale

# ── Paths ──────────────────────────────────────────────────────────────────
V2_MODELS_DIR = ROOT / "models"
RV_PATH       = ROOT / "training" / "v2_artifacts" / "run_value_table.json"
COMMIT_PATH   = (
    ROOT / "project_artifacts" / "training_artifacts"
    / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv"
)
OUT_DIR  = ROOT / "training" / "v2_artifacts"
OUT_PATH = OUT_DIR / "sequencing_plus_v1_diagnostic.json"
LOG_PATH = ROOT / "training" / "part_b_sequencing_plus_v1.log"
DB_PATH  = "mlb_data.db"

# ── Seasons ────────────────────────────────────────────────────────────────
# Commit table covers 2021-2025. Use 2021-2025 for LOSO CV (5 folds).
# Use full 2015-2025 for final scoring and validation.
COMMIT_SEASONS = list(range(2021, 2026))
SCORING_SEASONS = list(range(2015, 2026))
MIN_PITCHES = 300

# ── Feature lists ──────────────────────────────────────────────────────────
LAG_COLS        = lag_feature_columns()
PT_LAG_COLS     = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CAT     = ["pitch_type", "p_throws", "stand", "season"]
ALL_CAT_V2      = PT_LAG_COLS + CURRENT_CAT  # no geometry numerics

# New sequencing features (numerics/bools)
NEW_FEATURES    = ["same_as_lag1", "pitch_n_in_pa", "n_distinct_prior_types", "arsenal_entropy"]

# Full feature list for the new model
ALL_FEATURES    = NEW_FEATURES + ALL_CAT_V2  # numerics first, then categoricals

CATEGORICAL_COLS = ALL_CAT_V2  # these are categorical dtype in the DataFrame

# ── Outcome definitions ────────────────────────────────────────────────────
SWING_DESCS = {
    "swinging_strike","swinging_strike_blocked","foul_tip","bunt_foul_tip",
    "missed_bunt","foul","foul_bunt","foul_pitchout",
    "hit_into_play","hit_into_play_no_out","hit_into_play_score","swinging_pitchout",
}
WHIFF_DESCS = {"swinging_strike","swinging_strike_blocked","foul_tip","bunt_foul_tip","missed_bunt","swinging_pitchout"}
FOUL_DESCS  = {"foul","foul_bunt","foul_pitchout"}
BIP_DESCS   = {"hit_into_play","hit_into_play_no_out","hit_into_play_score"}
CS_DESCS    = {"called_strike","automatic_strike"}
BALL_DESCS  = {"ball","blocked_ball","pitchout","automatic_ball","intent_ball","pitchout_hit_into_play","pitchout_hit_into_play_score"}
HBP_DESCS   = {"hit_by_pitch"}

EV_BINS   = [0, 90, 95, 100, 105, 999]
EV_LABELS = ["ev_lt90","ev_90_95","ev_95_100","ev_100_105","ev_105plus"]

CLASSIFIER_PARAMS = {
    "n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": 42, "verbose": -1,
}

_log_fh = None

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n"); _log_fh.flush()


# ── Feature engineering helpers ────────────────────────────────────────────

def build_arsenal_entropy(conn, seasons):
    """Shannon entropy of pitch-type distribution per (pitcher_id, season)."""
    df = pd.read_sql(f"""
        SELECT pitcher_id, season, pitch_type, COUNT(*) as n
        FROM pitches
        WHERE season BETWEEN {min(seasons)} AND {max(seasons)}
          AND game_type = 'R'
          AND pitch_type NOT IN ('PO','UN','EP')
        GROUP BY pitcher_id, season, pitch_type
    """, conn)
    df["pitch_type"] = df["pitch_type"].replace({"KC": "CU"})
    total = df.groupby(["pitcher_id","season"])["n"].transform("sum")
    df["p"] = df["n"] / total
    df["plogp"] = -df["p"] * np.log(df["p"].clip(lower=1e-9))
    entropy = df.groupby(["pitcher_id","season"])["plogp"].sum().reset_index()
    entropy.columns = ["pitcher_id","season","arsenal_entropy"]
    return entropy


def add_sequencing_features(df, arsenal_entropy_df):
    """Add same_as_lag1, pitch_n_in_pa, n_distinct_prior_types, arsenal_entropy."""
    df = df.copy()

    # same_as_lag1: current pitch type equals immediately-prior pitch type
    df["same_as_lag1"] = (
        (df["pitch_type"].astype(str) == df["lag1_pitch_type"].astype(str)).astype(float)
    )
    df.loc[df["lag1_pitch_type"].isna(), "same_as_lag1"] = np.nan

    # pitch_n_in_pa: count of non-null lag_k_pitch_type + 1
    pt_lag_present = df[PT_LAG_COLS].notna().sum(axis=1)
    df["pitch_n_in_pa"] = (pt_lag_present + 1).astype(float)

    # n_distinct_prior_types: count of unique non-null values across lag1..lag6 pitch_type
    def count_distinct(row):
        vals = set()
        for col in PT_LAG_COLS:
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.add(str(v))
        return float(len(vals))
    df["n_distinct_prior_types"] = df.apply(count_distinct, axis=1)

    # arsenal_entropy: join per pitcher-season
    df["season_int"] = df["season"].astype(str).str[:4].astype(int)
    df = df.merge(
        arsenal_entropy_df.rename(columns={"season": "season_int"}),
        on=["pitcher_id","season_int"], how="left"
    )
    df = df.drop(columns=["season_int"])
    return df


def stringify_keep_na(series):
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)


# ── Outcome labeling ───────────────────────────────────────────────────────

def add_outcome_labels(df):
    df = df.copy()
    df["is_swing"] = df["description"].isin(SWING_DESCS).astype(float)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCS).astype(float)
    df["is_foul"]  = df["description"].isin(FOUL_DESCS).astype(float)
    df["is_bip"]   = df["description"].isin(BIP_DESCS).astype(float)
    df["is_cs"]    = df["description"].isin(CS_DESCS).astype(float)

    # BIP class for multiclass target (only when bb_type/launch_speed are available)
    df["bip_class"] = None
    if "bb_type" in df.columns and "launch_speed" in df.columns:
        ls = pd.to_numeric(df["launch_speed"], errors="coerce")
        bb = pd.cut(ls, bins=EV_BINS, labels=EV_LABELS, right=False)
        df["bip_ev_bin"] = bb.astype(str)
        has_bip = df["is_bip"] == 1.0
        has_class = df["bb_type"].notna() & (df["bip_ev_bin"] != "nan")
        df.loc[has_bip & has_class, "bip_class"] = (
            df.loc[has_bip & has_class, "bb_type"].str.replace(" ", "_") + "__" +
            df.loc[has_bip & has_class, "bip_ev_bin"]
        )
    return df


# ── xRV combination ────────────────────────────────────────────────────────

def _pred_binary(m, X):
    """Probability of positive class, works for both LGBMClassifier and lgb.Booster."""
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.predict(X)  # Booster already returns probabilities


def _pred_bip(m, X, bip_classes):
    """Ordered probability array over bip_classes, works for both model types."""
    if hasattr(m, "predict_proba"):
        # LGBMClassifier — reorder columns to match bip_classes
        cls_list = list(m.classes_)
        proba = m.predict_proba(X)  # (n, n_clf_classes)
        ordered = np.zeros((len(X), len(bip_classes)))
        for i, c in enumerate(bip_classes):
            if c in cls_list:
                ordered[:, i] = proba[:, cls_list.index(c)]
        return ordered
    return m.predict(X)  # Booster predict on multiclass → (n, n_classes) in training order


def combine_xrv(X, models, rv, bip_classes, hbp_rate=0.0051):
    p_sw  = _pred_binary(models["xswing"], X)
    p_cs  = _pred_binary(models["xcs"],    X)
    p_wh  = _pred_binary(models["xwhiff"], X)
    p_fo  = _pred_binary(models["xfoul"],  X)
    p_bip = _pred_bip(models["bip"], X, bip_classes)
    bip_rvs = np.array([rv.get(c, 0.0) for c in bip_classes])
    xrv_bip     = p_bip @ bip_rvs
    xrv_contact = p_fo * rv["foul"] + (1.0 - p_fo) * xrv_bip
    xrv_swing   = p_wh * rv["whiff"] + (1.0 - p_wh) * xrv_contact
    p_ball      = np.clip(1.0 - p_cs - hbp_rate, 0.0, 1.0)
    xrv_no_swing = p_cs * rv["called_strike"] + hbp_rate * rv["hbp"] + p_ball * rv["ball"]
    return p_sw * xrv_swing + (1.0 - p_sw) * xrv_no_swing


# ── Data loading ───────────────────────────────────────────────────────────

def load_and_prepare(conn, seasons, commit_table, cat_levels, arsenal_entropy_df,
                     need_outcomes=True, need_bip_extras=False):
    """Load all seasons, build lag features, add sequencing features + outcomes."""
    chunks = []
    for season in seasons:
        t0 = time.time()
        log(f"  Season {season}...")
        df_s = load_pitches(conn, season)
        if len(df_s) == 0:
            log("    No pitches, skipping"); continue

        if need_bip_extras:
            # Fetch extra columns for BIP class assignment
            extra = pd.read_sql(f"""
                SELECT game_id, pitcher_id, at_bat_number, pitch_number,
                       bb_type, launch_speed
                FROM pitches WHERE season = {season} AND game_type = 'R'
            """, conn)
            df_s = df_s.merge(extra, on=["game_id","pitcher_id","at_bat_number","pitch_number"], how="left")

        feat = sequential_lag_features(df_s, commit_table, max_lag=MAX_LAG)
        feat = feat.dropna(subset=["lag1_commit_distance","lag1_plate_distance"])
        if len(feat) == 0:
            log("    No lag rows, skipping"); continue

        feat["season"] = str(season)

        # Add sequencing features BEFORE coercing categoricals
        feat = add_sequencing_features(feat, arsenal_entropy_df)

        # Coerce categoricals
        for col in CATEGORICAL_COLS:
            feat[col] = stringify_keep_na(feat[col])
            levels = cat_levels.get(col)
            feat[col] = pd.Categorical(feat[col], categories=levels) if levels else pd.Categorical(feat[col])

        if need_outcomes:
            feat = add_outcome_labels(feat)

        chunks.append(feat)
        log(f"    {len(feat):,} rows ({time.time()-t0:.0f}s)")

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ── LOSO CV ────────────────────────────────────────────────────────────────

def cv_binary_classifier(df_train, df_test, target, all_feats, cat_feats, params):
    sub_train = df_train.dropna(subset=[target])
    sub_test  = df_test.dropna(subset=[target])
    X_tr = sub_train[all_feats]
    y_tr = sub_train[target].astype(int)
    X_te = sub_test[all_feats]
    y_te = sub_test[target].astype(int)
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, categorical_feature=cat_feats)
    prob = m.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, prob) if len(y_te.unique()) > 1 else float("nan")
    ll  = log_loss(y_te, prob)
    return auc, ll, m


def cv_bip_classifier(df_train, df_test, all_feats, cat_feats, params, all_classes):
    sub_train = df_train.dropna(subset=["bip_class"])
    sub_test  = df_test.dropna(subset=["bip_class"])
    if len(sub_train) < 100 or len(sub_test) < 10:
        return float("nan"), float("nan"), None
    X_tr = sub_train[all_feats]
    y_tr = sub_train["bip_class"]
    X_te = sub_test[all_feats]
    y_te = sub_test["bip_class"]
    m = lgb.LGBMClassifier(objective="multiclass", num_class=len(all_classes),
                            **params)
    m.fit(X_tr, y_tr, categorical_feature=cat_feats)
    prob = m.predict_proba(X_te)
    try:
        ll = log_loss(y_te, prob, labels=m.classes_)
    except Exception:
        ll = float("nan")
    return float("nan"), ll, m  # AUC not defined for multiclass directly


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global _log_fh
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")
    log("Part B — Sequencing+ v1 diagnostic model")
    log(f"Features: {ALL_FEATURES}")

    rv_data    = json.loads(RV_PATH.read_text())
    rv_table   = rv_data["rv_table"]
    bip_classes = rv_data["bip_classes"]

    commit_table = pd.read_csv(COMMIT_PATH)
    log(f"Commit table: {len(commit_table):,} rows")

    conn = sqlite3.connect(DB_PATH)

    # Category levels from full range
    log("Deriving category levels...")
    raw_cats = pd.read_sql("""
        SELECT DISTINCT pitch_type, p_throws, stand, season FROM pitches
        WHERE season BETWEEN 2015 AND 2026
    """, conn)
    raw_cats["pitch_type"] = raw_cats["pitch_type"].replace({"KC": "CU"})
    raw_cats = raw_cats[~raw_cats["pitch_type"].isin({"PO","UN","EP"})]
    raw_cats["season"] = raw_cats["season"].astype(str)
    pt_levels = sorted(raw_cats["pitch_type"].dropna().unique().tolist())
    cat_levels = {
        "pitch_type": pt_levels,
        "p_throws":   sorted(raw_cats["p_throws"].dropna().unique().tolist()),
        "stand":      sorted(raw_cats["stand"].dropna().unique().tolist()),
        "season":     sorted(raw_cats["season"].dropna().unique().tolist()),
    }
    for col in PT_LAG_COLS:
        cat_levels[col] = pt_levels

    # Arsenal entropy (2015-2025)
    log("Building arsenal entropy...")
    arsenal_entropy_df = build_arsenal_entropy(conn, SCORING_SEASONS)
    log(f"  {len(arsenal_entropy_df):,} pitcher-season entropy values")

    # ── LOSO CV (2021-2025) ──────────────────────────────────────────────
    log("\nLoading CV data (2021-2025)...")
    t0 = time.time()
    df_cv = load_and_prepare(conn, COMMIT_SEASONS, commit_table, cat_levels,
                             arsenal_entropy_df, need_outcomes=True, need_bip_extras=True)
    log(f"CV data loaded: {len(df_cv):,} rows ({time.time()-t0:.1f}s)")

    # BIP class universe
    bip_universe = sorted(df_cv["bip_class"].dropna().unique().tolist())
    log(f"BIP classes: {len(bip_universe)}")

    # Check new feature coverage
    for f in NEW_FEATURES:
        if f in df_cv.columns:
            nn = df_cv[f].notna().sum()
            log(f"  {f}: {nn:,} non-null / {len(df_cv):,}  (mean={df_cv[f].mean():.3f})")

    loso_aucs = {n: [] for n in ("xswing","xcs","xwhiff","xfoul","bip")}
    cv_scores_list = []

    for test_season in COMMIT_SEASONS:
        train_seasons = [s for s in COMMIT_SEASONS if s != test_season]
        df_train = df_cv[df_cv["season"] == str(test_season).__class__(str(test_season))]
        # season is already str
        df_train = df_cv[~(df_cv["season"].astype(str) == str(test_season))]
        df_test  = df_cv[df_cv["season"].astype(str) == str(test_season)]
        log(f"\n  LOSO fold: test={test_season}  train={len(df_train):,}  test={len(df_test):,}")

        cat_feat_names = [c for c in CATEGORICAL_COLS if c in ALL_FEATURES]

        auc_sw, _, m_sw   = cv_binary_classifier(df_train, df_test, "is_swing", ALL_FEATURES, cat_feat_names, CLASSIFIER_PARAMS)
        auc_cs, _, m_cs   = cv_binary_classifier(df_train, df_test, "is_cs",    ALL_FEATURES, cat_feat_names, CLASSIFIER_PARAMS)
        auc_wh, _, m_wh   = cv_binary_classifier(df_train, df_test, "is_whiff", ALL_FEATURES, cat_feat_names, CLASSIFIER_PARAMS)
        auc_fo, _, m_fo   = cv_binary_classifier(df_train, df_test, "is_foul",  ALL_FEATURES, cat_feat_names, CLASSIFIER_PARAMS)
        _, _,    m_bip    = cv_bip_classifier(   df_train, df_test, ALL_FEATURES, cat_feat_names, CLASSIFIER_PARAMS, bip_universe)

        for name, auc in [("xswing",auc_sw),("xcs",auc_cs),("xwhiff",auc_wh),("xfoul",auc_fo)]:
            loso_aucs[name].append(auc)
            log(f"    {name} AUC = {auc:.4f}")

        # Score test fold for ERA/YoY analysis
        if all(m is not None for m in [m_sw, m_cs, m_wh, m_fo, m_bip]):
            X_te = df_test[ALL_FEATURES]
            xrv  = combine_xrv(X_te, {"xswing":m_sw,"xcs":m_cs,"xwhiff":m_wh,"xfoul":m_fo,"bip":m_bip},
                               rv_table, bip_classes)
            fold_scores = df_test[["pitcher_id","season"]].copy()
            fold_scores["xrv"] = xrv
            cv_scores_list.append(fold_scores)

    mean_aucs = {n: float(np.nanmean(v)) for n, v in loso_aucs.items() if v}
    log(f"\nMean LOSO AUCs: {mean_aucs}")

    # ── Full model training on CV seasons for scoring ────────────────────
    log("\nTraining full models on 2021-2025 for scoring...")
    cat_feat_names = [c for c in CATEGORICAL_COLS if c in ALL_FEATURES]

    def train_full(target, multiclass=False, bip_univ=None):
        sub = df_cv.dropna(subset=[target])
        X = sub[ALL_FEATURES]
        y = sub[target]
        if multiclass:
            m = lgb.LGBMClassifier(objective="multiclass", num_class=len(bip_univ),
                                    **CLASSIFIER_PARAMS)
        else:
            m = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
        m.fit(X, y, categorical_feature=cat_feat_names)
        return m

    m_sw  = train_full("is_swing")
    m_cs  = train_full("is_cs")
    m_wh  = train_full("is_whiff")
    m_fo  = train_full("is_foul")
    m_bip = train_full("bip_class", multiclass=True, bip_univ=bip_universe)
    log("Full models trained.")

    # ── Aggregate CV fold scores to pitcher-season ───────────────────────
    if cv_scores_list:
        df_cv_scores = pd.concat(cv_scores_list, ignore_index=True)
        pitcher_seasons = (
            df_cv_scores.groupby(["pitcher_id","season"])
            .agg(n=("xrv","size"), mean_xrv=("xrv","mean"))
            .reset_index()
        )
        pitcher_seasons = pitcher_seasons[pitcher_seasons["n"] >= MIN_PITCHES]
        # Plus-scale per season
        records = []
        for season, grp in pitcher_seasons.groupby("season"):
            vals = grp["mean_xrv"].values
            plus = to_plus_scale(vals, vals.mean(), vals.std())
            for i, (_, row) in enumerate(grp.iterrows()):
                records.append({"pitcher_id": row["pitcher_id"], "season": row["season"],
                                 "sequencing_plus": float(plus[i]), "n": int(row["n"])})
        df_ps = pd.DataFrame(records)
        log(f"\nCV pitcher-season scores: {len(df_ps):,}")
    else:
        df_ps = pd.DataFrame()

    # ── Validation metrics ───────────────────────────────────────────────
    log("\nComputing validation metrics...")

    # 1. Score spread
    if len(df_ps):
        spread = {
            "mean": float(df_ps["sequencing_plus"].mean()),
            "std":  float(df_ps["sequencing_plus"].std()),
            "p10":  float(np.percentile(df_ps["sequencing_plus"], 10)),
            "p90":  float(np.percentile(df_ps["sequencing_plus"], 90)),
        }
        log(f"  Score spread: mean={spread['mean']:.2f}  std={spread['std']:.2f}  p10={spread['p10']:.1f}  p90={spread['p90']:.1f}")
    else:
        spread = {}

    # 2. YoY stability
    yoy_pairs = []
    df_ps["season"] = df_ps["season"].astype(str)
    for i in range(len(COMMIT_SEASONS) - 1):
        s_t  = str(COMMIT_SEASONS[i])
        s_t1 = str(COMMIT_SEASONS[i + 1])
        df_t  = df_ps[df_ps["season"] == s_t]
        df_t1 = df_ps[df_ps["season"] == s_t1]
        merged = df_t.merge(df_t1, on="pitcher_id", suffixes=("_t","_t1"))
        if len(merged) > 30:
            r, _ = stats.pearsonr(merged["sequencing_plus_t"], merged["sequencing_plus_t1"])
            yoy_pairs.append({"season_t": s_t, "season_t1": s_t1, "n": len(merged), "r": round(float(r), 3)})
            log(f"  YoY {s_t}-{s_t1}: n={len(merged)}  r={r:.3f}")
    pooled_yoy = float(np.mean([p["r"] for p in yoy_pairs])) if yoy_pairs else None
    log(f"  Pooled YoY r = {pooled_yoy:.3f}" if pooled_yoy else "  YoY: insufficient data")

    # 3. Split-half reliability (odd vs even at-bats per pitcher-season within CV seasons)
    log("  Computing split-half reliability...")
    df_cv["is_odd_ab"] = (df_cv["at_bat_number"] % 2 == 1).astype(int) if "at_bat_number" in df_cv.columns else None

    split_half_r = None
    if "at_bat_number" in df_cv.columns:
        X_all = df_cv[ALL_FEATURES]
        xrv_all = combine_xrv(X_all,
                               {"xswing":m_sw,"xcs":m_cs,"xwhiff":m_wh,"xfoul":m_fo,"bip":m_bip},
                               rv_table, bip_classes)
        df_cv["xrv_full"] = xrv_all

        odd_ps  = df_cv[df_cv["is_odd_ab"] == 1].groupby(["pitcher_id","season"])["xrv_full"].mean().reset_index(name="odd")
        even_ps = df_cv[df_cv["is_odd_ab"] == 0].groupby(["pitcher_id","season"])["xrv_full"].mean().reset_index(name="even")
        sh = odd_ps.merge(even_ps, on=["pitcher_id","season"])
        sh = sh.groupby(["pitcher_id","season"]).size().reset_index()  # filter by count
        # re-merge
        sh = odd_ps.merge(even_ps, on=["pitcher_id","season"])
        if len(sh) > 30:
            r_sh, _ = stats.pearsonr(sh["odd"], sh["even"])
            split_half_r = float(r_sh)
            log(f"  Split-half r = {r_sh:.3f}  (n={len(sh)} pitcher-seasons)")

    # 4. Outcome validity (pitch-level: whiff%, swing%)
    if "xrv_full" not in df_cv.columns:
        X_all = df_cv[ALL_FEATURES]
        xrv_all = combine_xrv(X_all,
                               {"xswing":m_sw,"xcs":m_cs,"xwhiff":m_wh,"xfoul":m_fo,"bip":m_bip},
                               rv_table, bip_classes)
        df_cv["xrv_full"] = xrv_all

    # Pitcher-season: pred xRV vs real whiff%/swing%
    ps_outcomes = df_cv.groupby(["pitcher_id","season"]).agg(
        pred_xrv=("xrv_full","mean"),
        real_whiff=("is_whiff","mean"),
        real_swing=("is_swing","mean"),
        n=("is_whiff","size"),
    ).reset_index()
    ps_outcomes = ps_outcomes[ps_outcomes["n"] >= MIN_PITCHES]

    r_whiff, _  = stats.pearsonr(ps_outcomes["pred_xrv"], ps_outcomes["real_whiff"])
    r_swing, _  = stats.pearsonr(ps_outcomes["pred_xrv"], ps_outcomes["real_swing"])
    log(f"  xRV vs whiff%: r={r_whiff:.4f}   xRV vs swing%: r={r_swing:.4f}")

    # 5. ERA / xFIP
    conn2 = sqlite3.connect(DB_PATH)
    adv = pd.read_sql("""
        SELECT player_id AS pitcher_id, season, era, xfip
        FROM player_advanced_stats
        WHERE season BETWEEN 2021 AND 2025
          AND era IS NOT NULL AND xfip IS NOT NULL
    """, conn2)
    conn2.close()
    adv["season"] = adv["season"].astype(str)

    if len(df_ps) and len(adv):
        merged_era = df_ps.merge(adv, on=["pitcher_id","season"])
        if len(merged_era) > 30:
            r_era, _  = stats.pearsonr(merged_era["sequencing_plus"], merged_era["era"])
            r_xfip, _ = stats.pearsonr(merged_era["sequencing_plus"], merged_era["xfip"])
            log(f"  ERA r  = {r_era:.4f}  (n={len(merged_era)})")
            log(f"  xFIP r = {r_xfip:.4f}")
        else:
            r_era = r_xfip = None
            log("  ERA/xFIP: insufficient overlap")
    else:
        r_era = r_xfip = None
        merged_era = pd.DataFrame()

    # ── Tunneling+ v2 reference scores (same corpus, for comparison) ─────
    log("\nScoring Tunneling+ v2 on same 2021-2025 corpus for comparison...")
    v2_models = {n: lgb.Booster(model_file=str(V2_MODELS_DIR / f"tunneling_plus_v2_{n}.txt"))
                 for n in ("xswing","xcs","xwhiff","xfoul","bip")}
    v2_feat_cols = json.loads((V2_MODELS_DIR / "tunneling_plus_v2_meta.json").read_text())["feat_cols"]

    X_v2 = df_cv[v2_feat_cols]
    xrv_v2 = combine_xrv(X_v2,
                          {"xswing": v2_models["xswing"], "xcs": v2_models["xcs"],
                           "xwhiff": v2_models["xwhiff"], "xfoul": v2_models["xfoul"],
                           "bip":    v2_models["bip"]},
                          rv_table, bip_classes)
    df_cv["xrv_v2"] = xrv_v2

    ps_v2 = df_cv.groupby(["pitcher_id","season"]).agg(
        pred_xrv_v2=("xrv_v2","mean"), n=("xrv_v2","size")
    ).reset_index()
    ps_v2 = ps_v2[ps_v2["n"] >= MIN_PITCHES]

    v2_records = []
    for season, grp in ps_v2.groupby("season"):
        vals = grp["pred_xrv_v2"].values
        plus = to_plus_scale(vals, vals.mean(), vals.std())
        for i, (_, row) in enumerate(grp.iterrows()):
            v2_records.append({"pitcher_id": row["pitcher_id"], "season": row["season"],
                                "v2_plus": float(plus[i])})
    df_v2_ps = pd.DataFrame(v2_records)
    log(f"  v2 pitcher-seasons: {len(df_v2_ps):,}")

    # v2 ERA
    if len(adv):
        v2_era = df_v2_ps.merge(adv, on=["pitcher_id","season"])
        v2_r_era  = float(stats.pearsonr(v2_era["v2_plus"], v2_era["era"])[0])  if len(v2_era) > 30 else None
        v2_r_xfip = float(stats.pearsonr(v2_era["v2_plus"], v2_era["xfip"])[0]) if len(v2_era) > 30 else None
        log(f"  v2 ERA r = {v2_r_era:.4f}   xFIP r = {v2_r_xfip:.4f}")
    else:
        v2_r_era = v2_r_xfip = None

    # v2 YoY
    v2_yoy_pairs = []
    for i in range(len(COMMIT_SEASONS) - 1):
        s_t  = str(COMMIT_SEASONS[i])
        s_t1 = str(COMMIT_SEASONS[i + 1])
        df_t  = df_v2_ps[df_v2_ps["season"] == s_t]
        df_t1 = df_v2_ps[df_v2_ps["season"] == s_t1]
        merged_yoy = df_t.merge(df_t1, on="pitcher_id", suffixes=("_t","_t1"))
        if len(merged_yoy) > 30:
            r_yoy, _ = stats.pearsonr(merged_yoy["v2_plus_t"], merged_yoy["v2_plus_t1"])
            v2_yoy_pairs.append(float(r_yoy))
    v2_pooled_yoy = float(np.mean(v2_yoy_pairs)) if v2_yoy_pairs else None
    log(f"  v2 pooled YoY r = {v2_pooled_yoy:.3f}" if v2_pooled_yoy else "  v2 YoY: insufficient data")

    # v2 classifier AUCs (on the 2021-2025 data for reference)
    log("\nComputing v2 reference AUCs on full 2021-2025 corpus...")
    v2_ref_aucs = {}
    for name, col in [("xswing","is_swing"),("xcs","is_cs"),("xwhiff","is_whiff"),("xfoul","is_foul")]:
        sub = df_cv.dropna(subset=[col])
        prob = v2_models[name].predict(sub[v2_feat_cols])
        y = sub[col].astype(int)
        if len(y.unique()) > 1:
            v2_ref_aucs[name] = float(roc_auc_score(y, prob))
        log(f"  v2 {name} AUC = {v2_ref_aucs.get(name, 'n/a')}")

    # Seq+ classifier AUCs on full 2021-2025 corpus for comparison
    seq_full_aucs = {}
    for name, col, model in [("xswing","is_swing",m_sw),("xcs","is_cs",m_cs),("xwhiff","is_whiff",m_wh),("xfoul","is_foul",m_fo)]:
        sub = df_cv.dropna(subset=[col])
        prob = model.predict_proba(sub[ALL_FEATURES])[:, 1]
        y = sub[col].astype(int)
        if len(y.unique()) > 1:
            seq_full_aucs[name] = float(roc_auc_score(y, prob))
        log(f"  Seq+ {name} AUC = {seq_full_aucs.get(name, 'n/a')}")

    conn.close()

    # ── Results summary ──────────────────────────────────────────────────
    log("\n=== SIDE-BY-SIDE: Sequencing+ v1 vs Tunneling+ v2 ===")
    log(f"  {'Metric':30s} {'Seq+ v1':>10} {'Tun+ v2':>10}")
    log(f"  {'-'*52}")

    def sfmt(v): return f"{v:+.3f}" if v is not None else "   n/a"
    log(f"  {'ERA Pearson r':30s} {sfmt(r_era if len(merged_era)>30 else None):>10} {sfmt(v2_r_era):>10}")
    log(f"  {'xFIP Pearson r':30s} {sfmt(r_xfip if len(merged_era)>30 else None):>10} {sfmt(v2_r_xfip):>10}")
    log(f"  {'YoY stability (pooled r)':30s} {sfmt(pooled_yoy):>10} {sfmt(v2_pooled_yoy):>10}")
    for name in ["xswing","xcs","xwhiff","xfoul"]:
        loso_m = float(np.mean(loso_aucs[name])) if loso_aucs[name] else None
        log(f"  {'LOSO AUC '+name:30s} {sfmt(loso_m):>10} {'  ref:':>10}")

    # ── Save output ──────────────────────────────────────────────────────
    output = {
        "model_name": "sequencing_plus_v1",
        "note": "DIAGNOSTIC ONLY — provisional name; not promoted to production",
        "feature_set": {
            "kept_from_v2": list(ALL_CAT_V2),
            "dropped_from_v2": ["lagN_commit_distance", "lagN_plate_distance", "lagN_shape_break (all 6 lags)"],
            "added": NEW_FEATURES,
            "excluded": ["count (balls/strikes)", "plate location (plate_x/plate_z)"],
        },
        "cv_seasons": COMMIT_SEASONS,
        "n_pitcher_seasons": int(len(df_ps)),
        "score_spread": spread,
        "loso_mean_aucs": mean_aucs,
        "v2_reference_full_corpus_aucs": v2_ref_aucs,
        "seq_full_corpus_aucs": seq_full_aucs,
        "yoy_stability": {
            "pooled_r": round(pooled_yoy, 4) if pooled_yoy else None,
            "per_pair": yoy_pairs,
        },
        "v2_yoy_stability": {
            "pooled_r": round(v2_pooled_yoy, 4) if v2_pooled_yoy else None,
        },
        "split_half_r": round(split_half_r, 4) if split_half_r else None,
        "era_correlation": round(float(r_era), 4) if r_era else None,
        "xfip_correlation": round(float(r_xfip), 4) if r_xfip else None,
        "era_n": int(len(merged_era)) if len(merged_era) else 0,
        "v2_era_correlation": round(v2_r_era, 4) if v2_r_era else None,
        "v2_xfip_correlation": round(v2_r_xfip, 4) if v2_r_xfip else None,
        "pitch_level_validity": {
            "xrv_vs_whiff_r": round(float(r_whiff), 4),
            "xrv_vs_swing_r": round(float(r_swing), 4),
        },
        "run_date": datetime.now().strftime("%Y-%m-%d"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {OUT_PATH}")
    _log_fh.close()


if __name__ == "__main__":
    main()
