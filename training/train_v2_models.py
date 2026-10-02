"""
Location+ v2 and Tunneling+ v2 — decomposed event-classifier architecture.
DIAGNOSTIC BUILDS ONLY — do NOT touch models/location_plus_v1.txt or
models/tunneling_v3.txt. Save artifacts under v2 naming only.

Architecture (FanGraphs PitchingBot methodology):
  Five classifiers predict discrete-event probabilities:
    1. xSwing%               — P(swing | pitch features)
    2. xCalledStrike%        — P(called_strike | no swing, pitch features)
    3. xWhiff%               — P(miss | swing, pitch features)
    4. xFoul%                — P(foul | contact, pitch features)
    5. BIP type × EV bucket  — multiclass P(bb_type × ev_bucket | BIP)
  Combine with a FIXED run-value table (Part 0) to produce per-pitch xRV.
  Aggregate per pitcher-season, z-score, apply to_plus_scale.

Progress logs written to training/v2_models_progress.log after each stage.
Checkpoint files in training/v2_checkpoints/ — resume from last completed.

Usage:
    python training/train_v2_models.py [--skip-part0] [--skip-partA]
                                       [--skip-partB] [--only-partC]
"""
import argparse
import datetime
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, log_loss, r2_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analytics.location_features import (
    ALL_FEATURES as LOC_FEATURES,
    CATEGORICAL_FEATURES as LOC_CAT_FEATURES,
    RAW_QUERY_COLUMNS as LOC_RAW_COLS,
    engineer_features as loc_engineer_features,
    to_plus_scale,
)
from analytics.tunneling_features import (
    RAW_QUERY_COLUMNS as TUN_RAW_COLS,
    build_commit_point_table,
    lag_feature_columns,
    load_pitches as tun_load_pitches,
    sequential_lag_features,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "mlb_data.db"
COMPLETE_SEASONS = list(range(2015, 2026))  # 2015–2025
ALL_SEASONS = COMPLETE_SEASONS + [2026]
MIN_PITCHES = 100  # qualifier for plus-scale (matches live Location+)

ARTIFACTS_DIR = Path("training/v2_artifacts")
CHECKPOINT_DIR = Path("training/v2_checkpoints")
LOG_PATH = Path("training/v2_models_progress.log")

# Event description sets (built from the real description frequencies in the DB)
SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip",
    "missed_bunt", "foul", "foul_bunt", "foul_pitchout",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "swinging_pitchout",
}
WHIFF_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "bunt_foul_tip", "missed_bunt", "swinging_pitchout",
}
FOUL_DESCS = {"foul", "foul_bunt", "foul_pitchout"}
BIP_DESCS = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
CALLED_STRIKE_DESCS = {"called_strike", "automatic_strike"}
BALL_DESCS = {
    "ball", "blocked_ball", "pitchout", "automatic_ball", "intent_ball",
    "pitchout_hit_into_play", "pitchout_hit_into_play_score",
}
HBP_DESCS = {"hit_by_pitch"}

# BIP exit-velo bucket boundaries (mph)
EV_BINS = [0, 90, 95, 100, 105, 999]
EV_LABELS = ["ev_lt90", "ev_90_95", "ev_95_100", "ev_100_105", "ev_105plus"]

# LightGBM params — shared across classifiers; num_leaves kept small since
# the core signal for each individual-event classifier is shallow.
BINARY_PARAMS = dict(
    objective="binary", n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=200, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
    metric="binary_logloss",
)
MULTI_PARAMS = dict(
    objective="multiclass", n_estimators=400, learning_rate=0.05,
    num_leaves=31, min_child_samples=100, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
    metric="multi_logloss",
)
# Max rows used for CV training per fold (diagnostic purposes — 1M gives
# stable AUC estimates without running 7M-row fits 11 times per classifier)
CV_TRAIN_MAX_ROWS = 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING & CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str, also_print: bool = True):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def checkpoint_done(name: str) -> bool:
    return (CHECKPOINT_DIR / f"{name}.done").exists()


def mark_checkpoint(name: str, data: dict = None):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = CHECKPOINT_DIR / f"{name}.done"
    p.write_text(json.dumps(data or {}, indent=2))
    log(f"  [DONE] Checkpoint saved: {name}")


def load_checkpoint(name: str) -> dict:
    p = CHECKPOINT_DIR / f"{name}.done"
    return json.loads(p.read_text()) if p.exists() else {}


# ─────────────────────────────────────────────────────────────────────────────
# EVENT-LABEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def assign_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary event columns for classifier training."""
    desc = df["description"].fillna("")
    df = df.copy()
    df["is_swing"] = desc.isin(SWING_DESCS).astype(int)
    df["is_no_swing"] = (~desc.isin(SWING_DESCS)).astype(int)
    df["is_whiff"] = desc.isin(WHIFF_DESCS).astype(int)
    df["is_foul"] = desc.isin(FOUL_DESCS).astype(int)
    df["is_bip"] = desc.isin(BIP_DESCS).astype(int)
    df["is_called_strike"] = desc.isin(CALLED_STRIKE_DESCS).astype(int)
    df["is_ball"] = desc.isin(BALL_DESCS).astype(int)
    df["is_hbp"] = desc.isin(HBP_DESCS).astype(int)
    # contact = swing and not whiff
    df["is_contact"] = ((df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
    return df


def make_bip_class(df: pd.DataFrame) -> pd.Series:
    """BIP class label = bb_type + "_" + ev_bucket, for BIP rows."""
    ev = pd.cut(
        df["launch_speed"].astype(float),
        bins=EV_BINS, labels=EV_LABELS, right=False,
    ).astype(str)
    return df["bb_type"].fillna("unknown") + "__" + ev


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def coerce_cats(train_df, val_df, cat_cols):
    for col in cat_cols:
        cats = sorted(set(train_df[col].dropna()) | set(val_df[col].dropna()))
        train_df[col] = pd.Categorical(train_df[col], categories=cats)
        val_df[col] = pd.Categorical(val_df[col], categories=cats)
    return train_df, val_df


def fit_binary(X_tr, y_tr, X_val, y_val, cat_cols):
    model = lgb.LGBMClassifier(**BINARY_PARAMS)
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def fit_multiclass(X_tr, y_tr, X_val, y_val, cat_cols, n_classes):
    params = {**MULTI_PARAMS, "num_class": n_classes}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def cv_binary(df, label_col, feat_cols, cat_cols, name):
    """Leave-one-season-out CV. Categoricals must already be set on df.
    Training capped at CV_TRAIN_MAX_ROWS with simple random sample."""
    folds = []
    for holdout in COMPLETE_SEASONS:
        mask = df["season"] == str(holdout)
        tr = df[~mask]
        te = df[mask]
        if len(te) < 100 or te[label_col].nunique() < 2:
            continue
        if len(tr) > CV_TRAIN_MAX_ROWS:
            tr = tr.sample(CV_TRAIN_MAX_ROWS, random_state=42)
        tr_fit, tr_stop = train_test_split(tr, test_size=0.08, random_state=42)
        model = fit_binary(tr_fit[feat_cols], tr_fit[label_col],
                           tr_stop[feat_cols], tr_stop[label_col], cat_cols)
        probs = model.predict_proba(te[feat_cols])[:, 1]
        preds = (probs >= 0.5).astype(int)
        try:
            auc = roc_auc_score(te[label_col], probs)
        except Exception:
            auc = float("nan")
        ll = log_loss(te[label_col], probs)
        acc = accuracy_score(te[label_col], preds)
        folds.append({"holdout": holdout, "auc": auc, "log_loss": ll, "acc": acc, "n": len(te)})
        log(f"    {name} holdout={holdout}: AUC={auc:.4f} LogLoss={ll:.4f} Acc={acc:.4f} n={len(te):,}")
    return folds


def cv_multiclass(df, label_col, feat_cols, cat_cols, class_order, name):
    """Leave-one-season-out CV. Categoricals must already be set on df."""
    folds = []
    n_classes = len(class_order)
    for holdout in COMPLETE_SEASONS:
        mask = df["season"] == str(holdout)
        tr = df[~mask]
        te = df[mask]
        if len(te) < 100 or te[label_col].nunique() < 2:
            continue
        if len(tr) > CV_TRAIN_MAX_ROWS:
            tr = tr.sample(CV_TRAIN_MAX_ROWS, random_state=42)
        tr_codes = pd.Categorical(tr[label_col], categories=class_order).codes
        te_codes = pd.Categorical(te[label_col], categories=class_order).codes
        tr_fit_idx, tr_stop_idx = train_test_split(range(len(tr)), test_size=0.08, random_state=42)
        tr_arr = tr.iloc[tr_fit_idx]; tr_stop_arr = tr.iloc[tr_stop_idx]
        tr_codes_fit = tr_codes[list(tr_fit_idx)]; tr_codes_stop = tr_codes[list(tr_stop_idx)]
        model = fit_multiclass(tr_arr[feat_cols], tr_codes_fit,
                               tr_stop_arr[feat_cols], tr_codes_stop,
                               cat_cols, n_classes)
        proba = model.predict_proba(te[feat_cols])
        ll = log_loss(te_codes, proba, labels=list(range(n_classes)))
        acc = accuracy_score(te_codes, proba.argmax(axis=1))
        folds.append({"holdout": holdout, "log_loss": ll, "acc": acc, "n": len(te)})
        log(f"    {name} holdout={holdout}: LogLoss={ll:.4f} Acc={acc:.4f} n={len(te):,}")
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# PART 0 — Run-value table
# ─────────────────────────────────────────────────────────────────────────────
def build_run_value_table(conn) -> dict:
    """Compute context-neutral average delta_run_exp per event type.
    Returns dict: {event_type_str: float rv}."""
    log("PART 0: Building run-value table...")

    df = pd.concat(pd.read_sql("""
        SELECT description, delta_run_exp, bb_type, launch_speed
        FROM pitches
        WHERE delta_run_exp IS NOT NULL AND season BETWEEN 2015 AND 2026
    """, conn, chunksize=100_000), ignore_index=True)
    df = assign_event_labels(df)

    rv = {}

    # Terminal swing outcomes
    whiff_rows = df[df["is_whiff"] == 1]
    rv["whiff"] = float(whiff_rows["delta_run_exp"].mean())

    foul_rows = df[df["is_foul"] == 1]
    rv["foul"] = float(foul_rows["delta_run_exp"].mean())

    # No-swing outcomes
    cs_rows = df[df["is_called_strike"] == 1]
    rv["called_strike"] = float(cs_rows["delta_run_exp"].mean())

    ball_rows = df[df["is_ball"] == 1]
    rv["ball"] = float(ball_rows["delta_run_exp"].mean())

    hbp_rows = df[df["is_hbp"] == 1]
    rv["hbp"] = float(hbp_rows["delta_run_exp"].mean())

    # BIP: bb_type × exit-velo bucket
    bip_rows = df[(df["is_bip"] == 1) & df["bb_type"].notna() & df["launch_speed"].notna()].copy()
    bip_rows["bip_class"] = make_bip_class(bip_rows)
    bip_rv = bip_rows.groupby("bip_class")["delta_run_exp"].mean().to_dict()
    rv.update({k: float(v) for k, v in bip_rv.items()})

    # Normalize: subtract the global weighted mean so the expected xRV across
    # all pitches is zero (count-sensitivity lives in probability models).
    # Compute weighted mean using empirical event frequencies.
    event_counts = {
        "whiff": len(whiff_rows), "foul": len(foul_rows),
        "called_strike": len(cs_rows), "ball": len(ball_rows), "hbp": len(hbp_rows),
    }
    event_counts.update({k: int((bip_rows["bip_class"] == k).sum()) for k in bip_rv})
    total = sum(event_counts.values())
    global_mean = sum(rv[k] * event_counts[k] / total for k in rv)
    rv = {k: v - global_mean for k, v in rv.items()}

    log(f"  Run-value table entries: {len(rv)}")
    log("  Key event values (normalized):")
    for k in ["whiff", "called_strike", "ball", "foul", "hbp"]:
        log(f"    {k}: {rv.get(k, float('nan')):.4f}")
    log("  BIP values (bb_type × EV bucket):")
    for k, v in sorted(bip_rv.items()):
        raw_n = int((bip_rows["bip_class"] == k).sum())
        log(f"    {k}: {v - global_mean:.4f} (n={raw_n:,})")

    # Report EV bucket distribution
    bip_rows["ev_bucket"] = pd.cut(bip_rows["launch_speed"], bins=EV_BINS, labels=EV_LABELS, right=False)
    bucket_counts = bip_rows.groupby("ev_bucket", observed=True).size()
    log("  EV bucket distribution (BIP pitches):")
    for bucket, cnt in bucket_counts.items():
        pct = cnt / len(bip_rows) * 100
        log(f"    {bucket}: {cnt:,} ({pct:.1f}%)")

    return rv, bip_rows["bip_class"].unique().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# COMBINATION FORMULA
# ─────────────────────────────────────────────────────────────────────────────
def combine_xrv(
    df,
    p_swing, p_whiff_given_swing, p_foul_given_contact, p_cs_given_no_swing,
    p_bip_proba, bip_classes, rv_table, hbp_rate_no_swing,
):
    """Combine event probabilities + run-value table into per-pitch xRV."""
    # xRV given swing
    rv_whiff = rv_table.get("whiff", 0.0)
    rv_foul = rv_table.get("foul", 0.0)

    # BIP expected run value = sum over classes: P(class) * RV(class)
    bip_rvs = np.array([rv_table.get(c, 0.0) for c in bip_classes])
    xrv_bip = p_bip_proba @ bip_rvs  # shape [n_pitches]

    xrv_contact = p_foul_given_contact * rv_foul + (1 - p_foul_given_contact) * xrv_bip
    xrv_given_swing = p_whiff_given_swing * rv_whiff + (1 - p_whiff_given_swing) * xrv_contact

    # xRV given no swing
    rv_cs = rv_table.get("called_strike", 0.0)
    rv_ball = rv_table.get("ball", 0.0)
    rv_hbp = rv_table.get("hbp", 0.0)
    p_ball_given_no_swing = 1 - p_cs_given_no_swing - hbp_rate_no_swing
    xrv_given_no_swing = (p_cs_given_no_swing * rv_cs
                          + hbp_rate_no_swing * rv_hbp
                          + p_ball_given_no_swing * rv_ball)

    return p_swing * xrv_given_swing + (1 - p_swing) * xrv_given_no_swing


# ─────────────────────────────────────────────────────────────────────────────
# PLUS-SCALE AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_plus_scores(df, min_pitches=MIN_PITCHES):
    """Per pitcher-season: mean xRV -> plus-scale score."""
    results = []
    for season, sg in df.groupby("season"):
        g = sg.groupby("pitcher_id").agg(
            mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size")
        ).reset_index()
        q = g[g["n_pitches"] >= min_pitches].copy()
        if len(q) < 10:
            continue
        league_mean = q["mean_xrv"].mean()
        league_std = q["mean_xrv"].std()
        q["plus_score"] = to_plus_scale(q["mean_xrv"], league_mean, league_std)
        q["season"] = season
        results.append(q)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PART A — Location+ v2
# ─────────────────────────────────────────────────────────────────────────────
def run_part_a(conn, rv_table, bip_classes):
    log("\n" + "=" * 70)
    log("PART A: Location+ v2 — event-classifier architecture")
    log("=" * 70)

    # ── Load pitches ─────────────────────────────────────────────────────────
    # Two queries: lean query for all pitches (location features + description
    # for event labels), then a narrow BIP query for bb_type/launch_speed.
    # Pulling bb_type/launch_speed for all 8M rows is slow and wasteful —
    # only ~1.4M rows are BIPs and actually have those values.
    log("Loading location pitches (lean query, chunked)...")
    from analytics.stuff_features import exclude_junk_pitch_types, merge_pitch_type
    raw = pd.concat(pd.read_sql(f"""
        SELECT {', '.join(LOC_RAW_COLS)}, description
        FROM pitches
        WHERE season BETWEEN 2015 AND 2026
    """, conn, chunksize=100_000), ignore_index=True)
    raw = merge_pitch_type(raw)
    raw = exclude_junk_pitch_types(raw)
    raw["season"] = raw["season"].astype(str)
    for col in ["plate_x", "plate_z", "balls", "strikes"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["plate_x", "plate_z", "balls", "strikes", "pitch_type", "stand"])
    raw = assign_event_labels(raw)
    log(f"  {len(raw):,} pitches loaded")

    # BIP columns loaded separately — only the rows that actually have them
    log("Loading BIP rows (bb_type + launch_speed, chunked)...")
    bip_extra = pd.concat(pd.read_sql("""
        SELECT pitcher_id, season, pitch_type, stand, plate_x, plate_z,
               balls, strikes, bb_type, launch_speed
        FROM pitches
        WHERE season BETWEEN 2015 AND 2026
          AND bb_type IS NOT NULL AND launch_speed IS NOT NULL
    """, conn, chunksize=100_000), ignore_index=True)
    bip_extra["pitch_type"] = bip_extra["pitch_type"].replace({"KC": "CU"})
    bip_extra = bip_extra[~bip_extra["pitch_type"].isin({"PO", "UN", "EP"})]
    bip_extra["season"] = bip_extra["season"].astype(str)
    log(f"  {len(bip_extra):,} BIP rows loaded")

    feat_cols = ["plate_x", "plate_z", "balls", "strikes", "pitch_type", "stand", "season"]
    cat_cols = ["pitch_type", "stand", "season"]

    # ── HBP rate for combination ──────────────────────────────────────────────
    no_swing_rows = raw[raw["is_no_swing"] == 1]
    hbp_rate_no_swing = float(no_swing_rows["is_hbp"].mean())
    log(f"  League HBP rate given no swing: {hbp_rate_no_swing:.4f}")

    # ── Coerce categoricals ONCE on the full dataset ──────────────────────────
    # Done once here rather than per-fold to avoid repeated 8M-row operations.
    log("  Coercing categoricals...")
    all_cats = {col: sorted(set(raw[col].dropna())) for col in cat_cols}
    for col in cat_cols:
        raw[col] = pd.Categorical(raw[col], categories=all_cats[col])
    log("  Done. Building subsets...")

    # ── Build BIP class labels for classifier 5 ───────────────────────────────
    bip_rows = bip_extra.copy()
    # Apply same categorical treatment to BIP subset
    for col in cat_cols:
        if col in bip_rows.columns:
            bip_rows[col] = pd.Categorical(bip_rows[col], categories=all_cats[col])
    bip_rows["bip_class"] = make_bip_class(bip_rows)
    valid_classes = [c for c in bip_classes if c in bip_rows["bip_class"].values]
    bip_rows = bip_rows[bip_rows["bip_class"].isin(valid_classes)].copy()
    bip_rows["bip_class_code"] = pd.Categorical(bip_rows["bip_class"], categories=valid_classes).codes
    log(f"  BIP rows for classifier 5: {len(bip_rows):,}, classes: {len(valid_classes)}")

    # ── Build subsets (views, not copies — categoricals already set) ──────────
    raw_c = raw
    no_sw_c = raw[raw["is_no_swing"] == 1]
    sw_c = raw[raw["is_swing"] == 1]
    contact_c = raw[raw["is_contact"] == 1]
    bip_c = bip_rows

    cv_results = {}

    # ── Classifier 1: xSwing% ────────────────────────────────────────────────
    log("\nClassifier 1 — xSwing% (binary, all pitches):")
    cv_results["loc_xswing"] = cv_binary(raw_c, "is_swing", feat_cols, cat_cols, "xSwing%")

    # ── Classifier 4: xCalledStrike% given no swing ───────────────────────────
    log("\nClassifier 4 — xCalledStrike% (binary, no-swing pitches only):")
    cv_results["loc_xcs"] = cv_binary(no_sw_c, "is_called_strike", feat_cols, cat_cols, "xCS%")

    # ── Classifier 2: xWhiff% given swing ────────────────────────────────────
    log("\nClassifier 2 — xWhiff% (binary, swing pitches only):")
    cv_results["loc_xwhiff"] = cv_binary(sw_c, "is_whiff", feat_cols, cat_cols, "xWhiff%")

    # ── Classifier 3: xFoul% given contact ───────────────────────────────────
    log("\nClassifier 3 — xFoul% (binary, contact pitches only):")
    cv_results["loc_xfoul"] = cv_binary(contact_c, "is_foul", feat_cols, cat_cols, "xFoul%")

    # ── Classifier 5: BIP distribution (multiclass) ──────────────────────────
    log(f"\nClassifier 5 — BIP type×EV (multiclass {len(valid_classes)} classes):")
    cv_results["loc_bip"] = cv_multiclass(
        bip_c, "bip_class", feat_cols, cat_cols, valid_classes, "BIP-dist"
    )

    # ── Print CV summary ──────────────────────────────────────────────────────
    for name, folds in cv_results.items():
        if not folds:
            continue
        if "auc" in folds[0]:
            mean_auc = np.mean([f["auc"] for f in folds if not np.isnan(f["auc"])])
            mean_ll = np.mean([f["log_loss"] for f in folds])
            log(f"  CV summary {name}: mean AUC={mean_auc:.4f} mean LogLoss={mean_ll:.4f}")
        else:
            mean_ll = np.mean([f["log_loss"] for f in folds])
            mean_acc = np.mean([f["acc"] for f in folds])
            log(f"  CV summary {name}: mean LogLoss={mean_ll:.4f} mean Acc={mean_acc:.4f}")

    # ── Train final models on all seasons (full data, not subsampled) ─────────
    log("\nTraining final Location+ v2 models on all seasons...")

    def fit_final_binary(subset_df, label_col, name_):
        tr_fit, tr_stop = train_test_split(subset_df, test_size=0.08, random_state=42)
        model = fit_binary(tr_fit[feat_cols], tr_fit[label_col],
                           tr_stop[feat_cols], tr_stop[label_col], cat_cols)
        log(f"  {name_} best_iteration={model.best_iteration_}")
        return model

    m_swing = fit_final_binary(raw_c, "is_swing", "xSwing%")
    m_cs = fit_final_binary(no_sw_c, "is_called_strike", "xCS%")
    m_whiff = fit_final_binary(sw_c, "is_whiff", "xWhiff%")
    m_foul = fit_final_binary(contact_c, "is_foul", "xFoul%")

    # BIP multiclass final — use integer codes directly
    bip_final = bip_c.copy()
    bip_final["bip_class_code"] = pd.Categorical(bip_final["bip_class"], categories=valid_classes).codes
    tr_bip, stop_bip = train_test_split(bip_final, test_size=0.08, random_state=42)
    m_bip = fit_multiclass(tr_bip[feat_cols], tr_bip["bip_class_code"],
                           stop_bip[feat_cols], stop_bip["bip_class_code"],
                           cat_cols, len(valid_classes))
    log(f"  BIP multiclass best_iteration={m_bip.best_iteration_}")

    # ── Score all pitches ────────────────────────────────────────────────────
    log("\nScoring all pitches with Location+ v2...")

    def predict_on(model, df_):
        return model.predict_proba(df_[feat_cols])[:, 1]

    p_swing = predict_on(m_swing, raw_c)
    p_whiff = predict_on(m_whiff, raw_c)
    p_foul = predict_on(m_foul, raw_c)
    p_cs = predict_on(m_cs, raw_c)
    p_bip = m_bip.predict_proba(raw_c[feat_cols])  # [n, n_classes]

    xrv = combine_xrv(
        raw_c, p_swing, p_whiff, p_foul, p_cs,
        p_bip, valid_classes, rv_table, hbp_rate_no_swing,
    )
    raw_c["xrv"] = xrv
    raw_c["pitcher_id"] = raw["pitcher_id"].values

    log(f"  xRV distribution: mean={xrv.mean():.4f} std={xrv.std():.4f}")
    log(f"  xRV p5={np.percentile(xrv,5):.4f} p95={np.percentile(xrv,95):.4f}")

    # ── Aggregate per pitcher-season ─────────────────────────────────────────
    scores = aggregate_plus_scores(raw_c)
    log(f"  Location+ v2 scores: {len(scores):,} pitcher-seasons, "
        f"std={scores['plus_score'].std():.2f}, "
        f"min={scores['plus_score'].min():.1f}, max={scores['plus_score'].max():.1f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(ARTIFACTS_DIR / "location_plus_v2_scores.parquet", index=False)
    raw_c[["pitcher_id", "season", "xrv"]].to_parquet(
        ARTIFACTS_DIR / "location_plus_v2_per_pitch.parquet", index=False
    )
    m_swing.booster_.save_model(str(ARTIFACTS_DIR / "location_plus_v2_xswing.txt"))
    m_cs.booster_.save_model(str(ARTIFACTS_DIR / "location_plus_v2_xcs.txt"))
    m_whiff.booster_.save_model(str(ARTIFACTS_DIR / "location_plus_v2_xwhiff.txt"))
    m_foul.booster_.save_model(str(ARTIFACTS_DIR / "location_plus_v2_xfoul.txt"))
    m_bip.booster_.save_model(str(ARTIFACTS_DIR / "location_plus_v2_bip.txt"))

    meta = dict(
        feat_cols=feat_cols, cat_cols=cat_cols, bip_classes=valid_classes,
        hbp_rate_no_swing=hbp_rate_no_swing,
        cv_results={k: v for k, v in cv_results.items()},
        trained_at=datetime.datetime.now().isoformat(),
        seasons=ALL_SEASONS,
    )
    (ARTIFACTS_DIR / "location_plus_v2_meta.json").write_text(json.dumps(meta, indent=2))

    log(f"\nPart A artifacts saved to {ARTIFACTS_DIR}/")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# PART B — Tunneling+ v2
# ─────────────────────────────────────────────────────────────────────────────
def run_part_b(conn, rv_table, bip_classes):
    log("\n" + "=" * 70)
    log("PART B: Tunneling+ v2 — event-classifier architecture")
    log("=" * 70)
    log("NOTE: Input features deliberately EXCLUDE plate_x/plate_z.")
    log("xSwing% and xCalledStrike% are EXPECTED to show weaker signal here")
    log("(swing/take decisions are primarily location-driven).")
    log("xWhiff% is expected to show the strongest genuine tunneling signal.")

    # ── Load pitches and build lag features ───────────────────────────────────
    log("\nLoading tunneling pitches and building lag features...")
    log("(This is the expensive step — building sequential lag features for all seasons)")

    all_lag_dfs = []
    for season in ALL_SEASONS:
        t0_s = time.time()
        df_s = tun_load_pitches(conn, season)
        if len(df_s) == 0:
            log(f"  Season {season}: no pitches, skipping")
            continue
        commit_tbl = build_commit_point_table(df_s)
        if len(commit_tbl) == 0:
            log(f"  Season {season}: no commit table, skipping")
            continue
        lag_df = sequential_lag_features(df_s, commit_tbl)
        lag_df["season"] = str(season)
        lag_df["delta_run_exp"] = lag_df["delta_run_exp"] if "delta_run_exp" in lag_df.columns else None
        all_lag_dfs.append(lag_df)
        elapsed = time.time() - t0_s
        log(f"  Season {season}: {len(lag_df):,} lag-feature rows ({elapsed:.0f}s)")

    if not all_lag_dfs:
        log("ERROR: No lag feature rows built — aborting Part B")
        return None

    df = pd.concat(all_lag_dfs, ignore_index=True)
    log(f"  Total lag-feature rows: {len(df):,}")

    # Add event labels
    df = assign_event_labels(df)

    # Merge bb_type + launch_speed for Classifier 5 (not in TUN_RAW_COLS)
    log("  Merging bb_type/launch_speed for BIP classifier...")
    bip_meta = pd.concat(pd.read_sql("""
        SELECT game_id, at_bat_number, pitch_number, bb_type, launch_speed
        FROM pitches
        WHERE season BETWEEN 2015 AND 2026
          AND bb_type IS NOT NULL AND launch_speed IS NOT NULL
    """, conn, chunksize=100_000), ignore_index=True)
    df = df.merge(bip_meta, on=["game_id", "at_bat_number", "pitch_number"], how="left")
    log(f"  After BIP merge: {df['bb_type'].notna().sum():,} rows have bb_type")

    # Feature set: lag features + current pitch categoricals
    lag_cols = lag_feature_columns()  # 24 lag-derived columns
    lag_num_cols = [c for c in lag_cols if not c.endswith("pitch_type")]
    lag_cat_cols = [c for c in lag_cols if c.endswith("pitch_type")]
    curr_cat_cols = ["pitch_type", "p_throws", "stand", "season"]
    feat_cols = lag_num_cols + lag_cat_cols + curr_cat_cols
    cat_cols = lag_cat_cols + curr_cat_cols

    # Restrict to rows where we have at least lag1 features
    df = df.dropna(subset=["lag1_commit_distance", "lag1_plate_distance"])
    log(f"  After lag1 filter: {len(df):,} rows")

    # ── HBP rate ──────────────────────────────────────────────────────────────
    no_sw = df[df["is_no_swing"] == 1]
    hbp_rate_no_swing = float(no_sw["is_hbp"].mean()) if len(no_sw) > 0 else 0.01

    # ── Coerce categoricals ───────────────────────────────────────────────────
    all_cat_vals = {}
    for col in cat_cols:
        all_cat_vals[col] = sorted(set(df[col].dropna().astype(str)))

    def prep_tun_cats(df_):
        for col in cat_cols:
            df_[col] = df_[col].astype(str)
            df_[col] = pd.Categorical(df_[col], categories=all_cat_vals[col])
        return df_

    df_c = prep_tun_cats(df.copy())

    # Build subsets
    no_sw_c = prep_tun_cats(df[df["is_no_swing"] == 1].copy())
    sw_c = prep_tun_cats(df[df["is_swing"] == 1].copy())
    contact_c = prep_tun_cats(df[df["is_contact"] == 1].copy())

    # BIP subset with class labels
    bip_rows = df[(df["is_bip"] == 1) & df["bb_type"].notna() & df["launch_speed"].notna()].copy()
    bip_rows["bip_class"] = make_bip_class(bip_rows)
    valid_bip_classes = [c for c in bip_classes if c in bip_rows["bip_class"].values]
    bip_rows = bip_rows[bip_rows["bip_class"].isin(valid_bip_classes)].copy()
    bip_rows["bip_class_code"] = pd.Categorical(bip_rows["bip_class"], categories=valid_bip_classes).codes
    bip_c = prep_tun_cats(bip_rows.copy())
    log(f"  BIP rows for classifier 5: {len(bip_c):,}")

    cv_results = {}

    # ── Classifier 1: xSwing% ─────────────────────────────────────────────────
    log("\nClassifier 1 — xSwing% (EXPECTED: weak — location-driven behavior):")
    cv_results["tun_xswing"] = cv_binary(df_c, "is_swing", feat_cols, cat_cols, "Tun xSwing%")

    # ── Classifier 4: xCalledStrike% ─────────────────────────────────────────
    log("\nClassifier 4 — xCalledStrike% (EXPECTED: weak — location-driven):")
    cv_results["tun_xcs"] = cv_binary(no_sw_c, "is_called_strike", feat_cols, cat_cols, "Tun xCS%")

    # ── Classifier 2: xWhiff% ────────────────────────────────────────────────
    log("\nClassifier 2 — xWhiff% (EXPECTED: strongest signal — tunneling-specific):")
    cv_results["tun_xwhiff"] = cv_binary(sw_c, "is_whiff", feat_cols, cat_cols, "Tun xWhiff%")

    # ── Classifier 3: xFoul% ─────────────────────────────────────────────────
    log("\nClassifier 3 — xFoul%:")
    cv_results["tun_xfoul"] = cv_binary(contact_c, "is_foul", feat_cols, cat_cols, "Tun xFoul%")

    # ── Classifier 5: BIP distribution ───────────────────────────────────────
    log(f"\nClassifier 5 — BIP type×EV (multiclass {len(valid_bip_classes)} classes):")
    cv_results["tun_bip"] = cv_multiclass(
        bip_c, "bip_class", feat_cols, cat_cols, valid_bip_classes, "Tun BIP-dist"
    )

    # ── Print asymmetry diagnosis ─────────────────────────────────────────────
    log("\n--- Tunneling+ v2 Asymmetry Diagnosis ---")
    asymmetry_report = {}
    for name in ["tun_xswing", "tun_xcs", "tun_xwhiff", "tun_xfoul"]:
        folds = cv_results.get(name, [])
        if folds and "auc" in folds[0]:
            mean_auc = np.mean([f["auc"] for f in folds if not np.isnan(f["auc"])])
            mean_ll = np.mean([f["log_loss"] for f in folds])
            asymmetry_report[name] = {"mean_auc": mean_auc, "mean_logloss": mean_ll}
            log(f"  {name}: AUC={mean_auc:.4f} LogLoss={mean_ll:.4f}")
    log("  Interpretation: xSwing/xCS AUC near 0.5 = expected (no location features).")
    log("  xWhiff AUC materially above 0.5 = tunneling genuinely predicts bat-misses.")

    # ── Train final models ────────────────────────────────────────────────────
    log("\nTraining final Tunneling+ v2 models on all seasons...")

    def fit_tun_binary(subset_df, label_col, name_):
        tr_fit, tr_stop = train_test_split(subset_df, test_size=0.08, random_state=42)
        model = fit_binary(tr_fit[feat_cols], tr_fit[label_col],
                           tr_stop[feat_cols], tr_stop[label_col], cat_cols)
        log(f"  {name_} best_iteration={model.best_iteration_}")
        return model

    m_swing = fit_tun_binary(df_c, "is_swing", "Tun xSwing%")
    m_cs = fit_tun_binary(no_sw_c, "is_called_strike", "Tun xCS%")
    m_whiff = fit_tun_binary(sw_c, "is_whiff", "Tun xWhiff%")
    m_foul = fit_tun_binary(contact_c, "is_foul", "Tun xFoul%")

    tr_bip, stop_bip = train_test_split(bip_c, test_size=0.08, random_state=42)
    m_bip = fit_multiclass(tr_bip[feat_cols], tr_bip["bip_class_code"],
                           stop_bip[feat_cols], stop_bip["bip_class_code"],
                           cat_cols, len(valid_bip_classes))
    log(f"  BIP multiclass best_iteration={m_bip.best_iteration_}")

    # ── Score all pitches ────────────────────────────────────────────────────
    log("\nScoring all pitches with Tunneling+ v2...")

    def tun_predict(model, df_):
        return model.predict_proba(df_[feat_cols])[:, 1]

    p_swing = tun_predict(m_swing, df_c)
    p_whiff = tun_predict(m_whiff, df_c)
    p_foul = tun_predict(m_foul, df_c)
    p_cs = tun_predict(m_cs, df_c)
    p_bip = m_bip.predict_proba(df_c[feat_cols])

    xrv = combine_xrv(
        df_c, p_swing, p_whiff, p_foul, p_cs,
        p_bip, valid_bip_classes, rv_table, hbp_rate_no_swing,
    )
    df_c["xrv"] = xrv
    df_c["pitcher_id"] = df["pitcher_id"].values

    log(f"  xRV distribution: mean={xrv.mean():.4f} std={xrv.std():.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    scores = aggregate_plus_scores(df_c)
    log(f"  Tunneling+ v2 scores: {len(scores):,} pitcher-seasons, "
        f"std={scores['plus_score'].std():.2f}, "
        f"min={scores['plus_score'].min():.1f}, max={scores['plus_score'].max():.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    scores.to_parquet(ARTIFACTS_DIR / "tunneling_plus_v2_scores.parquet", index=False)
    df_c[["pitcher_id", "season", "xrv"]].to_parquet(
        ARTIFACTS_DIR / "tunneling_plus_v2_per_pitch.parquet", index=False
    )
    m_swing.booster_.save_model(str(ARTIFACTS_DIR / "tunneling_plus_v2_xswing.txt"))
    m_cs.booster_.save_model(str(ARTIFACTS_DIR / "tunneling_plus_v2_xcs.txt"))
    m_whiff.booster_.save_model(str(ARTIFACTS_DIR / "tunneling_plus_v2_xwhiff.txt"))
    m_foul.booster_.save_model(str(ARTIFACTS_DIR / "tunneling_plus_v2_xfoul.txt"))
    m_bip.booster_.save_model(str(ARTIFACTS_DIR / "tunneling_plus_v2_bip.txt"))

    meta = dict(
        feat_cols=feat_cols, cat_cols=cat_cols, bip_classes=valid_bip_classes,
        hbp_rate_no_swing=hbp_rate_no_swing,
        cv_results={k: v for k, v in cv_results.items()},
        asymmetry_report=asymmetry_report,
        trained_at=datetime.datetime.now().isoformat(),
        seasons=ALL_SEASONS,
    )
    (ARTIFACTS_DIR / "tunneling_plus_v2_meta.json").write_text(json.dumps(meta, indent=2))

    log(f"\nPart B artifacts saved to {ARTIFACTS_DIR}/")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# PART C — Four-way comparison
# ─────────────────────────────────────────────────────────────────────────────
def run_part_c(conn, loc_v2_scores, tun_v2_scores):
    log("\n" + "=" * 70)
    log("PART C: Four-way diagnostic comparison")
    log("=" * 70)

    # ── Load live v1 scores from DB ───────────────────────────────────────────
    log("Loading live model scores from pitcher_model_scores table...")
    live = pd.read_sql("""
        SELECT pitcher_id, season, stuff_plus, tunneling_plus, location_plus
        FROM pitcher_model_scores
        WHERE season BETWEEN 2015 AND 2025
    """, conn)  # small table, no chunking needed
    log(f"  Live scores: {len(live):,} pitcher-seasons")

    # Ensure column name consistency
    loc_v2 = loc_v2_scores.rename(columns={"plus_score": "location_plus_v2"})
    tun_v2 = tun_v2_scores.rename(columns={"plus_score": "tunneling_plus_v2"})

    # Live scores (v1/v3) already in pitcher_model_scores
    loc_v1 = live[live["location_plus"].notna()][["pitcher_id", "season", "location_plus"]].copy()
    tun_v3 = live[live["tunneling_plus"].notna()][["pitcher_id", "season", "tunneling_plus"]].copy()
    stuff = live[live["stuff_plus"].notna()][["pitcher_id", "season", "stuff_plus"]].copy()

    models = {
        "Location+ v1 (live)": loc_v1["location_plus"],
        "Location+ v2 (candidate)": loc_v2["location_plus_v2"],
        "Tunneling+ v3 (live)": tun_v3["tunneling_plus"],
        "Tunneling+ v2 (candidate)": tun_v2["tunneling_plus_v2"],
    }

    def scores_arr(scores_series):
        return scores_series.dropna().values

    # ── Metric 1: Score spread ────────────────────────────────────────────────
    log("\n--- C1: Score spread ---")
    spread_rows = []
    for name, s in models.items():
        arr = scores_arr(s)
        if len(arr) == 0:
            continue
        row = {
            "model": name, "n": len(arr),
            "mean": np.mean(arr), "std": np.std(arr),
            "min": np.min(arr), "p5": np.percentile(arr, 5),
            "p25": np.percentile(arr, 25), "p50": np.percentile(arr, 50),
            "p75": np.percentile(arr, 75), "p95": np.percentile(arr, 95),
            "max": np.max(arr),
            "pct_99_101": float(np.mean((arr >= 99) & (arr <= 101)) * 100),
            "pct_95_105": float(np.mean((arr >= 95) & (arr <= 105)) * 100),
        }
        spread_rows.append(row)
        log(f"  {name}: std={row['std']:.2f} [{row['min']:.1f},{row['max']:.1f}] "
            f"within[99,101]={row['pct_99_101']:.1f}% within[95,105]={row['pct_95_105']:.1f}%")

    # ── Metric 2: YoY r² ─────────────────────────────────────────────────────
    log("\n--- C2: Year-over-year r² ---")

    def yoy_r2(df_, score_col, pid_col="pitcher_id", season_col="season"):
        """Pool consecutive-year correlations: for each pitcher with scores
        in both year N and N+1, correlate. Pool all pairs."""
        seasons = sorted(int(s) for s in df_[season_col].unique())
        all_y1, all_y2 = [], []
        for s1, s2 in zip(seasons[:-1], seasons[1:]):
            if s2 - s1 != 1:
                continue
            d1 = df_[df_[season_col].astype(int) == s1][[pid_col, score_col]].rename(columns={score_col: "s1"})
            d2 = df_[df_[season_col].astype(int) == s2][[pid_col, score_col]].rename(columns={score_col: "s2"})
            merged = d1.merge(d2, on=pid_col)
            if len(merged) < 10:
                continue
            all_y1.extend(merged["s1"].tolist())
            all_y2.extend(merged["s2"].tolist())
        if len(all_y1) < 20:
            return float("nan")
        return float(r2_score(all_y2, all_y1))

    yoy_r2_results = {}
    for name, s_series in [("Location+ v1 (live)", loc_v1), ("Location+ v2 (candidate)", loc_v2),
                            ("Tunneling+ v3 (live)", tun_v3), ("Tunneling+ v2 (candidate)", tun_v2)]:
        if name.startswith("Location"):
            sc_col = "location_plus" if "v1" in name else "location_plus_v2"
            r2 = yoy_r2(s_series, sc_col)
        else:
            sc_col = "tunneling_plus" if "v3" in name else "tunneling_plus_v2"
            r2 = yoy_r2(s_series, sc_col)
        yoy_r2_results[name] = r2
        log(f"  {name}: YoY r²={r2:.4f}")

    # ── Metric 3: Split-half reliability ─────────────────────────────────────
    log("\n--- C3: Split-half reliability ---")
    log("  (Loading per-pitch data for v2 models; using live model pitches for v1/v3)")

    # For v2 models, use per-pitch xRV files
    shr_results = {}

    def split_half_from_per_pitch(pp_df, score_col="xrv"):
        """Split each pitcher-season's pitches 50/50, compute mean xRV each half, correlate."""
        np.random.seed(42)
        results_h1, results_h2 = [], []
        pid_list = []
        for (pid, season), grp in pp_df.groupby(["pitcher_id", "season"]):
            if len(grp) < MIN_PITCHES * 2:
                continue
            idx = np.random.permutation(len(grp))
            h1 = grp.iloc[idx[:len(grp)//2]][score_col].mean()
            h2 = grp.iloc[idx[len(grp)//2:]][score_col].mean()
            results_h1.append(h1)
            results_h2.append(h2)
            pid_list.append((pid, season))
        if len(results_h1) < 10:
            return float("nan")
        r = np.corrcoef(results_h1, results_h2)[0, 1]
        # Spearman-Brown correction for full-length reliability
        return float(2 * r / (1 + r))

    for model_name, pp_path in [
        ("Location+ v2 (candidate)", ARTIFACTS_DIR / "location_plus_v2_per_pitch.parquet"),
        ("Tunneling+ v2 (candidate)", ARTIFACTS_DIR / "tunneling_plus_v2_per_pitch.parquet"),
    ]:
        if pp_path.exists():
            pp = pd.read_parquet(pp_path)
            pp["season"] = pp["season"].astype(int)
            shr = split_half_from_per_pitch(pp)
            shr_results[model_name] = shr
            log(f"  {model_name}: split-half (SB-corrected) r={shr:.4f}")
        else:
            shr_results[model_name] = float("nan")
            log(f"  {model_name}: per-pitch file not found — skipping")

    # For v1/v3 live models, use raw pitches from DB to generate predictions
    # via the live model artifacts
    log("  Loading live model predictions for split-half (v1 Location+, v3 Tunneling+)...")
    try:
        import lightgbm as _lgb
        from analytics.location_features import engineer_features as _loc_eng, ALL_FEATURES as _loc_feat, CATEGORICAL_FEATURES as _loc_cat

        loc_v1_model = _lgb.Booster(model_file="models/location_plus_v1.txt")
        loc_raw = pd.concat(pd.read_sql("""
            SELECT pitcher_id, pitch_type, stand, season, plate_x, plate_z, balls, strikes
            FROM pitches WHERE delta_run_exp IS NOT NULL AND season BETWEEN 2015 AND 2025
        """, conn, chunksize=100_000), ignore_index=True)
        loc_raw_eng = _loc_eng(loc_raw)
        cats = sorted(set(loc_raw_eng["pitch_type"].dropna()) | set())
        for col in _loc_cat:
            loc_raw_eng[col] = pd.Categorical(loc_raw_eng[col])
        preds_v1 = loc_v1_model.predict(loc_raw_eng[_loc_feat])
        loc_pp_v1 = loc_raw_eng[["pitcher_id", "season"]].copy()
        loc_pp_v1["season"] = loc_pp_v1["season"].astype(str).astype(int)
        loc_pp_v1["xrv"] = preds_v1
        shr_v1 = split_half_from_per_pitch(loc_pp_v1)
        shr_results["Location+ v1 (live)"] = shr_v1
        log(f"  Location+ v1 (live): split-half (SB-corrected) r={shr_v1:.4f}")
    except Exception as e:
        shr_results["Location+ v1 (live)"] = float("nan")
        log(f"  Location+ v1 (live) split-half failed: {e}")

    try:
        from analytics.tunneling_v3_model import pitcher_score as _tun_score
        # Split-half for tunneling v3 is expensive (requires lag features) — approximate
        # by using stored per-pitcher-season scores and bootstrapped half-seasons
        # This is a known limitation reported in output.
        shr_results["Tunneling+ v3 (live)"] = float("nan")
        log("  Tunneling+ v3 (live): split-half requires full lag rebuild — not computed here")
    except Exception as e:
        shr_results["Tunneling+ v3 (live)"] = float("nan")

    # ── Metric 4: Predictive validity (prior season → following outcomes) ────
    log("\n--- C4: Predictive validity (prior-season score → next-season outcomes) ---")

    # Compute next-season whiff%, K-BB%, chase% from DB
    log("  Computing next-season outcome stats from pitches table...")
    outcomes = pd.read_sql("""
        SELECT pitcher_id, season,
            SUM(CASE WHEN description IN ('swinging_strike','swinging_strike_blocked','foul_tip','bunt_foul_tip','missed_bunt') THEN 1 ELSE 0 END) as whiffs,
            SUM(CASE WHEN description IN ('swinging_strike','swinging_strike_blocked','foul_tip','bunt_foul_tip','missed_bunt',
                'foul','foul_bunt','foul_pitchout','hit_into_play','hit_into_play_no_out','hit_into_play_score') THEN 1 ELSE 0 END) as swings,
            COUNT(*) as total_pitches
        FROM pitches
        WHERE season BETWEEN 2015 AND 2025
        GROUP BY pitcher_id, season
    """, conn)  # aggregated query returns small result, no chunking needed
    outcomes["whiff_pct"] = outcomes["whiffs"] / outcomes["swings"].replace(0, np.nan)
    outcomes["swstr_pct"] = outcomes["whiffs"] / outcomes["total_pitches"]
    outcomes = outcomes[outcomes["total_pitches"] >= MIN_PITCHES].copy()

    def pred_validity(score_df, score_col, outcome_df, outcome_col):
        """r² of prior-season score predicting next-season outcome."""
        all_pred, all_out = [], []
        for season in COMPLETE_SEASONS[:-1]:
            next_s = season + 1
            if next_s not in COMPLETE_SEASONS:
                continue
            prior = score_df[score_df["season"] == season][["pitcher_id", score_col]]
            nxt = outcome_df[outcome_df["season"] == next_s][["pitcher_id", outcome_col]]
            merged = prior.merge(nxt, on="pitcher_id").dropna()
            if len(merged) < 10:
                continue
            all_pred.extend(merged[score_col].tolist())
            all_out.extend(merged[outcome_col].tolist())
        if len(all_pred) < 20:
            return float("nan")
        return float(r2_score(all_out, all_pred))

    pv_results = {}
    for (mname, score_df, scol) in [
        ("Location+ v1 (live)", loc_v1, "location_plus"),
        ("Location+ v2 (candidate)", loc_v2, "location_plus_v2"),
        ("Tunneling+ v3 (live)", tun_v3, "tunneling_plus"),
        ("Tunneling+ v2 (candidate)", tun_v2, "tunneling_plus_v2"),
    ]:
        score_df = score_df.copy()
        score_df["season"] = score_df["season"].astype(int)
        outcomes["season"] = outcomes["season"].astype(int)
        r2_whiff = pred_validity(score_df, scol, outcomes, "whiff_pct")
        r2_swstr = pred_validity(score_df, scol, outcomes, "swstr_pct")
        pv_results[mname] = {"whiff_pct": r2_whiff, "swstr_pct": r2_swstr}
        log(f"  {mname}: r²(next-year whiff%)={r2_whiff:.4f} r²(next-year SwStr%)={r2_swstr:.4f}")

    # ── Metric 5: Real-skill regression (ERA, xFIP) ──────────────────────────
    log("\n--- C5: Real-skill regression vs ERA and xFIP ---")

    era_data = pd.read_sql("""
        SELECT player_id as pitcher_id, season, era, xfip
        FROM player_advanced_stats
        WHERE season BETWEEN 2015 AND 2025
    """, conn)
    era_data = era_data.dropna(subset=["era"])

    skill_r2 = {}
    for (mname, score_df, scol) in [
        ("Location+ v1 (live)", loc_v1, "location_plus"),
        ("Location+ v2 (candidate)", loc_v2, "location_plus_v2"),
        ("Tunneling+ v3 (live)", tun_v3, "tunneling_plus"),
        ("Tunneling+ v2 (candidate)", tun_v2, "tunneling_plus_v2"),
    ]:
        score_df = score_df.copy()
        score_df["season"] = score_df["season"].astype(int)
        era_data["season"] = era_data["season"].astype(int)
        merged = score_df.merge(era_data, on=["pitcher_id", "season"]).dropna(subset=[scol, "era"])
        if len(merged) < 20:
            skill_r2[mname] = {"era": float("nan"), "xfip": float("nan")}
            log(f"  {mname}: insufficient data for skill regression")
            continue
        # ERA is "lower is better" for pitchers; score is "higher is better"
        r2_era = float(r2_score(merged["era"], merged[scol]))
        r2_xfip = float("nan")
        if merged["xfip"].notna().sum() >= 20:
            merged2 = merged.dropna(subset=["xfip"])
            r2_xfip = float(r2_score(merged2["xfip"], merged2[scol]))
        skill_r2[mname] = {"era": r2_era, "xfip": r2_xfip}
        log(f"  {mname}: r²(ERA)={r2_era:.4f} r²(xFIP)={r2_xfip:.4f} n={len(merged):,}")

    # ── Metric 6: Cross-model correlations ───────────────────────────────────
    log("\n--- C6: Cross-model correlations ---")

    # Build wide table: pitcher_id, season, stuff_plus, loc_v1, loc_v2, tun_v3, tun_v2
    wide = stuff.copy()
    wide["season"] = wide["season"].astype(int)

    for df_, col in [(loc_v1, "location_plus"), (loc_v2, "location_plus_v2"),
                     (tun_v3, "tunneling_plus"), (tun_v2, "tunneling_plus_v2")]:
        df_ = df_.copy()
        df_["season"] = df_["season"].astype(int)
        wide = wide.merge(df_[["pitcher_id", "season", col]], on=["pitcher_id", "season"], how="outer")

    wide = wide.dropna()
    log(f"  Wide table for cross-correlations: {len(wide):,} pitcher-seasons")

    pairs = [
        ("location_plus", "stuff_plus", "Loc v1 × Stuff+"),
        ("location_plus_v2", "stuff_plus", "Loc v2 × Stuff+"),
        ("tunneling_plus", "stuff_plus", "Tun v3 × Stuff+"),
        ("tunneling_plus_v2", "stuff_plus", "Tun v2 × Stuff+"),
        ("location_plus", "tunneling_plus", "Loc v1 × Tun v3"),
        ("location_plus_v2", "tunneling_plus_v2", "Loc v2 × Tun v2"),
        ("location_plus", "tunneling_plus_v2", "Loc v1 × Tun v2"),
        ("location_plus_v2", "tunneling_plus", "Loc v2 × Tun v3"),
    ]

    corr_results = {}
    for col_a, col_b, label in pairs:
        if col_a not in wide.columns or col_b not in wide.columns:
            continue
        r = float(np.corrcoef(wide[col_a], wide[col_b])[0, 1])
        corr_results[label] = r
        log(f"  {label}: r={r:.4f}")

    # ── FINAL COMPARISON TABLE ────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("PART C FINAL: Four-way comparison table")
    log("=" * 70)

    comparison = []
    for row in spread_rows:
        mname = row["model"]
        r = {
            "Model": mname,
            "N pitcher-seasons": row["n"],
            "Score std": f"{row['std']:.2f}",
            "Score min": f"{row['min']:.1f}",
            "Score max": f"{row['max']:.1f}",
            "p5": f"{row['p5']:.1f}",
            "p95": f"{row['p95']:.1f}",
            "% in [99,101]": f"{row['pct_99_101']:.1f}%",
            "% in [95,105]": f"{row['pct_95_105']:.1f}%",
            "YoY r²": f"{yoy_r2_results.get(mname, float('nan')):.4f}",
            "Split-half r": f"{shr_results.get(mname, float('nan')):.4f}",
            "PV r²(whiff%)": f"{pv_results.get(mname, {}).get('whiff_pct', float('nan')):.4f}",
            "PV r²(SwStr%)": f"{pv_results.get(mname, {}).get('swstr_pct', float('nan')):.4f}",
            "r²(ERA)": f"{skill_r2.get(mname, {}).get('era', float('nan')):.4f}",
            "r²(xFIP)": f"{skill_r2.get(mname, {}).get('xfip', float('nan')):.4f}",
        }
        comparison.append(r)

    # Print table
    if comparison:
        comp_df = pd.DataFrame(comparison)
        log("\n" + comp_df.to_string(index=False))

        # Save
        comp_df.to_csv(ARTIFACTS_DIR / "v2_model_comparison.csv", index=False)
        (ARTIFACTS_DIR / "v2_cross_correlations.json").write_text(json.dumps(corr_results, indent=2))

    # ── Plain-language summary ─────────────────────────────────────────────────
    log("\n--- Plain-language summary ---")

    def _std(name):
        for r in spread_rows:
            if r["model"] == name:
                return r["std"]
        return 0.0

    loc_v1_std = _std("Location+ v1 (live)")
    loc_v2_std = _std("Location+ v2 (candidate)")
    tun_v3_std = _std("Tunneling+ v3 (live)")
    tun_v2_std = _std("Tunneling+ v2 (candidate)")

    loc_improved = loc_v2_std > loc_v1_std * 1.05
    tun_improved = tun_v2_std > tun_v3_std * 1.05

    log(f"Location+: v1 std={loc_v1_std:.2f} → v2 std={loc_v2_std:.2f} "
        f"({'WIDER — candidate architecture shows improvement' if loc_improved else 'comparable or narrower — not clearly better'})")
    log(f"Tunneling+: v3 std={tun_v3_std:.2f} → v2 std={tun_v2_std:.2f} "
        f"({'WIDER — candidate architecture shows improvement' if tun_improved else 'comparable or narrower — not clearly better'})")

    loc_lv_v1 = corr_results.get("Loc v1 × Tun v3", float("nan"))
    loc_lv_v2 = corr_results.get("Loc v2 × Tun v2", float("nan"))
    log(f"Location+↔Tunneling+ cross-correlation: v1/v3={loc_lv_v1:.4f} → v2/v2={loc_lv_v2:.4f}")
    log("(Previous: r≈0.563 despite shared no raw features — verify whether this persists, changes, or disappears)")

    # Tunneling asymmetry hypothesis
    if "tun_xswing" in cv_results and "tun_xwhiff" in cv_results:
        pass  # reported inline above
    log("\nTunneling+ asymmetry hypothesis: see classifier AUC values above.")
    log("  Hypothesis: xSwing%/xCS% near-chance (AUC~0.5) due to no location features.")
    log("  xWhiff% materially above chance = sequential context predicts bat-miss timing.")
    log("\nNOTE: Neither v2 candidate is promoted to production.")
    log("Both live models (location_plus_v1.txt, tunneling_v3.txt) are UNCHANGED.")

    mark_checkpoint("partC")
    return comp_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-part0", action="store_true")
    parser.add_argument("--skip-partA", action="store_true")
    parser.add_argument("--skip-partB", action="store_true")
    parser.add_argument("--only-partC", action="store_true")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("v2 Models Training Run — Location+ v2 and Tunneling+ v2")
    log("DIAGNOSTIC BUILDS ONLY. Live models untouched.")
    log("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    t0_total = time.time()

    # ── Part 0 ────────────────────────────────────────────────────────────────
    if checkpoint_done("part0") or args.skip_part0 or args.only_partC:
        log("Part 0: Loading from checkpoint...")
        rv_path = ARTIFACTS_DIR / "run_value_table.json"
        if rv_path.exists():
            rv_data = json.loads(rv_path.read_text())
            rv_table = rv_data["rv_table"]
            bip_classes = rv_data["bip_classes"]
        else:
            log("ERROR: Part 0 checkpoint present but run_value_table.json missing")
            return
    else:
        rv_table, bip_classes = build_run_value_table(conn)
        rv_data = {"rv_table": rv_table, "bip_classes": bip_classes}
        (ARTIFACTS_DIR / "run_value_table.json").write_text(json.dumps(rv_data, indent=2))
        log(f"\nRun-value table saved to {ARTIFACTS_DIR}/run_value_table.json")
        mark_checkpoint("part0")

    # ── Part A ────────────────────────────────────────────────────────────────
    if checkpoint_done("partA") or args.skip_partA or args.only_partC:
        log("Part A: Loading Location+ v2 scores from checkpoint...")
        loc_path = ARTIFACTS_DIR / "location_plus_v2_scores.parquet"
        if loc_path.exists():
            loc_v2_scores = pd.read_parquet(loc_path)
        else:
            log("ERROR: Part A checkpoint but scores file missing")
            return
    else:
        loc_v2_scores = run_part_a(conn, rv_table, bip_classes)
        mark_checkpoint("partA")
        log(f"\n--- CHECKPOINT A: Location+ v2 complete ---")
        log(f"Total elapsed: {(time.time()-t0_total)/60:.1f} minutes")

    # ── Part B ────────────────────────────────────────────────────────────────
    if checkpoint_done("partB") or args.skip_partB or args.only_partC:
        log("Part B: Loading Tunneling+ v2 scores from checkpoint...")
        tun_path = ARTIFACTS_DIR / "tunneling_plus_v2_scores.parquet"
        if tun_path.exists():
            tun_v2_scores = pd.read_parquet(tun_path)
        else:
            log("ERROR: Part B checkpoint but scores file missing")
            return
    else:
        tun_v2_scores = run_part_b(conn, rv_table, bip_classes)
        mark_checkpoint("partB")
        log(f"\n--- CHECKPOINT B: Tunneling+ v2 complete ---")
        log(f"Total elapsed: {(time.time()-t0_total)/60:.1f} minutes")

    # ── Part C ────────────────────────────────────────────────────────────────
    run_part_c(conn, loc_v2_scores, tun_v2_scores)

    conn.close()
    total_elapsed = time.time() - t0_total
    log(f"\nAll parts complete. Total elapsed: {total_elapsed/60:.1f} minutes.")
    log(f"Artifacts in: {ARTIFACTS_DIR}/")
    log(f"Log at: {LOG_PATH}")


if __name__ == "__main__":
    main()
