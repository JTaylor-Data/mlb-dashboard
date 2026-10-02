"""
train_stuff_decomposed_v1.py  -- Part D of the v2 validation task.

Applies the same five-classifier decomposed-event architecture used for
Location+ v2 and Tunneling+ v2 to the Stuff+ feature set.

SAFETY: Read-only for live models. Writes ONLY to training/v2_artifacts/
under "stuff_decomposed_v1_*" naming. Does NOT modify models/, api/, or
any live model file. This is a DIAGNOSTIC build only.

Run from the mlb_dashboard root:
    python training/train_stuff_decomposed_v1.py

Expected runtime: 60-90 minutes (11 LOSO folds x 5 classifiers x
~1M samples per fold).
"""
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from analytics.location_features import to_plus_scale
from analytics.stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, EXCLUDED_PITCH_TYPES,
    RAW_QUERY_COLUMNS, add_movement_columns, add_separation_features,
    add_vaa_residual, build_fb_reference_table,
    engineer_features, exclude_junk_pitch_types, merge_pitch_type,
)
from analytics.spin_physics import fit_spin_axis_calibration
from analytics.metrics import compute_vaa

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "mlb_data.db"
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("training/v2_artifacts")
LOG_PATH = Path("training/train_stuff_decomposed_v1.log")
COMPLETE_SEASONS = list(range(2015, 2026))
MIN_PITCHES = 100
CV_TRAIN_MAX_ROWS = 1_000_000

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

# BIP exit-velo buckets (same as v2)
EV_BINS = [0, 90, 95, 100, 105, 999]
EV_LABELS = ["ev_lt90", "ev_90_95", "ev_95_100", "ev_100_105", "ev_105plus"]

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip",
    "missed_bunt", "foul", "foul_bunt", "foul_pitchout",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "swinging_pitchout",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "foul_tip",
               "bunt_foul_tip", "missed_bunt", "swinging_pitchout"}
FOUL_DESCS = {"foul", "foul_bunt", "foul_pitchout"}
BIP_DESCS = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
CALLED_STRIKE_DESCS = {"called_strike", "automatic_strike"}
BALL_DESCS = {"ball", "blocked_ball", "pitchout", "automatic_ball", "intent_ball",
              "pitchout_hit_into_play", "pitchout_hit_into_play_score"}
HBP_DESCS = {"hit_by_pitch"}

# ── Logging ───────────────────────────────────────────────────────────────────
_log_fh = None


def setup_log():
    global _log_fh
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


# ── Event labels ──────────────────────────────────────────────────────────────
def assign_event_labels(df):
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
    df["is_contact"] = ((df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
    return df


def make_bip_class(df):
    ev = pd.cut(df["launch_speed"].astype(float),
                bins=EV_BINS, labels=EV_LABELS, right=False).astype(str)
    return df["bb_type"].fillna("unknown") + "__" + ev


# ── LightGBM helpers ──────────────────────────────────────────────────────────
def fit_binary(X_tr, y_tr, X_val, y_val, cat_cols):
    model = lgb.LGBMClassifier(**BINARY_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              categorical_feature=cat_cols,
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def fit_multiclass(X_tr, y_tr, X_val, y_val, cat_cols, n_classes):
    p = {**MULTI_PARAMS, "num_class": n_classes}
    model = lgb.LGBMClassifier(**p)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              categorical_feature=cat_cols,
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def cv_binary(df, label_col, feat_cols, cat_cols, name):
    folds = []
    for holdout in COMPLETE_SEASONS:
        mask = df["season"] == str(holdout)
        tr = df[~mask]; te = df[mask]
        if len(te) < 100 or te[label_col].nunique() < 2:
            continue
        if len(tr) > CV_TRAIN_MAX_ROWS:
            tr = tr.sample(CV_TRAIN_MAX_ROWS, random_state=42)
        tr_fit, tr_stop = train_test_split(tr, test_size=0.08, random_state=42)
        model = fit_binary(tr_fit[feat_cols], tr_fit[label_col],
                           tr_stop[feat_cols], tr_stop[label_col], cat_cols)
        probs = model.predict_proba(te[feat_cols])[:, 1]
        try:
            auc = roc_auc_score(te[label_col], probs)
        except Exception:
            auc = float("nan")
        ll = log_loss(te[label_col], probs)
        acc = accuracy_score(te[label_col], (probs >= 0.5).astype(int))
        folds.append({"holdout": holdout, "auc": auc, "log_loss": ll, "acc": acc, "n": len(te)})
        log(f"    {name} holdout={holdout}: AUC={auc:.4f} LL={ll:.4f} n={len(te):,}")
    return folds


def cv_multiclass(df, label_col, feat_cols, cat_cols, class_order, name):
    folds = []
    n_classes = len(class_order)
    for holdout in COMPLETE_SEASONS:
        mask = df["season"] == str(holdout)
        tr = df[~mask]; te = df[mask]
        if len(te) < 100 or te[label_col].nunique() < 2:
            continue
        if len(tr) > CV_TRAIN_MAX_ROWS:
            tr = tr.sample(CV_TRAIN_MAX_ROWS, random_state=42)
        tr_codes = pd.Categorical(tr[label_col], categories=class_order).codes
        te_codes = pd.Categorical(te[label_col], categories=class_order).codes
        tr_fit_idx, tr_stop_idx = train_test_split(range(len(tr)), test_size=0.08, random_state=42)
        tr_arr = tr.iloc[list(tr_fit_idx)]; tr_stop_arr = tr.iloc[list(tr_stop_idx)]
        tr_codes_fit = tr_codes[list(tr_fit_idx)]; tr_codes_stop = tr_codes[list(tr_stop_idx)]
        model = fit_multiclass(tr_arr[feat_cols], tr_codes_fit,
                               tr_stop_arr[feat_cols], tr_codes_stop, cat_cols, n_classes)
        proba = model.predict_proba(te[feat_cols])
        ll = log_loss(te_codes, proba, labels=list(range(n_classes)))
        acc = accuracy_score(te_codes, proba.argmax(axis=1))
        folds.append({"holdout": holdout, "log_loss": ll, "acc": acc, "n": len(te)})
        log(f"    {name} holdout={holdout}: LL={ll:.4f} Acc={acc:.4f} n={len(te):,}")
    return folds


# ── xRV combination formula (same as v2) ─────────────────────────────────────
def combine_xrv(df, p_swing, p_whiff_given_swing, p_foul_given_contact,
                p_cs_given_no_swing, p_bip_proba, bip_classes, rv_table, hbp_rate_no_swing):
    rv_whiff = rv_table.get("whiff", 0.0)
    rv_foul = rv_table.get("foul", 0.0)
    bip_rvs = np.array([rv_table.get(c, 0.0) for c in bip_classes])
    xrv_bip = p_bip_proba @ bip_rvs
    xrv_contact = p_foul_given_contact * rv_foul + (1 - p_foul_given_contact) * xrv_bip
    xrv_given_swing = p_whiff_given_swing * rv_whiff + (1 - p_whiff_given_swing) * xrv_contact
    rv_cs = rv_table.get("called_strike", 0.0)
    rv_ball = rv_table.get("ball", 0.0)
    rv_hbp = rv_table.get("hbp", 0.0)
    p_ball_given_no_swing = 1 - p_cs_given_no_swing - hbp_rate_no_swing
    xrv_given_no_swing = (p_cs_given_no_swing * rv_cs
                          + hbp_rate_no_swing * rv_hbp
                          + p_ball_given_no_swing * rv_ball)
    return p_swing * xrv_given_swing + (1 - p_swing) * xrv_given_no_swing


def aggregate_plus_scores(df, min_pitches=MIN_PITCHES):
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


# ── YoY r² (Pearson r and sklearn R²) ────────────────────────────────────────
def yoy_metrics(df_, score_col):
    seasons = sorted(df_["season"].astype(int).unique())
    all_y1, all_y2 = [], []
    for s1, s2 in zip(seasons[:-1], seasons[1:]):
        if s2 - s1 != 1:
            continue
        d1 = df_[df_["season"].astype(int) == s1][["pitcher_id", score_col]].rename(columns={score_col: "s1"})
        d2 = df_[df_["season"].astype(int) == s2][["pitcher_id", score_col]].rename(columns={score_col: "s2"})
        m = d1.merge(d2, on="pitcher_id").dropna()
        if len(m) < 10:
            continue
        all_y1.extend(m["s1"].tolist())
        all_y2.extend(m["s2"].tolist())
    if len(all_y1) < 20:
        return float("nan"), float("nan"), float("nan")
    sk_r2 = float(r2_score(all_y2, all_y1))
    pr, _ = stats.pearsonr(all_y1, all_y2)
    return sk_r2, float(pr), float(pr ** 2)


def split_half_r(per_pitch_df, score_col="xrv"):
    """Spearman-Brown corrected split-half reliability."""
    results = []
    for season, sg in per_pitch_df.groupby("season"):
        g = sg.copy().reset_index(drop=True)
        even_idx = g.index[g.index % 2 == 0]
        odd_idx = g.index[g.index % 2 != 0]
        half1 = g.loc[even_idx].groupby("pitcher_id")[score_col].mean()
        half2 = g.loc[odd_idx].groupby("pitcher_id")[score_col].mean()
        m = half1.reset_index().merge(half2.reset_index(), on="pitcher_id", suffixes=("_h1", "_h2"))
        if len(m) < 10:
            continue
        r, _ = stats.pearsonr(m[f"{score_col}_h1"], m[f"{score_col}_h2"])
        results.append(r)
    if not results:
        return float("nan")
    mean_r = float(np.mean(results))
    return float(2 * mean_r / (1 + mean_r))  # Spearman-Brown correction


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    setup_log()
    log("=" * 70)
    log("train_stuff_decomposed_v1.py  -- Part D diagnostic")
    log("DIAGNOSTIC ONLY. Live models untouched. Writing to v2_artifacts/")
    log("=" * 70)

    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load run-value table (DO NOT REBUILD) ─────────────────────────────────
    log("\nLoading run-value table from checkpoint (do not rebuild)...")
    rv_path = ARTIFACTS_DIR / "run_value_table.json"
    with open(rv_path) as f:
        rv_data = json.load(f)
    rv_table = rv_data["rv_table"]
    bip_classes = rv_data["bip_classes"]
    log(f"  Loaded {len(rv_table)} run-value entries, {len(bip_classes)} BIP classes")

    # ── Load Stuff+ v3 calibrations (reuse, not refit) ────────────────────────
    log("\nLoading Stuff+ v3 calibrations from models/stuff_plus_v3_lgbm_meta.json...")
    meta = json.loads((MODELS_DIR / "stuff_plus_v3_lgbm_meta.json").read_text())
    calibrations = {
        "vaa_coefs": meta["vaa_coefs"],
        "spin_calibration": meta["spin_calibration"],
    }
    log("  Calibrations loaded.")

    # ── Load pitches with Stuff+ raw columns ─────────────────────────────────
    extra_cols = ["description", "game_id", "at_bat_number", "pitch_number"]
    query_cols = RAW_QUERY_COLUMNS + extra_cols
    log(f"\nLoading pitches (chunked, {len(query_cols)} columns)...")
    t0 = time.time()
    raw = pd.concat(pd.read_sql(f"""
        SELECT {', '.join(query_cols)}
        FROM pitches
        WHERE season BETWEEN 2015 AND 2025
          AND release_speed IS NOT NULL
          AND spin_axis IS NOT NULL
    """, conn, chunksize=100_000), ignore_index=True)
    log(f"  Loaded {len(raw):,} rows in {time.time()-t0:.0f}s")

    # ── Feature engineering ───────────────────────────────────────────────────
    log("\nApplying Stuff+ feature engineering pipeline...")
    # Build fb reference table (needs hb/ivb which engineer_features computes)
    prepped = merge_pitch_type(raw.copy())
    prepped = exclude_junk_pitch_types(prepped)
    prepped = add_movement_columns(prepped)
    fb_ref = build_fb_reference_table(prepped)
    log(f"  FB reference table: {len(fb_ref)} pitcher-seasons")

    # engineer_features drops rows with NaN in CORE_NUMERIC_FEATURES; extra columns
    # (description, game_id, etc.) are NOT in that list so they survive dropna.
    # They do get coerced to category if they match CATEGORICAL_FEATURES, but
    # description/game_id are not in that list either, so they pass through intact.
    raw_for_eng = raw.copy()
    df = engineer_features(raw_for_eng, calibrations, fb_ref)
    log(f"  After feature engineering: {len(df):,} rows")

    # description survives as a pass-through column
    if "description" not in df.columns:
        log("  WARNING: description dropped by engineer_features -- adding empty fallback")
        df["description"] = ""

    df = assign_event_labels(df)

    # BIP extra data (bb_type + launch_speed)
    log("Loading BIP rows for bb_type/launch_speed...")
    bip_extra = pd.concat(pd.read_sql("""
        SELECT game_id, at_bat_number, pitch_number, bb_type, launch_speed
        FROM pitches
        WHERE season BETWEEN 2015 AND 2025
          AND bb_type IS NOT NULL AND launch_speed IS NOT NULL
    """, conn, chunksize=100_000), ignore_index=True)
    log(f"  BIP rows: {len(bip_extra):,}")

    # Merge bb_type/launch_speed into df using original game key columns
    if "game_id" in df.columns and "at_bat_number" in df.columns:
        df = df.merge(bip_extra, on=["game_id", "at_bat_number", "pitch_number"], how="left")
    else:
        log("  WARNING: game_id not in df -- BIP merge skipped")
        df["bb_type"] = np.nan
        df["launch_speed"] = np.nan

    log(f"  After BIP merge: {df['bb_type'].notna().sum():,} rows have bb_type")

    # ── Feature / category setup ──────────────────────────────────────────────
    feat_cols = ALL_FEATURES
    cat_cols = CATEGORICAL_FEATURES

    # HBP rate from no-swing rows
    no_swing_rows = df[df["is_no_swing"] == 1]
    hbp_rate_no_swing = float(no_swing_rows["is_hbp"].mean())
    log(f"\nLeague HBP rate given no swing: {hbp_rate_no_swing:.4f}")

    # engineer_features already sets categoricals, but we need consistent
    # vocab across seasons for LOSO-CV
    log("  Re-coercing categoricals for consistent CV vocab...")
    all_cats = {}
    for col in cat_cols:
        all_cats[col] = sorted(set(df[col].dropna().astype(str)))
        df[col] = pd.Categorical(df[col].astype(str), categories=all_cats[col])

    # BIP subset
    bip_rows = df[(df["is_bip"] == 1) & df["bb_type"].notna() & df["launch_speed"].notna()].copy()
    bip_rows["bip_class"] = make_bip_class(bip_rows)
    valid_bip_classes = [c for c in bip_classes if c in bip_rows["bip_class"].values]
    bip_rows = bip_rows[bip_rows["bip_class"].isin(valid_bip_classes)].copy()
    log(f"  BIP rows for classifier 5: {len(bip_rows):,}, classes: {len(valid_bip_classes)}")

    log(f"\nDataset: {len(df):,} pitches for training")
    log(f"  is_swing: {df['is_swing'].mean():.3f}  is_whiff_given_swing: {(df[df['is_swing']==1]['is_whiff'].mean()):.3f}")
    log(f"  is_cs_given_no_swing: {(df[df['is_no_swing']==1]['is_called_strike'].mean()):.3f}")

    # ── LOSO Cross-validation ─────────────────────────────────────────────────
    cv_results = {}

    log("\n" + "=" * 70)
    log("Classifier 1 -- xSwing% (EXPECTED: similar to Location+ v2 ~0.66)")
    log("=" * 70)
    cv_results["stuff_xswing"] = cv_binary(
        df, "is_swing", feat_cols, cat_cols, "Stuff xSwing%")

    log("\n" + "=" * 70)
    log("Classifier 4 -- xCalledStrike% (EXPECTED: weak -- no plate location in Stuff+)")
    log("=" * 70)
    no_sw_c = df[df["is_no_swing"] == 1].copy()
    cv_results["stuff_xcs"] = cv_binary(
        no_sw_c, "is_called_strike", feat_cols, cat_cols, "Stuff xCS%")

    log("\n" + "=" * 70)
    log("Classifier 2 -- xWhiff% (EXPECTED: strongest signal for Stuff+)")
    log("=" * 70)
    sw_c = df[df["is_swing"] == 1].copy()
    cv_results["stuff_xwhiff"] = cv_binary(
        sw_c, "is_whiff", feat_cols, cat_cols, "Stuff xWhiff%")

    log("\n" + "=" * 70)
    log("Classifier 3 -- xFoul%")
    log("=" * 70)
    contact_c = df[df["is_contact"] == 1].copy()
    cv_results["stuff_xfoul"] = cv_binary(
        contact_c, "is_foul", feat_cols, cat_cols, "Stuff xFoul%")

    log("\n" + "=" * 70)
    log(f"Classifier 5 -- BIP type x EV (multiclass, {len(valid_bip_classes)} classes)")
    log("=" * 70)
    cv_results["stuff_bip"] = cv_multiclass(
        bip_rows, "bip_class", feat_cols, cat_cols, valid_bip_classes, "Stuff BIP-dist")

    # Print CV summary
    log("\n--- CV Summary ---")
    for name, folds in cv_results.items():
        if not folds:
            continue
        if "auc" in folds[0]:
            mean_auc = float(np.nanmean([f["auc"] for f in folds]))
            mean_ll = float(np.mean([f["log_loss"] for f in folds]))
            log(f"  {name}: mean AUC={mean_auc:.4f}  mean LogLoss={mean_ll:.4f}")
        else:
            mean_ll = float(np.mean([f["log_loss"] for f in folds]))
            mean_acc = float(np.mean([f["acc"] for f in folds]))
            log(f"  {name}: mean LogLoss={mean_ll:.4f}  mean Acc={mean_acc:.4f}")

    # ── Train final models ────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("Training final models on ALL seasons (full data)...")
    log("=" * 70)

    def fit_final_binary(subset_df, label_col, name_):
        tr_fit, tr_stop = train_test_split(subset_df, test_size=0.08, random_state=42)
        model = fit_binary(tr_fit[feat_cols], tr_fit[label_col],
                           tr_stop[feat_cols], tr_stop[label_col], cat_cols)
        log(f"  {name_} best_iteration={model.best_iteration_}")
        return model

    m_swing = fit_final_binary(df, "is_swing", "xSwing%")
    m_cs = fit_final_binary(no_sw_c, "is_called_strike", "xCS%")
    m_whiff = fit_final_binary(sw_c, "is_whiff", "xWhiff%")
    m_foul = fit_final_binary(contact_c, "is_foul", "xFoul%")

    bip_final = bip_rows.copy()
    bip_final["bip_class_code"] = pd.Categorical(bip_final["bip_class"], categories=valid_bip_classes).codes
    tr_bip, stop_bip = train_test_split(bip_final, test_size=0.08, random_state=42)
    m_bip = fit_multiclass(tr_bip[feat_cols], tr_bip["bip_class_code"],
                           stop_bip[feat_cols], stop_bip["bip_class_code"],
                           cat_cols, len(valid_bip_classes))
    log(f"  BIP multiclass best_iteration={m_bip.best_iteration_}")

    # ── Score all pitches ─────────────────────────────────────────────────────
    log("\nScoring all pitches with Stuff+ decomposed v1...")
    p_swing = m_swing.predict_proba(df[feat_cols])[:, 1]
    p_whiff = m_whiff.predict_proba(df[feat_cols])[:, 1]
    p_foul = m_foul.predict_proba(df[feat_cols])[:, 1]
    p_cs = m_cs.predict_proba(df[feat_cols])[:, 1]
    p_bip = m_bip.predict_proba(df[feat_cols])

    xrv = combine_xrv(df, p_swing, p_whiff, p_foul, p_cs,
                      p_bip, valid_bip_classes, rv_table, hbp_rate_no_swing)
    df["xrv"] = xrv
    log(f"  xRV: mean={xrv.mean():.4f} std={xrv.std():.4f} p5={np.percentile(xrv,5):.4f} p95={np.percentile(xrv,95):.4f}")

    # ── Aggregate per pitcher-season ─────────────────────────────────────────
    scores = aggregate_plus_scores(df)
    scores["season"] = scores["season"].astype(int)
    log(f"\nStuff+ decomposed v1: {len(scores):,} pitcher-seasons")
    log(f"  std={scores['plus_score'].std():.2f}  min={scores['plus_score'].min():.1f}  max={scores['plus_score'].max():.1f}")

    # Save artifacts
    scores_path = ARTIFACTS_DIR / "stuff_decomposed_v1_scores.parquet"
    pp_path = ARTIFACTS_DIR / "stuff_decomposed_v1_per_pitch.parquet"
    scores.to_parquet(scores_path, index=False)
    df[["pitcher_id", "season", "xrv"]].to_parquet(pp_path, index=False)

    # Save models
    m_swing.booster_.save_model(str(ARTIFACTS_DIR / "stuff_decomposed_v1_xswing.txt"))
    m_cs.booster_.save_model(str(ARTIFACTS_DIR / "stuff_decomposed_v1_xcs.txt"))
    m_whiff.booster_.save_model(str(ARTIFACTS_DIR / "stuff_decomposed_v1_xwhiff.txt"))
    m_foul.booster_.save_model(str(ARTIFACTS_DIR / "stuff_decomposed_v1_xfoul.txt"))
    m_bip.booster_.save_model(str(ARTIFACTS_DIR / "stuff_decomposed_v1_bip.txt"))
    log(f"\nArtifacts saved to {ARTIFACTS_DIR}/stuff_decomposed_v1_*")

    # ── Comparison battery vs live Stuff+ v3 ─────────────────────────────────
    log("\n" + "=" * 70)
    log("COMPARISON BATTERY: Stuff+ decomposed v1 vs live Stuff+ v3")
    log("=" * 70)

    import sqlite3 as _sq3
    live = pd.read_sql(
        "SELECT pitcher_id, season, stuff_plus FROM pitcher_model_scores "
        "WHERE season BETWEEN 2015 AND 2025 AND stuff_plus IS NOT NULL",
        conn)
    live["season"] = live["season"].astype(int)

    # Spread
    log("\n--- Spread comparison ---")
    log(f"  Stuff+ v3 (live):      std={live['stuff_plus'].std():.2f}  [{live['stuff_plus'].min():.1f}, {live['stuff_plus'].max():.1f}]")
    log(f"  Stuff+ decomposed v1:  std={scores['plus_score'].std():.2f}  [{scores['plus_score'].min():.1f}, {scores['plus_score'].max():.1f}]")

    # YoY
    log("\n--- Year-over-year stability ---")
    sk_v3, pr_v3, pr2_v3 = yoy_metrics(live, "stuff_plus")
    sk_d1, pr_d1, pr2_d1 = yoy_metrics(scores, "plus_score")
    log(f"  Stuff+ v3 (live):      sklearn_R2={sk_v3:.4f}  Pearson_r={pr_v3:.4f}  Pearson_r2={pr2_v3:.4f}")
    log(f"  Stuff+ decomposed v1:  sklearn_R2={sk_d1:.4f}  Pearson_r={pr_d1:.4f}  Pearson_r2={pr2_d1:.4f}")

    # Split-half
    log("\n--- Split-half reliability (Spearman-Brown corrected) ---")
    shr_d1 = split_half_r(df[["pitcher_id", "season", "xrv"]])
    log(f"  Stuff+ decomposed v1:  split-half r={shr_d1:.4f}")
    log("  Stuff+ v3 (live):      requires raw predictions (not computed here)")

    # Predictive validity (prior-season score -> next-season whiff%)
    log("\n--- Predictive validity (prior-season score -> next-season outcomes) ---")
    outcomes = pd.read_sql("""
        SELECT pitcher_id, season,
            SUM(CASE WHEN description IN
                ('swinging_strike','swinging_strike_blocked','foul_tip','bunt_foul_tip','missed_bunt')
                THEN 1 ELSE 0 END) as whiffs,
            COUNT(*) as total_pitches
        FROM pitches WHERE season BETWEEN 2015 AND 2025
        GROUP BY pitcher_id, season
    """, conn)
    outcomes["season"] = outcomes["season"].astype(int)
    outcomes["swstr_pct"] = outcomes["whiffs"] / outcomes["total_pitches"]
    outcomes = outcomes[outcomes["total_pitches"] >= MIN_PITCHES]

    def pred_val(score_df, score_col, out_df, out_col):
        preds, outs = [], []
        for s in COMPLETE_SEASONS[:-1]:
            ns = s + 1
            if ns not in COMPLETE_SEASONS:
                continue
            p = score_df[score_df["season"] == s][["pitcher_id", score_col]]
            n = out_df[out_df["season"] == ns][["pitcher_id", out_col]]
            m = p.merge(n, on="pitcher_id").dropna()
            if len(m) < 10: continue
            preds.extend(m[score_col].tolist())
            outs.extend(m[out_col].tolist())
        if len(preds) < 20: return float("nan"), float("nan")
        r, _ = stats.pearsonr(preds, outs)
        return float(r), float(r**2)

    r_v3, r2_v3 = pred_val(live, "stuff_plus", outcomes, "swstr_pct")
    r_d1, r2_d1 = pred_val(scores, "plus_score", outcomes, "swstr_pct")
    log(f"  Stuff+ v3 (live)     -> next-year SwStr%: Pearson r={r_v3:.4f}  r2={r2_v3:.4f}")
    log(f"  Stuff+ decomposed v1 -> next-year SwStr%: Pearson r={r_d1:.4f}  r2={r2_d1:.4f}")

    # ERA/xFIP same-season regression
    log("\n--- Same-season ERA/xFIP correlation ---")
    era = pd.read_sql(
        "SELECT player_id as pitcher_id, season, era, xfip FROM player_advanced_stats "
        "WHERE season BETWEEN 2015 AND 2025", conn)
    era["season"] = era["season"].astype(int)

    for label, score_df, scol in [("v3 live", live, "stuff_plus"), ("decomposed v1", scores, "plus_score")]:
        merged = score_df.merge(era.dropna(subset=["era"]), on=["pitcher_id", "season"])
        merged = merged.dropna(subset=[scol, "era"])
        if len(merged) < 20:
            log(f"  Stuff+ {label}: insufficient data (n={len(merged)})")
            continue
        r_era, _ = stats.pearsonr(merged[scol], -merged["era"])
        r2_era = r_era ** 2
        m2 = merged.dropna(subset=["xfip"])
        r_xfip = r2_xfip = float("nan")
        if len(m2) >= 20:
            rx, _ = stats.pearsonr(m2[scol], -m2["xfip"])
            r_xfip = rx; r2_xfip = rx ** 2
        log(f"  Stuff+ {label}: ERA r={r_era:.4f} (r2={r2_era:.4f})  xFIP r={r_xfip:.4f} (r2={r2_xfip:.4f})  n={len(merged):,}")

    # Save comparison meta
    comp_meta = {
        "created_at": datetime.now().isoformat(),
        "cv_classifiers": {
            name: {"mean_auc": float(np.nanmean([f["auc"] for f in folds])) if folds and "auc" in folds[0] else None,
                   "mean_ll": float(np.mean([f["log_loss"] for f in folds])) if folds else None}
            for name, folds in cv_results.items()
        },
        "scores_summary": {
            "n": len(scores),
            "std": float(scores["plus_score"].std()),
            "min": float(scores["plus_score"].min()),
            "max": float(scores["plus_score"].max()),
        },
        "yoy": {"sklearn_r2": sk_d1, "pearson_r": pr_d1, "pearson_r2": pr2_d1},
        "split_half_r": shr_d1,
        "pred_validity_swstr": {"pearson_r": r_d1, "pearson_r2": r2_d1},
        "note": "Part D: Stuff+ decomposed v1 -- diagnostic only, not for production"
    }
    meta_path = ARTIFACTS_DIR / "stuff_decomposed_v1_meta.json"
    with open(meta_path, "w") as f:
        json.dump(comp_meta, f, indent=2)
    log(f"\nComparison meta saved to {meta_path}")

    log("\n" + "=" * 70)
    log("Part D complete.")
    log("=" * 70)
    conn.close()
    if _log_fh:
        _log_fh.close()


if __name__ == "__main__":
    main()
