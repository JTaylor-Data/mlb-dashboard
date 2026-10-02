"""
Tunneling+ v2.1 — Drop geometry from xFoul% only

General rule applied here and going forward:
    Drop lagN_{commit_distance, plate_distance, shape_break} from any individual
    classifier where those features show a statistically significant WRONG-SIGNED
    effect on that classifier's own real target outcome. Keep geometry everywhere
    it is null or correctly signed.

xFoul% is the only current trigger (Part A2: lag1-geom z=-21.8, alllag z=-44.6).
The same rule would apply uniformly to any future classifier showing the same pattern.

Parts:
  A — Retrain xFoul% without the 18 geometry columns. Save new booster.
  B — Validate on a CLEAN split (complement of original Part A2 seed=42 sample —
      guaranteed zero overlap; stated explicitly below).
  C — LOSO AUC comparison (xFoul% only, 5 folds) + full-corpus ERA/YoY/split-half
      comparison: v2.1 (new xFoul%) vs v2 baseline (original xFoul%).

Constraints:
  - No production changes. Not wired into api/. Tunneling+ v2 stays live regardless.
  - Live model files (tunneling_plus_v2_*.txt) are not modified.
  - Output saved to training/v2_artifacts/tunneling_plus_v2_1_diagnostic.json
"""

import json
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from analytics.tunneling_features import (
    load_pitches,
    sequential_lag_features,
    MAX_LAG,
)
from analytics.location_features import to_plus_scale

MODELS_DIR  = ROOT / "models"
META_PATH   = MODELS_DIR / "tunneling_plus_v2_meta.json"
RV_PATH     = ROOT / "training" / "v2_artifacts" / "run_value_table.json"
COMMIT_PATH = (
    ROOT / "project_artifacts" / "training_artifacts"
    / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv"
)
ERA_DB_PATH = "mlb_data.db"
OUT_DIR     = ROOT / "training" / "v2_artifacts"
OUT_PATH    = OUT_DIR / "tunneling_plus_v2_1_diagnostic.json"
LOG_PATH    = ROOT / "training" / "tunneling_plus_v2_1.log"
NEW_MODEL_PATH = MODELS_DIR / "tunneling_plus_v2_1_xfoul_nogeom.txt"

TRAIN_SEASONS  = list(range(2021, 2026))
MIN_PA_PITCHES = 300

# Part A2 original sample parameters — used to derive the clean split
ORIGINAL_SHAP_N    = 200_000
ORIGINAL_SHAP_SEED = 42
CLEAN_N            = 100_000
CLEAN_SEED         = 99

CLASSIFIER_PARAMS = dict(
    n_estimators=500, num_leaves=31, learning_rate=0.05,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)

FOUL_DESCS  = {"foul", "foul_bunt", "foul_pitchout"}
SWING_DESCS = {
    "swinging_strike","swinging_strike_blocked","foul_tip","bunt_foul_tip",
    "missed_bunt","foul","foul_bunt","foul_pitchout",
    "hit_into_play","hit_into_play_no_out","hit_into_play_score","swinging_pitchout",
}
WHIFF_DESCS = {
    "swinging_strike","swinging_strike_blocked","foul_tip",
    "bunt_foul_tip","missed_bunt","swinging_pitchout",
}
BIP_DESCS = {"hit_into_play","hit_into_play_no_out","hit_into_play_score"}
CS_DESCS  = {"called_strike","automatic_strike"}
HBP_RATE  = 0.0051

_log_fh = None


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def stringify_keep_na(s):
    na = s.isna()
    return pd.Series(np.where(na, np.nan, s.astype(str)), index=s.index)


def prop_z(p_hi, n_hi, p_lo, n_lo):
    se = np.sqrt(p_hi * (1 - p_hi) / n_hi + p_lo * (1 - p_lo) / n_lo)
    return float((p_hi - p_lo) / se) if se > 0 else 0.0


def tercile_z(df, score_col, outcome_col):
    sub = df[[score_col, outcome_col]].dropna()
    sub = sub.copy()
    sub["_t"] = pd.qcut(sub[score_col], 3, labels=["lo","mid","hi"], duplicates="drop")
    g = sub.groupby("_t", observed=True)[outcome_col].agg(["mean", "count"])
    if "lo" not in g.index or "hi" not in g.index:
        return None
    return {
        "lo_rate": float(g.loc["lo","mean"]), "mid_rate": float(g.loc["mid","mean"]),
        "hi_rate": float(g.loc["hi","mean"]), "lo_n": int(g.loc["lo","count"]),
        "hi_n": int(g.loc["hi","count"]),
        "z": prop_z(g.loc["hi","mean"], g.loc["hi","count"],
                    g.loc["lo","mean"], g.loc["lo","count"]),
    }


def stratified_sample_idx(df, outcome_col, n, seed):
    rng  = np.random.default_rng(seed)
    pos  = df.index[df[outcome_col] == 1.0].tolist()
    neg  = df.index[df[outcome_col] == 0.0].tolist()
    rate = df[outcome_col].mean()
    n_p  = int(n * rate)
    n_n  = n - n_p
    return (
        rng.choice(pos, size=min(n_p, len(pos)), replace=False).tolist() +
        rng.choice(neg, size=min(n_n, len(neg)), replace=False).tolist()
    )


def _pred_binary(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.predict(X)


def _pred_bip(m, X, bip_classes):
    if hasattr(m, "predict_proba"):
        cls_list = list(m.classes_)
        proba    = m.predict_proba(X)
        out      = np.zeros((len(X), len(bip_classes)))
        for i, c in enumerate(bip_classes):
            if c in cls_list:
                out[:, i] = proba[:, cls_list.index(c)]
        return out
    return m.predict(X)   # Booster multiclass → (n, n_classes) in training order


def compute_xrv(df_X, v2_models, v2_fcols, xfoul_model, xfoul_fcols, rv, bip_classes):
    """Vectorised xRV for a DataFrame; returns shape (n,)."""
    X_v2 = df_X[v2_fcols]
    X_fo = df_X[xfoul_fcols]
    p_sw  = _pred_binary(v2_models["xswing"], X_v2)
    p_cs  = _pred_binary(v2_models["xcs"],    X_v2)
    p_wh  = _pred_binary(v2_models["xwhiff"], X_v2)
    p_fo  = _pred_binary(xfoul_model,          X_fo)
    p_bip = _pred_bip(v2_models["bip"], X_v2, bip_classes)
    bip_rvs      = np.array([rv.get(c, 0.0) for c in bip_classes])
    xrv_bip      = p_bip @ bip_rvs
    xrv_contact  = p_fo * rv["foul"] + (1.0 - p_fo) * xrv_bip
    xrv_swing    = p_wh * rv["whiff"] + (1.0 - p_wh) * xrv_contact
    p_ball       = np.clip(1.0 - p_cs - HBP_RATE, 0.0, 1.0)
    xrv_no_swing = p_cs * rv["called_strike"] + HBP_RATE * rv["hbp"] + p_ball * rv["ball"]
    return p_sw * xrv_swing + (1.0 - p_sw) * xrv_no_swing


def apply_cat_levels(df, col_list, cat_levels):
    for col in col_list:
        if col not in df.columns:
            continue
        df[col] = stringify_keep_na(df[col])
        lvls = cat_levels.get(col)
        df[col] = pd.Categorical(df[col], categories=lvls) if lvls else pd.Categorical(df[col])
    return df


def load_season_data(conn, season, commit_table, cat_levels, v2_feat_cols):
    df_s = load_pitches(conn, season)
    if len(df_s) == 0:
        return pd.DataFrame()
    feat = sequential_lag_features(df_s, commit_table, max_lag=MAX_LAG)
    feat = feat.dropna(subset=["lag1_commit_distance", "lag1_plate_distance"])
    if len(feat) == 0:
        return pd.DataFrame()
    feat["season"] = str(season)
    lag_pt_cols = [c for c in v2_feat_cols if c.endswith("_pitch_type")]
    curr_cat    = ["pitch_type", "p_throws", "stand", "season"]
    feat = apply_cat_levels(feat, lag_pt_cols + curr_cat, cat_levels)
    feat["is_foul"]  = feat["description"].isin(FOUL_DESCS).astype(float)
    feat["is_swing"] = feat["description"].isin(SWING_DESCS).astype(float)
    feat["is_whiff"] = feat["description"].isin(WHIFF_DESCS).astype(float)
    feat["is_bip"]   = feat["description"].isin(BIP_DESCS).astype(float)
    feat["is_cs"]    = feat["description"].isin(CS_DESCS).astype(float)
    return feat


def pitcher_season_scores(pitch_df, xrv_arr):
    """Aggregate per-pitch xRV to pitcher-season xRV sum + count."""
    tmp = pitch_df[["pitcher_id", "season"]].copy()
    tmp["xrv"] = xrv_arr
    agg = tmp.groupby(["pitcher_id", "season"])["xrv"].agg(["sum", "count"]).reset_index()
    agg.columns = ["pitcher_id", "season", "xrv_sum", "n_pitches"]
    return agg[agg["n_pitches"] >= MIN_PA_PITCHES].copy()


def to_plus(xrv_sum, n_pitches):
    """Per-season plus-scale from raw xRV per 100 pitches."""
    per_100 = xrv_sum / n_pitches * 100
    seas_mean = per_100.mean()
    seas_std  = per_100.std()
    return 100 - 10 * (per_100 - seas_mean) / (seas_std + 1e-9)


def main():
    global _log_fh
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")

    log("=" * 65)
    log("Tunneling+ v2.1 — Drop geometry from xFoul% only")
    log("=" * 65)
    log("General rule: drop lagN_{commit_distance,plate_distance,shape_break}")
    log("from any classifier with statistically significant WRONG-SIGNED geometry")
    log("effect on its own real target. Keep geometry where null or correct.")
    log("xFoul% triggers this rule: lag1-geom z=-21.8, alllag z=-44.6 (Part A2).")

    meta         = json.loads(META_PATH.read_text())
    v2_feat_cols = meta["feat_cols"]
    cat_cols     = meta["cat_cols"]
    bip_classes  = meta["bip_classes"]

    rv_data   = json.loads(RV_PATH.read_text())
    rv_table  = rv_data["rv_table"]

    commit_table = pd.read_csv(COMMIT_PATH)
    log(f"Commit table: {len(commit_table):,} rows, seasons {sorted(commit_table['season'].unique())}")

    conn = sqlite3.connect(ERA_DB_PATH)

    # ── Category levels ────────────────────────────────────────────────────
    log("Deriving category levels...")
    raw = pd.read_sql(
        "SELECT DISTINCT pitch_type,p_throws,stand,season FROM pitches WHERE season BETWEEN 2015 AND 2026",
        conn,
    )
    raw["pitch_type"] = raw["pitch_type"].replace({"KC": "CU"})
    raw = raw[~raw["pitch_type"].isin({"PO","UN","EP"})]
    raw["season"] = raw["season"].astype(str)
    pt_lvls = sorted(raw["pitch_type"].dropna().unique())
    lag_pt_cols = [c for c in v2_feat_cols if c.endswith("_pitch_type")]
    cat_levels  = {
        "pitch_type": pt_lvls,
        "p_throws":   sorted(raw["p_throws"].dropna().unique()),
        "stand":      sorted(raw["stand"].dropna().unique()),
        "season":     sorted(raw["season"].dropna().unique()),
    }
    for col in lag_pt_cols:
        cat_levels[col] = pt_lvls

    # ── New feature set: v2 minus geometry ────────────────────────────────
    geom_cols = [
        f"lag{n}_{m}" for n in range(1, MAX_LAG + 1)
        for m in ("commit_distance", "plate_distance", "shape_break")
    ]
    geom_set  = set(geom_cols)
    xfo_fcols = [c for c in v2_feat_cols if c not in geom_set]  # 10 features
    xfo_cats  = [c for c in xfo_fcols if c in cat_levels or c.endswith("_pitch_type")]
    log(f"v2 features: {len(v2_feat_cols)}   xFoul%-nogeom features: {len(xfo_fcols)}")
    log(f"Geometry columns removed: {len(geom_set)}")

    # ── Load v2 boosters (untouched) ───────────────────────────────────────
    v2_models = {
        name: lgb.Booster(model_file=str(MODELS_DIR / f"tunneling_plus_v2_{name}.txt"))
        for name in ("xswing", "xcs", "xwhiff", "xfoul", "bip")
    }
    log("v2 boosters loaded (not modified).")

    # ── Load training data (2021-2025) ─────────────────────────────────────
    log("\nLoading training data 2021-2025...")
    chunks = []
    t0 = time.time()
    for s in TRAIN_SEASONS:
        t1 = time.time()
        log(f"  Season {s}...")
        feat = load_season_data(conn, s, commit_table, cat_levels, v2_feat_cols)
        if len(feat) == 0:
            log("    No data, skipping")
            continue
        chunks.append(feat)
        log(f"    {len(feat):,} rows ({time.time()-t1:.0f}s)")
    df_all = pd.concat(chunks, ignore_index=True)
    log(f"Total: {len(df_all):,} rows in {time.time()-t0:.0f}s")
    log(f"xFoul% base rate: {df_all['is_foul'].mean():.4f}")

    results = {"general_rule": (
        "Drop lagN_{commit_distance,plate_distance,shape_break} from any classifier "
        "where those features show a statistically significant WRONG-SIGNED effect on "
        "that classifier's own real target outcome. Keep geometry everywhere null or correct. "
        "xFoul% is the only current trigger (Part A2: lag1-geom z=-21.8, alllag z=-44.6)."
    )}

    # ═══════════════════════════════════════════════════════════════════════
    # PART A — Retrain xFoul% without geometry
    # ═══════════════════════════════════════════════════════════════════════
    log("\n" + "="*65)
    log("PART A — Retrain xFoul% without geometry (2021-2025 full)")
    log("="*65)

    sub_tr = df_all.dropna(subset=["is_foul"])
    X_tr   = sub_tr[xfo_fcols]
    y_tr   = sub_tr["is_foul"].astype(int)

    log(f"Training on {len(X_tr):,} pitches with {len(xfo_fcols)} features (no geometry)...")
    t0 = time.time()
    m_xfoul_new = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    m_xfoul_new.fit(X_tr, y_tr, categorical_feature=xfo_cats)
    log(f"Done in {time.time()-t0:.0f}s")

    m_xfoul_new.booster_.save_model(str(NEW_MODEL_PATH))
    log(f"Saved: {NEW_MODEL_PATH}")

    results["part_a"] = {
        "model_path": str(NEW_MODEL_PATH),
        "feat_cols": xfo_fcols,
        "geom_cols_removed": geom_cols,
        "train_rows": int(len(X_tr)),
        "train_seasons": TRAIN_SEASONS,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # PART B — Clean-split validation
    # Clean split: complement of the original Part A2 SHAP sample
    # (seed=42, n=200K from the same 2021-2025 corpus, identical data loading).
    # Complement size ≈ 2.4M rows — guaranteed zero overlap with the original
    # SHAP diagnostic. A 100K stratified sample (seed=99) is drawn from the
    # complement for SHAP computation.
    # ═══════════════════════════════════════════════════════════════════════
    log("\n" + "="*65)
    log("PART B — Clean-split validation (no overlap with Part A2 diagnostic)")
    log("="*65)
    log("Clean split: complement of original Part A2 seed=42 200K sample")
    log("(identical data loading → same integer indices → guaranteed zero overlap)")

    orig_idx     = set(stratified_sample_idx(df_all, "is_foul", ORIGINAL_SHAP_N, ORIGINAL_SHAP_SEED))
    df_complement = df_all[~df_all.index.isin(orig_idx)]
    log(f"Complement size: {len(df_complement):,}  (original sample: {len(orig_idx):,})")

    clean_idx  = stratified_sample_idx(df_complement, "is_foul", min(CLEAN_N, len(df_complement)), CLEAN_SEED)
    df_clean   = df_complement.loc[clean_idx].copy()
    log(f"Clean sample: {len(df_clean):,}  foul base rate: {df_clean['is_foul'].mean():.4f}")

    part_b = {"clean_split_description": (
        "Complement of Part A2's original seed=42 200K stratified sample from 2021-2025. "
        "Uses seed=99 for a 100K draw from the complement (~2.4M rows). "
        "Overlap with original SHAP diagnostic: zero (same data loading → same indices)."
    )}

    # B-1: v2 xFoul% on clean split
    log("\nB-1: v2 xFoul% (WITH geometry) on clean split")
    X_clean_v2  = df_clean[v2_feat_cols]
    p_fo_v2     = v2_models["xfoul"].predict(X_clean_v2)
    auc_v2      = roc_auc_score(df_clean["is_foul"].astype(int), p_fo_v2)
    df_clean["p_foul_v2"] = p_fo_v2
    cal_v2 = tercile_z(df_clean, "p_foul_v2", "is_foul")
    log(f"  AUC: {auc_v2:.4f}")
    if cal_v2:
        log(f"  Calibration z: lo={cal_v2['lo_rate']:.4f} hi={cal_v2['hi_rate']:.4f} z={cal_v2['z']:.3f}")

    # Geometry SHAP on v2 to confirm wrong-direction on new data
    log("  Computing geometry SHAP (v2 xFoul%)...")
    t0 = time.time()
    expl_v2 = shap.TreeExplainer(v2_models["xfoul"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv_v2 = expl_v2.shap_values(X_clean_v2)
    if isinstance(sv_v2, list):
        sv_v2 = sv_v2[1] if len(sv_v2) == 2 else sv_v2[-1]
    log(f"  SHAP done in {time.time()-t0:.0f}s")

    fidx = {f: i for i, f in enumerate(v2_feat_cols)}
    lag1g_idx  = [fidx[c] for c in ["lag1_commit_distance","lag1_plate_distance","lag1_shape_break"] if c in fidx]
    allag_idx  = [fidx[c] for c in geom_cols if c in fidx]
    df_clean["lag1_geom_shap_v2"]  = sv_v2[:, lag1g_idx].sum(axis=1)
    df_clean["alllag_geom_shap_v2"] = sv_v2[:, allag_idx].sum(axis=1)
    r_lag1  = tercile_z(df_clean, "lag1_geom_shap_v2",   "is_foul")
    r_allag = tercile_z(df_clean, "alllag_geom_shap_v2", "is_foul")
    for label, r in [("lag1-geom", r_lag1), ("alllag-geom", r_allag)]:
        if r:
            log(f"  v2 {label} SHAP z: lo={r['lo_rate']:.4f} hi={r['hi_rate']:.4f} z={r['z']:.3f}")

    part_b["v2_xfoul"] = {
        "auc_clean_split": round(auc_v2, 4),
        "calibration": cal_v2,
        "lag1_geom_shap_z":  r_lag1,
        "alllag_geom_shap_z": r_allag,
    }

    # B-2: new xFoul% (no geometry) on clean split
    log("\nB-2: new xFoul% (WITHOUT geometry) on clean split")
    X_clean_new = df_clean[xfo_fcols]
    p_fo_new    = m_xfoul_new.predict_proba(X_clean_new)[:, 1]
    auc_new     = roc_auc_score(df_clean["is_foul"].astype(int), p_fo_new)
    df_clean["p_foul_new"] = p_fo_new
    cal_new = tercile_z(df_clean, "p_foul_new", "is_foul")
    log(f"  AUC: {auc_new:.4f}")
    if cal_new:
        log(f"  Calibration z: lo={cal_new['lo_rate']:.4f} hi={cal_new['hi_rate']:.4f} z={cal_new['z']:.3f}")

    # pt-lag SHAP on new model
    log("  Computing pt-lag SHAP (new xFoul%)...")
    t0 = time.time()
    expl_new = shap.TreeExplainer(m_xfoul_new.booster_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv_new = expl_new.shap_values(X_clean_new)
    if isinstance(sv_new, list):
        sv_new = sv_new[1] if len(sv_new) == 2 else sv_new[-1]
    log(f"  SHAP done in {time.time()-t0:.0f}s")

    nfidx    = {f: i for i, f in enumerate(xfo_fcols)}
    pt_idx   = [nfidx[c] for c in xfo_fcols if c.endswith("_pitch_type") and c in nfidx]
    df_clean["pt_lag_shap_new"] = sv_new[:, pt_idx].sum(axis=1)
    r_pt = tercile_z(df_clean, "pt_lag_shap_new", "is_foul")
    if r_pt:
        log(f"  new pt-lag SHAP z: lo={r_pt['lo_rate']:.4f} hi={r_pt['hi_rate']:.4f} z={r_pt['z']:.3f}")

    auc_delta = auc_new - auc_v2
    log(f"\n  AUC clean split: v2={auc_v2:.4f}  new={auc_new:.4f}  Δ={auc_delta:+.4f}")

    part_b["new_xfoul"] = {
        "auc_clean_split": round(auc_new, 4),
        "calibration": cal_new,
        "pt_lag_shap_z": r_pt,
    }
    part_b["auc_delta_new_minus_v2"] = round(auc_delta, 4)
    results["part_b"] = part_b

    # ═══════════════════════════════════════════════════════════════════════
    # PART C — LOSO AUC + full-corpus ERA/YoY comparison
    # ═══════════════════════════════════════════════════════════════════════
    log("\n" + "="*65)
    log("PART C — Full validation: v2.1 vs v2 baseline")
    log("="*65)

    # ── C-1: LOSO xFoul% AUC (5 folds, retrain xFoul% only) ───────────────
    log("\nC-1: LOSO xFoul% AUC (5 folds)...")
    log("  Note: 4 non-xFoul% classifiers are fixed live-v2 models throughout.")
    log("  Comparison: v2.1 xFoul% (retrained per fold) vs v2 xFoul% (full live model).")

    loso_new, loso_v2 = [], []
    fold_xrv_v21 = []
    fold_xrv_v2  = []

    for test_s in TRAIN_SEASONS:
        train_ss = [s for s in TRAIN_SEASONS if s != test_s]
        mask_te  = df_all["season"] == str(test_s)
        mask_tr  = df_all["season"].isin([str(s) for s in train_ss])
        df_te = df_all[mask_te].dropna(subset=["is_foul"])
        df_tr = df_all[mask_tr].dropna(subset=["is_foul"])

        log(f"  Fold {test_s}: train {len(df_tr):,}  test {len(df_te):,}")
        t0 = time.time()
        m_fold = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
        m_fold.fit(df_tr[xfo_fcols], df_tr["is_foul"].astype(int), categorical_feature=xfo_cats)
        log(f"    xFoul% retrained in {time.time()-t0:.0f}s")

        p_new_fold = m_fold.predict_proba(df_te[xfo_fcols])[:, 1]
        p_v2_fold  = v2_models["xfoul"].predict(df_te[v2_feat_cols])
        y_te       = df_te["is_foul"].astype(int)

        if len(y_te.unique()) > 1:
            a_new = roc_auc_score(y_te, p_new_fold)
            a_v2  = roc_auc_score(y_te, p_v2_fold)
            loso_new.append(a_new)
            loso_v2.append(a_v2)
            log(f"    AUC — new={a_new:.4f}  v2={a_v2:.4f}  Δ={a_new-a_v2:+.4f}")

        # Shared predictions (same for both)
        X_te_v2  = df_te[v2_feat_cols]
        p_sw     = v2_models["xswing"].predict(X_te_v2)
        p_cs     = v2_models["xcs"].predict(X_te_v2)
        p_wh     = v2_models["xwhiff"].predict(X_te_v2)
        p_bip    = _pred_bip(v2_models["bip"], X_te_v2, bip_classes)
        bip_rvs  = np.array([rv_table.get(c, 0.0) for c in bip_classes])
        xrv_bip  = p_bip @ bip_rvs

        def calc_xrv(p_fo):
            xrv_c  = p_fo * rv_table["foul"] + (1 - p_fo) * xrv_bip
            xrv_sw = p_wh * rv_table["whiff"] + (1 - p_wh) * xrv_c
            p_ball = np.clip(1 - p_cs - HBP_RATE, 0, 1)
            xrv_ns = p_cs * rv_table["called_strike"] + HBP_RATE * rv_table["hbp"] + p_ball * rv_table["ball"]
            return p_sw * xrv_sw + (1 - p_sw) * xrv_ns

        xrv_new = calc_xrv(p_new_fold)
        xrv_v2r = calc_xrv(p_v2_fold)

        for lst, xrv_arr in [(fold_xrv_v21, xrv_new), (fold_xrv_v2, xrv_v2r)]:
            tmp = df_te[["pitcher_id", "season"]].copy()
            tmp["xrv"] = xrv_arr
            lst.append(tmp)

    mean_loso_new = float(np.mean(loso_new)) if loso_new else None
    mean_loso_v2  = float(np.mean(loso_v2))  if loso_v2  else None
    log(f"  Mean LOSO AUC: new={mean_loso_new:.4f}  v2={mean_loso_v2:.4f}  Δ={mean_loso_new-mean_loso_v2:+.4f}")

    loso_result = {
        "per_fold_new": [round(x, 4) for x in loso_new],
        "per_fold_v2":  [round(x, 4) for x in loso_v2],
        "mean_new": round(mean_loso_new, 4) if mean_loso_new else None,
        "mean_v2":  round(mean_loso_v2,  4) if mean_loso_v2  else None,
        "mean_delta": round(mean_loso_new - mean_loso_v2, 4) if (mean_loso_new and mean_loso_v2) else None,
    }

    # ── C-2: Pitcher-season ERA/YoY comparison ────────────────────────────
    log("\nC-2: Pitcher-season ERA/YoY comparison (LOSO xRV aggregated)...")

    df_v21_pitch = pd.concat(fold_xrv_v21, ignore_index=True)
    df_v2_pitch  = pd.concat(fold_xrv_v2,  ignore_index=True)

    def aggregate_scores(pitch_df):
        agg = pitch_df.groupby(["pitcher_id","season"])["xrv"].agg(["sum","count"]).reset_index()
        agg.columns = ["pitcher_id","season","xrv_sum","n_pitches"]
        agg = agg[agg["n_pitches"] >= MIN_PA_PITCHES].copy()
        for s, grp in agg.groupby("season"):
            p100 = grp["xrv_sum"] / grp["n_pitches"] * 100
            agg.loc[grp.index, "plus"] = to_plus_scale(p100.values, p100.mean(), p100.std())
        return agg

    def to_plus_scale(vals, mean, std):
        return 100 - 10 * (vals - mean) / (std + 1e-9)

    scores_v21 = aggregate_scores(df_v21_pitch)
    scores_v2  = aggregate_scores(df_v2_pitch)
    log(f"  Pitcher-seasons: v2.1={len(scores_v21):,}  v2={len(scores_v2):,}")

    # ERA/xFIP from DB
    era_df = pd.read_sql("""
        SELECT player_id as pitcher_id, season,
               era, xfip
        FROM player_advanced_stats
        WHERE era IS NOT NULL AND xfip IS NOT NULL
          AND season BETWEEN 2021 AND 2025
    """, conn)
    era_df["season"] = era_df["season"].astype(str)
    era_df["pitcher_id"] = era_df["pitcher_id"].astype(str)
    scores_v21["pitcher_id"] = scores_v21["pitcher_id"].astype(str)
    scores_v21["season"]     = scores_v21["season"].astype(str)
    scores_v2["pitcher_id"]  = scores_v2["pitcher_id"].astype(str)
    scores_v2["season"]      = scores_v2["season"].astype(str)

    def era_r(sc):
        m = sc.merge(era_df, on=["pitcher_id","season"])
        if len(m) < 20:
            return None, None, len(m)
        r_era  = float(stats.pearsonr(m["plus"], m["era"])[0])
        r_xfip = float(stats.pearsonr(m["plus"], m["xfip"])[0])
        return r_era, r_xfip, len(m)

    r_era_v21, r_xfip_v21, n_v21 = era_r(scores_v21)
    r_era_v2,  r_xfip_v2,  n_v2  = era_r(scores_v2)
    log(f"  ERA  r: v2.1={r_era_v21:.3f}  v2={r_era_v2:.3f}  (n={n_v21})")
    log(f"  xFIP r: v2.1={r_xfip_v21:.3f}  v2={r_xfip_v2:.3f}")

    def yoy_r(sc):
        rs = []
        seasons = sorted(sc["season"].unique())
        for i in range(len(seasons) - 1):
            s1, s2 = seasons[i], seasons[i+1]
            a = sc[sc["season"] == s1][["pitcher_id","plus"]].rename(columns={"plus":"plus_t"})
            b = sc[sc["season"] == s2][["pitcher_id","plus"]].rename(columns={"plus":"plus_t1"})
            m = a.merge(b, on="pitcher_id")
            if len(m) >= 20:
                r = float(stats.pearsonr(m["plus_t"], m["plus_t1"])[0])
                rs.append({"pair": f"{s1}-{s2}", "r": round(r, 3), "n": len(m)})
        return rs

    yoy_v21 = yoy_r(scores_v21)
    yoy_v2  = yoy_r(scores_v2)
    mean_yoy_v21 = float(np.mean([x["r"] for x in yoy_v21])) if yoy_v21 else None
    mean_yoy_v2  = float(np.mean([x["r"] for x in yoy_v2]))  if yoy_v2  else None
    log(f"  YoY pooled r: v2.1={mean_yoy_v21:.3f}  v2={mean_yoy_v2:.3f}")
    for p, q in zip(yoy_v21, yoy_v2):
        log(f"    YoY {p['pair']}: v2.1={p['r']:.3f}  v2={q['r']:.3f}")

    def split_half_r(sc):
        g = sc.groupby("pitcher_id").filter(lambda x: len(x) >= 2)
        if len(g) == 0:
            return None, 0
        rng    = np.random.default_rng(42)
        halves = g.groupby("pitcher_id")["plus"].apply(
            lambda x: pd.Series({"h1": x.iloc[:len(x)//2].mean(), "h2": x.iloc[len(x)//2:].mean()})
        ).reset_index()
        halves = halves.dropna()
        if len(halves) < 10:
            return None, len(halves)
        return float(stats.pearsonr(halves["h1"], halves["h2"])[0]), len(halves)

    sh_v21, n_sh_v21 = split_half_r(scores_v21)
    sh_v2,  n_sh_v2  = split_half_r(scores_v2)
    log(f"  Split-half r: v2.1={sh_v21:.3f}  v2={sh_v2:.3f}  (n={n_sh_v21})")

    results["part_c"] = {
        "loso_xfoul_auc": loso_result,
        "pitcher_season_n": {"v21": int(len(scores_v21)), "v2": int(len(scores_v2))},
        "era_r": {"v21": round(r_era_v21, 4) if r_era_v21 else None, "v2": round(r_era_v2, 4) if r_era_v2 else None},
        "xfip_r": {"v21": round(r_xfip_v21, 4) if r_xfip_v21 else None, "v2": round(r_xfip_v2, 4) if r_xfip_v2 else None},
        "yoy": {"v21": yoy_v21, "v2": yoy_v2,
                "mean_pooled_v21": round(mean_yoy_v21, 4) if mean_yoy_v21 else None,
                "mean_pooled_v2":  round(mean_yoy_v2,  4) if mean_yoy_v2  else None},
        "split_half": {"v21": round(sh_v21, 4) if sh_v21 else None, "v2": round(sh_v2, 4) if sh_v2 else None},
        "era_n": int(n_v21),
    }

    # ── Summary ────────────────────────────────────────────────────────────
    log("\n" + "="*65)
    log("SUMMARY")
    log("="*65)
    log(f"  xFoul% AUC (clean split): v2={auc_v2:.4f}  v2.1={auc_new:.4f}  Δ={auc_delta:+.4f}")
    log(f"  xFoul% LOSO AUC:          v2={mean_loso_v2:.4f}  v2.1={mean_loso_new:.4f}  Δ={mean_loso_new-mean_loso_v2:+.4f}")
    log(f"  ERA r:                    v2={r_era_v2:.3f}  v2.1={r_era_v21:.3f}")
    log(f"  xFIP r:                   v2={r_xfip_v2:.3f}  v2.1={r_xfip_v21:.3f}")
    log(f"  YoY pooled r:             v2={mean_yoy_v2:.3f}  v2.1={mean_yoy_v21:.3f}")
    log(f"  Split-half r:             v2={sh_v2:.3f}  v2.1={sh_v21:.3f}")
    log(f"  v2 geometry SHAP z on clean split: lag1={r_lag1['z']:.3f}  alllag={r_allag['z']:.3f}")

    results["summary"] = {
        "xfoul_auc_clean_split": {"v2": round(auc_v2, 4), "v21": round(auc_new, 4), "delta": round(auc_delta, 4)},
        "xfoul_loso_auc": {"v2": round(mean_loso_v2, 4), "v21": round(mean_loso_new, 4),
                           "delta": round(mean_loso_new - mean_loso_v2, 4)},
        "era_r":  {"v2": round(r_era_v2,  4), "v21": round(r_era_v21,  4)},
        "xfip_r": {"v2": round(r_xfip_v2, 4), "v21": round(r_xfip_v21, 4)},
        "yoy_pooled": {"v2": round(mean_yoy_v2, 4), "v21": round(mean_yoy_v21, 4)},
        "split_half": {"v2": round(sh_v2, 4) if sh_v2 else None, "v21": round(sh_v21, 4) if sh_v21 else None},
        "v2_geom_shap_on_clean_split": {
            "lag1_z": round(r_lag1["z"], 3) if r_lag1 else None,
            "alllag_z": round(r_allag["z"], 3) if r_allag else None,
        },
        "constraint": "No production changes. Tunneling+ v2 stays live. Not wired into api/.",
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    log(f"\nSaved: {OUT_PATH}")
    log("Done.")
    _log_fh.close()


if __name__ == "__main__":
    main()
