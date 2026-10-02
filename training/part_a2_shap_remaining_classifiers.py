"""
Part A2 — SHAP geometry isolation: remaining four classifiers.

Extends the xWhiff% geometry isolation test (part_a_tunneling_shap_geometry.py)
to the other four Tunneling+ v2 classifiers: xSwing%, xCS%, xFoul%, and BIP type×EV.

Same methodology throughout:
  - Load data once, sample 200K rows stratified on each classifier's own outcome rate
  - Compute SHAP via TreeExplainer
  - Isolate three scores: lag-1 geometry SHAP, all-lag geometry SHAP, pitch-type-lag SHAP
  - Tercile z-test (high − low) on real outcome rate per tercile

BIP handling: multiclass, so geometry SHAP is aggregated as a run-value-weighted sum
across all 20 BIP classes (this gives "geometry's contribution to expected BIP xRV").
Outcome tested: real delta_run_exp on BIP events (t-test, not proportion z-test).
For consistency, also report is_bip proportion z-test alongside.

Combined 15-z-score table (5 classifiers × 3 scores) at end, including the already-
completed xWhiff% numbers for reference.

Run from mlb_dashboard root:
    python training/part_a2_shap_remaining_classifiers.py
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
import shap
from scipy import stats

from analytics.tunneling_features import (
    lag_feature_columns, load_pitches, sequential_lag_features, MAX_LAG,
)

MODELS_DIR  = ROOT / "models"
META_PATH   = MODELS_DIR / "tunneling_plus_v2_meta.json"
RV_PATH     = ROOT / "training" / "v2_artifacts" / "run_value_table.json"
COMMIT_PATH = (
    ROOT / "project_artifacts" / "training_artifacts"
    / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv"
)
OUT_PATH   = ROOT / "training" / "v2_artifacts" / "part_a2_shap_remaining.json"
LOG_PATH   = ROOT / "training" / "part_a2_shap_remaining.log"
DB_PATH    = "mlb_data.db"

COMMIT_SEASONS  = list(range(2021, 2026))
SHAP_SAMPLE_N   = 200_000
BIP_SHAP_SAMPLE = 30_000   # BIP is 20-class multiclass; smaller sample to keep SHAP runtime <5 min
RNG_SEED        = 42

LAG_COLS          = lag_feature_columns()
CATEGORICAL_LAG   = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CAT       = ["pitch_type", "p_throws", "stand", "season"]
ALL_CAT           = CATEGORICAL_LAG + CURRENT_CAT
LAG1_GEOM_COLS    = ["lag1_commit_distance", "lag1_plate_distance", "lag1_shape_break"]
ALL_LAG_GEOM_COLS = [f"lag{n}_{m}" for n in range(1, MAX_LAG + 1)
                     for m in ("commit_distance", "plate_distance", "shape_break")]
PITCH_TYPE_LAG_COLS = CATEGORICAL_LAG  # lag1..lag6 pitch_type

SWING_DESCS = {
    "swinging_strike","swinging_strike_blocked","foul_tip","bunt_foul_tip",
    "missed_bunt","foul","foul_bunt","foul_pitchout",
    "hit_into_play","hit_into_play_no_out","hit_into_play_score","swinging_pitchout",
}
WHIFF_DESCS = {"swinging_strike","swinging_strike_blocked","foul_tip","bunt_foul_tip","missed_bunt","swinging_pitchout"}
FOUL_DESCS  = {"foul","foul_bunt","foul_pitchout"}
BIP_DESCS   = {"hit_into_play","hit_into_play_no_out","hit_into_play_score"}
CS_DESCS    = {"called_strike","automatic_strike"}

_log_fh = None


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n"); _log_fh.flush()


def stringify_keep_na(series):
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)


def prop_z(p_high, n_high, p_low, n_low):
    se = np.sqrt(p_high*(1-p_high)/n_high + p_low*(1-p_low)/n_low)
    return float((p_high - p_low) / se) if se > 0 else 0.0


def tercile_prop_z(df, score_col, outcome_col):
    """Binary proportion z-test: high_score - low_score on real outcome rate."""
    sub = df.dropna(subset=[score_col, outcome_col]).copy()
    sub["_tercile"] = pd.qcut(sub[score_col], 3, labels=["low","mid","high"], duplicates="drop")
    by_t = sub.groupby("_tercile", observed=True)[outcome_col].agg(["mean","count"])
    if "low" not in by_t.index or "high" not in by_t.index:
        return None
    p_lo, n_lo = by_t.loc["low","mean"], by_t.loc["low","count"]
    p_hi, n_hi = by_t.loc["high","mean"], by_t.loc["high","count"]
    z = prop_z(p_hi, n_hi, p_lo, n_lo)
    return {
        "low_rate": float(p_lo), "mid_rate": float(by_t.loc["mid","mean"]),
        "high_rate": float(p_hi), "low_n": int(n_lo), "high_n": int(n_hi),
        "z_high_minus_low": z,
    }


def tercile_delta_re(df, score_col, outcome_col):
    """Continuous outcome z-test (delta_run_exp on BIP events): t-test on high vs low tercile means."""
    sub = df.dropna(subset=[score_col, outcome_col]).copy()
    sub["_tercile"] = pd.qcut(sub[score_col], 3, labels=["low","mid","high"], duplicates="drop")
    by_t = sub.groupby("_tercile", observed=True)[outcome_col].agg(["mean","std","count"])
    if "low" not in by_t.index or "high" not in by_t.index:
        return None
    lo = sub.loc[sub["_tercile"] == "low", outcome_col]
    hi = sub.loc[sub["_tercile"] == "high", outcome_col]
    t, _ = stats.ttest_ind(hi, lo, equal_var=False)
    return {
        "low_mean": float(by_t.loc["low","mean"]), "mid_mean": float(by_t.loc["mid","mean"]),
        "high_mean": float(by_t.loc["high","mean"]),
        "low_n": int(by_t.loc["low","count"]), "high_n": int(by_t.loc["high","count"]),
        "t_high_minus_low": float(t),
    }


def stratified_sample(df, outcome_col, n=None, seed=RNG_SEED):
    if n is None:
        n = SHAP_SAMPLE_N
    rng = np.random.default_rng(seed)
    pos_idx = df.index[df[outcome_col] == 1.0].tolist()
    neg_idx = df.index[df[outcome_col] == 0.0].tolist()
    base_rate = df[outcome_col].mean()
    n_pos = int(n * base_rate)
    n_neg = n - n_pos
    idx = (
        rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False).tolist() +
        rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False).tolist()
    )
    return df.loc[idx].copy()


def run_classifier_shap(name, booster, X_samp, feat_cols, rv_weights=None, bip_classes=None):
    """Compute SHAP for one booster. Returns per-pitch scores dict."""
    log(f"  Computing SHAP for {name}...")
    t0 = time.time()
    explainer = shap.TreeExplainer(booster)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X_samp)
    log(f"  SHAP done in {time.time()-t0:.0f}s")
    log(f"  sv type={type(sv).__name__}  shape={np.shape(sv)}")

    # Normalize to sv_array shape (n, n_feats, n_classes) for multiclass
    # or sv_2d shape (n, n_feats) for binary.
    feat_idx = {f: i for i, f in enumerate(feat_cols)}

    is_multiclass = rv_weights is not None

    if is_multiclass:
        # sv is either: list of n_classes arrays (n, n_feats), or ndarray (n, n_feats, n_classes)
        if isinstance(sv, list):
            sv_array = np.stack(sv, axis=2)  # (n, n_feats, n_classes)
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv_array = sv  # already (n, n_feats, n_classes)
        else:
            raise ValueError(f"Unexpected SHAP output for multiclass: type={type(sv)}, shape={np.shape(sv)}")

        def weighted_shap_sum(col_list):
            idx_list = [feat_idx[c] for c in col_list if c in feat_idx]
            geom_by_class = sv_array[:, idx_list, :].sum(axis=1)  # (n, n_classes)
            return (geom_by_class * rv_weights).sum(axis=1)        # (n,)

        lag1_scores   = weighted_shap_sum(LAG1_GEOM_COLS)
        alllag_scores = weighted_shap_sum(ALL_LAG_GEOM_COLS)
        pt_scores     = weighted_shap_sum(PITCH_TYPE_LAG_COLS)
    else:
        # Binary: sv is (n, n_feats) or list [class0, class1]
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[-1]
        # sv is now (n, n_feats)
        def geom_shap_sum(col_list):
            idx_list = [feat_idx[c] for c in col_list if c in feat_idx]
            return sv[:, idx_list].sum(axis=1)

        lag1_scores   = geom_shap_sum(LAG1_GEOM_COLS)
        alllag_scores = geom_shap_sum(ALL_LAG_GEOM_COLS)
        pt_scores     = geom_shap_sum(PITCH_TYPE_LAG_COLS)

    return {
        "lag1_geom_shap":   lag1_scores,
        "alllag_geom_shap": alllag_scores,
        "pt_lag_shap":      pt_scores,
    }


def main():
    global _log_fh
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")
    log("Part A2 — SHAP geometry isolation: remaining 4 classifiers")

    meta       = json.loads(META_PATH.read_text())
    feat_cols  = meta["feat_cols"]
    rv_data    = json.loads(RV_PATH.read_text())
    rv_table   = rv_data["rv_table"]
    bip_classes = rv_data["bip_classes"]
    rv_bip     = np.array([rv_table[c] for c in bip_classes])

    commit_table = pd.read_csv(COMMIT_PATH)
    log(f"Commit table: {len(commit_table):,} rows")

    # Derive category levels
    conn = sqlite3.connect(DB_PATH)
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
    for col in CATEGORICAL_LAG:
        cat_levels[col] = pt_levels

    # Build full feature + outcome matrix
    log(f"\nBuilding lag features for {COMMIT_SEASONS}...")
    all_chunks = []
    t_total = time.time()
    for season in COMMIT_SEASONS:
        t0 = time.time()
        log(f"  Season {season}...")
        df_s = load_pitches(conn, season)
        if len(df_s) == 0:
            continue
        feat = sequential_lag_features(df_s, commit_table, max_lag=MAX_LAG)
        feat = feat.dropna(subset=["lag1_commit_distance","lag1_plate_distance"])
        if len(feat) == 0:
            continue
        feat["season"] = str(season)
        for col in ALL_CAT:
            feat[col] = stringify_keep_na(feat[col])
            levels = cat_levels.get(col)
            feat[col] = pd.Categorical(feat[col], categories=levels) if levels else pd.Categorical(feat[col])

        keep = feat_cols + ["description", "delta_run_exp"]
        all_chunks.append(feat[[c for c in keep if c in feat.columns]].copy())
        log(f"    {len(feat):,} lag rows ({time.time()-t0:.0f}s)")

    conn.close()
    log(f"Load time: {(time.time()-t_total)/60:.1f} min")

    df_full = pd.concat(all_chunks, ignore_index=True)
    log(f"Total rows: {len(df_full):,}")

    # Outcome columns
    df_full["is_swing"] = df_full["description"].isin(SWING_DESCS).astype(float)
    df_full["is_cs"]    = df_full["description"].isin(CS_DESCS).astype(float)
    df_full["is_foul"]  = df_full["description"].isin(FOUL_DESCS).astype(float)
    df_full["is_bip"]   = df_full["description"].isin(BIP_DESCS).astype(float)

    results = {}

    # xWhiff% reference (from part_a_tunneling_shap_geometry.py)
    results["xwhiff"] = {
        "lag1_geom_z":   -1.706,
        "alllag_geom_z": +1.635,
        "pt_lag_z":      +18.622,
        "outcome": "is_whiff",
        "note": "from part_a_tunneling_shap_geometry.py (prior run)",
    }
    # xSwing%, xCS%, xFoul% already completed (200K sample each); hardcode to skip re-running
    results["xswing"] = {
        "outcome": "is_swing", "base_rate": 0.5340, "sample_n": 200000,
        "lag1_geom_z": +73.969, "alllag_geom_z": +153.024, "pt_lag_z": +28.139,
        "note": "from prior run (failed before BIP, results captured from stdout)",
    }
    results["xcs"] = {
        "outcome": "is_cs", "base_rate": 0.1156, "sample_n": 200000,
        "lag1_geom_z": +40.566, "alllag_geom_z": +58.378, "pt_lag_z": +20.694,
        "note": "from prior run (failed before BIP, results captured from stdout)",
    }
    results["xfoul"] = {
        "outcome": "is_foul", "base_rate": 0.2028, "sample_n": 200000,
        "lag1_geom_z": -21.760, "alllag_geom_z": -44.573, "pt_lag_z": -2.055,
        "note": "from prior run (failed before BIP, results captured from stdout)",
    }

    # ---- Only run BIP (the remaining classifier) ----
    classifiers = [
        ("bip",    "is_bip",   True),   # True = multiclass BIP
    ]

    for clf_name, outcome_col, is_bip in classifiers:
        log(f"\n{'='*60}")
        log(f"Classifier: {clf_name}  outcome: {outcome_col}")

        booster = lgb.Booster(model_file=str(MODELS_DIR / f"tunneling_plus_v2_{clf_name}.txt"))

        base_rate = df_full[outcome_col].mean()
        log(f"  Base rate: {base_rate:.4f}")

        # Stratified sample (BIP uses smaller N to keep multiclass SHAP tractable)
        sample_n = BIP_SHAP_SAMPLE if is_bip else SHAP_SAMPLE_N
        df_samp = stratified_sample(df_full, outcome_col, n=sample_n)
        log(f"  Sample: {len(df_samp):,}  (base rate in sample: {df_samp[outcome_col].mean():.4f})")

        X_samp = df_samp[feat_cols]

        # Compute SHAP
        score_dict = run_classifier_shap(
            clf_name, booster, X_samp, feat_cols,
            rv_weights=rv_bip if is_bip else None,
            bip_classes=bip_classes if is_bip else None,
        )

        # Attach scores to sample
        for k, v in score_dict.items():
            df_samp[k] = v

        # Distribution info
        for col, label in [("lag1_geom_shap","lag1-geom"),("alllag_geom_shap","alllag-geom"),("pt_lag_shap","pt-lag")]:
            arr = df_samp[col].values
            log(f"  {label}: mean={arr.mean():.5f}  std={arr.std():.5f}")

        # Tercile z-tests
        log(f"  --- Tercile z-tests ---")
        r1 = tercile_prop_z(df_samp, "lag1_geom_shap",   outcome_col)
        r2 = tercile_prop_z(df_samp, "alllag_geom_shap", outcome_col)
        r3 = tercile_prop_z(df_samp, "pt_lag_shap",      outcome_col)

        for label, r in [("lag1-geom", r1),("alllag-geom", r2),("pt-lag", r3)]:
            if r:
                log(f"    {label}: low={r['low_rate']:.4f}  mid={r['mid_rate']:.4f}  high={r['high_rate']:.4f}  z={r['z_high_minus_low']:.3f}")

        clf_result = {
            "outcome": outcome_col,
            "base_rate": float(base_rate),
            "sample_n": int(len(df_samp)),
            "lag1_geom_z":   float(r1["z_high_minus_low"]) if r1 else None,
            "alllag_geom_z": float(r2["z_high_minus_low"]) if r2 else None,
            "pt_lag_z":      float(r3["z_high_minus_low"]) if r3 else None,
            "lag1_geom_detail":   r1,
            "alllag_geom_detail": r2,
            "pt_lag_detail":      r3,
        }

        # BIP extra: delta_run_exp on BIP pitches
        if is_bip:
            log("  --- BIP xRV-weighted SHAP vs real delta_run_exp ---")
            bip_rows = df_samp[df_samp["is_bip"] == 1.0].copy()
            log(f"  BIP rows in sample: {len(bip_rows):,}")
            if len(bip_rows) > 100:
                for score_col, label in [("lag1_geom_shap","lag1-geom"),("alllag_geom_shap","alllag-geom"),("pt_lag_shap","pt-lag")]:
                    r_dre = tercile_delta_re(bip_rows, score_col, "delta_run_exp")
                    if r_dre:
                        log(f"    {label} vs delta_run_exp: low={r_dre['low_mean']:.4f}  high={r_dre['high_mean']:.4f}  t={r_dre['t_high_minus_low']:.3f}")
                clf_result["bip_dre_lag1"]   = tercile_delta_re(bip_rows, "lag1_geom_shap",   "delta_run_exp")
                clf_result["bip_dre_alllag"] = tercile_delta_re(bip_rows, "alllag_geom_shap", "delta_run_exp")
                clf_result["bip_dre_pt"]     = tercile_delta_re(bip_rows, "pt_lag_shap",      "delta_run_exp")

        results[clf_name] = clf_result

    # Summary table
    log("\n\n=== COMBINED 15-Z-SCORE TABLE (5 classifiers × 3 scores) ===")
    log(f"{'Classifier':12s}  {'lag1-geom z':>12}  {'alllag-geom z':>14}  {'pt-lag z':>10}  {'outcome'}")
    log("-" * 72)
    for name in ["xswing","xcs","xwhiff","xfoul","bip"]:
        r = results[name]
        z1 = r.get("lag1_geom_z")
        z2 = r.get("alllag_geom_z")
        z3 = r.get("pt_lag_z")
        outcome = r.get("outcome","")
        z1s = f"{z1:>+12.3f}" if z1 is not None else "      None  "
        z2s = f"{z2:>+14.3f}" if z2 is not None else "        None  "
        z3s = f"{z3:>+10.3f}" if z3 is not None else "    None  "
        log(f"{name:12s}  {z1s}  {z2s}  {z3s}  ({outcome})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    log(f"\nSaved to {OUT_PATH}")
    _log_fh.close()


if __name__ == "__main__":
    main()
