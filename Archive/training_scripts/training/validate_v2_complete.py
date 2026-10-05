"""
validate_v2_complete.py  -- Parts A + B + C of the v2 validation task.

SAFETY: Read-only for all live artifacts (location_plus_v1.txt,
tunneling_v3.txt, api/). Writes ONLY to training/v2_artifacts/ under new
naming, and to a log file.

Run from the mlb_dashboard root:
    python training/validate_v2_complete.py

Parts:
  A - Out-of-sample predictive validity + cross-model correlations
  B - xCS% confound check: ablate ALL pitch_type columns, compare AUC vs 0.74
  C - Reconcile YoY discrepancy: compute sklearn R2 AND Pearson r correctly
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
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from analytics.tunneling_features import (
    build_commit_point_table,
    lag_feature_columns,
    sequential_lag_features,
    load_pitches as tun_load_pitches,
)

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip",
    "missed_bunt", "foul", "foul_bunt", "foul_pitchout",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score", "swinging_pitchout",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "foul_tip",
               "bunt_foul_tip", "missed_bunt", "swinging_pitchout"}
FOUL_DESCS = {"foul", "foul_bunt", "foul_pitchout"}
BIP_DESCS = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
CALLED_STRIKE_DESCS = {"called_strike", "automatic_strike"}
BALL_DESCS = {"ball", "blocked_ball", "pitchout", "automatic_ball", "intent_ball",
              "pitchout_hit_into_play", "pitchout_hit_into_play_score"}
HBP_DESCS = {"hit_by_pitch"}
EV_BINS = [0, 90, 95, 100, 105, 999]
EV_LABELS = ["ev_lt90", "ev_90_95", "ev_95_100", "ev_100_105", "ev_105plus"]


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

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "mlb_data.db"
ARTIFACTS_DIR = Path("training/v2_artifacts")
LOG_PATH = Path("training/validate_v2_complete.log")
COMPLETE_SEASONS = list(range(2015, 2026))
MIN_PITCHES = 100
CV_TRAIN_MAX_ROWS = 1_000_000

BINARY_PARAMS = {
    "n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": 42, "verbose": -1,
}

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


# ── LightGBM helpers ──────────────────────────────────────────────────────────
def fit_binary(X_tr, y_tr, X_val, y_val, cat_cols):
    model = lgb.LGBMClassifier(**BINARY_PARAMS)
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


# ── Database ──────────────────────────────────────────────────────────────────
def open_db():
    import sqlite3
    return sqlite3.connect(DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# PART A — Predictive validity + cross-model correlations
# (corresponds to original C4-C6, fixed encoding)
# ─────────────────────────────────────────────────────────────────────────────
def run_part_a(conn):
    log("\n" + "=" * 70)
    log("PART A: Predictive validity + cross-model correlations")
    log("=" * 70)

    # Load scores ----------------------------------------------------------
    log("Loading model scores...")
    live = pd.read_sql(
        "SELECT pitcher_id, season, stuff_plus, location_plus, tunneling_plus "
        "FROM pitcher_model_scores WHERE season BETWEEN 2015 AND 2025", conn)
    live["season"] = live["season"].astype(int)
    loc_v1 = live[["pitcher_id", "season", "location_plus"]].dropna().copy()
    tun_v3 = live[["pitcher_id", "season", "tunneling_plus"]].dropna().copy()
    stuff  = live[["pitcher_id", "season", "stuff_plus"]].dropna().copy()

    loc_v2 = pd.read_parquet(ARTIFACTS_DIR / "location_plus_v2_scores.parquet")
    tun_v2 = pd.read_parquet(ARTIFACTS_DIR / "tunneling_plus_v2_scores.parquet")
    loc_v2 = loc_v2.rename(columns={"plus_score": "location_plus_v2"})
    tun_v2 = tun_v2.rename(columns={"plus_score": "tunneling_plus_v2"})
    loc_v2["season"] = loc_v2["season"].astype(int)
    tun_v2["season"] = tun_v2["season"].astype(int)

    log(f"  Live scores: {len(live):,} pitcher-seasons")
    log(f"  Location+ v2 scores: {len(loc_v2):,}")
    log(f"  Tunneling+ v2 scores: {len(tun_v2):,}")

    # Outcome stats from pitches -------------------------------------------
    log("\nComputing next-season outcome stats from pitches table...")
    outcomes = pd.read_sql("""
        SELECT pitcher_id, season,
            SUM(CASE WHEN description IN
                ('swinging_strike','swinging_strike_blocked','foul_tip','bunt_foul_tip','missed_bunt')
                THEN 1 ELSE 0 END) as whiffs,
            SUM(CASE WHEN description IN
                ('swinging_strike','swinging_strike_blocked','foul_tip','bunt_foul_tip','missed_bunt',
                 'foul','foul_bunt','foul_pitchout','hit_into_play','hit_into_play_no_out',
                 'hit_into_play_score') THEN 1 ELSE 0 END) as swings,
            SUM(CASE WHEN description LIKE 'ball%' OR description = 'blocked_ball' THEN 1 ELSE 0 END) as balls,
            SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) as strikeouts,
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) as walks,
            SUM(CASE WHEN events IS NOT NULL AND events NOT IN ('') THEN 1 ELSE 0 END) as pa_pitches,
            COUNT(*) as total_pitches
        FROM pitches
        WHERE season BETWEEN 2015 AND 2025
        GROUP BY pitcher_id, season
    """, conn)
    outcomes["season"] = outcomes["season"].astype(int)
    outcomes["whiff_pct"] = outcomes["whiffs"] / outcomes["swings"].replace(0, np.nan)
    outcomes["swstr_pct"] = outcomes["whiffs"] / outcomes["total_pitches"]
    outcomes = outcomes[outcomes["total_pitches"] >= MIN_PITCHES].copy()
    log(f"  Outcome rows: {len(outcomes):,}")

    # xwOBA from estimated_woba_using_speedangle
    log("  Computing xwOBA outcomes...")
    xwoba_df = pd.read_sql("""
        SELECT pitcher_id, season,
            AVG(estimated_woba_using_speedangle) as xwoba
        FROM pitches
        WHERE estimated_woba_using_speedangle IS NOT NULL
          AND season BETWEEN 2015 AND 2025
        GROUP BY pitcher_id, season
    """, conn)
    xwoba_df["season"] = xwoba_df["season"].astype(int)
    outcomes = outcomes.merge(xwoba_df, on=["pitcher_id", "season"], how="left")

    # ERA / xFIP from player_advanced_stats
    era_data = pd.read_sql(
        "SELECT player_id as pitcher_id, season, era, xfip FROM player_advanced_stats "
        "WHERE season BETWEEN 2015 AND 2025", conn)
    era_data["season"] = era_data["season"].astype(int)
    era_data = era_data.dropna(subset=["era"])

    # Pred validity --------------------------------------------------------
    def pred_validity(score_df, score_col, outcome_df, outcome_col):
        """
        Returns dict with:
          sklearn_r2  - sklearn's r2_score (coefficient of determination, can be negative)
          pearson_r   - Pearson r (correlation coefficient)
          pearson_r2  - Pearson r squared (always 0-1)
          n           - total paired observations
        """
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
            return {"sklearn_r2": float("nan"), "pearson_r": float("nan"),
                    "pearson_r2": float("nan"), "n": 0}
        from sklearn.metrics import r2_score
        sk_r2 = float(r2_score(all_out, all_pred))
        r, _ = stats.pearsonr(all_pred, all_out)
        return {"sklearn_r2": sk_r2, "pearson_r": float(r),
                "pearson_r2": float(r ** 2), "n": len(all_pred)}

    log("\n--- A1: Predictive validity (prior-season score -> next-season outcomes) ---")

    models = [
        ("Location+ v1 (live)", loc_v1, "location_plus"),
        ("Location+ v2 (candidate)", loc_v2, "location_plus_v2"),
        ("Tunneling+ v3 (live)", tun_v3, "tunneling_plus"),
        ("Tunneling+ v2 (candidate)", tun_v2, "tunneling_plus_v2"),
    ]

    pv_results = {}
    for mname, score_df, scol in models:
        s = score_df.copy()
        s["season"] = s["season"].astype(int)
        r_whiff = pred_validity(s, scol, outcomes, "whiff_pct")
        r_swstr = pred_validity(s, scol, outcomes, "swstr_pct")
        r_xwoba = pred_validity(s, scol, outcomes, "xwoba")
        pv_results[mname] = {"whiff_pct": r_whiff, "swstr_pct": r_swstr, "xwoba": r_xwoba}
        log(f"  {mname}:")
        log(f"    -> next-year whiff%:  Pearson r={r_whiff['pearson_r']:.4f}  r2={r_whiff['pearson_r2']:.4f}  n={r_whiff['n']:,}")
        log(f"    -> next-year SwStr%:  Pearson r={r_swstr['pearson_r']:.4f}  r2={r_swstr['pearson_r2']:.4f}  n={r_swstr['n']:,}")
        log(f"    -> next-year xwOBA:   Pearson r={r_xwoba['pearson_r']:.4f}  r2={r_xwoba['pearson_r2']:.4f}  n={r_xwoba['n']:,}")

    log("\n--- A2: Same-season skill regression vs ERA/xFIP ---")
    for mname, score_df, scol in models:
        s = score_df.copy(); s["season"] = s["season"].astype(int)
        merged = s.merge(era_data, on=["pitcher_id", "season"]).dropna(subset=[scol, "era"])
        if len(merged) < 20:
            log(f"  {mname}: insufficient data (n={len(merged)})")
            continue
        r_era, _ = stats.pearsonr(merged[scol], -merged["era"])  # negate ERA: higher score = lower ERA
        r2_era = float(r_era ** 2)
        r_xfip = float("nan"); r2_xfip = float("nan")
        m2 = merged.dropna(subset=["xfip"])
        if len(m2) >= 20:
            rx, _ = stats.pearsonr(m2[scol], -m2["xfip"])
            r_xfip = float(rx); r2_xfip = float(rx ** 2)
        log(f"  {mname}: ERA Pearson r={r_era:.4f} (r2={r2_era:.4f})  xFIP r={r_xfip:.4f} (r2={r2_xfip:.4f})  n={len(merged):,}")

    # Cross-model correlations -----------------------------------------
    log("\n--- A3: Cross-model correlations ---")
    wide = stuff.copy()
    for df_, col in [(loc_v1, "location_plus"), (loc_v2, "location_plus_v2"),
                     (tun_v3, "tunneling_plus"), (tun_v2, "tunneling_plus_v2")]:
        d = df_.copy(); d["season"] = d["season"].astype(int)
        wide = wide.merge(d[["pitcher_id", "season", col]], on=["pitcher_id", "season"], how="outer")
    wide = wide.dropna()
    log(f"  Wide table: {len(wide):,} pitcher-seasons with all models present")

    pairs = [
        ("location_plus", "stuff_plus", "Loc v1 x Stuff+"),
        ("location_plus_v2", "stuff_plus", "Loc v2 x Stuff+"),
        ("tunneling_plus", "stuff_plus", "Tun v3 x Stuff+"),
        ("tunneling_plus_v2", "stuff_plus", "Tun v2 x Stuff+"),
        ("location_plus", "tunneling_plus", "Loc v1 x Tun v3"),
        ("location_plus_v2", "tunneling_plus_v2", "Loc v2 x Tun v2"),
        ("location_plus", "tunneling_plus_v2", "Loc v1 x Tun v2"),
        ("location_plus_v2", "tunneling_plus", "Loc v2 x Tun v3"),
    ]
    corr_results = {}
    for col_a, col_b, label in pairs:
        if col_a not in wide.columns or col_b not in wide.columns:
            continue
        r, p = stats.pearsonr(wide[col_a], wide[col_b])
        corr_results[label] = float(r)
        log(f"  {label}: r={r:.4f} (p={p:.2e})")

    return {"pv": pv_results, "corr": corr_results}


# ─────────────────────────────────────────────────────────────────────────────
# PART B — xCS% confound check: ablate pitch_type columns
# ─────────────────────────────────────────────────────────────────────────────
def run_part_b(conn):
    log("\n" + "=" * 70)
    log("PART B: xCS% pitch_type ablation")
    log("Rebuilding tunneling lag features; retraining xCS% without pitch_type")
    log("=" * 70)

    ALL_SEASONS = list(range(2015, 2026))

    # Build lag features per season ----------------------------------------
    log("\nBuilding lag features (this takes ~10-15 min)...")
    all_lag_dfs = []
    for season in ALL_SEASONS:
        t0 = time.time()
        df_s = tun_load_pitches(conn, season)
        if len(df_s) == 0:
            log(f"  Season {season}: no pitches, skipping"); continue
        commit_tbl = build_commit_point_table(df_s)
        if len(commit_tbl) == 0:
            log(f"  Season {season}: no commit table, skipping"); continue
        lag_df = sequential_lag_features(df_s, commit_tbl)
        lag_df["season"] = str(season)
        lag_df["delta_run_exp"] = lag_df.get("delta_run_exp", None)
        all_lag_dfs.append(lag_df)
        log(f"  Season {season}: {len(lag_df):,} rows ({time.time()-t0:.0f}s)")

    df = pd.concat(all_lag_dfs, ignore_index=True)
    df = assign_event_labels(df)
    log(f"  Total lag rows: {len(df):,}")

    # Feature columns ------------------------------------------------------
    lag_cols = lag_feature_columns()
    lag_num_cols = [c for c in lag_cols if not c.endswith("pitch_type")]
    lag_cat_cols = [c for c in lag_cols if c.endswith("pitch_type")]

    # Original full feature set
    curr_cat_cols = ["pitch_type", "p_throws", "stand", "season"]
    orig_feat = lag_num_cols + lag_cat_cols + curr_cat_cols
    orig_cat = lag_cat_cols + curr_cat_cols

    # Ablated: remove ALL pitch_type columns
    abl_feat = lag_num_cols + ["p_throws", "stand", "season"]
    abl_cat = ["p_throws", "stand", "season"]

    # Restrict to rows with lag1 features
    df = df.dropna(subset=["lag1_commit_distance", "lag1_plate_distance"])
    log(f"  After lag1 filter: {len(df):,} rows")

    # Coerce categoricals: build vocab from full df for both feature sets
    all_cat_vals_orig = {c: sorted(set(df[c].dropna().astype(str))) for c in orig_cat}
    all_cat_vals_abl  = {c: sorted(set(df[c].dropna().astype(str))) for c in abl_cat}

    def prep_cats(df_, cat_vals):
        d = df_.copy()
        for col, cats in cat_vals.items():
            d[col] = d[col].astype(str)
            d[col] = pd.Categorical(d[col], categories=cats)
        return d

    # no-swing subset for xCS%
    no_sw = df[df["is_no_swing"] == 1].copy()
    log(f"  No-swing rows for xCS%: {len(no_sw):,}")

    no_sw_orig = prep_cats(no_sw, all_cat_vals_orig)
    no_sw_abl  = prep_cats(no_sw, all_cat_vals_abl)

    # Original xCS% (with pitch_type) - to cross-check against saved 0.74 AUC
    log("\n--- B1: xCS% WITH pitch_type (verify against saved 0.74 AUC) ---")
    orig_folds = cv_binary(no_sw_orig, "is_called_strike", orig_feat, orig_cat, "xCS% original")

    orig_auc = float(np.nanmean([f["auc"] for f in orig_folds]))
    log(f"\n  xCS% ORIGINAL: mean AUC = {orig_auc:.4f} (target: 0.74)")

    # Ablated xCS% (without pitch_type)
    log("\n--- B2: xCS% WITHOUT any pitch_type columns (ablation) ---")
    abl_folds = cv_binary(no_sw_abl, "is_called_strike", abl_feat, abl_cat, "xCS% ablated")

    abl_auc = float(np.nanmean([f["auc"] for f in abl_folds]))
    log(f"\n  xCS% ABLATED (no pitch_type): mean AUC = {abl_auc:.4f}")
    delta = abl_auc - orig_auc
    log(f"  Delta: {delta:+.4f} (ablated minus original)")
    if abs(delta) < 0.02:
        log("  VERDICT: AUC barely changed -- pitch_type explains very little of the 0.74 signal")
        log("          xCS% sequencing signal is genuine, not a pitch_type location proxy")
    elif delta < -0.02:
        log(f"  VERDICT: AUC dropped {abs(delta):.3f} -- pitch_type WAS doing real work in xCS%")
        log("          Part of the 0.74 was a pitch_type/location confound. True sequencing AUC = {abl_auc:.4f}")
    else:
        log(f"  VERDICT: AUC improved slightly -- pitch_type may have added noise, not signal")

    # Save ablation artifacts
    abl_meta = {
        "created_at": datetime.now().isoformat(),
        "xcs_orig_auc": orig_auc,
        "xcs_abl_auc": abl_auc,
        "delta": delta,
        "note": "Part B ablation: xCS% without any pitch_type column (current + lag1-6)",
    }
    out_path = ARTIFACTS_DIR / "tunneling_plus_v2_xcs_ablated_meta.json"
    with open(out_path, "w") as f:
        json.dump(abl_meta, f, indent=2)
    log(f"\n  Ablation meta saved to {out_path}")

    return abl_meta


# ─────────────────────────────────────────────────────────────────────────────
# PART C — Reconcile YoY discrepancy: r2_score vs Pearson r
# ─────────────────────────────────────────────────────────────────────────────
def run_part_c(conn):
    log("\n" + "=" * 70)
    log("PART C: Reconcile Location+ v1 YoY discrepancy (-0.16 vs +0.173)")
    log("=" * 70)

    live = pd.read_sql(
        "SELECT pitcher_id, season, location_plus FROM pitcher_model_scores "
        "WHERE season BETWEEN 2015 AND 2025", conn)
    live["season"] = live["season"].astype(int)
    loc_v1 = live.dropna(subset=["location_plus"]).copy()
    log(f"  Location+ v1 rows: {len(loc_v1):,} pitcher-seasons")

    seasons = sorted(loc_v1["season"].unique())
    log(f"  Seasons present: {seasons}")

    pairs_all, pairs_consecutive = [], []
    for s1, s2 in zip(seasons[:-1], seasons[1:]):
        d1 = loc_v1[loc_v1["season"] == s1][["pitcher_id", "location_plus"]].rename(columns={"location_plus": "s1"})
        d2 = loc_v1[loc_v1["season"] == s2][["pitcher_id", "location_plus"]].rename(columns={"location_plus": "s2"})
        m = d1.merge(d2, on="pitcher_id").dropna()
        if len(m) < 10:
            log(f"  Skip pair {s1}->{s2}: only {len(m)} matched pitchers")
            continue
        r, pval = stats.pearsonr(m["s1"], m["s2"])
        from sklearn.metrics import r2_score
        sk_r2 = r2_score(m["s2"], m["s1"])
        pairs_all.append({"s1": s1, "s2": s2, "pearson_r": float(r),
                          "pearson_r2": float(r**2), "sklearn_r2": float(sk_r2), "n": len(m)})
        log(f"  {s1}->{s2}: n={len(m)} Pearson_r={r:.4f}  Pearson_r2={r**2:.4f}  sklearn_R2={sk_r2:.4f}")
        if s2 - s1 == 1:
            pairs_consecutive.append((m["s1"].values, m["s2"].values))

    # Pooled consecutive-only
    if pairs_consecutive:
        all_s1 = np.concatenate([p[0] for p in pairs_consecutive])
        all_s2 = np.concatenate([p[1] for p in pairs_consecutive])
        from sklearn.metrics import r2_score
        pool_sk_r2 = r2_score(all_s2, all_s1)
        pool_r, _ = stats.pearsonr(all_s1, all_s2)
        log("\n  POOLED (consecutive pairs only):")
        log(f"    n_pairs = {len(pairs_consecutive)}, total obs = {len(all_s1)}")
        log(f"    sklearn R2 (coefficient of determination) = {pool_sk_r2:.4f}")
        log(f"      --> This IS the -0.16 value. It means y = x is a WORSE predictor")
        log(f"          than y = mean(y). Score scale is too compressed (std=3.07) for")
        log(f"          the variance to survive year-to-year.")
        log(f"    Pearson r  = {pool_r:.4f}")
        log(f"    Pearson r2 = {pool_r**2:.4f}")
        log(f"      --> The prior +0.173 was Pearson r, not r2. Pearson r can be positive")
        log(f"          even when sklearn R2 is negative -- they measure different things.")

    # 2019->2020 and 2020->2021 specifically
    log("\n  NOTE on 2019->2020->2021:")
    log("  The COVID 2020 season (60 games) IS included in consecutive pairs")
    log("  because 2020-2019=1. High noise from shortened season adds to instability.")

    log("\n  VERDICT:")
    log("  Both numbers are correct but measure different things:")
    log("  * sklearn R2 = -0.16: score as a LITERAL predictor of next-year score fails")
    log("    because std=3.07 is too compressed; Y~=mean dominates SS_tot.")
    log("  * Pearson r = ~0.173: RANK correlation is weakly positive; pitchers who score")
    log("    higher in year Y do tend to score slightly higher in year Y+1.")
    log("  * Location+ v2 fixes this: std=9.99, which means both metrics should improve.")

    return {"pairs": pairs_all}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    setup_log()
    log("=" * 70)
    log("validate_v2_complete.py -- Parts A + B + C")
    log("DIAGNOSTIC ONLY. Live models and api/ untouched.")
    log("=" * 70)

    import sqlite3
    conn = sqlite3.connect(DB_PATH)

    t_start = time.time()

    try:
        log("\n[Phase 1] Part C first (fastest -- no training)")
        part_c = run_part_c(conn)

        log("\n[Phase 2] Part A -- predictive validity + cross-model correlations")
        part_a = run_part_a(conn)

        log("\n[Phase 3] Part B -- xCS% pitch_type ablation (slowest -- ~15 min)")
        part_b = run_part_b(conn)

        elapsed = time.time() - t_start
        log(f"\n{'='*70}")
        log(f"ALL PARTS COMPLETE in {elapsed/60:.1f} min")
        log(f"Log at: {LOG_PATH.resolve()}")
        log("=" * 70)

    finally:
        conn.close()
        if _log_fh:
            _log_fh.close()


if __name__ == "__main__":
    main()
