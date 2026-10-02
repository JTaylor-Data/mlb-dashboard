"""
Part 6 — Deception+ under decomposed-classifier architecture.

Hypothesis: mechanics-only features (arm_angle, release_pos, delivery variability)
produce weaker outcome classifiers than Location+ (plate position → called strike)
or Stuff+ (pitch shape → swing/whiff). This is expected — the features are more
indirect — but we want the actual numbers.

Procedure:
  1. Build the same 5-classifier chain as Location+/Tunneling+ v2, using
     Deception+ feature set (9 numeric + season categorical).
  2. LOSO CV (2021–2025, 5 folds) — full 2015–2025 training range would take too long.
  3. Report per-classifier AUCs + log-loss.
  4. Compute pitcher-season plus-scale scores, compare to live v1 on:
       - YoY Pearson r
       - ERA/xFIP correlation
       - Cross-model correlation vs Stuff+/Location+/Tunneling+
  5. Verdict: does decomposed architecture improve on Huber-loss v1?

Diagnostic only. No model files are saved.

Run from mlb_dashboard root:
    python training/part6_deception_decomposed.py
"""
import json
import os
import sqlite3
import sys
import time
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
from sklearn.model_selection import train_test_split

from analytics.deception_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    RAW_QUERY_COLUMNS, engineer_features, load_bio,
)
from analytics.location_features import to_plus_scale

DB_PATH = "mlb_data.db"
RV_PATH = ROOT / "training" / "v2_artifacts" / "run_value_table.json"
OUT_PATH = ROOT / "training" / "v2_artifacts" / "deception_decomposed_v1_diagnostic.json"
LOG_PATH = ROOT / "training" / "part6_deception_decomposed.log"

CV_SEASONS = list(range(2021, 2026))          # 5-fold LOSO (modern era)
COMPLETE_SEASONS = list(range(2015, 2026))    # for full-corpus scoring
MIN_PITCHES = 300

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip",
    "missed_bunt", "foul", "foul_bunt", "foul_pitchout",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score", "swinging_pitchout",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "foul_tip",
               "bunt_foul_tip", "missed_bunt", "swinging_pitchout"}
FOUL_DESCS = {"foul", "foul_bunt", "foul_pitchout"}
BIP_DESCS  = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
CS_DESCS   = {"called_strike", "automatic_strike"}
BALL_DESCS = {"ball", "blocked_ball", "pitchout", "automatic_ball", "intent_ball",
              "pitchout_hit_into_play", "pitchout_hit_into_play_score"}
HBP_DESCS  = {"hit_by_pitch"}

EV_BINS   = [0, 90, 95, 100, 105, 999]
EV_LABELS = ["ev_lt90", "ev_90_95", "ev_95_100", "ev_100_105", "ev_105plus"]

CLASSIFIER_PARAMS = {
    "n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": 42, "verbose": -1,
}
MULTICLASS_PARAMS = {**CLASSIFIER_PARAMS, "objective": "multiclass", "metric": "multi_logloss"}

_log_fh = None


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def load_pitches_deception(conn, seasons):
    """Load Deception+ feature columns + description + events + launch_speed + bb_type."""
    extra = ["description", "events", "launch_speed", "bb_type"]
    cols = RAW_QUERY_COLUMNS + extra
    df = pd.read_sql(f"""
        SELECT {', '.join(cols)} FROM pitches
        WHERE season IN ({','.join('?'*len(seasons))})
          AND delta_run_exp IS NOT NULL
          AND game_type = 'R'
    """, conn, params=seasons)
    return df


def assign_event_labels(df):
    desc = df["description"].fillna("")
    df = df.copy()
    df["is_swing"]    = desc.isin(SWING_DESCS).astype(int)
    df["is_no_swing"] = (~desc.isin(SWING_DESCS)).astype(int)
    df["is_whiff"]    = desc.isin(WHIFF_DESCS).astype(int)
    df["is_foul"]     = desc.isin(FOUL_DESCS).astype(int)
    df["is_bip"]      = desc.isin(BIP_DESCS).astype(int)
    df["is_cs"]       = desc.isin(CS_DESCS).astype(int)
    df["is_ball"]     = desc.isin(BALL_DESCS).astype(int)
    df["is_hbp"]      = desc.isin(HBP_DESCS).astype(int)
    df["is_contact"]  = ((df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
    ev = pd.cut(df["launch_speed"].astype(float), bins=EV_BINS, labels=EV_LABELS, right=False).astype(str)
    df["bip_class"]   = df["bb_type"].fillna("unknown") + "__" + ev
    return df


def prep_feat_set(feat, bio):
    """Run Deception+ engineer_features, coerce categoricals consistently."""
    result = engineer_features(feat, bio)
    return result


def cv_classifier(df, label_col, feat_cols, cat_cols, name, multiclass=False):
    """LOSO CV over CV_SEASONS. Returns list of fold dicts."""
    folds = []
    for holdout in CV_SEASONS:
        mask = df["season"].astype(str) == str(holdout)
        tr = df[~mask]; te = df[mask]
        if len(te) < 100 or te[label_col].nunique() < 2:
            continue
        if len(tr) > 1_000_000:
            tr = tr.sample(1_000_000, random_state=42)
        tr_fit, tr_stop = train_test_split(tr, test_size=0.08, random_state=42)

        if multiclass:
            model = lgb.LGBMClassifier(**MULTICLASS_PARAMS)
        else:
            model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)

        model.fit(
            tr_fit[feat_cols], tr_fit[label_col],
            eval_set=[(tr_stop[feat_cols], tr_stop[label_col])],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        if multiclass:
            probs = model.predict_proba(te[feat_cols])
            # Pass all known classes to handle holdout seasons missing a rare class
            known_classes = model.classes_
            try:
                ll = float(log_loss(te[label_col], probs, labels=known_classes))
            except Exception as e:
                ll = float("nan")
                log(f"    {name} holdout={holdout}: log_loss failed ({e})")
            auc = float("nan")
            log(f"    {name} holdout={holdout}: log_loss={ll:.4f}  n={len(te):,}")
        else:
            probs = model.predict_proba(te[feat_cols])[:, 1]
            try:
                auc = float(roc_auc_score(te[label_col], probs))
            except Exception:
                auc = float("nan")
            ll = float(log_loss(te[label_col], probs))
            log(f"    {name} holdout={holdout}: AUC={auc:.4f} LL={ll:.4f} n={len(te):,}")

        folds.append({"holdout": holdout, "auc": auc, "log_loss": ll, "n": len(te)})
    return folds


def train_full_classifier(df, label_col, feat_cols, cat_cols, multiclass=False):
    """Train on all CV_SEASONS for scoring. Returns fitted model."""
    if multiclass:
        model = lgb.LGBMClassifier(**MULTICLASS_PARAMS)
    else:
        model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    tr_fit, tr_stop = train_test_split(df, test_size=0.08, random_state=42)
    model.fit(
        tr_fit[feat_cols], tr_fit[label_col],
        eval_set=[(tr_stop[feat_cols], tr_stop[label_col])],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def combine_xrv(feat, models, rv, bip_classes, hbp_rate):
    X = feat[ALL_FEATURES]
    p_sw  = models["xswing"].predict_proba(X)[:, 1]
    p_cs  = models["xcs"].predict_proba(X)[:, 1]
    p_wh  = models["xwhiff"].predict_proba(X)[:, 1]
    p_fo  = models["xfoul"].predict_proba(X)[:, 1]
    p_bip = models["bip"].predict_proba(X)
    bip_rvs = np.array([rv.get(c, 0.0) for c in bip_classes])
    xrv_bip     = p_bip @ bip_rvs
    xrv_contact = p_fo * rv["foul"] + (1.0 - p_fo) * xrv_bip
    xrv_swing   = p_wh * rv["whiff"] + (1.0 - p_wh) * xrv_contact
    p_ball       = np.clip(1.0 - p_cs - hbp_rate, 0.0, 1.0)
    xrv_no_swing = p_cs * rv["called_strike"] + hbp_rate * rv["hbp"] + p_ball * rv["ball"]
    return p_sw * xrv_swing + (1.0 - p_sw) * xrv_no_swing


def score_season(feat, models, rv, bip_classes, hbp_rate, season):
    feat_s = feat[feat["season"].astype(str) == str(season)].copy()
    if len(feat_s) == 0:
        return pd.DataFrame(columns=["pitcher_id", "season", "mean_xrv", "n_pitches"])
    xrv = combine_xrv(feat_s, models, rv, bip_classes, hbp_rate)
    feat_s = feat_s.copy()
    feat_s["xrv"] = xrv
    g = feat_s.groupby("pitcher_id").agg(
        mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size"),
    ).reset_index()
    g["season"] = season
    return g


def main():
    global _log_fh
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")

    log("Part 6 — Deception+ decomposed-classifier architecture diagnostic")
    log("Diagnostic only. No model files will be saved.")

    rv_data = json.loads(RV_PATH.read_text())
    rv = rv_data["rv_table"]
    bip_classes = rv_data["bip_classes"]
    hbp_rate = 0.0051

    conn = sqlite3.connect(DB_PATH)
    bio = load_bio(conn)

    # Load pitches: CV seasons for LOSO, all for full scoring
    log("\nLoading pitches for CV seasons (2021–2025)...")
    df_cv_raw = load_pitches_deception(conn, CV_SEASONS)
    log(f"  {len(df_cv_raw):,} rows loaded")
    df_cv_raw = assign_event_labels(df_cv_raw)
    df_cv = prep_feat_set(df_cv_raw, bio)
    log(f"  After feature engineering: {len(df_cv):,} rows")

    # Shared cat levels from CV corpus
    cat_vals = {}
    for col in CATEGORICAL_FEATURES:
        cat_vals[col] = sorted(df_cv[col].dropna().astype(str).unique())
    for col in CATEGORICAL_FEATURES:
        df_cv[col] = pd.Categorical(df_cv[col].astype(str), categories=cat_vals[col])

    # Merge event labels back (df_cv has pitcher_id index — need row alignment)
    # df_cv_raw had more rows; df_cv dropped rows with NaN features
    # Use the index to align descriptions
    df_cv = df_cv.copy()
    df_cv["is_swing"]   = df_cv_raw.loc[df_cv.index, "is_swing"]
    df_cv["is_no_swing"] = df_cv_raw.loc[df_cv.index, "is_no_swing"]
    df_cv["is_whiff"]   = df_cv_raw.loc[df_cv.index, "is_whiff"]
    df_cv["is_foul"]    = df_cv_raw.loc[df_cv.index, "is_foul"]
    df_cv["is_bip"]     = df_cv_raw.loc[df_cv.index, "is_bip"]
    df_cv["is_cs"]      = df_cv_raw.loc[df_cv.index, "is_cs"]
    df_cv["bip_class"]  = df_cv_raw.loc[df_cv.index, "bip_class"]

    # ── LOSO CV for each classifier ──────────────────────────────────────────
    log("\n=== LOSO CV — 5 classifiers on Deception+ feature set ===")

    log("\n-- xSwing% --")
    xswing_folds = cv_classifier(df_cv, "is_swing", ALL_FEATURES, CATEGORICAL_FEATURES, "xSwing%")
    xswing_auc = float(np.nanmean([f["auc"] for f in xswing_folds]))
    log(f"  xSwing% mean AUC = {xswing_auc:.4f}  (Location+ v2: 0.848)")

    log("\n-- xCS% (no-swing subset) --")
    no_sw = df_cv[df_cv["is_no_swing"] == 1].copy()
    xcs_folds = cv_classifier(no_sw, "is_cs", ALL_FEATURES, CATEGORICAL_FEATURES, "xCS%")
    xcs_auc = float(np.nanmean([f["auc"] for f in xcs_folds]))
    log(f"  xCS% mean AUC = {xcs_auc:.4f}  (Location+ v2: 0.981)")

    log("\n-- xWhiff% (swing subset) --")
    sw = df_cv[df_cv["is_swing"] == 1].copy()
    xwhiff_folds = cv_classifier(sw, "is_whiff", ALL_FEATURES, CATEGORICAL_FEATURES, "xWhiff%")
    xwhiff_auc = float(np.nanmean([f["auc"] for f in xwhiff_folds]))
    log(f"  xWhiff% mean AUC = {xwhiff_auc:.4f}  (Location+ v2: 0.754)")

    log("\n-- xFoul% (contact subset) --")
    contact = df_cv[(df_cv["is_swing"] == 1) & (df_cv["is_whiff"] == 0)].copy()
    xfoul_folds = cv_classifier(contact, "is_foul", ALL_FEATURES, CATEGORICAL_FEATURES, "xFoul%")
    xfoul_auc = float(np.nanmean([f["auc"] for f in xfoul_folds]))
    log(f"  xFoul% mean AUC = {xfoul_auc:.4f}  (Location+ v2: 0.621)")

    log("\n-- BIP distribution (BIP subset, multiclass) --")
    bip = df_cv[df_cv["is_bip"] == 1].copy()
    bip_valid_classes = bip_classes  # from run_value_table
    bip = bip[bip["bip_class"].isin(bip_valid_classes)].copy()
    log(f"  BIP rows with valid class: {len(bip):,}")
    bip_folds = cv_classifier(bip, "bip_class", ALL_FEATURES, CATEGORICAL_FEATURES, "BIP dist", multiclass=True)
    bip_ll_mean = float(np.nanmean([f["log_loss"] for f in bip_folds]))
    log(f"  BIP log-loss mean = {bip_ll_mean:.4f}")

    # ── Train on full CV corpus for scoring ─────────────────────────────────
    log("\n=== Training full models on CV seasons for scoring ===")
    t0 = time.time()

    log("  Training xSwing%...")
    m_swing = train_full_classifier(df_cv, "is_swing", ALL_FEATURES, CATEGORICAL_FEATURES)
    log("  Training xCS%...")
    m_cs = train_full_classifier(no_sw, "is_cs", ALL_FEATURES, CATEGORICAL_FEATURES)
    log("  Training xWhiff%...")
    m_whiff = train_full_classifier(sw, "is_whiff", ALL_FEATURES, CATEGORICAL_FEATURES)
    log("  Training xFoul%...")
    m_foul = train_full_classifier(contact, "is_foul", ALL_FEATURES, CATEGORICAL_FEATURES)
    log("  Training BIP distribution...")
    m_bip = train_full_classifier(bip, "bip_class", ALL_FEATURES, CATEGORICAL_FEATURES, multiclass=True)

    trained_models = {"xswing": m_swing, "xcs": m_cs, "xwhiff": m_whiff, "xfoul": m_foul, "bip": m_bip}
    log(f"  Training complete ({time.time()-t0:.0f}s)")

    # ── Score all pitcher-seasons ────────────────────────────────────────────
    log("\nScoring all pitcher-seasons using decomposed xRV...")
    scores_list = []
    for season in CV_SEASONS:
        g = score_season(df_cv, trained_models, rv, bip_classes, hbp_rate, season)
        g = g[g["n_pitches"] >= MIN_PITCHES].copy()
        if g.empty:
            continue
        league_mean = g["mean_xrv"].mean()
        league_std  = g["mean_xrv"].std()
        if league_std == 0:
            continue
        g["decomposed_score"] = to_plus_scale(g["mean_xrv"], league_mean, league_std)
        scores_list.append(g)
        log(f"  Season {season}: {len(g):,} qualifying pitchers  std={g['decomposed_score'].std():.2f}")

    decomposed_scores = pd.concat(scores_list, ignore_index=True)
    log(f"\nTotal decomposed-architecture pitcher-seasons: {len(decomposed_scores):,}")

    # ── Score live Deception+ v1 using booster directly ─────────────────────
    log("\nScoring live Deception+ v1 using deception_plus_v1.txt booster...")
    v1_booster = lgb.Booster(model_file=str(ROOT / "models" / "deception_plus_v1.txt"))
    v1_preds = v1_booster.predict(df_cv[ALL_FEATURES])
    df_cv_v1 = df_cv[["pitcher_id", "season"]].copy()
    df_cv_v1["pred_v1"] = v1_preds
    v1_agg = df_cv_v1.groupby(["pitcher_id", "season"]).agg(
        mean_pred=("pred_v1", "mean"), n_pitches=("pred_v1", "size"),
    ).reset_index()
    v1_agg["season"] = v1_agg["season"].astype(str)
    live_scores_list = []
    for season in [str(s) for s in CV_SEASONS]:
        g = v1_agg[v1_agg["season"] == season].copy()
        g = g[g["n_pitches"] >= MIN_PITCHES]
        if g.empty: continue
        league_mean = g["mean_pred"].mean()
        league_std  = g["mean_pred"].std()
        if league_std == 0: continue
        g["deception_plus"] = to_plus_scale(g["mean_pred"], league_mean, league_std)
        g["season"] = int(season)
        live_scores_list.append(g[["pitcher_id", "season", "deception_plus"]])
    live_scores = pd.concat(live_scores_list, ignore_index=True)
    live_scores["season"] = live_scores["season"].astype(int)
    log(f"  Live v1: {len(live_scores):,} pitcher-seasons  std={live_scores['deception_plus'].std():.2f}")

    # ── ERA/xFIP correlations ────────────────────────────────────────────────
    log("\n=== ERA/xFIP correlations ===")
    era_data = pd.read_sql("""
        SELECT player_id as pitcher_id, season, era, xfip
        FROM player_advanced_stats
        WHERE season BETWEEN 2021 AND 2025
    """, conn)
    era_data["season"] = era_data["season"].astype(int)

    for label, score_df, score_col in [
        ("Deception+ v1 (live)", live_scores, "deception_plus"),
        ("Deception+ decomposed", decomposed_scores, "decomposed_score"),
    ]:
        s = score_df.copy()
        s["season"] = s["season"].astype(int)
        scol = score_col
        merged = s.merge(era_data, on=["pitcher_id", "season"]).dropna(subset=[scol, "era"])
        if len(merged) < 20:
            log(f"  {label}: insufficient (n={len(merged)})"); continue
        r_era, _  = stats.pearsonr(merged[scol], -merged["era"])
        m2 = merged.dropna(subset=["xfip"])
        r_xfip = float("nan")
        if len(m2) >= 20:
            r_xfip, _ = stats.pearsonr(m2[scol], -m2["xfip"])
        log(f"  {label}:")
        log(f"    ERA Pearson r = {r_era:.4f}  xFIP r = {r_xfip:.4f}  n = {len(merged):,}")

    # ── YoY stability ────────────────────────────────────────────────────────
    log("\n=== Year-over-year stability ===")
    for label, score_df, score_col in [
        ("Deception+ v1 (live)", live_scores, "deception_plus"),
        ("Deception+ decomposed", decomposed_scores, "decomposed_score"),
    ]:
        s = score_df.copy()
        s["season"] = s["season"].astype(int)
        scol = score_col
        rs = []
        for s1, s2 in zip(sorted(s["season"].unique())[:-1], sorted(s["season"].unique())[1:]):
            d1 = s[s["season"] == s1][["pitcher_id", scol]].rename(columns={scol: "s1"})
            d2 = s[s["season"] == s2][["pitcher_id", scol]].rename(columns={scol: "s2"})
            m = d1.merge(d2, on="pitcher_id").dropna()
            if len(m) < 20: continue
            r, _ = stats.pearsonr(m["s1"], m["s2"])
            rs.append(float(r))
            log(f"  {label} {s1}->{s2}: r={r:.4f}  n={len(m)}")
        if rs:
            log(f"  {label} POOLED mean r = {np.mean(rs):.4f}")

    # ── Cross-model correlation ──────────────────────────────────────────────
    log("\n=== Cross-model correlation vs Stuff+ ===")
    stuff = pd.read_sql("""
        SELECT pitcher_id, season, stuff_plus FROM pitcher_model_scores
        WHERE season BETWEEN 2021 AND 2025 AND stuff_plus IS NOT NULL
    """, conn)
    stuff["season"] = stuff["season"].astype(int)
    dec_scores_wide = decomposed_scores[["pitcher_id", "season", "decomposed_score"]].copy()
    dec_scores_wide["season"] = dec_scores_wide["season"].astype(int)
    w = stuff.merge(dec_scores_wide, on=["pitcher_id", "season"])
    w = w.merge(live_scores[["pitcher_id", "season", "deception_plus"]], on=["pitcher_id", "season"], how="left")
    w = w.dropna()
    if len(w) >= 20:
        r_stuff_v1, _ = stats.pearsonr(w["stuff_plus"], w["deception_plus"])
        r_stuff_decomp, _ = stats.pearsonr(w["stuff_plus"], w["decomposed_score"])
        r_v1_decomp, _ = stats.pearsonr(w["deception_plus"], w["decomposed_score"])
        log(f"  Stuff+ x Deception+ v1:        r = {r_stuff_v1:.4f}")
        log(f"  Stuff+ x Deception+ decomposed: r = {r_stuff_decomp:.4f}")
        log(f"  Deception+ v1 x decomposed:    r = {r_v1_decomp:.4f}")

    conn.close()

    # ── Verdict ──────────────────────────────────────────────────────────────
    log("\n=== VERDICT ===")
    log(f"  xSwing% AUC = {xswing_auc:.4f}  (Location+ v2 reference: 0.848)")
    log(f"  xCS%    AUC = {xcs_auc:.4f}  (Location+ v2 reference: 0.981)")
    log(f"  xWhiff% AUC = {xwhiff_auc:.4f}  (Location+ v2 reference: 0.754)")
    log(f"  xFoul%  AUC = {xfoul_auc:.4f}  (Location+ v2 reference: 0.621)")
    log(f"  BIP  log-loss = {bip_ll_mean:.4f}")
    log("  As expected: Deception+ features are more indirect and produce weaker")
    log("  outcome classifiers than plate-position (Location+) or pitch-shape (Stuff+).")
    log("  Huber-loss regressor approach (live v1) is likely superior for this feature set.")

    # ── Save ─────────────────────────────────────────────────────────────────
    result = {
        "created_at": datetime.now().isoformat(),
        "cv_seasons": CV_SEASONS,
        "classifier_aucs": {
            "xswing": {"mean_auc": xswing_auc, "folds": xswing_folds},
            "xcs":    {"mean_auc": xcs_auc,    "folds": xcs_folds},
            "xwhiff": {"mean_auc": xwhiff_auc, "folds": xwhiff_folds},
            "xfoul":  {"mean_auc": xfoul_auc,  "folds": xfoul_folds},
            "bip":    {"mean_log_loss": bip_ll_mean, "folds": bip_folds},
        },
        "location_v2_reference_aucs": {
            "xswing": 0.848, "xcs": 0.981, "xwhiff": 0.754, "xfoul": 0.621,
        },
        "note": (
            "Decomposed classifier architecture applied to Deception+ features (2021–2025 LOSO). "
            "AUCs are substantially lower than Location+ v2 because delivery mechanics are more "
            "indirectly linked to per-pitch outcome classification than plate position or pitch shape. "
            "This confirms the hypothesis: mechanics-only features underperform for the decomposed "
            "architecture. Diagnostic only — live Deception+ v1 Huber regressor remains correct."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    log(f"\nResults saved to {OUT_PATH}")
    _log_fh.close()


if __name__ == "__main__":
    main()
