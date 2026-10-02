"""
Part B resume — ERA/xFIP section only.
Reuses all LOSO and scoring already done; just runs the ERA correlation
and v2 reference sections that failed due to x_fip column name bug.

Pre-computed results hardcoded from the LOSO output:
  LOSO mean AUCs: xswing=0.5850, xcs=0.6426, xwhiff=0.5923, xfoul=0.5822
  Score spread: mean=100, std=10, p10=88.6, p90=109.9
  YoY pooled r: 0.030
  Split-half r: 0.946
  xRV vs whiff% r: -0.1376
  xRV vs swing% r: -0.0138
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
from sklearn.metrics import roc_auc_score

from analytics.tunneling_features import (
    lag_feature_columns, load_pitches, sequential_lag_features, MAX_LAG,
)
from analytics.location_features import to_plus_scale

from training.part_b_sequencing_plus_v1 import (
    build_arsenal_entropy, add_sequencing_features, add_outcome_labels,
    stringify_keep_na, combine_xrv,
    ALL_FEATURES, CATEGORICAL_COLS, ALL_CAT_V2, PT_LAG_COLS, NEW_FEATURES,
    CLASSIFIER_PARAMS, EV_BINS, EV_LABELS,
    MIN_PITCHES, COMMIT_SEASONS, SCORING_SEASONS,
)

RV_PATH    = ROOT / "training" / "v2_artifacts" / "run_value_table.json"
COMMIT_PATH = (
    ROOT / "project_artifacts" / "training_artifacts"
    / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv"
)
V2_MODELS_DIR = ROOT / "models"
OUT_PATH   = ROOT / "training" / "v2_artifacts" / "sequencing_plus_v1_diagnostic.json"
DB_PATH    = "mlb_data.db"

LOG_PATH = ROOT / "training" / "part_b_era_resume.log"

_log_fh = None

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n"); _log_fh.flush()


def main():
    global _log_fh
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")
    log("Part B ERA resume — ERA/xFIP section + v2 reference")

    rv_data     = json.loads(RV_PATH.read_text())
    rv_table    = rv_data["rv_table"]
    bip_classes = rv_data["bip_classes"]

    commit_table = pd.read_csv(COMMIT_PATH)
    conn = sqlite3.connect(DB_PATH)

    # Category levels
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

    # Arsenal entropy
    log("Building arsenal entropy...")
    arsenal_entropy_df = build_arsenal_entropy(conn, SCORING_SEASONS)
    log(f"  {len(arsenal_entropy_df):,} pitcher-season entropy values")

    # Reload CV data (needed to retrain full models)
    log("Loading CV data (2021-2025)...")
    t0 = time.time()
    chunks = []
    for season in COMMIT_SEASONS:
        t1 = time.time()
        log(f"  Season {season}...")
        df_s = load_pitches(conn, season)
        if len(df_s) == 0:
            continue
        # BIP extras
        extra = pd.read_sql(f"""
            SELECT game_id, pitcher_id, at_bat_number, pitch_number,
                   bb_type, launch_speed
            FROM pitches WHERE season = {season} AND game_type = 'R'
        """, conn)
        df_s = df_s.merge(extra, on=["game_id","pitcher_id","at_bat_number","pitch_number"], how="left")
        feat = sequential_lag_features(df_s, commit_table, max_lag=MAX_LAG)
        feat = feat.dropna(subset=["lag1_commit_distance","lag1_plate_distance"])
        if len(feat) == 0:
            continue
        feat["season"] = str(season)
        feat = add_sequencing_features(feat, arsenal_entropy_df)
        for col in CATEGORICAL_COLS:
            feat[col] = stringify_keep_na(feat[col])
            levels = cat_levels.get(col)
            feat[col] = pd.Categorical(feat[col], categories=levels) if levels else pd.Categorical(feat[col])
        feat = add_outcome_labels(feat)
        chunks.append(feat)
        log(f"    {len(feat):,} rows ({time.time()-t1:.0f}s)")
    df_cv = pd.concat(chunks, ignore_index=True)
    log(f"CV data loaded: {len(df_cv):,} rows ({time.time()-t0:.1f}s)")

    bip_universe = sorted(df_cv["bip_class"].dropna().unique().tolist())
    cat_feat_names = [c for c in CATEGORICAL_COLS if c in ALL_FEATURES]

    # Train full models
    log("Training full models on 2021-2025...")

    def train_full(target, multiclass=False, bip_univ=None):
        sub = df_cv.dropna(subset=[target])
        X = sub[ALL_FEATURES]; y = sub[target]
        if multiclass:
            m = lgb.LGBMClassifier(objective="multiclass", num_class=len(bip_univ), **CLASSIFIER_PARAMS)
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

    # Score full CV data
    log("Scoring full CV corpus...")
    X_all = df_cv[ALL_FEATURES]
    xrv_all = combine_xrv(X_all, {"xswing":m_sw,"xcs":m_cs,"xwhiff":m_wh,"xfoul":m_fo,"bip":m_bip}, rv_table, bip_classes)
    df_cv["xrv_full"] = xrv_all

    ps_cv = df_cv.groupby(["pitcher_id","season"]).agg(
        mean_xrv=("xrv_full","mean"), n=("xrv_full","size")
    ).reset_index()
    ps_cv = ps_cv[ps_cv["n"] >= MIN_PITCHES]
    records = []
    for season, grp in ps_cv.groupby("season"):
        vals = grp["mean_xrv"].values
        plus = to_plus_scale(vals, vals.mean(), vals.std())
        for i, (_, row) in enumerate(grp.iterrows()):
            records.append({"pitcher_id": row["pitcher_id"], "season": str(row["season"]),
                             "sequencing_plus": float(plus[i]), "n": int(row["n"])})
    df_ps = pd.DataFrame(records)
    log(f"  {len(df_ps):,} pitcher-season scores")

    # ERA / xFIP
    log("Loading ERA/xFIP...")
    adv = pd.read_sql("""
        SELECT player_id AS pitcher_id, season, era, xfip
        FROM player_advanced_stats
        WHERE season BETWEEN 2021 AND 2025
          AND era IS NOT NULL AND xfip IS NOT NULL
    """, conn)
    adv["season"] = adv["season"].astype(str)

    r_era = r_xfip = None
    merged_era = pd.DataFrame()
    if len(adv):
        merged_era = df_ps.merge(adv, on=["pitcher_id","season"])
        if len(merged_era) > 30:
            r_era,  _ = stats.pearsonr(merged_era["sequencing_plus"], merged_era["era"])
            r_xfip, _ = stats.pearsonr(merged_era["sequencing_plus"], merged_era["xfip"])
            log(f"  Seq+ ERA r  = {r_era:.4f}  (n={len(merged_era)})")
            log(f"  Seq+ xFIP r = {r_xfip:.4f}")

    # v2 reference
    log("Scoring Tunneling+ v2 reference...")
    v2_models = {n: lgb.Booster(model_file=str(V2_MODELS_DIR / f"tunneling_plus_v2_{n}.txt"))
                 for n in ("xswing","xcs","xwhiff","xfoul","bip")}
    v2_feat_cols = json.loads((V2_MODELS_DIR / "tunneling_plus_v2_meta.json").read_text())["feat_cols"]

    X_v2 = df_cv[v2_feat_cols]
    xrv_v2 = combine_xrv(X_v2, {"xswing":v2_models["xswing"],"xcs":v2_models["xcs"],
                                  "xwhiff":v2_models["xwhiff"],"xfoul":v2_models["xfoul"],
                                  "bip":v2_models["bip"]}, rv_table, bip_classes)
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
            v2_records.append({"pitcher_id": row["pitcher_id"], "season": str(row["season"]),
                                "v2_plus": float(plus[i])})
    df_v2_ps = pd.DataFrame(v2_records)
    log(f"  v2 pitcher-seasons: {len(df_v2_ps):,}")

    v2_r_era = v2_r_xfip = None
    if len(adv):
        v2_era_df = df_v2_ps.merge(adv, on=["pitcher_id","season"])
        if len(v2_era_df) > 30:
            v2_r_era  = float(stats.pearsonr(v2_era_df["v2_plus"], v2_era_df["era"])[0])
            v2_r_xfip = float(stats.pearsonr(v2_era_df["v2_plus"], v2_era_df["xfip"])[0])
            log(f"  v2 ERA r  = {v2_r_era:.4f}  xFIP r = {v2_r_xfip:.4f}")

    # v2 YoY
    v2_yoy_pairs = []
    for i in range(len(COMMIT_SEASONS) - 1):
        s_t  = str(COMMIT_SEASONS[i])
        s_t1 = str(COMMIT_SEASONS[i + 1])
        merged_yoy = df_v2_ps[df_v2_ps["season"]==s_t].merge(
            df_v2_ps[df_v2_ps["season"]==s_t1], on="pitcher_id", suffixes=("_t","_t1"))
        if len(merged_yoy) > 30:
            r_yoy, _ = stats.pearsonr(merged_yoy["v2_plus_t"], merged_yoy["v2_plus_t1"])
            v2_yoy_pairs.append(float(r_yoy))
            log(f"  v2 YoY {s_t}-{s_t1}: r={r_yoy:.3f}")
    v2_pooled_yoy = float(np.mean(v2_yoy_pairs)) if v2_yoy_pairs else None
    log(f"  v2 pooled YoY r = {v2_pooled_yoy:.3f}" if v2_pooled_yoy else "  v2 YoY: n/a")

    # v2 AUCs on full corpus
    log("v2 reference AUCs on full corpus...")
    v2_ref_aucs = {}
    for name, col in [("xswing","is_swing"),("xcs","is_cs"),("xwhiff","is_whiff"),("xfoul","is_foul")]:
        sub = df_cv.dropna(subset=[col])
        prob = v2_models[name].predict(sub[v2_feat_cols])
        y = sub[col].astype(int)
        if len(y.unique()) > 1:
            v2_ref_aucs[name] = float(roc_auc_score(y, prob))
        log(f"  v2 {name} AUC = {v2_ref_aucs.get(name, 'n/a'):.4f}")

    # Seq+ full-corpus AUCs
    seq_full_aucs = {}
    for name, col, model in [("xswing","is_swing",m_sw),("xcs","is_cs",m_cs),("xwhiff","is_whiff",m_wh),("xfoul","is_foul",m_fo)]:
        sub = df_cv.dropna(subset=[col])
        prob = model.predict_proba(sub[ALL_FEATURES])[:, 1]
        y = sub[col].astype(int)
        if len(y.unique()) > 1:
            seq_full_aucs[name] = float(roc_auc_score(y, prob))
        log(f"  Seq+ {name} AUC = {seq_full_aucs.get(name, 'n/a'):.4f}")

    conn.close()

    # ── Side-by-side summary ────────────────────────────────────────────────
    log("\n=== SIDE-BY-SIDE: Sequencing+ v1 vs Tunneling+ v2 ===")
    def sfmt(v): return f"{v:+.3f}" if v is not None else "   n/a"
    log(f"  {'Metric':35s} {'Seq+ v1':>10} {'Tun+ v2':>10}")
    log(f"  {'-'*57}")
    log(f"  {'ERA Pearson r':35s} {sfmt(r_era):>10} {sfmt(v2_r_era):>10}")
    log(f"  {'xFIP Pearson r':35s} {sfmt(r_xfip):>10} {sfmt(v2_r_xfip):>10}")
    log(f"  {'YoY stability (pooled r)':35s} {'  +0.030':>10} {sfmt(v2_pooled_yoy):>10}")
    log(f"  {'Split-half r':35s} {'  +0.946':>10} {'     ---':>10}")
    log(f"  {'xRV vs whiff% r':35s} {'  -0.138':>10} {'     ---':>10}")
    for name in ["xswing","xcs","xwhiff","xfoul"]:
        log(f"  {'LOSO AUC '+name:35s} {'n/a (prev)':>10} {sfmt(v2_ref_aucs.get(name)):>10}")

    # Save
    output = {
        "model_name": "sequencing_plus_v1",
        "note": "DIAGNOSTIC ONLY — provisional name; not promoted to production",
        "feature_set": {
            "kept_from_v2": list(ALL_CAT_V2),
            "dropped_from_v2": ["lagN_commit_distance", "lagN_plate_distance", "lagN_shape_break (all 6 lags)"],
            "added": NEW_FEATURES,
            "excluded": ["count (balls/strikes)", "plate location"],
        },
        "cv_seasons": COMMIT_SEASONS,
        "n_pitcher_seasons": int(len(df_ps)),
        "score_spread": {"mean": 100.0, "std": 10.0, "p10": 88.6, "p90": 109.9},
        "loso_mean_aucs": {"xswing": 0.5850, "xcs": 0.6426, "xwhiff": 0.5923, "xfoul": 0.5822},
        "v2_reference_full_corpus_aucs": v2_ref_aucs,
        "seq_full_corpus_aucs": seq_full_aucs,
        "yoy_stability": {"pooled_r": 0.030, "per_pair": [
            {"season_t":"2021","season_t1":"2022","r":-0.041},
            {"season_t":"2022","season_t1":"2023","r":0.099},
            {"season_t":"2023","season_t1":"2024","r":-0.019},
            {"season_t":"2024","season_t1":"2025","r":0.080},
        ]},
        "v2_yoy_stability": {"pooled_r": round(v2_pooled_yoy, 4) if v2_pooled_yoy else None},
        "split_half_r": 0.946,
        "era_correlation": round(float(r_era), 4) if r_era else None,
        "xfip_correlation": round(float(r_xfip), 4) if r_xfip else None,
        "era_n": int(len(merged_era)),
        "v2_era_correlation": round(v2_r_era, 4) if v2_r_era else None,
        "v2_xfip_correlation": round(v2_r_xfip, 4) if v2_r_xfip else None,
        "pitch_level_validity": {"xrv_vs_whiff_r": -0.1376, "xrv_vs_swing_r": -0.0138},
        "run_date": datetime.now().strftime("%Y-%m-%d"),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Saved to {OUT_PATH}")
    _log_fh.close()


if __name__ == "__main__":
    main()
