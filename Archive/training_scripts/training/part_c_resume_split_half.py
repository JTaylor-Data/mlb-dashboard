"""
Resume: compute split-half reliability and save complete v2.1 diagnostic JSON.

All LOSO AUC, ERA, xFIP, YoY results are hardcoded from the log.
Only split-half is recomputed here (full-data scoring — both models trained on
all 2021-2025, scored in-sample; this is intentionally in-sample for this metric
since the LOSO fold that stalled prevents an OOS run in reasonable time).
"""

import json, os, sqlite3, sys, time, warnings
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from analytics.tunneling_features import load_pitches, sequential_lag_features, MAX_LAG
from analytics.location_features import to_plus_scale

MODELS_DIR   = ROOT / "models"
META_PATH    = MODELS_DIR / "tunneling_plus_v2_meta.json"
RV_PATH      = ROOT / "training" / "v2_artifacts" / "run_value_table.json"
COMMIT_PATH  = (ROOT / "project_artifacts" / "training_artifacts"
                / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv")
OUT_PATH     = ROOT / "training" / "v2_artifacts" / "tunneling_plus_v2_1_diagnostic.json"
NEW_MODEL    = MODELS_DIR / "tunneling_plus_v2_1_xfoul_nogeom.txt"

TRAIN_SEASONS = list(range(2021, 2026))
MIN_PITCHES   = 300
HBP_RATE      = 0.0051

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def stringify_keep_na(s):
    na = s.isna()
    return pd.Series(np.where(na, np.nan, s.astype(str)), index=s.index)

def apply_cat_levels(df, col_list, cat_levels):
    for col in col_list:
        if col not in df.columns: continue
        df[col] = stringify_keep_na(df[col])
        lvls = cat_levels.get(col)
        df[col] = pd.Categorical(df[col], categories=lvls) if lvls else pd.Categorical(df[col])
    return df

def load_season_data(conn, season, commit_table, cat_levels, v2_feat_cols):
    df_s = load_pitches(conn, season)
    if len(df_s) == 0: return pd.DataFrame()
    feat = sequential_lag_features(df_s, commit_table, max_lag=MAX_LAG)
    feat = feat.dropna(subset=["lag1_commit_distance","lag1_plate_distance"])
    if len(feat) == 0: return pd.DataFrame()
    feat["season"] = str(season)
    lag_pt = [c for c in v2_feat_cols if c.endswith("_pitch_type")]
    feat   = apply_cat_levels(feat, lag_pt + ["pitch_type","p_throws","stand","season"], cat_levels)
    return feat

def _pred_binary(m, X):
    return m.predict_proba(X)[:,1] if hasattr(m,"predict_proba") else m.predict(X)

def _pred_bip(m, X, bip_classes):
    if hasattr(m, "predict_proba"):
        cls = list(m.classes_)
        p   = m.predict_proba(X)
        out = np.zeros((len(X), len(bip_classes)))
        for i,c in enumerate(bip_classes):
            if c in cls: out[:,i] = p[:,cls.index(c)]
        return out
    return m.predict(X)

def compute_xrv(df_X, v2_models, v2_fcols, xfoul_model, xfoul_fcols, rv, bip_classes):
    X_v2 = df_X[v2_fcols]; X_fo = df_X[xfoul_fcols]
    p_sw  = _pred_binary(v2_models["xswing"], X_v2)
    p_cs  = _pred_binary(v2_models["xcs"],    X_v2)
    p_wh  = _pred_binary(v2_models["xwhiff"], X_v2)
    p_fo  = _pred_binary(xfoul_model, X_fo)
    p_bip = _pred_bip(v2_models["bip"], X_v2, bip_classes)
    bip_rvs = np.array([rv.get(c,0.0) for c in bip_classes])
    xrv_bip = p_bip @ bip_rvs
    xrv_c   = p_fo*rv["foul"] + (1-p_fo)*xrv_bip
    xrv_sw  = p_wh*rv["whiff"] + (1-p_wh)*xrv_c
    p_ball  = np.clip(1-p_cs-HBP_RATE, 0, 1)
    xrv_ns  = p_cs*rv["called_strike"] + HBP_RATE*rv["hbp"] + p_ball*rv["ball"]
    return p_sw*xrv_sw + (1-p_sw)*xrv_ns

def split_half_r(scores_df):
    pairs = []
    for pid, grp in scores_df.groupby("pitcher_id"):
        if len(grp) < 2: continue
        mid = len(grp)//2
        h1  = grp["plus"].iloc[:mid].mean()
        h2  = grp["plus"].iloc[mid:].mean()
        if pd.notna(h1) and pd.notna(h2):
            pairs.append((h1, h2))
    if len(pairs) < 10: return None, len(pairs)
    df_p = pd.DataFrame(pairs, columns=["h1","h2"])
    return float(stats.pearsonr(df_p["h1"], df_p["h2"])[0]), len(df_p)

def aggregate_scores(pitch_df):
    agg = pitch_df.groupby(["pitcher_id","season"])["xrv"].agg(["sum","count"]).reset_index()
    agg.columns = ["pitcher_id","season","xrv_sum","n_pitches"]
    agg = agg[agg["n_pitches"] >= MIN_PITCHES].copy()
    for s, grp in agg.groupby("season"):
        p100 = grp["xrv_sum"]/grp["n_pitches"]*100
        mn, sd = p100.mean(), p100.std()
        agg.loc[grp.index, "plus"] = 100 - 10*(p100-mn)/(sd+1e-9)
    return agg

def main():
    log("Resume: split-half + save final diagnostic JSON")

    meta         = json.loads(META_PATH.read_text())
    v2_feat_cols = meta["feat_cols"]
    bip_classes  = meta["bip_classes"]
    rv_table     = json.loads(RV_PATH.read_text())["rv_table"]
    commit_table = pd.read_csv(COMMIT_PATH)
    conn         = sqlite3.connect("mlb_data.db")

    raw = pd.read_sql(
        "SELECT DISTINCT pitch_type,p_throws,stand,season FROM pitches WHERE season BETWEEN 2015 AND 2026", conn)
    raw["pitch_type"] = raw["pitch_type"].replace({"KC":"CU"})
    raw = raw[~raw["pitch_type"].isin({"PO","UN","EP"})]
    raw["season"] = raw["season"].astype(str)
    pt_lvls = sorted(raw["pitch_type"].dropna().unique())
    lag_pt_cols = [c for c in v2_feat_cols if c.endswith("_pitch_type")]
    cat_levels  = {
        "pitch_type": pt_lvls, "p_throws": sorted(raw["p_throws"].dropna().unique()),
        "stand": sorted(raw["stand"].dropna().unique()), "season": sorted(raw["season"].dropna().unique()),
    }
    for col in lag_pt_cols: cat_levels[col] = pt_lvls

    geom_cols  = [f"lag{n}_{m}" for n in range(1,MAX_LAG+1) for m in ("commit_distance","plate_distance","shape_break")]
    xfo_fcols  = [c for c in v2_feat_cols if c not in set(geom_cols)]

    v2_models = {n: lgb.Booster(model_file=str(MODELS_DIR/f"tunneling_plus_v2_{n}.txt"))
                 for n in ("xswing","xcs","xwhiff","xfoul","bip")}
    m_xfoul_new = lgb.LGBMClassifier(verbose=-1)
    m_xfoul_new._Booster = lgb.Booster(model_file=str(NEW_MODEL))
    # Load as Booster directly for scoring
    booster_new = lgb.Booster(model_file=str(NEW_MODEL))

    log("Loading 2021-2025 data for full-corpus split-half scoring...")
    chunks = []
    for s in TRAIN_SEASONS:
        log(f"  Season {s}...")
        feat = load_season_data(conn, s, commit_table, cat_levels, v2_feat_cols)
        if len(feat): chunks.append(feat)
    df_all = pd.concat(chunks, ignore_index=True)
    log(f"  Total: {len(df_all):,} rows")

    log("Scoring v2.1 (full-data, in-sample)...")
    xrv_v21 = compute_xrv(df_all, v2_models, v2_feat_cols, booster_new, xfo_fcols, rv_table, bip_classes)
    log("Scoring v2 baseline (full-data, in-sample)...")
    xrv_v2  = compute_xrv(df_all, v2_models, v2_feat_cols, v2_models["xfoul"], v2_feat_cols, rv_table, bip_classes)

    pitch_df = df_all[["pitcher_id","season"]].copy()
    scores_v21 = aggregate_scores(pitch_df.assign(xrv=xrv_v21))
    scores_v2  = aggregate_scores(pitch_df.assign(xrv=xrv_v2))
    log(f"  v2.1 pitcher-seasons: {len(scores_v21):,}   v2: {len(scores_v2):,}")

    sh_v21, n_sh_v21 = split_half_r(scores_v21)
    sh_v2,  n_sh_v2  = split_half_r(scores_v2)
    log(f"  Split-half r: v2.1={sh_v21:.3f}  v2={sh_v2:.3f}  (n={n_sh_v21})")

    # ── Assemble and save final result ─────────────────────────────────────
    results = {
        "general_rule": (
            "Drop lagN_{commit_distance,plate_distance,shape_break} from any classifier "
            "where those features show a statistically significant WRONG-SIGNED effect on "
            "that classifier's own real target outcome. Keep geometry everywhere null or correct. "
            "xFoul% is the only current trigger (Part A2: lag1-geom z=-21.8, alllag z=-44.6)."
        ),
        "part_a": {
            "model_path": str(NEW_MODEL),
            "feat_cols": xfo_fcols,
            "geom_cols_removed": geom_cols,
            "train_seasons": TRAIN_SEASONS,
            "n_geom_removed": 18, "n_feat_total": len(xfo_fcols),
        },
        "part_b": {
            "clean_split_description": (
                "Complement of Part A2 seed=42 200K sample from 2021-2025 (zero overlap guaranteed). "
                "100K draw with seed=99 from the ~2.4M-row complement."
            ),
            "v2_xfoul": {
                "auc_clean_split": 0.4982,
                "calibration_z": -0.396,
                "lag1_geom_shap_z": -16.194,
                "alllag_geom_shap_z": -31.371,
                "note": "Geometry SHAP wrong-signed on clean held-out split — replicates Part A2 finding.",
            },
            "new_xfoul": {
                "auc_clean_split": 0.5863,
                "calibration_z": 34.439,
                "pt_lag_shap_z": 21.011,
                "note": "No geometry features → calibration now correct direction; pt-lag SHAP also positive.",
            },
            "auc_delta_new_minus_v2": 0.0881,
        },
        "part_c": {
            "loso_xfoul_auc": {
                "per_fold_new": [0.5823, 0.5828, 0.5817, 0.5832, 0.5830],
                "per_fold_v2":  [0.4977, 0.4989, 0.4997, 0.4998, 0.4994],
                "mean_new": 0.5826,
                "mean_v2":  0.4991,
                "mean_delta": 0.0835,
                "note": (
                    "v2 xFoul% AUC used here is from live full-data model (slightly in-sample), "
                    "v2.1 is properly out-of-sample LOSO. Comparison is conservative in favor of v2."
                ),
            },
            "era_r":  {"v2": -0.214, "v21": -0.163},
            "xfip_r": {"v2": -0.329, "v21": -0.271},
            "yoy": {
                "mean_pooled_v2":  0.692,
                "mean_pooled_v21": 0.756,
                "per_pair": [
                    {"pair":"2021-2022","v2":0.712,"v21":0.776},
                    {"pair":"2022-2023","v2":0.732,"v21":0.791},
                    {"pair":"2023-2024","v2":0.675,"v21":0.733},
                    {"pair":"2024-2025","v2":0.649,"v21":0.725},
                ],
            },
            "split_half": {
                "v2":  round(sh_v2,  4) if sh_v2  else None,
                "v21": round(sh_v21, 4) if sh_v21 else None,
                "n": int(n_sh_v21),
                "note": "Split-half computed on in-sample full-corpus scores (not LOSO).",
            },
            "pitcher_season_n": 2608,
        },
        "summary": {
            "xfoul_auc_clean_split":    {"v2": 0.4982, "v21": 0.5863, "delta": +0.0881},
            "xfoul_auc_loso_mean":      {"v2": 0.4991, "v21": 0.5826, "delta": +0.0835},
            "era_r":                    {"v2": -0.214,  "v21": -0.163},
            "xfip_r":                   {"v2": -0.329,  "v21": -0.271},
            "yoy_pooled":               {"v2": 0.692,   "v21": 0.756},
            "split_half_in_sample":     {"v2": round(sh_v2,4) if sh_v2 else None, "v21": round(sh_v21,4) if sh_v21 else None},
            "v2_geom_shap_clean_split": {"lag1_z": -16.194, "alllag_z": -31.371},
            "constraint": "No production changes. Tunneling+ v2 stays live. Not wired into api/.",
        },
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    log(f"Saved: {OUT_PATH}")
    log("Done.")

if __name__ == "__main__":
    main()
