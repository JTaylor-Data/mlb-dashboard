#!/usr/bin/env python3
"""
Cross-model validation v2.

Part A: Part 2 redo — fold-level AUC CIs, 5 contrasts with CIs,
        geometry regression with geometry/pitch_type/p_throws/stand and pitch_type-alone.
Part B: Part 3 redo — pitcher-clustered bootstrap CIs on all metrics,
        CI on V1-V0 and V-rand-V0 YoY r differences.
Part C: Deception+ v1.1 candidate (no height/weight) — full validation battery.

Constraints: No production changes. New artifacts only. Live models untouched.
New model: training/v2_artifacts/deception_plus_v1_1_no_bodysize.pkl
"""
import gc
import json
import logging
import pickle
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.location_features import to_plus_scale
from analytics.tunneling_v2_model import (
    NUMERIC_LAG_COLS,
    ALL_CATEGORICAL,
    ALL_FEATURES      as T_FEATURES,
    _cat_levels       as tun_cat_levels,
    _load_commit_table,
    load_pitches      as tun_load_pitches,
    MAX_LAG           as TUN_MAX_LAG,
)
from analytics.deception_features import (
    ALL_FEATURES         as DEC_FEATURES,
    RAW_QUERY_COLUMNS    as DEC_QUERY_COLS,
    engineer_features    as dec_eng,
    load_bio             as dec_load_bio,
)
from analytics.tunneling_features import sequential_lag_features

DB_PATH       = ROOT / "mlb_data.db"
V2_DIR        = ROOT / "training" / "v2_artifacts"
CP_PATH       = V2_DIR / "cross_model_validation_v2_checkpoint.json"
LOG_PATH      = ROOT / "training" / "cross_model_validation_v2.log"
HTML_OUTPUT   = ROOT / "Methodology" / "in_progress" / "CROSS_MODEL_VALIDATION_AND_LEAKAGE_TESTS_IN_PROGRESS.html"
MODEL_V11_PATH = V2_DIR / "deception_plus_v1_1_no_bodysize.pkl"
V1_CP_PATH    = V2_DIR / "cross_model_validation_checkpoint.json"

SEASONS   = [2021, 2022, 2023, 2024, 2025]
YOY_PAIRS = [(2021,2022),(2022,2023),(2023,2024),(2024,2025)]

THRESHOLDS = {
    "auc_leakage_delta": 0.010,
    "geom_to_loc_r2":    0.050,
    "ablation_yoy_delta":0.050,
}

SWING_DESCS = {
    "swinging_strike","swinging_strike_blocked","foul_tip","bunt_foul_tip",
    "missed_bunt","foul","foul_bunt","foul_pitchout",
    "hit_into_play","hit_into_play_no_out","hit_into_play_score","swinging_pitchout",
}
WHIFF_DESCS = {"swinging_strike","swinging_strike_blocked","foul_tip",
               "bunt_foul_tip","missed_bunt","swinging_pitchout"}
FOUL_DESCS  = {"foul","foul_bunt","foul_pitchout"}
CS_DESCS    = {"called_strike","automatic_strike"}

S_FEATURES = [c for c in ALL_CATEGORICAL if "lag" in c and "pitch_type" in c]
L_NUM = ["plate_x","plate_z","balls","strikes"]
L_CAT = ["pitch_type","stand","season"]

FEATURE_SETS = {
    "L":     {"num": L_NUM,                        "cat": L_CAT},
    "T":     {"num": NUMERIC_LAG_COLS,             "cat": list(ALL_CATEGORICAL)},
    "L+G":   {"num": L_NUM + NUMERIC_LAG_COLS,     "cat": list(dict.fromkeys(L_CAT + list(ALL_CATEGORICAL)))},
    "L+S":   {"num": L_NUM,                        "cat": list(dict.fromkeys(L_CAT + S_FEATURES + ["p_throws"]))},
    "L+G+S": {"num": L_NUM + NUMERIC_LAG_COLS,     "cat": list(dict.fromkeys(L_CAT + list(ALL_CATEGORICAL)))},
    "T-G":   {"num": [],                           "cat": list(ALL_CATEGORICAL)},
}

DEC_V11_FEATURES = [f for f in DEC_FEATURES if f not in ("height_in","weight_lb")]


# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

def log(msg):
    logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Checkpointing ──────────────────────────────────────────────────────────
def load_checkpoint():
    if CP_PATH.exists():
        return json.loads(CP_PATH.read_text())
    return {}

def save_checkpoint(cp):
    V2_DIR.mkdir(parents=True, exist_ok=True)
    CP_PATH.write_text(json.dumps(cp, indent=2, default=str))
    log(f"  Checkpoint saved → {CP_PATH.name}")


# ── Statistics ─────────────────────────────────────────────────────────────
def pearson_with_ci(x, y, alpha=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan"), float("nan"), n
    r, _ = stats.pearsonr(x, y)
    z  = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    ci = stats.norm.ppf(1 - alpha / 2)
    return float(r), float(np.tanh(z - ci*se)), float(np.tanh(z + ci*se)), n


def t_ci(vals, alpha=0.05):
    """t-distribution CI on a list of samples (for fold-level AUC CIs)."""
    a = np.asarray([v for v in vals if not np.isnan(v)], float)
    n = len(a)
    if n < 2:
        return float(np.mean(a)) if n else float("nan"), float("nan"), float("nan")
    m  = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n))
    t  = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return m, m - t * se, m + t * se


def contrast_t_ci(vals_a, vals_b, alpha=0.05):
    """Paired t-CI on mean(a - b)."""
    diffs = np.asarray(vals_a, float) - np.asarray(vals_b, float)
    return t_ci(diffs, alpha)


# ── Data loading ──────────────────────────────────────────────────────────
def add_targets(df):
    desc = df["description"]
    df["is_swing"]         = desc.isin(SWING_DESCS).astype(int)
    df["is_whiff"]         = desc.isin(WHIFF_DESCS).astype(int)
    df["is_foul"]          = desc.isin(FOUL_DESCS).astype(int)
    df["is_called_strike"] = desc.isin(CS_DESCS).astype(int)


def load_tunneling_with_location(conn, season):
    df = tun_load_pitches(conn, season)
    bs = pd.read_sql(
        f"SELECT game_id, pitcher_id, at_bat_number, pitch_number, balls, strikes, p_throws "
        f"FROM pitches WHERE season={season} AND game_type='R'", conn
    )
    df = df.merge(bs, on=["game_id","pitcher_id","at_bat_number","pitch_number"], how="left")
    commit = _load_commit_table()
    feat = sequential_lag_features(df, commit, max_lag=TUN_MAX_LAG)
    add_targets(feat)
    return feat


# ═══════════════════════════════════════════════════════════════════════════
# PART A — Part 2 redo with fold-level AUC CIs
# ═══════════════════════════════════════════════════════════════════════════

def run_part_a(conn, cp):
    if cp.get("part_a_done"):
        log("Part A: already complete — loading checkpoint.")
        return cp["part_a_results"]

    log("=" * 60)
    log("PART A: Tunneling+ location-leakage test (fold-level AUC + CIs)")
    log(f"  Threshold ΔAUC > {THRESHOLDS['auc_leakage_delta']:.3f} = non-trivial location contribution")
    log(f"  Threshold geom→loc R² > {THRESHOLDS['geom_to_loc_r2']:.2f} = direct encoding")
    log("=" * 60)

    classifiers = [
        ("xswing", "is_swing",        "all"),
        ("xcs",    "is_called_strike", "no_swing"),
        ("xwhiff", "is_whiff",         "swing"),
        ("xfoul",  "is_foul",          "contact"),
    ]

    # Cache seasonal data once — avoids 30 redundant loads
    log("\n  Loading all 5 seasons into cache (one-time)...")
    season_cache = {}
    for s in SEASONS:
        t0 = time.time()
        season_cache[s] = load_tunneling_with_location(conn, s)
        log(f"    {s}: {len(season_cache[s]):,} rows  ({time.time()-t0:.0f}s)")

    # Per-feature-set LOSO with fold-level AUC storage
    fold_auc_store = {}  # {set_name: {clf_name: [fold_aucs]}}

    for set_name, fset in FEATURE_SETS.items():
        ck_key = f"part_a_{set_name}"
        if cp.get(ck_key):
            log(f"\n  {set_name}: already done — loading checkpoint.")
            fold_auc_store[set_name] = cp[f"part_a_fold_aucs_{set_name}"]
            continue

        all_num = fset["num"]
        all_cat = fset["cat"]
        log(f"\n  Feature set: {set_name}  ({len(all_num)} numeric, {len(all_cat)} categorical)")

        fold_aucs = {clf: [] for clf, _, _ in classifiers}

        for test_season in SEASONS:
            train_seasons = [s for s in SEASONS if s != test_season]
            log(f"    Fold: test={test_season} ...")
            train_parts = [season_cache[s] for s in train_seasons]
            train_feat  = pd.concat(train_parts, ignore_index=True)
            test_feat   = season_cache[test_season]

            for clf_name, tgt, row_filter in classifiers:
                if row_filter == "swing":
                    tr_m = train_feat["is_swing"] == 1
                    te_m = test_feat["is_swing"]  == 1
                elif row_filter == "no_swing":
                    tr_m = train_feat["is_swing"] == 0
                    te_m = test_feat["is_swing"]  == 0
                elif row_filter == "contact":
                    tr_m = (train_feat["is_swing"]==1) & (train_feat["is_whiff"]==0)
                    te_m = (test_feat["is_swing"] ==1) & (test_feat["is_whiff"] ==0)
                else:
                    tr_m = slice(None)
                    te_m = slice(None)

                tr = train_feat.iloc[tr_m] if isinstance(tr_m, slice) else train_feat[tr_m]
                te = test_feat.iloc[te_m]  if isinstance(te_m, slice) else test_feat[te_m]

                avail    = [c for c in all_num + all_cat if c in tr.columns]
                cats_av  = [c for c in all_cat if c in avail]

                X_tr = tr[avail].copy()
                y_tr = tr[tgt].values
                X_te = te[avail].copy()
                y_te = te[tgt].values

                for c in cats_av:
                    X_tr[c] = X_tr[c].astype(str).astype("category")
                    X_te[c] = X_te[c].astype(str).astype("category")

                mdl = lgb.LGBMClassifier(
                    n_estimators=300, num_leaves=31, learning_rate=0.05,
                    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                    verbose=-1, n_jobs=-1,
                )
                mdl.fit(X_tr, y_tr, categorical_feature=cats_av)
                pred = mdl.predict_proba(X_te)[:, 1]
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_te, pred))
                fold_aucs[clf_name].append(auc)
                log(f"      {clf_name}: AUC={auc:.4f}")

            del train_feat

        fold_auc_store[set_name] = {k: v for k, v in fold_aucs.items()}
        cp[ck_key] = True
        cp[f"part_a_fold_aucs_{set_name}"] = fold_auc_store[set_name]
        save_checkpoint(cp)

    # ── Compute mean AUCs with fold-level t-CIs ──────────────────────────
    clf_names = [c for c, _, _ in classifiers]
    auc_means = {}
    auc_cis   = {}
    for fset_name, fold_data in fold_auc_store.items():
        auc_means[fset_name] = {}
        auc_cis[fset_name]   = {}
        for clf in clf_names:
            vals = fold_data.get(clf, [])
            m, lo, hi = t_ci(vals)
            auc_means[fset_name][clf] = m
            auc_cis[fset_name][clf]   = [lo, hi]

    # ── Contrasts (5 specified) ────────────────────────────────────────────
    contrasts = {}
    for clf in clf_names:
        def fold_vals(fset_name):
            return fold_auc_store.get(fset_name, {}).get(clf, [float("nan")]*5)

        v_L    = fold_vals("L")
        v_T    = fold_vals("T")
        v_LG   = fold_vals("L+G")
        v_LS   = fold_vals("L+S")
        v_LGS  = fold_vals("L+G+S")

        c_LG_L,   lo1, hi1 = contrast_t_ci(v_LG,  v_L)
        c_LS_L,   lo2, hi2 = contrast_t_ci(v_LS,  v_L)
        c_LGS_LS, lo3, hi3 = contrast_t_ci(v_LGS, v_LS)
        c_T_L,    lo4, hi4 = contrast_t_ci(v_T,   v_L)

        contrasts[clf] = {
            "LG_minus_L":   {"d": c_LG_L,   "lo": lo1, "hi": hi1},
            "LS_minus_L":   {"d": c_LS_L,   "lo": lo2, "hi": hi2},
            "LGS_minus_LS": {"d": c_LGS_LS, "lo": lo3, "hi": hi3},
            "T_minus_L":    {"d": c_T_L,    "lo": lo4, "hi": hi4},
        }
        log(f"  {clf} contrasts: (L+G)−L={c_LG_L:+.4f}[{lo1:+.4f},{hi1:+.4f}]"
            f"  T−L={c_T_L:+.4f}[{lo4:+.4f},{hi4:+.4f}]")

    # ── Interpretation ────────────────────────────────────────────────────
    # Pre-registered rule: for xSwing and xCS,
    #   LG_minus_L < 0.005 with CI including 0 → leakage verdict
    #   LG_minus_L >= 0.02 with CI clearly > 0 → real geometry information
    #   otherwise → ambiguous
    lg_swing = contrasts["xswing"]["LG_minus_L"]
    lg_cs    = contrasts["xcs"]["LG_minus_L"]

    def interpret_lg(c):
        d, lo, hi = c["d"], c["lo"], c["hi"]
        if not (isinstance(d, float) and not np.isnan(d)):
            return "ambiguous"
        if d < 0.005 and lo <= 0 <= hi:
            return "leakage"
        elif d >= 0.02 and lo > 0:
            return "real_geometry_info"
        else:
            return "ambiguous"

    verdict_swing = interpret_lg(lg_swing)
    verdict_cs    = interpret_lg(lg_cs)
    if verdict_swing == "leakage" and verdict_cs == "leakage":
        leakage_verdict = "leakage"
    elif verdict_swing == "real_geometry_info" and verdict_cs == "real_geometry_info":
        leakage_verdict = "real_geometry_info"
    else:
        leakage_verdict = "ambiguous"
    log(f"\n  Leakage verdict: {leakage_verdict} "
        f"(xSwing={verdict_swing}, xCS={verdict_cs})")

    # ── Geometry → plate position Ridge regression ────────────────────────
    log("\n  Geometry → plate position regression (Ridge, 2024):")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder
    import warnings

    sample = season_cache[2024].copy()
    geom_r2 = {}

    for target in ["plate_x", "plate_z"]:
        cols_geom     = NUMERIC_LAG_COLS
        cat_cols_reg  = ["pitch_type", "p_throws", "stand"]
        all_needed = cols_geom + cat_cols_reg + [target]
        available_cat = [c for c in cat_cols_reg if c in sample.columns]

        sub = sample.dropna(subset=[c for c in all_needed if c in sample.columns])
        if len(sub) < 100:
            geom_r2[target] = {"geom_only": float("nan"), "geom_plus_cat": float("nan"), "pt_only": float("nan")}
            continue

        y = sub[target].values

        # Model 1: geometry only
        X_geom = sub[cols_geom].values
        r2_geom = Ridge(alpha=1.0).fit(X_geom, y).score(X_geom, y) if X_geom.shape[1] > 0 else float("nan")

        # Model 2: pitch_type alone (one-hot)
        r2_pt = float("nan")
        if "pitch_type" in sub.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                X_pt = enc.fit_transform(sub[["pitch_type"]])
            r2_pt = Ridge(alpha=1.0).fit(X_pt, y).score(X_pt, y)

        # Model 3: geometry + pitch_type + p_throws + stand
        r2_geom_cat = float("nan")
        if available_cat:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enc2 = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                X_cat = enc2.fit_transform(sub[available_cat])
            X_full = np.hstack([X_geom, X_cat])
            r2_geom_cat = Ridge(alpha=1.0).fit(X_full, y).score(X_full, y)

        geom_r2[target] = {
            "geom_only":    float(r2_geom),
            "geom_plus_cat": float(r2_geom_cat),
            "pt_only":      float(r2_pt),
        }
        log(f"    {target}: geom={r2_geom:.4f}  geom+cat={r2_geom_cat:.4f}  pt_only={r2_pt:.4f}")

    results = {
        "auc_means":       auc_means,
        "auc_cis":         auc_cis,
        "fold_aucs":       fold_auc_store,
        "contrasts":       contrasts,
        "geom_r2":         geom_r2,
        "leakage_verdict": leakage_verdict,
    }
    cp["part_a_done"]    = True
    cp["part_a_results"] = results
    save_checkpoint(cp)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PART B — Part 3 redo with bootstrapped CIs
# ═══════════════════════════════════════════════════════════════════════════

def run_part_b(conn, cp):
    if cp.get("part_b_done"):
        log("Part B: already complete — loading checkpoint.")
        return cp["part_b_results"]

    log("=" * 60)
    log("PART B: Deception+ identity-proxy ablation with bootstrap CIs")
    log("  Bootstrap: pitcher-clustered (resample pitcher IDs), n=1000")
    log(f"  Threshold |ΔYoY r| > {THRESHOLDS['ablation_yoy_delta']:.3f} = meaningful memorization")
    log("=" * 60)

    bio = dec_load_bio(conn)

    log("\n  Loading 2021-2025 deception features...")
    raw_by_season = {}
    for s in SEASONS:
        df   = pd.read_sql(
            f"SELECT {', '.join(DEC_QUERY_COLS)} FROM pitches "
            f"WHERE season={s} AND game_type='R' AND delta_run_exp IS NOT NULL", conn
        )
        feat = dec_eng(df, bio)
        feat["season"] = feat["season"].astype(str).astype(int)
        raw_by_season[s] = feat
        log(f"    {s}: {len(feat):,} rows")

    outcomes = pd.read_sql(
        "SELECT player_id AS pitcher_id, season, era, xfip FROM player_advanced_stats "
        "WHERE era IS NOT NULL AND xfip IS NOT NULL AND season BETWEEN 2021 AND 2025", conn
    )

    def run_variant_loso(vname, feature_list, randomize_cols=None, seed=None):
        ck_key = f"part_b_oos_{vname}"
        if cp.get(ck_key):
            log(f"  Variant {vname}: LOSO already done — loading.")
            return pd.DataFrame(cp[f"part_b_oos_data_{vname}"])

        log(f"\n  Variant: {vname}  features={len(feature_list)}")
        rng = np.random.default_rng(seed) if seed is not None else None

        try:
            oos_parts = []
            for test_season in SEASONS:
                train_seasons = [s for s in SEASONS if s != test_season]
                train_all     = pd.concat([raw_by_season[s] for s in train_seasons], ignore_index=True)
                test_data     = raw_by_season[test_season]

                if randomize_cols and rng is not None:
                    train_all = train_all.copy()
                    for col in randomize_cols:
                        if col in train_all.columns:
                            train_all[col] = train_all[col].values[rng.permutation(len(train_all))]

                avail = [f for f in feature_list if f in train_all.columns]
                mdl = lgb.LGBMRegressor(
                    objective="huber", n_estimators=500, num_leaves=31, learning_rate=0.05,
                    min_child_samples=50, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1,
                )
                mdl.fit(train_all[avail], train_all["delta_run_exp"])
                del train_all

                if randomize_cols and rng is not None:
                    extra = [c for c in ["pitcher_id","season","delta_run_exp"] if c not in avail]
                    test_copy = test_data[avail + extra].copy()
                    for col in randomize_cols:
                        if col in test_copy.columns:
                            test_copy[col] = test_copy[col].values[rng.permutation(len(test_copy))]
                    preds  = mdl.predict(test_copy[avail])
                    fold_df = pd.DataFrame({
                        "pitcher_id": test_data["pitcher_id"].values,
                        "season":     test_data["season"].values,
                        "pred":       preds,
                        "actual":     test_data["delta_run_exp"].values,
                    })
                else:
                    preds  = mdl.predict(test_data[avail])
                    fold_df = pd.DataFrame({
                        "pitcher_id": test_data["pitcher_id"].values,
                        "season":     test_data["season"].values,
                        "pred":       preds,
                        "actual":     test_data["delta_run_exp"].values,
                    })

                oos_parts.append(fold_df)
                log(f"    Fold {test_season}: done")

        except Exception as e:
            import traceback
            log(f"  FAILED {vname}: {e}\n{traceback.format_exc()}")
            return pd.DataFrame()

        oos_df = pd.concat(oos_parts, ignore_index=True)
        cp[ck_key] = True
        cp[f"part_b_oos_data_{vname}"] = oos_df.to_dict(orient="list")
        save_checkpoint(cp)
        return oos_df

    v1_feat  = [f for f in DEC_FEATURES if f not in ("height_in","weight_lb")]
    oos_data = {
        "V0":       run_variant_loso("V0",       DEC_FEATURES),
        "V1":       run_variant_loso("V1",       v1_feat),
        "V-rand-0": run_variant_loso("V-rand-0", DEC_FEATURES, ["height_in","weight_lb"], 0),
        "V-rand-1": run_variant_loso("V-rand-1", DEC_FEATURES, ["height_in","weight_lb"], 1),
        "V-rand-2": run_variant_loso("V-rand-2", DEC_FEATURES, ["height_in","weight_lb"], 2),
    }

    # ── Bootstrap CIs ─────────────────────────────────────────────────────
    def compute_metrics_from_ps(ps, outcomes_df):
        ps_scaled = []
        for s in SEASONS:
            sub = ps[ps.season == s].copy()
            if len(sub) < 2:
                continue
            m, std = sub["mean_pred"].mean(), sub["mean_pred"].std()
            sub["plus"] = to_plus_scale(sub["mean_pred"], m, std) if std > 0 else 100.0
            ps_scaled.append(sub[["pitcher_id","season","plus"]])
        if not ps_scaled:
            return float("nan"), float("nan"), float("nan")
        ps_plus = pd.concat(ps_scaled, ignore_index=True)

        yoy_rows = []
        for s1, s2 in YOY_PAIRS:
            a = ps_plus[ps_plus.season==s1][["pitcher_id","plus"]].rename(columns={"plus":"p1"})
            b = ps_plus[ps_plus.season==s2][["pitcher_id","plus"]].rename(columns={"plus":"p2"})
            yoy_rows.append(a.merge(b, on="pitcher_id"))
        if not yoy_rows:
            return float("nan"), float("nan"), float("nan")
        yoy_df = pd.concat(yoy_rows, ignore_index=True)
        if len(yoy_df) < 5:
            return float("nan"), float("nan"), float("nan")
        yoy_r = float(np.corrcoef(yoy_df["p1"], yoy_df["p2"])[0,1])

        merged = ps_plus.merge(outcomes_df, on=["pitcher_id","season"])
        if len(merged) < 10:
            return yoy_r, float("nan"), float("nan")
        era_r  = float(np.corrcoef(merged["plus"], merged["era"])[0,1])
        xfip_r = float(np.corrcoef(merged["plus"], merged["xfip"])[0,1])
        return yoy_r, era_r, xfip_r

    def bootstrap_variant(oos_df, n_boot=1000, seed=42):
        if oos_df.empty:
            return {}
        rng  = np.random.default_rng(seed)
        ps   = oos_df.groupby(["pitcher_id","season"])["pred"].mean().reset_index()
        ps.columns = ["pitcher_id","season","mean_pred"]

        from sklearn.metrics import r2_score
        cv_r2_pt = float(r2_score(oos_df["actual"], oos_df["pred"]))

        yoy_r, era_r, xfip_r = compute_metrics_from_ps(ps, outcomes)

        all_pids = ps["pitcher_id"].unique()
        pid_grps = {pid: sub for pid, sub in ps.groupby("pitcher_id")}

        b_yoy, b_era, b_xfip = [], [], []
        for _ in range(n_boot):
            boot_pids = rng.choice(all_pids, size=len(all_pids), replace=True)
            boot_ps   = pd.concat([pid_grps[p] for p in boot_pids], ignore_index=True)
            yr, er, xr = compute_metrics_from_ps(boot_ps, outcomes)
            b_yoy.append(yr); b_era.append(er); b_xfip.append(xr)

        def pci(v):
            a = np.array([x for x in v if not np.isnan(x)])
            if len(a) < 10:
                return [float("nan"), float("nan")]
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

        return {
            "cv_r2":    cv_r2_pt,
            "yoy_r":    yoy_r,   "yoy_ci":  pci(b_yoy),
            "era_r":    era_r,   "era_ci":  pci(b_era),
            "xfip_r":   xfip_r,  "xfip_ci": pci(b_xfip),
            "_boot_yoy": b_yoy, "_boot_era": b_era,
        }

    log("\n  Running bootstrap CIs (1000 iters, pitcher-clustered)...")
    variant_metrics = {}
    for vname, oos_df in oos_data.items():
        ck_key = f"part_b_metrics_{vname}"
        if cp.get(ck_key):
            log(f"    {vname}: bootstrap already done.")
            variant_metrics[vname] = cp[f"part_b_metrics_data_{vname}"]
            continue
        log(f"    {vname}...")
        vm = bootstrap_variant(oos_df)
        log(f"      YoY r={vm.get('yoy_r',float('nan')):.3f} [{vm['yoy_ci'][0]:.3f},{vm['yoy_ci'][1]:.3f}]"
            f"  ERA r={vm.get('era_r',float('nan')):.3f}")
        # Don't store raw bootstrap arrays in checkpoint (too large)
        vm_store = {k: v for k, v in vm.items() if not k.startswith("_")}
        variant_metrics[vname] = vm_store
        cp[ck_key] = True
        cp[f"part_b_metrics_data_{vname}"] = vm_store
        save_checkpoint(cp)
        # Keep boot arrays in memory for difference CIs
        variant_metrics[vname]["_boot_yoy"] = vm.get("_boot_yoy", [])
        variant_metrics[vname]["_boot_era"] = vm.get("_boot_era", [])

    # ── Difference CIs ────────────────────────────────────────────────────
    # Re-run bootstrap jointly so same pitcher resamples are used for differences
    log("\n  Computing joint bootstrap for difference CIs (V1-V0, V-rand-V0)...")

    def joint_bootstrap_diff(oos_v0, oos_v1_list, n_boot=1000, seed=42):
        """Same pitcher resamples for V0 and each V1/V-rand."""
        if oos_v0.empty:
            return {}
        rng = np.random.default_rng(seed)
        ps_v0 = oos_v0.groupby(["pitcher_id","season"])["pred"].mean().reset_index()
        ps_v0.columns = ["pitcher_id","season","mean_pred"]
        all_pids = ps_v0["pitcher_id"].unique()

        other_ps = {}
        for name, oos_df in oos_v1_list:
            ps = oos_df.groupby(["pitcher_id","season"])["pred"].mean().reset_index()
            ps.columns = ["pitcher_id","season","mean_pred"]
            other_ps[name] = ps

        grps_v0 = {pid: sub for pid, sub in ps_v0.groupby("pitcher_id")}
        other_grps = {}
        for name, ps in other_ps.items():
            other_grps[name] = {pid: sub for pid, sub in ps.groupby("pitcher_id")}

        diff_yoy  = {name: [] for name in other_ps}
        diff_era  = {name: [] for name in other_ps}

        for _ in range(n_boot):
            boot_pids = rng.choice(all_pids, size=len(all_pids), replace=True)
            boot_v0   = pd.concat([grps_v0[p] for p in boot_pids if p in grps_v0], ignore_index=True)
            yr0, er0, _ = compute_metrics_from_ps(boot_v0, outcomes)

            for name in other_ps:
                grps_other = other_grps[name]
                boot_other = pd.concat([grps_other[p] for p in boot_pids if p in grps_other], ignore_index=True)
                yr1, er1, _ = compute_metrics_from_ps(boot_other, outcomes)
                diff_yoy[name].append(yr1 - yr0)
                diff_era[name].append(er1 - er0)

        def pci(v):
            a = np.array([x for x in v if not np.isnan(x)])
            if len(a) < 10:
                return [float("nan"), float("nan")]
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

        return {
            name: {
                "diff_yoy_r": float(np.nanmean(diff_yoy[name])),
                "diff_yoy_ci": pci(diff_yoy[name]),
                "diff_era_r":  float(np.nanmean(diff_era[name])),
                "diff_era_ci": pci(diff_era[name]),
                "yoy_excludes_zero": (pci(diff_yoy[name])[1] < 0 or pci(diff_yoy[name])[0] > 0),
            }
            for name in other_ps
        }

    diff_ck = "part_b_diffs_done"
    if cp.get(diff_ck):
        log("  Difference CIs already done — loading.")
        diff_results = cp["part_b_diff_results"]
    else:
        v_rand_pooled_oos = []
        for name in ["V-rand-0","V-rand-1","V-rand-2"]:
            if not oos_data[name].empty:
                v_rand_pooled_oos.append(oos_data[name])
        # Pool V-rand predictions: average across seeds per pitcher-pitch
        if v_rand_pooled_oos:
            vrand_pool = pd.concat(v_rand_pooled_oos, ignore_index=True)
            vrand_pool = vrand_pool.groupby(["pitcher_id","season"])["pred"].mean().reset_index()
            # Expand back to pitch level with pooled pred for convenience: just use first seed's structure
            vrand_oos_combined = oos_data["V-rand-0"].copy()
            # Map pitcher-season mean from pooled
            vrand_oos_combined = vrand_oos_combined.merge(
                vrand_pool.rename(columns={"pred":"pred_pool"}),
                on=["pitcher_id","season"], how="left"
            )
            vrand_oos_combined["pred"] = vrand_oos_combined["pred_pool"]
            vrand_oos_combined = vrand_oos_combined.drop(columns=["pred_pool"])
        else:
            vrand_oos_combined = pd.DataFrame()

        compare_list = [("V1", oos_data["V1"]), ("V-rand", vrand_oos_combined)]
        diff_results = joint_bootstrap_diff(oos_data["V0"], compare_list)

        for name, d in diff_results.items():
            log(f"    {name} − V0: ΔYoY r={d['diff_yoy_r']:+.3f} "
                f"[{d['diff_yoy_ci'][0]:+.3f},{d['diff_yoy_ci'][1]:+.3f}]"
                f"  excludes_0={d['yoy_excludes_zero']}")
        cp[diff_ck] = True
        cp["part_b_diff_results"] = diff_results
        save_checkpoint(cp)

    results = {
        "variants":    {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in variant_metrics.items()},
        "differences": diff_results,
    }
    cp["part_b_done"]    = True
    cp["part_b_results"] = results
    save_checkpoint(cp)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PART C — Deception+ v1.1 candidate (no height/weight)
# ═══════════════════════════════════════════════════════════════════════════

def run_part_c(conn, cp):
    if cp.get("part_c_done"):
        log("Part C: already complete — loading checkpoint.")
        return cp["part_c_results"]

    log("=" * 60)
    log("PART C: Deception+ v1.1 candidate — no height/weight")
    log(f"  Features: {DEC_V11_FEATURES}")
    log("=" * 60)

    bio = dec_load_bio(conn)
    MIN_P = 300

    # ── Train v1.1 on all 2021-2025 ───────────────────────────────────────
    if not cp.get("part_c_model_trained"):
        log("\n  Training Deception+ v1.1 on 2021-2025...")
        parts = []
        for s in SEASONS:
            df   = pd.read_sql(
                f"SELECT {', '.join(DEC_QUERY_COLS)} FROM pitches "
                f"WHERE season={s} AND game_type='R' AND delta_run_exp IS NOT NULL", conn
            )
            feat = dec_eng(df, bio)
            feat["season"] = feat["season"].astype(str).astype(int)
            parts.append(feat)
        all_data = pd.concat(parts, ignore_index=True)
        del parts

        model_v11 = lgb.LGBMRegressor(
            objective="huber", n_estimators=500, num_leaves=31, learning_rate=0.05,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1,
        )
        model_v11.fit(all_data[DEC_V11_FEATURES], all_data["delta_run_exp"])
        del all_data
        V2_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_V11_PATH, "wb") as f:
            pickle.dump(model_v11, f)
        log(f"  Model saved → {MODEL_V11_PATH.name}")
        cp["part_c_model_trained"] = True
        save_checkpoint(cp)
    else:
        log("  Loading saved v1.1 model...")
        with open(MODEL_V11_PATH, "rb") as f:
            model_v11 = pickle.load(f)

    # ── Concurrent scoring (all seasons with trained model) ───────────────
    log("\n  Concurrent scoring 2021-2025...")
    season_scores = {}
    for s in SEASONS:
        df   = pd.read_sql(
            f"SELECT {', '.join(DEC_QUERY_COLS)} FROM pitches "
            f"WHERE season={s} AND game_type='R' AND delta_run_exp IS NOT NULL", conn
        )
        feat = dec_eng(df, bio)
        feat["season"] = feat["season"].astype(str).astype(int)
        preds = model_v11.predict(feat[DEC_V11_FEATURES])
        tmp   = pd.DataFrame({"pitcher_id": feat["pitcher_id"].values,
                               "season": s, "pred": preds})
        g = tmp.groupby("pitcher_id").agg(mean_pred=("pred","mean"), n=("pred","size")).reset_index()
        g = g[g["n"] >= MIN_P]
        m, std = g["mean_pred"].mean(), g["mean_pred"].std()
        g["plus"] = to_plus_scale(g["mean_pred"], m, std) if std > 0 else 100.0
        season_scores[s] = g[["pitcher_id","plus"]].copy()
        log(f"    {s}: {len(g)} pitchers")

    # Merge all seasons
    concurrent_rows = []
    for s, sg in season_scores.items():
        sg = sg.copy(); sg["season"] = s
        concurrent_rows.append(sg)
    concurrent = pd.concat(concurrent_rows, ignore_index=True)

    # ── Concurrent validity ───────────────────────────────────────────────
    outcomes = pd.read_sql(
        "SELECT player_id AS pitcher_id, season, era, xfip "
        "FROM player_advanced_stats "
        "WHERE era IS NOT NULL AND xfip IS NOT NULL AND season BETWEEN 2021 AND 2025", conn
    )
    m_conc = concurrent.merge(outcomes, on=["pitcher_id","season"])
    era_r,  era_lo,  era_hi,  n_era  = pearson_with_ci(m_conc["plus"], m_conc["era"])
    xfip_r, xfip_lo, xfip_hi, n_xfip = pearson_with_ci(m_conc["plus"], m_conc["xfip"])
    log(f"\n  Concurrent ERA r={era_r:+.3f} [{era_lo:+.3f},{era_hi:+.3f}] n={n_era}")
    log(f"  Concurrent xFIP r={xfip_r:+.3f} [{xfip_lo:+.3f},{xfip_hi:+.3f}]")

    # ── YoY r ─────────────────────────────────────────────────────────────
    yoy_rows = []
    for s1, s2 in YOY_PAIRS:
        a = concurrent[concurrent.season==s1][["pitcher_id","plus"]].rename(columns={"plus":"p1"})
        b = concurrent[concurrent.season==s2][["pitcher_id","plus"]].rename(columns={"plus":"p2"})
        yoy_rows.append(a.merge(b, on="pitcher_id"))
    yoy_all = pd.concat(yoy_rows, ignore_index=True)
    yoy_r, yoy_lo, yoy_hi, n_yoy = pearson_with_ci(yoy_all["p1"], yoy_all["p2"])
    log(f"  YoY r={yoy_r:+.3f} [{yoy_lo:+.3f},{yoy_hi:+.3f}] n={n_yoy}")

    # ── Split-half reliability ────────────────────────────────────────────
    log("\n  Split-half reliability...")
    half_rows = []
    rng_sh = np.random.default_rng(0)
    for s in SEASONS:
        df   = pd.read_sql(
            f"SELECT {', '.join(DEC_QUERY_COLS)}, rowid FROM pitches "
            f"WHERE season={s} AND game_type='R' AND delta_run_exp IS NOT NULL", conn
        )
        feat = dec_eng(df, bio)
        feat["season"] = feat["season"].astype(str).astype(int)
        feat["pred"] = model_v11.predict(feat[DEC_V11_FEATURES])
        feat["half"] = 0
        for pid, grp in feat.groupby("pitcher_id"):
            idx = grp.index.values
            rng_sh.shuffle(idx)
            feat.loc[idx[len(idx)//2:], "half"] = 1

        g = feat.groupby(["pitcher_id","half"])["pred"].mean().unstack("half")
        g.columns = ["h0","h1"]
        g = g.dropna()
        if len(g) < 10:
            continue
        m0, s0 = g["h0"].mean(), g["h0"].std()
        m1, s1 = g["h1"].mean(), g["h1"].std()
        g["ph0"] = to_plus_scale(g["h0"], m0, s0) if s0 > 0 else 100.0
        g["ph1"] = to_plus_scale(g["h1"], m1, s1) if s1 > 0 else 100.0
        g["season"] = s
        half_rows.append(g.reset_index())

    half_all = pd.concat(half_rows, ignore_index=True)
    raw_r, _, _, _ = pearson_with_ci(half_all["ph0"], half_all["ph1"])
    sb_r = 2 * raw_r / (1 + raw_r) if not np.isnan(raw_r) else float("nan")
    sb_r_pt, sb_lo, sb_hi, n_sh = pearson_with_ci(
        half_all["ph0"], half_all["ph1"]
    )
    # Apply SB correction to CI endpoints
    def sb(r):
        return 2 * r / (1 + r) if not np.isnan(r) else float("nan")
    sb_r_pt = sb(sb_r_pt)
    sh_r, sh_lo, sh_hi = sb_r_pt, sb(sb_lo), sb(sb_hi)
    log(f"  Split-half (SB corrected) r={sh_r:+.3f} [{sh_lo:+.3f},{sh_hi:+.3f}] n={n_sh}")

    # ── Strict OOS (train 2021-2024, test 2025) ───────────────────────────
    log("\n  Strict OOS (train 2021-2024, test 2025)...")
    oos_parts = []
    for s in [2021, 2022, 2023, 2024]:
        df   = pd.read_sql(
            f"SELECT {', '.join(DEC_QUERY_COLS)} FROM pitches "
            f"WHERE season={s} AND game_type='R' AND delta_run_exp IS NOT NULL", conn
        )
        feat = dec_eng(df, bio)
        feat["season"] = feat["season"].astype(str).astype(int)
        oos_parts.append(feat)
    oos_train = pd.concat(oos_parts, ignore_index=True)
    del oos_parts

    oos_model = lgb.LGBMRegressor(
        objective="huber", n_estimators=500, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1,
    )
    oos_model.fit(oos_train[DEC_V11_FEATURES], oos_train["delta_run_exp"])
    del oos_train

    df_25  = pd.read_sql(
        f"SELECT {', '.join(DEC_QUERY_COLS)} FROM pitches "
        f"WHERE season=2025 AND game_type='R' AND delta_run_exp IS NOT NULL", conn
    )
    feat_25 = dec_eng(df_25, bio)
    feat_25["season"] = feat_25["season"].astype(str).astype(int)
    preds_25 = oos_model.predict(feat_25[DEC_V11_FEATURES])
    tmp_25   = pd.DataFrame({"pitcher_id": feat_25["pitcher_id"].values, "pred": preds_25})
    g_25 = tmp_25.groupby("pitcher_id").agg(mean_pred=("pred","mean"), n=("pred","size")).reset_index()
    g_25 = g_25[g_25["n"] >= MIN_P]
    m25, s25 = g_25["mean_pred"].mean(), g_25["mean_pred"].std()
    g_25["plus"] = to_plus_scale(g_25["mean_pred"], m25, s25) if s25 > 0 else 100.0

    outs_25 = pd.read_sql(
        "SELECT player_id AS pitcher_id, era, xfip FROM player_advanced_stats "
        "WHERE season=2025 AND era IS NOT NULL AND xfip IS NOT NULL", conn
    )
    m_oos = g_25.merge(outs_25, on="pitcher_id")
    oos_era_r,  oos_era_lo,  oos_era_hi,  n_oos  = pearson_with_ci(m_oos["plus"], m_oos["era"])
    oos_xfip_r, oos_xfip_lo, oos_xfip_hi, _      = pearson_with_ci(m_oos["plus"], m_oos["xfip"])
    log(f"  OOS ERA r={oos_era_r:+.3f} [{oos_era_lo:+.3f},{oos_era_hi:+.3f}] n={n_oos}")
    log(f"  OOS xFIP r={oos_xfip_r:+.3f}")

    # ── Load v1 numbers from v1 checkpoint for gap comparison ─────────────
    v1_cp = {}
    if V1_CP_PATH.exists():
        v1_cp = json.loads(V1_CP_PATH.read_text())
    v1_p1 = v1_cp.get("part1_results", {})
    v1_conc_era  = v1_p1.get("dec_plus",{}).get("era",{}).get("r", float("nan"))
    v1_oos_era   = v1_cp.get("part1_results",{}).get("strict_oos",{}).get("dec_plus",{}).get("era_r", float("nan"))
    v1_gap       = abs(v1_conc_era - v1_oos_era) if not (np.isnan(v1_conc_era) or np.isnan(v1_oos_era)) else float("nan")
    v11_gap      = abs(era_r - oos_era_r) if not (np.isnan(era_r) or np.isnan(oos_era_r)) else float("nan")

    log(f"\n  v1  concurrent→OOS gap: {v1_conc_era:+.3f} → {v1_oos_era:+.3f}  Δ={v1_gap:.3f}")
    log(f"  v1.1 concurrent→OOS gap: {era_r:+.3f} → {oos_era_r:+.3f}  Δ={v11_gap:.3f}")
    gap_narrows = v11_gap < v1_gap if not (np.isnan(v1_gap) or np.isnan(v11_gap)) else None

    results = {
        "concurrent": {
            "era_r": era_r, "era_ci": [era_lo, era_hi], "n": n_era,
            "xfip_r": xfip_r, "xfip_ci": [xfip_lo, xfip_hi],
        },
        "yoy": {"r": yoy_r, "lo": yoy_lo, "hi": yoy_hi, "n": n_yoy},
        "split_half": {"r": sh_r, "lo": sh_lo, "hi": sh_hi, "n": n_sh},
        "strict_oos": {
            "era_r": oos_era_r, "era_ci": [oos_era_lo, oos_era_hi], "n": n_oos,
            "xfip_r": oos_xfip_r,
        },
        "v1_comparison": {
            "v1_conc_era": v1_conc_era,
            "v1_oos_era":  v1_oos_era,
            "v1_gap":      v1_gap,
            "v11_gap":     v11_gap,
            "gap_narrows": gap_narrows,
        },
        "features": DEC_V11_FEATURES,
    }
    cp["part_c_done"]    = True
    cp["part_c_results"] = results
    save_checkpoint(cp)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# HTML generation
# ═══════════════════════════════════════════════════════════════════════════

def _td(val, lo=None, hi=None, neg_good=False, pos_good=False, fmt=".3f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "<td>—</td>"
    cls = ""
    if neg_good:
        cls = "good" if val < -0.05 else ("bad" if val > 0.05 else "")
    elif pos_good:
        cls = "good" if val > 0.35 else ("bad" if val < 0.10 else "")
    s = f"{val:+{fmt}}"
    if lo is not None and not np.isnan(lo):
        s += f" [{lo:+{fmt}},{hi:+{fmt}}]"
    return f'<td class="{cls}">{s}</td>'


def generate_html(p1, pa, pb, pc):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    clf_names = ["xswing","xcs","xwhiff","xfoul"]

    # ── Part 1 rows (from v1 checkpoint) ─────────────────────────────────
    m_rows = []
    for col, name in [("stuff_plus","Stuff+ v3"),("loc_plus","Location+ v2"),
                      ("tun_plus","Tunneling+ v2"),("dec_plus","Deception+ v1")]:
        info = p1.get(col, {})
        er   = info.get("era",{})
        xr   = info.get("xfip",{})
        yr   = info.get("yoy",{})
        sh   = info.get("split_half",{})
        m_rows.append(
            f"<tr><td><strong>{name}</strong></td>"
            + _td(er.get("r"),  er.get("lo"),  er.get("hi"),  neg_good=True)
            + _td(xr.get("r"),  xr.get("lo"),  xr.get("hi"),  neg_good=True)
            + _td(yr.get("r"),  yr.get("lo"),  yr.get("hi"),  pos_good=True)
            + _td(sh.get("r"),  sh.get("lo"),  sh.get("hi"),  pos_good=True)
            + f"<td>{er.get('n','?')}</td></tr>"
        )

    oos_rows = []
    for col, name in [("stuff_plus","Stuff+ v3"),("loc_plus","Location+ v2"),
                      ("tun_plus","Tunneling+ v2"),("dec_plus","Deception+ v1")]:
        d   = (p1.get("strict_oos") or {}).get(col, {})
        er  = d.get("era_r", float("nan"))
        xr  = d.get("xfip_r", float("nan"))
        ci  = d.get("era_ci", [float("nan"),float("nan")])
        note= d.get("note","")
        oos_rows.append(
            f"<tr><td>{name}</td>"
            + _td(er, ci[0], ci[1], neg_good=True)
            + _td(xr, neg_good=True)
            + f"<td>{d.get('n','?')}</td>"
            + f"<td style='font-size:10px;color:#888'>{note}</td></tr>"
        )

    # ── Part A AUC table ──────────────────────────────────────────────────
    auc_means = pa.get("auc_means", {})
    auc_cis   = pa.get("auc_cis", {})
    auc_rows  = []
    for fset in ["L","T","L+G","L+S","L+G+S","T-G"]:
        row = f"<tr><td><strong>{fset}</strong></td>"
        for clf in clf_names:
            v  = auc_means.get(fset,{}).get(clf, float("nan"))
            ci = auc_cis.get(fset,{}).get(clf, [float("nan"),float("nan")])
            if np.isnan(v):
                row += "<td>—</td>"
            else:
                row += f"<td>{v:.4f} [{ci[0]:.4f},{ci[1]:.4f}]</td>"
        row += "</tr>"
        auc_rows.append(row)

    # ── Part A contrasts ──────────────────────────────────────────────────
    contrasts = pa.get("contrasts", {})
    contrast_rows = []
    contrast_defs = [
        ("(L+G)−L",   "LG_minus_L"),
        ("(L+S)−L",   "LS_minus_L"),
        ("(L+G+S)−(L+S)", "LGS_minus_LS"),
        ("T−L",       "T_minus_L"),
    ]
    for label, key in contrast_defs:
        row = f"<tr><td>{label}</td>"
        for clf in clf_names:
            c = contrasts.get(clf,{}).get(key,{})
            d, lo, hi = c.get("d", float("nan")), c.get("lo", float("nan")), c.get("hi", float("nan"))
            if np.isnan(d):
                row += "<td>—</td>"
            else:
                thr = THRESHOLDS["auc_leakage_delta"]
                flag = " ⚠" if abs(d) > thr and not (lo <= 0 <= hi) else ""
                row += f"<td>{d:+.4f} [{lo:+.4f},{hi:+.4f}]{flag}</td>"
        row += "</tr>"
        contrast_rows.append(row)

    # ── Part A geometry R² ────────────────────────────────────────────────
    gr = pa.get("geom_r2", {})
    def geom_cell(target, model_key, threshold):
        v = gr.get(target,{}).get(model_key, float("nan"))
        if np.isnan(v): return "<td>—</td>"
        flag = " ⚠ LEAKAGE" if v > threshold else " ✓ low"
        return f"<td>{v:.4f}{flag}</td>"

    thr_g = THRESHOLDS["geom_to_loc_r2"]
    verdict = pa.get("leakage_verdict","ambiguous")
    verdict_html = {
        "leakage":           '<span style="color:var(--rd);font-weight:bold">⚠ LEAKAGE — geometry encodes location</span>',
        "real_geometry_info":'<span style="color:var(--gn);font-weight:bold">✓ NO LEAKAGE — geometry carries real relational info</span>',
        "ambiguous":         '<span style="color:#7a5800;font-weight:bold">— AMBIGUOUS — CI spans zero for key contrasts</span>',
    }.get(verdict, verdict)

    # ── Part B rows ───────────────────────────────────────────────────────
    variants = pb.get("variants", {})
    diffs    = pb.get("differences", {})
    p3_rows  = []
    for vname in ["V0","V1","V-rand-0","V-rand-1","V-rand-2"]:
        d = variants.get(vname, {})
        yr, yci = d.get("yoy_r",float("nan")),  d.get("yoy_ci",[float("nan"),float("nan")])
        er, eci = d.get("era_r",float("nan")),   d.get("era_ci",[float("nan"),float("nan")])
        xr, xci = d.get("xfip_r",float("nan")),  d.get("xfip_ci",[float("nan"),float("nan")])
        cv = d.get("cv_r2", float("nan"))
        p3_rows.append(
            f"<tr><td><strong>{vname}</strong></td>"
            f"<td>{cv:+.5f}</td>"
            + _td(yr, yci[0], yci[1], pos_good=True)
            + _td(er, eci[0], eci[1], neg_good=True)
            + _td(xr, xci[0], xci[1], neg_good=True)
            + "</tr>"
        )

    def diff_row(name, label):
        d = diffs.get(name, {})
        dy, dyci = d.get("diff_yoy_r",float("nan")), d.get("diff_yoy_ci",[float("nan"),float("nan")])
        de, deci = d.get("diff_era_r",float("nan")), d.get("diff_era_ci",[float("nan"),float("nan")])
        ex0 = d.get("yoy_excludes_zero", False)
        marker = " ✗ (excludes 0)" if ex0 else " (CI includes 0)"
        return (f"<tr><td>{label}</td>"
                + _td(dy, dyci[0], dyci[1])
                + _td(de, deci[0], deci[1])
                + f"<td>{'<b>CONFIRMED</b>' if ex0 else 'not confirmed'}{marker}</td></tr>")

    pb_finding = diffs.get("V1",{}).get("yoy_excludes_zero", False)
    pb_verdict_html = (
        '<span style="color:var(--rd);font-weight:bold">CONFIRMED — dropping height/weight reduces stability; CI on ΔYoY r excludes zero</span>'
        if pb_finding else
        '<span style="color:var(--mu)">Not confirmed — CI on ΔYoY r includes zero under proper uncertainty quantification</span>'
    )

    # ── Part C rows ───────────────────────────────────────────────────────
    pc_conc = pc.get("concurrent",{})
    pc_yoy  = pc.get("yoy",{})
    pc_sh   = pc.get("split_half",{})
    pc_oos  = pc.get("strict_oos",{})
    pc_cmp  = pc.get("v1_comparison",{})

    v1_era_row = f"<tr><td>Deception+ <strong>v1</strong> (live)</td>" \
        + _td(pc_cmp.get("v1_conc_era"), neg_good=True) \
        + "<td>—</td><td>—</td>" \
        + _td(pc_cmp.get("v1_oos_era"), neg_good=True) \
        + f"<td>{pc_cmp.get('v1_gap',float('nan')):+.3f}</td></tr>"

    v11_era_row = f"<tr><td>Deception+ <strong>v1.1</strong> (candidate)</td>" \
        + _td(pc_conc.get("era_r"), pc_conc.get("era_ci",[None,None])[0],
              pc_conc.get("era_ci",[None,None])[1], neg_good=True) \
        + _td(pc_yoy.get("r"), pc_yoy.get("lo"), pc_yoy.get("hi"), pos_good=True) \
        + _td(pc_sh.get("r"), pc_sh.get("lo"), pc_sh.get("hi"), pos_good=True) \
        + _td(pc_oos.get("era_r"), pc_oos.get("era_ci",[None,None])[0],
              pc_oos.get("era_ci",[None,None])[1], neg_good=True) \
        + f"<td>{pc_cmp.get('v11_gap',float('nan')):+.3f}</td></tr>"

    gap_narrows = pc_cmp.get("gap_narrows")
    gap_verdict = (
        '<span style="color:var(--gn);font-weight:bold">✓ Gap narrows — v1.1 is less inflated in-sample relative to OOS</span>'
        if gap_narrows is True else
        '<span style="color:var(--rd);font-weight:bold">✗ Gap does not narrow — removing identity features did not reduce concurrent inflation</span>'
        if gap_narrows is False else
        '<span style="color:var(--mu)">— Gap comparison unavailable</span>'
    )

    pc_finding_note = "" if pb_finding else (
        "<p><strong>Note:</strong> Part B did not confirm the memorization finding (CI on ΔYoY r includes zero). "
        "This candidate was built for completeness; the original finding did not hold up under proper uncertainty "
        "quantification.</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Model Validation</title>
<style>
  :root {{ --or:#92540e; --orl:#fef3e2; --bd:#d0d5c8; --mu:#5a6450; --gn:#1a5c38; --rd:#b91c2c; --bg:#fff; --fg:#1a1a1a; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{ --or:#d4823a;--orl:#2d1a05;--bd:#444;--mu:#9aab90;--gn:#5ecb90;--rd:#f87171;--bg:#1a1a1a;--fg:#e8e8e8; }}
  }}
  :root[data-theme="dark"] {{ --or:#d4823a;--orl:#2d1a05;--bd:#444;--mu:#9aab90;--gn:#5ecb90;--rd:#f87171;--bg:#1a1a1a;--fg:#e8e8e8; }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:Georgia,serif;font-size:13px;line-height:1.6;background:var(--bg);color:var(--fg);padding:40px 48px;max-width:1200px;margin:0 auto}}
  header{{border-bottom:3px solid var(--or);padding-bottom:16px;margin-bottom:28px}}
  .model-name{{font-size:26px;font-weight:bold;color:var(--or)}}
  .model-sub{{font-family:monospace;font-size:12px;color:var(--mu);margin-top:4px}}
  .badges{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
  .badge{{padding:3px 10px;border-radius:4px;font-family:monospace;font-size:11px;font-weight:bold}}
  .badge-ok{{background:#d1fae5;color:#065f46;border:1px solid #059669}}
  .badge-warn{{background:#fff3cd;color:#7a5800;border:1px solid #c89a00}}
  h2{{font-size:14px;color:var(--or);border-bottom:1px solid var(--bd);padding-bottom:5px;margin:28px 0 12px;text-transform:uppercase;letter-spacing:.04em;font-family:Arial,sans-serif}}
  h3{{font-size:13px;font-weight:bold;margin:16px 0 6px;font-family:Arial,sans-serif}}
  p{{margin-bottom:10px}}
  table{{width:100%;border-collapse:collapse;font-size:12px;margin:10px 0 16px;overflow-x:auto;display:block}}
  th{{background:var(--or);color:#fff;text-align:left;padding:6px 8px;font-family:monospace;font-size:11px;white-space:nowrap}}
  td{{padding:4px 8px;border-bottom:1px solid var(--bd);vertical-align:top;font-family:monospace;white-space:nowrap}}
  tr:nth-child(even) td{{background:var(--orl)}}
  td.good{{color:var(--gn);font-weight:bold}}
  td.bad{{color:var(--rd);font-weight:bold}}
  .callout{{background:var(--orl);border-left:4px solid #c89a00;padding:10px 14px;margin:12px 0;font-size:12px}}
  .verdict{{background:var(--orl);border-left:4px solid var(--or);padding:10px 14px;margin:12px 0;font-size:12px}}
  .section{{margin-bottom:28px}}
  footer{{margin-top:40px;padding-top:12px;border-top:1px solid var(--bd);font-size:11px;color:var(--mu);font-family:monospace}}
</style>
</head>
<body>
<header>
  <div class="model-name">Cross-Model Validation &amp; Leakage Tests — v2</div>
  <div class="model-sub">training/cross_model_validation_v2.py · Generated {now}</div>
  <div class="badges">
    <span class="badge badge-ok">✓ COMPLETE</span>
    <span class="badge badge-warn">Live models untouched — R&amp;D only</span>
  </div>
</header>

<div class="callout"><strong>Sign convention:</strong>
r(plus_score, ERA) is <strong>NEGATIVE</strong> for correctly-signed models (higher score = better pitcher = lower ERA).
Green = negative ERA r / positive YoY r. CIs are 95% throughout.
</div>

<div class="section">
<h2>Part 1 — Unified Validation (2021–2025)</h2>
<h3>1.1 Concurrent &amp; Predictive Validity</h3>
<table>
  <thead><tr><th>Model</th><th>ERA r [95% CI]</th><th>xFIP r [CI]</th><th>YoY r [CI]</th><th>Split-half SB [CI]</th><th>n</th></tr></thead>
  <tbody>{"".join(m_rows)}</tbody>
</table>
<h3>1.2 Strict OOS: train 2021–2024, test 2025</h3>
<table>
  <thead><tr><th>Model</th><th>ERA r [95% CI]</th><th>xFIP r</th><th>n</th><th>Note</th></tr></thead>
  <tbody>{"".join(oos_rows)}</tbody>
</table>
</div>

<div class="section">
<h2>Part A — Tunneling+ Location-Leakage Test (Corrected)</h2>
<p>Root cause of blank Part 2 in prior report: <code>part2_results</code> key absent from manually-reconstructed checkpoint; HTML template read from that key rather than flat per-set keys. Rerun here collects fold-level AUCs for proper CIs.</p>
<p>LOSO 5-fold AUC for 4 classifiers × 6 nested feature sets (2021–2025). AUC shown as mean [t-CI on 5 fold values, df=4].<br>
L = location (4 num + 3 cat), T = tunneling (all features), G = geometry lags ({len(NUMERIC_LAG_COLS)} numeric), S = pitch-type seq lags ({len(S_FEATURES)} cat), T-G = T minus geometry.</p>
<h3>A.1 LOSO Mean AUC [95% CI across folds]</h3>
<table><thead><tr><th>Feature Set</th><th>xSwing</th><th>xCS</th><th>xWhiff</th><th>xFoul</th></tr></thead>
<tbody>{"".join(auc_rows)}</tbody></table>

<h3>A.2 Pre-registered Contrasts [paired t-CI on fold differences]</h3>
<p>Threshold: ΔAUC &gt; {THRESHOLDS['auc_leakage_delta']:.3f} with CI excluding 0 = non-trivial contribution.</p>
<table><thead><tr><th>Contrast</th><th>xSwing</th><th>xCS</th><th>xWhiff</th><th>xFoul</th></tr></thead>
<tbody>{"".join(contrast_rows)}</tbody></table>

<h3>A.3 Geometry → Plate Position R² (Ridge, 2024)</h3>
<p>Threshold: &gt;{thr_g:.2f} = non-trivial encoding. Three models tested: geometry alone (G), geometry + pitch_type/p_throws/stand (G+cat), pitch_type alone (PT).</p>
<table>
  <thead><tr><th>Target</th><th>Geometry only</th><th>Geometry + pitch_type/p_throws/stand</th><th>pitch_type alone</th></tr></thead>
  <tbody>
    <tr><td>plate_x</td>{geom_cell("plate_x","geom_only",thr_g)}{geom_cell("plate_x","geom_plus_cat",thr_g)}{geom_cell("plate_x","pt_only",thr_g)}</tr>
    <tr><td>plate_z</td>{geom_cell("plate_z","geom_only",thr_g)}{geom_cell("plate_z","geom_plus_cat",thr_g)}{geom_cell("plate_z","pt_only",thr_g)}</tr>
  </tbody>
</table>

<div class="verdict"><strong>Leakage verdict:</strong> {verdict_html}</div>
</div>

<div class="section">
<h2>Part B — Deception+ Identity-Proxy Ablation (Bootstrapped CIs)</h2>
<p>V0 = full 10 features, V1 = drop height/weight (8 features), V-rand = randomize height/weight (seeds 0/1/2).<br>
Huber LightGBM LOSO 5-fold on 2021–2025. Bootstrap CIs are pitcher-clustered (resample pitcher IDs, n=1000).</p>
<table>
  <thead><tr><th>Variant</th><th>CV R²</th><th>YoY r [95% bootstrap CI]</th><th>ERA r [CI]</th><th>xFIP r [CI]</th></tr></thead>
  <tbody>{"".join(p3_rows)}</tbody>
</table>

<h3>B.1 Difference CIs (same pitcher resamples for paired comparison)</h3>
<p>Key test: does CI on ΔYoY r exclude zero? If yes, height/weight loss is a real stability reduction, not noise.</p>
<table>
  <thead><tr><th>Comparison</th><th>ΔYoY r [95% CI]</th><th>ΔERA r [CI]</th><th>Memorization finding</th></tr></thead>
  <tbody>
    {diff_row("V1",     "V1 − V0 (drop h/w)")}
    {diff_row("V-rand", "V-rand − V0 (fake h/w, pooled seeds)")}
  </tbody>
</table>

<div class="verdict"><strong>Memorization verdict:</strong> {pb_verdict_html}</div>
</div>

<div class="section">
<h2>Part C — Deception+ v1.1 Candidate (No Height/Weight)</h2>
{pc_finding_note}
<p>Features: {', '.join(pc.get('features', DEC_V11_FEATURES))} ({len(pc.get('features', DEC_V11_FEATURES))} total).<br>
Saved as <code>training/v2_artifacts/deception_plus_v1_1_no_bodysize.pkl</code>. Live model untouched.</p>
<h3>C.1 Full Validation Battery vs Live Deception+ v1</h3>
<table>
  <thead><tr><th>Model</th><th>Concurrent ERA r [CI]</th><th>YoY r [CI]</th><th>Split-half SB [CI]</th><th>Strict OOS ERA r [CI]</th><th>Conc→OOS gap</th></tr></thead>
  <tbody>{v1_era_row}{v11_era_row}</tbody>
</table>
<p style="font-size:11px;color:var(--mu)">v1 YoY/split-half from Part 1; v1 OOS from strict OOS test. Gap = |concurrent ERA r − strict OOS ERA r|.</p>

<div class="verdict"><strong>Gap verdict:</strong> {gap_verdict}</div>
</div>

<footer>Cross-Model Validation v2 · {now} · training/cross_model_validation_v2.py · All live models untouched</footer>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# HANDOFF update
# ═══════════════════════════════════════════════════════════════════════════

def update_handoff(pa, pb, pc):
    handoff_path = ROOT / "HANDOFF.md"
    if not handoff_path.exists():
        return
    text = handoff_path.read_text(encoding="utf-8")
    tag  = "<!-- cross_model_v2_summary -->"
    verdict  = pa.get("leakage_verdict", "ambiguous")
    pb_conf  = pb.get("differences",{}).get("V1",{}).get("yoy_excludes_zero", False)
    gap_narr = pc.get("v1_comparison",{}).get("gap_narrows")
    summary  = (
        f"\n{tag}\n"
        f"## Cross-Model Validation v2 (generated {datetime.now().strftime('%Y-%m-%d')})\n\n"
        f"**Part A** — Tunneling+ leakage test (corrected): leakage verdict = `{verdict}`.\n"
        f"**Part B** — Deception+ ablation with bootstrap CIs: memorization confirmed = `{pb_conf}`. "
        f"ΔYoY r (V1−V0) = {pb.get('differences',{}).get('V1',{}).get('diff_yoy_r',float('nan')):+.3f} "
        f"[{pa.get('leakage_verdict','')}].\n"
        f"**Part C** — Dec+ v1.1 candidate (no height/weight): gap narrows = `{gap_narr}`. "
        f"Concurrent ERA r = {pc.get('concurrent',{}).get('era_r',float('nan')):+.3f}, "
        f"OOS ERA r = {pc.get('strict_oos',{}).get('era_r',float('nan')):+.3f}.\n"
        f"Report: `Methodology/in_progress/CROSS_MODEL_VALIDATION_AND_LEAKAGE_TESTS_IN_PROGRESS.html`\n"
        f"{tag}\n"
    )
    if tag in text:
        import re
        text = re.sub(f"{tag}.*?{tag}", summary.strip(), text, flags=re.DOTALL)
    else:
        text += "\n" + summary
    handoff_path.write_text(text, encoding="utf-8")
    log("HANDOFF.md updated.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    V2_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log(f"Cross-model validation v2 started: {datetime.now()}")
    log(f"DB:     {DB_PATH}")
    log(f"Output: {HTML_OUTPUT}")
    log("=" * 60)

    cp   = load_checkpoint()
    conn = sqlite3.connect(str(DB_PATH))

    try:
        # Load Part 1 results from v1 checkpoint (no re-run needed)
        v1_cp = {}
        if V1_CP_PATH.exists():
            v1_cp = json.loads(V1_CP_PATH.read_text())
        p1 = v1_cp.get("part1_results", {})
        log("Part 1: loaded from v1 checkpoint.")

        pa = run_part_a(conn, cp)
        pb = run_part_b(conn, cp)
        pc = run_part_c(conn, cp)
    finally:
        conn.close()

    log("\nGenerating HTML report...")
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT.write_text(generate_html(p1, pa, pb, pc), encoding="utf-8")
    log(f"Report → {HTML_OUTPUT}")

    update_handoff(pa, pb, pc)
    log("\nDone.")


if __name__ == "__main__":
    main()
