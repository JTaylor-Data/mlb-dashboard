#!/usr/bin/env python3
"""
Cross-model validation and leakage tests.

Part 1: Unified four-model validation table (Stuff+ v3, Location+ v2,
        Tunneling+ v2, Deception+ v1) with one sign convention.
        Sign anchor: r(plus, ERA) NEGATIVE = correct (higher plus = lower ERA = better).

Part 2: Tunneling+ location-leakage test — nested feature sets LOSO AUC.
        Feature sets: L, T, L+G, L+S, L+G+S, T-G.

Part 3: Deception+ identity-proxy ablation — drop/randomize height/weight.

Constraints enforced:
  - No production changes. Live models stay untouched.
  - Effect sizes (r, ΔAUC) with 95% CIs are primary; z-scores secondary.
  - Decision thresholds stated before each test.
  - Exclude 2020. YoY pairs consecutive only (2021→2022 ... 2024→2025).
  - xfip column is `xfip`. Filter: era IS NOT NULL AND xfip IS NOT NULL.
  - Avoid unnecessary .copy() calls to minimize memory pressure.
  - Checkpoint after Part 1, after each feature set in Part 2, each variant in Part 3.
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sqlite3
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.location_features import to_plus_scale
from analytics.tunneling_v2_model import (
    _raw_predictions  as tun_raw,
    NUMERIC_LAG_COLS,
    ALL_CATEGORICAL,
    ALL_FEATURES as T_FEATURES,
    _cat_levels  as tun_cat_levels,
    _prepare_features as tun_prep,
    _combine_xrv as tun_combine,
    _models      as tun_models,
    _rv          as tun_rv,
    _meta        as tun_meta,
    _load_commit_table,
    load_pitches as tun_load_pitches,
    MAX_LAG      as TUN_MAX_LAG,
)
from analytics.location_v2_model import (
    _raw_predictions  as loc_raw,
    _cat_levels  as loc_cat_levels,
    _prep        as loc_prep,
    _combine_xrv as loc_combine,
    _models      as loc_models,
    _rv          as loc_rv,
    _meta        as loc_meta,
    MIN_PITCHES  as LOC_MIN_PITCHES,
)
from analytics.deception_model    import _raw_predictions as dec_raw
from analytics.deception_features import (
    ALL_FEATURES      as DEC_FEATURES,
    CATEGORICAL_FEATURES as DEC_CAT,
    RAW_QUERY_COLUMNS as DEC_QUERY_COLS,
    engineer_features as dec_eng,
    load_bio          as dec_load_bio,
)
from analytics.tunneling_features import sequential_lag_features

DB_PATH        = ROOT / "mlb_data.db"
V2_ARTIFACTS   = ROOT / "training" / "v2_artifacts"
CHECKPOINT_PATH= V2_ARTIFACTS / "cross_model_validation_checkpoint.json"
LOG_PATH       = ROOT / "training" / "cross_model_validation.log"
HTML_OUTPUT    = ROOT / "Methodology" / "in_progress" / "CROSS_MODEL_VALIDATION_AND_LEAKAGE_TESTS_IN_PROGRESS.html"

SEASONS = [2021, 2022, 2023, 2024, 2025]
YOY_PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
MIN_PITCHES_VALIDATION = 300

# Description sets — same as train_v2_models.py
SWING_DESCS = {
    "swinging_strike","swinging_strike_blocked","foul_tip","bunt_foul_tip",
    "missed_bunt","foul","foul_bunt","foul_pitchout",
    "hit_into_play","hit_into_play_no_out","hit_into_play_score","swinging_pitchout",
}
WHIFF_DESCS = {
    "swinging_strike","swinging_strike_blocked","foul_tip",
    "bunt_foul_tip","missed_bunt","swinging_pitchout",
}
FOUL_DESCS = {"foul","foul_bunt","foul_pitchout"}
BIP_DESCS  = {"hit_into_play","hit_into_play_no_out","hit_into_play_score"}
CS_DESCS   = {"called_strike","automatic_strike"}

# ─── Decision thresholds (stated before running each test) ──────────────────
THRESHOLDS = {
    "yoy_r_adequate":      0.40,   # below = inadequate predictive stability
    "era_r_adequate":     -0.10,   # must be more negative (correct sign anchor)
    "split_half_adequate": 0.50,   # Spearman-Brown corrected r below = unreliable
    "auc_leakage_delta":   0.010,  # ΔAUC > 0.010 = non-trivial location contribution
    "geom_to_loc_r2":      0.05,   # R² > 0.05: geometry encodes plate location
    "ablation_cv_r2_delta":0.010,  # |ΔCV R²| > 0.01 = meaningful identity memorization
    "ablation_yoy_delta":  0.05,   # |ΔYoY r| > 0.05 = meaningful identity memorization
}

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

def log(msg):
    logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ─── Checkpointing ──────────────────────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {}

def save_checkpoint(cp):
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2, default=str))
    log(f"  Checkpoint saved → {CHECKPOINT_PATH.name}")


# ─── Statistics ─────────────────────────────────────────────────────────────
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


def auc_from_fold(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def spearman_brown(r):
    return 2 * r / (1 + r)


def fmt_r(r, lo=None, hi=None, n=None):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "—"
    s = f"{r:+.3f}"
    if lo is not None and hi is not None:
        s += f" [{lo:+.3f}, {hi:+.3f}]"
    if n is not None:
        s += f" (n={n})"
    return s


# ─── Target column helper ───────────────────────────────────────────────────
def add_targets(df):
    """Add boolean target columns from description. Modifies df in-place."""
    desc = df["description"]
    df["is_swing"]          = desc.isin(SWING_DESCS).astype(int)
    df["is_whiff"]          = desc.isin(WHIFF_DESCS).astype(int)
    df["is_foul"]           = desc.isin(FOUL_DESCS).astype(int)
    df["is_bip"]            = desc.isin(BIP_DESCS).astype(int)
    df["is_called_strike"]  = desc.isin(CS_DESCS).astype(int)
    return df


# ─── Model scoring utilities ────────────────────────────────────────────────
def score_location_plus_v2(conn, season):
    raw = loc_raw(conn, season)
    if raw.empty:
        return pd.DataFrame(columns=["pitcher_id","plus","n_pitches"])
    g = raw.groupby("pitcher_id").agg(mean_xrv=("xrv","mean"), n=("xrv","size"))
    g = g[g["n"] >= LOC_MIN_PITCHES]
    m, s = g["mean_xrv"].mean(), g["mean_xrv"].std()
    if s == 0:
        return pd.DataFrame(columns=["pitcher_id","plus","n_pitches"])
    g["plus"] = to_plus_scale(g["mean_xrv"], m, s)
    g = g.reset_index().rename(columns={"n":"n_pitches"})
    return g[["pitcher_id","plus","n_pitches"]]


def score_tunneling_plus_v2(conn, season):
    raw = tun_raw(conn, season)
    if raw.empty:
        return pd.DataFrame(columns=["pitcher_id","plus","n_pitches"])
    g = raw.groupby("pitcher_id").agg(mean_xrv=("xrv","mean"), n=("xrv","size"))
    m, s = g["mean_xrv"].mean(), g["mean_xrv"].std()
    if s == 0:
        return pd.DataFrame(columns=["pitcher_id","plus","n_pitches"])
    g["plus"] = to_plus_scale(g["mean_xrv"], m, s)
    g = g.reset_index().rename(columns={"n":"n_pitches"})
    return g[["pitcher_id","plus","n_pitches"]]


def score_deception_plus_v1(conn, season):
    raw = dec_raw(conn, season)
    if raw.empty:
        return pd.DataFrame(columns=["pitcher_id","plus","n_pitches"])
    g = raw.groupby("pitcher_id").agg(mean_pred=("pred","mean"), n=("pred","size"))
    m, s = g["mean_pred"].mean(), g["mean_pred"].std()
    if s == 0:
        return pd.DataFrame(columns=["pitcher_id","plus","n_pitches"])
    g["plus"] = to_plus_scale(g["mean_pred"], m, s)
    g = g.reset_index().rename(columns={"n":"n_pitches"})
    return g[["pitcher_id","plus","n_pitches"]]


def score_stuff_plus_v3(conn, season):
    df = pd.read_sql(
        "SELECT pitcher_id, stuff_plus AS plus FROM pitcher_model_scores "
        "WHERE season=? AND stuff_plus IS NOT NULL",
        conn, params=(season,),
    )
    df["n_pitches"] = None
    return df


def build_all_scores(conn):
    log("Part 1a: Scoring all 4 models for 2021-2025...")
    rows = []
    for season in SEASONS:
        t0 = time.time()
        log(f"  Season {season}...")
        loc = score_location_plus_v2(conn, season).rename(columns={"plus":"loc_plus"})
        tun = score_tunneling_plus_v2(conn, season).rename(columns={"plus":"tun_plus"})
        dec = score_deception_plus_v1(conn, season).rename(columns={"plus":"dec_plus"})
        stu = score_stuff_plus_v3(conn, season).rename(columns={"plus":"stuff_plus"})

        for df in (loc, tun, dec, stu):
            df["season"] = season

        merged = (
            stu[["pitcher_id","season","stuff_plus"]]
            .merge(loc[["pitcher_id","season","loc_plus"]],   on=["pitcher_id","season"], how="outer")
            .merge(tun[["pitcher_id","season","tun_plus"]],   on=["pitcher_id","season"], how="outer")
            .merge(dec[["pitcher_id","season","dec_plus"]],   on=["pitcher_id","season"], how="outer")
        )
        rows.append(merged)
        log(f"    {season}: Stuff={len(stu)}, Loc={len(loc)}, Tun={len(tun)}, Dec={len(dec)}  ({time.time()-t0:.0f}s)")

    scores = pd.concat(rows, ignore_index=True)
    log(f"  Total pitcher-seasons: {len(scores)}")
    return scores


def build_outcomes(conn):
    log("Part 1b: Computing pitcher-season outcomes...")
    era_xfip = pd.read_sql(
        "SELECT player_id AS pitcher_id, season, era, xfip "
        "FROM player_advanced_stats "
        "WHERE era IS NOT NULL AND xfip IS NOT NULL AND season BETWEEN 2021 AND 2025",
        conn,
    )
    log(f"  ERA/xFIP rows: {len(era_xfip)}")

    log("  Aggregating pitch-level outcomes (whiff%, swstr%) from pitches...")
    pitch_rows = []
    for season in SEASONS:
        df = pd.read_sql(
            "SELECT pitcher_id, description, estimated_woba_using_speedangle "
            "FROM pitches "
            "WHERE season=? AND game_type='R' AND delta_run_exp IS NOT NULL",
            conn, params=(season,),
        )
        df["season"] = season
        df["is_swing"] = df["description"].isin(SWING_DESCS).astype(float)
        df["is_whiff"] = df["description"].isin(WHIFF_DESCS).astype(float)

        g = df.groupby(["pitcher_id","season"]).agg(
            n_pitches=("description","size"),
            n_swings=("is_swing","sum"),
            n_whiffs=("is_whiff","sum"),
            mean_xwoba=("estimated_woba_using_speedangle","mean"),
        ).reset_index()
        g["whiff_pct"] = g["n_whiffs"] / g["n_swings"].clip(lower=1)
        g["swstr_pct"] = g["n_whiffs"] / g["n_pitches"]
        pitch_rows.append(g)

    pitch_df = pd.concat(pitch_rows, ignore_index=True)
    log(f"  Pitch-level outcome rows: {len(pitch_df)}")

    outcomes = era_xfip.merge(
        pitch_df[["pitcher_id","season","n_pitches","whiff_pct","swstr_pct","mean_xwoba"]],
        on=["pitcher_id","season"], how="inner"
    )
    log(f"  Merged outcomes rows: {len(outcomes)}")
    return outcomes


# ─── Part 1 validation ──────────────────────────────────────────────────────
def compute_yoy(scores_all, model_col):
    """Pool consecutive-season pairs, return pooled r with CI."""
    all_pairs = []
    for s1, s2 in YOY_PAIRS:
        a = scores_all[scores_all.season==s1][["pitcher_id",model_col]].dropna(subset=[model_col])
        b = scores_all[scores_all.season==s2][["pitcher_id",model_col]].dropna(subset=[model_col])
        m = a.merge(b, on="pitcher_id", suffixes=("_y1","_y2"))
        if len(m) >= 30:
            all_pairs.append(m[[f"{model_col}_y1",f"{model_col}_y2"]].rename(
                columns={f"{model_col}_y1":"y1",f"{model_col}_y2":"y2"}))
    if not all_pairs:
        return float("nan"), float("nan"), float("nan"), 0
    pooled = pd.concat(all_pairs, ignore_index=True)
    return pearson_with_ci(pooled["y1"], pooled["y2"])


def compute_split_half_raw(raw_by_season, rng):
    """Split-half given dict {season: DataFrame with pitcher_id + pred}."""
    h1_scores, h2_scores = [], []
    for season, raw_df in raw_by_season.items():
        if raw_df.empty:
            continue
        pred_col = "pred"
        df = raw_df[["pitcher_id",pred_col]].copy()
        df["rnd"] = rng.random(len(df))
        for pid, grp in df.groupby("pitcher_id"):
            if len(grp) < 20:
                continue
            half = grp.sort_values("rnd")
            mid  = len(half) // 2
            h1_scores.append(half.iloc[:mid][pred_col].mean())
            h2_scores.append(half.iloc[mid:][pred_col].mean())
    if len(h1_scores) < 10:
        return float("nan"), float("nan"), float("nan"), 0
    r, lo, hi, n = pearson_with_ci(h1_scores, h2_scores)
    r_sb  = spearman_brown(r)  if not np.isnan(r)  else float("nan")
    lo_sb = spearman_brown(lo) if not np.isnan(lo) else float("nan")
    hi_sb = spearman_brown(hi) if not np.isnan(hi) else float("nan")
    return r_sb, lo_sb, hi_sb, n


def run_part1(conn, cp):
    if cp.get("part1_done"):
        results = cp["part1_results"]
        # Re-run strict OOS if any model's result contains an error
        oos = results.get("strict_oos", {})
        has_errors = any("error" in v for v in oos.values()) if oos else True
        if has_errors:
            log("Part 1: already complete — re-running strict OOS (previous run had errors).")
            results["strict_oos"] = run_strict_oos(conn, None)
            cp["part1_results"] = results
            save_checkpoint(cp)
        else:
            log("Part 1: already complete — loading checkpoint.")
        return results

    log("=" * 60)
    log("PART 1: Unified cross-model validation")
    log(f"Sign anchor: r(plus, ERA) NEGATIVE = correct (higher plus = better pitcher = lower ERA)")
    log(f"Thresholds: YoY r >= {THRESHOLDS['yoy_r_adequate']}, "
        f"ERA r <= {THRESHOLDS['era_r_adequate']}, "
        f"SB split-half >= {THRESHOLDS['split_half_adequate']}")
    log("=" * 60)

    scores  = build_all_scores(conn)
    outcomes= build_outcomes(conn)

    data = scores.merge(
        outcomes[outcomes["n_pitches"] >= MIN_PITCHES_VALIDATION],
        on=["pitcher_id","season"]
    )
    log(f"Analysis set (n_pitches >= {MIN_PITCHES_VALIDATION}): {len(data)} pitcher-seasons")

    # Verify std
    for mc in ["stuff_plus","loc_plus","tun_plus","dec_plus"]:
        sub = data.dropna(subset=[mc])
        s_per = sub.groupby("season")[mc].std()
        log(f"  {mc} std per season: " + ", ".join(f"{s}:{v:.2f}" for s,v in s_per.items()))

    models_list = [
        ("Stuff+ v3",     "stuff_plus"),
        ("Location+ v2",  "loc_plus"),
        ("Tunneling+ v2", "tun_plus"),
        ("Deception+ v1", "dec_plus"),
    ]

    results = {}
    for name, col in models_list:
        log(f"\n{name}:")
        sub = data.dropna(subset=[col])
        era_r,  era_lo,  era_hi,  era_n   = pearson_with_ci(sub[col], sub["era"])
        xfip_r, xfip_lo, xfip_hi, xfip_n  = pearson_with_ci(sub[col], sub["xfip"])
        wh_r,   wh_lo,   wh_hi,   wh_n    = pearson_with_ci(sub[col], sub["whiff_pct"])
        xw_r,   xw_lo,   xw_hi,   xw_n    = pearson_with_ci(sub[col], sub["mean_xwoba"])
        yoy_r,  yoy_lo,  yoy_hi,  yoy_n   = compute_yoy(scores, col)

        ok_era  = era_r  < THRESHOLDS["era_r_adequate"]  if not np.isnan(era_r)  else False
        ok_xfip = xfip_r < 0                              if not np.isnan(xfip_r) else False

        log(f"  ERA r    = {fmt_r(era_r, era_lo, era_hi, era_n)}  {'✓' if ok_era else '✗ SIGN FLIP'}")
        log(f"  xFIP r   = {fmt_r(xfip_r, xfip_lo, xfip_hi, xfip_n)}  {'✓' if ok_xfip else '✗ SIGN FLIP'}")
        log(f"  whiff% r = {fmt_r(wh_r, wh_lo, wh_hi, wh_n)}")
        log(f"  xwOBA r  = {fmt_r(xw_r, xw_lo, xw_hi, xw_n)}")
        log(f"  YoY r    = {fmt_r(yoy_r, yoy_lo, yoy_hi, yoy_n)}  "
            f"{'✓ adequate' if (not np.isnan(yoy_r) and yoy_r >= THRESHOLDS['yoy_r_adequate']) else 'WEAK'}")

        results[col] = {
            "name": name,
            "era":   {"r":era_r,  "lo":era_lo,  "hi":era_hi,  "n":era_n,  "sign_ok":ok_era},
            "xfip":  {"r":xfip_r, "lo":xfip_lo, "hi":xfip_hi, "n":xfip_n, "sign_ok":ok_xfip},
            "whiff": {"r":wh_r,   "lo":wh_lo,   "hi":wh_hi,   "n":wh_n},
            "xwoba": {"r":xw_r,   "lo":xw_lo,   "hi":xw_hi,   "n":xw_n},
            "yoy":   {"r":yoy_r,  "lo":yoy_lo,  "hi":yoy_hi,  "n":yoy_n},
        }

    # Split-half
    log("\nSplit-half reliability (Spearman-Brown corrected):")
    rng = np.random.default_rng(42)

    for col, raw_fn in [
        ("stuff_plus",  lambda s: _stuff_raw_season(conn, s)),
        ("loc_plus",    lambda s: loc_raw(conn, s).rename(columns={"xrv":"pred"})),
        ("tun_plus",    lambda s: tun_raw(conn, s).rename(columns={"xrv":"pred"})),
        ("dec_plus",    lambda s: dec_raw(conn, s)),
    ]:
        try:
            raw_by_s = {s: raw_fn(s) for s in SEASONS}
            r_sb, lo_sb, hi_sb, n = compute_split_half_raw(raw_by_s, rng)
            ok = r_sb >= THRESHOLDS["split_half_adequate"] if not np.isnan(r_sb) else False
            log(f"  {col}: SB r = {fmt_r(r_sb, lo_sb, hi_sb, n)}  {'✓' if ok else 'WEAK'}")
            results[col]["split_half"] = {"r":r_sb,"lo":lo_sb,"hi":hi_sb,"n":n}
        except Exception as e:
            log(f"  {col}: split-half FAILED: {e}")
            results[col]["split_half"] = {"r":None,"lo":None,"hi":None,"n":0}

    # Strict OOS
    log("\nStrict OOS: train on 2021-2024, test on 2025...")
    results["strict_oos"] = run_strict_oos(conn, scores)

    cp["part1_done"]    = True
    cp["part1_results"] = results
    save_checkpoint(cp)
    return results


def _stuff_raw_season(conn, season):
    from analytics.stuff_model import _raw_predictions as stuff_raw
    raw = stuff_raw(conn, season)
    if raw.empty:
        return pd.DataFrame(columns=["pitcher_id","pred"])
    # stuff _raw_predictions returns per-(pitcher_id, pitch_type) rows with "pred" column
    return raw[["pitcher_id","pred"]].copy()


# ─── Strict OOS 2024→2025 ───────────────────────────────────────────────────
def run_strict_oos(conn, all_scores):
    log("  Threshold: strict OOS ERA r within 25% of concurrent r = acceptable (no leakage)")
    oos = {}

    # Stuff+: live model IS trained on 2025-era data. Report live-model 2025 ERA r as a baseline.
    log("  Stuff+ v3 (live-model 2025 score, in training range):")
    t0 = time.time()
    try:
        stu_2025 = score_stuff_plus_v3(conn, 2025)
        outs = pd.read_sql(
            "SELECT player_id AS pitcher_id, era, xfip FROM player_advanced_stats "
            "WHERE season=2025 AND era IS NOT NULL AND xfip IS NOT NULL", conn
        )
        stu_2025 = stu_2025.rename(columns={"plus": "stuff_plus"})
        m = stu_2025.merge(outs, on="pitcher_id")
        er, elo, ehi, en = pearson_with_ci(m["stuff_plus"], m["era"])
        xr, *_ = pearson_with_ci(m["stuff_plus"], m["xfip"])
        log(f"    ERA r={er:.3f} [{elo:.3f},{ehi:.3f}] xFIP r={xr:.3f} (n={en})")
        oos["stuff_plus"] = {"era_r":er,"era_ci":[elo,ehi],"xfip_r":xr,"n":en,"note":"live model (2025 in training range)"}
    except Exception as e:
        log(f"    FAILED: {e}")
        oos["stuff_plus"] = {"error":str(e)}
    log(f"    ({time.time()-t0:.0f}s)")

    for model_name, model_col, retrain_fn in [
        ("Location+ v2", "loc_plus",  retrain_location_oos),
        ("Tunneling+ v2","tun_plus",  retrain_tunneling_oos),
        ("Deception+ v1","dec_plus",  retrain_deception_oos),
    ]:
        log(f"  {model_name} strict OOS (2021-2024 train → 2025 test):")
        t0 = time.time()
        try:
            result = retrain_fn(conn)
            oos[model_col] = result
            log(f"    ERA r={result.get('era_r',float('nan')):.3f}, xFIP r={result.get('xfip_r',float('nan')):.3f} (n={result.get('n','?')}) ({time.time()-t0:.0f}s)")
        except Exception as e:
            log(f"    FAILED: {e}")
            oos[model_col] = {"error":str(e)}

    return oos


def _fit_classifier(X_tr, y_tr, cat_cols):
    model = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    return model._Booster


def retrain_location_oos(conn):
    TRAIN = [2021, 2022, 2023, 2024]
    TEST  = 2025
    cat_lv = loc_cat_levels(conn)
    rv, bip_cls = loc_rv()
    hbp_rate = loc_meta().get("hbp_rate_no_swing", 0.0051)
    feat_cols = loc_meta()["feat_cols"]
    cat_cols  = loc_meta()["cat_cols"]

    # Build train set
    train_parts = []
    for s in TRAIN:
        df = pd.read_sql(
            f"SELECT pitcher_id, plate_x, plate_z, balls, strikes, pitch_type, stand, season, description "
            f"FROM pitches WHERE season={s} AND game_type='R'", conn
        )
        feat = loc_prep(df, cat_lv)   # _prep — returns feature columns only, keeps 'description' if present
        add_targets(feat)
        train_parts.append(feat)
    train_all = pd.concat(train_parts, ignore_index=True)

    # Classifier-specific row filters (same as train_v2_models.py)
    sw_mask     = train_all["is_swing"]         == 1
    no_sw_mask  = ~sw_mask
    contact_mask= sw_mask & (train_all["is_whiff"] == 0)

    boosters = {}
    for clf, row_mask, tgt in [
        ("xswing", "all",        "is_swing"),
        ("xcs",    no_sw_mask,   "is_called_strike"),
        ("xwhiff", sw_mask,      "is_whiff"),
        ("xfoul",  contact_mask, "is_foul"),
    ]:
        sub = train_all if isinstance(row_mask, str) else train_all[row_mask]
        boosters[clf] = _fit_classifier(sub[feat_cols], sub[tgt], cat_cols)
        log(f"    {clf} trained ({len(sub):,} rows)")

    # BIP multiclass — just use live booster for simplicity
    boosters["bip"] = loc_models()["bip"]

    # Score 2025
    test_df = pd.read_sql(
        f"SELECT pitcher_id, plate_x, plate_z, balls, strikes, pitch_type, stand, season, description "
        f"FROM pitches WHERE season={TEST} AND game_type='R'", conn
    )
    test_feat = loc_prep(test_df, cat_lv)
    xrv_arr = loc_combine(test_feat, boosters, rv, bip_cls, hbp_rate)
    test_feat = test_feat[["pitcher_id"]].copy()
    test_feat["xrv"] = xrv_arr

    g = test_feat.groupby("pitcher_id").agg(mean_xrv=("xrv","mean"), n=("xrv","size")).reset_index()
    g = g[g["n"] >= LOC_MIN_PITCHES]
    mean, std = g["mean_xrv"].mean(), g["mean_xrv"].std()
    g["plus"] = to_plus_scale(g["mean_xrv"], mean, std)
    outs = pd.read_sql(
        "SELECT player_id AS pitcher_id, era, xfip FROM player_advanced_stats "
        f"WHERE season={TEST} AND era IS NOT NULL AND xfip IS NOT NULL", conn
    )
    m = g.merge(outs, on="pitcher_id")
    er, elo, ehi, en = pearson_with_ci(m["plus"], m["era"])
    xr, xlo, xhi, _ = pearson_with_ci(m["plus"], m["xfip"])
    return {"era_r":er,"era_ci":[elo,ehi],"xfip_r":xr,"n":en}


def retrain_tunneling_oos(conn):
    TRAIN = [2021, 2022, 2023, 2024]
    TEST  = 2025
    cat_lv = tun_cat_levels(conn)
    rv, bip_cls = tun_rv()
    hbp_rate = tun_meta().get("hbp_rate_no_swing", 0.0051)
    cat_cols_list = [c for c in ALL_CATEGORICAL if c != "season"]  # season stays but as cat

    # Build train features (includes sequential_lag_features internally via tun_prep)
    train_parts = []
    for s in TRAIN:
        df   = tun_load_pitches(conn, s)
        feat = tun_prep(df, cat_lv)  # calls sequential_lag_features internally
        add_targets(feat)
        train_parts.append(feat)
    train_all = pd.concat(train_parts, ignore_index=True)
    del train_parts

    sw_mask     = train_all["is_swing"]       == 1
    no_sw_mask  = ~sw_mask
    contact_mask= sw_mask & (train_all["is_whiff"] == 0)

    boosters = {}
    for clf, row_mask, tgt in [
        ("xswing", "all",        "is_swing"),
        ("xcs",    no_sw_mask,   "is_called_strike"),
        ("xwhiff", sw_mask,      "is_whiff"),
        ("xfoul",  contact_mask, "is_foul"),
    ]:
        sub = train_all if isinstance(row_mask, str) else train_all[row_mask]
        boosters[clf] = _fit_classifier(sub[T_FEATURES], sub[tgt], list(ALL_CATEGORICAL))
        log(f"    {clf} trained ({len(sub):,} rows)")

    boosters["bip"] = tun_models()["bip"]
    del train_all

    test_df   = tun_load_pitches(conn, TEST)
    test_feat = tun_prep(test_df, cat_lv)
    xrv_arr   = tun_combine(test_feat, boosters, rv, bip_cls, hbp_rate)
    pid_col   = test_feat["pitcher_id"].values
    test_out  = pd.DataFrame({"pitcher_id": pid_col, "xrv": xrv_arr})

    g = test_out.groupby("pitcher_id").agg(mean_xrv=("xrv","mean"), n=("xrv","size")).reset_index()
    mean, std = g["mean_xrv"].mean(), g["mean_xrv"].std()
    g["plus"] = to_plus_scale(g["mean_xrv"], mean, std)
    outs = pd.read_sql(
        "SELECT player_id AS pitcher_id, era, xfip FROM player_advanced_stats "
        f"WHERE season={TEST} AND era IS NOT NULL AND xfip IS NOT NULL", conn
    )
    m = g.merge(outs, on="pitcher_id")
    er, elo, ehi, en = pearson_with_ci(m["plus"], m["era"])
    xr, xlo, xhi, _ = pearson_with_ci(m["plus"], m["xfip"])
    return {"era_r":er,"era_ci":[elo,ehi],"xfip_r":xr,"n":en}


def retrain_deception_oos(conn):
    TRAIN = [2021, 2022, 2023, 2024]
    TEST  = 2025
    bio   = dec_load_bio(conn)

    train_parts = []
    for s in TRAIN:
        df   = pd.read_sql(f"SELECT {', '.join(DEC_QUERY_COLS)} FROM pitches "
                           f"WHERE season={s} AND game_type='R' AND delta_run_exp IS NOT NULL", conn)
        feat = dec_eng(df, bio)
        feat["season"] = feat["season"].astype(str).astype(int)  # restore actual year as int
        train_parts.append(feat)
    train_all = pd.concat(train_parts, ignore_index=True)

    model = lgb.LGBMRegressor(
        objective="huber", n_estimators=500, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1,
    )
    model.fit(train_all[DEC_FEATURES], train_all["delta_run_exp"])
    del train_all

    test_df   = pd.read_sql(f"SELECT {', '.join(DEC_QUERY_COLS)} FROM pitches "
                            f"WHERE season={TEST} AND game_type='R' AND delta_run_exp IS NOT NULL", conn)
    test_feat = dec_eng(test_df, bio)
    test_feat["season"] = test_feat["season"].astype(str).astype(int)
    preds = model.predict(test_feat[DEC_FEATURES])
    test_out = pd.DataFrame({"pitcher_id":test_feat["pitcher_id"].values, "pred":preds})

    g = test_out.groupby("pitcher_id").agg(mean_pred=("pred","mean"), n=("pred","size")).reset_index()
    mean, std = g["mean_pred"].mean(), g["mean_pred"].std()
    g["plus"] = to_plus_scale(g["mean_pred"], mean, std)
    outs = pd.read_sql(
        "SELECT player_id AS pitcher_id, era, xfip FROM player_advanced_stats "
        f"WHERE season={TEST} AND era IS NOT NULL AND xfip IS NOT NULL", conn
    )
    m = g.merge(outs, on="pitcher_id")
    er, elo, ehi, en = pearson_with_ci(m["plus"], m["era"])
    xr, xlo, xhi, _ = pearson_with_ci(m["plus"], m["xfip"])
    return {"era_r":er,"era_ci":[elo,ehi],"xfip_r":xr,"n":en}


# ─── Part 2: Location-leakage test ──────────────────────────────────────────

# Pitch-type sequence lag columns (subset of ALL_CATEGORICAL)
S_FEATURES = [c for c in ALL_CATEGORICAL if "lag" in c and "pitch_type" in c]  # lag1..lag6_pitch_type
L_NUM  = ["plate_x","plate_z","balls","strikes"]
L_CAT  = ["pitch_type","stand","season"]

FEATURE_SETS = {
    "L":     {"num": L_NUM,                            "cat": L_CAT},
    "T":     {"num": NUMERIC_LAG_COLS,                 "cat": list(ALL_CATEGORICAL)},
    "L+G":   {"num": L_NUM + NUMERIC_LAG_COLS,         "cat": list(dict.fromkeys(L_CAT + list(ALL_CATEGORICAL)))},
    "L+S":   {"num": L_NUM,                            "cat": list(dict.fromkeys(L_CAT + S_FEATURES + ["p_throws"]))},
    "L+G+S": {"num": L_NUM + NUMERIC_LAG_COLS,         "cat": list(dict.fromkeys(L_CAT + list(ALL_CATEGORICAL)))},
    "T-G":   {"num": [],                               "cat": list(ALL_CATEGORICAL)},  # T minus geometry
}


def load_tunneling_with_location(conn, season):
    """Load Tunneling+ lag features for one season, joined with balls/strikes."""
    # Load base pitch data (includes plate_x, plate_z, description)
    df = tun_load_pitches(conn, season)

    # Add balls/strikes — join from pitches table
    bs = pd.read_sql(
        f"SELECT game_id, pitcher_id, at_bat_number, pitch_number, balls, strikes "
        f"FROM pitches WHERE season={season} AND game_type='R'", conn
    )
    df = df.merge(bs, on=["game_id","pitcher_id","at_bat_number","pitch_number"], how="left")

    # Compute lag features
    commit = _load_commit_table()
    feat = sequential_lag_features(df, commit, max_lag=TUN_MAX_LAG)
    add_targets(feat)
    return feat


def run_part2_loso(conn, cp):
    if cp.get("part2_done"):
        log("Part 2: already complete — loading checkpoint.")
        return cp.get("part2_results", {})

    log("=" * 60)
    log("PART 2: Tunneling+ location-leakage test — nested LOSO AUC")
    log(f"  Threshold: ΔAUC > {THRESHOLDS['auc_leakage_delta']:.3f} = non-trivial location contribution")
    log(f"  Threshold: geometry→location R² > {THRESHOLDS['geom_to_loc_r2']:.2f} = direct encoding")
    log("=" * 60)

    auc_table = {}
    classifiers = [
        ("xswing", "is_swing",         slice(None)),
        ("xcs",    "is_called_strike",  "no_swing"),
        ("xwhiff", "is_whiff",          "swing"),
        ("xfoul",  "is_foul",           "contact"),
    ]

    for set_name, fset in FEATURE_SETS.items():
        ck_key = f"part2_{set_name}"
        if cp.get(ck_key):
            log(f"  {set_name}: already done, skipping.")
            auc_table[set_name] = cp.get(f"part2_auc_{set_name}", {})
            continue

        all_num = fset["num"]
        all_cat = fset["cat"]
        log(f"\n  Feature set: {set_name}  ({len(all_num)} numeric, {len(all_cat)} categorical)")

        fold_aucs = {clf_name: [] for clf_name, _, _ in classifiers}

        for test_season in SEASONS:
            train_seasons = [s for s in SEASONS if s != test_season]
            log(f"    Fold: test={test_season} ...")

            # Load train data
            train_parts = []
            for s in train_seasons:
                train_parts.append(load_tunneling_with_location(conn, s))
            train_feat = pd.concat(train_parts, ignore_index=True)
            del train_parts

            test_feat = load_tunneling_with_location(conn, test_season)

            for clf_name, tgt, row_filter in classifiers:
                # Apply row filter
                if row_filter == "swing":
                    tr_mask = train_feat["is_swing"] == 1
                    te_mask = test_feat["is_swing"]  == 1
                elif row_filter == "no_swing":
                    tr_mask = train_feat["is_swing"] == 0
                    te_mask = test_feat["is_swing"]  == 0
                elif row_filter == "contact":
                    tr_mask = (train_feat["is_swing"]==1) & (train_feat["is_whiff"]==0)
                    te_mask = (test_feat["is_swing"] ==1) & (test_feat["is_whiff"] ==0)
                else:
                    tr_mask = slice(None)
                    te_mask = slice(None)

                tr = train_feat.iloc[tr_mask] if isinstance(tr_mask,slice) else train_feat[tr_mask]
                te = test_feat.iloc[te_mask]  if isinstance(te_mask,slice)  else test_feat[te_mask]

                available = [c for c in all_num + all_cat if c in tr.columns]
                cats_avail = [c for c in all_cat if c in available]

                X_tr = tr[available].copy()
                y_tr = tr[tgt].values
                X_te = te[available].copy()
                y_te = te[tgt].values

                for c in cats_avail:
                    X_tr[c] = X_tr[c].astype(str).astype("category")
                    X_te[c] = X_te[c].astype(str).astype("category")

                mdl = lgb.LGBMClassifier(
                    n_estimators=300, num_leaves=31, learning_rate=0.05,
                    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                    verbose=-1, n_jobs=-1,
                )
                mdl.fit(X_tr, y_tr, categorical_feature=cats_avail)
                pred = mdl.predict_proba(X_te)[:, 1]
                auc = auc_from_fold(y_te, pred)
                fold_aucs[clf_name].append(auc)
                log(f"      {clf_name}: AUC={auc:.4f} (n={len(y_te):,})")

            del train_feat, test_feat

        mean_aucs = {clf: float(np.mean(v)) for clf, v in fold_aucs.items() if v}
        auc_table[set_name] = mean_aucs
        log(f"    {set_name} mean AUCs: " + ", ".join(f"{k}={v:.4f}" for k,v in mean_aucs.items()))

        cp[ck_key] = True
        cp[f"part2_auc_{set_name}"] = mean_aucs
        save_checkpoint(cp)

    # Contrasts
    log("\n  ΔAUC contrasts:")
    contrasts = {}
    for clf_name, _, _ in classifiers:
        row = {s: auc_table.get(s,{}).get(clf_name, float("nan")) for s in FEATURE_SETS}
        dT_L   = row["T"]   - row["L"]
        dLG_L  = row["L+G"] - row["L"]
        dTG_T  = row["T-G"] - row["T"]
        log(f"  {clf_name}: T-L={dT_L:+.4f}  (L+G)-L={dLG_L:+.4f}  (T-G)-T={dTG_T:+.4f}")
        contrasts[clf_name] = {"T_vs_L":dT_L,"LG_vs_L":dLG_L,"TG_vs_T":dTG_T,"by_set":row}

    # Geometry → plate position R²
    log("\n  Geometry → plate position R² (Ridge, 2024 season):")
    log(f"  Threshold: R² > {THRESHOLDS['geom_to_loc_r2']:.2f} = non-trivial location encoding")
    geom_r2 = {}
    sample = load_tunneling_with_location(conn, 2024)
    from sklearn.linear_model import Ridge
    for target_loc in ["plate_x","plate_z"]:
        mask = sample[NUMERIC_LAG_COLS + [target_loc]].notna().all(axis=1)
        X = sample.loc[mask, NUMERIC_LAG_COLS].values
        y = sample.loc[mask, target_loc].values
        r2 = Ridge(alpha=1.0).fit(X, y).score(X, y)
        geom_r2[target_loc] = r2
        concern = r2 > THRESHOLDS["geom_to_loc_r2"]
        log(f"    {target_loc}: R²={r2:.4f}  {'⚠ LEAKAGE CONCERN' if concern else '✓ low'}")

    results = {
        "auc_table": auc_table,
        "contrasts": contrasts,
        "geom_to_loc_r2": geom_r2,
    }
    cp["part2_done"]    = True
    cp["part2_results"] = results
    save_checkpoint(cp)
    return results


# ─── Part 3: Deception+ identity-proxy ablation ─────────────────────────────
def run_part3_ablation(conn, cp):
    if cp.get("part3_done"):
        log("Part 3: already complete — loading checkpoint.")
        return cp.get("part3_results", {})

    log("=" * 60)
    log("PART 3: Deception+ identity-proxy ablation")
    log(f"  Threshold: |ΔCV R²| > {THRESHOLDS['ablation_cv_r2_delta']:.3f} or "
        f"|ΔYoY r| > {THRESHOLDS['ablation_yoy_delta']:.3f} = meaningful memorization")
    log("=" * 60)

    bio = dec_load_bio(conn)
    results = {}

    log("Loading 2021-2025 deception feature data...")
    raw_by_season = {}
    for season in SEASONS:
        df = pd.read_sql(
            f"SELECT {', '.join(DEC_QUERY_COLS)} FROM pitches "
            f"WHERE season={season} AND game_type='R' AND delta_run_exp IS NOT NULL", conn
        )
        feat = dec_eng(df, bio)
        feat["season"] = feat["season"].astype(str).astype(int)  # restore actual year as int
        raw_by_season[season] = feat
        log(f"  {season}: {len(feat):,} rows")

    def run_variant(vname, feature_list, randomize_cols=None, seed=None):
        ck_key = f"part3_{vname}"
        if cp.get(ck_key):
            log(f"  Variant {vname}: already done.")
            return cp.get(f"part3_result_{vname}", {})

        log(f"\n  Variant: {vname}  features={len(feature_list)}")
        rng = np.random.default_rng(seed) if seed is not None else None

        try:
            oos_preds = []
            for test_season in SEASONS:
                train_seasons = [s for s in SEASONS if s != test_season]
                train_parts   = [raw_by_season[s] for s in train_seasons]
                train_all     = pd.concat(train_parts, ignore_index=True)
                test_data     = raw_by_season[test_season]  # no copy — read only below

                if randomize_cols and rng is not None:
                    train_all = train_all.copy()  # copy once for randomization
                    for col in randomize_cols:
                        if col in train_all.columns:
                            train_all[col] = train_all[col].values[rng.permutation(len(train_all))]

                avail = [f for f in feature_list if f in train_all.columns]
                mdl = lgb.LGBMRegressor(
                    objective="huber", n_estimators=500, num_leaves=31, learning_rate=0.05,
                    min_child_samples=50, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1,
                )
                mdl.fit(train_all[avail], train_all["delta_run_exp"])

                # Randomize test data columns if needed — deduplicate column list to avoid
                # duplicate 'season' when season is already in avail (it's in DEC_FEATURES)
                if randomize_cols and rng is not None:
                    extra = [c for c in ["pitcher_id","season","delta_run_exp"] if c not in avail]
                    test_copy = test_data[avail + extra].copy()
                    for col in randomize_cols:
                        if col in test_copy.columns:
                            test_copy[col] = test_copy[col].values[rng.permutation(len(test_copy))]
                    preds  = mdl.predict(test_copy[avail])
                    fold_df= test_data[["pitcher_id","season"]].copy()
                    fold_df["pred"]   = preds
                    fold_df["actual"] = test_data["delta_run_exp"].values
                else:
                    preds   = mdl.predict(test_data[avail])
                    fold_df = test_data[["pitcher_id","season"]].copy()
                    fold_df["pred"]   = preds
                    fold_df["actual"] = test_data["delta_run_exp"].values

                oos_preds.append(fold_df)
                log(f"    Fold {test_season}: done")
        except Exception as e:
            import traceback
            log(f"  FAILED variant {vname}: {e}")
            log(traceback.format_exc())
            return {}

        all_oos = pd.concat(oos_preds, ignore_index=True)
        from sklearn.metrics import r2_score
        cv_r2 = float(r2_score(all_oos["actual"], all_oos["pred"]))

        # Plus-scale per season from LOSO predictions
        ps_rows = []
        for s in SEASONS:
            sub = all_oos[all_oos.season==s].groupby("pitcher_id")["pred"].mean().reset_index()
            sub.columns = ["pitcher_id","plus_raw"]
            if len(sub) < 2: continue
            mean, std = sub["plus_raw"].mean(), sub["plus_raw"].std()
            sub["plus"] = to_plus_scale(sub["plus_raw"], mean, std) if std > 0 else 100.0
            sub["season"] = s
            ps_rows.append(sub[["pitcher_id","season","plus"]])
        scores_df = pd.concat(ps_rows, ignore_index=True) if ps_rows else pd.DataFrame()

        yoy_r, yoy_lo, yoy_hi, yoy_n = compute_yoy(scores_df, "plus") if len(scores_df) else (float("nan"),)*4

        outs = pd.read_sql(
            "SELECT player_id AS pitcher_id, season, era, xfip FROM player_advanced_stats "
            "WHERE era IS NOT NULL AND xfip IS NOT NULL AND season BETWEEN 2021 AND 2025", conn
        )
        m = scores_df.merge(outs, on=["pitcher_id","season"]) if len(scores_df) else pd.DataFrame()
        if len(m) >= 10:
            er, elo, ehi, en = pearson_with_ci(m["plus"], m["era"])
            xr, xlo, xhi, _  = pearson_with_ci(m["plus"], m["xfip"])
        else:
            er = xr = en = float("nan"); elo=ehi=xlo=xhi=float("nan")

        log(f"    CV R²={cv_r2:.5f}, YoY r={yoy_r:.3f} [{yoy_lo:.3f},{yoy_hi:.3f}]")
        log(f"    ERA r={er:.3f}, xFIP r={xr:.3f} (n={en})")

        result = {
            "cv_r2": cv_r2,
            "yoy_r": yoy_r, "yoy_ci":[yoy_lo,yoy_hi], "yoy_n":yoy_n,
            "era_r": er,     "era_ci":[elo,ehi],
            "xfip_r":xr,     "n":en,
        }
        cp[ck_key]                   = True
        cp[f"part3_result_{vname}"] = result
        save_checkpoint(cp)
        return result

    results["V0"] = run_variant("V0", DEC_FEATURES)
    v1_feat = [f for f in DEC_FEATURES if f not in ("height_in","weight_lb")]
    results["V1"] = run_variant("V1", v1_feat)
    for seed in [0, 1, 2]:
        results[f"V-rand-{seed}"] = run_variant(f"V-rand-{seed}", DEC_FEATURES, ["height_in","weight_lb"], seed)

    # Compute deltas
    v0, v1 = results.get("V0",{}), results.get("V1",{})
    dcvr2 = abs((v1.get("cv_r2",0) or 0) - (v0.get("cv_r2",0) or 0))
    dyoy  = abs((v1.get("yoy_r",0) or 0) - (v0.get("yoy_r",0) or 0))
    log(f"\n  V1 vs V0: |ΔCV R²|={dcvr2:.5f} ({'CONCERN' if dcvr2>THRESHOLDS['ablation_cv_r2_delta'] else 'OK'}), "
        f"|ΔYoY r|={dyoy:.3f} ({'CONCERN' if dyoy>THRESHOLDS['ablation_yoy_delta'] else 'OK'})")
    results["deltas"] = {"V1_vs_V0":{"cv_r2":dcvr2,"yoy_r":dyoy}}

    cp["part3_done"]    = True
    cp["part3_results"] = results
    save_checkpoint(cp)
    return results


# ─── HTML report generation ─────────────────────────────────────────────────
def _rc(r, lo=None, hi=None, neg_good=False, pos_good=False):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "<td>—</td>"
    cls = ""
    if neg_good:
        cls = "good" if r < -0.05 else ("bad" if r > 0.05 else "")
    elif pos_good:
        cls = "good" if r > 0.35 else ("bad" if r < 0.10 else "")
    val = f"{r:+.3f}"
    ci  = f" [{lo:+.3f},{hi:+.3f}]" if lo is not None else ""
    return f'<td class="{cls}">{val}{ci}</td>'


def generate_html(p1, p2, p3):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    clf_names = ["xswing","xcs","xwhiff","xfoul"]

    # Part 1 model rows
    m_rows = []
    for col in ["stuff_plus","loc_plus","tun_plus","dec_plus"]:
        info = p1.get(col,{})
        name = info.get("name", col)
        er   = info.get("era",{})
        xr   = info.get("xfip",{})
        yr   = info.get("yoy",{})
        sh   = info.get("split_half",{})
        m_rows.append(
            f"<tr><td><strong>{name}</strong></td>"
            + _rc(er.get("r"),  er.get("lo"),  er.get("hi"),  neg_good=True)
            + _rc(xr.get("r"),  xr.get("lo"),  xr.get("hi"),  neg_good=True)
            + _rc(yr.get("r"),  yr.get("lo"),  yr.get("hi"),  pos_good=True)
            + _rc(sh.get("r"),  sh.get("lo"),  sh.get("hi"),  pos_good=True)
            + f"<td>{er.get('n','?')}</td></tr>"
        )

    # OOS rows
    oos_rows = []
    for col, name in [("stuff_plus","Stuff+ v3"),("loc_plus","Location+ v2"),("tun_plus","Tunneling+ v2"),("dec_plus","Deception+ v1")]:
        d = (p1.get("strict_oos") or {}).get(col,{})
        if "error" in d:
            oos_rows.append(f"<tr><td>{name}</td><td colspan=3>Error: {str(d['error'])[:60]}</td></tr>")
        else:
            er   = d.get("era_r", float("nan"))
            xr   = d.get("xfip_r", float("nan"))
            ci   = d.get("era_ci",[float("nan"),float("nan")])
            note = d.get("note","")
            oos_rows.append(
                f"<tr><td>{name}</td>"
                + _rc(er, ci[0], ci[1], neg_good=True)
                + _rc(xr, neg_good=True)
                + f"<td>{d.get('n','?')}</td>"
                + f"<td style='font-size:10px;color:#888'>{note}</td></tr>"
            )

    # Part 2 AUC table
    p2_auc = p2.get("auc_table",{})
    auc_hdr = "<tr><th>Feature Set</th>" + "".join(f"<th>{c}</th>" for c in clf_names) + "</tr>"
    auc_rows = []
    for fset in ["L","T","L+G","L+S","L+G+S","T-G"]:
        row = f"<tr><td><strong>{fset}</strong></td>"
        for clf in clf_names:
            v = p2_auc.get(fset,{}).get(clf, float("nan"))
            row += f"<td>{v:.4f}</td>" if not (isinstance(v,float) and np.isnan(v)) else "<td>—</td>"
        row += "</tr>"
        auc_rows.append(row)

    # Part 3 ablation
    p3_rows = []
    for vname in ["V0","V1","V-rand-0","V-rand-1","V-rand-2"]:
        d   = p3.get(vname,{})
        cv  = d.get("cv_r2", float("nan"))
        yr  = d.get("yoy_r", float("nan"))
        er  = d.get("era_r", float("nan"))
        xr  = d.get("xfip_r",float("nan"))
        p3_rows.append(
            f"<tr><td><strong>{vname}</strong></td>"
            f"<td>{cv:.5f}</td>"
            + _rc(yr, pos_good=True)
            + _rc(er, neg_good=True)
            + _rc(xr, neg_good=True)
            + "</tr>"
        )

    gx = (p2.get("geom_to_loc_r2") or {}).get("plate_x", float("nan"))
    gz = (p2.get("geom_to_loc_r2") or {}).get("plate_z", float("nan"))

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
  body{{font-family:Georgia,serif;font-size:13px;line-height:1.6;background:var(--bg);color:var(--fg);padding:40px 48px;max-width:1100px;margin:0 auto}}
  header{{border-bottom:3px solid var(--or);padding-bottom:16px;margin-bottom:28px}}
  .model-name{{font-size:26px;font-weight:bold;color:var(--or)}}
  .model-sub{{font-family:monospace;font-size:12px;color:var(--mu);margin-top:4px}}
  .badges{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
  .badge{{padding:3px 10px;border-radius:4px;font-family:monospace;font-size:11px;font-weight:bold}}
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
  code{{font-family:monospace;font-size:12px;background:var(--orl);padding:1px 4px;border-radius:3px}}
  .callout{{background:var(--orl);border-left:4px solid #c89a00;padding:10px 14px;margin:12px 0;font-size:12px}}
  .section{{margin-bottom:28px}}
  footer{{margin-top:40px;padding-top:12px;border-top:1px solid var(--bd);font-size:11px;color:var(--mu);font-family:monospace}}
</style>
</head>
<body>
<header>
  <div class="model-name">Cross-Model Validation &amp; Leakage Tests</div>
  <div class="model-sub">training/cross_model_validation.py · Generated {now}</div>
  <div class="badges">
    <span class="badge badge-warn">⚠ IN PROGRESS — not final</span>
    <span class="badge badge-warn">Live models untouched — R&amp;D only</span>
  </div>
</header>

<div class="callout"><strong>Sign convention:</strong>
r(plus_score, ERA) is <strong>NEGATIVE</strong> for correctly-signed models (higher score = better pitcher = lower ERA).
Green cells = negative ERA r / positive YoY r. The methodology docs' "+0.208"/"+0.182" ERA r values
used an inverted convention — values here supersede those.
</div>

<div class="section">
<h2>Part 1 — Unified Validation (2021–2025, min {MIN_PITCHES_VALIDATION} pitches)</h2>
<h3>1.1 Concurrent &amp; Predictive Validity</h3>
<table>
  <thead><tr><th>Model</th><th>ERA r [CI]</th><th>xFIP r [CI]</th><th>YoY r [CI]</th><th>Split-half SB [CI]</th><th>n</th></tr></thead>
  <tbody>{"".join(m_rows)}</tbody>
</table>
<h3>1.2 Strict OOS: train 2021–2024, test 2025</h3>
<table>
  <thead><tr><th>Model</th><th>ERA r [CI]</th><th>xFIP r</th><th>n</th><th>Note</th></tr></thead>
  <tbody>{"".join(oos_rows)}</tbody>
</table>
</div>

<div class="section">
<h2>Part 2 — Tunneling+ Location-Leakage Test</h2>
<p>LOSO 5-fold AUC for 4 classifiers across 6 nested feature sets (2021–2025).<br>
L = location (7), T = tunneling (28), G = geometry lags (18), S = pt-type seq lags (6), T-G = T minus geometry.</p>
<h3>2.1 LOSO Mean AUC</h3>
<table><thead>{auc_hdr}</thead><tbody>{"".join(auc_rows)}</tbody></table>
<h3>2.2 Geometry → Plate Position R²</h3>
<p>Ridge regression: {len(NUMERIC_LAG_COLS)} geometry lag features → plate_x or plate_z (2024 season).</p>
<p>plate_x R² = {gx:.4f} {'⚠ LEAKAGE' if gx>(THRESHOLDS['geom_to_loc_r2']) else '✓ low'} &nbsp;|&nbsp; plate_z R² = {gz:.4f} {'⚠ LEAKAGE' if gz>(THRESHOLDS['geom_to_loc_r2']) else '✓ low'}</p>
<p>Threshold: &gt;{THRESHOLDS['geom_to_loc_r2']:.2f} = non-trivial encoding. Threshold for AUC delta: &gt;{THRESHOLDS['auc_leakage_delta']:.3f}.</p>
</div>

<div class="section">
<h2>Part 3 — Deception+ Identity-Proxy Ablation</h2>
<p>V0 = full features, V1 = drop height/weight, V-rand = randomize height/weight (seeds 0/1/2).<br>
Huber LightGBM LOSO 5-fold on 2021–2025. |ΔCV R²| &gt;{THRESHOLDS['ablation_cv_r2_delta']:.3f} or |ΔYoY r| &gt;{THRESHOLDS['ablation_yoy_delta']:.3f} = meaningful memorization.</p>
<table>
  <thead><tr><th>Variant</th><th>CV R²</th><th>YoY r</th><th>ERA r</th><th>xFIP r</th></tr></thead>
  <tbody>{"".join(p3_rows)}</tbody>
</table>
</div>

<footer>Cross-Model Validation · {now} · training/cross_model_validation.py · All live models untouched</footer>
</body></html>"""


# ─── HANDOFF update ──────────────────────────────────────────────────────────
def update_handoff(p1, p2, p3):
    handoff_path = ROOT / "Methodology" / "HANDOFF.md"
    if not handoff_path.exists():
        return
    entry = f"""
---
## Cross-Model Validation ({datetime.now().strftime('%Y-%m-%d')})

**Sign convention:** r(plus, ERA) NEGATIVE = correctly-signed (higher plus = lower ERA).
Earlier methodology docs reported positive ERA r for v2 models — inverted convention; superseded here.

### Part 1 Summary (n_pitches ≥ {MIN_PITCHES_VALIDATION}, 2021–2025)
| Model | ERA r | xFIP r | YoY r | Split-half SB |
|---|---|---|---|---|
"""
    for col in ["stuff_plus","loc_plus","tun_plus","dec_plus"]:
        info = p1.get(col, {})
        entry += (
            f"| {info.get('name',col)} "
            f"| {info.get('era',{}).get('r',float('nan')):+.3f} "
            f"| {info.get('xfip',{}).get('r',float('nan')):+.3f} "
            f"| {info.get('yoy',{}).get('r',float('nan')):+.3f} "
            f"| {info.get('split_half',{}).get('r',float('nan')):+.3f} |\n"
        )

    gx = (p2.get("geom_to_loc_r2") or {}).get("plate_x", float("nan"))
    gz = (p2.get("geom_to_loc_r2") or {}).get("plate_z", float("nan"))
    v0 = p3.get("V0",{})
    v1 = p3.get("V1",{})
    deltas = p3.get("deltas",{}).get("V1_vs_V0",{})
    entry += f"""
### Part 2 Summary
Geometry → plate_x R²={gx:.4f}, plate_z R²={gz:.4f}
Full AUC table in: Methodology/in_progress/CROSS_MODEL_VALIDATION_AND_LEAKAGE_TESTS_IN_PROGRESS.html

### Part 3 Summary
V0 CV R²={v0.get('cv_r2',float('nan')):.5f}, YoY r={v0.get('yoy_r',float('nan')):+.3f}
V1 (no h/w) CV R²={v1.get('cv_r2',float('nan')):.5f}, YoY r={v1.get('yoy_r',float('nan')):+.3f}
|ΔCV R²|={deltas.get('cv_r2',float('nan')):.5f}  |ΔYoY r|={deltas.get('yoy_r',float('nan')):.3f}
"""
    content = handoff_path.read_text(encoding="utf-8")
    import re
    if "## Cross-Model Validation" not in content:
        content += entry
    else:
        content = re.sub(r"\n---\n## Cross-Model Validation.*",
                         "\n" + entry.strip(), content, flags=re.DOTALL)
    handoff_path.write_text(content, encoding="utf-8")
    log(f"HANDOFF.md updated.")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log(f"Cross-model validation started: {datetime.now()}")
    log(f"DB:     {DB_PATH}")
    log(f"Output: {HTML_OUTPUT}")
    log("=" * 60)

    cp   = load_checkpoint()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        p1 = run_part1(conn, cp)
        p2 = run_part2_loso(conn, cp)
        p3 = run_part3_ablation(conn, cp)
    finally:
        conn.close()

    log("\nGenerating HTML report...")
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT.write_text(generate_html(p1, p2, p3), encoding="utf-8")
    log(f"Report → {HTML_OUTPUT}")

    update_handoff(p1, p2, p3)
    log("\nDone.")


if __name__ == "__main__":
    main()
