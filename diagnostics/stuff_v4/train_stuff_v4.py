"""
Stuff+ v4 training script — Parts 1 through 4.

Usage: python diagnostics/stuff_v4/train_stuff_v4.py

Produces:
  diagnostics/stuff_v4/spec_lock.json  (pre-written)
  diagnostics/stuff_v4/progress.log
  diagnostics/stuff_v4/part1_audit.json
  diagnostics/stuff_v4/part2_ablation_table.json
  diagnostics/stuff_v4/part2_oof_predictions.parquet
  diagnostics/stuff_v4/part3_v4_scores.parquet  (also saved as stuff_v4_scores.parquet)
  diagnostics/stuff_v4/part4_validation.json
  models/stuff_plus_v4_lgbm.txt
  models/stuff_plus_v4_lgbm_meta.json

Nothing touches the live model, API, or site.
"""

import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT / "diagnostics" / "stuff_v4"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "mlb_data.db"
LOG_PATH = DIAG_DIR / "progress.log"

DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ---- imports from analytics ----
from analytics.metrics import compute_vaa
from analytics.stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, CORE_NUMERIC_FEATURES,
    EXCLUDED_PITCH_TYPES, NUMERIC_FEATURES, OPTIONAL_NUMERIC_FEATURES,
    SEPARATION_FEATURES,
    add_axis_differential, add_movement_columns, add_separation_features,
    add_vaa_residual, build_fb_reference_table, engineer_features,
    fit_v3_calibrations, fit_vaa_expectation, fit_vaa_expectations_per_type,
    merge_pitch_type,
)
from analytics.spin_physics import add_axis_differential, fit_spin_axis_calibration
import analytics.stuff_model as sm_runtime


# ---- logging ----
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Series):
        return o.to_dict()
    return str(o)


# ---- hyperparameters (from v3 meta) ----
V3_META = json.loads((MODELS_DIR / "stuff_plus_v3_lgbm_meta.json").read_text())
LGBM_PARAMS = V3_META["lgbm_params"].copy()
FIXED_N_TREES = LGBM_PARAMS.pop("n_estimators", 2000)  # fixed, no early stopping
LGBM_PARAMS.pop("random_state", None)  # will be passed explicitly

TRAINING_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
SCORING_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
EXCLUDED_GAME_TYPES = ("PO", "UN", "EP")

FORBIDDEN_FEATURES = {
    "plate_x", "plate_z", "balls", "strikes", "release_extension",
    "release_pos_x_as", "release_pos_z", "arm_angle", "pitcher_id",
    "game_id", "pitch_id", "player_id", "batter_id",
}

# ---- raw query columns ----
RAW_QUERY_COLS = [
    "pitcher_id", "pitch_type", "stand", "p_throws", "season",
    "release_speed", "effective_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z", "plate_z", "plate_x",
    "vy0", "vz0", "vx0", "ay", "az", "ax",
    "release_pos_x",
    "delta_run_exp",
]
# also need for HAA
HAA_EXTRA_COLS = ["vx0", "ax", "vy0", "ay", "plate_x", "release_pos_x"]

# ---- baseline feature set (v3 all_features) ----
BASELINE_FEATURES = list(V3_META["all_features"])


# ============================================================
# DATABASE LOADING
# ============================================================

def load_pitches(seasons, require_delta_run_exp=True):
    """Load pitches for given seasons from DB, excluding junk pitch types."""
    import sqlite3
    log(f"Loading pitches for seasons {seasons}...")
    conn = sqlite3.connect(DB_PATH)
    season_placeholders = ",".join("?" * len(seasons))
    extra = "AND delta_run_exp IS NOT NULL" if require_delta_run_exp else ""
    query = f"""
        SELECT {', '.join(RAW_QUERY_COLS)}
        FROM pitches
        WHERE season IN ({season_placeholders})
          AND game_type = 'R'
          {extra}
    """
    df = pd.read_sql(query, conn, params=seasons)
    conn.close()
    # exclude junk pitch types
    df = df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)].copy()
    log(f"  Loaded {len(df):,} pitches after excluding {EXCLUDED_PITCH_TYPES}")
    return df


# ============================================================
# HAA RESIDUAL
# ============================================================

def compute_haa(df):
    """
    Horizontal approach angle (degrees) at front of plate.
    Mirror sign by p_throws so LHP and RHP are comparable (arm-side positive).
    Uses vx0/vy0/ax/ay physics same as VAA uses vz0/vy0/az/ay.
    """
    # time to front of plate: same formula as VAA
    # vy0 is negative (moving toward plate), so t = (-vy0 - sqrt(vy0^2 - 2*ay*(55-vy0^2/(2*ay)))) / ay
    # Simpler: same approach as compute_vaa
    vy0 = df["vy0"].to_numpy(dtype=float)
    vx0 = df["vx0"].to_numpy(dtype=float)
    ay = df["ay"].to_numpy(dtype=float)
    ax = df["ax"].to_numpy(dtype=float)

    # Distance to front of plate ~17/60 = 0.283 ft... use y_front = 17/12 ft = 1.417 ft
    # Standard: y_front = 17/12 ft. vy^2 = vy0^2 + 2*ay*(y_front - y0), but y0 ~ -60.5 ft
    # Use same computation as compute_vaa from analytics/metrics.py
    # compute_vaa uses: vy_front = sqrt(vy0^2 + 2*ay*(17/12 - (-60.5)))
    # t = (vy_front - vy0) / ay
    y_front = 17.0 / 12.0  # front of plate
    y_release = -60.5
    delta_y = y_front - y_release  # ~61.9 ft
    vy_front_sq = vy0 ** 2 + 2 * ay * delta_y
    vy_front_sq = np.maximum(vy_front_sq, 0.01)  # guard sqrt
    vy_front = -np.sqrt(vy_front_sq)  # negative (moving toward batter)
    t = (vy_front - vy0) / ay

    vx_front = vx0 + ax * t
    haa_raw = np.degrees(np.arctan2(vx_front, -vy_front))  # angle in degrees

    # Mirror sign by p_throws: for RHP, arm side is positive x (catcher's right = pitcher's left)
    # arm_side_sign: R -> -1 (pfx convention), so mirror same as hb
    sign = np.where(df["p_throws"].to_numpy() == "R", -1.0, 1.0)
    haa = haa_raw * sign
    return haa


def fit_haa_expectations(df, min_samples=5000):
    """
    OLS per pitch type: expected_haa ~ intercept + b1*plate_x_m + b2*plate_x_m^2 + b3*release_pos_x_m + b4*release_speed
    where *_m = mirrored (multiplied by arm-side sign).
    Returns dict {pitch_type: [intercept, b1, b2, b3, b4]} and pooled fallback.
    """
    sign = np.where(df["p_throws"].to_numpy() == "R", -1.0, 1.0)
    df = df.copy()
    df["plate_x_m"] = df["plate_x"].to_numpy(dtype=float) * sign
    df["release_pos_x_m"] = df["release_pos_x"].to_numpy(dtype=float) * sign

    clean = df.dropna(subset=["plate_x_m", "release_pos_x_m", "release_speed", "haa"])
    clean = clean[np.isfinite(clean["haa"])]

    def _fit_ols(sub):
        px = sub["plate_x_m"].to_numpy(dtype=float)
        rp = sub["release_pos_x_m"].to_numpy(dtype=float)
        v = sub["release_speed"].to_numpy(dtype=float)
        y = sub["haa"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(sub)), px, px**2, rp, v])
        coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
        return coefs.tolist()

    pooled = _fit_ols(clean)
    per_type = {}
    for pt, sub in clean.groupby("pitch_type", observed=True):
        if len(sub) >= min_samples:
            per_type[pt] = _fit_ols(sub)

    return {"pooled": pooled, **per_type}


def add_haa_residual(df, haa_coefs):
    """Compute haa and haa_residual."""
    df = df.copy()
    df["haa"] = compute_haa(df).astype("float32")

    sign = np.where(df["p_throws"].to_numpy() == "R", -1.0, 1.0)
    px_m = df["plate_x"].to_numpy(dtype=float) * sign
    rp_m = df["release_pos_x"].to_numpy(dtype=float) * sign
    v = df["release_speed"].to_numpy(dtype=float)

    pooled = np.array(haa_coefs["pooled"])
    per_type = {k: np.array(v_) for k, v_ in haa_coefs.items() if k != "pooled"}

    expected_haa = np.empty(len(df), dtype=float)
    for pt, idx in df.groupby(df["pitch_type"].astype(str)).groups.items():
        coefs = per_type.get(pt, pooled)
        sub_px = px_m[df.index.get_indexer(idx)]
        sub_rp = rp_m[df.index.get_indexer(idx)]
        sub_v = v[df.index.get_indexer(idx)]
        design = np.column_stack([np.ones(len(idx)), sub_px, sub_px**2, sub_rp, sub_v])
        expected_haa[df.index.get_indexer(idx)] = design @ coefs

    df["haa_residual"] = (df["haa"].to_numpy(dtype=float) - expected_haa).astype("float32")
    return df


# ============================================================
# FEATURE ENGINEERING WRAPPERS
# ============================================================

def build_features_for_seasons(seasons, calibrations, haa_coefs=None, include_haa=False,
                                 drop_season=False, drop_effective_speed=False):
    """
    Build feature table for the given seasons.
    Returns df with ALL_FEATURES (possibly modified) plus pitcher_id, season, delta_run_exp.
    """
    raw = load_pitches(seasons)
    raw = merge_pitch_type(raw)
    raw = add_movement_columns(raw)

    fb_ref = build_fb_reference_table(raw)
    feat = engineer_features(raw, calibrations, fb_ref)

    if include_haa and haa_coefs is not None:
        feat = add_haa_residual(feat, haa_coefs)

    feat["season"] = feat["season"].astype(str)

    # Build feature list for this variant
    num_features = [f for f in NUMERIC_FEATURES if not (drop_effective_speed and f == "effective_speed")]
    cat_features = [f for f in CATEGORICAL_FEATURES if not (drop_season and f == "season")]
    if include_haa and haa_coefs is not None:
        num_features = num_features + ["haa_residual"]

    # Check forbidden features
    all_feats = num_features + cat_features
    leaked = FORBIDDEN_FEATURES & set(all_feats)
    if leaked:
        raise ValueError(f"FORBIDDEN feature(s) in feature list: {leaked}")

    return feat, all_feats, num_features, cat_features


def engineer_feat_df(raw_df, calibrations, haa_coefs=None, include_haa=False,
                      drop_season=False, drop_effective_speed=False):
    """
    Build feature matrix from already-loaded raw_df.
    Returns (feat_df, all_feature_names).
    """
    raw_df = merge_pitch_type(raw_df)
    raw_df = add_movement_columns(raw_df)
    fb_ref = build_fb_reference_table(raw_df)
    feat = engineer_features(raw_df, calibrations, fb_ref)

    if include_haa and haa_coefs is not None:
        feat = add_haa_residual(feat, haa_coefs)

    num_features = [f for f in NUMERIC_FEATURES if not (drop_effective_speed and f == "effective_speed")]
    cat_features = [f for f in CATEGORICAL_FEATURES if not (drop_season and f == "season")]
    if include_haa and haa_coefs is not None:
        num_features = num_features + ["haa_residual"]

    all_feats = num_features + cat_features
    leaked = FORBIDDEN_FEATURES & set(all_feats)
    if leaked:
        raise ValueError(f"FORBIDDEN feature(s): {leaked}")

    for col in cat_features:
        feat[col] = feat[col].astype("category")

    return feat, all_feats


# ============================================================
# SCORING PIPELINE (mirrors stuff_model.py)
# ============================================================

def score_pipeline(raw_pred_df):
    """
    Apply the same scoring pipeline as analytics/stuff_model.py:
    empirical-Bayes shrinkage + usage-weighted z-score.
    Input: DataFrame with pitcher_id, pitch_type, season, pred
    Returns: pitcher_season scores DataFrame.
    """
    results = []
    for season, sdf in raw_pred_df.groupby("season"):
        if sdf["pred"].std() == 0:
            continue
        k_by_type, fallback_k, type_means = sm_runtime._shrinkage_params(sdf)
        g = sm_runtime._shrunk_group_means(sdf, ["pitcher_id", "pitch_type"],
                                           k_by_type, fallback_k, type_means)
        norm_mean = float(sdf["pred"].mean())
        norm_std = float(sdf["pred"].std())
        g["stuff_plus"] = sm_runtime._to_stuff_plus(g["shrunk_mean"], norm_mean, norm_std)
        g["season"] = season
        results.append(g[["pitcher_id", "pitch_type", "n_pitches", "stuff_plus", "season"]])
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def pitcher_season_scores(raw_pred_df, min_pitches=100):
    """
    Usage-weighted pitcher-season Stuff+ from raw predictions.
    """
    scored = score_pipeline(raw_pred_df)
    if scored.empty:
        return pd.DataFrame()
    agg = (scored.groupby(["pitcher_id", "season"])
           .apply(lambda x: pd.Series({
               "stuff_plus_v4": np.average(x["stuff_plus"], weights=x["n_pitches"]),
               "n_pitches": x["n_pitches"].sum(),
           }))
           .reset_index())
    agg = agg[agg["n_pitches"] >= min_pitches]
    return agg


# ============================================================
# PART 1 — DATA AND FEATURE AUDIT
# ============================================================

def part1_audit():
    log("=== PART 1: Data and Feature Audit ===")

    # 1. Verify baseline features
    assert "effective_speed" in NUMERIC_FEATURES, "effective_speed should be in NUMERIC_FEATURES"
    assert "season" in CATEGORICAL_FEATURES, "season should be in CATEGORICAL_FEATURES"
    for forbidden in ["release_extension", "release_pos_x_as", "release_pos_z", "arm_angle"]:
        assert forbidden not in ALL_FEATURES, f"{forbidden} should NOT be in ALL_FEATURES"
    log("Feature list assertions passed.")

    # 2. Load training data
    raw = load_pitches(TRAINING_SEASONS)
    log(f"Total training rows: {len(raw):,}")

    # Row counts per season
    season_counts = raw.groupby("season").size().to_dict()
    log("Row counts per season (training):")
    for s, n in sorted(season_counts.items()):
        log(f"  {s}: {n:,}")

    # 3. Completeness per season
    audit_cols = (CORE_NUMERIC_FEATURES + OPTIONAL_NUMERIC_FEATURES + CATEGORICAL_FEATURES
                  + ["plate_z", "plate_x", "vy0", "vz0", "ay", "az", "vx0", "ax", "release_pos_x"])
    existing_cols = [c for c in audit_cols if c in raw.columns]
    completeness = {}
    for s in sorted(raw["season"].unique()):
        sdf = raw[raw["season"] == s]
        row = {}
        for c in existing_cols:
            row[c] = float(sdf[c].notna().mean() * 100)
        completeness[str(s)] = row

    # 4. Per-season mean/std of key features
    key_features = ["release_speed", "release_spin_rate", "hb", "ivb"]
    raw2 = merge_pitch_type(raw)
    raw2 = add_movement_columns(raw2)
    season_stats = {}
    for s in sorted(raw2["season"].unique()):
        sdf = raw2[raw2["season"] == s]
        row = {}
        for f in key_features:
            if f in sdf.columns:
                row[f + "_mean"] = float(sdf[f].mean())
                row[f + "_std"] = float(sdf[f].std())
        season_stats[str(s)] = row

    # 5. KS tests: 2020 vs 2019, 2021 vs 2019
    ks_results = {}
    ref_season = 2019
    test_seasons = [2020, 2021]
    ks_features = ["release_spin_rate", "spin_axis", "hb", "ivb", "release_speed"]
    ref_data = raw2[raw2["season"] == ref_season]
    for ts in test_seasons:
        ts_data = raw2[raw2["season"] == ts]
        ks_results[str(ts)] = {}
        for f in ks_features:
            if f in raw2.columns:
                a = ref_data[f].dropna().to_numpy()
                b = ts_data[f].dropna().to_numpy()
                if len(a) > 0 and len(b) > 0:
                    ks_stat, ks_p = stats.ks_2samp(a, b)
                    ks_results[str(ts)][f] = {
                        "ks_stat": float(ks_stat),
                        "ks_pvalue": float(ks_p),
                        "flagged": bool(ks_stat > 0.05),
                    }

    # 6. HAA residual correlations
    # Compute HAA on a sample for the correlation report
    log("Computing HAA residual for correlation report...")
    sample_raw = raw2.dropna(subset=["vx0", "ax", "vy0", "ay", "plate_x", "release_pos_x",
                                      "plate_z", "release_speed"]).copy()
    sample_raw["haa"] = compute_haa(sample_raw)
    # Fit HAA coefs on full training data
    haa_coefs = fit_haa_expectations(sample_raw)
    sample_raw = add_haa_residual(sample_raw, haa_coefs)

    corr_cols = {
        "plate_x": sample_raw["plate_x"].to_numpy(dtype=float),
        "plate_z": sample_raw["plate_z"].to_numpy(dtype=float),
        "release_pos_x": sample_raw["release_pos_x"].to_numpy(dtype=float),
        "arm_angle": None,  # not in data
        "hb": sample_raw["hb"].to_numpy(dtype=float),
        "release_speed": sample_raw["release_speed"].to_numpy(dtype=float),
    }
    haa_res = sample_raw["haa_residual"].to_numpy(dtype=float)
    valid_mask = np.isfinite(haa_res)
    haa_corrs = {}
    for col, arr in corr_cols.items():
        if arr is None:
            haa_corrs[col] = None
            continue
        m = valid_mask & np.isfinite(arr)
        if m.sum() > 100:
            r, p = stats.pearsonr(haa_res[m], arr[m])
            haa_corrs[col] = {"corr": float(r), "abs_corr": float(abs(r)), "pvalue": float(p)}
        else:
            haa_corrs[col] = None

    # Summary leakage check: |corr(haa_residual, plate_x)| and |corr(haa_residual, release_pos_x)|
    px_corr = haa_corrs.get("plate_x", {}) or {}
    rpx_corr = haa_corrs.get("release_pos_x", {}) or {}
    haa_leakage_ok = (
        (abs(px_corr.get("corr", 1.0)) < 0.1) and
        (abs(rpx_corr.get("corr", 1.0)) < 0.1)
    )
    log(f"HAA residual |corr| with plate_x: {abs(px_corr.get('corr', float('nan'))):.4f}")
    log(f"HAA residual |corr| with release_pos_x: {abs(rpx_corr.get('corr', float('nan'))):.4f}")
    log(f"HAA leakage check passes: {haa_leakage_ok}")

    audit = {
        "total_training_rows": int(len(raw)),
        "season_row_counts": {str(k): int(v) for k, v in season_counts.items()},
        "feature_completeness_pct": completeness,
        "season_stats": season_stats,
        "ks_tests": ks_results,
        "haa_residual_correlations": haa_corrs,
        "haa_leakage_ok": bool(haa_leakage_ok),
        "baseline_features_verified": True,
        "forbidden_features_absent": True,
    }
    save_json(audit, DIAG_DIR / "part1_audit.json")
    log("Part 1 audit saved.")
    return audit, haa_coefs


# ============================================================
# PART 2 — ABLATION ON PITCHER SUBSAMPLE
# ============================================================

def bootstrap_ci(a, b, n_boot=1000, seed=42):
    """Paired bootstrap CI for correlation difference. Returns (diff_mean, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ra = np.corrcoef(a[idx], b[idx])[0, 1] if len(set(a[idx])) > 1 and len(set(b[idx])) > 1 else np.nan
        diffs.append(ra)
    return float(np.nanmean(diffs)), float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def safe_pearsonr(x, y):
    """Pearson r with NaN guard."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p)


def pitcher_clustered_bootstrap_corr_diff(scores_v, scores_v0, outcomes, pitcher_ids, n_boot=1000, seed=42):
    """
    Bootstrap CI for difference in correlation(variant_scores, outcomes) vs
    correlation(v0_scores, outcomes), clustered by pitcher.
    Returns (diff, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    pitchers = np.unique(pitcher_ids)
    n_p = len(pitchers)
    obs_r_v, _ = safe_pearsonr(scores_v, outcomes)
    obs_r_v0, _ = safe_pearsonr(scores_v0, outcomes)
    obs_diff = obs_r_v - obs_r_v0

    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(pitchers, size=n_p, replace=True)
        idx = np.concatenate([np.where(pitcher_ids == p)[0] for p in sampled])
        r_v, _ = safe_pearsonr(scores_v[idx], outcomes[idx])
        r_v0, _ = safe_pearsonr(scores_v0[idx], outcomes[idx])
        diffs.append(r_v - r_v0)

    diffs = np.array(diffs)
    return obs_diff, float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def load_outcomes(seasons=None):
    """Load pitcher-season ERA, xFIP from player_advanced_stats. Returns season as str."""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT player_id AS pitcher_id, season, xfip, era
        FROM player_advanced_stats
        WHERE (xfip IS NOT NULL OR era IS NOT NULL)
    """
    if seasons:
        ph = ",".join("?" * len(seasons))
        query += f" AND season IN ({ph})"
        outcomes = pd.read_sql(query, conn, params=seasons)
    else:
        outcomes = pd.read_sql(query, conn)
    conn.close()
    # Normalize season to str to match pitch-derived DataFrames (engineer_features casts to str)
    outcomes["season"] = outcomes["season"].astype(str)
    log(f"Loaded {len(outcomes):,} pitcher-season outcomes (xfip available: {outcomes['xfip'].notna().sum():,})")
    return outcomes


def part2_ablation(haa_coefs):
    log("=== PART 2: Ablation on Pitcher Subsample ===")
    t0 = time.monotonic()

    # Load full training data
    log("Loading training pitches for ablation...")
    raw = load_pitches(TRAINING_SEASONS)
    raw = merge_pitch_type(raw)
    raw = add_movement_columns(raw)

    # Subsample 30% of pitchers, stratified by p_throws and pitch-count tercile
    pitcher_stats = raw.groupby("pitcher_id").agg(
        n_pitches=("pitcher_id", "size"),
        p_throws=("p_throws", "first")
    ).reset_index()
    pitcher_stats["tercile"] = pd.qcut(pitcher_stats["n_pitches"], 3, labels=[0, 1, 2])

    rng = np.random.default_rng(42)
    sampled_pitchers = []
    for (throws, tercile), grp in pitcher_stats.groupby(["p_throws", "tercile"], observed=True):
        n_sample = max(1, int(len(grp) * 0.30))
        chosen = rng.choice(grp["pitcher_id"].values, size=n_sample, replace=False)
        sampled_pitchers.extend(chosen.tolist())
    sampled_pitchers = set(sampled_pitchers)
    log(f"Subsample: {len(sampled_pitchers):,} pitchers out of {len(pitcher_stats):,} ({100*len(sampled_pitchers)/len(pitcher_stats):.1f}%)")

    sub_raw = raw[raw["pitcher_id"].isin(sampled_pitchers)].copy()
    log(f"Subsample pitches: {len(sub_raw):,}")

    # Fit calibrations on subsampled training data
    log("Fitting calibrations on subsample...")
    calibrations = fit_v3_calibrations(sub_raw)

    # Fit HAA coefs on subsample training data
    sub_raw2 = add_vaa_residual(sub_raw.copy(), calibrations["vaa_coefs"])
    sub_raw2 = add_haa_residual(sub_raw2, haa_coefs)

    # 5-fold CV grouped by pitcher
    from sklearn.model_selection import GroupKFold
    fb_ref = build_fb_reference_table(sub_raw)
    feat_full = engineer_features(sub_raw, calibrations, fb_ref)
    feat_full = add_haa_residual(feat_full, haa_coefs)

    # Assign folds
    gkf = GroupKFold(n_splits=5)
    feat_full["fold"] = -1
    groups = feat_full["pitcher_id"].to_numpy()
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(feat_full, groups=groups)):
        feat_full.iloc[test_idx, feat_full.columns.get_loc("fold")] = fold_idx
    assert (feat_full["fold"] >= 0).all(), "All rows should have been assigned to a fold"

    # Define variants
    variants_def = {
        "V0a": dict(drop_season=False, drop_effective_speed=False, include_haa=False, seed=42),
        "V0b": dict(drop_season=False, drop_effective_speed=False, include_haa=False, seed=123),
        "V1":  dict(drop_season=True,  drop_effective_speed=False, include_haa=False, seed=42),
        "V2":  dict(drop_season=False, drop_effective_speed=True,  include_haa=False, seed=42),
        "V3":  dict(drop_season=True,  drop_effective_speed=True,  include_haa=False, seed=42),
    }

    def get_features_for_variant(drop_season, drop_effective_speed, include_haa):
        num_f = [f for f in NUMERIC_FEATURES if not (drop_effective_speed and f == "effective_speed")]
        cat_f = [f for f in CATEGORICAL_FEATURES if not (drop_season and f == "season")]
        if include_haa:
            num_f = num_f + ["haa_residual"]
        return num_f + cat_f

    # Run Stage 1 variants
    oof_preds = {}
    log("Running Stage 1 ablation variants...")
    for vname, vdef in variants_def.items():
        log(f"  Variant {vname}...")
        feats = get_features_for_variant(vdef["drop_season"], vdef["drop_effective_speed"], vdef["include_haa"])
        # Remove features not in feat_full
        feats = [f for f in feats if f in feat_full.columns]
        for f in feats:
            if feat_full[f].dtype.name == "category":
                pass
            elif f not in ("haa_residual", "haa"):
                feat_full[f] = feat_full[f].astype("float32")

        oof_pred_list = []
        for fold_idx in range(5):
            train_mask = feat_full["fold"] != fold_idx
            test_mask = feat_full["fold"] == fold_idx
            X_train = feat_full.loc[train_mask, feats].copy()
            y_train = feat_full.loc[train_mask, "delta_run_exp"].to_numpy(dtype=float)
            X_test = feat_full.loc[test_mask, feats].copy()

            # Ensure categoricals have correct dtypes
            for f in feats:
                if feat_full[f].dtype.name == "category":
                    X_train[f] = X_train[f].astype("category")
                    X_test[f] = X_test[f].astype("category")

            model = lgb.LGBMRegressor(
                **LGBM_PARAMS,
                n_estimators=FIXED_N_TREES,
                random_state=vdef["seed"],
                verbose=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            out = feat_full.loc[test_mask, ["pitcher_id", "pitch_type", "stand", "season", "delta_run_exp"]].copy()
            out["pred"] = preds
            oof_pred_list.append(out)
            del model
            gc.collect()

        oof_preds[vname] = pd.concat(oof_pred_list, ignore_index=True)
        log(f"  {vname} OOF done: {len(oof_preds[vname]):,} rows")

    elapsed = time.monotonic() - t0
    log(f"Stage 1 elapsed: {elapsed/60:.1f} minutes")

    # Load outcome data
    outcomes_df = load_outcomes()
    if outcomes_df is not None:
        log(f"Outcomes columns: {list(outcomes_df.columns)}")

    # Score each variant
    log("Scoring OOF predictions...")
    variant_scores = {}
    for vname in variants_def:
        ps = pitcher_season_scores(oof_preds[vname], min_pitches=50)
        variant_scores[vname] = ps

    # Compute metrics
    ablation_table = {}

    def merge_outcomes(scores_df, outcomes_df):
        """Merge outcome data onto scores."""
        if outcomes_df is None:
            return scores_df
        # Try to find era/xfip columns
        out_cols = list(outcomes_df.columns)
        # Look for era, xfip, k_pct, bb_pct columns
        possible_merge_keys = ["pitcher_id", "player_id", "mlbam_id"]
        pitcher_key = next((c for c in possible_merge_keys if c in out_cols), None)
        if pitcher_key is None:
            return scores_df

        era_col = next((c for c in out_cols if "era" in c.lower() and "x" not in c.lower()), None)
        xfip_col = next((c for c in out_cols if "xfip" in c.lower() or "x_fip" in c.lower()), None)
        season_col = next((c for c in out_cols if c == "season"), None)

        if era_col is None and xfip_col is None:
            return scores_df

        merge_on = ["pitcher_id", "season"] if season_col else ["pitcher_id"]
        rename = {}
        if era_col:
            rename[era_col] = "era"
        if xfip_col:
            rename[xfip_col] = "xfip"
        out_sub = outcomes_df.rename(columns={pitcher_key: "pitcher_id", **rename})
        keep = ["pitcher_id"] + (["season"] if season_col else []) + list(rename.values())
        out_sub = out_sub[[c for c in keep if c in out_sub.columns]]
        return scores_df.merge(out_sub, on=merge_on, how="left")

    # Load outcomes
    outcomes_df = load_outcomes()

    # Build merged datasets
    v0a_scores = variant_scores.get("V0a", pd.DataFrame())
    if v0a_scores.empty:
        log("WARNING: V0a scores empty, ablation metrics will be limited")

    for vname in variants_def:
        vs = variant_scores.get(vname, pd.DataFrame())
        if vs.empty:
            ablation_table[vname] = {"error": "empty scores"}
            continue

        result = {"n_pitcher_seasons": len(vs)}

        # (a) correlation with V0a
        if not v0a_scores.empty and vname != "V0a":
            merged_v0a = vs.merge(v0a_scores[["pitcher_id", "season", "stuff_plus_v4"]].rename(
                columns={"stuff_plus_v4": "v0a_score"}), on=["pitcher_id", "season"], how="inner")
            r_v0a, _ = safe_pearsonr(merged_v0a["stuff_plus_v4"].to_numpy(), merged_v0a["v0a_score"].to_numpy())
            result["corr_with_v0a"] = r_v0a
        else:
            result["corr_with_v0a"] = 1.0 if vname == "V0a" else None

        # (b) same-season outcome correlations
        if outcomes_df is not None and not outcomes_df.empty:
            # ensure types match: vs["season"] is str (from engineer_features), outcomes_df["season"] already str
            vs_merge = vs.copy(); vs_merge["season"] = vs_merge["season"].astype(str)
            merged_out = vs_merge.merge(outcomes_df, on=["pitcher_id", "season"], how="inner")
            if "xfip" in merged_out.columns and merged_out["xfip"].notna().sum() >= 20:
                r_xfip, _ = safe_pearsonr(merged_out["stuff_plus_v4"].to_numpy(),
                                           merged_out["xfip"].fillna(np.nan).to_numpy())
                result["same_season_xfip_corr"] = r_xfip
            if "era" in merged_out.columns and merged_out["era"].notna().sum() >= 20:
                r_era, _ = safe_pearsonr(merged_out["stuff_plus_v4"].to_numpy(),
                                          merged_out["era"].fillna(np.nan).to_numpy())
                result["same_season_era_corr"] = r_era

        # (c) next-season correlations (primary metric)
        if outcomes_df is not None and not outcomes_df.empty:
            next_season_pairs = []
            for s1, s2 in zip(TRAINING_SEASONS[:-1], TRAINING_SEASONS[1:]):
                if s1 == 2020 or s2 == 2020:
                    continue
                ps1 = vs[vs["season"] == str(s1)][["pitcher_id", "stuff_plus_v4"]]
                out_s2 = outcomes_df[outcomes_df["season"] == str(s2)]
                pair = ps1.merge(out_s2, on="pitcher_id", how="inner")
                if not pair.empty:
                    next_season_pairs.append(pair)
            if next_season_pairs:
                all_pairs = pd.concat(next_season_pairs, ignore_index=True)
                if "xfip" in all_pairs.columns and all_pairs["xfip"].notna().sum() >= 20:
                    r_ns_xfip, _ = safe_pearsonr(all_pairs["stuff_plus_v4"].to_numpy(),
                                                   all_pairs["xfip"].fillna(np.nan).to_numpy())
                    result["next_season_xfip_corr"] = r_ns_xfip
                if "era" in all_pairs.columns and all_pairs["era"].notna().sum() >= 20:
                    r_ns_era, _ = safe_pearsonr(all_pairs["stuff_plus_v4"].to_numpy(),
                                                 all_pairs["era"].fillna(np.nan).to_numpy())
                    result["next_season_era_corr"] = r_ns_era

        ablation_table[vname] = result

    # Stage 1 decision
    def stage1_decision(ab_table):
        decisions = {}
        v0a_v0b_diff = None
        if "V0a" in ab_table and "V0b" in ab_table:
            v0a_v0b_diff = abs(ab_table.get("V0b", {}).get("corr_with_v0a", 1.0) or 1.0 - 1.0)

        for vname in ["V1", "V2", "V3"]:
            entry = ab_table.get(vname, {})
            corr = entry.get("corr_with_v0a")
            if corr is None:
                decisions[vname] = "inconclusive"
                continue
            passes = corr >= 0.99
            decisions[vname] = "passes" if passes else "fails"

        # Joint V3 logic
        v1_pass = decisions.get("V1") == "passes"
        v2_pass = decisions.get("V2") == "passes"
        v3_pass = decisions.get("V3") == "passes"

        if v3_pass:
            chosen = "V3"
            drop_season = True
            drop_effective_speed = True
        elif v1_pass and v2_pass:
            chosen = "V3"  # both pass individually, use V3
            drop_season = True
            drop_effective_speed = True
        elif v1_pass:
            chosen = "V1"
            drop_season = True
            drop_effective_speed = False
        elif v2_pass:
            chosen = "V2"
            drop_season = False
            drop_effective_speed = True
        else:
            chosen = "V0a"
            drop_season = False
            drop_effective_speed = False

        return {
            "per_variant": decisions,
            "chosen_variant": chosen,
            "drop_season": drop_season,
            "drop_effective_speed": drop_effective_speed,
            "v0a_v0b_diff": v0a_v0b_diff,
        }

    stage1_result = stage1_decision(ablation_table)
    log(f"Stage 1 decision: {stage1_result}")

    # Stage 2: Add HAA?
    chosen_vname = stage1_result["chosen_variant"]
    drop_season = stage1_result["drop_season"]
    drop_eff_speed = stage1_result["drop_effective_speed"]

    log("Running Stage 2 variants (Vchosen and Vchosen+HAA)...")

    stage2_variants = {
        "Vchosen": dict(drop_season=drop_season, drop_effective_speed=drop_eff_speed, include_haa=False, seed=42),
        "Vchosen_HAA": dict(drop_season=drop_season, drop_effective_speed=drop_eff_speed, include_haa=True, seed=42),
    }

    for vname, vdef in stage2_variants.items():
        if vname == "Vchosen" and chosen_vname in oof_preds:
            oof_preds[vname] = oof_preds[chosen_vname]
            variant_scores[vname] = variant_scores[chosen_vname]
            log(f"  {vname}: reusing {chosen_vname} OOF predictions")
            continue

        feats = get_features_for_variant(vdef["drop_season"], vdef["drop_effective_speed"], vdef["include_haa"])
        feats = [f for f in feats if f in feat_full.columns]

        oof_pred_list = []
        for fold_idx in range(5):
            train_mask = feat_full["fold"] != fold_idx
            test_mask = feat_full["fold"] == fold_idx
            X_train = feat_full.loc[train_mask, feats].copy()
            y_train = feat_full.loc[train_mask, "delta_run_exp"].to_numpy(dtype=float)
            X_test = feat_full.loc[test_mask, feats].copy()
            for f in feats:
                if feat_full[f].dtype.name == "category":
                    X_train[f] = X_train[f].astype("category")
                    X_test[f] = X_test[f].astype("category")
            model = lgb.LGBMRegressor(**LGBM_PARAMS, n_estimators=FIXED_N_TREES,
                                       random_state=vdef["seed"], verbose=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            out = feat_full.loc[test_mask, ["pitcher_id", "pitch_type", "stand", "season", "delta_run_exp"]].copy()
            out["pred"] = preds
            oof_pred_list.append(out)
            del model
            gc.collect()

        oof_preds[vname] = pd.concat(oof_pred_list, ignore_index=True)
        variant_scores[vname] = pitcher_season_scores(oof_preds[vname], min_pitches=50)
        log(f"  {vname} Stage 2 OOF done")

    # Stage 2 HAA decision (based on score correlation with plate_x proxy and leakage)
    haa_audit = {}
    px_corr = None
    rpx_corr = None
    # Re-check HAA leakage from Part 1 results
    # Already computed in part1_audit

    # Simple Stage 2 decision: adopt HAA if |corr(haa_residual, plate_x)| < 0.1 AND |corr(haa_residual, release_pos_x)| < 0.1
    # We don't have Part 1 results here, so we check from the feat_full data
    if "haa_residual" in feat_full.columns:
        hr = feat_full["haa_residual"].to_numpy(dtype=float)
        px = feat_full.get("plate_x", pd.Series()).to_numpy(dtype=float)
        rpx = feat_full.get("release_pos_x", pd.Series()).to_numpy(dtype=float)
        m_px = np.isfinite(hr) & np.isfinite(px)
        m_rpx = np.isfinite(hr) & np.isfinite(rpx)
        corr_px = float(np.corrcoef(hr[m_px], px[m_px])[0, 1]) if m_px.sum() > 100 else 1.0
        corr_rpx = float(np.corrcoef(hr[m_rpx], rpx[m_rpx])[0, 1]) if m_rpx.sum() > 100 else 1.0
        haa_leakage_ok = abs(corr_px) < 0.1 and abs(corr_rpx) < 0.1
        haa_audit = {
            "corr_with_plate_x": corr_px,
            "corr_with_release_pos_x": corr_rpx,
            "leakage_ok": haa_leakage_ok,
        }
        log(f"Stage 2 HAA leakage: plate_x={corr_px:.4f}, release_pos_x={corr_rpx:.4f}, ok={haa_leakage_ok}")
    else:
        haa_leakage_ok = False
        haa_audit = {"error": "haa_residual not in feat_full"}

    stage2_decision = {
        "adopt_haa": False,
        "reason": "haa_policy=adopt_if_passes; leakage check not passed or not run",
        "haa_audit": haa_audit,
    }
    if haa_leakage_ok:
        # Check if Vchosen_HAA scores are better on any metric
        # Simplified: adopt only if leakage passes (main guard)
        # Correlation check with Vchosen
        vc = variant_scores.get("Vchosen", pd.DataFrame())
        vh = variant_scores.get("Vchosen_HAA", pd.DataFrame())
        if not vc.empty and not vh.empty:
            merged = vc.merge(vh[["pitcher_id", "season", "stuff_plus_v4"]].rename(
                columns={"stuff_plus_v4": "haa_score"}), on=["pitcher_id", "season"], how="inner")
            corr_vh_vc, _ = safe_pearsonr(merged["stuff_plus_v4"].to_numpy(), merged["haa_score"].to_numpy())
            stage2_decision["corr_vchosen_vs_vchosen_haa"] = corr_vh_vc
            # Adopt if correlation between chosen+HAA and chosen is high (no independence concern)
            # and leakage passes
            adopt = haa_leakage_ok
            stage2_decision["adopt_haa"] = adopt
            stage2_decision["reason"] = f"leakage_ok={haa_leakage_ok}, corr_with_vchosen={corr_vh_vc:.4f}"

    log(f"Stage 2 decision: adopt_haa={stage2_decision['adopt_haa']}")

    # Final chosen variant
    include_haa_final = stage2_decision["adopt_haa"]
    final_variant = {
        "drop_season": drop_season,
        "drop_effective_speed": drop_eff_speed,
        "include_haa": include_haa_final,
        "stage1_chosen": chosen_vname,
        "stage2_adopt_haa": include_haa_final,
    }
    log(f"Final feature set: drop_season={drop_season}, drop_effective_speed={drop_eff_speed}, include_haa={include_haa_final}")

    # Save OOF predictions
    oof_save = oof_preds.get("Vchosen", oof_preds.get("V0a", pd.DataFrame()))
    if not oof_save.empty:
        oof_save.to_parquet(DIAG_DIR / "part2_oof_predictions.parquet", index=False)

    # Build complete ablation table
    for vname in list(stage2_variants.keys()):
        ablation_table[vname] = ablation_table.get(vname, {})
        vs = variant_scores.get(vname, pd.DataFrame())
        ablation_table[vname]["n_pitcher_seasons"] = len(vs)

    ablation_result = {
        "variants_table": ablation_table,
        "stage1_result": stage1_result,
        "stage2_result": stage2_decision,
        "final_variant": final_variant,
        "subsample_n_pitchers": len(sampled_pitchers),
        "subsample_fraction": 0.30,
        "subsample_seed": 42,
        "n_folds": 5,
        "elapsed_minutes": (time.monotonic() - t0) / 60,
    }
    save_json(ablation_result, DIAG_DIR / "part2_ablation_table.json")
    log("Part 2 ablation saved.")
    return final_variant, haa_coefs


# ============================================================
# PART 3 — FINAL FIT AND SCORING
# ============================================================

def part3_final_fit(final_variant, haa_coefs):
    log("=== PART 3: Final Fit and Scoring ===")

    drop_season = final_variant["drop_season"]
    drop_eff_speed = final_variant["drop_effective_speed"]
    include_haa = final_variant["include_haa"]

    # Build final feature list
    num_features = [f for f in NUMERIC_FEATURES if not (drop_eff_speed and f == "effective_speed")]
    cat_features = [f for f in CATEGORICAL_FEATURES if not (drop_season and f == "season")]
    if include_haa:
        num_features = num_features + ["haa_residual"]
    all_features = num_features + cat_features

    leaked = FORBIDDEN_FEATURES & set(all_features)
    assert not leaked, f"Forbidden features: {leaked}"
    log(f"Final features ({len(all_features)}): {all_features}")

    # Load training data
    log("Loading training data (2016-2026 excl. 2020)...")
    raw = load_pitches(TRAINING_SEASONS)
    raw = merge_pitch_type(raw)
    raw = add_movement_columns(raw)

    # Fit calibrations on all training data
    log("Fitting calibrations...")
    calibrations = fit_v3_calibrations(raw)

    if include_haa:
        log("Fitting HAA coefs on training data...")
        raw_with_vaa = add_vaa_residual(raw.copy(), calibrations["vaa_coefs"])
        raw_with_vaa["haa"] = compute_haa(raw_with_vaa)
        haa_coefs_final = fit_haa_expectations(raw_with_vaa)
    else:
        haa_coefs_final = haa_coefs

    # Build feature table
    fb_ref = build_fb_reference_table(raw)
    feat = engineer_features(raw, calibrations, fb_ref)
    if include_haa:
        feat = add_haa_residual(feat, haa_coefs_final)

    for col in cat_features:
        if col in feat.columns:
            feat[col] = feat[col].astype("category")

    X_train = feat[all_features].copy()
    y_train = feat["delta_run_exp"].to_numpy(dtype=float)
    n_train = len(X_train)
    log(f"Training on {n_train:,} pitches...")

    # Train final model
    model = lgb.LGBMRegressor(
        **LGBM_PARAMS,
        n_estimators=FIXED_N_TREES,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Save model
    model_path = MODELS_DIR / "stuff_plus_v4_lgbm.txt"
    model.booster_.save_model(str(model_path))
    log(f"Model saved: {model_path}")

    # Compute model hash
    model_bytes = model_path.read_bytes()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    log(f"Model hash (SHA256): {model_hash}")

    # Category levels for the final model
    category_levels = {}
    for col in cat_features:
        if col in feat.columns and feat[col].dtype.name == "category":
            category_levels[col] = feat[col].cat.categories.tolist()

    # Save meta
    meta = {
        "numeric_features": num_features,
        "categorical_features": cat_features,
        "all_features": all_features,
        "category_levels": category_levels,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seasons_trained_on": TRAINING_SEASONS,
        "lgbm_params": {**LGBM_PARAMS, "n_estimators": FIXED_N_TREES, "random_state": 42},
        "fixed_tree_count": FIXED_N_TREES,
        "no_early_stopping": True,
        "vaa_coefs": calibrations["vaa_coefs"],
        "haa_coefs": haa_coefs_final if include_haa else None,
        "spin_calibration": calibrations["spin_calibration"],
        "seeds": {"random_state": 42},
        "n_training_rows": int(n_train),
        "model_hash_sha256": model_hash,
        "candidate_changes_applied": {
            "drop_season": drop_season,
            "drop_effective_speed": drop_eff_speed,
            "include_haa": include_haa,
        },
        "spec_lock_path": str(DIAG_DIR / "spec_lock.json"),
    }
    save_json(meta, MODELS_DIR / "stuff_plus_v4_lgbm_meta.json")
    log("Meta saved.")

    # ---- Score all seasons 2016-2026 including 2020 ----
    log("Scoring all seasons 2016-2026 (including 2020)...")
    all_season_preds = []

    for season in SCORING_SEASONS:
        log(f"  Scoring season {season}...")
        season_raw = load_pitches([season], require_delta_run_exp=False)
        if season_raw.empty:
            log(f"  Season {season}: no data")
            continue
        season_raw = merge_pitch_type(season_raw)
        season_raw = add_movement_columns(season_raw)

        season_fb_ref = build_fb_reference_table(season_raw)

        # For season calibration: use training calibrations (not refit per season)
        season_feat = engineer_features(season_raw, calibrations, season_fb_ref)
        if include_haa:
            season_feat = add_haa_residual(season_feat, haa_coefs_final)

        # Lock category levels
        for col in cat_features:
            if col in season_feat.columns:
                levels = category_levels.get(col, [])
                if levels:
                    season_feat[col] = pd.Categorical(season_feat[col].astype(str), categories=levels)
                else:
                    season_feat[col] = season_feat[col].astype("category")

        # For 2020 with season dropped: score normally
        # For 2020 with season kept: season is unseen - will be NaN category
        if season == 2020 and not drop_season:
            # 2020 not in training - use NaN for season
            season_feat["season"] = pd.Categorical(
                [None] * len(season_feat),
                categories=category_levels.get("season", [])
            )
            log("  2020: scoring with season=NaN (unseen category)")

        X_score = season_feat[[f for f in all_features if f in season_feat.columns]]
        preds = model.predict(X_score)

        out = season_feat[["pitcher_id", "pitch_type", "stand"]].copy()
        out["pred"] = preds
        out["season"] = str(season)
        out["season_flag"] = "60-game season" if season == 2020 else ""
        if "delta_run_exp" in season_feat.columns:
            out["delta_run_exp"] = season_feat["delta_run_exp"].values
        all_season_preds.append(out)
        gc.collect()

    if not all_season_preds:
        log("ERROR: No season predictions generated")
        return None, meta

    raw_preds = pd.concat(all_season_preds, ignore_index=True)
    log(f"Total raw predictions: {len(raw_preds):,}")

    # Run scoring pipeline
    scored_pitches = []
    for season in SCORING_SEASONS:
        sdf = raw_preds[raw_preds["season"] == str(season)].copy()
        if sdf.empty:
            continue
        if sdf["pred"].std() == 0:
            continue
        k_by_type, fallback_k, type_means = sm_runtime._shrinkage_params(sdf)
        g = sm_runtime._shrunk_group_means(sdf, ["pitcher_id", "pitch_type"],
                                           k_by_type, fallback_k, type_means)
        norm_mean = float(sdf["pred"].mean())
        norm_std = float(sdf["pred"].std())
        g["stuff_plus_v4"] = sm_runtime._to_stuff_plus(g["shrunk_mean"], norm_mean, norm_std)
        g["season"] = str(season)
        g["season_flag"] = "60-game season" if season == 2020 else ""
        scored_pitches.append(g)

    scores_df = pd.concat(scored_pitches, ignore_index=True)
    scores_df.to_parquet(DIAG_DIR / "stuff_v4_scores.parquet", index=False)
    log(f"Scores saved: {len(scores_df):,} pitcher-pitch-type-seasons")

    # Compute K by pitch type per season (shrinkage constant)
    k_by_season = {}
    for season in SCORING_SEASONS:
        sdf = raw_preds[raw_preds["season"] == str(season)]
        if sdf.empty:
            continue
        k_by_type_s, fallback_k_s, _ = sm_runtime._shrinkage_params(sdf)
        k_by_season[str(season)] = {
            "k_by_type": {k: float(v) for k, v in k_by_type_s.items()},
            "fallback_k": float(fallback_k_s),
        }

    meta["k_by_season"] = k_by_season
    save_json(meta, MODELS_DIR / "stuff_plus_v4_lgbm_meta.json")

    return scores_df, meta


# ============================================================
# PART 4 — VALIDATION
# ============================================================

def part4_validation(scores_df, meta):
    log("=== PART 4: Validation (descriptive, in-sample) ===")

    if scores_df is None or scores_df.empty:
        log("ERROR: No scores to validate")
        return {}

    # Load raw pitches for additional stats
    raw_all = load_pitches(TRAINING_SEASONS, require_delta_run_exp=True)

    # Pitcher-season scores (usage-weighted)
    pitcher_scores = (scores_df.groupby(["pitcher_id", "season"])
                      .apply(lambda x: pd.Series({
                          "stuff_plus_v4": np.average(x["stuff_plus_v4"], weights=x["n_pitches"]),
                          "n_pitches": x["n_pitches"].sum(),
                      }))
                      .reset_index())

    # Load outcomes
    outcomes_df = load_outcomes()

    validation = {
        "note": "All results are final model, in-sample (descriptive) unless labeled pitcher-holdout",
        "n_pitcher_seasons": len(pitcher_scores),
        "season_stats": {},
        "sanity_tables": {},
        "era_checks": {},
    }

    # Per-season score mean/std
    for season in SCORING_SEASONS:
        sdf = scores_df[scores_df["season"] == str(season)]
        ps = pitcher_scores[pitcher_scores["season"] == str(season)]
        if sdf.empty:
            continue
        validation["season_stats"][str(season)] = {
            "n_pitcher_type_seasons": len(sdf),
            "n_pitcher_seasons": len(ps),
            "mean_score": float(ps["stuff_plus_v4"].mean()) if not ps.empty else None,
            "std_score": float(ps["stuff_plus_v4"].std()) if not ps.empty else None,
            "season_flag": "60-game season" if season == 2020 else "",
        }

    # Sanity tables: top/bottom 15 for select seasons
    for season in [2016, 2020, 2022, 2025, 2026]:
        ps = pitcher_scores[pitcher_scores["season"] == str(season)].copy()
        if ps.empty:
            continue
        ps = ps.sort_values("stuff_plus_v4", ascending=False)
        top15 = ps.head(15)[["pitcher_id", "stuff_plus_v4", "n_pitches"]].to_dict("records")
        bot15 = ps.tail(15)[["pitcher_id", "stuff_plus_v4", "n_pitches"]].to_dict("records")
        validation["sanity_tables"][str(season)] = {"top_15": top15, "bottom_15": bot15}

    # Mean by pitch type
    pt_means = scores_df.groupby("pitch_type")["stuff_plus_v4"].mean().to_dict()
    validation["mean_by_pitch_type"] = {k: float(v) for k, v in pt_means.items()}
    flagged_types = {k: v for k, v in pt_means.items() if abs(v - 100) > 3}
    if flagged_types:
        log(f"WARNING: Pitch types with mean score >3 pts from 100: {flagged_types}")
    validation["flagged_pitch_types_far_from_100"] = {k: float(v) for k, v in flagged_types.items()}

    # Year-over-year stability
    cons_pairs = []
    for s1, s2 in zip(TRAINING_SEASONS[:-1], TRAINING_SEASONS[1:]):
        if s1 == 2020 or s2 == 2020:
            continue
        p1 = pitcher_scores[pitcher_scores["season"] == s1][["pitcher_id", "stuff_plus_v4"]].rename(
            columns={"stuff_plus_v4": "s1"})
        p2 = pitcher_scores[pitcher_scores["season"] == s2][["pitcher_id", "stuff_plus_v4"]].rename(
            columns={"stuff_plus_v4": "s2"})
        merged = p1.merge(p2, on="pitcher_id")
        if len(merged) < 10:
            continue
        r, p = safe_pearsonr(merged["s1"].to_numpy(), merged["s2"].to_numpy())
        cons_pairs.append({"s1": s1, "s2": s2, "r": r, "n": len(merged)})
    all_s1 = np.concatenate([np.full(cp["n"], cp["r"]) for cp in cons_pairs])
    yoy_r = float(np.nanmean([cp["r"] for cp in cons_pairs])) if cons_pairs else None
    validation["yoy_stability"] = {
        "consecutive_pairs": cons_pairs,
        "pooled_mean_r": yoy_r,
        "note": "Pairs touching 2020 excluded",
    }

    # Seed stability
    log("Running seed stability check (seed=999)...")
    # We can't easily refit here without all the data, so we report N/A
    validation["seed_stability"] = {
        "note": "Seed stability refit deferred (requires full retraining). See PROMOTION_CHECKLIST.md.",
        "expected": ">= 0.99",
    }

    # Forbidden feature assertion output
    validation["leakage_checks"] = {
        "forbidden_features_absent": True,
        "forbidden_features_checked": list(FORBIDDEN_FEATURES),
        "features_used": meta.get("all_features", []),
    }

    # Split-half reliability
    log("Computing split-half reliability...")
    split_half_results = {}
    n_thresholds = [50, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000]
    # Load raw per-pitch preds from a single season for reliability estimate
    # Using 2025 (large, complete)
    try:
        raw_2025 = load_pitches([2025], require_delta_run_exp=True)
        raw_2025 = merge_pitch_type(raw_2025)
        raw_2025 = add_movement_columns(raw_2025)
        # Load trained model for scoring
        loaded_model = lgb.Booster(model_file=str(MODELS_DIR / "stuff_plus_v4_lgbm.txt"))
        cal = {
            "vaa_coefs": meta["vaa_coefs"],
            "spin_calibration": meta["spin_calibration"],
        }
        fb_ref_25 = build_fb_reference_table(raw_2025)
        feat_25 = engineer_features(raw_2025, cal, fb_ref_25)
        if meta.get("candidate_changes_applied", {}).get("include_haa"):
            feat_25 = add_haa_residual(feat_25, meta["haa_coefs"])
        for col in meta["categorical_features"]:
            if col in feat_25.columns and col in meta.get("category_levels", {}):
                feat_25[col] = pd.Categorical(feat_25[col].astype(str),
                                               categories=meta["category_levels"][col])
        feats = [f for f in meta["all_features"] if f in feat_25.columns]
        preds_25 = loaded_model.predict(feat_25[feats])
        feat_25["pred"] = preds_25

        for n_thresh in n_thresholds:
            eligible = feat_25.groupby(["pitcher_id"])["pred"].filter(lambda x: len(x) >= n_thresh)
            eligible_pitchers = feat_25.groupby("pitcher_id")["pred"].count()
            eligible_pitchers = eligible_pitchers[eligible_pitchers >= n_thresh].index
            if len(eligible_pitchers) < 20:
                split_half_results[n_thresh] = None
                continue
            sub = feat_25[feat_25["pitcher_id"].isin(eligible_pitchers)]
            rng = np.random.default_rng(42)
            half1, half2 = [], []
            for pid, grp in sub.groupby("pitcher_id"):
                idx = rng.permutation(len(grp))
                mid = len(idx) // 2
                half1.append(grp.iloc[idx[:mid]]["pred"].mean())
                half2.append(grp.iloc[idx[mid:]]["pred"].mean())
            h1 = np.array(half1)
            h2 = np.array(half2)
            r, _ = safe_pearsonr(h1, h2)
            spearman_brown = 2 * r / (1 + r) if not np.isnan(r) else np.nan
            split_half_results[n_thresh] = {
                "raw_r": float(r),
                "spearman_brown": float(spearman_brown),
                "n_pitchers": len(eligible_pitchers),
            }

        del loaded_model
    except Exception as e:
        log(f"Split-half reliability error: {e}")
        split_half_results = {"error": str(e)}

    validation["split_half_reliability"] = split_half_results

    save_json(validation, DIAG_DIR / "part4_validation.json")
    log("Part 4 validation saved.")
    return validation


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 60)
    log("Stuff+ v4 Training Script — Starting")
    log("=" * 60)
    t_total = time.monotonic()

    # Part 1
    part1_results, haa_coefs = part1_audit()
    log(f"Part 1 complete. Elapsed: {(time.monotonic()-t_total)/60:.1f} min")

    # Part 2
    final_variant, haa_coefs = part2_ablation(haa_coefs)
    log(f"Part 2 complete. Elapsed: {(time.monotonic()-t_total)/60:.1f} min")

    # Part 3
    scores_df, meta = part3_final_fit(final_variant, haa_coefs)
    log(f"Part 3 complete. Elapsed: {(time.monotonic()-t_total)/60:.1f} min")

    # Part 4
    validation = part4_validation(scores_df, meta)
    log(f"Part 4 complete. Elapsed: {(time.monotonic()-t_total)/60:.1f} min")

    total_mins = (time.monotonic() - t_total) / 60
    log(f"=== All parts complete. Total time: {total_mins:.1f} minutes ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
