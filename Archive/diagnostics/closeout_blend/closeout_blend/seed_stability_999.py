"""
Targeted seed stability check: train Stuff+ v4 final configuration with seed=999.
Produces stuff_v4_scores_seed999.parquet for comparison with seed=42.
"""
import sys, os, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
import sqlite3

BASE = str(ROOT)
DIAG = f"{BASE}/diagnostics/stuff_v4"
OUT = f"{BASE}/diagnostics/closeout_blend"
LOG = f"{OUT}/progress.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("Seed stability: loading model meta...")

# Load the final model meta to get config
meta_path = f"{BASE}/models/stuff_plus_v4_lgbm_meta.json"
if not os.path.exists(meta_path):
    log("ERROR: stuff_plus_v4_lgbm_meta.json not found")
    sys.exit(1)

with open(meta_path) as f:
    meta = json.load(f)

log(f"Model features: {meta.get('all_features', 'not found')}")
log(f"Cat features: {meta.get('categorical_features', 'not found')}")

# Load analytics modules
from analytics.stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES,
    EXCLUDED_PITCH_TYPES,
    add_movement_columns, build_fb_reference_table, engineer_features,
    merge_pitch_type, add_vaa_residual,
)
try:
    from analytics.stuff_features import add_haa_residual
    HAS_HAA = True
except ImportError:
    HAS_HAA = False
    log("  add_haa_residual not available")

from analytics.spin_physics import fit_spin_axis_calibration

spec = json.load(open(f"{DIAG}/spec_lock.json"))
TRAINING_SEASONS = spec["training_seasons"]
EXCL_TYPES = spec["excluded_pitch_types"]
FIXED_N_TREES = spec.get("fixed_tree_count", 2000)

log(f"Training seasons: {TRAINING_SEASONS}")

# Load pitches
log("Loading pitches...")
conn = sqlite3.connect(f"{BASE}/mlb_data.db")
seasons_str = ",".join(str(s) for s in TRAINING_SEASONS)
df = pd.read_sql(f"""
    SELECT pitcher_id, season, pitch_type, p_throws, stand,
           release_speed, effective_speed, release_spin_rate, spin_axis,
           pfx_x, pfx_z, plate_x, plate_z, balls, strikes,
           release_pos_x, release_pos_y, release_pos_z, release_extension,
           vx0, vy0, vz0, ax, ay, az, delta_run_exp
    FROM pitches
    WHERE season IN ({seasons_str})
      AND pitch_type NOT IN ({','.join("'"+t+"'" for t in EXCL_TYPES)})
      AND delta_run_exp IS NOT NULL
      AND release_speed IS NOT NULL
      AND release_spin_rate IS NOT NULL
""", conn)
conn.close()
log(f"Loaded {len(df):,} pitches")

# Feature engineering
log("Engineering features...")
df = merge_pitch_type(df)
df = add_movement_columns(df)

# Calibrations from meta
cal = {}
if "vaa_coefs" in meta:
    cal["vaa_coefs"] = meta["vaa_coefs"]
if "spin_calibration" in meta:
    cal["spin_calibration"] = meta["spin_calibration"]

# Build FB reference
fb_ref = build_fb_reference_table(df)
feat_df = engineer_features(df, cal, fb_ref)

# HAA if used in final model
include_haa = meta.get("candidate_changes_applied", {}).get("include_haa", False)
if include_haa and HAS_HAA and "haa_coefs" in meta:
    feat_df = add_haa_residual(feat_df, meta["haa_coefs"])
    log("  HAA residual added")

# Feature lists
all_features = meta.get("all_features", ALL_FEATURES)
cat_features = meta.get("categorical_features", CATEGORICAL_FEATURES)
cat_levels = meta.get("category_levels", {})

# Encode categoricals
for col in cat_features:
    if col in feat_df.columns:
        if col in cat_levels:
            feat_df[col] = pd.Categorical(feat_df[col].astype(str), categories=cat_levels[col])
        else:
            feat_df[col] = feat_df[col].astype("category")

feat_df = feat_df.dropna(subset=["delta_run_exp"])
available_feats = [f for f in all_features if f in feat_df.columns]
log(f"Available features ({len(available_feats)}): {available_feats}")

X = feat_df[available_feats]
y = feat_df["delta_run_exp"].values

# LGBM params from spec
lgbm_params = {
    "objective": "huber",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": FIXED_N_TREES,
    "random_state": 999,  # THE EXTRA SEED
}

log(f"Training LightGBM with seed=999, n_estimators={FIXED_N_TREES}...")
model = lgb.LGBMRegressor(**lgbm_params)
model.fit(X, y)
log("Training complete")

# Score all seasons (including 2020 and 2026)
conn2 = sqlite3.connect(f"{BASE}/mlb_data.db")
all_seasons = [2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026]
season_scores = []
for s in all_seasons:
    df_s = pd.read_sql(f"""
        SELECT pitcher_id, season, pitch_type, p_throws, stand,
               release_speed, effective_speed, release_spin_rate, spin_axis,
               pfx_x, pfx_z, plate_x, plate_z, balls, strikes,
               release_pos_x, release_pos_y, release_pos_z, release_extension,
               vx0, vy0, vz0, ax, ay, az, delta_run_exp
        FROM pitches
        WHERE season={s}
          AND pitch_type NOT IN ({','.join("'"+t+"'" for t in EXCL_TYPES)})
          AND release_speed IS NOT NULL AND release_spin_rate IS NOT NULL
    """, conn2)
    if len(df_s) == 0: continue
    df_s = merge_pitch_type(df_s)
    df_s = add_movement_columns(df_s)

    # For 2020: use policy (average of 2019/2021 season encoding)
    # For 2026: use 2025 season encoding (already in model as training season)

    feat_s = engineer_features(df_s, cal, build_fb_reference_table(df_s))
    if include_haa and HAS_HAA and "haa_coefs" in meta:
        feat_s = add_haa_residual(feat_s, meta["haa_coefs"])
    for col in cat_features:
        if col in feat_s.columns:
            if col in cat_levels:
                feat_s[col] = pd.Categorical(feat_s[col].astype(str), categories=cat_levels[col])
            else:
                feat_s[col] = feat_s[col].astype("category")

    feats_avail = [f for f in available_feats if f in feat_s.columns]
    feat_s["pred_s999"] = model.predict(feat_s[feats_avail])

    # Aggregate to pitcher-season
    ps = (feat_s.dropna(subset=["pred_s999"])
        .groupby(["pitcher_id","pitch_type"])
        .agg(n_pitches=("pred_s999","count"), mean_pred=("pred_s999","mean"))
        .reset_index())

    if len(ps) == 0: continue

    # Apply shrinkage: same as final model
    # Load shrinkage params from meta
    shrinkage = meta.get("shrinkage", {})
    if shrinkage:
        for pt, shrink_meta in shrinkage.items():
            mask = ps["pitch_type"] == pt
            if mask.any():
                mu = shrink_meta.get("mu", ps.loc[mask,"mean_pred"].mean())
                kappa = shrink_meta.get("kappa", 0)
                n0 = shrink_meta.get("n0", 100)
                ps.loc[mask, "shrunk_mean"] = (ps.loc[mask,"n_pitches"] * ps.loc[mask,"mean_pred"] + n0 * mu) / (ps.loc[mask,"n_pitches"] + n0)
        if "shrunk_mean" not in ps.columns:
            ps["shrunk_mean"] = ps["mean_pred"]
    else:
        ps["shrunk_mean"] = ps["mean_pred"]

    # Convert to Stuff+ scale: z-score then 100-mean, 10-std
    # Use league means from meta
    league = meta.get("league_stats", {})
    mu_league = league.get("mean_shrunk", ps["shrunk_mean"].mean())
    std_league = league.get("std_shrunk", ps["shrunk_mean"].std())
    if std_league < 1e-6: std_league = 1.0
    ps["stuff_plus_v4"] = (ps["shrunk_mean"] - mu_league) / std_league * (-10) + 100
    ps["season"] = s
    ps["season_flag"] = "60-game season" if s == 2020 else ""
    season_scores.append(ps)
    log(f"  Season {s}: {len(ps):,} pitcher-pitch-type rows")

conn2.close()

if not season_scores:
    log("ERROR: no scores produced")
    sys.exit(1)

all_scores = pd.concat(season_scores, ignore_index=True)
out_path = f"{BASE}/diagnostics/stuff_v4/stuff_v4_scores_seed999.parquet"
all_scores.to_parquet(out_path, index=False)
log(f"Saved {len(all_scores):,} rows to {out_path}")

with open(LOG, "a") as f:
    f.write("SEED999_COMPLETE\n")
print("DONE")
