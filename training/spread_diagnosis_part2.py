"""
Part 2: Stuff+ reference stats + Part D Huber vs MSE comparison.
Run from mlb_dashboard/ directory.
"""
import json
import sqlite3
import sys
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from analytics.location_features import (
    ALL_FEATURES as LOC_ALL_FEATURES,
    CATEGORICAL_FEATURES as LOC_CAT_FEATURES,
    RAW_QUERY_COLUMNS as LOC_RAW_COLS,
    engineer_features as loc_engineer,
    to_plus_scale,
)
from analytics.tunneling_features import (
    lag_feature_columns, load_pitches as tun_load_pitches, sequential_lag_features, MAX_LAG,
)
from analytics.stuff_features import RAW_QUERY_COLUMNS as STUFF_RAW_COLS
from analytics.stuff_model import (
    compute_stuff_plus, _raw_predictions as stuff_raw_preds,
    _shrinkage_params, MODELS_DIR as STUFF_MODELS_DIR,
)

SEASON = 2025
HOLDOUT = 2024
DB_PATH = "mlb_data.db"
MIN_PITCHES_STUFF = 200
MIN_PITCHES_TUN = 100
MIN_PITCHES_LOC = 100

MODELS_DIR = Path("models")
TUN_META_PATH = MODELS_DIR / "tunneling_v3_meta.json"
TUN_COMMIT_TABLE_PATH = (
    Path("project_artifacts/training_artifacts/tunneling_v2_artifacts/200ms/pitcher_season_commit_point.csv")
)

LGBM_BASE = dict(
    n_estimators=2000, learning_rate=0.03,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
)

LAG_COLS = lag_feature_columns()
NUMERIC_LAG_COLS = [c for c in LAG_COLS if not c.endswith("_pitch_type")]
CATEGORICAL_LAG_COLS = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CATEGORICAL_TUN = ["pitch_type", "p_throws", "stand", "season"]
CATEGORICAL_FEATURES_TUN = CATEGORICAL_LAG_COLS + CURRENT_CATEGORICAL_TUN
ALL_FEATURES_TUN = NUMERIC_LAG_COLS + CATEGORICAL_FEATURES_TUN


def pct_summary(scores, label):
    s = np.array(scores, dtype=float)
    p = [5, 25, 50, 75, 95]
    pcts = np.percentile(s, p)
    print(f"\n=== {label} (n={len(s)}) ===")
    print(f"  mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.2f}  max={s.max():.2f}")
    for pct, val in zip(p, pcts):
        print(f"  p{pct:02d}: {val:.2f}")
    in_99_101 = np.mean((s >= 99) & (s <= 101)) * 100
    in_95_105 = np.mean((s >= 95) & (s <= 105)) * 100
    print(f"  within [99,101]: {in_99_101:.1f}%")
    print(f"  within [95,105]: {in_95_105:.1f}%")


def _stringify_keep_na(series):
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)


conn = sqlite3.connect(DB_PATH)


# ─────────────────────────────────────────────────────────────────
# STUFF+ reference (Part B/C)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STUFF+ REFERENCE (Part B/C)")
print("="*70)

from analytics.stuff_model import overall_stuff_plus
stuff_scores = compute_stuff_plus(conn, season=SEASON)
# stuff_scores is indexed by (pitcher_id, pitch_type); filter pitchers with enough total pitches
pitcher_totals = stuff_scores.groupby("pitcher_id")["n_pitches"].sum()
qualifying_pitchers = pitcher_totals[pitcher_totals >= MIN_PITCHES_STUFF].index
stuff_q = stuff_scores[stuff_scores.index.get_level_values("pitcher_id").isin(qualifying_pitchers)]

per_pitcher = pd.DataFrame([
    {"pitcher_id": pid, "score": overall_stuff_plus(stuff_q, pid),
     "n_pitches": int(stuff_q.xs(pid, level="pitcher_id")["n_pitches"].sum())}
    for pid in qualifying_pitchers
])
per_pitcher = per_pitcher.dropna(subset=["score"])
pct_summary(per_pitcher["score"].values, f"Stuff+ {SEASON} per-pitcher (≥{MIN_PITCHES_STUFF} pitches)")

# Raw predictions for variance comparison
raw = stuff_raw_preds(conn, SEASON)
raw_preds_arr = raw["pred"].values
targets = raw["delta_run_exp"].values if "delta_run_exp" in raw.columns else None

print(f"\n  Single-pitch raw pred std:  {raw_preds_arr.std():.6f}")
per_p_raw = raw.groupby("pitcher_id")["pred"].mean()
counts = raw.groupby("pitcher_id").size()
per_p_q = per_p_raw[counts >= MIN_PITCHES_STUFF]
print(f"  Per-pitcher mean raw pred std (qualified): {per_p_q.std():.6f}")

# Target comparison: load actual targets
stuff_df_raw = pd.read_sql(f"""
    SELECT {', '.join(STUFF_RAW_COLS)} FROM pitches
    WHERE season = ? AND delta_run_exp IS NOT NULL
""", conn, params=(SEASON,))
target_std = stuff_df_raw["delta_run_exp"].std()
print(f"\n  Target (delta_run_exp) std:  {target_std:.6f}")
print(f"  Raw pred / target ratio:     {raw_preds_arr.std() / target_std:.4f}")
print(f"  CV R² (from report, mean):   0.000247  (per models/stuff_plus_v3_lgbm_report.json)")


# ─────────────────────────────────────────────────────────────────
# PART D: Huber vs MSE — Tunneling+
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"PART D — Tunneling+: Huber vs MSE (train on {HOLDOUT-1}/{HOLDOUT+1}, test on {HOLDOUT})")
print("="*70)

tun_meta = json.loads(TUN_META_PATH.read_text())
commit_table = pd.read_csv(TUN_COMMIT_TABLE_PATH)

print(f"Loading pitches for seasons {HOLDOUT-1}, {HOLDOUT}, {HOLDOUT+1}...")
dfs = []
for s in [HOLDOUT - 1, HOLDOUT, HOLDOUT + 1]:
    df_s = tun_load_pitches(conn, s)
    totals = df_s.groupby("pitcher_id").size()
    qualifying = totals[totals >= MIN_PITCHES_TUN].index
    dfs.append(df_s[df_s["pitcher_id"].isin(qualifying)])
tun_all = pd.concat(dfs, ignore_index=True)

print(f"Building lag features for {len(tun_all):,} pitches...")
feat_all = sequential_lag_features(tun_all, commit_table, max_lag=MAX_LAG)
feat_all = feat_all.dropna(subset=["delta_run_exp"]).copy()
feat_all["season"] = feat_all["season"].astype(str)
for col in CATEGORICAL_FEATURES_TUN:
    feat_all[col] = _stringify_keep_na(feat_all[col])

train_tun = feat_all[feat_all["season"] != str(HOLDOUT)].copy()
test_tun = feat_all[feat_all["season"] == str(HOLDOUT)].copy()
print(f"  train rows: {len(train_tun):,}  test rows: {len(test_tun):,}")

# share category levels
for col in CATEGORICAL_FEATURES_TUN:
    cats = sorted(set(train_tun[col][train_tun[col].notna()]) | set(test_tun[col][test_tun[col].notna()]))
    train_tun[col] = pd.Categorical(train_tun[col], categories=cats)
    test_tun[col] = pd.Categorical(test_tun[col], categories=cats)

def fit_model(train, obj, n_leaves=63):
    fr, sr = train_test_split(train, test_size=0.08, random_state=42)
    m = lgb.LGBMRegressor(**{**LGBM_BASE, "objective": obj, "num_leaves": n_leaves})
    m.fit(
        fr[ALL_FEATURES_TUN], fr["delta_run_exp"],
        eval_set=[(sr[ALL_FEATURES_TUN], sr["delta_run_exp"])],
        categorical_feature=CATEGORICAL_FEATURES_TUN,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return m

print(f"\nFitting Tunneling+ with Huber loss...")
m_tun_huber = fit_model(train_tun, "huber", n_leaves=63)
preds_tun_huber = m_tun_huber.predict(test_tun[ALL_FEATURES_TUN])

print(f"Fitting Tunneling+ with MSE loss...")
m_tun_mse = fit_model(train_tun, "mse", n_leaves=63)
preds_tun_mse = m_tun_mse.predict(test_tun[ALL_FEATURES_TUN])

tun_target = test_tun["delta_run_exp"].values
print(f"\nTunneling+ holdout={HOLDOUT} results:")
print(f"  Target std:                  {tun_target.std():.6f}")
print(f"  Huber raw pred std:          {preds_tun_huber.std():.6f}  R²={r2_score(tun_target, preds_tun_huber):.5f}")
print(f"  MSE   raw pred std:          {preds_tun_mse.std():.6f}  R²={r2_score(tun_target, preds_tun_mse):.5f}")
print(f"  Huber compression ratio:     {preds_tun_huber.std()/tun_target.std():.4f}")
print(f"  MSE   compression ratio:     {preds_tun_mse.std()/tun_target.std():.4f}")

def pitcher_score_std_from_preds(feat_df, preds_arr, min_p=MIN_PITCHES_TUN):
    df = feat_df[["pitcher_id"]].copy()
    df["pred"] = preds_arr
    m_ = df["pred"].mean()
    s_ = df["pred"].std()
    g = df.groupby("pitcher_id")["pred"].agg(["mean", "count"]).reset_index()
    g.columns = ["pitcher_id", "mean_pred", "n"]
    g = g[g["n"] >= min_p]
    scores = to_plus_scale(g["mean_pred"].values, m_, s_)
    return float(np.std(scores)), float(scores.max() - scores.min())

std_h, rng_h = pitcher_score_std_from_preds(test_tun, preds_tun_huber)
std_m, rng_m = pitcher_score_std_from_preds(test_tun, preds_tun_mse)
print(f"\n  Tunneling+ pitcher-score std   Huber={std_h:.3f}  MSE={std_m:.3f}")
print(f"  Tunneling+ pitcher-score range Huber={rng_h:.2f}  MSE={rng_m:.2f}")


# ─────────────────────────────────────────────────────────────────
# PART D: Huber vs MSE — Location+
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"PART D — Location+: Huber vs MSE (train on {HOLDOUT-1}/{HOLDOUT+1}, test on {HOLDOUT})")
print("="*70)

loc_df_d = pd.read_sql(f"""
    SELECT {', '.join(LOC_RAW_COLS)} FROM pitches
    WHERE season IN ({HOLDOUT-1}, {HOLDOUT}, {HOLDOUT+1}) AND delta_run_exp IS NOT NULL
""", conn)

print(f"Engineering features for {len(loc_df_d):,} pitches...")
loc_feat_d = loc_engineer(loc_df_d)
loc_feat_d["season"] = loc_feat_d["season"].astype(str)
for col in LOC_CAT_FEATURES:
    loc_feat_d[col] = loc_feat_d[col].astype(str)

train_loc = loc_feat_d[loc_feat_d["season"] != str(HOLDOUT)].copy()
test_loc = loc_feat_d[loc_feat_d["season"] == str(HOLDOUT)].copy()
print(f"  train rows: {len(train_loc):,}  test rows: {len(test_loc):,}")

for col in LOC_CAT_FEATURES:
    cats = sorted(set(train_loc[col][train_loc[col].notna()]) | set(test_loc[col][test_loc[col].notna()]))
    train_loc[col] = pd.Categorical(train_loc[col], categories=cats)
    test_loc[col] = pd.Categorical(test_loc[col], categories=cats)

def fit_loc(train, obj):
    fr, sr = train_test_split(train, test_size=0.08, random_state=42)
    m = lgb.LGBMRegressor(**{**LGBM_BASE, "objective": obj, "num_leaves": 31})
    m.fit(
        fr[LOC_ALL_FEATURES], fr["delta_run_exp"],
        eval_set=[(sr[LOC_ALL_FEATURES], sr["delta_run_exp"])],
        categorical_feature=LOC_CAT_FEATURES,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return m

print(f"\nFitting Location+ with Huber loss...")
m_loc_huber = fit_loc(train_loc, "huber")
preds_loc_huber = m_loc_huber.predict(test_loc[LOC_ALL_FEATURES])

print(f"Fitting Location+ with MSE loss...")
m_loc_mse = fit_loc(train_loc, "mse")
preds_loc_mse = m_loc_mse.predict(test_loc[LOC_ALL_FEATURES])

loc_target = test_loc["delta_run_exp"].values
print(f"\nLocation+ holdout={HOLDOUT} results:")
print(f"  Target std:                  {loc_target.std():.6f}")
print(f"  Huber raw pred std:          {preds_loc_huber.std():.6f}  R²={r2_score(loc_target, preds_loc_huber):.5f}")
print(f"  MSE   raw pred std:          {preds_loc_mse.std():.6f}  R²={r2_score(loc_target, preds_loc_mse):.5f}")
print(f"  Huber compression ratio:     {preds_loc_huber.std()/loc_target.std():.4f}")
print(f"  MSE   compression ratio:     {preds_loc_mse.std()/loc_target.std():.4f}")

def loc_pitcher_score_std(feat, preds, min_p=MIN_PITCHES_LOC):
    df = feat[["pitcher_id"]].copy()
    df["pred"] = preds
    m_ = df["pred"].mean()
    s_ = df["pred"].std()
    g = df.groupby("pitcher_id")["pred"].agg(["mean", "count"]).reset_index()
    g.columns = ["pitcher_id", "mean_pred", "n"]
    g = g[g["n"] >= min_p]
    scores = to_plus_scale(g["mean_pred"].values, m_, s_)
    return float(np.std(scores)), float(scores.max() - scores.min())

std_lh, rng_lh = loc_pitcher_score_std(test_loc, preds_loc_huber)
std_lm, rng_lm = loc_pitcher_score_std(test_loc, preds_loc_mse)
print(f"\n  Location+ pitcher-score std   Huber={std_lh:.3f}  MSE={std_lm:.3f}")
print(f"  Location+ pitcher-score range Huber={rng_lh:.2f}  MSE={rng_lm:.2f}")

conn.close()
print("\n\nDone.")
