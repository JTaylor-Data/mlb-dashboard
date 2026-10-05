"""
Score spread diagnosis for Tunneling+ and Location+.
Parts B, C, D of the task. Run from mlb_dashboard/ directory.
"""
import json
import sqlite3
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from analytics.location_features import (
    ALL_FEATURES as LOC_ALL_FEATURES,
    CATEGORICAL_FEATURES as LOC_CAT_FEATURES,
    NUMERIC_FEATURES as LOC_NUM_FEATURES,
    RAW_QUERY_COLUMNS as LOC_RAW_COLS,
    engineer_features as loc_engineer,
    shrinkage_params, shrunk_group_means, to_plus_scale,
)
from analytics.tunneling_features import (
    lag_feature_columns, load_pitches as tun_load_pitches, sequential_lag_features, MAX_LAG,
)
from analytics.stuff_model import compute_stuff_plus

SEASON = 2025
DB_PATH = "mlb_data.db"

MODELS_DIR = Path("models")
TUN_BOOSTER_PATH = MODELS_DIR / "tunneling_v3.txt"
TUN_META_PATH = MODELS_DIR / "tunneling_v3_meta.json"
TUN_COMMIT_TABLE_PATH = (
    Path("project_artifacts/training_artifacts/tunneling_v2_artifacts/200ms/pitcher_season_commit_point.csv")
)
LOC_BOOSTER_PATH = MODELS_DIR / "location_plus_v1.txt"
LOC_META_PATH = MODELS_DIR / "location_plus_v1_meta.json"

MIN_PITCHES_TUN = 100
MIN_PITCHES_LOC = 100
MIN_PITCHES_STUFF = 200


def pct_summary(scores, label):
    s = np.array(scores, dtype=float)
    p = [5, 25, 50, 75, 95]
    pcts = np.percentile(s, p)
    print(f"\n=== {label} score distribution (n={len(s)}) ===")
    print(f"  mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.2f}  max={s.max():.2f}")
    for pct, val in zip(p, pcts):
        print(f"  p{pct:02d}: {val:.2f}")
    in_99_101 = np.mean((s >= 99) & (s <= 101)) * 100
    in_95_105 = np.mean((s >= 95) & (s <= 105)) * 100
    print(f"  within [99,101]: {in_99_101:.1f}%")
    print(f"  within [95,105]: {in_95_105:.1f}%")
    return s


def raw_vs_target_summary(raw_preds, targets, label):
    print(f"\n=== {label} raw prediction vs. target variance ===")
    print(f"  raw pred std:   {np.std(raw_preds):.6f}")
    print(f"  target std:     {np.std(targets):.6f}")
    ratio = np.std(raw_preds) / np.std(targets) if np.std(targets) > 0 else float('nan')
    print(f"  ratio (pred/target): {ratio:.4f}  (1.0 = perfectly calibrated; << 1 = predictions too compressed)")
    r2 = r2_score(targets, raw_preds)
    print(f"  R² on this population: {r2:.6f}")


# ─────────────────────────────────────────────────────────────────
# TUNNELING+
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("PART B/C — TUNNELING+")
print("="*70)

conn = sqlite3.connect(DB_PATH)
tun_df = tun_load_pitches(conn, SEASON)

# qualification: same as live scoring
totals = tun_df.groupby("pitcher_id").size()
qualifying_ids = totals[totals >= MIN_PITCHES_TUN].index
tun_df_q = tun_df[tun_df["pitcher_id"].isin(qualifying_ids)]

commit_table = pd.read_csv(TUN_COMMIT_TABLE_PATH)
tun_model = lgb.Booster(model_file=str(TUN_BOOSTER_PATH))
tun_meta = json.loads(TUN_META_PATH.read_text())

LAG_COLS = lag_feature_columns()
NUMERIC_LAG_COLS = [c for c in LAG_COLS if not c.endswith("_pitch_type")]
CATEGORICAL_LAG_COLS = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CATEGORICAL = ["pitch_type", "p_throws", "stand", "season"]
CATEGORICAL_FEATURES_TUN = CATEGORICAL_LAG_COLS + CURRENT_CATEGORICAL
ALL_FEATURES_TUN = NUMERIC_LAG_COLS + CATEGORICAL_FEATURES_TUN

def _stringify_keep_na(series):
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)

def prepare_tun(df):
    feat = sequential_lag_features(df, commit_table, max_lag=MAX_LAG)
    feat["season"] = feat["season"].astype(str)
    for col in CATEGORICAL_FEATURES_TUN:
        feat[col] = _stringify_keep_na(feat[col])
        feat[col] = pd.Categorical(feat[col], categories=tun_meta["category_levels"][col])
    return feat

print(f"\nBuilding lag features for {SEASON}, {len(tun_df_q):,} qualifying pitches...")
tun_feat = prepare_tun(tun_df_q)

print(f"Scoring {len(tun_feat):,} lag-paired rows...")
tun_raw_preds = tun_model.predict(tun_feat[ALL_FEATURES_TUN])

# Per-pitcher scores, same as live leaderboard
tun_pred_df = tun_feat[["pitcher_id"]].copy()
tun_pred_df["pred"] = tun_raw_preds
league_mean = tun_pred_df["pred"].mean()
league_std = tun_pred_df["pred"].std()
g = tun_pred_df.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n=("pred", "size")).reset_index()
g["score"] = to_plus_scale(g["mean_pred"], league_mean, league_std)
tun_scores = pct_summary(g["score"].values, f"Tunneling+ {SEASON}")

# Part C: raw preds vs target
tun_feat_with_target = tun_feat.copy()
tun_feat_with_target["pred"] = tun_raw_preds
tun_target_col = tun_df_q.set_index(tun_df_q.index)["delta_run_exp"]
# Need to align: lag features drop first pitch of each PA
tun_feat_with_target = tun_feat_with_target.dropna(subset=["delta_run_exp"])
raw_vs_target_summary(
    tun_feat_with_target["pred"].values,
    tun_feat_with_target["delta_run_exp"].values,
    f"Tunneling+ {SEASON} (lag-paired rows with target)"
)

# Also: pitcher-mean raw pred std vs overall pitch-level raw pred std
print(f"\n  Per-pitcher MEAN raw pred std (what spreads the leaderboard): {g['mean_pred'].std():.6f}")
print(f"  Single-pitch raw pred std (full dist): {tun_raw_preds.std():.6f}")


# ─────────────────────────────────────────────────────────────────
# LOCATION+
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("PART B/C — LOCATION+")
print("="*70)

loc_df = pd.read_sql(f"""
    SELECT {', '.join(LOC_RAW_COLS)} FROM pitches
    WHERE season = ? AND delta_run_exp IS NOT NULL
""", conn, params=(SEASON,))

loc_meta = json.loads(LOC_META_PATH.read_text())
loc_model = lgb.Booster(model_file=str(LOC_BOOSTER_PATH))

def prepare_loc(df):
    feat = loc_engineer(df)
    for col in LOC_CAT_FEATURES:
        feat[col] = pd.Categorical(feat[col].astype(str), categories=loc_meta["category_levels"][col])
    return feat

print(f"\nEngineering Location+ features for {SEASON}, {len(loc_df):,} pitches...")
loc_feat = prepare_loc(loc_df)
loc_raw_preds = loc_model.predict(loc_feat[LOC_ALL_FEATURES])
loc_feat = loc_feat.copy()
loc_feat["pred"] = loc_raw_preds

# Shrinkage + per-pitcher scores (same as live official_location_plus)
raw_for_shrinkage = loc_feat[["pitcher_id", "pitch_type", "pred"]].copy()
k_by_type, fallback_k, type_means = shrinkage_params(raw_for_shrinkage)
shrunk = shrunk_group_means(raw_for_shrinkage, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
loc_league_mean = raw_for_shrinkage["pred"].mean()
loc_league_std = raw_for_shrinkage["pred"].std()
shrunk["score"] = to_plus_scale(shrunk["shrunk_mean"], loc_league_mean, loc_league_std)

# usage-weight to one score per pitcher
def _wavg(grp):
    return pd.Series({
        "score": np.average(grp["score"], weights=grp["n_pitches"]),
        "total_pitches": grp["n_pitches"].sum(),
    })
loc_pitcher_scores = shrunk.groupby("pitcher_id").apply(_wavg, include_groups=False).reset_index()
loc_pitcher_scores_q = loc_pitcher_scores[loc_pitcher_scores["total_pitches"] >= MIN_PITCHES_LOC]
loc_scores = pct_summary(loc_pitcher_scores_q["score"].values, f"Location+ {SEASON} (shrunk, qualified)")

# Also show unshrunk
per_pitcher_unshrunk = loc_feat.groupby("pitcher_id")["pred"].agg(["mean", "count"]).reset_index()
per_pitcher_unshrunk.columns = ["pitcher_id", "mean_pred", "n"]
per_pitcher_unshrunk_q = per_pitcher_unshrunk[per_pitcher_unshrunk["n"] >= MIN_PITCHES_LOC].copy()
per_pitcher_unshrunk_q["score"] = to_plus_scale(per_pitcher_unshrunk_q["mean_pred"], loc_league_mean, loc_league_std)
pct_summary(per_pitcher_unshrunk_q["score"].values, f"Location+ {SEASON} (UNSHRUNK, qualified)")

# Part C
raw_vs_target_summary(
    loc_raw_preds,
    loc_feat["delta_run_exp"].values,
    f"Location+ {SEASON} pitch-level"
)
print(f"\n  Per-pitcher MEAN raw pred std: {per_pitcher_unshrunk_q['mean_pred'].std():.6f}")
print(f"  Single-pitch raw pred std:     {loc_raw_preds.std():.6f}")
print(f"  Shrinkage K for FF (most common type): {k_by_type.get('FF', fallback_k):.1f} pitches")
print(f"  Shrinkage fallback K: {fallback_k:.1f}")


# ─────────────────────────────────────────────────────────────────
# STUFF+ (reference)
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("PART B/C — STUFF+ (reference)")
print("="*70)

from analytics.stuff_model import _shrinkage_params, _shrunk_group_means
from analytics.stuff_features import engineer_features as stuff_engineer, EXCLUDED_PITCH_TYPES, merge_pitch_type

stuff_df = pd.read_sql(f"""
    SELECT pitcher_id, pitch_type, p_throws, stand, season,
           release_speed, effective_speed, release_spin_rate, spin_axis,
           hb, ivb, vaa_residual, delta_run_exp
    FROM pitches
    WHERE season = ? AND delta_run_exp IS NOT NULL
""", conn, params=(SEASON,))
stuff_df = stuff_df.dropna()

from analytics.stuff_features import engineer_features as stuff_eng_fn
stuff_feat = stuff_eng_fn(stuff_df)

from analytics.stuff_model import MODELS_DIR as STUFF_MODELS_DIR, BOOSTER_PATH as STUFF_BOOSTER_PATH, META_PATH as STUFF_META_PATH
stuff_booster = lgb.Booster(model_file=str(STUFF_BOOSTER_PATH))
stuff_meta = json.loads(STUFF_META_PATH.read_text())

STUFF_CAT = ["pitch_type", "p_throws", "stand", "season"]
STUFF_NUM = ["release_speed", "effective_speed", "release_spin_rate", "spin_axis", "hb", "ivb", "vaa_residual",
             "velo_sep_vs_primary_fb", "hb_sep_vs_primary_fb", "ivb_sep_vs_primary_fb", "axis_differential"]
STUFF_ALL = STUFF_NUM + STUFF_CAT

# prep categoricals
for col in STUFF_CAT:
    stuff_feat[col] = pd.Categorical(stuff_feat[col].astype(str), categories=stuff_meta["category_levels"][col])

stuff_raw = stuff_booster.predict(stuff_feat[STUFF_ALL])
stuff_feat_out = stuff_feat[["pitcher_id", "pitch_type"]].copy()
stuff_feat_out["pred"] = stuff_raw

# Compute Stuff+ via shrinkage (reuse from analytics.stuff_model)
conn2 = sqlite3.connect(DB_PATH)
stuff_scores_all = compute_stuff_plus(conn2, season=SEASON)
conn2.close()
stuff_scores_q = stuff_scores_all[stuff_scores_all["n_pitches"] >= MIN_PITCHES_STUFF]
pct_summary(stuff_scores_q["score"].values, f"Stuff+ {SEASON} (shrunk, qualified ≥{MIN_PITCHES_STUFF})")

raw_vs_target_summary(
    stuff_raw,
    stuff_feat["delta_run_exp"].values,
    f"Stuff+ {SEASON} pitch-level"
)
per_pitcher_stuff = stuff_feat_out.groupby("pitcher_id")["pred"].agg(["mean", "count"]).reset_index()
per_pitcher_stuff.columns = ["pitcher_id", "mean_pred", "n"]
per_pitcher_stuff_q = per_pitcher_stuff[per_pitcher_stuff["n"] >= MIN_PITCHES_STUFF].copy()
stuff_league_mean = stuff_raw.mean()
stuff_league_std = stuff_raw.std()
per_pitcher_stuff_q["score_unshrunk"] = to_plus_scale(per_pitcher_stuff_q["mean_pred"], stuff_league_mean, stuff_league_std)
pct_summary(per_pitcher_stuff_q["score_unshrunk"].values, f"Stuff+ {SEASON} (UNSHRUNK, qualified)")
print(f"\n  Per-pitcher MEAN raw pred std: {per_pitcher_stuff_q['mean_pred'].std():.6f}")
print(f"  Single-pitch raw pred std:     {stuff_raw.std():.6f}")


# ─────────────────────────────────────────────────────────────────
# PART D: Huber vs MSE comparison for Tunneling+ and Location+
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("PART D — Huber vs MSE loss comparison (on held-out 2024)")
print("="*70)

HOLDOUT = 2024
LGBM_BASE = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
)

# ── Tunneling+ ──
print("\n--- Tunneling+ Huber vs MSE ---")
feat_all = None
try:
    import pandas as pd
    # Load a 2-season slice to keep runtime reasonable
    tun_df_d = pd.concat([
        tun_load_pitches(conn, s) for s in [HOLDOUT - 1, HOLDOUT, HOLDOUT + 1]
    ])
    tun_df_d = tun_df_d[tun_df_d["pitcher_id"].isin(
        tun_df_d.groupby("pitcher_id").size()[lambda x: x >= MIN_PITCHES_TUN].index
    )]
    all_seasons = sorted(tun_df_d["season"].unique())
    feat_all = prepare_tun(tun_df_d)
    feat_all = feat_all.dropna(subset=["delta_run_exp"])
    feat_all["season"] = feat_all["season"].astype(str)
    for col in CATEGORICAL_FEATURES_TUN:
        feat_all[col] = _stringify_keep_na(feat_all[col])
        cats = sorted(set(feat_all[col][feat_all[col].notna()]))
        feat_all[col] = pd.Categorical(feat_all[col], categories=cats)
except Exception as e:
    print(f"  WARNING: Failed to load multi-season data for Part D Tunneling: {e}")
    feat_all = None

if feat_all is not None and len(feat_all) > 1000:
    train_tun = feat_all[feat_all["season"] != str(HOLDOUT)].copy()
    test_tun = feat_all[feat_all["season"] == str(HOLDOUT)].copy()
    # share same category levels
    for col in CATEGORICAL_FEATURES_TUN:
        cats = sorted(set(train_tun[col][train_tun[col].notna()]) | set(test_tun[col][test_tun[col].notna()]))
        train_tun[col] = pd.Categorical(train_tun[col], categories=cats)
        test_tun[col] = pd.Categorical(test_tun[col], categories=cats)

    try:
        from sklearn.model_selection import train_test_split as tts
        def _fit_model(train, obj, n_leaves=63):
            fr, sr = tts(train, test_size=0.08, random_state=42)
            m = lgb.LGBMRegressor(**{**LGBM_BASE, "objective": obj, "num_leaves": n_leaves})
            m.fit(fr[ALL_FEATURES_TUN], fr["delta_run_exp"],
                  eval_set=[(sr[ALL_FEATURES_TUN], sr["delta_run_exp"])],
                  categorical_feature=CATEGORICAL_FEATURES_TUN,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            return m

        print(f"  Training Tunneling+ Huber on {len(train_tun):,} rows...")
        m_huber = _fit_model(train_tun, "huber")
        preds_huber = m_huber.predict(test_tun[ALL_FEATURES_TUN])
        r2_huber = r2_score(test_tun["delta_run_exp"], preds_huber)
        print(f"  Huber  → raw pred std on holdout: {preds_huber.std():.6f}  R²={r2_huber:.5f}")

        print(f"  Training Tunneling+ MSE on {len(train_tun):,} rows...")
        m_mse = _fit_model(train_tun, "mse")
        preds_mse = m_mse.predict(test_tun[ALL_FEATURES_TUN])
        r2_mse = r2_score(test_tun["delta_run_exp"], preds_mse)
        print(f"  MSE    → raw pred std on holdout: {preds_mse.std():.6f}  R²={r2_mse:.5f}")

        print(f"\n  Huber/MSE raw pred std ratio: {preds_huber.std()/preds_mse.std():.4f}")
        # Per-pitcher scores under each loss
        def pitcher_scores_from_preds(feat, preds, min_p=MIN_PITCHES_TUN):
            df = feat[["pitcher_id"]].copy()
            df["pred"] = preds
            m = df["pred"].mean(); s = df["pred"].std()
            g = df.groupby("pitcher_id")["pred"].agg(["mean", "count"]).reset_index()
            g.columns = ["pitcher_id", "mean_pred", "n"]
            g = g[g["n"] >= min_p]
            g["score"] = to_plus_scale(g["mean_pred"], m, s)
            return g["score"].std()

        std_huber = pitcher_scores_from_preds(test_tun, preds_huber)
        std_mse = pitcher_scores_from_preds(test_tun, preds_mse)
        print(f"\n  Tunneling+ pitcher-score std  Huber={std_huber:.3f}  MSE={std_mse:.3f}")
        tun_target_std = test_tun["delta_run_exp"].std()
        print(f"  Target std on holdout: {tun_target_std:.6f}")
        print(f"  Huber compression ratio: {preds_huber.std()/tun_target_std:.4f}")
        print(f"  MSE   compression ratio: {preds_mse.std()/tun_target_std:.4f}")
    except Exception as e:
        print(f"  WARNING: Part D Tunneling fit failed: {e}")
        import traceback; traceback.print_exc()
else:
    print("  Skipped (insufficient data loaded)")

# ── Location+ ──
print("\n--- Location+ Huber vs MSE ---")
try:
    loc_df_d = pd.read_sql(f"""
        SELECT {', '.join(LOC_RAW_COLS)} FROM pitches
        WHERE season IN ({HOLDOUT-1}, {HOLDOUT}, {HOLDOUT+1}) AND delta_run_exp IS NOT NULL
    """, conn)
    loc_feat_d = loc_engineer(loc_df_d)
    for col in LOC_CAT_FEATURES:
        cats = sorted(set(loc_feat_d[col][loc_feat_d[col].notna()]))
        loc_feat_d[col] = pd.Categorical(loc_feat_d[col].astype(str), categories=cats)
    loc_feat_d["season"] = loc_feat_d["season"].astype(str)
    train_loc = loc_feat_d[loc_feat_d["season"] != str(HOLDOUT)].copy()
    test_loc = loc_feat_d[loc_feat_d["season"] == str(HOLDOUT)].copy()
    for col in LOC_CAT_FEATURES:
        cats = sorted(set(train_loc[col][train_loc[col].notna()]) | set(test_loc[col][test_loc[col].notna()]))
        train_loc[col] = pd.Categorical(train_loc[col].astype(str), categories=cats)
        test_loc[col] = pd.Categorical(test_loc[col].astype(str), categories=cats)

    def _fit_loc(train, obj):
        fr, sr = tts(train, test_size=0.08, random_state=42)
        m = lgb.LGBMRegressor(**{**LGBM_BASE, "objective": obj, "num_leaves": 31})
        m.fit(fr[LOC_ALL_FEATURES], fr["delta_run_exp"],
              eval_set=[(sr[LOC_ALL_FEATURES], sr["delta_run_exp"])],
              categorical_feature=LOC_CAT_FEATURES,
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return m

    print(f"  Training Location+ Huber on {len(train_loc):,} rows...")
    m_loc_huber = _fit_loc(train_loc, "huber")
    lp_huber = m_loc_huber.predict(test_loc[LOC_ALL_FEATURES])
    r2_loc_huber = r2_score(test_loc["delta_run_exp"], lp_huber)
    print(f"  Huber  → raw pred std on holdout: {lp_huber.std():.6f}  R²={r2_loc_huber:.5f}")

    print(f"  Training Location+ MSE on {len(train_loc):,} rows...")
    m_loc_mse = _fit_loc(train_loc, "mse")
    lp_mse = m_loc_mse.predict(test_loc[LOC_ALL_FEATURES])
    r2_loc_mse = r2_score(test_loc["delta_run_exp"], lp_mse)
    print(f"  MSE    → raw pred std on holdout: {lp_mse.std():.6f}  R²={r2_loc_mse:.5f}")

    print(f"\n  Huber/MSE raw pred std ratio: {lp_huber.std()/lp_mse.std():.4f}")
    def loc_pitcher_score_std(feat, preds, min_p=MIN_PITCHES_LOC):
        df = feat[["pitcher_id"]].copy(); df["pred"] = preds
        m = df["pred"].mean(); s = df["pred"].std()
        g = df.groupby("pitcher_id")["pred"].mean()
        cts = df.groupby("pitcher_id").size()
        g = g[cts >= min_p]
        scores = to_plus_scale(g.values, m, s)
        return float(np.std(scores))
    std_l_huber = loc_pitcher_score_std(test_loc, lp_huber)
    std_l_mse = loc_pitcher_score_std(test_loc, lp_mse)
    print(f"\n  Location+ pitcher-score std  Huber={std_l_huber:.3f}  MSE={std_l_mse:.3f}")
    loc_target_std = test_loc["delta_run_exp"].std()
    print(f"  Target std on holdout: {loc_target_std:.6f}")
    print(f"  Huber compression ratio: {lp_huber.std()/loc_target_std:.4f}")
    print(f"  MSE   compression ratio: {lp_mse.std()/loc_target_std:.4f}")

except Exception as e:
    print(f"  WARNING: Part D Location fit failed: {e}")
    import traceback; traceback.print_exc()

conn.close()
print("\n\nDone.")
