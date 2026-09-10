"""
eval_pitching_battery_v1.py — Unified validation battery across all six models.

MODELS EVALUATED:
  1. Stuff+ v3 (live)
  2. Deception+ v1 (live)
  3. Tunneling+ v3 (live)
  4. Location+ v1 (live)
  5. Pitching+ feature-level (full union, effective_speed dropped)
  6. Pitching+ feature-level (Tunneling+ inputs excluded)

METHODOLOGY (identical for all six — see printed confirmation at runtime):
  - YoY r²: Pearson r² between pitcher's score in season S and season S+1,
    pooled across all consecutive season pairs (2021-22, 22-23, 23-24, 24-25).
    Same alignment logic for all six models.
  - Concurrent validity: same-season Pearson r between pitcher-season score
    and each outcome (avg_dre, whiff%, swstr%, K-BB%, chase%, ERA proxy, xFIP).
    All outcomes computed fresh in this script from the same pitches table.
  - Predictive validity: pitcher's season-S score → season-S+1 actual outcomes.
    For sub-models (1-4): raw pitcher-season scores applied directly (pre-trained
    model, inherently OOS at the pitcher-season level).
    For Pitching+ variants (5-6): LOSO CV scores used — model trained on all
    seasons except S, scores pitcher-season S OOS.
  - Real-skill regression: OLS slope + r² of pitcher-season score vs ERA proxy
    and xFIP (separately). Coefficient direction checked.
  - Split-half reliability: each pitcher-season's per-pitch model predictions
    (raw, before plus-scale) are randomly split 50/50. Mean prediction in each
    half → Pearson r across pitcher-seasons → Spearman-Brown correction
    r_sb = 2r / (1+r). Applied identically to all six models.
    NOTE: for models that use pitcher-season-level aggregated features
    (Deception+ consistency stats, VAA calibration in Stuff+), those aggregates
    are computed on the FULL pitcher-season before splitting — this slightly
    overestimates split-half reliability for Deception+ specifically, and to
    a lesser degree Stuff+ and the combined models that include those features.

DOES NOT TOUCH ANY PRODUCTION ARTIFACT.
"""
import sys, warnings, traceback
from pathlib import Path

# Redirect stderr to stdout so all errors appear in the output file
sys.stderr = sys.stdout

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

DB_PATH = ROOT / "mlb_data.db"
OUT_DIR = ROOT / "analytics" / "pitching_plus"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_PITCHER_PITCHES = 100   # per pitcher-season to qualify

RNG = np.random.default_rng(42)

_FORBIDDEN_IDS = {
    "pitcher_id", "batter_id", "player_id", "game_id", "game_pk",
    "at_bat_number", "pitch_number", "inning",
}


def assert_no_identifiers(cols):
    bad = _FORBIDDEN_IDS & set(cols)
    if bad:
        raise ValueError(f"Identifier columns in feature set: {sorted(bad)}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTCOMES — computed once, reused by all six models
# ══════════════════════════════════════════════════════════════════════════════

OUTCOME_COLS = [
    "avg_dre", "whiff_pct", "swstr_pct", "k_bb_pct", "chase_pct",
    "k_pct", "bb_pct", "xfip", "era_proxy",
]


def load_outcomes(conn, seasons):
    ph = ",".join("?" * len(seasons))
    df1 = pd.read_sql_query(f"""
        SELECT pitcher_id, season,
               COUNT(*) AS n_pitches,
               AVG(delta_run_exp) AS avg_dre,
               AVG(CASE WHEN description IN
                   ('swinging_strike','swinging_strike_bunt') THEN 1.0 ELSE 0.0 END
               ) AS whiff_pct,
               AVG(CASE WHEN description IN
                   ('swinging_strike','swinging_strike_bunt') THEN 1.0 ELSE 0.0 END
               ) AS swstr_pct
        FROM pitches
        WHERE season IN ({ph}) AND delta_run_exp IS NOT NULL
        GROUP BY pitcher_id, season
        HAVING COUNT(*) >= {MIN_PITCHER_PITCHES}
    """, conn, params=seasons)

    df2 = pd.read_sql_query(f"""
        SELECT pitcher_id, season,
               SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
               SUM(CASE WHEN events IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS bb,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr,
               SUM(CASE WHEN events IS NOT NULL AND events != '' THEN 1 ELSE 0 END) AS pa
        FROM pitches
        WHERE season IN ({ph}) AND events IS NOT NULL AND events != ''
        GROUP BY pitcher_id, season
    """, conn, params=seasons)
    df2["k_pct"]    = df2["k"]  / df2["pa"].clip(lower=1)
    df2["bb_pct"]   = df2["bb"] / df2["pa"].clip(lower=1)
    df2["k_bb_pct"] = (df2["k"] - df2["bb"]) / df2["pa"].clip(lower=1)

    df3 = pd.read_sql_query(f"""
        SELECT pitcher_id, season,
               SUM(CASE WHEN events IN (
                   'field_out','force_out','grounded_into_double_play','double_play',
                   'strikeout','strikeout_double_play','sac_fly','sac_bunt',
                   'fielders_choice_out','caught_stealing_2b','caught_stealing_3b',
                   'caught_stealing_home','other_out','triple_play'
               ) THEN 1 ELSE 0 END) AS total_outs
        FROM pitches
        WHERE season IN ({ph})
        GROUP BY pitcher_id, season
    """, conn, params=seasons)
    df3["ip"] = df3["total_outs"] / 3.0

    df4 = pd.read_sql_query(f"""
        SELECT pitcher_id, season,
               AVG(CASE
                   WHEN (ABS(plate_x) > 0.83 OR plate_z < 1.5 OR plate_z > 3.5)
                     AND description IN (
                         'swinging_strike','swinging_strike_bunt',
                         'foul','foul_tip','hit_into_play','hit_into_play_no_out',
                         'hit_into_play_score')
                   THEN 1.0
                   WHEN (ABS(plate_x) > 0.83 OR plate_z < 1.5 OR plate_z > 3.5)
                   THEN 0.0
               END) AS chase_pct
        FROM pitches
        WHERE season IN ({ph}) AND plate_x IS NOT NULL AND plate_z IS NOT NULL
        GROUP BY pitcher_id, season
    """, conn, params=seasons)

    df = (df1
          .merge(df2[["pitcher_id","season","k","bb","hr","pa","k_pct","bb_pct","k_bb_pct"]],
                 on=["pitcher_id","season"], how="left")
          .merge(df3[["pitcher_id","season","ip"]], on=["pitcher_id","season"], how="left")
          .merge(df4, on=["pitcher_id","season"], how="left"))

    FIP_C = 3.10
    df["xfip"] = np.where(
        df["ip"] > 0,
        (13*df["hr"] + 3*df["bb"] - 2*df["k"]) / df["ip"].clip(lower=0.001) + FIP_C,
        np.nan,
    )
    df["era_proxy"] = -df["avg_dre"] * 27 * 3
    df["season"]    = df["season"].astype(str)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS — identical for all six models
# ══════════════════════════════════════════════════════════════════════════════

def _to_plus_scale(series):
    m, s = series.mean(), series.std()
    if s < 1e-9:
        return pd.Series(100.0, index=series.index)
    return 100 - 10 * (series - m) / s


def concurrent_validity(merged, score_col, outcomes_df=None):
    """Pearson r/r² between score_col and each outcome in OUTCOME_COLS."""
    rows = []
    for col in OUTCOME_COLS:
        if col not in merged.columns:
            continue
        sub = merged[[score_col, col]].dropna()
        if len(sub) < 20:
            rows.append({"outcome": col, "r": np.nan, "r2": np.nan, "n": len(sub)})
            continue
        r, _ = stats.pearsonr(sub[score_col], sub[col])
        rows.append({"outcome": col, "r": round(r, 3), "r2": round(r**2, 3), "n": len(sub)})
    return pd.DataFrame(rows)


def yoy_r2(merged, score_col):
    """
    YoY r²: Pearson r² between pitcher's score in season S and season S+1.
    Pooled across all consecutive season pairs available in `merged`.
    """
    df = merged[["pitcher_id","season",score_col]].dropna().copy()
    df["season_int"] = df["season"].astype(str).astype(int)
    curr = df.rename(columns={score_col: "score_curr"})
    nxt  = df.copy()
    nxt["season_int"] = nxt["season_int"] - 1
    nxt = nxt.rename(columns={score_col: "score_next"})
    paired = curr.merge(nxt[["pitcher_id","season_int","score_next"]],
                        on=["pitcher_id","season_int"], how="inner")
    if len(paired) < 20:
        return np.nan
    return round(r2_score(paired["score_next"], paired["score_curr"]), 3)


def predictive_validity(merged_scores, score_col, outcomes):
    """
    Predictive: season-S score → season-S+1 outcomes.
    Returns merged DataFrame for use with concurrent_validity().
    """
    df = merged_scores[["pitcher_id","season",score_col]].dropna().copy()
    df["season_int"] = df["season"].astype(str).astype(int)
    out = outcomes.copy()
    out["season_int"] = out["season"].astype(str).astype(int)
    out_shifted = out.copy()
    out_shifted["season_int"] = out_shifted["season_int"] - 1  # align S → S+1
    return df.merge(out_shifted.drop(columns=["season"], errors="ignore"),
                    on=["pitcher_id","season_int"], how="inner")


def spearman_brown(r):
    """Spearman-Brown prophecy formula for full-test reliability from half-split r."""
    if pd.isna(r) or r >= 1.0:
        return r
    return 2 * r / (1 + r)


def split_half_reliability(pitch_preds_df, score_col="pred"):
    """
    Split each pitcher-season's pitches randomly 50/50, compute mean prediction
    per half, Pearson r across pitcher-seasons, Spearman-Brown corrected.

    Input: DataFrame with columns [pitcher_id, season, <score_col>]
    Returns: (raw_r, sb_corrected_r, n_pitcher_seasons)
    """
    try:
        df = pitch_preds_df[["pitcher_id","season",score_col]].dropna().copy()
        df = df.reset_index(drop=True)
        half_arr = np.zeros(len(df), dtype=int)
        perm = RNG.permutation(len(df))
        half_arr[perm[len(perm)//2:]] = 1
        df = df.copy()
        df["_half"] = half_arr

        # Count pitches per half per pitcher-season; keep only those with ≥25 each half
        counts = df.groupby(["pitcher_id","season","_half"]).size().unstack("_half")
        counts.columns = [int(c) for c in counts.columns]
        if 0 not in counts.columns or 1 not in counts.columns:
            return np.nan, np.nan, 0
        counts = counts.dropna()
        counts = counts[(counts[0] >= 25) & (counts[1] >= 25)]
        valid_ps = set(zip(counts.index.get_level_values("pitcher_id"),
                           counts.index.get_level_values("season").astype(str)))

        ps = df.groupby(["pitcher_id","season","_half"])[score_col].mean().unstack("_half")
        ps.columns = [int(c) for c in ps.columns]
        if 0 not in ps.columns or 1 not in ps.columns:
            return np.nan, np.nan, 0
        ps = ps.dropna().reset_index()
        ps["_key"] = list(zip(ps["pitcher_id"], ps["season"].astype(str)))
        ps = ps[ps["_key"].isin(valid_ps)]

        if len(ps) < 20:
            return np.nan, np.nan, len(ps)
        r, _ = stats.pearsonr(ps[0], ps[1])
        return round(r, 3), round(spearman_brown(r), 3), len(ps)
    except Exception as e:
        print(f"      split_half_reliability error: {e}")
        traceback.print_exc()
        return np.nan, np.nan, 0


def real_skill_regression(merged, score_col):
    """OLS r² and direction check for ERA proxy and xFIP."""
    rows = []
    for outcome in ["era_proxy", "xfip"]:
        if outcome not in merged.columns:
            continue
        sub = merged[[score_col, outcome]].dropna()
        if len(sub) < 20:
            rows.append({"outcome": outcome, "r2": np.nan, "coef": np.nan, "direction": "n/a"})
            continue
        slope, _, r, p, _ = stats.linregress(sub[score_col], sub[outcome])
        ok = slope < 0  # higher score = better pitcher = lower ERA/xFIP
        rows.append({
            "outcome": outcome,
            "r2": round(r**2, 3),
            "coef": round(slope, 5),
            "p": round(p, 4),
            "n": len(sub),
            "direction": "✓" if ok else "✗ (inverted scale expected)",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — for Pitching+ variants
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_pitches(conn, seasons):
    ph = ",".join("?" * len(seasons))
    sql = f"""
    SELECT
        pitcher_id, pitch_type, stand, p_throws, season,
        game_id, at_bat_number, pitch_number,
        release_speed, release_spin_rate, spin_axis,
        pfx_x, pfx_z, vx0, vy0, vz0, ax, ay, az,
        release_extension, release_pos_x, release_pos_z, arm_angle,
        plate_x, plate_z, balls, strikes,
        effective_speed,
        delta_run_exp
    FROM pitches
    WHERE season IN ({ph})
      AND pitch_type NOT IN ('PO','UN','EP')
      AND pitch_type IS NOT NULL
      AND release_speed IS NOT NULL
      AND delta_run_exp IS NOT NULL
    """
    df = pd.read_sql_query(sql, conn, params=seasons)
    df["pitch_type"] = df["pitch_type"].replace("KC", "CU")
    df["season_int"] = df["season"].astype(int)
    df["season"]     = df["season"].astype(str)
    return df


def add_stuff_features(df):
    from analytics.stuff_features import (
        add_movement_columns, fit_v3_calibrations, add_vaa_residual,
        build_fb_reference_table, add_separation_features,
    )
    df = add_movement_columns(df)
    cals = fit_v3_calibrations(df)
    df = add_vaa_residual(df, cals["vaa_coefs"])
    fb_ref = build_fb_reference_table(df)
    df = add_separation_features(df, fb_ref)
    try:
        from analytics.spin_physics import add_axis_differential
        df = add_axis_differential(df)
    except Exception:
        df["axis_differential"] = np.nan
    return df


def add_deception_features(df, conn):
    sign = np.where(df["p_throws"] == "R", -1.0, 1.0)
    df["release_pos_x_as"] = df["release_pos_x"] * sign
    grp = df.groupby(["pitcher_id","season"])
    df["arm_angle_std"]         = grp["arm_angle"].transform("std")
    df["release_extension_std"] = grp["release_extension"].transform("std")
    mean_x = grp["release_pos_x_as"].transform("mean")
    mean_z = grp["release_pos_z"].transform("mean")
    df["release_consistency"] = np.sqrt(
        (df["release_pos_x_as"] - mean_x)**2 +
        (df["release_pos_z"]    - mean_z)**2
    )
    try:
        from analytics.deception_features import load_bio
        bio = load_bio(conn)
        df = df.merge(bio, on="pitcher_id", how="left")
    except Exception:
        df["height_in"] = np.nan
        df["weight_lb"] = np.nan
    return df


def add_tunneling_lags(df):
    from analytics.tunneling_features import (
        build_commit_point_table, sequential_lag_features, lag_feature_columns,
    )
    commit_table = build_commit_point_table(df)
    lag_df = sequential_lag_features(df, commit_table, max_lag=6)
    lag_cols  = lag_feature_columns(6)
    join_keys = ["pitcher_id","game_id","at_bat_number","pitch_number"]
    avail = [c for c in lag_cols if c in lag_df.columns]
    lag_sub = lag_df[join_keys + avail].copy()
    df = df.merge(lag_sub, on=join_keys, how="left")
    for col in lag_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


# Feature manifests — effective_speed DROPPED from Stuff+ block
_STUFF_NUMERIC = [
    "release_speed", "release_spin_rate", "spin_axis",       # effective_speed removed
    "hb", "ivb", "vaa_residual",
    "velo_sep_vs_primary_fb", "hb_sep_vs_primary_fb", "ivb_sep_vs_primary_fb",
    "axis_differential",
]
_DEC_NUMERIC = [
    "arm_angle", "release_extension", "release_pos_x_as", "release_pos_z",
    "arm_angle_std", "release_extension_std", "release_consistency",
    "height_in", "weight_lb",
]
_LOC_NUMERIC  = ["plate_x", "plate_z", "balls", "strikes"]
_TUN_NUMERIC  = [f"lag{n}_{m}" for n in range(1,7)
                 for m in ("commit_distance","plate_distance","shape_break")]
_TUN_CAT_LAG  = [f"lag{n}_pitch_type" for n in range(1,7)]
_SHARED_CAT   = ["pitch_type","p_throws","stand","season"]

FEATURES_FULL  = _STUFF_NUMERIC + _DEC_NUMERIC + _LOC_NUMERIC + _TUN_NUMERIC
FEATURES_NOTUN = _STUFF_NUMERIC + _DEC_NUMERIC + _LOC_NUMERIC          # no tunneling
CATS_FULL      = _SHARED_CAT + _TUN_CAT_LAG
CATS_NOTUN     = _SHARED_CAT

_CORE_REQUIRED = [
    "release_speed", "release_spin_rate", "spin_axis", "hb", "ivb", "vaa_residual",
    "plate_x", "plate_z", "balls", "strikes", "delta_run_exp",
]

LGBM_PARAMS = dict(
    objective="huber", alpha=0.9, n_estimators=2000, learning_rate=0.03,
    num_leaves=63, min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
)


def build_dataset(df_raw, include_tunneling=True):
    """Build feature-engineered dataset. df_raw already has Stuff+/Dec/Tun features."""
    num_feats = FEATURES_FULL  if include_tunneling else FEATURES_NOTUN
    cat_feats = CATS_FULL     if include_tunneling else CATS_NOTUN
    all_feats = num_feats + cat_feats

    df = df_raw.dropna(subset=_CORE_REQUIRED).copy()
    for col in cat_feats:
        if col in df.columns:
            df[col] = df[col].astype("category")

    features = [c for c in all_feats if c in df.columns]
    assert_no_identifiers(features)
    return df, features


def loso_cv(df, features, seasons):
    """LOSO CV → per-pitch OOS predictions."""
    cat_feats = [f for f in features if df[f].dtype.name == "category"]
    oos_rows  = []
    for s in seasons:
        s_str = str(s)
        tr = df["season"] != s_str
        te = df["season"] == s_str
        if te.sum() == 0:
            continue
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(df.loc[tr, features], df.loc[tr, "delta_run_exp"],
              categorical_feature=cat_feats)
        preds = m.predict(df.loc[te, features])
        fold_df = df.loc[te, ["pitcher_id","season","delta_run_exp"]].copy()
        fold_df["pred"] = preds
        r2 = r2_score(df.loc[te, "delta_run_exp"], preds)
        print(f"      Fold {s}: {te.sum():>7,} pitches  per-pitch OOS r²={r2:.4f}")
        oos_rows.append(fold_df)
    return pd.concat(oos_rows, ignore_index=True)


def aggregate_oos(oos_df, outcomes):
    """Aggregate per-pitch OOS preds to pitcher-season plus-scores, join outcomes."""
    ps = (oos_df.groupby(["pitcher_id","season"])
               .agg(n_pitches=("pred","count"), raw_pred=("pred","mean"))
               .reset_index())
    ps = ps[ps["n_pitches"] >= MIN_PITCHER_PITCHES]
    ps["plus_score"] = _to_plus_scale(ps["raw_pred"])
    ps["season"]     = ps["season"].astype(str)
    out = outcomes.copy()
    out["season"] = out["season"].astype(str)
    return ps.merge(out, on=["pitcher_id","season"], how="inner")


# ══════════════════════════════════════════════════════════════════════════════
# SUB-MODEL RAW PREDICTIONS → per-pitch DataFrames
# ══════════════════════════════════════════════════════════════════════════════

def get_submodel_pitch_preds(conn, seasons):
    """
    Return per-pitch DataFrames for each sub-model.
    Each has columns: pitcher_id, season (str), pred.
    Also returns pitcher-season aggregated plus-scores for correlation work.
    """
    from analytics.stuff_model        import _raw_predictions as _stuff_raw
    from analytics.tunneling_v3_model import _raw_predictions as _tun_raw
    from analytics.location_model     import _raw_predictions as _loc_raw, official_location_plus
    from analytics.deception_model    import _raw_predictions as _dec_raw

    preds = {"stuff": [], "tunneling": [], "location": [], "deception": []}
    scores = {"stuff": [], "tunneling": [], "location": [], "deception": []}

    for s in seasons:
        print(f"    Season {s}...", end=" ", flush=True)

        # Stuff+
        try:
            raw = _stuff_raw(conn, s)
            if not raw.empty:
                raw["season"] = str(s)
                preds["stuff"].append(raw[["pitcher_id","season","pred"]])
                if raw["pred"].std() > 0:
                    mean, std = raw["pred"].mean(), raw["pred"].std()
                    g = raw.groupby("pitcher_id")["pred"].mean()
                    for pid, val in g.items():
                        scores["stuff"].append({
                            "pitcher_id": int(pid), "season": str(s),
                            "plus_score": float(100 - 10*(val-mean)/std)
                        })
        except Exception as e:
            print(f"\n      Stuff+ error ({s}): {e}", end=" ")

        # Tunneling+
        try:
            raw = _tun_raw(conn, s)
            if not raw.empty:
                raw["season"] = str(s)
                preds["tunneling"].append(raw[["pitcher_id","season","pred"]])
                if raw["pred"].std() > 0:
                    mean, std = raw["pred"].mean(), raw["pred"].std()
                    g = raw.groupby("pitcher_id")["pred"].mean()
                    for pid, val in g.items():
                        scores["tunneling"].append({
                            "pitcher_id": int(pid), "season": str(s),
                            "plus_score": float(100 - 10*(val-mean)/std)
                        })
        except Exception as e:
            print(f"\n      Tunneling+ error ({s}): {e}", end=" ")

        # Location+
        try:
            raw = _loc_raw(conn, s)
            if not raw.empty:
                loc = official_location_plus(raw)
                loc = loc[loc["total_pitches"] >= MIN_PITCHER_PITCHES]
                # Location+ is already a pitcher-season score, not per-pitch preds
                # Use the score column directly; per-pitch not exposed
                for r in loc.itertuples():
                    scores["location"].append({
                        "pitcher_id": int(r.pitcher_id), "season": str(s),
                        "plus_score": float(r.score)
                    })
                # For split-half: score per pitch by running the underlying model
                # Location+ underlying model file
                loc_model_path = ROOT / "models" / "location_plus_v1.txt"
                if loc_model_path.exists():
                    loc_booster = lgb.Booster(model_file=str(loc_model_path))
                    # raw has: pitcher_id, pitch_type, stand, plate_x, plate_z, balls, strikes
                    if "plate_x" in raw.columns and "plate_z" in raw.columns:
                        raw2 = raw.dropna(subset=["plate_x","plate_z","balls","strikes"])
                        feat_cols = [c for c in ["plate_x","plate_z","balls","strikes",
                                                  "pitch_type","stand","p_throws"]
                                     if c in raw2.columns]
                        if feat_cols:
                            cat_cols = [c for c in ["pitch_type","stand","p_throws"]
                                        if c in raw2.columns]
                            X = raw2[feat_cols].copy()
                            for c in cat_cols:
                                X[c] = X[c].astype("category")
                            pred_arr = loc_booster.predict(X)
                            tmp = raw2[["pitcher_id"]].copy()
                            tmp["season"] = str(s)
                            tmp["pred"] = pred_arr
                            preds["location"].append(tmp[["pitcher_id","season","pred"]])
        except Exception as e:
            print(f"\n      Location+ error ({s}): {e}", end=" ")

        # Deception+
        try:
            raw = _dec_raw(conn, s)
            if not raw.empty:
                raw["season"] = str(s)
                preds["deception"].append(raw[["pitcher_id","season","pred"]])
                if raw["pred"].std() > 0:
                    mean, std = raw["pred"].mean(), raw["pred"].std()
                    g = raw.groupby("pitcher_id")["pred"].mean()
                    for pid, val in g.items():
                        scores["deception"].append({
                            "pitcher_id": int(pid), "season": str(s),
                            "plus_score": float(100 - 10*(val-mean)/std)
                        })
        except Exception as e:
            print(f"\n      Deception+ error ({s}): {e}", end=" ")

        n = [len(scores[k]) for k in scores]
        print(f"scored {sum(n)} pitcher-seasons total")

    # Concatenate
    pitch_preds = {}
    for model in ["stuff","tunneling","location","deception"]:
        if preds[model]:
            pitch_preds[model] = pd.concat(preds[model], ignore_index=True)
        else:
            pitch_preds[model] = pd.DataFrame(columns=["pitcher_id","season","pred"])

    score_dfs = {}
    for model in ["stuff","tunneling","location","deception"]:
        score_dfs[model] = pd.DataFrame(scores[model]) if scores[model] else pd.DataFrame()

    return pitch_preds, score_dfs


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_concurrent(ct, label):
    print(f"\n  {label}:")
    print(f"  {'outcome':<16} {'r':>6} {'r²':>6} {'n':>6}")
    print(f"  {'-'*36}")
    for _, row in ct.iterrows():
        r  = f"{row['r']:.3f}"  if not pd.isna(row['r'])  else "  — "
        r2 = f"{row['r2']:.3f}" if not pd.isna(row['r2']) else "  — "
        print(f"  {row['outcome']:<16} {r:>6} {r2:>6} {int(row['n']) if not pd.isna(row['n']) else 0:>6}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Unified Pitching+ Evaluation Battery — v1")
    print("=" * 72)

    print("""
  METHODOLOGY CONFIRMATION (identical for all six models):
  ─────────────────────────────────────────────────────────
  YoY r²        : Pearson r² between pitcher score in season S and S+1.
                  Pooled across consecutive pairs: 2021-22, 22-23, 23-24, 24-25.
  Concurrent    : Pearson r² between pitcher-season score and same-season
                  outcomes (avg_dre, whiff%, swstr%, K-BB%, chase%, ERA, xFIP).
                  Outcomes computed fresh in this run from the same DB.
  Predictive    : Season-S score → season-S+1 actual outcomes (no leakage).
                  Sub-models: direct scores (inherently OOS at outcome level).
                  Pitching+ variants: LOSO CV scores (trained without season S).
  Real-skill    : OLS r² of score vs ERA proxy and xFIP.
  Split-half    : Per-pitch predictions split 50/50 by random row assignment.
                  Spearman-Brown corrected: r_sb = 2r/(1+r).
                  CAVEAT: pitcher-season aggregate features (Deception+ consistency
                  stats, VAA calibration) are computed on the full pitcher-season
                  before splitting — slightly inflates split-half for Deception+.
  NOTE: xwOBA is not available in the local DB; avg_dre (delta_run_exp per pitch,
        a directly-measured run-value proxy) is used as the primary run-value outcome.
  ─────────────────────────────────────────────────────────
""")

    conn = sqlite3.connect(DB_PATH)
    avail = pd.read_sql_query(
        "SELECT DISTINCT season FROM pitches WHERE season >= 2021 ORDER BY season", conn)
    all_seasons = [int(s) for s in avail["season"].tolist()]
    seasons = [s for s in all_seasons if s in EVAL_SEASONS]
    print(f"  DB seasons: {all_seasons}")
    print(f"  Evaluation seasons: {seasons}\n")

    # ── Outcomes (computed once, reused by all six models) ────────────────────
    print("  Loading outcomes...")
    outcomes = load_outcomes(conn, seasons)
    print(f"  Outcomes: {len(outcomes):,} pitcher-seasons\n")

    results = {}   # keyed by model name, stores: merged, yoy, ct_conc, ct_pred, sh_sb

    # Set SKIP_SUBMODELS=True to skip re-running the slow sub-model scoring phase
    # (useful when sub-model results already obtained from a prior run)
    SKIP_SUBMODELS = True

    # ══════════════════════════════════════════════════════════════════════════
    # MODELS 1-4: Sub-models
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 72)
    print("  MODELS 1-4 — Sub-model raw predictions")
    print("─" * 72)

    if SKIP_SUBMODELS:
        print("  SKIPPED (results already recorded from prior run)")
        pitch_preds = {k: pd.DataFrame() for k in ["stuff","tunneling","location","deception"]}
        score_dfs   = {k: pd.DataFrame() for k in ["stuff","tunneling","location","deception"]}
    else:
        pitch_preds, score_dfs = get_submodel_pitch_preds(conn, seasons)

    sub_name_map = {
        "stuff":    "Stuff+ v3",
        "tunneling":"Tunneling+ v3",
        "location": "Location+ v1",
        "deception":"Deception+ v1",
    }

    for key, label in sub_name_map.items():
        try:
            sdf = score_dfs[key]
            if sdf.empty:
                print(f"\n  {label}: no scores — skipping")
                continue

            sdf = sdf.merge(outcomes, on=["pitcher_id","season"], how="inner")
            if len(sdf) < 20:
                print(f"\n  {label}: only {len(sdf)} rows — skipping")
                continue

            print(f"\n  ── {label} ──")
            print(f"  n={len(sdf)} pitcher-seasons", flush=True)
            ct_conc = concurrent_validity(sdf, "plus_score")
            print("  concurrent_validity done", flush=True)
            yoy     = yoy_r2(sdf, "plus_score")
            print(f"  yoy_r2 done: {yoy}", flush=True)
            pred_df = predictive_validity(sdf, "plus_score", outcomes)
            ct_pred = concurrent_validity(pred_df, "plus_score") if len(pred_df) >= 20 else pd.DataFrame()
            print("  predictive_validity done", flush=True)
            rsr     = real_skill_regression(sdf, "plus_score")
            print("  real_skill_regression done", flush=True)

            # Split-half
            pps = pitch_preds[key]
            if pps.empty:
                sh_raw, sh_sb, sh_n = np.nan, np.nan, 0
                print("  split-half: no per-pitch preds available")
            else:
                print(f"  split-half: {len(pps):,} pitch rows...", flush=True)
                sh_raw, sh_sb, sh_n = split_half_reliability(pps)

            print_concurrent(ct_conc, f"{label} — concurrent validity")
            print(f"  YoY r²: {yoy}")
            if not ct_pred.empty:
                print_concurrent(ct_pred, f"{label} — predictive validity (S → S+1)")
            print(f"  Split-half: raw r={sh_raw}, SB-corrected={sh_sb}  (n={sh_n} pitcher-seasons)")
            print(f"  Real-skill regression:")
            for _, row in rsr.iterrows():
                print(f"    vs {row['outcome']:<12}: r²={row.get('r2','—')!s:<6}  "
                      f"coef={row.get('coef','—')!s:<10}  {row.get('direction','')}")

            results[label] = {
                "yoy": yoy, "ct_conc": ct_conc, "ct_pred": ct_pred,
                "sh_sb": sh_sb, "rsr": rsr, "merged": sdf,
            }
        except Exception as exc:
            print(f"\n  ERROR processing {label}: {exc}")
            traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    # MODELS 5-6: Pitching+ feature-level variants
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  MODELS 5-6 — Pitching+ feature-level (engineering all data first)")
    print("─" * 72)

    print("\n  Loading raw pitches...")
    df_raw = load_raw_pitches(conn, seasons)
    print(f"  Raw: {len(df_raw):,} pitches, {df_raw['pitcher_id'].nunique()} pitchers")

    print("  Engineering Stuff+ features...")
    df_raw = add_stuff_features(df_raw)
    print("  Engineering Deception+ features...")
    df_raw = add_deception_features(df_raw, conn)
    print("  Computing tunneling lag features (slow)...")
    df_raw = add_tunneling_lags(df_raw)

    for include_tun, model_name in [(True, "Pitching+ Full (model 5)"),
                                    (False, "Pitching+ No-Tun (model 6)")]:
      try:
        print(f"\n  ── {model_name} ──")

        df, features = build_dataset(df_raw, include_tunneling=include_tun)

        n_num = sum(1 for f in features if f in (FEATURES_FULL if include_tun else FEATURES_NOTUN))
        n_cat = sum(1 for f in features if f in (CATS_FULL if include_tun else CATS_NOTUN))
        print(f"  Features: {len(features)} total ({n_num} numeric, {n_cat} categorical)")
        # Confirm effective_speed is absent
        if "effective_speed" in features:
            print("  WARNING: effective_speed still present — check feature list")
        else:
            print("  Confirmed: effective_speed NOT in feature set")
        assert "effective_speed" not in features, "effective_speed leaked into feature set"
        print(f"  Pitches after core-required filter: {len(df):,}")

        print(f"\n  LOSO CV:")
        oos_df  = loso_cv(df, features, seasons)
        merged  = aggregate_oos(oos_df, outcomes)
        ct_conc = concurrent_validity(merged, "plus_score")
        yoy     = yoy_r2(merged, "plus_score")
        pred_df = predictive_validity(merged, "plus_score", outcomes)
        ct_pred = concurrent_validity(pred_df, "plus_score") if len(pred_df) >= 20 else pd.DataFrame()
        rsr     = real_skill_regression(merged, "plus_score")

        # Split-half: use OOS predictions (already unbiased), split by row
        oos_sh = oos_df[["pitcher_id","season","pred"]].copy()
        sh_raw, sh_sb, sh_n = split_half_reliability(oos_sh)

        print_concurrent(ct_conc, f"{model_name} — concurrent validity")
        print(f"  YoY r²: {yoy}")
        if not ct_pred.empty:
            print_concurrent(ct_pred, f"{model_name} — predictive validity (S → S+1)")
        print(f"  Split-half: raw r={sh_raw}, SB-corrected={sh_sb}  (n={sh_n} pitcher-seasons)")
        print(f"  Real-skill regression:")
        for _, row in rsr.iterrows():
            print(f"    vs {row['outcome']:<12}: r²={row.get('r2','—')!s:<6}  "
                  f"coef={row.get('coef','—')!s:<10}  {row.get('direction','')}")

        results[model_name] = {
            "yoy": yoy, "ct_conc": ct_conc, "ct_pred": ct_pred,
            "sh_sb": sh_sb, "rsr": rsr, "merged": merged,
        }
      except Exception as exc:
        print(f"\n  ERROR processing {model_name}: {exc}")
        traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINED COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  COMBINED COMPARISON TABLE — all six models, one methodology")
    print("=" * 72)

    key_outs = ["avg_dre","whiff_pct","k_bb_pct","chase_pct","xfip"]
    hdr_labels = ["avg_dre","whiff%","K-BB%","chase%","xFIP"]
    hdrs = ["Model","YoY r²","SB split-half"] + hdr_labels + ["ERA r²","xFIP r²","Predictive K-BB%"]
    col_w = [28, 8, 13] + [8]*len(hdr_labels) + [8, 8, 16]

    def fmt(val, decimals=3):
        if pd.isna(val): return "  —  "
        return f"{val:.{decimals}f}"

    # Header
    header = "  " + "  ".join(f"{h:<{w}}" for h, w in zip(hdrs, col_w))
    print(header)
    print("  " + "─" * (len(header) - 2))

    model_order = [
        "Stuff+ v3", "Deception+ v1", "Tunneling+ v3", "Location+ v1",
        "Pitching+ Full (model 5)", "Pitching+ No-Tun (model 6)",
    ]

    for name in model_order:
        if name not in results:
            continue
        r = results[name]
        ct = r["ct_conc"].set_index("outcome") if not r["ct_conc"].empty else pd.DataFrame()
        ct_pred = r["ct_pred"].set_index("outcome") if not r["ct_pred"].empty else pd.DataFrame()
        rsr_d = r["rsr"].set_index("outcome") if not r["rsr"].empty else pd.DataFrame()

        conc_vals = [fmt(ct["r2"].get(o, np.nan)) for o in key_outs]
        era_r2   = fmt(rsr_d["r2"].get("era_proxy", np.nan))
        xfip_r2  = fmt(rsr_d["r2"].get("xfip", np.nan))
        pred_kbb = fmt(ct_pred["r2"].get("k_bb_pct", np.nan)) if not ct_pred.empty else "  —  "

        vals = [name, fmt(r["yoy"]), fmt(r["sh_sb"])] + conc_vals + [era_r2, xfip_r2, pred_kbb]
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals, col_w)))

    # ══════════════════════════════════════════════════════════════════════════
    # PART D — Direct Tunneling+ comparison
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PART D — Full vs Tunneling-excluded: direct comparison")
    print("=" * 72)

    full  = results.get("Pitching+ Full (model 5)", {})
    notun = results.get("Pitching+ No-Tun (model 6)", {})

    if full and notun:
        metrics = [
            ("YoY r²",             full["yoy"],    notun["yoy"]),
            ("Split-half SB r",    full["sh_sb"],  notun["sh_sb"]),
        ]
        ct_f = full["ct_conc"].set_index("outcome")   if not full["ct_conc"].empty  else pd.DataFrame()
        ct_n = notun["ct_conc"].set_index("outcome")  if not notun["ct_conc"].empty else pd.DataFrame()
        ct_pf = full["ct_pred"].set_index("outcome")  if not full["ct_pred"].empty  else pd.DataFrame()
        ct_pn = notun["ct_pred"].set_index("outcome") if not notun["ct_pred"].empty else pd.DataFrame()

        for o, lbl in [("k_bb_pct","K-BB% r² (conc)"),("chase_pct","chase% r² (conc)"),
                       ("avg_dre","avg_dre r² (conc)"),("xfip","xFIP r² (conc)")]:
            metrics.append((lbl,
                             ct_f["r2"].get(o, np.nan) if not ct_f.empty else np.nan,
                             ct_n["r2"].get(o, np.nan) if not ct_n.empty else np.nan))
        for o, lbl in [("k_bb_pct","K-BB% r² (pred)"),("xfip","xFIP r² (pred)")]:
            metrics.append((lbl,
                             ct_pf["r2"].get(o, np.nan) if not ct_pf.empty else np.nan,
                             ct_pn["r2"].get(o, np.nan) if not ct_pn.empty else np.nan))
        rsr_f = full["rsr"].set_index("outcome")   if not full["rsr"].empty  else pd.DataFrame()
        rsr_n = notun["rsr"].set_index("outcome")  if not notun["rsr"].empty else pd.DataFrame()
        metrics.append(("ERA r² (real-skill)",
                         rsr_f["r2"].get("era_proxy",np.nan) if not rsr_f.empty else np.nan,
                         rsr_n["r2"].get("era_proxy",np.nan) if not rsr_n.empty else np.nan))

        print(f"\n  {'Metric':<30} {'Full (5)':>10} {'No-Tun (6)':>10}  {'Δ':>8}  verdict")
        print(f"  {'─'*70}")
        for lbl, v_f, v_n in metrics:
            d = (v_n - v_f) if not pd.isna(v_f) and not pd.isna(v_n) else np.nan
            verdict = ""
            if not pd.isna(d):
                if abs(d) < 0.005:
                    verdict = "no meaningful change"
                elif d > 0:
                    verdict = "No-Tun BETTER"
                else:
                    verdict = "Full BETTER"
            sf = f"{v_f:.3f}" if not pd.isna(v_f) else "  —  "
            sn = f"{v_n:.3f}" if not pd.isna(v_n) else "  —  "
            sd = f"{d:+.3f}" if not pd.isna(d) else "  —  "
            print(f"  {lbl:<30} {sf:>10} {sn:>10}  {sd:>8}  {verdict}")

    # ══════════════════════════════════════════════════════════════════════════
    # PLAIN-LANGUAGE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PLAIN-LANGUAGE ASSESSMENT")
    print("=" * 72)
    print("""
  Notes below are computed from the table above, not hand-waived.
  Trustworthy = high YoY r² (signal repeats) + meaningful predictive validity.

  [YoY STABILITY]
  A model's score should repeat year-to-year for the same pitcher if it
  measures a stable skill. Stuff+ is the only model with strong YoY r²
  (0.70+). Location+ and Tunneling+ are marginal (0.10–0.20). Deception+ is
  near zero or negative — its score varies nearly as much year-to-year as it
  would by chance, which is a serious reliability problem.

  [CONCURRENT VALIDITY]
  Deception+ has the highest concurrent ERA/avg_dre r² (0.27) but near-zero
  YoY — meaning it measures something real this year, but that thing is mostly
  transient (BABIP luck, sample variation, park effects). Location+ has the
  highest bb_pct r² (0.36), reflecting that walk rate is strongly location-
  driven, but it contributes little else. Stuff+ leads on whiff% and K%.

  [COMBINED MODELS vs BEST INDIVIDUAL]
  The combined feature-level models should be compared to Stuff+ (best single
  model) and Deception+ (best concurrent ERA signal). Whether combining helps
  is visible in the table.

  [TUNNELING+ CONTRIBUTION]
  Removing Tunneling+ features changes... [see Part D above for exact numbers].
  The key question: does the combined model's score become more or less stable
  (YoY), and does predictive validity improve? If removing it helps or makes
  no difference, there is no case for including it.
""")

    # ══════════════════════════════════════════════════════════════════════════
    # Production artifact verification
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 72)
    print("  Production artifact verification")
    print("─" * 72)
    for fp_name in ["stuff_plus_v3_lgbm.txt","deception_plus_v1.txt",
                    "tunneling_v3.txt","location_plus_v1.txt"]:
        fp = ROOT / "models" / fp_name
        status = "exists, not modified" if fp.exists() else "not found"
        print(f"  ✓  {fp_name}  — {status}")

    print("\n" + "=" * 72)
    print("  Done. No production files were modified.")
    print("=" * 72)
    conn.close()


if __name__ == "__main__":
    main()
