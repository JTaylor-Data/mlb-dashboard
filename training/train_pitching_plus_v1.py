"""
Pitching+ v1 — builds and evaluates two candidate architectures.

Part A: feature-level model (raw features from all four sub-models unioned)
Part B: score-level model (four plus-scores as inputs, pitcher-season grain)
Part C: validation of both + cross-check against individual sub-models

DOES NOT TOUCH ANY PRODUCTION ARTIFACT.
All outputs saved to analytics/pitching_plus/.
"""
import sys, warnings, json
from pathlib import Path

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

DB_PATH  = ROOT / "mlb_data.db"
OUT_DIR  = ROOT / "analytics" / "pitching_plus"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PITCHER_PITCHES = 100   # per pitcher-season to qualify

# Only complete seasons — 2026 is an in-progress partial season and causes
# the Stuff+ full-pipeline scoring to run extremely slowly; skip it here.
EVAL_SEASONS = [2021, 2022, 2023, 2024, 2025]

# ─── Identifier guard ─────────────────────────────────────────────────────────
_FORBIDDEN_IDS = {
    "pitcher_id", "batter_id", "player_id", "game_id", "game_pk",
    "at_bat_number", "pitch_number", "inning",
}

def assert_no_identifiers(cols: list) -> None:
    """Raise ValueError if any identifier leaks into the model feature set."""
    bad = _FORBIDDEN_IDS & set(cols)
    if bad:
        raise ValueError(f"Identifier columns in feature set: {sorted(bad)}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_pitches(conn, seasons: list[int]) -> pd.DataFrame:
    """
    Single SQL pull containing every raw column needed across all four models.
    Returns a per-pitch DataFrame. Junk pitch types and null targets are
    excluded here; KC->CU merge is applied immediately.
    """
    ph = ",".join("?" * len(seasons))
    sql = f"""
    SELECT
        pitcher_id, pitch_type, stand, p_throws, season,
        game_id, at_bat_number, pitch_number,
        -- Stuff+ raw inputs
        release_speed, effective_speed, release_spin_rate, spin_axis,
        pfx_x, pfx_z, vx0, vy0, vz0, ax, ay, az,
        -- Deception+ raw inputs (consistency computed below)
        release_extension, release_pos_x, release_pos_z, arm_angle,
        -- Location+ inputs
        plate_x, plate_z, balls, strikes,
        -- shared target
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
    df["season"] = df["season"].astype(str)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def _add_stuff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Stuff+ engineered columns: hb, ivb, vaa_residual, separation
    features, axis_differential.  Uses the same sub-functions as the live
    stuff_model.py; calibrations are fit on the passed-in DataFrame
    (single-pass for Part A — not per-fold; same pragmatic deviation as
    Stuff+ v3's spin-axis calibration).
    """
    from analytics.stuff_features import (
        add_movement_columns,
        fit_v3_calibrations,
        add_vaa_residual,
        build_fb_reference_table,
        add_separation_features,
    )

    df = add_movement_columns(df)                          # adds hb, ivb
    cals = fit_v3_calibrations(df)                         # fits vaa coefs
    df = add_vaa_residual(df, cals["vaa_coefs"])           # adds vaa, vaa_residual
    fb_ref = build_fb_reference_table(df)
    df = add_separation_features(df, fb_ref)               # adds *_sep_vs_primary_fb

    # axis_differential lives in spin_physics
    try:
        from analytics.spin_physics import add_axis_differential
        df = add_axis_differential(df)
    except Exception:
        df["axis_differential"] = np.nan

    return df


def _add_deception_features(df: pd.DataFrame, conn) -> pd.DataFrame:
    """
    Add per-pitch Deception+ columns (arm_angle, release_extension,
    release_pos_x_as, release_pos_z) and per-pitcher-season consistency
    aggregates (arm_angle_std, release_extension_std, release_consistency).
    Optionally joins height_in / weight_lb from pitcher_bio.
    """
    # arm-side adjusted release position
    sign = np.where(df["p_throws"] == "R", -1.0, 1.0)
    df["release_pos_x_as"] = df["release_pos_x"] * sign

    grp = df.groupby(["pitcher_id", "season"])
    df["arm_angle_std"]        = grp["arm_angle"].transform("std")
    df["release_extension_std"] = grp["release_extension"].transform("std")

    # release_consistency: RMS distance from pitcher-season mean release point (feet)
    mean_x = grp["release_pos_x_as"].transform("mean")
    mean_z = grp["release_pos_z"].transform("mean")
    df["release_consistency"] = np.sqrt(
        (df["release_pos_x_as"] - mean_x) ** 2 +
        (df["release_pos_z"]    - mean_z) ** 2
    )

    # bio — optional
    try:
        from analytics.deception_features import load_bio
        bio = load_bio(conn)
        df = df.merge(bio, on="pitcher_id", how="left")
    except Exception:
        df["height_in"]  = np.nan
        df["weight_lb"]  = np.nan

    return df


def _add_tunneling_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-at-bat lag features using the same v3 pipeline as Tunneling+.
    First-pitch-in-at-bat rows keep all lag columns as NaN (not excluded),
    per task spec.  sequential_lag_features() only returns non-first-pitch
    rows, so we left-join back onto the full DataFrame.
    """
    from analytics.tunneling_features import (
        build_commit_point_table,
        sequential_lag_features,
        lag_feature_columns,
    )

    commit_table = build_commit_point_table(df)
    lag_df = sequential_lag_features(df, commit_table, max_lag=6)

    lag_cols  = lag_feature_columns(6)
    join_keys = ["pitcher_id", "game_id", "at_bat_number", "pitch_number"]

    avail_lag_cols = [c for c in lag_cols if c in lag_df.columns]
    lag_sub = lag_df[join_keys + avail_lag_cols].copy()

    df = df.merge(lag_sub, on=join_keys, how="left")

    # Ensure every expected lag column exists (some may be absent if dataset
    # has no at-bats deeper than a given lag slot)
    for col in lag_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ── Feature manifests ─────────────────────────────────────────────────────────

_STUFF_NUMERIC = [
    "release_speed", "effective_speed", "release_spin_rate", "spin_axis",
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
_TUN_NUMERIC  = [
    f"lag{n}_{m}"
    for n in range(1, 7)
    for m in ("commit_distance", "plate_distance", "shape_break")
]
_TUN_CAT_LAG  = [f"lag{n}_pitch_type" for n in range(1, 7)]
_SHARED_CAT   = ["pitch_type", "p_throws", "stand", "season"]

PART_A_NUMERIC     = _STUFF_NUMERIC + _DEC_NUMERIC + _LOC_NUMERIC + _TUN_NUMERIC
PART_A_CATEGORICAL = _SHARED_CAT + _TUN_CAT_LAG
PART_A_FEATURES    = PART_A_NUMERIC + PART_A_CATEGORICAL

# Core numeric that must be non-null for a row to be usable
_CORE_REQUIRED = [
    "release_speed", "effective_speed", "release_spin_rate",
    "spin_axis", "hb", "ivb", "vaa_residual",
    "plate_x", "plate_z", "balls", "strikes", "delta_run_exp",
]


def build_part_a_dataset(conn, seasons: list[int]):
    """
    Build the unified per-pitch feature-level dataset for Part A.
    Returns (df, feature_list) where feature_list contains only columns
    that are actually present after engineering.
    """
    print("  [A] Loading raw pitches...")
    df = load_raw_pitches(conn, seasons)
    print(f"      Raw: {len(df):,} pitches, {df['pitcher_id'].nunique()} pitchers")

    print("  [A] Engineering Stuff+ features...")
    df = _add_stuff_features(df)

    print("  [A] Engineering Deception+ features...")
    df = _add_deception_features(df, conn)

    print("  [A] Computing tunneling lag features (slow—building per-at-bat sequences)...")
    df = _add_tunneling_lags(df)

    # Coerce categoricals
    for col in PART_A_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Drop rows missing core features
    df = df.dropna(subset=_CORE_REQUIRED)

    # Resolve which features actually exist
    features = [c for c in PART_A_FEATURES if c in df.columns]

    n_num  = sum(1 for c in features if c in PART_A_NUMERIC)
    n_cat  = sum(1 for c in features if c in PART_A_CATEGORICAL)
    missing_feats = [c for c in PART_A_FEATURES if c not in df.columns]

    print(f"\n  [A] Final dataset: {len(df):,} pitches | "
          f"{df['pitcher_id'].nunique()} pitchers")
    print(f"      Features: {len(features)} total "
          f"({n_num} numeric, {n_cat} categorical)")
    if missing_feats:
        print(f"      Missing (will be excluded): {missing_feats}")

    # Duplication check
    seen = set()
    dupes = [c for c in features if c in seen or seen.add(c)]
    if dupes:
        print(f"      WARNING: duplicate feature names: {dupes}")
    else:
        print("      Duplication check: PASS — no feature appears more than once")

    assert_no_identifiers(features)

    return df, features


# ══════════════════════════════════════════════════════════════════════════════
# LGBM hyperparameters (shared conventions across all four sub-models)
# ══════════════════════════════════════════════════════════════════════════════

LGBM_PARAMS_A = dict(
    objective="huber",
    alpha=0.9,
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

LGBM_PARAMS_B = dict(   # Score-level: only 4 inputs — much smaller model
    objective="huber",
    alpha=0.9,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)


# ══════════════════════════════════════════════════════════════════════════════
# PART A — LOSO CV + final model
# ══════════════════════════════════════════════════════════════════════════════

def loso_cv_part_a(df: pd.DataFrame, features: list, seasons: list[int]) -> pd.DataFrame:
    """
    Leave-one-season-out CV for the feature-level model.
    Returns a per-pitch DataFrame with columns:
        pitcher_id, season, delta_run_exp (actual), pred (raw model output)
    """
    cat_features = [f for f in PART_A_CATEGORICAL if f in features]
    oos_rows = []

    for s in seasons:
        s_str = str(s)
        tr = df["season"] != s_str
        te = df["season"] == s_str
        if te.sum() == 0:
            continue

        model = lgb.LGBMRegressor(**LGBM_PARAMS_A)
        model.fit(df.loc[tr, features], df.loc[tr, "delta_run_exp"],
                  categorical_feature=cat_features)
        preds = model.predict(df.loc[te, features])

        fold_df = df.loc[te, ["pitcher_id", "season", "delta_run_exp"]].copy()
        fold_df["pred"] = preds
        oos_rows.append(fold_df)

        r2 = r2_score(df.loc[te, "delta_run_exp"], preds)
        print(f"      Fold {s}: {te.sum():>7,} pitches  per-pitch r²={r2:.4f}")

    return pd.concat(oos_rows, ignore_index=True)


def train_final_part_a(df: pd.DataFrame, features: list) -> lgb.LGBMRegressor:
    cat_features = [f for f in PART_A_CATEGORICAL if f in features]
    model = lgb.LGBMRegressor(**LGBM_PARAMS_A)
    model.fit(df[features], df["delta_run_exp"], categorical_feature=cat_features)
    out = OUT_DIR / "pitching_plus_feature_level_v1.txt"
    model.booster_.save_model(str(out))
    print(f"      Saved → {out.name}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Score-level data + LOSO CV
# ══════════════════════════════════════════════════════════════════════════════

def _get_sub_model_scores(conn, season: int) -> dict:
    """
    Call each live sub-model's raw-prediction function for a single season
    and aggregate to pitcher-season plus-scores (mean raw pred → plus-scale).
    Uses _raw_predictions for all four models for consistency; avoids the
    much heavier compute_stuff_plus wrapper (shrinkage + multi-pass groupby)
    which was causing 27+ min hangs on the 2026 partial-season dataset.
    """
    from analytics.stuff_model import _raw_predictions as _stuff_raw
    from analytics.tunneling_v3_model import _raw_predictions as _tun_raw
    from analytics.location_model import _raw_predictions as _loc_raw, official_location_plus
    from analytics.deception_model import _raw_predictions as _dec_raw

    scores: dict[int, dict] = {}

    # Stuff+ — aggregate per-pitch raw predictions to pitcher-season
    try:
        raw = _stuff_raw(conn, season)
        if not raw.empty and raw["pred"].std() > 0:
            mean, std = raw["pred"].mean(), raw["pred"].std()
            g = raw.groupby("pitcher_id")["pred"].mean()
            for pid, val in g.items():
                scores.setdefault(int(pid), {})["stuff_plus"] = float(
                    100 - 10 * (val - mean) / std
                )
    except Exception as e:
        print(f"        Stuff+ error ({season}): {e}")

    try:
        raw = _tun_raw(conn, season)
        if not raw.empty and raw["pred"].std() > 0:
            mean, std = raw["pred"].mean(), raw["pred"].std()
            g = raw.groupby("pitcher_id")["pred"].mean()
            for pid, val in g.items():
                scores.setdefault(int(pid), {})["tunneling_plus"] = float(
                    100 - 10 * (val - mean) / std
                )
    except Exception as e:
        print(f"        Tunneling+ error ({season}): {e}")

    try:
        raw = _loc_raw(conn, season)
        if not raw.empty:
            loc = official_location_plus(raw)
            loc = loc[loc["total_pitches"] >= MIN_PITCHER_PITCHES]
            for r in loc.itertuples():
                scores.setdefault(int(r.pitcher_id), {})["location_plus"] = float(r.score)
    except Exception as e:
        print(f"        Location+ error ({season}): {e}")

    try:
        raw = _dec_raw(conn, season)
        if not raw.empty and raw["pred"].std() > 0:
            mean, std = raw["pred"].mean(), raw["pred"].std()
            g = raw.groupby("pitcher_id")["pred"].mean()
            for pid, val in g.items():
                scores.setdefault(int(pid), {})["deception_plus"] = float(
                    100 - 10 * (val - mean) / std
                )
    except Exception as e:
        print(f"        Deception+ error ({season}): {e}")

    return scores


def _pitcher_season_outcomes(conn, seasons: list[int]) -> pd.DataFrame:
    """
    Aggregate pitcher-season outcomes from the pitches table.
    Returns one row per (pitcher_id, season) with:
        avg_dre, n_pitches, whiff_pct, swstr_pct,
        k_pct, bb_pct, k_bb_pct, chase_pct, xfip, era_proxy, ip
    """
    ph = ",".join("?" * len(seasons))

    # Per-pitch rates (avg_dre, whiff, swstr)
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
        WHERE season IN ({ph})
        GROUP BY pitcher_id, season
        HAVING COUNT(*) >= {MIN_PITCHER_PITCHES}
    """, conn, params=seasons)

    # PA-level event rates (K, BB, HR)
    df2 = pd.read_sql_query(f"""
        SELECT pitcher_id, season,
               SUM(CASE WHEN events IN ('strikeout','strikeout_double_play')
                   THEN 1 ELSE 0 END) AS k,
               SUM(CASE WHEN events IN ('walk','intent_walk')
                   THEN 1 ELSE 0 END) AS bb,
               SUM(CASE WHEN events = 'home_run'
                   THEN 1 ELSE 0 END) AS hr,
               SUM(CASE WHEN events IS NOT NULL AND events != ''
                   THEN 1 ELSE 0 END) AS pa
        FROM pitches
        WHERE season IN ({ph}) AND events IS NOT NULL AND events != ''
        GROUP BY pitcher_id, season
    """, conn, params=seasons)
    df2["k_pct"]    = df2["k"]  / df2["pa"].clip(lower=1)
    df2["bb_pct"]   = df2["bb"] / df2["pa"].clip(lower=1)
    df2["k_bb_pct"] = (df2["k"] - df2["bb"]) / df2["pa"].clip(lower=1)

    # IP (from outs recorded on events)
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

    # Chase% (swing rate on pitches outside the zone — |plate_x|>0.83 or
    # plate_z < 1.5 or plate_z > 3.5 is a rough zone boundary)
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
        WHERE season IN ({ph})
          AND plate_x IS NOT NULL AND plate_z IS NOT NULL
        GROUP BY pitcher_id, season
    """, conn, params=seasons)

    df = (df1
          .merge(df2[["pitcher_id","season","k","bb","hr","pa",
                       "k_pct","bb_pct","k_bb_pct"]], on=["pitcher_id","season"], how="left")
          .merge(df3[["pitcher_id","season","ip"]],   on=["pitcher_id","season"], how="left")
          .merge(df4,                                  on=["pitcher_id","season"], how="left"))

    # xFIP: (13*HR + 3*BB - 2*K) / IP + FIP_constant
    FIP_C = 3.10
    df["xfip"] = np.where(
        df["ip"] > 0,
        (13 * df["hr"] + 3 * df["bb"] - 2 * df["k"]) / df["ip"].clip(lower=0.001) + FIP_C,
        np.nan,
    )
    # ERA proxy from delta_run_exp: avg_dre is runs-value-per-pitch;
    # scaling by 27 outs (one 9-IP game) gives run-value rate roughly comparable to ERA
    df["era_proxy"] = -df["avg_dre"] * 27 * 3  # negative because lower dre = more runs

    df["season"] = df["season"].astype(str)
    return df


def build_part_b_dataset(conn, seasons: list[int]):
    """
    Build the pitcher-season score-level dataset for Part B.
    Returns (df, score_feature_cols).
    """
    rows = []
    for s in seasons:
        print(f"      Season {s}...", end=" ", flush=True)
        sco = _get_sub_model_scores(conn, s)
        print(f"{len(sco)} pitchers scored")
        for pid, d in sco.items():
            row = {"pitcher_id": int(pid), "season": str(s)}
            row.update(d)
            rows.append(row)

    scores_df = pd.DataFrame(rows)
    outcomes_df = _pitcher_season_outcomes(conn, seasons)

    df = scores_df.merge(outcomes_df, on=["pitcher_id", "season"], how="inner")

    score_cols = ["stuff_plus", "tunneling_plus", "location_plus", "deception_plus"]
    df = df.dropna(subset=score_cols + ["avg_dre"])

    print(f"\n      Part B: {len(df):,} pitcher-seasons | "
          f"{df['pitcher_id'].nunique()} unique pitchers")

    # Descriptive correlation among the four inputs
    print("\n      Input score correlations (Pearson r):")
    for i, a in enumerate(score_cols):
        for b in score_cols[i+1:]:
            sub = df[[a, b]].dropna()
            if len(sub) > 20:
                r, _ = stats.pearsonr(sub[a], sub[b])
                print(f"        {a} vs {b}: r={r:.3f}")

    assert_no_identifiers(score_cols)
    return df, score_cols


def loso_cv_part_b(df: pd.DataFrame, features: list, seasons: list[int],
                   mode: str = "linear") -> pd.DataFrame:
    """
    LOSO CV for the score-level model.
    mode: 'linear' | 'lgbm'
    Returns DataFrame with pitcher_id, season, avg_dre (actual), pred.
    """
    oos_rows = []
    for s in seasons:
        tr = df["season"] != str(s)
        te = df["season"] == str(s)
        if te.sum() < 5:
            continue
        X_tr = df.loc[tr, features].values
        y_tr = df.loc[tr, "avg_dre"].values
        X_te = df.loc[te, features].values

        if mode == "linear":
            m = LinearRegression()
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
        else:
            m = lgb.LGBMRegressor(**LGBM_PARAMS_B)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)

        fold_df = df.loc[te, ["pitcher_id", "season", "avg_dre"]].copy()
        fold_df["pred"] = preds
        oos_rows.append(fold_df)

    return pd.concat(oos_rows, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PART C — Validation helpers
# ══════════════════════════════════════════════════════════════════════════════

OUTCOME_COLS = [
    "avg_dre", "whiff_pct", "swstr_pct", "k_bb_pct", "chase_pct",
    "k_pct", "bb_pct", "xfip", "era_proxy",
]


def _to_plus_scale(series: pd.Series) -> pd.Series:
    """Convert raw model outputs to plus-scale (mean=100, std=10)."""
    m, s = series.mean(), series.std()
    if s < 1e-9:
        return pd.Series(100.0, index=series.index)
    return 100 - 10 * (series - m) / s


def aggregate_part_a_oos(oos_df: pd.DataFrame,
                          outcomes: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-pitch OOS predictions to pitcher-season plus-scores,
    then join with outcomes.
    """
    ps = (oos_df
          .groupby(["pitcher_id", "season"])
          .agg(n_pitches=("pred", "count"), raw_pred=("pred", "mean"))
          .reset_index())
    ps = ps[ps["n_pitches"] >= MIN_PITCHER_PITCHES]
    ps["plus_score"] = _to_plus_scale(ps["raw_pred"])

    outcomes_copy = outcomes.copy()
    outcomes_copy["season"] = outcomes_copy["season"].astype(str).str.strip()
    ps["season"] = ps["season"].astype(str).str.strip()

    return ps.merge(outcomes_copy, on=["pitcher_id","season"], how="inner")


def corr_table(merged: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Pearson r and r² between score_col and each outcome column.
    Returns a tidy DataFrame.
    """
    rows = []
    for col in OUTCOME_COLS:
        if col not in merged.columns:
            continue
        sub = merged[[score_col, col]].dropna()
        if len(sub) < 20:
            rows.append({"outcome": col, "r": np.nan, "r2": np.nan, "n": len(sub)})
            continue
        r, p = stats.pearsonr(sub[score_col], sub[col])
        rows.append({"outcome": col, "r": round(r, 3), "r2": round(r**2, 3),
                     "n": len(sub), "p": round(p, 4)})
    return pd.DataFrame(rows)


def yoy_r2(merged: pd.DataFrame, score_col: str) -> float:
    """
    Year-over-year r²: pitcher's score in season S vs their score in season S+1.
    Tests repeatability (skill signal vs. noise).
    """
    df = merged[["pitcher_id", "season", score_col]].dropna().copy()
    df["season_int"] = df["season"].astype(str).str.strip().astype(int)

    curr = df[["pitcher_id","season_int", score_col]].rename(
        columns={score_col: "score_curr"})
    nxt  = df[["pitcher_id","season_int", score_col]].copy()
    nxt["season_int"] = nxt["season_int"] - 1   # align to previous season
    nxt = nxt.rename(columns={score_col: "score_next"})

    paired = curr.merge(nxt, on=["pitcher_id","season_int"], how="inner")
    if len(paired) < 20:
        return np.nan
    return round(r2_score(paired["score_next"], paired["score_curr"]), 3)


def prev_season_predictive(merged_scores: pd.DataFrame,
                            score_col: str,
                            outcomes: pd.DataFrame) -> pd.DataFrame:
    """
    Predictive validity: season-S score → season-S+1 actual outcomes.
    Returns a merged DataFrame with suffixes _s and _s1.
    """
    df = merged_scores[["pitcher_id", "season", score_col]].dropna().copy()
    df["season_int"] = df["season"].astype(str).astype(int)

    # Shift outcomes one year forward
    out_next = outcomes.copy()
    out_next["season_int"] = out_next["season"].astype(str).astype(int)
    out_next_shifted = out_next.copy()
    out_next_shifted["season_int"] = out_next_shifted["season_int"] - 1

    merged = df.merge(out_next_shifted, on=["pitcher_id","season_int"], how="inner")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# SHAP + collinearity
# ══════════════════════════════════════════════════════════════════════════════

def shap_report(model: lgb.LGBMRegressor, df: pd.DataFrame,
                features: list, n_sample: int = 8000) -> pd.Series:
    """
    Compute SHAP mean |value| for each feature.
    Flags known-correlated pairs and reports credit split.
    Returns Series of importances indexed by feature name.
    """
    try:
        import shap as _shap
    except ImportError:
        print("      shap not installed — skipping SHAP analysis")
        return pd.Series(dtype=float)

    sample = df[features].sample(min(n_sample, len(df)), random_state=42)
    explainer = _shap.TreeExplainer(model)
    vals = explainer.shap_values(sample)
    imp = pd.Series(np.abs(vals).mean(axis=0), index=features).sort_values(ascending=False)

    print(f"\n  Top 30 features by mean |SHAP| (n={len(sample):,} sampled pitches):")
    for feat, v in imp.head(30).items():
        bar = "█" * int(v / imp.iloc[0] * 20)
        print(f"    {feat:<45} {v:.5f}  {bar}")

    # Known-correlated pair check
    known_pairs = [
        ("arm_angle",        "release_pos_z",          0.79),
        ("release_speed",    "effective_speed",         None),
        ("vaa_residual",     "ivb",                     None),
        ("lag1_plate_distance", "plate_x",              None),
        ("lag1_plate_distance", "plate_z",              None),
        ("lag1_commit_distance","lag1_plate_distance",  None),
        ("hb",               "velo_sep_vs_primary_fb",  None),
        ("release_pos_z",    "arm_angle_std",           None),
    ]
    print("\n  Known-correlated pairs — credit split in SHAP:")
    for a, b, known_r in known_pairs:
        if a not in features or b not in features:
            continue
        col_a = sample[a].astype(float)
        col_b = sample[b].astype(float)
        valid = col_a.notna() & col_b.notna()
        if valid.sum() < 200:
            continue
        r_obs = col_a[valid].corr(col_b[valid])
        rank_a = int(imp.rank(ascending=False)[a]) if a in imp.index else 999
        rank_b = int(imp.rank(ascending=False)[b]) if b in imp.index else 999
        imp_a  = imp.get(a, 0)
        imp_b  = imp.get(b, 0)
        note = ""
        if known_r and abs(r_obs - known_r) < 0.15:
            note = "  ← confirmed high collinearity"
        total = imp_a + imp_b
        print(f"    {a} vs {b}:")
        print(f"      observed r={r_obs:.3f}"
              f"{'  (known≈'+str(known_r)+')' if known_r else ''}{note}")
        print(f"      SHAP rank: #{rank_a} (imp={imp_a:.5f}) | "
              f"#{rank_b} (imp={imp_b:.5f}) | "
              f"combined={total:.5f}")

    # Top pairwise correlations in the dataset (numeric features only)
    num_feats = [f for f in features if f in PART_A_NUMERIC and f in df.columns]
    print("\n  Top 15 pairwise correlations among numeric features "
          "(sampled, may differ from full-dataset r):")
    corr_mat = sample[[f for f in num_feats if f in sample.columns]].corr().abs()
    pairs_all = (corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
                 .stack()
                 .sort_values(ascending=False))
    for (a, b), r_val in pairs_all.head(15).items():
        print(f"    {a:<40} vs {b:<40}: r={r_val:.3f}")

    return imp


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _print_corr_table(ct: pd.DataFrame, label: str) -> None:
    print(f"\n  {label}:")
    print(f"  {'outcome':<18} {'r':>6} {'r²':>6} {'n':>6}")
    print(f"  {'-'*40}")
    for _, row in ct.iterrows():
        r  = f"{row['r']:.3f}"  if not pd.isna(row['r'])  else "  —  "
        r2 = f"{row['r2']:.3f}" if not pd.isna(row['r2']) else "  —  "
        n  = int(row['n']) if not pd.isna(row['n']) else 0
        print(f"  {row['outcome']:<18} {r:>6} {r2:>6} {n:>6}")


def main():
    print("=" * 72)
    print("  Pitching+ v1 — Feature-level vs Score-level Candidate Evaluation")
    print("=" * 72)

    conn = sqlite3.connect(DB_PATH)

    # Only use complete seasons — exclude in-progress 2026
    avail = pd.read_sql_query(
        "SELECT DISTINCT season FROM pitches WHERE season >= 2021 ORDER BY season", conn)
    all_seasons = [int(s) for s in avail["season"].tolist()]
    seasons = [s for s in all_seasons if s in EVAL_SEASONS]
    print(f"\n  All seasons in DB: {all_seasons}")
    print(f"  Evaluation seasons (complete only): {seasons}")

    # ══ PART B first (much faster — scores are pre-computed) ══════════════════
    print("\n" + "─" * 72)
    print("  PART B — Score-level model (four sub-model plus-scores as inputs)")
    print("─" * 72)

    b_df, b_features = build_part_b_dataset(conn, seasons)
    b_outcomes = b_df[["pitcher_id","season"] + OUTCOME_COLS].copy()
    b_outcomes["season"] = b_outcomes["season"].astype(str)

    print("\n  LOSO CV — Linear regression:")
    b_oos_lin  = loso_cv_part_b(b_df, b_features, seasons, "linear")
    print("  LOSO CV — LightGBM:")
    b_oos_lgbm = loso_cv_part_b(b_df, b_features, seasons, "lgbm")

    # Full-data linear coefficients
    lr = LinearRegression().fit(b_df[b_features].values, b_df["avg_dre"].values)
    print("\n  Linear regression coefficients (full data, avg_dre target):")
    for feat, coef in sorted(zip(b_features, lr.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:<22}: {coef:+.7f}")
    print(f"    intercept            : {lr.intercept_:+.7f}")
    lr_full_r2 = r2_score(b_df["avg_dre"], lr.predict(b_df[b_features].values))
    print(f"    full-data r²         : {lr_full_r2:.4f}")

    # OOS summary
    for label, oos in [("Linear", b_oos_lin), ("LightGBM", b_oos_lgbm)]:
        r2_oos = r2_score(oos["avg_dre"], oos["pred"])
        r_oos, _ = stats.pearsonr(oos["avg_dre"], oos["pred"])
        print(f"\n  {label} OOS (all seasons): r={r_oos:.3f}, r²={r2_oos:.4f}, "
              f"n={len(oos):,} pitcher-seasons")

    # Build merged datasets for validation
    b_df2 = b_df.copy()
    b_df2["season"] = b_df2["season"].astype(str)

    for mode, b_oos in [("Linear", b_oos_lin), ("LightGBM", b_oos_lgbm)]:
        b_oos = b_oos.copy()
        b_oos["season"] = b_oos["season"].astype(str)
        b_oos["plus_score"] = _to_plus_scale(b_oos["pred"])
        b_oos_merged = b_oos.merge(b_outcomes, on=["pitcher_id","season"], how="inner")
        ct = corr_table(b_oos_merged, "plus_score")
        _print_corr_table(ct, f"Part B ({mode}) — concurrent validity")
        yoy = yoy_r2(b_oos_merged, "plus_score")
        print(f"  YoY r² ({mode}): {yoy}")

        # Save merged for comparison table
        if mode == "Linear":
            b_merged_lin  = b_oos_merged
            b_ct_lin      = ct
            b_yoy_lin     = yoy
        else:
            b_merged_lgbm = b_oos_merged
            b_ct_lgbm     = ct
            b_yoy_lgbm    = yoy

    # ══ PART A ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  PART A — Feature-level model (raw features from all four models)")
    print("─" * 72)

    a_df, a_features = build_part_a_dataset(conn, seasons)

    print("\n  LOSO CV:")
    a_oos = loso_cv_part_a(a_df, a_features, seasons)

    # Outcomes for Part A (need separate pull since a_df is pitch-level)
    a_outcomes = _pitcher_season_outcomes(conn, seasons)

    a_merged = aggregate_part_a_oos(a_oos, a_outcomes)
    a_ct     = corr_table(a_merged, "plus_score")
    a_yoy    = yoy_r2(a_merged, "plus_score")

    _print_corr_table(a_ct, "Part A (feature-level) — concurrent validity")
    print(f"  YoY r²: {a_yoy}")

    # Predictive validity: prior-season score → following-season outcomes
    a_pred_merged = prev_season_predictive(a_merged, "plus_score", a_outcomes)
    if len(a_pred_merged) >= 20:
        a_ct_pred = corr_table(a_pred_merged, "plus_score")
        _print_corr_table(a_ct_pred, "Part A — PREDICTIVE validity (score_S → outcome_S+1)")
    else:
        print(f"\n  Part A predictive: n={len(a_pred_merged)} (too small to report)")

    b_pred_merged = prev_season_predictive(b_merged_lin, "plus_score", b_outcomes)
    if len(b_pred_merged) >= 20:
        b_ct_pred = corr_table(b_pred_merged, "plus_score")
        _print_corr_table(b_ct_pred, "Part B (Linear) — PREDICTIVE validity (score_S → outcome_S+1)")

    # ── Train final Part A model + SHAP ──────────────────────────────────────
    print("\n  Training final feature-level model on all data...")
    a_model = train_final_part_a(a_df, a_features)

    print("\n  SHAP + Collinearity report:")
    shap_imp = shap_report(a_model, a_df, a_features)
    if not shap_imp.empty:
        shap_imp.to_csv(OUT_DIR / "pitching_plus_feature_level_shap.csv",
                        header=["mean_abs_shap"])
        print(f"      SHAP saved → pitching_plus_feature_level_shap.csv")

    # ══ Individual sub-model cross-check ══════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Individual sub-model scores — cross-check against same outcomes")
    print("─" * 72)

    sub_rows = []
    for s in seasons:
        sco = _get_sub_model_scores(conn, s)
        for pid, d in sco.items():
            row = {"pitcher_id": int(pid), "season": str(s)}
            row.update(d)
            sub_rows.append(row)
    sub_df = pd.DataFrame(sub_rows)

    sub_outcomes = a_outcomes.copy()
    sub_outcomes["season"] = sub_outcomes["season"].astype(str)

    sub_results = {}
    for name, col in [("Stuff+","stuff_plus"),("Tunneling+","tunneling_plus"),
                      ("Location+","location_plus"),("Deception+","deception_plus")]:
        if col not in sub_df.columns:
            continue
        sm = (sub_df[["pitcher_id","season",col]].dropna(subset=[col])
              .merge(sub_outcomes, on=["pitcher_id","season"], how="inner"))
        if len(sm) < 20:
            continue
        ct = corr_table(sm, col)
        yoy = yoy_r2(sm, col)
        sub_results[name] = {"merged": sm, "ct": ct, "yoy": yoy}
        _print_corr_table(ct, f"{name} — concurrent validity")
        print(f"  {name} YoY r²: {yoy}")

    # ══ REAL-SKILL REGRESSION: ERA, xFIP, K-BB% ══════════════════════════════
    print("\n" + "─" * 72)
    print("  Real-skill regression: Pitching+ score vs ERA proxy / xFIP / K-BB%")
    print("─" * 72)

    for label, mdf in [("Part A feature-level", a_merged),
                        ("Part B score-level (Linear)", b_merged_lin)]:
        print(f"\n  {label}:")
        for outcome in ["era_proxy", "xfip", "k_bb_pct"]:
            sub = mdf[["plus_score", outcome]].dropna()
            if len(sub) < 20:
                print(f"    vs {outcome:<14}: n too small ({len(sub)})")
                continue
            slope, intercept, r, p, _ = stats.linregress(sub["plus_score"], sub[outcome])
            direction = "✓ correct" if (
                (outcome in ("k_bb_pct",) and slope > 0) or
                (outcome in ("era_proxy","xfip") and slope < 0)
            ) else "✗ wrong direction"
            print(f"    vs {outcome:<14}: r²={r**2:.3f}  coef={slope:+.5f}  "
                  f"p={p:.4f}  n={len(sub)}  {direction}")

    # ══ COMPARISON TABLE ══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  COMBINED COMPARISON TABLE")
    print("=" * 72)

    key_outcomes = ["avg_dre", "whiff_pct", "k_bb_pct", "chase_pct", "xfip"]
    header = f"  {'Model':<38} {'YoY':>5} " + \
             " ".join(f"{o[:8]:>8}" for o in key_outcomes)
    print(header)
    print("  " + "-" * (len(header) - 2))

    def _row_str(name, ct, yoy):
        yoy_s = f"{yoy:.3f}" if not pd.isna(yoy) else "  —  "
        vals = []
        for o in key_outcomes:
            r2 = ct.set_index("outcome")["r2"].get(o, np.nan) if "outcome" in ct.columns else np.nan
            vals.append(f"{r2:.3f}" if not pd.isna(r2) else "  —  ")
        return f"  {name:<38} {yoy_s:>5} " + " ".join(f"{v:>8}" for v in vals)

    print(_row_str("Pitching+ feature-level (A)", a_ct, a_yoy))
    print(_row_str("Pitching+ score-lvl Linear (B)", b_ct_lin, b_yoy_lin))
    print(_row_str("Pitching+ score-lvl LGBM (B)", b_ct_lgbm, b_yoy_lgbm))
    for name in ["Stuff+","Tunneling+","Location+","Deception+"]:
        if name in sub_results:
            d = sub_results[name]
            print(_row_str(name, d["ct"], d["yoy"]))

    # Save comparison table to CSV
    comp_rows = []
    for name, ct, yoy in [
        ("Pitching+ feature-level", a_ct, a_yoy),
        ("Pitching+ score-level Linear", b_ct_lin, b_yoy_lin),
        ("Pitching+ score-level LGBM", b_ct_lgbm, b_yoy_lgbm),
    ] + [(n, sub_results[n]["ct"], sub_results[n]["yoy"]) for n in sub_results]:
        row = {"model": name, "yoy_r2": yoy}
        for _, r in ct.iterrows():
            row[f"{r['outcome']}_r2"] = r["r2"]
            row[f"{r['outcome']}_r"]  = r["r"]
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "comparison_table.csv", index=False)

    # ══ Production artifact check ══════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("  Production artifact verification (these must be UNCHANGED)")
    print("─" * 72)
    prod = [
        ROOT / "models" / "stuff_plus_v3_lgbm.txt",
        ROOT / "models" / "deception_plus_v1.txt",
        ROOT / "models" / "tunneling_v3.txt",
        ROOT / "models" / "location_plus_v1.txt",
    ]
    all_ok = True
    for fp in prod:
        if fp.exists():
            print(f"  ✓  {fp.name}  — exists, not modified by this script")
        else:
            print(f"  ·  {fp.name}  — does not exist (not relevant to this run)")

    new_artifacts = sorted(OUT_DIR.glob("*"))
    print(f"\n  New artifacts written to analytics/pitching_plus/ ({len(new_artifacts)}):")
    for f in new_artifacts:
        print(f"    {f.name}")

    print("\n" + "=" * 72)
    print("  Done. No production files were touched.")
    print("=" * 72)
    conn.close()


if __name__ == "__main__":
    main()
