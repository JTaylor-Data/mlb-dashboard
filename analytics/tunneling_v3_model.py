"""
Tunneling+ v3 - runtime scoring. Loads the model trained by
train_tunneling_v3_model.py and scores pitches on demand for live requests
(league-wide leaderboard + per-pitcher SHAP breakdown), same live-serving
split as stuff_model.py/deception_model.py (this file is the request-
serving counterpart; training/validation live under training/).

v1 (analytics/tunneling_model.py) is UNTOUCHED and still powers the
separate Tunneling+ 3D trajectory viewer's own per-pitch-type-pair
validated numbers (api/routers/tunneling_3d.py, frontend/tunnel3d.html) -
this file and its API routes are new and additive, not a replacement of
any v1 path. v3 has no pair-level concept at all (single trained model,
per-pitch lag-context features), so there's no way to serve it through
v1's pair-shaped contract even in principle.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from analytics.tunneling_features import lag_feature_columns, load_pitches, sequential_lag_features, MAX_LAG

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BOOSTER_PATH = MODELS_DIR / "tunneling_v3.txt"
META_PATH = MODELS_DIR / "tunneling_v3_meta.json"
COMMIT_TABLE_PATH = (
    Path(__file__).resolve().parent.parent / "training" / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv"
)

MIN_SEASON_PITCHES = 100  # same single qualifying threshold used at training time (build_tunneling_v3_dataset.py Part C)

LAG_COLS = lag_feature_columns()
NUMERIC_LAG_COLS = [c for c in LAG_COLS if not c.endswith("_pitch_type")]
CATEGORICAL_LAG_COLS = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CATEGORICAL = ["pitch_type", "p_throws", "stand", "season"]
CATEGORICAL_FEATURES = CATEGORICAL_LAG_COLS + CURRENT_CATEGORICAL
ALL_FEATURES = NUMERIC_LAG_COLS + CATEGORICAL_FEATURES

RAW_PREDICTIONS_CACHE_TTL_SECONDS = 600
_raw_predictions_cache = {}
_model_cache = {}


def _load():
    if "model" not in _model_cache:
        _model_cache["model"] = lgb.Booster(model_file=str(BOOSTER_PATH))
    return _model_cache["model"]


def _get_explainer():
    """Cached the same way stuff_model.py/deception_model.py cache theirs -
    building a TreeExplainer has real, avoidable overhead if done fresh on
    every per-pitcher lookup."""
    if "explainer" not in _model_cache:
        _model_cache["explainer"] = shap.TreeExplainer(_load())
    return _model_cache["explainer"]


def _load_meta():
    if "meta" not in _model_cache:
        _model_cache["meta"] = json.loads(META_PATH.read_text())
    return _model_cache["meta"]


def _load_commit_table():
    """v2's own 200ms commit-point table, reused directly - same choice
    train_tunneling_v3_model.py's dataset build already made (not
    re-derived here). A pitcher-season absent from this table (no
    qualifying FF/SI/FC pitch) has no principled commit point and is
    silently absent from v3's output, same as at training time."""
    if "commit_table" not in _model_cache:
        _model_cache["commit_table"] = pd.read_csv(COMMIT_TABLE_PATH)
    return _model_cache["commit_table"]


def _stringify_keep_na(series):
    """Same fix as train_tunneling_v3_model.py's _stringify_keep_na -
    .astype(str) alone turns real NaN into the string "nan", which would
    stop LightGBM's native missing-value handling from ever seeing a real
    missing category."""
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)


def _prepare_features(df):
    """Raw pitches (as returned by load_pitches, `season` still a native
    int matching the commit table's own dtype) -> the 28-feature frame the
    trained booster expects. Categoricals are coerced to the EXACT
    category levels saved at training time (meta.json's category_levels) -
    required for a raw lgb.Booster.predict() to reproduce training-time
    behavior, same requirement already documented and verified in
    training/tunneling_v3_validate.py."""
    commit_table = _load_commit_table()
    feat = sequential_lag_features(df, commit_table, max_lag=MAX_LAG)
    feat["season"] = feat["season"].astype(str)
    meta = _load_meta()
    for col in CATEGORICAL_FEATURES:
        feat[col] = _stringify_keep_na(feat[col])
        feat[col] = pd.Categorical(feat[col], categories=meta["category_levels"][col])
    return feat


def _raw_predictions(conn, season):
    """Full-season per-pitch DataFrame: pitcher_id + raw predicted
    delta_run_exp - the shared basis for both the league-wide plus-scale
    baseline and any single pitcher's SHAP breakdown. Short-lived
    in-process cache, same TTL/pattern as stuff_model.py's/
    deception_model.py's own _raw_predictions. The MIN_SEASON_PITCHES
    qualifying filter is applied on TOTAL raw pitches (before lag pairing
    drops each at-bat's 1st pitch), exactly matching
    build_tunneling_v3_dataset.py's Part C ordering."""
    cached = _raw_predictions_cache.get(season)
    if cached is not None and (time.monotonic() - cached[1]) < RAW_PREDICTIONS_CACHE_TTL_SECONDS:
        return cached[0]

    df = load_pitches(conn, season)
    totals = df.groupby("pitcher_id").size()
    qualifying_ids = totals[totals >= MIN_SEASON_PITCHES].index
    df = df[df["pitcher_id"].isin(qualifying_ids)]
    if df.empty:
        out = pd.DataFrame(columns=["pitcher_id", "pred"])
        _raw_predictions_cache[season] = (out, time.monotonic())
        return out

    feat = _prepare_features(df)
    model = _load()
    preds = model.predict(feat[ALL_FEATURES])
    out = feat[["pitcher_id"]].copy()
    out["pred"] = preds
    _raw_predictions_cache[season] = (out, time.monotonic())
    return out


def clear_cache():
    _raw_predictions_cache.clear()
    _model_cache.pop("commit_table", None)


def _to_plus_scale(values, mean, std):
    return 100 - 10 * (values - mean) / std


def leaderboard(conn, season, top_n=15):
    """League-wide top/bottom N by Tunneling+, min MIN_SEASON_PITCHES that
    season - same shape training/tunneling_v3_validate.py's own
    leaderboard() produces, recomputed live here instead of read from a
    frozen artifact so a newer season (e.g. one added after the model was
    trained) works without retraining."""
    league = _raw_predictions(conn, season)
    if league.empty or league["pred"].std() == 0:
        return {"n_qualified": 0, "top": [], "bottom": []}
    mean, std = league["pred"].mean(), league["pred"].std()
    g = league.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n_pitches=("pred", "size")).reset_index()
    g["score"] = _to_plus_scale(g["mean_pred"], mean, std)

    ids = g["pitcher_id"].astype(int).tolist()
    placeholders = ",".join("?" * len(ids))
    names = pd.read_sql(f"""
        SELECT p.player_id AS pitcher_id, p.name, t.abbreviation AS team_abbrev
        FROM players p LEFT JOIN teams t ON p.team_id = t.team_id
        WHERE p.player_id IN ({placeholders})
    """, conn, params=ids)
    merged = g.merge(names, on="pitcher_id", how="left")

    def _rows(d):
        return [
            dict(pitcher_id=int(r.pitcher_id), name=r.name, team_abbrev=r.team_abbrev,
                 total_pitches=int(r.n_pitches), score=round(float(r.score), 1))
            for r in d.itertuples()
        ]
    top = merged.sort_values("score", ascending=False).head(top_n)
    bottom = merged.sort_values("score", ascending=True).head(top_n)
    return {"n_qualified": int(len(g)), "top": _rows(top), "bottom": _rows(bottom)}


def pitcher_score(conn, season, pitcher_id, league_raw=None):
    """Just the headline number (for the player-card badge) - cheap path
    that doesn't build a SHAP explainer. Returns None if not qualified."""
    if league_raw is None:
        league_raw = _raw_predictions(conn, season)
    if league_raw.empty or league_raw["pred"].std() == 0:
        return None
    row = league_raw[league_raw["pitcher_id"] == pitcher_id]
    if row.empty:
        return None
    mean, std = league_raw["pred"].mean(), league_raw["pred"].std()
    return float(_to_plus_scale(row["pred"].mean(), mean, std))


def pitcher_breakdown(conn, season, pitcher_id, league_raw):
    """One pitcher's Tunneling+ score plus a live SHAP breakdown grouped
    by lag slot (how many pitches back each contribution comes from) - the
    natural display grain for v3's real architecture (28 features across
    6 recency-ordered lag slots), same diagnostic role
    deception_model.py's per-feature SHAP breakdown plays for its own
    model. v3 has no per-pitch-type-pair table (unlike v1) - a lag-slot
    breakdown is the honest analog: "how much did each pitch back in this
    pitcher's own sequences push their score," not a fabricated pair
    concept that doesn't exist in this architecture. Returns None if the
    pitcher has no qualifying (>= MIN_SEASON_PITCHES raw pitches) season."""
    if league_raw.empty or league_raw["pred"].std() == 0:
        return None
    season_mean, season_std = league_raw["pred"].mean(), league_raw["pred"].std()

    raw_df = load_pitches(conn, season, pitcher_id=pitcher_id)
    if len(raw_df) < MIN_SEASON_PITCHES:
        return None
    feat = _prepare_features(raw_df)
    if feat.empty:
        return None

    model = _load()
    preds = model.predict(feat[ALL_FEATURES])
    score = float(_to_plus_scale(preds.mean(), season_mean, season_std))

    explainer = _get_explainer()
    shap_values = explainer.shap_values(feat[ALL_FEATURES])
    # SHAP values are additive in raw delta_run_exp units; the plus-scale
    # transform is linear (100 - 10*z), so each feature's mean SHAP
    # contribution converts to plus-points by the same -10/std factor -
    # identical convention to deception_model.py's own breakdown.
    plus_contributions = dict(zip(ALL_FEATURES, (-10 * shap_values.mean(axis=0) / season_std).tolist()))

    lag_slot_breakdown = []
    for lag in range(1, MAX_LAG + 1):
        commit = plus_contributions[f"lag{lag}_commit_distance"]
        plate = plus_contributions[f"lag{lag}_plate_distance"]
        shape = plus_contributions[f"lag{lag}_shape_break"]
        ptype = plus_contributions[f"lag{lag}_pitch_type"]
        lag_slot_breakdown.append(dict(
            lag=lag, commit_distance=commit, plate_distance=plate, shape_break=shape,
            pitch_type=ptype, total=commit + plate + shape + ptype,
        ))

    return dict(
        n_pitches=len(feat),  # lag-paired rows actually scored - same meaning as leaderboard()'s "total_pitches"
        raw_pitches=len(raw_df),  # unfiltered season pitch count, before the 1st-pitch-of-PA lag exclusion
        score=score,
        lag_slot_breakdown=lag_slot_breakdown,
    )
