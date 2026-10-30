"""
Location+ v1 - runtime scoring. Loads the model trained offline by
training/train_location_model.py and scores pitches on demand for live
requests (league-wide leaderboard + per-pitcher SHAP breakdown), same
live-serving split as stuff_model.py/deception_model.py/
tunneling_v3_model.py (this file is the request-serving counterpart;
training/validation live under training/).

Promoted to live-serving per direct instruction - previously an offline-
only proof of concept (see Methodology/ - no write-up yet exists for this
one; the training-time report/validation JSONs under
training/location_v1_artifacts/ are the record of how this model was
built and checked).
"""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from analytics.location_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, RAW_QUERY_COLUMNS, engineer_features,
    shrinkage_params, shrunk_group_means, to_plus_scale,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BOOSTER_PATH = MODELS_DIR / "location_plus_v1.txt"
META_PATH = MODELS_DIR / "location_plus_v1_meta.json"

RAW_PREDICTIONS_CACHE_TTL_SECONDS = 600
_raw_predictions_cache = {}
_model_cache = {}


def raw_predictions_if_cached(season):
    """Return cached raw predictions if available and fresh, else None. Non-blocking."""
    import time as _time
    cached = _raw_predictions_cache.get(season)
    if cached is not None and (_time.monotonic() - cached[1]) < RAW_PREDICTIONS_CACHE_TTL_SECONDS:
        return cached[0]
    return None


def _load():
    if "model" not in _model_cache:
        _model_cache["model"] = lgb.Booster(model_file=str(BOOSTER_PATH))
    return _model_cache["model"]


def _get_explainer():
    if "explainer" not in _model_cache:
        _model_cache["explainer"] = shap.TreeExplainer(_load())
    return _model_cache["explainer"]


def _load_meta():
    if "meta" not in _model_cache:
        import json
        _model_cache["meta"] = json.loads(META_PATH.read_text())
    return _model_cache["meta"]


def _prepare_features(raw_df):
    meta = _load_meta()
    feat = engineer_features(raw_df)
    for col in CATEGORICAL_FEATURES:
        feat[col] = pd.Categorical(feat[col].astype(str), categories=meta["category_levels"][col])
    return feat


def _raw_predictions(conn, season):
    """Full-season per-pitch DataFrame: pitcher_id, pitch_type, pred - the
    shared basis for the league-wide plus-scale baseline, the shrinkage
    constants, and any single pitcher's SHAP breakdown. Short-lived
    in-process cache, same TTL/pattern as the other three models' own
    _raw_predictions."""
    cached = _raw_predictions_cache.get(season)
    if cached is not None and (time.monotonic() - cached[1]) < RAW_PREDICTIONS_CACHE_TTL_SECONDS:
        return cached[0]

    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season = ? AND delta_run_exp IS NOT NULL
    """, conn, params=(season,))
    if df.empty:
        out = pd.DataFrame(columns=["pitcher_id", "pitch_type", "pred"])
        _raw_predictions_cache[season] = (out, time.monotonic())
        return out

    feat = _prepare_features(df)
    model = _load()
    preds = model.predict(feat[ALL_FEATURES])
    out = feat[["pitcher_id", "pitch_type"]].copy()
    out["pred"] = preds
    _raw_predictions_cache[season] = (out, time.monotonic())
    return out


def clear_cache():
    _raw_predictions_cache.clear()


def official_location_plus(raw):
    """The 'real' displayed score for every pitcher in `raw`: each
    (pitcher, pitch_type) mean shrunk toward that pitch type's own league
    mean (empirical-Bayes, Location+-specific K's - see
    analytics/location_features.py), then usage-weight-averaged across
    pitch types to one overall per-pitcher score. Same shrink-then-
    aggregate shape as Stuff+'s own compute_stuff_plus."""
    if raw.empty or raw["pred"].std() == 0:
        return pd.DataFrame(columns=["pitcher_id", "score", "total_pitches"])
    k_by_type, fallback_k, type_means = shrinkage_params(raw)
    shrunk = shrunk_group_means(raw, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
    shrunk["score"] = to_plus_scale(shrunk["shrunk_mean"], raw["pred"].mean(), raw["pred"].std())

    def _wavg(g):
        return pd.Series({
            "score": np.average(g["score"], weights=g["n_pitches"]),
            "total_pitches": g["n_pitches"].sum(),
        })
    return shrunk.groupby("pitcher_id").apply(_wavg, include_groups=False).reset_index()


def leaderboard(conn, season, min_pitches=100, top_n=15):
    league = _raw_predictions(conn, season)
    scores = official_location_plus(league)
    q = scores[scores["total_pitches"] >= min_pitches]
    if q.empty:
        return {"n_qualified": 0, "top": [], "bottom": []}

    ids = q["pitcher_id"].astype(int).tolist()
    placeholders = ",".join("?" * len(ids))
    names = pd.read_sql(f"""
        SELECT p.player_id AS pitcher_id, p.name, t.abbreviation AS team_abbrev
        FROM players p LEFT JOIN teams t ON p.team_id = t.team_id
        WHERE p.player_id IN ({placeholders})
    """, conn, params=ids)
    merged = q.merge(names, on="pitcher_id", how="left")

    def _rows(d):
        return [
            dict(pitcher_id=int(r.pitcher_id), name=r.name, team_abbrev=r.team_abbrev,
                 total_pitches=int(r.total_pitches), score=round(float(r.score), 1))
            for r in d.itertuples()
        ]
    top = merged.sort_values("score", ascending=False).head(top_n)
    bottom = merged.sort_values("score", ascending=True).head(top_n)
    return {"n_qualified": int(len(q)), "top": _rows(top), "bottom": _rows(bottom)}


def pitcher_score(conn, season, pitcher_id, league_raw=None, min_pitches=100):
    if league_raw is None:
        league_raw = _raw_predictions(conn, season)
    scores = official_location_plus(league_raw)
    row = scores[(scores["pitcher_id"] == pitcher_id) & (scores["total_pitches"] >= min_pitches)]
    if row.empty:
        return None
    return float(row["score"].iloc[0])


def pitcher_breakdown(conn, season, pitcher_id, league_raw, min_pitches=100):
    """One pitcher's Location+ score plus a live SHAP breakdown of which
    inputs (plate_x/plate_z/balls/strikes/pitch_type/stand/season) are
    driving it, in plus-scale points - same role Stuff+'s/Deception+'s own
    per-pitcher breakdowns play. Returns None if the pitcher doesn't clear
    min_pitches that season."""
    if league_raw.empty or league_raw["pred"].std() == 0:
        return None
    season_mean, season_std = league_raw["pred"].mean(), league_raw["pred"].std()

    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season = ? AND pitcher_id = ? AND delta_run_exp IS NOT NULL
    """, conn, params=(season, pitcher_id))
    if len(df) < min_pitches:
        return None
    feat = _prepare_features(df)
    if feat.empty:
        return None

    model = _load()
    preds = model.predict(feat[ALL_FEATURES])
    score = float(to_plus_scale(preds.mean(), season_mean, season_std))

    explainer = _get_explainer()
    shap_values = explainer.shap_values(feat[ALL_FEATURES])
    plus_contributions = dict(zip(ALL_FEATURES, (-10 * shap_values.mean(axis=0) / season_std).tolist()))

    def _typical(f):
        if f not in ("plate_x", "plate_z", "balls", "strikes"):
            return feat[f].astype(str).mode().iat[0]
        return float(feat[f].mean())

    contributions = sorted(
        ((f, plus_contributions[f], _typical(f)) for f in ALL_FEATURES),
        key=lambda row: -abs(row[1]),
    )

    return dict(n_pitches=len(feat), score=score, contributions=contributions)
