"""
Deception+ v1 - runtime scoring. Loads the model trained by
train_deception_model.py and scores pitches on demand (per-pitcher lookups
for the model-internals page's "Inspect a Pitcher" search) - the
training/validation-report side lives in train_deception_model.py, this
file is the live-request-serving counterpart, same split stuff_model.py /
train_stuff_model.py already use.
"""
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import shap

from analytics.deception_features import ALL_FEATURES, CATEGORICAL_FEATURES, RAW_QUERY_COLUMNS, engineer_features, load_bio
from analytics.location_features import to_plus_scale as _to_plus_scale

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BOOSTER_PATH = MODELS_DIR / "deception_plus_v1.txt"

RAW_PREDICTIONS_CACHE_TTL_SECONDS = 600
_raw_predictions_cache = {}
_model_cache = {}


def _load():
    if "model" not in _model_cache:
        _model_cache["model"] = lgb.Booster(model_file=str(BOOSTER_PATH))
    return _model_cache["model"]


def _get_explainer():
    """Cached the same way stuff_model.py's _get_explainer() is - building
    a TreeExplainer has real, avoidable overhead if done fresh on every
    per-pitcher lookup."""
    if "explainer" not in _model_cache:
        _model_cache["explainer"] = shap.TreeExplainer(_load())
    return _model_cache["explainer"]


def _prepare_features(df, bio):
    feat = engineer_features(df, bio)
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype(str).astype("category")
    return feat


def _raw_predictions(conn, season):
    """Full-season per-pitch DataFrame: pitcher_id + raw predicted
    delta_run_exp - the shared basis for both the league-wide plus-scale
    baseline and any single pitcher's SHAP breakdown. Short-lived
    in-process cache, same pattern and TTL as stuff_model.py's
    _raw_predictions."""
    cached = _raw_predictions_cache.get(season)
    if cached is not None and (time.monotonic() - cached[1]) < RAW_PREDICTIONS_CACHE_TTL_SECONDS:
        return cached[0]

    conn_bio = load_bio(conn)
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season = ? AND delta_run_exp IS NOT NULL
    """, conn, params=(season,))
    if df.empty:
        out = pd.DataFrame(columns=["pitcher_id", "pred"])
        _raw_predictions_cache[season] = (out, time.monotonic())
        return out
    feat = _prepare_features(df, conn_bio)
    model = _load()
    preds = model.predict(feat[ALL_FEATURES])
    out = feat[["pitcher_id"]].copy()
    out["pred"] = preds
    _raw_predictions_cache[season] = (out, time.monotonic())
    return out


def clear_cache():
    _raw_predictions_cache.clear()


def pitcher_deception_breakdown(conn, season, pitcher_id, league_raw):
    """One pitcher's Deception+ score plus a SHAP feature breakdown (in
    plus-scale points, so it reads on the same terms as the headline
    number) - which inputs are pushing their prediction toward better or
    worse run value, and by how much. Returns None if the pitcher has no
    scored pitches that season."""
    if league_raw.empty or league_raw["pred"].std() == 0:
        return None
    season_mean, season_std = league_raw["pred"].mean(), league_raw["pred"].std()

    bio = load_bio(conn)
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season = ? AND pitcher_id = ? AND delta_run_exp IS NOT NULL
    """, conn, params=(season, pitcher_id))
    if df.empty:
        return None
    feat = _prepare_features(df, bio)
    if feat.empty:
        return None

    model = _load()
    preds = model.predict(feat[ALL_FEATURES])
    deception_plus = float(_to_plus_scale(preds.mean(), season_mean, season_std))

    explainer = _get_explainer()
    shap_values = explainer.shap_values(feat[ALL_FEATURES])
    # SHAP values are additive in raw delta_run_exp units; the plus-scale
    # transform is linear (100 - 10*z), so each feature's SHAP contribution
    # converts to plus-points by the same -10/std scale factor.
    plus_scale_contributions = -10 * shap_values.mean(axis=0) / season_std
    mean_feature_values = feat[ALL_FEATURES].apply(
        lambda col: col.mean() if col.dtype.kind in "fc" else col.mode().iloc[0]
    )

    contributions = sorted(
        (
            (feat_name, float(plus_scale_contributions[i]), (float(mean_feature_values[feat_name]) if mean_feature_values[feat_name] is not None and not isinstance(mean_feature_values[feat_name], str) else mean_feature_values[feat_name]))
            for i, feat_name in enumerate(ALL_FEATURES)
        ),
        key=lambda x: -abs(x[1]),
    )

    return dict(
        n_pitches=len(feat),
        deception_plus=deception_plus,
        contributions=contributions,
    )
