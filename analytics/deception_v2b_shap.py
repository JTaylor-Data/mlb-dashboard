"""
Live per-pitcher SHAP breakdown for Deception+ v2b, the active cross-fitted
ensemble (see models/deception_plus_v2b_meta.json). v2b's own scores are
precomputed (api/routers/deception_report.py's _bracket() just reads the
parquet) because the full cross-fit requires scoring every pitcher-season
through the fold that held them out - too slow to redo league-wide on every
request. But a *single* pitcher's SHAP breakdown is cheap: pull that one
pitcher-season's four mean features from the DB, feed them through the three
rep models for their fold (pitcher_id % 5, the same fold assignment used at
training time), and average. This reuses the exact eb_weight/season_mean/
season_std already stored in the scores parquet for that pitcher-season, so
the breakdown is consistent with the precomputed headline score rather than
a second, slightly different computation.
"""
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURES = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle"]
N_FOLDS = 5
REPS = [0, 1, 2]

_model_cache: dict = {}


def _load_fold_model(fold: int, rep: int) -> lgb.Booster:
    key = (fold, rep)
    if key not in _model_cache:
        path = MODELS_DIR / f"deception_plus_v2b_fold{fold}_rep{rep}.txt"
        booster = lgb.Booster(model_file=str(path))
        _model_cache[key] = (booster, shap.TreeExplainer(booster))
    return _model_cache[key]


def _pitcher_season_features(conn, season: str, pitcher_id: int) -> pd.Series | None:
    """Mean of the four raw release features over this pitcher's qualifying
    pitches that season - same aggregation train_cv_pitcher_season() uses."""
    df = pd.read_sql("""
        SELECT p_throws, release_pos_x, release_pos_z, release_extension, arm_angle
        FROM pitches
        WHERE season = ? AND pitcher_id = ?
          AND pitch_type NOT IN ('PO','UN','EP')
          AND release_pos_x IS NOT NULL AND release_pos_z IS NOT NULL
          AND release_extension IS NOT NULL AND arm_angle IS NOT NULL
    """, conn, params=(int(season), pitcher_id))
    if df.empty:
        return None
    sign = -1 if (df["p_throws"] == "R").iloc[0] else 1
    df["release_pos_x_as"] = df["release_pos_x"] * sign
    return df[FEATURES].mean()


def pitcher_shap_breakdown(conn, season: str, pitcher_id: int, score_row: dict) -> dict | None:
    """Live SHAP contributions (in Deception+ points) for one pitcher-season.

    score_row: the matching row from deception_v2b_scores.parquet (as a
    dict), supplying eb_weight/season_mean/season_std so the point-scale
    conversion matches the pitcher's actual published score exactly."""
    feats = _pitcher_season_features(conn, season, pitcher_id)
    if feats is None:
        return None

    fold = pitcher_id % N_FOLDS
    row = feats[FEATURES].to_frame().T

    shap_sum = None
    raw_preds = []
    for rep in REPS:
        booster, explainer = _load_fold_model(fold, rep)
        values = explainer.shap_values(row)[0]
        raw_preds.append(booster.predict(row)[0])
        shap_sum = values if shap_sum is None else shap_sum + values
    mean_shap = shap_sum / len(REPS)
    mean_raw_pred = sum(raw_preds) / len(raw_preds)

    eb_weight = score_row["eb_weight"]
    season_std = score_row["season_std"] or 1.0
    points_factor = -10.0 * eb_weight / season_std

    contributions = sorted(
        (
            (feat_name, float(mean_shap[i] * points_factor), float(feats[feat_name]))
            for i, feat_name in enumerate(FEATURES)
        ),
        key=lambda x: -abs(x[1]),
    )

    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "fold": fold,
        "n_pitches_live": None,
        "raw_pred_check": float(mean_raw_pred),
        "contributions": contributions,
    }
