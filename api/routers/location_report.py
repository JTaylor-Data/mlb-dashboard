"""
/api/location/* - Location+ v1, promoted to live-serving. The static parts
of GET /report (features, CV, SHAP importance, the validation battery -
reliability incl. the direct Stuff+ comparison, year-over-year, cross-model
correlation, out-of-sample predictive validity, the count/count-interaction
sanity check, the stand-inclusion ablation) come straight off the
training-time artifacts in training/location_v1_artifacts/ and
models/location_plus_v1_report.json (regenerate by re-running
training/train_location_model.py + validate_location_model.py +
location_predictive_validity.py + location_count_interaction_check.py +
location_stand_ablation.py); the leaderboard and per-pitcher breakdown are
scored live per season via analytics/location_model.py so a season added
after the model was last trained still works, same pattern
api/routers/tunneling_report.py's v3 endpoints already use.
"""
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from api.cache import ttl_cache
from api.db import get_conn
from analytics.location_model import _raw_predictions, leaderboard as location_leaderboard, pitcher_breakdown

router = APIRouter(prefix="/api/location")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "training" / "location_v1_artifacts"

MIN_PITCHES_QUALIFY = 100


@ttl_cache()
def _raw_predictions_table(conn, season):
    """League-wide raw predictions for one season - cached because scoring
    a full season is the expensive part; the leaderboard and any
    per-pitcher SHAP lookup below reuse this instead of re-scoring the
    league each time. Same pattern as the other three models' own report
    routers."""
    return _raw_predictions(conn, season)


@router.get("/report")
def get_report(season: int, conn=Depends(get_conn)):
    """Model architecture (7 features, CV, SHAP importance, Location+-
    specific shrinkage K's) and the full validation battery, read straight
    off the training-time artifacts - plus a live-scored leaderboard for
    the requested season."""
    report_path = MODELS_DIR / "location_plus_v1_report.json"
    validation_path = ARTIFACTS_DIR / "location_v1_validation.json"
    predictive_path = ARTIFACTS_DIR / "location_v1_predictive_validity.json"
    interaction_path = ARTIFACTS_DIR / "location_v1_count_interaction.json"
    ablation_path = ARTIFACTS_DIR / "location_v1_stand_ablation.json"
    if not report_path.exists() or not validation_path.exists():
        raise HTTPException(404, "No Location+ artifacts found - run train_location_model.py + validate_location_model.py first")

    report = json.loads(report_path.read_text())
    validation = json.loads(validation_path.read_text())
    predictive = json.loads(predictive_path.read_text()) if predictive_path.exists() else None
    interaction = json.loads(interaction_path.read_text()) if interaction_path.exists() else None
    ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else None

    lb = location_leaderboard(conn, season, min_pitches=MIN_PITCHES_QUALIFY)

    return {
        "season": season,
        "n_training_rows": report["n_training_rows"],
        "all_features": report["all_features"],
        "cv_folds": report["cv_folds"],
        "cv_r2_mean": report["cv_r2_mean"],
        "cv_mae_mean": report["cv_mae_mean"],
        "mean_abs_shap": report["mean_abs_shap"],
        "shrinkage_k_by_pitch_type": report["shrinkage_k_by_pitch_type"],
        "shrinkage_fallback_k": report["shrinkage_fallback_k"],
        "reliability": validation["reliability"],
        "year_over_year": validation["year_over_year"],
        "cross_model_correlation": validation["cross_model_correlation"],
        "fastest_fb_velo_correlation": validation["fastest_fb_velo_correlation"],
        "predictive_validity": predictive,
        "count_interaction": interaction,
        "stand_ablation": ablation,
        "leaderboard": lb,
    }


@router.get("/pitcher/{pitcher_id}")
def get_pitcher_breakdown(pitcher_id: int, season: int, conn=Depends(get_conn)):
    """One pitcher's Location+ score plus a live SHAP breakdown of which
    inputs (location, count, pitch type, stand, season) are driving it."""
    league = _raw_predictions_table(conn, season)
    if league.empty:
        raise HTTPException(404, f"No Location+ scored pitches for season {season}")
    breakdown = pitcher_breakdown(conn, season, pitcher_id, league, min_pitches=MIN_PITCHES_QUALIFY)
    if not breakdown:
        raise HTTPException(404, f"No Location+ data for pitcher {pitcher_id} in {season} (needs {MIN_PITCHES_QUALIFY}+ pitches that season)")
    return breakdown
