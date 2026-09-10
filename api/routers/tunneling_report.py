"""
/api/tunneling/* - v1 endpoints serve the Tunneling+ v1 report (from
build_tunneling_report.py, read straight off disk). These stay exactly as
they are - api/routers/tunneling_3d.py's 3D trajectory viewer
(frontend/tunnel3d.html) depends on GET /pitcher/{id}'s exact pair-shaped
contract directly and unmodified.

/api/tunneling/v3/* are NEW, additive endpoints for Tunneling+ v3 (the
model promoted to "the standard" on the main Tunneling+ Model page,
frontend/tunneling-model.html) - a single trained model with no pair-level
concept at all, so it cannot reuse v1's endpoint shape. The static parts of
GET /v3/report (features, CV, SHAP/lag-slot importance, the validation
battery) come straight off the training-time artifacts in
training/tunneling_v3_artifacts/ (regenerate by re-running
training/train_tunneling_v3_model.py + tunneling_v3_validate.py +
tunneling_v3_predictive_validity.py); the leaderboard is scored live per
season via analytics/tunneling_v3_model.py so a season added after the
model was last trained still works.
"""
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from api.cache import ttl_cache
from api.db import get_conn
from analytics.tunneling_v3_model import _raw_predictions, leaderboard as v3_leaderboard, pitcher_breakdown

router = APIRouter(prefix="/api/tunneling")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
V3_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "training" / "tunneling_v3_artifacts"


def _pitchers_path(season: int) -> Path:
    return MODELS_DIR / f"tunneling_v1_pitchers_{season}.json"


@router.get("/pitcher/{pitcher_id}")
def get_pitcher_detail(pitcher_id: int, season: int):
    """One pitcher's pair-level detail for one season: every qualifying
    pitch-type pair with release/tunnel/plate distances and the resulting
    tunneling_plus (plus its two components and whether it's an FF/SI
    anchor pair)."""
    path = _pitchers_path(season)
    if not path.exists():
        raise HTTPException(404, f"No tunneling report for season {season} - run build_tunneling_report.py first")
    all_pitchers = json.loads(path.read_text())
    detail = all_pitchers.get(str(pitcher_id))
    if not detail:
        raise HTTPException(404, f"No tunneling detail for pitcher {pitcher_id} in {season} (may not qualify)")
    return detail


@ttl_cache()
def _v3_raw_predictions_table(conn, season):
    """League-wide raw predictions for one season - cached because scoring
    a full season's lag-feature pipeline is the expensive part; the
    leaderboard and any per-pitcher SHAP lookup below reuse this instead of
    re-scoring the league each time. Same pattern as
    stuff_model_report.py/deception_report.py."""
    return _raw_predictions(conn, season)


@router.get("/v3/report")
def get_v3_report(season: int, conn=Depends(get_conn)):
    """Tunneling+ v3: model architecture (28 lag-derived features, CV,
    SHAP/lag-slot importance) and the validation battery, read straight off
    the training-time artifacts - plus a live-scored leaderboard for the
    requested season (analytics/tunneling_v3_model.py), so the season
    selector has real effect even for a season scored after the model was
    last trained."""
    report_path = MODELS_DIR / "tunneling_v3_report.json"
    validation_path = V3_ARTIFACTS_DIR / "tunneling_v3_validation.json"
    predictive_path = V3_ARTIFACTS_DIR / "tunneling_v3_predictive_validity.json"
    if not report_path.exists() or not validation_path.exists():
        raise HTTPException(404, "No Tunneling+ v3 artifacts found - run train_tunneling_v3_model.py + tunneling_v3_validate.py first")

    report = json.loads(report_path.read_text())
    validation = json.loads(validation_path.read_text())
    predictive = json.loads(predictive_path.read_text()) if predictive_path.exists() else None

    lb = v3_leaderboard(conn, season)

    return {
        "season": season,
        "n_training_rows": report["n_training_rows"],
        "all_features": report["all_features"],
        "cv_folds": report["cv_folds"],
        "cv_r2_mean": report["cv_r2_mean"],
        "cv_mae_mean": report["cv_mae_mean"],
        "lag_slot_shap_summary": report["lag_slot_shap_summary"],
        "sequential_same_at_bat_test": validation["sequential_same_at_bat_test"],
        "year_over_year": validation["year_over_year"],
        "cross_model_correlation": validation["cross_model_correlation"],
        "fastest_fb_velo_correlation": validation["fastest_fb_velo_correlation"],
        "predictive_validity": predictive,
        "leaderboard": lb,
    }


@router.get("/v3/pitcher/{pitcher_id}")
def get_v3_pitcher_breakdown(pitcher_id: int, season: int, conn=Depends(get_conn)):
    """One pitcher's Tunneling+ v3 score plus a live SHAP breakdown grouped
    by lag slot - v3's real per-pitch architecture has no pair-level
    concept, so this is a lag-slot table rather than v1's pair table."""
    league = _v3_raw_predictions_table(conn, season)
    if league.empty:
        raise HTTPException(404, f"No Tunneling+ v3 scored pitches for season {season}")
    breakdown = pitcher_breakdown(conn, season, pitcher_id, league)
    if not breakdown:
        raise HTTPException(404, f"No Tunneling+ v3 data for pitcher {pitcher_id} in {season} (needs {season}'s own 100+ pitches and a qualifying commit point)")
    return breakdown
