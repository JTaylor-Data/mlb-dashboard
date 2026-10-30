"""
/api/location/* - Location+ v3, promoted to live-serving 2026-10-02.

Architecture: five LightGBM classifiers (xSwing, xCalledStrike, xWhiff, xFoul,
BIP×EV) combined with a count-specific run-value table to produce per-pitch
expected run value (xRV). Pitcher-season aggregation → plus-scale (100 = league
mean, 10 pts = 1 SD). S1 baseline adjustment adopted (R2_mix=0.338 >= 0.10;
S1 improves BB% corr vs S0). No shrinkage (split-half r_SB at 300 = 0.815 >= 0.70).
2020 excluded from training; scored by averaging season=2019 and season=2021
model predictions.

v2 fallback: if models/location_plus_v3_*.txt are missing the scorer falls
back to v2 automatically — LOCATION_ACTIVE_VERSION in location_v3_model.py
reflects the live version. Rollback: delete or rename location_plus_v3_*.txt
and the module falls back to v2.

Leaderboard / pitcher endpoints scored live per season via
analytics/location_v3_model.py so future seasons work without retraining.
Static validation artefacts served from diagnostics/location_v3/
(gate_results.json, validation_results.json, scoring_config.json).
"""
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from api.cache import ttl_cache
from api.db import get_conn
from analytics.location_v3_model import (
    LOCATION_ACTIVE_VERSION,
    _raw_predictions,
    leaderboard as location_leaderboard,
    pitcher_breakdown,
    model_meta,
)

router = APIRouter(prefix="/api/location")

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
DIAG_V3 = ROOT / "diagnostics" / "location_v3"

MIN_PITCHES_QUALIFY = 300   # v3 qualification threshold


@ttl_cache()
def _raw_predictions_table(conn, season):
    """League-wide raw predictions for one season — cached because scoring
    a full season is the expensive part; leaderboard and per-pitcher
    endpoints reuse this result."""
    return _raw_predictions(conn, season)


@ttl_cache()
def _pitcher_breakdown_cached(conn, season, pitcher_id):
    """Per-pitcher Location+ breakdown — cached."""
    league = _raw_predictions_table(conn, season)
    if league.empty:
        return None
    return pitcher_breakdown(conn, season, pitcher_id, league,
                             min_pitches=MIN_PITCHES_QUALIFY)


@router.get("/meta")
def get_model_meta():
    """Model version, hash, features, training seasons, scoring config,
    gate results. Used by the frontend banner and under-the-hood cards."""
    return model_meta()


@router.get("/report")
def get_report(season: int, conn=Depends(get_conn)):
    """Validation battery (scoring config, gate results, outcome
    correlations, split-half reliability, YoY stability, next-season
    predictive validity) plus a live-scored leaderboard for the season."""
    scoring_path = DIAG_V3 / "scoring_config.json"
    gate_path = DIAG_V3 / "gate_results.json"
    validation_path = DIAG_V3 / "validation_results.json"

    if not scoring_path.exists():
        raise HTTPException(
            404,
            "Location+ v3 scoring config not found — run "
            "training/train_location_v3.py first",
        )

    # Use utf-8-sig to handle optional BOM from manual file creation
    scoring = json.loads(scoring_path.read_text(encoding="utf-8-sig"))
    gates = (
        json.loads(gate_path.read_text(encoding="utf-8-sig")) if gate_path.exists() else {}
    )
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8-sig"))
        if validation_path.exists() else {}
    )

    lb = location_leaderboard(conn, season, min_pitches=MIN_PITCHES_QUALIFY)

    return {
        "season": season,
        "model_version": LOCATION_ACTIVE_VERSION,
        "scoring_config": scoring,
        "gates": gates,
        "validation": validation,
        "leaderboard": lb,
    }


@router.get("/leaderboard")
def get_leaderboard(season: int, n: int = 10, conn=Depends(get_conn)):
    """Season leaderboard — top/bottom by Location+ score.
    Qualification: 300 pitches (111 for 2020)."""
    min_p = 111 if season == 2020 else MIN_PITCHES_QUALIFY
    return location_leaderboard(conn, season, min_pitches=min_p, top_n=n)


@router.get("/pitcher/{pitcher_id}")
def get_pitcher_breakdown(pitcher_id: int, season: int, conn=Depends(get_conn)):
    """One pitcher's Location+ v3 score and event-component breakdown."""
    breakdown = _pitcher_breakdown_cached(conn, season, pitcher_id)
    if not breakdown:
        raise HTTPException(
            404,
            f"No Location+ data for pitcher {pitcher_id} in {season} "
            f"(needs {MIN_PITCHES_QUALIFY}+ pitches that season; "
            "2020 minimum is 111)",
        )
    return breakdown
