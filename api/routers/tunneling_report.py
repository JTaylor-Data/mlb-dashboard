"""
/api/tunneling/* - the live Tunneling+ model: a hand-built, pair-type-
normalized geometric composite (analytics/tunneling_pair_composite.py), NOT
a trained ML model - no SHAP, no lag slots, no per-request scoring of a
fitted estimator. The exact same module (and, via _cached_season_pairs()
below, the exact same cached per-season pair table) that already feeds the
per-pitch-type-pair comparison on each pitcher's Overview page
(api/routers/players.py's _tunneling_pair_season), so the model page and the
player page are guaranteed to agree.

/api/tunneling/pitcher/{id} (no /v3/) is a SEPARATE, older, no-longer-
promoted hand-built composite (analytics/tunneling_model.py, decision/plate
geometric-mean scoring) read from static per-season JSON files - kept only
because it still powers the 3D tunneling viewer (frontend/tunnel3d.html via
api/routers/tunneling_3d.py's docstring reference). Left untouched here;
not the model this page or the About page describe.

The /v3/ path prefix on the endpoints below is legacy naming (this used to
serve a trained lag-feature regressor, since retired) - kept so the
frontend and any bookmarked links don't need to change again.
"""
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from api.cache import ttl_cache
from api.db import get_conn
from analytics.tunneling_pair_composite import (
    composite_leaderboard, pitcher_composite_detail, season_all_pairs,
)
from analytics.tunneling_v2_model import _raw_predictions as _v2_raw_predictions

router = APIRouter(prefix="/api/tunneling")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _pitchers_path(season: int) -> Path:
    return MODELS_DIR / f"tunneling_v1_pitchers_{season}.json"


@router.get("/pitcher/{pitcher_id}")
def get_pitcher_detail(pitcher_id: int, season: int):
    """One pitcher's pair-level detail from the OLDER, separate hand-built
    composite (decision_similarity_plus/plate_separation_plus geometric
    mean) that still powers the 3D tunneling viewer only - see module
    docstring. Not the model /v3/* below serves."""
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
    """NOT used by this page's own endpoints below (those use the pair
    composite). Kept only because api/warmup.py and other pages still badge
    off the older, separate five-classifier model (analytics/
    tunneling_v2_model.py) elsewhere on the site (probable pitchers, career,
    team pages) - this just keeps that model's cache warm. Do not wire new
    callers to this; see module docstring."""
    return _v2_raw_predictions(conn, season)


@ttl_cache()
def _cached_season_pairs(conn, season):
    """The season-wide pair table (every qualifying pitch-type pair, every
    pitcher, with composite scores) - the slow call (~minutes). Cached here
    AND separately in api/routers/players.py's _tunneling_pair_season();
    the two caches are independent stores (ttl_cache keys per decorated
    function), so each pays its own first-request cost, but both call the
    identical season_all_pairs() and therefore always agree."""
    return season_all_pairs(conn, season)


@router.get("/v3/report")
def get_v3_report(season: int, conn=Depends(get_conn)):
    """Leaderboard for the live Tunneling+ composite, for the requested
    season."""
    all_pairs = _cached_season_pairs(conn, season)
    lb = composite_leaderboard(conn, season, all_pairs_df=all_pairs)
    return {"season": season, "leaderboard": lb}


@router.get("/v3/pitcher/{pitcher_id}")
def get_v3_pitcher_breakdown(pitcher_id: int, season: int, conn=Depends(get_conn)):
    """One pitcher's live Tunneling+ composite breakdown: the headline
    (usage-weighted) score plus every qualifying pitch-type pair's own
    commit/plate/divergence distances and composite score - the same
    numbers shown on that pitcher's Overview page pair table."""
    all_pairs = _cached_season_pairs(conn, season)
    detail = pitcher_composite_detail(conn, season, pitcher_id, all_pairs_df=all_pairs)
    if not detail:
        raise HTTPException(404, f"No Tunneling+ data for pitcher {pitcher_id} in {season} (needs 2+ qualifying pitch types, {season} pitches, and a qualifying commit point)")
    return detail
