"""
/api/deception/* - serves Deception+ v2 scores (pre-computed via cross-fitted
training pipeline; no per-request model scoring). Season selector enabled.

v2 scores live in diagnostics/deception_v2/deception_v2_scores.parquet
and are resolved via _bracket() which reads from that file.
Use DECEPTION_ACTIVE_VERSION from config/model_policy.py to switch versions.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from config.model_policy import DECEPTION_ACTIVE_VERSION, DECEPTION_PLUS

router = APIRouter(prefix="/api/deception")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DIAG_DIR = Path(__file__).resolve().parent.parent.parent / "diagnostics"

# Scores paths by version
_SCORES_PATHS = {
    "v2": DIAG_DIR / "deception_v2" / "deception_v2_scores.parquet",
}
# Fallback: v1 report JSON for legacy
_V1_REPORT_PATH = MODELS_DIR / "deception_plus_v1_report_2026.json"


@lru_cache(maxsize=4)
def _load_scores(version: str) -> pd.DataFrame:
    """Load pre-computed scores for a given version. Cached per version."""
    path = _SCORES_PATHS.get(version)
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["pitcher_id"] = df["pitcher_id"].astype(int)
    df["season"] = df["season"].astype(str)
    return df


def _bracket(pitcher_id: int, season: str, version: str = None) -> dict | None:
    """Read one pitcher's score from the pre-computed parquet. No per-request model scoring."""
    if version is None:
        version = DECEPTION_ACTIVE_VERSION
    df = _load_scores(version)
    if df.empty:
        return None
    row = df[(df["pitcher_id"] == pitcher_id) & (df["season"] == season)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "pitcher_id": int(r["pitcher_id"]),
        "season": r["season"],
        "deception_plus": float(r.get("deception_plus_v2", r.get("deception_plus", 100.0))),
        "n_pitches": int(r.get("n_pitches", 0)),
        "qualified": bool(r.get("qualified", False)),
        "percentile": float(r["percentile"]) if pd.notna(r.get("percentile")) else None,
        "flags": r.get("flags", ""),
        "version": version,
    }


@router.get("/report")
def get_report():
    """The full validation report -- v2 returns the part4 validation results."""
    version = DECEPTION_ACTIVE_VERSION
    if version == "v2":
        report_path = DIAG_DIR / "deception_v2" / "validation_results.json"
        if report_path.exists():
            return json.loads(report_path.read_text())
    # Fallback to v1
    if _V1_REPORT_PATH.exists():
        return json.loads(_V1_REPORT_PATH.read_text())
    raise HTTPException(404, "No deception report found")


@router.get("/meta")
def get_meta():
    """Model meta: version, features, training config, seasons."""
    version = DECEPTION_ACTIVE_VERSION
    if version == "v2":
        meta_path = MODELS_DIR / "deception_plus_v2_meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
    raise HTTPException(404, f"No meta for version {version}")


@router.get("/leaderboard")
def get_leaderboard(season: str = None):
    """Top pitchers by Deception+ for a given season."""
    version = DECEPTION_ACTIVE_VERSION
    df = _load_scores(version)
    if df.empty:
        raise HTTPException(404, "No deception scores found")
    available = sorted(df["season"].unique())
    if season is None:
        season = available[-1]
    sub = df[(df["season"] == season) & (df["qualified"] == True)].copy()
    if sub.empty:
        raise HTTPException(404, f"No qualified scores for season {season}")
    score_col = "deception_plus_v2" if "deception_plus_v2" in sub.columns else "deception_plus"
    sub = sub.sort_values(score_col, ascending=False)
    return {
        "season": season,
        "available_seasons": available,
        "pitchers": sub[["pitcher_id", score_col, "n_pitches", "percentile"]].rename(
            columns={score_col: "deception_plus"}
        ).to_dict("records"),
    }


@router.get("/pitcher/{pitcher_id}")
def get_pitcher_breakdown(pitcher_id: int, season: str = None):
    """One pitcher's Deception+ score for a given season (defaults to latest available)."""
    version = DECEPTION_ACTIVE_VERSION
    df = _load_scores(version)
    if df.empty:
        raise HTTPException(404, "No deception scores found")
    available_seasons = sorted(df[df["pitcher_id"] == pitcher_id]["season"].unique(), reverse=True)
    if not available_seasons:
        raise HTTPException(404, f"No scores for pitcher {pitcher_id}")
    if season is None:
        season = available_seasons[0]
    result = _bracket(pitcher_id, season, version)
    if result is None:
        raise HTTPException(404, f"No score for pitcher {pitcher_id} in {season}")
    result["available_seasons"] = available_seasons
    return result
