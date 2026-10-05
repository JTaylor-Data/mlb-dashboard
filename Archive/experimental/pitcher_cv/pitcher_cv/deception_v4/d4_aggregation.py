"""
Deception+ aggregation - rework to median + IQR, replacing mean (Part D).

For a given pitcher-season, aggregate plant/release timing and derived
features (arm speed, stride angle, etc.) using median + IQR instead of mean.
This is more robust against outliers (bad detections, pitch-specific outliers)
and surfaces the spread (IQR) so downstream analysis can weight certainty.
"""
import numpy as np


def aggregate_features(times_list, feature_name="feature"):
    """Aggregate a list of feature values (e.g., plant times across valid reps).

    Returns a dict with:
    - median: 50th percentile
    - iqr: interquartile range (75th - 25th percentile)
    - q25, q75: the bounds themselves
    - n_valid: count of non-None values used
    """
    values = [v for v in times_list if v is not None]
    n_valid = len(values)

    if n_valid == 0:
        return {"median": None, "iqr": None, "q25": None, "q75": None, "n_valid": 0}

    if n_valid == 1:
        return {
            "median": float(values[0]),
            "iqr": 0.0,
            "q25": float(values[0]),
            "q75": float(values[0]),
            "n_valid": 1,
        }

    q25 = float(np.percentile(values, 25))
    q50 = float(np.percentile(values, 50))
    q75 = float(np.percentile(values, 75))
    iqr = q75 - q25

    return {
        "median": q50,
        "iqr": iqr,
        "q25": q25,
        "q75": q75,
        "n_valid": n_valid,
    }


def per_pitcher_season_aggregation(results):
    """Given a list of per-pitch results (each with plant/release times + derived
    features), aggregate by pitcher-season.

    results: list of dicts from d4_pipeline.process_one(), each with:
    - tag: str (pitcher name, or "pitcher_id-season")
    - ok: bool
    - phase_times: dict with foot_plant, release times in seconds
    - (other features as added)

    Returns a dict keyed by pitcher-season, with aggregated medians + IQRs.
    """
    by_pitcher_season = {}

    for res in results:
        if not res.get("ok"):
            continue

        tag = res.get("tag")
        if not tag:
            continue

        # For now, assume tag is pitcher_id or pitcher_name.
        # In a real pipeline, extract season from metadata.
        key = tag  # Would be f"{pitcher_id}-{season}" in real use

        if key not in by_pitcher_season:
            by_pitcher_season[key] = {"plants": [], "releases": []}

        pt = res.get("phase_times", {})
        if pt.get("foot_plant") is not None:
            by_pitcher_season[key]["plants"].append(pt["foot_plant"])
        if pt.get("release") is not None:
            by_pitcher_season[key]["releases"].append(pt["release"])

    # Aggregate per pitcher-season
    aggregated = {}
    for pitcher_season, features in by_pitcher_season.items():
        aggregated[pitcher_season] = {
            "plant": aggregate_features(features["plants"], "plant"),
            "release": aggregate_features(features["releases"], "release"),
        }

    return aggregated
