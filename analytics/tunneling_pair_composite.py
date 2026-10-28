"""
Tunneling+ per-pitch-type-pair composite (hand-built, no ML model).

Composite formula: (-commit_distance_z + plate_distance_z + shape_break_plus_z) / 3
displayed as plus-scale (100-centered, 10=1SD; higher = better tunnel).

Sign decisions from Part 2 sign check (14,727 pairs, pooled 2021-2026):
  commit_distance:  r=+0.138 [+0.122, +0.153] -> use NEGATIVE (larger at commit = worse)
  divergence:       r=+0.006 [-0.010, +0.023] -> was DROPPED (CI crosses zero)
  plate_distance:   not separately tested; larger separation at plate = better, use POSITIVE
  shape_break_plus: r=-0.021 [-0.037, -0.004] -> use POSITIVE (larger = better)

plate_distance replaces divergence per user spec (2026-10-28).
"""
import numpy as np
import pandas as pd

from analytics.tunneling_features import (
    load_pitches, build_commit_point_table,
    pairwise_distance_features, add_pair_features,
    pitch_type_centroids,
)

MIN_PITCHES_PER_TYPE = 100


def _raw_pairs(conn, season, pitcher_id=None):
    """Load pitches, compute pairwise distance features + add_pair_features,
    return pairs DataFrame with divergence column. Returns empty DataFrame
    when no qualifying pairs exist."""
    df = load_pitches(conn, season, pitcher_id=pitcher_id)
    if df.empty:
        return pd.DataFrame()
    commit_table = build_commit_point_table(df)
    if commit_table.empty:
        return pd.DataFrame()
    centroids = pitch_type_centroids(df)
    pairs = pairwise_distance_features(df, centroids, commit_table,
                                       min_pitches_per_type=MIN_PITCHES_PER_TYPE)
    if pairs.empty:
        return pd.DataFrame()
    pairs = add_pair_features(pairs)
    pairs["divergence"] = pairs["plate_distance"] - pairs["commit_distance"]
    return pairs


def _compute_composite(pairs_df, pop_stats):
    """Apply z-score composite to a pairs DataFrame in place of a copy.
    pop_stats keys: commit_distance_mean/std, plate_distance_mean/std, shape_break_plus_mean/std."""
    df = pairs_df.copy()
    df["commit_distance_z"] = (
        (df["commit_distance"] - pop_stats["commit_distance_mean"])
        / pop_stats["commit_distance_std"]
    )
    df["plate_distance_z"] = (
        (df["plate_distance"] - pop_stats["plate_distance_mean"])
        / pop_stats["plate_distance_std"]
    )
    df["shape_break_plus_z"] = (
        (df["shape_break_plus"] - pop_stats["shape_break_plus_mean"])
        / pop_stats["shape_break_plus_std"]
    )
    raw_z = (-df["commit_distance_z"] + df["plate_distance_z"] + df["shape_break_plus_z"]) / 3
    df["composite_score"] = 100.0 + 10.0 * raw_z
    return df


def _pop_stats_from_pairs(pairs):
    """Extract population stats from a pairs DataFrame."""
    return {
        "commit_distance_mean": float(pairs["commit_distance"].mean()),
        "commit_distance_std": float(pairs["commit_distance"].std()),
        "plate_distance_mean": float(pairs["plate_distance"].mean()),
        "plate_distance_std": float(pairs["plate_distance"].std()),
        "shape_break_plus_mean": float(pairs["shape_break_plus"].mean()),
        "shape_break_plus_std": float(pairs["shape_break_plus"].std()),
    }


def season_all_pairs(conn, season):
    """All qualifying pitcher-season pairs for a season with composite scores.
    Z-scores are within-season (population = all qualifying pairs that season).
    Returns DataFrame sorted by composite_score descending.

    This is the slow call (~minutes for a full season). Cache it in callers."""
    pairs = _raw_pairs(conn, season)
    if pairs.empty:
        return pd.DataFrame()
    pop_stats = _pop_stats_from_pairs(pairs)
    pairs = _compute_composite(pairs, pop_stats)
    cols = ["pitcher_id", "season", "type_a", "type_b", "n_a", "n_b",
            "commit_distance", "plate_distance", "divergence", "shape_break_plus", "composite_score"]
    return pairs[cols].sort_values("composite_score", ascending=False).reset_index(drop=True)


def pitcher_pairs_from_season(all_pairs_df, pitcher_id):
    """Filter a precomputed season_all_pairs DataFrame to one pitcher.
    Returns list of dicts for JSON serialization, sorted by composite_score desc."""
    sub = all_pairs_df[all_pairs_df["pitcher_id"] == pitcher_id].copy()
    if sub.empty:
        return []
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "type_a": r["type_a"],
            "type_b": r["type_b"],
            "n_a": int(r["n_a"]),
            "n_b": int(r["n_b"]),
            "commit_distance": round(float(r["commit_distance"]), 2),
            "plate_distance": round(float(r["plate_distance"]), 2),
            "divergence": round(float(r["divergence"]), 2),
            "shape_break_plus": round(float(r["shape_break_plus"]), 2),
            "composite_score": round(float(r["composite_score"]), 1),
        })
    return rows
