"""
Tunneling+ per-pitch-type-pair composite (hand-built, no ML model).

Composite formula: (-commit_residual_z + divergence_residual_z + shape_break_plus_z) / 3
displayed as plus-scale (100-centered, 10=1SD; higher = better tunnel).

Pair-type normalization (added 2026-10-28): commit_distance and divergence are
de-meaned by (type_a, type_b) before z-scoring. This removes the confound where
CU/FF pairs score high simply because curveballs break more in absolute terms.
The composite answers "is this CU/FF better-tunneled than the average CU/FF?"
rather than "does CU/FF tunnel better than CH/SI by physics?"

  commit_residual   = commit_distance - mean_commit_for_(type_a,type_b)_that_season
  divergence_residual = divergence - mean_divergence_for_(type_a,type_b)_that_season

shape_break_plus is already pair-type-normalized (actual minus expected mid-trajectory
distance given pitch types), so it is z-scored directly.

Sign decisions from Part 2 sign check (14,727 pairs, pooled 2021-2026):
  commit_distance:  r=+0.138 -> use NEGATIVE residual (larger at commit = worse)
  divergence:       r=+0.006, but divergence > plate_distance on whiff CI and YoY xFIP
  shape_break_plus: r=-0.021 -> use POSITIVE (larger = better)
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


def _compute_pair_type_means(pairs):
    """Mean commit_distance and divergence per (type_a, type_b) pair type."""
    return pairs.groupby(["type_a", "type_b"])[["commit_distance", "divergence"]].mean().rename(
        columns={"commit_distance": "mean_commit", "divergence": "mean_divergence"}
    )


def _residual_stats(pairs_df, pair_type_means):
    """Compute residuals (de-meaned by pair type) and return population stats."""
    df = pairs_df.copy()
    df = df.join(pair_type_means, on=["type_a", "type_b"])
    df["commit_residual"] = df["commit_distance"] - df["mean_commit"]
    df["divergence_residual"] = df["divergence"] - df["mean_divergence"]
    return {
        "commit_residual_mean": float(df["commit_residual"].mean()),
        "commit_residual_std": float(df["commit_residual"].std()),
        "divergence_residual_mean": float(df["divergence_residual"].mean()),
        "divergence_residual_std": float(df["divergence_residual"].std()),
        "shape_break_mean": float(df["shape_break_plus"].mean()),
        "shape_break_std": float(df["shape_break_plus"].std()),
    }


def _compute_composite(pairs_df, pair_type_means, residual_stats):
    """
    Apply pair-type-normalized z-score composite.
    pair_type_means: DataFrame indexed by (type_a, type_b) with mean_commit, mean_divergence
    residual_stats: dict with commit_residual_mean/std, divergence_residual_mean/std, shape_break_mean/std
    """
    df = pairs_df.copy()
    df = df.join(pair_type_means, on=["type_a", "type_b"])
    df["commit_residual"] = df["commit_distance"] - df["mean_commit"]
    df["divergence_residual"] = df["divergence"] - df["mean_divergence"]

    df["commit_z"] = (
        (df["commit_residual"] - residual_stats["commit_residual_mean"])
        / residual_stats["commit_residual_std"]
    )
    df["divergence_z"] = (
        (df["divergence_residual"] - residual_stats["divergence_residual_mean"])
        / residual_stats["divergence_residual_std"]
    )
    df["shape_break_z"] = (
        (df["shape_break_plus"] - residual_stats["shape_break_mean"])
        / residual_stats["shape_break_std"]
    )

    raw_z = (-df["commit_z"] + df["divergence_z"] + df["shape_break_z"]) / 3
    df["composite_score"] = 100.0 + 10.0 * raw_z
    return df


def season_all_pairs(conn, season):
    """All qualifying pitcher-season pairs for a season with composite scores.
    Scores are pair-type-normalized within-season.
    Returns DataFrame sorted by composite_score descending.

    This is the slow call (~minutes for a full season). Cache it in callers."""
    pairs = _raw_pairs(conn, season)
    if pairs.empty:
        return pd.DataFrame()
    pair_type_means = _compute_pair_type_means(pairs)
    res_stats = _residual_stats(pairs, pair_type_means)
    pairs = _compute_composite(pairs, pair_type_means, res_stats)
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
