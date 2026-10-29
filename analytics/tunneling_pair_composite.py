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


def _raw_pairs_and_centroids(conn, season, pitcher_id=None):
    """Load pitches, compute pairwise distance features + add_pair_features,
    return (pairs, centroids). pairs has the divergence column added.
    Both empty when no qualifying pairs exist. centroids (per pitcher_id,
    season, pitch_type pitch counts) is exposed so callers can derive a
    pitcher's total qualifying-pitch count without a second DB pass."""
    df = load_pitches(conn, season, pitcher_id=pitcher_id)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    commit_table = build_commit_point_table(df)
    if commit_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    centroids = pitch_type_centroids(df)
    pairs = pairwise_distance_features(df, centroids, commit_table,
                                       min_pitches_per_type=MIN_PITCHES_PER_TYPE)
    if pairs.empty:
        return pd.DataFrame(), centroids
    pairs = add_pair_features(pairs)
    pairs["divergence"] = pairs["plate_distance"] - pairs["commit_distance"]
    return pairs, centroids


def _raw_pairs(conn, season, pitcher_id=None):
    """Load pitches, compute pairwise distance features + add_pair_features,
    return pairs DataFrame with divergence column. Returns empty DataFrame
    when no qualifying pairs exist."""
    pairs, _ = _raw_pairs_and_centroids(conn, season, pitcher_id=pitcher_id)
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
    cols = ["pitcher_id", "season", "type_a", "type_b", "n_a", "n_b", "pair_weight",
            "commit_distance", "tunnel_spread_plus", "plate_distance", "late_divergence_plus",
            "divergence", "shape_break_plus", "composite_score"]
    return pairs[cols].sort_values("composite_score", ascending=False).reset_index(drop=True)


def pitcher_pairs_from_season(all_pairs_df, pitcher_id, total_pitches=None):
    """Filter a precomputed season_all_pairs DataFrame to one pitcher.
    Returns list of dicts for JSON serialization, sorted by composite_score desc.

    total_pitches, when given, is this pitcher-season's overall pitch count -
    used to add pct_a/pct_b (each pitch type's share of the season), which
    the player page uses to hide pairs built from a pitch thrown under 5% of
    the time. n_a/n_b alone aren't comparable across pitchers without it."""
    sub = all_pairs_df[all_pairs_df["pitcher_id"] == pitcher_id].copy()
    if sub.empty:
        return []
    rows = []
    for _, r in sub.iterrows():
        row = {
            "type_a": r["type_a"],
            "type_b": r["type_b"],
            "n_a": int(r["n_a"]),
            "n_b": int(r["n_b"]),
            "commit_distance": round(float(r["commit_distance"]), 2),
            "commit_distance_plus": round(float(r["tunnel_spread_plus"]), 0),
            "plate_distance": round(float(r["plate_distance"]), 2),
            "plate_distance_plus": round(float(r["late_divergence_plus"]), 0),
            "divergence": round(float(r["divergence"]), 2),
            "shape_break_plus": round(float(r["shape_break_plus"]), 2),
            "composite_score": round(float(r["composite_score"]), 1),
        }
        if total_pitches:
            row["pct_a"] = round(row["n_a"] / total_pitches * 100, 1)
            row["pct_b"] = round(row["n_b"] / total_pitches * 100, 1)
        rows.append(row)
    return rows


def _qualifying_total_pitches(conn, season):
    """Each qualifying pitcher's total pitch count across qualifying pitch
    types only (>= MIN_PITCHES_PER_TYPE) - cheap (a groupby count, no
    pairwise distances), so a caller that already has a cached pairs
    DataFrame doesn't need to repeat the O(n^2) distance pass just for
    this count."""
    df = load_pitches(conn, season)
    if df.empty:
        return pd.Series(dtype=float, name="total_pitches")
    centroids = pitch_type_centroids(df)
    qualifying = centroids[centroids["n_pitches"] >= MIN_PITCHES_PER_TYPE]
    return qualifying.groupby("pitcher_id")["n_pitches"].sum().rename("total_pitches")


def pitcher_composite_scores(conn, season, all_pairs_df=None):
    """League-wide, one row per qualifying pitcher: usage-weighted mean of
    that pitcher's own pair composite_scores (the SAME composite the player
    page's pair table shows, aggregated up into a single headline number).
    Weight per pair is pair_weight = sqrt(usage_a * usage_b), the same
    convention the pair-level league baselines already use - no FF/SI
    anchor bonus (unlike the old, no-longer-live tunneling_model.py v1).

    all_pairs_df, when given (a cached season_all_pairs() result, which now
    carries pair_weight), is reused instead of recomputing - this is the
    slow call (~minutes for a full season) otherwise."""
    if all_pairs_df is None:
        pairs, centroids = _raw_pairs_and_centroids(conn, season)
        if pairs.empty:
            return pd.DataFrame(columns=["pitcher_id", "tunneling_plus", "n_pairs", "total_pitches"])
        pair_type_means = _compute_pair_type_means(pairs)
        res_stats = _residual_stats(pairs, pair_type_means)
        pairs = _compute_composite(pairs, pair_type_means, res_stats)
        qualifying = centroids[centroids["n_pitches"] >= MIN_PITCHES_PER_TYPE]
        total_pitches = qualifying.groupby("pitcher_id")["n_pitches"].sum().rename("total_pitches")
    else:
        if all_pairs_df.empty:
            return pd.DataFrame(columns=["pitcher_id", "tunneling_plus", "n_pairs", "total_pitches"])
        pairs = all_pairs_df
        total_pitches = _qualifying_total_pitches(conn, season)

    pairs = pairs.copy()
    pairs["weighted_score"] = pairs["composite_score"] * pairs["pair_weight"]
    agg = pairs.groupby("pitcher_id").agg(
        _weight_sum=("pair_weight", "sum"),
        _weighted_sum=("weighted_score", "sum"),
        n_pairs=("pitcher_id", "size"),
    )
    agg["tunneling_plus"] = agg["_weighted_sum"] / agg["_weight_sum"]

    out = agg[["tunneling_plus", "n_pairs"]].join(total_pitches)
    return out.reset_index().dropna(subset=["tunneling_plus"])


MIN_PAIRS_LEADERBOARD = 2  # a pitcher with exactly 1 qualifying pair has no
    # averaging to smooth out pair-level noise - the Methodology doc's own
    # validation run flagged single-pair pitchers as dominating the extremes
    # for exactly this reason. Leaderboard-only gate; the player page's own
    # pair table has no such filter (a 1-pair pitcher's pairs are still real).


def composite_leaderboard(conn, season, top_n=15, all_pairs_df=None):
    """Top/bottom N pitchers by the pitcher-level composite score, with
    names/teams joined in - same shape (name, team_abbrev, score,
    total_pitches) every other model's leaderboard endpoint already returns,
    so the frontend leaderboard renderer is unchanged. Requires 2+ qualifying
    pairs (see MIN_PAIRS_LEADERBOARD). all_pairs_df: see pitcher_composite_scores."""
    g = pitcher_composite_scores(conn, season, all_pairs_df=all_pairs_df)
    g = g[g["n_pairs"] >= MIN_PAIRS_LEADERBOARD]
    if g.empty:
        return {"n_qualified": 0, "top": [], "bottom": []}

    ids = g["pitcher_id"].astype(int).tolist()
    names = pd.read_sql(
        f"SELECT p.player_id AS pitcher_id, p.name, t.abbreviation AS team_abbrev "
        f"FROM players p LEFT JOIN teams t ON p.team_id = t.team_id "
        f"WHERE p.player_id IN ({','.join('?' * len(ids))})",
        conn, params=ids,
    )
    merged = g.merge(names, on="pitcher_id", how="left")

    def _rows(d):
        return [
            dict(pitcher_id=int(r.pitcher_id), name=r.name, team_abbrev=r.team_abbrev,
                 total_pitches=int(r.total_pitches), n_pairs=int(r.n_pairs),
                 score=round(float(r.tunneling_plus), 1))
            for r in d.itertuples()
        ]
    top = merged.sort_values("tunneling_plus", ascending=False).head(top_n)
    bottom = merged.sort_values("tunneling_plus", ascending=True).head(top_n)
    return {"n_qualified": int(len(g)), "top": _rows(top), "bottom": _rows(bottom)}


def pitcher_composite_detail(conn, season, pitcher_id, all_pairs_df=None):
    """One pitcher's composite Tunneling+ breakdown for the model page's
    pitcher lookup: the headline (usage-weighted) score plus every
    qualifying pitch-type pair's own distances and composite_score -
    exactly the numbers the player page's pair table shows, scored against
    the whole league's population for that season.

    all_pairs_df, when given (a cached season_all_pairs() result), is reused
    instead of recomputing - callers serving many pitcher lookups for one
    season should pass it."""
    if all_pairs_df is None:
        all_pairs_df = season_all_pairs(conn, season)
    if all_pairs_df.empty:
        return None
    sub = all_pairs_df[all_pairs_df["pitcher_id"] == pitcher_id]
    if sub.empty:
        return None

    df = load_pitches(conn, season, pitcher_id=pitcher_id)
    centroids = pitch_type_centroids(df) if not df.empty else pd.DataFrame()
    total_pitches = (
        int(centroids[centroids["n_pitches"] >= MIN_PITCHES_PER_TYPE]["n_pitches"].sum())
        if not centroids.empty else None
    )

    pairs = pitcher_pairs_from_season(all_pairs_df, pitcher_id, total_pitches)
    if not pairs:
        return None

    # Same pair_weight = sqrt(usage_a * usage_b) convention
    # pitcher_composite_scores() uses league-wide, applied here per-pitcher.
    if total_pitches:
        weights = [np.sqrt((p["pct_a"] / 100) * (p["pct_b"] / 100)) for p in pairs]
    else:
        weights = [1.0] * len(pairs)
    weight_sum = sum(weights) or 1.0
    headline = sum(p["composite_score"] * w for p, w in zip(pairs, weights)) / weight_sum

    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "tunneling_plus": round(float(headline), 1),
        "n_pairs": len(pairs),
        "total_pitches": total_pitches,
        "pairs": pairs,
    }
