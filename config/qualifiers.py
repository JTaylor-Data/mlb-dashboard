"""
config/qualifiers.py — Single source of truth for qualifying thresholds.

Consolidates min-pitch counts, min-PA thresholds, pitch-type exclusions,
2020 short-season scaling, and usage-filter constants.

Callers should import from here rather than define their own thresholds.
"""

# ---------------------------------------------------------------------------
# Minimum pitch counts by model / leaderboard
# ---------------------------------------------------------------------------

# Stuff+ v4 — leaderboard only (no scoring minimum; shrinkage handles small N)
STUFF_MIN_PITCHES_LEADERBOARD: int = 200

# Location+ v3 — both scoring and leaderboard
LOCATION_MIN_PITCHES_SCORING: int = 300
LOCATION_MIN_PITCHES_LEADERBOARD: int = 300

# 2020 short-season scaling: 60 games / 162 games ≈ 0.37
LOCATION_MIN_PITCHES_2020: int = 111  # 300 * 0.37

# Tunneling+ v2 — raw pitches before lag-pair exclusion
TUNNELING_MIN_PITCHES_SCORING: int = 100
TUNNELING_MIN_PITCHES_LEADERBOARD: int = 100

# Deception+ v2b — no explicit minimum; returns empty if features are empty after dropna
DECEPTION_MIN_PITCHES_SCORING = None

# ---------------------------------------------------------------------------
# 2020 short-season scaling
# ---------------------------------------------------------------------------
SHORT_SEASON_YEAR: int = 2020
SHORT_SEASON_SCALE: float = 60 / 162  # ≈ 0.370

def scale_min_pitches_for_year(min_pitches: int, year: int) -> int:
    """Return the minimum pitch count adjusted for 2020's shortened season."""
    if year == SHORT_SEASON_YEAR and min_pitches is not None:
        return max(1, round(min_pitches * SHORT_SEASON_SCALE))
    return min_pitches


# ---------------------------------------------------------------------------
# Minimum plate appearances (batters)
# ---------------------------------------------------------------------------
MIN_PA_LEADERBOARD: int = 100  # batters; used in hitzones, similar players

# ---------------------------------------------------------------------------
# Pitch-type exclusions (not tracked / not scored)
# ---------------------------------------------------------------------------
EXCLUDED_PITCH_TYPES: frozenset = frozenset({
    "PO",   # pickoff
    "UN",   # unknown
    "EP",   # eephus (rare)
    "FA",   # generic fastball alias (maps to FF in pipeline)
})

# Pitch-type mapping applied before scoring
PITCH_TYPE_MAP: dict = {
    "KC": "CU",   # knuckle-curve → curveball
}

# Minimum per-pitch-type usage fraction (1%) to include in pitch-mix display
MIN_PITCH_TYPE_USAGE_FRACTION: float = 0.01

# ---------------------------------------------------------------------------
# Qualifier functions
# ---------------------------------------------------------------------------

def is_qualified_pitcher_season(n_pitches: int, season: int, min_pitches: int = LOCATION_MIN_PITCHES_SCORING) -> bool:
    """True if a pitcher-season meets the minimum pitch threshold (with 2020 scaling)."""
    threshold = scale_min_pitches_for_year(min_pitches, season)
    return n_pitches >= threshold


def is_qualified_batter_season(n_pa: int) -> bool:
    """True if a batter-season meets the minimum PA threshold."""
    return n_pa >= MIN_PA_LEADERBOARD


def usage_filter(pitch_counts: dict, total_pitches: int) -> dict:
    """
    Return a filtered dict keeping only pitch types above MIN_PITCH_TYPE_USAGE_FRACTION.
    pitch_counts: {pitch_type: count}
    total_pitches: total pitches thrown (denominator)
    """
    if total_pitches <= 0:
        return {}
    return {
        pt: ct for pt, ct in pitch_counts.items()
        if ct / total_pitches >= MIN_PITCH_TYPE_USAGE_FRACTION
    }
