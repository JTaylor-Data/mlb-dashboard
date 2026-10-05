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
# Game-type constants
# ---------------------------------------------------------------------------

# The pitches table is ingested with only regular-season pitches (R).
# pull_statcast.filter_game_types() enforces this at ingest, so the DB
# never contains postseason, spring training, or exhibition rows.
# These constants let analytics code document its intent explicitly and
# provide a shared SQL helper rather than inlining the literal 'R'.
REGULAR_SEASON_GAME_TYPES: tuple = ("R",)


def regular_season_filter(alias: str = "") -> str:
    """Return a SQL WHERE fragment that restricts to regular-season pitches.

    Parameters
    ----------
    alias : str
        Optional table alias to prefix the column, e.g. ``alias='p'`` →
        ``"AND p.game_type = 'R'"``.  Empty (default) → ``"AND game_type = 'R'"``.
    """
    col = f"{alias}.game_type" if alias else "game_type"
    return f"AND {col} = 'R'"


# ---------------------------------------------------------------------------
# Pitch-type exclusions (not tracked / not scored / not displayed)
# ---------------------------------------------------------------------------

# MODEL_EXCLUDED: pitch types the LightGBM models (Stuff+ v4, Location+ v3)
# do NOT score.  Matches the EXCLUDED_PITCH_TYPES set inside
# analytics/stuff_features.py::exclude_junk_pitch_types(), plus FA which is
# a generic fastball alias that the PITCH_TYPE_MAP normalises away before
# features are computed (so it never reaches the model but should be excluded
# from any raw-pitch iteration too).
#
# AB (auto-batted ball, 3 rows total) and AS (auto-strike, 0 rows) are not in
# this set because they do not appear in the live DB in any significant count.
# If they did appear, they would belong here.
#
# Verified counts in live DB (2026-10-05):
#   PO: 977   UN: 58   EP: 4,568   FA: 6,332   AB: 3   AS: 0
MODEL_EXCLUDED: frozenset = frozenset({
    "PO",   # pickoff — not a pitch to the batter
    "UN",   # unclassified — Statcast classifier failure, not a real pitch type
    "EP",   # eephus — excluded by design (position-player mop-up innings etc.)
})

# DISPLAY_EXCLUDED: superset of MODEL_EXCLUDED.
# These types appear in the DB (may even be model-scored) but should NEVER
# be shown to users in splits, tables, heatmaps, or leaderboards.
# Additions beyond MODEL_EXCLUDED:
#   FA — generic fastball alias; maps to FF in the pipeline. Displaying it
#        as a separate row would confuse users who expect FF/SI/etc.
DISPLAY_EXCLUDED: frozenset = frozenset(MODEL_EXCLUDED | {
    "FA",   # generic fastball alias — normalised to FF in PITCH_TYPE_MAP
})

# Legacy alias kept for backward compatibility with callers that import
# EXCLUDED_PITCH_TYPES by name (analytics/splits.py, etc.).  Points to the
# display-excluded set (superset of model-excluded), which is what UI
# callers need.
EXCLUDED_PITCH_TYPES: frozenset = DISPLAY_EXCLUDED

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
