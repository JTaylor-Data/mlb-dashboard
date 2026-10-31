"""
config/season_phase.py — Single source of truth for season phase.

SEASON_PHASE controls which behaviors are active:
  REGULAR_SEASON : team-stat pulls run; all caches cleared on refresh;
                   standings/race sections live; percentile pools updating
  POSTSEASON     : team-stat pulls skipped (RS records frozen); only
                   daily-data caches cleared; bracket/playoff sections live
  OFFSEASON      : most pulls skipped; standings/race frozen; model-page
                   season defaults to completed season; no percentile updating

To transition phases, change SEASON_PHASE below (or set env var SEASON_PHASE).
See docs/methodology/09_season_phase_guide.md for the full transition checklist.
"""
import os

# ---------------------------------------------------------------------------
# Phase definition
# ---------------------------------------------------------------------------
REGULAR_SEASON = "REGULAR_SEASON"
POSTSEASON     = "POSTSEASON"
OFFSEASON      = "OFFSEASON"

# Read from environment; fall back to the value set here.
# Set env var SEASON_PHASE=REGULAR_SEASON before 2027 Opening Day.
_env = os.environ.get("SEASON_PHASE", "").strip().upper()
SEASON_PHASE: str = _env if _env in (REGULAR_SEASON, POSTSEASON, OFFSEASON) else POSTSEASON

# ---------------------------------------------------------------------------
# Derived helpers — single place to query what changes with the phase
# ---------------------------------------------------------------------------

def is_postseason() -> bool:
    """True during the playoff window (postseason pitch data still arriving)."""
    return SEASON_PHASE == POSTSEASON


def is_regular_season() -> bool:
    return SEASON_PHASE == REGULAR_SEASON


def is_offseason() -> bool:
    return SEASON_PHASE == OFFSEASON


def skip_team_stat_pulls() -> bool:
    """RS win/loss records are frozen once the regular season ends."""
    return SEASON_PHASE in (POSTSEASON, OFFSEASON)


def should_pull_statcast() -> bool:
    """Postseason Statcast data still arrives; stop after WS ends (offseason)."""
    return SEASON_PHASE in (REGULAR_SEASON, POSTSEASON)


def cache_ttl_live_seconds() -> int:
    """TTL for surfaces that update with new games (standings, top-performers)."""
    if SEASON_PHASE == REGULAR_SEASON:
        return 12 * 3600   # 12 h — daily games
    elif SEASON_PHASE == POSTSEASON:
        return 12 * 3600   # 12 h — postseason games not every day
    else:  # OFFSEASON
        return 7 * 24 * 3600  # 1 week — nothing changes


def show_playoff_bracket() -> bool:
    """Whether to show the playoff bracket section."""
    return SEASON_PHASE in (POSTSEASON, OFFSEASON)


def show_il_plaques() -> bool:
    """IL badges and injury data are meaningful only during the regular season.

    POSTSEASON: IL transactions are uncommon/frozen after the roster locks
    (September expanded rosters). OFFSEASON: no active rosters.
    When False:
      - Backend must not fetch or return il_status data.
      - Frontend receives il_status=null and renders no badge.
    """
    return SEASON_PHASE == REGULAR_SEASON

# Exported constant for use in templates and frontend checks.
SHOW_IL_PLAQUES: bool = show_il_plaques()


def percentile_pool_live() -> bool:
    """Whether the current season's percentile pool should be rebuilt on refresh."""
    return SEASON_PHASE in (REGULAR_SEASON, POSTSEASON)


# ---------------------------------------------------------------------------
# Optional: suggest phase from the schedule (logs warning only, never mutates)
# ---------------------------------------------------------------------------
def suggest_phase_from_schedule(conn) -> str:
    """
    Query the DB to suggest what SEASON_PHASE should be right now.
    Returns a string suggestion. NEVER changes SEASON_PHASE — that is manual.
    Logs a warning if the suggestion differs from the current value.
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        from datetime import date
        today = date.today().isoformat()
        # Postseason: schedule has postseason game types (F, D, L, W)
        ps_count = conn.execute(
            "SELECT COUNT(*) FROM schedule_games WHERE game_type IN ('F','D','L','W') "
            "AND game_date <= ?", (today,)
        ).fetchone()[0]
        # Regular season: recent regular-season games
        rs_count = conn.execute(
            "SELECT COUNT(*) FROM schedule_games WHERE game_type='R' "
            "AND game_date > date(?, '-14 days') AND game_date <= ?",
            (today, today)
        ).fetchone()[0]

        if ps_count > 0:
            suggestion = POSTSEASON
        elif rs_count > 0:
            suggestion = REGULAR_SEASON
        else:
            suggestion = OFFSEASON

        if suggestion != SEASON_PHASE:
            _log.warning(
                "season_phase.py: current SEASON_PHASE=%s but schedule suggests %s. "
                "Update SEASON_PHASE manually when ready (see docs/methodology/09_season_phase_guide.md).",
                SEASON_PHASE, suggestion,
            )
        return suggestion
    except Exception as e:
        _log.warning("suggest_phase_from_schedule failed: %s", e)
        return SEASON_PHASE
