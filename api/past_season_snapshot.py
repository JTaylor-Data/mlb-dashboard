"""Reader for parquets/past_season_pitcher_page.json, the season-wide tables the
pitcher page and Trends page need for COMPLETE seasons (see
tools/precompute_past_season_pitcher_page.py). A finished season never changes,
so these are computed once offline instead of scanning the season on the first
view after every restart. get() returns None for the current season or when the
file or entry is missing, and callers then fall back to the live query."""
import json
import logging
from pathlib import Path

from config.model_policy import CURRENT_SEASON

log = logging.getLogger(__name__)

PATH = Path(__file__).resolve().parent.parent / "parquets" / "past_season_pitcher_page.json"
_snapshot = None


def pitch_rows(season):
    """The season's pitch-row count recorded when the snapshot was built (None for the
    current season or when unknown). Percentile pools key their files on this count, so
    reading it here spares a season-wide COUNT(*) scan on every pool load."""
    if season >= CURRENT_SEASON:
        return None
    get("_meta", season)  # loads the file
    n = (_snapshot or {}).get("_meta", {}).get("pitch_rows_by_season", {}).get(str(season))
    return int(n) if n is not None else None


def get(kind, season):
    global _snapshot
    if season >= CURRENT_SEASON:
        return None
    if _snapshot is None:
        try:
            _snapshot = json.loads(PATH.read_text())
        except Exception as e:
            log.warning("%s unavailable (%s: %s) - past seasons will be scanned live",
                        PATH.name, type(e).__name__, e)
            _snapshot = {}
    return _snapshot.get(kind, {}).get(str(season))
