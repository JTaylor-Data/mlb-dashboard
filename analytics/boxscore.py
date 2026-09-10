"""
Live per-game box score stats from the MLB Stats API - official,
scorer-verified stats (RBI, SB, ER) that aren't reliably derivable from
Statcast pitch data alone (Statcast has no earned/unearned run distinction
at all, and no stolen base events attached to pitch rows). Fetched on
demand for whichever players land in the top-performers list, not
bulk-ingested ahead of time - there's no need to store a box score for
every player on every day when only ~10 a day ever get looked at.
"""
import requests

BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
PERSON_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{player_id}/stats"

# A finished game's box score is immutable (MLB doesn't retroactively revise
# final stats), so this is a plain permanent dict, not a TTL cache - same
# module-level-dict pattern stuff_model.py/deception_model.py already use
# for their own internal caches, with the same clear_cache() hook wired
# into api/pull.py's refresh cycle for consistency (in practice this only
# ever matters for the current day's still-being-played games). Before this,
# every caller independently re-fetched the same game's box score over the
# network - real, measured cost: a single day's Top Performers leaderboard
# alone fetches ~10-15 distinct games, and a player's Last 5 Games/Outings
# can overlap with games other players (or that same leaderboard) already
# fetched this session.
_boxscore_cache = {}


def fetch_boxscore(game_id):
    if game_id in _boxscore_cache:
        return _boxscore_cache[game_id]
    resp = requests.get(BOXSCORE_URL.format(game_id=game_id), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    _boxscore_cache[game_id] = data
    return data


def clear_cache():
    _boxscore_cache.clear()


def season_era(player_id, season=2026):
    """Live season ERA from the MLB Stats API - same reasoning as the rest
    of this module: Statcast pitch data has no earned/unearned run
    distinction at all, so ERA can't be derived from our own DB, only
    fetched as an official number. A pitcher who was traded mid-season can
    have multiple splits (one per team); the first is that team's own ERA,
    not a combined one, but that's what the Stats API itself returns as the
    "current" split and is what a scoreboard would show."""
    resp = requests.get(
        PERSON_STATS_URL.format(player_id=player_id),
        params={"stats": "season", "season": season, "group": "pitching"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None
    return splits[0].get("stat", {}).get("era")


def season_primary_position(player_id, season):
    """Live position actually played in `season`, from the MLB Stats API's
    fielding-by-position season split - NOT the same as players.position,
    which is only ever the player's CURRENT real-world primary position
    (overwritten on every roster pull, see pulls/pull_players_games.py) and
    is wrong for any season before a position change. Real example: Bo
    Bichette's players.position is '3B' (his current Mets position, 2026),
    but he was really the Blue Jays' SS in 2025 - using the static field
    for a 2025 lineup collided him with the Jays' real 3B and bumped that
    player to the bench. Whichever position has the most games that season
    wins (a late-season position switch still counts as a real, if partial,
    split - this just picks the dominant one, same "most usage wins" logic
    _team_affiliation already uses for team)."""
    resp = requests.get(
        PERSON_STATS_URL.format(player_id=player_id),
        params={"stats": "season", "season": season, "group": "fielding"},
        timeout=30,
    )
    resp.raise_for_status()
    splits = resp.json().get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None
    best = max(splits, key=lambda s: s.get("stat", {}).get("games") or 0)
    return best.get("position", {}).get("abbreviation")


def _find_player(boxscore, player_id):
    key = f"ID{player_id}"
    for side in ("home", "away"):
        players = boxscore["teams"][side]["players"]
        if key in players:
            return players[key]
    return None


def batter_daily_stats(boxscore, player_id):
    p = _find_player(boxscore, player_id)
    if p is None:
        return None
    daily = p["stats"].get("batting", {})
    season = p.get("seasonStats", {}).get("batting", {})
    return dict(
        h=daily.get("hits", 0), ab=daily.get("atBats", 0), hr=daily.get("homeRuns", 0),
        sb=daily.get("stolenBases", 0), rbi=daily.get("rbi", 0),
        season_ops=season.get("ops", "-"),
    )


def pitcher_daily_stats(boxscore, player_id):
    p = _find_player(boxscore, player_id)
    if p is None:
        return None
    daily = p["stats"].get("pitching", {})
    season = p.get("seasonStats", {}).get("pitching", {})
    return dict(
        ip=daily.get("inningsPitched", "0.0"), k=daily.get("strikeOuts", 0),
        h=daily.get("hits", 0), hr=daily.get("homeRuns", 0),
        r=daily.get("runs", 0), er=daily.get("earnedRuns", 0),
        season_era=season.get("era", "-"),
    )
