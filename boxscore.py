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


def fetch_boxscore(game_id):
    resp = requests.get(BOXSCORE_URL.format(game_id=game_id), timeout=30)
    resp.raise_for_status()
    return resp.json()


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
