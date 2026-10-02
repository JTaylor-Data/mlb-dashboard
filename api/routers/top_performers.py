"""
/api/top-performers - wraps wpa.py + boxscore.py exactly as
pages/3_Top_Performers.py did. Cached per-date: a completed day's box
scores don't change, and even "today" changing mid-game is low-stakes for a
personal tool (the Refresh button / TTL expiry covers it).

Ranking is a per-day weighted box-score score, not WPA (WPA is still
computed and displayed, just no longer what determines the top 10 - it
reflects game-context leverage, not raw production, so a monster batting
line in a blowout could be totally invisible under it). Weights are ordinal
placeholders standing in for a priority order the user specified without
magnitudes (highest to lowest weight: HR > H > RBI > SB for batters, ER
[lower better] > IP > K > H for pitchers) - a judgment call, not a derived
number.
"""
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, Query


from analytics.boxscore import batter_daily_stats, fetch_boxscore, pitcher_daily_stats
from analytics.wpa import all_daily_participants, find_walkoff_batter

from api.cache import ttl_cache
from api.db import get_conn

router = APIRouter(prefix="/api")

BATTER_SCORE_WEIGHTS = dict(hr=4, h=3, rbi=2, sb=1)
PITCHER_SCORE_WEIGHTS = dict(er=4, ip=3, k=2, h=1)  # er applied negatively (lower is better)


def _parse_ip(ip_str):
    """MLB's innings-pitched notation isn't decimal - ".1"/".2" mean a
    third/two-thirds of an inning, not a tenth. "6.1" is 6 1/3 innings
    (6.333), not 6.1. Naive float() would silently understate every partial
    inning and skew the score."""
    try:
        whole, _, frac = str(ip_str).partition(".")
        whole = int(whole)
        thirds = {"": 0, "0": 0, "1": 1, "2": 2}.get(frac, 0)
        return whole + thirds / 3
    except (ValueError, TypeError):
        return 0.0


def _batter_score(stats):
    w = BATTER_SCORE_WEIGHTS
    return w["hr"] * stats.get("hr", 0) + w["h"] * stats.get("h", 0) + w["rbi"] * stats.get("rbi", 0) + w["sb"] * stats.get("sb", 0)


def _pitcher_score(stats):
    w = PITCHER_SCORE_WEIGHTS
    return (
        -w["er"] * stats.get("er", 0)
        + w["ip"] * _parse_ip(stats.get("ip", "0.0"))
        + w["k"] * stats.get("k", 0)
        - w["h"] * stats.get("h", 0)
    )


def _build_opponent_map(conn, entries):
    """Batch-fetch home/away and team abbreviations for all entries at once."""
    game_ids = list({e["game_id"] for e in entries if e.get("team_id") is not None})
    if not game_ids:
        return {}
    placeholders = ",".join("?" * len(game_ids))
    games = {
        row[0]: (int(row[1]), int(row[2]))
        for row in conn.execute(
            f"SELECT game_id, home_team, away_team FROM games WHERE game_id IN ({placeholders})", game_ids
        )
    }
    # Postseason games live in schedule_games (not the Statcast games table)
    missing = [gid for gid in game_ids if gid not in games]
    if missing:
        mh = ",".join("?" * len(missing))
        for row in conn.execute(
            f"SELECT game_pk, home_team_id, away_team_id FROM schedule_games WHERE game_pk IN ({mh})", missing
        ):
            games[row[0]] = (int(row[1]), int(row[2]))
    all_team_ids = {tid for home, away in games.values() for tid in (home, away)}
    if not all_team_ids:
        return {}
    tp = ",".join("?" * len(all_team_ids))
    abbrevs = {
        row[0]: row[1]
        for row in conn.execute(f"SELECT team_id, abbreviation FROM teams WHERE team_id IN ({tp})", list(all_team_ids))
    }
    result = {}
    for e in entries:
        if e.get("team_id") is None:
            result[(e["game_id"], e["team_id"])] = ""
            continue
        gm = games.get(e["game_id"])
        if gm is None:
            result[(e["game_id"], e["team_id"])] = ""
            continue
        home, away = gm
        is_home = e["team_id"] == home
        opp_id = away if is_home else home
        abbrev = abbrevs.get(opp_id, "")
        result[(e["game_id"], e["team_id"])] = f"{'vs' if is_home else '@'} {abbrev}" if abbrev else ""
    return result


@ttl_cache()
def _leaderboard(conn, day_str):
    participants = all_daily_participants(conn, day_str)
    if participants.empty:
        return {"entries": [], "walkoffs": []}

    boxscore_cache = {}

    def boxscore_for(game_id):
        if game_id not in boxscore_cache:
            boxscore_cache[game_id] = fetch_boxscore(game_id)
        return boxscore_cache[game_id]

    # Pre-fetch every distinct game that day IN PARALLEL before the scoring
    # loop below - was one sequential network round-trip per game (a real
    # day can have ~10-15), all blocking the request thread one after
    # another. fetch_boxscore() has its own persistent cache (see
    # boxscore.py), so this just warms it up-front instead of leaving the
    # main loop to trigger fetches one at a time as it goes.
    unique_game_ids = participants["game_id"].astype(int).unique().tolist()
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(boxscore_for, unique_game_ids))

    scored = []
    for _, row in participants.iterrows():
        game_id = int(row["game_id"])
        _tid = row["team_id"]
        team_id = int(_tid) if _tid is not None and _tid == _tid else None  # NaN != NaN
        bx = boxscore_for(game_id)

        if row["role"] == "batter":
            stats = batter_daily_stats(bx, int(row["player_id"])) or {}
            score = _batter_score(stats)
        else:
            stats = pitcher_daily_stats(bx, int(row["player_id"])) or {}
            score = _pitcher_score(stats)

        scored.append(dict(
            player_id=int(row["player_id"]), name=row["name"], role=row["role"],
            wpa=round(float(row["wpa"]), 3), team_id=team_id, stats=stats, score=score,
            game_id=game_id,
        ))

    batters = sorted([e for e in scored if e["role"] == "batter"], key=lambda e: e["score"], reverse=True)[:10]
    pitchers = sorted([e for e in scored if e["role"] == "pitcher"], key=lambda e: e["score"], reverse=True)[:10]
    top = batters + pitchers
    opponent_map = _build_opponent_map(conn, top)
    for e in top:
        e["walkoff"] = e["role"] == "batter" and find_walkoff_batter(conn, e["game_id"]) == e["player_id"]
        e["opponent"] = opponent_map.get((e["game_id"], e["team_id"]), "")

    # Walk-offs are reported independent of leaderboard rank: one per game
    # (at most), for every game that day, regardless of whether that
    # player's score also happens to land in the top 10 above (it's fine if
    # it does - not mutually exclusive).
    walkoff_entries = []
    for game_id in participants["game_id"].astype(int).unique():
        walkoff_batter_id = find_walkoff_batter(conn, int(game_id))
        if walkoff_batter_id is None:
            continue
        bx = boxscore_for(int(game_id))
        stats = batter_daily_stats(bx, walkoff_batter_id) or {}
        player_row = participants[
            (participants["player_id"] == walkoff_batter_id) & (participants["game_id"].astype(int) == int(game_id))
        ]
        name = player_row.iloc[0]["name"] if not player_row.empty else None
        _wtid = player_row.iloc[0]["team_id"] if not player_row.empty else None
        team_id = int(_wtid) if _wtid is not None and _wtid == _wtid else None
        if name is None:
            row = conn.execute("SELECT name, team_id FROM players WHERE player_id = ?", (walkoff_batter_id,)).fetchone()
            if row is not None:
                name, team_id = row[0], row[1]
        walkoff_entries.append(dict(
            player_id=int(walkoff_batter_id), name=name, team_id=team_id, stats=stats,
            game_id=game_id,
        ))

    walkoff_opp_map = _build_opponent_map(conn, walkoff_entries)
    walkoffs = []
    for e in walkoff_entries:
        e["opponent"] = walkoff_opp_map.get((e["game_id"], e["team_id"]), "")
        walkoffs.append({k: v for k, v in e.items() if k != "game_id"})

    return {"entries": top, "walkoffs": walkoffs}


@router.get("/top-performers")
def get_top_performers(date: str = Query(...), conn=Depends(get_conn)):
    result = _leaderboard(conn, date)
    return {"date": date, "entries": result["entries"], "walkoffs": result["walkoffs"]}
