"""
Populate players and games reference tables from the MLB Stats API, keyed off
the distinct player_ids and game_ids already present in the pitches table.

Usage:
    python pull_players_games.py

Both tables are upserted (INSERT ... ON CONFLICT DO UPDATE), so re-running is
safe and just refreshes team/active-status/final-score info.

`run(since_date=...)` is gap-aware, not just recency-aware (see
get_distinct_player_ids/get_distinct_games): it always checks recent
activity AND any player/game that's ever appeared in `pitches` but is
missing from `players`/`games` entirely, regardless of age. That second
check is a cheap local JOIN, not a network call, so "fast" and "actually
thorough" aren't in tension - the automated refresh pipeline (api/pull.py)
can call this the same way on every run, whether it's the nightly schedule
or someone clicking the button, without needing a separate "full rescan"
mode.
"""
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import requests

DB_PATH = "mlb_data.db"
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
PEOPLE_BATCH_SIZE = 100


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def get_distinct_player_ids(conn, since_date=None):
    """Gap-aware, not just recency-aware: returns players active since
    `since_date` (recent trades/promotions/status changes - the common
    case) UNION ANY player who appears in `pitches` at all but has no row
    in `players` yet, no matter how long ago they first appeared (a real
    gap - e.g. a past pull that silently missed a batch). The second half
    is a cheap local JOIN, not a network call, so checking for it costs
    nothing extra - it just means "only recent" and "genuinely thorough"
    are no longer in tension the way a pure date cutoff would be. Omit
    `since_date` for a full historical pass (every player, every re-check)."""
    if since_date:
        recent = (
            "SELECT DISTINCT pitcher_id FROM pitches WHERE game_date >= ? "
            "UNION SELECT DISTINCT batter_id FROM pitches WHERE game_date >= ?"
        )
        params = (since_date, since_date)
    else:
        recent = "SELECT DISTINCT pitcher_id FROM pitches UNION SELECT DISTINCT batter_id FROM pitches"
        params = ()
    missing = (
        "SELECT p.pid FROM ("
        "  SELECT pitcher_id AS pid FROM pitches UNION SELECT batter_id AS pid FROM pitches"
        ") p LEFT JOIN players pl ON pl.player_id = p.pid WHERE pl.player_id IS NULL"
    )
    rows = conn.execute(f"{recent} UNION {missing}", params).fetchall()
    return sorted({r[0] for r in rows if r[0] is not None})


def get_distinct_games(conn, since_date=None):
    """Same gap-aware pattern as get_distinct_player_ids: recent activity
    UNION any game_id in `pitches` with no matching row in `games` yet at
    all, regardless of age."""
    if since_date:
        recent = "SELECT DISTINCT game_id, game_date FROM pitches WHERE game_date >= ?"
        params = (since_date,)
    else:
        recent = "SELECT DISTINCT game_id, game_date FROM pitches"
        params = ()
    missing = (
        "SELECT DISTINCT pi.game_id, pi.game_date FROM pitches pi "
        "LEFT JOIN games g ON g.game_id = pi.game_id WHERE g.game_id IS NULL"
    )
    rows = conn.execute(f"{recent} UNION {missing}", params).fetchall()
    return sorted(set(rows), key=lambda r: r[1])


def _fetch_people_batch(batch):
    """One /people lookup for a single <=100-id batch - split out so
    fetch_people can run every batch concurrently instead of one HTTP
    round-trip after another."""
    ids_param = ",".join(str(i) for i in batch)
    url = f"{STATSAPI_BASE}/people?personIds={ids_param}&hydrate=currentTeam"
    people = []
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ERROR: people lookup failed for {len(batch)} ids "
              f"({batch[0]}..{batch[-1]}): {e}")
        return people, (batch, str(e))

    for person in data.get("people", []):
        current_team = person.get("currentTeam") or {}
        # currentTeam reflects live roster assignment, which can be a minor-league
        # affiliate for an optioned/rehabbing player. When that happens the API
        # embeds parentOrgId on the affiliate - use that to resolve the MLB parent
        # club instead of storing the minor-league team.
        team_id = current_team.get("parentOrgId") or current_team.get("id")
        people.append({
            "player_id": person.get("id"),
            "name": person.get("fullName"),
            "team_id": team_id,
            "position": (person.get("primaryPosition") or {}).get("abbreviation"),
            "bats": (person.get("batSide") or {}).get("code"),
            "throws": (person.get("pitchHand") or {}).get("code"),
            "active_flag": 1 if person.get("active") else 0,
        })
    return people, None


def fetch_people(player_ids):
    """Batches of PEOPLE_BATCH_SIZE fetched IN PARALLEL - each is an
    independent HTTP round-trip to the same read-only endpoint, so there's
    no ordering/shared-state reason to run them one after another. A
    refresh touching ~650 recent players (7 sequential batches at
    ~1-1.5s each, ~10s total) was the single biggest chunk of "Refresh
    data takes forever" - measured directly, not assumed. Same
    ThreadPoolExecutor pattern already used for boxscore fetches in
    api/routers/top_performers.py."""
    batches = list(chunked(player_ids, PEOPLE_BATCH_SIZE))
    people = []
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for batch_people, failure in pool.map(_fetch_people_batch, batches):
            people.extend(batch_people)
            if failure:
                failures.append(failure)
    return people, failures


def fetch_schedule_games(start_date, end_date):
    """Returns dict game_pk -> row dict. final_score is 'home-away' when final, else None.

    gameType=R,S,F,D,L,W (regular season, Spring Training, Wild Card,
    Division Series, League Championship, World Series) - not R alone.
    Real bug, confirmed directly: pull_statcast.py pulls every real pitch
    with no game-type filter at all (Spring Training buffer days AND every
    postseason game both land in `pitches`), but this schedule lookup used
    to be R-only, so Spring Training and postseason game_ids could NEVER
    match and stayed permanently "missing" from `games`. get_distinct_games's
    gap-aware check has no age cutoff on "missing", so those same
    unresolvable games got pulled into the date range on EVERY refresh,
    forcing a full multi-year schedule re-fetch (one HTTP call per year,
    2021-2026) instead of just checking the last few days - the actual
    reason refreshes were slow even when there was nothing new to find.
    Confirmed live: a 2021-2025 backfill with R-only left 1319 unresolved
    games (all Spring Training); adding S dropped that to 208 (all real
    postseason games, every October 2021-2025); adding F/D/L/W here closes
    the gap completely."""
    url = (f"{STATSAPI_BASE}/schedule?sportId=1&startDate={start_date}"
           f"&endDate={end_date}&gameType=R,S,F,D,L,W")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    games = {}
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            game_pk = g.get("gamePk")
            teams = g.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            is_final = g.get("status", {}).get("abstractGameState") == "Final"
            final_score = None
            if is_final and "score" in home and "score" in away:
                final_score = f"{home['score']}-{away['score']}"  # home-away
            games[game_pk] = {
                "game_date": date_entry.get("date"),
                "home_team": (home.get("team") or {}).get("id"),
                "away_team": (away.get("team") or {}).get("id"),
                "final_score": final_score,
            }
    return games


def run(since_date=None):
    """`since_date` (an ISO date string) restricts both the players and
    games lookups to recent activity only - see `get_distinct_player_ids`'s
    docstring for why. Omit for a full historical pass (the default, used
    by manual `python pull_players_games.py` runs and one-off backfills).
    Returns a small summary dict the caller can surface to a user."""
    conn = sqlite3.connect(DB_PATH)

    # --- players ---
    player_ids = get_distinct_player_ids(conn, since_date)
    print(f"Looking up {len(player_ids)} distinct players...")
    people, failures = fetch_people(player_ids)

    conn.executemany(
        "INSERT INTO players (player_id, name, team_id, position, bats, throws, active_flag) "
        "VALUES (:player_id, :name, :team_id, :position, :bats, :throws, :active_flag) "
        "ON CONFLICT(player_id) DO UPDATE SET "
        "name=excluded.name, team_id=excluded.team_id, position=excluded.position, "
        "bats=excluded.bats, throws=excluded.throws, active_flag=excluded.active_flag",
        people,
    )
    conn.commit()

    found_ids = {p["player_id"] for p in people}
    missing_players = sorted(set(player_ids) - found_ids)
    print(f"Inserted/updated {len(people)} players.")
    if failures:
        print(f"  WARNING: {len(failures)} batch(es) failed outright (see errors above)")
    if missing_players:
        preview = missing_players[:10]
        print(f"  WARNING: {len(missing_players)} player_id(s) returned no match from the "
              f"Stats API: {preview}{' ...' if len(missing_players) > 10 else ''}")

    # --- games ---
    pitch_games = get_distinct_games(conn, since_date)
    if not pitch_games:
        print("No games in range - nothing to look up.")
        conn.close()
        return dict(players_updated=len(people), games_updated=0)
    distinct_game_ids = {g[0] for g in pitch_games}
    start_date = min(g[1] for g in pitch_games)
    end_date = max(g[1] for g in pitch_games)
    print(f"Looking up schedule/score data from {start_date} to {end_date}...")

    # The Stats API's /schedule endpoint silently truncates/misbehaves on a
    # multi-year date range in one call (confirmed: a single 2021-2026 call
    # matched only ~2,400 of ~14,000 games) - fetch one calendar year at a
    # time and merge, same chunking idea as pull_statcast.py's weekly chunks.
    schedule = {}
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        year_start = max(start_date, f"{year}-01-01")
        year_end = min(end_date, f"{year}-12-31")
        try:
            schedule.update(fetch_schedule_games(year_start, year_end))
        except Exception as e:
            print(f"  ERROR: schedule lookup failed for {year_start} to {year_end}: {e}")

    game_rows = []
    missing_games = []
    for game_id in sorted(distinct_game_ids):
        info = schedule.get(game_id)
        if info is None:
            missing_games.append(game_id)
            continue
        game_rows.append((
            game_id, info["game_date"], int(info["game_date"][:4]), info["home_team"],
            info["away_team"], info["final_score"],
        ))

    conn.executemany(
        "INSERT INTO games (game_id, game_date, season, home_team, away_team, final_score) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(game_id) DO UPDATE SET "
        "game_date=excluded.game_date, season=excluded.season, home_team=excluded.home_team, "
        "away_team=excluded.away_team, final_score=excluded.final_score",
        game_rows,
    )
    conn.commit()

    print(f"Inserted/updated {len(game_rows)} games.")
    if missing_games:
        preview = missing_games[:10]
        print(f"  WARNING: {len(missing_games)} game_id(s) from pitches had no schedule "
              f"match: {preview}{' ...' if len(missing_games) > 10 else ''}")

    conn.close()
    return dict(players_updated=len(people), games_updated=len(game_rows))


if __name__ == "__main__":
    run()
