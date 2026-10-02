"""
Shared MLB Stats API integration module.

Uses direct requests to statsapi.mlb.com — the MLB-StatsAPI pip package is
NOT installed in this environment, so all calls go through requests directly.
This was confirmed by attempting `import statsapi` in the project Python
(3.14 / system Python) before writing this file.

Caches pulled data in mlb_data.db in two new tables:
  - rosters       (team_id, player_id, player_name, position, roster_type,
                   jersey_number, pulled_at)
  - schedule_games (game_pk, game_date, home_team_id, away_team_id,
                    home_probable_pitcher_id, away_probable_pitcher_id,
                    venue_id, pulled_at)

All functions write to the DB and also return the fetched data for immediate
use by callers that don't want to hit the DB just to get what was just pulled.
"""
import sqlite3
from datetime import datetime, date
from typing import Optional
import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 30  # seconds per request
DB_PATH = "mlb_data.db"

VALID_ROSTER_TYPES = {"fullRoster", "40Man", "active", "depthChart",
                      "postseasonRoster"}
# Postseason round-specific roster_type keys used in the local DB.
# Map these to the MLB API roster type to request when pulling.
POSTSEASON_ROUND_MAP = {
    "postseason_WC":  "postseasonRoster",
    "postseason_DS":  "postseasonRoster",
    "postseason_LCS": "postseasonRoster",
    "postseason_WS":  "postseasonRoster",
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create the two caching tables if they don't already exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rosters (
            team_id       INTEGER,
            player_id     INTEGER,
            player_name   TEXT,
            position      TEXT,
            roster_type   TEXT,
            jersey_number TEXT,
            status_code   TEXT,
            pulled_at     TEXT,
            PRIMARY KEY (team_id, player_id, roster_type)
        );

        CREATE TABLE IF NOT EXISTS schedule_games (
            game_pk                      INTEGER PRIMARY KEY,
            game_date                    TEXT,
            game_time_et                 TEXT,
            home_team_id                 INTEGER,
            away_team_id                 INTEGER,
            home_team_name               TEXT,
            away_team_name               TEXT,
            home_probable_pitcher_id     INTEGER,
            away_probable_pitcher_id     INTEGER,
            venue_id                     INTEGER,
            venue_name                   TEXT,
            game_status                  TEXT,
            pulled_at                    TEXT
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# team_roster
# ---------------------------------------------------------------------------

def team_roster(team_id: int, roster_type: str = "active",
                season: int = 2026, db_path: str = DB_PATH) -> list[dict]:
    """
    Fetch the roster for one team and upsert into the `rosters` table.

    roster_type must be one of: fullRoster, 40Man, active, depthChart.
    Returns a list of player dicts.
    """
    # Postseason round keys (postseason_WC etc.) are stored in the local DB
    # under those keys but fetched from the MLB API as "postseasonRoster".
    api_roster_type = POSTSEASON_ROUND_MAP.get(roster_type, roster_type)
    if api_roster_type not in VALID_ROSTER_TYPES:
        raise ValueError(f"roster_type must be one of {VALID_ROSTER_TYPES | set(POSTSEASON_ROUND_MAP)}, got {roster_type!r}")

    url = f"{BASE_URL}/teams/{team_id}/roster"
    params = {"rosterType": api_roster_type, "season": season}
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    roster = data.get("roster", [])
    pulled_at = datetime.utcnow().isoformat()

    rows = []
    for entry in roster:
        person = entry.get("person", {})
        pos = entry.get("position", {})
        status = entry.get("status", {})
        rows.append({
            "team_id": team_id,
            "player_id": person.get("id"),
            "player_name": person.get("fullName"),
            "position": pos.get("abbreviation"),
            "roster_type": roster_type,
            "jersey_number": entry.get("jerseyNumber"),
            "status_code": status.get("code"),
            "pulled_at": pulled_at,
        })

    conn = _get_conn(db_path)
    ensure_tables(conn)
    conn.executemany(
        """INSERT INTO rosters
           (team_id, player_id, player_name, position, roster_type, jersey_number, status_code, pulled_at)
           VALUES (:team_id, :player_id, :player_name, :position, :roster_type, :jersey_number, :status_code, :pulled_at)
           ON CONFLICT(team_id, player_id, roster_type) DO UPDATE SET
             player_name=excluded.player_name, position=excluded.position,
             jersey_number=excluded.jersey_number, status_code=excluded.status_code,
             pulled_at=excluded.pulled_at""",
        rows,
    )
    conn.commit()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------

def schedule(start_date: str, end_date: str, sport_id: int = 1,
             db_path: str = DB_PATH) -> list[dict]:
    """
    Fetch the schedule between two dates (YYYY-MM-DD) and upsert into
    `schedule_games`. Returns a list of game dicts.

    Includes probable pitchers via the `probablePitcher` hydration.
    """
    url = f"{BASE_URL}/schedule"
    params = {
        "sportId": sport_id,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "probablePitcher,venue",
    }
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    pulled_at = datetime.utcnow().isoformat()
    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            venue = g.get("venue", {})
            # gameDate is full ISO UTC datetime e.g. "2026-09-01T17:05:00Z"
            game_datetime = g.get("gameDate", "")
            game_time_et = ""
            if game_datetime:
                try:
                    from datetime import timezone, timedelta as _td
                    dt_utc = datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
                    # MLB season is roughly EDT (UTC-4) Mar-Nov, EST (UTC-5) otherwise
                    mo = dt_utc.month
                    offset_h = -4 if 3 <= mo <= 11 else -5
                    dt_et = dt_utc.astimezone(timezone(_td(hours=offset_h)))
                    h = dt_et.hour % 12 or 12
                    ampm = "AM" if dt_et.hour < 12 else "PM"
                    game_time_et = f"{h}:{dt_et.strftime('%M')} {ampm} ET"
                except Exception:
                    pass
            rows.append({
                "game_pk": g["gamePk"],
                "game_date": g.get("officialDate", day["date"]),
                "game_time_et": game_time_et,
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_team_name": home["team"].get("name"),
                "away_team_name": away["team"].get("name"),
                "home_probable_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                "away_probable_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
                "game_status": g.get("status", {}).get("detailedState"),
                "pulled_at": pulled_at,
            })

    conn = _get_conn(db_path)
    ensure_tables(conn)
    # Migration: add columns that may not exist in older DBs
    for col, typedef in [("venue_id", "INTEGER"), ("game_time_et", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE schedule_games ADD COLUMN {col} {typedef}")
            conn.commit()
        except Exception:
            pass  # Column already exists

    # Migration: add name columns that may not exist in older DBs
    for col in [("home_team_name", "TEXT"), ("away_team_name", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE schedule_games ADD COLUMN {col[0]} {col[1]}")
            conn.commit()
        except Exception:
            pass  # Column already exists

    conn.executemany(
        """INSERT INTO schedule_games
           (game_pk, game_date, game_time_et, home_team_id, away_team_id,
            home_team_name, away_team_name,
            home_probable_pitcher_id, away_probable_pitcher_id,
            venue_id, venue_name, game_status, pulled_at)
           VALUES (:game_pk, :game_date, :game_time_et, :home_team_id, :away_team_id,
                   :home_team_name, :away_team_name,
                   :home_probable_pitcher_id, :away_probable_pitcher_id,
                   :venue_id, :venue_name, :game_status, :pulled_at)
           ON CONFLICT(game_pk) DO UPDATE SET
             home_team_name=excluded.home_team_name,
             away_team_name=excluded.away_team_name,
             home_probable_pitcher_id=excluded.home_probable_pitcher_id,
             away_probable_pitcher_id=excluded.away_probable_pitcher_id,
             venue_id=excluded.venue_id,
             venue_name=excluded.venue_name,
             game_time_et=excluded.game_time_et,
             game_status=excluded.game_status, pulled_at=excluded.pulled_at""",
        rows,
    )
    conn.commit()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# team_info
# ---------------------------------------------------------------------------

def team_info(team_id: int, season: int = 2026) -> dict:
    """
    Fetch team metadata from the MLB Stats API.
    Returns the raw team dict (not persisted — use pull_teams.py for that;
    this is for on-demand enrichment).
    """
    url = f"{BASE_URL}/teams/{team_id}"
    resp = requests.get(url, params={"season": season}, timeout=TIMEOUT)
    resp.raise_for_status()
    teams = resp.json().get("teams", [])
    if not teams:
        raise ValueError(f"No team found for team_id={team_id}")
    t = teams[0]
    return {
        "team_id": t["id"],
        "name": t.get("name"),
        "abbreviation": t.get("abbreviation"),
        "league": t.get("league", {}).get("name"),
        "division": t.get("division", {}).get("name"),
        "venue_id": t.get("venue", {}).get("id"),
        "venue_name": t.get("venue", {}).get("name"),
        "city": t.get("locationName"),
        "first_year": t.get("firstYearOfPlay"),
    }


# ---------------------------------------------------------------------------
# Convenience: bulk-pull all teams' rosters for a given roster_type
# ---------------------------------------------------------------------------

def pull_all_rosters(roster_type: str = "active", season: int = 2026,
                     db_path: str = DB_PATH) -> int:
    """Pull the given roster_type for every team in the `teams` table.
    Returns total player rows upserted."""
    conn = _get_conn(db_path)
    team_ids = [r[0] for r in conn.execute("SELECT team_id FROM teams").fetchall()]
    conn.close()

    total = 0
    for tid in team_ids:
        try:
            rows = team_roster(tid, roster_type, season, db_path)
            total += len(rows)
        except Exception as e:
            print(f"  Warning: roster pull failed for team {tid}: {e}")
    return total


# ---------------------------------------------------------------------------
# Convenience: pull today's (or a given date's) schedule
# ---------------------------------------------------------------------------

def pull_today(today: Optional[str] = None, db_path: str = DB_PATH) -> list[dict]:
    """Pull schedule for a single date (defaults to today UTC)."""
    if today is None:
        today = date.today().isoformat()
    return schedule(today, today, db_path=db_path)
