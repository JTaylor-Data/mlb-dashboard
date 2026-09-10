"""
Pulls the current injured-list snapshot (who's on the IL right now, across
all 30 teams) into il_status - powers the team card's "Players on IL" count/
rank and the player page's IL badge.

There is no dedicated "injured list" roster endpoint on the public MLB Stats
API (confirmed directly: GET /api/v1/rosterTypes lists every valid
rosterType value - 40Man/fullSeason/fullRoster/nonRosterInvitees/active/
allTime/depthChart/gameday/coach - "injuredList" is NOT one of them; passing
it silently falls back to the active roster). The 40-man roster view
(rosterType=40Man) is the one roster call that DOES carry each player's real
status code, including the IL ones: D7/D10/D15/D60 ("Injured 7/10/15/60-Day").
That's step one, per team.

Step two: for each player found on the IL, real start date + a real injury
description come from that player's own /transactions history - MLB Stats
API transactions have a stable, parseable sentence for this move type: e.g.
"New York Yankees placed RF Aaron Judge on the 10-day injured list. Right
rib stress fracture." (confirmed directly against real 2026 data). The most
recent such transaction (not necessarily the very first one this season - a
player can cycle on/off the IL) is used. No "expected return date" exists
anywhere in this API - min_return_date is instead a real, COMPUTED lower
bound (start_date + that IL type's minimum stint length in days, a hard
roster rule), not a guess.

Usage:
    python pull_injuries.py [SEASON]   # defaults to 2026
"""
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

DB_PATH = Path(__file__).resolve().parent.parent / "mlb_data.db"
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_SEASON = 2026

# Minimum stint length per IL type - a real MLB roster rule, not a guess.
MIN_STINT_DAYS = {"D7": 7, "D10": 10, "D15": 15, "D60": 60}

IL_TRANSACTION_RE = re.compile(
    r"placed .* on the \d+-day injured list(?: retroactive to [A-Za-z]+ \d{1,2}, \d{4})?\.\s*(.*)$"
)


def _fetch_il_players(team_id, season):
    url = f"{STATSAPI_BASE}/teams/{team_id}/roster"
    resp = requests.get(url, params={"rosterType": "40Man", "season": season}, timeout=20)
    resp.raise_for_status()
    roster = resp.json().get("roster", [])
    return [
        {"player_id": r["person"]["id"], "team_id": team_id, "il_type": r["status"]["code"],
         "il_label": r["status"]["description"]}
        for r in roster
        if r["status"]["code"] in MIN_STINT_DAYS
    ]


def _fetch_il_detail(player_id, season):
    """Real start_date + injury description, parsed from this player's own
    transaction history - the latest 'placed ... on the N-day injured
    list.' move this season. Returns (start_date, injury), either possibly
    None if the transaction feed doesn't have a matching entry (e.g. IL
    placement happened in a season the API otherwise reports fine but the
    parse doesn't match some edge-case phrasing - fails soft rather than
    fabricating a date)."""
    url = f"{STATSAPI_BASE}/transactions"
    resp = requests.get(url, params={
        "playerId": player_id, "startDate": f"{season}-01-01", "endDate": date.today().isoformat(),
    }, timeout=20)
    resp.raise_for_status()
    txns = resp.json().get("transactions", [])
    il_moves = [t for t in txns if "injured list" in (t.get("description") or "") and "placed" in t["description"]]
    if not il_moves:
        return None, None
    latest = sorted(il_moves, key=lambda t: t.get("date") or "")[-1]
    m = IL_TRANSACTION_RE.search(latest["description"])
    injury = m.group(1).strip() or None if m else None
    return latest.get("effectiveDate") or latest.get("date"), injury


def fetch_one_team(team_id, season):
    rows = []
    for entry in _fetch_il_players(team_id, season):
        start_date, injury = _fetch_il_detail(entry["player_id"], season)
        min_return = None
        if start_date:
            try:
                min_return = (date.fromisoformat(start_date) + timedelta(days=MIN_STINT_DAYS[entry["il_type"]])).isoformat()
            except ValueError:
                pass
        rows.append({**entry, "start_date": start_date, "injury": injury, "min_return_date": min_return})
    return rows


def run(season: int):
    conn = sqlite3.connect(DB_PATH)
    team_ids = [r[0] for r in conn.execute("SELECT team_id FROM teams").fetchall()]
    if not team_ids:
        print("No teams in the `teams` table - run pull_teams.py first.")
        conn.close()
        return

    print(f"Pulling current IL status for {len(team_ids)} teams (season {season})...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_one_team, tid, season): tid for tid in team_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 10 == 0:
                print(f"  ...{i}/{len(team_ids)} teams")

    now = datetime.now().isoformat()
    conn.execute("DELETE FROM il_status")
    conn.executemany(
        "INSERT INTO il_status (player_id, team_id, il_type, il_label, start_date, injury, min_return_date, pulled_at) "
        "VALUES (:player_id, :team_id, :il_type, :il_label, :start_date, :injury, :min_return_date, :pulled_at)",
        [dict(r, pulled_at=now) for r in rows],
    )
    conn.commit()
    conn.close()
    print(f"\nReplaced il_status with {len(rows)} currently-injured players across {len(team_ids)} teams.")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
