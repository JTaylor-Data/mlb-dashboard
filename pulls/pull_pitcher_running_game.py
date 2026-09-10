"""
Pulls real pitcher holding-runners / running-game data into
player_advanced_stats.running_game_runs_prevented, for the percentile
rankings' Holding Runners slot (replaces FB% there - see
analytics/percentiles.py's PITCHER_POLARITY).

Source: Baseball Savant's own pitcher-running-game leaderboard
(baseballsavant.mlb.com/leaderboard/pitcher-running-game?csv=true) - a
direct CSV export, unauthenticated, not Cloudflare-gated (confirmed by a
plain requests.get during initial feasibility research), same pull pattern
as pull_sabermetrics.py's arm-strength leaderboard. NOT wrapped by
pybaseball - pybaseball's own statcast_running/_sprint_speed/_running_splits
functions are batter-only and don't cover this pitcher-side data at all.

Metric chosen: `runs_prevented_on_running_attr` - a single runs-value
composite (Savant's own "runs prevented via running-game attribution"
figure), rather than a raw stolen-base success rate or time-to-plate alone.
Chosen because it's the one column here that's already denominated in runs
(directly comparable to other run-value stats on this page, like WAR) and
folds together the several real components of "preventing the running
game" (caught-stealing rate, pickoffs, lead distance allowed) into one
number, instead of picking just one of those pieces and ignoring the
others. A reasonable alternative would have been raw CS% against; this
project went with the composite instead - flag this if it turns out to
read confusingly small/large next to the more familiar percentile-scale
metrics it now sits beside.

NOTE ON THE `year`/`season` QUERY PARAM: this leaderboard's CSV endpoint
was found (by direct testing) to ignore season-filter query params
entirely (year=, season=, start_year=/end_year= all returned identical,
current-season-only data) - it appears to only ever serve the live current
season, unlike Savant's other historical leaderboards. This pull script
therefore only ever fetches CURRENT-season data regardless of what SEASON
argument is passed, and stores it under whatever season is actually live
right now - re-running this script for a past season will NOT backfill
that season; it will just re-fetch today's current season again. This is a
real limitation of the source, not a bug in this script - flagged
explicitly since it differs from every other pull_*.py script in this
project, which do support arbitrary historical seasons.

Usage:
    python pull_pitcher_running_game.py [SEASON]   # SEASON is stored as
                                                     # given, but the value
                                                     # is always the live
                                                     # current-season data
                                                     # per the note above
"""
import csv
import io
import sqlite3
import sys

import requests

DB_PATH = "mlb_data.db"
DEFAULT_SEASON = 2026
RUNNING_GAME_URL = "https://baseballsavant.mlb.com/leaderboard/pitcher-running-game"


def _running_game_map():
    resp = requests.get(RUNNING_GAME_URL, params={"csv": "true"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    # utf-8-sig, not resp.text/plain utf-8: this endpoint's CSV starts with a
    # UTF-8 BOM, which (undecoded) glues itself onto the first header cell
    # ("﻿player_id" instead of "player_id") and silently breaks every
    # row.get("player_id") lookup below - confirmed by a 0-row pull before
    # this fix, despite the raw response genuinely containing 474 data rows.
    reader = csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig")))
    out = {}
    for row in reader:
        val = row.get("runs_prevented_on_running_attr")
        pid = row.get("player_id")
        if val and pid:
            out[int(pid)] = float(val)
    return out


def run(season: int):
    conn = sqlite3.connect(DB_PATH)

    print(f"Pulling real running-game runs-prevented (Baseball Savant, current season only) "
          f"-> storing under season={season}...", end=" ", flush=True)
    rg_map = _running_game_map()
    print(f"done, {len(rg_map)} pitchers")

    rows = [(pid, season, val) for pid, val in rg_map.items()]
    conn.executemany("""
        INSERT INTO player_advanced_stats (player_id, season, running_game_runs_prevented)
        VALUES (?, ?, ?)
        ON CONFLICT(player_id, season) DO UPDATE SET
            running_game_runs_prevented=excluded.running_game_runs_prevented
    """, rows)
    conn.commit()

    print(f"Upserted {len(rows)} pitchers.")
    conn.close()


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
