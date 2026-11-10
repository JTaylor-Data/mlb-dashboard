# pulls/

Scripts that pull data into `mlb_data.db`. Run from the repo root.

## daily/ (run automatically by the app's daily pull, `api/pull.py`)

| Script | What it pulls |
|---|---|
| `pull_statcast.py` | Statcast pitches (regular season) |
| `pull_players_games.py` | `players` and `games` reference tables |
| `pull_team_stats.py` | Team-card snapshot (in-season only) |
| `pull_remaining_schedule.py` | Remaining regular-season schedule |
| `refresh_mlb_stats.py` | Rosters, schedule, news |
| `pull_prospects.py`, `pull_prospect_scouting.py` | Prospect rankings and scouting (Mondays) |

## annual/ (manual: run by hand during or after each season)

| Script | What it pulls | When |
|---|---|---|
| `pull_injuries.py` | Injured list snapshot (`il_status`) | Repeatedly in-season; nothing schedules it |
| `pull_sabermetrics.py` | wRC+, xFIP, Barrel%, arm strength, WAR | In-season and at season end |
| `pull_advanced_stats.py` | Sprint speed, outs above average | In-season and at season end |
| `pull_pitcher_running_game.py` | Holding-runners value | In-season and at season end |
| `pull_awards.py` | Awards (All-Star, MVP, Cy Young, ...) | After the awards are announced (November) |
| `backfill_career_stats.py` | ERA, WAR, primary position | After each season: `python pulls/annual/backfill_career_stats.py 2027` |
| `pull_teams.py` | Static team list | Only after a realignment |

Retired scripts (stadium photos, pitcher bio, FanGraphs fWAR) are kept in the local `Archive/pulls_archive/`, not in the repo.
