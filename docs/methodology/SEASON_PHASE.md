# Season Phase — Transition Guide

**Current phase**: POSTSEASON (as of 2026-10-05)

## Phase Definitions

| Phase | When | Description |
|-------|------|-------------|
| `REGULAR_SEASON` | Opening Day → last regular-season game | Active season; daily Statcast, team stats, standings |
| `POSTSEASON` | Wild-Card Game → World Series ends | Postseason pitch data ingested; RS records frozen |
| `OFFSEASON` | After World Series → Opening Day | No new pitch data; all surfaces frozen |

## What Changes With Each Phase

| Behavior | REGULAR_SEASON | POSTSEASON | OFFSEASON |
|---------|---------------|------------|-----------|
| `skip_team_stat_pulls()` | False | **True** | **True** |
| `should_pull_statcast()` | True | True | **False** |
| `show_playoff_bracket()` | False | **True** | True (frozen) |
| `show_il_plaques()` / `SHOW_IL_PLAQUES` | **True** | False | False |
| `percentile_pool_live()` | True | True | **False** |
| `cache_ttl_live_seconds()` | 12h | 12h | 1 week |
| Standings section | Live | Frozen (RS final) | Frozen |
| IL badge (player page, last 5) | Shown | **Hidden** | **Hidden** |
| Backend `il_status` fetch | Runs | **Skipped** | **Skipped** |
| Deception REPORT_SEASON | CURRENT_SEASON | CURRENT_SEASON | CURRENT_SEASON-1 |
| Team stat pulls | Run | Skipped | Skipped |

## How to Change Phase

1. Set the env var before starting the server:
   ```
   set SEASON_PHASE=REGULAR_SEASON   (Windows)
   export SEASON_PHASE=REGULAR_SEASON  (Linux/Mac)
   ```
   — OR —
   Edit `config/season_phase.py` line `POSTSEASON` at the end of the `_env = ...` ternary.

2. Restart the server (`python -m uvicorn api.main:app --port 8010`).

3. Run the smoke tests: `pytest tests/test_routes_smoke.py -q`

4. For REGULAR_SEASON transition specifically:
   - Run `pulls/pull_team_stats.py` once manually to get fresh W/L records
   - Re-run `tools/build_percentile_pools.py` if the new season's pool is stale

5. For OFFSEASON transition:
   - Set `CURRENT_SEASON` in `config/model_policy.py` to the next year's value after Opening Day
   - Run the full rescore pipeline (`scripts/rescore_models.py`) for the completed season

## Phase Verification (flip test)

After changing phase, verify in the API:
- `/api/admin/pull-status` — team-stat pull step should appear/disappear based on phase
- `/api/playoffs/bracket` — should return data in POSTSEASON/OFFSEASON, empty bracket in REGULAR_SEASON
- `/api/player/{id}` with any player_id — `il_status` field should be `null` in POSTSEASON/OFFSEASON, populated in REGULAR_SEASON (for IL players)
- Player page — IL badge should appear only in REGULAR_SEASON; no badge, no reserved space in POSTSEASON/OFFSEASON

**Current POSTSEASON behavior (verified 2026-10-05):** `il_status` returns `null` for all players
(SHOW_IL_PLAQUES=False). No IL badges render on player pages or last-5 cards.

## Current State (2026-10-05 POSTSEASON)

- `SEASON_PHASE = POSTSEASON` (default in config/season_phase.py)
- `CURRENT_SEASON = 2026` (in config/model_policy.py)
- Team stat pulls: skipped ✓
- Postseason bracket: live ✓
- Deception REPORT_SEASON: 2026 (was 2025, fixed in audit DEF-01)
- IL plaques: hidden (SHOW_IL_PLAQUES=False) ✓
