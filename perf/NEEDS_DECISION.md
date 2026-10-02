# Needs Decision

(No open items — Part 0 push blocker resolved by owner manually running git push.)

---

## ND-1: team_stats_1 returns 500 (pre-existing SQL bug)

`GET /api/teams/111/stats` returns 500 with OperationalError.

**Root cause**: `api/routers/teams.py` line ~328 has a dead first `pitching` query that references
`g.inning_topbot` — a column that does not exist in the `games` table (schema: game_id, game_date,
home_team, away_team, final_score, season). A second `pitching_stats` query below it uses the
`players` table instead and is structurally sound, but the dead first query runs first and raises.

**Impact**: Benchmark baseline shows `team_stats_1 500`. The dead query should be removed; only
the second query (`pitching_stats`) is needed.

**Decision needed**: Fix the dead query (remove it) or leave the 500 as-is?

---

## ND-2: Baseline benchmark captured with broken similar/stuff_leader/top_performers URLs

The baseline benchmark (bkziggsp9) started before F4 (benchmark URL) fixes were committed.
Runs 1-3 all show 422 for `similar_starter1`, `similar_batter1`, `stuff_leader`, `top_performers`.
These are benchmark-URL bugs, not real endpoint failures.

The "after" benchmark will use the fixed benchmark.py and will show 200 for those endpoints.
Comparison numbers for those 4 endpoints will show status change 422→200 which reflects the URL fix, not a performance change.
