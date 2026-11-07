# Scheduled and background processes

What runs automatically, what each one pulls or precomputes, where the result is
stored, and what will break (or need a manual step) when the season rolls over to
2027. Written 2026-11-07 from the code as deployed on Fly (app `thewholenine`,
one 1 GB shared-CPU machine, volume `/data`).

## 1. The big picture

```
 MLB sources (pybaseball/Statcast, MLB Stats API, MLB.com RSS)
        |
        v   daily 04:00 Toronto, and at boot if the last pull is >2h old
   api/pull.py  ->  SQLite DB on the /data volume (mlb_data.db, ~4.4 GB)
        |                      + pull_state.json (when the last pull finished)
        v   only if new pitches arrived
   precompute tunneling pairs (subprocess) -> parquets/ on the CONTAINER disk
        |
        v
   warm-up (current season only) -> in-memory caches (lost on restart)
        |
        v
   FastAPI routers -> frontend (static JS pages)
```

Two kinds of storage matter:

| Storage | Survives restart/deploy? | Holds |
|---|---|---|
| `/data` volume | yes | `mlb_data.db` (+ WAL), `pull_state.json` |
| Container disk (`/app/...`, incl. `parquets/`) | **no, rebuilt from the deployed image** | score parquets, tunneling parquets, percentile pools, the two snapshot JSONs |
| Process memory (TTL caches) | no | pools, model tables, leaderboards |

Everything under `parquets/` that the app writes at runtime (weekly rescore, daily
tunneling-pairs precompute, newly built percentile pools) lands on the container
disk, so a restart or redeploy goes back to whatever was in the image. See
"Risks" below.

## 2. Schedule at a glance

| When | What | Where it's defined |
|---|---|---|
| Every server start | Full data pull + warm-up, **unless** a pull finished <2h ago (then warm-up only) | `api/scheduler.py` `start()` |
| Daily 04:00 America/Toronto | Full data pull + warm-up | `api/scheduler.py` `_daily_loop` |
| Mondays, inside the scheduled pull | Prospect rankings + scouting refresh (uses the server's date, i.e. UTC on Fly) | `api/pull.py` `_run` |
| Sundays 05:00 Toronto | Model rescore for the current season; **skipped unless `SEASON_PHASE=REGULAR_SEASON`** | `api/scheduler.py` `_weekly_rescore_loop` |
| On demand | "Refresh data" button (`POST /api/admin/pull-data`, light warm-up) and `POST /api/admin/refresh-cache` (clear everything + warm); both need the admin token | `api/main.py` |

Not scheduled at all: the Stats-API/FanGraphs side pulls (section 5).

## 3. The daily pull, step by step (`api/pull.py` `_run`)

| # | Step | Source | Writes to | Notes |
|---|---|---|---|---|
| 1 | Statcast pitches | pybaseball | `pitches`, `ingested_dates` | Checks 2021-03-20 -> today against a skip-list; always re-checks the last 3 days. **Regular season only** (`KEEP_GAME_TYPES = {"R"}` in `pulls/pull_statcast.py`). Postseason pitches are never ingested. |
| 2 | Players and games | MLB Stats API | `players`, `games` | Recent-days window plus anything never seen. |
| 3 | Team stats snapshot | MLB Stats API | `team_stats_snapshot` | **Skipped when `SEASON_PHASE` is POSTSEASON/OFFSEASON.** |
| 4 | Upcoming schedule (14 days) | MLB Stats API | `schedule_games` | Includes probable pitchers. |
| 5 | Remaining schedule | MLB Stats API (`gameType=R`) | `remaining_schedule` | Full replace each run. Powers strength of schedule. |
| 6 | Rosters, schedule, news | MLB Stats API, MLB.com RSS | `rosters`, `schedule_games`, `news_items` | News `fetched_at` is stamped only when a **new** story is inserted. Summaries come from the Anthropic API only if a key is set; otherwise the RSS blurb is used. |
| 7 | Prospects (Mondays only) | mlb.com Pipeline ranking pages + Stats API stat lines | `prospects`, `prospect_scouting` | |
| 8 | Record completion | - | `/data/pull_state.json` | Written whenever the pull succeeds, even with zero new pitches. This is what the nav's "Refreshed ..." label shows. |
| 9 | If new pitches arrived: clear caches | - | memory | Now clears only entries involving the current season (or no season); finished seasons' entries are kept. |
| 10 | If new pitches arrived: tunneling-pairs precompute | local DB | `parquets/tunneling_pairs_<season>.parquet`, `tunneling_totals_<season>.parquet` | Runs in a subprocess (memory is released). |
| 11 | Warm-up | - | memory | See section 4. |

A failed step logs and the pull continues where it can; a failure before step 8
leaves the "Refreshed" time unchanged.

## 4. Warm-up (`api/warmup.py`) - current season only

Builds, for `CURRENT_SEASON`, the things the first visitor would otherwise wait
for: Stuff+ tables (overall and by batter stand), pitcher roles, league HR/FB, Deception+,
Tunneling+ and Location+ raw predictions (injected from score parquets), qualified
and unqualified percentile pools for ALL/L/R, league-average rows, the similar-players
pools (which span all seasons but read each past season's pool from disk), the
pitcher arsenal pool, season-wide pitcher-page tables (arm angles, peer movement,
approach-angle pools), Trends league averages, probable-pitcher score dictionaries,
standings, the player list, and the last 10 days' top-performer leaderboards.

Past seasons are not rebuilt: they come from the persisted pools in
`parquets/percentile_pools/`, the two snapshot files, or are computed on first view
and cached.

## 5. Precomputed files (built by hand with tools, shipped in the deploy)

| File(s) | Built by | Contents | Used for |
|---|---|---|---|
| `parquets/career_season_aggregates.json` | `tools/precompute_career_aggregates.py` | per-season HR/FB, sorted K and HR counts, per-pitcher Stuff+/Tunneling+/Location+, pitcher arsenals | Career page, similar-players arsenal pool |
| `parquets/past_season_pitcher_page.json` | `tools/precompute_past_season_pitcher_page.py` | arm angles, peer movement, approach-angle pools, league batter/pitcher rows, swing profile, wOBA/xwOBA, Trends league averages, per past season | pitcher page, player page, Trends |
| `parquets/percentile_pools/<season>/*.parquet` | built lazily on first request, or `tools/build_percentile_pools.py` / `tools/build_unqualified_pools.py` | qualified and unqualified percentile pools (version hash includes the season's row count, so a changed season rebuilds itself) | every percentile on the player page |
| `parquets/tunneling_pairs_*.parquet`, `tunneling_totals_*.parquet` | `tools/precompute_tunneling_pairs.py` (also run by the daily pull when new pitches arrive) | per-pitcher tunneling pair tables | player page tunneling card |
| `parquets/stuff_plus_v4_<season>.parquet`, `location_plus_v3_<season>.parquet`, tunneling raw predictions, `score_meta.json` | `tools/maintenance/rescore_models.py` (weekly in-season; past seasons are skipped once a file exists) | per-pitch model predictions + normalization params | skips live LightGBM scoring at boot |
| `diagnostics/deception_v2/*.parquet`, `diagnostics/location_v3/*.json` | model work, hand-copied | Deception+ scores, Location+ scoring tables | Deception+/Location+ pages |

Pulls that are **not** scheduled and only run when someone runs them:
`pull_sabermetrics.py` (wRC+, bWAR -> `player_advanced_stats`), `pull_advanced_stats.py`,
`pull_fangraphs_stats.py`, `pull_pitcher_running_game.py`, `pull_injuries.py`
(`il_status`), `pull_awards.py` (`player_awards`), `pull_pitcher_bio.py`, and the
backfill scripts. Anything that reads those tables is as fresh as the last manual run.

## 6. Where the postseason lives today

- **Pitches:** nowhere. Only game type `R` is ingested, so `pitches` ends at the last
  regular-season game (2026-09-27). (`docs/methodology/09_season_phase_guide.md` says
  postseason pitch data is ingested; the code does not do that.)
- **Bracket and series results:** fetched live from the MLB Stats API
  (`analytics/playoffs.py`, `gameType` F/D/L/W) with a 5-minute cache. Nothing is stored.
- **Box scores / top performers:** fetched live from the Stats API per game
  (`analytics/boxscore.py`, in-memory dict).
- **Rosters and the schedule rows (statuses can go stale):** `rosters`, `schedule_games`.

## 7. Season rollover: what flips by itself and what doesn't

Flips by itself once 2027 pitches are ingested:
- Season pickers and pages that use "latest season with data" (player, compare,
  splits, standings, playoffs, teams, trends, team card ranking lists).
- Warm-up targets, percentile pools for the new season (built lazily, then persisted).
- The nav's "games through" date.

Needs a manual change (in this order, before Opening Day):
1. `config/model_policy.py`: `CURRENT_SEASON = 2027`. Backend defaults, the pulls,
   warm-up and cache clearing all key off this.
2. `fly.toml` `SEASON_PHASE`: set to `REGULAR_SEASON` (currently `POSTSEASON`).
   **If this is forgotten:** team-stat pulls stay skipped, new pitches take the
   postseason branch that only clears the leaderboard and standings caches (pools
   and model tables go stale), and the weekly rescore never runs. Also the freeze banners keep showing.
3. Rerun the precompute tools so 2026 becomes a "past" season:
   `precompute_career_aggregates.py`, `precompute_past_season_pitcher_page.py`,
   `build_unqualified_pools.py`, `build_percentile_pools.py`, and make sure
   `tunneling_pairs_2026/totals_2026` exist. Until you do, 2026 falls back to the
   slow live scans.
4. Run `rescore_models.py --season 2027` (or wait for the first Sunday) so the new
   season has score parquets. Before that, 2027 scores are computed live.
5. `analytics/splits.py` `ALL_STAR_DATES`: add 2027 (otherwise it warns and uses July 14).
6. Re-run the prospect pull once the new list posts if you don't want to wait for Monday.

Things that look season-bound but are fine: model training ranges (2015-2026 in the
Location/Tunneling level derivation) are fixed on purpose; `EARLIEST_SEASON_START`
(2021-03-20) only bounds how far back the skip-list checks.

## 8. Risks worth knowing about

1. **Derived files are on the container disk, not the volume.** A weekly rescore, a
   daily tunneling precompute, or a newly built current-season pool survives only until
   the next restart or deploy; afterwards the image's copy is used. Fixes: ship fresh
   parquets with each deploy, or point those directories at `/data`.
2. **Postseason is not stored.** The Outings tab cannot show postseason games until
   postseason pitches are ingested into a separate place (the main `pitches` table is
   read by every regular-season stat and by model training, so postseason rows must not
   go there).
3. **Phase handling is only partly wired.** `OFFSEASON` is defined, but the pull only
   checks `skip_team_stat_pulls` and `is_postseason`; `should_pull_statcast()` is not
   consulted, so the daily pull still runs in the offseason (cheaply: it finds nothing new).
4. **Manual-only data goes stale silently** (wRC+/bWAR, injuries, awards): nothing
   schedules them.
5. **Memory:** one 1 GB machine. The out-of-memory kill earlier on 2026-11-07 wiped the
   caches; `MALLOC_ARENA_MAX=2` and chunked league averages address the known cause, but
   a new unbounded per-season scan could bring it back.
