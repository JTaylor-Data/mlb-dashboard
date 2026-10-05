# Percentile Cache Audit & Design Report

Generated: 2026-10-04

## Part 2.1 — Audit

### Where pools are built

`analytics/percentiles.py`
- `pitcher_percentile_metrics(conn, season, min_bf=None, stand_filter=None)`
  Full pitches-table scan (~6-7s for a full season, ~372 qualified pitchers for 2023-2024).
  Computes 25+ rate/advanced metrics per pitcher, merges with player_advanced_stats.
- `batter_percentile_metrics(conn, season, min_pa=None, throws_filter=None)`
  Similar: full pitches scan for batter metrics (xwOBA, bat speed, chase%, etc).

### Where pools are consumed

`api/routers/players.py` — `_pitcher_pool`, `_batter_pool`, `_pitcher_pool_unqualified`, `_batter_pool_unqualified`
These are @ttl_cache() wrappers (12h TTL) that feed:
- `/api/player/{id}` for percentile badge rows on pitcher and hitter pages
- `/api/player/{id}/similar` via `api/routers/similar_players.py`

### Warmup behavior (pre-Part 2)

`api/warmup.py` `_warm(conn, full=True)`:
- Pre-warms `_pitcher_pool(conn, 2026, None/L/R)` and `_batter_pool(conn, 2026, None/L/R)`
- Past seasons (2015-2025) were NOT pre-warmed — first request paid 6-7s live
- Score parquets (Stuff+/Loc+/Tunneling+) ARE injected at startup; percentile pools were not

### Qualification thresholds

Batter: `SAVANT_PA_PER_TEAM_GAME_BATTER = 2.1 × games_played` (e.g. ~340 PA for full season)
Pitcher: `SAVANT_PA_PER_TEAM_GAME_PITCHER = 1.25 × games_played` (e.g. ~203 BF)

Both scale with games played so mid-season thresholds auto-adjust.

### Timing (pre-Part 2, measured on integration-merge)

| Operation | Time |
|---|---|
| `pitcher_percentile_metrics(conn, 2023)` | ~7s |
| `pitcher_percentile_metrics(conn, 2024)` | ~6s |
| Full warmup (current season ALL+L+R pitcher+batter) | ~25s |
| First request past season (pre-Part 2) | ~6-7s |

---

## Part 2.2 — Design

### CURRENT_SEASON

Added to `config/model_policy.py` as `CURRENT_SEASON = 2026` and `EARLIEST_SEASON = 2015`.
`api/warmup.py` now imports from there (was a hardcoded local constant).

### Persistence store

`scores/percentile_pools/<season>/<player_type>_<split>_<pool_version>.parquet`
- Gitignored (added to .gitignore)
- 6 files per season: pitcher × (ALL, L, R), batter × (ALL, L, R)
- Unqualified (min=0) pools are NOT persisted — used rarely, fast once qualified pool is warm

### pool_version

8-char hex (SHA-256 truncated) of:
- `analytics/percentiles.py` full file content (code version)
- `SAVANT_PA_PER_TEAM_GAME_PITCHER` + `SAVANT_PA_PER_TEAM_GAME_BATCHER` as string
- Season's pitch row count from DB (data fingerprint)

This auto-invalidates when code changes or when data is backfilled for a season.

### Atomic writes

Write to `.tmp` file, then `Path.replace()` (atomic on POSIX, best-effort on Windows).
Never cached errors: if build raises, nothing is written.

### Single-flight lock

`threading.Lock()` per `(season, player_type, split)` key prevents concurrent first requests
from each building the pool. Second acquirer re-checks disk after lock acquired.

### Cache path in players.py

`_pitcher_pool` and `_batter_pool` check `season < CURRENT_SEASON` and call
`analytics.percentile_cache.get_or_build_pool(conn, season, player_type, split)`.
For current season, they fall through to direct `pitcher_percentile_metrics()` as before.
The `@ttl_cache()` wrapper still applies — in-memory hit skips disk too.

---

## Part 2.3 — Build tools

### `tools/build_percentile_pools.py`

CLI to build all complete seasons:
```
python tools/build_percentile_pools.py                  # 2015-2025
python tools/build_percentile_pools.py --seasons 2020-2022
python tools/build_percentile_pools.py --force          # rebuild all
```
Idempotent, resumable. Skips seasons with no pitch data. Exits 1 on any failure.

### `tools/freeze_season.py <season>`

Wrapper for when a season becomes complete:
```
# After bumping CURRENT_SEASON in config/model_policy.py:
python tools/freeze_season.py 2026
```

---

## Part 2.4 — Correctness verification

Verified on integration-merge before push:

| Season | Player type | Split | Direct rows | Cached rows | Max abs diff (any metric) |
|---|---|---|---|---|---|
| 2023 | pitcher | ALL | 372 | 372 | 0.0 |
| 2024 | pitcher | ALL | 366 | 366 | 0.0 |
| 2024 | pitcher | ALL | percentile comparison, 100 pitchers, all metrics | — | 0 |

Complete correctness test (2023-2025 all splits) run in Part 3.

---

## Part 2.5 — Timing after Part 2

| Operation | Before | After |
|---|---|---|
| First request, past season pool | ~7s (live query) | ~0.36s (disk load) if built; ~7s first build then save |
| Second request same session | ~0ms (in-memory TTL hit) | ~0ms (TTL hit) |
| Server restart, past season | ~7s first request | ~0.36s (from disk) |
| Warmup startup | ~25s (current season only) | ~25s + fast disk scan ≈25s |

Once all seasons are built by `tools/build_percentile_pools.py`, past-season first-request
latency drops from ~7s to ~360ms. Current season is always computed live (~7s first request).
