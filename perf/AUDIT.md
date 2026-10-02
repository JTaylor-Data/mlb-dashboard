# Performance Audit — MLB Dashboard

Generated: 2026-10-02  Branch: perf-cleanup

## Executive Summary

Five findings account for essentially all measured slowness. Three are bugs
(wrong cache decorator, NaN serialization failure, missing `role` param).
Two are design issues (no server-side search filtering, PRAGMA overhead).
All are fixable without touching model logic or DB schema.

---

## Findings (ranked by impact)

| # | Severity | File(s) | Warm before | Expected after |
|---|----------|---------|-------------|----------------|
| F1 | CRITICAL | heatmaps.py | 3100ms every call | ~2100ms after first hit |
| F2 | CRITICAL | career.py | cold 22–37s | <5s cold |
| F3 | BUG | analytics/player_page.py | 500 error | 200 |
| F4 | BUG | perf/benchmark.py | 422 error | 200 |
| F5 | HIGH | api/routers/players.py | 2100ms every search | <200ms cached |
| F6 | MEDIUM | api/db.py | ~200ms floor | ~50ms floor |

---

### F1 — Heatmaps cache never hits (CRITICAL)

**File**: `api/routers/heatmaps.py` lines 35, 83; `api/routers/hitzones.py` line 40

**Root cause**: Three functions decorated with `@ttl_cache_all_args()` all take
`conn` as their first argument. `ttl_cache_all_args` includes ALL args in the
cache key — including the `conn` object, which is a fresh `sqlite3.Connection`
created per request. The key never repeats, so every request is a cache miss.

```python
# WRONG — conn is a unique object each request; key never matches
@ttl_cache_all_args()
def _league_command_pool(conn, season, stand_filter=None): ...

@ttl_cache_all_args()
def _pitch_location_summary(conn, season, pitcher_id, stand_filter=None): ...

@ttl_cache_all_args()   # hitzones.py
def _hit_zone_summary(conn, season, batter_id, throws_filter=None): ...
```

**Fix**: Change all three to `@ttl_cache()`, which explicitly excludes the first
`conn` argument from the key (see `api/cache.py` docstring: *"every cached
function in this app takes a fresh sqlite3 connection as its first argument...
that argument is excluded from the cache key on purpose"*).

**Impact**: `_league_command_pool` runs a full-table scan:
`SELECT pitcher_id, pitch_type, plate_x, plate_z FROM pitches WHERE season = ?`
on 8M rows. Currently this runs on every heatmap request. After fix, runs once
per season per stand split (12h TTL), dropping warm latency from ~3100ms to
~2100ms (the connection-overhead floor).

---

### F2 — Career N+1 uncached league scans (CRITICAL)

**File**: `api/routers/career.py`, `_pitcher_career()` and `_batter_career()`

**Root cause**: For each season in the player's career, the code calls
`league_hr_per_fb(conn, int(season))` directly (the uncached version in
`analytics/player_page.py`). This does a full pitches-table scan per season.
A pitcher with 8 career seasons triggers 8 full scans cold (~22s), a batter
with 15 seasons triggers 15 (~37s).

```python
# In _pitcher_career loop:
lg_hr_fb = league_hr_per_fb(conn, int(season))  # uncached, full scan each time
```

A cached wrapper `_league_hr_fb` already exists in `api/routers/players.py`
with `@ttl_cache()`. The career router just needs to call that instead, or
`league_hr_per_fb` itself needs `@ttl_cache()`.

**Impact**: Cold career load 22–37s → estimated 3–5s after fix (single cached
scan per season rather than per player per request).

---

### F3 — Rolling xwOBA 500: NaN not JSON-serializable (BUG)

**File**: `analytics/player_page.py`, `batter_rolling_xwoba()` ~line 709

**Root cause**: `estimated_woba_using_speedangle` contains NULL values in the
DB; `astype(float)` converts them to `float('nan')`. `np.mean()` propagates NaN
through the rolling window. `float(np.nan)` is not JSON-serializable and raises
`ValueError: Out of range float values are not JSON compliant: nan`.

```python
qpa["_xw"] = qpa["estimated_woba_using_speedangle"].astype(float)  # NULL → NaN
xw_vals = qpa["_xw"].values
for i in range(total):
    pt[str(w)] = float(np.mean(xw_vals[i + 1 - w: i + 1]))  # NaN propagates!
```

**Fix**: Use `np.nanmean()` and map all-NaN slices to `None`.

---

### F4 — Similar player endpoints 422: missing `role` param (BUG)

**File**: `perf/benchmark.py` lines 56–57

**Root cause**: Benchmark URLs for `similar_starter1` and `similar_batter1`
omit the required `role` query parameter. The `/api/player/{id}/similar`
endpoint declares `role: str = Query(...)` as required with no default.

```python
# WRONG — missing role=pitcher
("similar_starter1", "GET", f"/api/player/{STARTER_1}/similar?season=2026"),
```

**Fix**: Add `role=pitcher` / `role=batter` to benchmark endpoint URLs.

---

### F5 — Search ignores `q` param; always returns 14K records (HIGH)

**File**: `api/routers/players.py`, `list_all_players()` at `/api/players/all`

**Root cause**: `list_all_players()` calls `_all_players(conn, season)` which
returns all players regardless of the `q` query parameter. Filtering happens
client-side in JS. The full 14K-player list is ~2-3MB and takes ~2100ms every
request (even warm, because the response is large and GZip compression adds
latency).

**Fix**: Filter by `q` server-side in the SQL query (case-insensitive prefix
match on player name) when `q` is present and len ≥ 2. Return full list only
when `q` is absent (to support the initial empty-search display).

**Note**: `_all_players` is `@ttl_cache()`-wrapped so the full 14K list is
returned instantly on warm hits — the 2100ms includes GZip + serialization of
a large payload. Server-side filtering drastically reduces response size.

---

### F6 — Per-request PRAGMA overhead floor (MEDIUM)

**File**: `api/db.py`, `get_conn()`

**Root cause**: Every request creates a new `sqlite3.Connection` and executes 5
PRAGMA statements before handing it to the endpoint. On Windows with WAL mode
and mmap, each PRAGMA round-trips to SQLite. This contributes an observed
~200–400ms floor to every endpoint latency, even those that only read a cache.

```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=5000")
conn.execute("PRAGMA cache_size=-32000")
conn.execute("PRAGMA temp_store=MEMORY")
conn.execute("PRAGMA mmap_size=268435456")
```

**Approach**: A connection pool (even a pool of 1–4) with PRAGMAs set once at
open time would eliminate per-request PRAGMA cost entirely. Alternatively, batch
the 5 PRAGMAs into a single `executescript()` call for a minor improvement.

**Risk**: SQLite connection pools need care under async Uvicorn. Recommend
Tier 2 (pool of 1 for now, expand if load warrants).

---

## Other observations

- `_league_command_pool` uses `itertuples()` in a Python loop to build a dict.
  Vectorizing with pandas groupby would be cleaner, but the bottleneck is the
  cache miss (F1), not the loop itself. Address after F1 is fixed and
  re-benchmarked if still slow.

- `heatmaps_s1_2026` (`stand=ALL`) calls `_pitch_location_summary` with
  `stand_filter=None` but then passes that to `_league_command_pool(conn, season, None)`.
  After fixing F1, this all caches correctly.

- Trends endpoints (~2100–2200ms warm) appear to be at the PRAGMA floor with no
  additional per-request scan. No structural issue found in `trends.py`.

- Career warm (~2100ms) is fine — outer `_pitcher_career` is `@ttl_cache()`-
  wrapped, cold only hits once. F2 matters only for first-load.
