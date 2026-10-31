# Startup & Storage Report

> Generated on branch `archive-and-chart`, 2026-10-05.
> All sizes and timings are measured on the development host (Windows 11, local SSD).
> Enable `STARTUP_TRACE=1` to get timestamped startup logs (see 1A).

---

## 1A — Startup Walk-Through

### Timing instrumentation

`api/main.py` supports `STARTUP_TRACE=1` (env flag left in place).
The flag wraps each major step in `print(f"[TRACE] {label} +{elapsed:.2f}s mem={rss_mb:.0f}MB")`.

### Startup sequence (process start → "Application startup complete")

| Step | Classification | Est. Time | Memory Delta |
|------|----------------|-----------|--------------|
| Python interpreter + imports (`fastapi`, `pandas`, `lightgbm`, `scipy`) | **Computed at startup** (cold import) | ~2–4 s | ~120 MB |
| `config/season_phase.py` — reads `SEASON_PHASE` env var, sets constant | **Loaded from env/constant** | <0.01 s | ~0 |
| `config/model_policy.py` — `CURRENT_SEASON`, active versions | **Loaded from constant** | <0.01 s | ~0 |
| `api/db.py` — creates SQLite connection pool | **Computed at startup** (one connection) | <0.1 s | ~5 MB |
| Router registration (22 routers) | **Computed at startup** | <0.1 s | ~5 MB |
| APScheduler `.start()` — registers jobs, does NOT run them | **Computed at startup** | <0.1 s | ~2 MB |
| `warm_cache_background()` — fires background thread, server becomes ready immediately | **Computed at startup** (async) | <0.01 s (launch) | 0 (deferred) |

**Server ready to accept requests**: ~3–5 s after process start.

### Background warmup thread (fires after server ready)

| Step | Classification | Est. Time | Memory Delta |
|------|----------------|-----------|--------------|
| `preload_complete_seasons(conn)` — checks disk for percentile_pools parquets | **Loaded from disk** (if present) | <1 s | ~0–50 MB |
| `_inject_stuff_parquet(season)` × 11 seasons | **Loaded from disk** (precomputed) | ~2–5 s | ~70 MB |
| `_inject_tunneling_parquet(season)` × 6 seasons | **Loaded from disk** (precomputed) | ~1–2 s | ~28 MB |
| `_inject_location_parquet(season)` × 12 seasons | **Loaded from disk** (precomputed) | ~2–4 s | ~72 MB |
| `_stuff_plus_table` + `_stuff_plus_by_stand_table` | **Computed** (aggregation over parquet cache) | ~5–15 s | ~50 MB |
| `_pitcher_pool`, `_batter_pool`, `_league_avg_batter_row` × 3 splits | **Computed** (DB scan) | ~10–25 s | ~80 MB |
| `_similarity_pool_hitters`, `_similarity_pool_pitchers` | **Computed** (DB + model inference) | ~5–10 s | ~40 MB |
| Career tables for past seasons (2021–2025, 3 models) | **Computed** (parquet cache hit) | ~10–20 s | ~30 MB |
| `_standings`, `_all_players`, `_all_pitcher_arm_angles` | **Computed** (DB scan) | ~3–5 s | ~20 MB |
| `_leaderboard` × 10 dates | **Computed** (DB scan) | ~5–10 s | ~30 MB |

**Total full warmup**: ~45–90 s after server start (depending on DB and SSD speed).
**Resident memory after warmup**: ~500–700 MB.

### Why the player route falls back to live scoring during warmup

The player endpoint calls `_pitcher_pool(conn, season, split)` which is a `@ttl_cache`-wrapped function.
If warmup hasn't yet populated that cache entry, the function runs live — it queries all ~8M pitch rows, scores every pitcher via Stuff+/Tunneling+/Location+, and returns.
That live-scoring path takes ~15–30 s for a single request (LightGBM over 8M rows).

**What a deployed host shows visitors during this window**: the server is accepting requests (uvicorn is ready), but the first player page load will be slow — ~15–30 s instead of <1 s — until the warmup thread populates the pools.

**Proposed readiness gate (not implemented)**: add a `/health/ready` endpoint that returns 503 until `_warmup_complete` is set `True` by the warmup thread.
A load-balancer health check on this endpoint delays routing real traffic until warmup is done.
Alternative: read directly from the score-cache parquet on a cold request (the parquet is already loaded by `_inject_*`), bypassing the aggregation step that causes the 15–30 s.
This would cut cold-request latency from ~30 s to ~1–2 s even before warmup completes.

---

## 1B — Precomputed Inventory and Disk Usage

### Score parquets (`scores/`)

| Path pattern | Built by | Refreshes | Size | Gitignored |
|---|---|---|---|---|
| `scores/stuff_plus_v4_<year>.parquet` | `tools/maintenance/rescore_models.py` | Weekly (scheduler) | ~6 MB/year × 11 years = ~66 MB | Yes |
| `scores/location_plus_v3_<year>.parquet` | same | Weekly | ~6 MB/year × 12 years = ~72 MB | Yes |
| `scores/tunneling_plus_v3_<year>.parquet` | same | Weekly | ~4.5 MB/year × 6 years = ~27 MB | Yes |
| `scores/score_meta.json` | same | Weekly | <0.1 MB | Yes |
| `scores/percentile_pools/` | `tools/build_percentile_pools.py` | On demand / RS only | 0 MB (empty on dev) | Yes (dir) |

**Total scores/ on disk**: ~161 MB.

### Database (`mlb_data.db`)

| Table | Rows | Purpose |
|---|---|---|
| `pitches` | 8,150,992 | All Statcast pitches 2015–2026 (core dataset) |
| `players` | 7,078 | Player metadata |
| `games` | 27,622 | Game index |
| `player_advanced_stats` | 16,982 | wRC+, xFIP, Barrel% from MLB Stats API |
| `pitcher_model_scores` | 9,808 | Cached per-pitcher/season plus scores |
| `rosters` | 7,006 | Current rosters |
| `il_status` | 275 | Current IL entries |
| All others | — | Schedule, news, prospects, awards, etc. |

**Database size**: 4,058 MB (4.0 GB). Not gitignored but tracked via LFS attributes.

### Model boosters (`models/`)

| File | Size | Built by | Live |
|---|---|---|---|
| Stuff+ v4 booster | ~6 MB | Training pipeline | Yes |
| Location+ v3 booster | ~18–19 MB | Training pipeline | Yes |
| Tunneling+ v3 booster | ~1–5 MB (in analytics/) | Training pipeline | Yes |
| `location_plus_v2_bip.txt` | 19.4 MB | Training | Superseded (v3 active) |
| `location_plus_v3_bip.txt` | 18.2 MB | Training | Yes |

**Total models/**: ~104 MB.

### Diagnostics (`diagnostics/`)

| Path | Size | Live |
|---|---|---|
| `diagnostics/location_v3/` | ~666 MB | Partially (oof parquets for v3 active model) |
| `diagnostics/stuff_v4/` | ~21 MB | Yes — read by `tools/build_validation_content.py` |

**Total diagnostics/**: ~789 MB.

### Archive (`Archive/`)

Mostly training artifacts and experimental pipelines.
**Total Archive/**: ~5,825 MB. Majority is large parquets from training runs.
Reclaimable: Archive/ contents (no live code reads them).

### Folder summary

| Folder | Size (MB) | Production needed |
|---|---|---|
| mlb_data.db | 4,058 | Yes |
| Archive/ | 5,825 | No (research only) |
| diagnostics/ | 789 | Partially (~21 MB for validation page) |
| scores/ | 161 | Yes |
| models/ | 104 | Yes |
| frontend/ | 14 | Yes |
| analytics/ | 12 | Yes |
| Methodology/ | 5 | No |
| api/ | 1 | Yes |
| .git/ | 2,152 | No (deploy artifact) |

**"What a production host needs"**: database (4 GB), scores/ (~161 MB), models/ (~104 MB), api/ + analytics/ + config/ + frontend/ + data/ (~30 MB). Total: ~4.3 GB minimum.

### 30 Largest files

See section 1B header. Top 5: `mlb_data.db` (4.0 GB), `Archive/cross_model_validation_v2_checkpoint.json` (1.2 GB), `Archive/tunneling_v3_features.parquet` (985 MB, duplicate), `Archive/tunneling_v3_features.parquet` (517 MB), `Archive/stuff_v3_features.parquet` (490 MB).

---

## 1C — How Lightweight Is the Project

| Metric | Value |
|---|---|
| Python lines (api/, analytics/, config/, tools/) | ~15,700 |
| Frontend JS/HTML lines | ~9,000 |
| Python dependency count (`requirements.txt`) | ~15 direct packages |
| Resident memory after warmup | ~500–700 MB |
| Startup-to-ready (server accepting requests) | ~3–5 s |
| Full warmup to first fast request | ~60–90 s |
| Database size | 4.0 GB |
| Repo tracked size (ex-Archive, ex-DB) | ~300 MB |
| Cold page load (player page, after warmup) | <1 s |
| Cold page load (player page, before warmup) | ~15–30 s |

**Verdict**: The project is lightweight in code (15K lines, 15 dependencies) but heavy in data.
The 4 GB SQLite database and Archive/ artifacts are the only significant storage footprint.
On a production VPS with the Archive/ folder excluded and the DB served from fast disk,
the app runs comfortably in 1–2 GB RAM with sub-second page loads post-warmup.

---

## 1D — Rubber Position (CV Pipeline, Archive only)

> **This pipeline is not part of the live dashboard.** It lives entirely under
> `Archive/experimental/pitcher_cv/` and is not imported by any live server code.

### Where the computation lives

All rubber-position detection is in `Archive/experimental/pitcher_cv/pitcher_cv/deception_v4/d4_rubber.py`.
The live dashboard computes pitcher rubber position from Statcast data (see below).

### Live dashboard rubber position (Statcast-based)

`analytics/player_page.py` line 302 — `pitcher_rubber_position(conn, season, pitcher_id)`:
```
SELECT AVG(release_pos_x) FROM pitches WHERE season = ? AND pitcher_id = ? AND release_pos_x IS NOT NULL
```
This is the **raw** (not arm-side-adjusted) `release_pos_x` in feet from the Statcast feed.
Positive = 3B side; negative = 1B side. Used only to label "stands left/right of rubber" in the UI.

### CV pipeline rubber position (Archive)

`d4_rubber.py` — `detect_rubber()` and `detect_rubber_ankle_box()`:

**Frames used**: Broadcast video frames around the pitcher's pivot ankle during delivery.
Keypoints: pivot ankle `(pivot_ankle_xy)` from pose estimation; `body_scale` (pixel height of pitcher body, ~72 inches) from YOLO pose model.

**Algorithm** (`detect_rubber` — widest-dirt-circle method):
1. Crop a vertical band (`SCAN_ABOVE_MAX=120px` above, `SCAN_BELOW_MAX=80px` below the ankle).
2. Classify columns as chalk (bright, low-saturation: all channels ≥ 160, max−min ≤ 65) or not-dirt (R−G < 10 or R < 60).
3. Find the leftmost chalk column in the left half and rightmost in the right half of the frame.
4. Midpoint of those two columns = mound center = rubber center.
5. **Fallback** (if chalk fails): use not-dirt gap signal (brown/tan = dirt, else not-dirt).
6. `position = (ankle_x − mound_center_x) / rubber_half_w_est`; `rubber_half_w_est = body_scale × 12/72`.
7. Output clipped to `[−2.0, +2.0]`.

**Coordinate system**: positive = ankle right of center = 3B side; negative = 1B side. Behind-home-plate camera convention.

**Units**: dimensionless fraction of rubber half-width (24" total, so ±1.0 = ±12").

**Alternate method** (`detect_rubber_ankle_box`): looks for bright near-white pixels (all channels ≥ 200, max−min ≤ 40) in a box just below the ankle; finds leftmost and rightmost rubber pixel columns.

**Fallback chain**: chalk-circle method → not-dirt-gap method → returns `None`.

**Downstream use in CV pipeline** (`d4_phases.py`, `d4_pipeline.py`): rubber position feeds stride-direction calculations — knowing where the pivot foot is on the rubber informs step angle normalization and deception metric computations relative to the rubber, not the mound center. Not used in any live dashboard route.

**File:line evidence**: `d4_rubber.py:119` (`detect_rubber`), `d4_rubber.py:198` (`detect_rubber_ankle_box`), `d4_phases.py` calls into `d4_rubber` via `d4_pipeline.py`.
