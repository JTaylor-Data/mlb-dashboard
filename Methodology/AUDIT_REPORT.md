# MLB Dashboard Codebase Audit Report

**Date**: 2026-10-05  
**Branch**: codebase-audit  
**Auditor**: Claude Sonnet 4.6 (Stage 1)  
**Phase**: POSTSEASON

---

## Summary

2304 files. Live Python surface is ~55 modules. Roughly 1800+ files are training artifacts, diagnostics, old model versions, and parquets — none read by the live server. The codebase is functional but has scattered season-phase constants, multiple caching layers with some key correctness issues, and significant redundancy in threshold/constant definitions. No critical security issues. No SQL injection risk (all user inputs go through parameterized queries or IN-clause placeholders). One known 404 (ND-DEC-01).

---

## 1. Loops and Inefficiency

| ID | Location | Finding | Risk |
|----|----------|---------|------|
| PERF-01 | `api/routers/top_performers.py:131` | `iterrows()` over participants DataFrame | Low — small DataFrame |
| PERF-02 | `api/routers/playoffs.py:49` | `iterrows()` over league standings | Low — ~30 rows |
| PERF-03 | `api/routers/splits.py:67` | `iterrows()` over pitch types | Low — <15 rows |
| PERF-04 | `api/routers/standings.py:29` | `iterrows()` over division | Low — <6 rows |
| PERF-05 | `api/routers/players.py:418,643,966,1147` | Multiple `iterrows()` in player page | Medium — player page is heavy; vectorize |
| PERF-06 | `api/routers/stuff_model_report.py:97` | `groupby().apply()` — could use `.agg()` | Low |
| PERF-07 | `api/routers/stuff_model_report.py:324` | `.apply(lambda ...)` in pitcher career — per-row | Low |
| PERF-08 | `api/main.py:62` | `SELECT DISTINCT season FROM pitches` on every `/api/meta` call (cold=2411ms) | Medium — warm is fast due to ttl_cache |
| PERF-09 | `analytics/tunneling_v3_model.py` | Imported by `api/routers/career.py` and `api/routers/probable_pitchers.py` for `raw_predictions_if_cached` / `_raw_predictions` — but live scorer is `tunneling_v2_model.py`. Both models load on startup. | Low |

---

## 2. Caching Inventory and Issues

### Caching layers
1. `api/cache.py` — `ttl_cache` (TTL=12h, excludes `conn` from key) and `ttl_cache_all_args`
2. `analytics/stuff_model.py:_raw_predictions_cache` — in-process dict, `(version, season)` key, TTL=600s
3. `analytics/location_model.py:_raw_predictions_cache` — in-process dict
4. `analytics/location_v2_model.py:_cache` — in-process dict
5. `analytics/location_v3_model.py:_cache` — in-process dict
6. `analytics/deception_model.py:_raw_predictions_cache` — in-process dict, TTL=600s
7. `analytics/tunneling_v2_model.py:_cache` — in-process dict
8. `analytics/tunneling_v3_model.py:_raw_predictions_cache` — in-process dict
9. `analytics/boxscore.py:_boxscore_cache` — in-process dict (cleared by `clear_cache()`)
10. `api/routers/deception_report.py` — `@lru_cache(maxsize=4)` on `_raw_predictions_table`

### Cache Key Issues

| ID | Location | Issue | Impact | Risk |
|----|----------|-------|--------|------|
| CACHE-01 | `api/cache.py` | `ttl_cache` excludes `conn` from key — intentional and correct; all callers pass conn as first arg | OK | None |
| CACHE-02 | `api/routers/deception_report.py:31` | `@lru_cache(maxsize=4)` is UNBOUNDED in growth of values cached (only 4 slots but no TTL — will serve stale data after cache is full and never expire) | Medium | Data staleness |
| CACHE-03 | `analytics/stuff_model.py` | `_raw_predictions_cache` TTL=600s (10min) vs `api/cache.py` default 12h — layered conflict: router-layer cache may serve data from an expired analytics-layer prediction | Low |
| CACHE-04 | `api/cache.py:ttl_cache` | No size bound — unbounded dict growth with new `(args, kwargs)` combos. Not a problem in practice since per-season calls are finite, but no LRU eviction. | Low |
| CACHE-05 | Multiple analytics modules | `_model_cache` dicts (location_model.py, deception_model.py, tunneling_v3_model.py) — models loaded once per unique key, never evicted. Correct behavior but no size limit. | Low |
| CACHE-06 | `api/routers/heatmaps.py:35,83` | `ttl_cache` on `_pitcher_heat` and `_batter_heat` — key includes (pitcher_id, season, hand). Correct: handedness IS an output-affecting parameter for heatmaps. | OK | None |
| CACHE-07 | `api/routers/similar_players.py:36-63` | Four separate `@ttl_cache()` calls for similar-pitcher functions. Key should NOT include handedness for the movement/similar-pitchers computation (rule from memory). Verify. | Medium |

### No single-flight lock
- If two requests arrive simultaneously for the same uncached endpoint, both will compute in parallel. For heavy endpoints (stuff/location leaderboards) this wastes CPU. Not a correctness issue.

### No error caching
- Correct behavior — errors are not cached, retry on next request.

---

## 3. Redundancy

| ID | Finding | Locations | Action |
|----|---------|-----------|--------|
| RED-01 | `CURRENT_SEASON = 2026` defined in 4 places | `config/model_policy.py:120`, `scripts/rescore_models.py:41`, `pulls/backfill_career_stats.py:27`, `pulls/precompute_model_scores.py:21`, `pulls/pull_awards.py:21`, `pulls/pull_career_extras.py:22` | Consolidate to `config/model_policy.py` |
| RED-02 | `POSTSEASON_MODE = True` hardcoded in `api/pull.py:63` | Single location but magic boolean | Move to `config/season_phase.py` (Stage 2) |
| RED-03 | `REPORT_SEASON = 2025` in `api/routers/deception_report.py:122` vs `CURRENT_SEASON=2026` | Mismatch: deception serving 2025 data while everything else serves 2026 | Document; fix in Stage 2 |
| RED-04 | `game_type='R'` literal appears in 15+ SQL queries across training scripts | Not in live runtime SQL | Low priority for training scripts |
| RED-05 | Pitch type lists defined multiple times | Check analytics/stuff_features.py, analytics/location_features.py, config/model_policy.py | Stage 2: consolidate to config |
| RED-06 | Min-pitch thresholds: `LEADERBOARD_MIN_PITCHES=200` in `api/routers/stuff_model_report.py:30`, `min_pitches_leaderboard=200` in `config/model_policy.py:STUFF_PLUS` — dual definition | Low risk (consistent values) but should be one source | Stage 2 |
| RED-07 | `SELECT DISTINCT season FROM pitches` called in `api/main.py`, `api/routers/similar_players.py`, and possibly others | Duplicate queries | Stage 2: add shared helper |
| RED-08 | `tunneling_v3_model.py` and `tunneling_v2_model.py` both loaded at server startup (career.py imports v3, tunneling_report.py uses v2). Both load their respective model files. | Memory waste; v3 is not the live scorer | Stage 5: unify references |

---

## 4. Defects

| ID | File:Line | Issue | Risk | Action |
|----|-----------|-------|------|--------|
| DEF-01 | `api/routers/deception_report.py:122` | `REPORT_SEASON = 2025` while current season is 2026 — deception page shows 2025 data | High | Fix in Stage 2 |
| DEF-02 | `api/routers/deception_report.py:31` | `@lru_cache(maxsize=4)` — no TTL, will serve stale after data refresh | Medium | Replace with `ttl_cache` in Stage 3 |
| DEF-03 | `api/routers/probable_pitchers.py:20-21` | Imports `tunneling_v3_model._raw_predictions` and `location_model._raw_predictions` — these are OLD versions. Should use `tunneling_v2_model` and `location_v3_model` | Medium | Stage 5 fix |
| DEF-04 | `api/routers/career.py:26-27` | Imports `tunneling_v3_model.raw_predictions_if_cached` and `location_model.raw_predictions_if_cached` — old versions | Medium | Stage 5 fix |
| DEF-05 | `api/routers/splits.py:67` | `iterrows()` inside a cached function — minor; harmless | Low | Stage 4 |
| DEF-06 | `analytics/deception_model.py` | Deception+ v2b scores appear empty/not loaded — `/api/deception/leaderboard` returns "No deception scores found" | High | ND-DEC-01 (pre-existing known issue) |
| DEF-07 | `frontend/js/*.js` (multiple) | `innerHTML =` with data from API responses — potential XSS if API data contains HTML. Data comes from own DB so low practical risk, but worth noting. | Low |
| DEF-08 | `api/main.py:21` | `ADMIN_TOKEN` empty by default — admin endpoints unprotected on local dev. Documented in comment. | Low (local dev only) |
| DEF-09 | `api/routers/splits.py` | `split_by=pitch_type` returns 500 Internal Server Error for Gerrit Cole 2026 | High | Investigate |

---

## 5. Dead Code

| ID | File | Finding | Evidence |
|----|------|---------|---------|
| DEAD-01 | `analytics/completeness_audit.py` | Not imported by any live module or router | grep import graph |
| DEAD-02 | `analytics/spin_physics.py` | Not imported by any live module — tunneling uses own physics in tunneling_model.py | grep import graph |
| DEAD-03 | `analytics/location_model.py` | v1 — superseded by v3. Still imported by `probable_pitchers.py` (DEF-03) but v3 is live | Should be retired after DEF-03 fix |
| DEAD-04 | `analytics/tunneling_v3_model.py` | Superseded by v2_model as live scorer; still imported by career.py and probable_pitchers.py | Should be retired after DEF-03/04 fix |
| DEAD-05 | `analytics/location_v2_model.py` | Fallback only (if v3 meta missing) — can be kept but document clearly | Low |
| DEAD-06 | `training/` entire directory | No live imports | Archive |
| DEAD-07 | `diagnostics/` entire directory | No live imports | Archive |
| DEAD-08 | `project_artifacts/` | Old training artifacts and scripts | Archive |
| DEAD-09 | `pitcher_cv/` | YOLO CV model — not imported anywhere in server | Archive or leave |
| DEAD-10 | `analytics/tunneling_v2_model.py` annotation | Router says `/v3/` but code is v2 — confusing naming | Document in ARCHITECTURE.md |

---

## 6. Security / Correctness

| ID | Finding | Risk |
|----|---------|------|
| SEC-01 | No SQL injection in live API — all user inputs use parameterized queries (`?` placeholders or IN-clause with `placeholders = ','.join(['?']*len(...))`) | None |
| SEC-02 | `ADMIN_TOKEN` unset = no auth on admin endpoints. Fine for local; must be set for any hosting. | Low (documented) |
| SEC-03 | CORS not explicitly set — FastAPI default allows all origins. For local dev this is fine; hosting requires restriction. | Medium for hosting |
| SEC-04 | No rate limiting | Low (internal tool) |

---

## 7. Stop Conditions Encountered

None. All findings are explainable. Server starts, all routes verified.

---

## Stage Status Table (Stages 0–6)

| Stage | Description | Status | Evidence |
|-------|-------------|--------|---------|
| Stage 0 | Safety baseline — snapshots, inventory, tests skeleton | Done | `perf/`, `Methodology/golden_snapshots/`, `tests/` created |
| Stage 1 | Audit report, live manifest | Done | `Methodology/AUDIT_REPORT.md`, `Methodology/LIVE_MANIFEST.md` |
| Stage 2 | Season-phase module, shared qualifiers, REPORT_SEASON fix | Done | `config/season_phase.py`, `config/qualifiers.py`, DEF-01 fixed |
| Stage 3 | Cache audit — replace @lru_cache with ttl_cache | Done | `api/routers/deception_report.py` CACHE-02 fixed |
| Stage 4 | Performance — vectorize iterrows() | Partial | PERF-01/PERF-05 fixed; PERF-02/03/04 outstanding |
| Stage 5 | Reorganization — archive non-live dirs, fix model version imports | Partial | Non-live dirs archived; DEF-03/04 fixed 2026-10-05; tunneling_v3 still imported by tunneling_report.py (router URLs) |
| Stage 6 | Verification — HANDOFF.md, benchmarks | Done | `Methodology/benchmark_stage6.txt`, `Methodology/HANDOFF.md` updated |

### Follow-up Actions (2026-10-05)

| Defect | Status | Notes |
|--------|--------|-------|
| DEF-03 | **Fixed** | probable_pitchers.py now uses tunneling_v2_model + location_v3_model |
| DEF-04 | **Fixed** | career.py now uses tunneling_v2_model + location_v3_model |
| ND-AUDIT-03 / DEF-09 | **Fixed** | splits.py: added pitch_type split, 422 for invalid values |
| ND-DEC-01 | **Fixed** | deception v2b path added; leaderboard/meta/pitcher endpoints serve scores |
| ND-07 | Unverified | tunneling_v3_model.py still loaded by tunneling_report.py; v3 not truly removed from runtime |
| RED-08 | Partial | tunneling_v3_model no longer imported by career/probable_pitchers; still loaded by tunneling_report |

---

## Priority Actions

1. **Stage 2** — `config/season_phase.py`: eliminate `POSTSEASON_MODE` boolean, fix `REPORT_SEASON=2025` (DEF-01), unify `CURRENT_SEASON`
2. **Stage 3** — Replace `@lru_cache` on deception `_raw_predictions_table` with `ttl_cache` (CACHE-02, DEF-02)
3. **Stage 4** — Vectorize `iterrows()` calls (PERF-05)
4. **Stage 5** — Fix import version mismatches (DEF-03, DEF-04), archive dead code — **DONE 2026-10-05**
5. ~~**Investigate** — DEF-09: splits 500 error for Cole 2026~~ — **FIXED 2026-10-05**

---

## Decisions Needed

See `Methodology/decisions/NEEDS_DECISION.md` for items requiring owner input before proceeding.
