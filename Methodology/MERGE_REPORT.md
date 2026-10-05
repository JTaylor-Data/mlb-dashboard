# Merge Report — 2026-10-04

## What Was Merged

**Integration branch**: `integration-merge`
**Source**: `closeout-blend` (which itself contains all of `deception-v2`)
**Target**: `master`

### Merge Order

1. `git checkout -b integration-merge master` (from d509519)
2. `git merge --no-ff closeout-blend` — clean, no conflicts (ort strategy)
3. Part 2 commits: percentile cache implementation (bc42d18)

### Files Changed (93 files in closeout-blend merge)

Major additions:
- `models/deception_plus_v2b_*` — 15 LightGBM fold files (v2b model artifacts)
- `models/deception_plus_v2_*` — 16 fold files (v2 interim artifacts, kept per Rule 3)
- `api/routers/deception_report.py` — full v2b parquet-based router
- `api/warmup.py` — score-cache parquet injection
- `api/scheduler.py` — weekly rescore cron
- `scripts/rescore_models.py` — rescore CLI
- `diagnostics/deception_v2/` — training and validation artifacts
- `diagnostics/closeout_blend/` — Stuff+/Location+ closeout validation
- `scores/score_meta.json` — season metadata for all scored seasons
- `Methodology/CLOSEOUT_AND_BLEND_REPORT.md`, `DECEPTION_COMPLETION.md`, etc.

Part 2 additions:
- `analytics/percentile_cache.py` — disk-persisted percentile pools
- `tools/build_percentile_pools.py`, `tools/freeze_season.py`
- `config/model_policy.py` — added CURRENT_SEASON=2026, EARLIEST_SEASON=2015
- `Methodology/PERCENTILE_CACHE_REPORT.md`

## Conflicts

None — clean merge.

## Lost-Change Check

`deception-v2` is a strict ancestor of `closeout-blend`. Master had no commits between
the `pre-deception-v2 cleanup` tag and the merge point. No lost changes.

## Verification Results

### Part 3.1 — Model meta

| Model | Expected version | Actual |
|---|---|---|
| Stuff+ | v4 | v4 (hash 0addb8e9) |
| Location+ | v3 | v3 |
| Deception+ | v2b | v2b (from deception/report) |
| Tunneling+ | v2/v3 routes | responds OK |

Note: `/api/deception/meta` returns 404 for "No meta for version v2b". This is a pre-existing
issue: the meta endpoint looks for a specific JSON file format that v2b doesn't supply.
The `/api/deception/report` endpoint (which is the canonical data source) works correctly.
Logged in NEEDS_DECISION.md as ND-DEF-01.

### Part 3.2 — Golden comparison

| Snapshot | Status |
|---|---|
| standings | MATCH |
| meta_main | MATCH |
| playoffs_bracket | MATCH |
| player_pitcher (543037) | MATCH |
| player_hitter (545361) | MATCH |
| stuff_meta | MATCH |
| location_meta | MATCH |

### Part 3.3 — Smoke tests

All 14 endpoints tested passed (similar players required `role=` query param — expected).

### Part 3.4 — Percentile cache correctness

| Test | Values checked | Max abs diff |
|---|---|---|
| Pitcher 2023-2025 ALL/L/R | 47,889 | 0 |
| Batter 2023-2025 ALL | 21,099 | 0 |

### Part 3.5 — Tunneling byte-identity

PASSED — MD5 checksums of all 26 tunneling files identical before and after merge.

### Part 3.6 — Score cache integrity

| Model | Seasons in meta | Status |
|---|---|---|
| Stuff+ v4 | 11 (2016-2026) | OK |
| Location+ v3 | 12 (2015-2026) | OK |
| Tunneling+ v3 | 6 (2021-2026) | OK |

## Deception+ meta endpoint note

`/api/deception/meta` returns 404 ("No meta for version v2b"). The v2b model uses a
different meta format — the DECEPTION_COMPLETION.md file and the `/api/deception/report`
endpoint are the authoritative sources. This is not a new regression; the endpoint was
never implemented for v2b. Logged in NEEDS_DECISION.md.

## Percentile Cache Timings (Part 2)

| Metric | Before | After |
|---|---|---|
| Past-season first request | ~7s | ~360ms (disk hit) |
| Past-season first build | N/A | ~6-7s then saved |
| Current season (2026) | ~7s | ~7s (always live) |

## Push Result

Pushed in Part 4. See below.

## Rollback Instructions

If rollback is needed after push:
```bash
git checkout master
git reset --hard pre-merge/main
git push origin master --force  # only if needed and only with team approval
```

Pre-merge state tagged as `pre-merge/main` (d509519), pushed to remote.
Safety bundle at `../pre-merge-backup.bundle` contains all branches.
