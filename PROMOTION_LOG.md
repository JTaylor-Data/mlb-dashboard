# PROMOTION LOG — Stuff+ v4

**Promoted**: 2026-10-02  
**Branch**: `stuff-v4-live`  
**Promoted by**: Automated promotion session

---

## What changed

Stuff+ v4 adds `haa_residual` (horizontal approach angle, arm-side mirrored, OLS residual) as a 16th feature. The `season` and `effective_speed` features from v3 were retained (ablation ruled drops as significantly worse or underpowered). HAA leakage checks passed (|r| with plate_x = 0.009, |r| with release_pos_x = 0.084, both below 0.10 threshold).

## Files changed

| File | Change |
|------|--------|
| `analytics/stuff_model.py` | `STUFF_ACTIVE_VERSION = "v4"` (one-line rollback flag); loads v4 model files; cache key includes model version |
| `analytics/stuff_features.py` | Added `compute_haa()`, `add_haa_residual()`; updated `engineer_features()` to call HAA when `haa_coefs` in calibrations; added `RAW_QUERY_COLUMNS_V4` (includes vx0, ax, plate_x, release_pos_x) |
| `api/routers/stuff_model_report.py` | Added `/api/stuff-model/meta` endpoint; updated leaderboard to support season 2016–2026, pitch type filter, n, direction; 2020 asterisk on all leaderboard rows; top/bottom 15 |
| `mlb_data.db` | `pitcher_model_scores.stuff_plus` overwritten with v4 scores; `stuff_plus_v4` column added; `n_pitches` column added |

## Rollback instructions

**One step**: change `STUFF_ACTIVE_VERSION = "v3"` in `analytics/stuff_model.py` and restart the server.

- The v3 model file (`stuff_plus_v3_lgbm.txt`) is unchanged on disk.
- The DB has both `stuff_plus` (now v4) and `stuff_plus_v4` columns; reverting the DB requires re-running v3 scoring.
- The `RAW_QUERY_COLUMNS` default is now v4 (includes HAA columns); switching back to v3 routing requires either: (a) re-pointing the import to `RAW_QUERY_COLUMNS_V3`, or (b) confirming the extra v4 columns are safely ignored by the v3 feature list (they are — `all_features` in v3 meta does not include `haa_residual`).

## Version metadata

- Model hash (SHA-256): `0addb8e96ea438ec196b3addc19fd350f1d76e241a9291d3719773f2b9766c28`
- Short hash: `0addb8e9`
- Trained: 2026-10-02 14:54:17
- Training seasons: 2016–2026 (excl. 2020)
- Scored seasons: 2016–2026 (all 11, incl. 2020 with asterisk)

## Stop-ship gates

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| v4 vs v3 corr (2021–2025, ≥300 pitches) | 0.9728 | ≥ 0.85 | ✓ |
| Fairbanks (2022) in top 25 | Rank 1 | ≤ 25 | ✓ |
| Clase (2022) in top 25 | Rank 2 | ≤ 25 | ✓ |
| All 11 seasons present (2016–2026) | 11/11 | all | ✓ |
| 2020 season_flag = "60-game season" | ✓ | required | ✓ |
| KN pitcher below average | 97.94 vs 99.98 avg | below avg | ✓ |

## Status badge

`stuff_v4_status: "provisional"` — badge shows "v4 — validation in progress". Do not auto-flip.
