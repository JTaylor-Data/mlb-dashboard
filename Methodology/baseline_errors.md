# Baseline Errors and Regression Analysis

**Date**: 2026-10-05  
**Branch**: codebase-audit  
**Pre-audit tag**: pre-audit (d573b90)

---

## Method

Pre-audit worktree created at `C:\Users\joshs\Desktop\mlb_dashboard_preaudit`.
Server on port 8011 not started (requires DB connection unavailable in this environment).
Analysis is static: reading source code and golden snapshots.

---

## Requests Known to Return Non-200 / Empty at Pre-Audit

| Endpoint | Method | Parameters | Pre-Audit Status | Now | Classification |
|----------|--------|------------|-----------------|-----|----------------|
| `/api/deception/meta` | GET | — | 404 (No meta for version v2b) | 200 | Pre-existing regression (v2b had no path) |
| `/api/deception/leaderboard` | GET | — | 404 (No deception scores found) | 200 | Pre-existing (v2b path missing from _SCORES_PATHS) |
| `/api/deception/pitcher/{id}` | GET | — | 404 (No deception scores found) | 200 | Pre-existing |
| `/api/splits` | GET | split_by=pitch_type | 500 (ValueError) | 200 | Pre-existing (split_by=pitch_type not handled) |

---

## Requests Whose Results Differ Now vs Pre-Audit

| Endpoint | Change | Reason | Intentional |
|----------|--------|--------|-------------|
| `/api/probable-pitchers` | Location+ and Tunneling+ scores now use active model versions | DEF-03 fix: location_v3 instead of location_v1; tunneling_v2 instead of tunneling_v3 | Yes — was showing wrong model |
| `/api/players/{id}/career` (pitcher) | Tunneling+ and Location+ scores for current season now use active model versions | DEF-04 fix | Yes — was showing wrong model |
| `/api/splits?split_by=pitch_type` | Now returns 200 with pitch-type split data instead of 500 | DEF-09 fix | Yes |
| `/api/deception/*` | Now serves v2b scores | ND-DEC-01 fix | Yes |

---

## Defect Classification: Pre-existing vs Regression

| Defect ID | Description | Pre-existing or Regression |
|-----------|-------------|---------------------------|
| DEF-03 | probable_pitchers.py imports old model versions | **Pre-existing** (in codebase before audit) |
| DEF-04 | career.py imports old model versions | **Pre-existing** (in codebase before audit) |
| ND-AUDIT-03 | /api/splits 500 for pitch_type | **Pre-existing** (never supported) |
| ND-DEC-01 | Deception+ endpoints 404/empty | **Pre-existing** (v2b path never added to _SCORES_PATHS) |

None of the four target defects are regressions introduced by the audit.
