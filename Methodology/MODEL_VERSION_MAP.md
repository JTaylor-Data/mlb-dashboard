# Model Version Map

**Date**: 2026-10-05  
**Branch**: codebase-audit

Active model versions:
- Stuff+: v4 (`analytics/stuff_model.py`)
- Location+: v3 (`analytics/location_v3_model.py`)
- Tunneling+: v2 (`analytics/tunneling_v2_model.py`) — router URLs still say /v3/
- Deception+: v2b (`analytics/deception_model.py` for live scoring; pre-computed scores in `diagnostics/deception_v2/deception_v2b_scores.parquet`)

---

## Per-Consumer Version Map

### Stuff+ (active: v4 via `analytics/stuff_model.py`)

| Consumer | File | Function | Version |
|----------|------|----------|---------|
| Player page plaque | `api/routers/players.py` | `compute_stuff_plus`, `overall_stuff_plus` | v4 |
| Per-pitch-type table | `api/routers/players.py` | `compute_stuff_plus` | v4 |
| Leaderboard | `api/routers/stuff_model_report.py` | `compute_stuff_plus` | v4 |
| Probable pitchers | `api/routers/probable_pitchers.py` | `compute_stuff_plus`, `overall_stuff_plus` | v4 |
| Career page | `api/routers/career.py` | `compute_stuff_plus_if_cached`, `overall_stuff_plus` | v4 |
| Team cards | `api/routers/team_card.py` | `compute_stuff_plus` (if present) | v4 |
| Top performers | `api/routers/top_performers.py` | `compute_stuff_plus` | v4 |
| Similar players | `api/routers/similar_players.py` | `compute_stuff_plus` | v4 |
| Trends viewer | `api/routers/trends.py` | `compute_stuff_plus` | v4 |
| Model report / meta | `api/routers/stuff_model_report.py` | `compute_stuff_plus` | v4 |

### Location+ (active: v3 via `analytics/location_v3_model.py`)

| Consumer | File | Function | Version |
|----------|------|----------|---------|
| Location report | `api/routers/location_report.py` | `location_v3_model.pitcher_breakdown` | v3 |
| Probable pitchers | `api/routers/probable_pitchers.py` | `location_v3_model._raw_predictions` | v3 (fixed) |
| Career page | `api/routers/career.py` | `location_v3_model.raw_predictions_if_cached` | v3 (fixed) |
| Player page (if applicable) | `api/routers/players.py` | — | Check separately |

### Tunneling+ (version info only — no assertions)

| Consumer | File | Function | Version |
|----------|------|----------|---------|
| Tunneling report | `api/routers/tunneling_report.py` | `tunneling_v2_model` | **v2** |
| Tunneling 3D viewer | `api/routers/tunneling_3d.py` | `tunneling_model` (v1) | **v1** (separate; never replaced) |
| Probable pitchers | `api/routers/probable_pitchers.py` | `tunneling_v2_model._raw_predictions` | **v2** (fixed from v3) |
| Career page | `api/routers/career.py` | `tunneling_v2_model.raw_predictions_if_cached` | **v2** (fixed from v3) |

Note: `tunneling_v3_model.py` is now unused by any live route. Can be archived.

### Deception+ (active: v2b, pre-computed scores)

| Consumer | File | Function | Version |
|----------|------|----------|---------|
| Deception report / leaderboard | `api/routers/deception_report.py` | `_load_scores("v2b")` | **v2b** (fixed) |
| Player page plaque | `api/routers/players.py` | `deception_report._pitcher_breakdown_cached` | v2b via parquet |
| Probable pitchers | `api/routers/probable_pitchers.py` | `deception_report._raw_predictions_table` | v2b (fixed) |
| Career page | `api/routers/career.py` | `deception_report._pitcher_breakdown_cached` | v2b via parquet |
| Warmup | `api/warmup.py` | — | Check separately |
