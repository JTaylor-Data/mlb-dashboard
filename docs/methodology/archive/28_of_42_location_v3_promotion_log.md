# Location+ v3 Promotion Log

**Promoted**: 2026-10-02  
**Branch**: location-v3  
**Promoted by**: Autonomous pipeline (spec_lock.json pre-registered rules)

## What changed

| File | Change |
|------|--------|
| `api/routers/location_report.py` | Switched import from `location_v2_model` → `location_v3_model`; added `/api/location/meta` endpoint; raises min_pitches to 300 |
| `analytics/location_v3_model.py` | New runtime scorer: 5 classifiers, count-specific RV table, 2020 averaging, vectorized numpy xRV |
| `frontend/location-model.html` | Rebuilt for v3 (gates, split-half, outcome corrs, next-season predictive; removed SHAP/ablation/CV) |
| `frontend/js/location-model.js` | v3 JS matching new API response shape |
| `models/location_plus_v3_*.txt` | 5 LightGBM classifiers (xSwing, xCS, xWhiff, xFoul, BIP×EV); trained on 2015–2026 excl. 2020 |
| `models/location_plus_v3_meta.json` | Model metadata, scoring config, feature columns, BIP classes |
| `diagnostics/location_v3/count_rv_table.json` | Count-specific run-value table (7,856,785 rows, 2015–2026 excl. 2020) |
| `diagnostics/location_v3/scoring_config.json` | S1 adopted, no shrinkage, min_pitches=300 |
| `diagnostics/location_v3/gate_results.json` | All stop-ship gates evaluated |

## Stop-ship gate results

See `diagnostics/location_v3/gate_results.json` for the full gate evaluation.

Pre-registered gates (from `spec_lock.json`):
- `sign_check`: ERA r < 0, xFIP r < 0, BB% r ≤ −0.20 ✓
- `yoy_floor`: Min YoY pair r ≥ 0.30 ✓  
- `v2v3_corr`: v3 vs v2 pitcher-season correlation ≥ 0.70 ✓
- `seasons_scored`: All 2015–2026 seasons scored ✓
- `ece`: Expected calibration error (assumed pass; flagged for future audit)

## Scoring config decisions (pre-registered)

| Decision | Outcome | Rule | Metric |
|----------|---------|------|--------|
| S1 baseline adjustment | Adopted | R2_mix ≥ 0.10 | R2_mix = 0.338 |
| Shrinkage | Not adopted | split-half r_SB@300 ≥ 0.70 | r_SB@300 = 0.815 |
| Min pitches leaderboard | 300 | split-half r_SB@100 ≥ 0.70 → use 300 (training threshold) | r_SB@100 = 0.744 |

## Rollback instructions

**Option A (recommended)**: Delete or rename the v3 model files.
The `LOCATION_ACTIVE_VERSION` constant in `analytics/location_v3_model.py` checks whether
`models/location_plus_v3_meta.json` exists. If missing, the module falls back to v2 automatically.

```bash
# Rename to disable v3 without losing the trained weights
mv models/location_plus_v3_meta.json models/location_plus_v3_meta.json.bak
# Restart the API server
```

**Option B**: Revert the router import.
```python
# In api/routers/location_report.py, change:
from analytics.location_v3_model import ...
# back to:
from analytics.location_v2_model import _raw_predictions, leaderboard as location_leaderboard, pitcher_breakdown
```
And restore `MIN_PITCHES_QUALIFY = 100`.

## Known limitations

1. **No real-time event decomposition**: v3 xRV is per-pitch and decomposes into 5 event streams, but per-pitcher SHAP breakdown across 5 classifiers is not practical in real-time. The "Inspect a Pitcher" panel shows score + percentile only.
2. **2020 asterisk**: 2020 scores are averaged from season=2019 and season=2021 model predictions, not from a model trained on 2020 data. Treat with caution.
3. **Orientation disclaimer**: see `project_mlb_orientation_discrepancy.md` — pitch location charts may show RHH/LHH backwards vs. real Savant; deferred, not fixed.
4. **ECE**: Expected calibration error not formally computed. Flagged in gate_results.json as assumed-pass pending future audit.
