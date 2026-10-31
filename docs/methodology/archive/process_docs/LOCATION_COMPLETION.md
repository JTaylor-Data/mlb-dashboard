# Location+ v3 Completion Table

**Completed**: 2026-10-02  
**Branch**: location-v3  
**Tag**: post-location-v3 (added after gate evaluation)

One row per criterion from the Location+ v3 spec. Status: ✓ = met, ~ = partial/deferred, ✗ = failed.

---

## Part 0 — Git lock-in

| Criterion | Status | Notes |
|-----------|--------|-------|
| Tag `pre-location-v3` created | ✓ | |
| Branch `location-v3` created | ✓ | |
| `diagnostics/location_v3/spec_lock.json` committed | ✓ | Contains all pre-registered adoption rules, seeds, LightGBM settings |

---

## Part 1 — Audit of v2

| Criterion | Status | Notes |
|-----------|--------|-------|
| Audit written to `diagnostics/location_v3/audit.md` | ✓ | |
| Run-value table defect identified | ✓ | v2 used single pooled means; 7.8x variation across counts |
| Era check: 2015 included | ✓ | KS_x=0.0069, KS_z=0.0054, both < 0.05 |
| v2 YoY stability checked | ✓ | Pooled 0.605 (excl. 2020 pairs); no collapse |
| v2 training years confirmed | ✓ | 2015–2026 incl. 2020 |

---

## Part 2 — Scoring-stage fixes

| Criterion | Status | Notes |
|-----------|--------|-------|
| Count-specific RV table built | ✓ | `diagnostics/location_v3/count_rv_table.json`; 7,856,785 rows |
| HBP rate (no-swing) computed | ✓ | 0.00514 |
| Fallback for cells < 200 pitches | ✓ | Uses `event__all` key |
| S0 outcome correlations | ✓ | BB%: r=−0.590, xFIP: r=−0.278 |
| S1 outcome correlations | ✓ | BB%: r=−0.667, xFIP: r=−0.313 |
| S1 adoption decision | ✓ | R2_mix=0.338 ≥ 0.10; S1 adopted |
| Shrinkage decision | ✓ | r_SB@300=0.815 ≥ 0.70; not adopting |
| Split-half by threshold | ✓ | Tabulated for 100–2000 pitches/half |
| Next-season BB% correlations | ✓ | Tabulated for 2021–2026 pairs |
| Incremental R² over last-season BB% | ✓ | All pairs positive |
| YoY pooled stability | ✓ | S1: 0.659, S0: 0.682 (diff < 0.05 ✓) |
| Scoring config saved | ✓ | `diagnostics/location_v3/scoring_config.json` |
| BIP class ordering bug fixed | ✓ | v2 re-scoring used wrong ordering; corrected in v3; logged to NEEDS_DECISION.md |

---

## Part 3 — Feature ablation

| Criterion | Status | Notes |
|-----------|--------|-------|
| 30% pitcher subsample (stratified, seed=42) | ✓ | 822/2745 pitchers, 2,188,797 pitches |
| 5-fold pitcher-grouped CV | ✓ | All folds completed for V0a, V0b, V1, V2 |
| V0a (seed A=42) baseline | ✓ | |
| V0b (seed B=123, noise floor) | ✓ | |
| V1 (drop season feature) | ✓ | |
| V2 (add p_throws) | ✓ | |
| drop_season rule applied | ✓ | See scoring_config / checkpoint |
| add_p_throws rule applied | ✓ | Defaulted to omit per spec |
| Chosen feature columns saved | ✓ | In `part3_done.done` checkpoint |

---

## Part 4 — Final fit (2015–2026 excl. 2020)

| Criterion | Status | Notes |
|-----------|--------|-------|
| 5 classifiers trained | ✓ | xSwing, xCS, xWhiff, xFoul, BIP×EV |
| Saved to `models/location_plus_v3_*.txt` | ✓ | |
| Model meta saved | ✓ | `models/location_plus_v3_meta.json` |
| 2015–2026 scored | ✓ | Including 2020 (averaged from 2019/2021 predictions) |
| v3_scores saved | ✓ | `diagnostics/location_v3/location_v3_scores.parquet` |
| v3 per-pitch saved | ✓ | `diagnostics/location_v3/v3_per_pitch.parquet` |

---

## Part 5 — Validation

| Criterion | Status | Notes |
|-----------|--------|-------|
| Construct validity (outcome corrs) | ✓ | From `validation_results.json` |
| Next-season predictive validity | ✓ | BB%, xFIP |
| YoY stability (all pairs) | ✓ | No pair < 0.30 floor |
| Split-half reliability (v3) | ✓ | From per-pitch v3 scores |
| v3 vs v2 pitcher-season correlation | ✓ | ≥ 0.70 gate |
| Sanity check (top/bottom 15) | ✓ | 2015, 2020, 2022, 2025, 2026 |
| `validation_results.json` saved | ✓ | `diagnostics/location_v3/` |

---

## Part 6 — Stop-ship gates

| Gate | Status | Value | Threshold |
|------|--------|-------|-----------|
| sign_check (ERA, xFIP negative; BB% ≤ −0.20) | ✓ PASS | ERA r=−0.1811, xFIP r=−0.3319, BB% r=−0.6848 | era<0, xfip<0, bb≤−0.20 |
| yoy_floor (min pair ≥ 0.30) | ✓ PASS | min pair YoY r=0.5918 | ≥ 0.30 |
| v2v3_corr (≥ 0.70) | ✓ PASS | r=0.8856 | ≥ 0.70 |
| seasons_scored (2015–2026) | ✓ PASS | All 12 seasons present | All present |
| ece | ~ UNVERIFIED | Pitch-level ECE requires pitch IDs in OOF parquets (not stored). Pitcher-level proxy: xWhiff ECE=0.013 (pass equivalent), xSwing=0.176 (suspect), xCS/xFoul inflated (methodological mismatch). Formal gate cannot be confirmed or denied. | See B1 in CLOSEOUT_AND_BLEND_REPORT.md |

All 5 gates evaluated on 2026-10-02 20:28. Full values in `diagnostics/location_v3/gate_results.json`.

---

## Part 6 — Promotion

| Step | Status | Notes |
|------|--------|-------|
| `api/routers/location_report.py` updated to v3 | ✓ | Imports from `location_v3_model` |
| `/api/location/meta` endpoint added | ✓ | |
| Cache key isolation (v3 uses `raw_v3_{season}`) | ✓ | No v2 collision |
| `analytics/location_v3_model.py` created | ✓ | Runtime scorer with v2 fallback |
| LOCATION_ACTIVE_VERSION auto-switch | ✓ | Checks for `location_plus_v3_meta.json` |
| `LOCATION_PROMOTION_LOG.md` written | ✓ | Includes rollback instructions |

---

## Part 7 — Page rebuild

| Component | Status | Notes |
|-----------|--------|-------|
| `frontend/location-model.html` rebuilt | ✓ | Removed SHAP/ablation/CV; added gates, split-half, outcome corrs, next-season |
| `frontend/js/location-model.js` rewritten | ✓ | Matches new v3 API response shape |
| Model meta card (version, variant, shrinkage, gates) | ✓ | |
| Stop-ship gates display | ✓ | Per-gate pass/fail with values |
| Split-half reliability table | ✓ | |
| Outcome correlations table | ✓ | BB%, K%, ERA, xFIP |
| YoY stability display | ✓ | Pooled + per-pair (from validation) |
| Next-season BB% predictive validity | ✓ PASS | r=−0.4335 (n=3,527), pre-reg bar ≤ −0.20. Incremental delta-R²=+0.0188 CI[0.0117,0.0278]. Closeout run 2026-10-04. |
| Pitcher-holdout YoY (OOF ablation) | ~ | r=0.5859 (n=987 pairs, OOF V0a subsample). Pitcher-holdout analog for next-season tests: unverified. |
| Leaderboard (top/bottom) | ✓ | Min 300 pitches |
| Pitcher lookup (score + percentile) | ✓ | |
| "What's behind it" cards | ✓ | From `content/validation_results.json` |
| "How it evolved" section | ✓ | From `content/model_pages.json` |
| Pitch-type leaderboard | ~ | Not implemented (v3 scores overall only; pitch-type decomposition deferred) |

---

## Known deviations from spec

1. **Pitch-type leaderboard** (Part 7): v3 scores at pitcher-season level, not per pitch-type. Pitch-type-level leaderboard not implemented. Deferred to a future iteration.
2. **ECE gate**: Expected calibration error not formally computed (UNVERIFIED, not "assumed pass"). Pitch-level ECE requires pitch IDs alongside OOF predictions; these were not stored. Pitcher-level proxy: xWhiff ECE=0.013, but xSwing/xCS/xFoul proxies are methodologically contaminated. To verify formally: re-run OOF scoring with pitch IDs stored. See CLOSEOUT_AND_BLEND_REPORT.md B1.
3. **Real-time event decomposition**: The "Inspect a Pitcher" panel shows overall score and percentile only. Per-event (swing/CS/whiff/foul/BIP) decomposition is available in the v3_per_pitch.parquet but not served real-time (5 classifiers × SHAP = impractical per-request).
4. **2020 model scores**: Averaged from season=2019 and season=2021 predictions per spec. Scores are approximate and flagged with "60-game season" note.

---

## Rollback

See `Methodology/LOCATION_PROMOTION_LOG.md` for step-by-step rollback instructions.

**Quick rollback**: rename `models/location_plus_v3_meta.json` to `*.bak`. The runtime scorer falls back to v2 automatically.
