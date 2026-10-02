# Location+ v2 Audit
**Date**: 2026-10-02  
**Branch**: location-v3  
**Script**: `training/train_v2_models.py`, `analytics/location_v2_model.py`

---

## 1. Model Architecture

**Boosters**: Five LightGBM classifiers
- `location_plus_v2_xswing.txt` — P(swing | pitch features)
- `location_plus_v2_xcs.txt` — P(called_strike | no_swing, pitch features)
- `location_plus_v2_xwhiff.txt` — P(whiff | swing, pitch features)
- `location_plus_v2_xfoul.txt` — P(foul | contact, pitch features)
- `location_plus_v2_bip.txt` — P(BIP type × EV bucket | BIP, pitch features)

**Training seasons**: 2015–2026 (including 2020)
> v3 CHANGE: Exclude 2020 from training per spec; score 2020 by averaging 2019/2021 predictions.

**Features**: `plate_x`, `plate_z`, `balls`, `strikes`, `pitch_type`, `stand`, `season`

**Scored seasons**: 2015–2026 from `mlb_data.db`

**Qualification threshold**: 100 pitches (MIN_PITCHES in scoring code)
> v3 CHANGE: Raise to 300 pitches per spec (2020: ~111 pitches scaled)

**Shrinkage**: None applied in v2

**Standardization**: Per-season z-score mapped to plus-scale (100 = league mean, 10 points = 1 SD)

**New season handling**: season passed directly as categorical; unseen 2026 training season included

---

## 2. Run-Value Table — **DEFECT CONFIRMED**

**Source**: `training/v2_artifacts/run_value_table.json`  
**Construction**: Context-neutral pooled mean `delta_run_exp` per event type (2015–2026)  
**Bug**: Table is NOT count-specific — single pooled values for whiff, foul, called_strike, ball, HBP

**Magnitude of error**: Count-specific run values vary substantially by count:
- called_strike at 0-0: -0.042, at 3-2: -0.328 (ratio ~7.8x)
- ball at 0-0: +0.037, at 3-2: +0.286 (ratio ~7.7x)

**Action**: Build count-specific run-value table in v3 pipeline. Defect correction, not optimization.

**v3 RV table specification**: For each (terminal event, balls, strikes): mean `delta_run_exp` pooled over 2015–2026 excl. 2020. Fallback to all-count mean when cell <200 pitches. BIP outcomes broken down by (bb_type × EV bucket × balls × strikes) where n ≥ 200, else fall back to (bb_type × EV bucket) mean.

---

## 3. Classifier Training Populations — **CORRECT**

Verified by reading `training/train_v2_models.py`:
- **xSwing**: all pitches ✓
- **xCalledStrike**: no-swing pitches only ✓
- **xWhiff**: swing pitches only ✓
- **xFoul**: contact pitches only ✓
- **BIP classifier**: balls-in-play only ✓

No mismatch bugs found.

---

## 4. Era and Zone Drift

**2015 vs 2016 KS statistics** (ERA CHECK):
- plate_x: KS stat = 0.0069 (< 0.05 threshold → **include 2015 in training**)
- plate_z: KS stat = 0.0054 (< 0.05 threshold → **include 2015 in training**)

**Conclusion**: 2015 passes era check; include in v3 training.

---

## 5. Pitch Codes

Codes in DB (2015–2026): FF, SI, SL, CH, CU, FC, ST, KC, FS, SV, KN, FA, IN, EP, FO, CS, PO, SC, UN, NaN

**v2 exclusions applied**: PO, UN, EP; KC remapped to CU; NaN excluded  
**v3 will apply same exclusions** plus require non-null plate_x/plate_z

---

## 6. YoY Stability Analysis

Per-consecutive-pair YoY (≥100 pitches, excl. 2020 pairs):

| Pair      | r     | n   |
|-----------|-------|-----|
| 2015-2016 | 0.633 | 468 |
| 2016-2017 | 0.607 | 488 |
| 2017-2018 | 0.591 | 497 |
| 2018-2019 | 0.663 | 515 |
| 2021-2022 | 0.578 | 516 |
| 2022-2023 | 0.625 | 517 |
| 2023-2024 | 0.572 | 507 |
| 2024-2025 | 0.584 | 518 |
| 2025-2026 | 0.586 | 510 |
| **Pooled** | **0.605** | — |

No post-2023 collapse (unlike v1). v1 had a collapse to ~0.009 by 2024-2025.

**Reconciliation of reported figures**:
- Old page "0.361" was from v1 (which suffered sign-flip and stability collapse)
- Cross-model validation V0 reported 0.495 (different population: ≥100 pitch threshold with different pairing logic)
- Current computation (above): 0.605 pooled excluding 2020 pairs (at ≥100 pitch threshold)

At ≥300 pitch threshold (v3 standard), pooled YoY expected to be similar or slightly higher.

---

## 7. v2 Classifier Performance (from meta.json)

Mean held-out AUC by classifier:
- xSwing: 0.850
- xCalledStrike: 0.982
- xWhiff: 0.753
- xFoul: 0.622
- BIP: multiclass, mean acc ≈ 0.237

---

## 8. Bugs Found

**BUG-001 (CONFIRMED)**: Run-value table is not count-specific.  
- Location: `training/train_v2_models.py` PART 0, `training/v2_artifacts/run_value_table.json`
- Effect: xRV values under-weight the count context (3-2 called strike ~7.8x more valuable than pooled)
- Fix: Build count-specific table in v3 pipeline

No other bugs found in classifier training populations.

---

## 9. Saving Per-Pitch Outputs

Per-pitch outputs (v2 boosters) are saved to `training/v2_artifacts/location_plus_v2_per_pitch.parquet` from prior training run. This will be used for Part 2 scoring variants without refitting.

---

## 10. Summary for v3

Changes relative to v2:
1. Count-specific run-value table (defect correction)
2. Exclude 2020 from training; score 2020 by averaging 2019/2021 predictions
3. S1 adjustment (baseline subtraction) — evaluate per spec rule
4. Empirical-Bayes shrinkage — evaluate per spec rule
5. Qualification threshold raised to 300 pitches
6. Possible: drop `season` feature (if ablation criteria met)
7. Possible: add `p_throws` (if ablation criteria met, default omit)
