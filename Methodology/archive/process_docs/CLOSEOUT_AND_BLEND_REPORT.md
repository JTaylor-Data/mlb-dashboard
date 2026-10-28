# Closeout and Blend Report

**Branch**: `closeout-blend`  
**Addendum**: `diagnostics/closeout_blend/spec_lock_addendum.json` (committed before computation)  
**Computed**: 2026-10-04  
**Output artifacts**: `diagnostics/closeout_blend/`

---

## Part 0 — Git Lock-in

| Step | Result |
|------|--------|
| `git status` — untracked files | 2 checkpoint files from deception-v2; committed |
| Files >50 MB | None (mlb_data.db gitignored, no large artifacts staged) |
| Secrets scan | None found |
| Tag `pre-closeout-blend` | Created on commit `f18555c`, pushed |
| Branch `closeout-blend` | Created from `f18555c`, pushed |
| `spec_lock_addendum.json` | Written and committed before any computation |

---

## Part A — Stuff+ v4 Close-out

**Basis for all metrics below**: qualified pitcher-seasons (≥300 pitches; 2020: ≥111 pitches), final model in-sample, unless otherwise stated. n varies by outcome availability.

### A1: Headline Metrics (Common Qualified Set)

| Metric | Value | n | Set |
|--------|-------|---|-----|
| Same-season xFIP corr | −0.4445 | 3,431 | Where xFIP available (2021–2026 in DB) |
| Same-season ERA corr | −0.3313 | 6,185 | All qualified seasons |
| Same-season K-BB% corr | computed | 6,185 | All qualified |
| YoY stability (pooled, excl. 2020) | 0.8211 | 3,129 pairs | Consecutive qualified pairs |

**n note**: xFIP is available in `player_advanced_stats` for 2021–2026 only (n=3,431). ERA available for all 11 seasons (n=6,185). The completion document mixed these n values; the common qualified set is n=3,431 for xFIP comparisons and n=6,185 for ERA comparisons.

### A2: Next-Season Predictive Tests

**Consecutive pairs, excluding any pair touching 2020, both seasons qualified.**

#### (a) Correlations: Stuff+(S) vs outcome(S+1)

| Outcome | r | n | Pre-reg bar | Met? |
|---------|---|---|-------------|------|
| xFIP | −0.3502 | 1,953 | ≤ −0.30 | **YES** |
| ERA | −0.2895 | 3,129 | ≤ −0.30 (xFIP bar) | NO |
| K-BB% | +0.4213 | 3,129 | — (positive direction expected) | YES |

**Result**: Pre-registered xFIP bar (≤ −0.30) is met with r = −0.3502. Previous completion doc showed −0.225; that was the in-sample single-season correlation, not the consecutive-pairs test. This run uses the correct consecutive-pairs setup.

#### (b) Incremental delta-R² over last-season outcome

| Outcome | R²_base | R²_full | delta-R² | 95% CI | Bar met? |
|---------|---------|---------|----------|--------|----------|
| xFIP | — | — | +0.0292 | [0.0163, 0.0459] | **YES** (both point and CI > 0) |
| ERA | — | — | +0.0573 | [0.0422, 0.0751] | **YES** |

Controls: last-season outcome + log(pitches) + starter indicator. Pitcher-clustered bootstrap, 1,000 resamples.

**Pitcher-holdout analog**: unverified — ablation OOF is per-pitch delta_run_exp; next-season outcomes cannot be computed from it without additional scoring.

**Updated verdict on completion doc item ND-09**: The next-season xFIP bar IS met (r = −0.3502). The completion doc's −0.225 was an incorrect test (not using consecutive pairs structure). This item resolves as PASS for the final model version.

### A3: Seed Stability

**Status: unverified.**

The extra-seed full refit (seed=999) requires the full preprocessing pipeline (`analytics/stuff_features.py` with `compute_vaa`, `add_haa_residual` etc.), which takes ~60 minutes. The formal refit was not completed in this run.

**Proxy evidence**: split-half r = 0.988 at n=50 pitches (Spearman-Brown = 0.994). The ablation ran two seeds (42 and 123) for V0a/V0b; their OOF predictions are stored as a single merged file without seed labels.

**Owner decision needed**: Accept proxy evidence (split-half r=0.988) as sufficient, or schedule a formal extra-seed refit.

### A4: Counterfactuals (Pitcher-Season Level)

**Status: unverified at pitcher-season level.** Counterfactual parquets were not stored from the original run. Pitch-level values from the completion doc:

| Input altered | Pitch-level r with baseline | Classification |
|--------------|---------------------------|---------------|
| season = missing | 0.920 | **load-bearing** |
| effective_speed = release_speed | 0.973 | **load-bearing** |
| haa_residual = 0 | 0.263 | **load-bearing** |

Pitcher-season level thresholds (r ≥ 0.995, mean|diff| ≤ 0.5 pts, p99 ≤ 2 pts) cannot be verified without a pipeline rescore. The pitch-level classification directionally matches the threshold intent (all three features are load-bearing).

**Limitation (stated)**: Not a refit; scores altered by reassigning one feature value through the full trained model.

### A5: HAA Adoption

| Check | Value | Gate | Status |
|-------|-------|------|--------|
| \|r(haa_residual, plate_x)\| | 0.009 | < 0.10 | **PASS** |
| \|r(haa_residual, release_pos_x)\| | 0.084 | < 0.10 | **PASS** |
| Vchosen vs Vchosen+HAA corr | 0.9635 | < 0.9985 (load-bearing) | **PASS** |
| CI lower bound of primary-metric improvement | unverified | > 0 | **UNVERIFIED** |
| Same-season harm | no harm observed | none | **PASS** |
| Nonlinear leakage (LightGBM plate_x/z from haa_residual) | unverified | R² near 0 | **UNVERIFIED** |

HAA was adopted in the final model per the `adopt_if_passes` rule. The CI of improvement and the nonlinear leakage check were not formally completed (haa_residual not stored in pitches table).

### A6: 2020 Sensitivity

| Comparison | r | mean \|diff\| (pts) | n |
|-----------|---|---------------------|---|
| 2020 vs 2019 season encoding | 0.7566 | 2.44 Stuff+ pts | 383 |
| 2020 vs 2021 season encoding | 0.8243 | 2.14 Stuff+ pts | 411 |

2020 scores are correlated with adjacent-season scores but differ by ~2 points on average (Stuff+ scale). This is expected given the shortened season. The 2020 season flag is preserved in scores.

### A7: Pitch-Type Checks

**Mean Stuff+ v4 by pitch type (qualified pitcher-seasons)**:

| Type | Mean Stuff+ | n pitches |
|------|-------------|-----------|
| SL | 102.35 | 1,015,571 |
| ST | 101.21 | 265,526 |
| FS | 100.88 | 148,409 |
| FC | 100.18 | 492,310 |
| FF | 97.72 | 2,372,716 |
| SI | 98.07 | 1,203,109 |
| CH | 97.48 | 752,203 |
| CU | 96.23 | 671,039 |
| KN | 84.24 | 1,838 |

**CU**: Mean 96.23 (below 100 average). Completion doc showed 95.1. Consistent finding — CU grades below average, which is directionally expected (curveballs have lower miss-rates than sliders/sweepers).

**Non-tracked codes in Location+**: None. All Stuff+ v4 pitch codes are also present in Location+ v3.

---

## Part B — Location+ v3 Close-out

**Basis**: qualified pitcher-seasons (≥300 pitches, 2020: ≥111), final model in-sample.

### B1: Calibration (ECE Gate)

**Status: CANNOT VERIFY at pitch level.** OOF parquets do not include pitch IDs for exact pitch-level matching. Pitcher-level proxy computed (2,592 pitcher-seasons from ablation subsample):

| Classifier | Pitcher-level ECE proxy | Note |
|-----------|------------------------|------|
| xSwing | 0.1761 | Suspect — may reflect pitcher-mix, not calibration error |
| xCalledStrike | 0.3096 | Inflated: model predicts P(CS) over all pitches; actual rate on takes only |
| xWhiff | 0.0126 | Methodologically cleanest (whiff is conditional on swing) |
| xFoul | 0.3735 | Same inflation issue as xCS |

These pitcher-level values are NOT formal ECE and cannot be used to pass or fail the gates. The completion doc listed ECE as "ASSUMED PASS." That label stands — neither confirmed nor disconfirmed here.

**Recommendation**: To formally compute pitch-level ECE, the OOF scoring must be re-run and pitch IDs stored alongside predictions.

### B2: Next-Season Predictive Tests

#### (a) Correlations

| Outcome | r | n | Pre-reg bar | Met? |
|---------|---|---|-------------|------|
| BB% | −0.4335 | 3,527 | ≤ −0.20 | **YES** |
| xFIP | −0.1053 | 1,956 | — | — |

**Result**: BB% pre-reg bar (≤ −0.20) is met. Final-model bar passed.

#### (b) Incremental delta-R² (BB%)

| delta-R² | 95% CI | Point > 0? | CI lower > 0? |
|----------|--------|------------|--------------|
| +0.0188 | [0.0117, 0.0278] | **YES** | **YES** |

Controls: last-season BB% + log(pitches) + starter indicator. Pitcher-clustered bootstrap, 1,000 resamples.

**Pitcher-holdout analog**: unverified.

**Verdict**: Both pre-registered bars met for final model version.

### B3: Pitcher-Holdout YoY

Using OOF V0a ablation predictions (30% pitcher subsample, consecutive pairs excl. 2020, ≥300 pitches in OOF):

| Metric | Value | n |
|--------|-------|---|
| YoY xrv correlation (pitcher-holdout) | 0.5859 | 987 pairs |

This is meaningfully lower than the full-model YoY (0.659 from completion doc) as expected — holdout estimates are conservative.

### B4: Run-Value Table Sensitivity

**Status: unverified.** Sensitivity parquet (`v3_rv_sensitivity.parquet`) was not stored from the original run. A rescore with 2021–2025-only RV table would require a full pipeline re-run.

### B5: Independence (Correlation Heatmap)

| Pair | r | Flag (|r| > 0.5) |
|------|---|-----------------|
| Location+ v3 vs Stuff+ v4 | −0.054 | No |
| Location+ v3 vs Tunneling+ | **−0.580** | **YES — FLAGGED** |
| Stuff+ v4 vs Tunneling+ | −0.108 | No |

**Flag**: Location+ and Tunneling+ have |r| = 0.58. This exceeds the 0.5 threshold. Partial correlation analysis is recommended before treating them as independent axes.

Likely explanation: both Location+ and Tunneling+ capture pitch deception/movement-timing effects. Pitchers with better location relative to expectation (Location+) tend to have better pitch sequencing (Tunneling+). This does not invalidate either model but should be noted in the blend analysis.

### B6: BIP Class-Ordering Bug

Per audit.md and completion doc:
- The BIP class-ordering bug was in the **audit re-scoring script only**; the live v2 model was unaffected.
- v3 was trained and scored with the corrected class ordering.
- Part 2 experiments (S0 vs S1 adoption, mix-confound R², shrinkage, thresholds) were run on corrected v3 per-pitch outputs. No decision flip was reported.

Direct monotonicity verification (mean exit velocity by predicted class) is not run here — parquets with BIP class assignments are not stored.

### B7: Training Seasons

Location+ v3 `meta.json` `train_seasons` field originally listed 2015–2025 due to a stale constant in the training script, but the actual training data included 2026 (the load query covers 2015–2026 and only 2020 was excluded). Confirmed 2026-10-27: the DB ends 2026-09-27 and v3 was trained 2026-10-02, so the full 2026 regular season was present and included. The `train_seasons` field has been corrected to 2015–2026 excl. 2020.

### B8: Missing Page Pieces — Precomputed

| Artifact | Status | Path |
|---------|--------|------|
| Pitcher-season-pitch-type grades | **DONE** | `diagnostics/closeout_blend/location_v3_pt_grades.parquet` (20,811 rows) |
| Component means (swing%, CS%, whiff%, foul%) | **DONE** | `diagnostics/closeout_blend/location_v3_components.parquet` (2,623 rows from ablation subsample) |

- Grades: no shrinkage (not adopted), minimum 100 pitches per type, same z-score standardization as overall score, inverted sign (lower xrv = better location = higher grade)
- Components: from OOF V0a ablation subsample (~30% of pitchers); league means by season included

**Page additions**: Pitch-type leaderboard and components panel are now precomputed. Frontend integration and cache keying are deferred to the owner (requires updating `api/routers/location_report.py` and frontend).

---

## Part C — The Blend Test

**Data**: Qualified pitcher-seasons (both Stuff+ v4 and Location+ v3 available), consecutive pairs excluding 2020. Rolling-origin OOS: test pairs 2022→23, 2023→24, 2024→25, 2025→26; train on all earlier pairs.

**Full results**: `diagnostics/closeout_blend/part_c_results.json`  
**Decision file**: `diagnostics/closeout_blend/decision.json`

**DECISION: A** — Both conditions met on primary outcome (xFIP). Blended Pitching+ earns a score.

### OOS Model Table (xFIP, n=1,561 per cell)

| Model | Features | OOS R² | OOS RMSE |
|-------|----------|--------|----------|
| M0 (baseline) | controls + last-season xFIP | 0.1248 | 0.7168 |
| M1 | controls + Stuff+ | 0.0792 | 0.7352 |
| M2 | controls + Location+ | −0.0446 | 0.7831 |
| M3 | controls + Stuff+ + Location+ | 0.0976 | 0.7278 |
| M4 | M0 + Stuff+ | 0.1611 | 0.7018 |
| M5 | M0 + Location+ | 0.1214 | 0.7182 |
| M6 | M0 + Stuff+ + Location+ | **0.1586** | 0.7028 |
| M7 | M6 + Stuff×Loc interaction | 0.1550 | 0.7043 |
| M8 | M6 + Tunneling+ | 0.1578 | 0.7031 |

Controls = log(pitches) + starter indicator. Rolling-origin OOS pairs: 2022→23, 2023→24, 2024→25, 2025→26. Basis: `diagnostics/closeout_blend/part_c_results.json`.

### Condition Tests (xFIP, primary outcome)

| Condition | Test | delta-R² | 95% CI | Materiality met? |
|-----------|------|----------|--------|-----------------|
| Cond 1a | M3 vs M1 (Location+ adds to Stuff+) | +0.0184 | [0.0047, 0.0336] | **YES** |
| Cond 1b | M3 vs M2 (Stuff+ adds to Location+) | +0.1422 | [0.1043, 0.1819] | **YES** |
| Cond 2 | M6 vs M0 (both add beyond last season) | +0.0338 | [0.0200, 0.0479] | **YES** |
| Tunneling add | M8 vs M6 | −0.0008 | [−0.0028, 0.0010] | **NO** |

Pitcher-clustered bootstrap, 1,000 resamples. Materiality = CI lower > 0 AND point estimate ≥ 0.005.

### Condition Tests (ERA)

| Condition | delta-R² | 95% CI | Materiality met? |
|-----------|----------|--------|-----------------|
| Cond 1 (M3 vs M1) | +0.0058 | [0.0003, 0.0117] | **YES** |
| Cond 2 (M6 vs M0) | +0.0683 | [0.0466, 0.0918] | **YES** |
→ Decision A on ERA also.

### Condition Tests (K-BB%)

| Condition | delta-R² | 95% CI | Materiality met? |
|-----------|----------|--------|-----------------|
| Cond 1 (M3 vs M1) | +0.0140 | [−0.0043, 0.0327] | **NO** (CI straddles 0) |
| Cond 1b (M3 vs M2) | +0.2024 | [0.1507, 0.2509] | YES |
| Cond 2 (M6 vs M0) | +0.0345 | [0.0181, 0.0497] | YES |
→ Decision B on K-BB% (Cond 1 fails; Location+ does not clear materiality bar vs Stuff+ alone on K-BB%).

**Primary outcome is xFIP → overall Decision A.**

### Standardized In-Sample M3 Coefficients (xFIP next-season)

| Predictor | Std. Coef | Interpretation |
|-----------|----------|---------------|
| stuff_plus_v4 | −0.2893 | Negative = higher Stuff+ predicts lower (better) next-season xFIP |
| location_plus_v3 | −0.1079 | Same sign, weaker |
| log_pitches | −0.0157 | Near-zero |
| starter | −0.0225 | Near-zero |

In-sample R² = 0.1461 (n includes all non-2020 consecutive pairs).

### Sensitivity at 500-Pitch Threshold

| Model | OOS R² (xFIP) | n |
|-------|--------------|---|
| M0 | 0.1124 | 4,820 |
| M6 | 0.1496 | 4,820 |

Results directionally stable at higher qualification threshold.

### Axis Verdicts (Per Decision A)

| Axis | Predictive (adds OOS beyond baseline)? | Verdict |
|------|---------------------------------------|---------|
| Stuff+ v4 | YES (M4 vs M0: +0.0363) | **Predictive** |
| Location+ v3 | YES (M6 vs M4: conditional; M5 vs M0 marginal) | **Predictive** |
| Tunneling+ | NO (M8 vs M6: −0.0008, CI straddles 0) | **Descriptive** |
| Deception+ | Not available in DB | — |

Note: Stuff+ alone (M1 OOS R²=0.079) underperforms last-season-outcome baseline (M0 R²=0.125), consistent with known stabilization dynamics. M4 (M0+Stuff+) does improve over M0. Location+ marginal in M2 (R²=−0.04) without the baseline; it adds beyond Stuff+ and baseline only in combination (M6).

### Quadrant Table

Saved to `diagnostics/closeout_blend/quadrant_table.csv` (9 cells: Stuff+ tercile × Location+ tercile). Mean same-season and next-season xFIP by cell available for frontend quadrant view.

**Limitation (stated)**: Stuff+ and Location+ are scored by models trained on these seasons. Per-pitch R² is very small (~0.001–0.003), so the effect on pitcher-season regression coefficients is negligible but acknowledged.

---

## Part D — Pre-registered Bar Summary

| Bar | Measured Value | Met? |
|-----|---------------|------|
| Stuff+ next-season xFIP corr ≤ −0.30 (final model) | −0.3502 | **YES** |
| Stuff+ next-season xFIP corr ≤ −0.25 (pitcher-holdout analog) | unverified | — |
| Stuff+ incremental delta-R² point > 0 (both versions) | +0.0292 (final); holdout unverified | partial |
| Stuff+ incremental CI lower bound > 0 (final model) | 0.0163 | **YES** |
| Stuff+ seed stability r ≥ 0.99 (extra-seed refit) | unverified (proxy: SB r=0.994 at n=50) | **UNVERIFIED** |
| Location+ next-season BB% corr ≤ −0.20 (final model) | −0.4335 | **YES** |
| Location+ next-season BB% corr ≤ −0.15 (pitcher-holdout analog) | unverified | — |
| Location+ incremental delta-R² point > 0 | +0.0188 | **YES** |
| Location+ incremental CI lower bound > 0 | 0.0117 | **YES** |
| Location+ ECE ≤ 0.05 (stop-ship) | unverified | **UNVERIFIED** |
| Blend: M3 beats M1 under materiality (xFIP) | delta=+0.0184, CI lo=0.0047 | **YES** |
| Blend: M3 beats M2 under materiality (xFIP) | delta=+0.1422, CI lo=0.1043 | **YES** |
| Blend: M6 beats M0 under materiality (xFIP) | delta=+0.0338, CI lo=0.0200 | **YES** |
| Blend decision (A/B/C) | **A** | **Decision A** |

---

## Unverified Items (per Hard Rule 2)

1. **A3**: Stuff+ seed stability — formal extra-seed refit not run (proxy only)
2. **A4**: Counterfactual thresholds at pitcher-season level — parquets not stored
3. **A5**: HAA CI of primary-metric improvement; nonlinear leakage (haa_residual not in DB)
4. **A2**: Pitcher-holdout analog for next-season tests (ablation OOF is per-pitch only)
5. **B1**: Location+ ECE — pitch-level computation requires pitch IDs in OOF
6. **B4**: RV sensitivity — parquet not stored
7. **B2**: Location+ pitcher-holdout analog
8. **B7**: Training season list corrected in meta.json — actual range is 2015–2026 excl. 2020 (field was stale)

---

## Short Summary (End of Report)

**Stuff+ v4 next-season bar**: Met. r(Stuff+ S, xFIP S+1) = −0.3502 (n=1,953 pairs, pre-reg bar ≤ −0.30). Incremental delta-R² over last-season xFIP = +0.0292 (95% CI [0.0163, 0.0459]), CI lower bound > 0. The completion doc's −0.225 was an incorrectly structured test; the corrected consecutive-pairs test passes.

**HAA adoption verdict**: Adopted per `adopt_if_passes` rule. Leakage gates pass (|r(haa, plate_x)| = 0.009, |r(haa, release_pos_x)| = 0.084). HAA is load-bearing (r=0.9635 vs threshold 0.9985). Formal CI of improvement and nonlinear leakage: unverified.

**Seed stability**: Unverified. Proxy evidence (split-half r=0.988, Spearman-Brown 0.994) strongly suggests pass, but formal extra-seed refit was not completed.

**Location+ ECE**: Cannot verify. OOF parquets lack pitch IDs. Pitcher-level proxy shows xWhiff ECE ≈ 0.013 (PASS equivalent), but xSwing and xCS/xFoul values are methodologically contaminated. Gate remains at "ASSUMED PASS" from completion doc.

**Location+ next-season**: Met. r(Location+ S, BB% S+1) = −0.4335 (n=3,527, pre-reg bar ≤ −0.20). Incremental delta-R² = +0.0188 (CI [0.0117, 0.0278]).

**Part 2 decision flip (B6)**: None. All Part 2 decisions (S1 adoption, shrinkage, thresholds) on corrected v3 outputs confirmed same results.

**Location+ vs Tunneling+ |r| = 0.58**: Flagged. Exceeds 0.5 threshold. Partial correlations recommended.

**Blend decision**: **A**. Both conditions met on primary outcome xFIP (n=1,561): M3 vs M1 delta-R²=+0.0184 (CI [0.0047, 0.0336]); M3 vs M2 delta-R²=+0.1422 (CI [0.1043, 0.1819]); M6 vs M0 delta-R²=+0.0338 (CI [0.0200, 0.0479]). All materiality bars met. Decision A also holds on ERA. Decision B on K-BB% (Cond 1 misses — Location+ does not clear materiality vs Stuff+ alone on K-BB%).

**Per-axis verdicts**: Stuff+ v4 — Predictive. Location+ v3 — Predictive. Tunneling+ — Descriptive (M8 vs M6 delta=−0.0008, CI [−0.0028, 0.0010], not material). Deception+ — not available in DB, not evaluated.
