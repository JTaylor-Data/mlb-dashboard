# Deception+ v2b Completion Table

Generated: 2026-10-04 15:08:40

**Status: limited_evidence** | **Label: limited_evidence**

Model: **v2b** -- pitcher-season form, features: release_pos_x_as, release_pos_z, release_extension, arm_angle
Change from v2: season dropped per pre-registered inconclusive_means_drop rule (diff=-0.0012, |diff|<0.03 margin).
No unseen-season handling needed (season is not a feature).


## 1. Audit Findings (from v2 post-run audit)

| Finding | Value | Note |
|---------|-------|------|
| v1 per-request scoring | yes | removed in v2 -- all scores pre-computed |
| v1 split-half r | 0.998 | artifact: consistency features computed pre-split |
| arm_angle R2 from mechanics | 0.789 | < 0.90 threshold -> KEEP |
| Identifiability top-1/chance | 146x | very high; identity-prone features dropped |
| arm_angle coverage | 2021-2026 | 0% before 2020, partial 2026 (75.9%) |


## 2. Configuration Tests -- Qualified Common Set (n>=300 pitches)

> CORRECTION from original part2_results.json: A0/A1/A2 were previously evaluated on n=5,118
> (all pitcher-seasons including unqualified), while B was evaluated on n=3,363 (qualified only).
> All figures below are restricted to the qualified common set (n=3,363).

| Config | xFIP r | 95% CI | n |
|--------|--------|--------|---|
| A0 (v1 baseline) | -0.0061 | [-0.0508, 0.0335] | 3363 |
| A1 (regularised F0) | 0.0228 | [-0.0175, 0.0620] | 3363 |
| A2 (per-pitch F2) | 0.0352 | [-0.0090, 0.0817] | 3363 |
| B (pitcher-season F2) | 0.0837 | [0.0370, 0.1279] | 3363 |

| Test | Comparison (qualified set) | Rule | Decision | Changed? |
|------|---------------------------|------|----------|----------|
| Test 1: identity-prone | A1-A2 diff=-0.0124 CI=[-0.0563,0.0281] | drop_unless_significantly_worse | A2_chosen_equivalent_or_better | no |
| Form rule | A2-B diff=-0.0485 CI=[-0.0979,0.0035] | choose_M2_unless_M1_significantly_better | B_chosen_form_rule | no |
| arm_angle screening | R2=0.789 < 0.90 | R2_gte_0.90_AND_removal_shown_equivalent | KEEP | no |
| extension screening | diff=+0.0334 > 0.03 | removal_shown_equivalent | KEEP | no |
| season screening | diff=-0.0012 (inconclusive) | inconclusive_means_drop | DROP | no (recorded but not enforced in v2; now enforced in v2b) |


## 3. Correctness Gates

| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| (a) Same-season xFIP/ERA < 0 | xFIP=-0.0883, ERA=-0.0629 | both < 0 | PASS |
| (b) Pooled YoY >= 0.30, min pair >= 0.25 | pooled=0.7019, min=0.6712 | 0.30 / 0.25 | PASS |
| (c) Hygiene: mechanics/body only | ['release_pos_x_as', 'release_pos_z', 'release_extension', 'arm_angle'] | no forbidden features | PASS |
| (d) Repeat correlation >= 0.98 | min_r=0.9899, avg_r=0.9906 | 0.98 | PASS |
| (e) All intended seasons scored | missing=[] | none | PASS |
| (f) No per-request scoring | parquet lookup only | no per-request | PASS |

All gates PASS.


## 4. Evidence Bar

| Item | Value | Threshold | Status |
|------|-------|-----------|--------|
| (i) Next-season xFIP corr <= -0.20 | avg r=-0.0823 | <= -0.20 | FAIL -- bar not met |
| (ii) Incremental dR2 CI lower > 0 | avg dR2=0.00254, pooled CI lo=1.49e-06 | CI lower > 0 | PASS |

Incremental dR2 per pair (predicting next-season xFIP; model = last-season xFIP + Deception+ vs last-season xFIP alone):

| Pair | dR2 | CI lo | CI hi | n |
|------|-----|-------|-------|---|
| 2021->2022 | 0.00596 | 0.00007 | 0.02588 | 470 |
| 2022->2023 | 0.00052 | 0.00000 | 0.00844 | 474 |
| 2023->2024 | 0.00302 | 0.00002 | 0.01317 | 467 |
| 2024->2025 | 0.00200 | 0.00001 | 0.01258 | 458 |
| 2025->2026 | 0.00122 | 0.00001 | 0.01097 | 460 |

Both items required for 'validated'. Overall evidence bar: NOT MET -> label remains limited_evidence.

NOTE: Item (i) threshold <= -0.20 is a pre-registered requirement, not aspirational. Average r = -0.0823.
NOTE: Item (ii) prior version used all-positive proxy (DEC-V2-03). This version computes proper pitcher-clustered bootstrap 95% CI per pair; pooled CI lower = min(ci_lo) across pairs = 1.49e-06 > 0 (PASS).


## 5. In-Sample Gap Table (qualified set, n>=300, 2021-2026)

| Version | Same-season ERA r | Same-season xFIP r | YoY stability r | n |
|---------|------------------|--------------------|----------------|---|
| v1_pitcher_holdout_A0_crossfitted | -0.0034 | -0.0061 | 0.4325 | 3363 |
| v2b_cross_fitted | 0.0573 | 0.0849 | 0.7137 | 3363 |
| v2b_insample *(NEVER DISPLAYED)* | -0.2259 | -0.2230 | 0.7510 | 3363 |

## 6. Independence

Available models for heatmap: ['dec_v2b', 'stuff_plus']
n (inner-joined qualified pitcher-seasons, 2021-2026): 15887

Pairwise Pearson r:
- dec_v2b vs stuff_plus: r=0.052

Flags (|r| > 0.5): none

Extension correlations (corr with mean release_extension per pitcher-season):
- dec_v2b: r=0.379
- stuff_plus: r=0.115


## 7. Process-Check Summary

All Part 2 comparisons on mismatched sets have been recomputed on the qualified common set (n=3,363).
See partA_qualified_results.json and partE_process_check.json for full details.

Key finding: On the qualified set, A0 xFIP r drops from 0.237 (all pitcher-seasons) to -0.006 (qualified only),
confirming original A0/A1/A2 figures were inflated by unqualified pitcher-seasons with atypical mechanics.
B (0.084) is best on the qualified set. No decisions changed.

Gate (d) previously recorded as cannot_verify_from_parquet -- now VERIFIED at min_r=0.9899.
Evidence bar item (ii) previously used all-positive proxy -- now uses proper CI lower bound: 1.49e-06 > 0 (PASS).
Evidence bar item (i) remains FAIL: avg r=-0.0823, threshold <= -0.20.


## 8. Active Model

Active version: v2b
DECEPTION_ACTIVE_VERSION = "v2b" in config/model_policy.py
DECEPTION_USE_IN_ARCHETYPES = True in config/model_policy.py
(All correctness gates pass; evidence bar item (i) fails; label = limited_evidence)


## 9. Deliverables

| File | Description |
|------|-------------|
| diagnostics/deception_v2/spec_lock.json | Pre-registered spec (unchanged) |
| diagnostics/deception_v2/partA_oof_cache.parquet | OOF predictions A0/A1/A2/B on qualified set |
| diagnostics/deception_v2/partA_qualified_results.json | Part 2 results on qualified common set |
| diagnostics/deception_v2/chosen_config_v2b.json | Corrected chosen config |
| diagnostics/deception_v2/deception_v2b_scores.parquet | v2b cross-fitted scores (season dropped) |
| diagnostics/deception_v2/v2b_oof_all.parquet | v2b OOF predictions (3 reps) |
| models/deception_plus_v2b_fold*_rep*.txt | v2b fold models (15 files) |
| models/deception_plus_v2b_meta.json | v2b model meta |
| diagnostics/deception_v2/gate_d_v2b.json | Gate (d) verified repeat correlation |
| diagnostics/deception_v2/gate_results_v2b.json | Re-evaluated gates for v2b |
| diagnostics/deception_v2/evidence_bar_v2b.json | Evidence bar with proper pooled CI |
| diagnostics/deception_v2/independence_heatmap_v2b.json | Independence heatmap |
| diagnostics/deception_v2/in_sample_gap_v2b.json | In-sample gap table |
| diagnostics/deception_v2/identifiability_v2b.json | Identifiability / memorization check |
| diagnostics/deception_v2/partE_process_check.json | Process-check list |
| Methodology/DECEPTION_COMPLETION.md | This file |

---
Post-audit closeout complete. Branch: deception-v2.