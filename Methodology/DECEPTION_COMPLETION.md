# Deception+ v2 Completion Table

Generated: 2026-10-03 18:01:03

Status: **limited_evidence**  |  Label: **limited_evidence**


## 1. Audit Findings
| Finding | Value | Note |
|---------|-------|------|
| v1 per-request scoring | yes | removed in v2 -- all scores pre-computed |
| v1 split-half r | 0.998 | artifact: consistency features computed pre-split |
| arm_angle R² from mechanics | 0.789 | < 0.90 threshold -> KEEP |
| Identifiability (top-1/chance) | 146× | very high; identity-prone features dropped |
| arm_angle coverage | 2021-2026 | 0% before 2020, partial 2026 (75.9%) |
| bio coverage (height/weight) | 98.6% | good coverage for 2021+ pitchers |
| qualified pitcher-seasons/yr | 812-905 | season range 2021-2026, min 300 pitches |

## 2. Configuration Tests
| Test | Comparison | Decision | Rule |
|------|-----------|---------|------|
| Test 1 (A2 vs A1) | A1-A2 diff=0.0238 | A2_chosen_equivalent_or_better | drop_unless_significantly_worse |
| Test 2 (form rule) | per_pitch-B diff=-0.0459 | B_chosen_form_rule | choose_M2_unless_M1_significantly_better |
| Screening: arm_angle | -- | KEEP: R2=0.789 < 0.90 from Part 1 | R2<0.90 keep |
| Screening: extension | -- | KEEP release_extension: diff=0.0334 | equivalence |
| Screening: season | -- | DECISION: inconclusive (diff=-0.0012) -> DROP per pre-registered rule | inconclusive_means_drop |
| Chosen config | B | features=release_pos_x_as, release_pos_z, release_extension, arm_angle, season... | -- |

## 3. Correctness Gates
| Gate | Value | Threshold | Pass/Fail |
|------|-------|-----------|-----------|
| (a) Same-season xFIP/ERA < 0 | xFIP=-0.0883, ERA=-0.0629 | both < 0 | PASS |
| (b) Pooled YoY ≥ 0.30, min pair ≥ 0.25 | pooled=0.7019, min=0.6712 | 0.30 / 0.25 | PASS |
| (c) Hygiene: only mechanics/body features | violations=[] | none | PASS |
| (d) Repeat correlation ≥ 0.98 | averaged_across_repeats_cannot_verify_from_parquet | 0.98 | True |
| (e) All intended seasons scored | missing=[] | none | PASS |
| (f) No per-request scoring | router uses parquet lookup post-promotion | no per-request | PASS (post-promotion) |

## 4. Evidence Bar
| Item | Value | Threshold | Pass/Fail |
|------|-------|-----------|-----------|
| (i) Next-season xFIP corr ≤ -0.20 | avg=-0.0847 | ≤ -0.20 | **FAIL** (bar not met) |
| (ii) Incremental delta-R² CI lower > 0 | avg=0.0022, all pairs positive | CI lower > 0 | UNVERIFIED (CI not computed; all-positive proxy used; bar not confirmed) |

Note: Evidence bar item (i) fails the pre-registered threshold (avg r = −0.085, required ≤ −0.20). This is not an aspirational target — it is a required threshold. The bar was not met.
Evidence bar item (ii) requires a pooled CI lower bound strictly above zero. The validation only checked that all individual pair values are positive; the actual CI lower bound was not computed. The bar has not been confirmed for item (ii).

## 5. Additional Validation
| Metric | Value | Note |
|--------|-------|------|
| Split-half reliability at 300 pitches | 0.0617 | target ≥ 0.70 |
| Independence from Stuff+ v4 | r=0.1452 | should be low |
| Score within-pitcher std | 4.9403 | nonzero confirms no pure memorization |
| Score between-pitcher std | 9.1845 | -- |

## 6. Deliverables
| File | Description |
|------|-------------|
| diagnostics/deception_v2/spec_lock.json | Pre-registered spec |
| diagnostics/deception_v2/train_part1.py | Audit script |
| diagnostics/deception_v2/audit.md | Audit findings |
| diagnostics/deception_v2/train_part2.py | Training: configurations A0/A1/A2/B |
| diagnostics/deception_v2/part2_results.json | Config test results |
| diagnostics/deception_v2/chosen_config.json | Selected config |
| diagnostics/deception_v2/train_part3.py | Cross-fitted scoring |
| diagnostics/deception_v2/deception_v2_scores.parquet | All pitcher-season scores |
| models/deception_plus_v2_fold*_rep*.txt | Fold models |
| models/deception_plus_v2_meta.json | Model meta |
| diagnostics/deception_v2/train_part4.py | Validation script |
| diagnostics/deception_v2/validation_results.json | Validation results |
| diagnostics/deception_v2/train_part5.py | Gates + promotion |
| diagnostics/deception_v2/gate_results.json | Gate evaluation |
| api/routers/deception_report.py | Updated router (parquet lookup, no per-request scoring) |
| config/model_policy.py | DECEPTION_ACTIVE_VERSION = v2 |
| Methodology/DECEPTION_COMPLETION.md | This file |
| Methodology/DECEPTION_PROMOTION_LOG.md | Promotion log |
| frontend/deception-model.html | Updated page |
| frontend/js/deception-model.js | Updated JS |

---

Deception+ v2 is complete with a limited-evidence label.

---

## Post-Run Rule Audit (2026-10-03)

A rule audit was conducted after the run summary raised a concern that B was incorrectly selected. Full audit in `diagnostics/deception_v2/DECEPTION_RULE_AUDIT.md`.

**Outcome: B is confirmed as the correct selection.** No rerun was needed.

Key findings:
1. The form rule test computed a **paired diff on the common subset** (inner join of A2 OOF + B OOF + xFIP): diff = −0.0459, CI = [−0.0963, +0.0050]. B was favored on this set.
2. The marginal xfip_corr values (A2=0.2525, B=0.0837, apparent diff=+0.169) are NOT directly comparable: per-pitch configs evaluated on all pitcher-seasons (n=5118), B evaluated only on ≥300-pitch pitcher-seasons (n=3363). On the common qualified set, the gap reverses.
3. The run summary description "within ±0.03 equivalence margin" was inaccurate. The actual diff was −0.0459 (outside the margin, B favored). The correct description: A2 is not significantly better than B (CI upper = +0.0050), form rule selects B.
4. Evidence bar item (i) **was not met** (threshold ≤ −0.20, actual avg = −0.085). This is a pre-registered rule, not an aspirational target.
5. Evidence bar item (ii) CI lower bound > 0 **was not confirmed** — only an all-positive proxy was used.
6. Gate (d) repeat correlation was recorded as PASS but marked "cannot_verify_from_parquet."
7. In-sample gap table was not populated (validation_results.json shows empty dict).
8. Season feature DROP decision was recorded in the note but not enforced in B's training (B still uses season). See NEEDS_DECISION.md.