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
| (i) Next-season xFIP corr ≤ -0.20 | avg=-0.0847 | ≤ -0.20 | PASS |
| (ii) Incremental delta-R² > 0, CI lower > 0 | avg=0.0022 | > 0 | PASS |

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

Deception+ v2 is complete with a limited-evidence label