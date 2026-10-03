# Deception+ v2 Rule Audit

**Date:** 2026-10-03  
**Branch:** deception-v2  
**Auditor:** Post-run audit (automated)

---

## PART A.1 — Part 2 Config Table

Sign convention: `xfip_corr` is Pearson r(predicted_delta_run_exp, xFIP). Both delta_run_exp and xFIP measure pitcher "badness," so a positive correlation means the model correctly identifies pitchers who allow more runs. **Higher xfip_corr = better model fit in Part 2.** This is the OPPOSITE sign from Part 4 validation (where plus-scale scores produce negative xFIP correlation).

| Config | Form | Features | xfip_corr | CI (95%) | n (qualified PS) | ERA corr |
|--------|------|----------|-----------|----------|-----------------|----------|
| A0 | per_pitch | F0 (full incl. body/consistency) | 0.2365 | [0.1992, 0.2769] | 5118 | 0.1425 |
| A1 | per_pitch | F0, regularized | 0.2764 | [0.2376, 0.3178] | 5118 | 0.1733 |
| A2 | per_pitch | F2 (mechanics only: x, z, ext, arm, season) | 0.2525 | [0.2073, 0.3010] | 5118 | 0.1504 |
| B  | pitcher_season | F2 | 0.0837 | [0.0371, 0.1275] | 3363 | 0.0598 |

Note: A0/A1/A2 are per-pitch configs aggregated to pitcher-season at eval time, without a ≥300-pitch filter on the evaluation set. B is trained and evaluated at pitcher-season level with ≥300-pitch filter. This creates different evaluation sets (n=5118 vs 3363), making the marginal xfip_corr values NOT directly comparable.

---

## PART A.2 — Paired Differences (Common Subset)

The `fast_paired_diff` function computes paired diffs on the **inner join** of both configs' OOF predictions with xFIP. This common subset controls for the different evaluation sets.

| Comparison | Point diff | 95% CI | Interpretation |
|-----------|-----------|--------|---------------|
| A2 − B (form rule test) | −0.0459 | [−0.0963, +0.0050] | B marginally better on common set; CI upper barely crosses zero |
| A1 − A2 (Test 1) | +0.0238 | [−0.0015, +0.0517] | A1 marginally better; CI includes zero → equivalent |
| A1 − A0 | approx. +0.040 | not stored | regularization helps |
| A2 − A0 | approx. +0.017 | not stored | feature reduction marginal |

Marginal apparent gap A2 − B = 0.2525 − 0.0837 = **+0.1688**. This is NOT the same as the paired diff (−0.0459) because the marginals use different evaluation sets. On the common/matched pitcher-season set, B performs marginally better than A2.

---

## PART A.3 — Evaluation Set Consistency

**Confirmed:** Per-pitch configs (A0, A1, A2) train on a 30% within-pitcher sample (`subsample_pitches(train_df, 0.30, seed+fold)`) but evaluate on ALL pitches of the held-out fold (`test_df = df[df['fold'] == fold]`). This is correct per the spec.

**Issue:** The per-pitch OOF set is then aggregated to pitcher-season level WITHOUT a ≥300-pitch qualification filter. B's evaluation set uses ≥300-pitch pitcher-seasons only. Consequently:
- A2 evaluates on 5,118 pitcher-seasons (includes low-pitch-count entries)
- B evaluates on 3,363 pitcher-seasons (≥300 pitches only)
- Paired test uses the intersection (~3,363)

**Impact:** The marginal xfip_corr values reported in part2_results.json for per-pitch configs include pitcher-seasons that B cannot evaluate. On the common qualified set, A2's advantage disappears (diff = −0.0459). The form rule was applied to the paired diff, which is the correct comparison.

---

## PART A.4 — Rule Application

### Test 1: A2 vs A1 (identity-prone feature screening)

**Spec rule (spec_lock.json):** `"identity_prone_default": "drop_unless_significantly_worse"`  
**Code condition:** `diff > 0.03 AND ci_lo > 0`  
**Result:** diff(A1−A2) = +0.0238, CI = [−0.0015, +0.0517], ci_lo = −0.0015 (NOT > 0)  
**Decision:** `A2_chosen_equivalent_or_better` ✓  
**Verdict:** Rule applied correctly. A1 is not significantly better, so simpler A2 chosen.

### Test 2: Form Rule (B vs A2)

**Spec rule (spec_lock.json):** `"form_rule": "choose_M2_unless_M1_significantly_better"`  
**Code condition:** `diff_PP_B > 0.03 AND cilo_ppb > 0`  
**Result:** diff(A2−B) on common set = −0.0459, CI = [−0.0963, +0.0050]  
**Decision:** `B_chosen_form_rule` ✓  
**Verdict:** Rule applied correctly. A2 is NOT significantly better (diff is negative). B selected.

### Screening Rules

| Feature | Rule | Condition | Result | Decision |
|---------|------|-----------|--------|----------|
| arm_angle | drop if R²≥0.90 AND removal equivalent | R²=0.789 < 0.90 | threshold not met | KEEP ✓ |
| release_extension | drop if removal equivalent (diff ≤ 0.03) | diff=0.0334 > 0.03 | not equivalent | KEEP ✓ |
| season | inconclusive_means_drop (diff ≤ 0.03) | diff=−0.0012, |diff|=0.0012 ≤ 0.03 | inconclusive | DROP ✓ |

Note: After the season DROP decision, the final feature set for B should be F2 without season. However, `chosen_config.json` still lists `season` in the feature list, and `gate_results.json` shows season in the features used. The season drop rule was recorded as a note (`season_drop_note`) but the feature list was not updated. This is a mild discrepancy — the season column is present in B's features in the actual trained model. The drop note says "DROP per pre-registered rule" but the chosen config used B which includes season. **This means the pre-registered season-drop rule was recorded but not enforced in the training of B.** Flag logged to NEEDS_DECISION.md.

### Chosen Config

**Rules select: B**  
**Pipeline selected: B**  
**Match: YES ✓**

---

## PART A.5 — "Within ±0.03" Claim Analysis

The run summary claimed per-pitch configs were "within the ±0.03 equivalence margin" of B. This claim is **INCORRECT as stated.**

- The actual paired diff (A2 − B on common set) = **−0.0459**
- |−0.0459| > 0.03 → OUTSIDE the equivalence margin
- Direction: B performs better than A2 on the common set, not "equivalent"

The correct description is: "A2 is not significantly better than B (CI upper = +0.0050, barely crossing zero), and B performs marginally better on the common qualified set, so the form rule selects B."

The marginal values (A2=0.2525, B=0.0837) look like A2 >> B, but this is an artifact of different evaluation sets. The marginals are not directly comparable.

---

## PART A.6 — Live Site and Promotion Status

**Active version:** `DECEPTION_ACTIVE_VERSION = "v2"` in `config/model_policy.py` line 105  
**Use in archetypes:** `DECEPTION_USE_IN_ARCHETYPES = False` in `config/model_policy.py` line 114  
**Status label:** `limited_evidence`  
**Promotion date:** 2026-10-03 18:00 (per DECEPTION_PROMOTION_LOG.md)  
**Router:** `api/routers/deception_report.py` — uses parquet lookup (pre-computed scores, no per-request scoring). Post-promotion check passed per gate (f).  
**Page rebuild:** Frontend pages updated (per DECEPTION_COMPLETION.md deliverables table).

---

## PART B — Rerun Determination

**Rules select: B** (confirmed above)  
**Pipeline selected: B**  
**PART B IS NOT TRIGGERED. No rerun needed.**

---

## PART C — Items from Existing Artifacts

### C.1 In-Sample Gap Table

`validation_results.json` contains: `"in_sample_gap_table": {}` — the gap table was not populated in Part 4 validation. No cross-comparison between v1 live scores, v1 pitcher-holdout, v2 cross-fitted, and v2 in-sample is available from existing artifacts.

### C.2 Memorization and Identifiability

**Memorization check (from validation_results.json):**
| Metric | Value | Interpretation |
|--------|-------|---------------|
| Score between-pitcher SD | 9.18 | large between-pitcher spread |
| Score within-pitcher SD | 4.94 | nonzero within-pitcher variation → no pure memorization |
| Score ICC | 0.776 | high ICC expected for mechanics-based model |

**Identifiability probe (from audit.md):**  
Cross-season matching 2021→2022 on standardized mechanics centroid (N=309 pitcher-season pairs):
- Top-1 accuracy: 0.472 (vs chance 0.003 → **146× above chance**)
- Top-5 accuracy: 0.835

Warning: mechanics features identify individual pitchers at 146× chance. This supports dropping identity-prone features, which was done.

### C.3 Gate Results

| Gate | ID | Value | Threshold | Pass |
|------|----|-------|-----------|------|
| (a) Same-season xFIP/ERA negative | a_same_season_negative | xFIP r=−0.0883, ERA r=−0.0629 | both < 0 | PASS |
| (b) Pooled YoY ≥0.30, min pair ≥0.25 | b_yoy_stability | pooled=0.7019, min=0.6712 | 0.30 / 0.25 | PASS |
| (c) Hygiene: mechanics/body only | c_hygiene | violations=[] | none | PASS |
| (d) Repeat correlation ≥0.98 | d_repeat_correlation | "cannot_verify_from_parquet" | 0.98 | PASS (unverified) |
| (e) All intended seasons scored | e_all_seasons_scored | missing=[] | none | PASS |
| (f) No per-request scoring | f_no_per_request_scoring | router uses parquet | none | PASS (post-promotion) |

**Note on gate (d):** The pass was recorded without actual verification from the parquet file. The repeat correlation ≥0.98 threshold cannot be confirmed from stored artifacts.

### C.4 Pooled Incremental Delta-R² with CI

| Year Pair | Delta-R² | n |
|-----------|---------|---|
| 2021→2022 | 0.00496 | 470 |
| 2022→2023 | 0.00043 | 474 |
| 2023→2024 | 0.00189 | 467 |
| 2024→2025 | 0.00239 | 458 |
| 2025→2026 | 0.00157 | 460 |
| **Pooled avg** | **0.00225** | — |

**CI status:** The gate_results.json records `"all_positive": true` and `"note": "CI approximated: all_positive used as proxy for CI_lower > 0"`. The spec_lock evidence bar requires **CI lower bound > 0**, not just all-positive. A proper pooled CI was not computed. Individual pair values range from 0.00043 to 0.00496 — all positive but the smallest (0.00043 for 2022→2023) is very close to zero. A true pooled CI lower bound has not been established.

### C.5 Heatmap Row (Independence from Stuff+)

From validation_results.json `independence` section:
- Deception+ v2 vs Stuff+ v4: r = 0.1452 (n = 3363)

No full heatmap with other models (Location+, Tunneling+) was computed in existing artifacts.

### C.6 Qualified Pitcher-Seasons Per Season

From audit.md:
| Season | Qualified Pitchers (≥300 clean pitches, all features present) |
|--------|--------------------------------------------------------------|
| 2021 | 905 |
| 2022 | 870 |
| 2023 | 859 |
| 2024 | 850 |
| 2025 | 864 |
| 2026 | 812 |

Note: 2026 has fewer qualifiers (812 vs ~860-905 in prior years) and arm_angle is only 75.9% populated in 2026 (partial season).

### C.7 "Aspirational" Language Review

No text in `DECEPTION_COMPLETION.md` calls a pre-registered threshold "aspirational." Evidence bar item (i) correctly shows `FAIL`. No rewording needed on this point.

However, evidence bar item (ii) in `DECEPTION_COMPLETION.md` shows `PASS` with threshold "> 0" but the actual spec requirement is "CI lower bound > 0." The table entry is misleading because it elides the CI requirement. See updated DECEPTION_COMPLETION.md.

---

## Summary

| Question | Answer |
|----------|--------|
| Which config do the rules select? | **B** (pitcher-season model, F2 features) |
| Did the pipeline select the same? | **Yes** |
| Was the "within ±0.03" description accurate? | **No** — actual paired diff = −0.0459 (B favored, outside ±0.03) |
| Was the marginal gap (0.169) a real signal A2 > B? | **No** — artifact of different evaluation sets; paired test on common set reverses the direction |
| Is a rerun (Part B) triggered? | **No** |
| Is the limited_evidence label correct? | **Yes** — evidence bar item (i) fails (avg next-season xFIP r = −0.085, threshold ≤ −0.20) |
| Does evidence bar item (ii) fully meet spec? | **Uncertain** — CI lower bound was approximated, not computed |
| Does the live site serve B? | **Yes**, DECEPTION_ACTIVE_VERSION = v2, model B |
