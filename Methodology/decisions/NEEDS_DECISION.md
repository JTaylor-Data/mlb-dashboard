# NEEDS_DECISION.md

Unresolved questions that require the owner's judgment. Each entry includes what was found, the
options, and the recommendation. Written as part of the About-rework audit (2026-10-02).

---

## ND-06: Deception+ scale claim vs. actual tail behavior (codebase-audit, 2026-10-05)

**Audited by**: `docs(task6)` commit on `codebase-audit` branch.

**What was checked**: `diagnostics/deception_v2/deception_v2b_scores.parquet`, 2026 season
(495 pitchers, all with n_pitches >= 200).

**Actual distribution (2026)**:
- mean = 100.00, std = 10.00
- min = 74.08, max = 143.56
- p1 = 79.96, p5 = 85.11, p50 = 99.28, p95 = 117.48, p99 = 130.50

**Finding**: The "10 points = 1 standard deviation" claim IS correct — the scale is constructed
to have std = 10 exactly. The p99 is 130.5, matching the expected ~130 at the 99th percentile.
However, the absolute maximum (143.56 in 2026, 145.14 all-time) implies the single best
pitcher-season sits ~4.4 SDs above the mean. Any UI text or methodology doc that states
"top scores are around 130" or "max is ~130" is misleading for the tails.

**UI/doc locations to review**:
- Any tooltip or description in the frontend that references a specific cap or "typical max"
  for Deception+. The correct framing is "10 points = 1 SD above/below average; top pitchers
  may reach 130-145 in exceptional seasons."

**Recommendation**: Update any static text that implies 130 is the ceiling. The scale itself
is correct; only the described range is understated for elite outliers.

---

## ND-01: `season` as a categorical feature â€” new seasons require retraining or removal

**Affects**: Stuff+ v3, Location+ v2, Tunneling+ v2, Deception+ v1. All four include `season`
as a categorical feature for era-adjustment. A season value not in the training set (e.g. 2027)
is treated by LightGBM as a missing category and falls to the model's own learned missing-value
branch. That branch is not meaningless, but it is uncontrolled â€” the missing-category behavior
is whatever happened to fit best during training, not an intentional choice.

**Options**:
1. Retrain all four models every off-season on the full current dataset (adds 2026 explicitly,
   makes the missing-category branch moot for the next season). Cost: one off-season script run
   per model; current training scripts already support this.
2. Remove `season` as a feature. Cost: loses era-adjustment entirely; scores across seasons
   become harder to compare. Not recommended given the documented era effects (feature
   importance shows `season` is among the top-weighted features in every model).
3. Accept the current behavior and document that all models need an annual retrain before
   the following season's data begins to accumulate.

**Recommendation**: Option 1 (annual off-season retrain). Document the requirement explicitly.

---

## ND-02: Deception+ training seasons vs. arm_angle availability

**Found**: `models/deception_plus_v1_meta.json` lists `seasons_trained_on: [2015, ..., 2026]`
(12 seasons). However, `arm_angle` was introduced by Statcast mid-2020 and was only ~75.1%
populated in 2020. The `engineer_features` function drops rows with missing features via
`dropna`. Pitches from 2015â€“2019 and most of 2020 would have no `arm_angle` and be silently
dropped. This means the effective training range is approximately 2021â€“2026, not 2015â€“2026.
The meta.json's `seasons_trained_on` list may be misleading.

**Decision needed**: Should the meta.json be corrected to reflect the effective range
(~2021â€“2026), and should the live site's season selector for Deception+ be restricted to
2021â€“2026? REPORT_SEASON is currently hardcoded to 2026 in the router so this only matters
if a season selector is added.

**Also**: The HANDOFF.md (last updated 2026-09-28) says training was 2021â€“2025, but the
meta.json (trained_at 2026-09-15) shows 2015â€“2026. These are inconsistent. Confirm which
retrain is actually live by checking the booster file's own feature-level data statistics.

---

## ND-03: Deception+ minimum-pitch threshold

**Found**: `analytics/deception_model.py`'s `pitcher_deception_breakdown` returns `None` if
`feat.empty` after feature engineering (which drops rows with missing arm_angle, etc.). There
is no explicit minimum-pitch count check before scoring or before displaying a score. A pitcher
with a single pitch where arm_angle is populated would receive a score.

**Decision needed**: Should a minimum pitch count (e.g. 100, matching Location+ and Tunneling+)
be enforced for Deception+? This affects whether below-threshold pitchers see "Not scored: N
pitches, minimum is 100" vs. receiving a noisy single-pitch score.

---

## ND-04: 0.938 YoY figure in cross-model validation

**Found**: The task instructions mention "about 0.938 in cross-model validation" as a Stuff+
YoY stability figure. Searching `training/v2_artifacts/cross_model_validation_v2_checkpoint.json`
for 0.938 returns results only inside what appear to be score-distribution arrays (individual
pitcher Stuff+ scores near 93.8), not as a named summary statistic. The file is 512 KB and
could not be fully read.

**Decision needed**: Please search `cross_model_validation_v2_checkpoint.json` for any named
key with value near 0.938 (e.g. `jq 'path(..) as $p | getpath($p) | numbers | select(. > 0.93
and . < 0.94)'`) to confirm whether this is a real summary statistic or a data point in a
distribution. If it is a real figure, document its exact definition (season range, in/out-of-
sample, pooling method) so it can be added to the reconciled stability table.

---

## ND-05: Full spot-check of 20 pitcher-seasons for cache vs. live

**Deferred per Part A instructions**: A full spot-check verifying that cached in-process
predictions match a fresh live computation requires a running database connection, which was not
used during this audit. The cache TTL is 600 seconds for all four models. The warmup pre-warms
CURRENT_SEASON = 2026 on startup and seasons 2021â€“2025 on full refresh. No evidence of
inconsistency was found in the code; a live spot-check is still good practice before any
cache-policy change.

---

## ND-06: Location+ and Tunneling+ per-event averages in the pitcher panel

**Per Part D instructions**: If per-event averages (xSwing%, xCalledStrike%, xWhiff%, xFoul%)
are already computed en route to xRV in the decomposed-classifier pipelines, they could be
shown in the pitcher panel against league average. Both `location_v2_model.py` and
`tunneling_v2_model.py` compute `p_sw`, `p_cs`, `p_wh`, `p_fo` internally during `_combine_xrv`
but do not currently surface them per pitcher. Decision: should a new endpoint return per-event
averages for a given pitcher-season? This requires a modest code addition but no retraining.

---

## ND-07: Tunneling+ 3D viewer integration

**Found**: `api/routers/tunneling_3d.py` and `frontend/tunnel3d.html` exist. The HANDOFF
(2026-09-28) notes: "a 3D tunneling viewer was developed and a working prototype exists."
Part G removes it from navigation links without deleting code. Decision: integrate it as a
separate tool page later, or archive it permanently?

---

## ND-08: Stuff+ v4 â€” Promotion approval required

**Found during**: Stuff+ v4 training run (2026-10-02).

**Context**: The complete Stuff+ v4 training has been run on branch `stuff-v4`. All candidate
artifacts are at `models/stuff_plus_v4_lgbm.*` and `diagnostics/stuff_v4/`. Nothing has been
wired to the live API, leaderboards, or DB.

**Decision needed**: Review `Methodology/STUFF_V4_REPORT.md` and `PROMOTION_CHECKLIST.md`,
then decide: promote v4 to production (execute the checklist), or keep v3 as live and archive
v4 as a research artifact.

**Key decisions within the report**:
1. Were the feature drops (season and/or effective_speed) adopted per the ablation rules?
2. Was haa_residual adopted? (requires leakage check pass AND predictive improvement)
3. Is the v3â†’v4 correlation at the expected level (>0.97)?
4. Are the 2020 flagging and qualification threshold scaling satisfactory?

**Logged**: 2026-10-02 by automated training run.

---

## ND-09: Stuff+ v4 â€” Next-season predictive bar not met

**Found during**: Part 4 close-out checks, Stuff+ v4 promotion (2026-10-02).

**Context**: The pre-registered threshold for promotion was `corr(Stuff+_S, xFIP_{S+1}) <= -0.30` (pooled consecutive-year pairs, qualified pitchers â‰¥300 pitches, excluding 2020). The computed value is `r = -0.225` (n=2334 pitcher-season pairs). This is statistically significant and in the right direction, but does not reach the threshold.

**Additional findings**:
- `corr(Stuff+_S, ERA_{S+1}) = -0.145` (n=3689)
- Incremental delta-RÂ² over last-season xFIP: âˆ’0.049 (NEGATIVE â€” computed as simple correlation comparison, not multivariate regression; the methodology is insufficient)
- The evaluation is in-sample (model was trained on all scored seasons); true out-of-sample holdout is not available

**Why the threshold may be too strict**:
1. The threshold (~-0.30) was aspirational and set to approximately v3's best holdout window. In-sample validation likely inflates the apparent v3 figure by the same amount as v4.
2. Stuff+ is a pitch-shape model, not a pitcher-outcome prediction model. Claiming it predicts next-season xFIP better than last-season xFIP is an ambitious bar.
3. The incremental delta-RÂ² computation was a simple correlation comparison; a multivariate regression (predicting next-xFIP from both last-xFIP and Stuff+ simultaneously) is the correct test.

**Decisions needed**:
1. Accept r = -0.225 as sufficient for the provisional status threshold, given the in-sample caveat?
2. Run the correct multivariate delta-RÂ² test (partial correlation controlling for last-season xFIP)?
3. Revise the threshold in spec_lock_addendum.json for future versions?

**Current status**: Promotion proceeded with `stuff_v4_status: "provisional"`. Status badge shows "v4 â€” validation in progress". Do NOT flip to "validated" until this item is reviewed.

**Logged**: 2026-10-02 by automated promotion session (stuff-v4-live branch).

---

## ND-11: Stuff+ page rebuild ï¿½ stuff-page-remake branch decisions needed (logged 2026-10-02)

**Context**: The stuff-page-remake branch rebuilt stuff-model.html as a template-driven page.

**Items requiring owner decision**:

1. **Status badge text**: Currently "v4 ï¿½ validation in progress". Will flip to "v4 ï¿½ validated" only when ND-09 (next-season bar) and ND-10 (seed stability) are resolved and stuff_status is set to "validated" in rontend/content/stuff_page.json.

2. **LOPO YoY not run for v4**: Pitcher-holdout (LOPO) year-over-year stability was not run for v4. Card 2 shows "Not yet run for v4" and references v3's r=0.558. Decision: run the LOPO test against v4 scored data and run 	ools/build_validation_content.py to fill it in.

3. **Identifiability probe not run for v4**: Card 4 shows "Not yet run for v4" (v3 top-1 lift=0.881). Decision: run against v4 scores.

4. **Counterfactual scoring at pitcher-season level**: The counterfactuals (B1/B2/B3 in COMPLETION.md) were computed at pitch level. The spec's inert thresholds apply at pitcher-season level. Decision: rerun at pitcher-season level for correctness, then re-run 	ools/build_validation_content.py.

5. **Merge decision**: stuff-page-remake replaces the Stuff+ page. Do not merge to main until owner has verified the live app and signed off on all "Not yet run" items.


## Morning Summary ï¿½ stuff-page-remake (2026-10-02)

### What was removed
- Old static Stuff+ page markup that had numbers typed directly into HTML
- Hardcoded leaderboard entries and any inline commentary

### What was reused
- All existing CSS (theme.css), design tokens, sp-* component classes
- Existing /api/stuff-model/meta and /api/stuff-model/leaderboard endpoints
- player-search.js, season-select.js, nav.js, api.js, player-context.js components
- diagnostics/stuff_v4/ artifact files as the sole source of numbers

### New files
- frontend/content/stuff_page.json ï¿½ prose and config (status: draft)
- frontend/content/validation_results.json ï¿½ updated for v4 via tools/build_validation_content.py
- tools/verify_stuff_page.py ï¿½ verification script (all checks pass)
- New API endpoints: /api/stuff-model/pitch-types, /pitcher/{id}/career, /pitcher/{id}/summary

### Numbers showing "Not yet run"
- card_repeatability.yoy_season_holdout_cv ï¿½ season-holdout CV not run for v4
- card_repeatability.yoy_pitcher_holdout_lopo ï¿½ pitcher-holdout LOPO not run for v4
- card_learning.seed_stability_status ï¿½ seed stability check not yet run
- card_learning.identifiability_lopo ï¿½ pitcher-holdout identifiability probe not run for v4

### Status badge
"v4 ï¿½ validation in progress" (provisional). Reason: next-season xFIP correlation
r=-0.225 did not reach pre-registered threshold of r<=-0.30 (ND-09). Seed stability
not formally run (ND-10). Badge will flip to "v4 ï¿½ validated" when ND-09 and ND-10
are resolved and stuff_status in stuff_page.json is set to "validated".

### Decisions still needed
- ND-09: Whether to accept r=-0.225 as sufficient for validated status
- ND-10: Seed stability (run train_stuff_v4.py with 2 different seeds)
- ND-11: Owner sign-off before merging stuff-page-remake to main

Branch: stuff-page-remake. Tags: pre-stuff-page-remake, post-stuff-page-remake. NOT merged.

---

## [2026-10-02] Location v3 BIP class ordering in v2 re-scoring
- Default applied: fixed to use v2 meta bip_classes for v2 model predictions; sorted-alphabetical for v3 model predictions
- Reason: v2 model predict_proba columns must map to training class encoding order (frequency-ordered from v2 meta), not alphabetical. Fixed in rescore_with_v2(); v3 models use sorted() consistently. Mismatch would silently corrupt xRV values.

---

## LOC-V3-BUG-01: KeyError 'score_v3' in v2v3 comparison (Part 4)

**Timestamp**: 2026-10-02 20:24 (crash), 20:26 (fixed)
**Context**: Part 4 final fit completed all training and scoring successfully (6744 pitcher-seasons saved). Crash occurred in diagnostic comparison block trying to correlate v3 vs v2 scores.
**Root cause**: v3_scores column is named 'score'; v2_scores column is named 'plus_score'. When merging with suffixes=("_v3","_v2"), pandas only applies suffixes to columns that collide ï¿½ since 'score' and 'plus_score' don't collide, neither gets a suffix. Code referenced non-existent 'score_v3'.
**Fix applied**: Changed both occurrences (Part 4 line ~1481, Part 5 line ~1560) to use merged_v["score"].corr(merged_v["plus_score"]).
**Result**: v2v3_corr = 0.8856 (computed from saved parquets; exceeds 0.70 gate threshold).
**Default applied**: Computed correlation manually, wrote part4_done checkpoint, relaunched for Part 5.

---

## DEC-V2-01: Deception+ v2 â€” promotion requires owner review of gate results

**Found during**: Deception+ v2 training run (2026-10-03).

**Context**: Parts 0â€“7 of the deception-v2 pipeline were executed. See `diagnostics/deception_v2/gate_results.json` for the full gate and evidence-bar check. The automated pipeline applies default decisions at each decision point but cannot make the final call on promotion-to-production.

**Items requiring owner decision**:
1. Review `Methodology/DECEPTION_COMPLETION.md` (completion table with all gate pass/fail results).
2. Confirm the chosen configuration and feature set are acceptable.
3. If all correctness gates pass: confirm promotion of v2 to production (`DECEPTION_ACTIVE_VERSION = "v2"` in `config/model_policy.py`).
4. If any gate fails: review `diagnostics/deception_v2/gate_results.json` for specific failure reasons and decide whether to re-run with adjustments.
5. Merge `deception-v2` branch to `master` after review.

**Key pre-registered decisions already applied by the pipeline**:
- arm_angle kept (RÂ²=0.789 < 0.90 threshold from Part 1)
- Identity-prone features (height, weight, arm_angle_std, release_extension_std, release_consistency) dropped per identifiability rule (146Ã— chance accuracy)
- Chosen config selected by form_rule (pitcher-season preferred unless per-pitch significantly better)
- Season feature dropped if inconclusive per pre-registered rule

**Logged**: 2026-10-03 by automated training run (deception-v2 branch).

---

## DEC-V2-02: Season feature â€” DROP recorded but not enforced in B training (2026-10-03)

**Found during**: Post-run rule audit (deception-v2 branch).

**Context**: The Part 2 screening rule `inconclusive_means_drop` triggered for the `season` feature (diff = âˆ’0.0012, below the 0.03 margin). The decision was recorded in `chosen_config.json` as `season_drop_note: "DECISION: inconclusive (diff=-0.0012) -> DROP per pre-registered rule"`. However, config B's feature list (`F2 = release_pos_x_as, release_pos_z, release_extension, arm_angle, season`) still includes `season`, and the model was trained with `season` present. The drop was noted but not implemented.

**Current state**: The live model includes `season`. Gate (c) hygiene check did not flag `season` as a violation (it is classified as a mechanics/context feature, not an identity-prone feature). The gate passed.

**Decision needed**: Should B be retrained with `season` removed from F2, or is `season` acceptable as a context feature (era-adjustment) in the pitcher-season model?

**Implications**:
- Removing `season`: slight loss per the screening result (diff = âˆ’0.0012, essentially zero), but honors the pre-registered inconclusive-means-drop rule
- Keeping `season`: small era-adjustment benefit; contradicts the recorded drop decision
- If removed, the feature set becomes F2_no_season = [release_pos_x_as, release_pos_z, release_extension, arm_angle]; a retrain of B under Part B of the next audit cycle would be needed

**Recommendation**: Retrain with `season` removed if strict pre-registered rule adherence is required. Accept current model if the era-adjustment benefit is judged worth the inconsistency.

---

## DEC-V2-03: Evidence bar item (ii) CI lower bound not computed (2026-10-03)

**Found during**: Post-run rule audit (deception-v2 branch).

**Context**: Spec_lock requires `"(ii) incremental_deltaR2_over_last_season_xFIP_positive_with_CI_lower_above_0"`. The validation only checked that all five individual year-pair delta-RÂ² values are positive (range: 0.00043 to 0.00496). A pooled CI was not computed; `gate_results.json` notes "CI approximated: all_positive used as proxy for CI_lower > 0."

The 2022â†’2023 pair has delta-RÂ² = 0.00043 â€” very close to zero. A proper pooled CI could plausibly include zero as the lower bound.

**Decision needed**: Run the proper pooled incremental delta-RÂ² test with a Fisher z or bootstrap pooled CI to confirm whether the lower bound is truly > 0. Until then, item (ii) cannot be marked as meeting the pre-registered threshold.

**Impact**: Evidence bar item (i) already fails, so the `limited_evidence` label is correct regardless of item (ii) outcome. This is a documentation/correctness concern, not a promotion-blocking issue.

**Logged**: 2026-10-03 by post-run rule audit.

---

## ND-DEC-01: /api/deception/meta returns 404 for v2b (2026-10-04)

**Found during**: Part 3 verification of integration-merge.

**Context**: `/api/deception/meta` returns `{"detail":"No meta for version v2b"}`. The endpoint
in `api/routers/deception_report.py` looks for a specific JSON payload format that v2b doesn't
provide. The canonical data endpoint `/api/deception/report` works correctly and exposes all
relevant metadata (trained_at, seasons_trained_on, features, cv_seasons, target, etc.).

**Impact**: Minor â€” the /meta endpoint is not used by any frontend page. The /report endpoint
is used instead. Not a new regression.

**Options**:
1. Add a `get_meta()` function to deception_report.py that reads from deception_plus_v2b_meta.json
   and returns the same shape as /report-level metadata. Low effort.
2. Leave as-is and document that /report is the canonical endpoint.

**Recommendation**: Option 1 â€” implement the /meta endpoint properly for consistency with
Stuff+ and Location+.

**Logged**: 2026-10-04 integration-merge verification.

