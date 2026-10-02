# NEEDS_DECISION.md

Unresolved questions that require the owner's judgment. Each entry includes what was found, the
options, and the recommendation. Written as part of the About-rework audit (2026-10-02).

---

## ND-01: `season` as a categorical feature — new seasons require retraining or removal

**Affects**: Stuff+ v3, Location+ v2, Tunneling+ v2, Deception+ v1. All four include `season`
as a categorical feature for era-adjustment. A season value not in the training set (e.g. 2027)
is treated by LightGBM as a missing category and falls to the model's own learned missing-value
branch. That branch is not meaningless, but it is uncontrolled — the missing-category behavior
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
`dropna`. Pitches from 2015–2019 and most of 2020 would have no `arm_angle` and be silently
dropped. This means the effective training range is approximately 2021–2026, not 2015–2026.
The meta.json's `seasons_trained_on` list may be misleading.

**Decision needed**: Should the meta.json be corrected to reflect the effective range
(~2021–2026), and should the live site's season selector for Deception+ be restricted to
2021–2026? REPORT_SEASON is currently hardcoded to 2026 in the router so this only matters
if a season selector is added.

**Also**: The HANDOFF.md (last updated 2026-09-28) says training was 2021–2025, but the
meta.json (trained_at 2026-09-15) shows 2015–2026. These are inconsistent. Confirm which
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
CURRENT_SEASON = 2026 on startup and seasons 2021–2025 on full refresh. No evidence of
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

## ND-08: Stuff+ v4 — Promotion approval required

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
3. Is the v3→v4 correlation at the expected level (>0.97)?
4. Are the 2020 flagging and qualification threshold scaling satisfactory?

**Logged**: 2026-10-02 by automated training run.
