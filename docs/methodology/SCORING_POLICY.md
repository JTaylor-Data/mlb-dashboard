# SCORING_POLICY.md

Audit of how each live model scores pitchers. Produced 2026-10-02 as Part A of the About rework.
All evidence cited with file:line or artifact path.

---

## 1. Which model scores which season

### Stuff+ v3
- **Booster**: `models/stuff_plus_v3_lgbm.txt`
- **Trained on**: seasons 2016–2026 (11 seasons). 2015 is excluded because `spin_axis` is 0%
  populated that year. Source: `docs/methodology/HANDOFF.md` line 76; `models/stuff_plus_v3_lgbm_report.json`
  — earliest holdout season in `cv_folds` is 2016.
- **One booster scores all seasons**: yes. Scoring calls `model.predict(feat[meta["all_features"]])`
  on whatever season is requested. Source: `analytics/stuff_model.py:168`.
- **Older seasons scored on training data**: yes, for seasons 2016–2026. No season is ever scored by
  a model that has never seen it (the model was trained on all 11 seasons; a LOSO holdout is only
  the validation framework, not the deployed booster).
- **2026 scored**: yes.
- **Available seasons**: 2016–2026. 2015 would return an empty DataFrame (no spin_axis data).

### Location+ v2
- **Boosters**: five classifiers — `models/location_plus_v2_{xswing,xcs,xwhiff,xfoul,bip}.txt`
- **Trained on**: 2015–2026. Source: `training/v2_artifacts/location_plus_v2_meta.json` (not
  separately listed in a seasons field, but `_cat_levels` in `analytics/location_v2_model.py:63`
  reads `WHERE season BETWEEN 2015 AND 2026`).
- **Available seasons**: 2015–2026 (hardcoded range in `_cat_levels`; a season outside this range
  would have its `season` category coerced to NaN).
- **2026 scored**: yes.

### Tunneling+ v2 (LIVE — promoted 2026-09-21)
- **Note**: despite the router path `/api/tunneling/v3/`, the live scorer since 2026-09-21 is
  `analytics/tunneling_v2_model.py`. The old `tunneling_v3.txt` single-regressor is no longer
  loaded. Source: `api/routers/tunneling_report.py:27` imports from `tunneling_v2_model`.
- **Boosters**: five classifiers — `models/tunneling_plus_v2_{xswing,xcs,xwhiff,xfoul,bip}.txt`
- **Trained on**: 2015–2026. Source: `models/tunneling_plus_v2_meta.json:472` `"seasons"` array.
- **Commit table**: `training/tunneling_v2_artifacts/200ms/pitcher_season_commit_point.csv`.
  This table was built from the same 12-season dataset; it is re-read by the scorer at runtime.
  Source: `analytics/tunneling_v2_model.py:32`.
- **Available seasons**: 2015–2026.
- **2026 scored**: yes.

### Deception+ v1
- **Booster**: `models/deception_plus_v1.txt`
- **`seasons_trained_on` in meta.json**: [2015, ..., 2026]. Source: `models/deception_plus_v1_meta.json`.
- **Effective training range**: approximately 2021–2026. `arm_angle` was introduced by Statcast
  mid-2020 (~75% populated in 2020). Pitches without `arm_angle` are dropped by `dropna` in
  `analytics/deception_features.py`. The HANDOFF.md (line 115) notes training on 2021–2025 for
  the original model; the September 2026 retrain extended the dataset but arm_angle gaps remain.
  **See NEEDS_DECISION.md ND-02.**
- **Live scoring**: only REPORT_SEASON = 2026. Source: `api/routers/deception_report.py:25`.
  No season selector is wired up.
- **2026 scored**: yes (the only season currently served).

---

## 2. When scores are computed

All four models compute scores **at request time** against the live SQLite database, with an
**in-process cache** (TTL 600 seconds). Source: each model's `_raw_predictions` function checks
a module-level dict keyed by season and falls back to a full DB query on cache miss.

The API warmup (`api/warmup.py`) pre-warms CURRENT_SEASON = 2026 on server startup and for
past seasons 2021–2025 (Stuff+, Tunneling+, Location+ only; Deception+ warms REPORT_SEASON = 2026
only). After a manual "Refresh data" click, the model caches are **not** cleared (only standings,
player list, and non-model pools are re-warmed). Source: `api/warmup.py:103`.

No frozen cached artifacts are read for scoring. Leaderboard numbers and SHAP breakdowns are
always computed live from the current DB contents.

**Full spot-check (20 pitcher-seasons, cache vs. live) deferred** — requires a running database
connection. See NEEDS_DECISION.md ND-05.

---

## 3. Minimum-pitch thresholds

| Model | Scoring threshold | Leaderboard threshold | Enforced where |
|---|---|---|---|
| Stuff+ v3 | None (all pitchers scored) | 200 pitches | `api/routers/stuff_model_report.py:30` `LEADERBOARD_MIN_PITCHES` |
| Location+ v2 | 100 pitches | 100 pitches | `analytics/location_v2_model.py:28` `MIN_PITCHES`; applied in `leaderboard()` line 160 and `pitcher_score()` line 198 |
| Tunneling+ v2 | 100 pitches (raw, before lag) | 100 pitches | `analytics/tunneling_v2_model.py:36` `MIN_SEASON_PITCHES`; applied in `_raw_predictions()` line 132 |
| Deception+ v1 | None explicit | N/A (no leaderboard) | `pitcher_deception_breakdown` returns None if `feat.empty`; no integer count check. See ND-03. |

**Stuff+**: scores are computed for every pitcher regardless of pitch count. The score for a
one-pitch pitcher is technically valid (and shrinkage would heavily shrink it toward the type mean)
but it is filtered out of the leaderboard at 200 pitches.

**Location+ and Tunneling+**: below-threshold pitchers are excluded from `_raw_predictions` before
any scoring happens. The scorer returns None / empty, so those pitchers are "not scored," not merely
hidden.

**Additional eligibility conditions**:
- **Tunneling+**: requires a commit point entry in the 200ms commit table (a pitcher must have
  qualifying FF/SI/FC pitches). A pitcher absent from the commit table produces no scored rows.
  Source: `analytics/tunneling_v2_model.py:84` comment.
- **Deception+**: requires `arm_angle` to be populated for a pitch (any pitch without it is
  silently dropped). Requires bio record in `pitcher_bio` table for `height_in`/`weight_lb`;
  pitchers without a bio record are also silently dropped.
- **Excluded pitch types**: `PO`, `UN`, `EP` are excluded by `exclude_junk_pitch_types()` in
  `analytics/stuff_features.py`, called from `location_v2_model.py:83`. `KC` is mapped to `CU`.
  Source: `analytics/location_v2_model.py:67-70`.
- **Postseason**: `pull_statcast.py` now keeps only `game_type = "R"` (regular season).
  Source: `docs/methodology/HANDOFF.md` line 37. Postseason pitches are not in the database; this
  gate was added in September 2026 retroactively.

---

## 4. Shrinkage

| Model | Shrinkage applied | Method |
|---|---|---|
| Stuff+ v3 | Yes | Empirical-Bayes per pitch type. K estimated fresh from each season's own raw predictions via method-of-moments variance decomposition. Values came out 0.4–2.8 pitches (very small; mainly relevant for rare pitch types). Falls back to median K for pitch types too rare to estimate their own. Source: `analytics/stuff_model.py:186-242`. |
| Location+ v2 | No | Decomposed-classifier architecture computes mean xRV over all qualified pitches directly. No shrinkage step. |
| Tunneling+ v2 | No | Same decomposed architecture. No shrinkage step. |
| Deception+ v1 | No | Direct mean over scored pitches, no shrinkage. |

**Location+ v1 vs. v2 note**: The old v1 single-regressor (no longer live) may have included
shrinkage via the v1 feature pipeline. The current live v2 does NOT apply shrinkage. Confirmed
by reading `analytics/location_v2_model.py` in full.

---

## 5. Standardization

All four models use `to_plus_scale(value, league_mean, league_std)` defined as
`100 - 10 * (value - mean) / std`. Source: `analytics/location_features.py`.

"100 = league average, 10 points = 1 standard deviation" for each model, computed within a
requested season over the qualified population:

| Model | Population for z-score | Notes |
|---|---|---|
| Stuff+ v3 | All per-pitch raw predictions in that season (not filtered to qualified pitchers) | `norm_mean = raw["pred"].mean()`, `norm_std = raw["pred"].std()`. Source: `analytics/stuff_model.py:297-298`. Cross-type normalization (all pitch types pooled). |
| Location+ v2 | Qualified pitchers (>= 100 pitches) that season | `league_mean = q["mean_xrv"].mean()`, `league_std = q["mean_xrv"].std()`. Source: `analytics/location_v2_model.py:164-166`. |
| Tunneling+ v2 | All pitchers who passed the 100-pitch filter (no additional leaderboard filter before standardization) | `mean, std = league["pred"].mean(), league["pred"].std()`. Source: `analytics/tunneling_v2_model.py:162`. |
| Deception+ v1 | All per-pitch raw predictions for that season | `season_mean, season_std = league_raw["pred"].mean(), league_raw["pred"].std()`. Source: `analytics/deception_model.py:89`. |

Standard deviations of pitcher-season scores from stored artifacts: **not computed in this
static audit** — requires a live database query to compute pitcher-level score distributions.

---

## 6. New pitchers and new seasons

All four models include `season` as a categorical feature. At scoring time:

- **Known season value (in training range)**: normal prediction, no issue.
- **Unknown season value** (e.g. 2027): the feature is coerced to NaN (category not in training
  set). LightGBM's native categorical handling routes NaN categories to the "missing-value branch"
  learned during training. The prediction is not an error but its behavior is uncontrolled.
  Source: see `analytics/stuff_model.py:117-118`, `analytics/tunneling_v2_model.py:114`.
- **New pitcher with no prior data**: all models exclude pitcher identity as a feature by design.
  A new pitcher is scored identically to any other pitcher with the same physical features.

**See NEEDS_DECISION.md ND-01** for the decision between annual retraining and removing the
`season` feature.

---

## 7. Entropy / arsenal_entropy

`arsenal_entropy` (Shannon entropy over pitch-type usage shares) appears in:

- `analytics/tunneling_features.py:33-34, 338-364` — computed inside `_season_level_features()`
  as part of the v1 tunneling feature pipeline. Used in Tunneling+ v1 (`tunneling_model.py`).
- `training/part_b_era_resume.py` and `training/part_b_sequencing_plus_v1.py` — used in
  exploratory sequencing research (not a live model).
- `docs/methodology/HANDOFF.md`, old diagnostic reports.

**Live models that use it**: Tunneling+ v1 only (which is no longer the live scorer; it still
serves the Tunneling+ 3D viewer via `api/routers/tunneling_3d.py`).

**NOT used** in the live: Stuff+ v3, Location+ v2, Tunneling+ v2, Deception+ v1. None of these
models include arsenal_entropy or any usage-mix feature. Source: confirmed by reading each
model's `ALL_FEATURES` / `feat_cols` list.

---

## 8. Reliability by sample size

**Split-half curves**: none found as stored artifacts in this read-only audit. The HANDOFF.md
reports split-half reliability for Tunneling+ v2 as ~0.85–0.91 (source: HANDOFF.md line 169)
and notes that Stuff+ stabilizes at r=0.98+ even at n=20 (source: `analytics/stuff_model.py:218`
docstring). No JSON artifacts with explicit split-half-by-sample-size curves were found.

**Deferred** — see NEEDS_DECISION.md. Recommended action: run `tools/build_validation_content.py`
(Part E) which will pull these from existing artifacts and surface any gaps.

---

## 9. Reconciled Stuff+ stability figures

Four figures are in circulation. Here is each one's exact source and meaning:

| Figure | Source file | What it actually is |
|---|---|---|
| **r = 0.907** | `docs/methodology/HANDOFF.md:97` | Pearson r computed on consecutive-season pairs for the **live v3 model** (trained on all seasons, scored against full population). This is an in-sample YoY figure — the model has seen every season. |
| **r = 0.831 / 0.845** | `diagnostics/stuff_memorization/memorization_report.json` — `section_A.loso_yoy_mean = 0.8314` and `interpretation.loso_yoy = 0.845` | Leave-one-season-out YoY: model retrained each time with that season held out, scored on the holdout season, then correlated with the following season. Slightly honest than 0.907 but still within-pitcher (the same pitcher appears in the training set under other seasons). Mean of individual season-pair LOS correlations listed in `loso_yoy_reference`. |
| **r = 0.558** | `diagnostics/stuff_memorization/memorization_report.json` — `section_A.yoy_mean_r = 0.5577` | Leave-one-pitcher-out YoY (LOPO): scores are from the 5-fold CV model which never trained on the scored pitcher's own data at all. Mean across 2016–2025 consecutive-season pairs. This is the most honest generalization figure; range 0.484–0.619 by pair. 95% CI: [0.538, 0.587]. |
| **r ≈ 0.938** | **Not confirmed as a stability figure** | The value 0.938 appears in `training/v2_artifacts/cross_model_validation_v2_checkpoint.json` only within arrays of individual pitcher Stuff+ score values (scores near 93.8 on the plus scale), not as a named summary statistic. See NEEDS_DECISION.md ND-04. |

**Plain-language summary**: The "0.907" stability figure in the HANDOFF is real but represents
the in-sample model (the model has seen those seasons). The honest out-of-sample figure is 0.558
(LOPO). The LOSO figure 0.831 is intermediate — the model saw other seasons from the same
pitcher. All three are legitimate but answer different questions.

---

## 10. One-table summary per model

| | Stuff+ v3 | Location+ v2 | Tunneling+ v2 | Deception+ v1 |
|---|---|---|---|---|
| What it measures | Pitch physical shape: velo, movement, spin, release speed | Where the pitch was thrown relative to the zone | How indistinguishable pitches look from each other at the commit point | Pitcher body and release consistency |
| Booster(s) | `stuff_plus_v3_lgbm.txt` | 5 classifiers `location_plus_v2_*.txt` | 5 classifiers `tunneling_plus_v2_*.txt` | `deception_plus_v1.txt` |
| Training seasons | 2016–2026 | 2015–2026 | 2015–2026 | ~2021–2026 (arm_angle constraint; see ND-02) |
| Available scoring seasons | 2016–2026 | 2015–2026 | 2015–2026 | 2026 only (no selector wired) |
| Score timing | Request-time (600s cache) | Request-time (600s cache) | Request-time (600s cache) | Request-time (600s cache) |
| Min pitches (scoring) | None (leaderboard: 200) | 100 | 100 (raw, pre-lag) | None explicit (see ND-03) |
| Shrinkage | Yes (EB per pitch type) | No | No | No |
| YoY stability | r=0.907 in-sample; r=0.558 LOPO (honest) | Sign-flip fixed in v2; v1 collapsed to r=0.009 | r²=0.28–0.35 (v2) | r²=0.17–0.34 (least stable) |
| Entropy used | No | No | No | No |
| Season categorical | Yes (2016–2026) | Yes (2015–2026) | Yes (2015–2026) | Yes (2015–2026 in meta, eff. 2021+) |
