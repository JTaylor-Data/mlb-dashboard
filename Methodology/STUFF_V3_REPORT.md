# Stuff+ v3 — LightGBM + GAM comparison report

Offline comparison only. Neither model has been promoted to production —
`models/stuff_plus_v2.txt` is untouched, `api/routers/` was not modified.
Artifacts: `models/stuff_plus_v3_lgbm.txt` (+ `_meta.json` / `_report.json`),
`models/stuff_plus_v3_gam.pkl` (+
`training/stuff_v3_artifacts/gam_partial_dependence.json`), and the
slot-count ablation record (Part C, historical):
`training/stuff_v3_artifacts/slot_ablation_results.json`.

## Final architecture: primary-only, no threshold (locked in)

**This report was updated after a direct architecture decision.** Part C's
ablation (below) found that a single reference fastball beat both 2-slot
and 3-slot schemes on year-over-year score stability while matching them on
raw per-pitch accuracy. Based on that evidence, the reference classification
was simplified further: no dominance/qualify threshold at all — a
pitcher-season's `primary_fb_type` is simply whichever of FF/SI/FC he threw
most that season, full stop. This closes the small (~0.2%) "no qualifying
reference" edge case entirely (every pitcher-season with at least one
FF/SI/FC pitch now gets a reference — confirmed empirically below: 100.0%
coverage across all 4,853 pitcher-seasons) and removes two now-unused
constants (`DOMINANT_USAGE_THRESHOLD`, `QUALIFY_USAGE_THRESHOLD`) along with
all `secondary_fb`/`tertiary_fb` machinery from `analytics/
stuff_features_v3.py`.

**Everything below Part A/B was re-run against this final architecture**:
the feature table was rebuilt (`training/build_stuff_v3_dataset.py`), the
LightGBM model was retrained (`training/train_stuff_v3_lgbm.py`), and the
validation battery was rerun (`training/validate_stuff_v3.py`). Part C is
kept as a historical record of the ablation that motivated this decision,
but its own "1-slot" numbers used the *older*, threshold-based primary
classification (60%-dominant, no fallback below 10%) — not this final
no-threshold rule. The two are close in practice (YoY r² mean 0.752 now vs.
0.754 in the ablation) since coverage only changed by ~0.2 percentage
points, but they are not the identical run; see Part C's closing note.

**Identifier-leakage guard**: `assert_no_identifiers()` (banned set:
`pitcher_id`, `game_id`, `pitch_id`, `player_id`, `batter_id`) is now
checked immediately before every `.fit()` call in `train_stuff_v3_lgbm.py`
as well as in the ablation script — not just inspected once by eye. Raised
zero times during the final retraining run (confirmed clean).

**Feature set, confirmed against the actual saved table before training**:
**11 numeric** (`release_speed`, `effective_speed`, `release_spin_rate`,
`spin_axis`, `hb`, `ivb`, `vaa_residual`, `axis_differential`, and the 3
`{velo,hb,ivb}_sep_vs_primary_fb` separation features) **+ 4 categorical**
(`pitch_type`, `p_throws`, `stand`, `season`) = **15 total**.
`release_extension`, `release_pos_x_as`, `release_pos_z`, `arm_angle`
confirmed absent from the table (not just unused). `secondary_fb`/
`tertiary_fb` columns no longer exist anywhere in the pipeline.

**One flagged methodology deviation from v2's CV**: v2 refits its VAA
regression per CV fold to prevent held-out-season leakage; v3 does the
same for `vaa_residual`, but reuses the precomputed fastball-reference
classification and spin-axis calibration across all folds (built on all 6
seasons combined). Judged low-risk — both are structural/physical facts,
not parameters fit to the run-value target — but not silently glossed over.

---

## Part A: LightGBM validation battery (final, primary-only architecture)

### Cross-validation: v3 vs. v2, same folds, same holdout seasons

| Holdout | v2 R² | v2 MAE | v3 R² | v3 MAE |
|---|---|---|---|---|
| 2021 | 0.0011 | 0.1233 | 0.0007 | 0.1234 |
| 2022 | 0.0007 | 0.1172 | 0.0004 | 0.1173 |
| 2023 | 0.0007 | 0.1220 | 0.0007 | 0.1221 |
| 2024 | 0.0002 | 0.1191 | 0.0002 | 0.1192 |
| 2025 | 0.0005 | 0.1190 | 0.0001 | 0.1199 |
| **Mean** | **0.00065** | **0.12014** | **0.00042** | **0.12038** |

Per-pitch R² this small is expected and matches the project's established
pattern (single-pitch outcomes are noise-dominated; see Deception+/
Tunneling+'s own per-pitch CV). **v3 is marginally, consistently a touch
behind v2 on raw per-pitch predictive accuracy** — removing 4 real
predictive features (mechanics/delivery properties, now Deception+-
exclusive) cost slightly more than the new features recovered. This is an
honest, expected tradeoff given the whole point of v3 was an architectural
change (removing feature overlap with Deception+), not a pure accuracy
play — but it is a real cost, not zero.

### Year-over-year score stability: v3 is *more* stable than v2, despite the small CV R² dip

This measures something different from the CV table above — not "how well
does the model predict a held-out pitch," but "does a pitcher's season
SCORE predict his own score the following year" (the real trait-stability
question).

| Seasons | v2 r² | v3 r² |
|---|---|---|
| 2021→2022 | 0.615 | **0.735** |
| 2022→2023 | 0.669 | **0.752** |
| 2023→2024 | 0.581 | **0.720** |
| 2024→2025 | 0.605 | **0.772** |
| 2025→2026 | 0.592 | **0.780** |
| **Mean** | **0.612** | **0.752** |

v3 beats v2 on **every single season pair**, by a 0.12–0.19 margin — a
wider gap than the earlier 17-feature version showed (that version's mean
was 0.720; primary-only closes another ~3 points). This is a genuinely
notable finding given v3's CV R² was slightly *worse*: it suggests the
new feature set produces a score that's less sensitive to single-pitch
noise at the season-aggregate level, even though individual pitch
predictions are marginally less accurate. Plausible mechanism: the
fastball-separation features compare a pitcher against his OWN arsenal
(an inherently more stable, self-referential quantity than raw
release-point mechanics, which can shift year to year for reasons
unrelated to true pitch quality) — and, per Part C, dropping the noisier
secondary/tertiary references removes exactly the instability they were
independently shown to add.

### By pitch type, 2025

| Type | v2 avg | v3 avg | n pitches | n pitchers |
|---|---|---|---|---|
| FF | 100.4 | 100.5 | 225,618 | 758 |
| SI | 101.2 | 101.5 | 109,895 | 570 |
| SL | 101.8 | 102.3 | 101,825 | 569 |
| CH | 98.3 | 97.4 | 73,047 | 540 |
| CU | 96.1 | 95.5 | 59,994 | 407 |
| ST | 100.7 | 101.2 | 54,304 | 356 |
| FC | 100.3 | 100.6 | 53,308 | 356 |
| FS | 100.8 | 100.6 | 23,110 | 164 |
| EP | 46.0 | **24.8** | 951 | 57 |
| FA | 52.8 | 39.1 | 900 | 51 |
| KN | 72.0 | 59.5 | 137 | 11 |

Common arsenal pitches: nearly identical between versions, as expected
(standardization is league-wide, not per-type). The rare/novelty codes
(eephus, unclassified, knuckleball — almost entirely position players in
mop-up innings or true specialists, same story as v2) score **more
extreme** under v3, not less. Worth noting, not over-interpreting given
tiny samples (57 pitchers on EP).

### Cross-model correlation: the core architectural prediction, confirmed

| Pair | v2 | v3 |
|---|---|---|
| Stuff+ vs. Deception+ | 0.535 | **0.377** |
| Stuff+ vs. Tunneling+ | 0.120 | 0.128 |

This is the single cleanest piece of evidence in this whole report.
Removing the four features literally shared with Deception+
(`release_extension`, `release_pos_x_as`, `release_pos_z`, `arm_angle`)
dropped the Stuff+/Deception+ correlation by 0.158 — almost exactly what
you'd expect from eliminating direct feature overlap while leaving the
genuine underlying relationship (better-stuff pitchers tend to have more
repeatable deliveries) intact. Tunneling+'s correlation is essentially
unchanged, as expected — it never shared features with Stuff+ either way.

### Collinearity check (run explicitly, before trusting SHAP)

**`axis_differential` vs. `hb`/`ivb`/`spin_axis`** — modest, not
alarming, and unchanged by the primary-only simplification (this
relationship doesn't depend on the fastball-reference scheme at all):

| | axis_differential | hb | ivb | spin_axis |
|---|---|---|---|---|
| axis_differential | 1.000 | -0.192 | -0.335 | -0.083 |
| hb | -0.192 | 1.000 | 0.485 | 0.369 |
| ivb | -0.335 | 0.485 | 1.000 | 0.289 |

The strongest relationship (-0.335 with `ivb`) is real but nowhere near
the r=0.79 that caused actual credit-splitting in Deception+'s
arm_angle/release_pos_z case. Confirmed further by the GAM in Part B.

**`velo/hb/ivb_sep_vs_primary_fb` vs. their own raw counterparts** — now
that there's only one slot, cross-slot correlation (Part A's original
collinearity concern) no longer applies by construction. What remains is
each separation feature's correlation with the raw signal it's built
from:

| | velo_sep | hb_sep | ivb_sep | hb | ivb | rel_speed | eff_speed |
|---|---|---|---|---|---|---|---|
| velo_sep_vs_primary_fb | 1.000 | 0.553 | 0.678 | 0.567 | 0.702 | 0.875 | 0.867 |
| hb_sep_vs_primary_fb | 0.553 | 1.000 | 0.285 | **0.894** | 0.464 | 0.482 | 0.480 |
| ivb_sep_vs_primary_fb | 0.678 | 0.285 | 1.000 | 0.415 | **0.853** | 0.535 | 0.525 |

Each separation feature correlates most strongly with the raw metric it's
literally derived from (`hb_sep` with `hb`, `ivb_sep` with `ivb`,
`velo_sep` with `release_speed`/`effective_speed`) — expected by
construction, not a new finding, since `X_sep = X - primary_reference_X`.
This is a different kind of relationship than the old cross-slot problem
(two features both trying to explain the same target-relevant variance);
here one feature is a linear transform of another, and LightGBM's
tree-splitting handles that natively without the redundant-credit failure
mode collinear *target-relevant* features can cause. SHAP/gain rankings
below back this up: `velo_sep_vs_primary_fb` still lands solidly mid-pack
(8th of 15 by SHAP), not buried by its correlation with `release_speed`.

### SHAP / gain importance, final model (all 15 features, ranked)

| Rank | Feature | Mean \|SHAP\| | Gain |
|---|---|---|---|
| 1 | effective_speed | 0.00219 | 719.8 |
| 2 | ivb | 0.00178 | 800.6 |
| 3 | hb | 0.00147 | 613.7 |
| 4 | **axis_differential** | **0.00125** | **507.6** |
| 5 | pitch_type | 0.00115 | 324.4 |
| 6 | release_spin_rate | 0.00113 | 489.9 |
| 7 | vaa_residual | 0.00092 | 695.8 |
| 8 | velo_sep_vs_primary_fb | 0.00078 | 422.7 |
| 9 | spin_axis | 0.00054 | 474.0 |
| 10 | release_speed | 0.00040 | 434.9 |
| 11 | season | 0.00039 | 232.4 |
| 12 | stand | 0.00037 | 102.2 |
| 13 | ivb_sep_vs_primary_fb | 0.00032 | 349.0 |
| 14 | p_throws | 0.00029 | 40.4 |
| 15 | hb_sep_vs_primary_fb | 0.00021 | 309.0 |

`axis_differential` ranks 4th of 15 by both measures — still genuinely
strong signal, consistent with the earlier 21-feature version (5th of 21
there). The separation features rank 8th/13th/15th — real but modest
contributors, with `velo_sep_vs_primary_fb` (the best-populated, since
velocity separation is meaningful even for pitchers whose secondary
shapes were previously sparse) clearly the strongest of the three.

### Named-pitcher SHAP spot-checks

Three real pitchers (2026 season pitches where available, else 2025),
chosen to sanity-check the model isn't doing anything degenerate on known
real arsenals:

**Clay Holmes** (real sinkerballer, primary_fb=SI, 1,063 2026 pitches) —
top drivers: `axis_differential` (-0.0027), `vaa_residual` (-0.0017),
`hb` (-0.0010), `ivb` (+0.0009). `velo_sep_vs_primary_fb` appears at
rank 8 (+0.0003) — small but present, expected since most of his SI
pitches self-exclude (they ARE his own reference) and it's his
non-sinker pitches driving what separation signal exists.

**Jacob Misiorowski** (elite velo, primary_fb=FF despite real cutter
usage, 2,178 2026 pitches) — dominated by `effective_speed` (-0.0091, far
larger than anything else), then `release_spin_rate`, `ivb`, `hb`. No
separation feature in his top 8 — expected, same self-exclusion logic
(most of his pitches are his own FF reference).

**Sonny Gray** (sinkerball-primary, 2,019 2026 pitches) —
`velo_sep_vs_primary_fb` appears at rank 5 (+0.0006), the most prominent
separation-feature showing of the three spot-checks — plausible given his
real, well-documented multi-pitch mix relative to his sinker.

**Plain finding**: nothing degenerate turned up. Each pitcher's top
drivers make physical sense given their known real arsenal, and
`axis_differential`/`velo_sep_vs_primary_fb` show up where expected
without one feature obviously swallowing another's credit.

---

## Part B: GAM diagnostic companion

**Tool**: `pygam` (`LinearGAM`), chosen over `rpy2`/`mgcv` for setup
simplicity — pure pip install, no R runtime needed in this environment.
Installing it downgraded `scipy` 1.17.1→1.16.3 (a pygam dependency pin);
verified the rest of the live app still imports and runs correctly under
1.16.3 before proceeding.

**Fit on a 500,000-row random sample**, not the full 4.1M — penalized-
spline fitting doesn't scale the way LightGBM's histogram algorithm does,
and this is explicitly a diagnostic, not a competing production
candidate. Fit completed in 127s. Default smoothing throughout, not
tuned. **Not rerun for the primary-only simplification** — the GAM's
purpose was to independently corroborate `axis_differential`'s signal
against hb/ivb, a question the primary-only change doesn't touch (the
separation-feature scheme changed; axis_differential's own construction
did not). The findings below stand as originally measured, on the
17-feature table that included secondary/tertiary separation features
alongside axis_differential; a GAM refit on the final 15-feature table
was judged not to change the axis_differential conclusion and wasn't run,
to avoid an unnecessary second multi-minute P-IRLS fit.

**Missing-value handling for the separation features (this model
only)**: median imputation + a companion binary `_was_missing` flag term
per feature, so the GAM can distinguish "no reference existed" from "a
real computed value." This is a real, stated distortion — LightGBM's
native NaN-split handling in Part A does not have this problem.

**Overall fit**: Pseudo R² = 0.0014 — about 3x LightGBM's CV R² in the
same ballpark of "very small," expected for a much simpler additive model
against this noise-dominated target. Not a competing accuracy claim.

### The explicit axis_differential vs. hb/ivb comparison

This is the main reason Part B exists — reported as its own subsection,
not left implicit.

| Feature | EDoF | p-value* |
|---|---|---|
| spin_axis | 18.3 | 0.75 |
| release_speed | 17.7 | 0.0017 |
| **axis_differential** | **17.5** | **9.8e-07** |
| hb | 16.0 | 0.0033 |
| ivb | 16.0 | 2.4e-14 |

*pygam's own summary() output carries an explicit warning that these
p-values run systematically too low and shouldn't be used for hard
significance claims — EDoF (effective degrees of freedom, i.e. how much
real curve complexity the smoother is using) is the more trustworthy
number here, p-values are a secondary signal only.

**Answer to the direct question asked**: axis_differential's EDoF (17.5)
is NOT near zero once hb/ivb are already in the model — it's actually
*higher* than either hb's or ivb's own EDoF (16.0 each). This is the
opposite of what a "fighting for the same signal" collinearity problem
would look like (that would show up as axis_differential's EDoF collapsing
toward the intercept, ~0-1, the same way every `_was_missing` flag term
did). Corroborated independently by Part A: axis_differential ranks 4th
of 15 features in both LightGBM gain importance and mean |SHAP| in the
final primary-only model (was 5th of 21 in the original 17-feature
version) — ahead of `release_speed`, `spin_axis`, and every separation
feature, in both versions. Two independent methods agree: this is real,
distinguishable signal, not overlap.

axis_differential's fitted partial-dependence shape is non-monotonic and
modest in magnitude (roughly -0.025 to +0.014 in raw delta_run_exp units
across its range) — a real but not enormous effect, consistent with a
genuine physical signal rather than a dominant driver.

### The `_was_missing` flags: reassuring, not concerning

Every separation feature's missingness-flag term came back with EDoF ≈
0–1 and a fitted effect size of roughly ±0.0002 to ±0.0006 — negligible
relative to the model's typical prediction scale. The GAM essentially
found nothing systematic in "did this reference exist or not" beyond what
the imputed/real values themselves already carry. This doesn't erase the
stated distortion (median-imputing a meaningful fraction of a feature's
values is still real information loss, and this diagnostic can't fully
undo that), but it is evidence the imputation choice isn't visibly
injecting a large, fake pattern into the GAM either.

---

## Part C: 1-slot vs. 2-slot vs. 3-slot ablation (historical — the evidence behind the final decision)

The original dataset-construction session deferred exactly this question:
does adding secondary/tertiary fastball references actually help, or is
primary-only enough? Run with the same CV + year-over-year harness as
Parts A/B, same LGBM_PARAMS, same per-fold VAA refit. **Note**: this
ablation used the *older*, threshold-based primary classification
(60%-dominant / 10%-qualify) — run before the final no-threshold
simplification described at the top of this report. Its "1-slot" result
is therefore not byte-for-byte the same run as Part A's final model above,
though the two land in the same neighborhood (YoY r² mean 0.754 here vs.
0.752 for the final retrained model), which is itself a useful
confirmation that removing the threshold didn't change the underlying
conclusion. **Identifier-leakage guard**: every feature list is checked
against a banned-identifier set (`pitcher_id`, `game_id`, `pitch_id`,
`player_id`, `batter_id`) immediately before every `.fit()`/`.predict()`
call, not just inspected once — confirmed to raise correctly on a real
leakage attempt before this run, and raised zero times during the actual
run (all three variants completed cleanly). `pitcher_id` is still used
elsewhere in this pipeline exactly as it was in Parts A/B — grouping
pitches into per-pitcher-season reference profiles, and aggregating
per-pitcher scores for the year-over-year check — that's data engineering
and evaluation bookkeeping, not a training feature, and was never in
scope for the guard to block.

| Variant | Features | CV R² | CV MAE | YoY r² mean |
|---|---|---|---|---|
| 1-slot (primary only) | 11 | 0.00041 | 0.12037 | **0.754** |
| 2-slot (primary+secondary) | 14 | 0.00045 | 0.12039 | 0.692 |
| 3-slot (all three, original model) | 17 | 0.00044 | 0.12040 | 0.720 |

**Per-pitch CV accuracy is flat across all three** — the differences
(0.00041 to 0.00045) are noise at this scale, confirming what the
original 17-feature model's cross-slot correlation matrix (0.65–0.85)
already suggested: secondary and tertiary separation features carry very
little NEW predictive information once primary is already in the model.

**Year-over-year stability is not flat, and not monotonic** — this is
the real finding. Going from 1-slot to 2-slot **drops** stability
substantially (0.754→0.692), and 3-slot partially recovers some of that
(0.692→0.720) but still ends up worse than 1-slot alone. The most likely
mechanism: secondary/tertiary reference profiles are built from smaller,
noisier per-pitcher-season samples (by construction — under the old
threshold rule they only existed at all for 47%/9% of pitcher-seasons,
and often from a pitcher's least-used fastball type when they do). That
noise doesn't meaningfully hurt per-pitch accuracy (LightGBM has 4M+ rows
to average it out) but it does appear to inject real season-to-season
instability into the resulting player-level score, which the CV metric
alone would never have caught — exactly why the year-over-year check was
worth running as its own thing rather than trusting CV R² as the only
accuracy signal.

Gain importance also told a consistent story in the original run: in
2-slot, `velo/hb/ivb_sep_vs_primary_fb`'s own gain roughly **doubles**
versus 1-slot (365→642 for velo) rather than getting "split" with the new
secondary features — the model isn't dividing a fixed pool of credit,
it's finding more (but apparently mostly redundant, per the flat CV R²)
ways to split on similar information once secondary is available.

**This was the direct evidence behind the final decision**: primary-only
(1-slot) is the strongest of the three — same raw predictive power,
meaningfully better year-over-year stability, and the simplest feature
set of the three. Adding secondary/tertiary references doesn't just fail
to help, it measurably hurts the metric this session's earlier work
identified as the more decision-relevant of the two. Per direct
instruction following this result, the reference classification was
simplified one step further (no threshold, always pick the single
highest-usage fastball) and the final model above was trained and
validated under that rule.

---

## Plain-language summary: is the new feature set adding real signal?

**Yes, with one clear caveat.** This is now describing the final,
locked-in primary-only architecture (11 numeric + 4 categorical = 15
total features), not the original 17-feature build.

- **`axis_differential` (the SSW proxy) is the strongest new-feature
  result in this report.** Two independent methods — LightGBM's own gain/
  SHAP importance (ranks 4th of 15 features in the final model) and the
  GAM's effective degrees of freedom (higher than hb's or ivb's) — agree
  it's carrying real, distinguishable signal, not a shadow of existing
  movement/spin features. The by-pitch-type pattern from the original
  dataset-construction session (FF lowest, sinkers/sliders elevated,
  matching known public SSW research) was already a good sign; this
  session's model-level evidence backs it up independently.
- **The separation features are real, modest contributors** —
  `velo_sep_vs_primary_fb` (8th of 15 by SHAP) is doing most of the real
  work of the three; `hb_sep_vs_primary_fb`/`ivb_sep_vs_primary_fb` are
  weaker (13th/15th) but not negligible. The original secondary/tertiary
  slots were dropped entirely, per Part C's ablation — they were shown to
  add real season-to-season instability without improving per-pitch
  accuracy, so simplifying to primary-only was a strict improvement, not
  a tradeoff.
- **The core architectural goal succeeded cleanly**: Stuff+/Deception+
  correlation dropped from 0.535 to 0.377, almost exactly matching what
  removing four literally-shared features should do, with Tunneling+'s
  correlation (already low) essentially unchanged.
- **The real cost**: a small, consistent per-pitch CV accuracy dip
  (0.00065→0.00042 R², both tiny in absolute terms) — offset by a
  meaningfully MORE stable year-over-year season score (v3 beats v2 on
  every single season pair, 0.72–0.78 vs. 0.58–0.67 r², mean 0.752 vs.
  0.612). Whether that trade is worth promoting to production is a
  separate decision from the architecture question this report answers.

**Final architecture, trained and validated**: `release_speed`,
`effective_speed`, `release_spin_rate`, `spin_axis`, `hb`, `ivb`,
`vaa_residual`, `axis_differential`, `velo_sep_vs_primary_fb`,
`hb_sep_vs_primary_fb`, `ivb_sep_vs_primary_fb` (11 numeric) +
`pitch_type`, `p_throws`, `stand`, `season` (4 categorical). Saved as
`models/stuff_plus_v3_lgbm.txt` / `_meta.json` / `_report.json`. Whether
to promote this to production (replacing `models/stuff_plus_v2.txt` and
wiring `api/routers/`) remains a separate decision, not made here.
