# Location+ v1 — build, validation, and design report

Fourth model in the "+" family (Stuff+, Deception+, Tunneling+, **Location+**).
Trained, not hand-built, against the same `delta_run_exp` target as the
other three, on the same plus-scale convention (100 = league average that
season, 10 points = 1 standard deviation). Offline build only — no
live-serving module was created, no existing model file was touched, and
nothing was pushed to production. This report is the record of what was
built and what the validation battery actually found.

## Design summary

Location+ measures where a pitch ended up and how valuable that location
was, **given the count** — nothing about physical shape (Stuff+), delivery
(Deception+), or arsenal relationships (Tunneling+). Follows FanGraphs'
own public Location+ methodology (their pitching primer): location, pitch
type, and count only, explicitly excluding velocity/movement/spin.

**Features (7 total)** — numeric: `plate_x`, `plate_z`, `balls`, `strikes`;
categorical: `pitch_type`, `stand`, `season`. `p_throws` and every
physical/mechanical/tunneling feature from the other three models are
excluded by design (see `analytics/location_features.py`'s module
docstring for the full list confirmed absent).

**Deliberate deviation from FanGraphs**: `stand` is kept in (they exclude
it, reserving handedness for their combined Pitching+ stack) — a pitch's
location value is genuinely relative to which side of the plate the
batter's standing on. `p_throws` stays excluded, following FanGraphs'
reasoning that pitcher-handedness-alone is a platoon/matchup concept, not
a location property. **This deviation was tested empirically, not just
argued for** — see §6.

**The accepted limitation**: Statcast only captures where a pitch actually
crossed, not catcher target or pitcher intent — this model measures
location-VALUE-ACHIEVED, conflating real command with outcome luck. Per
FanGraphs' own public writeup, they tested intent-based (stringer) data for
this exact purpose and found it added no predictive value over real-location
data alone. Cited here as precedent for accepting the same limitation
rather than treating it as a gap needing a data source this project doesn't
have either.

Same pitch-type exclusion as every other model here: `PO`/`UN`/`EP`
excluded (imported `EXCLUDED_PITCH_TYPES` from `analytics/stuff_features.py`
directly, not redefined), `KN` kept.

## 1. Model performance

| | Value |
|---|---|
| Training rows | 4,135,036 |
| CV mean R² (leave-one-season-out, 2021-2025) | **0.0333** |
| CV mean MAE | 0.1043 |
| Best iteration (final model, all seasons) | 1,977 |

**CV folds:**

| Holdout | R² | MAE | n |
|---|---|---|---|
| 2021 | 0.0356 | 0.1067 | 709,111 |
| 2022 | 0.0329 | 0.1012 | 707,601 |
| 2023 | 0.0329 | 0.1061 | 717,062 |
| 2024 | 0.0332 | 0.1033 | 708,591 |
| 2025 | 0.0318 | 0.1042 | 708,851 |

**This R² is roughly 4-5x higher than Stuff+'s or Tunneling+'s (~0.007
each)** — expected, not a fluke: where a pitch crosses the plate mechanically
determines a huge share of its outcome (a meatball down the middle vs. a
ball well off the plate is an enormous, direct difference), so a model with
direct access to `plate_x`/`plate_z` should out-predict models built from
indirect physical-quality proxies. A useful sanity check in its own right —
if Location+'s R² had come back similar to or lower than Stuff+'s, that
would have been a red flag that the feature engineering was broken.

**SHAP feature importance** (30k-pitch sample, mean |SHAP|):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `plate_x` | 0.02004 |
| 2 | `plate_z` | 0.01677 |
| 3 | `balls` | 0.00725 |
| 4 | `strikes` | 0.00485 |
| 5 | `pitch_type` | 0.00442 |
| 6 | `stand` | 0.00197 |
| 7 | `season` | 0.00118 |

Location dominates (as it should — this is a location model), count
matters meaningfully more than pitch-type identity, and `stand` carries a
real, if modest, independent contribution.

## 2. Empirical-Bayes shrinkage — Location+-specific K's, not Stuff+'s

Estimated fresh from Location+'s own predictions (same method-of-moments
methodology as Stuff+'s, deliberately different numeric constants):

| Pitch type | K (pitches) |
|---|---|
| CS | 10.9 |
| KN | 12.5 |
| FA | 44.8 |
| FS | 64.5 |
| CU | 78.7 |
| CH | 78.9 |
| SV | 86.0 |
| ST | 90.0 |
| SL | 112.7 |
| SI | 140.1 |
| FC | 142.0 |
| FF | 164.6 |
| **fallback** | **82.5** |

**Confirmed, not assumed**: these are dramatically larger than Stuff+'s own
K's (0.4–2.8 across every well-populated pitch type). Reusing Stuff+'s K's
here would have under-shrunk small-sample Location+ estimates badly — a
pitcher with 30 fastballs would be treated as almost fully reliable under
Stuff+'s K≈2, when Location+'s own data says a fastball sample needs
something like 165 pitches before it's even half-weighted toward the
individual pitcher's own mean rather than the league mean.

## 3. Stabilization — Location+ vs. Stuff+, identical thresholds, same infrastructure

The central empirical question. Split-half reliability computed the same
way for both models (unshrunk, pooled-across-pitch-type, random per-pitcher
halves) at identical per-half pitch-count thresholds, so this is a true
apples-to-apples comparison, not a citation of an external number:

| Per-half threshold | Location+ r | Stuff+ r |
|---|---|---|
| 50 | 0.597 | 0.992 |
| 100 | 0.585 | 0.992 |
| 200 | 0.611 | 0.994 |
| 400 | 0.686 | 0.996 |
| 600 | 0.794 | 0.995 |

**Crossing points:**

| | r ≥ 0.7 | r ≥ 0.9 |
|---|---|---|
| Location+ | 600 pitches/half | not reached within tested range |
| Stuff+ | 50 pitches/half (already crossed at the lowest threshold tested) |

**Confirms FanGraphs' finding directly on this project's own data**:
Location+ takes dramatically longer to stabilize than Stuff+ — at the
same 50-pitch-per-half sample, Stuff+ is already essentially fully
reliable (r=0.99) while Location+ is still around r=0.60. Even at 600
pitches per half (1,200 total pitches that season), Location+ hasn't
reached the 0.9 bar Stuff+ clears almost immediately. This has a direct,
practical consequence: **a Location+ leaderboard needs a much higher
minimum-pitch bar than Stuff+'s to be trusted**, and even the 100-pitch
qualifying threshold used for this report's own leaderboard (§7) should be
read as "a very noisy read," not a stable rating, per this model's own
numbers.

**Year-over-year**: pooled r² = 0.368 (n=2,558) — meaningfully lower than
what's typical for Stuff+, consistent with the reliability finding above
(a noisier season-level signal is also a less year-over-year-sticky one).
Per-pair breakdown:

| Seasons | n | r² |
|---|---|---|
| 2021→2022 | 516 | 0.418 |
| 2022→2023 | 517 | 0.370 |
| 2023→2024 | 507 | 0.309 |
| 2024→2025 | 518 | 0.336 |
| 2025→2026 | 500 | 0.419 |

## 4. Cross-model correlation

| vs. | r | n=723 (2025, min 100 pitches) |
|---|---|---|
| Stuff+ | **-0.070** | expected — near zero, clean separation |
| Deception+ | 0.210 | modest, plausible (both partly reflect overall pitcher quality) |
| Tunneling+ | **0.563** | **surprising — flagged below** |

Stuff+ shows the expected near-independence — pure pitch shape and where
it ends up really are close to separate skills. **The Tunneling+
correlation (0.563) is the real, flagged finding here** — Location+ and
Tunneling+ v3 share zero raw features (Location+ never sees a pitch's
relationship to any other pitch; Tunneling+ v3 never sees absolute plate
coordinates as such), so a correlation this size wasn't expected and isn't
explained by any shared input. The most plausible mechanism: both models
are downstream of the same underlying **command** skill, measured two
different ways — Tunneling+ v3's `plate_distance` lag features are
literally built from consecutive pitches' `plate_x`/`plate_z` differences,
so a pitcher who commands the zone well enough to consistently hit good
locations (driving a high Location+) plausibly also commands it well
enough to create deliberate plate separation between pitch types (driving
a good Tunneling+ score) — the same underlying command ability surfacing
in both, not a shared measurement artifact. This is a hypothesis, not a
proven mechanism — worth investigating further if a Pitching+ stack is
ever built, since two "independent" inputs correlating this much has real
implications for how they'd combine.

**Fastest-fastball-velocity correlation (leakage sanity check)**: **r =
-0.152**. Small in absolute terms, but not zero, and the sign is worth
noting rather than waving away: this is small enough that it does not look
like the count/pitch-type categoricals are smuggling in a real velocity
signal (a leakage concern would show up much stronger than -0.15), but
plausibly reflects a real, different phenomenon — harder throwers may pitch
more aggressively in/around the zone (trusting velocity over precision),
trading some location value for stuff. Noted for transparency, not
investigated further this session.

## 5. Out-of-sample predictive validity

Model trained on every season except 2024, scored on 2024's pitches,
correlated against REAL 2025 outcomes (never saw either season) — same
methodology and same five outcomes as Stuff+'s and Tunneling+ v3's own
checks:

| Outcome (min 100 2024 pitches, n=304) | r vs. 2025 outcome |
|---|---|
| xwOBA | 0.030 |
| whiff% | **-0.308** |
| swstr% | **-0.219** |
| K-BB% | 0.031 |
| chase% | 0.083 |

**The negative whiff%/swstr% correlations are a real, flagged finding, not
a bug** — confirmed directly: `whiff_pct`/`swstr_pct` in this project's own
`analytics/percentiles.py` are plain rates where higher = more whiffs =
better for the pitcher, no hidden sign flip. So this is a genuine negative
relationship: pitchers who score BETTER on Location+ tend to have LOWER
whiff rates out of sample. Plausible reading: Location+ is trained on
`delta_run_exp`, which rewards ANY run-suppressing outcome (called strikes,
weak contact, groundouts), not specifically swinging strikes — a pitcher
who commands the zone for called strikes and soft contact can score very
well here while genuinely not missing many bats, a real, different
archetype from a swing-and-miss Stuff+ specialist. Notably, Tunneling+
v3's own predictive-validity report showed this exact same sign pattern
(whiff%: -0.1, swstr%: -0.026) — this isn't a one-off, it may reflect a
real property of how `delta_run_exp`-trained models relate to whiff-rate
specifically, worth keeping in mind for any future model built the same
way. xwOBA and K-BB% land close to zero out of sample — a real, honest
limitation of this proof-of-concept feature set (7 features, no physical
quality signal at all) worth stating plainly rather than dressing up.

## 6. The `stand`-inclusion deviation — does it earn its keep?

Direct ablation: an identical model with `stand` dropped entirely (the
FanGraphs-exact feature set), same CV scheme, same LGBM_PARAMS:

| | With `stand` (promoted) | Without `stand` (FanGraphs-exact) | Δ |
|---|---|---|---|
| CV R² | 0.0333 | 0.0313 | **+0.0020** |
| CV MAE | 0.1043 | 0.1046 | -0.0003 |
| YoY pooled r² | 0.368 | 0.303 | **+0.065** |

**Yes, clearly.** Keeping `stand` improves both raw per-pitch accuracy
(R² +0.0020, a ~6% relative gain — meaningful at this scale, where the
whole model's R² is only 0.033) and, more importantly, season-to-season
stability (YoY pooled r² +0.065, a ~21% relative gain). This wasn't a
marginal or ambiguous result either way — the deliberate deviation from
FanGraphs' public methodology is earning its keep on real validation
metrics, not just theoretically justified. The intuition holds up
empirically: inside/outside are genuinely meaningless without knowing
which side of the plate the batter's standing on, and the model measurably
uses that information.

## 7. Count/pitch-type interaction sanity check (Part D.5)

The central qualitative claim of this model: the same physical location
should be worth different amounts in different counts. Reproduced with a
sinker (`SI`) vs. right-handed batters, comparing the model's
location-value heatmap at 3-1 vs. 3-2. Full grid + PNG:
`training/location_v1_artifacts/location_v1_count_interaction_sinker_R.png`
and `..._count_interaction.json`.

**Grid correlation between the two counts: 0.987** — the two heatmaps
share almost the same overall SHAPE (middle-of-zone good, edges/corners
bad), which is exactly what should happen — the physical strike zone
doesn't change between counts. But the interaction claim isn't about
overall shape, it's about whether the *residual* differs by location, and
it does, non-trivially:

**3-2 minus 3-1** (positive = location got more valuable in the deeper
count), mean |diff| = 0.018, max = 0.049 across the grid:

```
z=4.00 (top)     -0.031 -0.015 +0.003 +0.008 +0.007 -0.012 -0.032
z=3.50            -0.005 +0.020 +0.034 +0.037 +0.043 +0.034 +0.001
z=3.00            +0.001 +0.005 -0.009 -0.012 +0.001 +0.036 +0.013
z=2.50            +0.011 +0.001 -0.022 -0.031 -0.015 +0.045 +0.016
z=2.00            +0.010 +0.010 -0.011 -0.019 +0.010 +0.032 -0.008
z=1.50            +0.002 +0.007 +0.018 +0.015 +0.020 +0.014 -0.019
z=1.00 (bottom)  -0.028 -0.020 -0.017 -0.014 -0.015 -0.023 -0.049
        (inside)                                          (away)
```

**Not a flat rescale — a real, location-specific interaction**: misses off
the edges (top row, bottom row) get systematically WORSE in the deeper 3-2
count (walk risk is more costly with two strikes already banked and one
ball from a free base either way — actually the more literal read: a ball
off the edge in 3-2 is one pitch from a walk, raising its cost). The
upper-middle band (z≈3.5, center-to-slightly-inside) gets systematically
BETTER in 3-2 — plausible mechanism: a hitter protecting the zone in a full
count is more likely to be geared up to swing at anything competitive, so
a well-placed pitch up in the zone plays up when the batter can't simply
take it. This is exactly the qualitative signature the check was meant to
confirm: the model has learned count matters in a location-specific way,
not just as a flat additive shift.

## 8. Leaderboard — 2025, min 100 pitches (n=725 qualified)

**Read this leaderboard with §3's stabilization finding in mind** — at 100
pitches, Location+'s own split-half reliability is only ~0.59, meaningfully
noisier than a Stuff+ leaderboard at the same threshold.

**Top 15:**

| Pitcher | Team | Score | Pitches |
|---|---|---|---|
| Tyler Mahle | ATL | 101.4 | 1,381 |
| Shota Imanaga | CHC | 101.2 | 2,176 |
| Trevor Williams | WSH | 101.1 | 1,448 |
| Janson Junk | MIA | 101.1 | 1,622 |
| Chris Paddack | TEX | 101.1 | 2,565 |
| Joe Ryan | MIN | 101.0 | 2,794 |
| Chris Martin | TEX | 101.0 | 638 |
| Jameson Taillon | TOR | 101.0 | 1,979 |
| Tomoyuki Sugano | COL | 101.0 | 2,565 |
| Steven Matz | TB | 101.0 | 1,159 |
| Hoby Milner | CHC | 101.0 | 1,157 |
| Andrew Kittredge | BAL | 101.0 | 768 |
| Pierson Ohl | COL | 100.9 | 544 |
| Yoshinobu Yamamoto | LAD | 100.9 | 2,789 |
| Jakob Junis | TEX | 100.9 | 1,032 |

**Bottom 15:**

| Pitcher | Team | Score | Pitches |
|---|---|---|---|
| Javier Sanoja | MIA | 91.1 | 169 |
| Jonah Bride | TEX | 92.2 | 114 |
| Matt Waldron | TOR | 96.7 | 104 |
| Brendon Little | TOR | 98.2 | 1,240 |
| Triston McKenzie | SD | 98.4 | 123 |
| Luis Mey | CIN | 98.6 | 414 |
| Justin Martinez | AZ | 98.6 | 269 |
| Fernando Cruz | NYY | 98.6 | 792 |
| Lucas Sims | CWS | 98.6 | 279 |
| Félix Bautista | BAL | 98.7 | 626 |
| Hayden Birdsong | SF | 98.7 | 1,194 |
| Jack Kochanowicz | LAA | 98.8 | 1,858 |
| Chayce McDermott | NYM | 98.8 | 195 |
| Michael Kopech | LAD | 98.9 | 231 |
| Eduarniel Núñez | CHC | 98.9 | 249 |

**Note on the bottom two names**: Javier Sanoja and Jonah Bride are
position players who made rare mound appearances (real, if small,
mop-up-inning samples in the `pitches` table) — the same known edge case
already documented for why Stuff+ excludes PO/UN/EP (position players
throwing genuinely uncompetitive pitches). Their real pitch types (FF,
etc.) don't fall under this model's own exclusion set, so they legitimately
surface at the bottom of any location-value leaderboard — expected, not a
data-quality bug, consistent with how this project has handled the same
underlying phenomenon elsewhere.

## 9. What was and wasn't touched

**Created**: `analytics/location_features.py`, `training/
train_location_model.py`, `training/validate_location_model.py`,
`training/location_predictive_validity.py`, `training/
location_count_interaction_check.py`, `training/
location_stand_ablation.py`, `models/location_plus_v1.txt` (+ `_meta.json`
/ `_report.json`), everything under `training/location_v1_artifacts/`.

**Confirmed untouched** (file timestamps checked directly against this
session's own work): `models/stuff_plus_v3_lgbm.txt`, `models/
deception_plus_v1.txt`, `models/tunneling_v3.txt`, and every file under
`analytics/`, `api/`, `frontend/` belonging to Stuff+, Deception+, or
Tunneling+. No live-serving module was created for Location+, no API
route was added, no frontend page was touched, no production push
occurred — this was an offline build, validation, and report only, per
the task.
