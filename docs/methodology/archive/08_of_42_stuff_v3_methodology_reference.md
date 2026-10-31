# Stuff+ v3 — Methodology & Architecture Reference

**Purpose of this document**: a standalone refresher on how the current,
live Stuff+ model actually works — not a validation report (see
`Methodology/STUFF_V3_REPORT.md` for the full CV/GAM/ablation battery this
architecture was decided against). Read this if you've come back to the
project after a while and need to reconstruct the mental model of what
Stuff+ v3 measures and how, without re-deriving it from code.

**Status**: v3 (primary-only, no dominance threshold) is the standard, live
architecture as of this writing. v2's pipeline is preserved, unused, in
`analytics/stuff_features_v2_legacy.py` / `analytics/stuff_model_v2_legacy.py`.

---

## 1. What Stuff+ is supposed to isolate

Stuff+ grades the **physical shape** of a pitch — velocity, movement, spin,
release characteristics — independent of *where* it was thrown or *what
count* it was thrown in. Two deliberate exclusions follow directly from that
definition:

1. **Plate location and count are never features.** Those measure command
   and sequencing — real, different skills, not pitch quality.
2. **Pitcher identity is never a feature.** A tree model that could see
   `pitcher_id` would memorize which players are good rather than learning
   *why* a given pitch shape is good — which wouldn't generalize to a pitch
   that hasn't been thrown yet. This is a hard rule enforced by
   `assert_no_identifiers()`, checked immediately before every `.fit()` call.

## 2. The feature set (15 total)

**11 numeric:**

| Feature | What it is |
|---|---|
| `release_speed` | Raw velocity |
| `effective_speed` | Perceived velocity (adjusted for release extension) |
| `release_spin_rate` | Raw spin rate |
| `spin_axis` | Measured spin-axis clock angle |
| `hb` | Horizontal break, arm-side-adjusted (positive = toward arm side, either hand) |
| `ivb` | Induced vertical break |
| `vaa_residual` | Vertical Approach Angle vs. what a height/velocity-only expectation would predict (see §3) |
| `axis_differential` | Seam-shifted-wake / active-spin proxy (see §4) |
| `velo_sep_vs_primary_fb` | This pitch's velo vs. this pitcher-season's own primary fastball reference (see §5) |
| `hb_sep_vs_primary_fb` | Same, for horizontal break |
| `ivb_sep_vs_primary_fb` | Same, for induced vertical break |

**4 categorical:** `pitch_type`, `p_throws`, `stand`, `season`.

Notably **absent**: `release_extension`, `release_pos_x_as`, `release_pos_z`,
`arm_angle` — these were in the v2 feature set but are delivery/mechanics
properties, not pitch-shape properties, and were moved to being
Deception+-exclusive (see §7 for why).

## 3. VAA residual

Raw Vertical Approach Angle (VAA) is confounded with plate height and
velocity — a pitch just looks flatter/steeper depending on where it crosses
and how fast it's going, independent of anything about "how well thrown" it
is. `vaa_residual` fits an expectation for VAA from `plate_z`, `plate_z²`,
and `release_speed` (least squares, refit per CV fold to prevent held-out-
season leakage), then reports the **residual** — how much flatter or steeper
than expected the pitch's actual approach angle was. This isolates the part
of VAA that's a genuine property of the pitch, not an artifact of where it
happened to cross the zone.

## 4. `axis_differential` — the seam-shifted-wake proxy

Under a pure-Magnus assumption, a pitch's movement is fully explained by its
measured spin axis. Real pitches deviate: some spin is gyroscopic (no
movement contribution), and some real movement comes from seam-shifted-wake
aerodynamics that have nothing to do with the measured axis at all.
`axis_differential` is the gap between "the spin axis pure Magnus physics
*would require* to produce this pitch's actual movement" and the pitch's
*real* measured `spin_axis` — a proxy for how much of the movement isn't
explained by simple backspin/topspin/sidespin, same idea as public "active
spin" research.

**Calibrated empirically, not from a documented convention** — Statcast's
spin-axis clock-face convention is publicly documented, but trusting a
from-memory recollection of the exact sign/direction mapping is exactly the
kind of thing that produces a silent bug. Instead the mapping is fit
directly from real four-seam fastball data (FF is the standard
near-fully-active-spin reference pitch in public research; sinkers are
excluded from calibration since they're a primary SSW hypothesis case, not a
clean reference). The calibration is correct regardless of whether it
matches Statcast's documented zero-point, because it's fit empirically
either way.

**Handedness**: `hb` is already arm-side-adjusted (RHP's raw `pfx_x` is
negated, LHP's is not). `spin_axis` is a fixed field-relative measurement,
not adjusted this way by Statcast — so it's mirrored with the exact same
asymmetric rule before comparison, otherwise `axis_differential` would
silently reintroduce the handedness leakage `hb`'s own adjustment exists to
prevent.

**Validated as real, distinguishable signal, not overlap with hb/ivb** —
two independent methods agree: LightGBM's own SHAP/gain importance (ranks
4th of 15 features) and a GAM diagnostic's effective-degrees-of-freedom
comparison (`axis_differential`'s EDoF is *higher* than either `hb`'s or
`ivb`'s own, the opposite of what a "fighting for the same signal" collinearity
problem would show). See `STUFF_V3_REPORT.md` Part B for the full comparison.

## 5. Primary-fastball separation features

The idea: a secondary pitch's effectiveness partly comes from how it
compares to the fastball a hitter is calibrated to expect from *that*
pitcher. For each pitcher-season, one **primary fastball reference** is
picked — whichever of FF/SI/FC has the **highest usage** that season, no
minimum-dominance threshold. Every other pitch that pitcher throws that
season gets three separation features against that one reference:
`velo_sep_vs_primary_fb`, `hb_sep_vs_primary_fb`, `ivb_sep_vs_primary_fb`
(this pitch's own value minus the reference's mean value).

**True NaN, never zero or imputed**, when: (a) the pitcher-season has no
FF/SI/FC at all (no reference exists), or (b) the pitch itself *is* the
primary reference type (a pitch can't be separated from itself). LightGBM's
native missing-value handling deals with this directly — these NaNs are
real, meaningful absences, not data-quality gaps.

**Why primary-only, no threshold, no secondary/tertiary slots** — an earlier
build tested up to 3 reference slots (primary/secondary/tertiary) gated by a
60%-dominant / 10%-qualify threshold rule. A direct ablation
(`training/ablate_stuff_v3_slots.py`) compared 1-slot vs. 2-slot vs. 3-slot
on identical cross-validation + year-over-year harnesses:

- **Per-pitch CV accuracy was flat across all three** — differences were
  noise (secondary/tertiary references carry very little *new* predictive
  information once primary is already in the model).
- **Year-over-year score stability was not flat** — 1-slot: 0.754, 2-slot:
  0.692, 3-slot: 0.720. Adding secondary/tertiary references didn't just
  fail to help — it measurably *hurt* the metric that actually matters for
  "is this a stable trait." The likely mechanism: secondary/tertiary
  profiles are built from smaller, noisier per-pitcher-season samples,
  injecting real season-to-season instability without buying back per-pitch
  accuracy.

Per that finding, the current architecture is strictly primary-only, no
threshold at all: whichever of FF/SI/FC was thrown most, full stop. This is
also the simplest of the three options tested — same predictive power, best
stability, least code.

## 6. Junk pitch-type exclusion

`PO` (pitchout), `UN` (Statcast's own "unclassified" code), and `EP`
(eephus) are excluded from **both training and scoring** populations
entirely — not just hidden from display. Confirmed empirically before
excluding: these score systematically far above the population mean
predicted run value (they aren't real competitive pitch-quality attempts),
and including them in the season mean/std population was measuring "how bad
is a pitch that was never trying to be good," not stuff. Dropping them
shifted the season standard deviation down 2–4%, a real standardization
effect.

`KN` (knuckleball) is **deliberately kept** — a real, competitive pitch
type with a known, documented modeling difficulty (genuinely
hard-to-predict movement most Stuff+-style models struggle to grade well),
not junk data the way the three above are.

## 7. Why v3 exists at all — the Deception+ overlap problem

v2's feature set included four delivery/mechanics properties
(`release_extension`, `release_pos_x_as`, `release_pos_z`, `arm_angle`) that
were *also* features in Deception+. That overlap meant the two models
weren't measuring cleanly separate things — a pitcher scoring well on one
partly reflected the same underlying inputs driving the other. v3 removes
those four features from Stuff+ entirely (they remain Deception+-exclusive)
and adds the separation features + `axis_differential` in their place.

**Confirmed to work as intended**: Stuff+/Deception+ correlation dropped
from 0.535 (v2) to 0.377 (v3) — almost exactly what removing four literally-
shared features should produce, while the genuine underlying relationship
(better-stuff pitchers do tend to have somewhat more repeatable deliveries)
remains, appropriately, not fully zeroed out. Tunneling+'s own correlation
with Stuff+ was essentially unchanged (it never shared features with either
model).

**The honest cost**: a small, consistent per-pitch CV accuracy dip (mean R²
0.00065 → 0.00042, both tiny in absolute terms — single-pitch outcomes are
inherently noise-dominated). Offset by a meaningfully *more* stable
year-over-year season score (mean r² 0.612 → 0.752, v3 wins every single
season pair). The tradeoff was judged worth it — the whole point of v3 was
architectural separation from Deception+, and the trait-stability gain
suggests the new feature set isn't just different, it's a genuine
improvement at the season-score level.

## 8. Scoring: shrinkage, then the plus-scale

Two steps convert a raw per-pitch prediction into the number shown on the
site:

1. **Empirical-Bayes shrinkage toward the pitch type's league mean.** A
   pitcher's raw mean prediction for a pitch type is noisy when the sample
   is small (natural pitch-to-pitch variability, not outcome luck — `pred`
   is the model's prediction, not the real outcome). Shrinkage constants
   (`K`, in units of "pitches") are estimated fresh from each season's own
   data via a method-of-moments variance decomposition — never hardcoded.
   Reliability at sample size `n` is `n / (n + K)`. In practice `K` comes
   out tiny (0.4–2.8 pitches) for every well-populated pitch type, meaning
   the metric stabilizes almost immediately past a handful of pitches. Rare
   pitch types fall back to the median `K` across types that could be
   estimated.
2. **Z-score to the plus-scale**: `100 - 10 * (shrunk_pred - season_mean) /
   season_std`, computed fresh within each requested season — same
   convention as every other model on this site. 100 = league average that
   season; each 10 points = 1 standard deviation.

Per-pitcher overall Stuff+ is the usage-weighted average across their full
arsenal (`overall_stuff_plus`).

## 9. SHAP breakdown — what actually drives a pitcher's number

`pitcher_shap_breakdown()` explains the model's **raw, unshrunk** prediction
per pitch type — shrinkage is a statistical correction for small-sample
noise, not something attributable to any input feature, so it's
deliberately excluded from the per-feature attribution. For a low-sample
pitch type, the SHAP contributions will sum to a slightly different number
than `(stuff_plus - 100)`; that gap *is* the shrinkage adjustment, not an
error.

## 10. Where everything lives

**Live-serving:**
- `analytics/stuff_features.py` — shared feature engineering (training AND
  scoring both call this; the single-source-of-truth design that prevents
  train/serve skew).
- `analytics/stuff_model.py` — runtime scoring: loads
  `models/stuff_plus_v3_lgbm.txt` + `_meta.json`, builds the fastball
  reference table fresh per query, scores on demand, applies shrinkage,
  provides `compute_stuff_plus`, `compute_stuff_plus_by_stand`,
  `pitcher_shap_breakdown`.
- `api/routers/stuff_model_report.py` — `/api/stuff-model/*` endpoints
  (report, leaderboard, per-pitcher breakdown).
- `frontend/stuff-model.html` + `frontend/js/stuff-model.js` — the
  model-internals page.

**Training/offline:**
- `training/build_stuff_v3_dataset.py`, `training/train_stuff_v3_lgbm.py`,
  `training/validate_stuff_v3.py`, `training/ablate_stuff_v3_slots.py`.
- `Methodology/STUFF_V3_REPORT.md` — the full validation/decision report
  this architecture was locked in from.

**Preserved, unused (v2):** `analytics/stuff_features_v2_legacy.py`,
`analytics/stuff_model_v2_legacy.py`, `models/stuff_plus_v2.txt`.
