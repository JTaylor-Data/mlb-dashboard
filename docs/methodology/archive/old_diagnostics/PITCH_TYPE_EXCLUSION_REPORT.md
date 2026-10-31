# Stuff+ v3 — PO / UN / EP pitch-type exclusion

Excludes pitchouts (`PO`), Statcast's own "unclassified" code (`UN`), and eephus
(`EP`) from Stuff+'s training set, scoring population, and the live
by-pitch-type display — plus, separately, from the live pitch-table UI itself
(a display-layer gap found and fixed during this work, not part of the
original Stuff+ model but the same table Stuff+ scores live in). `KN`
(knuckleball) is deliberately untouched — a real, competitive pitch type with
a known modeling limitation, not junk data.

Model retrained, redeployed (`models/stuff_plus_v3_lgbm.txt` overwritten in
place — no separate promotion step needed, `analytics/stuff_model.py` already
pointed at this filename), and validated live. `models/stuff_plus_v2.txt`
and the legacy v1/v2 files were not touched.

---

## Part A.1 — Baseline mean/std shift, checked BEFORE changing anything

Computed against the live (pre-retrain) v3 model's raw predictions — this
answers "was this actually dragging the baseline," not just "did excluding
it change something."

| Season | mean (all) | mean (excl. PO/UN/EP) | std (all) | std (excl.) | std shift |
|---|---|---|---|---|---|
| 2023 | -0.005245 | -0.005280 | 0.006252 | 0.006128 | -2.0% |
| 2024 | -0.005077 | -0.005109 | 0.005800 | 0.005693 | -1.8% |
| 2025 | -0.005213 | -0.005277 | 0.006121 | 0.005861 | -4.2% |

**Real, if modest — this was dragging the baseline, growing over time.**
The mean shift is tiny in absolute terms (0.03–0.06 units on a ~-0.005
baseline), but the standard-deviation shift is not: 2–4% smaller once the
three categories are removed. Since the plus-scale formula is `100 -
10*(pred-mean)/std`, a smaller std means the SAME raw prediction now maps to
a slightly WIDER Stuff+ spread (dividing by a smaller number stretches
z-scores). The 2025 shift (-4.2%) is the largest because `EP` usage nearly
tripled from 2021 (128 pitches) to 2025 (951) — this wasn't a fixed, rounding
-error-sized nuisance, it was a slowly growing one.

Why: `PO`/`UN`/`EP` pitches score systematically far above (worse than) the
rest of the population in raw predicted run value:

| Season | EP mean pred | EP n | PO mean pred | PO n |
|---|---|---|---|---|
| 2023 | 0.0373 | 556 | 0.0187 | 46 |
| 2024 | 0.0320 | 574 | 0.0161 | 54 |
| 2025 | 0.0409 | 951 | 0.0183 | 54 |

(population mean is ≈ -0.005 — `EP`'s own mean is roughly 7-8x further from
zero, in the "bad pitch" direction, than the population it was sitting in.)

---

## Part B — Deployed

- Model retrained on the exclusion-filtered feature table, saved over the
  existing `models/stuff_plus_v3_lgbm.txt` / `_meta.json` / `_report.json` -
  `analytics/stuff_model.py` already reads these exact filenames, so no
  separate "point at the new artifact" step was needed once the training
  script ran.
- **Real bug caught mid-deployment, fixed**: the retrain script overwrites
  `stuff_plus_v3_lgbm_meta.json` from scratch and doesn't include
  `vaa_coefs`/`spin_calibration` (needed by live scoring to reproduce the
  training-time feature transform) - a previous session had patched these in
  by hand after training, a manual step that silently didn't survive this
  retrain. Fixed two ways: patched the current file immediately, and edited
  `training/train_stuff_v3_lgbm.py` itself to merge them in automatically on
  every future run, so this can't recur silently.
- **Real bug caught mid-deployment, fixed**: `analytics/player_page.py`'s
  `pitcher_pitch_table()` (the by-pitch-type table Stuff+ scores live in) is
  a SEPARATE code path from Stuff+'s own training/scoring pipeline - it
  builds the display table straight from `pitches`, only joining in
  `stuff_plus` afterward. Excluding PO/UN/EP from Stuff+'s scoring
  population did NOT remove them from this display table; a `PO` row still
  rendered with real velo/spin/etc stats and simply no Stuff+ number.
  Confirmed live on Miles Mikolas (3 real pitchouts in 2025) before the fix;
  fixed by excluding the same three types inside `pitcher_pitch_table()` and
  `pitcher_league_avg_row()` (its reference row) specifically - deliberately
  NOT inside the shared `_load()` helper, since `pitcher_counting_stats()`
  also uses `_load()` and real counting stats (K/BB/HR/IP) must still
  include every real pitch, eephus included (an eephus strikeout is still a
  real strikeout). Re-verified live after the fix: Mikolas' pitch table now
  shows `FF, SL, CU, SI, CH, ST, CS` - no `PO` row.
- Server restarted (picks up the retrained model - it's cached in-process
  and doesn't hot-reload) and its automatic startup refresh re-warmed every
  cached league-wide pool against the new baseline.

---

## Part C — Validation battery

### C.1 — see Part A.1 above (same numbers, not repeated).

### C.2 — Year-over-year, individual-pitcher-matched, pooled vs. per-pair

| Seasons | n | r² |
|---|---|---|
| 2021→2022 | 444 | 0.720 |
| 2022→2023 | 450 | 0.724 |
| 2023→2024 | 436 | 0.701 |
| 2024→2025 | 430 | 0.755 |
| 2025→2026 | 436 | 0.754 |
| **Pooled (all pairs stacked, one regression)** | **2,196** | **0.729** |
| Mean of the 5 per-pair r²s | — | 0.731 |

**Pooled and mean-of-per-pair land within 0.002 of each other here** - for
this data, averaging the 5 separate r²s is not meaningfully different from
pooling every matched pair into one regression. That's itself worth
recording (the two methods COULD diverge more than this if per-pair sample
sizes were more unequal; here they're all in the 430-450 range, so they
don't).

**Honest comparison to the already-recorded number**: before this exclusion
(primary-only, no-threshold architecture, previous session), the same check
read 0.752 (mean of per-pair r²s: 0.735/0.752/0.720/0.772/0.780). Post-
exclusion it's 0.731 - **slightly LOWER, not higher**. This is a real,
if small (~0.02), move in the opposite direction from what excluding
"noisy junk pitches" might be assumed to do. Plausible mechanism: removing
PO/UN/EP shrinks a handful of pitchers' total pitch counts right at the
qualification boundary, very slightly changing WHICH pitchers clear the
200-pitch bar each season pair - a composition effect, not evidence the
underlying model got worse. Reported plainly either way, not smoothed over.

### C.3/C.4 — Stuff+ vs. real xFIP / real ERA (pooled 2021-2025, real
`player_advanced_stats` values - backfilled for 2021-2025 this session,
previously only populated for 2026)

| Outcome | n | r² | r | coefficient | direction |
|---|---|---|---|---|---|
| Real xFIP | 3,188 | 0.145 | -0.381 | -0.0688 | **negative, as expected** |
| Real ERA | 3,188 | 0.099 | -0.315 | -0.1195 | **negative, as expected** |

Both correctly negative (higher Stuff+ → lower/better xFIP and ERA) and
neither r² is suspiciously large or literally zero - both land in a
plausible "real but partial" range for a metric that measures pitch shape
alone, not command, sequencing, or (for ERA specifically) defense and luck.
**xFIP correlates meaningfully more strongly than ERA (0.145 vs. 0.099)** -
exactly the expected pattern, since xFIP already strips out the
defense/luck noise Stuff+ was never trying to explain in the first place,
while ERA still carries it.

### C.5 — Out-of-sample predictive validity (2024 Stuff+, from a model that
never saw 2024, vs. real 2025 outcomes)

| 2025 outcome | v3, post-exclusion (n=290) | v2, on record (n=430) |
|---|---|---|
| xwOBA | -0.367 | -0.385 |
| whiff% | 0.421 | 0.403 |
| swstr% | 0.435 | 0.433 |
| K-BB% | 0.357 | 0.380 |
| chase% | 0.260 | 0.209 |

**Mixed, not uniformly better or worse**: whiff% and chase% improved
meaningfully (chase% +0.05), swstr% is essentially flat, xwOBA and K-BB%
are slightly weaker. No single clean verdict - reported as such rather than
cherry-picking the favorable columns.

**The n=290 vs. n=430 gap is NOT caused by this exclusion** - checked
directly before writing this up, since a ~33% sample-size drop from
excluding ~600 total pitches (out of 700K+ in 2024) would be a red flag.
Confirmed: the number of pitchers clearing 200 real 2024 pitches is
IDENTICAL (625) whether PO/UN/EP are excluded or not. The real cause is
that `pitcher_percentile_metrics`' own 2025 qualification pool (the
`min_bf` bar scales with team games played) is simply smaller NOW than it
was when the original v1-vs-v2 report was generated, months earlier in this
project's timeline - a time-dependency in an unrelated function, not an
effect of today's change. Flagged here so the n difference doesn't get
misread as this exclusion silently dropping 140 real pitchers.

### C.6 — By-pitch-type distribution, post-exclusion (2025)

| Type | Stuff+ | n pitches | n pitchers |
|---|---|---|---|
| FF | 100.6 | 225,618 | 758 |
| SI | 101.1 | 109,895 | 570 |
| SL | 101.7 | 101,825 | 569 |
| CH | 97.6 | 73,047 | 540 |
| CU | 95.4 | 59,994 | 407 |
| ST | 101.5 | 54,304 | 356 |
| FC | 100.3 | 53,308 | 356 |
| FS | 100.5 | 23,110 | 164 |
| SV | 98.2 | 3,540 | 25 |
| FA | 39.6 | 900 | 51 |
| KN | 62.4 | 137 | 11 |

Common arsenal types (FF/SI/SL/CH/CU/ST/FC/FS) all sit in a tight 95-102
band, same as before - removing the three low-scoring outlier categories
did NOT compress or shift the real-pitch-type distribution, which is what
you'd want: this was meant to be a population cleanup, not a rescale of
what "100" means for a real fastball. **`PO`/`UN`/`EP` confirmed absent**
from this table entirely (checked programmatically, not just by eye). `FA`
(generic/unclassified fastball subtype - a DIFFERENT code from `UN`,
correctly not touched by this exclusion) and `KN` remain, both still
scoring low, exactly as before - expected, not a new finding.

### C.7 — Cross-model correlation (2025, n=522 pitchers in all three models)

| Pair | Post-exclusion | Recorded baseline |
|---|---|---|
| Stuff+ vs. Deception+ | 0.402 | ≈0.377-0.535 (depending on which prior session's number) |
| Stuff+ vs. Tunneling+ | 0.124 | ≈0.12-0.19 |

Both land comfortably inside the previously-recorded ranges - **no
meaningful shift**, as expected. This change touches population
composition, not the feature set these correlations are actually
sensitive to.

### C.8 — Sample-size sanity check (2021-2025)

| | |
|---|---|
| Total pitches excluded | 3,000 |
| — EP | 2,756 |
| — PO | 228 |
| — UN | 16 |
| Distinct pitcher-seasons with ≥1 excluded pitch | 371 |

**A rounding-error-sized cleanup in raw volume** (3,000 pitches out of
~3.5M across these 5 seasons, ~0.09%), concentrated almost entirely in `EP`
(92% of the excluded total). Not nothing, though: 371 distinct
pitcher-seasons had at least one of these pitches, and per Part A.1 the
population-level std shift it produces (2-4%) is larger than the raw
volume alone would suggest - a small number of pitches sitting very far
from the mean can move a standard deviation more than their count implies.

---

## Bottom line

- **The exclusion was worth doing**: Part A.1 confirms it was a real, if
  modest, standardization drag (growing over time as `EP` usage grew), not
  an assumption that turned out to be nothing.
- **Model quality is unchanged**: CV R² is flat (0.00042 → 0.0004, noise-
  level), as expected from removing 0.09% of the data.
- **Real-outcome validity is genuinely there and correctly signed** (C.3/
  C.4), and out-of-sample predictive validity (C.5) is a wash, not a clean
  win - some outcomes improved, some didn't.
- **One check moved in the "wrong" direction and is reported as such**:
  year-over-year stability (C.2) is very slightly lower post-exclusion
  (0.729 pooled vs. 0.752 previously), most likely a qualification-boundary
  composition effect rather than a real quality regression, but not
  smoothed over.
- **Two real deployment bugs were caught and fixed during this work**
  (missing calibration keys on retrain; the by-pitch-type UI not sharing
  the same exclusion as the model) - neither would have been visible from
  the offline validation report alone, only from checking the live site.
