# Tunneling+ v3 — 150ms vs. 200ms commit-point experiment

**Question tested**: would moving the commit point closer to the plate (150ms
before arrival, ≈19.2 ft, vs. the live model's 200ms/≈25.7 ft) meaningfully
widen v3's score spread? Everything else — features, training code, CV
scheme, qualifying threshold, plus-scale conversion — is byte-for-byte
identical to the live model; the only input that changed is which of v2's
saved commit-point tables `sequential_lag_features()` was given.

**This is an offline experiment only.** Nothing here touches `models/
tunneling_v3.txt`, `analytics/tunneling_v3_model.py`, or any live-serving
path — the live site still runs 200ms. Artifacts live in
`training/tunneling_v3_150ms_artifacts/`.

## Verdict: the hypothesis doesn't hold

Spread is essentially unchanged — if anything, marginally *narrower*:

| | 200ms (live) | 150ms (experiment) |
|---|---|---|
| std dev | 1.15 | 1.13 |
| min | 94.9 | 94.5 |
| max | 103.9 | 104.0 |
| p10 | 98.4 | 98.4 |
| p90 | 101.2 | 101.2 |

(2025 season, min 100 qualifying pitches, n=688 both times.)

The leaderboards are nearly identical pitcher-for-pitcher — 8 of the top 10
and 8 of the bottom 10 are the same names in the same order, within a few
tenths of a point either way. The extra ~6.5 ft of divergence time between
150ms and 200ms just doesn't translate into more between-pitcher separation
at the score level.

**Why not, plausibly**: the model isn't scoring a single static distance —
it's already integrating information across up to 6 real lag slots per
pitch, and the SHAP importance ordering is nearly identical between the two
runs (lag1 dominates, decaying smoothly through lag6 either way — see §3).
Moving the reference point closer to the plate shifts each lag's raw
commit/plate distances by a roughly constant amount, but doesn't change
their *relative* informativeness to the model much, so the learned
prediction — and the resulting score spread — barely moves. This also lines
up with v2's own earlier finding (see below): 200ms beat 150ms on the
sequential test using the OLD hand-built composite, and here, with a
trained model, the two land within noise of each other on nearly every
check. The signal ceiling for this feature family looks like it's set by
what real sequential pitch relationships can predict about outcome at all
— not by exactly where within that flight path you sample them.

---

## 1. Leaderboard, 2025 season, min 100 pitches

### Top 10

| Rank | 200ms (live) | Score | 150ms (exp.) | Score |
|---|---|---|---|---|
| 1 | A.J. Puk (AZ) | 103.9 | A.J. Puk (AZ) | 104.0 |
| 2 | Jayden Murray (CHC) | 103.0 | Jayden Murray (CHC) | 102.9 |
| 3 | Jesus Tinoco (MIA) | 102.7 | Jesus Tinoco (MIA) | 102.6 |
| 4 | Andrew Kittredge (BAL) | 102.5 | Davis Daniel (CIN) | 102.5 |
| 5 | Trevor Williams (WSH) | 102.4 | Andrew Kittredge (BAL) | 102.4 |
| 6 | Davis Daniel (CIN) | 102.3 | Trevor Williams (WSH) | 102.4 |
| 7 | Eduard Bazardo (SEA) | 102.2 | Eduard Bazardo (SEA) | 102.4 |
| 8 | Ryan Johnson (LAA) | 102.2 | Tanner Scott (LAD) | 102.1 |
| 9 | Jacob deGrom (TEX) | 102.1 | Ryan Johnson (LAA) | 102.0 |
| 10 | Tanner Scott (LAD) | 102.1 | Taylor Rogers (MIN) | 102.0 |

### Bottom 10

| Rank | 200ms (live) | Score | 150ms (exp.) | Score |
|---|---|---|---|---|
| 1 | Michael Kopech (LAD) | 94.9 | Michael Kopech (LAD) | 94.5 |
| 2 | Luis Contreras (HOU) | 95.6 | Luis Contreras (HOU) | 95.4 |
| 3 | Ian Anderson (ATL) | 96.2 | Ian Anderson (ATL) | 95.8 |
| 4 | Félix Bautista (BAL) | 96.5 | Jalen Beeks (TEX) | 96.4 |
| 5 | Jalen Beeks (TEX) | 96.5 | Félix Bautista (BAL) | 96.6 |
| 6 | Carson Whisenhunt (SF) | 96.5 | Chayce McDermott (NYM) | 96.7 |
| 7 | Chayce McDermott (NYM) | 96.7 | Freddy Tarnok (MIA) | 96.9 |
| 8 | Mitch Farris (LAA) | 96.7 | Mitch Farris (LAA) | 96.9 |
| 9 | Sam Moll (CIN) | 96.8 | Sam Moll (CIN) | 97.1 |
| 10 | Erik Miller (BOS) | 97.0 | Carson Whisenhunt (SF) | 97.1 |

9 of 10 top names and 9 of 10 bottom names overlap between the two versions
(only Jacob deGrom drops out of 150ms's top 10, replaced by Taylor Rogers;
only Erik Miller drops out of 150ms's bottom 10, replaced by Freddy Tarnok)
— the identity of "who tunnels well" barely moves, not just the spread.

## 2. Full validation battery, side by side

| Check | 200ms (live) | 150ms (experiment) |
|---|---|---|
| CV mean R² | 0.007395 | 0.007395 (rounds identically; 0.0074 both) |
| CV mean MAE | 0.13211 | 0.13214 |
| Sequential same-at-bat z-score (model's own predicted tercile) | 84.68 | 83.66 |
| Year-over-year pooled r² | 0.299 (n=2,436) | 0.287 (n=2,436) |
| Cross-model r vs. Stuff+ | 0.074 | 0.047 |
| Cross-model r vs. Deception+ | 0.154 | 0.146 |
| Correlation vs. fastest FB velo | 0.085 | **0.023** |

Everything lands within noise of the live model **except** the fastest-
fastball-velocity correlation, which drops substantially (0.085 → 0.023).
That's the one place the commit-point choice visibly shows up: 200ms's own
distance-from-plate is itself more strongly velocity-driven (harder
throwers commit proportionally farther out, r=0.964 between velocity and
commit-point distance, per v2's own build), so a 200ms-based score
inherits a bit more of that velocity relationship than a 150ms-based one
does. Small in absolute terms either way, but the one honest directional
difference between the two runs.

### Out-of-sample predictive validity (trained ex-2024, scored on 2024,
correlated against real 2025 outcomes)

| Outcome (min 100 2024 pitches, n=302 both) | 200ms | 150ms |
|---|---|---|
| xwOBA | -0.051 | -0.032 |
| whiff% | -0.100 | **-0.135** |
| swstr% | -0.026 | -0.067 |
| K-BB% | 0.148 | 0.124 |
| chase% | 0.158 | 0.130 |

Mixed, small differences in both directions — 150ms edges out 200ms on
whiff%/swstr% (a closer point does track swing-and-miss slightly better
out of sample), 200ms edges out on K-BB%/chase%/xwOBA. Neither version
clearly dominates the other here; both are modest, directionally-plausible
out-of-sample signals, same as the live model's own report already
characterized them.

## 3. Lag-slot SHAP importance — nearly identical shape

| Lag | 200ms distance-group total | 150ms distance-group total |
|---|---|---|
| 1 back | 0.00888 | 0.00874 |
| 2 back | 0.00646 | 0.00648 |
| 3 back | 0.00498 | 0.00516 |
| 4 back | 0.00323 | 0.00309 |
| 5 back | 0.00204 | 0.00209 |
| 6 back | 0.00089 | 0.00087 |

Same smooth recency decay, same magnitudes to within rounding noise — the
model leans on the same lag structure regardless of which commit point
generated the underlying distances.

## 4. What this settles, and what it doesn't

- **Settled**: a closer decision point does not buy a meaningfully wider,
  more differentiated Tunneling+ leaderboard. The tight ~95-104 spread on
  the live model isn't an artifact of 200ms specifically being too
  conservative a commit point — moving it 6.5 ft closer to the plate
  produces essentially the same spread, the same top/bottom names, and the
  same lag-importance shape.
- **Not settled by this experiment**: whether an even more extreme point
  (very close to the plate, or the plate itself) would behave differently
  — this only tested the two commit-point definitions v2 already validated
  as its best time-based candidates. Given 150ms and 200ms already bracket
  a real ~6.5 ft difference and land this close together, a further move
  seems unlikely to change the conclusion, but wasn't tested here.
- **No promotion decision follows from this** — the live model stays on
  200ms. This was a targeted test of one specific hypothesis about spread,
  not a re-opening of the commit-point choice itself (which v2's own
  sequential-test comparison already settled in 200ms's favor on a
  different, more mechanism-specific metric).

## Where everything lives

`training/build_tunneling_v3_150ms_dataset.py`,
`training/train_tunneling_v3_150ms_model.py`,
`training/tunneling_v3_150ms_validate.py`,
`training/tunneling_v3_150ms_predictive_validity.py` — mirror the live
200ms scripts exactly except for the commit-point source and output paths.
All outputs in `training/tunneling_v3_150ms_artifacts/`:
`tunneling_v3_150ms.txt` (+ `_meta.json` / `_report.json`),
`tunneling_v3_150ms_scores.parquet`, `tunneling_v3_150ms_validation.json`,
`tunneling_v3_150ms_predictive_validity.json`.
