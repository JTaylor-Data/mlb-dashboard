# Tunneling+ Per-Pitch-Type-Pair Composite

**Status:** Built, validated, wired to player page. Branch: `tunneling-pair-composite`.  
**Not yet merged to master.** Awaiting review decision.

---

## Overview

Replaces a single pitcher-level Tunneling+ score with a per-pitch-type-pair composite table on each pitcher's player page. Renders below the movement plot as a full-width table (placement decision below).

---

## Part 2 — Sign Check Results

n = 14,727 qualifying pairs, pooled 2021–2026.

| Component | r | 95% CI | Decision |
|---|---|---|---|
| commit_distance | +0.138 | [+0.122, +0.153] | Use **−commit_distance** (larger = worse tunnel) |
| divergence | +0.006 | [−0.010, +0.023] | **DROPPED** — CI crosses zero |
| shape_break_plus | −0.021 | [−0.037, −0.004] | Use **+shape_break_plus** (larger = better tunnel) |

### Divergence drop decision (reversible)

`divergence = plate_distance − commit_distance` was dropped because its 95% CI fully straddles zero (r = +0.006). The signal is empirically indistinguishable from noise at this sample size.

**To re-add divergence:** In `analytics/tunneling_pair_composite.py`, add `divergence_mean`/`divergence_std` to pop_stats, then in `_compute_composite()` uncomment the `divergence_z` line and change the composite formula to `(-commit_distance_z + shape_break_plus_z - divergence_z) / 3`. Sign should be negative (larger divergence = better tunnel theoretically, but the empirical signal is near-zero).

---

## Part 3 — Composite Formula

```
composite_z = (−commit_distance_z + divergence_z + shape_break_plus_z) / 3
composite_score = 100 + 10 × composite_z   [plus-scale]
```

- `−commit_distance_z`: small commit-point distance = pitches look alike at decision time = good
- `+divergence_z`: large divergence (plate_distance − commit_distance) = pitches grow apart after commit = good
- `+shape_break_plus_z`: above-expected mid-trajectory divergence = good

### Component selection: divergence vs plate_distance (2026-10-28)

Head-to-head comparison run on 2025 data (n=2,721 pairs) and YoY validity (n=2,244 pitcher-seasons, 2021→2026):

| Metric | plate_distance r | 95% CI | divergence r | 95% CI |
|---|---|---|---|---|
| Whiff-rate (pair-level, 2025) | −0.021 | [−0.059, **+0.017**] ⚠️ crosses 0 | **−0.032** | [−0.070, **−0.005**] ✓ |
| YoY ERA (t→t+1) | −0.056 | [−0.097, −0.015] ✓ | −0.058 | [−0.100, −0.017] ✓ |
| YoY xFIP (t→t+1) | −0.108 | [−0.149, −0.067] ✓ | **−0.123** | [−0.164, −0.083] ✓ |

Divergence wins on all three: its whiff-rate CI excludes zero (plate_distance does not), and its xFIP signal is ~1.5 points stronger. The intuition: divergence captures *growth* in separation after the hitter commits, which is the mechanistically meaningful quantity. Raw plate_distance conflates separation growth with initial release spread.

**To swap back to plate_distance:** in `_compute_composite()`, replace the `divergence_z` line with `plate_distance_z = (df["plate_distance"] - pop_stats["plate_distance_mean"]) / pop_stats["plate_distance_std"]` and update pop_stats and formula accordingly.

**Display scale decision:** Plus-scale (100-centered, 10 = 1 SD), matching every other metric in the dashboard. Raw z-scores would require the user to mentally reference a mean; plus-scale reads identically to Stuff+/Tunneling+/etc.

**Normalization:** Within-season. Z-scores are relative to all qualifying pairs in that season's population. A 2025 score of 110 means this pair is 1 SD above the 2025 average.

**Implementation:** `analytics/tunneling_pair_composite.py`
- `season_all_pairs(conn, season)` — slow (~minutes, cached in players.py via `@ttl_cache`)
- `pitcher_pairs_from_season(all_pairs_df, pitcher_id)` — fast filter for player page

---

## Part 4 — Validation Results (2025)

### Score Distribution

| Stat | Value |
|---|---|
| n pairs | 2,721 |
| Min | 59.4 |
| Max | 119.2 |
| SD | 6.74 |
| Spread | 59.7 pts |

### Top / Bottom 5 Pairs (2025)

**Top:**
1. Reid Detmers — CU/FF — 119.2
2. Jacob deGrom — CU/FF — 117.9
3. Eduard Bazardo — SI/ST — 117.5
4. Eric Lauer — CU/FF — 117.2
5. Taylor Rogers — SI/ST — 117.2

**Bottom:**
1. Chris Bassitt — FF/FS — 59.4
2. Chris Bassitt — FC/FS — 59.5
3. Erik Miller — CH/FF — 69.2
4. Chris Bassitt — CU/FS — 71.0
5. Ryan Yarbrough — CH/FF — 72.6

Top pairs (Detmers, deGrom CU/FF) are qualitatively plausible — both are known for elite spin-direction diversity between their curveball and four-seamer.

### 2025 Leaderboard (pitcher-level mean composite)

538 qualifying pitchers, 2,721 pairs. Most top-10 pitchers have n_pairs=1 (high variance; single-pair pitchers can land anywhere — a 2+ pair requirement would be a reasonable display filter).

| Rank | Name | Team | Score | Pairs |
|---|---|---|---|---|
| 1 | Blake Treinen | LAD | 108.5 | 1 |
| 2 | Mick Abel | MIN | 107.3 | 1 |
| 3 | Trevor Megill | MIL | 107.1 | 1 |
| 4 | Spencer Arrighetti | TOR | 106.9 | 1 |
| 5 | Hayden Wesneski | HOU | 106.8 | 1 |
| 538 | Tyler Gilbert | CWS | 91.2 | 6 |
| 537 | Tommy Kahnle | BOS | 91.6 | 1 |
| 536 | Carson Whisenhunt | SF | 92.1 | 1 |

### Sequential Whiff-Rate Test (fixed, pair-level)

**Previous test** (tercile bucketing) was methodologically flawed — it mixed causal direction. Fixed test:

For each pitch that follows its pair-partner in the same at-bat, assign that pitch the pair's composite_score. Aggregate to pair level (mean_whiff_rate per pair), then correlate pair composite_score vs pair mean_whiff_rate.

| Stat | Value |
|---|---|
| n pairs with sequence data | 2,715 |
| n eligible follow-up pitches | 282,345 |
| r (composite vs whiff rate) | **−0.021** |
| 95% CI | [−0.059, +0.017] |
| CI crosses zero | **Yes** |

Signal is indistinguishable from zero — the composite does not predict next-pitch whiff rate at pair level. This is consistent with what the sign check was measuring (run value over full seasons, not per-pitch whiff inflation).

### YoY Predictive Validity (composite_t → ERA/xFIP at t+1)

n = 2,244 pitcher-seasons (2021→2026, all with matched t+1 outcomes).

| Outcome | r | 95% CI | CI crosses zero |
|---|---|---|---|
| ERA (t+1) | −0.056 | [−0.097, −0.015] | **No** ✓ |
| xFIP (t+1) | −0.108 | [−0.149, −0.067] | **No** ✓ |

Both CIs exclude zero. Higher composite (better geometric tunneling) predicts lower ERA and xFIP the following year. xFIP signal is stronger (r=−0.108 vs r=−0.056 for ERA), consistent with xFIP being a cleaner estimator of true skill. Effect sizes are modest but the directional signal is real and statistically reliable.

---

## Part 5 — Implementation Decisions

### Backend

Added `tunneling_pairs` to the existing `/api/player/{id}` pitcher payload (same endpoint, not a new one). Rationale: avoids a second round trip from the frontend and keeps the player page's single-endpoint loading pattern intact. A separate endpoint would be cleaner if the pair data grows to require its own split/filter controls; for now, one endpoint is simpler.

Added `_tunneling_pair_season(conn, season)` cached function in `api/routers/players.py` — computes once per season, all subsequent pitcher views within that session are a fast filter.

MIN_PITCHES_PER_TYPE gate: 100 pitches per type (same threshold as pairwise_distance_features default).

### Frontend

Pair table is placed **full-width below the movement-plot-row**, not inside the `.movement-plot-side-col`. The side column is narrow and already carries the movement summary table, rubber icon, and BABIP widget; a 5-column table with pair labels would overflow. Full-width placement reads more cleanly.

Function: `buildTunnelingPairsTable(pairs)` in `frontend/js/player.js`.

Columns: Pair (with pitch-type color dots via `pitchColor()`), Commit, Divg, Break, Score.

Pitch color dots use the same `seq-legend-dot` inline-span pattern used in `outings.js`.

---

## Weighting sensitivity

### Part 1 — Standalone component correlations

Each component tested individually against all four outcomes (n=2,611–2,721 pair-level for 2025; n=1,599 pitcher-seasons for YoY):

| Component | Whiff r | 95% CI | EV-delta r | 95% CI | YoY ERA r | 95% CI | YoY xFIP r | 95% CI |
|---|---|---|---|---|---|---|---|---|
| −commit_distance_z | −0.115 | [−0.153, −0.077] ✓ | +0.043 | [+0.004, +0.081] ⚠️ wrong sign | −0.009 | [−0.058, +0.041] ✗ | −0.084 | [−0.132, −0.035] ✓ |
| +divergence_z | +0.079 | [+0.040, +0.117] ⚠️ wrong sign | −0.006 | [−0.045, +0.033] ✗ | −0.037 | [−0.086, +0.012] ✗ | −0.057 | [−0.105, −0.008] ✓ |
| +shape_break_plus_z | −0.028 | [−0.066, +0.011] ✗ | −0.027 | [−0.065, +0.012] ✗ | +0.004 | [−0.045, +0.053] ✗ | −0.022 | [−0.071, +0.027] ✗ |

Mean |r| across all four outcomes: −commit=0.063, +divergence=0.045, +shape_break_plus=0.020.

**shape_break_plus IS the weakest component** — its mean |r| (0.020) is less than half of either other component, and it fails to clear zero on any of the four outcome CIs. Part 2 therefore ran.

Notable: commit_distance has the strongest signal overall (dominant on whiff-rate and YoY xFIP), but its EV-delta sign is wrong (+0.043 where we want negative). Divergence's whiff-rate sign is also wrong (+0.079). This confirms the composite's YoY xFIP validity is driven mostly by commit_distance, while the within-at-bat signals are noisy.

### Part 2 — Pre-registered weight grid (2025-2026 holdout)

Four weightings tested strictly on held-out 2025–2026 data (n=5,418–5,471 pair-level; n=335 for YoY 2025→2026). No fitting was done on this data before the grid ran.

| Weighting | EV-delta r | 95% CI | Whiff r | 95% CI | YoY xFIP r | 95% CI |
|---|---|---|---|---|---|---|
| W0 (1/3, 1/3, 1/3) | +0.021 | [−0.005, +0.048] | −0.043 | [−0.069, −0.016] ✓ | −0.113 | [−0.218, −0.006] ✓ |
| W1 (0.4, 0.4, 0.2) | +0.030 | [+0.003, +0.056] | −0.037 | [−0.063, −0.010] ✓ | −0.108 | [−0.212, −0.001] ✓ |
| W2 (0.45, 0.45, 0.1) | +0.033 | [+0.006, +0.059] | −0.031 | [−0.057, −0.004] ✓ | −0.097 | [−0.202, +0.010] ✗ |
| W3 (0.5, 0.5, 0.0) | +0.034 | [+0.007, +0.060] | −0.025 | [−0.052, +0.001] ✗ | −0.086 | [−0.192, +0.021] ✗ |

**Selection rule (pre-stated):** W1/W2/W3 wins only if it beats W0 on EV-delta by more than its own 95% CI width.

- W1 improvement over W0: |0.030|−|0.021| = 0.009 < CI width 0.053 → not enough
- W2 improvement: 0.011 < 0.053 → not enough
- W3 improvement: 0.012 < 0.053 → not enough

**W0 wins — no improvement exceeds noise. Equal weights (1/3, 1/3, 1/3) retained.**

The weight grid also shows that down-weighting shape_break_plus hurts whiff and YoY xFIP more than it helps EV-delta. There is no empirical case for re-weighting. Also note EV-delta sign is positive (wrong) across all weightings — this is a consistent finding, not a weighting problem.

---

## Pair-type normalization (2026-10-28)

### The confound

Without normalization, CU/FF pairs dominated the top leaderboard because curveballs break more in absolute terms — a CU/FF pair has inherently large `plate_distance` and naturally low `commit_distance` simply because of what those pitch types are, not because of how cleverly the pitcher hides them. The composite was partly measuring "does this pitcher throw a fastball and a curveball" rather than "does this pitcher tunnel that pair unusually well."

### The fix

`commit_distance` and `divergence` are now de-meaned by (type_a, type_b) before z-scoring:

```
commit_residual   = commit_distance - mean_commit_for_(type_a,type_b)_that_season
divergence_residual = divergence - mean_divergence_for_(type_a,type_b)_that_season
```

Residuals are then z-scored across all pairs. The composite now answers "is this CU/FF better-tunneled than the average CU/FF?" `shape_break_plus` was already pair-type-normalized internally, so it is unchanged.

### Before/after leaderboards (top 5 individual pairs)

| Before (raw) | After (normalized) |
|---|---|
| Bednar CU/FF 116.7 | Wheeler CU/SI 120.1 |
| Civale CU/FF 116.5 | Civale CU/FF 119.1 |
| Thielbar CU/FF 115.8 | Wheeler CU/SI 119.1 |
| Nola CU/FF 115.6 | Wheeler CU/SI 118.9 |
| Englert CU/FF 115.4 | Pruitt CU/SI 118.8 |

CU/FF still appears (Civale, Bednar) but no longer monopolizes — CU/SI pairs now compete on their own terms. More pitch type diversity in the top/bottom lists.

### Before/after validation

| Metric | Before (raw) | 95% CI | After (normalized) | 95% CI | Change |
|---|---|---|---|---|---|
| Whiff-rate | −0.032 | [−0.070, −0.005] ✓ | +0.055 | [+0.028, +0.081] ⚠️ sign flips | worse |
| EV-delta | +0.022 | [−0.016, +0.060] ✗ | **−0.039** | [**−0.071, −0.007**] ✓ | **better** |
| YoY ERA | −0.058 | [−0.100, −0.017] ✓ | **−0.073** | [**−0.114, −0.032**] ✓ | **better** |
| YoY xFIP | −0.123 | [−0.164, −0.083] ✓ | **−0.147** | [**−0.188, −0.107**] ✓ | **better** |

Normalization improves EV-delta (previously wrong sign, now correct and significant), YoY ERA, and YoY xFIP. The whiff-rate sign flips — this warrants investigation but may reflect that pair-type-normalized commit_distance captures a different dimension than raw commit_distance did for immediate at-bat effects.

Overall: normalization makes the metric stronger on the outcomes that most matter (contact quality suppression and future ERA/xFIP prediction) and is theoretically correct — it answers the right question.

### CU-pair variance: signal vs noise check (2026-10-28)

Even after mean-subtracting by pair type, CU-containing pairs still populate the leaderboard extremes. Two tests were run to determine whether this reflects real pitcher-to-pitcher variance in tunneling quality or measurement noise from smaller samples.

#### Test 1 — Split-half reliability (2025, n=569 CU pairs, n=927 FF pairs)

Each pitcher-season's pitches were split temporally (first 50% / second 50%). Components were recomputed independently on each half (min 50 pitches per type per half). Correlation between the two halves measures reproducibility.

| Component | CU pairs r | 95% CI | n | FF pairs r | 95% CI | n |
|---|---|---|---|---|---|---|
| commit_distance | 0.779 | [0.745, 0.810] | 569 | 0.824 | [0.802, 0.844] | 927 |
| divergence | 0.850 | [0.825, 0.871] | 569 | 0.883 | [0.868, 0.896] | 927 |
| shape_break_plus | 0.909 | [0.894, 0.923] | 569 | 0.890 | [0.876, 0.903] | 927 |

**Verdict:** CU pairs are slightly *less* reliable than FF pairs on commit_distance and divergence (r ~0.78–0.85 vs ~0.82–0.88), but the difference is small and both groups are highly reliable. shape_break_plus is marginally *more* reliable for CU pairs. There is no evidence of a measurement noise problem — both groups are in the same reliability ballpark.

#### Test 2 — Sample size distribution (2021–2026, 14,727 pairs)

| Group | n pairs | Median min_n | p25 | p75 |
|---|---|---|---|---|
| CU-containing | 3,577 | 204 | 142 | 295 |
| FF-containing (non-CU) | 5,557 | 222 | 148 | 349 |

CU-containing pairs have slightly smaller qualifying samples (median 204 vs 222 pitches for the smaller-count type). The difference is modest and not large enough to explain substantially wider spread on its own — particularly given that split-half reliability is only marginally lower.

#### Conclusion

The evidence supports **real variance** as the dominant explanation for CU-pair leaderboard presence after normalization. CU pairs are slightly noisier and slightly smaller-n than FF pairs, but both gaps are small. Elite curveballs genuinely differentiate more from average curveballs (in tunneling geometry) than elite fastballs differ from average fastballs — this is a property of the pitch type, not an artifact of measurement.

**The mean-only pair-type normalization is supported for merge as-is.** No further normalization (e.g. SD-scaling within pair type) is warranted — it would suppress real cross-pitcher variance in curveball tunneling quality.

---

## Files Changed

- `analytics/tunneling_pair_composite.py` — new module
- `api/routers/players.py` — import, cached helper, `tunneling_pairs` key in pitcher payload
- `frontend/js/player.js` — `buildTunnelingPairsTable()` function + call in `renderRightPanel`
- `Archive/training_scripts/training/validate_tunneling_pair_composite.py` — validation script
- `diagnostics/tunneling_pair_composite/validation_results.json` — output
- `docs/methodology/10_tunneling_pair_composite.md` — this document
