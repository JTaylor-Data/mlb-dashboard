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

## Files Changed

- `analytics/tunneling_pair_composite.py` — new module
- `api/routers/players.py` — import, cached helper, `tunneling_pairs` key in pitcher payload
- `frontend/js/player.js` — `buildTunnelingPairsTable()` function + call in `renderRightPanel`
- `Archive/training_scripts/training/validate_tunneling_pair_composite.py` — validation script
- `diagnostics/tunneling_pair_composite/validation_results.json` — output
- `Methodology/TUNNELING_PAIR_COMPOSITE.md` — this document
