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
composite_z = (−commit_distance_z + shape_break_plus_z) / 2
composite_score = 100 + 10 × composite_z   [plus-scale]
```

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

### ⚠ Sequential Whiff-Rate Tercile Test — FAILED

| Tercile | Whiff Rate | n |
|---|---|---|
| Low composite | 12.6% | 94,204 |
| High composite | 11.3% | 94,150 |
| Diff | −1.34 pp | |
| z-stat | −8.99 | |

The test shows the **opposite** of the expected direction: high-composite pairs yield fewer whiffs on the next pitch, not more. This is statistically significant.

**Interpretation:** The composite captures commit-point distance and shape break relative to a straight-line midpoint, which are geometric tunneling properties. These do not appear to translate to a same-at-bat sequential whiff lift. Possible reasons:

1. The outcome metric (mean_pair_dre from the sign check) reflects run value over full seasons, not pitch-to-pitch whiff tunneling. A pair that reduces run value may do so through chase rates, weaker contact, or count management rather than through induced swings-and-misses on the next pitch.
2. High-composite pairs often involve breaking ball + fastball combinations. Breaking balls thrown after fastballs may have structurally lower whiff rates than other orderings, confounding the test.
3. commit_distance tightness suppresses reactability for the batter in aggregate across many at-bats, without necessarily generating immediate per-pitch whiff inflation.

**Decision:** Build remains on branch. The face validity (top pairs look like real elite tunnelers; bottom pairs are clustered around pitchers with awkward shape combinations) is real. But the whiff-rate test failure is an honest red flag and is documented here. Recommend reviewing before merging.

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
