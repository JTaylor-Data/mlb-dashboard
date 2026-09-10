# Tunneling+ v3 — Methodology & Architecture Reference

**Purpose of this document**: a standalone refresher on how the current, live
Tunneling+ model actually works — not a validation report (see
`training/tunneling_v3_artifacts/TUNNELING_V3_REPORT.md` for the full battery
of numbers this version was built and checked against). Read this if you've
come back to the project after a while and need to reconstruct the mental
model of what Tunneling+ v3 measures and how, without re-deriving it from
code.

**Status**: v3 is the standard, live-serving version of Tunneling+ as of this
writing, powering the Tunneling+ Model page and the player-card badge. v1 and
v2 remain on disk, untouched, unused for scoring.

---

## 1. What tunneling is, in one paragraph

Two pitches from the same pitcher "tunnel" well if a hitter can't tell them
apart early enough to matter, but they end up doing very different things by
the time they reach the plate. v3 doesn't score this as an explicit
similarity ratio the way v1/v2 did — instead, it trains a model directly on
real outcomes, using engineered inputs that describe how each pitch relates
to what came immediately before it in the same at-bat. The model learns for
itself what "tunneling well" looks like, rather than being handed a
pre-defined formula for it.

## 2. The core architectural shift from v1/v2

| | v1 | v2 | **v3 (current)** |
|---|---|---|---|
| Scoring | Hand-built geometric-mean composite | 9 hand-built composite candidates + 2 trained-model grains | **Single trained LightGBM model, no hand-built composite** |
| Pairing | Every qualifying pitch-TYPE pair (aggregate) | Real consecutive different-type pairs only (lag-1) | **Real sequential pairs, lag 1 through 6 pitches back** |
| Same-type prior pitch | N/A (type-level aggregate) | Excluded | **Included** — a pitcher's 2nd straight fastball is real signal, not noise |
| Decision point | Fixed 23.5 ft from plate, every pitcher | Pitcher-specific, reused at scoring time (see §3) | **Same as v2** — reused directly, not re-derived |
| Output grain | One score per pitch-type pair, aggregated to a pitcher score | Per-pair or per-pitch, depending on grain | **One prediction per real pitch**, averaged to a pitcher-season score |

The single biggest conceptual change: **v3 has no "pair" concept at the
scoring level at all.** There is no `tunneling_plus` for "FF vs. CH" the way
v1 has. Every real pitch (from the 2nd pitch of an at-bat onward) is one
training row, described by its relationship to up to 6 real prior pitches in
that same at-bat, and the model predicts that pitch's own run-value outcome
directly.

## 3. The commit point (reused from v2, not re-derived)

"When does a hitter actually commit to a swing decision?" v1 used one fixed
answer for every pitcher (23.5 ft from the plate). v2 replaced that with a
**pitcher-specific** commit point: for each pitcher-season, find whichever of
FF/SI/FC that pitcher throws hardest (by `effective_speed`), then compute —
using that pitch's own real trajectory — how far from the plate it was
**200ms before it arrived**. That distance becomes this pitcher's own commit
point, applied uniformly to every pitch he throws that season (not just his
fastball).

v3 does not re-derive this — it directly reuses v2's saved 200ms commit-point
table (`training/tunneling_v2_artifacts/200ms/pitcher_season_commit_point.csv`,
also the file the live-serving module reads). A pitcher-season with no
qualifying FF/SI/FC pitch has no principled commit point and is silently
absent from v3's output — same edge case v2 had.

Confirmed relationship (not assumed): commit-point distance grows *with*
velocity — a harder thrower has covered more ground by a fixed time-before-
arrival, so his commit point sits farther from the plate, not closer.

## 4. The lag-pairing logic — the actual redesign in v3

This is the part that's genuinely new. For every real pitch from the **2nd
pitch of an at-bat onward** (the 1st pitch has no history and is excluded
entirely, not imputed), the model looks back up to **6 real prior pitches**
in that same at-bat. For each lag slot `N` (1 = immediately prior, up to 6 =
six pitches back), four columns are computed:

- **`lagN_commit_distance`** — real inches between this pitch and the pitch
  N-back, at *this pitcher's own* commit point (from §3).
- **`lagN_plate_distance`** — real inches between them at the plate.
- **`lagN_shape_break`** — how much the pair "pops apart early, tracks back
  late" (or the reverse), relative to a straight-line interpolation between
  release and the commit point. Positive means more separation appeared
  earlier than a straight-line path would predict.
- **`lagN_pitch_type`** — what pitch type was thrown N pitches back
  (categorical).

A lag slot beyond how many real pitches actually exist in that at-bat (e.g.
lag-4 for the at-bat's 2nd pitch) is **true NaN** across all 4 of that slot's
columns — never zero-filled, never imputed, never duplicated from another
slot. LightGBM's native missing-value handling deals with this directly.

6 lag slots × 4 columns = **24 lag-derived features**, plus 4 current-pitch
categoricals (`pitch_type`, `p_throws`, `stand`, `season`) = **28 features
total**. No pitch-type-pair identity feature exists anywhere — the model
only ever sees "this pitch" and "what happened N pitches before it."

**The deliberate difference from v2**: v2's sequential grain excluded a
same-type predecessor (two straight fastballs weren't treated as a
"tunneling pair" at all). v3 does **not** exclude same-type predecessors — a
pitcher's 2nd consecutive fastball is real information a hitter actually
experienced, not a case to throw out.

## 5. The qualifying threshold

A single rule, applied once: a pitcher-season needs **≥ 100 total pitches**
that season (after excluding PO/UN/EP, same junk-pitch-type set Stuff+ uses)
to be included at all. Unlike v1/v2, this is **not** a per-pitch-type or
per-pair count — just one number, applied to a pitcher-season's raw pitch
total before any lag-pairing happens.

## 6. Training target and model

- **Target**: `delta_run_exp` of the row's *own* pitch (the later pitch in
  each lag pair) — not a pair-level similarity score, not an aggregate. The
  model is asked directly: "given what this hitter just saw from this
  pitcher, how good was this specific pitch's real outcome?"
- **Model**: a single LightGBM regressor, Huber loss — same training
  conventions as every other model on this site (leave-one-season-out CV
  across 2021–2025, 2026 included as train-only since it's a partial
  season, `assert_no_identifiers()` guard against `pitcher_id`/`game_id`/etc.
  leaking in as features).
- **No hand-built composite anywhere alongside it** — this is a deliberate
  consistency choice, matching how Stuff+ and Deception+ are both single
  trained models with no separate hand-built formula running in parallel.

## 7. Scoring scale

Same convention as every other "+" model on this site: **100 = league
average for that season**, and **each 10 points = 1 standard deviation** from
that season's own mean predicted value. Scores are z-scored fresh within
each season (`100 - 10 * (pred - season_mean) / season_std`), so a 110 in
2023 and a 110 in 2025 mean the same relative standing, not necessarily the
same raw value.

**On the score spread**: v3's real leaderboard spread is tight (roughly
95–104 in a typical season) compared to Stuff+'s much wider range. This is
expected, not a defect — tunneling is a real skill, but an inherently
smaller lever a pitcher can pull independently compared to raw pitch
quality. A tight spread here is the honest shape of the underlying signal,
not evidence the model isn't differentiating pitchers.

## 8. What the validation battery actually checked (summary — see the full
report for numbers)

- **Sequential same-at-bat test, corrected for v3's own architecture**: v1/v2
  bucketed a single PAIR's score into terciles and checked real whiff rate on
  the second pitch. v3 has no such single pair-level score — instead, the
  model's own final prediction (which already reflects everything it learned
  from lag context, not one isolated pair) is bucketed into terciles across
  every real training row, and real whiff rate is checked on that same pitch.
  **Not a face-value apples-to-apples comparison to v1/v2's z-scores** — a
  structurally different question, answered honestly as such rather than
  glossed over.
- **Year-over-year stability**: does a pitcher's score this year predict his
  own score next year?
- **Cross-model correlation**: against Stuff+ and Deception+ (both expected
  to be low — tunneling is meant to be a distinct skill from either).
- **Fastest-fastball-velocity correlation**: a sanity check, not a target.
- **Out-of-sample predictive validity**: a model trained on every season
  except 2024, scored on 2024's pitches, correlated against each pitcher's
  REAL 2025 outcomes (xwOBA, whiff%, swstr%, K-BB%, chase%) — genuine
  out-of-sample, not in-sample fit.

## 9. Where everything lives

**Live-serving (what actually powers the site today)**:
- `analytics/tunneling_features.py` — shared feature engineering (v2 *and*
  v3; v3-specific pieces are `MAX_LAG`, `sequential_lag_features()`,
  `lag_feature_columns()`).
- `analytics/tunneling_v3_model.py` — runtime scoring module: loads
  `models/tunneling_v3.txt` + `_meta.json`, builds the 28-feature lag frame
  live per request, scores on demand (same on-demand pattern as
  `stuff_model.py`/`deception_model.py`). Provides the league-wide
  leaderboard, the player-card badge score, and per-pitcher SHAP breakdowns
  grouped by lag slot.
- `api/routers/tunneling_report.py` — the `/api/tunneling/v3/*` endpoints
  (additive; the original `/api/tunneling/*` v1 endpoints are untouched, see
  §10).
- `frontend/tunneling-model.html` + `frontend/js/tunneling-model.js` — the
  model-internals page.

**Training/offline (how the live model was produced)**:
- `training/build_tunneling_v3_dataset.py` — builds the lag-feature table.
- `training/train_tunneling_v3_model.py` — trains and saves the model.
- `training/tunneling_v3_validate.py` / `tunneling_v3_predictive_validity.py`
  — the validation battery, saved to `training/tunneling_v3_artifacts/`.

**Reused, not modified**: `training/tunneling_v2_artifacts/200ms/
pitcher_season_commit_point.csv` (v2's commit-point table — the one dependency
v3 has on v2's own artifacts).

## 10. What v3 deliberately does *not* touch

- **`analytics/tunneling_model.py` (v1)** is untouched — still the engine
  behind the Tunneling+ 3D trajectory viewer (`tunnel3d.html` /
  `api/routers/tunneling_3d.py`), which depends directly on v1's exact
  per-pitch-type-pair output shape (release/decision/plate distances per
  pair, anchor-pair flags) to render its hover detail. v3 has no equivalent
  quantity — there is no per-pitch-type-pair score in this architecture — so
  it was not (and structurally could not cleanly be) wired into that page.
  The 3D viewer's own blurb still correctly describes itself as showing v1's
  numbers.
- **v2's own hand-built composites and leaderboards** remain on disk,
  unused, as historical reference (`training/tunneling_v2_artifacts/`).
- Stuff+ and Deception+ — their own files, models, and features were never
  touched by any part of the v3 build.
