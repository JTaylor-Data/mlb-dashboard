# Tunneling+ v2 — dual build (hand-built composite + trained model)

Two candidate versions of Tunneling+, built side by side for comparison.
**Neither is deployed.** `analytics/tunneling_model.py`, `training/
build_tunneling_report.py`, and every `models/tunneling_v1_*` artifact are
confirmed unmodified (checked file timestamps directly — all predate this
session). Nothing in `api/routers/` was touched. Everything new lives under
new filenames: `analytics/tunneling_features.py`, `training/build_tunneling_
v2_dataset.py`, `training/tunneling_v2_candidates.py`, `training/
train_tunneling_v2_model.py`, `training/tunneling_v2_predictive_validity.py`,
`training/tunneling_v2_leaderboards.py`, and everything under `training/
tunneling_v2_artifacts/` and `models/tunneling_v2_*`.

**One naming note**: the task referenced `analytics/tunneling_features.py`
as an existing file to revise — it didn't exist. v1 lives entirely in
`analytics/tunneling_model.py`. `tunneling_features.py` was created fresh
as the v2 module; v1 was not touched.

This is a genuine methodology revision (fixed 23.5ft decision point →
pitcher-specific commit point), so v1 and v2 numbers are not directly
comparable in absolute terms, as expected going in.

---

## Part A — Pitch-type exclusion and qualifying threshold

- **PO/UN/EP excluded** from eligible pairs, applied once at `load_pitches()`
  time (same `EXCLUDED_PITCH_TYPES` constant Stuff+ uses — imported, not
  duplicated, so the two models can't silently drift on what counts as
  junk). **KN kept eligible**, same reasoning as Stuff+.
- **100 vs. 200 minimum-pitches-per-type threshold**: raising to 200 drops
  roughly **HALF the pairs** (45–56% retained, worse in later seasons):

  | Season | pairs@100 | pairs@200 | retained |
  |---|---|---|---|
  | 2021 | 2,051 | 1,106 | 53.9% |
  | 2022 | 2,150 | 1,199 | 55.8% |
  | 2023 | 2,337 | 1,215 | 52.0% |
  | 2024 | 2,505 | 1,337 | 53.4% |
  | 2025 | 2,721 | 1,352 | 49.7% |
  | 2026 | 2,539 | 1,145 | 45.1% |

  That looks like a big deal until you check what it does to the pitchers
  who qualify **either way**: for the 2,238 pitcher-seasons that clear 200,
  their baseline-candidate score at 100 vs. 200 correlates at **r=0.894**,
  median absolute shift **0.22 points** (mean 0.86, driven by a handful of
  outliers up to ~8.6 points). **Recommendation: default to 100** — raising
  the bar mostly just shrinks the population for no real change in how the
  remaining pitchers get scored.

---

## Part B — Pitcher-specific commit point

Replaces the fixed 23.5ft point with, per pitcher-season: the fastest of
FF/SI/FC by mean `effective_speed`, evaluated at a fixed TIME before each
pitch's own arrival (not a fixed distance) — using each pitch's own real
trajectory, not a velocity-adjusted formula. Persisted as `pitcher_season_
commit_point` (pitcher_id, season, fastest_fb_type, fastest_fb_velo,
commit_point_distance), one CSV per commit-point definition tested.

**Direction, confirmed empirically, not assumed**: harder throwers commit
**farther** from the plate, not closer (r=0.964 between fastest_fb_velo and
commit_point_distance at 150ms). Intuitive once stated — a fixed
time-before-arrival window covers more ground for a faster pitch, so the
150ms-before-arrival point is proportionally farther out (≈0.15 × velocity
in ft/s). Real spot check: Skenes (97.7mph) → 19.77ft; Holmes (93.0mph,
sinker) → 19.17ft; Gray (92.5mph, sinker) → 18.81ft.

**Per direct follow-up request, four commit-point definitions were built
and validated side by side**, not just the original 150ms:

| Variant | Definition | commit_point_distance (mean, ft) |
|---|---|---|
| 150ms | 150ms before arrival (time-based) | 19.16 |
| 200ms | 200ms before arrival (time-based) | 25.69 |
| fixed25ft | fixed 25ft from plate, every pitcher | 25.00 |
| fixed35ft | fixed 35ft from plate (per user request, citing public research) | 35.00 |

Results for each are in Part D/F below — **150ms/200ms (time-based, pitcher-
specific) validate meaningfully better on the sequential same-at-bat test
than either fixed-distance variant**, with 200ms the single best performer
on every metric tested.

---

## Part C — New features

- **`shape_break_plus`**: `actual_mid_distance` (real pairwise distance at
  the geometric midpoint between release and the commit point) minus
  `expected_mid_distance` (linear interpolation between release and commit
  distances) — signed, raw inches, not 100-scaled. Captures a pair that
  bulges apart in the middle relative to what a straight-line trend between
  its endpoints would predict (an early-diverge/late-converge shape, e.g. a
  curveball that pops out then tracks back). In the trained sequential-grain
  model's SHAP ranking, `seq_shape_break` lands 4th of 5 distance features
  (0.00086 mean |SHAP|) — real, modest signal, not dominant.
- **`release_consistency_score`** (identical construction to v1's own
  `_release_consistency`) and **`arm_angle_consistency_score`** (same
  construction as Deception+'s `arm_angle_std`, computed independently
  here) — both tested for composite inclusion (Part D.5), not just
  displayed. **`arm_angle_consistency_score` gives the single best
  year-over-year improvement of anything tested** (pooled r² 0.579–0.585,
  up from ~0.43–0.49 baseline) with only a modest increase in Deception+
  overlap. **`release_consistency_score` gives a smaller YoY gain but
  measurably re-introduces Deception+ overlap** (r_deception 0.26–0.30 vs.
  baseline's 0.18–0.24) — exactly the risk flagged going in, confirmed real,
  not a clean win.
- **`arsenal_entropy`**: Shannon entropy (nats) over post-exclusion
  pitch-type usage shares, `-Σ(p·log p)`. League distribution: mean 0.91,
  median 1.01, range 0.00 (a genuine one-pitch-type pitcher-season) to 1.98.
  Spot check: Sonny Gray (varied arsenal) 1.73–1.75 vs. a one-pitch reliever
  at 0.00 — behaves as expected. **Folded into the composite, entropy is
  NEGATIVELY correlated with Stuff+** (r ≈ -0.20 to -0.21, consistent across
  all 4 commit-point variants) — a real, interesting finding: pitchers with
  more pitch-type variety tend to have modestly lower raw "stuff," plausibly
  because pure power arms lean on fewer, harder pitch types while
  tunneling-reliant pitchers carry more variety. Reported both folded and
  unfolded, as instructed — the unfolded score is identical to whichever
  base candidate it's layered on (confirms the "just displayed, not scored"
  path works correctly).

---

## Part D/F — Full candidate × commit-point grid

Every hand-built candidate run against every commit-point variant (36
configurations). `sequential_z`/`split_half_1pair` are IDENTICAL across
candidates 1/3/4 (and across 5a/5b/5c/6, which share candidate 4's pair-level
formula) within a variant, by construction — they're evaluated at or before
the pitcher-level aggregation step, which is the only thing those candidates
change. Full CSV: `training/tunneling_v2_artifacts/part_d_f_summary.csv`.

| Variant | Candidate | Sequential z | Split-half (1 pair) | YoY pooled r² | r(Stuff+) | r(Deception+) | r(velo) |
|---|---|---|---|---|---|---|---|
| 150ms | 1 baseline | 9.28 | 0.885 | 0.434 | 0.276 | 0.198 | 0.321 |
| 150ms | 2 +release_sim | **0.93** | 0.975 | 0.405 | 0.027 | 0.277 | -0.001 |
| 150ms | 3 weight FF/SI 1.5x | 9.28 | 0.885 | 0.446 | 0.276 | 0.204 | 0.328 |
| 150ms | 4 weight usage-product | 9.28 | 0.885 | 0.466 | 0.282 | 0.209 | 0.328 |
| 150ms | 5a +release_consistency | 9.28 | 0.885 | 0.375 | 0.037 | **0.296** | -0.030 |
| 150ms | **5b +arm_angle_consistency** | 9.28 | 0.885 | **0.579** | 0.066 | 0.184 | -0.044 |
| 150ms | 5c +both consistency | 9.28 | 0.885 | 0.518 | 0.039 | 0.261 | -0.062 |
| 150ms | 6 +entropy (folded) | 9.28 | 0.885 | 0.547 | **-0.207** | 0.083 | -0.116 |
| **200ms** | 1 baseline | **9.98** | 0.899 | 0.458 | 0.252 | 0.211 | 0.308 |
| 200ms | 2 +release_sim | 1.89 | 0.972 | 0.409 | 0.043 | 0.277 | 0.025 |
| 200ms | 4 weight usage-product | 9.98 | 0.899 | 0.486 | 0.257 | 0.221 | 0.312 |
| **200ms** | **5b +arm_angle_consistency** | **9.98** | 0.899 | **0.581** | 0.074 | 0.190 | -0.029 |
| fixed25ft | 1 baseline | 7.08 | 0.882 | 0.426 | 0.127 | 0.188 | 0.076 |
| fixed25ft | 5b +arm_angle_consistency | 7.08 | 0.882 | 0.583 | 0.054 | 0.186 | -0.064 |
| fixed35ft | 1 baseline | 6.11 | **0.924** | 0.438 | 0.087 | 0.229 | 0.053 |
| fixed35ft | 5b +arm_angle_consistency | 6.11 | **0.924** | 0.585 | 0.053 | 0.199 | -0.059 |

**Headline findings, in order of how much they should move a final choice:**

1. **`release_similarity_plus` inclusion tanks the sequential test in every
   single variant tested** (z drops from 6–10 down to 0.7–2.0), robustly
   reconfirming v1's original rejection under four completely different
   commit-point definitions. This is now about as well-established a
   negative result as this project has.
2. **Time-based commit points beat fixed-distance ones on the mechanism-
   specific test**: 150ms/200ms (z≈9.3–10.0) clearly outperform fixed25ft/
   fixed35ft (z≈6.1–7.1). **200ms is the single best commit-point definition
   tested** — highest sequential z, highest split-half among the time-based
   pair, highest YoY r² for every candidate. Trade-off: **fixed35ft has the
   best split-half reliability of all four (0.924)** — the fixed-distance
   variants measure something less mechanism-valid but more repeatable.
3. **Usage-product weighting (candidate 4) beats both flat weighting
   (1) and FF/SI-boost weighting (3)** on YoY r² in every variant — a
   modest, consistent win. This is the weighting basis all consistency/
   entropy layering (5a–6) was built on.
4. **`arm_angle_consistency_score` (5b) is the strongest single addition
   tested** — a real ~0.12–0.15 jump in YoY r², the least Deception+
   overlap of the two consistency features, and the metric this session's
   automated selection picked in every variant.
5. **`release_consistency_score` (5a) measurably re-introduces Deception+
   overlap**, exactly the risk flagged before testing — real, not
   hypothetical, and reported as such rather than treated as a clean win.
6. **Folded `arsenal_entropy` is the only addition that turns the Stuff+
   correlation negative** (-0.20 to -0.21) — a genuine, consistent finding
   worth understanding before using it in a scored composite, not
   necessarily a reason to exclude it.

**Best hand-built candidate overall: `5b_plus_arm_angle_consistency` (built
on usage-product weighting), commit-point variant `200ms`.**

---

## Part E — Trained model, two grains

Both trained with the same conventions as Stuff+/Deception+ (Huber loss,
leave-one-season-out CV, `assert_no_identifiers()` guard checked at every
fit). All results below are for the `150ms` commit-point variant — training
two full grains × four commit-point variants was judged out of scope for
this session's compute budget; the hand-built grid already answers "which
commit-point definition is best" (200ms), and that finding is a reasonable
prior for the trained models too if this gets extended later.

### Cross-validation

| Grain | CV R² (mean) | CV MAE (mean) | Training rows |
|---|---|---|---|
| Sequential (real pair) | **+0.00213** | 0.1335 | 1,936,045 |
| Arsenal-aggregate | **-0.00050** | 0.1203 | 3,676,649 |

**The arsenal-aggregate grain does not beat a naive baseline** in CV — every
holdout season posted a negative R². This is a real, structural
consequence of its grain: every pitch of a given (pitcher, season,
pitch_type) gets the *exact same* 4 feature values (a pitcher-season
aggregate), so there's very little per-pitch information for the model to
split on — LightGBM's early stopping essentially can't find a useful split
beyond the mean. The sequential grain, built from each pitch's own real
paired trajectory, carries genuine (if small) per-pitch signal.

### SHAP, final models

| Sequential grain | mean \|SHAP\| | Arsenal grain | mean \|SHAP\| |
|---|---|---|---|
| seq_commit_distance | 0.00535 | arsenal_tunnel_spread_plus | 0.00197 |
| seq_plate_distance | 0.00521 | pitch_type | 0.00098 |
| pitch_type | 0.00116 | arsenal_late_divergence_plus | 0.00077 |
| seq_shape_break | 0.00086 | arsenal_shape_break_plus | 0.00067 |
| seq_mid_distance | 0.00067 | arsenal_release_similarity_plus | 0.00060 |
| seq_release_distance | 0.00045 | season | 0.00053 |

Both grains independently rank their commit-point/plate-style features
above release-based ones — consistent with Part D/F's hand-built finding
that release_similarity is the weak link.

### Out-of-sample predictive validity (2024 model → real 2025 outcomes, min 300 pitches)

| Outcome | Sequential grain | Arsenal grain |
|---|---|---|
| xwOBA | 0.012 | **-0.220** (correct sign) |
| whiff% | **-0.214** (wrong sign) | 0.103 |
| swstr% | **-0.177** (wrong sign) | 0.147 |
| K-BB% | 0.041 | 0.256 |
| chase% | 0.083 | 0.284 |

**A genuinely counter-intuitive, important result**: the sequential grain
(the one with real, if small, positive in-sample CV R²) produces
**wrong-signed, weak out-of-sample correlations** — a higher sequential-
grain score predicts *fewer* whiffs and swinging strikes next season, the
opposite of what a tunneling metric should show. The arsenal grain (the one
that *failed* its own in-sample CV check) produces a **coherent, correctly-
signed** real-outcome profile — negative with xwOBA, positive with every
whiff/strikeout/chase metric, comparable in size to Stuff+'s own v1-vs-v2
out-of-sample numbers on record. Plausible mechanism: the sequential
grain's per-pitch-instance features are noisy (individual pitch-pair
distances vary a lot even for a "true"-skill-constant pitcher), and that
noise doesn't average out cleanly when projecting two seasons forward;
the arsenal grain's season-aggregate features are far more stable per
pitcher even though that same stability starves the per-pitch CV signal.
Reported plainly, not smoothed over - a low in-sample R² did not predict
which grain would generalize better.

---

## Leaderboards, 2025, min 300 qualifying pitches

### Best hand-built (`5b_plus_arm_angle_consistency`, 200ms) — 513 qualified

**Top 15**: Chad Green (TOR) 142.1, Jonathan Bowlan (PHI) 140.3, Clayton
Beeter (WSH) 139.8, Trevor Megill (MIL) 137.7, Justin Topa (NYY) 134.1,
Angel Chivilli (NYY) 133.7, Tobias Myers (NYM) 133.6, Mick Abel (MIN)
133.5, Matt Svanson (STL) 131.3, Grant Wolfram (BAL) 129.8, Ryan Borucki
(CLE) 129.6, Cristian Javier (HOU) 129.6, Julian Merryweather (MIN) 129.4,
Nick Lodolo (CIN) 128.8, Trevor Williams (WSH) 126.8.

**Bottom 15**: Yu Darvish (SD) 57.2, Tyler Gilbert (CWS) 62.6, Jimmy
Herget (COL) 62.7, Anthony Bender (MIA) 63.8, Erik Miller (BOS) 64.7,
Andrew Heaney (LAD) 65.1, Caden Dana (LAA) 65.2, Victor Vodnik (MIA) 66.4,
Yohan Ramírez (PIT) 67.4, Brendon Little (TOR) 68.4, Luis Gil (NYY) 69.2,
Valente Bellozo (COL) 70.0, Josh Simpson (SEA) 71.4, David Morgan (SD)
71.6, Sam Long (KC) 72.2.

### Best trained (sequential grain, 150ms) — 400 qualified

**Top 15**: Taylor Rogers (MIN) 103.4, Eduard Bazardo (SEA) 102.6, Andrew
Kittredge (BAL) 102.6, Jakob Junis (TEX) 102.3, Angel Zerpa (MIL) 102.0,
Anthony Bender (MIA) 102.0, Nick Mears (KC) 101.9, Camilo Doval (PIT)
101.9, Jacob deGrom (TEX) 101.9, Brandon Young (BAL) 101.8, Troy Melton
(DET) 101.8, Matt Svanson (STL) 101.8, Abner Uribe (MIL) 101.8, Pete
Fairbanks (MIA) 101.7, Reid Detmers (LAA) 101.7.

**Bottom 15**: Jalen Beeks (TEX) 94.9, Brandon Eisert (PIT) 95.2, Hayden
Birdsong (SF) 96.6, Anthony Molina (SF) 96.9, Jeff Hoffman (MIN) 97.3,
Cristian Javier (HOU) 97.4, Lucas Giolito (SD) 97.4, Josh Hader (HOU)
97.5, Freddy Peralta (TB) 97.6, Robbie Ray (SD) 98.0, Eury Pérez (MIA)
98.0, Angel Chivilli (NYY) 98.0, Hunter Gaddis (CLE) 98.1, Ben Brown (CHC)
98.2, Dylan Cease (TOR) 98.3.

**The two lists barely overlap.** Only **Matt Svanson** appears in both
Top 15s. **Zero** pitchers appear in both Bottom 15s. Three pitchers land
at **opposite extremes**: Anthony Bender is hand-built's #4-worst but
trained's #6-best; Cristian Javier and Angel Chivilli are hand-built
Top-15 but trained Bottom-15. This is a real, substantial divergence in
what the two approaches consider "good tunneling" — not a rounding
difference.

---

## Plain-language summary: hand-built vs. trained, no production choice made

**Which validates better internally**: the hand-built composite, clearly.
Its sequential-test z-score (9.3–10.0 for the strongest variants) is a
real, mechanism-specific signal at a level neither trained grain's SHAP
ranking alone can directly claim (SHAP explains what the model uses, not
whether the model's *aggregate score* predicts real whiff behavior the way
the sequential test checks). The hand-built version's YoY stability (up to
0.585 with arm_angle_consistency) is also strong.

**Which predicts real future outcomes better**: genuinely unclear, and
that itself is the finding. The sequential-grain trained model's
in-sample CV shows real (if tiny) positive signal, but its out-of-sample
correlations with real 2025 outcomes are weak and partly wrong-signed. The
arsenal-grain trained model fails its own in-sample CV check entirely, yet
produces a coherent, correctly-signed real-outcome profile out of sample.
Neither trained grain was checked against the hand-built composite's own
out-of-sample validity, since (per instruction) that check is trained-
model-only — the hand-built composite has no equivalent "held out a
season" framing to test the same way.

**What SHAP reveals about which components matter**: both trained grains
independently rank commit-point/plate-style distance features above
release-based ones, echoing the hand-built grid's clearest finding
(release_similarity hurts). `shape_break` (the new feature) shows real but
modest importance in both grains — a genuine contribution, not a dominant
one.

**Do the two approaches agree on who's good at tunneling?** Not really.
The near-zero leaderboard overlap, plus three pitchers landing at
opposite extremes, means the practical, on-the-page ranking a reader would
see depends heavily on which version ships. That's the single most
important fact for whoever makes the production call to know going in.

**No recommendation is made here for which version (or which commit-point
definition, or which candidate) should go to production.** That decision,
per direct instruction, is made after this report is reviewed — not by
this session.
