# Tunneling+ v3 — trained model, real within-at-bat sequential pairs

Supersedes v2's sequential grain (`training/train_tunneling_v2_model.py`,
which only compared each pitch to the single immediately-preceding pitch).
**Per direct decision, Tunneling+ moves to a fully trained model going
forward — no hand-built composite alongside it.** v2's hand-built
candidates and leaderboards remain on disk as historical reference only,
untouched.

**Confirmed untouched**: `analytics/tunneling_model.py`, `training/
build_tunneling_report.py`, `models/tunneling_v1_*.json` all carry Aug 20
timestamps (before this project's Tunneling+ v2/v3 work began). v2's own
scripts and artifacts (`training/build_tunneling_v2_dataset.py`, `training/
tunneling_v2_candidates.py`, `models/tunneling_v2_*.txt`, `training/
tunneling_v2_artifacts/part_d_f_summary.csv`) carry timestamps from earlier
today's v2 session and were not modified by this session. The one shared
file touched this session is `analytics/tunneling_features.py` — v3's new
pairing functions were **added** to it (it's shared physics/feature code
both v2 and v3 legitimately build on, not a "v1 or v2 artifact" itself);
every v2 function already in that file is unchanged. No live-serving code
(`api/routers/`) was touched, nothing was pushed to production.

---

## Part A — Real within-at-bat pairing logic

Every pitch from the 2nd-onward in its at-bat pairs against **every** real
prior pitch in that at-bat, up to 6 pitches back — not filtered by
same-vs-different pitch type (a genuine, deliberate difference from v2's
sequential grain, which excluded same-type consecutive pairs). The 1st
pitch of every at-bat is excluded entirely, not imputed.

**Verified against real data before building anything**, using a real
4-pitch at-bat (FF, CH, FF, FF):

| Pitch | Type | lag1 | lag2 | lag3 | lag4-6 |
|---|---|---|---|---|---|
| #2 | CH | FF | — | — | — |
| #3 | FF | CH | FF | — | — |
| #4 | FF | FF | CH | FF | — |

Exact match to the worked example in the task spec, including the
same-type lag pair (pitch #4's lag1 is FF vs. its own FF) correctly
**not** excluded.

---

## Part B — Feature construction

Reused v2's **200ms** commit-point definition (the best-performing variant
from v2's hand-built grid) via v2's own `pitcher_season_commit_point`
table — not re-derived. For each of 6 lag slots: `lagN_commit_distance`,
`lagN_plate_distance`, `lagN_shape_break` (all real inches, computed
against that *specific* prior pitch, not an aggregate), `lagN_pitch_type`
(categorical). A lag slot beyond how many real prior pitches exist is true
NaN across all 4 of its columns.

**Schema, confirmed before training**: 24 lag-derived columns (6×4) + the
current pitch's own `pitch_type`/`p_throws`/`stand`/`season` = **28 total
features** (18 numeric lag columns + 6 categorical lag-pitch-type columns
+ 4 current-pitch categorical columns). Null rate by lag slot, exactly as
physically expected (fewer at-bats reach deeper lags):

| Lag | Null rate |
|---|---|
| 1 | 0.0% |
| 2 | 30.6% |
| 3 | 56.0% |
| 4 | 75.3% |
| 5 | 88.2% |
| 6 | 95.3% |

**3,044,748 training rows** (one per real 2nd-or-later pitch), before the
Part C threshold and target-null drop → **3,044,719** rows actually used.

---

## Part C — Qualifying threshold

Single threshold, exactly as instructed: exclude any pitcher-season with
< 100 **total** pitches that season (post PO/UN/EP exclusion, KN kept) —
not a per-type count, not a pair count. v1/v2's `MIN_PITCHES_PER_TYPE`
does not apply here. 5,169 total pitcher-seasons → **4,268 qualify**.
Applied to both training data and the leaderboard.

---

## Part D — Trained model

Single LightGBM model, Huber loss, leave-one-season-out CV,
`assert_no_identifiers()` guard, target = `delta_run_exp` of the row's own
pitch — same conventions as Stuff+/Deception+/v2's trained grains.

| | v3 | v2 sequential grain | v2 arsenal grain |
|---|---|---|---|
| CV R² (mean) | **0.0074** | 0.0021 | -0.0005 |
| CV MAE (mean) | 0.1321 | 0.1335 | 0.1203 |
| Training rows | 3,044,719 | 1,936,045 | 3,676,649 |

**v3's CV R² is more than 3x v2's sequential grain**, and comfortably
beats every other model in this project's Tunneling+/Stuff+ family on
this metric. The two most likely drivers: (a) not filtering to
different-type-only pairs roughly doubled the usable row count, and (b)
having up to 6 real lag slots of context per row, instead of just 1, gives
the model genuinely more signal to work with per prediction.

### Full per-feature SHAP (30k sample)

```
lag1_plate_distance   0.00595      lag5_commit_distance  0.00094
lag2_plate_distance   0.00324      lag5_plate_distance   0.00079
lag2_commit_distance  0.00273      lag1_shape_break      0.00056
lag3_commit_distance  0.00241      lag2_shape_break      0.00049
lag1_commit_distance  0.00237      lag6_commit_distance  0.00043
lag3_plate_distance   0.00216      lag2_pitch_type       0.00042
pitch_type            0.00167      lag3_shape_break      0.00041
lag4_commit_distance  0.00150      season                0.00037
lag4_plate_distance   0.00143      lag6_plate_distance   0.00033
lag1_pitch_type       0.00096      lag5_shape_break      0.00031
                                    lag4_shape_break      0.00030
                                    lag3_pitch_type       0.00023
                                    lag4_pitch_type       0.00019
                                    lag5_pitch_type       0.00018
                                    lag6_pitch_type       0.00017
                                    lag6_shape_break      0.00013
                                    stand                 0.00009
                                    p_throws              0.00002
```

### Lag-slot SHAP breakdown — the direct answer to the recency-decay question (Part E.2)

| Lag | Distance-group total (commit+plate+shape_break) | pitch_type |
|---|---|---|
| 1 | **0.00888** | 0.00096 |
| 2 | 0.00646 | 0.00042 |
| 3 | 0.00498 | 0.00023 |
| 4 | 0.00323 | 0.00019 |
| 5 | 0.00204 | 0.00018 |
| 6 | **0.00089** | 0.00017 |

**Clean, monotonic decay with lag, exactly as a real recency effect should
look.** But the answer to "is the 6-pitch cap well-supported, or cutting
off real signal": **mixed, and worth knowing**. Lag6's distance-group
total (0.00089) is genuinely small — about 10% of lag1's magnitude — but
it is **not zero, and not visually flattened out** relative to lag5
(0.00204). The ratio lag(N+1)/lagN is roughly 0.73, 0.77, 0.65, 0.63, 0.44
- decelerating but not yet fully flat by lag6. This suggests the cap
is reasonable (deeper lags are clearly diminishing) but **may still be
trimming a small amount of real, non-noise signal** rather than sitting
comfortably past the point where the recency effect has fully died out.
Not a reason to change the cap this session - just the honest empirical
read the task asked for.

**Does knowing the prior pitch TYPE add value beyond the raw distance
alone?** Yes, at every lag - `lagN_pitch_type`'s SHAP is real and non-zero
through lag6 (0.00017), and it decays alongside the distance features
rather than mattering only at short range. It's consistently smaller than
the distance-group total at every lag (roughly 5-15% of it), so type
context is a real but secondary contributor - the distances are doing most
of the work, but the model does use "what pitch was this" as more than
noise.

---

## Part E — Validation battery

### E.1 — Sequential same-at-bat test, corrected methodology

**Exactly how this is computed, stated explicitly per instruction**: v1/v2
bucketed a single PAIR-level score (one number per pitch-type-pair, or per
real different-type consecutive pair) into terciles and checked whiff rate
on the second pitch. v3 has no such single pair-level score - every
training row already IS "a real pitch with a real predecessor" by
construction, and its prediction already reflects up to 6 real lag slots
at once. The v3 version buckets the model's own final prediction (raw
predicted delta_run_exp, lower = better/more tunneled) into terciles
across all 3,044,719 real training rows and checks real whiff rate on that
same pitch.

**Result**: whiff rate by tercile - low_pred (best) **14.16%**, mid_pred
12.63%, high_pred (worst) **10.27%**. Correctly signed and monotonic - a
genuine ~38% relative spread between the best and worst terciles. z-score
for the low-vs-high difference: **84.68**.

**This z-score is explicitly NOT comparable to v1 (z=9.06) or v2 (z=9.3-
10.0) at face value, and that's said plainly rather than treated as "v3
crushed the old versions."** The z-score here is enormous almost entirely
because of sample size (n=3.04M vs. v1/v2's much smaller filtered
subsets of real consecutive pairs) - z scales with √n for a fixed effect
size, and this test's n is roughly 1,000x larger than what v1/v2 could use.
**The actually-comparable number is the whiff-rate spread itself** (14.16%
vs. 10.27% here), which is a real, correctly-signed, meaningfully-sized
effect - a legitimate positive result, just not one whose z-score should
be read as "9x more significant than v1."

### E.2 — Lag-slot SHAP breakdown

Covered above, under Part D - its own clearly labeled section as
instructed.

### E.3 — Year-over-year

| Seasons | n | r² |
|---|---|---|
| 2021→2022 | 492 | 0.277 |
| 2022→2023 | 494 | 0.349 |
| 2023→2024 | 490 | 0.292 |
| 2024→2025 | 481 | 0.283 |
| 2025→2026 | 479 | 0.298 |
| **Pooled** | **2,436** | **0.299** |

**Notably weaker than v2's best hand-built candidate** (pooled r² 0.585
with `arm_angle_consistency` layered on usage-product weighting), and also
weaker than v2's plain baseline candidates (0.43-0.49 range). Reported
plainly: whatever v3 gained in raw per-pitch CV fit, it did not carry
through to season-to-season score stability at the same level the best
hand-built composite achieved.

### E.4 — Cross-model correlation (2025, n=688)

r(Stuff+) = **0.074**, r(Deception+) = **0.154**. Both comfortably inside
or below v1's originally-cited 0.12-0.19 range - consistent with a
genuinely distinct signal from either of the other two models.

### E.5 — Out-of-sample predictive validity (2024 model → real 2025 outcomes, min 300 pitches, n=276)

| Outcome | v3 | v2 sequential grain (recorded) | v2 arsenal grain (recorded) |
|---|---|---|---|
| xwOBA | **-0.066** (correct) | 0.012 (near-zero) | -0.220 (correct) |
| whiff% | -0.092 (wrong sign) | -0.214 (wrong sign) | 0.103 (correct) |
| swstr% | -0.016 (wrong, ~negligible) | -0.177 (wrong sign) | 0.147 (correct) |
| K-BB% | **0.173** (correct) | 0.041 (near-zero) | 0.256 (correct) |
| chase% | **0.171** (correct) | 0.083 (weak) | 0.284 (correct) |

**A genuine mixed result, reported as such.** v3 is a real improvement
over v2's sequential grain on 4 of 5 outcomes (xwOBA, K-BB%, chase%
noticeably better; swstr% roughly a wash) and matches its whiff%
weakness in direction, though at smaller magnitude (-0.092 vs. -0.214).
It does not match v2's arsenal grain's clean, fully-correctly-signed
profile - whiff% and (marginally) swstr% still point the wrong way here.
The same plausible mechanism flagged in the v2 report likely still
applies: per-pitch-instance features carry real noise that a two-season-
ahead projection doesn't fully avoid, even with the richer 6-lag context.

### E.6 — fastest_fb_velo correlation

r = **0.085** - a small positive correlation, reported plainly per
instruction (an accepted design consequence of the pitcher-specific
commit point, not a bug).

### E.7 — Leaderboard, 2025, min 100 total pitches (688 qualified)

**Top 15**: A.J. Puk (AZ) 103.9, Jayden Murray (CHC) 103.0, Jesus Tinoco
(MIA) 102.7, Andrew Kittredge (BAL) 102.5, Trevor Williams (WSH) 102.4,
Davis Daniel (CIN) 102.3, Eduard Bazardo (SEA) 102.2, Ryan Johnson (LAA)
102.2, Jacob deGrom (TEX) 102.1, Tanner Scott (LAD) 102.1, Taylor Rogers
(MIN) 102.0, Max Kranick (WSH) 102.0, Tomoyuki Sugano (COL) 101.9, Kevin
Gausman (CHC) 101.8, Phil Maton (CHC) 101.8.

**Bottom 15**: Michael Kopech (LAD) 94.9, Luis Contreras (HOU) 95.6, Ian
Anderson (ATL) 96.2, Félix Bautista (BAL) 96.5, Jalen Beeks (TEX) 96.5,
Carson Whisenhunt (SF) 96.5, Chayce McDermott (NYM) 96.7, Mitch Farris
(LAA) 96.7, Sam Moll (CIN) 96.8, Erik Miller (BOS) 97.0, Albert Suárez
(BAL) 97.0, Danny Young (ATL) 97.0, Sean Guenther (DET) 97.0, Joey
Lucchesi (LAA) 97.1, Freddy Tarnok (MIA) 97.1.

**A structurally important difference from v2's leaderboards**: v3's score
range is **94.9-103.9** (a ~9-point spread), far tighter than v2's
hand-built leaderboard's **57.2-142.1** (an ~85-point spread) or even v2's
trained sequential grain's 94.9-103.4. This isn't a bug - it's a direct
consequence of averaging millions of individually-noisy per-pitch
predictions per pitcher rather than aggregating a smaller number of
season-level pair ratios, which naturally regresses individual pitchers'
displayed scores much closer to the mean. Worth knowing before comparing
"how extreme" any single pitcher's v3 score looks against a v2 number -
the scales are not interchangeable at face value.

---

## Part F — Deliverable summary

1. **New artifacts, all under `tunneling_v3_*` naming**: `models/
   tunneling_v3.txt` (+`_meta.json`/`_report.json`), `training/
   tunneling_v3_artifacts/tunneling_v3_features.parquet`, `_scores.parquet`,
   `_per_pitch_preds.parquet`, `_validation.json`,
   `_predictive_validity.json`. v1 and v2 artifacts confirmed untouched
   (file timestamps checked directly, not assumed).
2. **v2 comparison, in one place**:

   | Metric | v1 | v2 best hand-built | v2 sequential (trained) | v2 arsenal (trained) | **v3** |
   |---|---|---|---|---|---|
   | Sequential test | z=9.06 | z=9.3-10.0 | (not separately tested) | (not separately tested) | z=84.7\* / 14.2% vs 10.3% whiff spread |
   | YoY pooled r² | 0.31-0.53 | up to 0.585 | (not reported) | (not reported) | 0.299 |
   | r(Stuff+) / r(Deception+) | ~0.12-0.19 both | 0.05-0.28 range | (not reported) | (not reported) | 0.074 / 0.154 |
   | Out-of-sample whiff%/swstr% | — | — | wrong-signed | correctly-signed | wrong-signed (smaller) |
   | Out-of-sample xwOBA/K-BB%/chase% | — | — | weak/near-zero | correctly-signed, strong | correctly-signed, moderate |

   \*z=84.7 is not apples-to-apples with v1/v2's numbers - see E.1.

3. **Lag-slot SHAP breakdown** - its own section under Part D/E.2 above:
   clean recency decay, lag6 still ~10% of lag1's magnitude (not fully
   flattened - the 6-pitch cap is reasonable but may trim a little real
   signal), and prior-pitch-type context adds real value beyond raw
   distance at every lag.
4. **Top 15 / Bottom 15 leaderboard** - Part E.7 above, with the explicit
   note that v3's score range is much tighter than v2's.
5. **No production changes made.** v1 and v2 remain exactly as they were.
