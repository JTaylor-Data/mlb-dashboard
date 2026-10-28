# Tunneling+ v4 Completion Report

**Generated**: 2026-10-27 22:49:49
**Branch**: tunneling-v4
**Status**: Experimental — NOT promoted to production
**Spec**: Pitcher-season grain trained model (LightGBM, Huber loss, LOSO CV)

---

## Part 0 — Live v2/v3 Sign Diagnostic

**Critical question**: Does the live v2 model positively or negatively correlate with good pitching outcomes?
Convention: `score = 100 - 10*(pred - mean)/std` -> lower predicted xRV = better pitcher = **higher score**
Expected: `r(score, mean_dre) < 0` (mean_dre measures average run value per pitch; better pitchers have lower/more-negative mean_dre)

| Model | n | r(score, mean_dre) | 95% CI | Sign correct? |
|-------|---|---------------------|--------|---------------|
| v2 (live) | 723 | -0.1578 | [-0.228, -0.086] | YES |
| v3 | 723 | -0.1805 | [-0.250, -0.109] | YES |

Both v2 and v3 have correct sign (r < 0). v2 also shows expected correlations: r(whiff%)=+0.254, r(K%)=+0.170.

**Note on v2 spot-check**: See `diagnostics/tunneling_v4/part0_sign_diagnostic.json` for the top-10 K% pitchers and their v2 scores.

---

## Part 1 — Pre-registration

See `diagnostics/tunneling_v4/spec_lock.json` for pre-registered feature set, CV scheme, and adoption rules.

---

## Part 2 — Feature Construction

**Grain**: pitcher-season
**Qualifying threshold**: >=100 total pitches (after PO/UN/EP exclusion; KN kept)
**Feature set**: 18 lag-mean numerics (6 lag slots × commit_distance, plate_distance, shape_break) + arsenal_entropy + season
Final features: ['lag1_commit_distance', 'lag1_plate_distance', 'lag1_shape_break', 'lag2_commit_distance', 'lag2_plate_distance', 'lag2_shape_break', 'lag3_commit_distance', 'lag3_plate_distance', 'lag3_shape_break', 'lag4_commit_distance', 'lag4_plate_distance', 'lag4_shape_break', 'lag5_commit_distance', 'lag5_plate_distance', 'lag5_shape_break', 'lag6_commit_distance', 'lag6_plate_distance', 'lag6_shape_break']

**Ablation results** (LOSO CV MAE, lower = better):

| Config | Mean MAE | Mean R2 |
|--------|----------|---------|
| With season, with entropy | 0.008938876037361956 | 0.08450334837917868 |
| No season, with entropy | 0.008966113625748649 | 0.07888954390506134 |
| With season, no entropy | 0.009087893430153053 | 0.059439446073228284 |
| No season, no entropy | 0.009144654521212823 | 0.05378035998985653 |

Season feature: DROPPED (diff ≤ 0.03)
Arsenal entropy: DROPPED (diff ≤ 0.03)

---

## Part 3 — Model Training

**Architecture**: LightGBM, Huber loss, pitcher-season grain
**CV**: Leave-one-season-out on 2021–2025
**Ensemble**: 3 seeds ([42, 123, 456]), predictions averaged

LOSO CV results (modern era 2021–2025):

| Holdout | n | R2 | MAE |
|---------|---|-----|-----|
| 2021 | 736 | 0.10185 | 0.00929 |
| 2022 | 712 | -0.00553 | 0.00897 |
| 2023 | 711 | 0.08597 | 0.00860 |
| 2024 | 712 | 0.02851 | 0.00906 |
| 2025 | 723 | 0.05811 | 0.00980 |

| **Mean** | — | **0.05378** | **0.00914** |

---

## Part 4 — Validation Battery

### 4.1 Sign Check

| Metric | Value | Expected | Pass? |
|--------|-------|----------|-------|
| r(score, mean_dre) | **−0.820** | < 0 | ✓ PASS |
| r(score, K%) | +0.409 | > 0 | ✓ PASS |
| r(score, BB%) | −0.386 | < 0 | ✓ PASS |

**⚠ IN-SAMPLE NOTE**: This r = −0.820 is computed in-sample. The final model was trained on all qualifying pitcher-seasons 2015–2026 including 2025 (the season scored here), and mean_dre is the model's own training target. This correlation does NOT reflect out-of-fold predictive power — it shows the model fits its training data. The out-of-fold evidence is in Parts 4.3 (YoY stability) and 4.5 (predictive validity), both of which failed their pre-registered gates.

### 4.2 Score Distribution

| Metric | v4 | v3 (reference) |
|--------|----|----|
| Score range | 27.4–140.8 | 94.9–103.9 |
| Spread (max−min) | **113.4** pts | 9.0 pts |
| SD | 10.00 | ~1.5 (est.) |

### 4.3 Year-over-Year Stability

| Pair | n | Pearson r | r2 |
|------|---|-----------|-----|
| 2021->2022 | 516 | 0.3270 | 0.1069 |
| 2022->2023 | 517 | 0.2960 | 0.0876 |
| 2023->2024 | 507 | 0.3388 | 0.1148 |
| 2024->2025 | 518 | 0.2900 | 0.0841 |
| 2025->2026 | 514 | 0.2141 | 0.0458 |
| **Pooled** | 2572 | 0.28670973298229313 | **0.08220247098677783** |

### 4.4 Split-Half Reliability (2025)

r = 0.4750128405393574 (n = 694, split by random game assignment, seed=42)
✗ FAIL or inconclusive (r < 0.60)

### 4.5 Out-of-Sample Predictive Validity

| Pair | n | r(xwOBA) | r(whiff%) |
|------|---|-----------|-----------|
| 2021->2022 | 516 | 0.13646767103820892 | -0.19507256335455259 |
| 2022->2023 | 517 | 0.10847204682389902 | -0.1391329264398689 |
| 2023->2024 | 508 | 0.16198504882864517 | -0.11411234416461416 |
| 2024->2025 | 518 | 0.14958851208642193 | -0.08768449989906221 |
| 2025->2026 | 514 | 0.07912218238595671 | -0.1371282205986931 |

### 4.6 Independence

| vs. | r |
|-----|---|
| Stuff+ | -0.32385579183082897 (n=723) |
| Location+ | -0.2762797678509203 (n=723) |

### 4.7 Sequential Same-at-bat Whiff-Rate Tercile Test (2025)

Pitchers bucketed by v4 score tercile; whiff rate measured on at-bats following the 1st pitch.
(high_score = best tunneling, should suppress opponent whiff rates the most)

| Tercile | n pitches | n swings | Whiff rate |
|---------|-----------|----------|------------|
| high_score | 93278 | 47555 | 0.20157712122805174 |
| mid_score | 216648 | 112697 | 0.21280956902135814 |
| low_score | 214288 | 114134 | 0.2312720135980514 |

z(high vs low) = -13.079924116210485

### 4.8 Leaderboard

**2025 Top 5** (range: 54.7–124.4 (sd=10.2)):
  Andrew Kittredge: 124.4
  Eduard Bazardo: 124.3
  Abner Uribe: 122.5
  Jacob deGrom: 121.4
  Hunter Harvey: 119.3
**2026 Top 5** (range: 44.1–121.4 (sd=10.6)):
  Jonathan Loáisiga: 121.4
  Robert Suarez: 120.9
  Garrett Stallings: 120.7
  Jack Sinclair: 120.0
  Louis Varland: 119.9


---

## Part 5 — Side-by-Side Comparison

| Metric | v2 (live) | v3 | v4 (experimental) |
|--------|-----------|----|--------------------|
| Sign r(score, mean_dre) 2025 | −0.158 ✓ | −0.181 ✓ | −0.820 ✓ (in-sample — see §4.1) |
| Score spread | ~113 pts | 9.0 pts | **113.4 pts** |
| YoY r² (pooled) | — | 0.299 | **0.082** ✗ |
| Split-half r | — | — | 0.475 ✗ |
| Predictive r(whiff%) next season | — | — | −0.10 to −0.20 |

---

## Recommendation

**DO NOT SHIP v4 YET: YoY stability is below acceptable floor. Score is not stable enough for a reliable leaderboard metric.**

### Detailed reasoning

1. **Sign problem**: v2 sign inconclusive.
2. **Spread problem**: v4 score spread = 113.4 pts. Substantially better than v3 9 pts -- grain change helped.
3. **Stability**: Pooled YoY r2 = 0.08220247098677783. Below acceptable floor (< 0.25).
4. **Split-half reliability**: r = 0.4750128405393574. Below target (< 0.60).

This recommendation is advisory only. No production files have been modified.

---

## Known Deviations

1. Part 0 uses `mean(delta_run_exp)` as ERA/xFIP proxy (real ERA not available in DB; mean_dre is the same target the model trains on, so this is a consistency check rather than fully independent validation).
2. v3 sign comparison uses 2025 season only.
3. Out-of-sample predictive validity uses LOSO OOS predictions, which are themselves indirect (not a true held-out cohort).
