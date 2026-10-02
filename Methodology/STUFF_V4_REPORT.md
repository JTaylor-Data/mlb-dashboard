# Stuff+ v4 — Final Methodology

**Generated**: 2026-10-02 15:02:15  
**Model trained at**: 2026-10-02 14:54:17  
**Model SHA-256**: `0addb8e96ea438ec196b3addc19fd350f1d76e241a9291d3719773f2b9766c28`  
**Spec-lock commit**: `1ea4116ae4544ab359d982bdbbe2d40db6777236`  

---

## 1. Training Setup

- **Training seasons**: 2016–2026 excluding 2020 (7,023,077 pitches after filtering)
- **Scoring seasons**: 2016–2026 including 2020 (flagged `season_flag = '60-game season'`)
- **Excluded pitch types**: PO, UN, EP (junk/non-competitive)
- **KN (knuckleball)**: included
- **Postseason**: excluded (game_type = 'R' filter)
- **Objective**: Huber loss (predict `delta_run_exp`)
- **Fixed tree count**: 2000 (reused from v3 `n_estimators`, no early stopping)

### Hyperparameters (reused from v3)

| Parameter | Value |
|---|---|
| `objective` | `huber` |
| `learning_rate` | `0.03` |
| `num_leaves` | `63` |
| `min_child_samples` | `200` |
| `subsample` | `0.8` |
| `subsample_freq` | `1` |
| `colsample_bytree` | `0.8` |
| `reg_lambda` | `1.0` |
| `random_state` | `42` |

---

## 2. Final Feature Set

Candidate changes applied: `drop_season=False`, `drop_effective_speed=False`, `include_haa=True`

### Features used in final model

| # | Feature | Type |
|---|---|---|
| 1 | `release_speed` | numeric |
| 2 | `effective_speed` | numeric |
| 3 | `release_spin_rate` | numeric |
| 4 | `spin_axis` | numeric |
| 5 | `hb` | numeric |
| 6 | `ivb` | numeric |
| 7 | `vaa_residual` | numeric |
| 8 | `velo_sep_vs_primary_fb` | numeric |
| 9 | `hb_sep_vs_primary_fb` | numeric |
| 10 | `ivb_sep_vs_primary_fb` | numeric |
| 11 | `axis_differential` | numeric |
| 12 | `haa_residual` | numeric |
| 13 | `pitch_type` | categorical |
| 14 | `p_throws` | categorical |
| 15 | `stand` | categorical |
| 16 | `season` | categorical |

### Excluded features (vs v3 baseline)

- `release_extension`, `release_pos_x_as`, `release_pos_z`, `arm_angle` — excluded in v3 (Deception+ only); remain excluded in v4
- `season`: kept
- `effective_speed`: kept
- `haa_residual` (new): added (Stage 2 leakage check passed)

### Forbidden features (leakage guard)

The following columns are explicitly verified absent from the feature list before every training run:
`plate_x`, `plate_z`, `balls`, `strikes`, `release_extension`, `release_pos_x_as`,
`release_pos_z`, `arm_angle`, `pitcher_id`, `game_id`

---

## 3. HAA Residual (if adopted)

**Definition**: Residualized horizontal approach angle — actual HAA minus expected HAA from an OLS regression.

**HAA computation**: velocity vector angle at front of plate from `vx0/vy0/ax/ay` physics,
mirrored by `p_throws` so arm-side is positive for both LHP and RHP.

**Expected HAA**: OLS per pitch type on [mirrored `plate_x`, mirrored `plate_x`², mirrored `release_pos_x`, `release_speed`].
Fit on training seasons only; applied to all seasons including 2020.

**Leakage check (Part 1)**:

| Correlation | Value | Threshold | Pass? |
|---|---|---|---|
| `|corr(haa_residual, plate_x)|` | 0.0092 | < 0.1 | ✓ |
| `|corr(haa_residual, release_pos_x)|` | 0.0836 | < 0.1 | ✓ |

---

## 4. Ablation Results (Part 2 — Pitcher-Grouped 5-Fold CV, 30% Subsample)

**Setup**: 30% subsample of pitchers (all their seasons), stratified by `p_throws` and pitch-count tercile, seed=42.
5-fold CV grouped by pitcher (every pitch of a pitcher falls in exactly one fold).
Subsample: 776 pitchers, seed=42.

### Stage 1 — Drop candidates

| Variant | Description | Corr with V0a | Same-season xFIP r | Next-season xFIP r | Decision |
|---|---|---|---|---|---|
| V0a | Baseline (seed=42) | +1.000 | -0.216 | -0.173 | — |
| V0b | Baseline (seed=123; noise floor) | +0.979 | -0.199 | -0.169 | — |
| V1 | minus `season` | +0.934 | -0.221 | -0.173 | fails |
| V2 | minus `effective_speed` | +0.953 | -0.186 | -0.149 | fails |
| V3 | minus both | +0.899 | -0.199 | -0.140 | fails |

**Stage 1 chosen variant**: `V0a` — drop_season=False, drop_effective_speed=False

### Stage 2 — HAA residual

HAA leakage check: True
Decision: adopt_haa=True — leakage_ok=True, corr_with_vchosen=0.9635

---

## 5. Final Fit Details

- **Trained at**: 2026-10-02 14:54:17
- **Seeds**: random_state=42 (LightGBM)
- **Tree count**: 2000 (no early stopping)
- **Training rows**: 7,023,077
- **Model hash (SHA-256)**: `0addb8e96ea438ec196b3addc19fd350f1d76e241a9291d3719773f2b9766c28`

---

## 6. Validation (Part 4 — Final Model, In-Sample)

> **All tables in this section are FINAL MODEL, IN-SAMPLE (descriptive).**
> Out-of-sample evidence is in the ablation pitcher-holdout (Section 4, labeled conservative).

### Per-Season Score Distribution

| Season | N pitcher-seasons | Mean Stuff+ | Std Stuff+ | Note |
|---|---|---|---|---|
| 2016 | 742 | 99.8 | 4.7 |  |
| 2017 | 755 | 99.4 | 5.5 |  |
| 2018 | 798 | 98.7 | 7.6 |  |
| 2019 | 829 | 98.8 | 7.4 |  |
| 2020 | 735 | 99.3 | 6.5 | 60-game season |
| 2021 | 908 | 97.8 | 9.1 |  |
| 2022 | 870 | 96.6 | 13.3 |  |
| 2023 | 859 | 96.9 | 12.4 |  |
| 2024 | 851 | 98.1 | 8.4 |  |
| 2025 | 863 | 97.1 | 11.8 |  |
| 2026 | 860 | 97.1 | 12.8 |  |

### Year-over-Year Stability

| Pair | Pearson r | n |
|---|---|---|
| 2016→2017 | +0.766 | 539 |
| 2017→2018 | +0.761 | 557 |
| 2018→2019 | +0.818 | 588 |
| 2019→2021 | +0.581 | 531 |
| 2021→2022 | +0.831 | 611 |
| 2022→2023 | +0.939 | 602 |
| 2023→2024 | +0.869 | 608 |
| 2024→2025 | +0.884 | 609 |
| 2025→2026 | +0.903 | 606 |

**Pooled mean YoY r**: +0.817
*(Pairs touching 2020 excluded from pooled mean)*

### Mean Score by Pitch Type

| Pitch Type | Mean Stuff+ | Flagged (>3 pts from 100)? |
|---|---|---|
| AB | 105.7 | ⚠ YES |
| CH | 97.2 |  |
| CS | 62.0 | ⚠ YES |
| CU | 95.1 | ⚠ YES |
| FA | 59.9 | ⚠ YES |
| FC | 100.1 |  |
| FF | 97.4 |  |
| FO | 91.0 | ⚠ YES |
| FS | 100.5 |  |
| KN | 63.2 | ⚠ YES |
| SC | 88.1 | ⚠ YES |
| SI | 97.9 |  |
| SL | 101.8 |  |
| ST | 100.9 |  |
| SV | 98.4 |  |

### Split-Half Reliability (2025 season)

| Min Pitches | Spearman-Brown r | n pitchers |
|---|---|---|
| 50 | 0.994 | 777 |
| 100 | 0.994 | 725 |
| 200 | 0.994 | 640 |
| 300 | 0.995 | 573 |
| 400 | 0.996 | 521 |
| 600 | 0.997 | 435 |
| 800 | 0.997 | 346 |
| 1000 | 0.997 | 273 |
| 1500 | 0.998 | 141 |
| 2000 | 0.998 | 103 |

### Leakage Checks

| Check | Result |
|---|---|
| Forbidden features absent | True |
| HAA residual ↔ plate_x | 0.0092 (threshold < 0.1) |
| HAA residual ↔ release_pos_x | 0.0836 (threshold < 0.1) |

---

## 7. Open Issues / Known Limitations

1. **2020 exclusion from training**: The 60-game season is excluded from training but scored. Scores are flagged `season_flag='60-game season'`.
2. **Validation is in-sample**: All correlation figures in Section 6 are computed on the same data the model was trained on. The only out-of-sample evidence is the pitcher-holdout in the ablation (Section 4).
3. **No 2026 holdout**: Per pre-registration, no second retrain is performed with a 2026 holdout.
4. **xFIP availability**: xFIP data starts from 2021 in the DB; next-season xFIP correlations cover 2021-forward only.
5. **Split-half reliability uses 2025 only**: A multi-season estimate would be more stable.
6. **Seed stability**: Not yet run (requires a second full training pass). Expected >= 0.99.

---

## 8. Sanity Tables (Selected Seasons)

### 2016 — Top 15 / Bottom 15

| Rank | pitcher_id | Stuff+ v4 | N pitches |
|---|---|---|---|
| T1 | 547973 | 126.1 | 738 |
| T2 | 445276 | 118.8 | 955 |
| T3 | 476454 | 118.5 | 1,080 |
| T4 | 518886 | 114.5 | 899 |
| T5 | 501789 | 114.3 | 996 |
| T6 | 456713 | 113.1 | 909 |
| T7 | 592102 | 113.0 | 1,115 |
| T8 | 543766 | 112.9 | 1,112 |
| T9 | 542609 | 112.5 | 601 |
| T10 | 502154 | 112.0 | 909 |
| T11 | 477132 | 111.9 | 1,798 |
| T12 | 475054 | 111.6 | 400 |
| T13 | 489265 | 111.3 | 382 |
| T14 | 605218 | 111.3 | 585 |
| T15 | 621121 | 111.2 | 1,232 |
| ... | ... | ... | ... |
| B1 | 285064 | 90.6 | 1,268 |
| B2 | 500871 | 90.3 | 16 |
| B3 | 592593 | 90.2 | 1,142 |
| B4 | 462515 | 90.0 | 222 |
| B5 | 592433 | 89.1 | 774 |
| B6 | 471083 | 88.9 | 31 |
| B7 | 456124 | 88.8 | 36 |
| B8 | 542194 | 88.4 | 43 |
| B9 | 475247 | 87.6 | 19 |
| B10 | 534737 | 87.0 | 1,142 |
| B11 | 600301 | 86.8 | 6 |
| B12 | 592407 | 85.9 | 18 |
| B13 | 446381 | 85.2 | 16 |
| B14 | 534606 | 80.0 | 12 |
| B15 | 460269 | 78.8 | 7 |

### 2020 — Top 15 / Bottom 15

| Rank | pitcher_id | Stuff+ v4 | N pitches |
|---|---|---|---|
| T1 | 547973 | 119.2 | 199 |
| T2 | 676979 | 117.7 | 82 |
| T3 | 663770 | 117.0 | 54 |
| T4 | 607192 | 116.2 | 954 |
| T5 | 594798 | 115.3 | 1,134 |
| T6 | 605447 | 114.8 | 237 |
| T7 | 521230 | 114.7 | 370 |
| T8 | 642207 | 114.6 | 431 |
| T9 | 434378 | 113.7 | 73 |
| T10 | 664126 | 113.3 | 490 |
| T11 | 622251 | 112.1 | 471 |
| T12 | 475054 | 112.1 | 138 |
| T13 | 668687 | 111.8 | 404 |
| T14 | 621052 | 111.7 | 369 |
| T15 | 621111 | 111.5 | 600 |
| ... | ... | ... | ... |
| B1 | 571974 | 77.1 | 8 |
| B2 | 460077 | 76.6 | 30 |
| B3 | 606115 | 76.1 | 9 |
| B4 | 621532 | 73.9 | 16 |
| B5 | 669289 | 73.6 | 40 |
| B6 | 576397 | 73.1 | 11 |
| B7 | 570560 | 72.8 | 2 |
| B8 | 608348 | 72.3 | 12 |
| B9 | 621011 | 71.1 | 8 |
| B10 | 623168 | 70.6 | 2 |
| B11 | 571788 | 70.2 | 28 |
| B12 | 670712 | 70.2 | 4 |
| B13 | 592407 | 68.8 | 13 |
| B14 | 453943 | 63.5 | 13 |
| B15 | 641645 | 62.8 | 9 |

### 2022 — Top 15 / Bottom 15

| Rank | pitcher_id | Stuff+ v4 | N pitches |
|---|---|---|---|
| T1 | 664126 | 124.6 | 351 |
| T2 | 661403 | 123.8 | 920 |
| T3 | 594838 | 121.4 | 5 |
| T4 | 607192 | 118.9 | 114 |
| T5 | 642585 | 117.7 | 993 |
| T6 | 664854 | 117.2 | 954 |
| T7 | 547973 | 114.6 | 678 |
| T8 | 621363 | 113.1 | 973 |
| T9 | 594798 | 112.8 | 940 |
| T10 | 660271 | 112.6 | 2,623 |
| T11 | 605447 | 112.0 | 987 |
| T12 | 521230 | 112.0 | 948 |
| T13 | 605483 | 111.9 | 2,338 |
| T14 | 445276 | 111.9 | 998 |
| T15 | 621242 | 111.9 | 922 |
| ... | ... | ... | ... |
| B1 | 596117 | 42.9 | 49 |
| B2 | 641914 | 42.1 | 9 |
| B3 | 621433 | 41.1 | 58 |
| B4 | 545121 | 39.6 | 2 |
| B5 | 676391 | 38.4 | 42 |
| B6 | 640461 | 38.2 | 26 |
| B7 | 543281 | 37.7 | 57 |
| B8 | 592743 | 37.7 | 25 |
| B9 | 425877 | 34.4 | 39 |
| B10 | 571976 | 32.3 | 38 |
| B11 | 543829 | 31.0 | 28 |
| B12 | 643524 | 30.3 | 26 |
| B13 | 444876 | 27.6 | 15 |
| B14 | 622268 | 25.4 | 8 |
| B15 | 621011 | 14.5 | 26 |

### 2025 — Top 15 / Bottom 15

| Rank | pitcher_id | Stuff+ v4 | N pitches |
|---|---|---|---|
| T1 | 547973 | 119.9 | 917 |
| T2 | 695243 | 117.3 | 971 |
| T3 | 643511 | 116.8 | 976 |
| T4 | 661403 | 115.0 | 746 |
| T5 | 656730 | 114.0 | 752 |
| T6 | 695505 | 113.7 | 764 |
| T7 | 656629 | 112.4 | 218 |
| T8 | 642585 | 112.3 | 626 |
| T9 | 682254 | 111.9 | 858 |
| T10 | 691799 | 111.8 | 610 |
| T11 | 683618 | 111.5 | 258 |
| T12 | 445276 | 111.4 | 868 |
| T13 | 676879 | 111.0 | 1,033 |
| T14 | 661395 | 110.4 | 1,073 |
| T15 | 694819 | 109.8 | 1,162 |
| ... | ... | ... | ... |
| B1 | 683766 | 51.3 | 5 |
| B2 | 681987 | 51.1 | 8 |
| B3 | 670032 | 51.0 | 8 |
| B4 | 669707 | 50.6 | 6 |
| B5 | 694359 | 47.4 | 21 |
| B6 | 642215 | 47.3 | 11 |
| B7 | 650489 | 46.0 | 2 |
| B8 | 624431 | 43.6 | 33 |
| B9 | 622569 | 43.2 | 13 |
| B10 | 596142 | 40.9 | 11 |
| B11 | 691594 | 38.6 | 169 |
| B12 | 543592 | 37.5 | 16 |
| B13 | 608671 | 31.7 | 4 |
| B14 | 645305 | 19.5 | 15 |
| B15 | 672724 | 15.6 | 40 |

### 2026 — Top 15 / Bottom 15

| Rank | pitcher_id | Stuff+ v4 | N pitches |
|---|---|---|---|
| T1 | 695243 | 120.2 | 1,149 |
| T2 | 694819 | 118.4 | 2,646 |
| T3 | 695505 | 114.4 | 2,512 |
| T4 | 547973 | 113.8 | 816 |
| T5 | 594027 | 113.6 | 607 |
| T6 | 811315 | 113.1 | 81 |
| T7 | 683618 | 113.0 | 946 |
| T8 | 445276 | 112.9 | 725 |
| T9 | 656730 | 112.2 | 872 |
| T10 | 696519 | 112.1 | 580 |
| T11 | 671922 | 111.9 | 1,226 |
| T12 | 670970 | 111.8 | 1,312 |
| T13 | 669062 | 111.6 | 953 |
| T14 | 676879 | 111.5 | 1,580 |
| T15 | 679885 | 111.4 | 232 |
| ... | ... | ... | ... |
| B1 | 571771 | 40.7 | 3 |
| B2 | 691594 | 38.8 | 73 |
| B3 | 620443 | 37.2 | 4 |
| B4 | 656716 | 36.9 | 33 |
| B5 | 623168 | 35.7 | 26 |
| B6 | 655316 | 34.4 | 28 |
| B7 | 680474 | 32.2 | 14 |
| B8 | 624431 | 28.6 | 40 |
| B9 | 664774 | 24.0 | 11 |
| B10 | 677588 | 17.7 | 11 |
| B11 | 543309 | 16.5 | 18 |
| B12 | 683734 | 13.9 | 62 |
| B13 | 686475 | 9.3 | 38 |
| B14 | 678391 | 6.2 | 35 |
| B15 | 596117 | -0.8 | 36 |
