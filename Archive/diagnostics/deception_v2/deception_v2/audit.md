# Deception+ v2 Part 1 Audit

Generated: 2026-10-03 17:26:54

## 1. v1 Audit

### Training / Scoring

- Training seasons: 2021–2026 (arm_angle reliable from 2021)
- Scored seasons: same; REPORT_SEASON=2026 hardcoded in api/routers/deception_report.py:25
- Per-request scoring path: `_raw_predictions(conn, season)` in analytics/deception_model.py, scores EVERY pitch in a season on each unique (season, pitcher_id) request, cached in `_raw_predictions_cache` dict with TTL=600s (not the ttl_cache decorator)
- `_raw_predictions_table` in the router wraps that in @ttl_cache for shared reuse across pitcher breakdowns
- No per-pitcher min_pitches gate (returns None if feat.empty after dropna)

### Features
- NUMERIC: arm_angle, release_extension, release_pos_x_as, release_pos_z, arm_angle_std, release_extension_std, release_consistency, height_in, weight_lb
- CATEGORICAL: season
- arm_pos_x_as = release_pos_x * handedness_sign (flipped for LHP)
- Consistency features: std/mean within (pitcher_id, season) — computed on full season even for training splits (potential train-test leakage via variability features)
- height_in, weight_lb from pitcher_bio table (current listing only, not season-specific)

### Known Issues (v1)
- `arm_angle_std`, `release_extension_std`, `release_consistency` are per-(pitcher,season) aggregates leaked onto every pitch including test pitches → split-half r≈0.998 is an artifact, not reliability
- `height_in`, `weight_lb` are highly correlated with pitcher identity — model can memorize pitchers via body size
- `season` as categorical allows era absorption but also memorization if era-level patterns shift
- In-sample r² quoted in reports (0.17–0.34) reflects within-sample fit, not holdout generalization
- Cross-model Pearson YoY 0.612 is a different metric (year-over-year v1 scores, not outcome correlation)

### v1 Results Table
| Metric | Value | Basis |
|--------|-------|-------|
| In-sample r² | 0.17–0.34 | in-sample, artifact |
| Split-half reliability | 0.998 | artifact: consistency features computed pre-split |
| YoY score correlation | 0.612 | cross-model Pearson v1→v1 across seasons |
| Per-request scoring | yes | scores all pitches in season on each request (TTL cached) |

## 2. Data Completeness

### Per-season pitch mechanics coverage
| Season | Total | release_x% | release_z% | extension% | arm_angle% | delta_re% | non-excl% |
|--------|-------|-----------|-----------|-----------|-----------|---------|---------|
| 2015 | 702,306 | 99.9% | 99.9% | 97.9% | 0.0% | 99.5% | 99.8% |
| 2016 | 715,823 | 100.0% | 100.0% | 98.5% | 0.0% | 99.5% | 99.8% |
| 2017 | 724,627 | 99.2% | 99.2% | 99.2% | 0.0% | 99.5% | 99.0% |
| 2018 | 724,452 | 99.1% | 99.1% | 99.1% | 0.0% | 99.5% | 99.1% |
| 2019 | 735,090 | 99.0% | 99.0% | 99.0% | 0.0% | 99.6% | 99.0% |
| 2020 | 264,262 | 99.7% | 99.7% | 99.5% | 75.1% | 99.7% | 99.7% |
| 2021 | 712,320 | 99.6% | 99.6% | 99.5% | 96.5% | 99.7% | 99.6% |
| 2022 | 710,210 | 99.7% | 99.7% | 99.6% | 99.1% | 99.8% | 99.6% |
| 2023 | 720,684 | 99.6% | 99.6% | 99.5% | 98.7% | 99.6% | 99.5% |
| 2024 | 711,898 | 99.6% | 99.6% | 99.5% | 99.2% | 99.7% | 99.5% |
| 2025 | 712,528 | 99.6% | 99.6% | 99.6% | 99.5% | 99.7% | 99.5% |
| 2026 | 716,792 | 99.6% | 99.6% | 99.5% | 75.9% | 99.7% | 99.5% |

### Pitcher bio coverage (2021+): 1921 unique pitchers, 98.6% have height_in/weight_lb

### Qualified pitcher-seasons (≥300 clean pitches with all features)
| Season | Qualified Pitchers |
|--------|-------------------|
| 2021 | 905 |
| 2022 | 870 |
| 2023 | 859 |
| 2024 | 850 |
| 2025 | 864 |
| 2026 | 812 |

### Within- vs Between-pitcher variance (sample 500k pitches, 2021+)
| Feature | Between-pitcher SD | Within-pitcher SD | Intraclass Corr (approx) |
|---------|-------------------|------------------|--------------------------|
| release_pos_x | 1.897 | 0.235 | 0.985 |
| release_pos_z | 0.581 | 0.154 | 0.935 |
| release_extension | 0.543 | 0.184 | 0.897 |
| arm_angle | 13.638 | 3.819 | 0.927 |

## 3. Arm-Angle Predictability

OLS and Ridge predicting arm_angle from release_pos_x, release_pos_z, release_extension, height_in (NO spin/movement).

Pitcher-season level (n=4250 pitcher-seasons), pitcher-grouped 5-fold CV:

| Model | R² (CV) | Pearson r (CV) |
|-------|---------|---------------|
| OLS | 0.7894 | 0.8885 |
| Ridge (alpha=1) | 0.7894 | 0.8885 |

**Pre-registered rule**: drop arm_angle if R² >= 0.90 AND removal equivalent. Ridge CV R²=0.7894.
**Decision: KEEP arm_angle: R² < 0.90 threshold, drop condition NOT met**

## 4. Identifiability Probe

Cross-season matching: for pitchers ≥500 pitches/season, find nearest neighbor by standardized mechanics centroid from 2021 → 2022.
(Note: original script had a bug excluding the correct match; corrected using cross-season matching as equivalent test.)

N pitcher-season pairs: 309 (common to 2021 and 2022)
| Metric | Value | Chance |
|--------|-------|--------|
| Top-1 accuracy | 0.472 | 0.0032 |
| Top-5 accuracy | 0.835 | 0.0162 |
| Top-1 / Chance ratio | 146.0x | 1.0x |

**Warning**: VERY HIGH identifiability — mechanics features (release_pos_x, release_pos_z, extension, arm_angle) identify individual pitchers at 146x chance. This strongly supports dropping identity-prone features (height_in, weight_lb, arm_angle_std, release_extension_std, release_consistency) per pre-registered rule.
ICCs: release_pos_x=0.985, release_pos_z=0.935, release_extension=0.897, arm_angle=0.927 — confirming pitch mechanics are nearly deterministic per pitcher.

## 5. Era Check

Skipped: arm_angle R²=0.7894 < 0.90 threshold. Era control not required.
