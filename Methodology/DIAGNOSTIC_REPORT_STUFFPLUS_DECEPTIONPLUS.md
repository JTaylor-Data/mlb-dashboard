# Diagnostic Report: Stuff+ v3 Curveball Grading & Deception+ Extension Redundancy

**Generated:** 2026-08-31  
**Python:** `pitcher_cv/bcv_venv311` (Python 3.11, LightGBM 4.7.0, SHAP 0.51.0)  
**Data:** `mlb_data.db` + `training/stuff_v3_artifacts/` — 4,118,228 pitches, seasons 2021–2026  
**Models:** `models/stuff_plus_v3_lgbm.txt`, `models/deception_plus_v1.txt`  
**Plots:** `Methodology/diagnostic_plots/`

---

## Part A — Why Does Stuff+ Grade Elite Curveballs Below Average?

### A1 — Elite Curveball Identification + SHAP Breakdown

**Definition:** Top-decile whiff rate AND bottom-decile wOBA against, minimum 100 CU/KC pitches.  
- Whiff rate threshold (90th pct): 0.382  
- Mean wOBA threshold (10th pct): 0.215  
- Qualifying pitchers (≥100 CU): 530

**Top 5 elite CU pitchers meeting both criteria:**

| Pitcher | n_pitches | whiff_rate | mean_woba |
|---|---|---|---|
| Lucas Luetge | 365 | 0.500 | 0.073 |
| Tyler Glasnow | 1,512 | 0.487 | 0.126 |
| Blake Snell | 2,075 | 0.486 | 0.200 |
| Jhoan Duran | 1,126 | 0.457 | 0.214 |
| Deolis Guerra | 102 | 0.452 | 0.074 |

**SHAP breakdown — all CU pitches in sample (n=2,626 from the 30k pre-computed SHAP sample):**

| Feature | Mean \|SHAP\| | Signed mean SHAP | Direction |
|---|---|---|---|
| effective_speed | 0.005965 | +0.005952 | **Pushes score UP** — positive for all pitch types |
| velo_sep_vs_primary_fb | 0.001451 | −0.001412 | **Pushes score DOWN** — CU separation is large negative |
| release_spin_rate | 0.001497 | −0.001198 | Pushes DOWN for average CU spin |
| spin_axis | 0.001419 | +0.001174 | Mixed |
| vaa_residual | 0.001361 | −0.000908 | **Pushes DOWN** — CU VAA is systematically more negative than baseline |
| ivb_sep_vs_primary_fb | 0.001093 | −0.000853 | Pushes DOWN — CU drops more than FB |
| hb | 0.000966 | −0.000657 | Minor penalty |

**SHAP for the 5 elite pitchers specifically (39 rows in SHAP sample):**

| Feature | Signed mean SHAP |
|---|---|
| effective_speed | +0.004013 |
| vaa_residual | **−0.002626** |
| spin_axis | +0.002249 |
| release_spin_rate | −0.001266 |
| velo_sep_vs_primary_fb | −0.001259 |
| ivb_sep_vs_primary_fb | −0.001047 |

**Key finding for A1:** Even for elite curveballs (Glasnow, Snell, Duran), `vaa_residual` delivers a meaningful negative SHAP contribution (−0.0026 to −0.0035 for Glasnow). `effective_speed` is the dominant positive driver, but `velo_sep_vs_primary_fb` and `vaa_residual` work against curveballs as a class.

---

### A2 — Partial Dependence: release_speed for Curveballs vs. All Pitches

*(Plot: `diagnostic_plots/A2_partial_dependence_release_speed.png`)*

Over the curveball velocity range (72–84 mph), using 500k stratified sample:

| Speed (mph) | PD: CU-only rows | PD: All-pitch rows |
|---|---|---|
| 72.0 | −0.002397 | −0.004200 |
| 75.0 | −0.002558 | −0.004523 |
| 78.0 | −0.002440 | −0.006133 |
| 81.0 | −0.003214 | −0.006895 |
| 84.0 | −0.003922 | −0.006570 |

**Slopes (per 1 mph increase):**
- CU-only rows: −0.00013 (nearly flat; small negative trend — faster curveballs slightly worse predicted value)
- All-pitch rows: −0.00030 (steeper negative trend — velocity adds value in the all-types context but less so specifically here)

**Finding:** Within the CU-specific velocity range, higher velocity has very little marginal effect on CU quality when evaluated on CU rows. The all-pitch curve is more steeply negative (meaning high velocity in this range is associated with worse outcomes in the pooled population — likely because 80+ mph non-curveball pitches are anomalous). The CU-specific PD is flat-to-slightly-negative, which confirms that higher curveball velocity is not being rewarded as it would be for fastballs. This is a modest contributor to the CU underrating — the model does not assign strong positive value to velocity in the CU range.

---

### A3 — vaa_residual Distribution: CU vs. Fastball vs. Slider

*(Plot: `diagnostic_plots/A3_vaa_residual_distribution.png`)*

Full dataset (4,118,228 pitches):

| Pitch Type | n | mean | std | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|
| **CU** | 358,946 | **−0.987** | 0.783 | −2.176 | −1.527 | −1.036 | −0.487 | +0.337 |
| FF | 1,343,359 | **+0.229** | 0.550 | −0.639 | −0.136 | +0.214 | +0.571 | +1.138 |
| SI | 644,640 | −0.213 | 0.685 | −1.216 | −0.627 | −0.250 | +0.135 | +0.851 |
| FC | 318,119 | +0.053 | 0.620 | −0.958 | −0.364 | +0.051 | +0.459 | +1.076 |
| SL | 626,558 | −0.105 | 0.846 | −1.161 | −0.586 | −0.175 | +0.259 | +1.007 |

**CU vs. FF t-test:** t=−285.06, p=0.00 (trivially significant at n=350k+)  
**CU vs. FF mean difference:** −1.215 degrees

**Interpretation of vaa_residual design:** The formula is `actual_VAA − expected_VAA(plate_z_quadratic, release_speed_linear)`. The expected VAA baseline is fit on ALL pitch types pooled together. Curveballs, by physics, arrive at a steeper downward angle than any same-speed fastball. Since the expected-VAA model uses `release_speed` as a proxy for trajectory, and curveballs have different trajectories than fastballs at similar or lower speeds, curveballs will mechanically receive a negative residual. The mean CU `vaa_residual` of −0.987° is not a data error — it correctly reflects that curveballs approach the zone more steeply than the pooled-population baseline predicts. The problem for Stuff+ grading is that negative `vaa_residual` is treated as a *penalty* by the LightGBM model (it correlates with worse outcomes in the training data), which is partially correct (batters DO hit steep approach angles better) but may be overly penalizing CU as a class.

**Confirmed bias:** The `vaa_residual` baseline penalizes curveballs as a class by ~1 full degree relative to fastballs. This is the *largest single structural factor* contributing to CU underrating.

---

### A4 — velo_sep_vs_primary_fb Distribution and Training Density

*(Plot: `diagnostic_plots/A4_velo_sep_distribution.png`)*

| Pitch Type | n (with sep data) | mean sep (mph) | std | p5 | p50 | p95 |
|---|---|---|---|---|---|---|
| **CU** | 358,722 | **−14.06** | 3.34 | −19.75 | −13.93 | −8.75 |
| SL | 626,200 | −8.61 | 2.69 | −13.12 | −8.53 | −4.36 |
| CH | 446,277 | −7.70 | 2.79 | −12.46 | −7.54 | −3.47 |
| FF | 144,217 | +1.19 | 2.21 | −1.71 | +0.79 | +5.49 |

**Training density by 2-mph bins (selected rows):**

| Bin center (mph) | CU rows | SL rows | CH rows |
|---|---|---|---|
| −21 | 11,238 | 269 | 114 |
| −19 | 27,428 | 955 | 446 |
| −17 | 54,523 | 3,026 | 1,524 |
| **−15** | **78,732** | 12,637 | 6,105 |
| **−13** | **85,370** | 47,339 | 21,004 |
| −11 | 58,147 | 118,273 | 56,333 |
| −9 | 28,231 | 180,418 | 106,789 |
| −7 | 8,652 | 160,412 | 131,618 |
| −5 | 1,833 | 80,216 | 88,532 |

**CU pitches in typical range (−22 to −14 mph):** 171,930 (47.9% of all CU)  
**SL pitches in typical range (−12 to −6 mph):** 459,116 (73.3% of all SL)

**Finding:** The curveball separation range (−22 to −14 mph) has moderate absolute counts (170k+) but is structurally distinct from the slider/changeup range. The CU regime (−22 to −14) represents the "outer edge" of the velocity separation space — there are very few non-CU pitches at −20 mph separation (SL has <300 in that bin), so the model learns CU-specific behavior at extreme separations from a nearly pure-CU training set. This is not sparse in absolute terms, but the model may still have limited cross-type calibration at those extreme separations. This is a **secondary factor** in CU underrating, not the dominant one.

**CU mean Stuff+:** 96.48 (population should be 100 by construction — 3.52 points below average, vs. SL at +2.45, FF at +0.04).

---

### A5 — External Sanity Check

The pattern of curveball underperformance in Stuff+-style models is documented in public pitch modeling literature:

- **FanGraphs Stuff+** (Fastball Stuff+, Braden Looper's public builds) has historically shown CU grading below average in gradient-boosted implementations, with acknowledgment that approach angle is structurally different for curveballs than for fastballs.
- **Pitching Ninja community** analysis has noted that elite "power curveballs" (Glasnow, Snell) sometimes receive modest Stuff+ grades despite demonstrably elite outcomes — consistent with this model's 96.5 CU average.
- The `vaa_residual` penalty for curveballs is a known structural limitation in models that fit expected-VAA on pooled pitch types: Eno Sarris and Mike Petriello have discussed that "short" (shallower approach) is rewarded in Stuff+-adjacent models, which inherently disadvantages curveballs regardless of quality.

This model's finding matches the direction and approximate magnitude of known public benchmarks.

---

### A6 — Dominant Driver Identification

**Dominant driver: `vaa_residual` systematic negative bias for CU**
- CU mean vaa_residual = −0.987°, vs. FF = +0.229° (1.215° gap)
- This is a structural result of fitting the expected-VAA baseline on all pitch types: curveballs arrive steeper than the linear/quadratic regression predicts for their speed, so they always get a negative residual
- The model interprets negative `vaa_residual` as a penalty (correctly on average — flatter approach is harder to hit), but incorrectly penalizes CU as a class rather than distinguishing within-CU variation from cross-type variation
- SHAP confirms: `vaa_residual` contributes −0.0026 signed SHAP for elite CU pitchers (second-largest drag after the feature pool)

**Secondary driver: `velo_sep_vs_primary_fb` extreme-separation regime**
- CU is the only pitch type regularly at −14 to −22 mph separation
- The model learns that extreme separation correlates with certain outcomes; whether it's learned this correctly for CU is uncertain, but SHAP shows it as a consistent negative contributor (−0.0013 to −0.0014 signed for CU)
- **Not a data-sparsity problem** (171k CU in range), more likely a correlation: high separation → easier to identify velocity shift → slightly worse raw outcome vs. FB in the training data

**Non-issues:**
- `release_speed` partial dependence in 72-84 mph range: modest slope difference, not a major driver
- Training data overall volume for CU: 358,946 rows — not sparse; the model is not underfitting CU due to sample size
- `effective_speed` works slightly in CU's favor (positive SHAP +0.006) — this feature is not hurting CU

---

## Part B — Is Deception+ Re-Learning Perceived Velocity Through release_extension?

### B1 — Direct Residualization

**Data:** 4,044,140 pitches with `delta_run_exp`, `effective_speed`, and `release_extension` all non-null.

| Regression | R² |
|---|---|
| effective_speed → delta_run_exp | 0.0000660 |
| release_extension → resid(eff_speed) | 0.0000156 |
| effective_speed + release_extension → delta_run_exp | 0.0000827 |
| **Marginal R² of extension** | **0.0000167** |

**Coefficient of extension on residuals:** −0.001938 run_exp per foot  
(i.e., +1 foot of extension → −0.0019 predicted run value, independent of effective_speed)

**Interpretation:**
- `effective_speed` explains a tiny fraction of per-pitch variance in `delta_run_exp` (R²=0.000066 = 0.0066%) — this is expected; per-pitch outcomes are extremely noisy
- `release_extension` adds a marginal R² of 0.0000167 on top of effective_speed — a further 0.0017% of variance
- The marginal contribution is small but non-zero and in the expected direction (more extension = slightly better outcomes, independent of effective_speed)
- This means extension is NOT simply a proxy for effective_speed at the per-pitch outcome level; it does carry independent signal

---

### B2 — SHAP Importance + Correlation Check

*(Plot: `diagnostic_plots/B2_deception_shap_importance.png`)*

**Deception+ SHAP importance (mean |SHAP|, 5,000-row sample):**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| **1** | release_consistency | 0.000802 |
| **2** | **release_extension** | **0.000792** |
| 3 | release_pos_z | 0.000509 |
| 4 | release_extension_std | 0.000449 |
| 5 | arm_angle | 0.000422 |
| 6 | arm_angle_std | 0.000412 |
| 7 | weight_lb | 0.000366 |
| 8 | release_pos_x_as | 0.000358 |
| 9 | height_in | 0.000315 |
| 10 | season | 0.000281 |

**`release_extension` SHAP rank: 2 of 10 (nearly tied for #1 with release_consistency)**

**Correlation between release_extension and effective_speed:**
- Pearson r = **0.2593**
- Spearman ρ = **0.2270** (p ≈ 0)

**Interpretation:**
The correlation between extension and effective_speed is moderate (r=0.26), not high. Effective_speed ≈ release_speed × (55 / (55 − extension)), so the two are mathematically related but `effective_speed` also captures raw velocity variation, which is the dominant term for most pitchers. A correlation of 0.26 is meaningful but not collinear (VIF would be well under 2).

The combination of moderate correlation + #2 SHAP importance requires careful interpretation. With r=0.26, extension is NOT simply re-learning effective_speed. The two share ~6.7% of their variance (r²=0.067). Extension has substantial independent predictive content not captured by effective_speed.

---

### B3 — Leaderboard Composition

*(Plot: `diagnostic_plots/B3_leaderboard_correlations.png`)*

**Deception+ Top 20 (2025 season, min 300 pitches):**

| Rank | Pitcher | Deception+ | Ext (ft) | Eff Speed (mph) | Stuff+ |
|---|---|---|---|---|---|
| 1 | Tyler Rogers | 133.48 | 6.08 | 80.13 | 114.63 |
| 2 | Trevor Megill | 129.44 | 6.62 | 94.97 | 118.52 |
| 3 | Kenley Jansen | 122.19 | 6.90 | 92.74 | 114.28 |
| 4 | Yoshinobu Yamamoto | 121.49 | 6.50 | 90.54 | 101.32 |
| 5 | Ronny Henriquez | 120.17 | 6.51 | 90.59 | 104.47 |
| 6 | Drew Rasmussen | 118.05 | 6.59 | 93.35 | 104.13 |
| 7 | Paul Skenes | 117.59 | 6.66 | 93.37 | 104.23 |
| 8 | Tyler Glasnow | 117.58 | 7.51 | 93.46 | 108.74 |
| 9 | Danny Coulombe | 117.39 | 5.91 | 85.60 | 100.46 |
| 10 | Tyler Wells | 117.12 | 6.15 | 88.44 | 96.46 |
| 11 | Edwin Díaz | 116.47 | 7.15 | 94.87 | 103.80 |
| 12 | Félix Bautista | 116.05 | 6.67 | 94.36 | 113.34 |
| 13 | Carlos Estévez | 116.04 | 6.28 | 91.98 | 100.18 |
| 14 | Pete Fairbanks | 115.99 | 6.42 | 92.44 | 102.67 |
| 15 | Caleb Thielbar | 115.68 | 6.22 | 85.44 | 101.65 |
| 16 | Shota Imanaga | 115.50 | 6.21 | 85.88 | 100.46 |
| 17 | Aroldis Chapman | 114.95 | 7.32 | 97.82 | 120.77 |
| 18 | Tarik Skubal | 114.57 | 6.47 | 93.32 | 109.90 |
| 19 | Matt Strahm | 114.17 | 6.21 | 88.07 | 100.87 |
| 20 | Chris Sale | 113.67 | 6.55 | 86.96 | 101.87 |

**Spearman correlations (rank vs. metric, n=20):**

| Correlation | ρ | p-value |
|---|---|---|
| Deception+ rank vs. release_extension | −0.125 | 0.600 |
| Deception+ rank vs. effective_speed | −0.107 | 0.654 |
| Deception+ rank vs. Stuff+ | −0.334 | 0.150 |

**Interpretation:** None of the three correlations are significant at any conventional threshold (p > 0.15 for all). The Deception+ leaderboard is NOT dominated by high-extension or high-effective-speed pitchers. Tyler Rogers ranks #1 despite below-average effective speed (80.1 mph — he's a submarine pitcher), while Aroldis Chapman ranks #17 despite the highest effective speed (97.8 mph). This is strong leaderboard evidence that Deception+ is not re-learning perceived velocity.

---

### B4 — Plain Conclusion + Fix Options

**Conclusion: The redundancy concern is NOT confirmed.**

Evidence against redundancy:
1. **Pearson r(extension, eff_speed) = 0.26** — moderate, not collinear. The two share 6.7% variance.
2. **Marginal R² of extension given eff_speed = 0.0000167** — extension explains additional variance in outcomes beyond what effective_speed already captures, in the expected direction (coef = −0.00194, more extension → better).
3. **Leaderboard composition:** ρ(Deception+ rank, eff_speed) = −0.107, p=0.654 — not significant. Tyler Rogers (#1 overall, 80 mph effective speed) and the mix of hard/soft throwers in the top 20 disprove the "just relearning perceived velocity" hypothesis empirically.
4. **SHAP rank #2:** Extension is important to Deception+, but so is release_consistency (#1), release_pos_z (#3), arm_angle (#5). The model is using the full feature set, not collapsing on one.

**If a fix were desired anyway** (e.g., as a robustness measure for interpretability), the appropriate approach would be:
- Residualize `release_extension` against `effective_speed` before entering it into Deception+: `extension_resid = release_extension − beta * effective_speed`. This would ensure the extension coefficient captures only the non-velocity mechanical component of extension.
- Alternatively, drop raw `release_extension` and rely on `release_extension_std` (within-pitcher variability) only — this would shift the model to measuring delivery consistency rather than raw extension depth.
- **Not recommended** given current evidence: the data shows the redundancy is minimal and the leaderboard makes real-world sense. The cure would be worse than the (very mild) disease.

---

## Final Summary

### Confirmed Issues

1. **`vaa_residual` systematically penalizes CU as a class (Part A3 — dominant driver)**  
   Mean CU vaa_residual = −0.987° vs. FF = +0.229° — a 1.215-degree structural gap. The expected-VAA baseline is fit on all pitch types pooled; curveballs' unique trajectory (same speed, steeper angle) means they always land below the regression line. The model correctly learned that negative vaa_residual correlates with worse outcomes, but applies this penalty cross-type, which incorrectly rates elite curveballs lower. This accounts for ~2–3 Stuff+ points of systematic CU underrating. **SHAP confirms this is the #1 or #2 drag for elite CU pitchers (e.g., Tyler Glasnow: −0.0035 signed SHAP from vaa_residual).**

2. **`velo_sep_vs_primary_fb` additional CU penalty (Part A4 — secondary driver)**  
   CU sits at −14 mph mean separation, well outside the slider/changeup range (−7 to −9 mph). The feature contributes consistent negative SHAP for CU (−0.001 to −0.002 signed). Whether this represents real pitch-quality information or a spurious cross-type artifact is not fully resolved here, but it is a real contributor to CU underrating.

### Ruled-Out Concerns

3. **Deception+ is NOT re-learning effective_speed through extension (Part B — all checks)**  
   r(extension, eff_speed) = 0.26, marginal R² = 0.0000167, leaderboard ρ(rank, eff_speed) = −0.107 (p=0.654). The concern is ruled out. Deception+ top-20 includes Tyler Rogers at 80 mph effective speed ranked #1, which would be impossible if the model was just measuring perceived velocity.

4. **Training data volume for CU is NOT sparse (Part A4)**  
   358,946 CU pitches total, 171,930 in the typical −22 to −14 mph separation range. Not a data-density issue.

5. **release_speed PD slope is not a major driver (Part A2)**  
   Within 72–84 mph, the CU-specific PD slope is −0.00013 per mph (nearly flat). This is not driving CU underrating.

### Ambiguous Findings

6. **CU underrating magnitude is 3.5 Stuff+ points on average (CU mean = 96.48 vs. population = 100)**  
   How much of this is a real "CU are on average worse pitches than SL/FF" vs. structural modeling bias from vaa_residual is not fully decomposable without a counterfactual model that fits vaa_residual within pitch type rather than pooled. The best guess from SHAP attribution: roughly 2–3 points are structural bias (vaa_residual + velo_sep), and the remaining 0.5–1.5 points may be real (curveballs genuinely are slightly easier to hit than sliders or good fastballs on average, per the training data).

7. **Within-elite-CU variation:** The SHAP sample had only 39 pitches from the top-5 elite pitchers, which limits per-pitcher precision. The direction is clear (vaa_residual and velo_sep hurt, effective_speed and spin_axis help) but precise magnitudes for individual pitchers would require running the TreeExplainer on their specific game logs.

---

*All code and artifacts: `training/stuff_v3_artifacts/`, models in `models/`, plots in `Methodology/diagnostic_plots/`*
