# MLB Dashboard — Models Reference

**Last updated: 2026-10-26 | Authoritative alongside `HANDOFF.md`**

---

## Live Models Summary

| Model | Version | Status | Min Pitches |
|---|---|---|---|
| Stuff+ | v4 | ACTIVE | None (EB shrinkage) |
| Location+ | v3 | ACTIVE | 300 |
| Deception+ | v2b | ACTIVE | None (EB shrinkage, K=300) |
| Tunneling+ | v3 | OUT OF SCOPE | 100 |

Active version controlled by `config/model_policy.py`.

---

## Stuff+ v4

**Algorithm:** LightGBM ensemble  
**Trained:** 2026-10-06  
**Training data:** 2016–2026, 7,023,077 pitches  
**Features (16):** release_speed, effective_speed, release_spin_rate, spin_axis, hb, ivb, vaa_residual, velo_sep_vs_primary_fb, hb_sep_vs_primary_fb, ivb_sep_vs_primary_fb, axis_differential, haa_residual + pitch_type, p_throws, stand, season (categorical)  
**Unit:** pitcher-pitch-type (scaled to mean=100, sd=15 within each pitch type)  
**Missing data:** Empirical Bayes shrinkage toward league mean (no hard minimum)  
**Inference:** `analytics/stuff_model.py` → `compute_stuff_plus()`

### Validation
| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| Split-half SB @100p | 0.9938 | ≥0.98 | ✓ |
| YoY pooled r | 0.817 | ≥0.75 | ✓ |
| Concurrent xFIP r | −0.444 | ≤−0.35 | ✓ |
| Next-season xFIP r | −0.225 | ≤−0.30 | ✗ (missed) |

**Known issue:** Next-season xFIP missed threshold (−0.225 vs −0.30). Accepted provisionally — the metric is noisy at short sample sizes, and concurrent validity is strong.

---

## Location+ v3

**Algorithm:** LightGBM ensemble (5 sub-models: one per handedness × pitch category)  
**Trained:** 2026-10-02  
**Training data:** 2015–2026 (excluding 2020), split per sub-model  
**Features:** plate_x, plate_z, count, batter_hand, pitcher_hand, pitch_type, zone  
**Unit:** pitcher-season (scaled to mean=100, sd=15)  
**Min pitches:** 300 (hard cutoff for qualification)  
**Inference:** `analytics/location_model.py` → `_raw_predictions()`

### Validation
| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| YoY S1 pooled r | 0.659 | ≥0.55 | ✓ |
| Split-half SB @300p | 0.8145 | ≥0.75 | ✓ |
| v2→v3 correlation | 0.8856 | ≥0.80 | ✓ |
| Concurrent BB% r | −0.685 | ≤−0.50 | ✓ |
| Concurrent xFIP r | −0.332 | ≤−0.25 | ✓ |

---

## Deception+ v2b

**Algorithm:** LightGBM cross-validated ensemble (3 repeats × 5 folds = 15 models)  
**Created:** 2026-10-04  
**Features (4):** release_pos_x_as, release_pos_z, release_extension, arm_angle  
**Season dropped from v2:** Pre-registered rule — inconclusive season effect → drop  
**Unit:** pitcher-season (scaled to mean=100, sd=15)  
**Fold assignment:** pitcher_id mod 5 (pitcher-level CV, no data leakage)  
**EB shrinkage K:** 300  
**Scored seasons:** 2021–2026  
**Inference:** `analytics/deception_plus_model.py`

**Note:** Deception+ measures arm-slot disguise (release position + extension + arm angle). It does NOT capture delivery timing — that is a separate research direction (video-based, not yet scaled).

---

## Tunneling+ (OUT OF SCOPE)

**Status:** v3 model exists in `analytics/tunneling_v3_model.py` but is excluded from the dashboard pending validation. v2 (5-classifier) is the last fully validated version but its training artifact (`pitcher_season_commit_point.csv`) lives in `Archive/`. Do not wire Tunneling+ into any new routes without explicit sign-off.

The `tunneling_plus` column in `pitcher_model_scores` will be NULL for all seasons until this is resolved.

---

## Scoring Pipeline

Pre-computed scores are stored in `pitcher_model_scores` (SQLite):

```
pitcher_id | season | stuff_plus | tunneling_plus | location_plus | computed_at
```

Populated by: `python pulls/precompute_model_scores.py [season]`  
Re-run after each weekly model retrain for the current season.

The career page reads from this table. Individual pitch pages call the model directly.

---

## Model Policy

See `config/model_policy.py` for:
- `DECEPTION_ACTIVE_VERSION = "v2b"`
- `CURRENT_SEASON = 2026`
- Min-pitch thresholds per model

---

## Known Outstanding Issues

1. **Stuff+ next-season xFIP** — missed threshold (−0.225 vs −0.30). Provisional acceptance. Track in future retrains.
2. **Tunneling+ null** — `pitcher_model_scores.tunneling_plus` is NULL for all seasons. Expected until Tunneling+ is brought back in scope.
3. **Pitch location chart orientation** — may show RHH/LHH reversed vs Savant. Deferred (see `Methodology/decisions/NEEDS_DECISION.md`).

---

## Deferred Decisions

See `Methodology/decisions/NEEDS_DECISION.md` for the full log. Key open items:
- ND-LOC-01: Pitch location chart orientation (deferred, low priority)

---

## Document Hierarchy

1. `HANDOFF.md` — operational context, who owns what, deployment notes
2. **`Methodology/MODELS_REFERENCE.md`** (this file) — authoritative model specs and validation
3. `Methodology/live_models/STUFF_PLUS_V4_FINAL_METHODOLOGY.html` — detailed Stuff+ v4 methodology
4. `Methodology/decisions/NEEDS_DECISION.md` — deferred decision log
5. `Methodology/archive/` — superseded model docs
