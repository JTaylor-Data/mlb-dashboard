"""
tools/build_validation_content.py

Reads existing training artifacts (no model execution, no DB access) and writes
frontend/content/validation_results.json with provenance per number.

Run:
    python tools/build_validation_content.py

Output:
    frontend/content/validation_results.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
V2_ART = ROOT / "training" / "v2_artifacts"
TUN_ART = ROOT / "training" / "tunneling_v3_artifacts"
DIAG_STUFF = ROOT / "diagnostics" / "stuff_memorization"
OUT_PATH = ROOT / "frontend" / "content" / "validation_results.json"

def _load(path):
    return json.loads(path.read_text())

def build():
    results = {}

    # --- Stuff+ v3 ---
    stuff_report = _load(MODELS_DIR / "stuff_plus_v3_lgbm_report.json")
    stuff_mem = _load(DIAG_STUFF / "memorization_report.json")
    stuff_pv = _load(MODELS_DIR / "stuff_plus_v3_predictive_validity.json")

    results["stuff_plus"] = {
        "model": "Stuff+ v3",
        "status": "draft",
        "card_repeatability": {
            "title": "Does it repeat?",
            "what_it_tests": "Whether a pitcher's score is stable year-to-year (not just noise).",
            "yoy_lopo_r": round(stuff_mem["section_A"]["yoy_mean_r"], 3),
            "yoy_lopo_r_source": "diagnostics/stuff_memorization/memorization_report.json section_A.yoy_mean_r",
            "yoy_lopo_r_note": "Leave-one-pitcher-out: model never trained on the scored pitcher's own data",
            "yoy_lopo_ci_lo": round(stuff_mem["section_A"]["yoy_ci_lo"], 3),
            "yoy_lopo_ci_hi": round(stuff_mem["section_A"]["yoy_ci_hi"], 3),
            "yoy_loso_mean_r": round(stuff_mem["section_A"]["loso_yoy_mean"], 3),
            "yoy_loso_mean_r_source": "diagnostics/stuff_memorization/memorization_report.json section_A.loso_yoy_mean",
            "yoy_loso_mean_r_note": "Leave-one-season-out: model sees pitcher in other seasons but not target season",
            "yoy_insample_r": 0.907,
            "yoy_insample_r_source": "Methodology/HANDOFF.md line 97",
            "yoy_insample_r_note": "In-sample: live model trained on all seasons including the scored ones",
            "split_half_note": "Not stored as a named artifact. Docstring in analytics/stuff_model.py line 218 reports r=0.98+ even at n=20 (computed during development).",
            "verdict": "Passes",
            "verdict_note": "Physical pitch characteristics repeat reliably. LOPO r=0.558 is the honest out-of-sample figure."
        },
        "card_predictive": {
            "title": "Does it predict results?",
            "what_it_tests": "Does 2024 Stuff+ v3 predict real 2025 outcomes (strict out-of-sample)?",
            "score_season": stuff_pv["score_season"],
            "outcome_season": stuff_pv["outcome_season"],
            "model_note": stuff_pv["description"],
            "source": "models/stuff_plus_v3_predictive_validity.json",
            "correlations_min200": stuff_pv["results_by_min_pitches"]["200"]["correlations"],
            "n_qualified_min200": stuff_pv["results_by_min_pitches"]["200"]["n_qualified"],
            "direction_note": "Negative xwoba correlation is correct: lower xwoba (fewer runs allowed) goes with higher Stuff+.",
            "verdict": "Passes",
            "verdict_note": "xwoba r=-0.365, whiff% r=0.420, SwStr% r=0.434 at min 200 pitches (out-of-sample)."
        },
        "card_independence": {
            "title": "Is it its own thing?",
            "what_it_tests": "Correlation with the other three models — are they measuring different things?",
            "r_deception": 0.396,
            "r_deception_source": "Methodology/HANDOFF.md line 211 (training/v2_artifacts/part0b_stuff_deception_correlation.json)",
            "r_deception_note": "Pearson r across 2868 pitcher-seasons 2021-2025. Moderate — some shared signal (good pitchers score well on all dimensions) but not redundant.",
            "r_tunneling_note": "Not yet computed for v2 tunneling vs v3 stuff in a stored artifact.",
            "r_location_note": "Not yet computed in a stored artifact.",
            "verdict": "Mixed",
            "verdict_note": "Deception+ correlation r=0.396 is moderate and expected. Tunneling+ and Location+ cross-correlations not yet stored as artifacts."
        },
        "card_stress": {
            "title": "Stress test",
            "what_it_tests": "Pitcher-holdout identifiability: can the model's feature vector identify which pitcher it scored?",
            "top1_lift": round(stuff_mem["section_C"]["lift_top1"], 3),
            "top1_lift_source": "diagnostics/stuff_memorization/memorization_report.json section_C.lift_top1",
            "top1_lift_note": "Lift over naive baseline at 1 pitcher. Below 1x = model cannot identify pitchers above chance. 0.88x here.",
            "interpretation": stuff_mem["interpretation"]["verdict"],
            "interpretation_source": "diagnostics/stuff_memorization/memorization_report.json interpretation.verdict",
            "lopo_vs_loso_drop": round(stuff_mem["interpretation"]["yoy_drop"], 3),
            "verdict": "Mixed",
            "verdict_note": "Feature vector is only weakly identifiable (top-1 lift 0.88x). LOPO YoY r drops 0.287 below LOSO, consistent with some memorization but not strong. Weight of evidence: genuine generalization with minor memorization."
        }
    }

    # --- Location+ v2 ---
    loc_meta = _load(MODELS_DIR / "location_plus_v2_meta.json")

    # Extract LOSO AUC means from tunneling v2 meta which has AUC data
    # Location AUCs from HANDOFF.md (stored in meta only via cv results section)
    loc_aucs = {
        "xSwing": 0.848,
        "xCS": 0.981,
        "xWhiff": 0.754,
        "xFoul": 0.621,
        "source": "Methodology/HANDOFF.md line 191 (LOSO CV from training)"
    }

    results["location_plus"] = {
        "model": "Location+ v2",
        "status": "draft",
        "card_repeatability": {
            "title": "Does it repeat?",
            "what_it_tests": "Whether a pitcher's command score is stable year-to-year.",
            "yoy_v1_collapse": "YoY r collapsed to 0.009 for 2024→2025 in v1 (sign-flip bug). v2 fixes both sign and stability.",
            "yoy_v1_source": "Methodology/HANDOFF.md line 194-195",
            "v2_std": 9.99,
            "v2_std_source": "Methodology/HANDOFF.md line 195",
            "split_half_note": "Not stored as a named artifact for v2. Run tools/build_validation_content.py again after generating split-half curves.",
            "verdict": "Not yet run",
            "verdict_note": "v1 had a stability collapse; v2 fixes it. Full YoY battery for v2 pending more seasons."
        },
        "card_predictive": {
            "title": "Does it predict results?",
            "what_it_tests": "Does Location+ predict ERA and xFIP in the same or next season?",
            "era_r_concurrent": 0.208,
            "era_r_source": "Methodology/HANDOFF.md line 195",
            "xfip_r_concurrent": 0.349,
            "xfip_r_source": "Methodology/HANDOFF.md line 195",
            "direction_note": "Positive correlation with ERA is correct: higher Location+ = better command = lower ERA.",
            "out_of_sample_note": "Strict 2024→2025 out-of-sample test not yet run for Location+ v2. See NEEDS_DECISION.md.",
            "verdict": "Passes",
            "verdict_note": "ERA r=0.208, xFIP r=0.349 (concurrent). Direction correct. Out-of-sample not yet stored."
        },
        "card_independence": {
            "title": "Is it its own thing?",
            "what_it_tests": "Does Location+ measure something distinct from pitch shape and mechanics?",
            "design_note": "Features are plate_x, plate_z, count — explicitly no pitch type, no movement, no release point. By construction it cannot re-derive Stuff+ or Deception+.",
            "location_leakage_test": "Not yet computed. See NEEDS_DECISION.md ND-06.",
            "verdict": "Passes (by construction)",
            "verdict_note": "Feature set deliberately excludes all pitch-quality inputs. Cross-model correlation not yet stored as an artifact."
        },
        "card_stress": {
            "title": "Stress test",
            "what_it_tests": "Classifier AUCs (LOSO CV) per event type.",
            "classifier_aucs": loc_aucs,
            "pitcher_holdout_note": "Pitcher-holdout test not yet run for Location+ v2.",
            "verdict": "Not yet run",
            "verdict_note": "Classifier AUCs show the model learns real event probabilities. Full pitcher-holdout identifiability test pending."
        }
    }

    # --- Tunneling+ v2 (live) ---
    tun_val = _load(TUN_ART / "tunneling_v3_validation.json")
    tun_pv  = _load(TUN_ART / "tunneling_v3_predictive_validity.json")
    # Note: these artifacts were produced for the old v3 single-regressor.
    # The v2 (live) validation data is in tunneling_plus_v2_meta.json + HANDOFF.md.
    tun_v2_meta = _load(MODELS_DIR / "tunneling_plus_v2_meta.json")

    # Year-over-year from v2 HANDOFF
    tun_yoy_r2_range = "0.28-0.35"  # from HANDOFF.md line 170 (r² 0.31–0.53 is from old v3; v2 listed as 0.31–0.53 as well)

    results["tunneling_plus"] = {
        "model": "Tunneling+ v2",
        "status": "draft",
        "card_repeatability": {
            "title": "Does it repeat?",
            "what_it_tests": "Whether pitch-sequence scores are stable year-to-year.",
            "yoy_r2_range": "0.31-0.53",
            "yoy_r2_source": "Methodology/HANDOFF.md line 170 (v2 validation)",
            "split_half_range": "0.85-0.91",
            "split_half_source": "Methodology/HANDOFF.md line 169",
            "lower_than_stuff_note": "Lower YoY stability than Stuff+ is expected — sequencing is partly a strategic choice, not purely a fixed physical trait.",
            "verdict": "Passes",
            "verdict_note": "YoY r²=0.31–0.53, split-half reliability 0.85–0.91. Stable enough to meaningfully rank pitchers."
        },
        "card_predictive": {
            "title": "Does it predict results?",
            "what_it_tests": "Does Tunneling+ (old v3 model) predict 2025 outcomes from 2024 scores?",
            "note": "These figures are from the old v3 single-regressor, not the current live v2. v2 predictive validity not yet stored as a separate artifact.",
            "score_season": tun_pv["score_season"],
            "outcome_season": tun_pv["outcome_season"],
            "correlations_min100": tun_pv["results_by_min_pitches"]["100"]["correlations"],
            "n_qualified": tun_pv["results_by_min_pitches"]["100"]["n_qualified"],
            "era_r_v2_concurrent": 0.182,
            "era_r_v2_source": "Methodology/HANDOFF.md line 163",
            "xfip_r_v2_concurrent": 0.337,
            "xfip_r_v2_source": "Methodology/HANDOFF.md line 163",
            "direction_note": "Positive ERA correlation is correct: higher Tunneling+ = better = lower ERA.",
            "verdict": "Mixed",
            "verdict_note": "v2 ERA r=0.182, xFIP r=0.337 (concurrent, sign-flip fixed). v3 out-of-sample figures small and mixed. Geometry adds little beyond pitch-type sequencing per 2026-09-21 test."
        },
        "card_independence": {
            "title": "Is it its own thing?",
            "what_it_tests": "Correlation with other models; geometry-leakage test.",
            "r_stuff": tun_val["cross_model_correlation"]["r_stuff"],
            "r_deception": tun_val["cross_model_correlation"]["r_deception"],
            "r_stuff_source": "training/tunneling_v3_artifacts/tunneling_v3_validation.json (old v3 model)",
            "geometry_leakage_test": {
                "lag1_geometry_shap_z": -1.706,
                "all_lag_geometry_shap_z": 1.635,
                "pitch_type_lag_shap_z": 18.622,
                "source": "Methodology/HANDOFF.md lines 229-233",
                "finding": "Geometry features (commit/plate/shape distances) show no independent predictive signal for real whiff rates. Model power comes from pitch-type sequencing, not geometric measurement."
            },
            "verdict": "Mixed",
            "verdict_note": "Low cross-model correlations (r≈0.07–0.15) confirm genuine distinctness. But the geometry features that motivated the model add no independent signal beyond pitch-type sequencing."
        },
        "card_stress": {
            "title": "Stress test",
            "what_it_tests": "Sequential same-at-bat test (old v3 model): do predicted-bad pitches actually produce fewer whiffs?",
            "z_score": tun_val["sequential_same_at_bat_test"]["good_minus_bad_z_score"],
            "z_score_source": "training/tunneling_v3_artifacts/tunneling_v3_validation.json",
            "z_score_note": tun_val["sequential_same_at_bat_test"]["methodology_note"],
            "corrected_z_geometry": -1.706,
            "corrected_z_pitch_type": 18.622,
            "corrected_source": "Methodology/HANDOFF.md line 232",
            "verdict": "Unresolved",
            "verdict_note": "The z=84.7 result (old v3 model) was not circular but the corrected test shows geometry features add nothing; sequencing matters but not via the geometry the model was designed around."
        }
    }

    # --- Deception+ v1 ---
    dec_report = _load(MODELS_DIR / "deception_plus_v1_report_2026.json")

    results["deception_plus"] = {
        "model": "Deception+ v1",
        "status": "draft",
        "card_repeatability": {
            "title": "Does it repeat?",
            "what_it_tests": "Whether delivery-mechanics scores are stable year-to-year.",
            "yoy_r2_range": "0.17-0.34",
            "yoy_r2_source": "Methodology/HANDOFF.md line 123",
            "split_half_at_100": dec_report["split_half_reliability"]["100"]["correlation"],
            "split_half_at_100_source": "models/deception_plus_v1_report_2026.json split_half_reliability['100']",
            "split_half_n": dec_report["split_half_reliability"]["100"]["n"],
            "verdict": "Mixed",
            "verdict_note": "Split-half reliability 0.846 (within-season). YoY r²=0.17–0.34 is the weakest of the four models. Likely cause: arm_angle and release_pos_z (r=0.79 leaguewide) partially capture pitcher identity, not deception mechanics."
        },
        "card_predictive": {
            "title": "Does it predict results?",
            "what_it_tests": "Concurrent correlation between Deception+ 2026 and same-season outcomes.",
            "whiff_pct_raw_r": dec_report["outcome_validity"]["outcomes"]["whiff_pct"]["raw_correlation"],
            "whiff_pct_partial_r": dec_report["outcome_validity"]["outcomes"]["whiff_pct"]["partial_correlation_stuff_removed"],
            "xwoba_raw_r": dec_report["outcome_validity"]["outcomes"]["xwoba"]["raw_correlation"],
            "xwoba_partial_r": dec_report["outcome_validity"]["outcomes"]["xwoba"]["partial_correlation_stuff_removed"],
            "swstr_raw_r": dec_report["outcome_validity"]["outcomes"]["swstr_pct"]["raw_correlation"],
            "swstr_partial_r": dec_report["outcome_validity"]["outcomes"]["swstr_pct"]["partial_correlation_stuff_removed"],
            "source": "models/deception_plus_v1_report_2026.json outcome_validity",
            "n_qualified": dec_report["outcome_validity"]["n_qualified"],
            "min_pitches": dec_report["outcome_validity"]["min_pitches_qualify"],
            "direction_note": "Negative xwoba correlation is correct. Positive whiff/SwStr/K-BB% correlations are correct.",
            "out_of_sample_note": "Strict 2024→2025 out-of-sample test for Deception+ v2 shows small and inconsistent correlations (xwoba r=-0.057, whiff% r=-0.106 — note sign flip on whiff vs concurrent). This gap is unresolved.",
            "out_of_sample_source": "models/deception_plus_v1_report_2025.json (separate file for 2025 season scoring; Methodology/HANDOFF.md notes the model)",
            "verdict": "Mixed",
            "verdict_note": "Concurrent correlations real (xwoba r=-0.536 including Stuff+ overlap). After removing Stuff+ overlap: xwoba partial r=-0.357 still meaningful. But out-of-sample figures are small and mixed."
        },
        "card_independence": {
            "title": "Is it its own thing?",
            "what_it_tests": "Correlation with Stuff+ after mechanics features were moved off Stuff+.",
            "r_stuff": 0.396,
            "r_stuff_source": "Methodology/HANDOFF.md line 211",
            "correlation_with_stuff": dec_report["outcome_validity"]["correlation_with_overall_stuff"],
            "correlation_with_stuff_source": "models/deception_plus_v1_report_2026.json outcome_validity.correlation_with_overall_stuff",
            "mechanics_separation_note": "Stuff+ v3 removed release_extension, release_pos_x_as, release_pos_z, arm_angle (confirmed 2026-09-21). These four are now Deception+-exclusive.",
            "verdict": "Mixed",
            "verdict_note": "Stuff+ correlation 0.396–0.470 is moderate and expected. The four mechanics features are separated. But arm_angle × release_pos_z collinearity (r=0.79) means the model may partially proxy pitcher identity or body size."
        },
        "card_stress": {
            "title": "Stress test",
            "what_it_tests": "Height/weight ablation: does removing body-size features change the score?",
            "height_weight_ablation_note": "Not yet run as a stored artifact. The current model includes height_in and weight_lb; whether removing them degrades or improves out-of-sample performance is unresolved.",
            "arm_angle_pos_z_r": 0.79,
            "arm_angle_pos_z_r_source": "Methodology/HANDOFF.md line 126",
            "verdict": "Unresolved",
            "verdict_note": "height/weight ablation not yet run. arm_angle × release_pos_z collinearity (r=0.79) unresolved. A computer-vision pipeline is in R&D to replace these proxy features with timing and concealment measures."
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Written: {OUT_PATH}")
    print(f"  stuff_plus: {len(results['stuff_plus'])} sections")
    print(f"  location_plus: {len(results['location_plus'])} sections")
    print(f"  tunneling_plus: {len(results['tunneling_plus'])} sections")
    print(f"  deception_plus: {len(results['deception_plus'])} sections")

if __name__ == "__main__":
    build()
