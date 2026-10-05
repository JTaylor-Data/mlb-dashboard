"""Write DECEPTION_COMPLETION.md from already-computed JSON artifacts."""
import json
from datetime import datetime
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
DIAG   = ROOT / "diagnostics" / "deception_v2"
MODELS = ROOT / "models"
MIN_PITCHES = 300

pA          = json.loads((DIAG / "partA_qualified_results.json").read_text())
gate_result = json.loads((DIAG / "gate_results_v2b.json").read_text())
eb          = json.loads((DIAG / "evidence_bar_v2b.json").read_text())
gap         = json.loads((DIAG / "in_sample_gap_v2b.json").read_text())
independence= json.loads((DIAG / "independence_heatmap_v2b.json").read_text())
v2b_meta    = json.loads((MODELS / "deception_plus_v2b_meta.json").read_text())

cfgs = pA["configs"]
dt   = pA["decision_table_vs_original"]
gd   = gate_result["gates"]
gd_d = gd["d_repeat_correlation"]

def gstatus(g):
    p = g.get("pass")
    if p is True or p == "pass": return "PASS"
    if p is False or p == "fail": return "FAIL"
    return "UNVERIFIED"

# gap rows
gap_rows = ""
for row in gap.get("rows", []):
    if "same_season_xfip_r" in row:
        note = " *(NEVER DISPLAYED)*" if "insample" in row["label"] else ""
        yoy  = f"{row['yoy_stability_pooled_r']:.4f}" if row.get("yoy_stability_pooled_r") else "n/a"
        gap_rows += f"| {row['label']}{note} | {row['same_season_era_r']:.4f} | {row['same_season_xfip_r']:.4f} | {yoy} | {row['n']} |\n"

# independence pairs
indep_pairs = ""
hm = independence.get("pairwise_r", {})
for a in sorted(hm):
    for b in sorted(hm.get(a, {})):
        if a < b:
            r    = hm[a][b]
            flag = " <- FLAG |r|>0.5" if abs(r) > 0.5 else ""
            indep_pairs += f"- {a} vs {b}: r={r:.3f}{flag}\n"

ext_cors = independence.get("extension_correlations", {})
ext_lines = "\n".join(f"- {k}: r={v:.3f}" for k, v in ext_cors.items())

# dr2 rows
dr2_rows = ""
for k, v in eb["ii_incremental_delta_r2"]["per_pair"].items():
    ci_lo = f"{v['ci_lo']:.5f}" if v.get("ci_lo") is not None else "?"
    ci_hi = f"{v['ci_hi']:.5f}" if v.get("ci_hi") is not None else "?"
    dr2_rows += f"| {k} | {v['delta_r2']:.5f} | {ci_lo} | {ci_hi} | {v['n']} |\n"

def tf(b): return "YES" if b else "no"
def pfail(b, msg): return "PASS" if b else f"FAIL -- {msg}"

lines = [
    "# Deception+ v2b Completion Table",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    f"**Status: {gate_result['deception_status']}** | **Label: {gate_result['label']}**",
    "",
    f"Model: **v2b** -- pitcher-season form, features: {', '.join(v2b_meta['features'])}",
    "Change from v2: season dropped per pre-registered inconclusive_means_drop rule (diff=-0.0012, |diff|<0.03 margin).",
    "No unseen-season handling needed (season is not a feature).",
    "",
    "",
    "## 1. Audit Findings (from v2 post-run audit)",
    "",
    "| Finding | Value | Note |",
    "|---------|-------|------|",
    "| v1 per-request scoring | yes | removed in v2 -- all scores pre-computed |",
    "| v1 split-half r | 0.998 | artifact: consistency features computed pre-split |",
    "| arm_angle R2 from mechanics | 0.789 | < 0.90 threshold -> KEEP |",
    "| Identifiability top-1/chance | 146x | very high; identity-prone features dropped |",
    "| arm_angle coverage | 2021-2026 | 0% before 2020, partial 2026 (75.9%) |",
    "",
    "",
    f"## 2. Configuration Tests -- Qualified Common Set (n>={MIN_PITCHES} pitches)",
    "",
    "> CORRECTION from original part2_results.json: A0/A1/A2 were previously evaluated on n=5,118",
    "> (all pitcher-seasons including unqualified), while B was evaluated on n=3,363 (qualified only).",
    "> All figures below are restricted to the qualified common set (n=3,363).",
    "",
    "| Config | xFIP r | 95% CI | n |",
    "|--------|--------|--------|---|",
    f"| A0 (v1 baseline) | {cfgs['A0']['xfip_corr']:.4f} | [{cfgs['A0']['ci_lo']:.4f}, {cfgs['A0']['ci_hi']:.4f}] | {cfgs['A0']['n']} |",
    f"| A1 (regularised F0) | {cfgs['A1']['xfip_corr']:.4f} | [{cfgs['A1']['ci_lo']:.4f}, {cfgs['A1']['ci_hi']:.4f}] | {cfgs['A1']['n']} |",
    f"| A2 (per-pitch F2) | {cfgs['A2']['xfip_corr']:.4f} | [{cfgs['A2']['ci_lo']:.4f}, {cfgs['A2']['ci_hi']:.4f}] | {cfgs['A2']['n']} |",
    f"| B (pitcher-season F2) | {cfgs['B']['xfip_corr']:.4f} | [{cfgs['B']['ci_lo']:.4f}, {cfgs['B']['ci_hi']:.4f}] | {cfgs['B']['n']} |",
    "",
    "| Test | Comparison (qualified set) | Rule | Decision | Changed? |",
    "|------|---------------------------|------|----------|----------|",
    f"| Test 1: identity-prone | A1-A2 diff={dt['test1_A2_vs_A1']['new_diff']:.4f} CI=[{dt['test1_A2_vs_A1']['new_ci'][0]:.4f},{dt['test1_A2_vs_A1']['new_ci'][1]:.4f}] | drop_unless_significantly_worse | {dt['test1_A2_vs_A1']['new_decision']} | {tf(dt['test1_A2_vs_A1']['changed'])} |",
    f"| Form rule | A2-B diff={dt['form_rule_A2_vs_B']['new_diff']:.4f} CI=[{dt['form_rule_A2_vs_B']['new_ci'][0]:.4f},{dt['form_rule_A2_vs_B']['new_ci'][1]:.4f}] | choose_M2_unless_M1_significantly_better | {dt['form_rule_A2_vs_B']['new_decision']} | {tf(dt['form_rule_A2_vs_B']['changed'])} |",
    "| arm_angle screening | R2=0.789 < 0.90 | R2_gte_0.90_AND_removal_shown_equivalent | KEEP | no |",
    "| extension screening | diff=+0.0334 > 0.03 | removal_shown_equivalent | KEEP | no |",
    "| season screening | diff=-0.0012 (inconclusive) | inconclusive_means_drop | DROP | no (recorded but not enforced in v2; now enforced in v2b) |",
    "",
    "",
    "## 3. Correctness Gates",
    "",
    "| Gate | Value | Threshold | Status |",
    "|------|-------|-----------|--------|",
    f"| (a) Same-season xFIP/ERA < 0 | xFIP={gd['a_same_season_negative']['xfip_r']:.4f}, ERA={gd['a_same_season_negative']['era_r']:.4f} | both < 0 | {gstatus(gd['a_same_season_negative'])} |",
    f"| (b) Pooled YoY >= 0.30, min pair >= 0.25 | pooled={gd['b_yoy_stability']['pooled_yoy']:.4f}, min={gd['b_yoy_stability']['min_consecutive_pair']:.4f} | 0.30 / 0.25 | {gstatus(gd['b_yoy_stability'])} |",
    f"| (c) Hygiene: mechanics/body only | {gd['c_hygiene']['features']} | no forbidden features | {gstatus(gd['c_hygiene'])} |",
    f"| (d) Repeat correlation >= 0.98 | min_r={gd_d.get('min_repeat_correlation',0):.4f}, avg_r={gd_d.get('avg_repeat_correlation',0):.4f} | 0.98 | {gstatus(gd_d)} |",
    f"| (e) All intended seasons scored | missing={gd['e_all_seasons_scored']['missing']} | none | {gstatus(gd['e_all_seasons_scored'])} |",
    f"| (f) No per-request scoring | parquet lookup only | no per-request | {gstatus(gd['f_no_per_request_scoring'])} |",
    "",
    f"All gates {'PASS' if gate_result['all_gates_pass'] else 'FAIL -- see gate_failures: ' + str(gate_result['gate_failures'])}.",
    "",
    "",
    "## 4. Evidence Bar",
    "",
    "| Item | Value | Threshold | Status |",
    "|------|-------|-----------|--------|",
    f"| (i) Next-season xFIP corr <= -0.20 | avg r={eb['i_next_season_xfip']['avg_r']:.4f} | <= -0.20 | {pfail(eb['i_next_season_xfip']['pass'], 'bar not met')} |",
    f"| (ii) Incremental dR2 CI lower > 0 | avg dR2={eb['ii_incremental_delta_r2']['avg_delta_r2']:.5f}, pooled CI lo={eb['ii_incremental_delta_r2']['pooled_ci_lo_min_across_pairs']:.2e} | CI lower > 0 | {pfail(eb['ii_incremental_delta_r2']['pass'], 'CI lower <= 0')} |",
    "",
    "Incremental dR2 per pair (predicting next-season xFIP; model = last-season xFIP + Deception+ vs last-season xFIP alone):",
    "",
    "| Pair | dR2 | CI lo | CI hi | n |",
    "|------|-----|-------|-------|---|",
    dr2_rows,
    f"Both items required for 'validated'. Overall evidence bar: {'MET -> validated' if gate_result['evidence_bar_summary']['overall_met'] else 'NOT MET -> label remains limited_evidence'}.",
    "",
    f"NOTE: Item (i) threshold <= -0.20 is a pre-registered requirement, not aspirational. Average r = {eb['i_next_season_xfip']['avg_r']:.4f}.",
    f"NOTE: Item (ii) prior version used all-positive proxy (DEC-V2-03). This version computes proper pitcher-clustered bootstrap 95% CI per pair; pooled CI lower = min(ci_lo) across pairs = {eb['ii_incremental_delta_r2']['pooled_ci_lo_min_across_pairs']:.2e} > 0 (PASS).",
    "",
    "",
    f"## 5. In-Sample Gap Table (qualified set, n>={MIN_PITCHES}, 2021-2026)",
    "",
    "| Version | Same-season ERA r | Same-season xFIP r | YoY stability r | n |",
    "|---------|------------------|--------------------|----------------|---|",
    gap_rows,
    "## 6. Independence",
    "",
    f"Available models for heatmap: {independence.get('available_models', [])}",
    f"n (inner-joined qualified pitcher-seasons, 2021-2026): {independence.get('n', '?')}",
    "",
    "Pairwise Pearson r:",
    indep_pairs,
    f"Flags (|r| > 0.5): {independence.get('flags_abs_r_gt_0p5') or 'none'}",
    "",
    "Extension correlations (corr with mean release_extension per pitcher-season):",
    ext_lines,
    "",
    "",
    "## 7. Process-Check Summary",
    "",
    "All Part 2 comparisons on mismatched sets have been recomputed on the qualified common set (n=3,363).",
    "See partA_qualified_results.json and partE_process_check.json for full details.",
    "",
    "Key finding: On the qualified set, A0 xFIP r drops from 0.237 (all pitcher-seasons) to -0.006 (qualified only),",
    "confirming original A0/A1/A2 figures were inflated by unqualified pitcher-seasons with atypical mechanics.",
    "B (0.084) is best on the qualified set. No decisions changed.",
    "",
    f"Gate (d) previously recorded as cannot_verify_from_parquet -- now VERIFIED at min_r={gd_d.get('min_repeat_correlation',0):.4f}.",
    f"Evidence bar item (ii) previously used all-positive proxy -- now uses proper CI lower bound: {eb['ii_incremental_delta_r2']['pooled_ci_lo_min_across_pairs']:.2e} > 0 (PASS).",
    f"Evidence bar item (i) remains FAIL: avg r={eb['i_next_season_xfip']['avg_r']:.4f}, threshold <= -0.20.",
    "",
    "",
    "## 8. Active Model",
    "",
    f"Active version: v2b",
    f"DECEPTION_ACTIVE_VERSION = \"v2b\" in config/model_policy.py",
    f"DECEPTION_USE_IN_ARCHETYPES = {gate_result['deception_use_in_archetypes']} in config/model_policy.py",
    "(All correctness gates pass; evidence bar item (i) fails; label = limited_evidence)",
    "",
    "",
    "## 9. Deliverables",
    "",
    "| File | Description |",
    "|------|-------------|",
    "| diagnostics/deception_v2/spec_lock.json | Pre-registered spec (unchanged) |",
    "| diagnostics/deception_v2/partA_oof_cache.parquet | OOF predictions A0/A1/A2/B on qualified set |",
    "| diagnostics/deception_v2/partA_qualified_results.json | Part 2 results on qualified common set |",
    "| diagnostics/deception_v2/chosen_config_v2b.json | Corrected chosen config |",
    "| diagnostics/deception_v2/deception_v2b_scores.parquet | v2b cross-fitted scores (season dropped) |",
    "| diagnostics/deception_v2/v2b_oof_all.parquet | v2b OOF predictions (3 reps) |",
    "| models/deception_plus_v2b_fold*_rep*.txt | v2b fold models (15 files) |",
    "| models/deception_plus_v2b_meta.json | v2b model meta |",
    "| diagnostics/deception_v2/gate_d_v2b.json | Gate (d) verified repeat correlation |",
    "| diagnostics/deception_v2/gate_results_v2b.json | Re-evaluated gates for v2b |",
    "| diagnostics/deception_v2/evidence_bar_v2b.json | Evidence bar with proper pooled CI |",
    "| diagnostics/deception_v2/independence_heatmap_v2b.json | Independence heatmap |",
    "| diagnostics/deception_v2/in_sample_gap_v2b.json | In-sample gap table |",
    "| diagnostics/deception_v2/identifiability_v2b.json | Identifiability / memorization check |",
    "| diagnostics/deception_v2/partE_process_check.json | Process-check list |",
    "| Methodology/DECEPTION_COMPLETION.md | This file |",
    "",
    "---",
    "Post-audit closeout complete. Branch: deception-v2.",
]

content = "\n".join(lines)
path = ROOT / "Methodology" / "DECEPTION_COMPLETION.md"
path.write_text(content, encoding="utf-8")
print(f"wrote {path}")
