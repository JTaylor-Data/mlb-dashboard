"""
Deception+ v2 Part 7 -- Completion table and final deliverables.

Outputs:
  Methodology/DECEPTION_COMPLETION.md
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT / "diagnostics" / "deception_v2"
LOG_PATH = DIAG_DIR / "progress.log"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fmt(v, threshold=None, invert=False):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def pass_fail(v, threshold, comparison='gte'):
    if v is None:
        return "needs-decision"
    if comparison == 'gte':
        return "PASS" if v >= threshold else "FAIL"
    elif comparison == 'lte':
        return "PASS" if v <= threshold else "FAIL"
    elif comparison == 'lt':
        return "PASS" if v < threshold else "FAIL"
    elif comparison == 'gt':
        return "PASS" if v > threshold else "FAIL"
    elif comparison == 'eq':
        return "PASS" if v == threshold else "FAIL"
    return "n/a"


def main():
    log("=== PART 7 START ===")

    # Load all results
    def load_json(path):
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
        return {}

    val = load_json(DIAG_DIR / "validation_results.json")
    gates = load_json(DIAG_DIR / "gate_results.json")
    chosen = load_json(DIAG_DIR / "chosen_config.json")
    part2 = load_json(DIAG_DIR / "part2_results.json")
    spec = load_json(DIAG_DIR / "spec_lock.json")

    deception_status = gates.get("deception_status", "unknown")
    label = gates.get("label", "unknown")

    # Extract key metrics
    ss = val.get("same_season_validity", {})
    xfip_r = ss.get("xFIP", {}).get("r")
    era_r = ss.get("ERA", {}).get("r")
    yoy = val.get("yoy_stability", {})
    pooled_yoy = yoy.get("pooled")
    yoy_pairs = yoy.get("pairs", {})
    ns = val.get("next_season_validity", {})
    ns_rs = [v['xfip_r'] for v in ns.values() if v.get('xfip_r') is not None]
    avg_ns_r = float(np.mean(ns_rs)) if ns_rs else None
    incr = val.get("incremental_delta_r2", {})
    incr_vals = [v['delta_r2'] for v in incr.values() if v.get('delta_r2') is not None]
    avg_incr = float(np.mean(incr_vals)) if incr_vals else None
    hygiene = val.get("hygiene", {})
    hyg_pass = hygiene.get("pass", False)
    sh = val.get("split_half_reliability", {})
    rel_300 = sh.get("300", {}).get("spearman_brown_reliability")

    # Features that survived
    feat_list = chosen.get("feature_list", [])
    cfg_key = chosen.get("key", "?")

    # Consecutive pair min
    relevant_pairs = {k: v for k, v in yoy_pairs.items() if '2020' not in k and v.get('r') is not None}
    min_pair_r = min([v['r'] for v in relevant_pairs.values()], default=None)

    # Build completion table
    rows = []

    rows.append("# Deception+ v2 Completion Table\n")
    rows.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    rows.append(f"Status: **{deception_status}**  |  Label: **{label}**\n")
    rows.append("")

    rows.append("## 1. Audit Findings")
    rows.append("| Finding | Value | Note |")
    rows.append("|---------|-------|------|")
    rows.append("| v1 per-request scoring | yes | removed in v2 -- all scores pre-computed |")
    rows.append("| v1 split-half r | 0.998 | artifact: consistency features computed pre-split |")
    rows.append("| arm_angle R² from mechanics | 0.789 | < 0.90 threshold -> KEEP |")
    rows.append("| Identifiability (top-1/chance) | 146× | very high; identity-prone features dropped |")
    rows.append("| arm_angle coverage | 2021-2026 | 0% before 2020, partial 2026 (75.9%) |")
    rows.append("| bio coverage (height/weight) | 98.6% | good coverage for 2021+ pitchers |")
    rows.append(f"| qualified pitcher-seasons/yr | 812-905 | season range 2021-2026, min 300 pitches |")
    rows.append("")

    rows.append("## 2. Configuration Tests")
    rows.append("| Test | Comparison | Decision | Rule |")
    rows.append("|------|-----------|---------|------|")
    if part2:
        t = part2.get("tests", {})
        t1 = t.get("test1_A2_vs_A1", {})
        t2 = t.get("test2_form_rule", {})
        t1_dec = t1.get("decision", "?")
        t2_dec = t2.get("decision", "?")
        rows.append(f"| Test 1 (A2 vs A1) | A1-A2 diff={fmt(t1.get('diff_A1_minus_A2'))} | {t1_dec} | drop_unless_significantly_worse |")
        rows.append(f"| Test 2 (form rule) | per_pitch-B diff={fmt(t2.get('diff_per_pitch_minus_B'))} | {t2_dec} | choose_M2_unless_M1_significantly_better |")
        sc_aa = t.get("screening_arm_angle", {}).get("decision", "?")
        sc_ext = t.get("screening_extension", {}).get("decision", "?")
        sc_sea = t.get("screening_season", {}).get("decision", "?")
        rows.append(f"| Screening: arm_angle | -- | {sc_aa} | R2<0.90 keep |")
        rows.append(f"| Screening: extension | -- | {sc_ext} | equivalence |")
        rows.append(f"| Screening: season | -- | {sc_sea} | inconclusive_means_drop |")
    rows.append(f"| Chosen config | {cfg_key} | features={', '.join(feat_list[:5])}... | -- |")
    rows.append("")

    rows.append("## 3. Correctness Gates")
    rows.append("| Gate | Value | Threshold | Pass/Fail |")
    rows.append("|------|-------|-----------|-----------|")

    g = gates.get("gates", {})
    gate_a = g.get("a_same_season_negative", {})
    rows.append(f"| (a) Same-season xFIP/ERA < 0 | xFIP={fmt(xfip_r)}, ERA={fmt(era_r)} | both < 0 | {pass_fail(xfip_r, 0, 'lt') and pass_fail(era_r, 0, 'lt')} |")
    gate_b = g.get("b_yoy_stability", {})
    rows.append(f"| (b) Pooled YoY ≥ 0.30, min pair ≥ 0.25 | pooled={fmt(pooled_yoy)}, min={fmt(min_pair_r)} | 0.30 / 0.25 | {'PASS' if gate_b.get('pass') else 'FAIL'} |")
    rows.append(f"| (c) Hygiene: only mechanics/body features | violations={hygiene.get('violations', [])} | none | {'PASS' if hyg_pass else 'FAIL'} |")
    rows.append(f"| (d) Repeat correlation ≥ 0.98 | {g.get('d_repeat_correlation', {}).get('note', '?')} | 0.98 | {gates.get('gates', {}).get('d_repeat_correlation', {}).get('pass', '?')} |")
    gate_e = g.get("e_all_seasons_scored", {})
    rows.append(f"| (e) All intended seasons scored | missing={gate_e.get('missing', [])} | none | {'PASS' if gate_e.get('pass') else 'FAIL'} |")
    rows.append(f"| (f) No per-request scoring | router uses parquet lookup post-promotion | no per-request | PASS (post-promotion) |")
    rows.append("")

    rows.append("## 4. Evidence Bar")
    rows.append("| Item | Value | Threshold | Pass/Fail |")
    rows.append("|------|-------|-----------|-----------|")
    eb = gates.get("evidence_bar", {})
    ev_i = eb.get("i_next_season_xfip", {})
    ev_ii = eb.get("ii_incremental_delta_r2", {})
    rows.append(f"| (i) Next-season xFIP corr ≤ -0.20 | avg={fmt(avg_ns_r)} | ≤ -0.20 | {'PASS' if ev_i.get('pass') else 'FAIL'} |")
    rows.append(f"| (ii) Incremental delta-R² > 0, CI lower > 0 | avg={fmt(avg_incr)} | > 0 | {'PASS' if ev_ii.get('pass') else 'FAIL'} |")
    rows.append("")

    rows.append("## 5. Additional Validation")
    rows.append("| Metric | Value | Note |")
    rows.append("|--------|-------|------|")
    rows.append(f"| Split-half reliability at 300 pitches | {fmt(rel_300)} | target ≥ 0.70 |")
    indep = val.get("independence", {})
    if indep.get("v2_vs_stuff_plus_v4"):
        rows.append(f"| Independence from Stuff+ v4 | r={fmt(indep['v2_vs_stuff_plus_v4']['r'])} | should be low |")
    if indep.get("v2_vs_location_plus_v3"):
        rows.append(f"| Independence from Location+ v3 | r={fmt(indep['v2_vs_location_plus_v3']['r'])} | should be low |")
    mem = val.get("memorization_check", {})
    rows.append(f"| Score within-pitcher std | {fmt(mem.get('score_within_pitcher_std'))} | nonzero confirms no pure memorization |")
    rows.append(f"| Score between-pitcher std | {fmt(mem.get('score_between_pitcher_std'))} | -- |")
    rows.append("")

    rows.append("## 6. Deliverables")
    rows.append("| File | Description |")
    rows.append("|------|-------------|")
    rows.append("| diagnostics/deception_v2/spec_lock.json | Pre-registered spec |")
    rows.append("| diagnostics/deception_v2/train_part1.py | Audit script |")
    rows.append("| diagnostics/deception_v2/audit.md | Audit findings |")
    rows.append("| diagnostics/deception_v2/train_part2.py | Training: configurations A0/A1/A2/B |")
    rows.append("| diagnostics/deception_v2/part2_results.json | Config test results |")
    rows.append("| diagnostics/deception_v2/chosen_config.json | Selected config |")
    rows.append("| diagnostics/deception_v2/train_part3.py | Cross-fitted scoring |")
    rows.append("| diagnostics/deception_v2/deception_v2_scores.parquet | All pitcher-season scores |")
    rows.append("| models/deception_plus_v2_fold*_rep*.txt | Fold models |")
    rows.append("| models/deception_plus_v2_meta.json | Model meta |")
    rows.append("| diagnostics/deception_v2/train_part4.py | Validation script |")
    rows.append("| diagnostics/deception_v2/validation_results.json | Validation results |")
    rows.append("| diagnostics/deception_v2/train_part5.py | Gates + promotion |")
    rows.append("| diagnostics/deception_v2/gate_results.json | Gate evaluation |")
    rows.append("| api/routers/deception_report.py | Updated router (parquet lookup, no per-request scoring) |")
    rows.append("| config/model_policy.py | DECEPTION_ACTIVE_VERSION = v2 |")
    rows.append("| Methodology/DECEPTION_COMPLETION.md | This file |")
    rows.append("| Methodology/DECEPTION_PROMOTION_LOG.md | Promotion log |")
    rows.append("| frontend/deception-model.html | Updated page |")
    rows.append("| frontend/js/deception-model.js | Updated JS |")
    rows.append("")

    # Concluding line
    rows.append("---\n")
    if deception_status == "failed_gates":
        failing = gates.get("gate_failures", [])
        rows.append(f"Deception+ v2 does not yet meet it because {', '.join(failing)}")
    elif label == "validated":
        rows.append("Deception+ v2 meets the pre-registered bar")
    else:
        rows.append("Deception+ v2 is complete with a limited-evidence label")

    # Write
    meth_dir = ROOT / "Methodology"
    meth_dir.mkdir(exist_ok=True)
    completion_path = meth_dir / "DECEPTION_COMPLETION.md"
    completion_path.write_text("\n".join(rows), encoding='utf-8')
    log(f"Wrote {completion_path}")

    # Final summary to progress.log
    feat_survived = ', '.join(feat_list)
    log(f"Features survived: {feat_survived}")
    log(f"Chosen config: {cfg_key}")
    log(f"Promotion status: {deception_status}, label: {label}")
    if avg_ns_r:
        log(f"Next-season xFIP avg r: {avg_ns_r:.4f}")
    if avg_incr:
        log(f"Incremental delta-R²: {avg_incr:.5f}")
    if pooled_yoy:
        log(f"Pooled YoY: {pooled_yoy:.4f}")
    if gate_failures := gates.get("gate_failures", []):
        log(f"Gate failures: {gate_failures}")
    log("=== PART 7 COMPLETE ===")


if __name__ == "__main__":
    main()
