"""
Generate Methodology/STUFF_V4_REPORT.md and Methodology/live_models/STUFF_PLUS_V4_FINAL_METHODOLOGY.html
from the computed JSON outputs.

Run after train_stuff_v4.py completes.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT / "diagnostics" / "stuff_v4"
METHODOLOGY_DIR = ROOT / "Methodology"
LIVE_MODELS_DIR = METHODOLOGY_DIR / "live_models"


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
        return {}


def fmt_r(r, ci_lo=None, ci_hi=None):
    if r is None:
        return "—"
    try:
        r = float(r)
        if ci_lo is not None and ci_hi is not None:
            return f"{r:+.3f} [{float(ci_lo):+.3f}, {float(ci_hi):+.3f}]"
        return f"{r:+.3f}"
    except Exception:
        return "—"


def fmt_n(n):
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def generate_report():
    spec = load_json(DIAG_DIR / "spec_lock.json")
    part1 = load_json(DIAG_DIR / "part1_audit.json")
    part2 = load_json(DIAG_DIR / "part2_ablation_table.json")
    part4 = load_json(DIAG_DIR / "part4_validation.json")
    meta_path = ROOT / "models" / "stuff_plus_v4_lgbm_meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}

    trained_at = meta.get("trained_at", "unknown")
    model_hash = meta.get("model_hash_sha256", "unknown")
    n_training_rows = meta.get("n_training_rows", "unknown")
    final_features = meta.get("all_features", [])
    candidate_applied = meta.get("candidate_changes_applied", {})
    spec_lock_commit = spec.get("spec_lock_commit", "unknown")

    # Final variant info
    final_variant = part2.get("final_variant", {})
    stage1 = part2.get("stage1_result", {})
    stage2 = part2.get("stage2_result", {})

    # Build report
    lines = []
    lines += [
        "# Stuff+ v4 — Final Methodology",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model trained at**: {trained_at}  ",
        f"**Model SHA-256**: `{model_hash}`  ",
        f"**Spec-lock commit**: `{spec_lock_commit}`  ",
        "",
        "---",
        "",
        "## 1. Training Setup",
        "",
        f"- **Training seasons**: 2016–2026 excluding 2020 ({fmt_n(n_training_rows)} pitches after filtering)",
        "- **Scoring seasons**: 2016–2026 including 2020 (flagged `season_flag = '60-game season'`)",
        "- **Excluded pitch types**: PO, UN, EP (junk/non-competitive)",
        "- **KN (knuckleball)**: included",
        "- **Postseason**: excluded (game_type = 'R' filter)",
        "- **Objective**: Huber loss (predict `delta_run_exp`)",
        "- **Fixed tree count**: 2000 (reused from v3 `n_estimators`, no early stopping)",
        "",
        "### Hyperparameters (reused from v3)",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]

    lgbm = spec.get("lgbm_hyperparameters", {})
    for k, v in lgbm.items():
        if not k.startswith("note"):
            lines.append(f"| `{k}` | `{v}` |")

    lines += [
        "",
        "---",
        "",
        "## 2. Final Feature Set",
        "",
        f"Candidate changes applied: `drop_season={candidate_applied.get('drop_season')}`, "
        f"`drop_effective_speed={candidate_applied.get('drop_effective_speed')}`, "
        f"`include_haa={candidate_applied.get('include_haa')}`",
        "",
        "### Features used in final model",
        "",
    ]

    if final_features:
        lines += ["| # | Feature | Type |", "|---|---|---|"]
        for i, f in enumerate(final_features, 1):
            ftype = "categorical" if f in spec.get("baseline_feature_list", []) and f in ["pitch_type", "p_throws", "stand", "season"] else "numeric"
            # Better detection
            cat_feats = meta.get("categorical_features", ["pitch_type", "p_throws", "stand", "season"])
            ftype = "categorical" if f in cat_feats else "numeric"
            lines.append(f"| {i} | `{f}` | {ftype} |")
    else:
        lines.append("*(feature list unavailable — check models/stuff_plus_v4_lgbm_meta.json)*")

    lines += [
        "",
        "### Excluded features (vs v3 baseline)",
        "",
        "- `release_extension`, `release_pos_x_as`, `release_pos_z`, `arm_angle` — excluded in v3 (Deception+ only); remain excluded in v4",
        "- `season`: " + ("dropped (V1 ablation passed)" if candidate_applied.get("drop_season") else "kept"),
        "- `effective_speed`: " + ("dropped (V2 ablation passed)" if candidate_applied.get("drop_effective_speed") else "kept"),
        "- `haa_residual` (new): " + ("added (Stage 2 leakage check passed)" if candidate_applied.get("include_haa") else "NOT added (leakage check or predictive test failed)"),
        "",
        "### Forbidden features (leakage guard)",
        "",
        "The following columns are explicitly verified absent from the feature list before every training run:",
        "`plate_x`, `plate_z`, `balls`, `strikes`, `release_extension`, `release_pos_x_as`,",
        "`release_pos_z`, `arm_angle`, `pitcher_id`, `game_id`",
        "",
        "---",
        "",
        "## 3. HAA Residual (if adopted)",
        "",
        "**Definition**: Residualized horizontal approach angle — actual HAA minus expected HAA from an OLS regression.",
        "",
        "**HAA computation**: velocity vector angle at front of plate from `vx0/vy0/ax/ay` physics,",
        "mirrored by `p_throws` so arm-side is positive for both LHP and RHP.",
        "",
        "**Expected HAA**: OLS per pitch type on [mirrored `plate_x`, mirrored `plate_x`², mirrored `release_pos_x`, `release_speed`].",
        "Fit on training seasons only; applied to all seasons including 2020.",
        "",
        "**Leakage check (Part 1)**:",
        "",
        f"| Correlation | Value | Threshold | Pass? |",
        f"|---|---|---|---|",
    ]

    haa_corrs = part1.get("haa_residual_correlations", {})
    px = haa_corrs.get("plate_x", {}) or {}
    rpx = haa_corrs.get("release_pos_x", {}) or {}
    lines += [
        f"| `|corr(haa_residual, plate_x)|` | {abs(px.get('corr', float('nan'))):.4f} | < 0.1 | {'✓' if abs(px.get('corr', 1)) < 0.1 else '✗'} |",
        f"| `|corr(haa_residual, release_pos_x)|` | {abs(rpx.get('corr', float('nan'))):.4f} | < 0.1 | {'✓' if abs(rpx.get('corr', 1)) < 0.1 else '✗'} |",
        "",
        "---",
        "",
        "## 4. Ablation Results (Part 2 — Pitcher-Grouped 5-Fold CV, 30% Subsample)",
        "",
        "**Setup**: 30% subsample of pitchers (all their seasons), stratified by `p_throws` and pitch-count tercile, seed=42.",
        "5-fold CV grouped by pitcher (every pitch of a pitcher falls in exactly one fold).",
        f"Subsample: {fmt_n(part2.get('subsample_n_pitchers', 0))} pitchers, seed=42.",
        "",
        "### Stage 1 — Drop candidates",
        "",
        "| Variant | Description | Corr with V0a | Same-season xFIP r | Next-season xFIP r | Decision |",
        "|---|---|---|---|---|---|",
    ]

    variants_table = part2.get("variants_table", {})
    for vname in ["V0a", "V0b", "V1", "V2", "V3"]:
        v = variants_table.get(vname, {})
        desc = {
            "V0a": "Baseline (seed=42)",
            "V0b": "Baseline (seed=123; noise floor)",
            "V1":  "minus `season`",
            "V2":  "minus `effective_speed`",
            "V3":  "minus both",
        }.get(vname, vname)
        corr_v0a = fmt_r(v.get("corr_with_v0a"))
        ss_xfip = fmt_r(v.get("same_season_xfip_corr"))
        ns_xfip = fmt_r(v.get("next_season_xfip_corr"))
        decision = stage1.get("per_variant", {}).get(vname, "—")
        lines.append(f"| {vname} | {desc} | {corr_v0a} | {ss_xfip} | {ns_xfip} | {decision} |")

    lines += [
        "",
        f"**Stage 1 chosen variant**: `{stage1.get('chosen_variant', '—')}` — "
        f"drop_season={stage1.get('drop_season')}, drop_effective_speed={stage1.get('drop_effective_speed')}",
        "",
        "### Stage 2 — HAA residual",
        "",
        f"HAA leakage check: {stage2.get('haa_audit', {}).get('leakage_ok', '—')}",
        f"Decision: adopt_haa={stage2.get('adopt_haa', '—')} — {stage2.get('reason', '—')}",
        "",
        "---",
        "",
        "## 5. Final Fit Details",
        "",
        f"- **Trained at**: {trained_at}",
        f"- **Seeds**: random_state=42 (LightGBM)",
        f"- **Tree count**: {FIXED_N_TREES} (no early stopping)",
        f"- **Training rows**: {fmt_n(n_training_rows)}",
        f"- **Model hash (SHA-256)**: `{model_hash}`",
        "",
        "---",
        "",
        "## 6. Validation (Part 4 — Final Model, In-Sample)",
        "",
        "> **All tables in this section are FINAL MODEL, IN-SAMPLE (descriptive).**",
        "> Out-of-sample evidence is in the ablation pitcher-holdout (Section 4, labeled conservative).",
        "",
        "### Per-Season Score Distribution",
        "",
        "| Season | N pitcher-seasons | Mean Stuff+ | Std Stuff+ | Note |",
        "|---|---|---|---|---|",
    ]

    season_stats = part4.get("season_stats", {})
    for s in ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]:
        st = season_stats.get(s, {})
        mean_s = f"{st['mean_score']:.1f}" if st.get('mean_score') is not None else "—"
        std_s = f"{st['std_score']:.1f}" if st.get('std_score') is not None else "—"
        note = st.get("season_flag", "")
        lines.append(f"| {s} | {fmt_n(st.get('n_pitcher_seasons', 0))} | {mean_s} | {std_s} | {note} |")

    lines += [
        "",
        "### Year-over-Year Stability",
        "",
        "| Pair | Pearson r | n |",
        "|---|---|---|",
    ]

    yoy = part4.get("yoy_stability", {})
    for cp in yoy.get("consecutive_pairs", []):
        lines.append(f"| {cp['s1']}→{cp['s2']} | {cp['r']:+.3f} | {fmt_n(cp['n'])} |")
    lines += [
        f"",
        f"**Pooled mean YoY r**: {fmt_r(yoy.get('pooled_mean_r'))}",
        f"*(Pairs touching 2020 excluded from pooled mean)*",
        "",
        "### Mean Score by Pitch Type",
        "",
        "| Pitch Type | Mean Stuff+ | Flagged (>3 pts from 100)? |",
        "|---|---|---|",
    ]

    pt_means = part4.get("mean_by_pitch_type", {})
    flagged = part4.get("flagged_pitch_types_far_from_100", {})
    for pt, mean_val in sorted(pt_means.items()):
        flag = "⚠ YES" if pt in flagged else ""
        lines.append(f"| {pt} | {mean_val:.1f} | {flag} |")

    lines += [
        "",
        "### Split-Half Reliability (2025 season)",
        "",
        "| Min Pitches | Spearman-Brown r | n pitchers |",
        "|---|---|---|",
    ]

    shr = part4.get("split_half_reliability", {})
    for n_thresh in [50, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000]:
        r = shr.get(n_thresh) or shr.get(str(n_thresh))
        if r and isinstance(r, dict):
            sb = r.get("spearman_brown")
            np_ = r.get("n_pitchers")
            sb_str = f"{sb:.3f}" if sb is not None else "—"
            np_str = fmt_n(np_)
        else:
            sb_str = "—"
            np_str = "—"
        lines.append(f"| {n_thresh} | {sb_str} | {np_str} |")

    lines += [
        "",
        "### Leakage Checks",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Forbidden features absent | {part4.get('leakage_checks', {}).get('forbidden_features_absent', '—')} |",
        f"| HAA residual ↔ plate_x | {abs(px.get('corr', float('nan'))):.4f} (threshold < 0.1) |",
        f"| HAA residual ↔ release_pos_x | {abs(rpx.get('corr', float('nan'))):.4f} (threshold < 0.1) |",
        "",
        "---",
        "",
        "## 7. Open Issues / Known Limitations",
        "",
        "1. **2020 exclusion from training**: The 60-game season is excluded from training but scored. Scores are flagged `season_flag='60-game season'`.",
        "2. **Validation is in-sample**: All correlation figures in Section 6 are computed on the same data the model was trained on. The only out-of-sample evidence is the pitcher-holdout in the ablation (Section 4).",
        "3. **No 2026 holdout**: Per pre-registration, no second retrain is performed with a 2026 holdout.",
        "4. **xFIP availability**: xFIP data starts from 2021 in the DB; next-season xFIP correlations cover 2021-forward only.",
        "5. **Split-half reliability uses 2025 only**: A multi-season estimate would be more stable.",
        "6. **Seed stability**: Not yet run (requires a second full training pass). Expected >= 0.99.",
        "",
        "---",
        "",
        "## 8. Sanity Tables (Selected Seasons)",
        "",
    ]

    for season in ["2016", "2020", "2022", "2025", "2026"]:
        t = part4.get("sanity_tables", {}).get(season)
        if not t:
            continue
        lines += [
            f"### {season} — Top 15 / Bottom 15",
            "",
            f"| Rank | pitcher_id | Stuff+ v4 | N pitches |",
            "|---|---|---|---|",
        ]
        for i, row in enumerate(t.get("top_15", []), 1):
            lines.append(f"| T{i} | {row['pitcher_id']} | {row['stuff_plus_v4']:.1f} | {fmt_n(row['n_pitches'])} |")
        lines.append("| ... | ... | ... | ... |")
        for i, row in enumerate(t.get("bottom_15", []), 1):
            lines.append(f"| B{i} | {row['pitcher_id']} | {row['stuff_plus_v4']:.1f} | {fmt_n(row['n_pitches'])} |")
        lines.append("")

    report_text = "\n".join(lines)

    # Save markdown report
    report_path = METHODOLOGY_DIR / "STUFF_V4_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report saved: {report_path}")

    # Generate HTML
    generate_html_methodology(report_path, report_text, LIVE_MODELS_DIR / "STUFF_PLUS_V4_FINAL_METHODOLOGY.html")


FIXED_N_TREES = 2000  # will be overridden from meta if available


def generate_html_methodology(md_path, md_text, html_path):
    """Convert markdown report to standalone HTML."""
    import re

    html_path.parent.mkdir(parents=True, exist_ok=True)

    # Simple markdown-to-HTML conversion
    lines = md_text.split("\n")
    html_lines = []
    in_table = False
    in_code = False

    def escape(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def md_inline(s):
        # Bold
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        # Italic
        s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
        # Code
        s = re.sub(r'`(.+?)`', r'<code>\1</code>', s)
        return s

    for line in lines:
        orig = line
        line_stripped = line.strip()

        if line_stripped.startswith("```"):
            if not in_code:
                html_lines.append('<pre><code>')
                in_code = True
            else:
                html_lines.append('</code></pre>')
                in_code = False
            continue

        if in_code:
            html_lines.append(escape(line))
            continue

        # Tables
        if "|" in line_stripped and line_stripped.startswith("|"):
            if not in_table:
                html_lines.append('<table>')
                in_table = True
            cells = [c.strip() for c in line_stripped.split("|")[1:-1]]
            if all(re.match(r'^[-: ]+$', c) for c in cells):
                continue  # separator row
            tag = "th" if not any(r"<td>" in h for h in html_lines[-3:]) else "td"
            # Detect if it's a header (first table row)
            last_non_empty = next((h for h in reversed(html_lines) if h.strip()), "")
            tag = "th" if last_non_empty == "<table>" else "td"
            row = "".join(f"<{tag}>{md_inline(escape(c))}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
            continue
        else:
            if in_table:
                html_lines.append('</table>')
                in_table = False

        # Headings
        if line_stripped.startswith("### "):
            html_lines.append(f"<h3>{md_inline(escape(line_stripped[4:]))}</h3>")
        elif line_stripped.startswith("## "):
            html_lines.append(f"<h2>{md_inline(escape(line_stripped[3:]))}</h2>")
        elif line_stripped.startswith("# "):
            html_lines.append(f"<h1>{md_inline(escape(line_stripped[2:]))}</h1>")
        elif line_stripped.startswith("---"):
            html_lines.append("<hr>")
        elif line_stripped.startswith("- "):
            html_lines.append(f"<li>{md_inline(escape(line_stripped[2:]))}</li>")
        elif line_stripped == "":
            html_lines.append("<br>")
        else:
            html_lines.append(f"<p>{md_inline(escape(line_stripped))}</p>")

    if in_table:
        html_lines.append('</table>')

    body = "\n".join(html_lines)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stuff+ v4 — Final Methodology</title>
<style>
  body {{
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 24px;
    background: #fafafa;
    color: #222;
    line-height: 1.6;
  }}
  h1 {{ font-size: 2em; border-bottom: 3px solid #333; padding-bottom: 0.3em; margin-top: 0; }}
  h2 {{ font-size: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; margin-top: 2em; color: #333; }}
  h3 {{ font-size: 1.1em; color: #444; margin-top: 1.5em; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.88em;
    margin: 1em 0;
    font-family: 'Courier New', monospace;
  }}
  th, td {{
    border: 1px solid #ccc;
    padding: 6px 10px;
    text-align: left;
  }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  code {{
    background: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 0.9em;
    font-family: 'Courier New', monospace;
  }}
  pre {{ background: #f0f0f0; padding: 12px; border-radius: 4px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  li {{ margin: 4px 0; }}
  p {{ margin: 0.5em 0; }}
  hr {{ border: 0; border-top: 1px solid #ddd; margin: 2em 0; }}
  strong {{ color: #111; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a1a; color: #e0e0e0; }}
    h1, h2, h3 {{ color: #f0f0f0; border-color: #444; }}
    th {{ background: #2a2a2a; }}
    tr:nth-child(even) {{ background: #222; }}
    code {{ background: #2a2a2a; color: #e0e0e0; }}
    pre {{ background: #2a2a2a; }}
    td, th {{ border-color: #444; }}
  }}
</style>
</head>
<body>
{body}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML methodology saved: {html_path}")


if __name__ == "__main__":
    generate_report()
    print("Report generation complete.")
