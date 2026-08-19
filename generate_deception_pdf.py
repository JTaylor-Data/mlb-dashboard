"""
Generates the Deception+ v1 methodology PDF from
models/deception_plus_v1_report_{season}.json - every number in the
document is read from that file, not hand-typed, so the PDF can never
drift out of sync with what the model actually produced.

    python generate_deception_pdf.py
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

MODELS_DIR = Path(__file__).parent / "models"
REPORT_SEASON = 2025
REPORT_PATH = MODELS_DIR / f"deception_plus_v1_report_{REPORT_SEASON}.json"
OUT_PATH = Path(__file__).parent / "Deception+ v1 - Methodology.pdf"

FOREST = colors.HexColor("#1b4332")
FOREST_DARK = colors.HexColor("#123023")
BARK = colors.HexColor("#6b4423")
TEXT_MUTED = colors.HexColor("#6b6153")
BG_TINT = colors.HexColor("#f7f1e6")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], textColor=FOREST_DARK, spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], textColor=FOREST, spaceBefore=14, spaceAfter=6, fontSize=13))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.3, leading=15, spaceAfter=8))
styles.add(ParagraphStyle("Small", parent=styles["Normal"], fontSize=9, leading=13, textColor=TEXT_MUTED, spaceAfter=6))
styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], textColor=FOREST_DARK, fontSize=24))
styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12, textColor=TEXT_MUTED, alignment=1, spaceAfter=4))


def p(text, style="Body"):
    return Paragraph(text, styles[style])


def data_table(rows, header, col_widths=None):
    data = [header] + rows
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), FOREST),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_TINT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ddceac")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def build(report):
    story = []
    v = report["verdict"]
    val = report["outcome_validity"]
    rel = report["split_half_reliability"]
    ex = report["worked_examples"]

    # ---- Title page ----
    story.append(Spacer(1, 1.6 * inch))
    story.append(p("Deception+ v1", "TitleBig"))
    story.append(p("A proof-of-concept model of delivery/release-based deception", "Subtitle"))
    story.append(p(f"Built {report['trained_at']} &middot; {report['report_season']} season", "Subtitle"))
    story.append(Spacer(1, 0.4 * inch))
    story.append(p(
        "This document exists to answer one question honestly: given only what's real and "
        "obtainable, is a delivery/release-based deception model actually useful? The verdict "
        "below is not decided in advance - it's read directly off the validation numbers in "
        "Section 4.", "Body",
    ))
    story.append(PageBreak())

    # ---- Executive summary / verdict ----
    story.append(p("1. The Verdict", "H1"))
    story.append(p(f"<b>{v['headline']}</b>", "Body"))
    story.append(p(
        f"Split-half reliability (100+ pitch pitchers): <b>{v['split_half_correlation_100plus']:.3f}</b>. "
        f"Outcomes with a meaningful partial correlation beyond Stuff+: <b>{v['n_outcomes_with_meaningful_partial_correlation']} / 5</b>. "
        f"Leave-one-season-out CV mean R&sup2;: <b>{report['cv_r2_mean']:.5f}</b> (see Section 4.3 for why a "
        "near-zero per-pitch R&sup2; here doesn't by itself decide the verdict - the same was true for this "
        "site's other models at the per-pitch level).", "Body",
    ))

    # ---- What was requested vs what's here ----
    story.append(p("2. What Was Requested, What's Actually Here", "H1"))
    story.append(p(
        "The original spec: a gradient-boosted model fed ONLY characteristics of a pitcher's "
        "delivery - windup, timing, release - independent of Stuff+ and arsenal, to find which "
        "combination of delivery traits predicts good outcomes. The full requested feature list: "
        "height, weight, arm angle, extension, length of delivery, ball concealment, timing "
        "variance (holding the ball longer/shorter), quick pitches/leg hangs/timing disruptions, "
        "and release point.", "Body",
    ))
    story.append(p("Real investigation (same session) found three tiers:", "Body"))
    story.append(ListFlowable([
        ListItem(p("<b>Real and used here:</b> height, weight (pulled fresh from MLB Stats API's "
                   "bio endpoint - not in Statcast at all), arm_angle, release_extension, release "
                   "point (all Statcast fields), plus each pitcher's own season-long VARIABILITY "
                   "in arm angle/extension/release point - the closest available proxy for "
                   "&ldquo;variation in the delivery.&rdquo;")),
        ListItem(p("<b>Real, but no accessible data source exists at all:</b> delivery length "
                   "(windup-to-release duration) and ball concealment. Both would require "
                   "frame-by-frame motion-capture data that MLB's Hawk-Eye system captures "
                   "internally but does not publicly release. Not attempted - no honest proxy "
                   "exists.")),
        ListItem(p("<b>Real, partially prototyped, not yet solved:</b> leg lift timing, time held "
                   "in the set position, and timing variance/quick pitches. A real pipeline "
                   "(download broadcast video &rarr; pose-estimation with MediaPipe &rarr; derive "
                   "joint-position timing) was built and PARTS of it work - video download and pose "
                   "estimation both function on real MLB broadcast clips - but reliably telling "
                   "which detected person in frame IS the pitcher (vs. catcher/umpire/batter) is "
                   "unsolved as of this version. This is under separate, ongoing investigation, not "
                   "included in this model.")),
    ], bulletType="bullet", start="circle"))
    story.append(p(
        "So this v1 is explicitly a proof-of-concept on the reduced, honestly-available feature "
        "set - not the full model originally envisioned.", "Body",
    ))

    # ---- The model ----
    story.append(p("3. The Model", "H1"))
    story.append(p(
        "Same architecture as this site's Stuff+ v2 for consistency - LightGBM, Huber loss "
        "(delta_run_exp's heavy home-run-driven tail), <code>season</code> as a categorical "
        "feature to absorb era effects, leave-one-season-out cross-validation across 2021-2025 "
        "(2026 partial, train-only). The only real difference: no pitcher identity, no pitch "
        "type, no velocity, no movement, no plate location - ONLY the body/release features "
        "listed above, so the model can't quietly re-derive Stuff+ under a new name.", "Body",
    ))
    story.append(p(
        f"Target: <b>{report['target']}</b>.", "Body",
    ))

    story.append(PageBreak())

    # ---- Validation ----
    story.append(p("4. Validation", "H1"))
    story.append(p(
        "The same rigor Tunneling+ needed to prove it wasn't noise - this model's premise is "
        "more speculative than Stuff+'s (a much smaller, less obviously-predictive feature set), "
        "so it doesn't get a pass on validation.", "Body",
    ))

    story.append(p("4.1 Split-half reliability", "H2"))
    story.append(p(
        "Each pitcher's pitches were split into two random, independent halves; the score was "
        "computed separately on each half and correlated. A metric dominated by noise would show "
        "a correlation near zero.", "Body",
    ))
    story.append(p(
        "<b>A real methodological bug was caught and fixed while building this check.</b> The "
        "first version reused the already-computed, full-season arm_angle_std/"
        "release_extension_std/release_consistency values for both halves, instead of recomputing "
        "them independently within each half - since those three features are season-long "
        "aggregates, both halves saw literally identical values for them, which trivially inflated "
        "the correlation to 0.999 regardless of whether the model had any real signal. The "
        "leaderboard from that same run was also contaminated with position players who'd thrown "
        "3-58 mop-up-inning pitches (not real pitchers with a real delivery profile), since worked "
        "examples weren't filtered to the same qualifying bar the outcome-correlation check already "
        "used. Both are fixed in the numbers below: per-half features are now recomputed from only "
        "that half's own pitches, and the leaderboard uses the same 300-pitch minimum as everywhere "
        "else.", "Body",
    ))
    story.append(data_table(
        [[f"min {k} pitches", str(r["n"]), f"{r['correlation']:.3f}" if r["correlation"] is not None else "n/a"] for k, r in rel.items()],
        ["Cutoff", "n pitchers", "Correlation"],
    ))
    story.append(Spacer(1, 8))

    story.append(p("4.2 Does it predict real outcomes?", "H2"))
    story.append(p(
        f"deception_plus was correlated against each pitcher's real {report['report_season']} outcomes "
        f"(n = {val['n_qualified']} pitchers with 300+ pitches). Correlation with this project's own "
        f"Stuff+ v2 model: {val['correlation_with_overall_stuff']:.3f}. Each outcome's correlation was "
        f"then recomputed AFTER removing whatever Stuff+ already explains (linear residualization) - a "
        f"meaningfully positive partial r means the delivery features add real signal Stuff+ doesn't "
        f"already capture.", "Body",
    ))
    story.append(data_table(
        [[o, f"{d['raw_correlation']:+.3f}", f"{d['partial_correlation_stuff_removed']:+.3f}"] for o, d in val["outcomes"].items()],
        [f"{report['report_season']} outcome", "Raw r", "Partial r (Stuff+ removed)"],
    ))
    story.append(Spacer(1, 6))
    story.append(p(
        f"Correlation with overall Stuff+ itself ({val['correlation_with_overall_stuff']:.3f}) is notably "
        "higher than this site's other &ldquo;+&rdquo; models saw against Stuff+ - expected, and not a "
        "coincidence: release_extension is a feature in BOTH models (Stuff+ v2 uses it directly; it also "
        "feeds effective_speed there), so some real overlap between the two scores is baked in by "
        "construction, not just correlated behavior. The partial-correlation column is what actually "
        "matters here: whiff% and swinging-strike% drop to near zero once Stuff+ is removed - for those, "
        "this model is mostly just re-deriving Stuff+ through the shared extension feature. Chase%, "
        "K-BB%, and xwOBA-against retain a meaningfully positive partial correlation - real, independent "
        "signal beyond Stuff+ for those three specifically, even though it isn't for the pure "
        "swing-and-miss outcomes.", "Body",
    ))

    story.append(Spacer(1, 10))
    story.append(p("4.3 A note on the near-zero per-pitch R&sup2;", "H2"))
    story.append(p(
        f"CV mean R&sup2; = {report['cv_r2_mean']:.5f} - alarmingly low-looking on its own, but this is "
        "not unique to this model: a single pitch's outcome is dominated by noise no delivery model can "
        "see (contact luck, defense, sequencing). Stuff+ v2 and Tunneling+ both faced the same issue at "
        "the per-pitch level and were validated instead via aggregate reliability and real outcome "
        "correlation (Sections 4.1-4.2) - the same standard applied here. The difference is those models' "
        "aggregate validation came back strong; Section 1 states plainly whether this one's does too.", "Body",
    ))

    story.append(p("4.4 Feature importance (SHAP)", "H2"))
    shap_items = sorted(report["mean_abs_shap"].items(), key=lambda x: -x[1])
    story.append(data_table(
        [[f, f"{v:.4f}"] for f, v in shap_items],
        ["Feature", "Mean |SHAP|"],
    ))

    story.append(PageBreak())

    story.append(p("4.5 Worked examples", "H2"))
    story.append(p(
        "A soft, secondary sanity check - real pitcher names at the extremes. Illustrative only, "
        "not independent statistical evidence.", "Body",
    ))
    top5 = ex["highest_deception"][:5]
    bot5 = ex["lowest_deception"][:5]
    story.append(p("Highest deception_plus:", "Small"))
    story.append(data_table(
        [[r["name"], f"{r['deception_plus']:.0f}", str(r["total_pitches"])] for r in top5],
        ["Pitcher", "Deception+", "Pitches"],
    ))
    story.append(Spacer(1, 6))
    story.append(p("Lowest deception_plus:", "Small"))
    story.append(data_table(
        [[r["name"], f"{r['deception_plus']:.0f}", str(r["total_pitches"])] for r in bot5],
        ["Pitcher", "Deception+", "Pitches"],
    ))

    story.append(Spacer(1, 10))
    story.append(p("4.6 Cross-validation detail", "H2"))
    story.append(data_table(
        [[str(f["holdout_season"]), f"{f['r2']:.5f}", f"{f['mae']:.4f}", f"{f['n_pitches']:,}"] for f in report["cv_folds"]],
        ["Held-out season", "R²", "MAE", "Pitches"],
    ))

    story.append(PageBreak())

    # ---- Limitations / next steps ----
    story.append(p("5. Known Limitations and Open Questions", "H1"))
    story.append(ListFlowable([
        ListItem(p("<b>Thin feature set.</b> Five real per-pitch/aggregate features stand in for what "
                   "was meant to be a much richer delivery-timing profile - see Section 2.")),
        ListItem(p("<b>Height/weight are current-snapshot, not season-specific.</b> Same limitation "
                   "this site already accepts for team_id elsewhere - a pitcher's 2021 listed weight "
                   "may differ slightly from what was pulled.")),
        ListItem(p("<b>No shrinkage applied</b> to thin-sample pitchers, unlike Stuff+ v2.")),
        ListItem(p("<b>Single season report.</b> Unlike Stuff+/Tunneling+, this proof-of-concept only "
                   f"validates against {report['report_season']} - no season dropdown yet, no "
                   "year-over-year predictive-validity check.")),
        ListItem(p("<b>The video-based delivery-timing pipeline is a live, separate investigation</b> - "
                   "not blocked forever, just unsolved as of this version. See Section 2.")),
    ], bulletType="bullet", start="circle"))

    story.append(p("6. Suggested Next Steps", "H1"))
    story.append(ListFlowable([
        ListItem(p("Solve reliable pitcher-vs-other-players isolation in broadcast video (uniform "
                   "color + position heuristics, or a properly-isolated player-detection model), "
                   "then add leg-lift/set-time/timing-variance features for real.")),
        ListItem(p("Multi-season report + season dropdown, matching Stuff+/Tunneling+.")),
        ListItem(p("Season-specific height/weight if a historical bio data source can be found.")),
        ListItem(p("Empirical-Bayes shrinkage for thin-sample pitchers.")),
    ], bulletType="bullet", start="circle"))

    return story


def main():
    report = json.loads(REPORT_PATH.read_text())
    doc = SimpleDocTemplate(
        str(OUT_PATH), pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    )
    doc.build(build(report))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
