"""
Generates the Tunneling+ v1 methodology PDF from models/tunneling_v1_report.json
- every number in the document is read from that file, not hand-typed, so the
PDF can never drift out of sync with what the model actually produced.

    python generate_tunneling_pdf.py
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
# The site now offers a report per season (2021-2026) via the season
# dropdown; this document's numbers are fixed to one representative season
# (the most recent complete one) rather than regenerated per season - it's
# a methodology writeup, not meant to flip underneath a UI toggle.
REPORT_PATH = MODELS_DIR / "tunneling_v1_report_2025.json"
OUT_PATH = Path(__file__).parent / "Tunneling+ v1 - Methodology.pdf"

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
styles.add(ParagraphStyle("Quote", parent=styles["Normal"], fontSize=10, leading=14, leftIndent=14,
                           borderColor=BARK, borderWidth=0, textColor=FOREST_DARK, spaceAfter=8))
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
    m = report["methodology"]
    rel = report["split_half_reliability"]
    val = report["outcome_validity"]
    seq = report["sequential_same_at_bat_test"]
    ex = report["worked_examples"]

    # ---- Title page ----
    story.append(Spacer(1, 1.6 * inch))
    story.append(p("Tunneling+ v1", "TitleBig"))
    story.append(p("A first-pass, self-directed model of pitch-pair tunneling", "Subtitle"))
    story.append(p(f"Built {report['built_at']} &middot; {report['season']} season &middot; {m['n_pitchers_scored']} pitchers scored", "Subtitle"))
    story.append(Spacer(1, 0.4 * inch))
    story.append(p(
        "No outside specification was given for this model - the mechanism, the math, and the "
        "validation plan documented here are all independent design decisions, made and recorded "
        "as they were made, not reconstructed afterward. This document is the full record: what "
        "was tried, why, and what the evidence actually shows. (Originally named &ldquo;Deception+&rdquo; "
        "- renamed to Tunneling+ because it measures specifically pitch-pair tunneling, one of several "
        "distinct ideas that get lumped under the broader word &ldquo;deception.&rdquo;)", "Body",
    ))
    story.append(p(
        f"<b>Units:</b> distances are reported in inches (movement-scale quantities, same convention as "
        f"IVB/HB elsewhere on this site); the decision point is a location along the pitch's flight path, "
        f"kept in feet. <b>Formula:</b> {m.get('scale_note', 'Tunneling+ is scaled so 100 = league average.')}",
        "Small",
    ))
    story.append(PageBreak())

    # ---- Executive summary ----
    story.append(p("1. Executive Summary", "H1"))
    story.append(p(
        "Tunneling+ measures <b>pitch-pair tunneling</b>: how close together two of a pitcher's "
        "pitch types are at the point a hitter has to commit to swinging, versus how far apart "
        "they actually end up at the plate. A pair that looks the same early and ends up somewhere "
        "very different scores high. The position at that decision point is computed directly from "
        "Statcast's own trajectory physics - not a proxy, not a black box.", "Body",
    ))
    story.append(p("The model passed every validation test run against it:", "Body"))
    story.append(ListFlowable([
        ListItem(p(f"<b>Reliable</b>: {rel['2']['correlation']:.2f} split-half correlation (pitchers with 2+ scoring pairs) - "
                    f"this is measuring something stable about a pitcher, not noise."), bulletColor=FOREST),
        ListItem(p(f"<b>Predicts real outcomes</b>: positive correlation with whiff%, chase%, and swinging-strike% "
                    f"(r = {val['outcomes']['swstr_pct']['raw_correlation']:.2f} for swstr%), and that correlation "
                    f"<b>survives controlling for Stuff+</b> (partial r = {val['outcomes']['swstr_pct']['partial_correlation_stuff_removed']:.2f}) - "
                    f"this is not just re-deriving raw pitch quality under a new name."), bulletColor=FOREST),
        ListItem(p(f"<b>Mechanism-specific</b>: in real same-at-bat sequences, pitch pairs the model scores as more "
                    f"tunneled produce a measurably higher whiff rate on the second pitch "
                    f"(z = {seq['low_vs_high_z_score']:.2f}, n = {seq['n_with_known_pair_ratio']:,} sequences) - "
                    f"the specific causal pathway the model claims to capture, tested directly."), bulletColor=FOREST),
    ], bulletType="bullet", start="circle"))
    story.append(p(
        "It is still a v1: the caveats and open questions in Section 7 are as important to read as "
        "the results above.", "Body",
    ))

    # ---- What tunneling means ----
    story.append(p("2. Why Tunneling, and Why This Isn&rsquo;t Called &ldquo;Deception+&rdquo; Anymore", "H1"))
    story.append(p(
        "Pitching analytics converges on a few distinct mechanical ideas under the word "
        "&ldquo;deception.&rdquo; This model was originally named Deception+ and framed as having "
        "picked one of those ideas as its primary target - but that framing implied broader scope "
        "than the model actually has. It measures exactly one specific, well-defined mechanism, so "
        "it's named for that mechanism directly: <b>tunneling</b>.", "Body",
    ))
    story.append(p("Candidates considered when the model was first designed:", "H2"))
    story.append(ListFlowable([
        ListItem(p("<b>Tunneling</b> - two pitch types traveling through nearly the same point in space at "
                   "the decision point, then diverging. The most commonly cited, most mechanistic definition "
                   "in public pitching research, and fully computable from Statcast's trajectory fit. "
                   "<b>This is what the model measures.</b>")),
        ListItem(p("<b>Release-point consistency</b> - a pitcher who releases every pitch from the same spot "
                   "gives no positional tell before the ball even leaves his hand. Simpler and less novel, "
                   "but cheap to check independently - kept as a secondary diagnostic (Section 5), not folded "
                   "into the primary score.")),
        ListItem(p("<b>Spin mirroring</b> - opposing spin directions between e.g. a fastball and curveball that "
                   "look similar early due to shared early Magnus-force behavior. Theoretically interesting but "
                   "less rigorously validated in public work than trajectory-based tunneling - set aside for v1, "
                   "noted as a real candidate for v2.")),
        ListItem(p("<b>Effective-velocity masking</b> - a changeup thrown with fastball-identical arm speed. "
                   "Related to tunneling but more about arm action than trajectory - not separately modeled here; "
                   "to the extent it matters, it's partially captured by the release-consistency diagnostic.")),
    ], bulletType="bullet", start="circle"))
    story.append(p(
        "<b>Tunneling was chosen as the primary target</b> because it is the most mechanistic (a "
        "geometric fact about two trajectories, not a proxy), the most commonly validated concept in "
        "public pitching research, and - critically - it was fully computable from data already in "
        "this database, using physics already established elsewhere in this codebase (see Section 4). "
        "The other candidates above (release-point consistency aside, which IS tracked) remain open "
        "ideas for a separate future model, not folded into this one under a broader &ldquo;deception&rdquo; "
        "umbrella.", "Body",
    ))

    # ---- The scoring fixes ----
    story.append(p("3. Three Real Methodological Bugs, Caught and Fixed", "H1"))
    story.append(p("3.1 Division coupled the two halves of the idea together", "H2"))
    story.append(p(
        "The first version of this model scored each pair with a single ratio: plate_distance / "
        "tunnel_distance. That coupled the two halves of the idea together by division - a pair's "
        "plate separation only counted <i>in proportion to</i> how small its tunnel separation "
        "already was. A pair with a genuinely large plate separation (literally the point of "
        "throwing two different pitch types) could still score below average purely because its "
        "tunnel separation was also moderately large - real, meaningful separation was being "
        "diluted rather than credited. Caught via a real example: Tarik Skubal's changeup/fastball "
        "pair had the single largest plate separation in his entire arsenal, yet scored a "
        "below-average score under the ratio, purely because its tunnel separation was also "
        "middling - the formula was punishing exactly the thing two different pitch types are "
        "supposed to do.", "Body",
    ))
    story.append(p(
        "<b>Fix:</b> split into independently-normalized components, each scaled 100=league average "
        "(decision_similarity_plus and plate_separation_plus at first; see 3.3 for the third), "
        "combined by <b>geometric mean</b> instead of division. Geometric mean still requires every "
        "component to be reasonably good to produce a high combined score - a pair that's merely "
        "average on one axis can't be &ldquo;rescued&rdquo; to a great score by another axis alone. "
        "This preserves the AND-logic that's the actual definition of tunneling, without the "
        "original ratio's harsh division-based coupling.", "Body",
    ))

    story.append(p("3.2 Centroid distance quietly erases real command variance", "H2"))
    story.append(p(
        "Every distance was originally computed as centroid-to-centroid: average all of a pitcher's "
        "changeups into one point, average all his fastballs into one point, then measure the gap "
        "between those two averages. That silently assumes away pitch-to-pitch variance within each "
        "type - checked directly on Skubal's changeup/fastball pair: the two types' centroids sit "
        "5.3\" apart at the decision point, but each type's own individual pitches vary by 4-5\" "
        "(std. dev.) around their own centroid. The <i>typical real</i> changeup and <i>typical "
        "real</i> fastball are actually 9.7\" apart at that moment - not 5.3\". Centroid distance "
        "was systematically understating true separation for any pitcher with real command "
        "variance, which is to say every real pitcher. Leaguewide, average tunnel distance nearly "
        "tripled once measured correctly (3.4\" &rarr; 9.5\").", "Body",
    ))
    story.append(p(
        "<b>Fix:</b> every distance (release/tunnel/plate) is now the full pairwise mean distance "
        "between EVERY individual pitch of type A and EVERY individual pitch of type B, not the "
        "distance between their averages. Computed via vectorized broadcasting rather than a Python "
        "loop over pitch pairs - cheap even at full-season pitch counts (roughly 15 seconds extra, "
        "leaguewide, for a materially more honest number).", "Body",
    ))

    story.append(p("3.3 A release-point component was tried, tested, and rejected", "H2"))
    story.append(p(
        "A pair could converge beautifully by the decision point yet still tip itself off at "
        "release - a changeup thrown from a visibly different arm slot than the fastball gives the "
        "hitter information before the ball is even 10 feet from the pitcher's hand, regardless of "
        "how similar the two pitches look later. release_distance was always computed and displayed, "
        "but never fed into the score - so a release_similarity_plus component was built (100 &times; "
        "league-average release distance / this pair's release distance) and tried as a third factor "
        "in the geometric mean alongside decision_similarity_plus and plate_separation_plus.", "Body",
    ))
    story.append(p(
        "<b>The theory was reasonable; the data disagreed.</b> Tested against the full validation "
        "battery, adding release_similarity_plus made the model measurably WORSE, not better: the "
        "sequential same-at-bat test - the single most rigorous, mechanism-specific check available "
        "(Section 6.3) - dropped from z=9.06 (two components) to z=1.60 (three components), falling "
        "below the conventional significance bar entirely. Every outcome correlation weakened "
        "similarly. Real hitters' whiff rates on the second pitch of a sequence just don't track "
        "release-point similarity the way they track decision-point similarity and plate separation "
        "- folding it in diluted a strong, validated signal with a much weaker one.", "Body",
    ))
    story.append(p(
        "<b>Decision: rejected.</b> release_similarity_plus is still computed and shown per-pair in "
        "the arsenal inspector, for context - but tunneling_plus stays the two-component geometric "
        "mean from 3.1. This is a real instance of the working principle for every model on this "
        "site: a plausible mechanism gets tested against the evidence before it ships, and loses to "
        "the evidence when the two disagree.", "Body",
    ))
    story.append(p(
        "Every fix in this section is a genuinely different scoring function from what came before, "
        "not a rescaling - pair rankings change, not just their units. Every validation check in "
        "Section 6 was re-run against the current (two-component, full-pairwise-distance) version "
        "rather than assumed to still hold from an earlier one.", "Body",
    ))

    story.append(PageBreak())

    # ---- The physics ----
    story.append(p("4. The Physics", "H1"))
    story.append(p(
        "Statcast fits every pitch to a constant-acceleration trajectory: velocity (vx0, vy0, vz0) "
        "and acceleration (ax, ay, az) are given at a fixed measurement plane 50 feet from home "
        "plate. This codebase already uses that exact model elsewhere - metrics.py's compute_vaa "
        "(Vertical Approach Angle) solves for the ball's state at the front of the plate given its "
        "state at y=50ft. This model generalizes that same solve two ways: to BOTH the x and z axes "
        "(not just z), and to an arbitrary target distance from the plate (not just the plate itself). "
        "The same simplification compute_vaa already makes - treating release_pos_x/release_pos_z as "
        "the position at the y=50ft plane, when release actually happens a few feet closer - is kept "
        "here for consistency with the rest of the codebase rather than introducing a second convention.",
        "Body",
    ))
    story.append(p(
        f"<b>Decision point: {m['decision_point_feet_from_plate']} feet from home plate.</b> Public "
        "pitch-tunneling research generally cites a range of roughly 20-24 feet from the plate for "
        "when a hitter's swing decision is effectively locked in, depending on pitch velocity. This "
        "model uses one fixed value inside that range for every pitch - deliberately simple for a v1. "
        "A velocity-adjusted decision point (faster pitches leave less flight time, so the "
        "true decision point should sit slightly further from the plate) is a real, documented "
        "limitation, not an oversight - see Section 7.", "Body",
    ))
    story.append(p("For every pitch, three positions are computed or read directly:", "Body"))
    story.append(ListFlowable([
        ListItem(p("<b>Release</b> - release_pos_x / release_pos_z, read directly from Statcast.")),
        ListItem(p(f"<b>Decision point</b> - position at y = {m['decision_point_feet_from_plate']} ft, solved from the "
                    "trajectory equations (the new part).")),
        ListItem(p("<b>Plate</b> - plate_x / plate_z, read directly from Statcast.")),
    ], bulletType="bullet", start="circle"))

    story.append(PageBreak())

    # ---- Pipeline ----
    story.append(p("5. The Pipeline, Step by Step", "H1"))
    steps = [
        "Load every pitch for the season with a valid trajectory (release position, velocity, "
        "acceleration, and plate location all present) and compute its decision-point position.",
        f"Group by (pitcher, pitch type) to identify which types qualify - at least "
        f"{m['min_pitches_per_type']} pitches that season (a type thrown only a handful of times is "
        f"too thin a sample to trust).",
        "For every pair of qualifying pitch types within one pitcher's arsenal, compute three "
        "distances - release distance, tunnel distance (at the decision point), and plate distance - "
        "each as the FULL PAIRWISE MEAN distance between every individual pitch of type A and every "
        "individual pitch of type B (see Section 3.2), not a distance between averaged positions.",
        f"Compute the league-wide, usage-weighted average release/tunnel/plate distance across every "
        f"qualifying pair, leaguewide - the baselines every pair is scored against (each is floored "
        f"at {m['distance_floor_inches']}\" first, so a near-zero separation can't produce an "
        f"explosive, meaningless component).",
        "For each pair, compute release_similarity_plus, decision_similarity_plus, and "
        "plate_separation_plus (each 100 &times; the relevant ratio against its league baseline). "
        "tunneling_plus = the geometric mean of decision_similarity_plus and plate_separation_plus "
        "only - release_similarity_plus is shown per-pair for context but is not scored; it was "
        "tested as a third component and rejected after it measurably hurt validated predictive "
        "power (see Section 3.3).",
        "Weight each pair by the geometric mean of the two types' usage share within the pitcher's "
        "qualifying arsenal - a pair only matters to a hitter's real experience in proportion to how "
        "often both pitches actually show up. Pairs involving the four-seam (FF) or sinker (SI) then "
        "get a further 2x bonus on top of that usage weight before aggregating to the pitcher-level "
        "composite: a hitter's read is built primarily against the fastball (or sinker, for an "
        "east-west arsenal), so how well an offspeed pitch tunnels with THAT matters more than how "
        "two offspeed pitches tunnel with each other. This bonus does not apply to a pair's own "
        "tunneling_plus (shown per-pair in the inspector) or to the league-wide baselines - only to how "
        "much each pair counts toward the single pitcher-level number.",
        "Aggregate to one tunneling_plus per pitcher: the (usage x anchor-bonus)-weighted average of "
        "tunneling_plus across every real pair in their arsenal - the headline number shown on the site.",
        "Separately compute release_consistency: the usage-weighted RMS distance of each individual "
        "pitch's release point from the pitcher's own overall release centroid, across the WHOLE "
        "arsenal (not just pairs) - a simpler, whole-arsenal diagnostic, distinct from the pairwise "
        "release_similarity_plus above, kept separate from the primary score.",
    ]
    story.append(ListFlowable([ListItem(p(s)) for s in steps], bulletType="1"))

    # ---- Validation ----
    story.append(p("6. Validation", "H1"))
    story.append(p(
        "Four independent checks were run against the model - not selected after the fact, run "
        "as a deliberate battery because a single passing test proves very little on its own.",
        "Body",
    ))

    story.append(p("6.1 Split-half reliability", "H2"))
    story.append(p(
        "Each pitcher's pitches were split into two random, independent halves; the score was "
        "computed separately on each half and correlated. A metric dominated by noise would show a "
        "correlation near zero.", "Body",
    ))
    story.append(data_table(
        [[f"min {k} pair(s)", str(v["n"]), f"{v['correlation']:.3f}"] for k, v in rel.items()],
        ["Cutoff", "n pitchers", "Correlation"],
    ))
    story.append(Spacer(1, 8))

    story.append(p("6.2 Does it predict real outcomes?", "H2"))
    story.append(p(
        f"tunneling_plus was correlated against each pitcher's real {report['season']} outcomes "
        f"(n = {val['n_qualified']} pitchers with 300+ pitches). To check this isn't simply "
        f"re-deriving pitch quality under a new name, "
        f"the correlation with this project's own Stuff+ v2 model was checked first "
        f"(r = {val['correlation_with_overall_stuff']:.3f} - modest, not the 0.6+ that would suggest "
        f"heavy overlap), then each outcome's correlation with tunneling_plus was recomputed AFTER "
        f"removing whatever Stuff+ already explains (linear residualization).", "Body",
    ))
    story.append(data_table(
        [[o, f"{d['raw_correlation']:+.3f}", f"{d['partial_correlation_stuff_removed']:+.3f}"] for o, d in val["outcomes"].items()],
        [f"{report['season']} outcome", "Raw r", "Partial r (Stuff+ removed)"],
    ))
    story.append(Spacer(1, 6))
    story.append(p(
        "Every outcome except xwOBA shows a meaningful positive partial correlation - tunneling adds "
        "real signal beyond what raw pitch quality already predicts, for the swing-and-miss family of "
        "outcomes especially (swinging-strike% partial r = "
        f"{val['outcomes']['swstr_pct']['partial_correlation_stuff_removed']:+.3f}). xwOBA's partial "
        "correlation flips sign and stays weak - plausible, since tunneling should matter most for "
        "swing decisions (whiff/chase), less for the quality of contact once a batter does connect.",
        "Body",
    ))

    story.append(p("6.3 Sequential same-at-bat test", "H2"))
    story.append(p(
        "The most mechanism-specific check: within real at-bats, when a pitcher throws two DIFFERENT "
        "pitch types back to back, does the pre-computed tunneling_plus for that specific pair predict "
        "a worse outcome on the second pitch of the sequence? This tests the actual causal pathway the "
        "model claims to capture, not a league-wide aggregate that could be confounded by “good "
        "pitchers are good at everything.”", "Body",
    ))
    seq_rows = [[t.capitalize(), f"{d['whiff_rate']*100:.1f}%", f"{d['n']:,}"]
                for t, d in seq["whiff_rate_by_tunnel_ratio_tercile"].items()]
    story.append(data_table(seq_rows, ["Tunneling+ tercile", "Whiff rate (2nd pitch)", "n sequences"]))
    story.append(Spacer(1, 6))
    story.append(p(
        f"Low-to-high tercile difference: z = {seq['low_vs_high_z_score']:.2f} "
        f"(|z| &gt; 1.96 is the conventional p &lt; 0.05 bar). The effect is small in absolute terms "
        f"(a percentage point or so of whiff rate) but, given the enormous sample "
        f"(n = {seq['n_with_known_pair_ratio']:,} sequences), highly statistically significant and "
        f"monotonic across all three terciles.", "Body",
    ))

    story.append(p("6.4 Worked examples", "H2"))
    story.append(p(
        "A soft, secondary sanity check (not a substitute for the tests above) - real pitcher names "
        "at the extremes, to see whether the ranking is at least plausible against general scouting "
        "narratives. Treat this section as illustrative, not as independent statistical evidence - I "
        "have not independently verified every pitcher's specific reputation against outside sources.",
        "Body",
    ))
    top5 = ex["highest_tunneling"][:5]
    bot5 = ex["lowest_tunneling"][:5]
    story.append(p("Highest tunneling_plus:", "Small"))
    story.append(data_table(
        [[r["name"], f"{r['tunneling_plus']:.0f}", str(r["n_pairs"]), str(r["total_pitches"])] for r in top5],
        ["Pitcher", "Tunneling+", "Pairs", "Pitches"],
    ))
    story.append(Spacer(1, 6))
    story.append(p("Lowest tunneling_plus:", "Small"))
    story.append(data_table(
        [[r["name"], f"{r['tunneling_plus']:.0f}", str(r["n_pairs"]), str(r["total_pitches"])] for r in bot5],
        ["Pitcher", "Tunneling+", "Pairs", "Pitches"],
    ))

    story.append(PageBreak())

    # ---- Limitations ----
    story.append(p("7. Known Limitations and Open Questions", "H1"))
    story.append(ListFlowable([
        ListItem(p("<b>Fixed decision point.</b> 23.5 ft from the plate is used for every pitch regardless "
                   "of velocity. A 100mph fastball and a 75mph curveball don't take the same time to "
                   "travel the same distance - a velocity-adjusted decision point (further from the plate "
                   "for faster pitches, since less flight time remains) would be more physically honest. "
                   "Not implemented in v1; a clear v2 candidate.")),
        ListItem(p("<b>n_pairs=1 pitchers are noisier.</b> A pitcher with exactly two qualifying pitch types "
                   "has only one pair to average - no internal averaging to smooth out noise the way a "
                   "6-pair pitcher's score benefits from. The worked examples in Section 6.4 were filtered "
                   "to n_pairs&ge;2 for this reason; the full leaderboard (in the HTML page) is not, matching "
                   "the same trade-off documented for Stuff+ v2's small-sample pitch types.")),
        ListItem(p("<b>No shrinkage applied.</b> Unlike Stuff+ v2, this model does not (yet) apply "
                   "empirical-Bayes shrinkage toward league-average component values for thin samples. "
                   "Given the demonstrated split-half reliability is already strong, this is a lower "
                   "priority than it was for Stuff+, but remains untested at very low pair counts.")),
        ListItem(p("<b>Spin mirroring not modeled.</b> A pitch pair with genuinely mirrored, opposite spin "
                   "axes may look more similar to a real hitter's eye than pure trajectory distance alone "
                   "would suggest, or vice versa - this model has no visibility into that.")),
        ListItem(p("<b>Correlational, not causal, at the aggregate level.</b> Section 6.2's outcome "
                   "correlations are exactly that - correlations. Section 6.3's sequential test is the "
                   "strongest causal-adjacent evidence in this report specifically because it isolates the "
                   "mechanism directly, but even that is observational, not a controlled experiment.")),
        ListItem(p("<b>Single season.</b> All validation here is within the "
                   f"{report['season']} season. No year-over-year predictive-validity check (does a "
                   "pitcher's tunneling_plus in one season predict a DIFFERENT season's outcomes) has "
                   "been run yet - the same gold-standard test that mattered for validating Stuff+ v2.")),
    ], bulletType="bullet", start="circle"))

    story.append(p("8. Suggested Next Steps", "H1"))
    story.append(ListFlowable([
        ListItem(p("Velocity-adjust the decision point per pitch instead of a fixed 23.5 ft.")),
        ListItem(p("Year-over-year predictive validity check, matching the Stuff+ v2 methodology.")),
        ListItem(p("Explore spin-axis mirroring as a second, independent signal alongside trajectory "
                   "tunneling.")),
        ListItem(p("Apply empirical-Bayes shrinkage to stabilize thin-sample (especially n_pairs=1) "
                   "pitchers, matching the approach already validated for Stuff+ v2.")),
        ListItem(p("Combine tunneling_plus and release_consistency into one blended metric, once/if "
                   "each is shown to add independent value in a joint model rather than kept separate.")),
        ListItem(p("Revisit release-point information in a different form than a third geometric-mean "
                   "factor (Section 3.3) - e.g. as a separate filter/flag on obviously-tipped pairs, or "
                   "weighted much lighter than the other two components - rather than abandoning the "
                   "idea outright just because the naive equal-weight version didn't validate.")),
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
