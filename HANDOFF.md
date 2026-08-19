# MLB Dashboard — Session Handoff (v4)

Context for continuing work on this project in a new conversation. Supersedes v3 for everything Deception+-v2
/ video-timing related — v3 covered the Tunneling+ rename/redesign (still true, not repeated here) and a
first-pass Deception+ v1 (still true as a MODEL, not repeated here, but its planned v2 feature set and the
video R&D behind it are substantially updated below — v3's "planned metrics" section is now largely built and
validated, not just planned). **This handoff covers a fourth, separate session** whose entire focus was
actually building and validating the video-based delivery-timing extraction pipeline that v3 only prototyped
on one pitcher — read the new "Deception+ v2 delivery-timing pipeline" section below in full before touching
any of this code; it replaces v3's "Video-based delivery-timing extraction" section wholesale, not just adds
to it. Several real methodology bugs were found and fixed along the way (documented below with what broke,
how it was caught, and the fix) — this is NOT a clean first-try build, it's the record of a real debugging
process, and the "known open issues" at the end are genuinely unresolved, not just disclaimers.

## Where everything lives

Project root: `C:\Users\joshs\Desktop\mlb_dashboard`. No git repo still — offer again if it comes up, don't nag.

- **Data**: `mlb_data.db` (SQLite). **New this session**: `pitcher_bio` table (`player_id`, `height_in`,
  `weight_lb`, `pulled_at`) — pulled from MLB Stats API's `/people/{id}` bio endpoint via `pull_pitcher_bio.py`
  (2948/2948 pitchers resolved), since height/weight aren't in Statcast at all. Current-listed bio only, not
  season-specific (same limitation `players.team_id` already carries).
- **`models/` — Tunneling+ files, now per-season** (the season dropdown, new this session, needs one pair of
  files per year): `tunneling_v1_report_{season}.json` / `tunneling_v1_pitchers_{season}.json` for each of
  2021-2026. The OLD flat `tunneling_v1_report.json`/`_pitchers.json` (no season suffix) were deleted — if you
  see code or a stale API call referencing those, it's wrong, fix it to the per-season path.
- **`models/` — Deception+ files, new this session, single-season only (2025)**: `deception_plus_v1.txt` (the
  trained LightGBM booster, human-readable native text dump), `deception_plus_v1_meta.json`,
  `deception_plus_v1_report_2025.json`. No season dropdown yet for this one — see Deception+ section.
- **Renamed this session**: `deception_model.py` → `tunneling_model.py`, `build_deception_report.py` →
  `build_tunneling_report.py`, `api/routers/deception_report.py` → `tunneling_report.py`,
  `frontend/deception-model.html`/`js/deception-model.js` → `tunneling-model.html`/`js/tunneling-model.js`.
  Old deception-named files were deleted outright, not kept as aliases.
- **New this session — the real Deception+**: `deception_features.py` (shared feature engineering, train/serve
  parity, same pattern as `stuff_features.py`), `train_deception_model.py` (offline training + full validation
  battery — mirrors `train_stuff_model.py`'s architecture), `deception_model.py` (**new file, NOT the old
  renamed-away one** — this is the live-scoring runtime counterpart, mirrors `stuff_model.py`'s split from
  `train_stuff_model.py`), `pull_pitcher_bio.py`, `generate_deception_pdf.py` → `Deception+ v1 - Methodology.pdf`
  (delivered to user, several times, as the model evolved — regenerate via `python generate_deception_pdf.py`
  after re-running `train_deception_model.py`).
- **New API files**: `api/routers/deception_report.py` (this is a NEW file with this name — not to be confused
  with the OLD file of the same name that got renamed away to `tunneling_report.py` earlier in the session;
  same filename, completely different model underneath).
- **New frontend pages**: `deception-model.html`/`js/deception-model.js` (this is a NEW file with this name,
  same "reused filename, different model" situation as the API router above).
- **`generate_tunneling_pdf.py`** was edited repeatedly this session as the Tunneling+ formula changed — always
  regenerate the PDF after any `build_tunneling_report.py` rerun, it drifts out of sync otherwise.

## Running it

```bash
python -m uvicorn api.main:app --port 8010
```
Unchanged — port 8010, no `--reload`, kill+relaunch after any backend `.py` edit. Still auto-refreshes on
startup (see v2 handoff for the scheduler design, unchanged this session).

## Tunneling+ — renamed from Deception+, then substantially redesigned (multiple real bugs found and fixed)

The user clarified early this session that "deception" was always meant to be about delivery/windup (see the
brand-new Deception+ section below for that model) — the OLD "Deception+ v1" only ever measured pitch-pair
**tunneling** specifically, so it was renamed to say exactly what it measures. The rename triggered a long
chain of real, evidence-based redesigns — **not just cosmetic**, several were genuine bugs caught by the user
asking pointed questions and demanding real numbers before trusting a change. Read `tunneling_model.py`'s
module docstring for the full reasoning; summary of what changed, roughly in order:

1. **Units**: distances switched from feet to inches (movement-scale quantities, matches IVB/HB convention
   elsewhere on the site). Decision point (23.5 ft from plate) deliberately STAYED in feet — it's a flight-path
   location, not a movement magnitude.
2. **Score rescaled to 100=league-average** (`tunneling_plus`), matching the Stuff+/wRC+ "+" convention site-wide
   — the original score was a raw, unbounded ratio with no fixed reference point, which the user correctly
   found unreadable.
3. **Real bug #1 — the original ratio formula punished exactly what tunneling is supposed to reward.**
   `plate_distance / tunnel_distance` coupled the two halves by division: a pair with a genuinely huge plate
   separation could still score below average just because its tunnel separation was also middling. Fixed by
   splitting into `decision_similarity_plus` and `plate_separation_plus` (each independently scaled
   100=league-avg) combined by **geometric mean**, not division. Caught via a real example (Tarik Skubal's
   CH↔FF, his single largest plate separation, scored below-average under the old ratio).
4. **Real bug #2 — centroid-to-centroid distance quietly erased real command variance.** Averaging all of a
   pitcher's changeups into one point, averaging all fastballs into another, then measuring ONE distance
   between the averages, silently assumes away pitch-to-pitch spread. Checked on real data: centroid distance
   said Skubal's CH/FF were 5.3" apart at the decision point; the FULL PAIRWISE mean distance (every real
   changeup vs. every real fastball) said 9.7". Leaguewide, average tunnel distance nearly tripled once measured
   correctly. Fixed: every distance is now `_pairwise_mean_dist()` — vectorized broadcasting, ~15s extra per
   season, not a meaningful cost for a materially more honest number. **This was an unambiguous improvement**
   across every validation metric.
5. **A third component (`release_similarity_plus`) was tried and REJECTED** — the user asked why release-point
   distance wasn't scored (a pair could tip itself off at release even if its later trajectory converges). Real
   idea, tested honestly: folding it into a 3-way geometric mean was tested against the FULL validation battery
   and made the model measurably WORSE — the sequential same-at-bat test (the single most rigorous check)
   dropped from z=9.06 (2-way) to z=1.60 (3-way, below the significance bar). Real hitters' whiff rates just
   don't track release-point similarity the way they track decision-point similarity and plate separation.
   **Kept as a displayed, unscored diagnostic** (shown per-pair in the arsenal table) — same role
   `release_consistency` already plays. This is documented in the PDF as a real instance of "a plausible
   mechanism gets tested against evidence and loses" — don't silently re-add it without re-validating.
6. **Fastball-anchor weighting**: pairs involving FF or SI count 2x (usage-weighted) toward the pitcher-level
   composite score (not toward the league baselines, and not toward a pair's own displayed score) — a hitter's
   read is built primarily against the fastball/sinker, so an offspeed pitch's tunnel with THAT matters more
   than two offspeed pitches tunneling with each other. Chosen 2x over 3x/anchor-only after showing the user
   real before/after numbers for a deep-arsenal pitcher (Justin Verlander).
7. **Season dropdown added** — `build_tunneling_report.py` now loops over every season with data (or one via
   CLI arg), writing season-suffixed files. Unlike Stuff+'s season selector (which scores live per request,
   since it's just model inference), Tunneling+'s validation battery is genuinely expensive (~85s/season with
   the full-pairwise fix) so it stays fully precomputed offline — changing the season on the page reloads the
   ENTIRE report, not just a per-pitcher panel. All 6 seasons (2021-2026) regenerated.
8. **`composite_weight_fn` bug fix**: split-half reliability was computed by reusing already-engineered,
   full-season aggregate features for BOTH random halves — since those are season-long aggregates, both halves
   saw identical values, trivially inflating the correlation to 0.999 regardless of real signal. Fixed by
   recomputing per-half from only that half's own raw pitches. (This exact class of bug recurred almost
   identically in Deception+ later the same session — see below. Watch for it in any future split-half check
   that touches season-aggregate features.)

**Final validated state**: split-half reliability ~0.85-0.91, sequential same-at-bat test z≈9 (very strong),
outcome correlations all improved over the original ratio version. Live at `tunneling-model.html` with a
compact arsenal table (Pair | N | Release | Decision | Plate | Decision Sim+ | Plate Sep+ | Tunneling+,
anchor-pair rows shaded), season dropdown, and per-pitcher search.

## Deception+ v1 — brand new this session, the REAL delivery/mechanics model

The freed-up name is now what it always should have meant: **pure pitching mechanics, body, and delivery
characteristics — deliberately independent of pitch trajectory/movement (that's Stuff+'s job) and of pitch-pair
relationships (that's Tunneling+'s job)**. This is the user's stated long-term architecture — see "Planned
future architecture" below, not yet executed.

### What's actually in v1 (a scoped-down proof of concept — read this before assuming more exists)

Real data-availability investigation was done BEFORE building anything (this session's established
pattern — verify, don't assume):
- **Real and included**: `arm_angle`, `release_extension` (both already in `pitches`, 90-97% coverage),
  release point, `height_in`/`weight_lb` (new pull), plus each pitcher's own season-long VARIABILITY in
  arm_angle/extension/release-point (`arm_angle_std`, `release_extension_std`, `release_consistency`).
- **Confirmed NOT obtainable from any accessible source**: delivery length (windup-to-release duration), ball
  concealment. Both would need frame-by-frame motion-capture data MLB doesn't publicly release. Documented as
  an honest gap in the PDF, not faked with a proxy.
- **Real, now SOLVED as of this session (see the video R&D section below) but NOT YET integrated into the
  model**: leg-lift timing, time held in set, delivery length, timing variance. The pipeline to extract these
  now works and was validated on one real pitcher — it just hasn't been run at scale or wired into
  `deception_features.py` yet.

Architecture mirrors `stuff_model.py`/`train_stuff_model.py`'s split exactly: `deception_features.py` (shared
feature engineering), `train_deception_model.py` (LightGBM, Huber loss, leave-one-season-out CV 2021-2025,
`season` as a categorical feature), `deception_model.py` (live-scoring runtime — loads the saved booster, scores
one pitcher's pitches on demand for the per-pitcher SHAP inspector).

### A real validation bug, caught before trusting the first run (same pattern as Tunneling+'s bug #8 above)

First run: split-half reliability 0.999 (suspiciously perfect) and a leaderboard topped by POSITION PLAYERS
with 3-58 mop-up-inning pitches (Garrett Hampson, several catchers/infielders) — not real pitcher signal.
Root cause: `arm_angle_std`/`release_extension_std`/`release_consistency` are season-long aggregates computed
BEFORE the random split-half, so both halves shared identical values, trivially inflating the correlation.
Fixed by recomputing those aggregates independently within each half (`engineer_features` called separately
per half's own raw rows) and filtering worked-examples to the same 300-pitch bar the outcome-correlation check
already used. **After the fix**: reliability 0.84-0.88 (real), leaderboard topped by real, sensible pitchers —
**Tyler Rogers #1**, matching his known extreme submarine delivery.

### Honest verdict — mixed, and the verdict logic actually reflects it

Correlation with overall Stuff+ itself: 0.544 (notably higher than Tunneling+'s ~0.1-0.2) — **not a bug**,
`release_extension`/`arm_angle`/`release_pos_x_as`/`release_pos_z` are LITERALLY shared raw input features with
Stuff+ v2 (see `stuff_features.py`'s `NUMERIC_FEATURES`), so real overlap is baked in by construction. Partial
correlations (Stuff+ removed) tell the real story: whiff_pct (0.038) and swstr_pct (0.097) wash out almost
entirely — for pure swing-and-miss, this model is mostly re-deriving Stuff+ through the shared extension
feature. But chase_pct (0.160), k_minus_bb_pct (0.307), and xwoba (-0.386) retain real, independent signal.
Per-pitch CV R² ≈ 0 (expected, same as Tunneling+/Stuff+ — single-pitch outcomes are noise-dominated; the
aggregate checks are what matter). Rendered verdict: "Useful, with real caveats."

### Per-pitcher SHAP inspector — added after the user pointed out it was missing

First version of the page shipped with only the aggregate report, no per-pitcher lookup (unlike
Stuff+/Tunneling+) — a real omission the user caught directly. `deception_model.py`'s
`pitcher_deception_breakdown()` scores one pitcher live and returns a SHAP feature breakdown in
Deception+-point units (`GET /api/deception/pitcher/{pitcher_id}`). Verified against real baseball knowledge:
Tyler Rogers's `release_pos_z` (1.33 ft — his submarine arm slot) is by far his single largest positive SHAP
driver, exactly matching how he actually pitches.

### The overlap-with-Stuff+ diagnosis — a real design conversation, redesign DEFERRED not abandoned

User asked directly whether Deception+ might be re-deriving VAA. It isn't (Deception+ never touches
`vaa_residual` or raw VAA) — the real overlap is the 4 literally-shared raw features listed above. **User's
stated end goal**: use Tunneling+ and Deception+ as INPUT FEATURES into a future/improved Stuff+ (a
stacking/ensemble approach), which requires each to carry information Stuff+ doesn't already have, not a
redundant re-encoding.

**CORRECTED in session 4 (superseding this section's original recommendation)**: the fix is NOT to drop
`arm_angle`/`release_extension`/`release_pos_x_as`/`release_pos_z` from Deception+ — those stay. Instead
**Stuff+ drops them** (see "Planned future architecture" below, already said this, just re-confirming it's the
live plan, not the paragraph above). Deception+ keeps its full raw-feature set AND gets the real video-timing
features added on top once wired in (see the new pipeline section below) — the overlap gets resolved from the
Stuff+ side of the stack, not by shrinking Deception+.

## Deception+ v2 delivery-timing pipeline — built and validated this session (session 4), NOT yet run at scale

v3 (prior session) did real but limited R&D: ONE pitcher (Cease), phases identified by manually paging through
frames, a "planned metrics" wishlist not yet built. **This session actually built the extraction pipeline and
the automated phase-detection logic**, iterated on it through many real bugs (documented below), and validated
it screenshot-by-screenshot on several real Keller pitches. Read this whole section before touching this code
— it is the current source of truth, not v3's section (deleted above, this replaces it).

### Where this lives — still NOT in the main project directory

Same isolation pattern as v3: everything lives under
`C:\Users\joshs\AppData\Local\Temp\claude\mlb_pitcher_isolation_investigation\deception_v2\`, using the
isolated Python 3.11 venv at `..\bcv_venv311\` (reuse it, don't rebuild — same baseballcv/mediapipe version
conflict with the main project's Python 3.14 env as before). **This is scratch/R&D territory, not wired into
the main project yet** — `deception_features.py` does not import any of this. Key files, current state:

- **`compute_metrics.py`** — the core engine. `PitchAnalysis` class: takes one pitch's landmarks JSON +
  `p_throws`, runs `detect_phases()` (returns 5 phase frame indices + which side is lift/pivot/throw), then
  `compute()` turns those into the actual metric dict. This is the file that got rewritten the most times this
  session — read its current state directly rather than trusting a summary, but the methodology section below
  reflects the final, validated logic as of session's end.
- **`extract_landmarks_fast.py`** — per-clip landmark extraction. `process_video()` is the current entry
  point: a cheap coarse pass (sparse frames, full detection+pose) locates the real leg-lift peak directly from
  knee height, then a fine pass extracts full-resolution landmarks in a window around it. `process_video_full()`
  still exists as a no-localization fallback/debug path (processes the entire clip) — slower, use only to rule
  out localization as a variable when debugging.
- **`visual_audit.py`** — **the tool to actually trust.** For one pitch: runs the full pipeline, then produces
  (a) a 4-panel matplotlib signal plot (lift knee/ankle height, ankle x-position, ankle x+y movement, throw-arm
  speed, all with the 5 detected phases marked as vertical lines) and (b) 5 annotated real video-frame
  screenshots, one per phase (coming_set, leg_lift_peak, foot_plant+stride, release, rubber position), each
  with the actual measurement drawn as a line/vector on the frame, not just a number. **User's explicit
  standing instruction: ALWAYS show these screenshots after any pull, every time — this is the only way bugs in
  this pipeline get caught, several real ones were only found by looking at a screenshot after the aggregate
  numbers looked "fine."**
- **`rubber_position.py`** — white-rubber-strip detector (color/shape heuristic, anchored to the pitcher's own
  ankle position, not a frame-wide search — see gotchas below for why frame-wide fails).
- **Round 1 / Round 2 artifacts (`selections.json`/`selections_r2.json`, `results.json`/`results_r2.json`,
  `screenshots/`/`screenshots_r2/`, and the two delivered PDFs "Deception+ v2 - Delivery Timing RD.pdf" /
  "Deception+ v2 - Round 2 (Hardened+Faster).pdf") — STALE.** Both were produced BEFORE the phase-detection
  rewrite documented below (the "take first peak not last," combined-arm-speed release, longest-quiet-run
  plant, etc.) — their actual pitch-level numbers do not reflect the current, validated methodology. Keep them
  as historical record of the debugging process (the PDFs literally document real bugs found along the way,
  still useful reading) but **do not treat their per-pitch results as trustworthy** — no full batch run has
  been done yet with the current code, only single-pitch `visual_audit.py` validation.

### Final validated methodology (as of this session's end)

1. **Throwing side, determined FIRST, independent of everything else**: whichever side (MediaPipe's own L/R
   label, not necessarily true anatomical left/right — see gotcha below) has the higher peak **combined
   elbow+wrist speed** anywhere in the clip is `pivot_side`/throwing side; the other is `lift_side`. This is
   the one signal that doesn't need to guess, and it's what makes lefty/righty handling correct without ever
   needing to trust `p_throws` against a MediaPipe label.
2. **`leg_lift_peak`**: find every prominent local max of the LIFT knee's height signal (`scipy.signal.find_peaks`
   with an adaptive prominence threshold, `max(8.0, 0.12*range)`), take the **FIRST** one. (Not the last — see
   bug list. MLB's clip generator anchors the clip to the actual pitch event, so the first prominent knee-lift
   is the real delivery; anything later is post-pitch activity, e.g. fielding a comebacker.)
3. **`coming_set`**: nearest prominent trough (same `find_peaks`, negated signal) before the peak. If none
   exists before it, honest fallback to the start of available data — the real quiet stance may have happened
   before pose tracking locked on; this makes `hold_time` a real underestimate for that specific pitch, which
   is a known, flagged limitation, not silently hidden.
4. **`leg_lift_start`**: first sustained rise in the LIFT ANKLE's height (not knee), scoped to
   `[coming_set, leg_lift_peak]` only — velocity-based onset detection with a threshold local to that window
   (not a whole-clip noise floor, see gotcha below for why that failed).
5. **`foot_plant`**: the longest contiguous stretch, anywhere in a bounded window after the peak (currently
   1.0s), where the LIFT ankle's total x+y movement stays below an adaptive per-window threshold (currently the
   45th percentile of that window's own movement values). "Longest," not "first" — a brief mid-air
   deceleration during descent is real but short; the actual landed-and-staying-still period is longer. Guarded
   against ever picking a stretch of MISSING pose data (which interpolates to a deceptively smooth, artificially
   "quiet" line) via a `has_landmarks` validity mask.
6. **`release`**: peak **combined elbow+wrist speed** (not wrist alone) within `[plant, plant+0.30s]` — release
   must follow foot-plant, never precede it, once peak/plant themselves are anchored correctly.
7. **Hard invariant, checked before returning anything**: `coming_set <= leg_lift_start <= leg_lift_peak <
   foot_plant <= release`. If violated, the pitch is rejected outright (`"phases out of order"`) rather than
   silently returning internally-inconsistent phases. This check is what caught two of the real index-rate
   bugs below.

### Metrics currently computed (per pitch)

`hold_time_s`, `delivery_time_s` (renamed from `drive_time_s` this session — same quantity, leg-lift-start to
release, clearer name), `peak_to_plant_s`, `plant_to_release_s` (**new this session** — foot-plant to release,
a proxy for torso-rotation speed once the front foot is down, separate signal from the whole-delivery
`delivery_time_s`), `leg_lift_height_ratio`, `torso_tilt_windup_deg`/`_plant_deg`/`_release_deg` (torso-line
angle off vertical, evaluated at exactly the coming_set / foot_plant / release frames respectively —
"windup" here means "at rest before anything moves," not mid-lift), `stride_length_norm`, `vertical_bob_norm`,
`hip_shoulder_separation_proxy` (diagnostic only, see below), `concealment_pct` (diagnostic only, see below),
**`arm_path_length_norm`** (**new this session** — total wrist travel distance from leg-lift-peak to release,
normalized by body height, i.e. literally tracking the arm through space, not just timing), **`arm_peak_speed_norm`**
(**new this session** — peak combined elbow+wrist speed in that same window, normalized by body height), and
`rubber_probe` / `pivot_fraction_along_rubber` (position on the rubber — see below, this is the ONE metric
still measured from a raw joint, deliberately: it uses the pivot foot's ARCH, approximated as the ankle/foot-index
midpoint, not the ankle joint itself, per explicit correction — "it should measure where on the rubber the
center point (arch) of the pitcher's foot falls," ankle alone is the heel/joint, not the arch).

`METRIC_BOUNDS` + `validate_metrics()` in `compute_metrics.py`: real physical min/max per metric (e.g.
`delivery_time_s` 0.30-1.5s, `peak_to_plant_s` 0.10-0.8s) — a pitch failing ANY bound gets rejected outright,
not just that one field. **This bounds table was built against the OLD (pre-rewrite) phase-detection logic** —
re-validate it against the new methodology before trusting it for a real batch run; some bounds may need
adjusting now that plant/release land differently.

### Real bugs found and fixed this session (roughly chronological — several are worth remembering for any
future CV/pose-timing work on this project, not just this pipeline)

- **Sign error nulled `leg_lift_height_ratio` on every single pitch** — subtracting two already-negated
  "height" quantities instead of raw pixel-y values, always came out negative, silently turned to `None` by a
  `>0` guard. Caught by a synthetic hand-built test BEFORE running on real data, not from real output — worth
  repeating this pattern (a small synthetic skeleton sequence with known expected phase timing) before trusting
  any future rewrite of this logic.
- **Foot-plant detection repeated the EXACT mistake v3's Cease validation already found and documented**: an
  early version gated plant on ankle HEIGHT returning to a baseline, which v3 explicitly warned against ("ankle
  height is a bad signal for a bent-knee lift"). Re-introduced by accident in a later rewrite, caught again the
  same way (a screenshot showing the foot still airborne). **Lesson: write hard-won lessons into code
  comments/tests, not just prior handoffs, or they get silently reintroduced.**
- **Unbounded phase-search windows repeatedly grabbed unrelated LATER motion** in these clips, which run
  6-22s with a lot of pre/post-pitch broadcast content — produced physically-impossible multi-second
  "drive times," then later a foot-plant that was actually a completely different, later moment (once even
  landing on a different person — a base-runner — after the fallback bbox drifted). Fixed by bounding every
  phase search to a real, tight, physically-justified time window relative to the previous phase, every time —
  this class of bug recurred multiple times across different search steps before the general principle stuck.
- **A REAL, recurring index-rate bug, worth remembering specifically**: `self.fps` on a `PitchAnalysis` object
  is the video's NATIVE fps, but the landmarks array itself advances by whatever extraction STRIDE was used
  (e.g. every 2nd native frame) — using `self.fps` directly to convert a time duration (like "0.3 seconds") into
  an index count overshoots by ~stride-x. This bit at least 3 different places in the code independently before
  being fully stamped out; the fix is always to derive the array's real sample rate from its own `self.t` values
  (`np.median(np.diff(self.t))`), never from `self.fps` directly, for any index-count-from-duration conversion.
- **MediaPipe's left/right landmark labels are assigned independently PER FRAME with no temporal identity
  tracking** — not "the same left leg" frame to frame, just whichever side looks anatomically left in that one
  image. On real data (a pitcher whose legs cross mid-delivery, common), this flips mid-clip and silently
  corrupts any cross-frame comparison (produced an ~8px "stride" on a ~250px-tall body). Fixed with a greedy
  temporal-consistency pass (`_fix_lr_continuity`): at each frame, keep-or-swap ALL left/right-paired landmarks
  together, whichever choice is more spatially continuous with the previous, already-corrected frame.
- **Also confirmed separately**: MediaPipe's L/R labels are NOT reliably tied to true anatomical sides at all on
  these rear-view "pitcher cam" broadcasts (trained mostly on front-facing subjects). Originally handled by
  hardcoding `throw_wrist` from `p_throws` — wrong, since that assumes a label mapping that doesn't hold. Fixed
  by determining throwing side empirically (peak arm speed, see methodology above) instead of ever trusting
  `p_throws` to pick a MediaPipe label.
- **"Take the LAST prominent knee-height peak" (an earlier version's rule) was confirmed WRONG on real
  screenshots** — it was landing on post-pitch activity, well after the ball was already gone. Flipped to FIRST
  peak once the real reason was understood (see methodology #2 above) — this is not a coin-flip choice, it
  follows from how MLB's own clip generator times these clips.
- **A pitcher's `x`-only ankle "stillness" check for foot-plant missed vertical motion entirely** — a striding
  foot can genuinely slow down sideways for a frame or two while still descending through the air, which isn't
  landed. Fixed by using combined x+y movement, per direct user instruction.
- **The "longest quiet run" plant detector's first version found a stretch of MISSING pose data**, not real
  stillness — linear interpolation across a long gap in real detections produces an artificially smooth,
  low-velocity signal. Fixed by (a) bounding the search window tightly (1.0s past peak, not a generous multi-
  second window) and (b) an explicit `has_landmarks` validity mask that keeps any interpolated stretch from
  ever counting as "quiet."
- **The adaptive stillness-threshold percentile needed real tuning, not a first guess** — 30th percentile let
  a brief post-touchdown weight-settling wobble fragment the true (longer) stillness period into two shorter
  pieces, causing the algorithm to pick a LATER secondary rest period instead of the real touchdown. Raised to
  45th percentile, confirmed correct on a real screenshot (foot down, arm still cocked — the right sequence).
- **The original coarse-localization speed optimization (bbox-area rate-of-change as an "activity" signal) did
  NOT reliably align with the real leg-lift peak** — replaced with a coarse pass that computes actual (sparse)
  knee-height directly and applies the SAME "first prominent peak" rule the real detector uses, so localization
  and detection agree by construction instead of by coincidence.

### Known open issues (real, unresolved as of session's end)

- **`stride_length_norm` measures too small, consistently, across multiple real pitchers/pitches** — current
  read: this is genuine camera-angle foreshortening on the standard "behind the pitcher" broadcast angle these
  clips all use (the pitcher's stride runs mostly ALONG the camera's line of sight toward the plate, not across
  it, so real 3D stride collapses into a small horizontal-pixel displacement), confirmed by the signal plot
  showing the lift ankle's x-position barely moving through an entire real delivery. **Not yet fixed.** User's
  own idea, not yet acted on: since some ballparks'/broadcasts' camera angles are less purely axial than others,
  it may be possible to deliberately target games/parks with a more favorable (more lateral) camera angle for
  stride specifically, rather than fixing this in software. Worth real investigation (which parks/broadcast
  feeds actually have a better angle) before writing more code here.
- **`hold_time_s` can be a real underestimate** when a clip's earliest usable landmark data starts after the
  pitcher has already begun moving (broadcast graphic overlays delay usable tracking on some clips) — the
  `coming_set=0` fallback is honest about this (doesn't fabricate a fake stillness point) but the resulting
  number is still short of the true value for that specific pitch.
- **Efficiency measured on exactly one clip so far (~90s for a ~12s clip)** — a real but modest win relative to
  full-clip processing for that specific LONG clip; not yet A/B tested on the more typical 6-7s clips where the
  fixed coarse-pass cost should be a much larger fraction of a bigger total win. Confirm before assuming the
  speed characteristics generalize.
- **`METRIC_BOUNDS` needs re-validation** against the current (rewritten) phase-detection logic — it was tuned
  against the OLD logic's output ranges.
- **Hip-shoulder separation and ball concealment remain unvalidated diagnostics**, computed but explicitly not
  trusted as real metrics (same status as v3 left them — hip-shoulder separation is fundamentally a rotation
  around a vertical axis, mostly invisible to one 2D camera; concealment is a crude 2D silhouette-overlap proxy).
  User wants both KEPT and computed every run regardless, with the eventual plan to test whether adding them
  to the model measurably helps before deciding whether they're worth trusting.
- **No batch run with the current, final pipeline has been done yet** — only single-pitch validation via
  `visual_audit.py`. **User will direct the timing/scope of the next validation batch — do not start one
  unprompted**, same standing instruction as v3.

### A real, non-obvious finding worth remembering: video download is cheap, processing is what's slow

Confirmed directly (not assumed): resolving a `playId` to a video URL and downloading it takes ~1-2 seconds per
clip once the game is a real regular-season/postseason game (see gotcha below re: Spring Training). The real
compute cost is the detection+pose EXTRACTION, not the download — so "download only part of the video" isn't
the right lever; reducing how much of the DOWNLOADED clip gets processed (the coarse-then-fine localization
above) is where the actual speed win lives.

## Planned future architecture (stated by user, NOT yet executed — just the plan)

Three-model separation, explicitly designed to minimize feature overlap so Tunneling+ and Deception+ can later
be fed INTO an improved Stuff+ as features:
- **Tunneling+**: purely how well/how long pitches track and stay indistinguishable from release to the plate.
  (Current state — no changes planned here beyond the existing v2 roadmap items.)
- **Deception+**: purely pitching mechanics, body, angles, delivery/timing — independent of the pitch's own
  trajectory. (Current state: proof-of-concept with a scoped-down feature set; real redesign deferred until the
  video-timing pipeline is production-ready, see above.)
- **Stuff+**: eventually "everything" — the user plans to (1) REMOVE raw release position (x/y/z) and
  `release_extension` from Stuff+'s feature set, since those are mechanics/delivery properties that belong to
  Deception+, not intrinsic pitch-shape properties; (2) ADD **HAA (horizontal approach angle)** alongside the
  existing VAA, as the two are "direct properties of the pitch" (trajectory at the plate), not mechanics; (3)
  eventually feed Tunneling+ and Deception+ themselves in as INPUT FEATURES to a next-generation Stuff+
  (stacking/ensemble). **None of this has been implemented yet** — explicitly told "don't do it yet." Perceived
  velocity (already computed, not displayed — see v2 handoff) is expected to partially absorb what extension
  currently contributes once extension itself is removed.

## Environment/tooling gotchas from this session (in addition to previous handoffs')

- **The video CDN (`sporty-clips.mlb.com`) requires real browser-like headers** (`User-Agent` + `Referer`) on
  direct `requests.get()` calls or it 403s, even with a correct, freshly-resolved URL.
- **Cross-origin canvas capture from a `<video>` element fails with `SecurityError: Tainted canvases may not be
  exported`** when the video's origin differs from the page's — confirmed hitting this trying to rasterize a
  Statcast video frame via canvas the same way the existing SVG-rasterize workaround works. Direct navigation
  to the video CDN's own origin was also denied by this environment's browsing policy. The actual fix that
  worked: download the video via plain `requests` (with the headers above) and process frames with `opencv`
  locally — bypasses the browser entirely for this use case.
- **This environment's Browser pane occasionally denies `navigate` on an existing tab** ("navigation ... was
  denied or failed") for no apparent reason — opening a fresh tab (`tabs_create`) and navigating that instead
  reliably works around it. Hit this multiple times this session.
- **`mediapipe`'s modern pip build ships ONLY the Tasks API** (`mediapipe.tasks.python.vision.PoseLandmarker`)
  — the legacy `mp.solutions.pose`/`mp.solutions.drawing_utils` API used in most online tutorials/examples is
  NOT present (`ModuleNotFoundError: No module named 'mediapipe.framework'` if you try `drawing_utils`). Draw
  landmarks manually with `cv2.circle` using the raw `.x`/`.y` coordinates instead.
- **A single detected "person" from MediaPipe with no player-identity model behind it is NOT reliable for
  isolating a specific role (pitcher vs. catcher vs. umpire) in a multi-person frame** — confirmed as a real,
  repeatable failure mode (not a one-off), fixed only by `baseballcv`'s dedicated detector, see the video R&D
  section above.
- **Raw joint height (e.g. ankle y-coordinate) is an unreliable signal for detecting a bent-knee leg lift** —
  visually verify any phase-detection heuristic against real frames before trusting the numbers; a "confident"
  numeric peak/dip can be pure pose-tracking noise, and this cost real time on the Cease validation before it
  was caught.
- `baseballcv`'s dependency chain (old pinned `mediapipe`, `pandas<3.0`, `pillow` build failures on very new
  Python versions) makes it fundamentally incompatible with keeping it in the same environment as a project
  already on newer pandas — isolated venv is not optional, it's required.
- Still no `requirements.txt`, still no git repo — now genuinely worth doing given three sessions of real,
  significant, undocumented dependency accumulation (`mediapipe`, `opencv-python`, `ultralytics`/YOLO in the
  isolated venv, `baseballcv`, on top of everything from the prior two sessions).

## Environment/tooling gotchas from session 4 (in addition to all of the above)

- **`MAX(game_id)` does NOT reliably return the most recent REGULAR-SEASON game for a pitcher** — gamePks are
  not chronological across game types; Spring Training games (Feb/March) get gamePks that interleave with real
  regular-season ones. First real attempt at "pick a pitcher's most recent game" this way silently picked
  Spring Training games for 5/6 pitchers (no video available on Savant's normal CDN for those, every download
  failed before this was diagnosed). **Fix: sort on `MAX(game_date)` instead**, and cross-check game type
  against the MLB schedule API (`https://statsapi.mlb.com/api/v1/schedule?gamePk=<id>` →
  `seriesDescription`) before trusting a pick for anything video-related.
- **Real, useful discovery**: MLB's own "sporty clip" generator appears to use a roughly FIXED pre-roll before
  the actual pitch event across different clips of similar length — confirmed by a coarse localization pass
  independently converging on nearly identical clip-relative timing (~t=3.2s) across multiple different
  pitchers'/games' clips before this was even understood. Don't rely on this being exact or universal, but it's
  consistent enough that "the pitch is probably in the first half of the clip, not scattered randomly
  throughout" is a safe assumption for any future localization work.
- **YOLO inference resolution matters a lot on CPU, batching barely does**: dropping `imgsz` from 640→384 for
  the `phc_detector` YOLO calls measured ~4x faster (0.56s→0.14s/frame) with essentially identical detection
  quality (same bboxes, confidence still comfortably above threshold) — checked directly on real frames, not
  assumed. Batching multiple frames into one `predict()` call barely helped by comparison (no GPU here to
  amortize kernel-launch overhead against) — resolution was the real lever, not batch size.
- **Frame-WIDE white-strip search for the pitching rubber fails immediately** — picks up the batter's white
  socks, catcher's gear, foul lines, and the rosin bag before it ever finds the actual rubber. The working
  version in `rubber_position.py` anchors a narrow search band directly below the pitcher's own already-known
  ankle landmarks. Even then, expect real fragmentation from partial occlusion by the foot itself and dirt
  specks — merge nearby fragments and filter by aspect ratio (the rubber reads as wide-and-thin, a rosin bag
  reads as roughly square) before trusting a detected span.
- **Random-access video seeking (`cap.set(CAP_PROP_POS_FRAMES, n)`) repeatedly, once per frame you want, is
  MUCH slower than one seek + sequential reads** on H.264 video — has to decode forward from the nearest
  keyframe on every seek. A first attempt using repeated seeks for a "coarse then fine" sampling pattern
  measured barely faster than processing every frame despite touching far fewer of them; switching to one seek
  + sequential `cap.read()` (skipping/discarding frames you don't want, not seeking past them) fixed it.
- **MediaPipe pose estimation itself is cheap (~0.05s/frame)** — if a pipeline stage feels slow, profile before
  assuming pose is the bottleneck; on this project it was consistently the YOLO detection step.

## Roadmap / open items (as of this session's end)

- **Video-timing pipeline (updated, session 4)**: phase-detection methodology is now built and validated at the
  single-pitch level (see the new section above) — the next step is a real multi-pitch/multi-pitcher batch run
  with the CURRENT code, screenshots reviewed for every pitch before trusting any aggregate number. **User will
  direct the timing/scope of that next batch — do not start one unprompted**, same standing instruction as
  before. Open sub-items specifically: re-validate `METRIC_BOUNDS` against the new logic, investigate the
  stride/camera-angle issue (possibly by targeting specific parks/broadcasts with better angles, user's own
  idea, not yet acted on), and only AFTER a batch run looks clean should `deception_features.py` actually be
  wired up to consume this data (still not done — this pipeline produces data, nothing reads it into the model
  yet).
- **User's standing PDF instruction**: wants a PDF like the ones already delivered (`build_report.py` /
  `build_report_r2.py` are the templates, though both need updating for the current metric set) generated
  after every real pull/validation batch going forward, documenting methodology + results + screenshots — not
  just this one time.
- **Deception+ v2 feature-overlap fix** (dropping shared raw features) — **CORRECTED this session**: this
  happens on the STUFF+ side, not Deception+'s. Deception+ keeps its full raw feature set. See the corrected
  "overlap-with-Stuff+ diagnosis" section above — the old recommendation to trim Deception+'s features was
  wrong and has been struck through/corrected in place, don't act on the original paragraph.
- **Stuff+ feature-set change** (drop release position/extension, add HAA, eventually stack in Tunneling+/
  Deception+ as features): planned, explicitly not started yet.
- Tunneling+ v2 candidates carried from the prior design: velocity-adjusted decision point, spin-axis
  mirroring, shrinkage for thin pair counts, year-over-year predictive-validity check.
- Older still-open items, unchanged: arsenal classification as its own feature, real FanGraphs season-aggregate
  table, league-average baseline for Splits' `_overall` columns, Player-page gauge archetype thresholds not
  yet tuned, no `requirements.txt`, no git repo.

## Working-style notes for whoever picks this up

- **Verify before trusting, always** — this session's two biggest wins (the Tunneling+ full-pairwise-distance
  fix and catching the Deception+ split-half bug) both came from refusing to accept a suspiciously-good number
  without checking the mechanism. A metric that looks "too good" (0.999 correlation, a leaderboard of unknown
  names) is a bug until proven otherwise, not a result to celebrate.
- **When a plausible mechanism doesn't survive validation, say so and drop it** — the release-similarity
  component in Tunneling+ is the clean example: real theory, tested honestly, measurably made things worse,
  documented as "tried and rejected" rather than quietly dropped or forced in anyway.
- User asks pointed, technical follow-up questions ("is this just re-deriving VAA," "why does this still
  correlate after residualizing") and expects real, checked answers, not reassurance — when the honest answer
  reveals a real design flaw (the Stuff+ feature overlap), say so plainly and propose a concrete fix.
- User is comfortable directing multi-hour background R&D (the video pipeline investigations) but wants
  explicit control over WHEN large/expensive runs actually happen — scope and propose, don't just execute at
  scale.
- Same as before: methodology documented as decisions are made, deliverables actually delivered (PDFs, freeze
  frames) as files the user can see directly, not just described in chat.

**Added, session 4:**
- **User will ask methodology questions ("at exactly which point...", "walk me through how X is designed") and
  expects a real, specific walkthrough of the actual current logic before more code gets written** — not a
  summary, not reassurance that it's handled. When a design looks wrong, they'll say exactly what's wrong and
  why, from having actually looked at a screenshot, not just distrusted a number on principle.
- **ALWAYS show screenshots of pitch-level results, every time, unprompted** — this is now a standing
  instruction, not a one-off ask. Several real bugs this session (wrong cycle selected, foot still airborne at
  "plant," tracker locked onto a base-runner) were ONLY caught by looking at an actual annotated frame, never
  by an aggregate number alone looking wrong.
- **When something is "mostly working but not clean," report back honestly with the specific remaining issue
  rather than declaring success or continuing to silently patch** — user explicitly asked for this workflow
  ("if that comes back clean then you're allowed to [proceed]... otherwise report back with any issues") and
  it's the reason several real bugs got caught before a larger, wasted batch run.
- **User is willing to go several rounds of "try X, show me, here's what's still wrong, try Y" on a single hard
  sub-problem** (foot-plant detection took ~6 real iterations this session) rather than accepting a "good
  enough" first attempt — don't rush to declare a fix validated after one clean-looking result if the
  underlying signal has shown itself to be noisy.
- **Prefers efficient, compact, "clean slate" rewrites over accumulating patches once a piece of logic has been
  patched several times** — when a function has grown a long comment trail of "bug found, fixed, bug found
  again," that's the signal to stop patching and rewrite it cleanly rather than add patch #7.
