# MLB Dashboard — Project Handoff

Rewritten from scratch on 2026-08-25 as a complete reference, not a session log. The previous
version had grown to 1000+ lines of chronological "Session N" entries, most of it about an
experimental video-timing pipeline that was never merged into the project. This version is
organized by TOPIC instead, so it stays useful regardless of when it's read next. A short pointer to
that unmerged thread is at the bottom; everything else here describes what's actually live.

**Immediate next step, per direct instruction**: continue in a plain claude.ai chat conversation
(not Claude Code) to work through model methodology — starting with Deception+, but the whole
picture below is meant to support that conversation without needing any files reopened.

## What this project is

A personal MLB analytics dashboard: real Statcast pitch-level data (2021 through the current
in-progress season) feeding three original "+"-scaled models (Stuff+, Deception+, Tunneling+) plus a
full player/team/prospect browsing site. Built as a learning project and, per user context, a
portfolio piece for baseball analytics job applications — methodology rigor and honest validation
matter here beyond just shipping features.

## Architecture, foundation to surface

```
pulls/  →  mlb_data.db  →  analytics/  →  api/  →  frontend/
(fetch)    (single SQLite) (compute)      (JSON)    (renders it)
```

- **`pulls/`** — one script per external data source: Statcast pitch-by-pitch (`pull_statcast.py`),
  players/games/schedule (`pull_players_games.py`), MLB Stats API sabermetrics and Savant leaderboard
  stats (`pull_sabermetrics.py`, `pull_advanced_stats.py`), injuries, prospects, pitcher bio
  (height/weight, needed by Deception+, not in Statcast at all).
- **`mlb_data.db`** — single SQLite file, gitignored (regenerable, ~2GB). `pitches` is the
  foundation: one row per real pitch, everything else on the site is a query or model on top of it.
  Reference tables (`players`, `games`, `teams`) are resolved from whatever's in `pitches`, not pulled
  independently. `player_advanced_stats` (Sprint Speed, OAA, real wRC+/xFIP/WAR/Barrel%) is genuinely
  external — not derivable from pitch-level data.
- **`analytics/`** — the compute layer: percentiles, splits, archetypes, and the three models.
- **`api/`** — FastAPI, one router per feature area. `api/warmup.py` pre-computes expensive
  league-wide pools at startup/daily so requests don't eat a cold-cache cost live. `api/scheduler.py`
  runs the full pull+warmup pipeline automatically at startup and every day at 6 AM.
- **`frontend/`** — plain HTML/JS/CSS, one JS file per page, hand-built SVG for every chart (no
  charting library).
- **`models/`** — trained model artifacts: LightGBM boosters (`stuff_plus_v2.txt`,
  `deception_plus_v1.txt`) plus their metadata/validation-report JSONs, and Tunneling+'s precomputed
  per-season score files (`tunneling_v1_pitchers_{season}.json`, all 6 seasons already built).
- **`training/`** — offline model-fitting scripts (`train_stuff_model.py`, `train_deception_model.py`,
  `build_tunneling_report.py`), run by hand, not part of the live request path.

Run with `python -m uvicorn api.main:app --port 8010` (no `--reload`, kill+relaunch after any
backend `.py` edit). Real git repo now, pushed to a private GitHub repo at
`github.com/JTaylor-Data/mlb-dashboard`.

## Recent data-quality work (this session)

- **Purged Spring Training + postseason contamination** that had leaked into `pitches`/`games` across
  ALL of 2021-2026 (1,319 non-regular-season games, 316,501 pitch rows) — root cause was
  `pull_statcast.py` pulling by date range with no `game_type` filter, plus a deliberately early
  `EARLIEST_SEASON_START` buffer to avoid missing real early openers (2025's Tokyo Series, etc.).
  Fixed by classifying every game_id against the real MLB Stats API schedule (not a date heuristic —
  date cutoffs would have wrongly flagged 2023's and 2025's real early-March openers). Verified zero
  real regular-season games were lost in the process. **The root cause in `pull_statcast.py` itself
  is still unfixed** — it'll happen again at the start of a future season unless the ingestion adds a
  real `game_type` filter.
- **Adopted Baseball Savant's real percentile-ranking qualifying thresholds** (2.1 PA/team-game for
  batters, 1.25 PA/team-game i.e. batters-faced for pitchers — confirmed directly off Savant's own
  page), replacing ad hoc flat thresholds (200 pitches / 100 PA) that had no real source.

## The three models

All three are "+"-scaled (100 = league average), all three are LightGBM-based except Tunneling+
(which is a geometric-mean composite of pairwise distance ratios, not a fitted model per se), and all
three exclude pitcher identity and plate location/count as features by design — the whole point is
to grade something that would generalize to a pitch that hasn't been thrown yet, not to memorize
"this is a good pitcher" or reward hitting a corner.

### Stuff+ — the pitch's own physical shape

**What it isolates**: velocity, movement, spin, release — independent of where the pitch was thrown
or the count (deliberately excludes `plate_x`/`plate_z`/`balls`/`strikes` — that's a future
Location+/Command+ model's job, mixing it in here would make a fastball down the middle look bad and
a lucky corner-catcher look great).

**Features** (`analytics/stuff_features.py`): `release_speed`, `effective_speed` (perceived velo,
already extension-adjusted), `release_spin_rate`, `spin_axis`, `hb` (arm-side-adjusted horizontal
break), `ivb`, `release_extension`, `release_pos_x_as`, `release_pos_z`, `arm_angle`, `vaa_residual`
— plus categoricals `pitch_type`, `p_throws`, `stand`, `season`.

**`vaa_residual` is the one non-obvious feature**, and worth understanding since it's directly
relevant to the Deception+ discussion below: raw VAA (vertical approach angle) is geometrically
confounded with plate height — a pitch crossing higher has less time for gravity to act on it, so it
reads flatter regardless of the pitcher's actual shape. `fit_vaa_expectation()` fits a plain
least-squares `vaa ~ plate_z + plate_z² + release_speed` across the league, and the residual (actual
minus expected) is "flatter or steeper than physics alone predicts," with the location component
removed. **This session independently built a similar but distinct height-adjustment** for the
player-page percentile display's FB VAA (a per-pitch-type linear regression against `plate_z` only,
applied to the pitcher's PRIMARY fastball type specifically) — the two aren't the same code and
weren't reconciled; worth comparing approaches if VAA methodology comes up.

**Scale**: `100 - 10*(pred - season_mean)/season_std`, z-scored fresh within whichever season is
requested. Small-sample pitch types are shrunk toward that PITCH TYPE's own league mean (not the
grand mean) via empirical-Bayes shrinkage — the reliability constant K is estimated fresh from each
season's own data, came out tiny (0.4-2.8 pitches) meaning the metric stabilizes almost immediately;
mainly matters for genuinely rare pitch types (a handful of pitchers throwing a screwball or eephus).

**This session's validation** (2021-2026, full history):
- **By far the most year-over-year stable of the three models** — r² between 0.58 and 0.67 across
  every consecutive season pair. Expected: physical characteristics are genuinely repeatable traits.
- **By pitch type** (2025): common arsenal pitches (FF/SI/SL/CH/CU/ST/FC/FS) cluster tightly 96-102,
  as expected since standardization is league-wide not per-type. The rare codes are the real signal:
  eephus/pitchout/unclassified score far below average (45-68), almost entirely position players in
  mop-up innings, not real pitcher signal. **Knuckleball (KN, ~72) is a genuine, known limitation, not
  a bug**: real knuckleballers throw slow with little conventional movement by design, so a
  physically-based model correctly reads their raw characteristics as bad stuff while missing the
  entire reason the pitch works — unpredictability.
- **2025 top 5 individual seasons (2021-2025 pool)**: Pete Fairbanks 2022 (124.2), Emmanuel Clase
  2022 (122.0), Félix Bautista 2023 (121.5), Clase 2021 (119.9), Clase 2024 (118.4).

### Deception+ — pitcher body and delivery mechanics, independent of pitch shape

**What it isolates**: does something about a pitcher's BODY and release consistency predict good
outcomes, independent of how good the pitch itself is (that's Stuff+'s job)? Never touches pitch
trajectory, movement, or spin.

**Features** (`analytics/deception_features.py`): `arm_angle`, `release_extension`,
`release_pos_x_as`, `release_pos_z` (all raw, per-pitch — **note all four are ALSO Stuff+ features**,
see the overlap discussion below), `arm_angle_std`, `release_extension_std`, `release_consistency`
(season-long per-pitcher variability in the above, joined onto every pitch that pitcher threw that
season), `height_in`/`weight_lb` (from a separate `pitcher_bio` table, pulled from the MLB Stats API
since Statcast has no biographical data — **time-invariant**, same value regardless of season
queried, a real limitation for year-over-year work), `season` (categorical).

**Confirmed NOT obtainable from any accessible source**: delivery length (windup-to-release
duration), leg-lift timing, time held in set, ball concealment — would need frame-by-frame
motion-capture data Statcast doesn't publish. (A separate, unmerged computer-vision pipeline
attempted to recover these — see the bottom of this document.)

**Model**: LightGBM, Huber loss, leave-one-season-out CV 2021-2025. Same z-score plus-scale formula
as Stuff+. **No league-wide table function exists in the runtime module** — unlike
`compute_stuff_plus()`, `deception_model.py` only scores one pitcher at a time, on demand
(`pitcher_deception_breakdown`). One was built ad hoc for this session's diagnostics but not merged
in permanently. **The live site also only ever scores season=2025**, hardcoded
(`REPORT_SEASON` in `api/routers/deception_report.py`) — the model itself is season-agnostic, there's
just no season selector wired up yet.

**This session's validation** (the main open thread for the next conversation):

1. **Least year-over-year stable of the three models, consistently** — r² between 0.17 and 0.34
   across every season pair (roughly a third to a half of Stuff+'s). Open question: is this because
   "deception" as currently operationalized is a real but inherently noisier signal, or because the
   feature set isn't capturing the right thing? The `_std`/`_consistency` features are season-long
   aggregates — a pitcher's true mechanical consistency probably doesn't swing much year to year, so
   if the SCORE swings a lot anyway, that's worth digging into.
2. **Correlates moderately with Stuff+ (r≈0.535)**, weakly with Tunneling+ (r≈0.12-0.19). Plausibly
   real (better-stuff pitchers tend to have more repeatable deliveries) but four raw features are
   LITERALLY shared with Stuff+ (see above), so some of this is overlap by construction. Open
   question: is 0.535 acceptable, or should those four features eventually move OFF Stuff+ and become
   Deception+-exclusive (the stated future-architecture plan below says yes, not yet executed)?
3. **A real methodological finding, worth walking through carefully**: arm_angle's SHAP contribution
   ALONE made extreme submarine deliveries look PENALIZED — contradicting the obvious real-world
   check (Tyler Rogers, a famous submarine reliever, should be rewarded for his weird slot, not
   punished). Investigated directly: Rogers's real 2025 SHAP breakdown shows `arm_angle` at -3.5
   points but `release_pos_z` (his release height — mechanically almost the same information as arm
   angle, since a lower slot means a lower release point) at **+17.7 points**, net strongly positive,
   part of a 133.5 overall score (elite). `arm_angle` and `release_pos_z` correlate at **r=0.79**
   league-wide — the tree splits "unusual delivery" credit between two nearly-redundant features. The
   corrected COMBINED (arm_angle + release_pos_z) SHAP curve, real per-pitch values across a full
   season (708,795 pitches), binned by arm angle:

   | Arm angle | Combined SHAP contribution |
   |---|---|
   | -62° to -58° (true submarine) | **+14.1 to +14.3 pts** |
   | -27° to -23° (moderate sidearm) | +3.7 to +4.0 pts |
   | -12° to +45° (the normal range) | roughly -1.4 to +0.3 pts, mostly flat |
   | ~57° (high three-quarters, most common slot) | +1.2 pts |
   | 67° to 72° (pure over-the-top) | **-2.8 to -5.9 pts** |

   Confirms the original intuition (true submarine IS strongly rewarded) but the relationship is
   **asymmetric**, not "extreme is always good" — pure over-the-top is penalized, not rewarded, even
   though it's just as rare. Open question: is that asymmetry real signal, or an artifact of how few
   true over-the-top pitchers exist to learn from reliably? And should `arm_angle`/`release_pos_z` be
   combined or orthogonalized before the next training run, given how collinear they are?
4. **2021-2025 outlier seasons**: top 5 — Félix Bautista 2023 (146.2), Jacob deGrom 2021 (140.8),
   Bautista 2022 (137.6), Tyler Rogers 2021 (136.2), Rogers 2025 (133.5). Bottom 5 — Noah Murdock
   2025 (54.2), Jake Reed 2022 (55.1), Madison Bumgarner 2023 (57.2), Ty Blach 2024 (65.2), Ty Madden
   2024 (65.9). Rogers appearing twice in the top 5 (different seasons) is a useful counterpoint to
   finding #1 — his own extreme-delivery signal is captured consistently at the individual level even
   though the aggregate year-over-year correlation is weak.

**Open questions for the next conversation, in priority order**: (1) why is Deception+ so much
noisier year-over-year than Stuff+ — real or fixable? (2) should the four Stuff+-shared features move
off one model or the other? (3) should arm_angle/release_pos_z be combined given r=0.79? (4) is
season-level aggregation the right grain for the `_std` features, or would a rolling recent-starts
window be more honest?

### Tunneling+ — how indistinguishable a pitcher's pitches are from each other

**What it isolates**: do a pitcher's pitches look alike out of the hand (release point) and separate
late (near the plate, close to when a hitter has to commit)? A real, separate skill from either raw
pitch quality or delivery mechanics.

**Methodology**: for every pair of pitch types a pitcher throws (minimum 100 pitches per type to
qualify), compute the real PAIRWISE mean distance (every real instance of pitch A vs. every real
instance of pitch B — not centroid-to-centroid, see the bug fix below) at three points: release,
the "decision point" (23.5 ft from the plate, a fixed flight-path location, kept in feet
deliberately — a location, not a movement magnitude, unlike the other distances which are in inches),
and the plate. `decision_similarity_plus` (100 × league-avg tunnel distance / this pair's tunnel
distance) and `plate_separation_plus` (100 × this pair's plate distance / league-avg plate distance)
combine via **geometric mean** into the pair's own `tunneling_plus`. A pitcher's overall
`tunneling_plus` is the usage-weighted average across his qualifying pairs, with pairs involving
FF/SI counted 2x (a hitter's read is built primarily against the fastball/sinker, so an offspeed
pitch's tunnel with THAT matters more than two offspeed pitches tunneling with each other).

**Two real bugs found and fixed during its build** (worth knowing since they're the kind of mistake
that could recur in similar geometry-based work): (1) the original formula was `plate_distance /
tunnel_distance` — a straight ratio that coupled the two halves by division, so a pair with genuinely
huge plate separation could still score below average just because its tunnel distance was also
middling (caught on a real example: Tarik Skubal's changeup/fastball, his single largest real plate
separation, scored below average under the old formula). Fixed by scoring each half independently
then combining by geometric mean. (2) Centroid-to-centroid distance (averaging all of a pitcher's
changeups into one point, averaging all fastballs into another, then measuring one distance) silently
erased real pitch-to-pitch command variance — checked on real data, centroid distance said Skubal's
CH/FF were 5.3" apart at the decision point, the full pairwise mean said 9.7". Fixed, and leaguewide
average tunnel distance nearly tripled once measured correctly.

**A third component (`release_similarity_plus`) was tried and rejected** — tested honestly against
the full validation battery, folding it into a 3-way geometric mean made the model measurably worse
(the sequential same-at-bat test, the most rigorous check available, dropped from z=9.06 to z=1.60,
below significance). Real hitters' whiff rates track decision-point similarity and plate separation,
not release-point similarity, as closely as the theory predicted. Kept as a displayed, unscored
diagnostic, not silently re-added without re-validating.

**Precomputed, not live-scored**: unlike Stuff+ (live inference) and Deception+ (live per-pitcher),
Tunneling+'s validation battery is genuinely expensive (~85s/season with the full-pairwise fix), so
it's fully precomputed offline into `models/tunneling_v1_pitchers_{season}.json` for all 6 seasons —
changing the season on the site's own page reloads the whole precomputed file, not a live query.

**Final validated state**: split-half reliability ~0.85-0.91, sequential same-at-bat test z≈9 (very
strong — real hitters' whiff rates genuinely track this in the same at-bat, the single most
convincing validation check across all three models).

**This session's validation**: year-over-year r² between 0.31 and 0.53 across season pairs — sits
between Stuff+ and Deception+ in stability. Correlates weakly with both other models (r≈0.12-0.19),
confirming it's capturing something genuinely distinct. **2021-2025 top 5 individual seasons**: Kevin
Ginkel 2025 (113.5), Cionel Pérez 2021 (113.2), Garrett Cleavinger 2022 (112.3), Alex Vesia 2023/2024
(112.1/111.3). **Bottom 5**: Matt Bowman 2024 (82.9), Rich Hill 2021/2022/2023 (84.8/86.6/87.5),
Yusmeiro Petit 2021 (85.6).

## Full validation artifact

All three models' score distributions, year-over-year regression scatter plots, cross-model
correlation grid, top/bottom-10 tables, and the two model-specific diagnostics (Stuff+ by pitch type,
Deception+'s decomposed arm-angle curve) are published as a real chart-based report:
**https://claude.ai/code/artifact/f908a7a9-a6a3-497f-b4a6-04f8464e05d7** ("Stuff, Deception &
Tunneling") — open directly if the next conversation can render links; the numbers above are the
prose summary either way.

## The rest of the site (not model-related)

### Player tab (sub-nav: Overview / Splits / Radar / Outing / Compare)

- **Overview** (`index.html`): real Statcast pitch-level breakdown per player, percentile rankings
  (Baseball Savant's qualifying thresholds — 2.1 PA/team-game batters, 1.25 for pitchers), Pitch
  Locations heatmap sized by league-wide command percentile within pitch type, batter swing archetype
  cards (Attack Angle/Direction, Swing Length, Swing Tilt — 0-30th/31-69th/70th+ buckets), Similar
  Players feature (nearest neighbor in percentile-space, any qualifying player-season 2021-2026).
- **Outing Breakdown** (`outings.html`): single-game breakdown for any pitcher. Entry point: button
  on Top Performers cards, or pitcher search on the page itself. **Design** (as of 2026-09-03):
  - *Hero header*: headshot + team logo + player name + matchup date/opponent + Max EV + Max Velo boxes.
  - *Table 1 — Usage, Velocity & Strikes*: Pitch | n | % | Yr% | vs RHH | vs LHH | Velo | ±Velo | K |
    SW | CS | Whiff (x/x) | W% | CS+W (x/x) | CS+W% — plus ALL summary row.
  - *Table 2 — Movement & Spin*: Pitch | Drop (in) | ±Drop | IVB (in) | ±IVB | HB (in) | ±HB |
    Spin (rpm) | ±Spin — plus ALL summary row.
  - Drop uses gravity formula: `drop = 11.4 - (pfx_z_feet × 12)`. All ±deltas compare today's outing
    average to that pitcher's season average for that pitch type. `pfx_x`/`pfx_z` stored in feet,
    displayed in inches. Explicitly excludes 1st Pitch Strike %.
- **Compare** (`compare.html`): side-by-side player comparison, accessible from the Player sub-nav
  (moved from top-level nav as of 2026-09-03).

### Other pages

- **Games / Probable Pitchers** (`probable-pitchers.html`): today's starting pitchers from the MLB
  Stats API. As of 2026-09-03, game start times are converted from ET to the viewer's local timezone
  using `Intl` / `toLocaleTimeString` — EDT (UTC-4) March–November, EST (UTC-5) otherwise.
- **News** (`news.html`): 8 articles ranked by priority (breaking news: trades/injuries/call-ups/
  walk-offs score 2; standard score 1; fluff scores 0), then recency within tier. Each article shows
  a substantive body text below the headline. The MLB RSS feed's `<description>` tags are empty —
  the fetcher scrapes each article's `og:description` meta tag on first fetch instead (stored in
  `news_items.description`). AI summarization (claude-haiku-4-5-20251001) is attempted when
  `ANTHROPIC_API_KEY` is set; falls back to the scraped og:description otherwise.
  `data/news_fetcher.py` handles all RSS parsing, scraping, and DB storage.
- **Team cards**: season + trailing-30-day OPS/ERA/FIP/bullpen ERA, park factor with rank, remaining
  strength of schedule, expandable full-league rankings per metric.
- **Prospects**: MLB Pipeline Top 100 + org Top 30 rankings, real season stat lines from the MLB
  Stats API, 20-80 scouting grades, plus a self-computed Farm Rankings score.
- **Standings, Top Performers**: real, officially-sourced daily/season stats, not reconstructed from
  Statcast where an official number exists.

### Known issues / deferred

- **Pitch Locations chart** (`outings.html` and player page): may show RHH/LHH backwards vs. real
  Baseball Savant — not yet investigated, deferred.
- **`pull_statcast.py` game_type filter**: still missing — Spring Training/postseason contamination
  will recur at the start of a future season unless a real `game_type` filter is added.

### Pending large redesign (plan saved, not yet started)

A full visual overhaul plan exists (saved as a plan file in this Claude Code session). Covers:
batter spray chart (hits-only, HR markers, pull-side shading), Attack Direction consistency (one
number driving gauge + spray chart + label), bat speed real percentile, Swing Length replacing Swing
Path Tilt, percentile bar visual consistency (pitcher's centered badge for batters too), movement plot
static on L/R split, pitcher rubber-position visual, new Heatmaps page, FanGraphs xFIP/wRC+
integration (staged), Stuff+ redesign into a compact banner widget. None of these are live yet.

## Future aspirations (stated direction, largely NOT yet executed)

The long-term architecture, explicitly designed to minimize feature overlap so Tunneling+ and
Deception+ can eventually feed INTO an improved Stuff+ as inputs (a stacking/ensemble approach):

1. **Stuff+ drops delivery/mechanics features** — `release_extension`, `release_pos_x_as`,
   `release_pos_z`, `arm_angle` currently live on BOTH Stuff+ and Deception+ (confirmed still present
   in `stuff_features.py` as of this rewrite — not yet removed). The stated plan is these move to be
   Deception+-exclusive, since they're mechanics/delivery properties, not intrinsic pitch-shape
   properties. This session's r≈0.535 Stuff+/Deception+ correlation is direct evidence for why this
   matters.
2. **Stuff+ adds HAA (horizontal approach angle)** alongside its existing VAA residual — the two
   are both "direct properties of the pitch's trajectory at the plate," a natural pair. Not yet
   implemented (confirmed not in the current `NUMERIC_FEATURES` list).
3. **Tunneling+ and Deception+ themselves become input features to a next-generation, stacked
   Stuff+** — the actual end goal of the three-model separation. Not started.
4. **Fix `pull_statcast.py`'s missing `game_type` filter** (see "Recent data-quality work" above) —
   the Spring Training contamination will recur at the start of every future season otherwise.
5. Older, lower-priority open items: real FanGraphs-sourced season aggregates where currently
   approximated, a Location+/Command+ model (explicitly out of scope for Stuff+ by design, see
   above), pitcher-side UI placement for the Similar Players feature (backend supports it, no card
   location decided yet).

## Separate, unmerged thread: video-based delivery-timing extraction

A computer-vision pipeline (MediaPipe pose tracking + YOLO player detection) was built across several
past sessions attempting to recover leg-lift timing, time-in-set, and delivery length directly from
Statcast broadcast video — confirmed above as NOT obtainable any other way. Lives entirely outside
this project in an isolated temp directory
(`...\AppData\Local\Temp\claude\mlb_pitcher_isolation_investigation\`) with its own Python 3.11 venv,
and was never wired into `deception_features.py` — nothing in the live model reads its output.

Status as of its last session: validated screenshot-by-screenshot on individual reference pitches,
but foot-plant detection (a core phase-timing signal) was still unsolved, and no production-scale
batch run had been done. Real, hard-won lessons are documented in that directory's own notes if this
thread gets picked back up (MediaPipe L/R labels aren't temporally stable frame-to-frame, raw joint
height is an unreliable plant signal, native-fps-vs-extraction-stride index bugs, etc.) — not
restated here since this document is about the model that's actually live. **Not a priority** unless
the methodology conversation concludes the feature set genuinely needs delivery-timing data badly
enough to justify finishing it.
