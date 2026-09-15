# MLB Dashboard — Project Handoff

Last updated: 2026-09-15. Describes what is actually live today.

---

## What this project is

A personal MLB analytics dashboard: real Statcast pitch-level data (2015 through the current
in-progress 2026 season) feeding four original "+"-scaled models (Stuff+, Deception+, Tunneling+,
Location+) plus a full player/team/prospect browsing site. Built as a learning project and portfolio
piece for baseball analytics job applications — methodology rigor and honest validation matter here
beyond just shipping features.

**GitHub**: `github.com/JTaylor-Data/mlb-dashboard` (private). All work is on `master`. The repo
contains the full codebase — large training artifacts and the SQLite DB are gitignored.

---

## Architecture

```
pulls/  →  mlb_data.db  →  analytics/  →  api/  →  frontend/
(fetch)    (single SQLite) (compute)      (JSON)    (renders it)
```

- **`pulls/`** — one script per external data source: Statcast pitch-by-pitch (`pull_statcast.py`),
  players/games/schedule (`pull_players_games.py`), MLB Stats API sabermetrics/Savant leaderboard
  stats (`pull_sabermetrics.py`, `pull_advanced_stats.py`), injuries, prospects, pitcher bio
  (height/weight, needed by Deception+, not in Statcast).
- **`mlb_data.db`** — single SQLite file, gitignored (regenerable, ~4.2 GB as of 2026-09-15).
  **8.1 million pitches** covering 2015–2026. `pitches` is the foundation: one row per real pitch,
  everything else is a query or model on top of it. `player_advanced_stats` (Sprint Speed, OAA,
  real wRC+/xFIP/WAR/Barrel%) is genuinely external — not derivable from pitch-level data.
  `game_type` column added to `pitches` in September 2026; pull_statcast.py now filters
  `KEEP_GAME_TYPES = frozenset({"R"})` to prevent spring training / postseason contamination.
- **`analytics/`** — the compute layer: percentiles, splits, archetypes, and the four models.
- **`api/`** — FastAPI, one router per feature area. `api/warmup.py` pre-computes expensive
  league-wide pools at startup/daily. `api/scheduler.py` runs the full pull+warmup pipeline
  automatically at startup and every day at 6 AM.
- **`frontend/`** — plain HTML/JS/CSS, one JS file per page, hand-built SVG for every chart (no
  charting library). `frontend/js/api.js` holds shared utilities including `getJSON`,
  `prefetchInBackground`, and `window._prefetchCache`.
- **`models/`** — trained model artifacts (LightGBM `.txt` boosters + metadata/validation JSONs).
  **Current live versions**: `stuff_plus_v3_lgbm.txt` (2016–2026), `tunneling_v3.txt` (2015–2026),
  `location_plus_v1.txt` (2015–2026), `deception_plus_v1.txt` (2021–2025). All models were
  retrained in September 2026 on the full backfilled dataset. Legacy v2 artifacts and the
  pre-backfill backup copies have been removed.
- **`training/`** — offline model-fitting scripts (ext variants), run by hand, not part of the
  live request path. All scripts at `training/*.py` level; `training/tunneling_v3_artifacts/`,
  `training/tunneling_v2_artifacts/200ms/`, and `training/location_v1_artifacts/` contain
  precomputed artifacts the live API reads at runtime.
- **`pitcher_cv/`** — computer-vision pipeline for Deception+ v4 (see bottom of this document).

Run with `python -m uvicorn api.main:app --port 8010` (no `--reload`, kill+relaunch after any
backend `.py` edit). The server process can get into a stale state after a crash — kill it and
restart rather than trying to recover.

---

## The four models

All are "+"-scaled (100 = league average). All exclude pitcher identity and plate location/count as
features by design — the whole point is to grade pitch quality in a way that generalizes, not
memorize "this is a good pitcher" or reward hitting a corner.

### Stuff+ v3 — the pitch's own physical shape

**What it isolates**: velocity, movement, spin, release — independent of where the pitch was thrown
or the count.

**Model**: LightGBM (v3, `stuff_plus_v3_lgbm.txt`). **Training data: 2016–2026** (11 seasons;
2015 excluded — `spin_axis` is 0% populated in 2015). Leave-one-season-out CV.

**Features** (`analytics/stuff_features.py`): `release_speed`, `effective_speed`, `release_spin_rate`,
`spin_axis`, `hb` (arm-side-adjusted horizontal break), `ivb`, `release_extension`,
`release_pos_x_as`, `release_pos_z`, `arm_angle`, `vaa_residual` — plus categoricals `pitch_type`,
`p_throws`, `stand`, `season`.

**`vaa_residual`**: raw VAA is geometrically confounded with plate height. `fit_vaa_expectation()`
fits `vaa ~ plate_z + plate_z² + release_speed` across the league; the residual is "flatter or
steeper than physics predicts." A separate per-pitch-type height-adjustment exists on the player-page
percentile display (applied to the primary fastball type specifically) — these are different code
paths, not reconciled.

**Scale**: z-scored within season, small-sample pitch types shrunk toward their PITCH TYPE's own
league mean via empirical-Bayes shrinkage (K constant estimated fresh from each season's data, came
out tiny: 0.4–2.8 pitches — mainly matters for genuinely rare pitch types like eephus/screwball).

**Known limitation**: Knuckleball (~72) scores far below average by construction — slow, little
conventional movement — while missing the reason the pitch works (unpredictability).

**Year-over-year stability**: r² 0.58–0.67 across consecutive season pairs. By far the most stable
of the three models. Top individual seasons: Pete Fairbanks 2022 (124.2), Emmanuel Clase 2022
(122.0), Félix Bautista 2023 (121.5), Clase 2021 (119.9), Clase 2024 (118.4).

---

### Deception+ v1 — pitcher body and delivery mechanics

**What it isolates**: does something about a pitcher's body and release consistency predict good
outcomes, independent of pitch shape?

**Features** (`analytics/deception_features.py`): `arm_angle`, `release_extension`,
`release_pos_x_as`, `release_pos_z` (per-pitch raw — NOTE: all four are also Stuff+ features),
`arm_angle_std`, `release_extension_std`, `release_consistency` (season-long per-pitcher variability),
`height_in`/`weight_lb` (from `pitcher_bio` table — time-invariant per pitcher, a real limitation
for year-over-year work), `season`.

**Model**: LightGBM, Huber loss, leave-one-season-out CV 2021–2025. **Training data: 2021–2025**
(`arm_angle` was introduced by Statcast mid-2020; only 75.1% populated in 2020 — not enough to
extend further back). The live site scores 2025 only (hardcoded `REPORT_SEASON = 2025` in
`api/routers/deception_report.py`) — the model is season-agnostic, no season selector is wired up.

**Confirmed NOT obtainable from any accessible source**: delivery length, leg-lift timing, time held
in set, ball concealment — need frame-by-frame motion-capture data Statcast doesn't publish.

**Year-over-year stability**: r² 0.17–0.34 across consecutive season pairs — least stable of the
four. Correlates moderately with Stuff+ (r≈0.535), weakly with Tunneling+ (r≈0.12–0.19).

**Key diagnostic finding**: arm_angle and release_pos_z correlate at r=0.79 league-wide. A naïve read
of arm_angle's SHAP alone makes extreme submarine deliveries look penalized, but the COMBINED
(arm_angle + release_pos_z) SHAP curve shows true submarine (~-60°) is strongly rewarded (+14 pts),
over-the-top (~70°) is penalized (-3 to -6 pts), and the normal range is roughly flat. Tyler Rogers's
2025 SHAP: arm_angle -3.5 pts, release_pos_z +17.7 pts, overall 133.5 (elite).

**Top/bottom individual seasons (2021–2025)**: Top 5 — Bautista 2023 (146.2), deGrom 2021 (140.8),
Bautista 2022 (137.6), T. Rogers 2021 (136.2), T. Rogers 2025 (133.5). Bottom 5 — N. Murdock 2025
(54.2), J. Reed 2022 (55.1), M. Bumgarner 2023 (57.2), Ty Blach 2024 (65.2), Ty Madden 2024 (65.9).

**Open methodology questions**: (1) Why so much noisier YoY than Stuff+ — real or fixable?
(2) Should the four Stuff+-shared features move off one model? (3) Should arm_angle / release_pos_z
be combined given r=0.79? (4) Is season-level aggregation the right grain for `_std` features?

---

### Tunneling+ v3 — how indistinguishable pitches are from each other

**What it isolates**: do pitches look alike out of the hand and separate late near the plate?

**Methodology**: for every pair of pitch types (min 100 pitches per type), compute real pairwise mean
distance (every real instance of A vs. every real instance of B — not centroid-to-centroid) at three
points: release, 23.5 ft from plate (decision point), and plate. `decision_similarity_plus` (100 ×
league-avg tunnel distance / this pair's tunnel distance) and `plate_separation_plus` (100 × this
pair's plate distance / league-avg plate distance) combine via geometric mean into the pair's
`tunneling_plus`. Overall score = usage-weighted average across qualifying pairs, with FF/SI pairs
double-weighted (the hitter's read is built against the fastball).

**Model**: LightGBM (`tunneling_v3.txt`). **Training data: 2015–2026** (12 seasons; full backfill).
The commit-point table (`training/tunneling_v2_artifacts/200ms/pitcher_season_commit_point.csv`)
covers all 12 seasons (9,348 pitcher-seasons, regenerated at training time). Precomputed offline;
live API reads precomputed scores.

**v3 changes from v1**: improved pairwise distance calculation, the full-pairwise bug fix (centroid
distance silently erased real command variance — leaguewide average tunnel distance nearly tripled
once measured correctly), `release_similarity_plus` component dropped (made the model worse —
sequential z-score dropped from 9.06 to 1.60 when included).

**Validation**: split-half reliability ~0.85–0.91, sequential same-at-bat z≈135.5 (substantially
improved over live model's 84.7 due to expanded training data). Year-over-year r² 0.31–0.53.
Correlates weakly with both other models (r≈0.12–0.19), confirming genuine distinctness.

**Top/bottom individual seasons**: Top — Kevin Ginkel 2025 (113.5), Cionel Pérez 2021 (113.2),
Garrett Cleavinger 2022 (112.3), Alex Vesia 2023/2024 (112.1/111.3). Bottom — Matt Bowman 2024
(82.9), Rich Hill 2021/2022/2023 (84.8/86.6/87.5).

---

### Location+ v1 — command quality

**What it isolates**: where a pitch is thrown relative to the strike zone, independent of what the
pitch is. Fills the deliberate gap left by Stuff+ and Tunneling+ (both exclude plate location).

**Features** (`analytics/location_features.py`): `plate_x`, `plate_z`, `sz_top`, `sz_bot`, count
state — with a sinker-vs-RHH interaction term confirmed to matter in the ablation.

**Model**: LightGBM (`location_plus_v1.txt`). **Training data: 2015–2026** (12 seasons; full
backfill). Validated in `training/validate_location_plus_ext.py` + artifacts in
`training/location_v1_artifacts/`.

**Status**: live in the API, displayed on the player page and probable pitchers cards.

---

## Future model architecture (stated direction, largely not yet executed)

1. **Stuff+ drops delivery/mechanics features** — `release_extension`, `release_pos_x_as`,
   `release_pos_z`, `arm_angle` live on BOTH Stuff+ and Deception+ right now. Plan: these move to
   Deception+-exclusive. The r≈0.535 Stuff+/Deception+ correlation is the evidence this matters.
2. **Stuff+ adds HAA (horizontal approach angle)** alongside the existing VAA residual. Not yet in
   `NUMERIC_FEATURES`.
3. **Tunneling+ and Deception+ become input features to a next-generation stacked Stuff+**. Not started.

---

## Data pipeline

**Statcast**: `pulls/pull_statcast.py`. Weekly chunked pull (Mon→Sun windows). Filters
`KEEP_GAME_TYPES = frozenset({"R"})` — regular season only. Spring training / postseason / all-star
games are dropped at ingest. The `game_type` column is stored in `pitches` for audit purposes.

**Backfill**: 2015–2020 data was backfilled in September 2026 using `pulls/backfill_orchestrator.py`
(resumable, checkpointed). 2021–2026 was already clean. Verification confirmed zero non-R rows in
all seasons.

**Refresh cadence**: `api/scheduler.py` runs the full pull+warmup pipeline at server startup and
daily at 6 AM. The frontend's "Refresh data" button triggers a manual pull.

---

## Site pages

### Player tab (sub-nav: Overview / Splits / Outing / Trends / Career / Compare)

- **Overview** (`index.html`): real Statcast pitch-level breakdown, percentile rankings (Baseball
  Savant qualifying thresholds: 2.1 PA/team-game batters, 1.25 for pitchers), Pitch Locations
  heatmap, batter swing archetype cards (Attack Angle/Direction, Swing Length, Swing Tilt),
  Similar Players (nearest neighbor in percentile-space). Movement plot, rubber-position visual.
  - Percentile box sits ABOVE the movement plot, BELOW the hero banner.
  - Last 5 Outings card has no scrollbar (overflow: hidden).
- **Outing Breakdown** (`outings.html`): single-game breakdown for any pitcher.
  - *Hero*: headshot + team logo + player name + matchup date/opponent + Max EV + Max Velo.
  - *Table 1 — Usage, Velocity & Strikes*: Pitch | n | % | Yr% | vs RHH | vs LHH | Velo | ±Velo | K |
    SW | CS | Whiff (x/x) | W% | CS+W (x/x) | CS+W% — plus ALL row.
  - *Table 2 — Movement & Spin*: Pitch | Drop (in) | ±Drop | IVB | ±IVB | HB | ±HB | Spin | ±Spin —
    plus ALL row. Drop uses gravity formula: `drop = 11.4 - (pfx_z_feet × 12)`.
  - *Pitch Sequence by At-Bat*: at-bats grouped by inning, each pitch shown as a colored dot (by
    pitch type) with hover tooltip (count, description), result badge (K/BB/HR/1B/etc) at right.
- **Compare** (`compare.html`): side-by-side player comparison.
- **Batter pitch type table**: includes **Faced%** column (% of that pitcher's total pitches that
  were this type).

### Other pages

- **Probable Pitchers** (`probable-pitchers.html`): today's starters from the MLB Stats API. Shows
  Stuff+/Tunneling+/Location+/Deception+ score blocks per pitcher, recent starts summary, Matchup
  of the Day feature.
- **Top Performers** (`top-performers.html`): pitchers and batters toggled separately. Separate top
  10 pitchers AND top 10 batters. Walk-Off panel above the leaderboard.
- **News** (`news.html`): 8 articles ranked by priority + recency. AI summarization with
  claude-haiku-4-5-20251001 when `ANTHROPIC_API_KEY` is set; fallback to scraped og:description.
- **Team Cards** (`teams.html`): season + trailing-30-day OPS/ERA/FIP/bullpen ERA, park factor with
  rank, remaining strength of schedule.
- **Prospects** (`prospects.html`): MLB Pipeline Top 100 + org Top 30, real season stats, 20-80
  scouting grades, Farm Rankings score.
- **Playoffs** (`playoffs.html`, `playoffs-bracket.html`, `playoffs-roster.html`): bracket and rosters.
- **Tunneling 3D** (`tunnel3d.html`): Three.js 3D visualization of pitch tunneling.
- **Location Model** (`location-model.html`): Location+ visualization.

---

## Startup caching / prefetch

Player page loads and caches first on launch. After the player page's `init()` completes, it fires
background prefetches for probable pitchers and top performers for the current date, populating
`window._prefetchCache`. Those pages check the cache on init so they render instantly when navigated
to. Everything else loads on click.

Cache is implemented in `frontend/js/api.js` (`prefetchInBackground()`, `window._prefetchCache`).
Server-side cache: `api/cache.py` (`ttl_cache`, 12-hour TTL), used by all expensive per-season
league-wide pool computations.

---

## Stadium photos

`frontend/img/stadiums/{venue_id}.jpg` — one per ballpark, keyed by MLB venue ID. **Yankee Stadium
and T-Mobile Park photos are still the originals from the initial download** — new high-home-view
photos (looking out at the outfield) are needed for both. This is a pending task.

---

## Known issues / deferred

- **Pitch Locations chart** (player page and outing): may show RHH/LHH backwards vs. Baseball
  Savant orientation — not yet investigated, deferred.
- **Deception+ season selector**: live site scores season=2025 only, hardcoded (`REPORT_SEASON = 2025`
  in `api/routers/deception_report.py`). The model itself is season-agnostic and covers 2021–2025;
  no UI selector is wired up yet.
- **No league-wide Deception+ table function in the runtime**: `deception_model.py` scores one
  pitcher at a time on demand; a league-wide scorer was built for diagnostics but not merged into
  the live module.

---

## Pending tasks (as of 2026-09-15)

1. **GitHub push**: `git push origin master` — push all accumulated changes to remote. Last force-
   push was blocked by auto mode; run manually from the terminal.
2. **Ballpark photos**: new high-home-view photos for Yankee Stadium (venue_id=3313) and T-Mobile
   Park (venue_id=680), looking out at the outfield.
3. **Prospect top performers stat spacing**: space out stats/outings the same way as the top
   performers tab (the `statCells` grid layout).

---

## Pending large redesign (plan saved, not yet started)

A full visual overhaul plan exists (saved as a Claude Code plan file). Covers:
batter spray chart (hits-only, HR markers, pull-side shading), Attack Direction consistency (one
number driving gauge + spray chart + label), bat speed real percentile, Swing Length replacing Swing
Path Tilt, percentile bar visual consistency (pitcher's centered badge for batters too), movement
plot static on L/R split, pitcher rubber-position visual, new Heatmaps page, FanGraphs xFIP/wRC+
integration (staged), Stuff+ redesign into a compact banner widget. None of these are live yet.

---

## Codebase notes (as of 2026-09-15 cleanup)

**What was cleaned up**: redundant `sys.path.insert()` boilerplate removed from all 15 API router
files (main.py sets the path once at startup, making these redundant); duplicate `_to_plus_scale()`
function removed from `deception_model.py` and `tunneling_v3_model.py` (both now import
`to_plus_scale` from `analytics/location_features`); inline `math.isnan()` checks in `players.py`
replaced with the existing `_clean()` helper; redundant second `SELECT ... FROM players` query
removed from the outing detail path in `players.py` (now uses already-fetched `name`/`team_id`
variables).

**Dead file candidates** (not yet deleted — require explicit confirmation):
- `data/mlb.db` — zero-byte empty file, the real DB is `mlb_data.db` at project root
- `models/pre_backfill_backup_20260914/` — pre-promotion safety backup, superseded
- `models/variants/` — ablation models from development, no live references
- `models/pitching_plus_ext/` — the `_ext` named copies used during training; promoted models
  are in `models/` root; no live code loads these
- `_restored_training/` — old training scripts recovered from git, superseded by `training/` ext
  variants; the trained models already exist in `models/`
- `training/pitching_plus_ext_artifacts/` — intermediate build artifacts, regenerable if needed
- `training/stuff_v3_artifacts/` — intermediate artifacts, not referenced by live code
- `training/tunneling_v3_150ms_artifacts/` — experimental, zero references anywhere
- `pulls/backfill_orchestrator.py`, `analytics/completeness_audit.py`, `pulls/backfill_years.py` —
  one-time maintenance CLI scripts; backfill is complete

**Shared utilities**:
- `_clean(v)` in `api/routers/players.py` — NaN/NaT → None, numpy scalar → Python native
- `to_plus_scale(values, mean, std)` in `analytics/location_features.py` — the single canonical
  `100 - 10 * (v - mean) / std` implementation; all four model runtime modules use this

---

## Separate, unmerged thread: video-based delivery-timing extraction (Deception+ v4 CV)

A computer-vision pipeline (`pitcher_cv/deception_v4/`) uses MediaPipe pose tracking + YOLO player
detection to recover leg-lift timing, time-in-set, and delivery length directly from Statcast
broadcast video — features confirmed NOT obtainable any other way. The v4 pipeline (`d4_pipeline.py`,
`d4_phases.py`, `d4_timing.py`, etc.) is structurally complete but not wired into `deception_features.py`
— nothing in the live model reads its output.

Real hard-won lessons (MediaPipe L/R labels aren't temporally stable frame-to-frame, raw joint
height is an unreliable plant signal, native-fps-vs-extraction-stride index bugs) are documented in
the pipeline scripts. Foot-plant detection (a core phase-timing signal) was still unsolved at last
session. Not a priority unless the methodology conversation concludes the feature set genuinely needs
delivery-timing data badly enough to justify finishing it.
