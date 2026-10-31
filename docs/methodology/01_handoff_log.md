# MLB Dashboard — Project Handoff

Last updated: 2026-10-05 (codebase-audit branch — all stages complete).

## WHAT CHANGED IN THIS SESSION (2026-10-05 codebase-audit)

### Branch: codebase-audit (not yet merged to master — owner reviews then merges)

Tags: `pre-audit` = d573b90 (baseline before any changes), `post-audit` = see below.

### Changes made (zero behavior change — refactors and cleanup only)

1. **`config/season_phase.py`** (NEW) — single source for SEASON_PHASE constant (REGULAR_SEASON/POSTSEASON/OFFSEASON). Set via env var `SEASON_PHASE`, defaults to POSTSEASON. Helpers: `is_postseason()`, `skip_team_stat_pulls()`, `cache_ttl_live_seconds()`, etc. `api/pull.py` POSTSEASON_MODE now derived from this instead of hardcoded True.

2. **`config/qualifiers.py`** (NEW) — canonical per-model min-pitch thresholds. All leaderboard/scoring qualifier constants live here.

3. **`api/routers/deception_report.py`** — REPORT_SEASON fixed from hardcoded 2025 → CURRENT_SEASON (2026). `_load_scores` replaced `@lru_cache` with `@ttl_cache_all_args()` so it participates in `/api/admin/refresh-cache`.

4. **`api/main.py`** — added `/api/admin/cache-stats` endpoint (admin-token protected).

5. **`api/routers/players.py`** — 3 of 4 `iterrows()` calls vectorized. 4th kept (downstream uses pandas Series attribute access).

6. **`Archive/`** (NEW directory) — non-live training/diagnostic/experimental code moved here with `git mv`. See `Archive/MOVED_FILES.csv` for full provenance. Notable: `diagnostics/location_v3/` was moved then REVERTED — it is live (read by `location_report.py` at runtime).

7. **`ARCHITECTURE.md`** (NEW at repo root) — folder tree, data flow, active model table, router index, cache docs.

### Deferred items (owner input needed — see docs/methodology/02_open_decisions.md)

- **ND-AUDIT-02/DEF-03/04**: `probable_pitchers.py` and `career.py` import `tunneling_v3_model` and the old `location_model` instead of active versions. Deferred — owner to decide if/when to fix.
- **ND-AUDIT-03**: `/api/splits?player_id=543037&season=2026&split_by=pitch_type` 500 error. Not investigated — not caused by this audit.

### Rollback instructions

```bash
git checkout master   # returns to pre-audit state immediately
# OR to revert codebase-audit after it's merged:
git revert <merge-commit-sha>
# OR restore from bundle backup:
git clone ..\pre-audit-backup.bundle mlb_dashboard_restored
```

---

## WHAT CHANGED IN THE PREVIOUS SESSION (2026-10-04 merge)

### Active model versions (updated)
- **Stuff+**: v4 (was v3). Hash 0addb8e9. Training seasons 2016-2026. See models/deception_plus_v2b_meta.json.
- **Location+**: v3 (was v2). Training seasons 2015-2019, 2021-2025. Models: location_plus_v3_*.txt.
- **Deception+**: v2b (was v1). Features: release_pos_x_as, release_pos_z, release_extension, arm_angle. Status: limited_evidence. Parquet-based scoring. `DECEPTION_ACTIVE_VERSION = "v2b"` in config/model_policy.py.
- **Tunneling+**: unchanged (out of scope). Byte-identity verified.

### New infrastructure
- **Score cache parquets**: Stuff+/Loc+/Tunneling+ pre-scored for all seasons, injected at warmup (api/warmup.py). Scripts: scripts/rescore_models.py. Weekly rescore: api/scheduler.py.
- **Percentile pool cache**: analytics/percentile_cache.py persists complete-season (< 2026) pools to scores/percentile_pools/. Reduces past-season first-request cost from ~7s to ~360ms. Build tool: tools/build_percentile_pools.py.
- **CURRENT_SEASON = 2026** added to config/model_policy.py as single source of truth.

### Open items from this merge
See docs/methodology/02_open_decisions.md — specifically:
- ND-DEC-01: /api/deception/meta returns 404 for v2b (needs meta endpoint fix)
- ND-01: annual model retrain reminder (season feature)
- ND-02/03: Deception+ arm_angle coverage 2021+ only

### Percentile cache: how to freeze a season
When season 2026 ends (after World Series), bump `CURRENT_SEASON` to 2027 in `config/model_policy.py`, then run:
```bash
python tools/freeze_season.py 2026
```
This builds all 6 pool files for 2026 and saves to scores/percentile_pools/2026/.

---

Last updated (original): 2026-09-28. Describes what is actually live today.

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
  **Note**: `data/mlb.db` is an empty zero-byte file — the real DB is `mlb_data.db` at project root.
- **`analytics/`** — the compute layer: percentiles, splits, archetypes, and the four models.
- **`api/`** — FastAPI, one router per feature area. `api/warmup.py` pre-computes expensive
  league-wide pools at startup/daily. `api/scheduler.py` runs the full pull+warmup pipeline
  automatically at startup and every day at 6 AM.
- **`frontend/`** — plain HTML/JS/CSS, one JS file per page, hand-built SVG for every chart (no
  charting library). `frontend/js/api.js` holds shared utilities including `getJSON`,
  `prefetchInBackground`, and `window._prefetchCache`.
- **`models/`** — trained model artifacts (LightGBM `.txt` boosters + metadata/validation JSONs).
  **Current live versions**: `stuff_plus_v3_lgbm.txt` (2016–2026), `tunneling_plus_v2_{xswing/xcs/xwhiff/xfoul/bip}.txt`
  (2015–2026, promoted 2026-09-21), `location_plus_v2_{xswing/xcs/xwhiff/xfoul/bip}.txt` (2015–2026,
  promoted 2026-09-21), `deception_plus_v1.txt` (2020–2026; 7 season categories confirmed in booster
  splits — 2015-2019 absent after arm_angle dropna, 2020 ~75% populated and retained). Old single-booster files `tunneling_v3.txt`
  and `location_plus_v1.txt` remain in `models/` for reference — the live API no longer loads them.
  All models were retrained in September 2026 on the full backfilled dataset.
- **`training/`** — offline model-fitting scripts (ext variants), run by hand, not part of the
  live request path. All scripts at `training/*.py` level; `training/tunneling_v3_artifacts/`,
  `training/tunneling_v2_artifacts/200ms/`, and `training/location_v1_artifacts/` contain
  precomputed artifacts the live API reads at runtime. `training/v2_artifacts/` contains the
  v2 candidate model artifacts (see V2 R&D section below).
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
`spin_axis`, `hb` (arm-side-adjusted horizontal break), `ivb`, `vaa_residual` — plus categoricals
`pitch_type`, `p_throws`, `stand`, `season`. **Note**: `release_extension`, `release_pos_x_as`,
`release_pos_z`, and `arm_angle` have been removed from Stuff+ and are now Deception+-exclusive
(confirmed in `analytics/stuff_features.py` module docstring lines 19–22).

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

**Year-over-year stability**: Pearson r = 0.907 (consecutive season pairs, pooled). By far the most
stable of the four models. Top individual seasons: Pete Fairbanks 2022 (124.2), Emmanuel Clase 2022
(122.0), Félix Bautista 2023 (121.5), Clase 2021 (119.9), Clase 2024 (118.4).

---

### Deception+ v1 — pitcher body and delivery mechanics

**What it isolates**: does something about a pitcher's body and release consistency predict good
outcomes, independent of pitch shape?

**Features** (`analytics/deception_features.py`): `arm_angle`, `release_extension`,
`release_pos_x_as`, `release_pos_z` (per-pitch raw — these are now Deception+-exclusive; removed
from Stuff+ in the current codebase), `arm_angle_std`, `release_extension_std`,
`release_consistency` (season-long per-pitcher variability), `height_in`/`weight_lb` (from
`pitcher_bio` table — time-invariant per pitcher, a real limitation for year-over-year work),
`season`.

**Model**: LightGBM, Huber loss. **Training data: 2020–2026** (September 2026 retrain extended
to 2020 — arm_angle ~75% populated in 2020 after dropna, retained; 2015–2019 absent. Booster
confirms 7 season categories. Original v1 used 2021–2025; meta.json updated to reflect retrain). The live site scores 2025 only (hardcoded `REPORT_SEASON = 2025` in
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

### Tunneling+ v2 — how indistinguishable pitches are from each other (LIVE — promoted 2026-09-21)

**What it isolates**: do pitches look alike out of the hand and separate late near the plate?

**Methodology**: for every pair of pitch types (min 100 pitches per type), compute real pairwise mean
distance (every real instance of A vs. every real instance of B — not centroid-to-centroid) at three
points: release, 23.5 ft from plate (decision point), and plate. `decision_similarity_plus` (100 ×
league-avg tunnel distance / this pair's tunnel distance) and `plate_separation_plus` (100 × this
pair's plate distance / league-avg plate distance) combine via geometric mean into the pair's
`tunneling_plus`. Overall score = usage-weighted average across qualifying pairs, with FF/SI pairs
double-weighted (the hitter's read is built against the fastball).

**Model**: Five LightGBM classifiers (xSwing%, xCS%, xWhiff%, xFoul%, BIP×EV), combined via fixed
run-value table → per-pitch xRV → to_plus_scale. Booster files: `tunneling_plus_v2_{xswing/xcs/xwhiff/xfoul/bip}.txt`.
**Training data: 2015–2026** (12 seasons; full backfill). The commit-point table
(`training/tunneling_v2_artifacts/200ms/pitcher_season_commit_point.csv`) covers all 12 seasons.
Scored live each request via `analytics/tunneling_v2_model.py`.

**Why v2**: the previous live model (`tunneling_v3.txt`, single regressor) had a sign-flip
(ERA r = −0.218, xFIP r = −0.348 — higher score = worse pitcher). v2 fixes this:
ERA r = +0.182, xFIP r = +0.337. The five-classifier decomposed architecture naturally produces
xRV in the correct direction (lower = bad outcomes), eliminating the sign ambiguity.

**Promoted**: 2026-09-21. Booster files copied to `models/`, wired into `api/routers/tunneling_report.py`
(import changed from `tunneling_v3_model` to `tunneling_v2_model`). Old `tunneling_v3.txt` remains
in `models/` for reference but is no longer loaded by any live code.

**Validation**: split-half reliability ~0.85–0.91, sequential same-at-bat z≈135.5 (substantially
improved over live model's 84.7 due to expanded training data). Year-over-year r² 0.31–0.53.
Correlates weakly with both other models (r≈0.12–0.19), confirming genuine distinctness.

**Top/bottom individual seasons**: Top — Kevin Ginkel 2025 (113.5), Cionel Pérez 2021 (113.2),
Garrett Cleavinger 2022 (112.3), Alex Vesia 2023/2024 (112.1/111.3). Bottom — Matt Bowman 2024
(82.9), Rich Hill 2021/2022/2023 (84.8/86.6/87.5).

---

### Location+ v2 — command quality (LIVE — promoted 2026-09-21)

**What it isolates**: where a pitch is thrown relative to the strike zone, independent of what the
pitch is. Fills the deliberate gap left by Stuff+ and Tunneling+ (both exclude plate location).

**Features**: `plate_x`, `plate_z`, `balls`, `strikes`, `pitch_type`, `stand`, `season` (7 total).

**Model**: Five LightGBM classifiers (xSwing%, xCS%, xWhiff%, xFoul%, BIP×EV), combined via fixed
run-value table → per-pitch xRV → to_plus_scale. Booster files:
`location_plus_v2_{xswing/xcs/xwhiff/xfoul/bip}.txt`. **Training data: 2015–2026** (12 seasons).
Scored live each request via `analytics/location_v2_model.py`.

**Classifier AUCs** (LOSO CV): xSwing% 0.848, xCS% 0.981, xWhiff% 0.754, xFoul% 0.621.

**Why v2**: the previous live model (`location_plus_v1.txt`, single regressor) had a sign-flip
(ERA r = −0.257, xFIP r = −0.507) AND a stability collapse (YoY r collapsed to r=0.009 for
2024→2025, std=3.07). v2 fixes both: ERA r = +0.208, xFIP r = +0.349, std = 9.99.

**Promoted**: 2026-09-21. Booster files copied to `models/`, wired into `api/routers/location_report.py`
(import changed from `location_model` to `location_v2_model`). Old `location_plus_v1.txt` remains
in `models/` for reference but is no longer loaded by any live code.

---

## Diagnostic studies

### Stuff+ feature reconciliation (Part 0) — confirmed 2026-09-21

**Feature list confirmed**: `release_extension`, `release_pos_x_as`, `release_pos_z`, `arm_angle`
are **absent** from live Stuff+ (`analytics/stuff_features.py` `ALL_FEATURES`). These four are
Deception+-exclusive. Confirmed Case A (mechanics features fully separated between models).

**Stuff+ × Deception+ cross-correlation** (Part 0b — run 2026-09-21): Pearson r = **0.396** across
2,868 pitcher-seasons (2021–2025). Moderate correlation — models share some signal (which is
expected: good pitchers throw well across all dimensions) but are not redundant. By-season range:
r = 0.32–0.44. Artifact: `training/v2_artifacts/part0b_stuff_deception_correlation.json`.

### Tunneling+ v2 sequential same-at-bat test (Part 4) — **superseded; see Part A below**

Original test bucketed pitches by full blended xRV (which embeds the model's own xWhiff% prediction
as one of five inputs), producing z = 180.1. This test was close to circular and not a genuine
validation of the lag-geometry signal. Artifact: `training/v2_artifacts/tunneling_v2_sequential_test.json`.

### Tunneling+ v2 corrected sequential test — SHAP geometry isolation (Part A) — 2026-09-21

**Replaces the Part 4 result.** Isolated the xWhiff% classifier's SHAP contributions by feature
group on a 200K-row sample to test whether the lag-geometry features specifically predict real whiff,
independent of pitch-type information.

| Score | z (high − low) | Interpretation |
|---|---|---|
| Lag-1 geometry SHAP (`commit_distance`, `plate_distance`, `shape_break`) | **−1.706** | **Null** — geometry features not predictive of real whiff |
| All-lag geometry SHAP (6 slots × 3 features) | **+1.635** | **Null** — extending to all lags recovers nothing |
| Pitch-type lag SHAP (confound control: `lagN_pitch_type` for N=1–6) | **+18.622** | **Real signal** — pitch-type sequencing is genuine |
| Prior Part 4 full-xRV (not informative) | 180.1 | Near-circular |

**Finding**: The tunneling geometry features (commit/plate/shape distances) show no independent
predictive signal for real whiff rates (z ≈ ±1.7). The model's discriminating power in xWhiff%
comes entirely from pitch-type sequencing (z = 18.6), not geometric measurement. The prior
z = 180.1 result was an artifact of the model's own xWhiff% prediction being embedded in the
xRV used for bucketing. **Tunneling+ v2 stays live** — this is a validation-quality diagnostic, not
a promotion decision, and the ERA/xFIP sign-flip fix that motivated v2 remains valid.

Artifact: `training/v2_artifacts/part_a_tunneling_shap_geometry.json`.

### Tunneling+ v2 full geometry isolation — all 5 classifiers (Part A2) — 2026-09-21

Extended the corrected geometry isolation to all five classifiers (200K sample each; 30K for BIP).

| Classifier | lag1-geom z | alllag-geom z | pt-lag z | Assessment |
|---|---|---|---|---|
| xSwing% | **+73.969** | **+153.024** | +28.139 | GEOMETRY REAL — strong correct direction |
| xCS% | **+40.566** | **+58.378** | +20.694 | GEOMETRY REAL — strong correct direction |
| xWhiff% | −1.706 | +1.635 | **+18.622** | GEOMETRY NULL — pitch-type only |
| xFoul% | **−21.760** | **−44.573** | −2.055 | GEOMETRY WRONG DIRECTION |
| BIP type×EV | **+9.361** | **+16.557** | +4.963 | GEOMETRY WEAK-REAL (BIP prob); δRE null (t≈1.4) |

**Finding**: Geometry predicts pitch-selection decisions (swing/take) strongly and BIP probability
weakly. It is null for whiff prediction and inverts for foul prediction. Contact quality (δRE on
BIP events) is uncorrelated with geometry (t≈1.4–1.6). The model's geometry features capture whether
hitters are fooled into a wrong decision, not what happens when they make contact.

Artifact: `training/v2_artifacts/part_a2_shap_remaining.json`.

### Sequencing+ v1 diagnostic (Part B) — 2026-09-21

Tested a sequencing-only model that drops all 18 lag-geometry features and adds four explicit
sequencing features (`same_as_lag1`, `pitch_n_in_pa`, `n_distinct_prior_types`, `arsenal_entropy`).
Same 5-classifier decomposed architecture. LOSO CV on 2021-2025, full validation vs Tunneling+ v2.

**Key findings:**

| Metric | Seq+ v1 | Tunneling+ v2 | Assessment |
|---|---|---|---|
| YoY stability (pooled r) | **0.030** | ~0.40–0.55 | **FAIL — essentially random** |
| Split-half reliability | 0.946 | ~0.85–0.91 | PASS |
| xRV vs real whiff% | **−0.138** | (positive) | **FAIL — wrong direction** |
| xSwing% LOSO AUC | 0.585 | 0.711 | −0.126 |
| xCS% LOSO AUC | 0.643 | 0.742 | −0.099 |
| xWhiff% LOSO AUC | 0.592 | 0.649 | −0.057 |
| xFoul% LOSO AUC | 0.582 | 0.570 | +0.012 (only gain) |
| ERA / xFIP r | **−0.049 / −0.044** | −0.214 / −0.329 | FAIL — near-zero vs v2 |

Note: ERA correlations are negative because higher plus score = better pitcher = lower ERA. Relative magnitude is the meaningful comparison.

**Verdict**: Sequencing+ v1 is NOT a viable replacement for Tunneling+ v2. Dropping geometry
collapses YoY stability (0.030 vs 0.692), kills ERA/xFIP signal (−0.049 vs −0.214), and inverts
outcome validity. One genuine finding: v2 xFoul% AUC = 0.499 on full corpus (coin-flip) — consistent
with the SHAP finding that geometry inverts for foul prediction. Seq+ gets xFoul% AUC = 0.591.
The correct v3 direction is targeted geometry removal: drop geometry from xFoul% only.

Artifact: `training/v2_artifacts/sequencing_plus_v1_diagnostic.json`.

### Deception+ feature-set audit (Part 5)

Collinearity, SHAP feature importance, and bio-feature analysis. No changes.

**Collinearity**: arm_angle × release_pos_z r = +0.812 (confirmed); arm_angle × release_pos_x_as
r = −0.546; arm_angle_std × release_consistency r = +0.575; height × weight r = +0.403.

**SHAP feature importance** (50k-row sample, season 2025):
- Raw delivery (arm_angle, release_extension, release_pos_x_as, release_pos_z): **42%** of total signal
- Variability (_std features + release_consistency): **38%** — nearly as important as raw delivery
- `release_consistency` is the single most important feature (mean |SHAP| 0.00096)
- Bio (height_in, weight_lb): **15%** — non-negligible but not dominant; mean SHAP ≈ 0 (no systematic direction)
- Season: 5%

**Rolling window recommendation**: defer. Season-level aggregation is the honest choice and
aligns with Deception+'s year-over-year framing.

Artifact: `training/v2_artifacts/deception_audit.json`.

### Deception+ under decomposed-classifier architecture (Part 6)

Applied the 5-classifier decomposed architecture to Deception+ features. 2021–2025 LOSO CV.

**Classifier AUCs** (all near coin-flip — as expected for indirect mechanics features):

| Classifier | Deception+ decomposed | Location+ v2 reference |
|---|---|---|
| xSwing% | 0.512 | 0.848 |
| xCS% | 0.520 | 0.981 |
| xWhiff% | 0.529 | 0.754 |
| xFoul% | 0.516 | 0.621 |

**Comparison vs live v1** (both scored on same 2021–2025 corpus, 2,868 pitcher-seasons each):

| Metric | Live v1 (Huber regressor) | Decomposed v1 |
|---|---|---|
| ERA r | +0.564 | +0.311 |
| xFIP r | +0.470 | +0.480 |
| YoY Pearson r (pooled) | 0.472 | **0.783** |
| Stuff+ cross-correlation | r = 0.396 | r = 0.524 |

The decomposed architecture dramatically improves YoY stability (0.472 → 0.783) but at the cost
of ERA signal (0.564 → 0.311). The reason: delivery mechanics are effectively time-invariant
per pitcher (arm slot doesn't change season to season) — so the decomposed model is highly
stable but its xRV predictions reflect pitcher identity more than true pitch-outcome contribution.
The live Huber regressor trades some stability for stronger ERA signal, which is the right call.

Deception+ v1 x decomposed correlation: r = 0.507 — they agree on about half the signal and
diverge on the rest. Decomposed is not an improvement overall. **Live v1 remains correct.**

Artifact: `training/v2_artifacts/deception_decomposed_v1_diagnostic.json`.

---

## V2 Model R&D — completed and promoted 2026-09-21

Full validation study covering four parts. All artifacts saved under `training/v2_artifacts/`.
**Location+ v2 and Tunneling+ v2 promoted to live on 2026-09-21** (see individual model sections above).

### Location+ v2 candidate

**Architecture**: same five-classifier decomposed design (xSwing%, xCS%, xWhiff%, xFoul%,
BIP-type×EV) combined via fixed run-value table → per-pitch xRV → to_plus_scale aggregation.
**Features**: `plate_x`, `plate_z`, `balls`, `strikes`, `pitch_type`, `stand`, `season`.

**Classifier AUCs** (LOSO CV, 2015–2025):
- xSwing%: mean AUC = **0.848**
- xCS%: mean AUC = **0.981**
- xWhiff%: mean AUC = **0.754**
- xFoul%: mean AUC = **0.621**

**Key stats**: std = 9.99, 8,247 pitcher-seasons.
**ERA/xFIP**: ERA r = +0.208, xFIP r = +0.349 (both positive — correct direction, compared to v1's
negative values). **Score files**: `training/v2_artifacts/location_plus_v2_*.parquet` + `_meta.json`.

### Tunneling+ v2 candidate

**Architecture**: same five-classifier design. **Features**: lag1–lag6 commit distance, plate
distance, shape break, + lag1–lag6 pitch type, current pitch type, p_throws, stand, season.

**Classifier AUCs** (LOSO CV, 2015–2025):
- xSwing%: mean AUC = **0.711**
- xCS%: mean AUC = **0.742** (note: ~0.028 is pitch-type confound — see Part B)
- xWhiff%: mean AUC = **0.649**
- xFoul%: mean AUC = **0.570**

**Key stats**: std ≈ 10, 7,888 pitcher-seasons.
**ERA/xFIP**: ERA r = +0.182, xFIP r = +0.337 (correct direction vs. v3's negative values).
**Score files**: `training/v2_artifacts/tunneling_plus_v2_*.parquet` + `_meta.json`.

### xCS% pitch-type confound (Part B)

Tunneling+ v2's xCS% classifier AUC includes a ~0.028 contribution from pitch-type columns
(pitch_type and its 6 lag columns). Removing all pitch-type info drops AUC from 0.741 → 0.713.
Most of the signal (96%) is genuine sequencing. Ablation meta at:
`training/v2_artifacts/tunneling_plus_v2_xcs_ablated_meta.json`.

### YoY discrepancy explanation (Part C)

The previously reported −0.16 (sklearn R²) and +0.173 (Pearson r) for Location+ v1 YoY stability
are both correct — they measure different things. sklearn R² = −0.160 (score as literal predictor
of next-year score fails due to std=3.07 compression). Pearson r = 0.361 pooled (rank correlation
is weakly positive). The "+0.173" was a single-season Pearson r (not pooled). Definitive numbers:
**sklearn R² = −0.160, Pearson r = 0.361** (pooled across all consecutive season pairs 2015–2025).

### Cross-model correlation finding

Location+ v1 and Tunneling+ v3 correlate at r = +0.605 (both pointing wrong direction).
Location+ v2 and Tunneling+ v2 correlate at r = +0.617 (both pointing correct direction).
Live-model × v2-candidate cross-correlations are **negative** (r ≈ −0.48 to −0.60), confirming
the old and new versions rank pitchers in opposite directions.

### Stuff+ decomposed v1 (Part D — diagnostic only, do not promote)

Applied the same five-classifier architecture to Stuff+ features. Result: no improvement.
- YoY stability: Pearson r = 0.830 vs live v3's 0.907 — **worse**
- Predictive validity (→ next-yr SwStr%): 0.443 vs 0.432 — essentially tied
- ERA/xFIP: essentially tied with live v3 within noise
- Split-half reliability: 0.992 (high within-season consistency, as expected)

Artifacts saved as `training/v2_artifacts/stuff_decomposed_v1_*`. The live Stuff+ v3 remains the
correct choice. Main architectural value is interpretability (per-event decomposition), not
predictive improvement.

### Validation scripts

- `training/validate_v2_complete.py` — Parts A + B + C (runs in ~12 min)
- `training/train_stuff_decomposed_v1.py` — Part D (runs in ~47 min)
- Log output: `training/validate_v2_complete.log`, `training/train_stuff_decomposed_v1.log`
- Consolidated report (HTML artifact, private): published 2026-09-21

---

## Future model architecture (stated direction, largely not yet executed)

1. **Stuff+ delivery/mechanics features — DONE**: `release_extension`, `release_pos_x_as`,
   `release_pos_z`, `arm_angle` have been removed from Stuff+ and are now Deception+-exclusive
   (confirmed live in `analytics/stuff_features.py`). The r≈0.535 Stuff+/Deception+ correlation
   was the motivation.
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
- **Location+ v2 / Tunneling+ v2: no SHAP breakdown**: the decomposed-classifier architecture
  produces per-event probabilities across five classifiers + a run-value table. There is no single
  SHAP model to query. The `/pitcher/{id}` endpoints return a `shap_note` field explaining this;
  the frontend will need to gracefully handle the missing contributions array.

---

### Tunneling+ v2.1 candidate (Part C) — 2026-09-28

Tested targeted geometry removal: drop geometry from xFoul% only (general rule: drop geometry from any
classifier where it shows a statistically significant WRONG-SIGNED effect on that classifier's own
real target). xFoul% is the only current trigger (Part A2: lag1 z=−21.8, alllag z=−44.6).

New xFoul% trained on 2,635,868 pitches (2021–2025), 10 features (no geometry). Clean split: complement
of original Part A2 seed=42 200K sample (zero overlap guaranteed).

**Part B (clean split) results:**

| Metric | v2 xFoul% (WITH geometry) | v2.1 xFoul% (no geometry) |
|---|---|---|
| AUC clean split | 0.4982 (coin-flip) | **0.5863** (+0.0881) |
| Calibration z | −0.396 (WRONG direction) | **+34.4** (correct) |
| Geometry SHAP z (lag1/alllag) | −16.2 / −31.4 (replicates Part A2) | N/A |
| pt-lag SHAP z | — | **+21.0** (correct) |

**Part C (full battery) results:**

| Metric | Tunneling+ v2 | Tunneling+ v2.1 | Δ |
|---|---|---|---|
| xFoul% LOSO AUC | 0.499 | **0.583** | +0.084 |
| YoY stability (pooled r) | 0.692 | **0.756** | +0.064 |
| Split-half reliability | 0.645 | **0.719** | +0.074 |
| ERA r | −0.214 | −0.163 | −0.051 |
| xFIP r | −0.329 | −0.271 | −0.058 |

**Finding**: xFoul% classifier dramatically improves (+0.084 AUC; coin-flip → genuinely predictive).
YoY and split-half also improve. ERA/xFIP correlations are slightly weaker — geometry in v2's xFoul%
contributed to full-model ERA correlation in ways not visible from the xFoul% outcome alone. Worth
understanding before promotion. Status: **candidate only — v2 stays live**.

Scripts: `training/part_c_tunneling_v2_1.py`, `training/part_c_resume_split_half.py`.
Artifact: `training/v2_artifacts/tunneling_plus_v2_1_diagnostic.json`.
New booster: `models/tunneling_plus_v2_1_xfoul_nogeom.txt` (not wired into `api/`).

---

## Pending tasks (as of 2026-09-28)

1. **GitHub push**: `git push origin master` — push all accumulated changes to remote.
2. **Ballpark photos**: new high-home-view photos for Yankee Stadium (venue_id=3313) and T-Mobile
   Park (venue_id=680), looking out at the outfield.
3. **Prospect top performers stat spacing**: space out stats/outings the same way as the top
   performers tab (the `statCells` grid layout).
4. **Frontend SHAP graceful degradation**: Location+ and Tunneling+ player breakdown pages currently
   expect a `contributions` / SHAP breakdown from the API. With v2 live, the endpoints return an
   empty array + `shap_note`. The frontend should display the note gracefully instead of an empty
   table.
5. **Tunneling+ v2.1 promotion decision**: ERA/xFIP slight regression needs explanation before
   promoting. Hypothesis: geometry in xFoul% was acting as a proxy for some other pitcher-quality
   signal at the full-model level. Investigate before wiring into `api/`.

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

---
## Cross-Model Validation (2026-09-28)

**Sign convention:** r(plus, ERA) NEGATIVE = correctly-signed (higher plus = lower ERA).
Earlier methodology docs reported positive ERA r for v2 models — inverted convention; superseded here.

### Part 1 Summary (n_pitches ≥ 300, 2021–2025)
| Model | ERA r | xFIP r | YoY r | Split-half SB |
|---|---|---|---|---|
| Stuff+ v3 | -0.333 | -0.409 | +0.938 | +0.995 |
| Location+ v2 | -0.182 | -0.279 | +0.589 | +0.801 |
| Tunneling+ v2 | -0.218 | -0.339 | +0.643 | +0.887 |
| Deception+ v1 | -0.551 | -0.463 | +0.612 | +0.998 |

### Part 2 Summary
Geometry → plate_x R²=nan, plate_z R²=nan
Full AUC table in: docs/methodology/in_progress/CROSS_MODEL_VALIDATION_AND_LEAKAGE_TESTS_IN_PROGRESS.html

### Part 3 Summary
V0 CV R²=-0.00049, YoY r=+0.495
V1 (no h/w) CV R²=-0.00054, YoY r=+0.423
|ΔCV R²|=0.00006  |ΔYoY r|=0.072
