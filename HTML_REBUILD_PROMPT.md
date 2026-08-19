I have a working MLB analytics dashboard built as a Streamlit Python app, and I want it rebuilt as a real HTML/CSS/JS frontend backed by a lightweight API — not a static page, and not a Streamlit app. It needs to keep pulling live data and computing things dynamically, the same as it does now. I'm going to describe the current system in full so you have everything you need without me re-explaining from scratch.

## Where everything lives

Project folder: `C:\Users\joshs\Desktop\mlb_dashboard`. Everything below already exists and works — your job is to build a new frontend + API layer on top of it, not to redo the data/analytics work.

**Data**: `mlb_data.db`, a SQLite file (~220MB), full 2026 MLB season Statcast pitch-by-pitch data (538K+ rows) plus reference tables: `players`, `games`, `teams`, `player_advanced_stats` (sprint speed, outs above average), `ingested_dates` (internal bookkeeping).

**Data pipeline scripts** (all idempotent, safe to re-run, upsert-based):
- `pull_statcast.py` — incremental Statcast pitch data pull, tracks what's already ingested
- `pull_players_games.py` — player bios + game schedule/scores from MLB Stats API
- `pull_teams.py` — static team/league/division reference data
- `pull_advanced_stats.py` — Sprint Speed + OAA from Baseball Savant leaderboards via `pybaseball`

**Analytics/computation modules** (pure Python, no Streamlit dependency — these are the reusable core, call them directly from your API layer):
- `metrics.py` — VAA (exact FanGraphs formula), pitcher SP/RP role heuristic, Barrel% (community-standard approximation), pitch-movement archetype (East-West/North-South), wRC+ approximation
- `splits.py` — pitcher/batter splits engine (home/away, month, half, L/R, archetype, pitch type)
- `percentiles.py` — league-wide percentile pools for pitchers/batters, handedness-aware (recomputes the whole qualifying pool per L/R split, not just re-ranks)
- `stuff_model.py` — Stuff+ model (per-pitch-type OLS regression), plus a usage-weighted overall aggregate
- `player_page.py` — data-assembly helpers: IP/xFIP derivation, pitch tables, movement plot data, batter pitch-type performance, spray/field zone percentages
- `archetypes.py` — rule-based batter archetype classifier (5-Tool, Glove Guy, Three True Outcomes, Masher, Burner, Waiter, Pest, Steady Eddie)
- `standings.py` — standings computed from our own `games` table (not the MLB API's standings endpoint)
- `wpa.py` — daily Win Probability Added ranking, straight from Statcast's `delta_home_win_exp`
- `boxscore.py` — live MLB Stats API boxscore fetcher for official per-game stats (RBI, SB, ER — not reconstructable from Statcast alone)
- `ui_helpers.py` — Streamlit-specific presentation helpers (colors, HTML snippets) — you'll be replacing this layer, not reusing it directly, but read it for the exact logic (gradient formulas, color choices) it currently encodes

**Current Streamlit app** (what you're replacing the *presentation* of): `dashboard.py` (entry point + simple splits explorer) and `pages/1_Player_Page.py`, `pages/2_Standings.py`, `pages/3_Top_Performers.py`. Read these to understand exactly what's currently rendered and how — they're your spec for what the new frontend needs to reproduce.

## The architecture you're building

A static-page-with-zero-backend approach won't work here: the MLB Stats API doesn't support arbitrary cross-origin browser requests (CORS), and rewriting the Stuff+ regression / WPA aggregation / archetype rules / wOBA math in JavaScript would mean maintaining two parallel implementations of the same logic.

Build instead:
- **Backend**: a lightweight Python API (FastAPI recommended) that imports and calls the existing modules above directly — don't reimplement their logic, wrap them. Expose endpoints like `/api/player/{id}`, `/api/standings`, `/api/top-performers?date=...`, etc. This is where all the SQLite querying, live MLB API calls, and computation stays.
- **Frontend**: plain HTML/CSS/JS (no React/Vue/build-step framework — genuinely static-feeling HTML pages) that calls the API with `fetch()` and renders the results. This gives full pixel-level layout control, which the Streamlit version has been fighting against all session (custom HTML injection via `unsafe_allow_html` to work around Streamlit's widget constraints).
- For the movement plot specifically, consider `vega-embed` (the JS-native version of the Altair/Vega-Lite charting already used) to avoid reinventing that chart from scratch — or a simple SVG/canvas scatter if you'd rather keep dependencies minimal.

This is a genuine rearchitecture, not a reskin — treat it as its own multi-step project. Reuse the Python computation wholesale; only the presentation layer is being replaced.

## Design system: dark green + off-white theme

Restyle all **structural/decorative** UI (backgrounds, cards, borders, headers, body text, navigation) to a dark green and off-white palette — something like a deep forest green (`#1b4332`-ish) for primary structural elements/accents and a warm off-white (`#f5f3ee`-ish) for backgrounds, with dark green or near-black text for readability. Exact hex values are your call — pick something that reads clean and professional, not neon.

**Do not** restyle colors that are carrying actual data meaning — these are functional, not decorative, and changing them would break the signal they're designed to communicate:
- The percentile bar gradient (blue = low percentile, grey = 50th, red = high) — this red/blue convention is standard in the baseball analytics world (matches Baseball Savant) and the user reads it fluently; don't reskin it to green.
- Pitch-type colors on the movement plot (FF=red, CU=blue, FC=brown, SI=orange, CH=green, FS=turquoise, ST=darker orange, SL=yellow) — these are fixed, deliberate, and specifically requested by the user; keep them as-is.
- Run differential coloring (green=positive, red=negative) on Standings.
- The gold/amber "best record" and "top WPA rank" highlight treatments — these could arguably shift toward the new theme's accent color, use judgment, but keep them visually distinct from the surrounding rows.

## What each page needs to reproduce

Read the corresponding Streamlit page file for the authoritative current spec — this is a summary, not the full detail.

**Player Page** (pitcher + batter mode toggle):
- Horizontal player card at the top: team logo (`https://www.mlbstatic.com/team-logos/{team_id}.svg` — confirmed public, no auth), name, counting stats laid out horizontally, Stuff+ badge (pitchers) or archetype badge (batters)
- Percentiles panel: grouped into logical clusters (not sorted by value), colored gradient bars with raw value + percentile shown inline, larger/fatter/narrower bars than a first draft ended up with
- Movement plot (pitchers): scatter of horizontal break vs. induced vertical break, colored by pitch type, crosshair axes at 0/0, grey dashed reference line for arm angle, **must use the arm-side-positive HB sign convention** — see gotcha below
- Pitch table (pitchers) / pitch-type performance table (batters): full width, pitch-type cell colored solid, rest of row faintly tinted the same color, includes a league-average comparison row at the bottom
- Batter "Swing & Field" visual: field-level (not bird's-eye) umpire's-eye view — horizon with sky/field, LF/CF/RF zones as a perspective fan colored by the batter's actual pull tendency, illustrative bat-swing arc in the foreground
- L/R toggle recomputes percentiles from a freshly-filtered pool (not just re-ranks the same numbers)

**Standings**: AL/NL toggle, grouped by division, team logos, W/L/win%/run-diff/one-run record/last-10/last-30, best-record-in-league highlighted distinctly

**Top Performers**: date picker (default to the actual most recent date in the database, not hardcoded), top 10 players by WPA for that day, mixed pitcher/batter leaderboard where each row shows role-appropriate stats, opponent shown small ("vs OPP" / "@ OPP"), walk-off badge when applicable, **columns must align in fixed positions across every row regardless of pitcher/batter stat-count differences** — see gotcha below, rank 1-3 get a gold background that darkens toward rank 1

## Known gotchas and decisions worth carrying forward

These were each discovered the hard way this session — worth not re-learning them:

- **Statcast's `pfx_x` (horizontal movement) is a fixed field coordinate, not handedness-adjusted.** A RHP's arm-side run and a LHP's arm-side run have opposite raw signs even though they mean the same thing. Fix: flip sign for RHP so positive HB = arm-side for everyone (Baseball Savant's own convention). Already implemented in `player_page.py`'s `_add_arm_side_hb()` — just reuse it.
- **Percentile ratios vs. percentages**: `percentiles.py` returns raw ratios (0–1), not 0–100. Format accordingly (multiply by 100 or use a percent format spec) — this bit us once already (a value silently showing "0.2%" instead of "20.0%").
- **CSS Grid column alignment**: if you build a mixed-row-type table (like Top Performers, where pitcher and batter rows have different stat counts), every row needs the *same* grid template with real placeholder elements for unused slots — an empty string doesn't reserve a grid column track, only an actual empty element does. Verify alignment isn't just "looks right" — check computed positions.
- **Team logo URL pattern**: `https://www.mlbstatic.com/team-logos/{team_id}.svg`, confirmed public and working, no auth needed.
- **MLB Stats API boxscore endpoint** (`https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore`) gives official per-game AND season-cumulative stats in one call — this is the source for RBI, stolen bases, and earned runs, none of which are reliably derivable from Statcast pitch data alone (confirmed: stolen base events don't even appear in Statcast's per-pitch data, and Statcast has zero earned/unearned run distinction).
- **KC (knuckle curve) is merged into CU (curveball)** everywhere pitch type is grouped or colored — treated as the same pitch, per explicit user request.
- **wRC+, xFIP, and Barrel% are all approximations, clearly labeled as such** — wRC+ uses self-computed league wOBA but hardcoded historical constants for wOBA_scale/lgR-per-PA (no park factor); xFIP uses a hardcoded FIP constant (~3.10) since there's no earned-run data to calibrate against; Barrel% uses the community-standard Bill Petti reverse-engineered formula since Statcast's real formula was never published. Preserve these caveats in the new UI somewhere (tooltip, footnote, whatever fits the design) rather than silently presenting them as authoritative.
- **The database can lag behind real time.** Games/pitches only exist through whatever date was last pulled — build the app to reflect `MAX(game_date)` from the database rather than assuming "today," and consider whether your API layer should expose a way to trigger a fresh pull (or at least surface how current the data is).
- Streamlit's caching meant data pulled while the app was running wouldn't show up until a restart — your new architecture should decide deliberately how "live" each endpoint actually is (query fresh every request vs. some caching layer with explicit invalidation).

## Suggested approach

1. Read the current Streamlit pages and the modules they import to build a precise feature inventory before writing any new code.
2. Stand up the FastAPI backend first — thin wrapper endpoints around the existing modules, verify each one returns sane data via direct requests before touching the frontend.
3. Build the frontend page by page, matching the Streamlit version's information density but with real layout control.
4. Verify against known-good examples already validated this session (e.g., Bobby Witt Jr. should classify as 5-Tool, Andrés Giménez as Glove Guy, a RHP's fastball should show positive horizontal break) rather than just eyeballing that it looks plausible.
