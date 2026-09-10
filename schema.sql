-- mlb_dashboard data layer
-- Statcast pitch-level data only. FanGraphs season aggregates and MLB Stats API
-- standings will live in their own separate tables in a later session.

PRAGMA foreign_keys = ON;

-- One row per MLB team. Static reference data (name/league/division), pulled
-- from the MLB Stats API /teams endpoint - not derived from Statcast.
CREATE TABLE IF NOT EXISTS teams (
    team_id      INTEGER PRIMARY KEY,   -- MLBAM team id
    name         TEXT,                  -- e.g. 'Athletics'
    abbreviation TEXT,                  -- e.g. 'ATH'
    league       TEXT,                  -- 'AL' / 'NL'
    division     TEXT                   -- e.g. 'AL West'
);

-- One row per player. Populated in a later session (roster ingestion).
CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,   -- MLBAM id
    name        TEXT,
    team_id     TEXT,
    position    TEXT,
    bats        TEXT,                  -- 'L' / 'R' / 'S'
    throws      TEXT,                  -- 'L' / 'R'
    active_flag INTEGER DEFAULT 1
);

-- One row per game. Populated in a later session (schedule/boxscore ingestion).
CREATE TABLE IF NOT EXISTS games (
    game_id     INTEGER PRIMARY KEY,   -- MLBAM game_pk
    game_date   TEXT,
    season      INTEGER,               -- calendar year of game_date; regular-season dates never cross a year boundary
    home_team   TEXT,
    away_team   TEXT,
    final_score TEXT
);

-- One row per pitch. Single source of truth for everything Statcast-derived:
-- splits, VAA, arsenal classification, xwOBA rollups, etc. are all queries/
-- views on top of this table, not separate raw tables.
CREATE TABLE IF NOT EXISTS pitches (
    pitch_id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id                           INTEGER,
    game_date                         TEXT,
    season                            INTEGER,
    at_bat_number                     INTEGER,   -- needed with pitch_number to uniquely id a pitch
    pitch_number                      INTEGER,
    pitcher_id                        INTEGER,
    batter_id                         INTEGER,
    pitch_type                        TEXT,
    release_speed                     REAL,
    release_spin_rate                 REAL,
    release_pos_x                     REAL,
    release_pos_y                     REAL,
    release_pos_z                     REAL,
    pfx_x                             REAL,
    pfx_z                             REAL,
    plate_x                           REAL,
    plate_z                           REAL,
    zone                              INTEGER,
    balls                             INTEGER,
    strikes                           INTEGER,
    outs_when_up                      INTEGER,
    inning                            INTEGER,
    stand                             TEXT,       -- batter side
    p_throws                          TEXT,       -- pitcher hand
    description                       TEXT,
    events                            TEXT,
    launch_speed                      REAL,
    launch_angle                      REAL,
    delta_run_exp                     REAL,
    woba_value                        REAL,
    estimated_woba_using_speedangle   REAL,       -- source of truth for xwOBA, aggregated later
    estimated_slg_using_speedangle    REAL,       -- source of truth for xSLG, aggregated later
    estimated_ba_using_speedangle     REAL,       -- source of truth for xBA, aggregated later
    vx0                                REAL,       -- trajectory velocity/accel components at
    vy0                                REAL,       -- release-point measurement plane (y=50ft),
    vz0                                REAL,       -- needed to project pitch to the plate for
    ax                                 REAL,       -- VAA/HAA (vertical/horizontal approach angle)
    ay                                 REAL,
    az                                 REAL,
    release_extension                 REAL,
    spin_axis                         REAL,
    effective_speed                   REAL,
    arm_angle                         REAL,
    sz_top                            REAL,
    sz_bot                            REAL,
    inning_topbot                     TEXT,       -- 'Top' = home team pitching, 'Bot' = away team pitching
    woba_denom                        INTEGER,     -- Statcast's own flag for PA qualifying toward wOBA/xwOBA denominator
    bb_type                           TEXT,       -- ground_ball / fly_ball / line_drive / popup, batted balls only
    hc_x                               REAL,       -- hit coordinates, for spray angle / pull-air%
    hc_y                               REAL,
    bat_speed                         REAL,
    swing_length                      REAL,
    attack_angle                      REAL,
    attack_direction                  REAL,
    swing_path_tilt                   REAL,
    delta_home_win_exp                REAL,       -- WPA source: change in home team's win probability on this pitch
    bat_score                         REAL,       -- batting team's score entering this pitch
    post_bat_score                    REAL,       -- batting team's score after this pitch (diff = runs scored on the play)
    post_home_score                   REAL,
    post_away_score                   REAL,
    FOREIGN KEY (pitcher_id) REFERENCES players(player_id),
    FOREIGN KEY (batter_id)  REFERENCES players(player_id),
    FOREIGN KEY (game_id)    REFERENCES games(game_id),
    UNIQUE (game_id, at_bat_number, pitch_number)
);

CREATE INDEX IF NOT EXISTS idx_pitches_pitcher   ON pitches(pitcher_id);
CREATE INDEX IF NOT EXISTS idx_pitches_batter    ON pitches(batter_id);
CREATE INDEX IF NOT EXISTS idx_pitches_game_date ON pitches(game_date);
CREATE INDEX IF NOT EXISTS idx_pitches_season    ON pitches(season);
CREATE INDEX IF NOT EXISTS idx_pitches_pitch_type ON pitches(pitch_type);
CREATE INDEX IF NOT EXISTS idx_pitches_stand      ON pitches(stand);
CREATE INDEX IF NOT EXISTS idx_pitches_p_throws   ON pitches(p_throws);
-- Composite, season-first: every league-wide pool query (player search,
-- percentile pools, similarity pools, ...) filters `WHERE season = ?` then
-- needs distinct pitcher_id/batter_id - the single-column indexes above
-- force SQLite to look up each matching row by rowid to read pitcher_id/
-- batter_id. These let it answer straight from the index (COVERING INDEX,
-- confirmed via EXPLAIN QUERY PLAN) - measured ~11x faster (1.7s -> 0.15s)
-- on the /api/players/all query for a single season.
CREATE INDEX IF NOT EXISTS idx_pitches_season_pitcher ON pitches(season, pitcher_id);
CREATE INDEX IF NOT EXISTS idx_pitches_season_batter  ON pitches(season, batter_id);

-- Season-level Savant/MLB Stats API leaderboard stats - not derivable from
-- the pitch-by-pitch pull, genuinely external aggregates pulled separately.
-- One row per player per season. wrc_plus/xfip/barrel_pct_* are real,
-- officially-sourced SEASON-TOTAL numbers (MLB Stats API's own sabermetrics
-- stat group for wrc_plus/xfip, Baseball Savant's exit-velo/barrels
-- leaderboard for barrel_pct_*) - FanGraphs itself is Cloudflare-blocked
-- (see the earlier investigation note in api/presentation.py), but these
-- are the same real numbers by a different, unblocked route. They have no
-- opposing-pitcher-hand split (MLB doesn't publish one), so the L/R-split
-- display still falls back to this project's own pitch-level approximation
-- - only the ALL/unsplit view uses these.
CREATE TABLE IF NOT EXISTS player_advanced_stats (
    player_id           INTEGER,
    season               INTEGER,
    sprint_speed         REAL,   -- ft/sec, Baseball Savant sprint speed leaderboard
    oaa                  REAL,   -- Outs Above Average, Baseball Savant fielding leaderboard (all positions)
    wrc_plus             REAL,   -- real wRC+, MLB Stats API sabermetrics (hitting), season total
    xfip                 REAL,   -- real xFIP, MLB Stats API sabermetrics (pitching), season total
    barrel_pct_batting   REAL,   -- real Barrel% (own batted balls), Baseball Savant, season total
    barrel_pct_pitching  REAL,   -- real Barrel% (allowed), Baseball Savant, season total
    arm_strength         REAL,   -- real blended arm strength (mph), Baseball Savant arm-strength leaderboard
    war_batting          REAL,   -- real batting WAR, MLB Stats API sabermetrics (hitting), season total
    war_pitching         REAL,   -- real pitching WAR, MLB Stats API sabermetrics (pitching), season total
    running_game_runs_prevented REAL,  -- real runs prevented via the running game (steals/pickoffs), Baseball Savant pitcher-running-game leaderboard's runs_prevented_on_running_attr - see pulls/pull_pitcher_running_game.py for why this column was chosen over the other candidate metrics
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- MLB Pipeline prospect rankings (Top 100 + each org's Top 30), scraped
-- separately from Statcast/statsapi since no official rankings API exists.
-- Season stat line (gp/k_pct/bb_pct/ops or era) comes from the real MLB Stats
-- API season-stats endpoint at the prospect's current level, keyed off the
-- MLBAM person id recovered from their MLB Pipeline profile link - genuinely
-- official numbers, not scraped off the rankings page itself (which has none).
CREATE TABLE IF NOT EXISTS prospects (
    mlb_id        INTEGER,              -- MLBAM person id
    season        INTEGER,              -- rankings list season (e.g. 2026)
    name          TEXT,
    position      TEXT,                 -- raw Pipeline position string, e.g. 'SS', 'RHP', '3B/2B/OF'
    is_pitcher    INTEGER,              -- derived from position starting with RHP/LHP
    team_id       INTEGER,              -- resolved from teams.name
    org_rank      INTEGER,              -- rank within the org's own Top 30
    top100_rank   INTEGER,              -- rank on the overall Top 100 list, NULL if unranked
    level         TEXT,                 -- e.g. 'AA', 'AAA', 'A+', 'A', 'ROK'
    eta           INTEGER,
    age           INTEGER,
    height_weight TEXT,
    bats          TEXT,
    throws        TEXT,
    headshot_url  TEXT,
    stats_season  INTEGER,              -- season the stat line below is from
    gp            INTEGER,
    k_pct         REAL,
    bb_pct        REAL,
    ops           REAL,                 -- batters only
    era           REAL,                 -- pitchers only
    updated_at    TEXT,
    PRIMARY KEY (mlb_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- 20-80 scouting grades + scouting report blurb per prospect, from Baseball
-- Savant's player page (which embeds MLB Pipeline's scouting write-ups -
-- Pipeline's own site has none of this, only bio/ranking info). Grade tool
-- names vary by role (Hit/Power/Run/Arm/Field for hitters, Fastball/
-- Curveball/Slider/Changeup/Control for pitchers, and pitch-mix names vary
-- pitcher to pitcher), so grades are kept as a JSON object rather than fixed
-- columns. One row per prospect - whichever dated report block on the page
-- is most recent, not the full multi-year history.
CREATE TABLE IF NOT EXISTS prospect_scouting (
    mlb_id        INTEGER PRIMARY KEY,
    report_label  TEXT,       -- e.g. '2026', or 'international' if that's all that exists yet
    grades_json   TEXT,       -- {"Hit": 60, "Power": 60, ...} or {"Fastball": 60, ...}
    overall_grade INTEGER,
    report_text   TEXT,       -- report paragraphs joined with blank lines
    video_url     TEXT,
    updated_at    TEXT
);

-- Internal bookkeeping only (not a data table): lets pull_statcast.py know
-- which calendar dates have already been pulled so re-runs skip them instead
-- of re-hitting the Statcast API. Dedup of actual rows is enforced by the
-- UNIQUE constraint on pitches above regardless of this table's state.
CREATE TABLE IF NOT EXISTS ingested_dates (
    game_date     TEXT PRIMARY KEY,
    rows_inserted INTEGER,
    ingested_at   TEXT
);

-- Team-level season + trailing-30-day snapshot for the team card popup
-- (click any team logo). OPS/ERA/bullpen ERA are official MLB Stats API
-- numbers; FIP is self-computed with the standard formula from official
-- HR/BB/HBP/K/IP (same FIP_CONSTANT_APPROX=3.10 already used for xFIP
-- elsewhere in this project - FanGraphs itself is unreachable, blocked by
-- a Cloudflare bot challenge even a real headless browser can't get past,
-- confirmed directly). No bullpen_era_30d: the Stats API's date-range stat
-- type silently ignores the relief-pitcher sitCode filter (confirmed
-- directly - it returns identical numbers with and without it), and our
-- own pitch data has no earned/unearned distinction to self-compute ERA
-- from, so a real 30-day bullpen split isn't available from any source
-- this project can reach. One row per team, overwritten on every pull.
CREATE TABLE IF NOT EXISTS team_stats_snapshot (
    team_id       INTEGER PRIMARY KEY,
    season        INTEGER,
    pulled_at     TEXT,
    ops           REAL,       -- team's own season batting OPS
    era           REAL,       -- team's own season pitching ERA
    bullpen_era   REAL,       -- season relief-pitching-only ERA (sitCodes=rp)
    fip           REAL,       -- self-computed, season
    ops_30d       REAL,
    era_30d       REAL,
    fip_30d       REAL,       -- self-computed, trailing 30 days
    baserunning_value REAL,   -- 0.2*SB - 0.4*CS, standard rough linear-weight
                              -- approximation of the SB/CS component of
                              -- baserunning value (runs) - NOT a full BsR
                              -- (no taking-extra-bases component; that needs
                              -- play-by-play data this project doesn't pull)
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Current injured-list snapshot - who is on the IL RIGHT NOW, per the MLB
-- Stats API's own 40-man-roster status codes (D10/D15/D60 - there is no
-- dedicated "injured list" roster endpoint; 40Man is the only roster view
-- that carries IL status on it, see pull_injuries.py). start_date and
-- injury are real, parsed from that player's own /transactions history (the
-- "placed <player> on the N-day injured list. <injury>." transaction text) -
-- not fabricated. There is no official "expected return date" published by
-- MLB Stats API anywhere; min_return_date is a REAL, computed lower bound
-- (start_date + the IL type's minimum stint length, a hard roster rule),
-- not a guess at when they'll actually be activated. Fully replaced on
-- every pull (a snapshot of "on the IL today", not a running history log).
CREATE TABLE IF NOT EXISTS il_status (
    player_id       INTEGER PRIMARY KEY,
    team_id         INTEGER,
    il_type         TEXT,       -- 'D10' / 'D15' / 'D60' / 'D7'
    il_label        TEXT,       -- 'Injured 10-Day' / 'Injured 15-Day' / etc, from the API's own status.description
    start_date      TEXT,       -- real, parsed from the transactions feed
    injury          TEXT,       -- real, parsed from the same transaction's description sentence, NULL if unavailable
    min_return_date TEXT,       -- start_date + this IL type's minimum stint length (a real roster rule, not a guess)
    pulled_at       TEXT,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Remaining (not-yet-played) regular-season schedule, for Remaining
-- Strength of Schedule on the team card - see pull_remaining_schedule.py.
-- Fully replaced each pull, not upserted incrementally (yesterday's
-- "remaining" set is stale, not additive).
CREATE TABLE IF NOT EXISTS remaining_schedule (
    game_id      INTEGER PRIMARY KEY,
    season       INTEGER,
    game_date    TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER
);
