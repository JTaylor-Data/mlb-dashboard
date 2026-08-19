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

-- Season-level Savant leaderboard stats (Sprint Speed, Outs Above Average) -
-- not derivable from the pitch-by-pitch pull, genuinely external aggregates
-- pulled separately, same category as the deferred FanGraphs table. One row
-- per player per season.
CREATE TABLE IF NOT EXISTS player_advanced_stats (
    player_id     INTEGER,
    season        INTEGER,
    sprint_speed  REAL,       -- ft/sec, Baseball Savant sprint speed leaderboard
    oaa           REAL,       -- Outs Above Average, Baseball Savant fielding leaderboard (all positions)
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
