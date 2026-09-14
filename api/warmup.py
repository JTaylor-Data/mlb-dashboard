"""
Pre-warms the TTL-cached league-wide pool computations (percentile pools,
Stuff+, pitcher roles, league-average rows, standings) so the player page -
and the first request after a cold cache in general - doesn't have to eat a
~15s+ full-table scan live in the request path. Runs in a background thread
so the server starts accepting requests immediately; requests that land
before warmup finishes still work, they just recompute (and cache) on
demand like before.
"""
import sqlite3
import threading

from api.db import DB_PATH

SPLITS = [None, "L", "R"]
# Only the current season is pre-warmed - warming all 6 backfilled seasons
# on every startup would multiply the warmup cost 6x for seasons that may
# never get clicked. Older seasons just pay the ~15s cost on first request
# instead, same as this app's normal cold-cache behavior everywhere else.
CURRENT_SEASON = 2026


def _warm(conn, full):
    """`full=True` (server startup - a one-time cost, worth spending fully)
    warms every split. `full=False` (after a "Refresh data" click, which a
    user can trigger repeatedly in one sitting) warms only the ALL split -
    the L/R percentile pools are a genuine ~25s chunk of warmup time that
    most refreshes don't need paid again immediately; a user who actually
    toggles to L/R shortly after a refresh just eats the same one-time cold
    cost this app already accepts for older seasons, rather than every
    refresh paying for splits that may not get viewed at all this session."""
    from api.routers.heatmaps import _league_command_pool
    from api.routers.players import (
        _all_players, _all_pitcher_arm_angles, _batter_pool, _league_avg_batter_row, _league_hr_fb,
        _pitcher_pool, _pitcher_roles, _stuff_plus_by_stand_table, _stuff_plus_table,
    )
    from api.routers.deception_report import REPORT_SEASON as DECEPTION_REPORT_SEASON, _raw_predictions_table as _deception_raw_predictions_table
    from api.routers.tunneling_report import _v3_raw_predictions_table
    from api.routers.location_report import _raw_predictions_table as _location_raw_predictions_table
    from api.routers.similar_players import _similarity_pool_hitters, _similarity_pool_pitchers
    from api.routers.probable_pitchers import _stuff_dict, _tunneling_dict, _location_dict, _deception_dict
    from api.routers.standings import _standings
    from api.routers.stuff_model_report import _stuff_leaderboard
    from api.routers.team_card import _cached_rows_and_ranks, _cached_sos
    from api.routers.top_performers import _leaderboard

    season = CURRENT_SEASON
    _stuff_plus_table(conn, season)
    _stuff_plus_by_stand_table(conn, season)
    _pitcher_roles(conn, season)
    _league_hr_fb(conn, season)
    # Every pitcher page unconditionally computes the Deception+ and
    # Tunneling+ v3 badges (see api/routers/players.py's
    # _deception_plus_badge/_tunneling_plus_badge), each needing this
    # season's full league-wide raw-prediction table the first time - a
    # real SHAP-explainer pass over Deception+'s pitches and a full
    # lag-feature-engineering + scoring pass for Tunneling+ v3
    # (confirmed directly: ~4s and ~9s respectively on a cold cache).
    # Neither was ever added here when those badges were wired in, so the
    # very first pitcher page load after every restart was eating both
    # costs live instead of hitting a warm cache - the same gap
    # _all_players/_all_pitcher_arm_angles below were already fixed for.
    # Not split-dependent (neither badge has a split), so warmed
    # unconditionally like Stuff+ above, not gated behind `full`.
    _deception_raw_predictions_table(conn, DECEPTION_REPORT_SEASON)
    _v3_raw_predictions_table(conn, season)
    # Location+'s own model page has a season-driven leaderboard, same
    # shape as Tunneling+ v3's - warmed here so the first visit to that
    # page after a restart isn't the one paying for the full-season scan.
    _location_raw_predictions_table(conn, season)
    # Stuff+'s own leaderboard endpoint keeps a SEPARATE @ttl_cache() entry
    # from players.py's _stuff_plus_table above (each @ttl_cache()-wrapped
    # function gets its own private store - see api/cache.py - even though
    # both call the same compute_stuff_plus()), so warming one doesn't warm
    # the other. The expensive inner model-scoring pass is still shared via
    # stuff_model.py's own internal cache either way; this just closes the
    # remaining shrinkage/groupby gap for whoever opens the Stuff+ page first.
    _stuff_leaderboard(conn, season)
    # Team card - event-delegated on every team logo, sitewide, and was
    # completely uncached (confirmed directly: WAR groupby + two independent
    # full `games` table scans, recomputed from scratch on every single
    # open). Warmed here so the first click of a session doesn't pay for it.
    _cached_rows_and_ranks(conn, season)
    _cached_sos(conn, season)
    for split in (SPLITS if full else [None]):
        _pitcher_pool(conn, season, split)
        _batter_pool(conn, season, split)
        _league_avg_batter_row(conn, season, split)
    # Command-percentile pool only exists for a real L/R split (the pitch
    # heatmap endpoint itself requires one, never "ALL" - see heatmaps.py) -
    # warm both real splits regardless of `full`, since unlike the L/R
    # percentile pools this one is cheap (4 narrow columns, one vectorized
    # pass, no per-player loop) and every pitcher heatmap request needs it.
    for split in ("L", "R"):
        _league_command_pool(conn, season, split)
    # Similar-players pool spans EVERY backfilled season (2021-current), not
    # just the current one - real, heavier than anything else warmed here
    # (a _pitcher_pool/_batter_pool build per historical season, most of
    # which nothing else ever warms). Startup/daily only (full=True), same
    # reasoning as the L/R split pools above - a "Refresh data" click
    # doesn't need to re-eat a 6-season scan every time, and this pool's
    # TTL means it stays warm across ordinary refreshes anyway.
    if full:
        _similarity_pool_hitters(conn)
        _similarity_pool_pitchers(conn)
        # Career page calls _stuff_table/_tun_table/_loc_table per season for
        # veteran players (2021-2025). These aren't warmed above (only current
        # season). Warm them now so the first career load for a veteran doesn't
        # pay ~2+ min of per-season model scoring live.
        from api.routers.stuff_model_report import _raw_predictions_table as _career_stuff_table
        from api.routers.tunneling_report import _v3_raw_predictions_table as _career_tun_table
        from api.routers.location_report import _raw_predictions_table as _career_loc_table
        past_seasons = [s for s in range(2021, CURRENT_SEASON)]
        for ps in past_seasons:
            try:
                _career_stuff_table(conn, ps)
            except Exception as e:
                print(f"[warmup] career stuff_table({ps}) failed: {e}")
            try:
                _career_tun_table(conn, ps)
            except Exception as e:
                print(f"[warmup] career tun_table({ps}) failed: {e}")
            try:
                _career_loc_table(conn, ps)
            except Exception as e:
                print(f"[warmup] career loc_table({ps}) failed: {e}")
    # Probable-pitchers page model score dicts — all four are @ttl_cache() in
    # probable_pitchers.py but were never warmed, so the first Probables page
    # visit after a restart paid full model cost for all four dicts cold.
    # _stuff_dict calls compute_stuff_plus (same LightGBM pass as
    # _stuff_plus_table above, deduped by stuff_model.py's own 10-min internal
    # cache); _tunneling/_location/_deception each run their own raw-prediction
    # passes, also already warmed above, so this is just the light groupby/
    # scoring aggregation on top of the cached raw tables.
    _stuff_dict(conn, season)
    _tunneling_dict(conn, season)
    _location_dict(conn, season)
    _deception_dict(conn)
    _standings(conn, season)
    # The player-search dropdown's own full-season UNION scan
    # (GET /api/players/all) is unconditionally hit on every Player-page
    # load - was missing from warmup, so the very first page load after
    # every restart paid this scan live instead of hitting a warm cache.
    _all_players(conn, season)
    # Season-wide arm-angle pool behind the movement plot's peer-group
    # feature - same reasoning as _all_players above: every pitcher's page
    # needs it, so warm it once here instead of paying its cost on
    # whichever pitcher happens to be the first one someone actually looks
    # at after a restart.
    _all_pitcher_arm_angles(conn, season)
    # Every player page's "Last 5 Games" card checks, per game, whether
    # this player was a top performer THAT DAY (api/routers/players.py's
    # _last_5_games -> _leaderboard(conn, date)) - _leaderboard is already
    # cached per-date, but a brand-new player's 5 most recent games are
    # near-certainly within the last ~2 weeks, so warming that handful of
    # real recent dates here means virtually every real player lookup hits
    # an already-warm leaderboard instead of a live ~1-1.5s/date fetch (up
    # to ~5-7s stacked across 5 games for whoever's unlucky enough to be
    # the first to ask about a given date). Confirmed directly as the
    # single largest remaining per-player cost after the badge/pool gaps
    # above were fixed. Sequential here (not threaded) - _leaderboard
    # already parallelizes its OWN internal boxscore fetches per date;
    # nesting another layer of concurrency on top of that was confirmed to
    # cause real contention (slower, not faster) rather than compounding
    # the speedup.
    recent_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM games ORDER BY game_date DESC LIMIT 10"
    ).fetchall()]
    for d in recent_dates:
        try:
            _leaderboard(conn, d)
        except Exception as e:
            print(f"[warmup] _leaderboard({d}) failed: {e}")  # a bad/incomplete date shouldn't abort the rest


def warm_cache_sync(full=True):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        _warm(conn, full)
    finally:
        conn.close()


def warm_cache_background():
    threading.Thread(target=warm_cache_sync, daemon=True).start()
