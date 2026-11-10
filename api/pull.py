"""
Data refresh pipeline - rebuilt from scratch (previous version's button
reportedly still didn't work reliably for the user, and was slow on top of
that). Two ways this gets triggered, both funnel through the exact same
`_run()`:

- Automatically, at server startup and every day at 6 AM (api/scheduler.py)
  - the primary mechanism now. The user shouldn't need to click anything
  for the site to stay current.
- The nav's "Refresh data" button (POST /api/admin/pull-data) - a manual
  "check right now" for when someone doesn't want to wait for the schedule.

There is deliberately no separate "fast" vs "full" pull mode anymore. The
previous version tried to make the button fast by only checking a recent
date window, which is exactly the kind of shortcut that can silently miss
a real gap. Instead:
  - pull_statcast.py's own `ingested_dates` skip-list means checking the
    ENTIRE season history on every single call costs almost nothing when
    everything's already covered (a few thousand cheap set lookups) - so
    every call here checks the full history, not a shortcut window.
  - pull_players_games.py's get_distinct_player_ids/get_distinct_games are
    gap-aware (recent activity UNION anything ever missing entirely), so
    a normal run stays fast (network calls only for what's actually new or
    genuinely never-seen) without needing a separate "full rescan" mode
    to catch real gaps.
The only thing that still varies by trigger is how thoroughly the display
cache gets rewarmed afterward (`full_warmup`) - see warmup.py's docstring -
since that's a pure UI-responsiveness tradeoff, not a data-correctness one.
"""
import logging
import os
import sqlite3
import json
import subprocess
import sys
import threading
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger("refresh")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulls.daily import pull_players_games
from pulls.daily import pull_team_stats
from pulls.daily import pull_remaining_schedule
from analytics import boxscore

from api import cache
from api.db import DB_PATH
from api.warmup import warm_cache_sync
from config.model_policy import CURRENT_SEASON
from config.season_phase import skip_team_stat_pulls as _skip_team_stats, is_postseason as _is_postseason

# A few days before 2021's actual opening day (~April 1) - close enough
# that this doesn't sweep in a needless extra month of guaranteed-empty
# offseason date checks, but early enough to safely cover every backfilled
# season. Checking already-covered dates is cheap (see get_uningested_dates
# in pull_statcast.py) but a date that's NEVER been checked before still
# costs one real (if fast, 0-row) pybaseball call before it gets marked and
# becomes cheap forever after - so this constant is a genuine, if small,
# one-time cost lever, not just documentation.
EARLIEST_SEASON_START = date(2021, 3, 20)
# Mirrors pull_statcast.RECENT_WINDOW_DAYS - not imported from there directly
# because pull_statcast pulls in pybaseball (and transitively matplotlib/Pillow),
# which would otherwise be loaded at every boot just for this one module-level
# constant. pull_statcast itself is imported lazily in _run() below.
PLAYERS_GAMES_RECENT_WINDOW_DAYS = 3

# POSTSEASON_MODE is now derived from config/season_phase.py (SEASON_PHASE env var
# or the default set there). Do not change this line; change SEASON_PHASE instead.
POSTSEASON_MODE = _is_postseason()

_status = {"state": "idle", "detail": "", "started_at": None, "finished_at": None}
_lock = threading.Lock()

# Persisted (survives a restart, unlike _status above) so a crash-loop -
# the process gets OOM-killed mid-warmup, restarts, and would otherwise
# immediately re-trigger the exact same pull+clear+rebuild every time -
# can be detected and the startup pull skipped. Lives next to the DB (same
# persistent volume), not in the DB itself, so it doesn't need a
# schema migration or risk corrupting real data.
_PULL_STATE_FILE = Path(DB_PATH).resolve().parent / "pull_state.json"
RESTART_PULL_SKIP_WINDOW_SECONDS = 2 * 3600  # 2 hours


def last_pull_finished_at():
    """Unix timestamp of the last pull cycle that completed successfully
    (didn't except), or None if unknown (first boot ever, or the state
    file is missing/corrupt - always safe to treat as None, which just
    means "pull as normal")."""
    try:
        return json.loads(_PULL_STATE_FILE.read_text())["finished_at"]
    except Exception:
        return None


def _record_pull_finished():
    try:
        _PULL_STATE_FILE.write_text(json.dumps({"finished_at": time.time()}))
    except Exception as e:
        log.warning("could not persist pull_state.json (non-fatal): %s", e)


def get_status():
    return dict(_status)


def _precompute_tunneling_pairs(season):
    """Regenerate the season's tunneling parquets so the player page reads
    fresh precomputed data. Runs in a subprocess (its memory is released when
    it exits) and has its own failure boundary: an error is logged and never
    blocks warm-up."""
    script = Path(__file__).resolve().parent.parent / "tools" / "precompute_tunneling_pairs.py"
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(script), str(season)],
            capture_output=True, text=True, timeout=1200,
            env={**os.environ, "DB_PATH": str(DB_PATH)},
        )
        elapsed = time.monotonic() - started
        if result.returncode == 0:
            summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            log.info("tunneling precompute ok: season=%s elapsed=%.1fs %s", season, elapsed, summary)
        else:
            log.error("tunneling precompute failed: season=%s exit=%s elapsed=%.1fs %s",
                      season, result.returncode, elapsed, result.stderr[-1000:])
    except Exception as e:
        log.error("tunneling precompute failed: season=%s elapsed=%.1fs %s: %s",
                  season, time.monotonic() - started, type(e).__name__, e)


def _run(full_warmup):
    from pulls.daily import pull_statcast

    end = date.today()
    pull_succeeded = False
    new_pitches = 0
    games_result = {"games_updated": 0, "players_updated": 0}

    try:
        start = EARLIEST_SEASON_START

        log.info("checking Statcast data %s to %s for anything missing...", start, end)
        # Deliberately doesn't show the full 2021-today range here even
        # though that IS what's being checked (see module docstring) - a
        # date that old reads as "why is this rescanning years of history"
        # even though the ingested_dates skip-list makes that check fast.
        # The real, current-day status is "is today's/recent data in yet."
        _status["detail"] = "Checking for new pitch data..."
        pitches_result = pull_statcast.run(start.isoformat(), end.isoformat())
        new_pitches = pitches_result["rows_inserted"]

        log.info("updating players/games reference data...")
        _status["detail"] = "Updating players/games reference data..."
        recent_cutoff = (end - timedelta(days=PLAYERS_GAMES_RECENT_WINDOW_DAYS)).isoformat()
        games_result = pull_players_games.run(since_date=recent_cutoff)

        season = CURRENT_SEASON

        if not POSTSEASON_MODE:
            log.info("updating team stats...")
            _status["detail"] = "Updating team stats..."
            pull_team_stats.run(season)
        else:
            log.info("postseason mode — skipping team stats (RS records frozen)")

        log.info("updating upcoming schedule / probable pitchers...")
        _status["detail"] = "Updating upcoming schedule..."
        from data.mlb_stats_api import schedule as _mlb_schedule
        _mlb_schedule(end.isoformat(), (end + timedelta(days=14)).isoformat(), db_path=str(DB_PATH))

        log.info("updating remaining schedule (SOS/playoffs)...")
        _status["detail"] = "Updating remaining schedule..."
        pull_remaining_schedule.run(season)

        log.info("updating rosters + schedule via MLB Stats API...")
        _status["detail"] = "Updating rosters/schedule..."
        try:
            from pulls.daily.refresh_mlb_stats import run as _refresh_mlb_stats
            _refresh_mlb_stats(season=season, skip_news=False, postseason=POSTSEASON_MODE)
        except Exception as _me:
            log.warning("refresh_mlb_stats failed (non-fatal): %s", _me)

        # Prospects: skip on manual refresh clicks; weekly on Mondays during scheduled runs.
        if full_warmup and date.today().weekday() == 0:
            log.info("updating prospect rankings + scouting (weekly Monday run)...")
            _status["detail"] = "Updating prospect rankings..."
            try:
                from pulls.daily import pull_prospects, pull_prospect_scouting
                pull_prospects.run(season)
                pull_prospect_scouting.run()
            except Exception as _pe:
                log.warning("prospect pull failed (non-fatal): %s", _pe)

        pull_succeeded = True
        _record_pull_finished()
        if new_pitches == 0:
            _status["detail"] = f"No new pitches found - already up to date through {end.isoformat()}."
        else:
            _status["detail"] = (
                f"Up to date through {end.isoformat()}: {new_pitches:,} new pitches, "
                f"{games_result['games_updated']} game(s) updated, "
                f"{games_result['players_updated']} player(s) checked."
            )
        log.info("pull done - %s", _status["detail"])
    except Exception as e:
        _status["detail"] = f"Refresh failed: {e}"
        log.error("pull failed (%s): %s", type(e).__name__, e)
        traceback.print_exc()

    # Warm-up runs in its OWN try/except, regardless of whether the pull
    # above succeeded - a network blip from Savant/the MLB Stats API used
    # to mean warm_cache_sync() never ran at all (it was the last step
    # inside the pull's own try block), so the site served nothing warmed
    # until the NEXT scheduled cycle. Now a pull failure just means this
    # cycle doesn't learn about new data; whatever's already in the DB
    # still gets (re)warmed.
    try:
        log.info("refreshing computed stats...")
        _status["detail"] = "Refreshing computed stats..."
        # Only clear caches when this cycle actually found new data -
        # unconditionally clearing on every full_warmup run (every boot,
        # every scheduled 6am refresh) defeated the point of the
        # disk-persisted percentile pools: every restart threw them away
        # and paid the full rebuild cost again, and a crash mid-warmup
        # just repeated the same expensive clear+rebuild on the next boot.
        # pull_succeeded=False (new_pitches stays 0) correctly leaves
        # existing caches alone - we don't know anything changed, so
        # don't act like it did.
        if full_warmup and not POSTSEASON_MODE and new_pitches > 0:
            # Scheduled/startup refresh with real new data: clear
            # everything including model caches, then do a full warmup
            # (includes L/R splits, similar players, etc.)
            from analytics import stuff_model

            # Only the current season's entries: finished seasons are unchanged
            # and rebuilding them is slow on the small production machine.
            cache.clear_current_season(CURRENT_SEASON)
            stuff_model.clear_cache(CURRENT_SEASON)
            boxscore.clear_cache()
        elif full_warmup and POSTSEASON_MODE and new_pitches > 0:
            # Postseason: RS model scores are final — don't nuke and recompute them.
            # Only clear daily-data caches that change with each new postseason game.
            boxscore.clear_cache()
            from api.routers.top_performers import _leaderboard
            from api.routers.standings import _standings
            _leaderboard.clear()
            _standings.clear()
        elif not full_warmup and new_pitches > 0:
            # Manual refresh button, and something actually changed: only
            # clear the daily-data caches. Model scoring caches (Stuff+,
            # Tunneling+, Deception+, Location+) don't meaningfully change
            # because of one day's new pitches — clearing and recomputing
            # them cold costs ~25-35s of LightGBM scoring that the user's
            # already paid for today. Keep them warm. Only clear what
            # actually needs to reflect new data: box scores (completed
            # games), leaderboards (daily top performers), standings.
            from api.routers.top_performers import _leaderboard
            from api.routers.standings import _standings
            boxscore.clear_cache()
            _leaderboard.clear()
            _standings.clear()
        # else: nothing new (or the pull failed) - skip clearing, just
        # warm whatever's cold below (first boot ever, or a key that
        # expired on its own TTL).
        # Only when this pull injected new pitches: the precompute loads a
        # full season of pitches, which is memory-heavy, and the readers use
        # the committed parquet whenever it exists.
        if pull_succeeded and new_pitches > 0:
            _precompute_tunneling_pairs(season)
        warm_cache_sync(full=full_warmup)
        if pull_succeeded:
            _status["state"] = "done"
        else:
            _status["state"] = "error"
            _status["detail"] += " (warmed from existing data)"
        log.info("warm-up done")
    except Exception as e:
        _status["state"] = "error"
        _status["detail"] = f"Warm-up failed: {e}"
        log.error("warm_cache_sync failed (%s): %s", type(e).__name__, e)
        traceback.print_exc()
    finally:
        _status["finished_at"] = date.today().isoformat()
        _lock.release()


def start_pull(full_warmup=False):
    """Returns True if a pull was started, False if one was already
    running. `full_warmup=True` (used by the scheduled/startup triggers,
    where nobody's waiting) also rewarms the L/R percentile splits, not
    just the default ALL view - see warmup.py."""
    if not _lock.acquire(blocking=False):
        return False
    _status["state"] = "running"
    _status["detail"] = "Starting..."
    _status["started_at"] = date.today().isoformat()
    _status["finished_at"] = None
    threading.Thread(target=_run, args=(full_warmup,), daemon=True).start()
    return True
