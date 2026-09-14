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
import sqlite3
import sys
import threading
import traceback
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pulls import pull_players_games
from pulls import pull_statcast
from pulls import pull_team_stats
from pulls import pull_remaining_schedule
from analytics import boxscore, deception_model, stuff_model

from api import cache
from api.db import DB_PATH
from api.warmup import warm_cache_sync

# A few days before 2021's actual opening day (~April 1) - close enough
# that this doesn't sweep in a needless extra month of guaranteed-empty
# offseason date checks, but early enough to safely cover every backfilled
# season. Checking already-covered dates is cheap (see get_uningested_dates
# in pull_statcast.py) but a date that's NEVER been checked before still
# costs one real (if fast, 0-row) pybaseball call before it gets marked and
# becomes cheap forever after - so this constant is a genuine, if small,
# one-time cost lever, not just documentation.
EARLIEST_SEASON_START = date(2021, 3, 20)
PLAYERS_GAMES_RECENT_WINDOW_DAYS = pull_statcast.RECENT_WINDOW_DAYS

_status = {"state": "idle", "detail": "", "started_at": None, "finished_at": None}
_lock = threading.Lock()


def get_status():
    return dict(_status)


def _run(full_warmup):
    try:
        end = date.today()
        start = EARLIEST_SEASON_START

        print(f"[refresh] checking Statcast data {start} to {end} for anything missing...", flush=True)
        # Deliberately doesn't show the full 2021-today range here even
        # though that IS what's being checked (see module docstring) - a
        # date that old reads as "why is this rescanning years of history"
        # even though the ingested_dates skip-list makes that check fast.
        # The real, current-day status is "is today's/recent data in yet."
        _status["detail"] = "Checking for new pitch data..."
        pitches_result = pull_statcast.run(start.isoformat(), end.isoformat())

        print("[refresh] updating players/games reference data...", flush=True)
        _status["detail"] = "Updating players/games reference data..."
        recent_cutoff = (end - timedelta(days=PLAYERS_GAMES_RECENT_WINDOW_DAYS)).isoformat()
        games_result = pull_players_games.run(since_date=recent_cutoff)

        season = end.year

        print("[refresh] updating team stats...", flush=True)
        _status["detail"] = "Updating team stats..."
        pull_team_stats.run(season)

        print("[refresh] updating upcoming schedule / probable pitchers...", flush=True)
        _status["detail"] = "Updating upcoming schedule..."
        from data.mlb_stats_api import schedule as _mlb_schedule
        _mlb_schedule(end.isoformat(), (end + timedelta(days=14)).isoformat(), db_path=str(DB_PATH))

        print("[refresh] updating remaining schedule (SOS/playoffs)...", flush=True)
        _status["detail"] = "Updating remaining schedule..."
        pull_remaining_schedule.run(season)

        # Prospects don't change daily — skip on manual Refresh clicks (full_warmup=False)
        # to avoid a slow network call the user didn't ask for. Daily scheduler still runs it.
        if full_warmup:
            print("[refresh] updating prospect rankings + scouting...", flush=True)
            _status["detail"] = "Updating prospect rankings..."
            try:
                from pulls import pull_prospects, pull_prospect_scouting
                pull_prospects.run(season)
                pull_prospect_scouting.run()
            except Exception as _pe:
                print(f"[refresh] prospect pull failed (non-fatal): {_pe}", flush=True)

        print("[refresh] refreshing computed stats...", flush=True)
        _status["detail"] = "Refreshing computed stats..."
        cache.clear_all()
        stuff_model.clear_cache()
        # Was missing - stuff_model's own internal cache got cleared on every
        # refresh but deception_model's equivalent didn't, so a "Refresh
        # data" click updated Stuff+ immediately but could leave Deception+
        # showing stale scores for up to its own 10-minute internal TTL.
        deception_model.clear_cache()
        # Box scores for games that were still in progress at last fetch
        # (most relevantly, today's games) need a fresh pull after new data
        # comes in - a finished game's box score never changes, but the
        # cache can't tell "finished" from "was in progress last time we
        # looked" on its own.
        boxscore.clear_cache()
        warm_cache_sync(full=full_warmup)

        _status["state"] = "done"
        new_pitches = pitches_result["rows_inserted"]
        if new_pitches == 0:
            _status["detail"] = f"No new pitches found - already up to date through {end.isoformat()}."
        else:
            _status["detail"] = (
                f"Up to date through {end.isoformat()}: {new_pitches:,} new pitches, "
                f"{games_result['games_updated']} game(s) updated, "
                f"{games_result['players_updated']} player(s) checked."
            )
        print(f"[refresh] done - {_status['detail']}", flush=True)
    except Exception as e:
        _status["state"] = "error"
        _status["detail"] = f"Refresh failed: {e}"
        print(f"[refresh] ERROR: {e}", flush=True)
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
