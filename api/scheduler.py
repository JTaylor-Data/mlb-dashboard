"""
Automatic daily data refresh - runs the full pull + cache-warm pipeline
once at server startup and again every day at DAILY_REFRESH_HOUR (local
server time), so the site is current without anyone needing to click
"Refresh data" and wait on it. That manual button still exists (see
api/pull.py) as an on-demand fallback for "I want this exact second," but
the schedule is the primary mechanism now - clicking through a ~30-70s
wait every session was the actual complaint that led here.

Both triggers use full_warmup=True (see pull.start_pull's docstring) -
nobody's watching a 2am or startup refresh finish, so it's worth fully
warming every percentile split rather than optimizing for someone staring
at a button, which is what full_warmup=False (the button's default) is for.

Weekly rescore
--------------
Every Sunday at 4:00 AM the score cache is refreshed: scripts/rescore_models.py
is run for the current season (past seasons are static and never need a rescore).
The fresh parquets are picked up on the NEXT server restart or cache refresh.
To get them into the live cache immediately after the rescore finishes, the
weekly job also invalidates the model TTL caches so the parquets are re-injected
on the next request (which hits warmup's usual background path).
"""
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from api import pull
from config.season_phase import SEASON_PHASE, REGULAR_SEASON

DAILY_REFRESH_HOUR = 6  # 6:00 AM local server time - by then, essentially
                         # all of the previous day's games have finished
                         # and their Statcast data has settled.

WEEKLY_RESCORE_WEEKDAY = 6   # Sunday (0=Monday, 6=Sunday)
WEEKLY_RESCORE_HOUR    = 4   # 4:00 AM local server time

ROOT = Path(__file__).resolve().parent.parent


def _seconds_until_next_run():
    now = datetime.now()
    target = now.replace(hour=DAILY_REFRESH_HOUR, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _seconds_until_next_sunday_rescore():
    """Seconds until next Sunday 4 AM."""
    now = datetime.now()
    days_until_sunday = (WEEKLY_RESCORE_WEEKDAY - now.weekday()) % 7
    # If today is Sunday but we've already passed 4 AM, aim for NEXT Sunday.
    if days_until_sunday == 0 and now.hour >= WEEKLY_RESCORE_HOUR:
        days_until_sunday = 7
    target = (now + timedelta(days=days_until_sunday)).replace(
        hour=WEEKLY_RESCORE_HOUR, minute=0, second=0, microsecond=0
    )
    return max(0.0, (target - now).total_seconds())


def _wait_for_pull_to_finish(poll_seconds=5):
    while pull.get_status()["state"] == "running":
        time.sleep(poll_seconds)


def _run_rescore(season: int):
    """Run Archive/misc_scripts/scripts/rescore_models.py for the current season in a subprocess."""
    script = ROOT / "Archive" / "misc_scripts" / "scripts" / "rescore_models.py"
    if not script.exists():
        print(f"[scheduler] rescore script not found at {script}, skipping")
        return
    try:
        print(f"[scheduler] starting weekly rescore for season {season}...")
        result = subprocess.run(
            [sys.executable, str(script), "--season", str(season), "--model", "all"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if result.returncode == 0:
            print(f"[scheduler] weekly rescore complete:\n{result.stdout[-2000:]}")
            # Invalidate model raw-predictions caches so next warmup/request
            # picks up the newly written parquets.
            try:
                from analytics.stuff_model import _raw_predictions_cache, clear_cache as stuff_clear
                stuff_clear()
            except Exception:
                pass
            try:
                from analytics.tunneling_v2_model import _cache as tun_cache
                tun_cache.clear()
            except Exception:
                pass
            try:
                from analytics.location_v3_model import _cache as loc_cache
                loc_cache.clear()
            except Exception:
                pass
            # Trigger a background warmup so the fresh parquets are hot
            # before the next user request arrives.
            from api.warmup import warm_cache_background
            warm_cache_background()
        else:
            print(f"[scheduler] weekly rescore failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
    except Exception as e:
        print(f"[scheduler] weekly rescore exception: {e}")


def _daily_loop():
    while True:
        time.sleep(_seconds_until_next_run())
        pull.start_pull(full_warmup=True)
        _wait_for_pull_to_finish()


def _weekly_rescore_loop():
    """Background thread that fires the rescore script every Sunday at 4 AM.
    Gated on SEASON_PHASE=REGULAR_SEASON — skipped during POSTSEASON and OFFSEASON
    (postseason pitch data doesn't affect historical parquets; offseason has no
    new pitches; only the current regular season accumulates pitches week-to-week)."""
    while True:
        time.sleep(_seconds_until_next_sunday_rescore())
        if SEASON_PHASE != REGULAR_SEASON:
            print(f"[scheduler] weekly rescore skipped: SEASON_PHASE={SEASON_PHASE} "
                  f"(only runs during REGULAR_SEASON)")
            time.sleep(3600)
            continue
        from api.warmup import CURRENT_SEASON
        _run_rescore(CURRENT_SEASON)
        # After firing, sleep at least 1 hour before re-checking so we don't
        # re-trigger immediately (the next _seconds_until_next_sunday_rescore
        # call will be safe after the rescore has advanced past the 4 AM window).
        time.sleep(3600)


def start():
    """Kicks off an immediate startup refresh plus the recurring daily
    schedule and weekly rescore, all in background threads - the app starts
    accepting requests right away regardless (any request that lands before
    the startup refresh finishes just recomputes/caches on demand, same as
    any other cold-cache request on this site)."""
    pull.start_pull(full_warmup=True)
    threading.Thread(target=_daily_loop, daemon=True).start()
    threading.Thread(target=_weekly_rescore_loop, daemon=True).start()
