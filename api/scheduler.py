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
"""
import threading
import time
from datetime import datetime, timedelta

from api import pull

DAILY_REFRESH_HOUR = 6  # 6:00 AM local server time - by then, essentially
                         # all of the previous day's games have finished
                         # and their Statcast data has settled.


def _seconds_until_next_run():
    now = datetime.now()
    target = now.replace(hour=DAILY_REFRESH_HOUR, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _wait_for_pull_to_finish(poll_seconds=5):
    while pull.get_status()["state"] == "running":
        time.sleep(poll_seconds)


def _daily_loop():
    while True:
        time.sleep(_seconds_until_next_run())
        pull.start_pull(full_warmup=True)
        _wait_for_pull_to_finish()


def start():
    """Kicks off an immediate startup refresh plus the recurring daily
    schedule, both in background threads - the app starts accepting
    requests right away regardless (any request that lands before the
    startup refresh finishes just recomputes/caches on demand, same as
    any other cold-cache request on this site)."""
    pull.start_pull(full_warmup=True)
    threading.Thread(target=_daily_loop, daemon=True).start()
