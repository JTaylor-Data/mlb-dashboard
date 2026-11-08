"""Cap how many COLD heavy builds run at once, answer "busy" beyond that, and hand
freed memory back to the OS.

The player, trends and compare endpoints build season-wide tables on a cold
cache. On the 1 GB Fly machine a burst of those in parallel peaked at ~1 GB and
starved the one shared CPU. Only a cache miss is expensive, so the gate sits on
the cache-miss path (api/cache.py calls cold_slot() around the build), not on
the request: cached responses never touch it. At most HEAVY_LIMIT cold builds run
at once; a request that cannot get a slot within WAIT_S seconds gets BusyError,
which main.py turns into HTTP 503 {"busy": true, ...} with Retry-After. The
per-key single-flight lock in api/cache.py is unchanged.

glibc keeps freed pandas buffers in its arenas instead of returning them. After a
heavy request finishes (at most every MIN_TRIM_GAP_S seconds) and every
IDLE_TRIM_S seconds while no heavy request runs, gc.collect() + malloc_trim(0)
run and RSS before/after is logged.
"""
import asyncio
import contextlib
import contextvars
import ctypes
import gc
import logging
import re
import threading
import time

log = logging.getLogger("heavy_limiter")

HEAVY_LIMIT = 10
WAIT_S = 5.0
# "Busy" means the machine is actually maxed out: every slot taken for WAIT_S seconds AND
# memory near the 1 GB limit. If slots are full but memory is fine (a couple of visitors
# whose pages each start a few builds), the request keeps waiting, up to MAX_WAIT_S.
BUSY_RSS_MB = 800
MAX_WAIT_S = 40.0
RETRY_AFTER_S = 10
BUSY_MESSAGE = "Busy right now, try again in a moment"
IDLE_TRIM_S = 60
MIN_TRIM_GAP_S = 2.0

# /api/player/<id> and its streamed /progress twin, /similar, /tunneling-pairs;
# /api/players/<id>/trends; /api/trends/*; /api/compare
HEAVY_PATH = re.compile(
    r"^/api/(player/\d+(/progress|/similar|/tunneling-pairs)?|players/\d+/trends|trends/[^/]+|compare)/?$"
)

_is_heavy = contextvars.ContextVar("heavy_request", default=False)
_holding = contextvars.ContextVar("holding_cold_slot", default=False)
_slots = threading.BoundedSemaphore(HEAVY_LIMIT)
_in_flight = 0
_lock = threading.Lock()
_last_trim = 0.0
_libc = None


class BusyError(Exception):
    """No cold-build slot became free within WAIT_S."""


@contextlib.contextmanager
def cold_slot():
    """Wrap a cache-miss build. Only requests to heavy endpoints are gated (warm-up
    and other endpoints pass straight through), and a request that already holds a
    slot (nested cached calls) does not take a second one."""
    if not _is_heavy.get() or _holding.get():
        yield
        return
    waited = 0.0
    while not _slots.acquire(timeout=WAIT_S):
        waited += WAIT_S
        rss = _rss_mb()
        if (rss == rss and rss >= BUSY_RSS_MB) or waited >= MAX_WAIT_S:  # rss != rss: unreadable
            raise BusyError(BUSY_MESSAGE)
    token = _holding.set(True)
    try:
        yield
    finally:
        _holding.reset(token)
        _slots.release()


def busy_body():
    return {"busy": True, "message": BUSY_MESSAGE}


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def trim_memory(reason):
    """gc.collect() + malloc_trim(0), logging RSS before and after. No-op off Linux/glibc."""
    global _libc, _last_trim
    with _lock:
        if time.monotonic() - _last_trim < MIN_TRIM_GAP_S:
            return
        _last_trim = time.monotonic()
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return
    before = _rss_mb()
    gc.collect()
    _libc.malloc_trim(0)
    log.info("memory trim (%s): RSS %.0f MB -> %.0f MB", reason, before, _rss_mb())


def _idle_loop():
    while True:
        time.sleep(IDLE_TRIM_S)
        if _in_flight == 0:
            try:
                trim_memory("idle")
            except Exception:
                log.exception("idle memory trim failed")


class HeavyRequestMarker:
    """Pure-ASGI middleware: marks requests to heavy endpoints (so their cache
    misses go through cold_slot) and trims memory when one finishes. It never
    queues or delays a request itself."""

    def __init__(self, app, start_idle_timer=True):
        self.app = app
        if start_idle_timer:
            threading.Thread(target=_idle_loop, daemon=True, name="heavy-idle-trim").start()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not HEAVY_PATH.match(scope["path"]):
            return await self.app(scope, receive, send)
        global _in_flight
        with _lock:
            _in_flight += 1
        token = _is_heavy.set(True)
        try:
            await self.app(scope, receive, send)
        finally:
            _is_heavy.reset(token)
            with _lock:
                _in_flight -= 1
            asyncio.get_running_loop().run_in_executor(None, trim_memory, "after heavy request")
