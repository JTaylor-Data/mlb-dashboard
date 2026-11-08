"""
Tiny TTL cache for the handful of endpoints that scan the full pitches table
to build a league-wide pool (percentile pools, Stuff+, pitcher roles,
standings, top-performers-by-date). Everything else queries by player_id/
game_id and is cheap enough to run fresh every request.

Streamlit's equivalent (@st.cache_data) only ever invalidated on a full app
restart - this is deliberately better: a short TTL so data doesn't go stale
forever, plus a manual /api/admin/refresh-cache endpoint (wired to a
"Refresh data" button in the nav) for right after re-running a pull script.
"""
import threading
import time
from functools import wraps

DEFAULT_TTL_SECONDS = 36 * 3600

# Caps how many distinct (args) keys a single cached function can hold at
# once - without this, a cache keyed by things like player_id/season/split
# combinations grows without bound over a long-running process. Eviction is
# plain FIFO (oldest-inserted key goes first, via dict insertion order) -
# simple rather than true LRU, which is fine here since the point is just to
# put a ceiling on memory, not to optimize hit rate.
DEFAULT_MAX_ENTRIES = 5000

_registry = []


def _get_lock(locks, locks_mu, key):
    with locks_mu:
        lock = locks.get(key)
        if lock is None:
            lock = threading.Lock()
            locks[key] = lock
        return lock


def _store_set(store, key, value):
    """Insert/overwrite a key, then evict oldest entries past the cap."""
    store.pop(key, None)  # re-insert at the end so a refresh counts as recent
    store[key] = value
    while len(store) > DEFAULT_MAX_ENTRIES:
        store.pop(next(iter(store)))


def _is_past_season_key(key, current_season):
    """True when a cache key names a season (a 2000-2100 int argument) and every
    such season is a finished one. A finished season's numbers never change, so
    new pitches only invalidate keys that involve the current season or no
    season at all (cross-season aggregates, per-player and per-date entries).
    MLB player, game and team ids never fall in 2000-2100."""
    args, kwargs = key
    seasons = [a for a in list(args) + [v for _, v in kwargs]
               if isinstance(a, int) and not isinstance(a, bool) and 2000 <= a <= 2100]
    return bool(seasons) and all(x < current_season for x in seasons)


def ttl_cache(ttl_seconds=DEFAULT_TTL_SECONDS):
    """
    Every cached function in this app takes a fresh sqlite3 connection as its
    first argument (see db.get_conn - a new object every request), so that
    argument is excluded from the cache key on purpose; otherwise every call
    would miss the cache.

    Single-flight per key: a cache miss acquires a lock scoped to that one
    key before building, so concurrent requests hitting the SAME cold key
    (two visitors landing on the player page while warm-up is still running,
    say) don't each trigger their own expensive build and stack their
    memory peaks - the second caller blocks, then gets the first caller's
    now-cached result. Different keys (different seasons/splits) still run
    concurrently; this only serializes repeats of the identical call.

    Bounded to DEFAULT_MAX_ENTRIES distinct keys (FIFO eviction) so a
    long-running process can't accumulate an unbounded number of cached
    player/season/split combinations.
    """
    def decorator(fn):
        store = {}
        locks = {}
        locks_mu = threading.Lock()

        @wraps(fn)
        def wrapper(conn, *args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            if key in store:
                value, expires_at = store[key]
                if now < expires_at:
                    return value
            lock = _get_lock(locks, locks_mu, key)
            with lock:
                # Re-check: whoever held the lock just before us may have
                # already built this key while we were waiting for it.
                now = time.monotonic()
                if key in store:
                    value, expires_at = store[key]
                    if now < expires_at:
                        return value
                value = fn(conn, *args, **kwargs)
                _store_set(store, key, (value, time.monotonic() + ttl_seconds))
            return value

        def clear():
            store.clear()

        def clear_current(current_season):
            for key in [k for k in store if not _is_past_season_key(k, current_season)]:
                store.pop(key, None)

        wrapper.clear = clear
        wrapper.clear_current = clear_current
        _registry.append(wrapper)
        return wrapper

    return decorator


def ttl_cache_all_args(ttl_seconds=DEFAULT_TTL_SECONDS):
    """Like ttl_cache, but for functions with no leading `conn` argument to
    exclude from the key (e.g. live external-API calls keyed by player_id).
    Same single-flight-per-key locking and FIFO entry cap as ttl_cache - see
    its docstring."""
    def decorator(fn):
        store = {}
        locks = {}
        locks_mu = threading.Lock()

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            if key in store:
                value, expires_at = store[key]
                if now < expires_at:
                    return value
            lock = _get_lock(locks, locks_mu, key)
            with lock:
                now = time.monotonic()
                if key in store:
                    value, expires_at = store[key]
                    if now < expires_at:
                        return value
                value = fn(*args, **kwargs)
                _store_set(store, key, (value, time.monotonic() + ttl_seconds))
            return value

        def clear():
            store.clear()

        def clear_current(current_season):
            for key in [k for k in store if not _is_past_season_key(k, current_season)]:
                store.pop(key, None)

        wrapper.clear = clear
        wrapper.clear_current = clear_current
        _registry.append(wrapper)
        return wrapper

    return decorator


def clear_all():
    for cached_fn in _registry:
        cached_fn.clear()


def clear_current_season(current_season):
    """What a scheduled refresh with new pitches should clear: everything that
    could involve the current season, but not the finished seasons' entries,
    which are unchanged and costly to rebuild on a small machine."""
    for cached_fn in _registry:
        cached_fn.clear_current(current_season)
