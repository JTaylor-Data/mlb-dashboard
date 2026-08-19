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
import time
from functools import wraps

DEFAULT_TTL_SECONDS = 12 * 3600

_registry = []


def ttl_cache(ttl_seconds=DEFAULT_TTL_SECONDS):
    """
    Every cached function in this app takes a fresh sqlite3 connection as its
    first argument (see db.get_conn - a new object every request), so that
    argument is excluded from the cache key on purpose; otherwise every call
    would miss the cache.
    """
    def decorator(fn):
        store = {}

        @wraps(fn)
        def wrapper(conn, *args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            if key in store:
                value, expires_at = store[key]
                if now < expires_at:
                    return value
            value = fn(conn, *args, **kwargs)
            store[key] = (value, now + ttl_seconds)
            return value

        def clear():
            store.clear()

        wrapper.clear = clear
        _registry.append(wrapper)
        return wrapper

    return decorator


def ttl_cache_all_args(ttl_seconds=DEFAULT_TTL_SECONDS):
    """Like ttl_cache, but for functions with no leading `conn` argument to
    exclude from the key (e.g. live external-API calls keyed by player_id)."""
    def decorator(fn):
        store = {}

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            if key in store:
                value, expires_at = store[key]
                if now < expires_at:
                    return value
            value = fn(*args, **kwargs)
            store[key] = (value, now + ttl_seconds)
            return value

        def clear():
            store.clear()

        wrapper.clear = clear
        _registry.append(wrapper)
        return wrapper

    return decorator


def clear_all():
    for cached_fn in _registry:
        cached_fn.clear()
