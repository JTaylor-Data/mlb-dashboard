"""Cold heavy builds are capped, cached/non-heavy work bypasses the cap, and a request that
cannot get a slot in time raises BusyError (HTTP 503 via main.py's handler)."""
import threading
import time

import pytest

from api import heavy_limiter as hl
from api.cache import ttl_cache


def _small(monkeypatch, n=2, wait=0.3, rss=900):
    monkeypatch.setattr(hl, "_slots", threading.BoundedSemaphore(n))
    monkeypatch.setattr(hl, "WAIT_S", wait)
    monkeypatch.setattr(hl, "_rss_mb", lambda: rss)  # high by default: the machine is "maxed out"


def test_path_matching():
    for ok in ("/api/player/543037", "/api/player/543037/progress", "/api/player/543037/similar",
               "/api/players/1/trends", "/api/trends/league-averages", "/api/compare"):
        assert hl.HEAVY_PATH.match(ok), ok
    for no in ("/api/meta", "/api/standings", "/api/heatmaps/1", "/api/players/all", "/api/news"):
        assert not hl.HEAVY_PATH.match(no), no


def test_busy_body():
    assert hl.busy_body() == {"busy": True, "message": "Busy right now, try again in a moment"}


def test_non_heavy_requests_never_wait(monkeypatch):
    _small(monkeypatch, n=1)
    with hl.cold_slot(), hl.cold_slot():  # not a heavy request: no slot taken at all
        pass


def test_cold_builds_capped_and_excess_gets_busy(monkeypatch):
    _small(monkeypatch, n=2, wait=0.2)
    state = {"now": 0, "peak": 0}
    results = []

    @ttl_cache()
    def build(conn, key):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.6)
        state["now"] -= 1
        return key

    def request(key):
        hl._is_heavy.set(True)
        try:
            results.append(build(None, key))
        except hl.BusyError:
            results.append("busy")

    threads = [threading.Thread(target=request, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["peak"] <= 2
    assert results.count("busy") == 3 and len([r for r in results if r != "busy"]) == 2


def test_cached_hits_bypass_the_cap(monkeypatch):
    _small(monkeypatch, n=1, wait=0.1)

    @ttl_cache()
    def build(conn, key):
        return key

    hl._is_heavy.set(True)
    assert build(None, 1) == 1  # cold: takes and frees the only slot
    with hl._slots:  # now someone else holds the only slot...
        assert build(None, 1) == 1  # ...and the cached hit still answers instantly
    hl._is_heavy.set(False)


def test_nested_cold_builds_do_not_deadlock(monkeypatch):
    _small(monkeypatch, n=1, wait=0.2)

    @ttl_cache()
    def inner(conn, k):
        return k

    @ttl_cache()
    def outer(conn, k):
        return inner(conn, k) + 1

    hl._is_heavy.set(True)
    try:
        assert outer(None, 1) == 2
    finally:
        hl._is_heavy.set(False)


def test_slots_full_but_memory_fine_keeps_waiting(monkeypatch):
    _small(monkeypatch, n=1, wait=0.1, rss=300)
    monkeypatch.setattr(hl, "MAX_WAIT_S", 5.0)
    hl._is_heavy.set(True)
    try:
        threading.Timer(0.35, hl._slots.release).start()
        hl._slots.acquire()  # someone else holds the only slot for ~0.35 s
        with hl.cold_slot():  # waits several WAIT_S rounds, no BusyError, then runs
            pass
    finally:
        hl._is_heavy.set(False)
