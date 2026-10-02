"""
FastAPI dependency for SQLite access. Opens a fresh connection per request
and closes it after - avoids sharing one connection across FastAPI's
threadpool, and keeps every request's data as current as whatever's on disk
right now (subject to the TTL cache in cache.py for the expensive
league-wide queries).
"""
import os
import sqlite3

# Default to the repo-relative path for local dev; override with DB_PATH env
# var in hosted environments where the database lives on a persistent volume.
DB_PATH = os.environ.get("DB_PATH", "mlb_data.db")

# WAL mode persists in the DB file once set, so only needs to be enabled once
# at process start, not on every connection. All other PRAGMAs are
# per-connection and are batched into a single executescript() call to save
# 4 round-trips per request vs 5 individual execute() calls.
def _ensure_wal():
    _c = sqlite3.connect(DB_PATH)
    _c.execute("PRAGMA journal_mode=WAL")
    _c.close()

_ensure_wal()

_PER_CONN_PRAGMAS = """
PRAGMA busy_timeout=5000;
PRAGMA cache_size=-32000;
PRAGMA temp_store=MEMORY;
PRAGMA mmap_size=268435456;
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.executescript(_PER_CONN_PRAGMAS)
    try:
        yield conn
    finally:
        conn.close()
