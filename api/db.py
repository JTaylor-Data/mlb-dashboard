"""
FastAPI dependency for SQLite access. Opens a fresh connection per request
and closes it after - avoids sharing one connection across FastAPI's
threadpool, and keeps every request's data as current as whatever's on disk
right now (subject to the TTL cache in cache.py for the expensive
league-wide queries).
"""
import sqlite3

DB_PATH = "mlb_data.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()
