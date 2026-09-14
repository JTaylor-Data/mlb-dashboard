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
    # 32MB page cache (vs default 8MB). Full-season pitches scans read ~100MB+
    # of data; a larger cache means the second scan of the same pages (e.g.
    # pitcher pool then batter pool, both cold) hits OS/SQLite cache instead
    # of disk. Negative value = size in KB; 32000 = 32MB.
    conn.execute("PRAGMA cache_size = -32000")
    # Keep sort/GROUP BY/ORDER BY work in memory rather than spilling to a temp
    # file. Materializing a full-season aggregation to disk and reading it back
    # was a hidden cost on every cold pool build.
    conn.execute("PRAGMA temp_store = MEMORY")
    # Memory-map up to 256MB of the DB file. After the first full-table read,
    # subsequent reads of the same pages come from OS-mapped memory rather than
    # read() system calls — a meaningful win for repeated full-season scans
    # (percentile pools, stuff model, warmup) on the same server process.
    conn.execute("PRAGMA mmap_size = 268435456")
    try:
        yield conn
    finally:
        conn.close()
