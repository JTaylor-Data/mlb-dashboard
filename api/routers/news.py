"""
/api/news — serves MLB news items from the news_items table (populated by
data/news_fetcher.py, triggered by pulls/daily/refresh_mlb_stats.py daily).
"""
from pathlib import Path

from fastapi import APIRouter, Depends, Query


from api.db import get_conn

router = APIRouter(prefix="/api")


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            headline    TEXT,
            source_url  TEXT UNIQUE,
            pub_date    TEXT,
            summary     TEXT,
            summarized  INTEGER DEFAULT 0,
            fetched_at  TEXT
        )
    """)
    conn.commit()


@router.get("/news")
def get_news(limit: int = Query(10, ge=1, le=50), conn=Depends(get_conn)):
    """Latest MLB news, newest-first. Populated by pulls/daily/refresh_mlb_stats.py."""
    _ensure_table(conn)
    rows = conn.execute(
        """SELECT id, headline, source_url, pub_date, summary,
                  COALESCE(description, '') as description,
                  COALESCE(priority, 1) as priority,
                  summarized, fetched_at
           FROM news_items
           ORDER BY COALESCE(priority, 1) DESC, id DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return {
        "items": [
            {
                "id": r[0],
                "headline": r[1],
                "source_url": r[2],
                "pub_date": r[3],
                "summary": r[4],
                "description": r[5],
                "priority": r[6],
                "summarized": bool(r[7]),
                "fetched_at": r[8],
            }
            for r in rows
        ]
    }
