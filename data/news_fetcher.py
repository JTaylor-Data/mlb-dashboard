"""
News fetcher for the MLB Dashboard news page (Part E).

Source: MLB.com RSS feed at https://www.mlb.com/feeds/news/rss.xml
Confirmed working (200 OK, 25 items) via direct requests test on 2026-09-01.

Anthropic API / summarization:
  The `anthropic` Python package is NOT installed in this environment
  (confirmed by attempting `import anthropic` in the project Python 3.14).
  Summarization is therefore skipped — the summary field is populated with
  the first ~200 characters of the headline as a graceful fallback.

  To enable real 2-sentence summaries:
    1. pip install anthropic
    2. Set ANTHROPIC_API_KEY in the environment or in a .env file at the
       project root (mlb_dashboard/.env — loaded below via python-dotenv
       if present, no error if dotenv itself isn't installed).
    3. Re-run this module — it will automatically detect the package and
       key and use claude-haiku-4-5-20251001 for new, unsummarized stories.
       Already-stored summaries are not re-generated.

NewsAPI.org note:
  A paid NewsAPI.org subscription (~$50/mo for the Developer plan) would
  provide broader coverage across hundreds of baseball beat sources (beat
  writers, local papers, ESPN, The Athletic) rather than just MLB.com's own
  editorial feed. It would meaningfully improve story diversity but is not
  required for the core use case of tracking official MLB news.
"""
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime
import requests

RSS_URL = "https://www.mlb.com/feeds/news/rss.xml"
TIMEOUT = 15
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            headline    TEXT,
            source_url  TEXT UNIQUE,
            pub_date    TEXT,
            summary     TEXT,
            description TEXT,
            priority    INTEGER DEFAULT 1,
            summarized  INTEGER DEFAULT 0,
            fetched_at  TEXT
        )
    """)
    for col, typedef in [("description", "TEXT"), ("priority", "INTEGER DEFAULT 1")]:
        try:
            conn.execute(f"ALTER TABLE news_items ADD COLUMN {col} {typedef}")
        except Exception:
            pass
    conn.commit()


_HIGH_PRIORITY_TERMS = [
    "called up", "call-up", "callup", "promoted", "recalled",
    "walk-off", "walkoff", "no-hitter", "no hitter", "perfect game",
    "cycle", "injured list", "il ", " il,", "placed on", "optioned",
    "traded", "trade", "released", "designated for assignment", "dfa",
    "contract", "extension", "suspended", "ejected", "brawl",
    "marte", "ohtani", "judge", "acuna", "acuña",
]
_LOW_PRIORITY_TERMS = [
    "dog", "cat", "pet", "hug", "funny", "viral", "challenge", "dance",
    "prank", "mascot", "food", "giveaway", "bobblehead", "fireworks night",
]

def _priority(headline: str) -> int:
    h = headline.lower()
    if any(t in h for t in _LOW_PRIORITY_TERMS):
        return 0
    if any(t in h for t in _HIGH_PRIORITY_TERMS):
        return 2
    return 1

def _scrape_og_description(url: str) -> str:
    """Fetch an MLB.com article and extract its og:description meta tag."""
    try:
        import re as _re
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        m = _re.search(r'property="og:description"\s+content="([^"]+)"', r.text)
        if not m:
            m = _re.search(r'content="([^"]+)"\s+property="og:description"', r.text)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


def _parse_rss(text: str) -> list[dict]:
    root = ET.fromstring(text)
    channel = root.find("channel")
    items = []
    for item in channel.findall("item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        # description often has a 1-2 sentence article blurb
        desc = item.findtext("description") or ""
        # Strip HTML tags from description
        import re
        desc = re.sub(r"<[^>]+>", "", desc).strip()
        items.append({
            "headline": title.strip(),
            "source_url": link.strip(),
            "pub_date": pub_date,
            "description": desc[:400] if desc else "",
            "priority": _priority(title),
        })
    # Sort: high priority first, then by position in feed (most recent)
    items.sort(key=lambda x: x["priority"], reverse=True)
    return items


def _try_summarize(headline: str, description: str, anthropic_key: str) -> str | None:
    """Call claude-haiku-4-5-20251001 for a 3-sentence summary using both the
    headline and the RSS description (article blurb). Returns None on any
    failure so the fallback kicks in."""
    try:
        import anthropic  # noqa: F401
        client = anthropic.Anthropic(api_key=anthropic_key)
        context = f"Headline: {headline}"
        if description and description.strip() and description.strip() != headline:
            context += f"\nArticle snippet: {description.strip()[:600]}"
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=220,
            messages=[{
                "role": "user",
                "content": (
                    "Write a 3-sentence plain-English summary of this MLB news article "
                    "for a baseball fan briefing. Be specific — mention key players, "
                    "stats, outcomes, or context if present. Do not restate the headline "
                    "verbatim. Write in active voice.\n\n"
                    + context
                ),
            }],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


def fetch_and_store(db_path: str = "mlb_data.db", max_items: int = 10) -> int:
    """
    Fetch the MLB RSS feed, store new items in news_items, and attempt
    to summarize any that don't have a summary yet.

    Returns the number of NEW items inserted (not total).
    """
    # Try loading .env for ANTHROPIC_API_KEY
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    except ImportError:
        pass

    import os
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    resp = requests.get(RSS_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    items = _parse_rss(resp.text)[:max_items]

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_table(conn)

    fetched_at = datetime.utcnow().isoformat()
    new_count = 0

    for item in items:
        # Skip if already in DB (UNIQUE on source_url)
        existing = conn.execute(
            "SELECT id, summary, summarized FROM news_items WHERE source_url = ?",
            (item["source_url"],),
        ).fetchone()

        # If RSS description is empty, scrape og:description from the article page
        desc = (item.get("description") or "").strip()
        if not desc and item.get("source_url"):
            desc = _scrape_og_description(item["source_url"])
            item["description"] = desc

        if existing is None:
            # New story — generate summary
            summary = None
            summarized = 0
            if anthropic_key:
                summary = _try_summarize(item["headline"], desc, anthropic_key)
                if summary:
                    summarized = 1
            if not summary:
                # Fallback: scraped/RSS description blurb
                summary = desc[:300] if desc else item["headline"][:200]

            conn.execute(
                """INSERT INTO news_items (headline, source_url, pub_date, summary, description, priority, summarized, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (item["headline"], item["source_url"], item["pub_date"], summary,
                 desc, item.get("priority", 1), summarized, fetched_at),
            )
            new_count += 1
        else:
            # Existing story — update empty description and/or re-summarize
            row_id, existing_summary, summarized = existing
            existing_desc = conn.execute(
                "SELECT description FROM news_items WHERE id = ?", (row_id,)
            ).fetchone()
            existing_desc = (existing_desc[0] or "").strip() if existing_desc else ""
            if not existing_desc and desc:
                conn.execute(
                    "UPDATE news_items SET description = ? WHERE id = ?", (desc, row_id)
                )
                # Also update summary if it was the headline fallback
                if existing_summary == item["headline"] or existing_summary == item["headline"][:200]:
                    conn.execute(
                        "UPDATE news_items SET summary = ? WHERE id = ?", (desc[:300], row_id)
                    )
            if not summarized and anthropic_key:
                summary = _try_summarize(item["headline"], desc or existing_desc, anthropic_key)
                if summary:
                    conn.execute(
                        "UPDATE news_items SET summary = ?, summarized = 1 WHERE id = ?",
                        (summary, row_id),
                    )

    # Backfill any existing rows that still have empty descriptions
    stale = conn.execute(
        "SELECT id, headline, source_url, summary FROM news_items "
        "WHERE (description IS NULL OR description = '') AND source_url IS NOT NULL LIMIT 20"
    ).fetchall()
    for row_id, headline, url, existing_summary in stale:
        desc = _scrape_og_description(url)
        if desc:
            conn.execute("UPDATE news_items SET description = ? WHERE id = ?", (desc, row_id))
            if not existing_summary or existing_summary == headline or existing_summary == headline[:200]:
                conn.execute("UPDATE news_items SET summary = ? WHERE id = ?", (desc[:300], row_id))

    conn.commit()
    conn.close()
    return new_count


def get_news(db_path: str = "mlb_data.db", limit: int = 10) -> list[dict]:
    """Read stored news from the DB, newest first."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    _ensure_table(conn)
    rows = conn.execute(
        """SELECT id, headline, source_url, pub_date, summary, summarized, fetched_at
           FROM news_items ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0], "headline": r[1], "source_url": r[2], "pub_date": r[3],
            "summary": r[4], "summarized": bool(r[5]), "fetched_at": r[6],
        }
        for r in rows
    ]
