"""
Pulls 20-80 scouting grades + a scouting report blurb for every prospect
already in the prospects table (run pull_prospects.py first), into
prospect_scouting.

MLB Pipeline's own site has none of this - the rankings pages are bio/rank
only. Baseball Savant's player page (baseballsavant.mlb.com/savant-player/
<mlb_id>) embeds MLB Pipeline's actual scouting write-ups instead, server-
rendered plain HTML (no headless browser needed here, unlike
pull_prospects.py). The page can carry several dated report blocks per
player (an "international"/pre-debut one, then one per year as they're
re-scouted) - this keeps only the most recent (highest numeric year label,
falling back to the last block on the page if none are numeric).

Grade tool names vary by role (Hit/Power/Run/Arm/Field for hitters,
Fastball/Curveball/Slider/Changeup/Control-ish for pitchers, and the exact
pitch-mix names vary pitcher to pitcher), so they're parsed generically as
whatever "Name: NN" pairs appear on the "Scouting grades:" line rather than
assumed to be a fixed set, and stored as JSON.

Usage:
    python pull_prospect_scouting.py

Not every prospect has a report yet (very new, unscouted-since-signing
prospects genuinely have no "Scouting Report" section on the page at all) -
those are just skipped, not stored as empty rows. Uses a small thread pool
(these are 900 independent page fetches, not writes) since this is a
sizeable page per player; a fully sequential pull would take much longer for
no benefit. Upserts, so re-running refreshes everything.
"""
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

DB_PATH = "mlb_data.db"
SAVANT_PLAYER_URL = "https://baseballsavant.mlb.com/savant-player/{}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_WORKERS = 10


def _parse_blocks(soup):
    header = soup.find(lambda tag: tag.name == "h2" and tag.get_text(strip=True) == "Scouting Report")
    if header is None:
        return []

    blocks = []
    current = None
    for sib in header.find_all_next():
        if sib.name == "h2":
            break
        if sib.name == "b":
            if current is not None:
                blocks.append(current)
            current = {"label": sib.get_text(strip=True), "paragraphs": []}
        elif sib.name == "p" and current is not None:
            current["paragraphs"].append(sib)
    if current is not None:
        blocks.append(current)

    parsed = []
    for b in blocks:
        grades, report_paras, video_url = {}, [], None
        for p in b["paragraphs"]:
            text = p.get_text(" ", strip=True)
            if text.startswith("Scouting grades"):
                for tool, val in re.findall(r"([A-Za-z][A-Za-z ]*?):\s*(\d+)", text):
                    grades[tool.strip()] = int(val)
            elif "Video scouting report" in text:
                a = p.find("a")
                video_url = a.get("href") if a else None
            else:
                report_paras.append(text)
        if grades:  # skip stray non-report blocks (shouldn't normally happen)
            parsed.append({
                "label": b["label"], "grades": grades,
                "report": "\n\n".join(report_paras), "video_url": video_url,
            })
    return parsed


def fetch_scouting(mlb_id):
    """Returns a dict for the most recent report block, or None if this
    prospect has no scouting report on the page at all."""
    resp = requests.get(SAVANT_PLAYER_URL.format(mlb_id), headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    blocks = _parse_blocks(soup)
    if not blocks:
        return None

    numeric = [b for b in blocks if b["label"].isdigit()]
    latest = max(numeric, key=lambda b: int(b["label"])) if numeric else blocks[-1]
    return latest


def run():
    conn = sqlite3.connect(DB_PATH)
    mlb_ids = sorted({r[0] for r in conn.execute("SELECT DISTINCT mlb_id FROM prospects")})
    print(f"Fetching scouting reports for {len(mlb_ids)} prospects...")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    no_report = []
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_scouting, mlb_id): mlb_id for mlb_id in mlb_ids}
        done = 0
        for future in as_completed(futures):
            mlb_id = futures[future]
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(mlb_ids)}")
            try:
                latest = future.result()
            except Exception as e:
                failures.append((mlb_id, str(e)))
                continue
            if latest is None:
                no_report.append(mlb_id)
                continue
            rows.append((
                mlb_id, latest["label"], json.dumps(latest["grades"]),
                latest["grades"].get("Overall"), latest["report"], latest["video_url"], now,
            ))

    conn.executemany(
        "INSERT INTO prospect_scouting (mlb_id, report_label, grades_json, overall_grade, "
        "report_text, video_url, updated_at) VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(mlb_id) DO UPDATE SET "
        "report_label=excluded.report_label, grades_json=excluded.grades_json, "
        "overall_grade=excluded.overall_grade, report_text=excluded.report_text, "
        "video_url=excluded.video_url, updated_at=excluded.updated_at",
        rows,
    )
    conn.commit()
    conn.close()

    print(f"\nUpserted {len(rows)} scouting reports.")
    print(f"  {len(no_report)} prospect(s) had no scouting report on the page yet.")
    if failures:
        print(f"  WARNING: {len(failures)} request(s) failed: {failures[:10]}")


if __name__ == "__main__":
    run()
