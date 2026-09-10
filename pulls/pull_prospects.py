"""
Pulls MLB Pipeline prospect rankings (overall Top 100 + each org's Top 30)
into the prospects table, then attaches a real season stat line (GP, K%,
BB%, OPS/ERA) from the official MLB Stats API for each prospect.

There's no official rankings API - MLB Pipeline's rankings pages
(mlb.com/milb/prospects/<team-slug>) render their tables client-side (the
first 5 rows are server-rendered for SEO, the rest only appear in the DOM
after the "Show Full List" button is clicked and React renders from data it
already fetched on mount - confirmed by inspecting network activity, no new
request fires on click). That means a plain `requests.get()` only ever sees
5 rows; this script drives a real (headless) browser with Playwright instead
and reads the rendered table out of the DOM, same as a human clicking around
the site would see.

Each prospect's MLBAM person id is recovered from their profile link
(mlb.com/stories/<slug>-<id>), not from their headshot image (some
prospects' headshot URLs don't embed an id at all). That id is then used to
pull real season stats from statsapi.mlb.com at the sportId matching their
Pipeline-listed level (AAA/AA/A+/A/ROK) - genuinely official numbers, not
scraped off the rankings page (which has none).

Usage:
    python pull_prospects.py [SEASON]   # defaults to 2026

Rankings pages take a few seconds each to render (31 page loads: Top 100 +
30 team pages) - that part is inherently sequential (one shared Playwright
page navigating site to site) and stays the slower half of a run. The
~900-prospect stat-line lookup against statsapi is a completely
independent per-player GET, so it runs on a small thread pool (same
pattern as pull_pitcher_bio.py's bio lookups) instead of one request at a
time - a real speedup on what used to be "expect this to take several
minutes" almost entirely from that sequential loop. Upserts, so
re-running (e.g. after a midseason Pipeline update) just refreshes
everything. Not wired into the nav's "Refresh data" button -
rankings/stats don't change day to day the way live games do, so this is
meant to be run by hand occasionally, same as pull_teams.py.
"""
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from playwright.sync_api import sync_playwright

DB_PATH = "mlb_data.db"
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_SEASON = 2026

TEAM_SLUGS = {
    "Arizona Diamondbacks": "dbacks",
    "Athletics": "athletics",
    "Atlanta Braves": "braves",
    "Baltimore Orioles": "orioles",
    "Boston Red Sox": "redsox",
    "Chicago Cubs": "cubs",
    "Chicago White Sox": "whitesox",
    "Cincinnati Reds": "reds",
    "Cleveland Guardians": "guardians",
    "Colorado Rockies": "rockies",
    "Detroit Tigers": "tigers",
    "Houston Astros": "astros",
    "Kansas City Royals": "royals",
    "Los Angeles Angels": "angels",
    "Los Angeles Dodgers": "dodgers",
    "Miami Marlins": "marlins",
    "Milwaukee Brewers": "brewers",
    "Minnesota Twins": "twins",
    "New York Mets": "mets",
    "New York Yankees": "yankees",
    "Philadelphia Phillies": "phillies",
    "Pittsburgh Pirates": "pirates",
    "San Diego Padres": "padres",
    "San Francisco Giants": "giants",
    "Seattle Mariners": "mariners",
    "St. Louis Cardinals": "cardinals",
    "Tampa Bay Rays": "rays",
    "Texas Rangers": "rangers",
    "Toronto Blue Jays": "bluejays",
    "Washington Nationals": "nationals",
}

# MLB Stats API sportId for each Pipeline-listed level. Includes MLB (1) -
# some Top 30 prospects are rookie-eligible call-ups already in the majors.
LEVEL_TO_SPORT_ID = {"AAA": 11, "AA": 12, "A+": 13, "A": 14, "ROK": 16, "MLB": 1}

ID_FROM_STORY_LINK = re.compile(r"-(\d+)$")

ROW_EXTRACT_JS = """
() => [...document.querySelectorAll('table.rankings__table tbody tr')].map(tr => {
    const cells = [...tr.children].map(td => td.textContent.trim());
    const link = tr.querySelector('a[href]');
    return { cells, href: link ? link.getAttribute('href') : null };
})
"""


def _mlb_id_from_href(href):
    if not href:
        return None
    m = ID_FROM_STORY_LINK.search(href.split("?")[0])
    return int(m.group(1)) if m else None


def _is_pitcher(position):
    return position.startswith("RHP") or position.startswith("LHP")


def _scrape_table(page, url):
    """Loads a rankings page, expands it via "Show Full List" if present,
    and returns the rendered rows as raw cell-text lists."""
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        page.wait_for_selector("table.rankings__table tbody tr", timeout=15000)
    except Exception:
        return []
    # The table always renders with just its first 5 rows; "Show Full List"
    # doesn't fetch anything new (confirmed - no request fires on click), it
    # just reveals rows React already has in memory. Two gotchas found by
    # trial and error: Playwright's normal page.click() (real pointer
    # actions) doesn't register with this button's handler in headless mode
    # - a dispatched MouseEvent does. And the button exists in the DOM (SSR)
    # before React finishes hydrating and attaches its onClick listener, so
    # clicking too early is a silent no-op - a short unconditional wait
    # first, plus a retry if the row count didn't grow, covers both.
    try:
        page.wait_for_selector('[data-testid="load-more-button"]', timeout=8000)
    except Exception:
        return page.evaluate(ROW_EXTRACT_JS)  # no load-more button - this list already fit in 5 rows

    page.wait_for_timeout(3000)
    for _ in range(2):
        page.evaluate("""() => {
            const btn = document.querySelector('[data-testid="load-more-button"]');
            if (btn) btn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: window}));
        }""")
        page.wait_for_timeout(2500)
        if len(page.query_selector_all("table.rankings__table tbody tr")) > 5:
            break
    return page.evaluate(ROW_EXTRACT_JS)


def scrape_rankings(season):
    """Returns (rows, top100_ranks) - rows is a list of raw prospect dicts
    (one per org Top 30 appearance), top100_ranks maps mlb_id -> overall rank."""
    rows = []
    top100_ranks = {}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        print("Scraping overall Top 100...", end=" ", flush=True)
        top100 = _scrape_table(page, "https://www.mlb.com/milb/prospects")
        for r in top100:
            if len(r["cells"]) < 10:
                continue
            mlb_id = _mlb_id_from_href(r["href"])
            if mlb_id is not None:
                top100_ranks[mlb_id] = int(r["cells"][0])
        print(f"{len(top100_ranks)} ranked players")

        for team_name, slug in TEAM_SLUGS.items():
            print(f"Scraping {team_name} Top 30...", end=" ", flush=True)
            team_rows = _scrape_table(page, f"https://www.mlb.com/milb/prospects/{slug}")
            count = 0
            for r in team_rows:
                cells = r["cells"]
                if len(cells) < 10:
                    continue
                mlb_id = _mlb_id_from_href(r["href"])
                if mlb_id is None:
                    continue
                rows.append({
                    "mlb_id": mlb_id,
                    "team_name": team_name,
                    "org_rank": int(cells[0]),
                    "name": cells[1],
                    "position": cells[2],
                    "level": cells[4],
                    "eta": int(cells[5]) if cells[5].isdigit() else None,
                    "age": int(cells[6]) if cells[6].isdigit() else None,
                    "height_weight": cells[7],
                    "bats": cells[8],
                    "throws": cells[9],
                })
                count += 1
            print(f"{count} rows")

        browser.close()

    return rows, top100_ranks


def fetch_stat_line(session, mlb_id, level, is_pitcher, season):
    """Returns (gp, k_pct, bb_pct, ops, era) from statsapi at the sportId
    matching the prospect's Pipeline-listed level, or all-None if the level
    isn't recognized or the player has no season stats there (e.g. injured
    all year, just signed)."""
    sport_id = LEVEL_TO_SPORT_ID.get(level)
    if sport_id is None:
        return None, None, None, None, None

    group = "pitching" if is_pitcher else "hitting"
    url = f"{STATSAPI_BASE}/people/{mlb_id}/stats"
    params = {"stats": "season", "group": group, "season": season, "sportId": sport_id}
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None, None, None, None, None

    splits = (data.get("stats") or [{}])[0].get("splits") or []
    if not splits:
        return None, None, None, None, None
    stat = splits[0]["stat"]

    gp = stat.get("gamesPlayed")
    if is_pitcher:
        bf = stat.get("battersFaced") or 0
        k_pct = round(100 * stat["strikeOuts"] / bf, 1) if bf else None
        bb_pct = round(100 * stat["baseOnBalls"] / bf, 1) if bf else None
        era = float(stat["era"]) if stat.get("era") not in (None, "-.--") else None
        return gp, k_pct, bb_pct, None, era
    else:
        pa = stat.get("plateAppearances") or 0
        k_pct = round(100 * stat["strikeOuts"] / pa, 1) if pa else None
        bb_pct = round(100 * stat["baseOnBalls"] / pa, 1) if pa else None
        ops = float(stat["ops"]) if stat.get("ops") not in (None, "-.--") else None
        return gp, k_pct, bb_pct, ops, None


def run(season: int):
    conn = sqlite3.connect(DB_PATH)
    team_ids = {name: tid for tid, name in conn.execute("SELECT team_id, name FROM teams")}

    rows, top100_ranks = scrape_rankings(season)

    # Some prospects have offseason/complex-league stints on top of their
    # main Pipeline-listed level, or appear on the team page more than once
    # across a re-scrape; keep the first (highest org-ranked) occurrence.
    by_id = {}
    for r in rows:
        by_id.setdefault(r["mlb_id"], r)

    print(f"\nFetching season stats for {len(by_id)} prospects (10 concurrent)...")
    session = requests.Session()  # thread-safe for concurrent .get() - urllib3's
                                   # connection pooling underneath is what makes this
                                   # a real speedup rather than just more overhead
    now = datetime.now(timezone.utc).isoformat()
    unmatched_teams = set()
    unmapped_levels = set()

    def fetch_one(mlb_id, r):
        is_pitcher = _is_pitcher(r["position"])
        gp, k_pct, bb_pct, ops, era = fetch_stat_line(session, mlb_id, r["level"], is_pitcher, season)
        return mlb_id, r, is_pitcher, (gp, k_pct, bb_pct, ops, era)

    results = {}
    with ThreadPoolExecutor(10) as pool:
        futures = {pool.submit(fetch_one, mlb_id, r): mlb_id for mlb_id, r in by_id.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            mlb_id, r, is_pitcher, stat_line = fut.result()
            results[mlb_id] = (r, is_pitcher, stat_line)
            if i % 100 == 0:
                print(f"  ...{i}/{len(by_id)}")

    db_rows = []
    for mlb_id, r in by_id.items():  # stable order, not thread-completion order
        _r, is_pitcher, (gp, k_pct, bb_pct, ops, era) = results[mlb_id]

        team_id = team_ids.get(r["team_name"])
        if team_id is None:
            unmatched_teams.add(r["team_name"])
        if r["level"] not in LEVEL_TO_SPORT_ID:
            unmapped_levels.add(r["level"])

        headshot = f"https://img.mlbstatic.com/mlb-photos/image/upload/c_fill,g_auto/w_360/v1/people/{mlb_id}/headshot/milb/current"

        db_rows.append((
            mlb_id, season, r["name"], r["position"], int(is_pitcher), team_id,
            r["org_rank"], top100_ranks.get(mlb_id), r["level"], r["eta"], r["age"],
            r["height_weight"], r["bats"], r["throws"], headshot,
            season, gp, k_pct, bb_pct, ops, era, now,
        ))

    conn.executemany(
        "INSERT INTO prospects (mlb_id, season, name, position, is_pitcher, team_id, "
        "org_rank, top100_rank, level, eta, age, height_weight, bats, throws, headshot_url, "
        "stats_season, gp, k_pct, bb_pct, ops, era, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(mlb_id, season) DO UPDATE SET "
        "name=excluded.name, position=excluded.position, is_pitcher=excluded.is_pitcher, "
        "team_id=excluded.team_id, org_rank=excluded.org_rank, top100_rank=excluded.top100_rank, "
        "level=excluded.level, eta=excluded.eta, age=excluded.age, "
        "height_weight=excluded.height_weight, bats=excluded.bats, throws=excluded.throws, "
        "headshot_url=excluded.headshot_url, stats_season=excluded.stats_season, gp=excluded.gp, "
        "k_pct=excluded.k_pct, bb_pct=excluded.bb_pct, ops=excluded.ops, era=excluded.era, "
        "updated_at=excluded.updated_at",
        db_rows,
    )
    conn.commit()
    conn.close()

    have_stats = sum(1 for row in db_rows if row[16] is not None)
    print(f"\nUpserted {len(db_rows)} prospects ({len(top100_ranks)} on the Top 100, "
          f"{have_stats} with a season stat line).")
    if unmatched_teams:
        print(f"  WARNING: {len(unmatched_teams)} team name(s) didn't match the teams table: {sorted(unmatched_teams)}")
    if unmapped_levels:
        print(f"  WARNING: {len(unmapped_levels)} level string(s) had no sportId mapping (stats skipped): {sorted(unmapped_levels)}")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
