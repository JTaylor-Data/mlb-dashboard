"""
Downloads high-home-plate aerial stadium photos for each MLB venue.
Saves to frontend/img/stadiums/{venue_id}.jpg.

Search strategy: queries Wikimedia Commons for each stadium using
progressively broader angle-specific terms ("aerial", "panorama", "bird",
"overhead") and picks the widest landscape-oriented result — wide-and-short
images are almost always overhead/press-box shots, not fan-level snaps.
Re-run with --force to replace existing files.
"""
import sys, time, requests
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "img" / "stadiums"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORCE = "--force" in sys.argv

HEADERS = {"User-Agent": "mlb-dashboard-local/1.0 (contact: josh.scott.tay@gmail.com)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# venue_id → (primary stadium name, [search suffixes in priority order])
# Suffixes are appended to the stadium name when querying Wikimedia Commons.
VENUES = {
    32:   ("American Family Field",         ["aerial", "panorama", "overhead", "bird"]),
    1:    ("Angel Stadium",                 ["aerial", "panorama", "overhead"]),
    2889: ("Busch Stadium",                 ["aerial", "panorama", "overhead"]),
    15:   ("Chase Field",                   ["aerial", "panorama", "overhead"]),
    3289: ("Citi Field",                    ["aerial", "panorama", "overhead"]),
    2681: ("Citizens Bank Park",            ["aerial", "panorama", "overhead"]),
    2394: ("Comerica Park",                 ["aerial", "panorama", "overhead"]),
    19:   ("Coors Field",                   ["aerial", "panorama", "overhead"]),
    2392: ("Minute Maid Park",              ["aerial", "panorama", "overhead"]),
    3:    ("Fenway Park",                   ["aerial", "panorama", "overhead"]),
    5325: ("Globe Life Field",              ["aerial", "panorama", "overhead"]),
    2602: ("Great American Ball Park",      ["aerial", "panorama", "overhead"]),
    7:    ("Kauffman Stadium",              ["aerial", "panorama", "overhead"]),
    3309: ("Nationals Park",               ["aerial", "panorama", "overhead"]),
    2395: ("Oracle Park San Francisco",     ["aerial", "panorama", "overhead"]),
    2:    ("Oriole Park Camden Yards",      ["aerial", "panorama", "overhead"]),
    31:   ("PNC Park Pittsburgh",           ["aerial", "panorama", "overhead"]),
    2680: ("Petco Park",                    ["aerial", "panorama", "overhead"]),
    5:    ("Progressive Field Cleveland",   ["aerial", "panorama", "overhead"]),
    4:    ("Guaranteed Rate Field",         ["aerial", "panorama", "overhead"]),
    14:   ("Rogers Centre Toronto",         ["aerial", "panorama", "overhead"]),
    2529: ("Sutter Health Park Sacramento", ["aerial", "panorama", "overhead"]),
    680:  ("T-Mobile Park Seattle",         ["aerial", "panorama", "overhead"]),
    3312: ("Target Field Minneapolis",      ["aerial", "panorama", "overhead"]),
    12:   ("Tropicana Field",               ["aerial", "panorama", "overhead"]),
    4705: ("Truist Park Atlanta",           ["aerial", "panorama", "overhead"]),
    22:   ("Dodger Stadium",                ["aerial", "panorama", "overhead"]),
    17:   ("Wrigley Field",                 ["aerial", "panorama", "overhead"]),
    3313: ("Yankee Stadium",                ["aerial", "panorama", "overhead"]),
    4169: ("loanDepot park Miami",          ["aerial", "panorama", "overhead"]),
}


def search_commons(query: str, limit: int = 8) -> list[dict]:
    """Search Wikimedia Commons File namespace, return list of {url, width, height}."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size",
        "iiurlwidth": 800,
        "format": "json",
        "formatversion": "2",
    }
    try:
        r = SESSION.get("https://commons.wikimedia.org/w/api.php",
                        params=params, timeout=12)
        data = r.json()
        pages = data.get("query", {}).get("pages", [])
        results = []
        for p in pages:
            info = (p.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url", "")
            # Skip SVGs and non-photo formats
            if not url or url.lower().endswith(".svg") or url.lower().endswith(".png"):
                continue
            results.append({
                "title": p.get("title", ""),
                "url": url,
                "w": info.get("thumbwidth") or info.get("width", 0),
                "h": info.get("thumbheight") or info.get("height", 1),
            })
        return results
    except Exception as e:
        print(f"  Commons search error: {e}")
        return []


def best_aerial(results: list[dict]) -> dict | None:
    """Pick the most landscape-dominant (widest aspect ratio) result.
    High-home aerial shots are almost always wider than tall."""
    if not results:
        return None
    # Score = aspect ratio (w/h); wide panoramas score highest
    scored = sorted(results, key=lambda r: r["w"] / max(r["h"], 1), reverse=True)
    return scored[0]


def download(url: str, dest: Path) -> bool:
    try:
        r = SESSION.get(url, timeout=25, stream=True)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"  Saved {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


ok = skipped = failed = 0

for venue_id, (name, suffixes) in sorted(VENUES.items(), key=lambda x: x[1][0]):
    dest = OUT_DIR / f"{venue_id}.jpg"
    if dest.exists() and not FORCE:
        print(f"[skip] {name} ({venue_id})")
        skipped += 1
        continue

    print(f"[fetch] {name} (venue_id={venue_id})")
    chosen = None
    for suffix in suffixes:
        query = f"{name} baseball {suffix}"
        results = search_commons(query, limit=10)
        candidate = best_aerial(results)
        if candidate:
            ratio = candidate["w"] / max(candidate["h"], 1)
            print(f"  [{suffix}] best ratio={ratio:.2f}  {candidate['title'][:60]}")
            # Accept if reasonably wide (>= 1.2:1)
            if ratio >= 1.2:
                chosen = candidate
                break
        time.sleep(0.2)

    if not chosen:
        print(f"  No suitable aerial found — falling back to best available")
        # Fall back to any landscape image from first search
        for suffix in suffixes:
            results = search_commons(f"{name} baseball stadium", limit=10)
            candidate = best_aerial(results)
            if candidate:
                chosen = candidate
                break

    if chosen:
        print(f"  → {chosen['url'][:90]}...")
        if download(chosen["url"], dest):
            ok += 1
        else:
            failed += 1
    else:
        print(f"  FAILED — no image found")
        failed += 1

    time.sleep(0.3)

print(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed")
