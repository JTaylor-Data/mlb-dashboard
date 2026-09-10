"""
Targeted fix: downloads specific known-good Wikimedia Commons files
for venues where automated search picked wrong images.
Each file was verified to be a high-angle stadium shot.
"""
import requests, time
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "img" / "stadiums"
HEADERS = {"User-Agent": "mlb-dashboard-local/1.0 (contact: josh.scott.tay@gmail.com)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# Direct Wikimedia Commons thumb URLs for known-good aerial photos.
# Format: venue_id -> (description, commons_file_title)
EXACT_FILES = {
    # Globe Life Field — interior aerial from upper concourse, full diamond visible
    5325: ("Globe Life Field",        "GlobeLifeField2021.jpg"),
    # American Family Field — aerial from above showing full roof and field
    32:   ("American Family Field",   "American Family Field aerial.jpg"),
    # Great American Ball Park — aerial from press box level
    2602: ("Great American",          "Great American Ball Park - opening day 2003.jpg"),
    # Rogers Centre — interior view from upper deck (retractable roof open)
    14:   ("Rogers Centre",           "June 2012 Rogers Centre (aka Skydome).jpg"),
    # Guaranteed Rate Field — aerial of full stadium complex
    4:    ("Guaranteed Rate Field",   "Guaranteed Rate Field 2019.jpg"),
    # Minute Maid Park — panoramic aerial looking from behind home plate
    2392: ("Minute Maid Park",        "PanoramicMinuteMaidParkHouston.jpg"),
    # Oracle Park — iconic aerial showing bay behind the outfield
    2395: ("Oracle Park",             "Oracle Park aerial photo.jpg"),
    # PNC Park — aerial from above home plate with downtown Pittsburgh
    31:   ("PNC Park",                "PNC Park aerial (cropped).jpg"),
    # Tropicana Field — aerial exterior dome shot
    12:   ("Tropicana Field",         "Tropicana Field in St. Petersburg, Florida.jpg"),
    # Truist Park — aerial of Cumberland, GA stadium
    4705: ("Truist Park",             "Truist Park 2025.jpg"),
}


def get_url(filename: str) -> str | None:
    params = {
        "action": "query",
        "titles": f"File:{filename}",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 960,
        "format": "json",
        "formatversion": "2",
    }
    try:
        r = SESSION.get("https://commons.wikimedia.org/w/api.php",
                        params=params, timeout=12)
        pages = r.json().get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        info = (pages[0].get("imageinfo") or [{}])[0]
        return info.get("thumburl") or info.get("url")
    except Exception as e:
        print(f"  API error: {e}")
        return None


def search_fallback(venue_name: str) -> str | None:
    """Search Commons for the venue, pick the widest landscape image."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": f'"{venue_name}" baseball filetype:bitmap',
        "gsrlimit": 20,
        "prop": "imageinfo",
        "iiprop": "url|size",
        "iiurlwidth": 960,
        "format": "json",
        "formatversion": "2",
    }
    try:
        r = SESSION.get("https://commons.wikimedia.org/w/api.php",
                        params=params, timeout=12)
        pages = r.json().get("query", {}).get("pages", [])
        best_url, best_ratio = None, 0
        for p in pages:
            title = p.get("title", "").lower()
            # Skip clearly wrong files
            if any(k in title for k in ["player", "mascot", "logo", ".svg", "portrait"]):
                continue
            info = (p.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url", "")
            if not url or url.lower().endswith(".svg"):
                continue
            w = info.get("thumbwidth") or info.get("width", 1)
            h = info.get("thumbheight") or info.get("height", 1)
            ratio = w / max(h, 1)
            if ratio > best_ratio:
                best_ratio, best_url = ratio, url
        return best_url
    except Exception:
        return None


def download(url: str, dest: Path) -> bool:
    try:
        r = SESSION.get(url, timeout=25, stream=True)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} from {url[:60]}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        kb = dest.stat().st_size // 1024
        print(f"  Saved {dest.name} ({kb} KB)")
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


ok = failed = 0
for venue_id, (name, filename) in EXACT_FILES.items():
    dest = OUT_DIR / f"{venue_id}.jpg"
    print(f"\n[fix] {name} (venue_id={venue_id})  file='{filename}'")
    url = get_url(filename)
    if url:
        print(f"  Found: {url[:90]}...")
        if download(url, dest):
            ok += 1
            time.sleep(0.3)
            continue
    # Fall back to search
    print(f"  File not found on Commons — searching...")
    url = search_fallback(name)
    if url:
        print(f"  Search fallback: {url[:90]}...")
        if download(url, dest):
            ok += 1
            time.sleep(0.3)
            continue
    print(f"  FAILED")
    failed += 1
    time.sleep(0.3)

print(f"\nDone: {ok} fixed, {failed} failed")
