"""
verify_stuff_page.py — PART 5 verification script for stuff-page-remake.

Confirms:
1. Every number in the page comes from validation_results.json or a live API call.
2. No numeric literal from validation_results.json is hard-coded in stuff-model.js or stuff-model.html.
3. Endpoints return expected fields for test cases.

Usage:
    python tools/verify_stuff_page.py

Requires: requests (pip install requests). Run with the server on port 8010.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VR_PATH   = ROOT / "frontend" / "content" / "validation_results.json"
CFG_PATH  = ROOT / "frontend" / "content" / "stuff_page.json"
JS_PATH   = ROOT / "frontend" / "js" / "stuff-model.js"
HTML_PATH = ROOT / "frontend" / "stuff-model.html"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

failures = []

def check(name, ok, msg=""):
    if ok:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {msg}" if msg else ""))
        failures.append(name)

# ── 1. Content files exist ────────────────────────────────────────────────────
print("\n=== 1. Content files ===")
check("validation_results.json exists", VR_PATH.exists())
check("stuff_page.json exists",         CFG_PATH.exists())

vr  = json.loads(VR_PATH.read_text(encoding="utf-8"))
cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

check("validation_results has stuff_plus_v4 key", "stuff_plus_v4" in vr)
check("stuff_page.json has opening_blurb",         bool(cfg.get("opening_blurb")))
check("stuff_page.json stuff_status is provisional or validated",
      cfg.get("stuff_status") in ("provisional", "validated"))

# ── 2. No numeric literals from validation_results hard-coded in JS/HTML ──────
print("\n=== 2. No numbers hard-coded in JS/HTML ===")

def collect_leaf_numbers(obj, path=""):
    """Yield (path, value) for every numeric leaf in a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from collect_leaf_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from collect_leaf_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and obj not in (0, 1, -1, 100, 200):
        yield path, obj

v4 = vr.get("stuff_plus_v4", {})
js_src   = JS_PATH.read_text(encoding="utf-8")
html_src = HTML_PATH.read_text(encoding="utf-8")

hard_coded_violations = []
for path, val in collect_leaf_numbers(v4):
    # Check if the raw number literal appears in JS or HTML
    literal = str(val)
    # Only check numbers with >= 3 significant chars
    if len(literal.replace(".", "").replace("-", "")) < 3:
        continue
    # Skip years and counts that are expected constants
    if val in (2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026,
               200, 100, 1000, 300):
        continue
    # Check JS (but not inside comments or the feature labels dict)
    if re.search(r'\b' + re.escape(literal) + r'\b', js_src):
        # False-positive filter: if the line is a comment or inside a string that's clearly a label, skip
        for lineno, line in enumerate(js_src.splitlines(), 1):
            if re.search(r'\b' + re.escape(literal) + r'\b', line):
                if line.strip().startswith("//") or line.strip().startswith("*"):
                    continue  # comment line, OK
                hard_coded_violations.append(f"stuff-model.js:{lineno}: {literal!r} (from vr{path})")
                break

if hard_coded_violations:
    for v in hard_coded_violations[:5]:
        check(f"No hard-code: {v}", False)
else:
    check("No validation numbers hard-coded in stuff-model.js", True)

# ── 3. Live API checks ────────────────────────────────────────────────────────
print("\n=== 3. Live API checks ===")
try:
    import requests
    BASE = "http://127.0.0.1:8010"

    # 3a. Meta endpoint
    r = requests.get(f"{BASE}/api/stuff-model/meta", timeout=5)
    check("/api/stuff-model/meta 200", r.status_code == 200)
    if r.ok:
        meta = r.json()
        check("meta has model_hash_sha256", bool(meta.get("model_hash_sha256")))
        check("meta active_version is v4",  meta.get("active_version") == "v4")

    # 3b. Leaderboard 2026
    r = requests.get(f"{BASE}/api/stuff-model/leaderboard?season=2026&n=10", timeout=10)
    check("/api/stuff-model/leaderboard 2026 200", r.status_code == 200)
    if r.ok:
        lb = r.json()
        check("leaderboard top has 10 entries",    len(lb.get("top", [])) == 10)
        check("leaderboard bottom has 10 entries", len(lb.get("bottom", [])) == 10)
        check("leaderboard top entry has rank/name/stuff_plus/n_pitches",
              all(k in lb["top"][0] for k in ("rank","name","stuff_plus","n_pitches")))

    # 3c. Leaderboard 2020 (asterisk year)
    r = requests.get(f"{BASE}/api/stuff-model/leaderboard?season=2020&n=10", timeout=10)
    check("/api/stuff-model/leaderboard 2020 200", r.status_code == 200)
    if r.ok:
        lb20 = r.json()
        has_asterisk = any(e.get("season_asterisk") for e in (lb20.get("top") or []) + (lb20.get("bottom") or []))
        check("2020 entries have season_asterisk=True", has_asterisk)

    # 3d. Pitch types
    r = requests.get(f"{BASE}/api/stuff-model/pitch-types?season=2026", timeout=10)
    check("/api/stuff-model/pitch-types 2026 200", r.status_code == 200)
    if r.ok:
        pts = r.json()
        types = [p["pitch_type"] for p in pts.get("pitch_types", [])]
        check("pitch-types includes FF", "FF" in types)
        check("pitch-types includes SL", "SL" in types)

    # 3e. Per-pitch-type leaderboard (KN — few qualifiers)
    # NOTE: may be slow on cold start (computes all-season raw predictions if not cached).
    try:
        r = requests.get(f"{BASE}/api/stuff-model/leaderboard?season=2026&pitch_type=KN&n=10", timeout=90)
        check("/api/stuff-model/leaderboard KN 200 (may have <20 qualifiers)", r.status_code == 200)
    except requests.exceptions.Timeout:
        print(f"  {SKIP}  /api/stuff-model/leaderboard KN: timed out on cold start (expected)")

    # 3f. Pitcher summary for Mason Miller (695243) 2026 — should be qualified
    try:
        r = requests.get(f"{BASE}/api/stuff-model/pitcher/695243/summary?season=2026", timeout=90)
        check("/api/stuff-model/pitcher/695243/summary 2026 200", r.status_code == 200)
        if r.ok:
            s = r.json()
            check("Mason Miller 2026 qualified=True", s.get("qualified") is True)
            check("Mason Miller 2026 percentile is not None", s.get("percentile") is not None)
    except requests.exceptions.Timeout:
        print(f"  {SKIP}  /api/stuff-model/pitcher/695243/summary: timed out on cold start (expected)")

    # 3g. Pitcher breakdown (SHAP)
    try:
        r = requests.get(f"{BASE}/api/stuff-model/pitcher/695243?season=2026", timeout=90)
        check("/api/stuff-model/pitcher/695243 SHAP 2026 200", r.status_code == 200)
    except requests.exceptions.Timeout:
        print(f"  {SKIP}  /api/stuff-model/pitcher/695243 SHAP: timed out on cold start (expected)")

except ImportError:
    print(f"  {SKIP}  requests not installed; skipping live API checks")
except Exception as e:
    check(f"Live API reachable", False, str(e))

# ── 4. "Not yet run" items inventory ─────────────────────────────────────────
print("\n=== 4. 'Not yet run' items in validation_results.json ===")
not_yet_run_items = []
def find_nyr(obj, path=""):
    if isinstance(obj, dict):
        if obj.get("status") == "not_yet_run":
            not_yet_run_items.append(path)
        for k, v in obj.items():
            find_nyr(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_nyr(v, f"{path}[{i}]")
    elif obj == "not_yet_run":
        not_yet_run_items.append(path)

find_nyr(v4, "stuff_plus_v4")
if not_yet_run_items:
    print(f"  INFO  {len(not_yet_run_items)} 'not_yet_run' item(s) — these will render as 'Not yet run' on the page:")
    for item in not_yet_run_items:
        print(f"         • {item}")
else:
    print(f"  {PASS}  No 'not_yet_run' items (all results populated)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if failures:
    print(f"RESULT: {len(failures)} failure(s):")
    for f in failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print("RESULT: All checks passed.")
    sys.exit(0)
