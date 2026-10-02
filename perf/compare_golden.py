"""
perf/compare_golden.py — Diff current API responses against golden snapshots.
Usage:
    python perf/compare_golden.py [--baseline baseline.json] [--tolerance 1e-9]

Exit code 0 = all identical (or only registered intentional changes).
Exit code 1 = unexpected differences found.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import requests
import yaml

PERF_DIR   = Path(__file__).parent
GOLDEN_DIR = PERF_DIR / "golden"
BASE       = "http://localhost:8010"


def load_intentional(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        data = yaml.safe_load(f) or []
    return data if isinstance(data, list) else []


def approx_equal(a, b, tol: float) -> bool:
    if type(a) != type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return a == b
        return abs(a - b) <= tol
    return a == b


def deep_diff(a, b, path: str, tol: float, diffs: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            key_path = f"{path}.{k}"
            if k not in a:
                diffs.append(("added", key_path, None, b[k]))
            elif k not in b:
                diffs.append(("removed", key_path, a[k], None))
            else:
                deep_diff(a[k], b[k], key_path, tol, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(("length", path, len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_diff(x, y, f"{path}[{i}]", tol, diffs)
    elif isinstance(a, float) or isinstance(b, float):
        if not approx_equal(a, b, tol):
            diffs.append(("changed", path, a, b))
    else:
        if a != b:
            diffs.append(("changed", path, a, b))


def is_registered(diff_path: str, intentional: list[dict]) -> str | None:
    """Return reason string if diff is covered by an intentional_changes entry."""
    for entry in intentional:
        field = entry.get("field", "")
        if field and (field in diff_path or diff_path.endswith(field)):
            return entry.get("reason", "(no reason)")
    return None


def check_endpoint(name: str, url: str, tol: float, intentional: list[dict]) -> dict:
    golden_path = GOLDEN_DIR / f"{name}.json"
    if not golden_path.exists():
        return {"name": name, "status": "NO_GOLDEN", "diffs": []}

    try:
        r = requests.get(url, timeout=60)
    except Exception as e:
        return {"name": name, "status": "FETCH_ERROR", "error": str(e), "diffs": []}

    if r.status_code != 200:
        return {"name": name, "status": f"HTTP_{r.status_code}", "diffs": []}

    try:
        current = r.json()
        golden  = json.loads(golden_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"name": name, "status": "PARSE_ERROR", "error": str(e), "diffs": []}

    raw_diffs: list = []
    deep_diff(golden, current, "root", tol, raw_diffs)

    if not raw_diffs:
        return {"name": name, "status": "IDENTICAL", "diffs": []}

    reported, skipped = [], []
    for d in raw_diffs:
        reason = is_registered(d[1], intentional)
        if reason:
            skipped.append({"path": d[1], "reason": reason})
        else:
            reported.append({"type": d[0], "path": d[1],
                             "before": str(d[2])[:120], "after": str(d[3])[:120]})

    if reported:
        return {"name": name, "status": "DIFF", "diffs": reported,
                "intentional_skipped": len(skipped)}
    return {"name": name, "status": "INTENTIONAL_ONLY", "diffs": [],
            "intentional_count": len(skipped)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",  default="baseline.json")
    ap.add_argument("--tolerance", type=float, default=1e-9)
    args = ap.parse_args()

    baseline_path = PERF_DIR / args.baseline
    if not baseline_path.exists():
        print(f"ERROR: {baseline_path} not found — run benchmark.py first.")
        sys.exit(1)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    intentional = load_intentional(PERF_DIR / "intentional_changes.yaml")

    results = baseline.get("results", [])
    print(f"Comparing {len(results)} endpoints against golden snapshots...\n")

    summary = {"IDENTICAL": 0, "INTENTIONAL_ONLY": 0, "DIFF": 0, "NO_GOLDEN": 0,
               "FETCH_ERROR": 0, "HTTP_*": 0, "PARSE_ERROR": 0}
    failures = []

    for ep in results:
        name = ep["name"]
        # Reconstruct URL from the stored url field
        url  = ep.get("url", BASE + "/api/meta")
        res  = check_endpoint(name, url, args.tolerance, intentional)
        s    = res["status"]
        summary[s if s in summary else "HTTP_*"] = summary.get(s if s in summary else "HTTP_*", 0) + 1
        if s == "DIFF":
            failures.append(res)
            print(f"  DIFF      {name}")
            for d in res["diffs"][:5]:
                print(f"            {d['type']} {d['path']}: {d['before']!r} -> {d['after']!r}")
        elif s == "IDENTICAL":
            print(f"  OK        {name}")
        else:
            print(f"  {s:12s} {name}")

    print("\n-- Summary -------------------------------------------------")
    for k, v in summary.items():
        if v:
            print(f"  {k}: {v}")
    print(f"\nFailures (unexpected diffs): {len(failures)}")

    if failures:
        report_path = PERF_DIR / "golden_diff_report.md"
        lines = ["# Golden Diff Report\n"]
        for f in failures:
            lines.append(f"## {f['name']}\n")
            for d in f["diffs"]:
                lines.append(f"- **{d['type']}** `{d['path']}`")
                lines.append(f"  - before: `{d['before']}`")
                lines.append(f"  - after:  `{d['after']}`\n")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report: {report_path}")
        sys.exit(1)
    else:
        print("All responses identical (within tolerance + registered intentional changes). OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
