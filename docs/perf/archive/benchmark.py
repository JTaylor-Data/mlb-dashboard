"""
perf/benchmark.py — MLB Dashboard API benchmark harness.
Usage:
    python perf/benchmark.py [--output baseline.json] [--warm-reps 15] [--runs 3]

Writes: perf/<output> and perf/baseline.md (or after.md when --output after.json)
Golden snapshots: perf/golden/<hash>.json  (committed as manifest only)
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import psutil  # may not be installed; graceful fallback

BASE = "http://localhost:8010"
PERF_DIR = Path(__file__).parent
GOLDEN_DIR = PERF_DIR / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)

# ── Representative parameters ──────────────────────────────────────────────
STARTER_1  = 645261   # Sandy Alcantara (high volume 2026)
STARTER_2  = 650911   # Cristopher Sánchez
RELIEVER   = 689254   # Mason Fluharty
BATTER_1   = 656941   # Kyle Schwarber
BATTER_2   = 670623   # Isaac Paredes
KERSHAW    = 477132   # inactive 2026 (last: 2025)
ROOKIE     = 664059   # low-sample 2026
GAME_ID    = 822690   # Alcantara game
TEAM_1     = 111      # Boston Red Sox
TEAM_2     = 147      # NY Yankees

ENDPOINTS = [
    # Search
    ("search_kersh",       "GET", "/api/players/all?q=kersh"),
    ("search_judg",        "GET", "/api/players/all?q=judg"),
    # Player overviews
    ("player_starter1",    "GET", f"/api/player/{STARTER_1}"),
    ("player_starter2",    "GET", f"/api/player/{STARTER_2}"),
    ("player_reliever",    "GET", f"/api/player/{RELIEVER}"),
    ("player_batter1",     "GET", f"/api/player/{BATTER_1}"),
    ("player_batter2",     "GET", f"/api/player/{BATTER_2}"),
    ("player_kershaw",     "GET", f"/api/player/{KERSHAW}"),
    ("player_rookie",      "GET", f"/api/player/{ROOKIE}"),
    # Similar players
    ("similar_starter1",   "GET", f"/api/player/{STARTER_1}/similar?season=2026&role=pitcher"),
    ("similar_batter1",    "GET", f"/api/player/{BATTER_1}/similar?season=2026&role=batter"),
    # Career
    ("career_starter1",    "GET", f"/api/players/{STARTER_1}/career"),
    ("career_batter1",     "GET", f"/api/players/{BATTER_1}/career"),
    # Outings
    ("outing_game",        "GET", f"/api/players/{STARTER_1}/outing?game_id={GAME_ID}"),
    ("outings_list",       "GET", f"/api/players/{STARTER_1}/outings"),
    # Rolling xwOBA (batter)
    ("rolling_xwoba_2026", "GET", f"/api/players/{BATTER_1}/rolling-xwoba?season=2026"),
    ("rolling_xwoba_2025", "GET", f"/api/players/{BATTER_1}/rolling-xwoba?season=2025"),
    # Trends
    ("trends_starter_2026","GET", f"/api/players/{STARTER_1}/trends?season=2026"),
    ("trends_batter_2026", "GET", f"/api/players/{BATTER_1}/trends?season=2026"),
    ("trends_starter_2024","GET", f"/api/players/{STARTER_1}/trends?season=2024"),
    # Heatmaps / pitch locations (pitchers)
    ("heatmaps_s1_2026",   "GET", f"/api/heatmaps/{STARTER_1}?season=2026&stand=ALL"),
    ("heatmaps_s1_L",      "GET", f"/api/heatmaps/{STARTER_1}?season=2026&stand=L"),
    ("heatmaps_s1_R",      "GET", f"/api/heatmaps/{STARTER_1}?season=2026&stand=R"),
    ("heatmaps_s1_2024",   "GET", f"/api/heatmaps/{STARTER_1}?season=2024&stand=ALL"),
    ("location_s1_2026",   "GET", f"/api/location/pitcher/{STARTER_1}?season=2026"),
    ("location_s1_2025",   "GET", f"/api/location/pitcher/{STARTER_1}?season=2025"),
    # Hit zones (batters)
    ("hitzones_b1_2026",   "GET", f"/api/hitzones/{BATTER_1}?season=2026&stand=ALL"),
    ("hitzones_b1_L",      "GET", f"/api/hitzones/{BATTER_1}?season=2026&stand=L"),
    ("hitzones_b1_R",      "GET", f"/api/hitzones/{BATTER_1}?season=2026&stand=R"),
    # Radar / compare
    ("radar_starter1",     "GET", f"/api/radar/{STARTER_1}?season=2026"),
    # Model endpoints
    ("stuff_s1_2026",      "GET", f"/api/stuff-model/pitcher/{STARTER_1}?season=2026"),
    ("stuff_s1_2025",      "GET", f"/api/stuff-model/pitcher/{STARTER_1}?season=2025"),
    ("stuff_leader",       "GET", "/api/stuff-model/leaderboard?season=2026"),
    ("tunnel_s1",          "GET", f"/api/tunneling/pitcher/{STARTER_1}?season=2026"),
    ("tunnel_v3_s1",       "GET", f"/api/tunneling/v3/pitcher/{STARTER_1}?season=2026"),
    ("deception_s1",       "GET", f"/api/deception/pitcher/{STARTER_1}?season=2026"),
    # League / standings / schedule
    ("standings",          "GET", "/api/standings"),
    ("top_performers",     "GET", "/api/top-performers?date=2026-09-28"),
    ("probable_pitchers",  "GET", "/api/probable-pitchers"),
    ("league_avgs",        "GET", "/api/trends/league-averages"),
    # Playoffs
    ("bracket",            "GET", "/api/playoffs/bracket"),
    ("wildcard",           "GET", "/api/playoffs/wildcard"),
    ("playoff_teams",      "GET", "/api/playoffs/teams"),
    ("div_leaders",        "GET", "/api/playoffs/division-leaders"),
    # Teams
    ("teams",              "GET", "/api/teams"),
    ("team_card_1",        "GET", f"/api/team/{TEAM_1}/card"),
    ("team_card_2",        "GET", f"/api/team/{TEAM_2}/card"),
    ("team_stats_1",       "GET", f"/api/teams/{TEAM_1}/stats"),
    ("team_roster_1",      "GET", f"/api/teams/{TEAM_1}/roster"),
    # Meta
    ("meta",               "GET", "/api/meta"),
]

# Endpoints to SKIP (call external APIs, not safe to benchmark in loop)
SKIP_EXTERNAL = {"/api/admin/pull-data", "/api/admin/refresh-cache", "/api/news/refresh"}


def env_info() -> dict:
    try:
        cpu_pct = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        mem_mb = ram.available // (1024 * 1024)
    except Exception:
        cpu_pct = -1
        mem_mb = -1
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    try:
        db_mb = os.path.getsize("mlb_data.db") / (1024 * 1024)
    except Exception:
        db_mb = -1
    return {
        "timestamp": datetime.now().isoformat(),
        "git_hash": git_hash,
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_percent_at_start": cpu_pct,
        "available_ram_mb": mem_mb,
        "db_size_mb": round(db_mb, 1),
    }


def canonicalize(obj: Any) -> str:
    """JSON with sorted keys, no float rounding — for golden diffing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=True)


def save_golden(name: str, body: bytes) -> str:
    try:
        obj = json.loads(body)
        canonical = canonicalize(obj)
    except Exception:
        canonical = body.decode("utf-8", errors="replace")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    path = GOLDEN_DIR / f"{name}.json"
    path.write_text(canonical, encoding="utf-8")
    return digest


def measure_one(name: str, method: str, path: str, warm_reps: int) -> dict:
    url = BASE + path
    result: dict = {"name": name, "url": url, "method": method}

    # Cold request
    t0 = time.perf_counter()
    try:
        r = requests.request(method, url, timeout=60)
        cold_ms = (time.perf_counter() - t0) * 1000
        result["cold_status"] = r.status_code
        result["cold_ms"] = round(cold_ms, 1)
        body = r.content
        result["response_bytes"] = len(body)
        try:
            result["response_gz_bytes"] = len(gzip.compress(body))
        except Exception:
            result["response_gz_bytes"] = -1
        # Save golden on cold request
        if r.status_code == 200:
            result["golden_hash"] = save_golden(name, body)
        else:
            result["golden_hash"] = None
            result["error_body"] = body.decode("utf-8", errors="replace")[:200]
    except Exception as e:
        result["cold_status"] = -1
        result["cold_ms"] = -1
        result["error"] = str(e)
        return result

    # Warm requests
    warm_times = []
    for _ in range(warm_reps):
        t0 = time.perf_counter()
        try:
            requests.request(method, url, timeout=60)
            warm_times.append((time.perf_counter() - t0) * 1000)
        except Exception:
            pass

    if warm_times:
        result["warm_min_ms"]    = round(min(warm_times), 1)
        result["warm_median_ms"] = round(statistics.median(warm_times), 1)
        result["warm_p95_ms"]    = round(sorted(warm_times)[int(len(warm_times) * 0.95)], 1)
        result["warm_max_ms"]    = round(max(warm_times), 1)
        result["warm_reps"]      = len(warm_times)
        if len(warm_times) > 1:
            result["warm_stdev_ms"] = round(statistics.stdev(warm_times), 1)
    return result


def run_suite(warm_reps: int, runs: int) -> list[dict]:
    """Run the full endpoint suite `runs` times, return all results."""
    all_runs: list[list[dict]] = []
    for run_i in range(runs):
        print(f"\n-- Run {run_i + 1}/{runs} ---------------------------------")
        run_results = []
        for name, method, path in ENDPOINTS:
            if any(path.startswith(s) for s in SKIP_EXTERNAL):
                continue
            print(f"  {name:40s}", end=" ", flush=True)
            r = measure_one(name, method, path, warm_reps if run_i == 0 else 5)
            med = r.get("warm_median_ms", r.get("cold_ms", -1))
            status = r.get("cold_status", -1)
            print(f"{status:3d}  cold={r.get('cold_ms',-1):7.1f}ms  warm_med={med:7.1f}ms")
            run_results.append(r)
        all_runs.append(run_results)

    # Aggregate across runs (use warm median from each run)
    aggregated = []
    for i, ep in enumerate(all_runs[0]):
        agg = dict(ep)
        medians = [
            all_runs[r][i].get("warm_median_ms", -1)
            for r in range(runs)
            if all_runs[r][i].get("warm_median_ms", -1) >= 0
        ]
        if medians:
            agg["runs_warm_medians_ms"] = [round(m, 1) for m in medians]
            agg["aggregate_median_ms"]  = round(statistics.median(medians), 1)
            if len(medians) > 1:
                cv = statistics.stdev(medians) / (statistics.mean(medians) or 1)
                agg["run_to_run_cv"]    = round(cv, 4)
                # noise band: larger of 10% or 2× CV
                noise = max(0.10, 2 * cv)
                agg["noise_band_pct"]   = round(noise * 100, 1)
        aggregated.append(agg)
    return aggregated


def build_manifest(results: list[dict]) -> dict:
    return {ep["name"]: ep.get("golden_hash") for ep in results}


def save_markdown(results: list[dict], env: dict, path: Path) -> None:
    lines = [
        "# Benchmark Results",
        f"\nGenerated: {env['timestamp']}  Git: `{env['git_hash'][:8]}`",
        f"Platform: {env['platform']}  Python: {env['python'].split()[0]}",
        f"DB: {env['db_size_mb']} MB  CPU at start: {env['cpu_percent_at_start']}%",
        "\n## Per-Endpoint Results\n",
        "| Endpoint | Status | Cold ms | Warm med ms | Warm p95 ms | Bytes | GZ Bytes | Noise band |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ep in results:
        name   = ep["name"]
        status = ep.get("cold_status", -1)
        cold   = ep.get("cold_ms", -1)
        med    = ep.get("aggregate_median_ms", ep.get("warm_median_ms", -1))
        p95    = ep.get("warm_p95_ms", -1)
        byt    = ep.get("response_bytes", -1)
        gz     = ep.get("response_gz_bytes", -1)
        noise  = ep.get("noise_band_pct", "—")
        lines.append(f"| {name} | {status} | {cold} | {med} | {p95} | {byt} | {gz} | {noise}% |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",    default="baseline.json")
    ap.add_argument("--warm-reps", type=int, default=15)
    ap.add_argument("--runs",      type=int, default=3)
    args = ap.parse_args()

    print("Collecting environment info...")
    env = env_info()
    for k, v in env.items():
        print(f"  {k}: {v}")

    # Verify server is up
    try:
        requests.get(BASE + "/api/meta", timeout=10)
    except Exception as e:
        print(f"\nERROR: server not reachable at {BASE}: {e}")
        sys.exit(1)

    print(f"\nRunning {len(ENDPOINTS)} endpoints × {args.runs} runs × {args.warm_reps} warm reps")
    results = run_suite(warm_reps=args.warm_reps, runs=args.runs)

    out = {
        "env": env,
        "warm_reps": args.warm_reps,
        "runs": args.runs,
        "results": results,
        "golden_manifest": build_manifest(results),
    }

    out_path = PERF_DIR / args.output
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    md_stem = args.output.replace(".json", ".md")
    md_path = PERF_DIR / md_stem
    save_markdown(results, env, md_path)
    print(f"Saved: {md_path}")

    # Print summary
    print("\n-- Summary (aggregate warm median) ----------------------")
    slow = sorted(results, key=lambda x: x.get("aggregate_median_ms", 0), reverse=True)[:10]
    for ep in slow:
        print(f"  {ep['name']:40s} {ep.get('aggregate_median_ms',-1):8.1f} ms")


if __name__ == "__main__":
    main()
