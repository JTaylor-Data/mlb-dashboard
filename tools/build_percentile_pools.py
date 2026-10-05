"""
tools/build_percentile_pools.py — Build and persist percentile pools for all complete seasons.

Usage:
    python tools/build_percentile_pools.py [--seasons 2015-2025] [--splits all] [--force]

Idempotent: skips seasons/splits whose pool files are already current (same pool_version).
Resumable: on interrupt, already-written files are intact and the next run skips them.

pool_version is a hash of:
  - analytics/percentiles.py content
  - qualification thresholds
  - season's row count in the DB (data fingerprint)

Options:
  --seasons <start>-<end>  Override default range (default: EARLIEST_SEASON to CURRENT_SEASON-1)
  --season <year>          Build only this one season (used by freeze_season.py)
  --force                  Rebuild even if pool file already exists
  --splits all|pitcher|batter  Default: all
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.model_policy import CURRENT_SEASON, EARLIEST_SEASON
from api.db import DB_PATH
from analytics.percentile_cache import (
    _build_pool, _pool_version, _pool_path, load_pool, save_pool,
)

SPLITS = [None, "L", "R"]


def build_season(conn, season: int, player_types: list, splits: list, force: bool):
    version = _pool_version(conn, season)
    built = 0
    skipped = 0
    failed = 0

    for pt in player_types:
        for split in splits:
            # Check if already current
            if not force:
                existing = load_pool(conn, season, pt, split)
                if existing is not None:
                    print(f"  SKIP {pt:7s} split={split or 'ALL':3s} — already current")
                    skipped += 1
                    continue

            print(f"  BUILD {pt:7s} split={split or 'ALL':3s} ...", end=" ", flush=True)
            t0 = time.monotonic()
            df = _build_pool(conn, season, pt, split, None)
            elapsed = time.monotonic() - t0

            if df is None or df.empty:
                print(f"FAILED (no data) in {elapsed:.1f}s")
                failed += 1
                continue

            save_pool(conn, season, pt, split, df)
            print(f"OK {len(df):,} rows in {elapsed:.1f}s")
            built += 1

    return built, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Build percentile pool parquets for complete seasons")
    parser.add_argument("--seasons", help="Range like 2015-2025")
    parser.add_argument("--season", type=int, help="Build only this season")
    parser.add_argument("--force", action="store_true", help="Rebuild even if current")
    parser.add_argument("--splits", choices=["all", "pitcher", "batter"], default="all")
    args = parser.parse_args()

    # Determine season range
    if args.season:
        seasons = [args.season]
    elif args.seasons:
        start, end = map(int, args.seasons.split("-"))
        seasons = list(range(start, end + 1))
    else:
        seasons = list(range(EARLIEST_SEASON, CURRENT_SEASON))

    # Validate
    invalid = [s for s in seasons if s >= CURRENT_SEASON]
    if invalid:
        print(f"ERROR: Cannot persist pools for current/future season(s): {invalid}")
        print(f"  CURRENT_SEASON = {CURRENT_SEASON}. Only seasons < {CURRENT_SEASON} are persisted.")
        sys.exit(1)

    player_types = {
        "all": ["pitcher", "batter"],
        "pitcher": ["pitcher"],
        "batter": ["batter"],
    }[args.splits]

    print(f"Building percentile pools for seasons {min(seasons)}-{max(seasons)}")
    print(f"  Player types: {player_types}")
    print(f"  Splits: {SPLITS}")
    print(f"  Force rebuild: {args.force}")
    print()

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    total_built = total_skipped = total_failed = 0
    t_total = time.monotonic()

    try:
        for season in seasons:
            row_count = conn.execute(
                "SELECT COUNT(*) FROM pitches WHERE season = ?", (season,)
            ).fetchone()[0]
            if row_count == 0:
                print(f"Season {season}: no pitch data — skipping")
                continue

            print(f"Season {season} ({row_count:,} pitches):")
            built, skipped, failed = build_season(conn, season, player_types, SPLITS, args.force)
            total_built += built
            total_skipped += skipped
            total_failed += failed
            print()
    finally:
        conn.close()

    elapsed = time.monotonic() - t_total
    print(f"Done in {elapsed:.1f}s: {total_built} built, {total_skipped} skipped, {total_failed} failed")
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
