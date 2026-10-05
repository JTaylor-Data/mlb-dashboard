"""
tools/freeze_season.py <season> — Build and persist percentile pools when a season ends.

Run this once when a season becomes complete (i.e., after the World Series):
    python tools/freeze_season.py 2026

This builds all 6 pool files (pitcher/batter × ALL/L/R) for the given season
and saves them to scores/percentile_pools/<season>/.  After this, the server
will serve that season's percentile pools from disk without ever re-querying
the pitches table.

Safe to re-run: uses the same idempotency as build_percentile_pools.py.
Use --force to rebuild even if the pool file is already current.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.model_policy import CURRENT_SEASON


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Freeze a completed season's percentile pools")
    parser.add_argument("season", type=int, help="The season year to freeze (must be < CURRENT_SEASON after bump)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if pool file is current")
    args = parser.parse_args()

    if args.season >= CURRENT_SEASON:
        print(f"ERROR: Season {args.season} is not yet complete.")
        print(f"  CURRENT_SEASON = {CURRENT_SEASON}.")
        print(f"  To freeze season {args.season}, first bump CURRENT_SEASON in config/model_policy.py.")
        sys.exit(1)

    # Delegate to build_percentile_pools with --season flag
    import subprocess
    cmd = [sys.executable, str(Path(__file__).parent / "build_percentile_pools.py"),
           "--season", str(args.season)]
    if args.force:
        cmd.append("--force")

    print(f"Freezing season {args.season}...")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
