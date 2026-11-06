"""Build and persist the threshold-free (unqualified) percentile pools for every
complete season, both roles, all three splits.

The player page ranks an unqualified pitcher or batter against the qualified
pool but reads their own raw values from a threshold-free pool. Each of those
is a full-season scan, so without this the first unqualified player viewed per
season and split pays several seconds after every restart. Files go under
parquets/percentile_pools/<season>/ as pitcher_unq_* / batter_unq_*, next to the
qualified pools (build_percentile_pools.py); the pool version hash includes the
season's row count, so a changed season is rebuilt automatically.

    python tools/build_unqualified_pools.py
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("SKIP_WARMUP", "1")

from api.routers.players import _batter_pool_unqualified, _pitcher_pool_unqualified  # noqa: E402
from config.model_policy import CURRENT_SEASON, EARLIEST_SEASON  # noqa: E402


def main():
    db_path = os.environ.get("DB_PATH", str(REPO / "mlb_data.db"))
    conn = sqlite3.connect(db_path)
    for season in range(EARLIEST_SEASON, CURRENT_SEASON):
        t0 = time.time()
        for split in (None, "L", "R"):
            _pitcher_pool_unqualified(conn, season, split)
            _batter_pool_unqualified(conn, season, split)
        print(f"{season}: 6 pools in {time.time() - t0:.1f}s", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
