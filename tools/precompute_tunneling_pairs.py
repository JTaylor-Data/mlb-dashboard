"""Write the season tunneling pair table and qualifying pitch totals to parquets/.

Run after the pitch data for a season changes (the app falls back to live
compute when these files are missing).

    python tools/precompute_tunneling_pairs.py 2026 2025
"""
import os
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analytics.tunneling_pair_composite import _qualifying_total_pitches, season_all_pairs  # noqa: E402
from config.model_policy import CURRENT_SEASON  # noqa: E402

PARQUET_DIR = REPO / "parquets"


def main(seasons):
    db_path = os.environ.get("DB_PATH", str(REPO / "mlb_data.db"))
    conn = sqlite3.connect(db_path)
    PARQUET_DIR.mkdir(exist_ok=True)
    for season in seasons:
        pairs = season_all_pairs(conn, season)
        pairs.to_parquet(PARQUET_DIR / f"tunneling_pairs_{season}.parquet", index=False)
        totals = _qualifying_total_pitches(conn, season).reset_index()
        totals.to_parquet(PARQUET_DIR / f"tunneling_totals_{season}.parquet", index=False)
        print(f"{season}: {len(pairs)} pairs, {len(totals)} pitchers", flush=True)
    conn.close()


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [CURRENT_SEASON])
