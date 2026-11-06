"""Write per-season league aggregates for COMPLETE seasons to parquets/.

The Career page needs, for every season a player played, the league HR/FB rate
and the sorted per-pitcher strikeout and per-batter home-run counts. Each is a
full-season scan of the pitches table; this computes them once for every
season before CURRENT_SEASON. Rerun it if past-season pitch data is ever
re-pulled or corrected (the app falls back to live queries when an entry is
missing).

    python tools/precompute_career_aggregates.py
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analytics.player_page import league_hr_per_fb  # noqa: E402
from api.routers.career import _compute_season_batter_hr_counts, _compute_season_pitcher_k_counts  # noqa: E402
from config.model_policy import CURRENT_SEASON, EARLIEST_SEASON  # noqa: E402

OUT = REPO / "parquets" / "career_season_aggregates.json"


def main():
    db_path = os.environ.get("DB_PATH", str(REPO / "mlb_data.db"))
    conn = sqlite3.connect(db_path)
    snap = {"hr_fb": {}, "pitcher_k": {}, "batter_hr": {}}
    for season in range(EARLIEST_SEASON, CURRENT_SEASON):
        t0 = time.time()
        snap["hr_fb"][str(season)] = float(league_hr_per_fb(conn, season))
        snap["pitcher_k"][str(season)] = [int(c) for c in _compute_season_pitcher_k_counts(conn, season)]
        snap["batter_hr"][str(season)] = [int(c) for c in _compute_season_batter_hr_counts(conn, season)]
        print(f"{season}: {len(snap['pitcher_k'][str(season)])} pitchers, "
              f"{len(snap['batter_hr'][str(season)])} batters, {time.time() - t0:.1f}s", flush=True)
    conn.close()
    OUT.write_text(json.dumps(snap))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
