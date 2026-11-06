"""Write per-season league aggregates for COMPLETE seasons to parquets/.

The Career page needs, for every season a player played, the league HR/FB rate,
the sorted per-pitcher strikeout and per-batter home-run counts, and each
pitcher's Stuff+/Tunneling+/Location+. It also stores each pitcher's arsenal
(pitch types thrown at least ARSENAL_MIN_USAGE of the time) per season, which
the similar-players endpoint otherwise builds with a scan of every pitch. The first three are full-season scans of
the pitches table and the scores need full-season score parquets read and
normalized league-wide; this computes them once for every season before
CURRENT_SEASON. Rerun it if past-season pitch data is ever
re-pulled or corrected (the app falls back to live queries when an entry is
missing). The file records when it was built and each season's pitch-row count
at that time under "_meta", so a stale snapshot can be spotted by comparing
those counts with the database.

    python tools/precompute_career_aggregates.py
"""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analytics.player_page import league_hr_per_fb  # noqa: E402
from analytics.similarity import build_pitcher_arsenal_pool  # noqa: E402
from api.routers.career import (  # noqa: E402
    _compute_season_batter_hr_counts, _compute_season_pitcher_k_counts, _compute_season_pitcher_scores,
)
from config.model_policy import CURRENT_SEASON, EARLIEST_SEASON  # noqa: E402

OUT = REPO / "parquets" / "career_season_aggregates.json"


def main():
    db_path = os.environ.get("DB_PATH", str(REPO / "mlb_data.db"))
    conn = sqlite3.connect(db_path)
    snap = {"hr_fb": {}, "pitcher_k": {}, "batter_hr": {}, "pitcher_scores": {}, "pitcher_arsenal": {}}
    for season in range(EARLIEST_SEASON, CURRENT_SEASON):
        t0 = time.time()
        snap["hr_fb"][str(season)] = float(league_hr_per_fb(conn, season))
        snap["pitcher_k"][str(season)] = [int(c) for c in _compute_season_pitcher_k_counts(conn, season)]
        snap["batter_hr"][str(season)] = [int(c) for c in _compute_season_batter_hr_counts(conn, season)]
        snap["pitcher_scores"][str(season)] = {
            str(pid): scores for pid, scores in _compute_season_pitcher_scores(season).items()
        }
        arsenal = build_pitcher_arsenal_pool(conn, [season])
        snap["pitcher_arsenal"][str(season)] = {
            str(int(pid)): sorted(types) for (pid, _), types in arsenal.items()
        }
        print(f"{season}: {len(snap['pitcher_k'][str(season)])} pitchers, "
              f"{len(snap['batter_hr'][str(season)])} batters, "
              f"{len(snap['pitcher_scores'][str(season)])} scored pitchers, {time.time() - t0:.1f}s", flush=True)
    counts = dict(conn.execute("SELECT season, COUNT(*) FROM pitches WHERE season < ? GROUP BY season",
                               (CURRENT_SEASON,)).fetchall())
    snap["_meta"] = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pitch_rows_by_season": {str(k): int(v) for k, v in sorted(counts.items())},
    }
    conn.close()
    OUT.write_text(json.dumps(snap))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
