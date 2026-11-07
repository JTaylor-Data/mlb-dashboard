"""Write the season-wide tables the pitcher and Trends pages need, for every
COMPLETE season, to parquets/past_season_pitcher_page.json.

Per season: every pitcher's average arm angle, the per-(pitcher, pitch type)
movement sums behind the movement plot's peer circle, the same-hand
release-point pools behind the approach-angle percentile, and the Trends
league averages for both roles. Each is a full-season scan that a finished
season only ever needs once; without this the first pitcher or Trends view of
each past season pays it after every restart (minutes on the 1 GB Fly machine).
The app falls back to the live query when an entry is missing. Rerun it if
past-season pitch data is ever re-pulled or corrected; "_meta" records when it
was built and each season's pitch-row count at that time.

    python tools/precompute_past_season_pitcher_page.py
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
os.environ.setdefault("SKIP_WARMUP", "1")

from analytics.metrics import league_average_woba  # noqa: E402
from analytics.player_page import (  # noqa: E402
    all_pitcher_arm_angles, approach_angle_pool, league_avg_xwoba, league_average_batter_row,
    league_average_swing_profile, peer_movement_table, pitcher_league_avg_row,
)
from api.past_season_snapshot import PATH as OUT  # noqa: E402
from api.routers.trends import compute_league_averages  # noqa: E402
from config.model_policy import CURRENT_SEASON, EARLIEST_SEASON  # noqa: E402


def main():
    db_path = os.environ.get("DB_PATH", str(REPO / "mlb_data.db"))
    conn = sqlite3.connect(db_path)
    snap = {"arm_angles": {}, "peer_movement": {}, "approach_pool": {}, "league_avgs": {},
            "league_batter_row": {}, "league_pitcher_row": {}, "league_swing_profile": {},
            "league_woba": {}, "league_xwoba": {}}
    for season in range(EARLIEST_SEASON, CURRENT_SEASON):
        t0 = time.time()
        key = str(season)
        arm = all_pitcher_arm_angles(conn, season)
        snap["arm_angles"][key] = {str(int(p)): float(a) for p, a in zip(arm["pitcher_id"], arm["arm_angle"])}
        peer = peer_movement_table(conn, season)
        snap["peer_movement"][key] = [
            [int(p), pt, float(hb), float(ivb), int(n)]
            for p, pt, hb, ivb, n in zip(peer["pitcher_id"], peer["pt"], peer["hb_sum"], peer["ivb_sum"], peer["n"])
        ]
        snap["approach_pool"][key] = {
            hand: [float(v) for v in approach_angle_pool(conn, season, hand)] for hand in ("L", "R")
        }
        snap["league_avgs"][key] = {
            role: compute_league_averages(conn, role, season) for role in ("pitcher", "batter")
        }
        snap["league_batter_row"][key] = {
            name: league_average_batter_row(conn, season, split) for name, split in (("ALL", None), ("L", "L"), ("R", "R"))
        }
        snap["league_pitcher_row"][key] = {
            name: pitcher_league_avg_row(conn, season, split) for name, split in (("ALL", None), ("L", "L"), ("R", "R"))
        }
        snap["league_swing_profile"][key] = league_average_swing_profile(conn, season)
        snap["league_woba"][key] = float(league_average_woba(conn, season))
        snap["league_xwoba"][key] = float(league_avg_xwoba(conn, season))
        print(f"{season}: {len(arm)} arm angles, {len(peer)} peer rows, {time.time() - t0:.1f}s", flush=True)
    counts = dict(conn.execute("SELECT season, COUNT(*) FROM pitches WHERE season < ? GROUP BY season",
                               (CURRENT_SEASON,)).fetchall())
    snap["_meta"] = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pitch_rows_by_season": {str(k): int(v) for k, v in sorted(counts.items())},
    }
    conn.close()
    OUT.write_text(json.dumps(snap, default=lambda o: o.item()))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
