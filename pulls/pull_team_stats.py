"""
Pulls the team-card snapshot (season + trailing-30-day OPS/ERA/bullpen ERA/
FIP for all 30 teams) into team_stats_snapshot, for the "click any team
logo" popup card.

FanGraphs was the original plan (per direct request) but is unreachable -
confirmed directly, both a plain `requests` call (even with a real browser
User-Agent) and a real headless-Playwright page load both get stuck on a
Cloudflare "Just a moment..." bot challenge page, not real data. Sourced
from the MLB Stats API instead: OPS and ERA are official numbers straight
from the team stats endpoint; FIP is self-computed with the standard
formula from official HR/BB/HBP/K/IP (same FIP_CONSTANT_APPROX=3.10 this
project already uses for xFIP - a fixed historical approximation, not
fit to this data, same honest caveat as everywhere else it's used).

Bullpen ERA (sitCodes=rp) is season-only. Confirmed directly: the Stats
API's byDateRange stat type silently ignores sitCodes (returns identical
numbers with and without it - checked side by side, not assumed), and our
own pitch data has no earned/unearned distinction to self-compute ERA
from at all (documented dead end elsewhere in this project - see wpa.py).
No fabricated 30-day bullpen number; the card should show "season only"
for that one metric rather than guess.

Usage:
    python pull_team_stats.py [SEASON]   # defaults to 2026
"""
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

DB_PATH = Path(__file__).resolve().parent.parent / "mlb_data.db"
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_SEASON = 2026
FIP_CONSTANT_APPROX = 3.10   # matches analytics/player_page.py's xFIP constant
ROLLING_WINDOW_DAYS = 30


def _stat_block(team_id, season, group, stats_type, **params):
    url = f"{STATSAPI_BASE}/teams/{team_id}/stats"
    q = {"stats": stats_type, "group": group, "season": season, "sportId": 1, **params}
    resp = requests.get(url, params=q, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    splits = data.get("stats", [{}])[0].get("splits") or []
    return splits[0]["stat"] if splits else {}


def _fip(stat):
    hr, bb, hbp, k = stat.get("homeRuns"), stat.get("baseOnBalls"), stat.get("hitByPitch"), stat.get("strikeOuts")
    ip = float(stat.get("inningsPitched") or 0)
    if not ip or None in (hr, bb, hbp, k):
        return None
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT_APPROX, 2)


def _baserunning_value(stat):
    # Rough linear-weight approximation of the SB/CS component of
    # baserunning value, in runs - see schema.sql's column comment for what
    # this does and doesn't capture.
    sb, cs = stat.get("stolenBases"), stat.get("caughtStealing")
    if sb is None or cs is None:
        return None
    return round(0.2 * sb - 0.4 * cs, 1)


def fetch_one_team(team_id, season, start_date, end_date):
    hitting = _stat_block(team_id, season, "hitting", "season")
    pitching = _stat_block(team_id, season, "pitching", "season")
    bullpen = _stat_block(team_id, season, "pitching", "statSplits", sitCodes="rp")
    hitting_30d = _stat_block(team_id, season, "hitting", "byDateRange", startDate=start_date, endDate=end_date)
    pitching_30d = _stat_block(team_id, season, "pitching", "byDateRange", startDate=start_date, endDate=end_date)

    def f(stat, key):
        v = stat.get(key)
        return float(v) if v not in (None, "-.--") else None

    return dict(
        team_id=team_id, season=season,
        ops=f(hitting, "ops"), era=f(pitching, "era"), bullpen_era=f(bullpen, "era"), fip=_fip(pitching),
        ops_30d=f(hitting_30d, "ops"), era_30d=f(pitching_30d, "era"), fip_30d=_fip(pitching_30d),
        baserunning_value=_baserunning_value(hitting),
    )


def run(season: int):
    conn = sqlite3.connect(DB_PATH)
    team_ids = [r[0] for r in conn.execute("SELECT team_id FROM teams").fetchall()]
    if not team_ids:
        print("No teams in the `teams` table - run pull_teams.py first.")
        conn.close()
        return

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=ROLLING_WINDOW_DAYS)).isoformat()
    print(f"Pulling team stats for {len(team_ids)} teams (season {season}, "
          f"trailing {ROLLING_WINDOW_DAYS}d {start_date} to {end_date})...", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_one_team, tid, season, start_date, end_date): tid for tid in team_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 10 == 0:
                print(f"  ...{i}/{len(team_ids)}")

    now = __import__("datetime").datetime.now().isoformat()
    conn.executemany(
        "INSERT INTO team_stats_snapshot (team_id, season, pulled_at, ops, era, bullpen_era, fip, "
        "ops_30d, era_30d, fip_30d, baserunning_value) VALUES (:team_id, :season, :pulled_at, :ops, :era, "
        ":bullpen_era, :fip, :ops_30d, :era_30d, :fip_30d, :baserunning_value) "
        "ON CONFLICT(team_id) DO UPDATE SET season=excluded.season, pulled_at=excluded.pulled_at, "
        "ops=excluded.ops, era=excluded.era, bullpen_era=excluded.bullpen_era, fip=excluded.fip, "
        "ops_30d=excluded.ops_30d, era_30d=excluded.era_30d, fip_30d=excluded.fip_30d, "
        "baserunning_value=excluded.baserunning_value",
        [dict(r, pulled_at=now) for r in rows],
    )
    conn.commit()
    conn.close()
    print(f"\nUpserted {len(rows)} teams into team_stats_snapshot.")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
