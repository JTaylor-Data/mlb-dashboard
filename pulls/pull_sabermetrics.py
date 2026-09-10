"""
Pulls real, officially-sourced season-total wRC+, xFIP, Barrel%, Arm
Strength, and WAR into player_advanced_stats. FanGraphs itself (the
traditional source for wRC+/xFIP/arm strength/WAR) is Cloudflare-blocked
(confirmed via direct pybaseball call - a 403 from fangraphs.com, not
something to work around). All of these turn out to be available from
unblocked sources instead:

  - wRC+ / xFIP / WAR: the MLB Stats API's own "sabermetrics" stat group
    (statsapi.mlb.com/api/v1/stats?stats=sabermetrics), the same trusted API
    already used elsewhere in this project (season ERA, box scores). Its
    default leaderboard only returns players who clear a batting/pitching
    title's qualification bar; playerPool=all lifts that, returning ~750-900
    players instead of ~50-150 - still season-total only, no L/R split. WAR
    is pulled separately for hitting and pitching (war_batting/war_pitching)
    since a two-way player can carry a real value in both.
  - Barrel%: Baseball Savant's exit-velocity/barrels leaderboard, via
    pybaseball's statcast_batter_exitvelo_barrels / _pitcher_exitvelo_barrels
    (a different Savant endpoint than the blocked FanGraphs pages - not
    Cloudflare-gated). brl_percent comes back as a 0-100 number; stored here
    divided by 100 to match this project's own 0-1 barrel_pct convention.
  - Arm Strength: Baseball Savant's own arm-strength leaderboard
    (baseballsavant.mlb.com/leaderboard/arm-strength?csv=true) - a direct
    CSV export, not Cloudflare-gated either, and not wrapped by pybaseball
    so fetched here with a plain request. `arm_overall` is each fielder's
    blended average throw velocity (mph) across whatever positions they
    played that season - the single "how strong is this arm" number, vs.
    the per-position arm_1b/arm_of/etc. columns also in the CSV.

Usage:
    python pull_sabermetrics.py [SEASON]   # defaults to 2026

Upserts, so re-running just refreshes the numbers. Players missing from
these leaderboards (too few PA/BF/throws) simply keep NULL here - the API
layer falls back to this project's own pitch-level approximation where one
exists (wRC+/xFIP/Barrel%), or just shows "-" (Arm Strength, no fallback).
"""
import csv
import io
import sqlite3
import sys

import requests
from pybaseball import statcast_batter_exitvelo_barrels, statcast_pitcher_exitvelo_barrels

DB_PATH = "mlb_data.db"
DEFAULT_SEASON = 2026
STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
ARM_STRENGTH_URL = "https://baseballsavant.mlb.com/leaderboard/arm-strength"


def _sabermetrics_map(season, group, stat_key):
    resp = requests.get(STATS_URL, params={
        "stats": "sabermetrics", "group": group, "season": season,
        "sportId": 1, "limit": 1000, "playerPool": "all",
    }, timeout=30)
    resp.raise_for_status()
    splits = resp.json().get("stats", [{}])[0].get("splits", [])
    return {s["player"]["id"]: s["stat"].get(stat_key) for s in splits if s["stat"].get(stat_key) is not None}


def _arm_strength_map(season):
    resp = requests.get(ARM_STRENGTH_URL, params={"year": season, "csv": "true"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    out = {}
    for row in reader:
        overall = row.get("arm_overall")
        if overall:
            out[int(row["player_id"])] = float(overall)
    return out


def run(season: int):
    conn = sqlite3.connect(DB_PATH)

    print(f"Pulling real wRC+ (MLB Stats API sabermetrics) for {season}...", end=" ", flush=True)
    wrc_map = _sabermetrics_map(season, "hitting", "wRcPlus")
    print(f"done, {len(wrc_map)} players")

    print(f"Pulling real xFIP (MLB Stats API sabermetrics) for {season}...", end=" ", flush=True)
    xfip_map = _sabermetrics_map(season, "pitching", "xfip")
    print(f"done, {len(xfip_map)} players")

    print(f"Pulling real Barrel% - batting (Baseball Savant) for {season}...", end=" ", flush=True)
    bat_barrels = statcast_batter_exitvelo_barrels(season, minBBE=1)
    barrel_bat_map = dict(zip(bat_barrels["player_id"], bat_barrels["brl_percent"] / 100))
    print(f"done, {len(barrel_bat_map)} players")

    print(f"Pulling real Barrel% - pitching (Baseball Savant) for {season}...", end=" ", flush=True)
    pit_barrels = statcast_pitcher_exitvelo_barrels(season, minBBE=1)
    barrel_pit_map = dict(zip(pit_barrels["player_id"], pit_barrels["brl_percent"] / 100))
    print(f"done, {len(barrel_pit_map)} players")

    print(f"Pulling real Arm Strength (Baseball Savant) for {season}...", end=" ", flush=True)
    arm_map = _arm_strength_map(season)
    print(f"done, {len(arm_map)} players")

    print(f"Pulling real batting WAR (MLB Stats API sabermetrics) for {season}...", end=" ", flush=True)
    war_bat_map = _sabermetrics_map(season, "hitting", "war")
    print(f"done, {len(war_bat_map)} players")

    print(f"Pulling real pitching WAR (MLB Stats API sabermetrics) for {season}...", end=" ", flush=True)
    war_pit_map = _sabermetrics_map(season, "pitching", "war")
    print(f"done, {len(war_pit_map)} players")

    all_ids = set(wrc_map) | set(xfip_map) | set(barrel_bat_map) | set(barrel_pit_map) | set(arm_map) | set(war_bat_map) | set(war_pit_map)
    rows = [
        (pid, season, wrc_map.get(pid), xfip_map.get(pid), barrel_bat_map.get(pid), barrel_pit_map.get(pid),
         arm_map.get(pid), war_bat_map.get(pid), war_pit_map.get(pid))
        for pid in all_ids
    ]

    conn.executemany("""
        INSERT INTO player_advanced_stats
            (player_id, season, wrc_plus, xfip, barrel_pct_batting, barrel_pct_pitching, arm_strength, war_batting, war_pitching)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id, season) DO UPDATE SET
            wrc_plus=excluded.wrc_plus, xfip=excluded.xfip,
            barrel_pct_batting=excluded.barrel_pct_batting, barrel_pct_pitching=excluded.barrel_pct_pitching,
            arm_strength=excluded.arm_strength,
            war_batting=excluded.war_batting, war_pitching=excluded.war_pitching
    """, rows)
    conn.commit()

    print(f"\nUpserted {len(rows)} players ({len(wrc_map)} wRC+, {len(xfip_map)} xFIP, "
          f"{len(barrel_bat_map)} batting Barrel%, {len(barrel_pit_map)} pitching Barrel%, "
          f"{len(arm_map)} arm strength, {len(war_bat_map)} batting WAR, {len(war_pit_map)} pitching WAR).")
    conn.close()


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEASON
    run(season)
