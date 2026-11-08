"""The past-season roles snapshot must equal a live classify_pitcher_roles() scan, and
player_last_n_games must give the same games whether or not the caller hands it the roles."""
import os
import sqlite3

import pandas as pd
import pytest

from analytics.metrics import classify_pitcher_roles
from analytics.player_page import player_last_n_games
from api.routers.players import _pitcher_roles

DB = os.environ.get("DB_PATH", "mlb_data.db")
pytestmark = pytest.mark.skipif(not os.path.exists(DB), reason="needs the local database")


def test_snapshot_roles_match_live_scan():
    conn = sqlite3.connect(DB)
    live = classify_pitcher_roles(conn, 2022)
    snap = _pitcher_roles(conn, 2022)  # served from parquets/past_season_pitcher_page.json
    pd.testing.assert_frame_equal(
        snap.sort_index(), live.sort_index()[snap.columns], check_dtype=False, check_names=False)


def test_last_games_same_with_or_without_roles():
    conn = sqlite3.connect(DB)
    roles = _pitcher_roles(conn, 2022)
    for pid in (roles[roles.role == "SP"].index[0], roles[roles.role == "RP"].index[0]):
        assert player_last_n_games(conn, 2022, int(pid), "pitcher", 5) == \
            player_last_n_games(conn, 2022, int(pid), "pitcher", 5, roles=roles)


def test_snapshot_row_count_and_hr_fb_match_live():
    from analytics.percentile_cache import _season_row_count
    from analytics.player_page import league_hr_per_fb
    from api.routers.players import _league_hr_fb
    conn = sqlite3.connect(DB)
    for season in (2021, 2024):
        assert _season_row_count(conn, season) == conn.execute(
            "SELECT COUNT(*) FROM pitches WHERE season = ?", (season,)).fetchone()[0]
        assert _league_hr_fb(conn, season) == pytest.approx(league_hr_per_fb(conn, season))
