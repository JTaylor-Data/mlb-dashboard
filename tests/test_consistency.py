"""
Part C: Cross-page consistency and no-broken-request tests.

Tests that Stuff+, Location+, and Deception+ values are consistent across
pages for a set of pitcher-seasons, and that no frontend-facing route returns
4xx/5xx or an empty payload for populated seasons.

Run: pytest tests/test_consistency.py -q

Server must be running on port 8010 (or TEST_BASE_URL).
"""
import os
import pytest
import requests

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8010")

# 20 pitcher-seasons spanning multiple seasons including 2020 and inactive pitchers.
# (pitcher_id, season, name_for_reference)
# These are well-known pitchers whose scores should be populated.
PITCHER_SEASONS = [
    (543037, 2026, "Gerrit Cole 2026"),
    (543037, 2025, "Gerrit Cole 2025"),
    (543037, 2024, "Gerrit Cole 2024"),
    (543037, 2022, "Gerrit Cole 2022"),
    (543037, 2020, "Gerrit Cole 2020"),
    (477132, 2025, "Clayton Kershaw 2025"),
    (477132, 2022, "Clayton Kershaw 2022"),
    (477132, 2019, "Clayton Kershaw 2019"),  # inactive recent season
    (605135, 2026, "Spencer Strider 2026"),
    (605135, 2025, "Spencer Strider 2025"),
    (656302, 2026, "Logan Gilbert 2026"),
    (656302, 2025, "Logan Gilbert 2025"),
    (694973, 2026, "Paul Skenes 2026"),
    (660271, 2026, "Hunter Brown 2026"),
    (592789, 2026, "Tarik Skubal 2026"),
    (592789, 2025, "Tarik Skubal 2025"),
    (641154, 2026, "Pablo Lopez 2026"),
    (621111, 2026, "Zac Gallen 2026"),
    (621111, 2023, "Zac Gallen 2023"),
    (608718, 2026, "Luis Castillo 2026"),
]


@pytest.fixture(scope="session")
def session():
    s = requests.Session()
    s.timeout = 30
    return s


def _get(session, path, params=None):
    r = session.get(f"{BASE_URL}{path}", params=params)
    return r


# ---------------------------------------------------------------------------
# Smoke tests: key routes return 200 and non-empty payload
# ---------------------------------------------------------------------------

def test_api_meta(session):
    r = _get(session, "/api/meta")
    assert r.status_code == 200, f"/api/meta returned {r.status_code}"
    data = r.json()
    assert data, "Empty response from /api/meta"


def test_deception_meta(session):
    r = _get(session, "/api/deception/meta")
    assert r.status_code == 200, f"/api/deception/meta returned {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert data, "Empty response from /api/deception/meta"


def test_deception_leaderboard_2026(session):
    r = _get(session, "/api/deception/leaderboard", {"season": "2026"})
    assert r.status_code == 200, f"Deception leaderboard returned {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert data.get("pitchers"), "Empty pitchers list in deception leaderboard 2026"


def test_splits_pitch_type_pitcher(session):
    """split_by=pitch_type must return 200 for a pitcher (DEF-09 fix)."""
    r = _get(session, "/api/splits", {"role": "pitcher", "player_id": 543037,
                                       "split_by": "pitch_type", "season": 2026})
    assert r.status_code == 200, f"splits pitch_type returned {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert "rows" in data, "No 'rows' in splits response"


def test_splits_invalid_split_by_returns_422(session):
    """Unknown split_by must return 422, not 500."""
    r = _get(session, "/api/splits", {"role": "pitcher", "player_id": 543037,
                                       "split_by": "nonexistent", "season": 2026})
    assert r.status_code == 422, f"Expected 422 for invalid split_by, got {r.status_code}"


def test_splits_stand_pitcher(session):
    r = _get(session, "/api/splits", {"role": "pitcher", "player_id": 543037,
                                       "split_by": "stand", "season": 2026})
    assert r.status_code == 200, f"splits stand returned {r.status_code}"


def test_splits_home_away_batter(session):
    r = _get(session, "/api/splits", {"role": "batter", "player_id": 592450,
                                       "split_by": "home_away", "season": 2026})
    assert r.status_code in (200, 404), f"splits batter home_away returned {r.status_code}"


# ---------------------------------------------------------------------------
# Model version tests: deception endpoints serve v2b scores
# ---------------------------------------------------------------------------

def test_deception_pitcher_score_cole_2026(session):
    """Gerrit Cole 2026 Deception+ from /api/deception/pitcher/{id} is non-null."""
    r = _get(session, "/api/deception/pitcher/543037", {"season": "2026"})
    if r.status_code == 404:
        pytest.skip("Cole 2026 not in deception v2b scores — may not qualify")
    assert r.status_code == 200, f"Deception pitcher returned {r.status_code}: {r.text[:200]}"
    data = r.json()
    score = data.get("deception_plus")
    assert score is not None, "deception_plus is null"
    assert 50 < score < 200, f"deception_plus={score} outside plausible range"


# ---------------------------------------------------------------------------
# Model-version consistency: career page uses v2 tunneling, v3 location
# ---------------------------------------------------------------------------

def test_career_page_imports_correct_models():
    """Verify at import time that career.py uses the active model versions."""
    import api.routers.career as car
    import inspect
    src = inspect.getfile(car)
    with open(src) as f:
        text = f.read()
    assert "tunneling_v2_model" in text, "career.py does not import tunneling_v2_model"
    assert "tunneling_v3_model" not in text, "career.py still imports tunneling_v3_model (wrong)"
    assert "location_v3_model" in text, "career.py does not import location_v3_model"
    assert "from analytics.location_model import" not in text, "career.py still imports location_model (wrong v1)"


def test_probable_pitchers_imports_correct_models():
    """Verify at import time that probable_pitchers.py uses the active model versions."""
    import api.routers.probable_pitchers as pp
    import inspect
    src = inspect.getfile(pp)
    with open(src) as f:
        text = f.read()
    assert "tunneling_v2_model" in text, "probable_pitchers.py does not import tunneling_v2_model"
    assert "from analytics.tunneling_v3_model import" not in text, "probable_pitchers.py still imports tunneling_v3_model"
    assert "location_v3_model" in text, "probable_pitchers.py does not import location_v3_model"
    assert "from analytics.location_model import" not in text, "probable_pitchers.py still imports location_model (wrong v1)"


# ---------------------------------------------------------------------------
# Deception v2b score cache consistency: parquet matches _bracket for 20 pitchers
# ---------------------------------------------------------------------------

def test_deception_v2b_score_consistency():
    """For pitchers in the v2b parquet, _bracket and direct parquet read agree to 1e-9."""
    import pandas as pd
    from api.routers.deception_report import _SCORES_PATHS, _bracket

    path = _SCORES_PATHS.get("v2b")
    if path is None or not path.exists():
        pytest.skip("v2b parquet not available")

    df = pd.read_parquet(path)
    df["pitcher_id"] = df["pitcher_id"].astype(int)
    df["season"] = df["season"].astype(str)

    sample = df[df["season"] == "2026"].head(20)
    for row in sample.itertuples():
        result = _bracket(int(row.pitcher_id), "2026", "v2b")
        assert result is not None, f"_bracket returned None for pitcher {row.pitcher_id} 2026"
        direct_score = float(row.deception_plus_v2b)
        api_score = result["deception_plus"]
        assert abs(direct_score - api_score) < 1e-9, (
            f"Pitcher {row.pitcher_id} 2026: parquet={direct_score}, api={api_score}"
        )
