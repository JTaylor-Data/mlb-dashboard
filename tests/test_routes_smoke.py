"""
Route smoke tests — verify every main endpoint returns 200 with JSON.
These are the golden-snapshot routes; extend as routes are added.
"""
import pytest
import requests

BASE = "http://localhost:8010"

ROUTES = [
    "/api/meta",
    "/api/standings",
    "/api/probable-pitchers",
    "/api/playoffs/bracket",
    "/api/playoffs/division-leaders",
    "/api/playoffs/wildcard",
    "/api/playoffs/teams",
    "/api/news",
    "/api/stuff-model/meta",
    "/api/stuff-model/leaderboard?season=2026",
    "/api/stuff-model/leaderboard?season=2025",
    "/api/stuff-model/pitcher/543037?season=2026",
    "/api/location/meta",
    "/api/location/leaderboard?season=2026",
    "/api/location/pitcher/543037?season=2026",
    "/api/player/543037",
    "/api/players/543037/outings?season=2026",
    "/api/heatmaps/543037?season=2026",
    "/api/heatmaps/543037?season=2026&split=R",
    "/api/heatmaps/543037?season=2026&split=L",
    "/api/trends/league-averages",
    "/api/players/543037/trends?season=2026",
    "/api/radar/543037",
    "/api/stuff-model/predictive-validity",
    "/api/stuff-model/pitch-types?season=2026",
    "/api/team/147/card",
    "/api/hitzones/592518?season=2026",
    "/api/splits/pitch-types?player_id=543037&season=2026",
    "/api/top-performers?season=2026&date=2026-09-27",
    "/api/top-performers?season=2025&date=2025-09-28",
    "/api/player/543037/similar?season=2026&role=pitcher",
]

# Routes that return streaming NDJSON (not a single JSON object)
NDJSON_ROUTES = {"/api/player/543037/progress?season=2026"}

# Routes that require large precomputed artifacts not present on all machines
SKIP_IF_MISSING_ARTIFACTS = {
    "/api/tunneling/v3/report?season=2026",
    "/api/tunneling/v3/pitcher/543037?season=2026",
}


@pytest.mark.parametrize("route", ROUTES)
def test_route_200(route):
    resp = requests.get(f"{BASE}{route}", timeout=60)
    assert resp.status_code == 200, f"{route} returned {resp.status_code}: {resp.text[:200]}"
    if route not in NDJSON_ROUTES:
        data = resp.json()
        assert data is not None


@pytest.mark.parametrize("route", [
    "/api/player/543037/progress?season=2026",
    "/api/tunneling/v3/report?season=2026",
    "/api/tunneling/v3/pitcher/543037?season=2026",
])
def test_streaming_or_artifact_routes(route):
    """Routes that stream NDJSON or require precomputed artifacts — just check status code."""
    resp = requests.get(f"{BASE}{route}", timeout=60)
    if route in SKIP_IF_MISSING_ARTIFACTS and resp.status_code in (404, 500):
        pytest.skip(f"{route} artifacts not present on this machine (status {resp.status_code})")
    assert resp.status_code == 200, f"{route} returned {resp.status_code}: {resp.text[:200]}"
