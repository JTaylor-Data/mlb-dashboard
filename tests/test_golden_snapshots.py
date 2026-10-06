"""
Golden snapshot comparison tests.
Compare current API responses to stage0 snapshots — any difference is flagged.
"""
import json
import os
import pytest
import requests

BASE = "http://localhost:8010"
SNAP_DIR = os.path.join(os.path.dirname(__file__), "..", "Methodology", "golden_snapshots", "stage0")

SNAPSHOT_ROUTES = [
    ("meta", "/api/meta"),
    ("standings", "/api/standings"),
    ("probable_pitchers", "/api/probable-pitchers"),
    ("playoffs_bracket", "/api/playoffs/bracket"),
    ("stuff_model_meta", "/api/stuff-model/meta"),
    ("stuff_lb_2026", "/api/stuff-model/leaderboard?season=2026"),
    ("stuff_lb_2025", "/api/stuff-model/leaderboard?season=2025"),
    ("stuff_cole_2026", "/api/stuff-model/pitcher/543037?season=2026"),
    ("location_meta", "/api/location/meta"),
    ("location_cole_2026", "/api/location/pitcher/543037?season=2026"),
    ("player_cole", "/api/player/543037"),
    ("heatmaps_cole_2026", "/api/heatmaps/543037?season=2026"),
    ("tunneling_v3_2026", "/api/tunneling/v3/report?season=2026"),
    ("tunneling_cole_2026", "/api/tunneling/v3/pitcher/543037?season=2026"),
    ("radar_cole", "/api/radar/543037"),
    ("stuff_pred_validity", "/api/stuff-model/predictive-validity"),
]


def load_snapshot(name: str):
    path = os.path.join(SNAP_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"No snapshot file: {name}.json")
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


@pytest.mark.parametrize("name,route", SNAPSHOT_ROUTES)
def test_snapshot_match(name, route):
    snap = load_snapshot(name)
    resp = requests.get(f"{BASE}{route}", timeout=60)
    assert resp.status_code == 200
    current = resp.json()
    assert json.dumps(current, sort_keys=True) == json.dumps(snap, sort_keys=True), \
        f"Snapshot mismatch for {name} ({route})"
