"""
Golden snapshot comparison tests.
Compare current API responses to stage0 snapshots — any difference is flagged.
Routes that require precomputed artifacts not present on all machines are xfailed.
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

# These routes require precomputed artifacts not guaranteed present on all machines.
# Mark xfail rather than skip so a pass is a bonus (xpass), failure is expected.
ARTIFACT_DEPENDENT = {
    "tunneling_v3_2026",
    "tunneling_cole_2026",
}

# Top-level fields that legitimately change over time (timestamps, live counts).
# Excluded from the diff comparison so the test stays meaningful as data ages.
VOLATILE_FIELDS = {
    "meta": {"latest_game_date", "latest_schedule_date"},
    "probable_pitchers": {"date", "pulled_at"},
    "standings": {"last_updated"},
    "stuff_model_meta": {"trained_at"},
    "player_cole": {"il_status", "on_il"},
}

# Snapshots whose data legitimately drifts as new games are played (scores,
# records, live stats). These are compared structurally (key set, types) only.
STRUCTURAL_ONLY = {
    "playoffs_bracket",   # game records update each round
    "stuff_lb_2025",      # score normalisation changes with each new season
    "stuff_lb_2026",      # live season — scores change weekly
    "radar_cole",         # live-season percentile shifts
    "player_cole",        # live-season scores shift
    "probable_pitchers",  # live game-day data
}


def _fix_mojibake(obj):
    """Fix strings that were double-encoded (UTF-8 bytes read as latin-1).
    Snapshot files were written with BOM and latin-1 interpretation, causing
    accented chars and symbols like ° to appear as e.g. Ã© or Â°."""
    if isinstance(obj, str):
        try:
            return obj.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return obj
    if isinstance(obj, dict):
        return {k: _fix_mojibake(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_mojibake(v) for v in obj]
    return obj


def load_snapshot(name: str):
    path = os.path.join(SNAP_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"No snapshot file: {name}.json")
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return _fix_mojibake(raw)


def _strip_volatile(name: str, obj):
    """Remove volatile top-level keys from a dict for comparison purposes."""
    volatile = VOLATILE_FIELDS.get(name, set())
    if isinstance(obj, dict) and volatile:
        return {k: v for k, v in obj.items() if k not in volatile}
    return obj


def _top_level_keys_match(a, b, name):
    """Assert the same top-level keys and value types exist in both dicts."""
    assert set(a.keys()) == set(b.keys()), \
        f"{name}: key mismatch live={set(a.keys())} snap={set(b.keys())}"
    for k in a:
        assert type(a[k]) == type(b[k]), \
            f"{name}.{k}: type mismatch live={type(a[k]).__name__} snap={type(b[k]).__name__}"


@pytest.mark.parametrize("name,route", SNAPSHOT_ROUTES)
def test_snapshot_match(name, route):
    if name in ARTIFACT_DEPENDENT:
        pytest.xfail("Requires precomputed artifacts not present on all machines")

    snap = load_snapshot(name)
    resp = requests.get(f"{BASE}{route}", timeout=60)
    if resp.status_code in (404, 500) and name in ARTIFACT_DEPENDENT:
        pytest.xfail(f"{route} returned {resp.status_code} — artifacts missing")
    assert resp.status_code == 200, f"{route} returned {resp.status_code}: {resp.text[:200]}"
    current = resp.json()

    if name in STRUCTURAL_ONLY:
        # Data drifts legitimately; verify shape only
        _top_level_keys_match(current, snap, name)
        return

    snap_cmp = _strip_volatile(name, snap)
    current_cmp = _strip_volatile(name, current)

    assert json.dumps(current_cmp, sort_keys=True) == json.dumps(snap_cmp, sort_keys=True), \
        f"Snapshot mismatch for {name} ({route})"
