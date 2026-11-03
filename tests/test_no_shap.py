"""
Regression guard for the SHAP removal: SHAP added ~240MB to the web
process's boot baseline (lightgbm's TreeExplainer + shap's own sklearn/
scipy dependency chain) for a feature that is no longer served. These
tests make sure it doesn't come back in silently.
"""
import json
import os
import subprocess
import sys

import pytest
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8010")

# Representative routes that previously carried (or could plausibly carry)
# SHAP-shaped fields: the per-pitcher model breakdown endpoints.
SHAP_FREE_ROUTES = [
    "/api/stuff-model/pitcher/543037?season=2026",
    "/api/location/pitcher/543037?season=2026",
    "/api/deception/pitcher/543037",
]


def _contains_shap(obj) -> bool:
    """True if any dict key or string value contains 'shap' (case-insensitive)
    anywhere in a JSON-decoded response."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "shap" in str(k).lower():
                return True
            if _contains_shap(v):
                return True
        return False
    if isinstance(obj, list):
        return any(_contains_shap(v) for v in obj)
    if isinstance(obj, str):
        return "shap" in obj.lower()
    return False


def test_shap_never_imported_after_boot_and_warmup():
    """Importing the full app (same module graph api/main.py loads) and
    running a real warm-up must never pull `shap` into sys.modules.

    Runs in a FRESH subprocess, not the shared pytest process: tests/test_
    imports.py walks and imports every .py file in the repo (including
    Archive/, which still has a couple of modules that `import shap`
    directly) earlier in a full-suite run, which would otherwise poison
    this process's sys.modules with a false positive unrelated to the
    live app's own import graph."""
    probe = (
        "import os, sys; "
        f"os.environ.setdefault('DB_PATH', {os.path.join(ROOT, 'mlb_data.db')!r}); "
        f"sys.path.insert(0, {ROOT!r}); "
        "import api.main; "
        "from api.warmup import warm_cache_sync; "
        "warm_cache_sync(full=True); "
        "assert 'shap' not in sys.modules, 'shap was imported during app boot/warm-up'; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        "shap was imported during app boot/warm-up - it should never load; "
        "check for a reintroduced top-level `import shap` somewhere in the "
        "router/analytics import graph.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.parametrize("route", SHAP_FREE_ROUTES)
def test_route_has_no_shap_fields(route):
    """No response from a model-breakdown route should carry a SHAP-named
    key or value, directly or nested."""
    try:
        resp = requests.get(f"{BASE_URL}{route}", timeout=30)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Server not running at {BASE_URL}")
    if resp.status_code != 200:
        pytest.skip(f"{route} returned {resp.status_code}, nothing to check")
    data = resp.json()
    assert not _contains_shap(data), f"{route} response still contains a SHAP-named field: {data}"


def test_deception_shap_endpoint_is_gone():
    """The old /api/deception/pitcher/{id}/shap endpoint must not exist."""
    try:
        resp = requests.get(f"{BASE_URL}/api/deception/pitcher/543037/shap?season=2025", timeout=30)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Server not running at {BASE_URL}")
    assert resp.status_code == 404, (
        f"Expected the removed SHAP endpoint to 404, got {resp.status_code}"
    )
