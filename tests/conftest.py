"""
Pytest configuration and fixtures for MLB Dashboard audit tests.
Run with: pytest tests/ -q
Server must be running on port 8010.
"""
import json
import os
import pytest
import requests

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8010")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "golden_snapshots", "stage0")


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def client():
    session = requests.Session()
    session.timeout = 30
    return session


def load_snapshot(name: str) -> dict:
    path = os.path.join(SNAPSHOT_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
