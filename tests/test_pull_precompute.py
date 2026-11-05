from pathlib import Path
import subprocess

import pytest

from api import pull
from api.db import DB_PATH

REPO = Path(__file__).resolve().parent.parent
PAIRS_2026 = REPO / "parquets" / "tunneling_pairs_2026.parquet"


def test_precompute_writes_parquet_newer_than_db():
    db = Path(DB_PATH)
    if not db.exists():
        pytest.skip("mlb_data.db not present")
    pull._precompute_tunneling_pairs(2026)
    assert PAIRS_2026.exists()
    assert PAIRS_2026.stat().st_mtime >= db.stat().st_mtime


def test_precompute_failure_does_not_raise(monkeypatch):
    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="precompute", timeout=1)
    monkeypatch.setattr(pull.subprocess, "run", boom)
    pull._precompute_tunneling_pairs(2026)
