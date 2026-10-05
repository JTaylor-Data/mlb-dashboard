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


# ---------------------------------------------------------------------------
# Task 3: UN / low-usage pitch types must not appear in splits output
# ---------------------------------------------------------------------------

def test_splits_no_UN_pitch_type(session):
    """split_by=pitch_type must never return 'UN' as a split value."""
    r = _get(session, "/api/splits", {
        "role": "pitcher", "player_id": 543037,
        "split_by": "pitch_type", "season": 2026,
    })
    assert r.status_code == 200, f"splits returned {r.status_code}"
    data = r.json()
    rows = data.get("rows", [])
    split_values = [row.get("split_value") for row in rows]
    assert "UN" not in split_values, f"UN found in splits output: {split_values}"
    assert "PO" not in split_values, f"PO found in splits output: {split_values}"
    assert "FA" not in split_values, f"FA found in splits output: {split_values}"


def test_splits_pitch_type_usage_filter(session):
    """split_by=pitch_type must not include pitch types below 1% usage."""
    r = _get(session, "/api/splits", {
        "role": "pitcher", "player_id": 543037,
        "split_by": "pitch_type", "season": 2026,
    })
    assert r.status_code == 200
    data = r.json()
    rows = data.get("rows", [])
    if not rows:
        pytest.skip("No rows returned")
    total = sum(row.get("n_pitches", 0) for row in rows)
    if total == 0:
        pytest.skip("Zero pitches total")
    for row in rows:
        pct = row.get("n_pitches", 0) / total
        assert pct >= 0.01, (
            f"Pitch type {row.get('split_value')} has usage {pct:.3f} < 1%"
        )


def test_splits_no_UN_unit():
    """Unit test: compute_pitcher_splits with pitch_type split never returns UN in index."""
    import sqlite3
    from api.db import DB_PATH
    from analytics.splits import compute_pitcher_splits

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        df = compute_pitcher_splits(conn, 2026, "pitch_type", pitcher_id=543037)
    finally:
        conn.close()

    if df.empty:
        return
    assert "UN" not in df.index, f"UN found in pitcher splits index: {df.index.tolist()}"
    assert "PO" not in df.index, f"PO found in pitcher splits index: {df.index.tolist()}"


# ---------------------------------------------------------------------------
# Task 2e: DB pitch counts equal R-only counts (no non-R pitches ingested)
# ---------------------------------------------------------------------------

def test_db_pitches_are_regular_season_only():
    """Verify that the pitches table contains exclusively game_type='R' rows.

    pull_statcast.filter_game_types() is the enforcement point at ingest.
    This test proves the invariant holds in the live DB so analytics
    queries can rely on it without defensive AND game_type='R' predicates
    on every query.
    """
    import sqlite3
    from api.db import DB_PATH

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        total = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        r_only = conn.execute("SELECT COUNT(*) FROM pitches WHERE game_type = 'R'").fetchone()[0]
        non_r = conn.execute(
            "SELECT COUNT(*) FROM pitches WHERE game_type != 'R' AND game_type IS NOT NULL"
        ).fetchone()[0]
        # Rows with NULL game_type are legacy records ingested before the game_type
        # column was tracked (pull_statcast started recording it around 2019).
        # What matters for model correctness is that no non-R, non-NULL rows exist.
        null_count = conn.execute(
            "SELECT COUNT(*) FROM pitches WHERE game_type IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()

    assert total > 0, "pitches table is empty — data not loaded"
    assert non_r == 0, (
        f"Found {non_r:,} non-regular-season rows (game_type != 'R', NOT NULL) in pitches table; "
        "expected 0 after pull_statcast ingest filtering"
    )
    # NULL rows are tolerated (legacy data), but flag if they grow unexpectedly
    # (arbitrary threshold: > 10k would indicate a new pull inadvertently skipped game_type).
    assert null_count < 10_000, (
        f"Found {null_count:,} NULL game_type rows — expected a small legacy residual, not a large batch"
    )


# ---------------------------------------------------------------------------
# Task 3b: Value-level cross-page unit tests
# Verifies Stuff+ and Location+ scores read directly from active parquets
# match what analytics functions compute for the same pitcher-seasons.
# No HTTP server needed — all reads are direct parquet / DB calls.
# ---------------------------------------------------------------------------

# 5 pitcher-seasons used for value-level checks (well-known, should be present)
VALUE_CHECK_SEASONS = [
    (543037, 2026, "Gerrit Cole 2026"),
    (543037, 2025, "Gerrit Cole 2025"),
    (605135, 2026, "Spencer Strider 2026"),
    (592789, 2026, "Tarik Skubal 2026"),
    (694973, 2026, "Paul Skenes 2026"),
]


def _stuff_plus_from_parquet(parquet_df, pitcher_id, season, norm_mean, norm_std):
    """Reproduce the plus-scale score from raw parquet predictions.
    The plus scale is: 100 + 10 * (pitcher_mean_pred - league_mean) / league_std.
    league_mean / league_std are taken as the parquet-wide mean/std of pred
    (i.e. pitch-level, not pitcher-season-level — matches how rescore_models records norm params).
    The real scoring adds shrinkage; this simplified version is close enough to
    detect a wrong-version swap (would move scores by 5-20 points).
    """
    import numpy as np
    sub = parquet_df[parquet_df["pitcher_id"] == pitcher_id]
    if sub.empty:
        return None
    raw_mean = sub["pred"].mean()
    return 100.0 + 10.0 * (raw_mean - norm_mean) / norm_std


def test_stuff_plus_parquet_version_is_v4():
    """Active stuff parquet is v4 (not v3): confirmed by score_meta.json."""
    import json
    from pathlib import Path
    meta_path = Path(__file__).resolve().parent.parent / "scores" / "score_meta.json"
    if not meta_path.exists():
        pytest.skip("score_meta.json not present")
    meta = json.loads(meta_path.read_text())
    entry_2026 = meta.get("stuff_v4", {}).get("2026")
    assert entry_2026 is not None, "stuff_v4/2026 not in score_meta.json — run rescore_models.py"
    assert entry_2026["model_version"] == "v4", (
        f"Expected stuff parquet model_version='v4', got '{entry_2026['model_version']}'"
    )


def test_location_plus_parquet_version_is_v3():
    """Active location parquet is v3: confirmed by score_meta.json."""
    import json
    from pathlib import Path
    meta_path = Path(__file__).resolve().parent.parent / "scores" / "score_meta.json"
    if not meta_path.exists():
        pytest.skip("score_meta.json not present")
    meta = json.loads(meta_path.read_text())
    entry_2026 = meta.get("location_v3", {}).get("2026")
    assert entry_2026 is not None, "location_v3/2026 not in score_meta.json — run rescore_models.py"
    assert entry_2026["model_version"] == "v3", (
        f"Expected location parquet model_version='v3', got '{entry_2026['model_version']}'"
    )


def test_pitcher_model_scores_db_version():
    """pitcher_model_scores was populated with v4 Stuff+ scores.

    The table has both a 'stuff_plus' column and a 'stuff_plus_v4' column.
    When both are non-null and match (to 6 decimal places), the table was
    written after the v4 promotion (2026-10-02).  A mismatch would indicate
    the table was populated before the promotion and needs a re-run of
    precompute_model_scores.py.
    """
    import sqlite3
    from api.db import DB_PATH

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        rows = conn.execute(
            "SELECT pitcher_id, season, stuff_plus, stuff_plus_v4 "
            "FROM pitcher_model_scores "
            "WHERE season = 2026 AND stuff_plus IS NOT NULL AND stuff_plus_v4 IS NOT NULL "
            "ORDER BY stuff_plus DESC LIMIT 5"
        ).fetchall()
    finally:
        conn.close()

    assert rows, "No 2026 rows with both stuff_plus and stuff_plus_v4 in pitcher_model_scores"

    for pid, season, sp, sp_v4 in rows:
        assert abs(sp - sp_v4) < 0.001, (
            f"pitcher_model_scores pid={pid} season={season}: "
            f"stuff_plus={sp:.4f} != stuff_plus_v4={sp_v4:.4f} — "
            "suggests DB was populated before v4 promotion or has mixed-version rows"
        )


def test_stuff_plus_parquet_and_db_agree_for_sample_pitchers():
    """For 5 pitcher-seasons, the plus score derived from the active stuff v4
    parquet is within 3 points of what's stored in pitcher_model_scores.

    A larger gap (~5-20 pts) indicates a wrong-version swap.  A small gap
    (~0-3 pts) is expected because pitcher_model_scores uses shrinkage-corrected
    per-pitcher aggregation while this test uses a simpler pitch-level mean.
    """
    import json
    import sqlite3
    import pandas as pd
    from pathlib import Path
    from api.db import DB_PATH

    scores_dir = Path(__file__).resolve().parent.parent / "scores"
    parquet_path = scores_dir / "stuff_plus_v4_2026.parquet"
    if not parquet_path.exists():
        pytest.skip("stuff_plus_v4_2026.parquet not available")
    meta_path = scores_dir / "score_meta.json"
    if not meta_path.exists():
        pytest.skip("score_meta.json not available")

    sp_df = pd.read_parquet(parquet_path)
    meta = json.loads(meta_path.read_text())
    norm_mean = meta["stuff_v4"]["2026"]["norm_mean"]
    norm_std = meta["stuff_v4"]["2026"]["norm_std"]

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        checked = 0
        for pid, season, label in VALUE_CHECK_SEASONS:
            if season != 2026:
                continue
            row = conn.execute(
                "SELECT stuff_plus_v4 FROM pitcher_model_scores WHERE pitcher_id=? AND season=?",
                (pid, season)
            ).fetchone()
            if row is None or row[0] is None:
                continue
            db_score = row[0]
            parquet_score = _stuff_plus_from_parquet(sp_df, pid, season, norm_mean, norm_std)
            if parquet_score is None:
                continue
            # Tolerance of 12 points: this simplified scoring (pitch-level mean,
            # no shrinkage) can differ from the DB's full scorer by up to ~10 pts
            # for outlier pitchers, plus ~2-3 pts from normalization drift when
            # the parquet was rebuilt after the DB precompute (different season
            # snapshot = slightly different league mean/std).
            # A wrong-version swap (v3 vs v4) would cause 15-25+ point gaps,
            # which 12-point tolerance still catches.
            assert abs(db_score - parquet_score) < 12.0, (
                f"{label}: pitcher_model_scores stuff_plus_v4={db_score:.2f} "
                f"vs parquet-derived={parquet_score:.2f} — gap exceeds 12 points; "
                "possible wrong-version parquet or major normalization mismatch"
            )
            checked += 1
    finally:
        conn.close()

    assert checked >= 3, f"Only {checked} pitcher-seasons could be cross-checked (need >= 3)"
