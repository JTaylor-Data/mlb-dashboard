# verify-both verification report
Generated: 2026-10-06  
Branch: verify-both (master ← test-fixes ← archive-and-chart, both merged --no-ff, no conflicts)

---

## 1. `git diff --stat master..archive-and-chart` — reconciliation

20 files changed, 850 insertions(+), 123 deletions(-).  
**Zero renames, zero deletes.** Every entry is either a new file (9 new READMEs plus
`06_archive_policy.md`, `STARTUP_AND_STORAGE.md`, `check_archive_hygiene.py`, etc.) or an
in-place edit to an existing file (`player.js`, `player_page.py`, `players.py`,
`season_phase.py`, `index.html`, `trends.js`, `trends.html`).

The earlier statement "no files moved" is correct.

---

## 2. `pytest -q` — cold start from verify-both

```
198 passed, 1 skipped, 2 xfailed, 2 warnings in 3739.26s (1:02:19)
```

### Skipped (1)

| Test | Reason |
|---|---|
| `test_streaming_or_artifact_routes[/api/tunneling/v3/report?season=2026]` | Tunneling v3 report artifacts not present on this machine (route returns 404). Skip via `pytest.skip` in `test_routes_smoke.py`. |

### Expected failures / xfailed (2)

| Test | Reason |
|---|---|
| `test_snapshot_match[tunneling_v3_2026-...]` | Requires precomputed Tunneling+ v3 artifacts — marked `pytest.xfail` in `test_golden_snapshots.py`. |
| `test_snapshot_match[tunneling_cole_2026-...]` | Same reason. |

These are correct: Tunneling+ v3 is out of scope on this machine and the xfails are the intended handling.

### Warnings (2)

Both are the same FastAPI `on_event is deprecated` notice from `api/main.py:46`. Not a test failure; cosmetic only.

---

## 3. Golden snapshot results

16 routes in `SNAPSHOT_ROUTES`. Breakdown:

| Category | Routes | Result |
|---|---|---|
| **Exact match** | meta, standings, stuff_model_meta, stuff_cole_2026, location_meta, location_cole_2026, heatmaps_cole_2026, stuff_pred_validity | All **passed** — byte-identical after stripping volatile fields |
| **Structural-only** (data drifts legitimately) | playoffs_bracket, stuff_lb_2025, stuff_lb_2026, radar_cole, player_cole, probable_pitchers | All **passed** — same top-level keys and value types |
| **xfailed** (artifacts absent) | tunneling_v3_2026, tunneling_cole_2026 | Expected failures, not counted against green |

All 14 runnable snapshots matched. No unexpected differences.

Volatile fields excluded from exact comparison:
- `meta`: `latest_game_date`, `latest_schedule_date`
- `probable_pitchers`: `date`, `pulled_at` (None-safe)
- `standings`: `last_updated`
- `stuff_model_meta`: `trained_at`
- `player_cole`: `il_status`, `on_il`

---

## 4. Heatmap split=L vs split=R

```
split=L vs split=R payload same: False   ✓ (different batter populations)
default  vs split=R payload same: True   ✓ (default is R, as expected)
```

The handedness cache key is working correctly.

---

## 5. Tunneling+ page and endpoint check

| Route | Status | Notes |
|---|---|---|
| `/api/tunneling/v3/report?season=2026` | **404** | Expected — artifacts not present on this machine. Returns `{"detail": "No Tunneling+ v3 artifacts found — run train_tunneling_v3_model.py + tunneling_v3_validate.py first"}`. |
| `/api/tunneling/v3/pitcher/543037?season=2026` | **200** | Per-pitcher endpoint works (uses parquet that is present). |

No other page imports Tunneling+ code in a way that errors. All key modules import cleanly:

```
OK  api.routers.players
OK  api.routers.splits
OK  api.routers.probable_pitchers
OK  api.routers.deception_report
OK  api.routers.heatmaps
OK  analytics.player_page
OK  analytics.stuff_model
OK  analytics.location_v3_model
```

---

## 6. Branch state

| Branch | Status |
|---|---|
| `test-fixes` | Pushed to origin. Not merged into master. |
| `archive-and-chart` | Pushed to origin. Not merged into master. Visual checks still pending. |
| `verify-both` | Local only. Not pushed. Server running on port 8010. |

### pytest.ini

Added `pytest.ini` with `testpaths = tests` to prevent collection errors from
`Archive/experimental/` run scripts. This commit lives on `verify-both` and should be
cherry-picked or re-applied when either branch lands on master.

---

## Summary

**verify-both is green.** Both branches merge cleanly, all tests pass (198/198 runnable),
the two known Tunneling+ xfails are correct and expected, heatmap handedness cache keys
work, and no non-Tunneling page erroneously imports Tunneling code.

Ready to merge `test-fixes` then `archive-and-chart` into master once visual checks
on the xwOBA chart and IL plaque suppression are confirmed.
