# Decisions Needed

This file tracks items the codebase audit flagged as needing owner input before proceeding.
Audit continues past these items (marking them as deferred) rather than stopping.

---

## Open Items

### ND-01 (pre-existing) — Annual Stuff+ retrain
Stuff+ v4 uses LightGBM with 2016-2026 training data. An unseen season (2027) will use the
missing-category branch. Retrain recommended each offseason.
**Owner action**: Schedule retrain for after 2026 season ends.

### ND-02 (pre-existing) — Deception+ arm_angle availability
Deception+ meta lists 2015-2026 as trained seasons but arm_angle is only reliable from 2021.
Pre-2021 scores may be low-quality.
**Owner action**: Decide whether to restrict displayed seasons to 2021+.

### ND-03 (pre-existing) — Deception+ no minimum pitch count
No integer minimum enforced before scoring a pitcher for Deception+. Returns None if features
are empty after dropna, but very small samples may produce unstable scores.
**Owner action**: Decide minimum pitch count (suggest 100).

### ND-DEC-01 (pre-existing, known) — Deception+ v2b 404 and empty scores
`/api/deception/meta` returns 404 for v2b. Deception leaderboard returns "No deception scores
found." The v2b model and score parquets appear not to be loaded.
**Status**: Known pre-existing issue from HANDOFF.md.
**Owner action**: Run deception scoring pipeline to populate scores, then update DECEPTION_ACTIVE_VERSION.

### ND-AUDIT-01 — REPORT_SEASON mismatch
`api/routers/deception_report.py:122` has `REPORT_SEASON = 2025` while `CURRENT_SEASON = 2026`.
All other models serve 2026 data. This means the deception page (when it works) will show 2025.
**Proposed fix**: Change to `REPORT_SEASON = 2026` and pull from `config/model_policy.py.CURRENT_SEASON`.
**Owner action**: Confirm 2026 deception scores exist (or will after ND-DEC-01 resolved) before changing.

### ND-AUDIT-02 — Tunneling naming confusion
`analytics/tunneling_v2_model.py` is the live scorer. API routes say `/v3/`. Router
`api/routers/tunneling_report.py` imports `tunneling_v2_model` directly (correctly).
`api/routers/probable_pitchers.py` imports `tunneling_v3_model` (incorrectly, DEF-03).
**Proposed fix**: Audit every import of tunneling_v3_model; redirect to v2_model where live data is needed.
**Owner action**: Confirm v2 is definitively the active scorer (not v3 with a different purpose).

### ND-AUDIT-03 — Splits 500 error
`/api/splits?player_id=543037&season=2026&split_by=pitch_type` returns 500 Internal Server Error.
**Owner action**: Investigate `api/routers/splits.py` — check correct `split_by` values.

---

## Resolved Items

*(None yet — populate as decisions are made)*
