# diagnostics/

Diagnostic artifacts for model validation and quality tracking.

## Current (live)

- `stuff_v4/` — Stuff+ v4 diagnostic artifacts; **read by `tools/build_validation_content.py`** (required for the validation page to build). Keep all files here.
- `location_v3/` — Location+ v3 validation artifacts; **read by `api/routers/location_report.py`** at runtime (`gate_results.json`, `validation_results.json`, `scoring_config.json`). Keep all files here.

## Archive (dead / superseded)

- `closeout_blend/` — Old closeout blend diagnostics; not read by any live code. **Candidate for move to `Archive/diagnostics/`** but kept here per ARCHIVE_POLICY.md's "unproven" rule until confirmed safe.
- `deception_v2/` — Deception+ v2 score parquets; scores served from `scores/` parquets, not from here. **Candidate for move to `Archive/diagnostics/`**.

See `Methodology/ARCHIVE_POLICY.md` for the policy on when to move files to Archive/.
