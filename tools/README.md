# tools/

Build utilities and maintenance scripts. These are owner-run tools, not server code.

## Current (live)

- `build_validation_content.py` — Regenerates `frontend/content/validation_results.json` from `diagnostics/stuff_v4/` artifacts
- `build_percentile_pools.py` — Builds percentile pool parquets for complete past seasons
- `freeze_season.py` — Utility for freezing a season's worth of model scores
- `verify_stuff_page.py` — Smoke-checks the Stuff+ model page output
- `model_correlation_heatmap.py` — Generates model correlation heatmap for diagnostics
- `maintenance/rescore_models.py` — Scheduled weekly rescore pipeline (also run by APScheduler)
- `maintenance/check_archive_hygiene.py` — Lists unreferenced live files as Archive candidates

## Archive

No files archived yet. See `docs/methodology/ARCHIVE_POLICY.md`.
