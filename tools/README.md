# tools/

Build utilities and maintenance scripts. These are owner-run tools, not server code.

## Current (live)

- `build_validation_content.py` — Regenerates `frontend/content/validation_results.json` from `diagnostics/stuff_v4/` artifacts
- `build_percentile_pools.py` — Builds percentile pool parquets for complete past seasons
- `freeze_season.py` — Utility for freezing a season's worth of model scores
- `verify_stuff_page.py` — Smoke-checks the Stuff+ model page output
- `model_correlation_heatmap.py` — Generates model correlation heatmap for diagnostics
- `maintenance/rescore_models.py` — Scheduled weekly rescore pipeline (also run by APScheduler)
- `precompute_tunneling_pairs.py` — Writes the current season's tunneling pair and total parquets (run by the daily pull when new pitches arrive)
- `build_unqualified_pools.py` — Builds and persists the threshold-free percentile pools (`*_unq_*` under `parquets/percentile_pools/`) for every complete season, so the first unqualified player viewed per season/split doesn't rescan the pitches table. Safe to rerun; files are versioned by the season's row count
- `precompute_career_aggregates.py` — Writes `parquets/career_season_aggregates.json`, the per-season league aggregates the Career page reads for complete seasons. Rerun it whenever a past season's pitch data is re-pulled or corrected, and at each season rollover after bumping `CURRENT_SEASON`; the file's `_meta` records the build time and per-season pitch-row counts so staleness can be checked
- `maintenance/check_archive_hygiene.py` — Lists unreferenced live files as Archive candidates

## Archive

No files archived yet. See `docs/methodology/06_archive_policy.md`.
