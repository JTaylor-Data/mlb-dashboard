# analytics/

Business logic: model scoring, feature engineering, statistics, percentiles.

## Current (live)

All `.py` files in this folder are imported by live API routes. Active model scorers:

- `stuff_model.py` — Stuff+ v4 (ACTIVE)
- `location_v3_model.py` — Location+ v3 (ACTIVE)
- `tunneling_v2_model.py` — Tunneling+ v3 scorer (ACTIVE; out of scope for audit)
- `deception_model.py` — Deception+ v2b (ACTIVE)
- `player_page.py` — Player page data assembly
- `percentiles.py` / `percentile_cache.py` — Percentile computation

Legacy scorers kept for career-page backward compat: `location_model.py`, `location_v2_model.py`, `tunneling_model.py`, `tunneling_v3_model.py`.

## Archive

No files archived yet. See `docs/methodology/06_archive_policy.md`.
