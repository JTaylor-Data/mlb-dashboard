"""
One-off helper: runs pull_statcast.py's run() across several full regular
seasons in sequence (2021-2024, following the 2025 pilot pull). Not part of
the normal pull workflow - a throwaway script for the multi-season backfill
requested 2026-08-17. Safe to re-run (pull_statcast.run() is itself
idempotent per chunk/date).
"""
import pull_statcast

SEASONS = {
    2024: ("2024-03-28", "2024-09-29"),
    2023: ("2023-03-30", "2023-10-01"),
    2022: ("2022-04-07", "2022-10-05"),
    2021: ("2021-04-01", "2021-10-03"),
}

for year, (start, end) in SEASONS.items():
    print(f"\n=== {year} ({start} to {end}) ===", flush=True)
    pull_statcast.run(start, end)
