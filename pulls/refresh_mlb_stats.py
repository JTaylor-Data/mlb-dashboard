"""
Daily refresh script for MLB Stats API data (rosters, schedule) and news.

Run manually from the mlb_dashboard/ directory:
    python pulls/refresh_mlb_stats.py

There is no built-in scheduler or cron daemon in this Windows environment.
The existing api/scheduler.py runs a daily 6 AM data pull for Statcast data
via APScheduler (already wired to uvicorn startup in api/main.py) but it
does NOT call this script. Options for scheduling this separately:
  1. Add a call to `run()` here from api/scheduler.py's daily job.
  2. Create a Windows Task Scheduler entry pointing at this script.
  3. Run it manually as needed (daily cadence recommended for roster/schedule
     freshness).

The simplest path is option 1: add `from pulls.refresh_mlb_stats import run
as refresh_mlb_stats; refresh_mlb_stats()` inside api/scheduler.py's
daily job function. That is intentionally NOT done automatically here to
avoid modifying scheduler.py without the user's review.
"""
import sys
from pathlib import Path
from datetime import date, timedelta

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = "mlb_data.db"


def refresh_rosters(season: int = 2026) -> None:
    from data.mlb_stats_api import pull_all_rosters
    print("Pulling active rosters...")
    n = pull_all_rosters("active", season, DB_PATH)
    print(f"  active rosters: {n} rows upserted")
    print("Pulling 40-man rosters...")
    n = pull_all_rosters("40Man", season, DB_PATH)
    print(f"  40Man rosters: {n} rows upserted")


def refresh_schedule(days_ahead: int = 7, season: int = 2026) -> None:
    from data.mlb_stats_api import schedule
    today = date.today()
    start = today.isoformat()
    end = (today + timedelta(days=days_ahead)).isoformat()
    print(f"Pulling schedule {start} → {end}...")
    games = schedule(start, end, db_path=DB_PATH)
    print(f"  {len(games)} games upserted")


def refresh_news() -> None:
    from data.news_fetcher import fetch_and_store
    print("Fetching MLB news...")
    n = fetch_and_store(DB_PATH)
    print(f"  {n} new news items stored")


def run(season: int = 2026, skip_news: bool = False) -> None:
    refresh_schedule(days_ahead=7, season=season)
    refresh_rosters(season)
    if not skip_news:
        try:
            refresh_news()
        except Exception as e:
            print(f"  News fetch failed (non-fatal): {e}")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    run(season)
