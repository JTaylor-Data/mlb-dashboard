"""
FastAPI app: mounts the API routers plus the static frontend. Run from the
mlb_dashboard/ directory with:

    python -m uvicorn api.main:app --port 8000

(cwd on sys.path is what lets api/routers/*.py do `from metrics import ...`
etc. against the existing analytics modules, same as the Streamlit pages did.)
"""
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import cache, pull, scheduler
from api.db import get_conn
from api.routers import (
    analysis, deception_report, players, prospects, splits, standings, stuff_model_report, top_performers,
    tunneling_report,
)
from api.warmup import warm_cache_background

app = FastAPI(title="MLB Dashboard API")


@app.on_event("startup")
def _refresh_on_startup():
    # Pulls fresh data AND warms the cache (not just the cache-only warm
    # this used to do) - see api/scheduler.py. Also arms the daily
    # 6 AM auto-refresh, so this is the only automatic trigger needed;
    # the nav's "Refresh data" button is now a manual fallback on top of it.
    scheduler.start()


admin_router = APIRouter(prefix="/api")


@admin_router.get("/meta")
def get_meta(conn=Depends(get_conn)):
    row = conn.execute("SELECT MAX(game_date) FROM games").fetchone()
    seasons = [r[0] for r in conn.execute("SELECT DISTINCT season FROM pitches ORDER BY season DESC")]
    return {"latest_game_date": row[0] if row else None, "available_seasons": seasons}


@admin_router.post("/admin/refresh-cache")
def refresh_cache():
    """Cache-only refresh - use when you've re-run the pull scripts yourself
    and just want the app to pick up what's already on disk."""
    cache.clear_all()
    warm_cache_background()
    return {"status": "ok"}


@admin_router.post("/admin/pull-data")
def pull_data():
    """The nav's "Refresh data" button - actually fetches new Statcast/
    schedule data (not just cache invalidation), then re-warms the cache.
    Runs in the background; poll /api/admin/pull-status for progress."""
    started = pull.start_pull()
    return {"started": started, "status": pull.get_status()}


@admin_router.get("/admin/pull-status")
def pull_status():
    return pull.get_status()


app.include_router(admin_router)
app.include_router(players.router)
app.include_router(standings.router)
app.include_router(top_performers.router)
app.include_router(prospects.router)
app.include_router(splits.router)
app.include_router(analysis.router)
app.include_router(stuff_model_report.router)
app.include_router(tunneling_report.router)
app.include_router(deception_report.router)


@app.middleware("http")
async def no_cache_static(request, call_next):
    # This app is under active local iteration. Setting Cache-Control on the
    # way out isn't enough by itself: StaticFiles honors the browser's
    # If-None-Match/If-Modified-Since conditional headers and answers with a
    # bodyless 304, so a stale cached body keeps getting reused regardless
    # of what header rides along with the 304. Strip the conditional
    # headers off the incoming request so every response is a full 200 with
    # current content, then also mark it non-cacheable for next time.
    request.scope["headers"] = [
        (k, v) for k, v in request.scope["headers"]
        if k not in (b"if-none-match", b"if-modified-since")
    ]
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
