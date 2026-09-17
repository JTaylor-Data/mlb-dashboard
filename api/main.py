"""
FastAPI app: mounts the API routers plus the static frontend. Run from the
mlb_dashboard/ directory with:

    python -m uvicorn api.main:app --port 8010

(cwd on sys.path is what lets api/routers/*.py do `from analytics.metrics import ...`
etc. against the analytics/ package, same as the Streamlit pages did.)
"""
import sys
from pathlib import Path

import os

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

# Set ADMIN_TOKEN env var in production to secure the admin endpoints.
# Unset (or empty) means no auth required — fine for local dev, not for hosting.
_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()


def _require_admin(x_admin_token: str = Header(default="")):
    if _ADMIN_TOKEN and x_admin_token != _ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics import boxscore as _boxscore
from api import cache, pull, scheduler
from api.db import get_conn
from api.routers import (
    analysis, career, deception_report, heatmaps, hitzones, location_report, players, playoffs,
    prospects, similar_players, splits, standings, stuff_model_report, team_card, top_performers,
    trends, tunneling_report, tunneling_3d, teams, probable_pitchers, news,
)
from api.warmup import warm_cache_background

app = FastAPI(title="MLB Dashboard API")
# Compress API JSON responses ≥1KB — leaderboard/player payloads can be
# 100KB+; gzip typically cuts that 5-10x on the wire.
app.add_middleware(GZipMiddleware, minimum_size=1000)


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
def refresh_cache(_=Depends(_require_admin)):
    """Cache-only refresh - use when you've re-run the pull scripts yourself
    and just want the app to pick up what's already on disk."""
    cache.clear_all()
    _boxscore.clear_cache()
    warm_cache_background()
    return {"status": "ok"}


@admin_router.post("/admin/pull-data")
def pull_data(_=Depends(_require_admin)):
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
app.include_router(tunneling_3d.router)
app.include_router(deception_report.router)
app.include_router(location_report.router)
app.include_router(team_card.router)
app.include_router(heatmaps.router)
app.include_router(hitzones.router)
app.include_router(similar_players.router)
app.include_router(playoffs.router)
app.include_router(teams.router)
app.include_router(probable_pitchers.router)
app.include_router(news.router)
app.include_router(trends.router)
app.include_router(career.router)


@app.middleware("http")
async def no_cache_static(request, call_next):
    # HTML pages must never be served stale — strip conditional headers so the
    # browser can't 304 itself back to an old page structure. Versioned static
    # assets (JS/CSS with ?v=N query params) are safe to cache in the browser;
    # stripping their conditional headers just wastes bandwidth on a forced
    # re-download every time.
    path = request.url.path
    is_html = (
        path == "/"
        or path.endswith(".html")
        or (not path.startswith("/api/") and "." not in path.split("/")[-1])
    )
    # Stadium photos are replaced in place (same URL, new content) so the
    # browser must always revalidate them, not serve from cache.
    is_stadium_photo = path.startswith("/img/stadiums/")
    # JS/CSS served with a ?v=N version query param are fingerprinted — safe to
    # cache aggressively. Without the version param, fall back to no-store so
    # unversioned assets never go stale.
    is_versioned_asset = (
        (path.endswith(".js") or path.endswith(".css"))
        and bool(request.query_params.get("v"))
    )
    if is_html or is_stadium_photo:
        request.scope["headers"] = [
            (k, v) for k, v in request.scope["headers"]
            if k not in (b"if-none-match", b"if-modified-since")
        ]
    response = await call_next(request)
    if is_html or is_stadium_photo:
        response.headers["Cache-Control"] = "no-store"
    elif is_versioned_asset:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
