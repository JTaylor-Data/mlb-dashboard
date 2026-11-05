"""
FastAPI app: mounts the API routers plus the static frontend. Run from the
mlb_dashboard/ directory with:

    python -m uvicorn api.main:app --port 8010

(cwd on sys.path is what lets api/routers/*.py do `from analytics.metrics import ...`
etc. against the analytics/ package, same as the Streamlit pages did.)
"""
import logging
import sys
from pathlib import Path

import os

# Without this, log.info()/log.warning() calls across the app (pull.py,
# scheduler.py, warmup.py, etc.) never reach stdout - only bare log.error
# text makes it out, via Python's unconfigured-logging fallback, with no
# timestamp or logger name. This makes every level visible in the Fly log
# viewer.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics import boxscore as _boxscore
from api import cache, pull, scheduler
from api.auth import require_admin as _require_admin
from api.db import get_conn
from api.routers import (
    analysis, career, deception_report, heatmaps, hitzones, location_report, players, playoffs,
    prospects, similar_players, splits, standings, stuff_model_report, team_card, top_performers,
    trends, tunneling_report, tunneling_3d, probable_pitchers, news,
)
from api.warmup import warm_cache_background

# API docs (/docs, /redoc, /openapi.json) are off unless ENABLE_API_DOCS=1,
# so production doesn't publish the full route map.
_DOCS = os.environ.get("ENABLE_API_DOCS", "") == "1"
app = FastAPI(
    title="MLB Dashboard API",
    docs_url="/docs" if _DOCS else None,
    redoc_url="/redoc" if _DOCS else None,
    openapi_url="/openapi.json" if _DOCS else None,
)
# Compress API JSON responses ≥1KB — leaderboard/player payloads can be
# 100KB+; gzip typically cuts that 5-10x on the wire.
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _client_ip(request: Request) -> str:
    """Real per-visitor IP for rate limiting.

    The hosting platform's edge proxy is the only way any request reaches
    this process - there's no direct public path to the container - so
    request.client.host here is always that proxy, not the visitor,
    and slowapi's default get_remote_address would key every visitor on
    one shared IP (rate-limiting everyone together instead of per-visitor).
    The proxy sets X-Forwarded-For itself, with the real client IP as the
    first entry, so trusting its leftmost value is safe in this
    deployment (verify on each new platform); it would NOT be safe if this app were ever reachable by
    any other route in (e.g. direct container access), since a client
    could then forge that header.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


# Per-IP limit on everything. Admin routes get a tighter one below.
limiter = Limiter(key_func=_client_ip, default_limits=["240/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


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
    sched_row = conn.execute("SELECT MAX(game_date) FROM schedule_games WHERE game_date <= date('now')").fetchone()
    seasons = [r[0] for r in conn.execute("SELECT DISTINCT season FROM pitches ORDER BY season DESC")]
    return {
        "latest_game_date": row[0] if row else None,
        "latest_schedule_date": sched_row[0] if sched_row else None,
        "available_seasons": seasons,
    }


@admin_router.post("/admin/refresh-cache")
@limiter.limit("10/minute")
def refresh_cache(request: Request, _=Depends(_require_admin)):
    """Cache-only refresh - use when you've re-run the pull scripts yourself
    and just want the app to pick up what's already on disk."""
    cache.clear_all()
    _boxscore.clear_cache()
    warm_cache_background()
    return {"status": "ok"}


@admin_router.post("/admin/pull-data")
@limiter.limit("10/minute")
def pull_data(request: Request, _=Depends(_require_admin)):
    """The nav's "Refresh data" button - actually fetches new Statcast/
    schedule data (not just cache invalidation), then re-warms the cache.
    Runs in the background; poll /api/admin/pull-status for progress."""
    started = pull.start_pull()
    return {"started": started, "status": pull.get_status()}


@admin_router.get("/admin/pull-status")
@limiter.limit("10/minute")
def pull_status(request: Request, _=Depends(_require_admin)):
    return pull.get_status()


@admin_router.get("/admin/cache-stats")
@limiter.limit("10/minute")
def cache_stats(request: Request, _=Depends(_require_admin)):
    """Internal: report cache entry counts per registered function.
    Disabled externally by ADMIN_TOKEN when set (same as other admin endpoints)."""
    import time as _time
    stats = []
    for fn in cache._registry:
        store = fn.__closure__[0].cell_contents if fn.__closure__ else {}
        if not isinstance(store, dict):
            store = {}
        now = _time.monotonic()
        live = sum(1 for _, exp in store.values() if now < exp)
        stats.append({
            "function": fn.__name__,
            "total_entries": len(store),
            "live_entries": live,
            "expired_entries": len(store) - live,
        })
    return {"caches": stats, "total_registered": len(cache._registry)}


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
app.include_router(probable_pitchers.router)
app.include_router(news.router)
app.include_router(trends.router)
app.include_router(career.router)


# Report-only for now: several pages still have inline <script> blocks and
# JS-generated inline style="" attributes, so script-src/style-src need
# 'unsafe-inline' to avoid blocking those outright. This header only logs
# violations to the browser console — see security-report/csp-violations.txt
# for what it caught. Enforcement is a separate follow-up.
CSP_REPORT_ONLY_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: https://img.mlbstatic.com https://www.mlbstatic.com; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Content-Security-Policy-Report-Only", CSP_REPORT_ONLY_POLICY)
    # The platform terminates TLS in front of this app, so every request this
    # process actually sees has already gone over HTTPS - safe to tell
    # browsers to always use HTTPS for this host going forward.
    response.headers.setdefault(
        "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
    )
    return response


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
