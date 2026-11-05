# Hosting Guide

## Platform: Fly.io

The app deploys to [Fly.io](https://fly.io) using `fly.toml` and the repo's `Dockerfile`.
Full steps are in `deploy/FLY_DEPLOY.md`.

- App config (region `yyz`, 1 shared CPU, 1 GB RAM, one volume at `/data`): `fly.toml`
- `DB_PATH=/data/mlb_data.db` and `SEASON_PHASE=POSTSEASON` are set in `fly.toml [env]`
- `ADMIN_TOKEN` is a Fly secret, never in `fly.toml`
- The machine stays running; the daily refresh runs at 04:00 America/Toronto

---

## How Updates Work After Hosting

`fly deploy` rebuilds and replaces the machine. The volume keeps `mlb_data.db` across deploys.
The 04:00 Toronto scheduler keeps data current, so nothing needs to be triggered by hand.

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DB_PATH` | Yes | `/data/mlb_data.db` | Path to the SQLite database on the Fly volume |
| `SEASON_PHASE` | Yes | `POSTSEASON` | `REGULAR_SEASON`, `POSTSEASON`, or `OFFSEASON` |
| `ADMIN_TOKEN` | Recommended | (no auth) | Secures admin data-pull and cache-refresh endpoints |

---

## Local Dev

```bash
python -m uvicorn api.main:app --port 8010
```

No env vars needed — defaults to `mlb_data.db` in the repo root with no admin auth.

---

## Files That Don't Go to the Host

`.gitignore` already excludes:
- `mlb_data.db` — lives on the Fly volume, not in git
- `scores/*.parquet`, `Archive/`, `diagnostics/` — not needed at runtime (see `.dockerignore`)
- `__pycache__/`, `*.pyc`

---

## Alternative: Railway

Railway is a similar Docker-friendly platform. It works with this codebase too — add a
persistent volume at `/data`, set `DB_PATH=/data/mlb_data.db`, and use `./start.sh` as the
start command. Configure through Railway's dashboard.
