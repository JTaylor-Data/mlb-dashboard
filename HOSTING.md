# Hosting Guide

## Platform: Render

This app is configured for [Render](https://render.com). The repo includes a `Dockerfile` and
`render.yaml` that Render uses directly — no manual build config needed.

Why Render:
- Docker-native (uses the `Dockerfile` in this repo)
- **Persistent disks** — required for SQLite (`mlb_data.db` is ~4 GB and must survive redeploys)
- `render.yaml` auto-wires the service config (disk mount, env vars, start command)

---

## One-time Setup

1. Push this repo to GitHub (already done — remote is set)

2. Go to [render.com](https://render.com) → New → Web Service → Connect your GitHub repo

3. Render auto-detects `render.yaml` and pre-fills:
   - Build: Docker (uses the `Dockerfile`)
   - Disk: 10 GB, mounted at `/data`
   - `DB_PATH=/data/mlb_data.db`
   - `SEASON_PHASE=POSTSEASON`

4. Set `ADMIN_TOKEN` in the Render dashboard (Render can generate one) — secures the
   `/api/admin/pull-data` and `/api/admin/refresh-cache` endpoints.

5. Deploy. The app boots immediately; the startup scheduler runs a data pull in the background.

---

## First Deploy — Seeding the Database

`mlb_data.db` (~4 GB) is not in git. Two options:

**Option A — Smoke test with the stub DB** (verify mechanics first)

The `start.sh` entrypoint auto-copies `deploy/mlb_stub.db` to `/data/mlb_data.db` if the disk
is empty. Deploy as-is and `GET /api/meta` should respond — data will be minimal/empty but the
app boots cleanly.

**Option B — Upload the real DB before first use**

```bash
# Render CLI: npm install -g @render-com/cli
render ssh <service-id>
# in the shell:
rsync -avz mlb_data.db <render-ssh-user>@<host>:/data/mlb_data.db
```

After copying, restart the service — it picks up the DB on next boot.

---

## How Updates Work After Hosting

Push to GitHub → Render auto-redeploys. The persistent disk keeps `mlb_data.db` untouched
across deploys. The daily 6 AM scheduler keeps data current — nothing to trigger manually.

```bash
git add -p
git commit -m "..."
git push   # Render picks this up automatically
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DB_PATH` | Yes | `/data/mlb_data.db` | Path to the SQLite database on the persistent disk |
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
- `mlb_data.db` — lives on the persistent disk, not in git
- `scores/*.parquet`, `Archive/`, `diagnostics/` — not needed at runtime (see `.dockerignore`)
- `__pycache__/`, `*.pyc`

---

## Alternative: Railway

Railway is a similar Docker-friendly platform. It works with this codebase too — add a
persistent volume at `/data`, set `DB_PATH=/data/mlb_data.db`, and use `./start.sh` as the
start command. No `render.yaml` equivalent; configure through Railway's dashboard.
