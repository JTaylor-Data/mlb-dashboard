# Hosting Guide

## Recommended Platform: Railway

Railway is the best fit for this stack. It supports:
- Python/FastAPI natively (no Docker required)
- **Persistent volumes** — critical for SQLite (`mlb_data.db` is ~100MB+ and must survive redeploys)
- Free tier is enough for a portfolio project with low traffic
- Git-push-to-deploy once connected to GitHub

Alternatives in order of effort:
- **Render** — similar to Railway; persistent disks cost ~$1/month extra
- **Fly.io** — more control, requires a `fly.toml` config and Docker familiarity
- **DigitalOcean App Platform** — simple, but pricier for persistent storage

---

## One-time Setup (Railway)

1. Push this repo to GitHub (already done — remote is set)

2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo → select this repo

3. Add a **Volume** in Railway:
   - Mount path: `/data`
   - Then set environment variable: `DB_PATH=/data/mlb_data.db`

4. Set environment variables in Railway's dashboard:
   ```
   DB_PATH=/data/mlb_data.db
   ADMIN_TOKEN=<pick a long random string — keep it secret>
   PORT=8010
   HOST=0.0.0.0
   ```

5. Set the **Start Command** in Railway:
   ```
   sh start.sh
   ```
   Or equivalently:
   ```
   python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```

6. **First deploy — seed the database**: The app's startup pull will try to fetch data, but the volume starts empty so `mlb_data.db` won't exist yet. Options:
   - Let it run — the startup scheduler will pull everything from scratch via `pybaseball` (slow, ~30-60 min first time, but fully automated)
   - Or: upload your local `mlb_data.db` to the volume via Railway's SSH/CLI to skip the bootstrap

---

## How Updates Work After Hosting

Push to GitHub → Railway auto-redeploys. The persistent volume keeps `mlb_data.db` untouched across deploys.

```
# make changes locally, test them, then:
git add -p          # stage only what you want
git commit -m "..."
git push            # Railway picks this up automatically
```

The daily 6 AM scheduler keeps the data current — you don't need to trigger anything manually.

To update data right now from a browser: click the "Refresh data" button in the nav. With `ADMIN_TOKEN` set, the nav's refresh button won't work unless it sends the token — see the note below.

---

## Admin Token + the Refresh Button

When `ADMIN_TOKEN` is set, `POST /api/admin/pull-data` and `POST /api/admin/refresh-cache` require an `X-Admin-Token` header. The frontend's "Refresh data" button currently calls these without any header, so **set the token and the button will stop working for visitors** (returns 401 — non-fatal, just won't refresh).

To keep the button working for yourself after hosting, either:
- Don't set `ADMIN_TOKEN` if the site is just a portfolio demo with no sensitive data-pull concerns (the endpoints just kick off a pybaseball fetch — worst case a visitor slows the server for a minute)
- Or add the token to the nav's fetch call in `frontend/js/nav.js` (search for `refresh-cache` / `pull-data`) and keep the token client-side (fine for a personal portfolio, not for production apps)

---

## Files That Don't Go to the Host

`.gitignore` already excludes the big ones:
- `mlb_data.db` — lives on the persistent volume, not in git
- `training/` artifacts, `project_artifacts/`, `pitcher_cv/` — R&D files, not needed at runtime
- `__pycache__/`, `*.pyc`

The `.streamlit/` directory can be deleted — it's a leftover from the old Streamlit prototype and does nothing in the FastAPI app.

---

## Environment Variables Summary

| Variable | Required in prod? | Default | Purpose |
|---|---|---|---|
| `DB_PATH` | Yes | `mlb_data.db` (relative) | Path to the SQLite database |
| `ADMIN_TOKEN` | Recommended | (no auth) | Secures admin data-pull endpoints |
| `PORT` | Set by platform | `8010` | Uvicorn bind port |
| `HOST` | Set by platform | `0.0.0.0` | Uvicorn bind interface |

---

## Local Dev (unchanged)

```
python -m uvicorn api.main:app --port 8010
```

No env vars needed — defaults to `mlb_data.db` in the repo root and no admin auth.
