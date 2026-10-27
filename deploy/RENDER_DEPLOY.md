# Deploying to Render

## Files added

| File | Purpose |
|---|---|
| `Dockerfile` | Tells Render how to build and run the app |
| `render.yaml` | Render service config (runtime, disk, env vars) |
| `deploy/mlb_stub.db` | Minimal SQLite stub for smoke-test deploys |

---

## Smoke test deploy (verify mechanics work before uploading the real DB)

**Goal:** confirm the app boots and responds to requests — data doesn't need to be real.

1. **Create a Render account** at render.com → New → Web Service → Connect your GitHub repo
2. Render auto-detects `render.yaml` and pre-fills most settings
3. Under **Disks**, the 10GB disk mounts at `/data` — for the smoke test, upload `deploy/mlb_stub.db` as `/data/mlb_data.db` via the Render shell after first deploy, or use the Render CLI:
   ```
   render ssh <service-id>
   # then inside the shell:
   cp /path/to/mlb_stub.db /data/mlb_data.db
   ```
4. Set `SEASON_PHASE=POSTSEASON` (already in render.yaml)
5. Deploy — the app should boot and `GET /api/meta` should return `{"latest_game_date": null, ...}`

---

## Production deploy (real database)

The real `mlb_data.db` is ~4.2 GB — too large to commit to git. Two options:

**Option A: Render Shell upload**
```bash
# From your local machine (Render CLI required: npm install -g @render-com/cli)
render ssh <service-id>
# In the shell, use curl or scp to copy the DB to /data/mlb_data.db
```

**Option B: rsync on first deploy**
```bash
rsync -avz --progress mlb_data.db <render-ssh-user>@<host>:/data/mlb_data.db
```

After copying, the app picks it up immediately on the next restart (or `/api/admin/refresh-cache` call).

---

## Environment variables (set in Render dashboard or render.yaml)

| Variable | Required | Notes |
|---|---|---|
| `DB_PATH` | Yes | `/data/mlb_data.db` — points to the persistent disk |
| `SEASON_PHASE` | Yes | `POSTSEASON`, `REGULAR_SEASON`, or `OFFSEASON` |
| `ADMIN_TOKEN` | Recommended | Render auto-generates one; copy it to secure admin endpoints |

---

## Notes

- **Python version**: Dockerfile uses 3.11 (Render's 3.14 support is still limited; all packages support 3.11)
- **Scheduler**: The app starts a background scheduler on boot that fires a data pull. On the first smoke-test deploy with the stub DB this will fail silently (the pull will find no new data and log a warning). That's fine — it won't crash the server.
- **Disk size**: The `render.yaml` requests a 10GB disk. The real DB is ~4.2 GB; keep headroom for WAL files. Upgrade to 20GB if you plan to keep multiple seasons of raw pitch data.
- **Plan**: `starter` plan suspends after 15 min of inactivity (free tier behavior). Upgrade to `standard` ($7/mo) for always-on.
