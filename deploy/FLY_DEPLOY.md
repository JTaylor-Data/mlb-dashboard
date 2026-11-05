# Deploying to Fly.io

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the app image (Fly builds it from this file) |
| `fly.toml` | App config: region, volume mount, env, VM size, HTTP service |
| `start.sh` | Starts uvicorn on `0.0.0.0:8010` |

`fly.toml` contains no secrets. Set `ADMIN_TOKEN` with `fly secrets set`.

## One-time setup

1. Install flyctl and sign in: `fly auth login`
2. Set the app name in `fly.toml` (`app = "..."`) and create the app: `fly apps create <app-name>`
3. Create the volume that `fly.toml` mounts at `/data`:
   `fly volumes create mlb_data --region yyz --size 15`
4. Set the admin token as a secret (generate one locally; do not commit it):
   `fly secrets set ADMIN_TOKEN=<token>`
5. Deploy: `fly deploy`

## Loading the database

The app reads `/data/mlb_data.db`. Upload it to the volume through an SSH session:

```bash
fly ssh console
# then, on your machine, copy the file in with sftp or fly ssh sftp
```

Write to a temporary name first, then rename it on the volume so the app never sees a partial file.
Remove any stale `mlb_data.db-wal` / `mlb_data.db-shm` files and make the file owned by uid 1000.

## Operations

- Daily data pull runs at 04:00 America/Toronto (see `api/scheduler.py`).
- Logs: `fly logs`
- Machine stays running (`min_machines_running = 1`, `auto_stop_machines = false`).
