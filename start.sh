#!/bin/sh
# Production entry point. Environment variables:
#   PORT        - port to bind (default 8010)
#   HOST        - interface to bind (default 0.0.0.0 for hosted, 127.0.0.1 locally)
#   DB_PATH     - path to mlb_data.db (default: repo-relative mlb_data.db)
#   ADMIN_TOKEN - secret for POST /api/admin/* endpoints (leave unset for local dev)

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"

exec python -m uvicorn api.main:app --host "$HOST" --port "$PORT"
