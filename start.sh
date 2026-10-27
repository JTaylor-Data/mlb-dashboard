#!/bin/sh
# If no DB exists yet on the persistent disk, seed it with the stub.
if [ ! -f "$DB_PATH" ] && [ -f "deploy/mlb_stub.db" ]; then
    cp deploy/mlb_stub.db "$DB_PATH"
fi
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8010
