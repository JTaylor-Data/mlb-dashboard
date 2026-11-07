#!/bin/sh
# If no DB exists yet on the persistent disk, seed it with the stub.
if [ ! -f "$DB_PATH" ] && [ -f "deploy/mlb_stub.db" ]; then
    cp deploy/mlb_stub.db "$DB_PATH"
fi
# glibc gives every request thread its own malloc arena, and pandas buffers freed
# in one arena are not reused by another, so resident memory ballooned to ~865 MB
# (OOM kill on the 1 GB machine) where the same work needs ~440 MB. Two arenas
# keep it flat.
export MALLOC_ARENA_MAX=2
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8010
