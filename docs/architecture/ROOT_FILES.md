# Root Files

Files at the project root. Moved to `docs/` 2026-10-27.

## Build / deploy (stay at root — required by Fly and Docker)

| File | Purpose |
|---|---|
| `Dockerfile` | Docker image definition; read by the Fly build |
| `fly.toml` | Fly app config (region, volume mount, env, VM size) |
| `requirements.txt` | Python dependency list |
| `start.sh` | Container entrypoint; seeds stub DB on cold boot |
| `schema.sql` | Database DDL reference; used when re-creating DB from scratch |
| `pytest.ini` | pytest config; restricts test collection to `tests/` |

## Dotfiles

| File | Purpose |
|---|---|
| `.gitignore` | Controls what git tracks |
| `.dockerignore` | Excludes files from Docker build context |
| `.gitattributes` | Git LFS and diff attributes for binary files |

## Documentation (moved to `docs/`)

See `docs/` for ARCHITECTURE.md, HOSTING.md, PROMOTION_CHECKLIST.md,
PROMOTION_LOG.md, and VERIFY_BOTH.md.
