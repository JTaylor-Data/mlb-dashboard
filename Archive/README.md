# Archive/

Dead-code storage for the MLB Dashboard project. Files here are kept for history and reference only.
Nothing under Archive/ is imported or read by any live server code.

## Contents

- `artifacts/` — Old training run artifacts (parquets, checkpoints from model training)
- `diagnostics/` — Superseded model diagnostics (deception_v2, closeout_blend)
- `experimental/` — Experimental pipelines not part of the live dashboard:
  - `pitcher_cv/` — Computer-vision rubber-position and deception pipeline (video-based; superseded by Statcast-based approach). See `pitcher_cv/README.md` to restore.
- `methodology_old/` — Old methodology documents
- `misc_scripts/` — Retired utility scripts
- `models_retired/` — Retired model versions
- `training_scripts/` — Historical training scripts and training artifacts
- `unclassified/` — Files not yet classified

## How to Restore a File

All moves were done with `git mv`; history is fully preserved.
To restore: `git mv Archive/<path> <original_path>` and commit.
See `Archive/MOVED_FILES.csv` for the record of every move.

## Future: Gitignore Step

See `Methodology/ARCHIVE_POLICY.md` — "Gitignore Step (deferred)" for the process
to back up and stop tracking Archive/ contents.
