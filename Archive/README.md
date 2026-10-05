# Archive

This directory contains files that are NOT read by the live server at runtime.
Files were moved here by the 2026-10-05 codebase audit (Stage 5).

See `Archive/MOVED_FILES.csv` for a complete record of what moved where and why.

## Subfolders

| Folder | Contents |
|--------|---------|
| `models_retired/` | Old model versions no longer used as live scorers |
| `training_scripts/` | Training and scoring scripts for retired/old model versions |
| `diagnostics/` | Diagnostic outputs for models not currently in production |
| `methodology_old/` | Old methodology documents superseded by Methodology/ |
| `artifacts/` | Training artifacts (OOF predictions, feature parquets) not needed at runtime |
| `experimental/` | Experimental code, R&D scripts, one-off analyses |
| `misc_scripts/` | Utility scripts that are not part of the live pipeline |
| `unclassified/` | Files that don't fit other categories but are not live |

## Important Note

Files in this directory are NOT deleted. They are kept for reference and rollback.
If you need to restore a file, use `git mv Archive/<subfolder>/<file> <original_path>`.
