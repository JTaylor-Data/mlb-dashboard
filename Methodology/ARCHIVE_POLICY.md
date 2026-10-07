# Archive Policy

> Established on branch `archive-and-chart`, 2026-10-05.

---

## What "Important (stays)" Means

A file is **Important (stays in its current location)** if at least one of the following is true:

1. **Reachable from the live manifest** — reachable via the static import graph from `api/main.py`,
   from the runtime trace of a full page/route crawl, from `api/scheduler.py`, or from
   `tools/build_validation_content.py` (the content builder).
2. **Belongs to the active model version** — booster files, score parquets, or diagnostics that
   correspond to the currently-active model versions (Stuff+ v4, Location+ v3, Tunneling+ v3,
   Deception+ v2b, as listed in `config/model_policy.py`).
3. **Read by a live route, page, or builder** — any file opened via `open()`, `read_json()`,
   `read_parquet()`, or `Path(...)` in `api/`, `analytics/`, `tools/`, or `config/`.
   **Note:** an earlier reorganization broke a route that read
   `diagnostics/location_v3/*.json`; anything read by a route must stay where it is.
4. **A documented tool the owner runs** — any script in `tools/` with a docstring explaining
   when and why to run it.
5. **Listed KEEP with justification** in the folder's README.

Everything else should move to that folder's `Archive/` subdirectory.
**When unsure, keep the file in place** and list it as "unproven" in the README.

---

## Archive Conventions

- **Per-folder Archive/**: each project folder (`diagnostics/`, `analytics/`, `api/`, etc.)
  gets its own `Archive/` subdirectory, created at the top level of that folder.
  A second-level `Archive/` (e.g. `diagnostics/<model>/Archive/`) is created only where
  that subfolder mixes live and dead files.
- **Whole-dead-folder rule**: a folder whose *entire* contents are dead moves whole into its
  parent's `Archive/`. No Archive nested inside an Archive.
- **Root Archive/**: `Archive/` at the project root keeps only root-level files and entire
  dead folders (video pipeline, 3D viewer, etc.), each with its own `README.md` explaining
  how to restore.
- **git mv always**: files are moved with `git mv`, never deleted. History is preserved.
- **Large gitignored data stays**: databases, score parquets, and cache files that are
  gitignored must not be moved (they are not tracked and moving them would break paths).
- **Superseded files**: when a file is superseded by a new version, move it to the folder's
  `Archive/` in the same commit that introduces the new version.

---

## Archive Directory Layout

```
<folder>/
├── Archive/           # Dead files for this folder
│   └── README.md      # What is here and why; how to restore if needed
└── README.md          # 3 lines: what the folder is, what is current, what is in Archive/
```

Root:
```
Archive/               # Root Archive: entire dead subtrees and root-level dead files
├── experimental/      # Experimental pipelines (pitcher_cv, Tunneling+ 3D viewer)
│   └── pitcher_cv/    # Video pipeline; README explains how to restore
├── training_scripts/  # Old training artifacts (parquets, checkpoints)
├── diagnostics/       # Old per-model diagnostics (superseded model versions)
├── artifacts/         # Project artifacts (old training runs)
└── MOVED_FILES.csv    # Record of all moves: old_path, new_path, reason, evidence
```

---

## MOVED_FILES.csv Format

Every `git mv` that moves a file into or between Archive/ directories must have a row in
`Archive/MOVED_FILES.csv`:

| Column | Description |
|---|---|
| `old_path` | Path relative to project root, before move |
| `new_path` | Path relative to project root, after move |
| `reason` | One of: superseded, dead-code, training-artifact, experimental |
| `evidence` | What confirms it is dead (e.g. "no import in api/, analytics/") |

---

## What Goes Where

| Category | Live location | Dead goes to |
|---|---|---|
| Active model diagnostics | `diagnostics/<model>/` with subfolders `reports/`, `artifacts/`, `scripts/`, `logs/` | `diagnostics/Archive/` |
| Superseded model diagnostics | — | `diagnostics/Archive/<model>/` |
| Active boosters | `models/` | — |
| Superseded boosters | `models/Archive/` | — |
| Active training scripts | `training/` (if it exists) | — |
| Old training scripts | `Archive/training_scripts/` | (already there) |
| Experimental pipelines | `Archive/experimental/` | (already there) |
| Analytics modules for active models | `analytics/` | — |
| Superseded analytics modules | `analytics/Archive/` | — |
| Active tools | `tools/` | — |
| Retired tools | `tools/Archive/` | — |

---

## Guard: No Live Code May Reference Archive/

`tests/test_no_archive_refs.py` enforces that no `.py` file in `api/`, `analytics/`,
`config/`, `tools/`, or `scripts/` contains the string `Archive/` outside a comment.
This test must pass at every commit.

In addition, `tools/maintenance/check_archive_hygiene.py` (see Part 4.6) lists files in
live folders that nothing references — candidates for Archive. Run it before committing
any archive move.

---

## Gitignore Step (deferred)

When the Archive/ folders have stabilized:

1. Back up Archive/ contents outside the repo (or to an external drive / object storage).
2. `git rm -r --cached Archive/ <folder>/Archive/` (for each Archive subfolder).
3. Add `**/Archive/` to `.gitignore`.
4. Commit and push.

This keeps full history in git objects but stops the Archive folders from bloating future clones.
**Do not do this step on the `archive-and-chart` branch.**
