# Archive: Diagnostics

Diagnostic outputs from model development and validation runs.

`diagnostics/stuff_v4/` stays in the repo root (NOT here) because
`tools/build_validation_content.py` reads it at build time.

## Contents

| Subfolder | Contents | Reason archived |
|-----------|---------|-----------------|
| `closeout_blend/` | Final blend validation scripts | Not read at runtime |
| `deception_v2/` | Deception+ v2 training/validation | Not read at runtime (v2b not yet fully deployed) |
| `location_v3/` | REVERTED — kept at `diagnostics/location_v3/` (live read by location_report.py) | — |
| `stuff_memorization/` | Memorization diagnostic report | Not read at runtime |
