"""
Permanent guard: no runtime Python file in api/, analytics/, config/, tools/,
or scripts/ (if present) may contain the string 'Archive/' as a runtime path
reference.

Archive/ is dead-code storage; live server code must never import from it,
subprocess into it, or construct a path pointing into it.  Comments and
docstrings in non-runtime locations (tests, docs) are explicitly exempted —
only files in the scanned directories are checked.

Rationale: the scheduler previously referenced
Archive/misc_scripts/scripts/rescore_models.py; that script was moved to
tools/maintenance/ as part of the codebase-audit branch cleanup.  This test
prevents the same mistake from recurring silently.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories that contain live runtime code and must never reference Archive/
SCAN_DIRS = ["api", "analytics", "config", "tools", "scripts"]

# String that must not appear as a runtime path reference in any .py file
FORBIDDEN = "Archive/"


def _collect_files():
    files = []
    for dir_name in SCAN_DIRS:
        d = ROOT / dir_name
        if d.is_dir():
            files.extend(d.rglob("*.py"))
    return files


def test_no_archive_path_in_runtime_code():
    """
    Every .py file in api/, analytics/, config/, tools/, scripts/ must be
    free of the string 'Archive/' outside comments.

    The check is intentionally simple: it looks for the literal string
    'Archive/' in each line after stripping inline comments (# …).  This
    catches the most dangerous form — a path string — without triggering
    false positives from bare comments explaining WHY something moved.
    """
    violations = []
    for path in _collect_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            # Strip inline comments before checking.
            stripped = line.split("#", 1)[0]
            if FORBIDDEN in stripped:
                violations.append(f"{path.relative_to(ROOT)}:{lineno}  {line.rstrip()}")

    assert not violations, (
        f"Found {len(violations)} runtime Archive/ reference(s) in live code:\n"
        + "\n".join(violations)
    )
