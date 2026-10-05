"""
tools/maintenance/check_archive_hygiene.py

Lists files in live project folders that are not referenced by any live code,
config, scheduler job, build tool, or document. These are candidates for
moving to the folder's dead-code storage subdirectory.

Usage:
    python tools/maintenance/check_archive_hygiene.py

Output:
    A sorted list of unreferenced file paths, relative to the project root.
    These are CANDIDATES only — confirm each one before moving it.

Rules:
    - Only .py, .json, .txt (booster/BIP), .parquet (in diagnostics/) files are scanned.
    - Files in dead-code storage folders are ignored.
    - A file counts as "referenced" if its basename or relative path appears as
      a string in any .py, .md, .json, or .html file under api/, analytics/,
      config/, tools/, frontend/, tests/, or the root.
    - False positives are possible (a file referenced only by a dead script).
      This tool is advisory only.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent.parent

# Directories to scan for unreferenced files
LIVE_DIRS = ["api", "analytics", "config", "tools", "diagnostics", "models", "data"]

# Extensions to check
CHECK_EXTS = {".py", ".json", ".txt", ".parquet"}

# Directories to search for references (source files that might import/open things)
REFERENCE_DIRS = ["api", "analytics", "config", "tools", "frontend", "tests"]
REFERENCE_EXTS = {".py", ".md", ".json", ".html", ".js"}

# Build a set of all string tokens from reference files
def _build_reference_tokens():
    tokens = set()
    for d in REFERENCE_DIRS:
        p = ROOT / d
        if not p.is_dir():
            continue
        for f in p.rglob("*"):
            if f.suffix not in REFERENCE_EXTS:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                # Extract quoted strings and bare path-like tokens
                for m in re.findall(r'["\']([^"\']{4,})["\']', text):
                    tokens.add(m)
                # Also grab bare filenames (no quotes) in common patterns
                for m in re.findall(r'(?:open|read_json|read_parquet|Path)\s*\(["\']([^"\']+)["\']', text):
                    tokens.add(m)
            except OSError:
                pass
    return tokens


def _is_referenced(path: Path, tokens: set) -> bool:
    name = path.name
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    rel_no_ext = rel.rsplit(".", 1)[0]
    stem = path.stem
    return (
        name in tokens
        or rel in tokens
        or rel_no_ext in tokens
        or stem in tokens
        or any(name in t or rel in t for t in tokens)
    )


def main():
    tokens = _build_reference_tokens()
    candidates = []
    _dead = "Archive"
    for dir_name in LIVE_DIRS:
        d = ROOT / dir_name
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if _dead in f.parts:
                continue
            if not f.is_file():
                continue
            if f.suffix not in CHECK_EXTS:
                continue
            if not _is_referenced(f, tokens):
                candidates.append(f.relative_to(ROOT))

    if not candidates:
        print("No unreferenced files found.")
        return

    print(f"Unreferenced file candidates ({len(candidates)}):")
    for p in sorted(candidates):
        print(f"  {p}")


if __name__ == "__main__":
    main()
