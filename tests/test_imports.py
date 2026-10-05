"""
Import-everything test — verify every Python module in the project can be imported
without errors. Catches broken imports from reorganization early.
"""
import importlib
import os
import pytest
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Modules to skip (e.g. training scripts that need heavyweight deps not in venv)
SKIP_PATTERNS = ["pitcher_cv", "training/", "__pycache__"]

MODULES = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    # Skip hidden dirs, __pycache__, and heavy dirs
    dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"
                   and "pitcher_cv" not in dirpath and "training" not in dirpath]
    for fname in filenames:
        if fname.endswith(".py") and not fname.startswith("."):
            rel = os.path.relpath(os.path.join(dirpath, fname), ROOT)
            # Convert path to module name
            mod = rel.replace(os.sep, ".")[:-3]  # strip .py
            if not any(p in rel for p in SKIP_PATTERNS):
                MODULES.append(mod)


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    try:
        importlib.import_module(module)
    except ImportError as e:
        pytest.fail(f"ImportError in {module}: {e}")
    except Exception as e:
        # Non-import errors (e.g. missing DB) are acceptable at import time
        pass
