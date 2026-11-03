"""
Shared helper for reading a model's pre-scored parquet file directly,
on demand, instead of requiring api/warmup.py to have eagerly injected it
into the model's in-process cache first.

Why this exists: warmup.py used to eagerly load Stuff+/Tunneling+/
Location+ raw predictions for the current season AND every prior season
(2021-current) into memory at boot - about 230MB retained for the rest of
the process's life, every single boot, even for seasons nobody looks at
that session. Each model's own _raw_predictions() only ever checked its
in-process cache and fell back to live LightGBM scoring on a miss - it had
no way to read the parquet itself. This function gives it that fallback,
so a cold cache for ANY season (not just ones warmup pre-injected) is
still fast - read a file, not re-run the model - and past seasons can be
loaded lazily, on whichever request actually needs them, instead of all
at once at boot.
"""
import logging
from pathlib import Path

log = logging.getLogger("score_parquet_cache")

SCORES_DIR = Path(__file__).resolve().parent.parent / "parquets"


def try_load_parquet(filename: str):
    """Returns a DataFrame if `parquets/<filename>` exists and loads
    cleanly, else None (missing file, or any read error - never raises,
    callers should treat None exactly like a cache miss)."""
    path = SCORES_DIR / filename
    if not path.exists():
        return None
    try:
        import pandas as pd
        return pd.read_parquet(path)
    except Exception as e:
        log.warning("parquet load failed for %s: %s", filename, e)
        return None
