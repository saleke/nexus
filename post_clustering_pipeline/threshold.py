"""Dynamic global similarity threshold (with process-local TTL cache).

The threshold is read from the DB but refreshed at most once per TTL to avoid
a SELECT per batch; the daily control-plane autotuner (`run_threshold_autotune`
in quality.py) updates it at most once a day.
"""
from __future__ import annotations

import time

from .config import AUTO_ASSIGN_THRESHOLD, THRESHOLD_CACHE_TTL_SECONDS
from .db import extract_val

_THRESHOLD_SELECT = "SELECT value FROM system_config WHERE key = 'global_similarity_threshold';"

_cache: dict = {"value": None, "ts": 0.0}


def effective_threshold(dynamic_value, floor: float) -> float:
    """Resolve the operating threshold, bounded by the precision floor."""
    if dynamic_value is None:
        return float(floor)
    return max(float(dynamic_value), float(floor))


def get_effective_threshold(cur) -> float:
    """Fetch the cached global threshold, re-reading the DB on TTL expiry."""
    now = time.time()
    cached = _cache
    if cached["value"] is not None and now - cached["ts"] < THRESHOLD_CACHE_TTL_SECONDS:
        return cached["value"]

    cur.execute(_THRESHOLD_SELECT)
    row = cur.fetchone()
    value = extract_val(row, "value", 0) if row else None
    effective = effective_threshold(value, AUTO_ASSIGN_THRESHOLD)
    _cache["value"] = effective
    _cache["ts"] = now
    return effective