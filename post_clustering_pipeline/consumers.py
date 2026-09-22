"""Per-consumer API authentication (feedback + admin routes).

Complements the legacy shared bearer token: consumer tokens are opaque,
hashed at rest, rate-limited in-process, and revocable. Every feedback /
admin action is attributed to a consumer/actor so a noisy or hostile client
can be identified and severed without touching the shared token.
"""
from __future__ import annotations

import hashlib
import time

from .db import get_db_cursor, extract_val

# In-process token buckets per consumer for the control-plane routes. A
# process-local guard is enough: burst protection + a blocking hammer; not a
# billing-grade limiter. Keyed by consumer_id.
_buckets: dict[str, tuple[float, int]] = {}

# Minimum interval between committed last_used_at touches (see
# authenticate_consumer): a minute of granularity is ample for the "who's
# active" panel, and it keeps authenticate off the synchronous-commit path.
LAST_USED_THROTTLE_SECONDS = 60.0
_last_touch: dict[str, float] = {}


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def list_consumers() -> list[dict]:
    """All consumer rows for the settings integration card."""
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            "SELECT consumer_id, name, is_active, rate_limit_per_minute, last_used_at "
            "FROM api_consumers ORDER BY id;"
        )
        rows = cur.fetchall() or []
    return [
        {
            "consumer_id": extract_val(r, "consumer_id", 0),
            "name": extract_val(r, "name", 1),
            "is_active": bool(extract_val(r, "is_active", 2)),
            "rate_limit_per_minute": extract_val(r, "rate_limit_per_minute", 3),
            "last_used_at": extract_val(r, "last_used_at", 4),
        }
        for r in rows
    ]


def authenticate_consumer(token: str) -> dict | None:
    """Resolve a consumer token to its identity, or None.

    Also touches last_used_at so the admin panel can show who's active. The
    touch is throttled to one committed write per consumer per minute: this
    runs on EVERY authenticated request, so an unthrottled UPDATE would add a
    synchronous fsync to the hot path.
    """
    if not token:
        return None
    digest = hash_token(token)
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "SELECT consumer_id, name, is_active, rate_limit_per_minute, last_used_at FROM api_consumers WHERE token_hash = %s",
            (digest,)
        )
        row = cur.fetchone()
        if not row or not extract_val(row, "is_active", 2):
            return None
        now = time.time()
        last_touch = _last_touch.get(extract_val(row, "consumer_id", 0), 0.0)
        if now - last_touch >= LAST_USED_THROTTLE_SECONDS:
            cur.execute("UPDATE api_consumers SET last_used_at = NOW() WHERE consumer_id = %s;", (extract_val(row, "consumer_id", 0),))
            _last_touch[extract_val(row, "consumer_id", 0)] = now
        return {
            "consumer_id": extract_val(row, "consumer_id", 0),
            "name": extract_val(row, "name", 1),
            "rate_limit_per_minute": int(extract_val(row, "rate_limit_per_minute", 3) or 60),
        }


def rate_limit_check(consumer_id: str, rate_limit_per_minute: int) -> tuple[bool, int]:
    """Sliding-window allowance check. Returns ``(allowed, remaining)``."""
    now = time.time()
    window = 60.0
    if rate_limit_per_minute <= 0:
        return False, 0
    last, used = _buckets.get(consumer_id, (0.0, 0))
    if now - last >= window:
        last, used = now, 0
    remaining = rate_limit_per_minute - used
    allowed = remaining > 0
    _buckets[consumer_id] = (last, used + 1 if allowed else used)
    return allowed, max(remaining - 1 if allowed else remaining, 0)


def delete_consumer(consumer_id: str) -> bool:
    """Permanently remove a consumer and its token. Returns False if unknown."""
    _buckets.pop(consumer_id, None)
    with get_db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM api_consumers WHERE consumer_id = %s RETURNING id;", (consumer_id,))
        return cur.fetchone() is not None