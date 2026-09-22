"""Best-effort sliding-window rate limiting, Redis-first with a local fallback.

Counters live in Redis under ``nexus:rl:{scope}:{key}`` so a limit is shared by
every uvicorn worker/instance (a 4-worker API would otherwise give an attacker
four independent counters). Redis here is an accelerator, not a trust boundary:
if Redis is unreachable we degrade to a per-process counter so authentication
is never bricked by a broker outage. State is best-effort and self-healing - a
lost or uneven counter only means slightly more attempts allowed for one
window, which is the correct direction to fail for a guardrail.
"""

import time

from .queues import get_redis_client

_LOCAL_BUCKETS: dict[str, list[float]] = {}

# Atomic INCR + TTL guard. The previous INCR-then-EXPIRE pair was not atomic:
# if the EXPIRE was lost (Redis error/restart between the two commands) the
# counter key persisted with no TTL and never reset, permanently locking the
# account out. Running both in one Lua script makes the increment and the TTL
# inseparable; the TTL check also repairs any legacy key that somehow lacks
# one, while preserving the original "window starts at first failure"
# semantics (EXPIRE is not re-extended on every hit).
_INCR_WINDOW_LUA = """
local c = redis.call('INCR', KEYS[1])
if c == 1 or redis.call('TTL', KEYS[1]) < 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return c
"""


def _key(scope: str, key: str) -> str:
    return f"nexus:rl:{scope}:{key}"


def _prune_local(rkey: str, window: int) -> None:
    bucket = _LOCAL_BUCKETS.get(rkey)
    if not bucket:
        return
    now = time.time()
    kept = [t for t in bucket if now - t < window]
    if kept:
        _LOCAL_BUCKETS[rkey] = kept
    else:
        _LOCAL_BUCKETS.pop(rkey, None)


def failures(scope: str, key: str, window: int) -> int:
    """Number of failures recorded in the current sliding window."""
    rkey = _key(scope, key)
    try:
        value = get_redis_client().get(rkey)
        return max(0, int(value)) if value is not None else 0
    except Exception:
        _prune_local(rkey, window)
        return len(_LOCAL_BUCKETS.get(rkey, []))


def note_failure(scope: str, key: str, window: int) -> int:
    """Record one failure; returns the new count for the current window."""
    rkey = _key(scope, key)
    try:
        r = get_redis_client()
        count = r.eval(_INCR_WINDOW_LUA, 1, rkey, window)
        return int(count)
    except Exception:
        now = time.time()
        bucket = [t for t in _LOCAL_BUCKETS.get(rkey, []) if now - t < window]
        bucket.append(now)
        _LOCAL_BUCKETS[rkey] = bucket
        return len(bucket)


def blocked(scope: str, key: str, limit: int, window: int) -> bool:
    """True once the counter is at or above ``limit`` in the window."""
    return failures(scope, key, window) >= limit


def reset(scope: str, key: str) -> None:
    """Clear the counter (e.g. after a successful login)."""
    rkey = _key(scope, key)
    try:
        get_redis_client().delete(rkey)
    except Exception:
        pass
    _LOCAL_BUCKETS.pop(rkey, None)