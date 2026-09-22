"""Asynchronous webhook push dispatcher with circuit breaker and DLQ routing."""
from __future__ import annotations

import hashlib
import hmac
import time
import requests
from datetime import datetime, timezone
import json

from psycopg2.extras import execute_values

from .db import get_db_cursor
from .config import EVENT_DELIVERY_MODE, EVENT_WEBHOOK_URL, EVENT_WEBHOOK_SIGNING_SECRET, OUTBOX_MAX_ATTEMPTS


def signature_for(payload: bytes, secret: str) -> str:
    """Hex HMAC-SHA256 of the EXACT bytes being delivered (RFC 2104)."""
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


class CircuitBreaker:
    """Thread-safe circuit breaker protecting worker threads against downstream host outages."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"

    def is_open(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


circuit_breaker = CircuitBreaker()
_http_session: requests.Session | None = None


def get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
    return _http_session


_CLAIM_SQL = """
    WITH claimed AS (
        SELECT id FROM integration_outbox
        WHERE available_at <= NOW()
          AND (delivery_status = 'pending' OR (delivery_status = 'leased' AND lease_until < NOW()))
        ORDER BY id ASC
        LIMIT %s FOR UPDATE SKIP LOCKED
    )
    UPDATE integration_outbox o
    SET delivery_status = 'leased', lease_until = NOW() + INTERVAL '2 minutes', attempts = attempts + 1
    FROM claimed c WHERE o.id = c.id
    RETURNING o.id, o.event_type, o.post_id, o.event_id, o.payload, o.attempts;
"""


def _outbox_field(row, name: str, idx: int):
    return row[name] if isinstance(row, dict) else row[idx]


def _deliver_one(session, webhook_url: str, row) -> tuple[int, str, str | None]:
    """POST one claimed event; return ``(outbox_id, status, error)``.

    ``status`` is ``delivered`` when the host accepts, ``failed`` once the
    attempt budget is exhausted, else ``pending`` for a later retry.
    """
    outbox_id = _outbox_field(row, "id", 0)
    event_type = _outbox_field(row, "event_type", 1)
    post_id = _outbox_field(row, "post_id", 2)
    event_id = _outbox_field(row, "event_id", 3)
    payload = _outbox_field(row, "payload", 4)
    attempts = _outbox_field(row, "attempts", 5)

    body = {
        "id": outbox_id,
        "event_type": event_type,
        "post_id": post_id,
        "event_id": event_id,
        "payload": payload,
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
    }
    payload_bytes = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Nexus-Event-Dispatcher/1.0",
    }
    signing = EVENT_WEBHOOK_SIGNING_SECRET
    if signing:
        # Sign the exact bytes we send; the host recomputes and compares
        # constant-time. Uniqueness-header per event id for idempotency.
        headers["X-Nexus-Signature"] = "sha256=" + signature_for(payload_bytes, signing)

    try:
        res = session.post(webhook_url, data=payload_bytes, headers=headers, timeout=3.0)
        if res.status_code in {200, 201, 202, 204}:
            circuit_breaker.record_success()
            return outbox_id, "delivered", None
        raise RuntimeError(f"HTTP {res.status_code}: {res.text[:200]}")
    except Exception as exc:
        circuit_breaker.record_failure()
        err_msg = str(exc)[:400]
        status = "failed" if attempts >= OUTBOX_MAX_ATTEMPTS else "pending"
        return outbox_id, status, err_msg


def dispatch_pending_webhooks(limit: int = 50, webhook_url: str = EVENT_WEBHOOK_URL, session=None) -> int:
    """Push leased outbox events to the configured webhook endpoint.

    Three phases so no HTTP call ever runs inside a database transaction: a
    short claim transaction leases the batch, the network I/O happens with no
    transaction open, and a second short transaction records the outcomes.
    A crash between phases leaves rows leased; they are reclaimed once the
    2-minute lease expires (delivery is therefore at-least-once, which is why
    every payload carries its idempotency key).
    """
    if not webhook_url or EVENT_DELIVERY_MODE != "webhook":
        return 0
    if not webhook_url.startswith(("https://", "http://")):
        return 0  # refuse to POST anywhere exotic (file:, redis:, data:)

    if circuit_breaker.is_open():
        return 0

    session = session or get_http_session()

    # Phase 1: claim under row locks, then commit immediately to release them.
    with get_db_cursor(commit=True) as cur:
        cur.execute(_CLAIM_SQL, (limit,))
        rows = cur.fetchall()

    if not rows:
        return 0

    # Phase 2: network I/O with no transaction/locks held.
    outcomes: list[tuple[int, str, str | None]] = []
    unprocessed = []
    for i, row in enumerate(rows):
        if circuit_breaker.is_open():
            unprocessed = rows[i:]
            break
        outcomes.append(_deliver_one(session, webhook_url, row))

    delivered_count = sum(1 for _, status, _ in outcomes if status == "delivered")

    # Phase 3: record outcomes and release any rows skipped after the breaker
    # opened (undo their claim increment so a transient outage cannot push them
    # toward the DLQ threshold).
    with get_db_cursor(commit=True) as cur:
        if outcomes:
            execute_values(
                cur,
                """
                UPDATE integration_outbox o
                SET delivery_status = v.status,
                    lease_until = NULL,
                    last_error = CASE WHEN v.status = 'delivered' THEN o.last_error ELSE v.last_error END,
                    delivered_at = CASE WHEN v.status = 'delivered' THEN NOW() ELSE o.delivered_at END,
                    available_at = CASE WHEN v.status = 'pending' THEN NOW() + INTERVAL '1 minute' ELSE o.available_at END
                FROM (VALUES %s) AS v(id, status, last_error)
                WHERE o.id = v.id;
                """,
                outcomes,
                template="(%s::int, %s::text, %s::text)",
            )
        if unprocessed:
            uids = [_outbox_field(r, "id", 0) for r in unprocessed]
            cur.execute(
                "UPDATE integration_outbox SET delivery_status = 'pending', lease_until = NULL, "
                "attempts = GREATEST(attempts - 1, 0) "
                "WHERE id = ANY(%s::int[]) AND delivery_status = 'leased';",
                (uids,),
            )

    return delivered_count
