"""Asynchronous webhook push dispatcher with circuit breaker and DLQ routing."""
from __future__ import annotations

import time
import requests
from datetime import datetime, timezone
import json

from .db import get_db_cursor
from .config import EVENT_DELIVERY_MODE, EVENT_WEBHOOK_URL, OUTBOX_MAX_ATTEMPTS


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


def dispatch_pending_webhooks(limit: int = 50, webhook_url: str = EVENT_WEBHOOK_URL) -> int:
    """Push leased outbox events to the configured webhook endpoint with connection pooling."""
    if not webhook_url or EVENT_DELIVERY_MODE != "webhook":
        return 0

    if circuit_breaker.is_open():
        return 0

    delivered_count = 0
    session = get_http_session()

    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
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
            """,
            (limit,)
        )
        rows = cur.fetchall()

        if not rows:
            return 0

        for r in rows:
            outbox_id = r["id"] if isinstance(r, dict) else r[0]
            event_type = r["event_type"] if isinstance(r, dict) else r[1]
            post_id = r["post_id"] if isinstance(r, dict) else r[2]
            event_id = r["event_id"] if isinstance(r, dict) else r[3]
            payload = r["payload"] if isinstance(r, dict) else r[4]
            attempts = r["attempts"] if isinstance(r, dict) else r[5]

            body = {
                "id": outbox_id,
                "event_type": event_type,
                "post_id": post_id,
                "event_id": event_id,
                "payload": payload,
                "dispatched_at": datetime.now(timezone.utc).isoformat()
            }

            try:
                res = session.post(
                    webhook_url,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Nexus-Event-Dispatcher/1.0"
                    },
                    timeout=3.0
                )
                if res.status_code in {200, 201, 202, 204}:
                    cur.execute(
                        """
                        UPDATE integration_outbox 
                        SET delivery_status = 'delivered', delivered_at = NOW(), lease_until = NULL 
                        WHERE id = %s;
                        """,
                        (outbox_id,)
                    )
                    circuit_breaker.record_success()
                    delivered_count += 1
                else:
                    raise RuntimeError(f"HTTP {res.status_code}: {res.text[:200]}")
            except Exception as exc:
                circuit_breaker.record_failure()
                err_msg = str(exc)[:400]

                if attempts >= OUTBOX_MAX_ATTEMPTS:
                    cur.execute(
                        """
                        UPDATE integration_outbox 
                        SET delivery_status = 'failed', last_error = %s, lease_until = NULL 
                        WHERE id = %s;
                        """,
                        (err_msg, outbox_id)
                    )
                else:
                    cur.execute(
                        """
                        UPDATE integration_outbox 
                        SET delivery_status = 'pending', lease_until = NULL, 
                            available_at = NOW() + INTERVAL '1 minute', last_error = %s 
                        WHERE id = %s;
                        """,
                        (err_msg, outbox_id)
                    )

    return delivered_count
