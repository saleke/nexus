"""Durable integration events for host systems."""
import json


def write_outbox(cur, event_type: str, post_id: int | None = None, event_id: int | None = None, payload: dict | None = None):
    cur.execute(
        """
        INSERT INTO integration_outbox (event_type, post_id, event_id, payload)
        VALUES (%s, %s, %s, %s::jsonb)
        ON CONFLICT (post_id, event_type) WHERE post_id IS NOT NULL
        DO UPDATE SET 
            event_id = EXCLUDED.event_id,
            payload = EXCLUDED.payload,
            delivery_status = 'pending',
            attempts = 0,
            available_at = NOW()
        RETURNING id;
        """,
        (event_type, post_id, event_id, json.dumps(payload or {}))
    )
