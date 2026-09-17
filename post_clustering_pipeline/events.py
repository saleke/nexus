"""Durable integration events for host systems.

Every row carrying a ``post_id`` embeds a per-post monotone ``post_seq`` in
its payload. Consumers use ``(post_id, post_seq)`` as their idempotency key:
a post that moves hubA -> hubB (assigned -> unlinked -> assigned) yields
strictly increasing sequences, so a consumer can drop stale deliveries and
order events per post even when out-of-band delivery reorders them.
"""
import json


def _row_col(row, name: str, idx: int = 0):
    """Read a column from a psycopg row (dict -> by name, tuple -> by index)."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(name)
    return row[idx]


def bump_post_seqs(cur, post_ids) -> dict[int, int]:
    """Monotone per-post sequence bump: ``{post_id: new_seq}``.

    Runs as a single UPDATE .. RETURNING so the bulk assignment path pays one
    round-trip for the whole chunk instead of one per post.
    """
    ids = sorted({int(pid) for pid in post_ids if pid is not None})
    if not ids:
        return {}
    cur.execute(
        "UPDATE posts SET post_seq = post_seq + 1 WHERE id = ANY(%s::int[]) RETURNING id, post_seq;",
        (ids,)
    )
    return {
        int(_row_col(r, "id", 0)): int(_row_col(r, "post_seq", 1) or 0)
        for r in (cur.fetchall() or [])
    }


def _seq_payload(payload: dict, seq: int | None) -> dict:
    out = dict(payload or {})
    if seq is not None:
        out["post_seq"] = seq
    return out


def write_outbox(cur, event_type: str, post_id: int | None = None, event_id: int | None = None, payload: dict | None = None):
    payload = dict(payload or {})
    if post_id is not None:
        payload = _seq_payload(payload, bump_post_seqs(cur, [post_id]).get(post_id))
    cur.execute(
        """
        INSERT INTO integration_outbox (event_type, post_id, event_id, payload, schema_version)
        VALUES (%s, %s, %s, %s::jsonb, 2)
        ON CONFLICT (post_id, event_type) WHERE post_id IS NOT NULL
        DO UPDATE SET
            event_id = EXCLUDED.event_id,
            payload = EXCLUDED.payload,
            schema_version = 2,
            delivery_status = 'pending',
            attempts = 0,
            available_at = NOW()
        RETURNING id;
        """,
        (event_type, post_id, event_id, json.dumps(payload, default=str))
    )


def write_outbox_bulk(cur, events) -> None:
    """Write many outbox events in one seq-bump + one INSERT.

    ``events``: iterable of ``(event_type, post_id, event_id, payload)`` tuples,
    mirroring ``write_outbox`` row-for-row (same seq attachment, same conflict
    upsert). One round-trip for the seq bump and one for the insert regardless
    of batch size, so noise flaps and bulk assignments no longer pay N UPDATEs.
    """
    rows = [(e[0], e[1], e[2], e[3]) for e in events if e]
    if not rows:
        return
    seqs = bump_post_seqs(cur, [post_id for _, post_id, _, _ in rows])
    from psycopg2.extras import execute_values
    execute_values(
        cur,
        """
        INSERT INTO integration_outbox (event_type, post_id, event_id, payload, schema_version)
        VALUES %s
        ON CONFLICT (post_id, event_type) WHERE post_id IS NOT NULL
        DO UPDATE SET
            event_id = EXCLUDED.event_id,
            payload = EXCLUDED.payload,
            schema_version = 2,
            delivery_status = 'pending',
            attempts = 0,
            available_at = NOW();
        """,
        [
            (
                event_type,
                post_id,
                event_id,
                json.dumps(_seq_payload(dict(payload or {}), seqs.get(post_id)), default=str),
            )
            for event_type, post_id, event_id, payload in rows
        ],
        template="(%s, %s::int, %s::int, %s::jsonb, 2)",
    )
