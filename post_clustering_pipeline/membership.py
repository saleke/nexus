"""First-class membership ledger between posts and hubs.

The relationship is no longer just two pointer columns (``posts.event_id`` =
live membership, ``event_hubs.seed_post_id`` = lineage). Live membership
still lives on ``posts.event_id`` for the hot write path, but every mutation is
mirrored here as a role-aware, historical row. ``post_id`` rows are closed
(``departed_at``) instead of deleted, so "what was this hub at time T" is
answerable and the anchor role is explicit rather than implicit.

Invariants enforced:
  * a post has at most ONE live row (partial unique index on ``post_id`` where
    ``departed_at IS NULL``), matching the single-valued ``posts.event_id``;
  * identity never depends on membership: the hub's label lives on
    ``event_hubs.title/handle``, and its ``seed_post_id`` is only lineage.
"""
from __future__ import annotations

from psycopg2.extras import execute_values

from .db import extract_val

SEED_ROLE = "seed"
MEMBER_ROLE = "member"


def current_hub(cur, post_id: int) -> int | None:
    """The post's live hub (id) as the membership ledger currently knows it."""
    cur.execute(
        "SELECT hub_id FROM hub_members WHERE post_id = %s AND departed_at IS NULL;",
        (post_id,)
    )
    row = cur.fetchone()
    return extract_val(row, "hub_id", 0) if row else None


def _close_live(cur, post_id: int) -> None:
    cur.execute(
        "UPDATE hub_members SET departed_at = NOW() "
        "WHERE post_id = %s AND departed_at IS NULL;",
        (post_id,)
    )


def record_membership(cur, post_id: int, hub_id: int, role: str = MEMBER_ROLE) -> None:
    """Make ``post_id`` a live member of ``hub_id`` with the given role.

    Idempotent: a no-op when the post already holds that exact live row. A
    cross-hub move closes the old live row (history preserved) and adds the
    new one. Must run in the caller's transaction, next to the ``posts.event_id``
    update it mirrors.
    """
    cur.execute(
        "SELECT hub_id, role FROM hub_members "
        "WHERE post_id = %s AND departed_at IS NULL FOR UPDATE;",
        (post_id,)
    )
    row = cur.fetchone()
    if row and extract_val(row, "hub_id", 0) == hub_id and extract_val(row, "role", 1) == role:
        return
    if row:
        _close_live(cur, post_id)
    cur.execute(
        "INSERT INTO hub_members (post_id, hub_id, role) VALUES (%s, %s, %s);",
        (post_id, hub_id, role)
    )


def record_memberships(cur, rows) -> None:
    """Bulk form of ``record_membership`` for ``(post_id, hub_id, role)`` triples.

    Each triple moves/creates the post's live membership the same way the
    single-row path does, but in one round-trip. ``record_membership``-style
    idempotency is preserved per row by closing any existing live row first
    (a live row pointing at the same hub+role is left untouched).
    """
    rows = [(int(p), int(h), r or MEMBER_ROLE) for p, h, r in rows]
    if not rows:
        return
    execute_values(
        cur,
        """
        UPDATE hub_members hm
        SET departed_at = NOW()
        FROM (VALUES %s) AS v(post_id, hub_id, role)
        WHERE hm.post_id = v.post_id
          AND hm.departed_at IS NULL
          AND NOT (hm.hub_id = v.hub_id AND hm.role = v.role);
        """,
        rows,
        template="(%s::int, %s::int, %s::text)",
    )
    execute_values(
        cur,
        """
        INSERT INTO hub_members (post_id, hub_id, role)
        SELECT post_id, hub_id, role FROM (VALUES %s) AS v(post_id, hub_id, role)
        WHERE NOT EXISTS (
            SELECT 1 FROM hub_members hm
            WHERE hm.post_id = v.post_id AND hm.departed_at IS NULL
              AND hm.hub_id = v.hub_id AND hm.role = v.role
        );
        """,
        rows,
        template="(%s::int, %s::int, %s::text)",
    )


def close_membership(cur, post_id: int) -> None:
    """End the post's live membership (mirrors ``posts.event_id`` -> NULL)."""
    _close_live(cur, post_id)


def move_membership(cur, post_ids, to_hub: int, role: str = MEMBER_ROLE) -> None:
    """Move a batch of posts to ``to_hub`` in the ledger (merge/reopen paths)."""
    record_memberships(cur, [(pid, to_hub, role) for pid in (post_ids or [])])


def hub_member_roles(cur, hub_id: int) -> dict[int, str]:
    """All *live* memberships of a hub: ``{post_id: role}``."""
    cur.execute(
        "SELECT post_id, role FROM hub_members "
        "WHERE hub_id = %s AND departed_at IS NULL;",
        (hub_id,)
    )
    return {
        int(extract_val(r, "post_id", 0)): extract_val(r, "role", 1) or MEMBER_ROLE
        for r in (cur.fetchall() or [])
    }


def hub_membership_history(cur, hub_id: int, limit: int = 200) -> list[dict]:
    """Admission/departure history for a hub, newest first."""
    cur.execute(
        """
        SELECT post_id, hub_id, role, admitted_at, departed_at
        FROM hub_members
        WHERE hub_id = %s
        ORDER BY admitted_at DESC
        LIMIT %s;
        """,
        (hub_id, limit)
    )
    return [{
        "post_id": extract_val(r, "post_id", 0),
        "hub_id": extract_val(r, "hub_id", 1),
        "role": extract_val(r, "role", 2) or MEMBER_ROLE,
        "admitted_at": extract_val(r, "admitted_at", 3),
        "departed_at": extract_val(r, "departed_at", 4),
    } for r in (cur.fetchall() or [])]