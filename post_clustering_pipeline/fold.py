"""Same-hub duplicate-content folding (repost collapse).

A hub exposed to a repost storm (one author, N reposts of the same body) would
otherwise present N identical posts. Every post is fingerprinted at ingest
(``nlp.content_signature`` -> ``posts.content_sig``); the fold keeps ONE
canonical post per ``(event_id, content_sig)`` and marks the duplicates as
reposts of it (``repost_of_id`` -> canonical, ``assignment_status='repost'``).

The canonical post keeps ``event_id`` and its assigned status, so pairwise
recall/precision over the corpus is untouched - reposts still live in their
hub in the membership ledger and centroid (identical embeddings move nothing),
they are simply no longer *surfaced* as independent members. After a fold,
``event_hubs.member_count`` counts DISTINCT content signals and
``repost_count`` carries the folded copies.

Precision guards (mirroring the pipeline's precision-first ethos):

* Folding is scoped to a single hub (``event_id``). Two hubs legitimately
  sharing identical copy (e.g. a re-shared trending snippet across unrelated
  communities) are NEVER collapsed - the membership ledger of the second hub
  stays whole.
* Exact folds are byte-level by normalized content hash, so they cannot merge
  distinct topics.
* Near-dup folds require embedding cosine >= ``REPOST_NEAR_DUP_COSINE``
  (0.98) - far tighter than any assignment/birth threshold - and only run on
  the tidying sweep and at hub birth, off the streaming hot path.

Write paths call the fold inline so a folder is never surfaced to readers;
the scheduled sweep is the idempotent safety net for missed paths, legacy
rows (``content_sig`` = 0 backfilled first), and merge-after-fold collisions.
"""
from __future__ import annotations

import numpy as np

from .config import (
    REPOST_NEAR_DUP_COSINE,
    REPOST_FOLD_MAX_CANONICALS,
    REPOST_BACKFILL_BATCH,
)
from .db import get_db_cursor, extract_val
from .embed_io import parse_vector_literal
from .nlp import content_signature


def _fold_exact_for_hubs(cur, hub_ids) -> set[int]:
    """Fold byte-identical content within each hub; returns folded post ids.

    The canonical per ``(event_id, content_sig)`` is the earliest id among the
    whole body's rows: current canonicals PLUS the ``repost_of_id`` targets of
    already-folded copies. Resolving through the lineage matters because a
    storm's earlier copies may have been collapsed by the NEAR-DUP resolver
    onto a canonical whose own signature differs (edited repost) - a late
    byte-identical copy then shares a signature only with a REPOST, and must
    fold onto that repost's canonical, not linger as a second canonical
    member. Only rows with a real signature (``content_sig <> 0``) and no
    prior repost link are folded, so the statement is idempotent and cheap to
    re-run on an already-folded hub.
    """
    if not hub_ids:
        return set()
    cur.execute(
        """
        WITH keepers AS (
            -- candidates for the body's canonical: current canonicals AND
            -- the repost targets of folded copies (resolves the orphan case)
            SELECT event_id, content_sig, id AS keep_id
            FROM posts
            WHERE event_id = ANY(%s)
              AND content_sig <> 0
              AND repost_of_id IS NULL
              AND deleted_at IS NULL
            UNION
            SELECT p.event_id, p.content_sig, r.id AS keep_id
            FROM posts p
            JOIN posts r ON r.id = p.repost_of_id AND r.deleted_at IS NULL
            WHERE p.event_id = ANY(%s)
              AND p.content_sig <> 0
              AND p.repost_of_id IS NOT NULL
              AND p.deleted_at IS NULL
        ),
        picks AS (
            SELECT event_id, content_sig, MIN(keep_id) AS keep_id
            FROM keepers
            GROUP BY event_id, content_sig
        )
        UPDATE posts p
        SET repost_of_id = picks.keep_id,
            assignment_status = 'repost',
            assignment_updated_at = NOW()
        FROM picks
        WHERE p.event_id = picks.event_id
          AND p.content_sig = picks.content_sig
          AND p.id <> picks.keep_id
          AND p.repost_of_id IS NULL
          AND p.deleted_at IS NULL
        RETURNING p.id;
        """,
        (list(hub_ids), list(hub_ids))
    )
    return {extract_val(r, "id", 0) for r in (cur.fetchall() or [])}


def _near_dup_fold_map(posts: list[dict], threshold: float = REPOST_NEAR_DUP_COSINE) -> dict[int, int]:
    """Pure repost-resolver: which post ids fold to which canonical id.

    ``posts`` is a list of ``{"id": int, "embedding": np.ndarray}`` for the
    canonical (already exact-deduped) members of one hub. Pairwise cosine
    similarity is computed via one normalized matrix multiply; pairs at or
    above ``threshold`` form a connected graph, and each connected component
    resolves to its smallest id. Non-repeating output: never folds two posts
    that already share a body, and never creates chains.

    Bounded O(n^2) work; callers cap ``n`` with REPOST_FOLD_MAX_CANONICALS.
    """
    if len(posts) < 2:
        return {}
    posts = [p for p in posts if p.get("embedding") is not None]
    if len(posts) < 2:
        return {}
    ids = [int(p["id"]) for p in posts]
    matrix = np.stack([np.asarray(p["embedding"], dtype=np.float64) for p in posts])
    norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    sims = (matrix / norm) @ (matrix / norm).T
    edges = [
        (i, j)
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
        if float(sims[i, j]) >= threshold
    ]
    if not edges:
        return {}

    parent = list(range(len(ids)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for node in range(len(ids)):
        groups.setdefault(find(node), []).append(node)

    fold_map: dict[int, int] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = min(members, key=lambda i: ids[i])
        for i in members:
            if i != canonical:
                fold_map[ids[i]] = ids[canonical]
    return fold_map


def _fold_near_dup_for_hubs(cur, hub_ids, embeddings_by_id=None) -> set[int]:
    """Fold near-duplicate (edited repost) content within each hub.

    Only canonical rows (``repost_of_id IS NULL``, embedding present) are
    compared; hubs larger than REPOST_FOLD_MAX_CANONICALS are skipped because
    the exact pass has already collapsed byte-identical storms there and the
    added pairwise cost is not justified. ``embeddings_by_id`` lets birth-time
    callers reuse vectors they already hold (flush per hub to bound memory).
    Returns the set of newly folded post ids.
    """
    if not hub_ids:
        return set()
    fold_map: dict[int, int] = {}
    for hub_id in hub_ids:
        cur.execute(
            """
            SELECT id, embedding::text
            FROM posts
            WHERE event_id = %s
              AND repost_of_id IS NULL
              AND deleted_at IS NULL
              AND embedding IS NOT NULL
            ORDER BY id
            LIMIT %s;
            """,
            (hub_id, REPOST_FOLD_MAX_CANONICALS + 1)
        )
        rows = cur.fetchall() or []
        if len(rows) > REPOST_FOLD_MAX_CANONICALS:
            continue
        posts = []
        for r in rows:
            pid = extract_val(r, "id", 0)
            emb_text = extract_val(r, "embedding", 1)
            provided = (embeddings_by_id or {}).get(pid)
            posts.append({
                "id": pid,
                "embedding": provided if provided is not None else parse_vector_literal(emb_text),
            })
        fold_map.update(_near_dup_fold_map(posts))

    if not fold_map:
        return set()
    from psycopg2.extras import execute_values

    rows = list(fold_map.items())  # (repost_id, canonical_id)
    execute_values(
        cur,
        """
        UPDATE posts p
        SET repost_of_id = v.keep, assignment_status = 'repost', assignment_updated_at = NOW()
        FROM (VALUES %s) AS v(id, keep)
        WHERE p.id = v.id
          AND p.repost_of_id IS NULL
          AND p.assignment_status <> 'repost'
          AND p.deleted_at IS NULL
          AND EXISTS (
              SELECT 1 FROM posts k
              WHERE k.id = v.keep
                AND k.deleted_at IS NULL
                AND k.repost_of_id IS NULL
          )
        RETURNING p.id;
        """,
        rows,
        template="(%s::int, %s::int)",
    )
    return {extract_val(r, "id", 0) for r in (cur.fetchall() or [])}


def _reparent_repost_children(cur) -> None:
    """Collapse repost chains after a fold.

    Sequential fold passes can transiently create a chain (a later near-dup
    fold flips a canonical into a repost while earlier reposts still point at
    it). Repoint every child to the terminal canonical in one indexed pass;
    idempotent, and a no-op on an already-clean hub.
    """
    cur.execute(
        """
        UPDATE posts p
        SET repost_of_id = c.repost_of_id
        FROM posts c
        WHERE p.repost_of_id = c.id
          AND c.repost_of_id IS NOT NULL
          AND p.repost_of_id <> c.repost_of_id;
        """
    )


def fold_hubs(cur, hub_ids, *, near_dup: bool = False, embeddings_by_id=None) -> dict:
    """Fold reposts for ``hub_ids`` (exact + optional near-dup) and resync counts.

    Returns ``{"folded_exact": n, "folded_near_dup": n}``. Callers must wrap
    this in the transaction they already hold; count resync runs in the same
    statement as the fold so member_count/repost_count never diverge.
    """
    folded_exact = _fold_exact_for_hubs(cur, hub_ids)
    folded_near_dup = _fold_near_dup_for_hubs(cur, hub_ids, embeddings_by_id) if near_dup else set()
    _reparent_repost_children(cur)
    # Outbox truth at fold time, so inline folds never leave a stale
    # post.assigned event over-counting outward canonicals (the audit and any
    # consumer derive canonicals from status, not from the outbox).
    cur.execute(
        """
        UPDATE integration_outbox o
        SET delivery_status = 'superseded', last_error = 'folded_repost',
            attempts = attempts + 1
        FROM posts p
        WHERE o.event_type = 'post.assigned'
          AND p.repost_of_id IS NOT NULL
          AND p.id = o.post_id
          AND o.delivery_status <> 'superseded';
        """
    )
    _sync_hub_counts(cur, hub_ids)
    return {"folded_exact": len(folded_exact), "folded_near_dup": len(folded_near_dup)}


def _sync_hub_counts(cur, hub_ids) -> None:
    """Recompute member_count (distinct signals) and repost_count from live posts."""
    if not hub_ids:
        return
    cur.execute(
        """
        UPDATE event_hubs h
        SET member_count = s.canonicals,
            repost_count = s.reposts,
            last_updated_at = NOW()
        FROM (
            SELECT event_id,
                   COUNT(*) FILTER (WHERE repost_of_id IS NULL) AS canonicals,
                   COUNT(*) FILTER (WHERE repost_of_id IS NOT NULL) AS reposts
            FROM posts
            WHERE event_id = ANY(%s) AND deleted_at IS NULL
            GROUP BY event_id
        ) s
        WHERE h.id = s.event_id
          AND (h.member_count <> s.canonicals OR h.repost_count <> s.reposts);
        """,
        (list(hub_ids),)
    )


def sync_hub_counts(cur, hub_ids) -> None:
    """Public count-only resync (used where a fold did not change membership)."""
    _sync_hub_counts(cur, hub_ids)


def discover_foldable_hubs(cur, limit: int = 200) -> list[int]:
    """Hubs with at least one duplicated content signature (cheap, indexed)."""
    cur.execute(
        """
        SELECT DISTINCT event_id
        FROM (
            SELECT event_id, content_sig
            FROM posts
            WHERE content_sig <> 0
              AND repost_of_id IS NULL
              AND deleted_at IS NULL
              AND event_id IS NOT NULL
            GROUP BY event_id, content_sig
            HAVING COUNT(*) > 1
        ) d
        ORDER BY event_id
        LIMIT %s;
        """,
        (limit,)
    )
    return [extract_val(r, "event_id", 0) for r in (cur.fetchall() or [])]


def discover_near_dup_hubs(cur, limit: int = 200) -> list[int]:
    """Hubs worth a near-dup pass: 2..REPOST_FOLD_MAX_CANONICALS canonicals.

    Edited-repost variants ('!!!'/extra words) carry distinct content
    signatures, so the exact-only discovery never sees them; they are caught
    here instead. The cap keeps the pairwise cosine pass O(n^2)-bounded, and
    ``limit`` bounds how many hubs a single sweep pass visits.
    """
    cur.execute(
        """
        SELECT h.id
        FROM event_hubs h
        JOIN (
            SELECT event_id, COUNT(*) AS canonicals
            FROM posts
            WHERE deleted_at IS NULL
              AND repost_of_id IS NULL
              AND embedding IS NOT NULL
              AND event_id IS NOT NULL
            GROUP BY event_id
            HAVING COUNT(*) BETWEEN 2 AND %s
        ) s ON s.event_id = h.id
        WHERE h.is_active = TRUE
        ORDER BY s.canonicals DESC, h.id
        LIMIT %s;
        """,
        (REPOST_FOLD_MAX_CANONICALS, limit)
    )
    return [extract_val(r, "id", 0) for r in (cur.fetchall() or [])]


def backfill_content_signatures(cur, limit: int | None = None) -> int:
    """Compute content_sig for legacy rows minted before the column existed.

    A signature of exactly 0 cannot occur from ``content_signature`` (the
    sentinel is reserved), so ``sig = 0`` unambiguously means 'never computed'.
    """
    limit = limit or REPOST_BACKFILL_BATCH
    cur.execute(
        """
        SELECT id, content FROM posts
        WHERE content_sig = 0
          AND deleted_at IS NULL
        ORDER BY id
        LIMIT %s;
        """,
        (limit,)
    )
    rows = cur.fetchall() or []
    if not rows:
        return 0
    from psycopg2.extras import execute_values

    execute_values(
        cur,
        """
        UPDATE posts p
        SET content_sig = v.sig
        FROM (VALUES %s) AS v(id, sig)
        WHERE p.id = v.id;
        """,
        [(extract_val(r, "id", 0), content_signature(extract_val(r, "content", 1))) for r in rows],
        template="(%s::int, %s::bigint)",
    )
    return cur.rowcount


def run_fold_sweep(*, batch: int = 200, near_dup: bool = True) -> dict:
    """Idempotent safety-net sweep over hubs holding duplicates.

    Visits hubs with duplicated exact signatures (the cheap indexed signal)
    AND hubs with 2..REPOST_FOLD_MAX_CANONICALS canonicals (the near-dup
    edited-repost signal) - a hub without exact duplicates may still hold
    edited reposts, and those are only reachable via the near-dup pass.

    Backfills legacy signatures first, folds, and resyncs counts - all in one
    transaction. Returns on the no-op path without creating a transaction.
    """
    with get_db_cursor(commit=True) as cur:
        backfilled = backfill_content_signatures(cur, limit=REPOST_BACKFILL_BATCH)
        hub_ids = set(discover_foldable_hubs(cur, limit=batch))
        if near_dup:
            hub_ids.update(discover_near_dup_hubs(cur, limit=batch))
        hub_ids = sorted(hub_ids)[:batch]
        result = fold_hubs(cur, hub_ids, near_dup=near_dup) if hub_ids else {"folded_exact": 0, "folded_near_dup": 0}
        # Invariant repair: the fold canonicals a row by setting repost_of_id;
        # any status writer that later races back to 'assigned' must never win
        # over the fold, so stale resurrected statuses are re-corrected here
        # every sweep. Runs even when no hub was foldable, because resurrected
        # rows are invisible to the hub-discovery probes (both filter
        # repost_of_id IS NULL) and would otherwise persist un-fixed.
        cur.execute(
            """
            UPDATE posts
            SET assignment_status = 'repost', assignment_updated_at = NOW()
            WHERE repost_of_id IS NOT NULL AND assignment_status = 'assigned';
            """
        )
        status_repair = cur.rowcount
        # Outbox truth: a fold-after-assign leaves a stale `post.assigned`
        # outbox row that the audit's parity check (post.assigned == outward
        # canonicals) would over-count against. Any post that now carries a
        # repost_of_id is NOT canonical, so supersede its stale assignment
        # event here (one row per post/event_type by unique index).
        cur.execute(
            """
            UPDATE integration_outbox o
            SET delivery_status = 'superseded', last_error = 'folded_repost',
                attempts = attempts + 1
            FROM posts p
            WHERE o.event_type = 'post.assigned'
              AND p.repost_of_id IS NOT NULL
              AND p.id = o.post_id
              AND o.delivery_status <> 'superseded';
            """
        )
        outbox_superseded = cur.rowcount
        result.update({"backfilled": backfilled, "hubs_folded": len(hub_ids),
                       "status_repair": status_repair, "outbox_superseded": outbox_superseded})
        return result