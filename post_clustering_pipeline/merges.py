"""Shared soft-merge apply / reopen path.

Client requests, owner actions, and the internal auto-detect job all funnel
through the SAME guarded transaction so the outcome is identical regardless of
initiator (only the ``initiated_by`` tag differs on the audit row). Merges are:

  * canonical (redirect chains resolved before any mutation),
  * race-safe (callers hold pg_advisory_xact_lock on the ordered hub pair),
  * idempotent / conflict-avoiding (active-only, distinct, non-nested),
  * reversible (snapshot of post ids + source centroid lets reopen be exact).
"""
from __future__ import annotations

import json

import numpy as np

from .db import extract_val, get_db_cursor
from .events import write_outbox
from .membership import move_membership
from .centroid import normalize, bounded_rolling_centroid
from .embed_io import vector_to_array_literal, parse_vector_literal
from .refs import hub_reference, post_reference
from .audit import log_admin_action
from .config import CENTROID_MAX_MEMBERS

MERGE_LOCK_NAMESPACE = 0x4E455800  # "NEX" - collision-avoiding namespace


def resolve_canonical_hub(cur, hub_id: int) -> tuple[int, bool]:
    """Resolve canonical hub id following merged_into_id links (max 5 hops)."""
    current = int(hub_id)
    redirected = False
    for _ in range(5):
        cur.execute("SELECT id, is_active, merged_into_id FROM event_hubs WHERE id = %s;", (current,))
        row = cur.fetchone()
        if not row:
            break
        is_active = extract_val(row, "is_active", 1)
        merged_into = extract_val(row, "merged_into_id", 2)
        if is_active or not merged_into:
            return current, redirected
        current = int(merged_into)
        redirected = True
    return current, redirected


def _recompute_hub_centroid(cur, hub_id: int) -> None:
    """Recompute a hub's centroid from its live member posts.

    On reopen, BOTH hubs change membership: the source winnows the posts it
    regains (they were rebound from target, which now has fewer members).
    The stored ``snapshot_centroid_source`` only reflects the frozen members
    at merge time; reopening the merge must instead rebuild from the posts
    currently bound to the hub so the next assignment round matches the
    current, live set rather than the stale merge-time one.

    Uses the same bounded rolling centroid (anti-fossilization max cap) that
    the streaming encoder uses, so the reopened hub re-enters the pool with
    the correct live density instead of the pre-merge snapshot shape.
    """
    cur.execute(
        """
        SELECT embedding::text
        FROM posts
        WHERE event_id = %s
          AND deleted_at IS NULL
          AND assignment_status = 'assigned'
        ORDER BY id
        LIMIT %s;
        """,
        (hub_id, CENTROID_MAX_MEMBERS)
    )
    rows = cur.fetchall() or []
    vectors = [
        parse_vector_literal(extract_val(r, "embedding", 0))
        for r in rows
    ]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        cur.execute(
            "UPDATE event_hubs SET centroid = NULL, last_updated_at = NOW() WHERE id = %s;",
            (hub_id,)
        )
        return
    rolled = bounded_rolling_centroid(None, 0, vectors, CENTROID_MAX_MEMBERS)
    centroid_str = vector_to_array_literal(rolled)
    cur.execute(
        "UPDATE event_hubs SET centroid = %s::vector, last_updated_at = NOW() WHERE id = %s;",
        (centroid_str, hub_id)
    )


def _hub(cur, hub_id: int) -> dict | None:
    cur.execute(
        "SELECT id, is_active, member_count, centroid::text, last_updated_at FROM event_hubs WHERE id = %s;",
        (hub_id,)
    )
    r = cur.fetchone()
    if not r:
        return None
    centroid_text = extract_val(r, "centroid", 3)
    return {
        "id": extract_val(r, "id", 0),
        "is_active": bool(extract_val(r, "is_active", 1)),
        "member_count": int(extract_val(r, "member_count", 2) or 1),
        "centroid_text": centroid_text,
        "centroid": parse_vector_literal(centroid_text) if centroid_text else None,
    }


def _feedback_score(cur, source, target) -> float:
    src = _hub(cur, source)
    tgt = _hub(cur, target)
    if src and tgt and src["centroid"] is not None and tgt["centroid"] is not None:
        a = src["centroid"] / (np.linalg.norm(src["centroid"]) + 1e-9)
        b = tgt["centroid"] / (np.linalg.norm(tgt["centroid"]) + 1e-9)
        return round(float(np.dot(a, b)), 3)
    return 0.900


def apply_merge_soft(cur, source_event_id: int, target_event_id: int,
                     initiated_by: str = "client", actor: str | None = None,
                     note: str | None = None) -> dict:
    """Soft-merge ``source`` into ``target`` inside the caller's transaction.

    Caller must hold the advisory lock for the ordered pair (use
    ``lock_hub_pair``) and own the transaction commit. Returns a dict with the
    canonical target id, member counts, and the merge log id.
    """
    canonical_source, redirected_src = resolve_canonical_hub(cur, source_event_id)
    canonical_target, redirected_tgt = resolve_canonical_hub(cur, target_event_id)
    if canonical_source == canonical_target:
        raise ValueError(
            f"hub pair {source_event_id} -> {target_event_id} resolves to a single canonical hub "
            f"({canonical_target}); nothing to merge"
        )

    src = _hub(cur, canonical_source)
    tgt = _hub(cur, canonical_target)
    if not src or not src["is_active"]:
        raise ValueError(f"source hub {canonical_source} is not active")
    if not tgt or not tgt["is_active"]:
        raise ValueError(f"target hub {canonical_target} is not active")

    score = _feedback_score(cur, canonical_source, canonical_target)

    cur.execute(
        "SELECT id FROM posts WHERE event_id = %s AND deleted_at IS NULL ORDER BY id;",
        (canonical_source,)
    )
    snapshot_ids = sorted(int(extract_val(r, "id", 0)) for r in (cur.fetchall() or []))
    snapshot_count = len(snapshot_ids)

    # Rebind every live member to the target hub. Canonicals stay 'assigned';
    # folded reposts keep their 'repost' status, but MUST follow their
    # canonical into the target: a repost stranded on the merged-away source id
    # would (a) break the same-hub invariant (repost_of_id -> a canonical in a
    # different event), (b) punch a hole in the topic audit (membership rooted
    # at a merged hub id), and (c) block later exact-folds (the keeper's event
    # would not match where the late copy landed). Resurrecting a repost to
    # 'assigned' would leave an assigned row carrying repost_of_id (invisible
    # to the exact-fold and never re-fixed).
    if snapshot_ids:
        cur.execute(
            """
            UPDATE posts
            SET event_id = %s,
                assignment_status = CASE WHEN repost_of_id IS NULL
                                             THEN 'assigned' ELSE 'repost' END,
                assignment_updated_at = NOW()
            WHERE event_id = %s AND deleted_at IS NULL;
            """,
            (canonical_target, canonical_source)
        )
        # Mirror the rebind into the membership ledger (close on source, open
        # on target) in the same transaction.
        move_membership(cur, snapshot_ids, canonical_target)

    # Weighted centroid combine (bounded like the internal merge job).
    new_centroid_text = None
    if src["centroid"] is not None and tgt["centroid"] is not None:
        nA = min(src["member_count"], CENTROID_MAX_MEMBERS)
        nB = min(tgt["member_count"], CENTROID_MAX_MEMBERS)
        combined = normalize((src["centroid"] * nA + tgt["centroid"] * nB) / (nA + nB))
        new_centroid_text = vector_to_array_literal(combined)
        cur.execute(
            "UPDATE event_hubs SET centroid = %s::vector, last_updated_at = NOW() WHERE id = %s;",
            (new_centroid_text, canonical_target)
        )

    # A merge may co-locate two canonicals with identical bodies (source had a
    # repost flood whose exact copy also anchored the target, or vice versa).
    # Fold them now and resync the true counts instead of summing raw sizes.
    from .fold import fold_hubs
    fold_hubs(cur, [canonical_target], near_dup=True)

    cur.execute(
        """
        UPDATE event_hubs SET is_active = FALSE, status = 'merged',
                              merged_into_id = %s, last_updated_at = NOW()
        WHERE id = %s;
        """,
        (canonical_target, canonical_source)
    )

    cur.execute(
        """
        INSERT INTO hub_merges (source_event_id, target_event_id, initiated_by, status,
                                snapshot_post_ids, snapshot_member_count, snapshot_centroid_source, opened_note)
        VALUES (%s, %s, %s, 'merged', %s::jsonb, %s, %s, %s)
        RETURNING id;
        """,
        (canonical_source, canonical_target, initiated_by,
         json.dumps(snapshot_ids), snapshot_count, src["centroid_text"], note)
    )
    merge_id = extract_val(cur.fetchone(), "id", 0)

    # Feedback trace (post_id NULL, event_id = target). Human-initiated merges
    # (client apps that force a fold, panel merge desk) are tagged as HUMAN
    # signals; only the autodetect job is 'system_auto_merge'. This keeps the
    # rollups' human-vs-system separation honest — a panel merge must never be
    # counted as system behavior.
    if initiated_by in ("client",):
        feedback_type = "user_merged_source"
    elif initiated_by in ("panel", "owner", "admin"):
        feedback_type = "owner_merge"
    else:
        feedback_type = "system_auto_merge"
    cur.execute(
        """
        INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type, actor)
        VALUES (NULL, %s, %s, %s, %s);
        """,
        (canonical_target, score, feedback_type, actor or initiated_by)
    )

    write_outbox(cur, "event.merged", None, canonical_target, {
        "event_id": canonical_target,
        "source_event_id": canonical_source,
        "redirect_to": canonical_target,
        "initiated_by": initiated_by,
        "merge_id": merge_id,
        "member_count": tgt["member_count"] + snapshot_count,
        "source": hub_reference(cur, canonical_source),
        "target": hub_reference(cur, canonical_target),
        "moved_posts": snapshot_count,
    })

    return {
        "merge_id": merge_id,
        "source_event_id": canonical_source,
        "was_redirected": redirected_src or redirected_tgt,
        "target_event_id": canonical_target,
        "member_count": tgt["member_count"] + snapshot_count,
        "source": hub_reference(cur, canonical_source),
        "target": hub_reference(cur, canonical_target),
        "moved_posts": snapshot_count,
    }


def reopen_merge(cur, merge_id: int, actor: str, note: str | None = None) -> dict:
    """Reverse a soft merge: only snapshot members rebound, fresh target
    arrivals are never disturbed (they keep their current hub assignment)."""
    cur.execute(
        "SELECT source_event_id, target_event_id, snapshot_post_ids, snapshot_member_count, "
        "snapshot_centroid_source FROM hub_merges WHERE id = %s AND status = 'merged' FOR UPDATE;",
        (merge_id,)
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"hub merge {merge_id} not found or already reopened")

    source = int(extract_val(row, "source_event_id", 0))
    target = int(extract_val(row, "target_event_id", 1))
    _snap = extract_val(row, "snapshot_post_ids", 2)
    if isinstance(_snap, str):
        snapshot_ids = json.loads(_snap or "[]")
    elif _snap is None:
        snapshot_ids = []
    else:
        snapshot_ids = list(_snap)
    snapshot_count = int(extract_val(row, "snapshot_member_count", 3) or 0)
    source_centroid = extract_val(row, "snapshot_centroid_source", 4)

    tgt = _hub(cur, target)
    if not tgt or not tgt["is_active"]:
        raise ValueError(f"target hub {target} is not active - cannot reopen merge {merge_id}")

    # Chain guard: if the source hub was merged ONWARD into a THIRD hub after
    # this merge, reopening would snap the redirect chain - source members were
    # rebased again, and resurrecting source would strand posts behind the
    # later link. This merge's own redirect (source -> target) is fine to peel.
    cur.execute(
        "SELECT COUNT(*) AS moved_on FROM event_hubs WHERE id = %s AND merged_into_id IS NOT NULL AND merged_into_id <> %s;",
        (source, target)
    )
    moved_on = int(extract_val(cur.fetchone(), "moved_on", 0) or 0)
    if moved_on:
        raise ValueError(
            f"hub {source} was merged onward after merge {merge_id}; revert that merge first"
        )

    # Rebind only snapshot posts still on the target AND still assigned
    # (never ones moved onward, and never ones the operator later unlinked).
    rebound = []
    if snapshot_ids:
        cur.execute(
            """
            UPDATE posts SET event_id = %s, assignment_updated_at = NOW()
            WHERE id = ANY(%s::int[]) AND event_id = %s AND assignment_status = 'assigned'
            RETURNING id;
            """,
            (source, snapshot_ids, target)
        )
        rebound = [int(extract_val(r, "id", 0)) for r in (cur.fetchall() or [])]
        # Rehome folded reposts WITH their canonical: a repost folded to a source
        # canonical while it lived in target; now the canonical returns to
        # source, and leaving the repost behind would strand it posting across
        # hubs (repost_of_id -> a canonical in a different event). Reposts and
        # their canonical always share a hub by the fold invariant.
        if rebound:
            cur.execute(
                "UPDATE posts SET event_id = %s, assignment_updated_at = NOW() "
                "WHERE repost_of_id = ANY(%s::int[]) AND event_id = %s AND deleted_at IS NULL "
                "AND assignment_status = 'repost';",
                (source, rebound, target)
            )
        # Mirror the rebind into the membership ledger.
        move_membership(cur, rebound, source)

    rebound_count = len(rebound)

    cur.execute(
        "UPDATE event_hubs SET member_count = GREATEST(member_count - %s, 0), last_updated_at = NOW() WHERE id = %s;",
        (rebound_count, target)
    )
    # Recompute BOTH hubs from their live members, never restore frozen snapshots.
    # Reopening changes membership on both sides (source grows, target shrinks),
    # so a stale snapshot centroid would poison the next assignment round and the
    # target's stored centroid is equally stale after the rebind.
    _recompute_hub_centroid(cur, source)
    _recompute_hub_centroid(cur, target)
    cur.execute(
        """
        UPDATE event_hubs SET is_active = TRUE, status = 'active', merged_into_id = NULL,
                              member_count = %s, last_updated_at = NOW()
        WHERE id = %s;
        """,
        (rebound_count, source)
    )
    # Resync both hubs from their live membership after the rebind.
    from .fold import sync_hub_counts
    sync_hub_counts(cur, sorted({source, target}))
    cur.execute(
        "UPDATE hub_merges SET status = 'reopened', reopened_note = %s, reopened_at = NOW() WHERE id = %s;",
        (note, merge_id)
    )

    # Record the REVERSAL as a human signal. If the original merge was the
    # autodetect job's, rollups must not keep crediting it - this row makes a
    # human-stopped system action auditable and verifiable, not self-graded.
    reversal_score = _feedback_score(cur, source, target)
    cur.execute(
        """
        INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type, actor)
        VALUES (NULL, %s, %s, 'owner_reopened_merge', %s);
        """,
        (source, reversal_score, actor)
    )

    write_outbox(cur, "event.merged_reopened", None, source, {
        "event_id": source,
        "source_event_id": source,
        "target_event_id": target,
        "merge_id": merge_id,
        "rebound_posts": rebound_count,
        "source": hub_reference(cur, source),
        "target": hub_reference(cur, target),
    })
    log_admin_action(cur, actor, "hub.merge.reopen", "hub_merges", merge_id,
                     {"source": source, "target": target},
                     {"rebound_posts": rebound_count})
    return {"merge_id": merge_id, "source_event_id": source, "rebound_posts": rebound_count,
            "source": hub_reference(cur, source), "target": hub_reference(cur, target)}


def lock_hub_pair(cur, source_event_id: int, target_event_id: int) -> tuple[int, int]:
    """pg_advisory_xact_lock on the ordered pair - serializes concurrent merges
    between the same two hubs (client b + owner b + autodetect job)."""
    lo, hi = sorted((int(source_event_id), int(target_event_id)))
    cur.execute("SELECT pg_advisory_xact_lock(%s, %s);", (MERGE_LOCK_NAMESPACE + (lo % 65536), hi))
    return lo, hi


def client_merge(source_event_id: int, target_event_id: int, actor: str | None,
                 initiated_by: str = "client") -> dict:
    """Entry point for merge endpoints (owns the txn+lock).

    ``initiated_by`` distinguishes the human surfaces ('client' = operator
    app forcing a fold, 'panel'/'owner' = control panel) from the autodetect
    job ('system') so feedback tags never misattribute a human decision to
    the system.
    """
    with get_db_cursor(commit=True) as cur:
        lock_hub_pair(cur, source_event_id, target_event_id)
        return apply_merge_soft(cur, source_event_id, target_event_id, initiated_by, actor)