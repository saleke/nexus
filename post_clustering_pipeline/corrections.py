"""Shared, guarded correction paths (unlink / confirm).

The client API and the control-panel desk execute the SAME transaction logic
through these functions - only the transport differs (JSON for the API, HTML
form for the panel). Guards are expressed here, once:
  * canonical hub resolution (never write to a redirected hub),
  * scoped updates (never mutate a post that no longer sits in the stated hub),
  * feedback tagged with the policy/model in effect,
  * outbox events carrying content references for the operator's clients.
"""
from __future__ import annotations

from .db import get_db_cursor, extract_val
from .events import write_outbox
from .membership import close_membership, record_membership
from .policy import current_versions
from .refs import hub_reference, post_reference
from .merges import resolve_canonical_hub


class CorrectionError(ValueError):
    pass


def unlink_post(cur, post_id: int, event_id: int, actor: str = "user",
                detailed: bool = True) -> dict:
    """Remove a post from a hub and set it to 'candidate' (gradeable).

    Runs inside the caller's transaction. Returns references + similarity so
    both transports can render the same human-facing confirmation.
    """
    cur.execute("SELECT id, repost_of_id FROM posts WHERE id = %s AND deleted_at IS NULL;", (post_id,))
    row = cur.fetchone()
    if not row:
        raise CorrectionError(f"post #{post_id} does not exist")
    if extract_val(row, "repost_of_id", 1) is not None:
        # A folded repost is represented by its canonical; unsetting it would
        # resurrect an assigned row carrying repost_of_id and re-cluster a post
        # whose content is byte-identical to its canonical. Operators unlink
        # the canonical instead.
        raise CorrectionError(f"post #{post_id} is a folded repost; unlink its canonical first")

    canonical_id, redirected = resolve_canonical_hub(cur, event_id)
    cur.execute("SELECT id FROM event_hubs WHERE id = %s;", (canonical_id,))
    if not cur.fetchone():
        raise CorrectionError(f"hub #{event_id} does not exist")

    # Similarity to the hub CENTROID - the same metric the assignment path
    # used - so the recorded score is comparable, not a sibling-member proxy.
    cur.execute(
        """
        SELECT (1 - (p.embedding <=> eh.centroid)) AS similarity
        FROM posts p, event_hubs eh
        WHERE p.id = %s AND eh.id = %s;
        """,
        (post_id, canonical_id)
    )
    row = cur.fetchone()
    raw_sim = extract_val(row, "similarity", 0)
    sim_score = float(raw_sim) if raw_sim is not None else 0.0

    cur.execute(
        """
        UPDATE posts
        SET event_id = NULL, assignment_status = 'candidate',
            assignment_confidence = NULL, assignment_updated_at = NOW()
        WHERE id = %s AND event_id = %s AND repost_of_id IS NULL
        RETURNING id;
        """,
        (post_id, canonical_id)
    )
    if not cur.fetchone():
        # The post left this hub between the operator's read and now (e.g. a
        # concurrent merge moved it). Refuse instead of unlinking from a hub
        # the operator never saw.
        raise CorrectionError(f"post #{post_id} is not in hub #{canonical_id} anymore")

    # Membership count is authoritative for centroid weighting, merge combine
    # math, and the panel's labels - it must never drift from the posts table.
    cur.execute(
        "UPDATE event_hubs SET member_count = GREATEST(member_count - 1, 0), last_updated_at = NOW() WHERE id = %s;",
        (canonical_id,)
    )

    # Mirror the unlink in the membership ledger (close the live row).
    close_membership(cur, post_id)

    # Return the post to the birth queue: 'candidate' posts become eligible for
    # re-clustering by the hourly event-birth job, exactly like pipeline-produced
    # candidates. Without this, unlinked posts would sit as stranded candidates.
    cur.execute(
        """
        INSERT INTO unclustered_posts_buffer (post_id, embedding)
        SELECT id, embedding FROM posts WHERE id = %s
        ON CONFLICT (post_id) DO NOTHING;
        """,
        (post_id,)
    )

    pv, mv = current_versions(cur)
    cur.execute(
        """
        INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type, actor, model_version, policy_version)
        VALUES (%s, %s, %s, 'user_removed', %s, %s, %s);
        """,
        (post_id, canonical_id, sim_score, actor, mv, pv)
    )
    write_outbox(cur, "post.unlinked", post_id, canonical_id, {
        "post_id": post_id,
        "previous_event_id": canonical_id,
        "status": "unlinked",
        "post": post_reference(cur, post_id),
        "previous_hub": hub_reference(cur, canonical_id),
    })

    result = {
        "status": "unlinked",
        "post_id": post_id,
        "event_id": canonical_id,
        "requested_event_id": event_id,
        "was_redirected": redirected,
        "similarity": sim_score,
    }
    if detailed:
        result["post"] = post_reference(cur, post_id)
        result["hub"] = hub_reference(cur, canonical_id)
    return result


def confirm_post(cur, post_id: int, event_id: int, actor: str = "user",
                 detailed: bool = True) -> dict:
    """Pin a post into a hub as a human-confirmed assignment."""
    # Lock the row before reading so the hub bookkeeping below reflects the
    # post's real prior hub even under a concurrent confirm/assign/delete:
    # with FOR UPDATE the second writer blocks until the first commits, then
    # reads the committed event_id (no CAS drift -> no double decrement).
    cur.execute("SELECT event_id, assignment_confidence, assignment_status, repost_of_id FROM posts WHERE id = %s AND deleted_at IS NULL FOR UPDATE", (post_id,))
    row = cur.fetchone()
    if not row:
        raise CorrectionError("post not found")
    old_event = extract_val(row, "event_id", 0)
    old_status = extract_val(row, "assignment_status", 2)
    confidence = extract_val(row, "assignment_confidence", 1) or 0
    if extract_val(row, "repost_of_id", 3) is not None:
        raise CorrectionError(f"post #{post_id} is a folded repost; confirm its canonical instead")

    canonical_id, redirected = resolve_canonical_hub(cur, event_id)
    cur.execute("SELECT id FROM event_hubs WHERE id = %s AND is_active = TRUE;", (canonical_id,))
    if not cur.fetchone():
        raise CorrectionError("target hub not found or inactive")

    cur.execute(
        "UPDATE posts SET event_id = %s, assignment_status = 'assigned', assignment_updated_at = NOW() "
        "WHERE id = %s AND repost_of_id IS NULL",
        (canonical_id, post_id)
    )
    # Keep member_count in lockstep with the posts table:
    #   * crossing hubs: source loses one, target gains one;
    #   * fresh hub entry (post had no hub): target gains one;
    #   * same-hub reconfirmation: nothing to do.
    if old_event is not None and old_event != canonical_id:
        cur.execute(
            "UPDATE event_hubs SET member_count = GREATEST(member_count - 1, 0), last_updated_at = NOW() WHERE id = %s;",
            (old_event,)
        )
    if old_event != canonical_id:
        cur.execute(
            "UPDATE event_hubs SET member_count = member_count + 1, last_updated_at = NOW() WHERE id = %s;",
            (canonical_id,)
        )

    # A confirmed post leaves the birth queue (assigned members never re-cluster).
    if old_status != "assigned" or (old_event is not None and old_event != canonical_id):
        cur.execute(
            "DELETE FROM unclustered_posts_buffer WHERE post_id = %s;",
            (post_id,)
        )

    # Mirror the confirm in the membership ledger (cross-hub moves close the
    # old live row automatically).
    record_membership(cur, post_id, canonical_id)

    pv, mv = current_versions(cur)
    cur.execute(
        """INSERT INTO clustering_feedback_log
        (post_id, event_id, initial_similarity_score, feedback_type, actor, model_version, policy_version)
        VALUES (%s, %s, %s, 'user_confirmed', %s, %s, %s)""",
        (post_id, canonical_id, confidence, actor, mv, pv)
    )
    write_outbox(
        cur, "post.confirmed", post_id, canonical_id,
        {"post_id": post_id, "event_id": canonical_id, "requested_event_id": event_id,
         "status": "assigned", "confidence": float(confidence),
         "post": post_reference(cur, post_id), "hub": hub_reference(cur, canonical_id)}
    )

    result = {
        "status": "confirmed",
        "post_id": post_id,
        "event_id": canonical_id,
        "requested_event_id": event_id,
        "was_redirected": redirected,
        "previous_event_id": old_event,
    }
    if detailed:
        result["post"] = post_reference(cur, post_id)
        result["hub"] = hub_reference(cur, canonical_id)
    return result