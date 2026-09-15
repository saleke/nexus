"""Post-to-hub assignment logic (chunk-level SQL + pure decision helpers).

Owns every SQL statement that transitions a claimed post to a final status
(assigned / candidate / unassigned) plus its outbox/feedback/buffer writes.
Kept separate from the task orchestration in ``tasks`` so the hot path is
reviewable and independently testable.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from psycopg2.extras import execute_values

from .config import (
    SIMILARITY_MARGIN,
    CANDIDATE_THRESHOLD,
    CENTROID_UPDATE_THRESHOLD,
    ASSIGN_CHUNK_SIZE,
    BULK_ASSIGN,
)
from .chunks import iter_chunks
from .db import get_db_cursor, extract_val
from .decisions import log_decision, log_decisions_bulk
from .embed_io import vector_to_array_literal
from .events import write_outbox
from .membership import record_membership, record_memberships
from .nlp import entities_conflict
from .policy import current_versions
from .threshold import get_effective_threshold


def _hub_strong_entities(cur, hub_ids) -> dict[int, set[str]]:
    """Union of strong identity entities across a hub's member posts.

    Computed on the fly (never stored on the hub) so merge/unlink/correction
    paths cannot drift out of sync. An empty set means 'no identity evidence'
    -> the caller must not veto on that side."""
    hub_ids = [h for h in (hub_ids or []) if h is not None]
    if not hub_ids:
        return {}
    cur.execute(
        """
        SELECT DISTINCT p.event_id, x.ent
        FROM posts p
        CROSS JOIN LATERAL unnest(p.entities) AS x(ent)
        WHERE p.event_id = ANY(%s::int[]) AND p.entities IS NOT NULL
        """,
        (list(set(hub_ids)),)
    )
    out: dict[int, set[str]] = defaultdict(set)
    for r in cur.fetchall() or []:
        eid = extract_val(r, "event_id", 0)
        ent = extract_val(r, "ent", 1)
        if eid and ent:
            out[eid].add(ent)
    return out


def _post_strong_entities(cur, post_ids) -> dict[int, list[str]]:
    """Strong identity entities stored on the posts (captured at ingestion)."""
    if not post_ids:
        return {}
    cur.execute("SELECT id, entities FROM posts WHERE id = ANY(%s::int[]);", (post_ids,))
    out: dict[int, list[str]] = {}
    for r in cur.fetchall() or []:
        pid = extract_val(r, "id", 0)
        out[pid] = list(extract_val(r, "entities", 1) or [])
    return out


def _reason_code(sim: float | None, runner_sim: float | None, threshold: float, status: str) -> str:
    """Why was this final status chosen? Feeds the decisions inspector."""
    if status == "assigned":
        return "confident_assign"
    if status == "candidate":
        if runner_sim is not None and sim is not None and sim >= threshold and (sim - runner_sim) < SIMILARITY_MARGIN:
            return "margin_fail"
        return "below_threshold"
    return "below_candidate_floor"


def _margin_budget(sim: float | None, runner_sim: float | None, threshold: float) -> float | None:
    """Signed gap behind the decision: how close was the runner-up / how far
    below the assignment bar the top match really was."""
    if sim is None:
        return None
    if runner_sim is not None:
        return round(float(sim - runner_sim), 4)
    return round(float(sim - threshold), 4)


def decide_assignment(similarity: float | None, runner_similarity: float | None, threshold: float):
    """Pure decision: is the post confidently assigned to a hub?

    Returns ``(is_confident, status, confidence)``. ``event_id`` must be
    supplied by the caller when confident.
    """
    if similarity is None:
        return False, "unassigned", None

    sim = float(similarity)
    has_competitor = runner_similarity is not None
    confident = (
        sim >= threshold
        and (not has_competitor or sim - float(runner_similarity) >= SIMILARITY_MARGIN)
    )
    if confident:
        return True, "assigned", sim

    status = "candidate" if sim >= CANDIDATE_THRESHOLD else "unassigned"
    return False, status, sim


def _assign_payload_dict(event_type: str, pid: int, event_id: int | None, confidence: float | None) -> dict:
    return {
        "post_id": pid,
        "event_id": event_id,
        "status": "assigned" if event_type == "post.assigned" else event_type.split(".", 1)[1],
        "confidence": confidence,
    }


def _assign_payload(event_type: str, pid: int, event_id: int | None, confidence: float | None) -> str:
    """Legacy JSON-string form; kept as the stable outbox/API machine contract
    (tests pin its exact shape). New callers prefer ``_assign_payload_dict``
    so the outbox writer can inject ``post_seq`` into the payload object."""
    import json
    return json.dumps(_assign_payload_dict(event_type, pid, event_id, confidence))


def _bulk_match(cur, chunk_ids: list[int]) -> dict[int, list[tuple[int | None, float | None]]]:
    """Top-2 hub matches for every post via one LATERAL + HNSW statement.

    Returns ``{post_id: [(event_id, similarity), ...]}`` ordered best-first.
    A post with no active recent hub yields the single entry ``(None, None)``,
    matching the per-row query's empty match.
    """
    cur.execute(
        """
        SELECT p.id AS post_id, h.id AS event_id,
               1 - (h.centroid <=> p.embedding) AS similarity
        FROM posts p
        LEFT JOIN LATERAL (
            SELECT eh.id, eh.centroid
            FROM event_hubs eh
            WHERE eh.is_active = TRUE
              AND eh.centroid IS NOT NULL
              AND eh.last_updated_at >= NOW() - INTERVAL '48 hours'
            ORDER BY eh.centroid <=> p.embedding
            LIMIT 2
        ) h ON TRUE
        WHERE p.id = ANY(%s::int[])
        ORDER BY p.id;
        """,
        (chunk_ids,)
    )
    groups: dict[int, list[tuple[int | None, float | None]]] = defaultdict(list)
    for r in cur.fetchall() or []:
        pid = extract_val(r, "post_id", 0)
        eid = extract_val(r, "event_id", 1)
        sim = extract_val(r, "similarity", 2)
        groups[pid].append((eid, sim))
    return groups


def _bulk_update_status_assigned(cur, rows):
    if not rows:
        return []
    execute_values(
        cur,
        """
        UPDATE posts AS p
        SET event_id = v.eid, assignment_status = 'assigned',
            assignment_confidence = v.conf, assignment_updated_at = NOW()
        FROM (VALUES %s) AS v(id, eid, conf)
        WHERE p.id = v.id AND p.assignment_status = 'processing'
        RETURNING p.id
        """,
        rows,
        template="(%s::int, %s::int, %s::numeric)",
    )
    return [extract_val(r, "id", 0) for r in (cur.fetchall() or [])]


def _bulk_update_status_unassigned(cur, rows):
    if not rows:
        return []
    execute_values(
        cur,
        """
        UPDATE posts AS p
        SET assignment_status = v.st, assignment_confidence = v.conf,
            assignment_updated_at = NOW()
        FROM (VALUES %s) AS v(id, st, conf)
        WHERE p.id = v.id AND p.assignment_status = 'processing'
        RETURNING p.id
        """,
        rows,
        template="(%s::int, %s::text, %s::numeric)",
    )
    return [extract_val(r, "id", 0) for r in (cur.fetchall() or [])]


def _reclaim_chunk_posts(chunk_ids: list[int]):
    """Return posts back to 'pending' after a failed chunk (committed, so a
    retry/sweep re-claims them). Never marks processing errors as 'noise'."""
    if not chunk_ids:
        return
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE posts
            SET assignment_status = 'pending', assignment_updated_at = NOW()
            WHERE id = ANY(%s::int[]) AND assignment_status = 'processing';
            """,
            (chunk_ids,)
        )


def _assign_chunk_bulk(item_chunk, vec_chunk, *, assigned_count, hub_centroid_updates, hub_membership_increments):
    """Bulk assignment for one chunk: ~4 statements instead of ~6 per post.

    Isolation granularity is the chunk. On failure the chunk is rolled back and
    its posts returned to ``pending`` so a retry re-claims them - an error is
    never mislabeled as ``noise``.
    """
    chunk_ids = [item["id"] for item in item_chunk]
    try:
        with get_db_cursor(commit=True) as cur:
            threshold = get_effective_threshold(cur)

            # 1) Persist embeddings (idempotent: deterministic from content)
            execute_values(
                cur,
                """
                UPDATE posts AS p
                SET embedding = v.emb::vector
                FROM (VALUES %s) AS v(id, emb)
                WHERE p.id = v.id
                """,
                [(item["id"], vector_to_array_literal(vec)) for item, vec in zip(item_chunk, vec_chunk)],
                template="(%s::int, %s::vector)",
            )

            # 2) Bulk HNSW match - one round trip for the whole chunk
            groups = _bulk_match(cur, chunk_ids)

            # Preload entity-structure signals for the whole chunk: identity
            # entities of the matched hubs (so the veto is one query, not
            # one per post) and the chunk posts' own stored entities.
            matched_hub_ids = {eid for vals in groups.values() for eid, _ in vals if eid}
            hub_entities = _hub_strong_entities(cur, matched_hub_ids)
            post_entities = _post_strong_entities(cur, chunk_ids)

            decisions = []  # (pid, vector, is_assigned, event_id, status, confidence, sim, runner_sim, reason)
            for item, vector in zip(item_chunk, vec_chunk):
                pid = item["id"]
                matches = groups.get(pid, [])
                best = matches[0] if matches else (None, None)
                best_eid, sim = best
                runner_sim = matches[1][1] if len(matches) > 1 else None
                is_assigned, status, confidence = decide_assignment(sim, runner_sim, threshold)
                event_id = best_eid if is_assigned else None
                reason_override = None
                if is_assigned and entities_conflict(
                    post_entities.get(pid, []),
                    hub_entities.get(best_eid, set()),
                ):
                    # Embeddings cannot separate confusable actor threads
                    # (e.g. Apple vs Samsung launch) - identical semantics,
                    # different brands. Strong identity entities resolve it;
                    # the post is left out (candidate -> buffer) rather than
                    # silently co-clustered with a different actor.
                    is_assigned = False
                    status = "candidate"
                    event_id = None
                    reason_override = "entity_conflict"
                decisions.append((pid, vector, is_assigned, event_id, status, confidence, sim, runner_sim, reason_override))

            # 3) Status writes guarded so we only finalize posts we still own.
            pv, mv = current_versions(cur)
            assigned_raw = [(pid, event_id, confidence) for pid, _, is_assigned, event_id, _, confidence, _, _, _ in decisions if is_assigned]
            assigned_updated = _bulk_update_status_assigned(cur, assigned_raw)
            unassigned_raw = [(pid, status, confidence) for pid, _, is_assigned, _, status, confidence, _, _, _ in decisions if not is_assigned]
            unassigned_updated = _bulk_update_status_unassigned(cur, unassigned_raw)

            # 4) Outbox events / feedback / buffer / decision journal only for
            #    rows we actually updated (guarded against lost claims).
            assigned_ids = set(assigned_updated)
            unassigned_ids = set(unassigned_updated)

            # Mirror every confirmed assignment into the membership ledger in
            # the same transaction as the event_id pointer (one batch).
            record_memberships(cur, [
                (pid, event_id, "member")
                for pid, _, is_assigned, event_id, _, _, _, _, _ in decisions
                if is_assigned and pid in assigned_ids
            ])

            outbox_rows = []
            feedback_rows = []
            decision_rows = []
            for pid, vector, is_assigned, event_id, status, confidence, sim, runner_sim, reason_override in decisions:
                if is_assigned:
                    if pid not in assigned_ids:
                        continue
                    event_type = "post.assigned"
                else:
                    if pid not in unassigned_ids:
                        continue
                    event_type = f"post.{status}"
                    event_id = None
                outbox_rows.append(
                    (event_type, pid, event_id, _assign_payload_dict(event_type, pid, event_id, confidence))
                )
                reason = reason_override or _reason_code(sim, runner_sim, threshold, status)
                decision_rows.append(
                    (pid, event_id, sim, runner_sim, threshold,
                     _margin_budget(sim, runner_sim, threshold), status, confidence,
                     pv, mv, reason)
                )
                if is_assigned:
                    feedback_rows.append((pid, event_id, confidence))
                    if sim >= CENTROID_UPDATE_THRESHOLD:
                        hub_centroid_updates[event_id].append(vector)
                    hub_membership_increments[event_id] += 1
                    assigned_count += 1

            if outbox_rows:
                from .events import bump_post_seqs, _seq_payload
                outbox_seqs = bump_post_seqs(cur, [r[1] for r in outbox_rows])
                outbox_rows = [
                    (r[0], r[1], r[2], json.dumps(_seq_payload(r[3], outbox_seqs.get(r[1]))))
                    for r in outbox_rows
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO integration_outbox (event_type, post_id, event_id, payload, schema_version)
                    SELECT v.etype, v.pid, v.eid, v.payload::jsonb, 2
                    FROM (VALUES %s) AS v(etype, pid, eid, payload)
                    ON CONFLICT (post_id, event_type) WHERE post_id IS NOT NULL
                    DO UPDATE SET event_id = EXCLUDED.event_id, payload = EXCLUDED.payload,
                                  schema_version = 2,
                                  delivery_status = 'pending', attempts = 0, available_at = NOW()
                    """,
                    outbox_rows,
                    template="(%s::text, %s::int, %s::int, %s::jsonb)",
                )
            if feedback_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type)
                    SELECT v.pid, v.eid, v.conf, 'auto_confirmed'
                    FROM (VALUES %s) AS v(pid, eid, conf)
                    """,
                    feedback_rows,
                    template="(%s::int, %s::int, %s::numeric)",
                )
            if decision_rows:
                log_decisions_bulk(cur, decision_rows)
            if unassigned_ids:
                cur.execute(
                    """
                    INSERT INTO unclustered_posts_buffer (post_id, embedding)
                    SELECT id, embedding FROM posts WHERE id = ANY(%s::int[])
                    ON CONFLICT (post_id) DO NOTHING;
                    """,
                    (sorted(unassigned_ids),)
                )
    except Exception as exc:
        _reclaim_chunk_posts(chunk_ids)
        print(f"[Worker] Assignment chunk {chunk_ids[0]}..{chunk_ids[-1]} failed, re-queued: {exc}")
        raise
    return assigned_count


def _assign_chunk_rowwise(item_chunk, vec_chunk, *, assigned_count, hub_centroid_updates, hub_membership_increments):
    """Legacy per-post assignment path, kept as flag-gated fallback
    (BULK_ASSIGN=0) if the LATERAL matcher fails to use the HNSW index."""
    with get_db_cursor(commit=True) as cur:
        threshold = get_effective_threshold(cur)
        pv, mv = current_versions(cur)
        for item, vector in zip(item_chunk, vec_chunk):
            pid = item["id"]
            vector_str = vector_to_array_literal(vector)

            try:
                cur.execute("SAVEPOINT post_tx;")
                cur.execute("UPDATE posts SET embedding = %s WHERE id = %s;", (vector_str, pid))

                query_active = """
                    SELECT eh.id AS event_id,
                           eh.member_count,
                           (1 - (eh.centroid <=> %s::vector)) AS cosine_similarity
                    FROM event_hubs eh
                    WHERE eh.is_active = TRUE
                      AND eh.centroid IS NOT NULL
                      AND eh.last_updated_at >= NOW() - INTERVAL '48 hours'
                    ORDER BY eh.centroid <=> %s::vector
                    LIMIT 2;
                """
                cur.execute(query_active, (vector_str, vector_str))
                matches = cur.fetchall()

                match = matches[0] if matches else None
                event_id = extract_val(match, "event_id", 0)
                similarity = extract_val(match, "cosine_similarity", 2)
                second_similarity = extract_val(matches[1], "cosine_similarity", 2) if len(matches) > 1 else -1.0

                is_assigned, status, confidence = decide_assignment(
                    similarity,
                    second_similarity if second_similarity != -1.0 else None,
                    threshold,
                )
                reason_override = None

                # Entity-conflict veto (mirror of the bulk path): if the best
                # hub carries strong identity entities disjoint from this post's,
                # the similarity match is an embedding ambiguity, not an event
                # match - leave the post out (candidate) rather than mis-assign.
                if is_assigned and event_id:
                    cur.execute("SELECT entities FROM posts WHERE id = %s;", (pid,))
                    pe_ent_row = cur.fetchone()
                    pe = list(extract_val(pe_ent_row, "entities", 0) or [])
                    he = _hub_strong_entities(cur, [event_id]).get(event_id, set())
                    if entities_conflict(pe, he):
                        is_assigned = False
                        status = "candidate"
                        event_id = None
                        reason_override = "entity_conflict"
                    else:
                        reason_override = None

                if is_assigned:
                    cur.execute(
                        """
                        UPDATE posts
                        SET event_id = %s, assignment_status = 'assigned',
                            assignment_confidence = %s, assignment_updated_at = NOW()
                        WHERE id = %s AND assignment_status = 'processing';
                        """,
                        (event_id, confidence, pid)
                    )
                    write_outbox(cur, "post.assigned", pid, event_id, {
                        "post_id": pid,
                        "event_id": event_id,
                        "confidence": confidence,
                        "status": "assigned"
                    })
                    record_membership(cur, pid, event_id)
                    cur.execute(
                        """
                        INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type)
                        VALUES (%s, %s, %s, 'auto_confirmed');
                        """,
                        (pid, event_id, confidence)
                    )
                    if confidence >= CENTROID_UPDATE_THRESHOLD:
                        hub_centroid_updates[event_id].append(np.asarray(vector))
                    hub_membership_increments[event_id] += 1
                    assigned_count += 1
                    log_decision(cur, post_id=pid, event_id=event_id, similarity=similarity,
                                 runner_similarity=second_similarity if second_similarity != -1.0 else None,
                                 threshold_used=threshold, margin_budget=_margin_budget(
                                     similarity, second_similarity if second_similarity != -1.0 else None, threshold),
                                 status=status, confidence=confidence, policy_version=pv,
                                 model_version=mv, reason=reason_override or _reason_code(similarity, second_similarity, threshold, status))
                else:
                    cur.execute(
                        """
                        UPDATE posts
                        SET assignment_status = %s, assignment_confidence = %s, assignment_updated_at = NOW()
                        WHERE id = %s AND assignment_status = 'processing';
                        """,
                        (status, confidence, pid)
                    )
                    write_outbox(cur, f"post.{status}", pid, None, {
                        "post_id": pid,
                        "status": status,
                        "confidence": confidence
                    })
                    cur.execute(
                        """
                        INSERT INTO unclustered_posts_buffer (post_id, embedding)
                        VALUES (%s, %s::vector)
                        ON CONFLICT (post_id) DO NOTHING;
                        """,
                        (pid, vector_str)
                    )
                    log_decision(cur, post_id=pid, event_id=None, similarity=similarity,
                                 runner_similarity=second_similarity if second_similarity != -1.0 else None,
                                 threshold_used=threshold, margin_budget=_margin_budget(
                                     similarity, second_similarity if second_similarity != -1.0 else None, threshold),
                                 status=status, confidence=confidence, policy_version=pv,
                                 model_version=mv, reason=reason_override or _reason_code(similarity, second_similarity, threshold, status))

                cur.execute("RELEASE SAVEPOINT post_tx;")
            except Exception as row_exc:
                cur.execute("ROLLBACK TO SAVEPOINT post_tx;")
                cur.execute(
                    "UPDATE posts SET assignment_status = 'pending', assignment_updated_at = NOW() WHERE id = %s;",
                    (pid,)
                )
                print(f"[Worker] Isolated poison pill post #{pid}: {row_exc}")
    return assigned_count


def assign_valid_for_encoding(valid_for_encoding, embeddings):
    """Run the match & assign phase for a batch of encoded posts.

    Returns ``(assigned_count, hub_centroid_updates, hub_membership_increments)``.
    """
    hub_centroid_updates: dict[int, list[np.ndarray]] = defaultdict(list)
    hub_membership_increments: dict[int, int] = defaultdict(int)
    assigned_count = 0

    for item_chunk, start, end in iter_chunks(valid_for_encoding, ASSIGN_CHUNK_SIZE):
        vec_chunk = embeddings[start:end]
        if BULK_ASSIGN:
            assigned_count = _assign_chunk_bulk(
                item_chunk, vec_chunk,
                assigned_count=assigned_count,
                hub_centroid_updates=hub_centroid_updates,
                hub_membership_increments=hub_membership_increments,
            )
        else:
            assigned_count = _assign_chunk_rowwise(
                item_chunk, vec_chunk,
                assigned_count=assigned_count,
                hub_centroid_updates=hub_centroid_updates,
                hub_membership_increments=hub_membership_increments,
            )
    return assigned_count, hub_centroid_updates, hub_membership_increments