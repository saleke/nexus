import os
from collections import defaultdict
import numpy as np
import torch

from .queues import celery_app, distributed_task_lock
from .db import get_db_cursor
from .nlp import is_clusterable, cleanse_text, is_worth_seeing
from .models import get_embedding_engine
from .events import write_outbox
from .config import (
    SIMILARITY_MARGIN, AUTO_ASSIGN_THRESHOLD, CANDIDATE_THRESHOLD,
    CENTROID_UPDATE_THRESHOLD, CENTROID_MAX_MEMBERS, ANCHOR_WEIGHT,
)


def extract_val(row, key: str, idx: int):
    if not row:
        return None
    if isinstance(row, dict):
        return row.get(key)
    if isinstance(row, (list, tuple)):
        return row[idx]
    return row


@celery_app.task(
    name="post_clustering_pipeline.tasks.process_post_batch_ingestion",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
)
def process_post_batch_ingestion(posts: list[dict]):
    """Vectorized batch post ingestion with row-level SAVEPOINT poison-pill isolation."""
    if not posts:
        return {"status": "skipped", "count": 0}

    valid_for_encoding = []
    skipped_count = 0
    assigned_count = 0

    with get_db_cursor(commit=True) as cur:
        # Phase 1: Filter clusterability/discourse & atomic CAS claim
        for p in posts:
            pid = p["id"]
            content = p["content"]

            cur.execute(
                """
                UPDATE posts 
                SET assignment_status = 'processing', assignment_updated_at = NOW() 
                WHERE id = %s AND assignment_status IN ('pending', 'processing')
                RETURNING id;
                """,
                (pid,)
            )
            claimed = cur.fetchone()
            if not claimed:
                continue

            worth_seeing, reason = is_worth_seeing(content)
            if not worth_seeing:
                cur.execute(
                    "UPDATE posts SET assignment_status = 'noise', assignment_updated_at = NOW() WHERE id = %s;",
                    (pid,)
                )
                write_outbox(cur, "post.noise", pid, None, {
                    "post_id": pid,
                    "status": "noise",
                    "reason": reason
                })
                skipped_count += 1
                continue

            cleaned = cleanse_text(content)
            valid_for_encoding.append({"id": pid, "content": content, "cleaned": cleaned})

        if not valid_for_encoding:
            return {"status": "success", "processed": len(posts), "assigned": 0, "skipped": skipped_count}

        # Phase 2: Batch Vector Encoding in PyTorch
        texts_to_encode = [item["cleaned"] for item in valid_for_encoding]
        embeddings = get_embedding_engine().encode_batch(texts_to_encode, batch_size=64)

        # Query dynamic threshold
        cur.execute("SELECT value FROM system_config WHERE key = 'global_similarity_threshold';")
        config_row = cur.fetchone()
        raw_val = extract_val(config_row, "value", 0)
        threshold = max(float(raw_val) if raw_val is not None else AUTO_ASSIGN_THRESHOLD, AUTO_ASSIGN_THRESHOLD)

        # In-memory pre-aggregation container for centroid updates
        hub_centroid_updates: dict[int, list[np.ndarray]] = defaultdict(list)
        hub_membership_increments: dict[int, int] = defaultdict(int)

        # Phase 3: Match & Assign with SAVEPOINT isolation per post
        for idx, item in enumerate(valid_for_encoding):
            pid = item["id"]
            vector = embeddings[idx]
            vector_str = f"[{','.join(map(str, vector))}]"

            try:
                cur.execute("SAVEPOINT post_tx;")
                cur.execute("UPDATE posts SET embedding = %s WHERE id = %s;", (vector_str, pid))

                # Query active event hubs using partial HNSW index
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

                has_competitor = len(matches) > 1
                confident = (
                    match is not None
                    and similarity is not None
                    and float(similarity) >= threshold
                    and (not has_competitor or float(similarity) - float(second_similarity) >= SIMILARITY_MARGIN)
                )

                if confident:
                    sim_float = float(similarity)
                    cur.execute(
                        """
                        UPDATE posts 
                        SET event_id = %s, assignment_status = 'assigned', 
                            assignment_confidence = %s, assignment_updated_at = NOW() 
                        WHERE id = %s;
                        """,
                        (event_id, sim_float, pid)
                    )
                    write_outbox(cur, "post.assigned", pid, event_id, {
                        "post_id": pid,
                        "event_id": event_id,
                        "confidence": sim_float,
                        "status": "assigned"
                    })

                    cur.execute(
                        """
                        INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type)
                        VALUES (%s, %s, %s, 'auto_confirmed');
                        """,
                        (pid, event_id, sim_float)
                    )

                    # Stage centroid update in-memory
                    if sim_float >= CENTROID_UPDATE_THRESHOLD:
                        hub_centroid_updates[event_id].append(np.array(vector))
                    hub_membership_increments[event_id] += 1
                    assigned_count += 1
                else:
                    # Precision-first: If confidence is not rock-solid, mark as candidate or unassigned
                    status_value = 'candidate' if similarity is not None and float(similarity) >= CANDIDATE_THRESHOLD else 'unassigned'
                    cur.execute(
                        """
                        UPDATE posts 
                        SET assignment_status = %s, assignment_confidence = %s, assignment_updated_at = NOW() 
                        WHERE id = %s;
                        """,
                        (status_value, float(similarity) if similarity is not None else None, pid)
                    )
                    write_outbox(cur, f"post.{status_value}", pid, None, {
                        "post_id": pid,
                        "status": status_value,
                        "confidence": float(similarity) if similarity is not None else None
                    })
                    cur.execute(
                        """
                        INSERT INTO unclustered_posts_buffer (post_id, embedding)
                        VALUES (%s, %s::vector)
                        ON CONFLICT (post_id) DO NOTHING;
                        """,
                        (pid, vector_str)
                    )

                cur.execute("RELEASE SAVEPOINT post_tx;")
            except Exception as row_exc:
                cur.execute("ROLLBACK TO SAVEPOINT post_tx;")
                cur.execute(
                    "UPDATE posts SET assignment_status = 'noise', assignment_updated_at = NOW() WHERE id = %s;",
                    (pid,)
                )
                print(f"[Worker] Isolated poison pill post #{pid}: {row_exc}")

        # Phase 4: Execute In-Memory Aggregated Centroid Updates with Anchor Protection (Valid PostgreSQL pgvector)
        for eid, new_vectors in hub_centroid_updates.items():
            cur.execute(
                """
                SELECT centroid::text, member_count, anchor_post_id
                FROM event_hubs
                WHERE id = %s
                FOR UPDATE;
                """,
                (eid,)
            )
            hub_row = cur.fetchone()
            if not hub_row:
                continue

            c_text = extract_val(hub_row, "centroid", 0)
            old_count = extract_val(hub_row, "member_count", 1) or 1
            anchor_pid = extract_val(hub_row, "anchor_post_id", 2)

            k = len(new_vectors)
            sum_new = np.sum(new_vectors, axis=0)

            n_eff = min(old_count, CENTROID_MAX_MEMBERS)
            if c_text:
                old_centroid = np.array(list(map(float, c_text.strip("[]").split(","))))
                c_rolling = (old_centroid * n_eff + sum_new) / (n_eff + k)
            else:
                c_rolling = sum_new / k

            # Anchor protection: preserve catalyst direction for large events
            anchor_vec = None
            if anchor_pid:
                cur.execute("SELECT embedding::text FROM posts WHERE id = %s;", (anchor_pid,))
                anc_row = cur.fetchone()
                anc_text = extract_val(anc_row, "embedding", 0)
                if anc_text:
                    anchor_vec = np.array(list(map(float, anc_text.strip("[]").split(","))))

            if anchor_vec is not None:
                c_effective = ANCHOR_WEIGHT * anchor_vec + (1.0 - ANCHOR_WEIGHT) * c_rolling
            else:
                c_effective = c_rolling

            c_norm = c_effective / (np.linalg.norm(c_effective) + 1e-9)
            updated_str = f"[{','.join(map(str, c_norm.tolist()))}]"

            cur.execute(
                """
                UPDATE event_hubs
                SET centroid = %s::vector,
                    member_count = member_count + %s,
                    last_updated_at = NOW()
                WHERE id = %s;
                """,
                (updated_str, k, eid)
            )

        # For hubs that had matches below CENTROID_UPDATE_THRESHOLD, increment counts
        for eid, count in hub_membership_increments.items():
            if eid not in hub_centroid_updates:
                cur.execute(
                    "UPDATE event_hubs SET member_count = member_count + %s, last_updated_at = NOW() WHERE id = %s;",
                    (count, eid)
                )

    return {
        "status": "success",
        "processed": len(posts),
        "assigned": assigned_count,
        "skipped": skipped_count
    }


@celery_app.task(
    name="post_clustering_pipeline.tasks.process_post_ingestion",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
)
def process_post_ingestion(post_id: int, content: str):
    """Single post ingestion wrapper delegating to batch processor."""
    return process_post_batch_ingestion([{"id": post_id, "content": content}])


@celery_app.task(name="post_clustering_pipeline.tasks.reconcile_pending_posts")
def reconcile_pending_posts(batch_limit: int = 200):
    """Reconcile orphaned posts stuck in processing status longer than 2 minutes."""
    reconciled = 0
    with get_db_cursor() as cur:
        cur.execute(
            """
            SELECT id, content FROM posts
            WHERE assignment_status = 'processing'
              AND assignment_updated_at < NOW() - INTERVAL '2 minutes'
            ORDER BY id ASC
            LIMIT %s;
            """,
            (batch_limit,)
        )
        rows = cur.fetchall()
        if rows:
            batch_data = [{"id": extract_val(r, "id", 0), "content": extract_val(r, "content", 1)} for r in rows]
            celery_app.send_task("post_clustering_pipeline.tasks.process_post_batch_ingestion", args=[batch_data])
            reconciled = len(batch_data)
    return {"reconciled": reconciled}


@celery_app.task(name="post_clustering_pipeline.tasks.run_event_birth_scheduled")
def run_event_birth_scheduled():
    """Trigger periodic event birth job with distributed lock protection."""
    with distributed_task_lock("event_birth_job", timeout=3600) as acquired:
        if not acquired:
            return {"status": "skipped", "reason": "Job already running"}
        from .jobs.event_birth import run_clustering_pipeline
        run_clustering_pipeline()
        return {"status": "completed"}


@celery_app.task(name="post_clustering_pipeline.tasks.run_hub_merge_scheduled")
def run_hub_merge_scheduled():
    """Trigger periodic hub reconciliation with distributed lock protection."""
    with distributed_task_lock("hub_merge_job", timeout=1800) as acquired:
        if not acquired:
            return {"status": "skipped", "reason": "Job already running"}
        from .jobs.merge_hubs import reconcile_hub_merges
        count = reconcile_hub_merges()
        return {"status": "completed", "merged_hubs": count}


@celery_app.task(name="post_clustering_pipeline.tasks.prune_delivered_outbox")
def prune_delivered_outbox(retention_days: int = 7):
    """Prune acknowledged integration events older than retention period."""
    with get_db_cursor() as cur:
        cur.execute(
            """
            DELETE FROM integration_outbox 
            WHERE delivery_status = 'delivered' 
              AND delivered_at < NOW() - (%s * INTERVAL '1 day');
            """,
            (retention_days,)
        )
        deleted = cur.rowcount
    return {"pruned_outbox_events": deleted}


@celery_app.task(name="post_clustering_pipeline.tasks.dispatch_webhooks_scheduled")
def dispatch_webhooks_scheduled():
    """Periodically push pending outbox events to the configured webhook endpoint."""
    from .dispatch import dispatch_pending_webhooks
    count = dispatch_pending_webhooks(limit=50)
    return {"dispatched_webhooks": count}
