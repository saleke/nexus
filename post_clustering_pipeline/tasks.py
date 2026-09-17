from .queues import celery_app, distributed_task_lock, get_redis_client
from .db import get_db_cursor, extract_val
from .nlp import cleanse_text, analyze_discourse_batch
from .models import get_embedding_engine
from .events import write_outbox_bulk
from .embed_io import vector_to_array_literal, parse_vector_literal
from .chunks import iter_chunks
from .assignment import assign_valid_for_encoding, resolve_stale_candidates
from .decisions import log_decisions_bulk
from .centroid import bounded_rolling_centroid, anchor_blended_centroid, normalize
from .policy import current_versions
from .config import (
    CENTROID_MAX_MEMBERS,
    ANCHOR_WEIGHT,
    CLAIM_CHUNK_SIZE,
    CLAIM_DURABLE,
    INGEST_HINTS_KEY,
)

PROCESSING_STALE_SECONDS = 120
PENDING_STALL_SECONDS = 900  # 15 min: a post this old in 'pending' is a delivery anomaly


@celery_app.task(
    name="post_clustering_pipeline.tasks.process_post_batch_ingestion",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
)
def process_post_batch_ingestion(posts: list[dict]):
    """Vectorized batch post ingestion with idempotent claiming and chunked commits.

    Orchestration only: claim + discourse gate (Phase 1), batch encoding
    (Phase 2), matcher/assignment (Phase 3, delegated to ``assignment``), and
    aggregated centroid updates (Phase 4).

    Idempotency contract:
      * Only ``pending`` posts are claimable; an already-final post is never
        reprocessed, so a retried task performs no double assignment.
      * Progress commits per chunk, so a mid-batch crash keeps finished chunks
        durable; uncommitted posts remain ``processing`` and are reclaimed by
        ``reconcile_pending_posts`` (reset -> dispatch).
      * Outbox events are written in the same transaction as the status change
        they describe.

    Durability: Phase 1 claims are derived state guarded transaction-atomically
    with their gating result, so they commit with ``synchronous_commit = off``
    unless ``CLAIM_DURABLE`` forces full durability. Phase 3 (assignment + its
    contract writes) and Phase 4 (centroids, the assignment phase's semantic
    tail) always commit with full durability.
    """
    if not posts:
        return {"status": "skipped", "count": 0}

    valid_for_encoding = []
    skipped_count = 0
    entity_rows = []

    # Phase 1: CAS claim (pending -> processing) + discourse gating.
    # Commit per chunk so claimed/gated work survives process death without
    # holding a single giant transaction for the whole batch.
    for claim_chunk, _, _ in iter_chunks(posts, CLAIM_CHUNK_SIZE):
        with get_db_cursor(commit=True, durable=CLAIM_DURABLE) as cur:
            pv, mv = current_versions(cur)

            # Single-statement CAS claim for the whole chunk: one round-trip
            # instead of one per post. Rows already claimed/finalized by a
            # concurrent attempt are simply not returned, preserving the same
            # "skip" semantics as the old per-post CAS.
            cur.execute(
                """
                UPDATE posts
                SET assignment_status = 'processing', assignment_updated_at = NOW()
                WHERE id = ANY(%s::int[])
                  AND assignment_status = 'pending'
                RETURNING id;
                """,
                ([p["id"] for p in claim_chunk],)
            )
            claimed_ids = {int(extract_val(r, "id", 0)) for r in (cur.fetchall() or [])}
            claimed = [p for p in claim_chunk if p["id"] in claimed_ids]
            if not claimed:
                continue

            # Bulk discourse gate: verdicts are byte-identical to per-post
            # analyze_discourse calls (nlp.pipe is a pure throughput change),
            # so admission/noise outcomes are unchanged.
            verdicts = analyze_discourse_batch([p["content"] for p in claimed])
            noise_rows = []
            batch_entities = []
            for item, (worth_seeing, reason, entities) in zip(claimed, verdicts):
                pid = item["id"]
                if not worth_seeing:
                    noise_rows.append((pid, reason))
                    skipped_count += 1
                    continue
                valid_for_encoding.append({"id": pid, "content": item["content"], "cleaned": cleanse_text(item["content"])})
                if entities:
                    batch_entities.append((pid, entities))

            if noise_rows:
                cur.execute(
                    """
                    UPDATE posts
                    SET assignment_status = 'noise', assignment_updated_at = NOW()
                    WHERE id = ANY(%s::int[]);
                    """,
                    ([pid for pid, _ in noise_rows],)
                )
                write_outbox_bulk(cur, [
                    ("post.noise", pid, None, {"post_id": pid, "status": "noise", "reason": reason})
                    for pid, reason in noise_rows
                ])
                log_decisions_bulk(cur, [
                    (pid, None, None, None, None, None, "noise", None, pv, mv, "discourse_gate_noise")
                    for pid, _ in noise_rows
                ])

            entity_rows.extend(batch_entities)

    # Persist extracted strong entities (LABEL:text[]) for entity-conflict
    # disambiguation in assignment/birth. Separate update: failure here only
    # loses the veto signal, never the post's processing (entities stay NULL
    # and behave exactly like the pre-veto path).
    if entity_rows:
        with get_db_cursor(commit=True) as cur:
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """
                UPDATE posts AS p
                SET entities = v.ents::text[]
                FROM (VALUES %s) AS v(id, ents)
                WHERE p.id = v.id
                """,
                entity_rows,
                template="(%s::int, %s::text[])",
            )

    if not valid_for_encoding:
        return {"status": "success", "processed": len(posts), "assigned": 0, "skipped": skipped_count}

    # Phase 2: Batch Vector Encoding in PyTorch (single process-wide batch)
    embeddings = get_embedding_engine().encode_batch(
        [item["cleaned"] for item in valid_for_encoding], batch_size=64
    )

    # Phase 3: Match & Assign with per-chunk commits (bounds fsync + lock hold)
    assigned_count, hub_centroid_updates, hub_membership_increments = assign_valid_for_encoding(
        valid_for_encoding, embeddings
    )

    # Phase 4: One transaction for aggregated centroid updates (anchor rule)
    if hub_centroid_updates or hub_membership_increments:
        with get_db_cursor(commit=True) as cur:
            for eid, new_vectors in hub_centroid_updates.items():
                cur.execute(
                    """
                    SELECT centroid::text, member_count, seed_post_id
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
                anchor_pid = extract_val(hub_row, "seed_post_id", 2)

                c_rolling = bounded_rolling_centroid(
                    parse_vector_literal(c_text) if c_text else None,
                    old_count,
                    new_vectors,
                    CENTROID_MAX_MEMBERS,
                )

                # Anchor protection: preserve catalyst direction for large events
                anchor_vec = None
                if anchor_pid:
                    cur.execute("SELECT embedding::text FROM posts WHERE id = %s;", (anchor_pid,))
                    anc_row = cur.fetchone()
                    anc_text = extract_val(anc_row, "embedding", 0)
                    if anc_text:
                        anchor_vec = parse_vector_literal(anc_text)

                c_effective = anchor_blended_centroid(c_rolling, anchor_vec, ANCHOR_WEIGHT)
                updated_str = vector_to_array_literal(normalize(c_effective))

                cur.execute(
                    """
                    UPDATE event_hubs
                    SET centroid = %s::vector,
                        member_count = member_count + %s,
                        last_updated_at = NOW()
                    WHERE id = %s;
                    """,
                    (updated_str, len(new_vectors), eid)
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


def _reset_stale_processing() -> int:
    """Return stuck 'processing' claims to 'pending' so they become claimable again."""
    with get_db_cursor(commit=True, durable=CLAIM_DURABLE) as cur:
        cur.execute(
            """
            UPDATE posts
            SET assignment_status = 'pending', assignment_updated_at = NOW()
            WHERE assignment_status = 'processing'
              AND assignment_updated_at < NOW() - (INTERVAL '1 second' * %s);
            """,
            (PROCESSING_STALE_SECONDS,)
        )
        return cur.rowcount


def _dispatch_pending_posts(batch_limit: int) -> int:
    """Enqueue a batch of pending posts (safety net) for processing.

    Deliberately does *not* pre-claim the rows: the processor's Phase 1 CAS
    claim (``pending`` -> ``processing``) is the only claim path, so a sweep can
    never rob the worker of its work. Overlapping sweeps simply dispatch the
    same ids again; the atomic claim makes the losers no-ops.
    """
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, content FROM posts
            WHERE assignment_status = 'pending'
            ORDER BY id ASC
            LIMIT %s;
            """,
            (batch_limit,)
        )
        rows = cur.fetchall()

    if not rows:
        return 0

    batch_data = [
        {"id": extract_val(r, "id", 0), "content": extract_val(r, "content", 1)}
        for r in rows
    ]
    celery_app.send_task("post_clustering_pipeline.tasks.process_post_batch_ingestion", args=[batch_data])
    return len(batch_data)


@celery_app.task(name="post_clustering_pipeline.tasks.drain_ingest_hints")
def drain_ingest_hints(max_hints: int = 500):
    """Fast-path: drain API LPUSH hints into one batched ingest task.

    The API row insert (status 'pending') is the durable source of truth; this
    drain only shrinks latency by batching hint ids into few ``encode_batch``
    calls. ``reconcile_pending_posts`` (30s beat) backstops anything the drain
    misses (Redis loss, worker saturation) since it re-claims 'pending' rows.
    """
    r = get_redis_client()
    hint_ids: set[int] = set()
    for _ in range(max_hints):
        raw = r.rpop(INGEST_HINTS_KEY)
        if raw is None:
            break
        try:
            hint_ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    if not hint_ids:
        return {"status": "idle", "hints": 0, "dispatched": 0}

    with get_db_cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, content FROM posts WHERE id = ANY(%s::int[]) AND assignment_status = 'pending' ORDER BY id;",
            (sorted(hint_ids),)
        )
        rows = cur.fetchall() or []

    batch_data = [
        {"id": extract_val(r, "id", 0), "content": extract_val(r, "content", 1)}
        for r in rows
    ]
    for i in range(0, len(batch_data), CLAIM_CHUNK_SIZE):
        chunk = batch_data[i:i + CLAIM_CHUNK_SIZE]
        celery_app.send_task("post_clustering_pipeline.tasks.process_post_batch_ingestion", args=[chunk])
    return {"status": "ok", "hints": len(hint_ids), "dispatched": len(batch_data)}


@celery_app.task(name="post_clustering_pipeline.tasks.reconcile_pending_posts")
def reconcile_pending_posts(batch_limit: int = 200):
    """Reconcile stuck 'processing' claims and backstop missed API task hints.

    The API row insert is the durable source of truth; the Redis task send is
    only a fast-path hint. This sweeper guarantees no post stays 'pending'
    forever even if the hint enqueue failed or the worker crashed mid-batch.
    """
    reset_stale = _reset_stale_processing()
    dispatched = _dispatch_pending_posts(batch_limit)
    auto_resolve = resolve_stale_candidates(batch_limit)

    # Staleness advisory: pending rows that are far older than normal drain
    # latency indicate a stuck hint/sweeper path - one lightweight count so the
    # operator (and /health/ready) can observe ingestion health.
    stalled_pending = 0
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS stalled
            FROM posts
            WHERE assignment_status = 'pending'
              AND assignment_updated_at < NOW() - (INTERVAL '1 second' * %s);
            """,
            (PENDING_STALL_SECONDS,)
        )
        row = cur.fetchone()
        stalled_pending = extract_val(row, "stalled", 0) or 0

    return {"reconciled": dispatched, "reset_stale": reset_stale, "stalled_pending": stalled_pending,
            "auto_resolved": auto_resolve.get("resolved", 0)}


@celery_app.task(name="post_clustering_pipeline.tasks.run_event_birth_scheduled")
def run_event_birth_scheduled():
    """Trigger periodic event birth job with distributed lock protection."""
    with distributed_task_lock("event_birth_job", timeout=3600) as acquired:
        if not acquired:
            return {"status": "skipped", "reason": "Job already running"}
        from .jobs.event_birth import run_clustering_pipeline
        run_clustering_pipeline()
        # Repair is part of birth, not a separate beat: community detection
        # over-splits a topic into fragments (measured up to 34 hubs for 10
        # topics), so folding the duplicates immediately after creation keeps
        # the next assignment wave matching against merged centroids instead of
        # re-littering the candidate pile.
        from .jobs.merge_hubs import reconcile_hub_merges
        merged = reconcile_hub_merges()
        return {"status": "completed", "merged_hubs": merged}


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
    """Prune acknowledged integration events older than retention period, and
    bound outbox growth in pull mode: unconsumed 'pending' events older than
    OUTBOX_UNCONSUMED_TTL_DAYS that are NOT leased to an active consumer age out
    to 'failed' (an event the integration never claimed), then failed rows are
    pruned after OUTBOX_FAILED_TTL_DAYS. Otherwise an inactive integration pool
    (0 consumers, 1,316 rows and growing) would grow without bound."""
    from .config import OUTBOX_UNCONSUMED_TTL_DAYS, OUTBOX_FAILED_TTL_DAYS
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE integration_outbox
            SET delivery_status = 'failed', last_error = 'unconsumed_event',
                attempts = attempts + 1, lease_until = NULL
            WHERE delivery_status = 'pending'
              AND available_at < NOW() - (%s * INTERVAL '1 day')
              AND NOT (consumer IS NOT NULL AND lease_until > NOW());
            """,
            (OUTBOX_UNCONSUMED_TTL_DAYS,)
        )
        aged_out = cur.rowcount

        cur.execute(
            """
            DELETE FROM integration_outbox
            WHERE delivery_status = 'failed'
              AND last_error = 'unconsumed_event'
              AND created_at < NOW() - (%s * INTERVAL '1 day');
            """,
            (OUTBOX_FAILED_TTL_DAYS,)
        )
        pruned_failed = cur.rowcount

        cur.execute(
            """
            DELETE FROM integration_outbox
            WHERE delivery_status = 'delivered'
              AND delivered_at < NOW() - (%s * INTERVAL '1 day');
            """,
            (retention_days,)
        )
        deleted = cur.rowcount
    return {"pruned_outbox_events": deleted, "aged_out": aged_out, "pruned_failed": pruned_failed}


@celery_app.task(name="post_clustering_pipeline.tasks.run_quality_rollup_scheduled")
def run_quality_rollup_scheduled():
    """Roll up the anchored trailing window (hour-bounded, idempotent)."""
    with distributed_task_lock("quality_rollup_job", timeout=900) as acquired:
        if not acquired:
            return {"status": "skipped", "reason": "Job already running"}
        from .quality import run_window_rollup
        return run_window_rollup()


@celery_app.task(name="post_clustering_pipeline.tasks.run_threshold_autotune_scheduled")
def run_threshold_autotune_scheduled():
    """Replay human feedback and propose (or auto-apply) a threshold revision.
    Propose-only unless AUTO_APPLY_THRESHOLD is set."""
    with distributed_task_lock("threshold_autotune_job", timeout=600) as acquired:
        if not acquired:
            return {"status": "skipped", "reason": "Job already running"}
        from .quality import run_threshold_autotune
        return run_threshold_autotune()


@celery_app.task(name="post_clustering_pipeline.tasks.dispatch_webhooks_scheduled")
def dispatch_webhooks_scheduled():
    """Periodically push pending outbox events to the configured webhook endpoint."""
    from .dispatch import dispatch_pending_webhooks
    count = dispatch_pending_webhooks(limit=50)
    return {"dispatched_webhooks": count}