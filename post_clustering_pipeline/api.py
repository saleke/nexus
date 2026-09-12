import uuid
import hmac
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime

from .db import get_db_connection, get_db_cursor
from .queues import celery_app
from .config import MODEL_VERSION, POLICY_VERSION, OUTBOX_MAX_ATTEMPTS, API_AUTH_TOKEN
from .events import write_outbox

app = FastAPI(title="Nexus Event Clustering Platform API")


@app.middleware("http")
async def service_authentication(request: Request, call_next):
    """Require a configured bearer token for application endpoints.

    Authentication is opt-in for local development; production deployments should
    set API_AUTH_TOKEN and keep health probes publicly reachable.
    """
    public_paths = {"/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc"}
    if API_AUTH_TOKEN and request.url.path not in public_paths:
        supplied = request.headers.get("authorization", "")
        expected = f"Bearer {API_AUTH_TOKEN}"
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={"detail": "authentication required"}, headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


def run():
    import uvicorn
    uvicorn.run("post_clustering_pipeline.api:app", host="0.0.0.0", port=8000)


# --- Request Models ---
class PostCreateRequest(BaseModel):
    user_id: int
    content: str
    has_media: bool = False
    platform: str = "unknown"
    source_id: str = "legacy"
    external_post_id: str | None = None
    external_author_id: str | None = None
    published_at: datetime | None = None


class BatchPostCreateRequest(BaseModel):
    # Bound request size so one transaction cannot monopolize the database.
    posts: list[PostCreateRequest] = Field(default_factory=list, max_length=500)


class FeedbackRemoveRequest(BaseModel):
    post_id: int
    event_id: int


class FeedbackConfirmRequest(BaseModel):
    post_id: int
    event_id: int
    actor: str = "user"


# --- Helper Function for Cursors ---
def extract_field(row, dict_key: str, index: int = 0):
    """Safely extract values whether using RealDictCursor or standard tuple cursor."""
    if not row:
        return None
    if isinstance(row, dict):
        return row.get(dict_key)
    if isinstance(row, (list, tuple)):
        return row[index]
    return row


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT COUNT(*) AS dlq_count FROM integration_outbox WHERE delivery_status = 'failed'")
            row = cur.fetchone()
            dlq_count = extract_field(row, "dlq_count", 0) or 0
        return {
            "status": "ready",
            "dlq_failed_events": dlq_count
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")


@app.get("/posts/{post_id}/status")
def get_post_status(post_id: int):
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, event_id, assignment_status, assignment_confidence, assignment_updated_at FROM posts WHERE id = %s",
            (post_id,)
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")
    return row


@app.post("/posts", status_code=status.HTTP_201_CREATED)
def create_post(post: PostCreateRequest):
    try:
        with get_db_cursor() as cur:
            if post.external_post_id:
                cur.execute(
                    "SELECT id, deleted_at FROM posts WHERE source_id = %s AND external_post_id = %s",
                    (post.source_id, post.external_post_id)
                )
                existing = cur.fetchone()
                if existing and extract_field(existing, "deleted_at", 1) is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="This platform post was deleted and cannot be recreated"
                    )

            cur.execute(
                """INSERT INTO posts
                (user_id, content, has_media, platform, source_id, external_post_id,
                 external_author_id, published_at, assignment_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')
                ON CONFLICT (source_id, external_post_id) WHERE external_post_id IS NOT NULL
                DO UPDATE SET content = EXCLUDED.content, has_media = EXCLUDED.has_media,
                              published_at = COALESCE(EXCLUDED.published_at, posts.published_at)
                RETURNING id;""",
                (post.user_id, post.content, post.has_media, post.platform, post.source_id,
                 post.external_post_id, post.external_author_id, post.published_at)
            )
            row = cur.fetchone()
            post_id = extract_field(row, "id", 0)
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[API ERROR] Failed to create post: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        celery_app.send_task("post_clustering_pipeline.tasks.process_post_ingestion", args=[post_id, post.content])
    except Exception as e:
        print(f"\n[CELERY ERROR] Failed to send task to Redis: {e}\n")

    return {"status": "queued", "post_id": post_id}


@app.post("/posts/batch", status_code=status.HTTP_201_CREATED)
def create_posts_batch(payload: BatchPostCreateRequest):
    """High-throughput multi-post submission endpoint."""
    if not payload.posts:
        return {"status": "queued", "post_ids": []}

    created_posts: list[tuple[int, str]] = []
    try:
        with get_db_cursor() as cur:
            for post in payload.posts:
                if post.external_post_id:
                    cur.execute(
                        "SELECT id, deleted_at FROM posts WHERE source_id = %s AND external_post_id = %s",
                        (post.source_id, post.external_post_id)
                    )
                    existing = cur.fetchone()
                    if existing and extract_field(existing, "deleted_at", 1) is not None:
                        continue  # Skip deleted tombstones in batch

                cur.execute(
                    """INSERT INTO posts
                    (user_id, content, has_media, platform, source_id, external_post_id,
                     external_author_id, published_at, assignment_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')
                    ON CONFLICT (source_id, external_post_id) WHERE external_post_id IS NOT NULL
                    DO UPDATE SET content = EXCLUDED.content, has_media = EXCLUDED.has_media,
                                  published_at = COALESCE(EXCLUDED.published_at, posts.published_at)
                    RETURNING id;""",
                    (post.user_id, post.content, post.has_media, post.platform, post.source_id,
                     post.external_post_id, post.external_author_id, post.published_at)
                )
                row = cur.fetchone()
                pid = extract_field(row, "id", 0)
                if pid:
                    created_posts.append((pid, post.content))
    except Exception as e:
        print(f"\n[API ERROR] Failed in batch post ingestion: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))

    for i in range(0, len(created_posts), 64):
        chunk = [{"id": pid, "content": content} for pid, content in created_posts[i:i + 64]]
        try:
            celery_app.send_task("post_clustering_pipeline.tasks.process_post_batch_ingestion", args=[chunk])
        except Exception as e:
            print(f"\n[CELERY ERROR] Failed to dispatch batch to Redis: {e}\n")

    return {"status": "queued", "post_ids": [pid for pid, _ in created_posts]}


@app.delete("/posts/{post_id}")
def delete_post(post_id: int):
    with get_db_cursor() as cur:
        cur.execute(
            """
            UPDATE posts 
            SET deleted_at = COALESCE(deleted_at, NOW()), assignment_status = 'noise', 
                event_id = NULL, assignment_updated_at = NOW() 
            WHERE id = %s RETURNING id
            """,
            (post_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Post not found")
        cur.execute("DELETE FROM unclustered_posts_buffer WHERE post_id = %s;", (post_id,))
        write_outbox(cur, "post.deleted", post_id, None, {"status": "deleted"})
    return {"status": "deleted", "post_id": post_id}


@app.post("/posts/unlink", status_code=status.HTTP_200_OK)
def unlink_post(payload: FeedbackRemoveRequest):
    with get_db_cursor() as cur:
        cur.execute("SELECT id FROM event_hubs WHERE id = %s;", (payload.event_id,))
        event_row = cur.fetchone()
        if not event_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event Hub #{payload.event_id} does not exist."
            )

        cur.execute(
            """
            SELECT p.embedding, (1 - (p.embedding <=> (
                SELECT embedding FROM posts WHERE event_id = %s AND id != %s AND embedding IS NOT NULL LIMIT 1
            ))) AS similarity
            FROM posts p WHERE p.id = %s;
            """,
            (payload.event_id, payload.post_id, payload.post_id)
        )
        row = cur.fetchone()
        raw_sim = extract_field(row, "similarity", 1)
        sim_score = float(raw_sim) if raw_sim is not None else 0.0

        cur.execute(
            """
            UPDATE posts 
            SET event_id = NULL, assignment_status = 'candidate', 
                assignment_confidence = NULL, assignment_updated_at = NOW() 
            WHERE id = %s;
            """,
            (payload.post_id,)
        )
        cur.execute(
            """
            INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type)
            VALUES (%s, %s, %s, 'user_removed');
            """,
            (payload.post_id, payload.event_id, sim_score)
        )
        write_outbox(cur, "post.unlinked", payload.post_id, payload.event_id, {
            "post_id": payload.post_id,
            "previous_event_id": payload.event_id,
            "status": "unlinked"
        })

    return {"status": "unlinked", "post_id": payload.post_id, "event_id": payload.event_id}


@app.post("/posts/confirm", status_code=status.HTTP_200_OK)
def confirm_post(payload: FeedbackConfirmRequest):
    with get_db_cursor() as cur:
        cur.execute("SELECT event_id, assignment_confidence FROM posts WHERE id = %s", (payload.post_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Post not found")
        old_event = extract_field(row, "event_id", 0)
        confidence = extract_field(row, "assignment_confidence", 1) or 0

        cur.execute(
            "UPDATE posts SET event_id = %s, assignment_status = 'assigned', assignment_updated_at = NOW() WHERE id = %s",
            (payload.event_id, payload.post_id)
        )
        cur.execute(
            """INSERT INTO clustering_feedback_log
            (post_id, event_id, initial_similarity_score, feedback_type, actor, model_version, policy_version)
            VALUES (%s, %s, %s, 'user_confirmed', %s, %s, %s)""",
            (payload.post_id, payload.event_id, confidence, payload.actor, MODEL_VERSION, POLICY_VERSION)
        )
        write_outbox(
            cur, "post.confirmed", payload.post_id, payload.event_id,
            {"post_id": payload.post_id, "event_id": payload.event_id, "status": "assigned", "confidence": float(confidence)}
        )

    return {"status": "confirmed", "post_id": payload.post_id, "event_id": payload.event_id, "previous_event_id": old_event}


@app.get("/integration/events")
def integration_events(limit: int = 100, after: int = 0, consumer: str = "default"):
    """Pull interface with lease locking and dead-letter queue (DLQ) protection."""
    limit = max(1, min(limit, 500))
    token = uuid.uuid4().hex

    with get_db_cursor() as cur:
        # Move poisoned records exceeding max retries to Dead Letter Queue (failed)
        cur.execute(
            """
            UPDATE integration_outbox 
            SET delivery_status = 'failed', last_error = 'Max delivery attempts exceeded'
            WHERE attempts >= %s AND delivery_status IN ('pending', 'leased')
              AND (lease_until IS NULL OR lease_until < NOW());
            """,
            (OUTBOX_MAX_ATTEMPTS,)
        )

        cur.execute(
            """
            WITH claimed AS (
                SELECT id FROM integration_outbox
                WHERE id > %s AND available_at <= NOW()
                  AND (delivery_status = 'pending' OR (delivery_status = 'leased' AND lease_until < NOW()))
                ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED
            )
            UPDATE integration_outbox o SET delivery_status = 'leased', consumer = %s,
                lease_token = %s, lease_until = NOW() + INTERVAL '2 minutes', attempts = attempts + 1
            FROM claimed c WHERE o.id = c.id
            RETURNING o.id, o.id::text AS event_key, o.schema_version, o.event_type, o.post_id,
                o.event_id, o.payload, o.attempts, o.lease_token, o.created_at
            ORDER BY o.id
            """,
            (after, limit, consumer, token)
        )
        rows = cur.fetchall()

    return rows


@app.post("/integration/events/{outbox_id}/ack")
def acknowledge_integration_event(outbox_id: int, lease_token: str | None = None):
    with get_db_cursor() as cur:
        cur.execute(
            """
            UPDATE integration_outbox
            SET delivery_status = 'delivered', delivered_at = NOW(), lease_until = NULL
            WHERE id = %s AND delivery_status IN ('pending', 'leased')
              AND (%s IS NULL OR lease_token = %s)
            RETURNING id, id::text AS event_key, delivery_status
            """,
            (outbox_id, lease_token, lease_token)
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                "SELECT id, id::text AS event_key, delivery_status FROM integration_outbox WHERE id = %s",
                (outbox_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Integration event not found")

    return row


def resolve_canonical_hub(cur, hub_id: int) -> tuple[int, bool]:
    """Resolve canonical hub ID following merged_into_id links.
    
    Returns (canonical_hub_id, was_redirected).
    """
    current_id = hub_id
    redirected = False
    for _ in range(5):
        cur.execute("SELECT id, is_active, merged_into_id FROM event_hubs WHERE id = %s;", (current_id,))
        row = cur.fetchone()
        if not row:
            break
        is_active = extract_field(row, "is_active", 1)
        merged_into = extract_field(row, "merged_into_id", 2)
        if is_active or not merged_into:
            return current_id, redirected
        current_id = merged_into
        redirected = True
    return current_id, redirected


@app.get("/hubs/{hub_id}/view")
@app.get("/events/{hub_id}/view")
def get_hub_view(hub_id: int, limit: int = 50, offset: int = 0):
    """Unified 1-click portal view for a story, debate, topic, or event."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with get_db_cursor(commit=False) as cur:
        canonical_id, was_redirected = resolve_canonical_hub(cur, hub_id)
        cur.execute(
            """
            SELECT id, status, discourse_type, member_count, anchor_post_id, created_at, last_updated_at
            FROM event_hubs
            WHERE id = %s;
            """,
            (canonical_id,)
        )
        hub_row = cur.fetchone()
        if not hub_row:
            raise HTTPException(status_code=404, detail="Event Hub not found")

        anchor_pid = extract_field(hub_row, "anchor_post_id", 4)
        status_val = extract_field(hub_row, "status", 1) or "active"
        discourse_type = extract_field(hub_row, "discourse_type", 2) or "event"

        # 1. Fetch catalyst anchor post
        catalyst = None
        if anchor_pid:
            cur.execute(
                "SELECT id, user_id, content, has_media, engagement_score, created_at FROM posts WHERE id = %s;",
                (anchor_pid,)
            )
            catalyst = cur.fetchone()

        if not catalyst:
            cur.execute(
                "SELECT id, user_id, content, has_media, engagement_score, created_at FROM posts WHERE event_id = %s ORDER BY created_at ASC LIMIT 1;",
                (canonical_id,)
            )
            catalyst = cur.fetchone()

        # 2. Fetch chronological timeline
        cur.execute(
            """
            SELECT id, user_id, content, has_media, engagement_score, created_at
            FROM posts
            WHERE event_id = %s AND deleted_at IS NULL
            ORDER BY created_at ASC, id ASC
            LIMIT %s OFFSET %s;
            """,
            (canonical_id, limit + 1, offset)
        )
        timeline_rows = cur.fetchall() or []
        has_more = len(timeline_rows) > limit
        timeline = timeline_rows[:limit]
        cur.execute("SELECT COUNT(*) AS total_posts FROM posts WHERE event_id = %s AND deleted_at IS NULL;", (canonical_id,))
        count_row = cur.fetchone()
        total_posts = int(extract_field(count_row, "total_posts", 0) or 0)

        # 3. Fetch key perspectives (top distinct viewpoints / high engagement posts)
        catalyst_id = extract_field(catalyst, "id", 0) if catalyst else -1
        cur.execute(
            """
            SELECT id, user_id, content, has_media, engagement_score, created_at
            FROM posts
            WHERE event_id = %s AND id != %s AND deleted_at IS NULL
            ORDER BY engagement_score DESC, created_at ASC
            LIMIT %s;
            """,
            (canonical_id, catalyst_id, min(limit, 20))
        )
        perspectives = cur.fetchall() or []

        # 4. Fetch evidence media
        cur.execute(
            """
            SELECT id, user_id, content, created_at
            FROM posts
            WHERE event_id = %s AND has_media = TRUE AND deleted_at IS NULL
            ORDER BY created_at ASC, id ASC
            LIMIT %s;
            """,
            (canonical_id, min(limit, 100))
        )
        media = cur.fetchall() or []

        # 5. Count unique voices
        cur.execute(
            "SELECT COUNT(DISTINCT user_id) AS unique_voices FROM posts WHERE event_id = %s AND deleted_at IS NULL;",
            (canonical_id,)
        )
        uv_row = cur.fetchone()
        unique_voices = extract_field(uv_row, "unique_voices", 0) or len(timeline)

    return {
        "hub_id": canonical_id,
        "requested_hub_id": hub_id,
        "was_redirected": was_redirected,
        "status": status_val,
        "discourse_type": discourse_type,
        "metrics": {
            "total_posts": total_posts,
            "unique_voices": unique_voices,
            "created_at": extract_field(hub_row, "created_at", 5),
            "last_updated_at": extract_field(hub_row, "last_updated_at", 6),
        },
        "pagination": {
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
        },
        "catalyst_anchor": catalyst,
        "key_perspectives": perspectives,
        "timeline": timeline,
        "evidence_media": media,
    }


@app.get("/events/{event_id}/anchor")
def get_anchor_post(event_id: int):
    with get_db_cursor(commit=False) as cur:
        canonical_id, _ = resolve_canonical_hub(cur, event_id)
        cur.execute("SELECT * FROM posts WHERE event_id = %s ORDER BY created_at ASC LIMIT 1;", (canonical_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Event hub standard anchor post not found")
    return row


@app.get("/events/{event_id}/media")
def get_event_media(event_id: int):
    with get_db_cursor(commit=False) as cur:
        canonical_id, _ = resolve_canonical_hub(cur, event_id)
        cur.execute(
            "SELECT id, content, created_at FROM posts WHERE event_id = %s AND has_media = TRUE ORDER BY created_at ASC;",
            (canonical_id,)
        )
        rows = cur.fetchall()
    return rows


@app.get("/events/{event_id}/timeline")
def get_event_timeline(event_id: int):
    with get_db_cursor(commit=False) as cur:
        canonical_id, _ = resolve_canonical_hub(cur, event_id)
        cur.execute("SELECT * FROM posts WHERE event_id = %s ORDER BY created_at ASC;", (canonical_id,))
        rows = cur.fetchall()
    return rows


@app.get("/events/{event_id}/top")
def get_event_top_discussion(event_id: int):
    with get_db_cursor(commit=False) as cur:
        canonical_id, _ = resolve_canonical_hub(cur, event_id)
        cur.execute("SELECT * FROM posts WHERE event_id = %s ORDER BY engagement_score DESC;", (canonical_id,))
        rows = cur.fetchall()
    return rows
