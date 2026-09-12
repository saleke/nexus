import sys
import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from datetime import datetime

# Ensure local project directory is in the import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from .db import get_db_connection
from .queues import celery_app
from .config import MODEL_VERSION, POLICY_VERSION
from .events import write_outbox

app = FastAPI(title="EventHub Clustering Platform API")


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

class FeedbackRemoveRequest(BaseModel):
    post_id: int
    event_id: int


class FeedbackConfirmRequest(BaseModel):
    post_id: int
    event_id: int
    actor: str = "user"


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")


@app.get("/posts/{post_id}/status")
def get_post_status(post_id: int):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, event_id, assignment_status, assignment_confidence, assignment_updated_at FROM posts WHERE id = %s", (post_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")
    return row


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


# --- Endpoints ---
@app.post("/posts", status_code=status.HTTP_201_CREATED)
def create_post(post: PostCreateRequest):
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if post.external_post_id:
            cur.execute("SELECT id, deleted_at FROM posts WHERE source_id = %s AND external_post_id = %s", (post.source_id, post.external_post_id))
            existing = cur.fetchone()
            if existing and existing["deleted_at"] is not None:
                raise HTTPException(status_code=409, detail="This platform post was deleted and cannot be recreated")
        
        cur.execute(
            """INSERT INTO posts
            (user_id, content, has_media, platform, source_id, external_post_id,
             external_author_id, published_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_id, external_post_id) WHERE external_post_id IS NOT NULL
            DO UPDATE SET content = EXCLUDED.content, has_media = EXCLUDED.has_media,
                          published_at = COALESCE(EXCLUDED.published_at, posts.published_at)
            RETURNING id;""",
            (post.user_id, post.content, post.has_media, post.platform, post.source_id,
             post.external_post_id, post.external_author_id, post.published_at)
        )
        row = cur.fetchone()
        post_id = extract_field(row, "id", 0)
        conn.commit()
    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\n[API ERROR] Failed to create post: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    try:
        celery_app.send_task("tasks.process_post_ingestion", args=[post_id, post.content])
    except Exception as e:
        print(f"\n[CELERY ERROR] Failed to send task to Redis: {e}\n")

    return {"status": "queued", "post_id": post_id}


@app.delete("/posts/{post_id}")
def delete_post(post_id: int):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE posts SET deleted_at = COALESCE(deleted_at, NOW()), assignment_status = 'noise', event_id = NULL, assignment_updated_at = NOW() WHERE id = %s RETURNING id", (post_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Post not found")
            cur.execute("INSERT INTO integration_outbox (event_type, post_id, payload) VALUES ('post.deleted', %s, %s::jsonb)", (post_id, '{"status":"deleted"}'))
        conn.commit()
    finally:
        conn.close()
    return {"status": "deleted", "post_id": post_id}


@app.post("/posts/unlink", status_code=status.HTTP_200_OK)
def unlink_post(payload: FeedbackRemoveRequest):
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Check if the event_id exists in event_hubs to avoid foreign key violation
        cur.execute("SELECT id FROM event_hubs WHERE id = %s;", (payload.event_id,))
        event_row = cur.fetchone()
        if not event_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event Hub #{payload.event_id} does not exist."
            )
        
        # 2. Extract baseline similarity score before breaking association
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

        # 3. Unlink post and record feedback log
        cur.execute("UPDATE posts SET event_id = NULL, assignment_status = 'candidate', assignment_confidence = NULL, assignment_updated_at = NOW() WHERE id = %s;", (payload.post_id,))
        cur.execute(
            """
            INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type)
            VALUES (%s, %s, %s, 'user_removed');
            """,
            (payload.post_id, payload.event_id, sim_score)
        )
        conn.commit()
    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\n[API ERROR] Failed to unlink post: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    return {"status": "unlinked", "post_id": payload.post_id, "event_id": payload.event_id}


@app.post("/posts/confirm", status_code=status.HTTP_200_OK)
def confirm_post(payload: FeedbackConfirmRequest):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT event_id, assignment_confidence FROM posts WHERE id = %s", (payload.post_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Post not found")
            old_event = row["event_id"]
            confidence = row["assignment_confidence"] or 0
            cur.execute("UPDATE posts SET event_id = %s, assignment_status = 'assigned', assignment_updated_at = NOW() WHERE id = %s", (payload.event_id, payload.post_id))
            cur.execute("""INSERT INTO clustering_feedback_log
                (post_id, event_id, initial_similarity_score, feedback_type, actor, model_version, policy_version)
                VALUES (%s, %s, %s, 'user_confirmed', %s, %s, %s)""", (payload.post_id, payload.event_id, confidence, payload.actor, MODEL_VERSION, POLICY_VERSION))
            write_outbox(cur, "post.confirmed", payload.post_id, payload.event_id, {"post_id": payload.post_id, "event_id": payload.event_id, "status": "assigned", "confidence": float(confidence)})
        conn.commit()
    finally:
        conn.close()
    return {"status": "confirmed", "post_id": payload.post_id, "event_id": payload.event_id, "previous_event_id": old_event}


@app.get("/integration/events")
def integration_events(limit: int = 100, after: int = 0, consumer: str = "default"):
    """Development-phase pull interface for reliable host integration."""
    limit = max(1, min(limit, 500))
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            token = uuid.uuid4().hex
            cur.execute("""WITH claimed AS (
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
            ORDER BY o.id""", (after, limit, consumer, token))
            rows = cur.fetchall()
        conn.commit()
    finally:
        conn.close()
    return rows


@app.post("/integration/events/{outbox_id}/ack")
def acknowledge_integration_event(outbox_id: int, lease_token: str | None = None):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""UPDATE integration_outbox
                SET delivery_status = 'delivered', delivered_at = NOW(), lease_until = NULL
                WHERE id = %s AND delivery_status IN ('pending', 'leased')
                  AND (%s IS NULL OR lease_token = %s)
                RETURNING id, id::text AS event_key, delivery_status""", (outbox_id, lease_token, lease_token))
            row = cur.fetchone()
            if not row:
                cur.execute("SELECT id, id::text AS event_key, delivery_status FROM integration_outbox WHERE id = %s", (outbox_id,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Integration event not found")
        conn.commit()
    finally:
        conn.close()
    return row


@app.get("/events/{event_id}/anchor")
def get_anchor_post(event_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM posts WHERE event_id = %s ORDER BY created_at ASC LIMIT 1;", (event_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Event hub standard anchor post not found")
    return row


@app.get("/events/{event_id}/media")
def get_event_media(event_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, content, created_at FROM posts WHERE event_id = %s AND has_media = TRUE ORDER BY created_at ASC;",
        (event_id,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


@app.get("/events/{event_id}/timeline")
def get_event_timeline(event_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM posts WHERE event_id = %s ORDER BY created_at ASC;", (event_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


@app.get("/events/{event_id}/top")
def get_event_top_discussion(event_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM posts WHERE event_id = %s ORDER BY engagement_score DESC;", (event_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
