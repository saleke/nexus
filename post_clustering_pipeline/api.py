import sys
import os
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

# Ensure local project directory is in the import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from .db import get_db_connection
from .queues import celery_app

app = FastAPI(title="EventHub Clustering Platform API")


def run():
    import uvicorn
    uvicorn.run("post_clustering_pipeline.api:app", host="0.0.0.0", port=8000)

# --- Request Models ---
class PostCreateRequest(BaseModel):
    user_id: int
    content: str
    has_media: bool = False

class FeedbackRemoveRequest(BaseModel):
    post_id: int
    event_id: int


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
        
        cur.execute(
            "INSERT INTO posts (user_id, content, has_media) VALUES (%s, %s, %s) RETURNING id;",
            (post.user_id, post.content, post.has_media)
        )
        row = cur.fetchone()
        post_id = extract_field(row, "id", 0)
        conn.commit()
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
        cur.execute("UPDATE posts SET event_id = NULL WHERE id = %s;", (payload.post_id,))
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
