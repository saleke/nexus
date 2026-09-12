import torch
try:
    from .queues import celery_app
    from .db import get_db_connection
    from .nlp import is_clusterable, cleanse_text
    from .models import embedding_engine
    from .events import write_outbox
except ImportError:
    from queues import celery_app
    from db import get_db_connection
    from nlp import is_clusterable, cleanse_text
    from models import embedding_engine
    from events import write_outbox
try:
    from .config import SIMILARITY_MARGIN, AUTO_ASSIGN_THRESHOLD, CANDIDATE_THRESHOLD, CENTROID_UPDATE_THRESHOLD
except ImportError:
    from config import SIMILARITY_MARGIN, AUTO_ASSIGN_THRESHOLD, CANDIDATE_THRESHOLD, CENTROID_UPDATE_THRESHOLD

def extract_val(row, key: str, idx: int):
    if not row:
        return None
    if isinstance(row, dict):
        return row.get(key)
    if isinstance(row, (list, tuple)):
        return row[idx]
    return row

@celery_app.task(
    name="tasks.process_post_ingestion",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
)
def process_post_ingestion(post_id: int, content: str):
    if not is_clusterable(content):
        return {"status": "skipped", "reason": "Not clusterable"}
    
    cleaned_text = cleanse_text(content)
    
    # Static Reasoning Logic Execution under strict memory gradient isolation
    vector = embedding_engine.encode(cleaned_text)
    vector_str = f"[{','.join(map(str, vector))}]"
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("UPDATE posts SET embedding = %s WHERE id = %s;", (vector_str, post_id))
        
        cur.execute("SELECT value FROM system_config WHERE key = 'global_similarity_threshold';")
        config_row = cur.fetchone()
        raw_val = extract_val(config_row, "value", 0)
        threshold = max(float(raw_val) if raw_val is not None else AUTO_ASSIGN_THRESHOLD, AUTO_ASSIGN_THRESHOLD)
        
        # Match against maintained event centroids and require a confidence margin.
        query_hookon = """
            SELECT eh.id AS event_id,
                   (1 - (eh.centroid <=> %s::vector)) AS cosine_similarity
            FROM event_hubs eh
            WHERE eh.centroid IS NOT NULL
              AND eh.last_updated_at >= NOW() - INTERVAL '48 hours'
            ORDER BY eh.centroid <=> %s::vector
            LIMIT 2;
        """
        cur.execute(query_hookon, (vector_str, vector_str))
        matches = cur.fetchall()
        match = matches[0] if matches else None
        event_id = extract_val(match, "event_id", 0)
        similarity = extract_val(match, "cosine_similarity", 1)
        second_similarity = extract_val(matches[1], "cosine_similarity", 1) if len(matches) > 1 else -1.0
        
        has_competitor = len(matches) > 1
        confident = (match and similarity is not None and float(similarity) >= threshold
                     and (not has_competitor or float(similarity) - float(second_similarity) >= SIMILARITY_MARGIN))
        if confident:
            similarity = float(similarity)
            cur.execute("UPDATE posts SET event_id = %s WHERE id = %s;", (event_id, post_id))
            cur.execute("UPDATE posts SET assignment_status = 'assigned', assignment_confidence = %s, assignment_updated_at = NOW() WHERE id = %s;", (float(similarity), post_id))
            write_outbox(cur, "post.assigned", post_id, event_id, {"post_id": post_id, "event_id": event_id, "confidence": similarity, "status": "assigned"})
            cur.execute("UPDATE event_hubs SET last_updated_at = NOW() WHERE id = %s;", (event_id,))
            if float(similarity) >= CENTROID_UPDATE_THRESHOLD:
                cur.execute("UPDATE event_hubs SET centroid = (SELECT AVG(embedding) FROM posts WHERE event_id = %s AND assignment_status = 'assigned') WHERE id = %s;", (event_id, event_id))
            cur.execute(
                """
                INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type)
                VALUES (%s, %s, %s, 'auto_confirmed');
                """,
                (post_id, event_id, similarity)
            )
        else:
            status_value = 'candidate' if similarity is not None and float(similarity) >= CANDIDATE_THRESHOLD else 'pending'
            cur.execute("UPDATE posts SET assignment_status = %s, assignment_confidence = %s, assignment_updated_at = NOW() WHERE id = %s;", (status_value, float(similarity) if similarity is not None else None, post_id))
            write_outbox(cur, f"post.{status_value}", post_id, None, {"post_id": post_id, "status": status_value, "confidence": similarity})
            cur.execute(
                """
                INSERT INTO unclustered_posts_buffer (post_id, embedding)
                VALUES (%s, %s::vector)
                ON CONFLICT (post_id) DO NOTHING;
                """,
                (post_id, vector_str)
            )
            
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()
        
    return {"status": "success", "post_id": post_id}
