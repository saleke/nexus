import torch
try:
    from .queues import celery_app
    from .db import get_db_connection
    from .nlp import is_clusterable, cleanse_text
    from .models import embedding_engine
except ImportError:
    from queues import celery_app
    from db import get_db_connection
    from nlp import is_clusterable, cleanse_text
    from models import embedding_engine
try:
    from .config import SIMILARITY_MARGIN
except ImportError:
    from config import SIMILARITY_MARGIN

def extract_val(row, key: str, idx: int):
    if not row:
        return None
    if isinstance(row, dict):
        return row.get(key)
    if isinstance(row, (list, tuple)):
        return row[idx]
    return row

@celery_app.task(name="tasks.process_post_ingestion")
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
        threshold = float(raw_val) if raw_val is not None else 0.75
        
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
        
        if (match and similarity is not None and float(similarity) >= threshold
                and float(similarity) - float(second_similarity) >= SIMILARITY_MARGIN):
            similarity = float(similarity)
            cur.execute("UPDATE posts SET event_id = %s WHERE id = %s;", (event_id, post_id))
            cur.execute("UPDATE event_hubs SET last_updated_at = NOW() WHERE id = %s;", (event_id,))
            cur.execute(
                "UPDATE event_hubs SET centroid = (SELECT AVG(embedding) FROM posts WHERE event_id = %s) WHERE id = %s;",
                (event_id, event_id),
            )
            cur.execute(
                """
                INSERT INTO clustering_feedback_log (post_id, event_id, initial_similarity_score, feedback_type)
                VALUES (%s, %s, %s, 'auto_confirmed');
                """,
                (post_id, event_id, similarity)
            )
        else:
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
