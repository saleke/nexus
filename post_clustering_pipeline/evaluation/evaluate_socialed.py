import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from ..jobs.event_birth import run_clustering_pipeline
from ..db import get_db_connection
from ..embed_io import vector_to_array_literal

# Simulated benchmark records representing real dataset distributions
BENCHMARK_DATASET = [
    {"text": "Massive AWS cloud outage reported in us-east-1 region today.", "timestamp": "2012-10-01 10:00:00", "label": 1},
    {"text": "Amazon Web Services confirms network degradation in Virginia.", "timestamp": "2012-10-01 10:05:00", "label": 1},
    {"text": "FC Barcelona secured a dramatic 3-2 victory over Real Madrid.", "timestamp": "2012-10-01 11:00:00", "label": 2},
    {"text": "Late goal sends Barcelona past Real Madrid in El Clasico.", "timestamp": "2012-10-01 11:02:00", "label": 2},
    {"text": "SpaceX Starship successfully completes orbital insertion test.", "timestamp": "2012-10-01 12:00:00", "label": 3},
    {"text": "Historic rocket launch today as Starship reaches space orbit.", "timestamp": "2012-10-01 12:05:00", "label": 3},
    {"text": "Federal Reserve cuts benchmark interest rates by 50 basis points.", "timestamp": "2012-10-01 13:00:00", "label": 4},
    {"text": "Stock market rallies sharply following Central Bank rate reduction.", "timestamp": "2012-10-01 13:05:00", "label": 4}
]

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    texts = [item["text"] for item in BENCHMARK_DATASET]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    for idx, item in enumerate(BENCHMARK_DATASET):
        emb_str = vector_to_array_literal(embeddings[idx].tolist())
        
        cur.execute(
            """
            INSERT INTO posts (content, embedding, created_at, ground_truth)
            VALUES (%s, %s::vector, %s, %s)
            RETURNING id;
            """,
            (item["text"], emb_str, item["timestamp"], item["label"])
        )
        pid = cur.fetchone()[0]
        
        cur.execute(
            """
            INSERT INTO unclustered_posts_buffer (post_id, embedding, created_at)
            VALUES (%s, %s::vector, %s);
            """,
            (pid, emb_str, item["timestamp"])
        )
    conn.commit()
    cur.close()
    conn.close()
    
    run_clustering_pipeline()
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT ground_truth, event_id FROM posts WHERE event_id IS NOT NULL;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if not rows:
        print("No clusters formed.")
        return
        
    y_true = [r[0] for r in rows]
    y_pred = [r[1] for r in rows]
    
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"Evaluation Metrics -> NMI: {nmi:.4f} | ARI: {ari:.4f}")

if __name__ == "__main__":
    main()
