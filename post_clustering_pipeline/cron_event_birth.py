import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timezone
try:
    from .db import get_db_connection
except ImportError:
    from db import get_db_connection

def parse_embedding(emb_text):
    return np.array(list(map(float, emb_text.strip("[]").split(","))))

def get_active_centroids(cur):
    cur.execute(
        """
        SELECT event_id, embedding::text 
        FROM posts 
        WHERE event_id IS NOT NULL 
        ORDER BY created_at DESC LIMIT 500;
        """
    )
    rows = cur.fetchall()
    
    hub_vectors = {}
    for r in rows:
        eid = r["event_id"] if isinstance(r, dict) else r[0]
        emb = parse_embedding(r["embedding"] if isinstance(r, dict) else r[1])
        if eid not in hub_vectors:
            hub_vectors[eid] = []
        hub_vectors[eid].append(emb)
        
    centroids = {}
    for eid, vectors in hub_vectors.items():
        centroids[eid] = np.mean(vectors, axis=0)
    return centroids

def temporal_dataloader(rows, slice_hours=1.0):
    if not rows:
        return []
    
    slices = []
    current_slice = []
    
    slice_start = rows[0]["created_at"] if isinstance(rows[0], dict) else rows[0][2]
    if slice_start.tzinfo is None:
        slice_start = slice_start.replace(tzinfo=timezone.utc)

    for r in rows:
        created_at = r["created_at"] if isinstance(r, dict) else r[2]
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
            
        time_diff_hours = abs((created_at - slice_start).total_seconds()) / 3600.0
        
        if time_diff_hours > slice_hours:
            slices.append(current_slice)
            current_slice = [r]
            slice_start = created_at
        else:
            current_slice.append(r)
            
    if current_slice:
        slices.append(current_slice)
        
    return slices

def construct_knn_event_graph(rows, k=4, min_similarity_floor=0.28, half_life_hours=12.0):
    g = nx.Graph()
    post_ids, embeddings, timestamps = [], [], []

    for r in rows:
        pid = r["post_id"] if isinstance(r, dict) else r[0]
        emb = parse_embedding(r["embedding"] if isinstance(r, dict) else r[1])
        created_at = r["created_at"] if isinstance(r, dict) else r[2]
        
        post_ids.append(pid)
        embeddings.append(emb)
        timestamps.append(created_at)
        g.add_node(pid)

    n = len(post_ids)
    if n < 2:
        return g, post_ids

    embeddings_arr = np.array(embeddings)
    sim_matrix = cosine_similarity(embeddings_arr)

    for i in range(n):
        sim_scores = sim_matrix[i]
        nearest_indices = np.argsort(sim_scores)[::-1]
        
        connected_count = 0
        for j in nearest_indices:
            if i == j: continue
            
            base_sim = sim_scores[j]
            if base_sim < min_similarity_floor: break
            if connected_count >= k and base_sim < 0.40: break

            t1, t2 = timestamps[i], timestamps[j]
            if t1.tzinfo is None: t1 = t1.replace(tzinfo=timezone.utc)
            if t2.tzinfo is None: t2 = t2.replace(tzinfo=timezone.utc)
            
            time_diff_hours = abs((t1 - t2).total_seconds()) / 3600.0
            decay_factor = np.exp(-np.log(2) * (time_diff_hours / half_life_hours))
            
            edge_weight = float(base_sim * decay_factor)
            g.add_edge(post_ids[i], post_ids[j], weight=edge_weight)
            connected_count += 1

    return g, post_ids

def run_clustering_pipeline():
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        active_centroids = get_active_centroids(cur)
        
        cur.execute(
            """
            SELECT post_id, embedding::text, created_at
            FROM unclustered_posts_buffer
            ORDER BY created_at ASC;
            """
        )
        buffer_rows = cur.fetchall()
        
        if not buffer_rows:
            return

        temporal_slices = temporal_dataloader(buffer_rows, slice_hours=1.0)
        
        for slice_rows in temporal_slices:
            remaining_for_birth = []
            
            for r in slice_rows:
                pid = r["post_id"] if isinstance(r, dict) else r[0]
                emb = parse_embedding(r["embedding"] if isinstance(r, dict) else r[1])
                
                best_hub = None
                best_sim = -1
                
                for eid, centroid in active_centroids.items():
                    sim = cosine_similarity([emb], [centroid])[0][0]
                    if sim > best_sim:
                        best_sim = sim
                        best_hub = eid
                
                if best_hub and best_sim >= 0.30:
                    cur.execute("UPDATE posts SET event_id = %s WHERE id = %s;", (best_hub, pid))
                    cur.execute("DELETE FROM unclustered_posts_buffer WHERE post_id = %s;", (pid,))
                    active_centroids[best_hub] = (active_centroids[best_hub] + emb) / 2.0
                else:
                    remaining_for_birth.append(r)
            
            if len(remaining_for_birth) >= 2:
                g, post_ids = construct_knn_event_graph(remaining_for_birth, k=4, min_similarity_floor=0.28)
                
                communities = nx_comm.louvain_communities(g, weight='weight', resolution=1.0)
                
                for comm in communities:
                    cluster_post_ids = list(comm)
                    if len(cluster_post_ids) < 2:
                        continue
                        
                    cur.execute("INSERT INTO event_hubs DEFAULT VALUES RETURNING id;")
                    event_row = cur.fetchone()
                    new_event_id = event_row["id"] if isinstance(event_row, dict) else event_row[0]
                    
                    cur.execute(
                        "UPDATE posts SET event_id = %s WHERE id = ANY(%s);",
                        (new_event_id, cluster_post_ids)
                    )
                    cur.execute(
                        "DELETE FROM unclustered_posts_buffer WHERE post_id = ANY(%s);",
                        (cluster_post_ids,)
                    )
                    
                    new_embeddings = [parse_embedding(r["embedding"] if isinstance(r, dict) else r[1]) 
                                      for r in remaining_for_birth if (r["post_id"] if isinstance(r, dict) else r[0]) in cluster_post_ids]
                    active_centroids[new_event_id] = np.mean(new_embeddings, axis=0)
                    centroid_str = "[" + ",".join(map(str, active_centroids[new_event_id].tolist())) + "]"
                    cur.execute("UPDATE event_hubs SET centroid = %s::vector WHERE id = %s;", (centroid_str, new_event_id))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_clustering_pipeline()
