import numpy as np
import torch
import networkx as nx
import networkx.algorithms.community as nx_comm
from datetime import datetime, timezone

from ..db import get_db_cursor
from ..events import write_outbox
from ..config import (
    BIRTH_ASSIGN_THRESHOLD,
    BIRTH_SIMILARITY_FLOOR,
    BIRTH_MIN_AUTHORS,
    BIRTH_COHESION_FLOOR,
    CENTROID_MAX_MEMBERS,
)


def parse_embedding(emb_text: str) -> np.ndarray:
    return np.array(list(map(float, emb_text.strip("[]").split(","))))


def get_active_centroids(cur):
    """Retrieve canonical active centroids and member counts directly from event_hubs."""
    cur.execute(
        """
        SELECT id, member_count, centroid::text 
        FROM event_hubs 
        WHERE is_active = TRUE
          AND centroid IS NOT NULL 
          AND last_updated_at >= NOW() - INTERVAL '48 hours';
        """
    )
    rows = cur.fetchall()

    centroids = {}
    member_counts = {}
    for r in rows:
        eid = r["id"] if isinstance(r, dict) else r[0]
        m_count = r["member_count"] if isinstance(r, dict) else r[1]
        emb = parse_embedding(r["centroid"] if isinstance(r, dict) else r[2])
        centroids[eid] = emb
        member_counts[eid] = m_count or 1

    return centroids, member_counts


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


def construct_knn_event_graph(rows, k=4, min_similarity_floor=BIRTH_SIMILARITY_FLOOR, half_life_hours=12.0):
    """Vectorized k-NN graph construction with temporal exponential decay."""
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

    # Vectorized similarity in PyTorch
    emb_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
    emb_norm = torch.nn.functional.normalize(emb_tensor, p=2, dim=-1)
    sim_matrix = torch.mm(emb_norm, emb_norm.t())

    k_val = min(k + 1, n)
    topk_sims, topk_indices = torch.topk(sim_matrix, k=k_val, dim=-1)

    ts_seconds = [
        (t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t).timestamp()
        for t in timestamps
    ]

    for i in range(n):
        for sim_val, j in zip(topk_sims[i].tolist(), topk_indices[i].tolist()):
            if i == j:
                continue
            if sim_val < min_similarity_floor:
                continue

            time_diff_hours = abs(ts_seconds[i] - ts_seconds[j]) / 3600.0
            decay_factor = np.exp(-np.log(2) * (time_diff_hours / half_life_hours))
            edge_weight = float(sim_val * decay_factor)

            if edge_weight >= min_similarity_floor:
                g.add_edge(post_ids[i], post_ids[j], weight=edge_weight)

    return g, post_ids


def run_clustering_pipeline():
    with get_db_cursor(commit=True) as cur:
        # Age out old unclustered posts and prune buffer
        cur.execute(
            """
            WITH aged AS (
                UPDATE posts SET assignment_status = 'noise', assignment_updated_at = NOW()
                WHERE event_id IS NULL AND assignment_status IN ('pending', 'candidate', 'unassigned')
                  AND created_at < NOW() - INTERVAL '24 hours'
                RETURNING id
            )
            DELETE FROM unclustered_posts_buffer WHERE post_id IN (SELECT id FROM aged);
            """
        )

        active_centroids, member_counts = get_active_centroids(cur)

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
                best_sim = -1.0

                for eid, centroid in active_centroids.items():
                    # Cosine similarity between normalized vectors
                    dot = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-9)
                    if dot > best_sim:
                        best_sim = dot
                        best_hub = eid

                # Assign if similarity meets calibrated birth-assign threshold (e.g. 0.82)
                if best_hub and best_sim >= BIRTH_ASSIGN_THRESHOLD:
                    cur.execute(
                        """
                        UPDATE posts 
                        SET event_id = %s, assignment_status = 'assigned', 
                            assignment_confidence = %s, assignment_updated_at = NOW() 
                        WHERE id = %s;
                        """,
                        (best_hub, float(best_sim), pid)
                    )
                    cur.execute("DELETE FROM unclustered_posts_buffer WHERE post_id = %s;", (pid,))
                    write_outbox(cur, "post.assigned", pid, best_hub, {
                        "post_id": pid,
                        "event_id": best_hub,
                        "confidence": float(best_sim),
                        "status": "assigned",
                        "source": "event_birth_fallback"
                    })

                    # Bounded incremental centroid update
                    n_eff = min(member_counts.get(best_hub, 1), CENTROID_MAX_MEMBERS)
                    updated_centroid = (active_centroids[best_hub] * n_eff + emb) / (n_eff + 1)
                    updated_centroid = updated_centroid / (np.linalg.norm(updated_centroid) + 1e-9)
                    active_centroids[best_hub] = updated_centroid
                    member_counts[best_hub] = member_counts.get(best_hub, 1) + 1

                    centroid_str = "[" + ",".join(map(str, updated_centroid.tolist())) + "]"
                    cur.execute(
                        """
                        UPDATE event_hubs 
                        SET centroid = %s::vector, member_count = member_count + 1, last_updated_at = NOW() 
                        WHERE id = %s;
                        """,
                        (centroid_str, best_hub)
                    )
                else:
                    remaining_for_birth.append(r)

            # Birthing new event hubs via community detection with multi-author and cohesion guards
            if len(remaining_for_birth) >= BIRTH_MIN_AUTHORS:
                g, post_ids = construct_knn_event_graph(remaining_for_birth, k=4, min_similarity_floor=BIRTH_SIMILARITY_FLOOR)
                communities = nx_comm.louvain_communities(g, weight='weight', resolution=1.0)

                for comm in communities:
                    cluster_post_ids = list(comm)
                    if len(cluster_post_ids) < BIRTH_MIN_AUTHORS:
                        continue

                    # Guardrail 1: Distinct author verification (multi-voice requirement)
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT COALESCE(external_author_id, user_id::text)) AS author_count
                        FROM posts
                        WHERE id = ANY(%s);
                        """,
                        (cluster_post_ids,)
                    )
                    auth_row = cur.fetchone()
                    author_count = (auth_row["author_count"] if isinstance(auth_row, dict) else auth_row[0]) if auth_row else 0
                    if author_count < BIRTH_MIN_AUTHORS:
                        continue

                    # Guardrail 2: Cluster cohesion check (average pairwise intra-cluster similarity >= 0.82)
                    cluster_embeddings = [
                        parse_embedding(r["embedding"] if isinstance(r, dict) else r[1])
                        for r in remaining_for_birth
                        if (r["post_id"] if isinstance(r, dict) else r[0]) in cluster_post_ids
                    ]
                    embs_arr = np.array(cluster_embeddings)
                    norms = np.linalg.norm(embs_arr, axis=1, keepdims=True) + 1e-9
                    embs_norm = embs_arr / norms
                    sim_matrix = np.dot(embs_norm, embs_norm.T)
                    cohesion = float(np.mean(sim_matrix))

                    if cohesion < BIRTH_COHESION_FLOOR:
                        continue

                    # Find anchor post: earliest post with highest engagement
                    cur.execute(
                        """
                        SELECT id FROM posts 
                        WHERE id = ANY(%s) 
                        ORDER BY engagement_score DESC, created_at ASC 
                        LIMIT 1;
                        """,
                        (cluster_post_ids,)
                    )
                    anchor_row = cur.fetchone()
                    anchor_pid = anchor_row["id"] if isinstance(anchor_row, dict) else anchor_row[0]

                    initial_centroid = np.mean(cluster_embeddings, axis=0)
                    initial_centroid = initial_centroid / (np.linalg.norm(initial_centroid) + 1e-9)
                    centroid_str = "[" + ",".join(map(str, initial_centroid.tolist())) + "]"

                    cur.execute(
                        """
                        INSERT INTO event_hubs (centroid, member_count, anchor_post_id, status) 
                        VALUES (%s::vector, %s, %s, 'active') 
                        RETURNING id;
                        """,
                        (centroid_str, len(cluster_post_ids), anchor_pid)
                    )
                    event_row = cur.fetchone()
                    new_event_id = event_row["id"] if isinstance(event_row, dict) else event_row[0]

                    cur.execute(
                        "UPDATE posts SET event_id = %s, assignment_status = 'assigned', assignment_updated_at = NOW() WHERE id = ANY(%s);",
                        (new_event_id, cluster_post_ids)
                    )
                    cur.execute(
                        "DELETE FROM unclustered_posts_buffer WHERE post_id = ANY(%s);",
                        (cluster_post_ids,)
                    )

                    active_centroids[new_event_id] = initial_centroid
                    member_counts[new_event_id] = len(cluster_post_ids)

                    # Emit outbox events for birth & assignment
                    write_outbox(cur, "event.created", None, new_event_id, {
                        "event_id": new_event_id,
                        "anchor_post_id": anchor_pid,
                        "initial_member_count": len(cluster_post_ids),
                        "cohesion": round(cohesion, 4),
                    })
                    for c_pid in cluster_post_ids:
                        write_outbox(cur, "post.assigned", c_pid, new_event_id, {
                            "post_id": c_pid,
                            "event_id": new_event_id,
                            "status": "assigned",
                            "source": "event_birth_community"
                        })


if __name__ == "__main__":
    run_clustering_pipeline()
