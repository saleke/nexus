import numpy as np
import torch
import networkx as nx
import networkx.algorithms.community as nx_comm
from datetime import datetime, timezone

from ..db import get_db_cursor, extract_val
from ..events import write_outbox
from ..embed_io import vector_to_array_literal, parse_vector_literal
from ..centroid import bounded_rolling_centroid, normalize
from ..refs import title_from_seed, unique_handle
from ..membership import record_membership
from ..nlp import identity_entity_tokens
from ..config import (
    BIRTH_ASSIGN_THRESHOLD,
    BIRTH_SIMILARITY_FLOOR,
    BIRTH_MIN_AUTHORS,
    BIRTH_COHESION_FLOOR,
    CENTROID_MAX_MEMBERS,
)


def parse_embedding(emb_text: str) -> np.ndarray:
    return parse_vector_literal(emb_text)


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


def _entity_split_components(cur, post_ids: list[int]) -> list[list[int]]:
    """Split a (candidate) community by strong identity entities.

    Embeddings cannot separate confusable actor threads (Apple vs Samsung
    launch read identically as 'phone launch'); the per-post identity
    entities captured at ingestion can. Posts are unioned when they share any
    strong entity; posts with NO strong entity are generic and attach to the
    largest component (they follow the embedding majority rather than strand).
    A community that is one entity thread (or no entities at all) returns a
    single component -> behaviour identical to the pre-split path.
    """
    if len(post_ids) <= 1:
        return [post_ids]

    cur.execute("SELECT id, entities FROM posts WHERE id = ANY(%s);", (post_ids,))
    ents: dict[int, set[str]] = {}
    for r in (cur.fetchall() or []):
        pid = extract_val(r, "id", 0)
        vals = extract_val(r, "entities", 1) or []
        # Sanitized identity evidence only: generic collective tokens
        # (insiders/fans/critics/observers) span every topic and would chain
        # distinct communities into one component, defeating the split.
        strong = identity_entity_tokens(vals)
        if strong:
            ents[pid] = strong

    parent = {p: p for p in post_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    entity_to_posts: dict[str, list[int]] = {}
    for pid, es in ents.items():
        for e in es:
            entity_to_posts.setdefault(e, []).append(pid)
    for pids in entity_to_posts.values():
        base = pids[0]
        for p in pids[1:]:
            union(base, p)

    comps: dict[int, list[int]] = {}
    for p in post_ids:
        comps.setdefault(find(p), []).append(p)
    groups = list(comps.values())

    if len(groups) == 1:
        return groups

    # Detach ONLY labeled sub-groups with >= 2 members. A 1-member labeled
    # group is far more likely a single post whose entity string differs from
    # the community's wording (e.g. "Amazon Web Services" vs "AWS") than a
    # second real actor, so it must NOT fracture the community.
    largest = max(groups, key=len)
    kept = []
    for g in groups:
        if g is largest:
            kept.append(g)
            continue
        if all(p not in ents for p in g):
            largest.extend(g)  # unlabeled/generic posts ride the majority
        elif len(g) >= 2:
            kept.append(g)
        else:
            largest.extend(g)  # singleton labeled post rides the majority
    return kept


def run_clustering_pipeline():
    with get_db_cursor(commit=True) as cur:
        # Age out old unclustered posts and prune buffer. Every final status is
        # journaled in the SAME transaction so 'noise' is explainable everywhere.
        from ..decisions import log_decisions_bulk
        from ..policy import current_versions
        pv, mv = current_versions(cur)
        cur.execute(
            """
            UPDATE posts SET assignment_status = 'noise', assignment_updated_at = NOW()
            WHERE event_id IS NULL AND assignment_status IN ('pending', 'candidate', 'unassigned')
              AND created_at < NOW() - INTERVAL '24 hours'
            RETURNING id;
            """
        )
        aged_ids = [int(extract_val(r, "id", 0)) for r in (cur.fetchall() or [])]
        if aged_ids:
            cur.execute(
                "DELETE FROM unclustered_posts_buffer WHERE post_id = ANY(%s::int[]);",
                (aged_ids,)
            )
            log_decisions_bulk(cur, [
                (pid, None, None, None, None, None, "noise", None, pv, mv, "aged_unclustered_noise")
                for pid in aged_ids
            ])

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
                    record_membership(cur, pid, best_hub)
                    write_outbox(cur, "post.assigned", pid, best_hub, {
                        "post_id": pid,
                        "event_id": best_hub,
                        "confidence": float(best_sim),
                        "status": "assigned",
                        "source": "event_birth_fallback"
                    })
                    log_decisions_bulk(cur, [(
                        pid, best_hub, float(best_sim), None, BIRTH_ASSIGN_THRESHOLD,
                        float(best_sim) - BIRTH_ASSIGN_THRESHOLD, "assigned", float(best_sim),
                        pv, mv, "event_birth_fallback"
                    )])

                    # Bounded incremental centroid update
                    updated_centroid = normalize(bounded_rolling_centroid(
                        active_centroids[best_hub],
                        member_counts.get(best_hub, 1),
                        [emb],
                        CENTROID_MAX_MEMBERS,
                    ))
                    active_centroids[best_hub] = updated_centroid
                    member_counts[best_hub] = member_counts.get(best_hub, 1) + 1

                    centroid_str = vector_to_array_literal(updated_centroid)
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

                emb_by_pid = {
                    (r["post_id"] if isinstance(r, dict) else r[0]):
                    parse_embedding(r["embedding"] if isinstance(r, dict) else r[1])
                    for r in remaining_for_birth
                }

                # Identity-entity disambiguation FIRST: a community that mixes
                # confusable actors (Apple vs Samsung launch reads identically
                # to the embedder) is split into per-actor components BEFORE the
                # multi-author and cohesion guards evaluate each one, so the
                # mixed community can never be born as one contaminated hub.
                candidate_groups = []
                for comm in communities:
                    cluster_post_ids = list(comm)
                    if len(cluster_post_ids) < BIRTH_MIN_AUTHORS:
                        continue
                    for group in _entity_split_components(cur, cluster_post_ids):
                        if len(group) >= BIRTH_MIN_AUTHORS:
                            candidate_groups.append(group)

                for cluster_post_ids in candidate_groups:
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

                    # Guardrail 2: Cluster cohesion check, measured per component
                    # (the merged community's cohesion was contaminated by the
                    # wrong-actor posts and is not a valid signal for either half)
                    cluster_embeddings = [emb_by_pid[p] for p in cluster_post_ids if p in emb_by_pid]
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

                    initial_centroid = normalize(np.mean(cluster_embeddings, axis=0))
                    centroid_str = vector_to_array_literal(initial_centroid)

                    # Hub identity is captured AT BIRTH onto the hub row itself
                    # (title + stable handle) instead of being re-derived from a
                    # member post's content on every render.
                    title = title_from_seed(cur, anchor_pid, fallback="hub")
                    handle = unique_handle(cur, title)

                    cur.execute(
                        """
                        INSERT INTO event_hubs (centroid, member_count, seed_post_id, status, title, handle) 
                        VALUES (%s::vector, %s, %s, 'active', %s, %s) 
                        RETURNING id;
                        """,
                        (centroid_str, len(cluster_post_ids), anchor_pid, title, handle)
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

                    # First-class membership ledger: the seed post carries the
                    # 'seed' role, everyone else 'member' - same transaction as
                    # the event_id pointer so the two never disagree.
                    for c_pid in cluster_post_ids:
                        record_membership(
                            cur, c_pid, new_event_id,
                            role="seed" if c_pid == anchor_pid else "member"
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

                    # Journal the membership decision: born-hub assignments must
                    # be as explainable in the decisions inspector as every
                    # other final status (they previously were not, which made
                    # born hubs invisible on the panel).
                    log_decisions_bulk(cur, [
                        (c_pid, new_event_id, cohesion, None, BIRTH_COHESION_FLOOR,
                         cohesion - BIRTH_COHESION_FLOOR, "assigned", cohesion,
                         pv, mv, "event_birth_community")
                        for c_pid in cluster_post_ids
                    ])


if __name__ == "__main__":
    run_clustering_pipeline()
