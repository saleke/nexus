import numpy as np
import torch
import networkx as nx
import networkx.algorithms.community as nx_comm
from datetime import datetime, timezone

from psycopg2.extras import execute_values

from ..db import get_db_cursor, extract_val
from ..events import write_outbox_bulk
from ..embed_io import vector_to_array_literal, parse_vector_literal
from ..centroid import bounded_rolling_centroid, normalize
from ..refs import title_from_seed, unique_handle
from ..membership import record_memberships
from ..nlp import identity_entity_tokens
from ..config import (
    BIRTH_ASSIGN_THRESHOLD,
    BIRTH_SIMILARITY_FLOOR,
    BIRTH_MIN_AUTHORS,
    BIRTH_COHESION_FLOOR,
    CENTROID_MAX_MEMBERS,
    ACTIVE_HUB_FRESHNESS_HOURS,
)


# Identity-entity atomization (see _entity_split_components): two entity
# tokens are treated as the same actor only when their post sets overlap by at
# least this fraction of the smaller set. Aliases / same-story actors overlap
# ~1.0; rival actors co-occur only on cross-mention posts (measured <= 0.56).
ENTITY_ATOM_MIN_OVERLAP = 0.6
# Below this many posts an entity token is too rare to synthesize an actor
# from; its posts are attached by embedding instead of federating tokens.
ENTITY_ATOM_MIN_SUPPORT = 2


def parse_embedding(emb_text: str) -> np.ndarray:
    return parse_vector_literal(emb_text)


def get_active_centroids(cur):
    """Retrieve canonical active centroids and member counts directly from event_hubs.

    The rolling-centroid weight uses TOTAL live members (member_count distinct
    signals + repost_count folded copies): a storm of 1000 identical reposts
    is one displayed member but 1000 posts of real mass, and the pre-fold
    rigidity semantics must be preserved so a new post moves the centroid the
    same way it always did.
    """
    cur.execute(
        """
        SELECT id, member_count, repost_count, centroid::text 
        FROM event_hubs 
        WHERE is_active = TRUE
          AND centroid IS NOT NULL 
          AND last_updated_at >= NOW() - %s::interval;
        """,
        (f"{ACTIVE_HUB_FRESHNESS_HOURS} hours",)
    )
    rows = cur.fetchall()

    centroids = {}
    member_counts = {}
    for r in rows:
        eid = r["id"] if isinstance(r, dict) else r[0]
        m_count = r["member_count"] if isinstance(r, dict) else r[1]
        reposts = r["repost_count"] if isinstance(r, dict) else r[2]
        emb = parse_embedding(r["centroid"] if isinstance(r, dict) else r[3])
        centroids[eid] = emb
        member_counts[eid] = (m_count or 1) + (reposts or 0)

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


def _entity_split_components(
    cur, post_ids: list[int], emb_by_pid: dict[int, "np.ndarray"] | None = None
) -> list[list[int]]:
    """Split a (candidate) community by strong identity entities.

    Embeddings cannot separate confusable actor threads (an Apple and a Samsung
    launch read identically as 'phone launch'); the per-post identity entities
    captured at ingestion can. The naive rule "union any two posts sharing an
    entity" fails, though: a single cross-mention post ("Apple's reveal is
    being compared to Samsung's event") welds two otherwise-disjoint actors
    into one thread. So entities are first ATOMIZED - two entity tokens belong
    to the same actor only when their post sets overlap substantially
    (min-set overlap >= ENTITY_ATOM_MIN_OVERLAP), i.e. they nearly always
    co-occur. Aliases and same-story actors overlap ~1.0; rivals co-occur only
    on comparison posts (measured <= 0.56). A post naming several actors, and
    a post with no strong entity, are then attached to the nearest actor
    centroid by embedding rather than riding the majority blindly.

    A community that is one entity thread (or has no entities at all) returns
    a single component -> behaviour identical to the pre-split path.
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

    if not ents:
        return [post_ids]

    entity_to_posts: dict[str, set[int]] = {}
    for pid, es in ents.items():
        for e in es:
            entity_to_posts.setdefault(e, set()).add(pid)

    parent = {e: e for e in entity_to_posts}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    names = list(entity_to_posts)
    for i in range(len(names)):
        a = entity_to_posts[names[i]]
        for j in range(i + 1, len(names)):
            b = entity_to_posts[names[j]]
            support = min(len(a), len(b))
            if support < ENTITY_ATOM_MIN_SUPPORT:
                continue
            if len(a & b) / support >= ENTITY_ATOM_MIN_OVERLAP:
                union(names[i], names[j])

    atoms: dict[str, set[int]] = {}
    for e, pids in entity_to_posts.items():
        atoms.setdefault(find(e), set()).update(pids)

    atom_ids = list(atoms)
    if len(atom_ids) == 1:
        return [post_ids]

    centroids: dict[str, "np.ndarray"] = {}
    if emb_by_pid:
        for aid in atom_ids:
            vecs = [emb_by_pid[p] for p in atoms[aid] if p in emb_by_pid]
            if not vecs:
                continue
            c = np.mean(np.array(vecs), axis=0)
            centroids[aid] = c / (np.linalg.norm(c) + 1e-9)

    groups: dict[str, list[int]] = {}
    for pid in post_ids:
        s = ents.get(pid, set())
        cands = {find(e) for e in s} if s else set(atom_ids)
        cands = [c for c in cands if c in atoms]
        if not cands:
            continue
        if len(cands) == 1 or not centroids or emb_by_pid is None or pid not in emb_by_pid:
            best = max(cands, key=lambda c: len(atoms[c]))
        else:
            v = emb_by_pid[pid]
            v = v / (np.linalg.norm(v) + 1e-9)
            best = max(
                cands,
                key=lambda c: float(v @ centroids[c]) if c in centroids else -1.0,
            )
        groups.setdefault(best, []).append(pid)

    result = list(groups.values())
    if len(result) <= 1:
        return result or [post_ids]

    # A single stray post must not fracture an otherwise coherent community
    # (e.g. one post whose entity string differs from the community's wording).
    largest = max(result, key=len)
    strays = [p for g in result if g is not largest and len(g) == 1 for p in g]
    if strays:
        largest.extend(strays)
        result = [g for g in result if g is largest or len(g) >= 2]
    return result


def _load_active_centroids():
    with get_db_cursor(commit=True) as cur:
        return get_active_centroids(cur)


def _load_buffer_rows():
    with get_db_cursor(commit=True) as cur:
        # A post can be deleted while its row still sits in the buffer (a
        # tombstone is never back-purged from it); re-clustering it here would
        # resurrect dead content into a fresh hub. Drop tombstoned buffer rows
        # and read only live posts.
        cur.execute(
            """
            DELETE FROM unclustered_posts_buffer b
            USING posts p
            WHERE b.post_id = p.id AND p.deleted_at IS NOT NULL;
            """
        )
        cur.execute(
            """
            SELECT b.post_id, b.embedding::text, b.created_at
            FROM unclustered_posts_buffer b
            JOIN posts p ON p.id = b.post_id AND p.deleted_at IS NULL
            ORDER BY b.created_at ASC;
            """
        )
        return cur.fetchall()


def _age_unclustered_posts() -> None:
    """Age old unclustered posts to 'noise' in its own checkpointed transaction."""
    from ..decisions import log_decisions_bulk
    from ..policy import current_versions

    with get_db_cursor(commit=True) as cur:
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
            # Ageing is a decision, so it is one the operator's clients must
            # hear about; otherwise a post can silently vanish from every hub
            # view while the client's copy still shows it assigned.
            write_outbox_bulk(cur, [
                ("post.noise", pid, None, {"post_id": pid, "status": "noise", "reason": "aged_unclustered"})
                for pid in aged_ids
            ])


def best_centroid(emb, centroid_ids, centroid_matrix, centroid_norms):
    """Nearest active centroid as ``(hub_id, similarity)``, vectorized.

    The centroid rows are ordered exactly like ``active_centroids`` and
    ``argmax`` returns the first maximum, preserving the strict ``>`` tie-break
    of the original per-hub Python loop (an earlier hub wins a tie). Returns
    ``(None, -1.0)`` when there are no centroids, matching the old ``best_sim``
    sentinel.
    """
    if not centroid_ids:
        return None, -1.0
    emb_norm = float(np.linalg.norm(emb))
    dots = centroid_matrix @ emb
    sims = dots / (emb_norm * centroid_norms + 1e-9)
    best_idx = int(np.argmax(sims))
    return centroid_ids[best_idx], float(sims[best_idx])


def _flush_fallback_assignments(cur, assigned, active_centroids, pv, mv) -> None:
    """Persist one slice's fallback assignments with bulk round-trips.

    ``assigned`` is the ordered ``[(post_id, hub_id, similarity), ...]`` chosen
    by the sequential in-memory loop; each hub's rolling centroid has already
    advanced to its end-of-slice value, so only that final centroid is written
    (with a ``member_count`` delta, never an absolute value, so concurrent
    increments are not clobbered).

    The UPDATE is CAS-guarded: it only wins on rows still in the assignment
    pool (``pending/processing/unassigned/candidate``) and never touches a
    folded repost. A row the candidate sweep (or another birth) already
    finalised elsewhere is skipped here - not clobbered - and its memberships /
    outbox / buffer-row are handled by whatever made it final. It is the
    sweep's 30s job, not this hourly job, to resolve contested candidates.
    """
    from ..decisions import log_decisions_bulk

    if not assigned:
        return
    post_rows = [(int(pid), int(hub), float(sim)) for pid, hub, sim in assigned]

    execute_values(
        cur,
        """
        UPDATE posts p
        SET event_id = v.event_id,
            assignment_status = 'assigned',
            assignment_confidence = v.confidence,
            assignment_updated_at = NOW()
        FROM (VALUES %s) AS v(post_id, event_id, confidence)
        WHERE p.id = v.post_id
          AND p.repost_of_id IS NULL
          AND p.assignment_status IN ('pending', 'processing', 'unassigned', 'candidate')
        RETURNING p.id, v.event_id AS event_id;
        """,
        post_rows,
        template="(%s::int, %s::int, %s::double precision)",
    )
    applied = {}
    for r in (cur.fetchall() or []):
        applied[int(extract_val(r, "id", 0))] = int(extract_val(r, "event_id", 1))
    if not applied:
        return
    applied_ids = sorted(applied)

    cur.execute(
        "DELETE FROM unclustered_posts_buffer WHERE post_id = ANY(%s::int[]);",
        (applied_ids,)
    )
    record_memberships(cur, [(pid, hub, "member") for pid, hub in applied.items()])

    # Fold repost floods (identical content reaching an existing hub) before
    # events are written, so duplicates are never emitted as separate
    # assignments. Near-dup folding is deferred to the sweep to keep this
    # path O(distinct posts) per slice.
    hub_ids = sorted(set(applied.values()))
    from ..fold import fold_hubs
    fold_hubs(cur, hub_ids, near_dup=False)

    cur.execute(
        "SELECT id FROM posts WHERE id = ANY(%s::int[]) AND repost_of_id IS NOT NULL;",
        (applied_ids,)
    )
    repost_ids = {extract_val(r, "id", 0) for r in (cur.fetchall() or [])}

    write_outbox_bulk(cur, [
        (
            "post.assigned", pid, hub,
            {
                "post_id": pid,
                "event_id": hub,
                "confidence": float(sim),
                "status": "assigned",
                "source": "event_birth_fallback",
            },
        )
        for pid, hub, sim in assigned
        if pid in applied and pid not in repost_ids
    ])
    log_decisions_bulk(cur, [
        (
            pid, hub, float(sim), None, BIRTH_ASSIGN_THRESHOLD,
            float(sim) - BIRTH_ASSIGN_THRESHOLD,
            "repost" if pid in repost_ids else "assigned", float(sim),
            pv, mv, "event_birth_fallback",
        )
        for pid, hub, sim in assigned
        if pid in applied
    ])

    for hub in hub_ids:
        cur.execute(
            """
            UPDATE event_hubs
            SET centroid = %s::vector, last_updated_at = NOW()
            WHERE id = %s;
            """,
            (vector_to_array_literal(active_centroids[hub]), hub),
        )


def _process_slice(slice_rows, active_centroids, member_counts) -> None:
    """Process one temporal slice in its own transaction (a checkpoint).

    A crash or statement timeout can therefore lose at most the current slice
    instead of rolling back the whole run, and no single transaction holds locks
    for the entire (possibly large) buffer.
    """
    from ..decisions import log_decisions_bulk
    from ..policy import current_versions

    with get_db_cursor(commit=True) as cur:
        pv, mv = current_versions(cur)

        centroid_ids = list(active_centroids.keys())
        centroid_index = {eid: i for i, eid in enumerate(centroid_ids)}
        if centroid_ids:
            centroid_matrix = np.array([active_centroids[e] for e in centroid_ids])
            centroid_norms = np.linalg.norm(centroid_matrix, axis=1)
        else:
            centroid_matrix = np.zeros((0, 0))
            centroid_norms = np.zeros(0)

        remaining_for_birth = []
        assigned = []

        for r in slice_rows:
            pid = r["post_id"] if isinstance(r, dict) else r[0]
            emb = parse_embedding(r["embedding"] if isinstance(r, dict) else r[1])

            best_hub, best_sim = best_centroid(emb, centroid_ids, centroid_matrix, centroid_norms)

            # Assign if similarity meets calibrated birth-assign threshold (e.g. 0.82)
            if best_hub and best_sim >= BIRTH_ASSIGN_THRESHOLD:
                assigned.append((pid, best_hub, best_sim))

                # Bounded incremental centroid update (order-sensitive rolling
                # centroid, advanced in memory only; flushed once per slice).
                updated_centroid = normalize(bounded_rolling_centroid(
                    active_centroids[best_hub],
                    member_counts.get(best_hub, 1),
                    [emb],
                    CENTROID_MAX_MEMBERS,
                ))
                active_centroids[best_hub] = updated_centroid
                member_counts[best_hub] = member_counts.get(best_hub, 1) + 1
                idx = centroid_index[best_hub]
                centroid_matrix[idx] = updated_centroid
                centroid_norms[idx] = np.linalg.norm(updated_centroid)
            else:
                remaining_for_birth.append(r)

        if assigned:
            _flush_fallback_assignments(cur, assigned, active_centroids, pv, mv)

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
                for group in _entity_split_components(cur, cluster_post_ids, emb_by_pid):
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
                    "UPDATE posts SET event_id = %s, assignment_status = 'assigned', assignment_updated_at = NOW() "
                    "WHERE id = ANY(%s) AND repost_of_id IS NULL AND event_id IS NULL "
                    "AND assignment_status IN ('pending', 'processing', 'unassigned', 'candidate') "
                    "RETURNING id;",
                    (new_event_id, cluster_post_ids)
                )
                applied_pids = [int(extract_val(r, "id", 0)) for r in (cur.fetchall() or [])]
                if not applied_pids:
                    continue
                cur.execute(
                    "DELETE FROM unclustered_posts_buffer WHERE post_id = ANY(%s);",
                    (applied_pids,)
                )

                # First-class membership ledger: the seed post carries the
                # 'seed' role, everyone else 'member' - same transaction as
                # the event_id pointer so the two never disagree.
                record_memberships(cur, [
                    (c_pid, new_event_id, "seed" if c_pid == anchor_pid else "member")
                    for c_pid in applied_pids
                ])

                # Repost-folding at birth: a storm of identical posts arriving
                # together would otherwise render as N members. Fold exact
                # copies by content signature, then near-copies by embedding
                # (the component's vectors are already in memory) while this
                # community is still small - before the hub hits the stream
                # assignment path. The canonical keeps event_id; duplicates
                # become reposts and are never emitted as separate events.
                from ..fold import fold_hubs
                fold_hubs(cur, [new_event_id], near_dup=True, embeddings_by_id=emb_by_pid)

                cur.execute(
                    "SELECT id FROM posts WHERE event_id = %s AND repost_of_id IS NULL ORDER BY id;",
                    (new_event_id,)
                )
                canonical_pids = [extract_val(r, "id", 0) for r in (cur.fetchall() or [])]
                cur.execute(
                    "SELECT id FROM posts WHERE event_id = %s AND repost_of_id IS NOT NULL;",
                    (new_event_id,)
                )
                repost_pids = {extract_val(r, "id", 0) for r in (cur.fetchall() or [])}

                active_centroids[new_event_id] = initial_centroid
                member_counts[new_event_id] = len(cluster_post_ids)

                # Emit outbox events for birth & assignment in one seq-bump +
                # one INSERT (event.created first, then each canonical
                # assignment only - folded reposts stay below the wire).
                write_outbox_bulk(cur, [
                    (
                        "event.created", None, new_event_id,
                        {
                            "event_id": new_event_id,
                            "anchor_post_id": anchor_pid,
                            "initial_member_count": len(canonical_pids),
                            "reposts_folded": len(repost_pids),
                            "cohesion": round(cohesion, 4),
                        },
                    ),
                    *[
                        (
                            "post.assigned", c_pid, new_event_id,
                            {
                                "post_id": c_pid,
                                "event_id": new_event_id,
                                "status": "assigned",
                                "source": "event_birth_community",
                            },
                        )
                        for c_pid in canonical_pids
                    ],
                ])

                # Journal the membership decision: born-hub assignments must
                # be as explainable in the decisions inspector as every
                # other final status (they previously were not, which made
                # born hubs invisible on the panel). Folded reposts are
                # journaled too, marked with their folded status.
                log_decisions_bulk(cur, [
                    (
                        c_pid, new_event_id, cohesion, None, BIRTH_COHESION_FLOOR,
                        0.0 if c_pid in repost_pids else cohesion - BIRTH_COHESION_FLOOR,
                        "repost" if c_pid in repost_pids else "assigned", cohesion,
                        pv, mv, "event_birth_community" if c_pid not in repost_pids else "repost_folded",
                    )
                    for c_pid in applied_pids
                ])


def run_clustering_pipeline():
    """Cluster buffered posts, checkpointing at each temporal slice.

    Aging runs in its own transaction, the active centroids and buffer are read
    once (snapshot), and every 1-hour slice is processed in its own transaction
    so a failure loses at most one slice and no lock is held for the full run.
    The caller (``run_event_birth_scheduled``) already holds the
    ``event_birth_job`` distributed lock for the whole call.
    """
    _age_unclustered_posts()

    active_centroids, member_counts = _load_active_centroids()
    buffer_rows = _load_buffer_rows()
    if not buffer_rows:
        return

    for slice_rows in temporal_dataloader(buffer_rows, slice_hours=1.0):
        _process_slice(slice_rows, active_centroids, member_counts)


if __name__ == "__main__":
    run_clustering_pipeline()
