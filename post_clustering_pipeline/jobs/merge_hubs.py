"""Reconciliation job to detect, validate, and merge duplicate event hubs."""
from __future__ import annotations

import argparse
import numpy as np
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity

from ..db import get_db_cursor
from ..events import write_outbox
from ..config import CENTROID_MAX_MEMBERS

MERGE_SIMILARITY_THRESHOLD = 0.90
MAX_TEMPORAL_DIFF_HOURS = 12.0


def parse_embedding(emb_text: str) -> np.ndarray:
    return np.array(list(map(float, emb_text.strip("[]").split(","))))


def reconcile_hub_merges(similarity_threshold: float = MERGE_SIMILARITY_THRESHOLD) -> int:
    """Detect and soft-merge overlapping event hubs meeting strict multi-factor guardrails."""
    merged_count = 0

    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            SELECT id, member_count, centroid::text, last_updated_at, anchor_post_id
            FROM event_hubs
            WHERE is_active = TRUE 
              AND centroid IS NOT NULL 
              AND last_updated_at >= NOW() - INTERVAL '48 hours'
            ORDER BY member_count DESC, id ASC;
            """
        )
        rows = cur.fetchall()

        if len(rows) < 2:
            return 0

        hub_ids = []
        member_counts = []
        centroids = []
        timestamps = []
        anchors = []

        for r in rows:
            hid = r["id"] if isinstance(r, dict) else r[0]
            m_count = r["member_count"] if isinstance(r, dict) else r[1]
            emb = parse_embedding(r["centroid"] if isinstance(r, dict) else r[2])
            ts = r["last_updated_at"] if isinstance(r, dict) else r[3]
            anc = r.get("anchor_post_id") if isinstance(r, dict) else (r[4] if len(r) > 4 else None)

            hub_ids.append(hid)
            member_counts.append(m_count or 1)
            centroids.append(emb / (np.linalg.norm(emb) + 1e-9))
            timestamps.append(ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts)
            anchors.append(anc)

        centroids_arr = np.array(centroids)
        sim_matrix = cosine_similarity(centroids_arr)

        merged_away = set()
        n = len(hub_ids)

        for i in range(n):
            if hub_ids[i] in merged_away:
                continue

            for j in range(i + 1, n):
                if hub_ids[j] in merged_away:
                    continue

                sim = sim_matrix[i, j]
                if sim < similarity_threshold:
                    continue

                # Guardrail: Check temporal co-occurrence window
                time_diff = abs((timestamps[i] - timestamps[j]).total_seconds()) / 3600.0
                if time_diff > MAX_TEMPORAL_DIFF_HOURS:
                    continue

                # Hub i survives (larger member_count or lower ID), Hub j merges in
                surviving_id = hub_ids[i]
                absorbed_id = hub_ids[j]

                nA = min(member_counts[i], CENTROID_MAX_MEMBERS)
                nB = min(member_counts[j], CENTROID_MAX_MEMBERS)

                # Combined weighted centroid
                c_combined = (centroids[i] * nA + centroids[j] * nB) / (nA + nB)
                c_combined = c_combined / (np.linalg.norm(c_combined) + 1e-9)
                c_str = "[" + ",".join(map(str, c_combined.tolist())) + "]"
                new_total_members = member_counts[i] + member_counts[j]

                # Soft merge in DB: point absorbed hub to surviving hub and update status
                cur.execute(
                    """
                    UPDATE event_hubs 
                    SET is_active = FALSE, status = 'merged', merged_into_id = %s, last_updated_at = NOW() 
                    WHERE id = %s;
                    """,
                    (surviving_id, absorbed_id)
                )

                # Reassign posts from absorbed hub to surviving hub
                cur.execute(
                    "UPDATE posts SET event_id = %s WHERE event_id = %s;",
                    (surviving_id, absorbed_id)
                )

                # Update surviving hub's centroid and member count (and adopt anchor if missing)
                surviving_anchor = anchors[i] or anchors[j]
                cur.execute(
                    """
                    UPDATE event_hubs 
                    SET centroid = %s::vector, member_count = %s, anchor_post_id = COALESCE(anchor_post_id, %s), last_updated_at = NOW() 
                    WHERE id = %s;
                    """,
                    (c_str, new_total_members, surviving_anchor, surviving_id)
                )

                # Emit outbox notification for downstream integration
                write_outbox(cur, "event.merged", None, surviving_id, {
                    "surviving_hub_id": surviving_id,
                    "absorbed_hub_id": absorbed_id,
                    "similarity": float(sim),
                    "new_member_count": new_total_members
                })

                merged_away.add(absorbed_id)
                member_counts[i] = new_total_members
                centroids[i] = c_combined
                merged_count += 1

    return merged_count


def main():
    p = argparse.ArgumentParser(description="Reconcile and merge duplicate event hubs.")
    p.add_argument("--threshold", type=float, default=MERGE_SIMILARITY_THRESHOLD)
    args = p.parse_args()

    count = reconcile_hub_merges(args.threshold)
    print(f"[Hub Reconciliation] Merged {count} duplicate event hubs.")


if __name__ == "__main__":
    main()
