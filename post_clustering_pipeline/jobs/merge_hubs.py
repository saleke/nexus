"""Reconciliation job to detect, validate, and merge duplicate event hubs.

Detection happens here (multi-factor guardrails: similarity + temporal
co-occurrence); the actual merge is delegated to ``merges.apply_merge_soft`` —
the SAME canonical, advisory-locked, snapshot-preserving path the client and
the panel use. There is exactly one merge implementation in the system; this
job only decides WHICH pairs to fold, never HOW to fold them.
"""
from __future__ import annotations

import argparse
import numpy as np
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity

from ..db import get_db_cursor, extract_val
from ..centroid import normalize
from ..merges import apply_merge_soft, lock_hub_pair

MERGE_SIMILARITY_THRESHOLD = 0.90
MAX_TEMPORAL_DIFF_HOURS = 12.0


def reconcile_hub_merges(similarity_threshold: float = MERGE_SIMILARITY_THRESHOLD) -> int:
    """Detect overlapping hubs and fold them through the shared merge path."""
    merged_count = 0

    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            SELECT id, member_count, centroid::text, last_updated_at
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

        from ..embed_io import parse_vector_literal

        hub_ids = []
        member_counts = []
        centroids = []
        timestamps = []

        for r in rows:
            hid = extract_val(r, "id", 0)
            centroids.append(normalize(parse_vector_literal(extract_val(r, "centroid", 2))))
            hub_ids.append(hid)
            member_counts.append(int(extract_val(r, "member_count", 1) or 1))
            ts = extract_val(r, "last_updated_at", 3)
            timestamps.append(ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts)

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

                sim = float(sim_matrix[i, j])
                if sim < similarity_threshold:
                    continue

                time_diff = abs((timestamps[i] - timestamps[j]).total_seconds()) / 3600.0
                if time_diff > MAX_TEMPORAL_DIFF_HOURS:
                    continue

                # Survivor = larger hub (or lower id on tie); the other folds in.
                if member_counts[i] > member_counts[j]:
                    target_id, source_id = hub_ids[i], hub_ids[j]
                else:
                    target_id, source_id = hub_ids[j], hub_ids[i]

                # One shared implementation: canonical resolution, advisory
                # lock on the pair, member snapshot, centroid rebalance,
                # hub_merges row, outbox event (with references), feedback tag.
                lock_hub_pair(cur, source_id, target_id)
                apply_merge_soft(cur, source_id, target_id, initiated_by="system",
                                 actor="hub-reconciliation", note=f"similarity {sim:.4f}")

                merged_away.add(source_id)
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