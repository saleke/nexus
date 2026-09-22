"""Reconciliation job to detect, validate, and merge duplicate event hubs.

Detection happens here (multi-factor guardrails: similarity + temporal
co-occurrence); the actual merge is delegated to ``merges.apply_merge_soft`` —
the SAME canonical, advisory-locked, snapshot-preserving path the client and
the panel use. There is exactly one merge implementation in the system; this
job only decides WHICH pairs to fold, never HOW to fold them.
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity

from ..db import get_db_cursor, extract_val
from ..centroid import normalize
from ..merges import apply_merge_soft, lock_hub_pair
from ..nlp import identity_entity_tokens
from ..config import (
    MERGE_SIMILARITY_THRESHOLD,
    MERGE_ENTITY_FLOOR,
    MERGE_ORG_VETO_MIN_ORGS,
    MERGE_NO_IDENTITY_SIM,
    ACTIVE_HUB_FRESHNESS_HOURS,
)

logger = logging.getLogger(__name__)

MAX_TEMPORAL_DIFF_HOURS = 12.0

# A hub's identity is what the hub is MOSTLY about, not every token any single
# member mentions. Cross-mention ("Apple's reveal vs Samsung's event") and
# venue/context tokens appear in only a few member posts; treating them as hub
# identity let one bridge post donate a rival's identity to a hub and chain
# bogus ``shared_identity`` merges (Apple+Samsung, Beyonce+Taylor all merged
# this way). An entity token now counts only when it is supported by at least
# this fraction of the hub's posts (and at least MERGE_IDENTITY_MIN_SUPPORT
# posts), so dominant actors keep their identity while bridges cannot weld.
MERGE_IDENTITY_MIN_FRACTION = 0.5
MERGE_IDENTITY_MIN_SUPPORT = 2


def _entity_sets_by_hub(cur, hub_ids: list[int]) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    """Per-hub identity entity profiles in one query (sanitized tokens only).

    Returns ``(orgs_by_hub, identity_by_hub)``:
      * ``orgs`` = dominant ORG tokens (actor identity; the veto basis).
      * ``identity`` = dominant ORG | PRODUCT | NORP tokens (merge-overlap basis).

    Only *dominant* tokens count (see MERGE_IDENTITY_MIN_FRACTION): an entity
    must be carried by a substantial share of the hub's member posts. This
    strips both generic collective tokens (insiders/fans/critics/observers...)
    and incidental cross-mention/venue tokens, either of which would otherwise
    fabricate universal identity overlap and chain distinct events together.
    """
    if not hub_ids:
        return {}, {}
    hub_ids = list(set(hub_ids))
    cur.execute(
        """
        SELECT event_id, COUNT(*) AS total
        FROM posts
        WHERE event_id = ANY(%s::int[]) AND deleted_at IS NULL
        GROUP BY event_id;
        """,
        (hub_ids,)
    )
    totals: dict[int, int] = {}
    for r in (cur.fetchall() or []):
        totals[extract_val(r, "event_id", 0)] = int(extract_val(r, "total", 1) or 0)

    cur.execute(
        """
        SELECT p.event_id, x.ent, COUNT(DISTINCT p.id) AS support
        FROM posts p
        CROSS JOIN LATERAL unnest(p.entities) AS x(ent)
        WHERE p.event_id = ANY(%s::int[]) AND p.entities IS NOT NULL
          AND p.deleted_at IS NULL
        GROUP BY p.event_id, x.ent;
        """,
        (hub_ids,)
    )
    orgs: dict[int, set[str]] = {}
    identity: dict[int, set[str]] = {}
    for r in (cur.fetchall() or []):
        eid = extract_val(r, "event_id", 0)
        ent = str(extract_val(r, "ent", 1) or "")
        support = int(extract_val(r, "support", 2) or 0)
        if not eid or not ent:
            continue
        total = totals.get(eid, 0)
        if support < MERGE_IDENTITY_MIN_SUPPORT:
            continue
        if total and support / total < MERGE_IDENTITY_MIN_FRACTION:
            continue
        orgs.setdefault(eid, set()).update(identity_entity_tokens([ent], org_only=True))
        identity.setdefault(eid, set()).update(identity_entity_tokens([ent]))
    return orgs, identity


def _should_merge(sim: float, org_i, identity_i, org_j, identity_j) -> tuple[bool, str]:
    """Multi-factor guardrail: is this pair the same event, not two similar ones?

    Identity rule (both bands share it):
      * dominant-ORG-conflict vetoes: disjoint ORG evidence means distinct
        actors even when embeddings agree (Tesla vs SpaceX, soccer vs cricket);
      * identity overlap online blesses the fold - a merge needs shared
        ORG/PRODUCT/NORP evidence OR (both sides entity-less AND very high
        embedding agreement). Distinct topics NEVER share sanitized identity
        tokens, so they can never merge no matter how high the similarity.
    """
    if (len(org_i) >= MERGE_ORG_VETO_MIN_ORGS
            and len(org_j) >= MERGE_ORG_VETO_MIN_ORGS
            and org_i.isdisjoint(org_j)):
        return False, "dominant_org_conflict"

    if identity_i and identity_j:
        if identity_i & identity_j:
            return True, f"shared_identity sim {sim:.4f}"
        return False, "identity_divergent"

    if sim >= MERGE_NO_IDENTITY_SIM:
        return True, f"high_similarity_no_identity sim {sim:.4f}"

    return False, "no_identity_evidence"


def reconcile_hub_merges() -> int:
    """Detect overlapping hubs and fold them through the shared merge path."""
    merged_count = 0

    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            SELECT id, member_count, centroid::text, last_updated_at
            FROM event_hubs
            WHERE is_active = TRUE
              AND centroid IS NOT NULL
              AND last_updated_at >= NOW() - %s::interval
            ORDER BY member_count DESC, id ASC;
            """,
            (f"{ACTIVE_HUB_FRESHNESS_HOURS} hours",)
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

        orgs_by_hub, identity_by_hub = _entity_sets_by_hub(cur, hub_ids)

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
                if sim < MERGE_ENTITY_FLOOR:
                    continue

                time_diff = abs((timestamps[i] - timestamps[j]).total_seconds()) / 3600.0
                if time_diff > MAX_TEMPORAL_DIFF_HOURS:
                    continue

                should, note = _should_merge(
                    sim,
                    orgs_by_hub.get(hub_ids[i], set()),
                    identity_by_hub.get(hub_ids[i], set()),
                    orgs_by_hub.get(hub_ids[j], set()),
                    identity_by_hub.get(hub_ids[j], set()),
                )
                if not should:
                    continue

                # Survivor = larger hub (or lower id on tie); the other folds in.
                if member_counts[i] > member_counts[j]:
                    target_id, source_id = hub_ids[i], hub_ids[j]
                else:
                    target_id, source_id = hub_ids[j], hub_ids[i]

                # One shared implementation: canonical resolution, advisory
                # lock on the pair, member snapshot, centroid rebalance,
                # hub_merges row, outbox event (with references), feedback tag.
                try:
                    lock_hub_pair(cur, source_id, target_id)
                    apply_merge_soft(cur, source_id, target_id, initiated_by="system",
                                     actor="hub-reconciliation", note=note)
                    merged_away.add(source_id)
                    merged_count += 1
                except ValueError as exc:
                    # A concurrent merge already resolved this pair to a single
                    # canonical (or the pair became non-mergeable mid-pass).
                    # Swallow per-pair only - the loser's O(n^2) pass must not
                    # roll back wholesale because one candidate vanished.
                    logger.warning("merge pair skipped (%s -> %s): %s", source_id, target_id, exc)

    return merged_count


def reconcile_hub_merges_to_fixpoint(max_rounds: int = 10) -> int:
    """Run reconciliation until no further folds are possible.

    A single pass computes its similarity matrix once, so a fold that shifts a
    survivor's centroid (weighted combine) can push another pair *above* the
    no-identity threshold that it missed earlier. Measuring the live corpus, a
    second pass folded a launch hub that the first pass could not see. The
    guardrails are unchanged - this only applies the same policy to convergence.
    """
    total = 0
    for _ in range(max_rounds):
        merged = reconcile_hub_merges()
        total += merged
        if merged == 0:
            break
    return total


def main():
    p = argparse.ArgumentParser(description="Reconcile and merge duplicate event hubs.")
    p.add_argument("--threshold", type=float, default=MERGE_SIMILARITY_THRESHOLD,
                   help="kept for CLI compatibility; identity-aware judgement is fixed")
    args = p.parse_args()

    count = reconcile_hub_merges_to_fixpoint()
    print(f"[Hub Reconciliation] Merged {count} duplicate event hubs.")


if __name__ == "__main__":
    main()