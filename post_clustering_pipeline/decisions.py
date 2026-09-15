"""Decision-journal writers (the de-black-boxer).

Every final status (assigned / candidate / unassigned / noise) appends one
row here IN THE SAME TRANSACTION as the status change, with the exact
similarity, runner, threshold, margin, and versions that produced it. This is
the raw material for the rollups, threshold tuning, and the owner-facing
decisions inspector. Speculative rows are never written (only finalized posts).
"""
from __future__ import annotations

from psycopg2.extras import execute_values


def log_decision(cur, *, post_id: int, event_id: int | None, similarity: float | None,
                 runner_similarity: float | None, threshold_used: float | None,
                 margin_budget: float | None, status: str, confidence: float | None,
                 policy_version: str | None, model_version: str | None, reason: str) -> None:
    cur.execute(
        """
        INSERT INTO assignment_decision_log
            (post_id, event_id, similarity, runner_similarity, threshold_used,
             margin_budget, status, confidence, policy_version, model_version, reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (post_id, event_id, similarity, runner_similarity, threshold_used,
         margin_budget, status, confidence, policy_version, model_version, reason)
    )


def log_decisions_bulk(cur, rows) -> None:
    """Bulk insert into the decision journal.

    ``rows``: iterable of (post_id, event_id, similarity, runner_similarity,
    threshold_used, margin_budget, status, confidence, policy_version,
    model_version, reason).
    """
    if not rows:
        return
    execute_values(
        cur,
        """
        INSERT INTO assignment_decision_log
            (post_id, event_id, similarity, runner_similarity, threshold_used,
             margin_budget, status, confidence, policy_version, model_version, reason)
        VALUES %s;
        """,
        rows,
        template="(%s::int, %s::int, %s::double precision, %s::double precision, "
                 "%s::double precision, %s::double precision, %s::text, %s::double precision, "
                 "%s::text, %s::text, %s::text)",
    )