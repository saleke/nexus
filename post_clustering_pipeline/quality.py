"""Quality rollups, drift, and threshold auto-tuning (the intelligence loop).

Pure estimators live here (unit-testable); the DB runners wrap them with the
window queries and upsert rows. Human labels never grade the system's own
'auto_confirmed' rows, so precision cannot be self-fulfilling.
"""
from __future__ import annotations

from .db import get_db_cursor, extract_val
from .policy import propose_policy_change, apply_policy_change, current_versions
from .config import (
    AUTO_ASSIGN_THRESHOLD,
    AUTO_APPLY_THRESHOLD, MIN_FEEDBACK_SAMPLES,
    AUTO_TUNE_TARGET_PRECISION, AUTO_TUNE_MIN_COVERAGE,
    ROLLUP_WINDOW_HOURS, TUNE_WINDOW_DAYS,
)


def threshold_curve(labeled: list[tuple[float, str]]) -> list[dict]:
    """Precision/coverage across candidate thresholds.

    ``labeled``: [(similarity, 'good' | 'bad')]. 'good' = confirmed assignment,
    'bad' = user-removed assignment. Returns ascending threshold curve.

    Coverage (goods kept / all goods) is monotone non-increasing as the
    threshold rises. Precision is NOT monotone: raising the threshold past a
    good's similarity drops a true positive while a bad sitting above it is
    retained, so a feasible band can open and close. Never optimize with a
    first-feasible-point scan - see ``pick_threshold_auto``.
    """
    goods = [s for s, label in labeled if label == "good"]
    bads = [s for s, label in labeled if label == "bad"]
    candidates = sorted({round(s, 4) for s, _ in labeled})
    curve = []
    for t in candidates:
        tp = sum(1 for s in goods if s >= t)
        fp = sum(1 for s in bads if s >= t)
        curve.append({
            "threshold": t,
            "true_positives": tp,
            "false_positives": fp,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "coverage": (tp / len(goods)) if goods else None,
        })
    return curve


def pick_threshold_auto(labeled, floor: float = AUTO_ASSIGN_THRESHOLD,
                        target_precision: float = AUTO_TUNE_TARGET_PRECISION,
                        min_coverage: float = AUTO_TUNE_MIN_COVERAGE,
                        min_samples: int = MIN_FEEDBACK_SAMPLES) -> float | None:
    """Precision-first feasible threshold, or None (no change).

    Returns None when samples are insufficient or no threshold meets both
    guards - silence is safer than drift on precision-first. Because precision
    is non-monotone (see ``threshold_curve``), we scan ALL feasible points and
    pick the best by (precision desc, coverage desc, threshold asc) - i.e. the
    highest precision that still clears the coverage floor.
    """
    if not labeled or len(labeled) < min_samples:
        return None
    best: tuple[tuple[float, float, float], float] | None = None
    for point in threshold_curve(labeled):
        t = point["threshold"]
        if t < floor or point["precision"] is None:
            continue
        if point["precision"] >= target_precision and (
            point["coverage"] is None or point["coverage"] >= min_coverage
        ):
            rank = (point["precision"], point["coverage"] or 0.0, -t)
            if best is None or rank > best[0]:
                best = (rank, t)
    return best[1] if best else None


def _upsert_rollup(cur, window_start, window_end, policy_version, model_version, source, metrics):
    cur.execute(
        """
        INSERT INTO feedback_rollups
            (window_start, window_end, policy_version, model_version, source,
             total_decisions, confirmed, removed, dismissed, precision_value,
             coverage_value, unlink_rate, drift_index)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (window_start, window_end, policy_version, model_version, source)
        DO UPDATE SET confirmed = EXCLUDED.confirmed, removed = EXCLUDED.removed,
                      dismissed = EXCLUDED.dismissed, precision_value = EXCLUDED.precision_value,
                      coverage_value = EXCLUDED.coverage_value, unlink_rate = EXCLUDED.unlink_rate,
                      drift_index = EXCLUDED.drift_index, total_decisions = EXCLUDED.total_decisions;
        """,
        (window_start.isoformat(), window_end.isoformat(), policy_version, model_version, source,
         metrics.get("total") or 0, metrics.get("confirmed") or 0,
         metrics.get("removed") or 0, metrics.get("dismissed") or 0,
         metrics.get("precision"), metrics.get("coverage"),
         metrics.get("unlink_rate"), metrics.get("drift_index"))
    )


def compute_rollups(cur, window_start, window_end):
    """Rollups for the given window: human-graded and system-decided, separate."""
    start_iso = window_start.isoformat()
    end_iso = window_end.isoformat()

    # --- Human side: graded feedback, tagged by the policy/model in effect ---
    cur.execute(
        """
        SELECT COALESCE(policy_version, 'unknown') AS policy_version,
               COALESCE(model_version, 'unknown') AS model_version,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE feedback_type = 'user_confirmed') AS confirmed,
               COUNT(*) FILTER (WHERE feedback_type = 'user_removed') AS removed,
               COUNT(*) FILTER (WHERE feedback_type = 'user_dismissed') AS dismissed
        FROM clustering_feedback_log
        WHERE created_at >= %s AND created_at < %s
        GROUP BY policy_version, model_version;
        """,
        (start_iso, end_iso)
    )
    human_drift = _drift_by_version(cur, start_iso, end_iso)
    for r in (cur.fetchall() or []):
        pv = extract_val(r, "policy_version", 0)
        mv = extract_val(r, "model_version", 1)
        total = int(extract_val(r, "total", 2) or 0)
        confirmed = int(extract_val(r, "confirmed", 3) or 0)
        removed = int(extract_val(r, "removed", 4) or 0)
        dismissed = int(extract_val(r, "dismissed", 5) or 0)
        denom = confirmed + removed + dismissed
        _upsert_rollup(cur, window_start, window_end, pv, mv, "human", {
            "total": total,
            "confirmed": confirmed,
            "removed": removed,
            "dismissed": dismissed,
            "precision": (confirmed / denom) if denom else None,
            "unlink_rate": (removed / denom) if denom else None,
            "drift_index": human_drift.get((pv, mv)),
        })

    # --- System side: what the pipeline actually decided over the window ---
    cur.execute(
        """
        SELECT COALESCE(policy_version, 'unknown') AS policy_version,
               COALESCE(model_version, 'unknown') AS model_version,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE status = 'assigned') AS assigned
        FROM assignment_decision_log
        WHERE created_at >= %s AND created_at < %s
        GROUP BY policy_version, model_version;
        """,
        (start_iso, end_iso)
    )
    for r in (cur.fetchall() or []):
        pv = extract_val(r, "policy_version", 0)
        mv = extract_val(r, "model_version", 1)
        total = int(extract_val(r, "total", 2) or 0)
        assigned = int(extract_val(r, "assigned", 3) or 0)
        _upsert_rollup(cur, window_start, window_end, pv, mv, "system", {
            "total": total,
            "coverage": (assigned / total) if total else None,
        })


def _drift_by_version(cur, start_iso, end_iso) -> dict[tuple[str, str], float]:
    """Margin separation (mean sim of confirmed - mean sim of removed) per version."""
    cur.execute(
        """
        WITH fb AS (
            SELECT post_id, feedback_type,
                   CASE WHEN feedback_type = 'user_removed' THEN 'bad'
                        WHEN feedback_type = 'user_confirmed' THEN 'good' END AS label,
                   COALESCE(policy_version, 'unknown') AS policy_version,
                   COALESCE(model_version, 'unknown') AS model_version,
                   ROW_NUMBER() OVER (PARTITION BY post_id ORDER BY created_at DESC) AS rn
            FROM clustering_feedback_log
            WHERE created_at >= %s AND created_at < %s
              AND feedback_type IN ('user_confirmed', 'user_removed')
        )
        SELECT f.policy_version, f.model_version, f.label, AVG(d.similarity) AS mean_sim
        FROM fb f
        JOIN assignment_decision_log d ON d.post_id = f.post_id AND d.status = 'assigned'
        WHERE f.rn = 1 AND d.similarity IS NOT NULL
        GROUP BY f.policy_version, f.model_version, f.label;
        """,
        (start_iso, end_iso)
    )
    means: dict[tuple[str, str], float] = {}
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for r in (cur.fetchall() or []):
        key = (extract_val(r, "policy_version", 0), extract_val(r, "model_version", 1))
        label = extract_val(r, "label", 2)
        mean_sim = extract_val(r, "mean_sim", 3)
        if mean_sim is not None:
            by_key.setdefault(key, {})[label] = float(mean_sim)
    for key, m in by_key.items():
        if "good" in m and "bad" in m:
            means[key] = m["good"] - m["bad"]
    return means


def run_window_rollup() -> dict:
    """Roll up the anchored trailing window (default 1h) - called by beat.

    ``end`` is truncated to the hour boundary so repeated runs upsert the same
    row (idempotent) instead of spawning a window that shifts every tick.

    A second, LIVE partial row covers [top of current hour, now) - its window_end
    grows each tick. Fresh zeros in that row mean 'idle with nothing arriving',
    the absence of the row means 'rollup never ran', and stale non-zero values
    mean 'ingestion stalled' - so the metrics panel can distinguish the three
    signals instead of reading a single closed-hour row that lags 60 minutes.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=ROLLUP_WINDOW_HOURS)
    live_start = end  # top of the current (partial) hour
    with get_db_cursor(commit=True) as cur:
        compute_rollups(cur, start, end)
        compute_rollups(cur, live_start, now)
    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "live_start": live_start.isoformat(),
        "live_end": now.isoformat(),
    }


def run_threshold_autotune() -> dict:
    """Replay human feedback against the decision journal and propose (or
    auto-apply) a threshold revision. Propose-only by default."""
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=TUNE_WINDOW_DAYS)
    with get_db_cursor(commit=True) as cur:
        pv, mv = current_versions(cur)
        cur.execute(
            """
            WITH fb AS (
                SELECT post_id,
                       CASE WHEN feedback_type = 'user_confirmed' THEN 'good'
                            WHEN feedback_type = 'user_removed' THEN 'bad' END AS label,
                       policy_version,
                       model_version,
                       ROW_NUMBER() OVER (PARTITION BY post_id ORDER BY created_at DESC) AS rn
                FROM clustering_feedback_log
                WHERE created_at >= %s AND feedback_type IN ('user_confirmed', 'user_removed')
            )
            SELECT d.similarity, f.label
            FROM fb f
            JOIN assignment_decision_log d ON d.post_id = f.post_id AND d.status = 'assigned'
            WHERE f.rn = 1 AND f.label IS NOT NULL AND d.similarity IS NOT NULL
              AND COALESCE(f.policy_version, 'unknown') = %s
              AND COALESCE(f.model_version, 'unknown') = %s;
            """,
            (cutoff.isoformat(), pv, mv)
        )
        labeled = [(float(extract_val(r, "similarity", 0)), extract_val(r, "label", 1))
                   for r in (cur.fetchall() or []) if extract_val(r, "label", 1)]

        cur.execute("SELECT value FROM system_config WHERE key = 'global_similarity_threshold';")
        row = cur.fetchone()
        current = float(extract_val(row, "value", 0) or AUTO_ASSIGN_THRESHOLD)

        proposed = pick_threshold_auto(labeled, floor=AUTO_ASSIGN_THRESHOLD)
        if proposed is None:
            return {"status": "no_change", "samples": len(labeled), "current": current}

        if abs(proposed - current) < 0.002:
            return {"status": "within_epsilon", "samples": len(labeled), "current": current, "proposed": proposed}

        rationale = (
            f"autotune over {TUNE_WINDOW_DAYS}d: {len(labeled)} human decisions; "
            f"{'auto-apply' if AUTO_APPLY_THRESHOLD else 'proposed'} {current:.4f} -> {proposed:.4f}"
        )
        if AUTO_APPLY_THRESHOLD:
            change = apply_policy_change(cur, "global_similarity_threshold", proposed,
                                         "autotune", "system", rationale)
            return {"status": "applied", "samples": len(labeled), **change}
        change = propose_policy_change(cur, "global_similarity_threshold", proposed,
                                       "autotune", "system", rationale)
        return {"status": "proposed", "samples": len(labeled), **change}