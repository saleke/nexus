import pytest
import numpy as np
from datetime import datetime, timezone

from post_clustering_pipeline.dispatch import CircuitBreaker
from post_clustering_pipeline.queues import distributed_task_lock
from post_clustering_pipeline.jobs.merge_hubs import MERGE_SIMILARITY_THRESHOLD, MAX_TEMPORAL_DIFF_HOURS


def test_circuit_breaker_trip_and_reset():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=60.0)
    assert cb.is_open() is False

    cb.record_failure()
    assert cb.is_open() is False

    cb.record_failure()
    assert cb.is_open() is False

    # 3rd failure trips the circuit
    cb.record_failure()
    assert cb.is_open() is True

    # Success resets circuit
    cb.record_success()
    assert cb.is_open() is False
    assert cb.failure_count == 0


def test_redis_distributed_task_locking():
    # Test non-blocking distributed task lock prevents overlapping executions
    with distributed_task_lock("unit_test_task_lock", timeout=30) as acquired1:
        assert acquired1 is True

        # Second concurrent attempt for the same lock name must be rejected
        with distributed_task_lock("unit_test_task_lock", timeout=30) as acquired2:
            assert acquired2 is False

    # After exiting the context, lock is released and can be acquired again
    with distributed_task_lock("unit_test_task_lock", timeout=30) as acquired3:
        assert acquired3 is True


def test_hub_merging_math_and_guardrails():
    # Centroid 1 & 2 with very high similarity (cosine > 0.95)
    c1 = np.ones(384) / np.sqrt(384)
    c2 = c1.copy()
    c2[0] += 0.01
    c2 = c2 / np.linalg.norm(c2)

    cos_sim = np.dot(c1, c2)
    assert cos_sim >= MERGE_SIMILARITY_THRESHOLD

    # Combined weighted centroid math
    nA, nB = 50, 30
    combined = (c1 * nA + c2 * nB) / (nA + nB)
    combined = combined / np.linalg.norm(combined)

    assert abs(np.linalg.norm(combined) - 1.0) < 1e-6
    # Similarity of combined with either parent should be extremely high
    assert np.dot(combined, c1) > 0.99
    assert np.dot(combined, c2) > 0.99


def test_hub_merging_rejection_below_threshold():
    # Centroid 1 along axis 0, Centroid 2 along axis 1 (orthogonal, sim = 0)
    c1 = np.zeros(384)
    c1[0] = 1.0

    c2 = np.zeros(384)
    c2[1] = 1.0

    cos_sim = np.dot(c1, c2)
    assert cos_sim < MERGE_SIMILARITY_THRESHOLD


def test_hub_merging_temporal_cutoff():
    t_new = datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc)
    t_old = datetime(2026, 9, 11, 20, 0, tzinfo=timezone.utc)

    delta_hours = abs((t_new - t_old).total_seconds()) / 3600.0
    assert delta_hours > MAX_TEMPORAL_DIFF_HOURS  # 18 hours > 12 hours cutoff
