"""Equivalence tests for the vectorized nearest-centroid lookup in event_birth.

The old path computed, per post, a Python loop over every active centroid with
``sim = dot(emb, centroid) / (|emb| * |centroid| + 1e-9)`` and a strict ``>``
comparison (so the first hub in iteration order wins ties). The refactor keeps
the same formula but batches it as one matmul + ``argmax``. These tests pin the
decision (hub id and similarity) against a reference implementation of the old
loop, including tie-breaking and the empty-centroid sentinel.
"""
import numpy as np

from post_clustering_pipeline.jobs.event_birth import best_centroid


def _reference(emb, centroid_ids, centroids):
    """The original per-hub loop, verbatim in behaviour."""
    best_hub = None
    best_sim = -1.0
    for eid in centroid_ids:
        centroid = centroids[eid]
        dot = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-9)
        if dot > best_sim:
            best_sim = float(dot)
            best_hub = eid
    return best_hub, best_sim


def _matrix(centroid_ids, centroids):
    mat = np.array([centroids[e] for e in centroid_ids])
    return mat, np.linalg.norm(mat, axis=1)


def test_empty_centroids_returns_sentinel():
    emb = np.array([1.0, 0.0, 0.0])
    hub, sim = best_centroid(emb, [], np.zeros((0, 0)), np.zeros(0))
    assert hub is None
    assert sim == -1.0


def test_matches_reference_over_random_vectors():
    rng = np.random.default_rng(1234)
    dim = 16
    for _ in range(200):
        n_hubs = int(rng.integers(1, 12))
        n_posts = int(rng.integers(1, 25))
        centroid_ids = list(range(1, n_hubs + 1))
        centroids = {eid: rng.normal(size=dim) for eid in centroid_ids}
        mat, norms = _matrix(centroid_ids, centroids)

        for _ in range(n_posts):
            emb = rng.normal(size=dim)
            exp_hub, exp_sim = _reference(emb, centroid_ids, centroids)
            got_hub, got_sim = best_centroid(emb, centroid_ids, mat, norms)
            assert got_hub == exp_hub
            assert abs(got_sim - exp_sim) < 1e-9


def test_first_hub_wins_exact_ties():
    # Two identical centroids: the first in iteration order must win, exactly
    # like the strict ``>`` comparison of the old loop.
    emb = np.array([1.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    ids = [7, 3, 9]
    centroids = {7: c, 3: c.copy(), 9: np.array([0.0, 1.0, 0.0])}
    mat, norms = _matrix(ids, centroids)

    exp_hub, exp_sim = _reference(emb, ids, centroids)
    got_hub, got_sim = best_centroid(emb, ids, mat, norms)
    assert (exp_hub, got_hub) == (7, 7)
    assert abs(got_sim - exp_sim) < 1e-12


def test_updated_row_tracks_the_new_centroid():
    # Mirrors the in-loop mutation: after a hub's row is replaced in the
    # matrix, the next post must match against the updated vector, not a
    # stale one.
    ids = [1, 2]
    c1 = np.array([1.0, 0.0, 0.0])
    c2 = np.array([0.0, 1.0, 0.0])
    mat = np.array([c1, c2])
    norms = np.linalg.norm(mat, axis=1)

    emb = np.array([0.0, 1.0, 0.0])
    assert best_centroid(emb, ids, mat, norms)[0] == 2

    # Rotate hub 1's row to point at the query, then recompute the cached norm.
    mat[0] = np.array([0.0, 1.0, 0.0])
    norms[0] = np.linalg.norm(mat[0])
    assert best_centroid(emb, ids, mat, norms)[0] == 1
