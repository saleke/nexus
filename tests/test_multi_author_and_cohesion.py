import pytest
import numpy as np
from post_clustering_pipeline.config import (
    BIRTH_MIN_AUTHORS,
    BIRTH_COHESION_FLOOR,
    ANCHOR_WEIGHT,
)


def test_cohesion_math_pass():
    # 4 embeddings with high pairwise similarity (> 0.90)
    base = np.random.RandomState(42).randn(384)
    base = base / np.linalg.norm(base)

    cluster_embeddings = []
    for _ in range(4):
        perturbed = base + 0.05 * np.random.RandomState(42).randn(384)
        cluster_embeddings.append(perturbed / np.linalg.norm(perturbed))

    embs_arr = np.array(cluster_embeddings)
    norms = np.linalg.norm(embs_arr, axis=1, keepdims=True)
    embs_norm = embs_arr / norms
    sim_matrix = np.dot(embs_norm, embs_norm.T)
    cohesion = float(np.mean(sim_matrix))

    assert cohesion >= BIRTH_COHESION_FLOOR
    assert cohesion > 0.95


def test_cohesion_math_fail_on_disjoint_topics():
    # 4 embeddings along orthogonal axes (unrelated grab-bag of topics)
    cluster_embeddings = []
    for i in range(4):
        vec = np.zeros(384)
        vec[i] = 1.0
        cluster_embeddings.append(vec)

    embs_arr = np.array(cluster_embeddings)
    sim_matrix = np.dot(embs_arr, embs_arr.T)
    cohesion = float(np.mean(sim_matrix))

    # Diagonal is 1.0, off-diagonal is 0.0. Average is (4*1.0 + 12*0.0) / 16 = 0.25
    assert cohesion < BIRTH_COHESION_FLOOR
    assert abs(cohesion - 0.25) < 1e-4


def test_anchor_weighted_centroid_math():
    # Anchor vector along axis 0
    anchor_vec = np.zeros(384)
    anchor_vec[0] = 1.0

    # Rolling centroid along axis 1
    c_rolling = np.zeros(384)
    c_rolling[1] = 1.0

    c_effective = ANCHOR_WEIGHT * anchor_vec + (1.0 - ANCHOR_WEIGHT) * c_rolling
    c_norm = c_effective / np.linalg.norm(c_effective)

    # Axis 0 (anchor) must have strong weight = 0.35 / norm
    expected_norm = np.sqrt(ANCHOR_WEIGHT**2 + (1.0 - ANCHOR_WEIGHT)**2)
    assert abs(c_norm[0] - (ANCHOR_WEIGHT / expected_norm)) < 1e-5
    assert abs(c_norm[1] - ((1.0 - ANCHOR_WEIGHT) / expected_norm)) < 1e-5

    # Dot product with anchor is guaranteed to be substantial
    assert np.dot(c_norm, anchor_vec) > 0.45
