"""Pure centroid arithmetic shared by the streaming assigner and event birth.

Single source of truth for the bounded incremental centroid, the anchor
blend, and L2 normalization - callers must not re-derive this math.
"""
from __future__ import annotations

import numpy as np

_EPSILON = 1e-9


def normalize(vector) -> np.ndarray:
    """L2-normalize a vector, guarding against the zero vector."""
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    return array / (norm + _EPSILON)


def bounded_rolling_centroid(old_centroid, old_member_count, new_vectors, max_members) -> np.ndarray:
    """Rolling average with a bounded effective ``n`` (anti-fossilization).

    ``new_vectors`` may be a single embedding or a pre-summed list; the count
    ``k`` is the number of new members. If ``old_centroid`` is None (fresh
    hub), the result is simply the mean of the new vectors.
    """
    new_vectors = [np.asarray(vec, dtype=np.float64) for vec in new_vectors]
    if not new_vectors:
        raise ValueError("bounded_rolling_centroid requires at least one new vector")

    k = len(new_vectors)
    sum_new = np.sum(new_vectors, axis=0)

    if old_centroid is None:
        return sum_new / k

    n_eff = min(int(old_member_count or 1), int(max_members))
    old = np.asarray(old_centroid, dtype=np.float64)
    return (old * n_eff + sum_new) / (n_eff + k)


def anchor_blended_centroid(rolling_centroid, anchor_vector, anchor_weight) -> np.ndarray:
    """Blend a rolling centroid toward the catalyst post's direction.

    ``anchor_vector`` None disables the blend (returns the rolling centroid).
    """
    if anchor_vector is None:
        return rolling_centroid
    anchor = np.asarray(anchor_vector, dtype=np.float64)
    return anchor_weight * anchor + (1.0 - anchor_weight) * np.asarray(rolling_centroid, dtype=np.float64)