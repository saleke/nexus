"""DB-free unit tests for repost folding primitives.

The fold write paths are SQL and exercised against the live DB by the runtime
suite; these tests pin the two pure pieces: the ingest-time content signature
(nlp.content_signature) and the near-dup resolver (fold._near_dup_fold_map).
"""
import numpy as np
import pytest

from post_clustering_pipeline.fold import _near_dup_fold_map
from post_clustering_pipeline.nlp import content_signature


def _vec(*components):
    arr = np.array(components, dtype=np.float64)
    norm = np.linalg.norm(arr) or 1.0
    return arr / norm


def test_content_signature_is_deterministic_and_normalized():
    a = content_signature("  BREAKING: Apple buys Tesla?! https://example.com/x  ")
    b = content_signature("breaking: apple buys tesla?! https://example.com/y")
    assert a == b
    assert a != content_signature("BREAKING: Apple buys Tesla")
    assert content_signature("hello world") == content_signature("hello   world")


def test_content_signature_never_returns_zero_sentinel():
    for text in ("", None, "a", "0" * 100, "https://x.com"):
        sig = content_signature(text)
        assert isinstance(sig, int)
        assert sig != 0  # 0 is reserved for legacy "never computed" rows


def test_content_signature_differs_across_topics():
    assert content_signature("Apple launches iPhone") != content_signature("Samsung launches Galaxy")


def test_near_dup_fold_map_keeps_oldest_and_folds_near_copies():
    # base vector; a slightly-edited repost must fold to the original
    base = _vec(1.0, 2.0, 3.0, -1.0, 0.5, 2.5, -0.5, 1.5)
    noisy = base + np.array([0.001, -0.001, 0.0, 0.001, 0.0, -0.001, 0.0, 0.0])
    posts = [
        {"id": 11, "embedding": base},     # canonical (oldest)
        {"id": 12, "embedding": noisy},    # near-copy -> folds to 11
        {"id": 13, "embedding": _vec(0, 0, 0, 1, 0, 0, 0, 1)},  # unrelated
    ]
    mapping = _near_dup_fold_map(posts)
    assert mapping == {12: 11}


def test_near_dup_fold_map_exact_duplicate_folds():
    base = _vec(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    posts = [
        {"id": 5, "embedding": base},
        {"id": 9, "embedding": base.copy()},
    ]
    assert _near_dup_fold_map(posts) == {9: 5}


def test_near_dup_fold_map_threshold_respected():
    import math
    base = _vec(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # ~0.9 cosine: clearly a DIFFERENT statement, below the 0.98 fold bar
    below = _vec(math.cos(0.45), math.sin(0.45), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    orthogonal = _vec(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    posts = [
        {"id": 1, "embedding": base},
        {"id": 2, "embedding": orthogonal},
        {"id": 3, "embedding": below},
    ]
    assert _near_dup_fold_map(posts) == {}


def test_near_dup_fold_map_component_single_canonical_no_chain():
    b0 = _vec(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    b1 = b0 + np.array([0.0005] * 8)
    b2 = b0 + np.array([0.0009] * 8)
    posts = [
        {"id": 100, "embedding": b0},
        {"id": 101, "embedding": b1},
        {"id": 102, "embedding": b2},
    ]
    mapping = _near_dup_fold_map(posts)
    # entire component resolves to the OLDEST id; both fold directly to it
    assert mapping == {101: 100, 102: 100}


def test_near_dup_fold_map_singletons_and_none_embeddings():
    assert _near_dup_fold_map([{"id": 1, "embedding": _vec(1, 1, 1, 1, 1, 1, 1, 1)}]) == {}
    assert _near_dup_fold_map([{"id": 1, "embedding": None}, {"id": 2, "embedding": None}]) == {}


def test_near_dup_fold_map_never_creates_chains():
    # A single pass must map every folded post directly to a terminal
    # canonical (a post that is not itself folded). Chains are a cross-pass
    # artifact handled by _reparent_repost_children, never by the map.
    base = _vec(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    eps = np.array([0.0004] * 8)
    posts = [
        {"id": 10, "embedding": base},
        {"id": 20, "embedding": base + eps},
        {"id": 30, "embedding": base + 2 * eps},
    ]
    mapping = _near_dup_fold_map(posts)
    canonicals = set(mapping.values())
    assert not (canonicals & set(mapping.keys()))  # no canonical is itself folded