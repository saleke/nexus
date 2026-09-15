"""Unit tests for the pipeline safety layer added in the hardening pass.

Covers the pieces that don't need a live database: the vector codec seam,
chunk iteration, the threshold floor, and the idempotent status predicate.
"""
import pytest
import numpy as np
import torch

from post_clustering_pipeline.embed_io import (
    vector_to_array_literal,
    parse_vector_literal,
    vector_to_binary,
    binary_to_vector,
)
from post_clustering_pipeline.chunks import iter_chunks
from post_clustering_pipeline.db import extract_val
from post_clustering_pipeline.threshold import effective_threshold
from post_clustering_pipeline.assignment import decide_assignment, _assign_payload
from post_clustering_pipeline.config import CANDIDATE_THRESHOLD


def test_vector_literal_roundtrip():
    rng = np.random.default_rng(42)
    vector = rng.standard_normal(384).astype(np.float32)
    literal = vector_to_array_literal(vector)
    parsed = parse_vector_literal(literal)
    assert len(parsed) == 384
    # float32 values stored in the vector type must round-trip losslessly
    np.testing.assert_array_equal(parsed.astype(np.float32), vector)


def test_vector_literal_roundtrip_python_floats():
    # model.encode_batch returns plain Python floats (float64 reps of float32)
    vector = [0.1] * 384
    literal = vector_to_array_literal(vector)
    assert literal.startswith("[") and literal.endswith("]")
    parsed = parse_vector_literal(literal)
    np.testing.assert_allclose(parsed, vector, atol=1e-9)


def test_vector_literal_is_locale_independent():
    # The codec must never emit a comma decimal separator; only the delimiter.
    literal = vector_to_array_literal([0.123456789, -0.000123456])
    assert literal == "[0.123456789,-0.000123456]"
    # every comma separates elements, never inside a number
    for token in literal.strip("[]").split(","):
        assert token.count(".") <= 1
        assert token.replace(".", "").replace("-", "").isdigit()


def test_parse_vector_literal_handles_none_and_empty():
    assert parse_vector_literal(None).size == 0
    assert parse_vector_literal("[]").size == 0
    assert parse_vector_literal("").size == 0


def test_pgvector_binary_roundtrip():
    rng = np.random.default_rng(7)
    vec = rng.standard_normal(384)
    data = vector_to_binary(vec)
    assert len(data) == 4 + 384 * 4
    back = binary_to_vector(data)
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, vec.astype(np.float32))


def test_pgvector_binary_matches_library_codec():
    from pgvector import Vector

    vec = [0.1, -0.25, 0.5, 1.0]
    assert vector_to_binary(vec) == Vector(vec).to_binary()
    decoded = binary_to_vector(Vector(vec).to_binary())
    np.testing.assert_allclose(decoded, Vector(vec).to_list(), atol=1e-6)


def test_pgvector_binary_rejects_bad_payload():
    with pytest.raises(ValueError):
        binary_to_vector(b"\x00\x01\x01\x01")
    with pytest.raises(ValueError):
        binary_to_vector(b"\x00\x01\x00\x00" + b"\x00" * 0)  # declares 1 dim, no payload


def test_iter_chunks_covers_every_element():
    posts = list(range(100))
    sizes = []
    collected = []
    for chunk, start, end in iter_chunks(posts, 30):
        sizes.append(len(chunk))
        collected.extend(chunk)
        assert end - start == len(chunk) == (30 if end < 100 else 10)
    assert sizes == [30, 30, 30, 10]
    assert collected == posts


def test_iter_chunks_parallel_lists_stay_aligned():
    left = [1, 2, 3, 4, 5]
    right = ["a", "b", "c", "d", "e"]
    for (lc, _, le), (rc, _, re) in zip(
        iter_chunks(left, 2), iter_chunks(right, 2)
    ):
        assert lc != rc
        assert le == re
    assert list(iter_chunks([], 16)) == []


def test_effective_threshold_never_below_floor():
    assert effective_threshold(0.80, 0.88) == 0.88
    assert effective_threshold(0.91, 0.88) == 0.91
    assert effective_threshold(None, 0.88) == 0.88
    assert effective_threshold(0.99, 0.88) == 0.99


def test_extract_val_covers_cursor_shapes():
    assert extract_val({"id": 7}, "id", 0) == 7
    assert extract_val((7, "x"), "id", 0) == 7
    assert extract_val([7, "x"], "id", 1) == "x"
    assert extract_val(None, "id", 0) is None


def test_embedding_engine_output_is_codec_compatible():
    # Guard the seam: whatever the engine emits, the codec round-trips it.
    from post_clustering_pipeline.models import EmbeddingEngine

    engine = EmbeddingEngine()
    vec = engine.encode("Major hurricane warning issued along the coast of Florida.")
    literal = vector_to_array_literal(vec)
    parsed = parse_vector_literal(literal)
    np.testing.assert_allclose(parsed, vec, atol=1e-6)
    assert parsed.shape == (384,)
    norm = torch.nn.functional.normalize(torch.tensor(parsed), p=2, dim=0)
    np.testing.assert_allclose(norm.numpy(), parsed / np.linalg.norm(parsed), atol=1e-6)


def test_decide_assignment_over_threshold_with_margin():
    # 0.93 beats 0.87 runner-up by 0.06 >= SIMILARITY_MARGIN (0.05)
    is_assigned, status, conf = decide_assignment(0.93, 0.87, 0.88)
    assert is_assigned is True
    assert status == "assigned"
    assert conf == 0.93


def test_decide_assignment_over_threshold_no_runner():
    # No competitor means the margin check is skipped
    is_assigned, status, _ = decide_assignment(0.90, None, 0.88)
    assert is_assigned is True


def test_decide_assignment_ambiguous_goes_candidate():
    # Above threshold but within the margin of the runner-up -> NOT assigned
    is_assigned, status, conf = decide_assignment(0.90, 0.88, 0.88)
    assert is_assigned is False
    assert status in {"candidate", "unassigned"}
    assert conf == 0.90


def test_decide_assignment_below_threshold_candidate_vs_unassigned():
    # Values straddling the configurable candidate floor, not magic numbers.
    thr = 0.88
    hi = CANDIDATE_THRESHOLD + 0.05
    lo = CANDIDATE_THRESHOLD - 0.05
    _, status, _ = decide_assignment(hi, None, thr)
    assert status == "candidate"          # >= CANDIDATE_THRESHOLD
    _, status, _ = decide_assignment(lo, None, thr)
    assert status == "unassigned"         # < CANDIDATE_THRESHOLD


def test_decide_assignment_no_match():
    is_assigned, status, conf = decide_assignment(None, None, 0.88)
    assert is_assigned is False
    assert status == "unassigned"
    assert conf is None


def test_assign_payload_shapes():
    import json as _json
    assigned = _assign_payload("post.assigned", 1, 42, 0.91)
    assert _json.loads(assigned) == {
        "post_id": 1, "event_id": 42, "status": "assigned", "confidence": 0.91
    }
    candidate = _assign_payload("post.candidate", 2, None, 0.80)
    assert _json.loads(candidate)["status"] == "candidate"
    unassigned = _assign_payload("post.unassigned", 3, None, None)
    assert _json.loads(unassigned)["status"] == "unassigned"
    assert _json.loads(unassigned)["confidence"] is None