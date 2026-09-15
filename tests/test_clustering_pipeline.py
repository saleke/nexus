import pytest
import numpy as np
import torch
from datetime import datetime, timezone

from post_clustering_pipeline.nlp import is_clusterable, cleanse_text
from post_clustering_pipeline.models import EmbeddingEngine
from post_clustering_pipeline.jobs.event_birth import (
    construct_knn_event_graph,
    parse_embedding,
    temporal_dataloader,
)
from post_clustering_pipeline.db import extract_val


def test_nlp_cleansing():
    raw = "Breaking: Check this out https://example.com/live and @nasa updates #space!"
    cleaned = cleanse_text(raw)
    assert "https" not in cleaned
    assert "@nasa" not in cleaned
    assert cleaned.startswith("Breaking Check this out")


def test_nlp_clusterability_valid():
    text = "NASA and SpaceX announced a joint rocket launch mission from Cape Canaveral Florida today."
    assert is_clusterable(text) is True


def test_nlp_clusterability_too_short():
    text = "NASA launched."
    assert is_clusterable(text) is False


def test_nlp_clusterability_no_entities_or_proper_nouns():
    text = "i was walking down the quiet street thinking about eating dinner alone tonight"
    assert is_clusterable(text) is False


def test_embedding_engine_batch_encoding():
    engine = EmbeddingEngine()

    texts = [
        "Major hurricane warning issued along the coast of Florida.",
        "FC Barcelona defeats Real Madrid in exciting match tonight.",
        "Federal Reserve decided to hold benchmark interest rates steady."
    ]

    vectors = engine.encode_batch(texts, batch_size=2)
    assert len(vectors) == 3
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384

    # Verify L2 normalization
    for vec in vectors:
        norm = sum(x * x for x in vec) ** 0.5
        assert 0.99 <= norm <= 1.01

    single_vec = engine.encode(texts[0])
    # Single and batch encoding should be identical within numerical precision
    diff = sum(abs(a - b) for a, b in zip(vectors[0], single_vec))
    assert diff < 1e-4


def test_embedding_engine_freeze_and_unfreeze():
    engine = EmbeddingEngine()

    # Inference mode: everything frozen
    engine.freeze_all()
    for name, param in engine.model.named_parameters():
        assert param.requires_grad is False, f"Parameter {name} should be frozen"

    # Training mode: LoRA unfreezes, base remains frozen
    engine.unfreeze_lora()
    for name, param in engine.model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True, f"LoRA parameter {name} should be unfrozen"
        else:
            assert param.requires_grad is False, f"Base parameter {name} must remain frozen"

    # Re-freeze for inference
    engine.freeze_all()
    for param in engine.model.parameters():
        assert param.requires_grad is False


def test_extract_val_helper():
    assert extract_val({"key": 42}, "key", 0) == 42
    assert extract_val((10, 20, 30), "key", 1) == 20
    assert extract_val([10, 20, 30], "key", 2) == 30
    assert extract_val(None, "key", 0) is None


def test_bounded_incremental_centroid_math():
    max_members = 150
    # Simulate an established event with 200 members
    member_count = 200
    n_eff = min(member_count, max_members)
    assert n_eff == 150

    # Old centroid
    c_old = np.zeros(384)
    c_old[0] = 1.0  # Unit vector along axis 0

    # New embedding
    e_new = np.zeros(384)
    e_new[1] = 1.0  # Unit vector along axis 1

    c_updated = (c_old * n_eff + e_new) / (n_eff + 1)
    c_normalized = c_updated / np.linalg.norm(c_updated)

    # Old axis should dominate with weight 150/151, new axis should have non-zero weight 1/151
    assert c_normalized[0] > 0.99
    assert c_normalized[1] > 0.006
    assert abs(np.linalg.norm(c_normalized) - 1.0) < 1e-6


def test_vectorized_graph_construction():
    t0 = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 12, 12, 30, tzinfo=timezone.utc)

    # 2 identical vectors (cosine sim = 1.0)
    v1 = "[" + ",".join(["0.1"] * 384) + "]"
    v2 = "[" + ",".join(["0.1"] * 384) + "]"

    rows = [
        {"post_id": 1, "embedding": v1, "created_at": t0},
        {"post_id": 2, "embedding": v2, "created_at": t1},
    ]

    g, post_ids = construct_knn_event_graph(rows, k=2, min_similarity_floor=0.5)
    assert len(post_ids) == 2
    assert g.has_node(1)
    assert g.has_node(2)
    assert g.has_edge(1, 2)
    edge_weight = g[1][2]["weight"]
    # Similarity is 1.0, decay for 30 mins with 12h half life should be > 0.97
    assert 0.95 <= edge_weight <= 1.0


def test_temporal_dataloader():
    t0 = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 12, 10, 30, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)

    rows = [
        {"post_id": 1, "created_at": t0},
        {"post_id": 2, "created_at": t1},
        {"post_id": 3, "created_at": t2},
    ]

    slices = temporal_dataloader(rows, slice_hours=1.0)
    assert len(slices) == 2
    assert len(slices[0]) == 2  # t0 and t1 in first slice
    assert len(slices[1]) == 1  # t2 in second slice
