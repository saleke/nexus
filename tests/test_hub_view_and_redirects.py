import pytest
from unittest.mock import MagicMock
from post_clustering_pipeline.api import resolve_canonical_hub


def test_resolve_canonical_hub_active():
    cur = MagicMock()
    # Hub 1 is active, not merged
    cur.fetchone.return_value = {"id": 1, "is_active": True, "merged_into_id": None}

    canonical_id, redirected = resolve_canonical_hub(cur, 1)
    assert canonical_id == 1
    assert redirected is False


def test_resolve_canonical_hub_single_redirect():
    cur = MagicMock()
    # Hub 2 merged into Hub 1; Hub 1 is active
    cur.fetchone.side_effect = [
        {"id": 2, "is_active": False, "merged_into_id": 1},
        {"id": 1, "is_active": True, "merged_into_id": None},
    ]

    canonical_id, redirected = resolve_canonical_hub(cur, 2)
    assert canonical_id == 1
    assert redirected is True


def test_resolve_canonical_hub_chained_redirect():
    cur = MagicMock()
    # Hub 3 merged into Hub 2, which merged into Hub 1 (active)
    cur.fetchone.side_effect = [
        {"id": 3, "is_active": False, "merged_into_id": 2},
        {"id": 2, "is_active": False, "merged_into_id": 1},
        {"id": 1, "is_active": True, "merged_into_id": None},
    ]

    canonical_id, redirected = resolve_canonical_hub(cur, 3)
    assert canonical_id == 1
    assert redirected is True


def test_resolve_canonical_hub_not_found():
    cur = MagicMock()
    cur.fetchone.return_value = None

    canonical_id, redirected = resolve_canonical_hub(cur, 999)
    assert canonical_id == 999
    assert redirected is False
