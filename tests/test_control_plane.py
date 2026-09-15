"""Unit tests for the control-plane intelligence layer (no live database).

Covers the pure estimators and guards: threshold curve / autotune picks,
policy knob validation and persistence rounding, webhook signature math, and
the assignment reason-code / margin helpers that feed the decisions inspector.
"""
import hmac
import hashlib
import json

from post_clustering_pipeline.quality import threshold_curve, pick_threshold_auto
from post_clustering_pipeline.policy import validate_knob, _persist_precision, KNOB_RANGES
from post_clustering_pipeline.dispatch import signature_for
from post_clustering_pipeline.assignment import _reason_code, _margin_budget


# --- threshold curve ------------------------------------------------------


def test_threshold_curve_coverage_monotone():
    labeled = [
        (0.95, "good"), (0.91, "good"), (0.88, "good"),
        (0.90, "bad"), (0.84, "bad"),
    ]
    curve = threshold_curve(labeled)
    coverages = [c["coverage"] for c in curve if c["coverage"] is not None]
    # Coverage (goods kept / all goods) never rises as the threshold rises.
    assert all(c2 <= c1 for c1, c2 in zip(coverages, coverages[1:]))
    # The curve only visits distinct observed similarities.
    assert len(curve) == len({s for s, _ in labeled})


def test_threshold_curve_precision_not_assumed_monotone():
    # Documented: precision can DIP (0.88 -> 0.90 removes a good while the
    # bad at 0.90 survives). The estimator must not assume monotonicity.
    labeled = [
        (0.95, "good"), (0.91, "good"), (0.88, "good"),
        (0.90, "bad"), (0.84, "bad"),
    ]
    by_t = {p["threshold"]: p for p in threshold_curve(labeled)}
    assert by_t[0.88]["precision"] == 0.75
    assert by_t[0.90]["precision"] == 2 / 3  # drops, despite the rise in t
    assert by_t[0.91]["precision"] == 1.0


def test_threshold_curve_bins_by_distinct_similarity():
    labeled = [(0.93, "good"), (0.91, "bad"), (0.93, "bad"), (0.91, "good")]
    curve = threshold_curve(labeled)
    assert len(curve) == 2
    by_t = {p["threshold"]: p for p in curve}
    assert by_t[0.91]["true_positives"] == 2
    assert by_t[0.93]["true_positives"] == 1
    assert by_t[0.93]["false_positives"] == 1


def test_pick_threshold_auto_precision_first():
    # 0.94 admits a false positive (bad 0.94); 0.96 cleans the set while
    # keeping every good. Precision-first picks 0.96 over the cleaner-but-lower-
    # coverage 0.97 (both precision 1.0; 0.96 keeps more coverage).
    labeled = [(0.98, "good"), (0.97, "good"), (0.96, "good"),
               (0.94, "bad"), (0.93, "bad")]
    chosen = pick_threshold_auto(labeled, floor=0.80, target_precision=1.0,
                                 min_coverage=0.4, min_samples=1)
    assert chosen == 0.96


def test_pick_threshold_auto_prefers_higher_precision_band():
    # The feasible band is NOT contiguous: 0.90 and 0.91 clear the 0.75 bar,
    # then a bad at 0.91 kills 0.92-0.96, and 0.97 alone gives precision 1.0.
    # A naive first-feasible scan would pick 0.90; precision-first picks 0.97.
    labeled = [(0.98, "good"), (0.97, "good"), (0.90, "good"), (0.89, "good"),
               (0.91, "bad"), (0.80, "bad")]
    chosen = pick_threshold_auto(labeled, floor=0.85, target_precision=0.75,
                                 min_coverage=0.5, min_samples=1)
    assert chosen == 0.97


def test_pick_threshold_auto_insufficient_samples_silent():
    labeled = [(0.95, "good"), (0.80, "bad")]
    assert pick_threshold_auto(labeled, min_samples=10) is None


def test_pick_threshold_auto_floor_respected():
    # Case 1: bads sit ABOVE the goods — no threshold >= 0.97 is clean, so the
    # floor is untouchable and the tuner stays silent rather than drift.
    labeled = [(0.95, "good"), (0.95, "good"), (0.99, "bad"), (0.99, "bad")]
    assert pick_threshold_auto(labeled, floor=0.97, target_precision=1.0,
                               min_coverage=0.5, min_samples=1) is None
    # Case 2: goods and bads separate cleanly at 0.95; the floor admits it.
    labeled = [(0.95, "good"), (0.95, "good"), (0.93, "bad"), (0.93, "bad")]
    assert pick_threshold_auto(labeled, floor=0.90, target_precision=1.0,
                               min_coverage=0.5, min_samples=1) == 0.95


# --- policy knob validation / persistence precision -----------------------


def test_validate_knob_bounds():
    assert validate_knob("global_similarity_threshold", 0.900)[0]
    assert validate_knob("global_similarity_threshold", 0.500)[0]  # inclusive low
    assert not validate_knob("global_similarity_threshold", 0.490)[0]
    assert not validate_knob("global_similarity_threshold", 1.000)[0]
    assert not validate_knob("not_a_knob", 0.9)[0]


def test_validate_knob_persistence_precision():
    # NUMERIC(4,3) storage: a 4-decimal value must round to 3, never overflow.
    ok, _ = validate_knob("global_similarity_threshold", 0.9155)
    assert ok
    assert _persist_precision("global_similarity_threshold") == 3
    assert KNOB_RANGES["global_similarity_threshold"][2] == 3


# --- webhook signature ----------------------------------------------------


def test_webhook_signature_known_vector():
    body = json.dumps({"a": 1}, separators=(",", ":")).encode("utf-8")
    expected = hmac.new(b"sekret", body, hashlib.sha256).hexdigest()
    assert signature_for(body, "sekret") == expected


def test_webhook_signature_deterministic_and_byte_exact():
    body = b'{"x": true}'
    assert signature_for(body, "s") == signature_for(body, "s")
    # A changed byte in the exact delivered body yields a different signature.
    assert signature_for(body, "s") != signature_for(b'{"x":tru}', "s")


# --- assignment reason / margin helpers -----------------------------------


def test_reason_codes():
    t = 0.88
    assert _reason_code(0.92, 0.90, t, "assigned") == "confident_assign"
    # At/above threshold but the runner-up is within margin -> margin_fail.
    assert _reason_code(0.90, 0.88, t, "candidate") == "margin_fail"
    assert _reason_code(0.80, None, t, "candidate") == "below_threshold"
    assert _reason_code(0.60, None, t, "unassigned") == "below_candidate_floor"


def test_margin_budget():
    assert _margin_budget(0.90, 0.88, 0.88) == 0.02
    assert _margin_budget(0.90, None, 0.88) == 0.02
    assert _margin_budget(None, None, 0.88) is None