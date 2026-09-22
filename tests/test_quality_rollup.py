"""Regression tests for the intelligence-layer rollup (quality.compute_rollups).

The historical bug: compute_rollups executed the human-feedback aggregate,
then called _drift_by_version (a second execute on the same cursor) BEFORE
fetching the aggregate. psycopg2 discards the previous result set as soon as a
new statement executes, so the human rollup was never written when drift had no
samples (silently), and crashed with KeyError when drift did have samples
(drift rows are (pv, mv, label, mean_sim), not aggregate rows).

A scripted fake cursor returns each result set immediately after its execute(),
exactly like psycopg2, and the REAL _drift_by_version runs so its statement
consumes the aggregate result the same way production does. Under the old
ordering the first fetchall() would read the drift rows instead.
"""
from datetime import datetime, timezone

from post_clustering_pipeline import quality

# Unique substrings identifying each statement quality.compute_rollups issues
# (the drift statement is NOT a substring of the aggregates and vice versa).
_HUMAN_AGG = "FILTER (WHERE feedback_type = 'user_confirmed')"
_DRIFT = "ROW_NUMBER() OVER (PARTITION BY post_id ORDER BY created_at DESC)"
_SYSTEM_AGG = "FILTER (WHERE status = 'assigned') AS assigned"


class ScriptedCursor:
    """fetchall() returns the result set from the most recent execute().

    Mirrors psycopg2: a new execute() discards whatever was fetched-or-not of
    the previous statement's result set.
    """

    def __init__(self):
        self._result = []
        self._queues = {}  # substr -> rows
        self.rollup_upserts = []
        self.executed = []

    def script(self, substr: str, rows):
        self._queues[substr] = rows

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self._result = []
        for substr, rows in self._queues.items():
            if substr in sql:
                self._result = rows
                return

    def fetchall(self):
        return list(self._result)

    def fetchone(self):
        return self._result[0] if self._result else None


def _sink_upsert(cur, window_start, window_end, policy_version, model_version,
                 source, metrics):
    cur.rollup_upserts.append({
        "window_start": window_start, "window_end": window_end,
        "policy_version": policy_version, "model_version": model_version,
        "source": source, "metrics": dict(metrics),
    })


def _run(monkeypatch, human_rows, drift_rows, system_rows) -> ScriptedCursor:
    cur = ScriptedCursor()
    cur.script(_HUMAN_AGG, human_rows)
    cur.script(_DRIFT, drift_rows)
    cur.script(_SYSTEM_AGG, system_rows)
    monkeypatch.setattr(quality, "_upsert_rollup", _sink_upsert)
    start = datetime(2026, 9, 22, 11, 0, tzinfo=timezone.utc)
    end = datetime(2026, 9, 22, 11, 4, tzinfo=timezone.utc)
    quality.compute_rollups(cur, start, end)
    return cur


def test_human_rollup_written_when_drift_has_no_samples(monkeypatch):
    """Only one label class exists -> drift empty; human row must still land."""
    human = [{
        "policy_version": "policy-v1", "model_version": "embedding-v1",
        "total": 3, "confirmed": 2, "removed": 1, "dismissed": 0,
    }]
    cur = _run(monkeypatch, human, drift_rows=[], system_rows=[])

    writes = [u for u in cur.rollup_upserts if u["source"] == "human"]
    assert len(writes) == 1, f"human rollup missing, got {cur.rollup_upserts}"
    m = writes[0]["metrics"]
    assert (m["total"], m["confirmed"], m["removed"], m["dismissed"]) == (3, 2, 1, 0)
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["unlink_rate"] - 1 / 3) < 1e-9
    assert m["drift_index"] is None


def test_human_rollup_does_not_read_drift_rows_as_aggregates(monkeypatch):
    """When drift IS non-empty, its columns must not be misparsed as counts."""
    human = [{
        "policy_version": "policy-v1", "model_version": "embedding-v1",
        "total": 4, "confirmed": 3, "removed": 1, "dismissed": 0,
    }]
    drift = [
        {"policy_version": "policy-v1", "model_version": "embedding-v1",
         "label": "good", "mean_sim": 0.99},
        {"policy_version": "policy-v1", "model_version": "embedding-v1",
         "label": "bad", "mean_sim": 0.80},
    ]
    cur = _run(monkeypatch, human, drift, system_rows=[])

    writes = [u for u in cur.rollup_upserts if u["source"] == "human"]
    assert len(writes) == 1, f"human rollup missing, got {cur.rollup_upserts}"
    m = writes[0]["metrics"]
    assert (m["total"], m["confirmed"], m["removed"], m["dismissed"]) == (4, 3, 1, 0)
    assert abs(m["drift_index"] - 0.19) < 1e-9  # good - bad margin


def test_system_rollup_coverage_row(monkeypatch):
    """System-side coverage row is computed from assignment_decision_log."""
    system = [{
        "policy_version": "policy-v1", "model_version": "embedding-v1",
        "total": 100, "assigned": 40,
    }]
    cur = _run(monkeypatch, human_rows=[], drift_rows=[], system_rows=system)

    writes = [u for u in cur.rollup_upserts if u["source"] == "system"]
    assert len(writes) == 1
    assert abs(writes[0]["metrics"]["coverage"] - 0.4) < 1e-9
