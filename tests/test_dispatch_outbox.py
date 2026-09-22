"""DB-free tests for the 3-phase webhook dispatcher (P5).

Verifies the core fix: the HTTP POST never runs while a database transaction
(and its ``FOR UPDATE`` locks) is open, outcomes are applied in a separate
short transaction, and rows skipped after the circuit breaker opens are
released and have their claim attempt undone.
"""
import post_clustering_pipeline.dispatch as dispatch
from post_clustering_pipeline.config import OUTBOX_MAX_ATTEMPTS
from types import SimpleNamespace


def _sqls(state):
    return [(s.decode("utf-8") if isinstance(s, bytes) else s) for s, _ in state["executed"]]


class FakeCursor:
    def __init__(self, state, rows):
        self.state = state
        self._rows = rows
        self.connection = SimpleNamespace(encoding="UTF8")

    def execute(self, sql, params=None):
        self.state["executed"].append((sql, params))

    def fetchall(self):
        return self._rows

    def mogrify(self, template, args):
        return (template % args).encode("utf-8")


class FakeCtx:
    def __init__(self, state, rows):
        self.state = state
        self.rows = rows

    def __enter__(self):
        self.state["open"] += 1
        return FakeCursor(self.state, self.rows)

    def __exit__(self, *exc):
        self.state["open"] -= 1
        return False


def make_db(state, first_rows):
    calls = {"n": 0}

    def _factory(commit=True):
        calls["n"] += 1
        state["db_calls"] += 1
        return FakeCtx(state, first_rows if calls["n"] == 1 else [])

    return _factory


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text


class FakeSession:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = []

    def post(self, url, data, headers, timeout):
        self.calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        st = self.statuses.pop(0) if self.statuses else 200
        return FakeResponse(st, "boom" if st >= 400 else "")


def _rows(n):
    return [
        {
            "id": i,
            "event_type": "post.assigned",
            "post_id": i,
            "event_id": 1,
            "payload": {"post_id": i},
            "attempts": 1,
        }
        for i in range(1, n + 1)
    ]


def _setup(monkeypatch, state_rows, statuses, threshold=5):
    monkeypatch.setattr(dispatch, "EVENT_DELIVERY_MODE", "webhook")
    monkeypatch.setattr(dispatch, "EVENT_WEBHOOK_SIGNING_SECRET", "")
    dispatch.circuit_breaker = dispatch.CircuitBreaker(failure_threshold=threshold)
    state = {"open": 0, "executed": [], "db_calls": 0}
    monkeypatch.setattr(dispatch, "get_db_cursor", make_db(state, state_rows))
    session = FakeSession(statuses)
    seen = {"open_during_post": []}
    real_post = session.post

    def tracked_post(*a, **k):
        seen["open_during_post"].append(state["open"])
        return real_post(*a, **k)

    session.post = tracked_post
    return state, session, seen


def test_http_runs_with_no_transaction_open_and_applies_outcomes(monkeypatch):
    state, session, seen = _setup(monkeypatch, _rows(2), [200, 200])

    delivered = dispatch.dispatch_pending_webhooks(
        limit=10, webhook_url="https://example.com/hook", session=session
    )

    assert delivered == 2
    assert seen["open_during_post"] == [0, 0], "HTTP ran while a DB transaction was open"
    assert state["open"] == 0
    assert state["db_calls"] == 2, "expected exactly a claim tx and an apply tx"
    assert any("delivered" in sql for sql in _sqls(state))
    # Claim statement must use SKIP LOCKED and be its own transaction.
    assert any("SKIP LOCKED" in sql for sql in _sqls(state))


def test_exhausted_attempts_marks_failed(monkeypatch):
    rows = _rows(1)
    rows[0]["attempts"] = OUTBOX_MAX_ATTEMPTS
    state, session, _ = _setup(monkeypatch, rows, [500])

    delivered = dispatch.dispatch_pending_webhooks(
        limit=10, webhook_url="https://example.com/hook", session=session
    )

    assert delivered == 0
    assert any("failed" in sql for sql in _sqls(state))


def test_breaker_open_releases_unprocessed_and_undoes_claim(monkeypatch):
    state, session, _ = _setup(monkeypatch, _rows(3), [500, 200, 200], threshold=1)

    delivered = dispatch.dispatch_pending_webhooks(
        limit=10, webhook_url="https://example.com/hook", session=session
    )

    assert delivered == 0
    assert len(session.calls) == 1, "breaker should stop after the first failure"
    assert any("GREATEST(attempts - 1, 0)" in sql for sql in _sqls(state))
    # The release must target a literal array of the two skipped ids (2 and 3).
    release = [
        p for s, p in state["executed"]
        if "GREATEST(attempts - 1, 0)" in (s.decode("utf-8") if isinstance(s, bytes) else s)
    ]
    assert release and release[0][0] == [2, 3], release


def test_signing_header_present_when_secret_configured(monkeypatch):
    state, session, _ = _setup(monkeypatch, _rows(1), [200])
    monkeypatch.setattr(dispatch, "EVENT_WEBHOOK_SIGNING_SECRET", "shh")

    dispatch.dispatch_pending_webhooks(
        limit=10, webhook_url="https://example.com/hook", session=session
    )

    sig = session.calls[0]["headers"].get("X-Nexus-Signature")
    assert sig and sig.startswith("sha256=")
    # Signature must be over the exact body bytes that were sent.
    assert dispatch.signature_for(session.calls[0]["data"], "shh") == sig.split("=", 1)[1]
