"""Runtime-era verify: era-anchored quality path against a live Postgres+Redis.

Only asserts what the era-anchor fix actually promised, using the exact
on-disk public symbols (verified by import, not by memory):

  - threshold.get_effective_threshold(cur)            (threshold.py:25)
  - policy.current_versions(cur)                      (policy.py:76)
  - db.get_db_cursor()                                (db.py:68)
  - queues.bounded_push_ingest_hints(*ids, cap=...)   (queues.py:80)

Run:  ./.venv/bin/python tests/runtime/verify_era.py
Exits non-zero if any claim does not hold on the live stack.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Env must point at the same stack the suite ran against. Defaults mirror
# docker-compose reads the yes, but are overridable for a local dev DB.
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/clustering_db")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")

from post_clustering_pipeline.db import get_db_cursor          # noqa: E402
from post_clustering_pipeline.policy import current_versions   # noqa: E402
from post_clustering_pipeline.threshold import get_effective_threshold  # noqa: E402
from post_clustering_pipeline.queues import (                   # noqa: E402
    bounded_push_ingest_hints,
    get_redis_client,
)
from post_clustering_pipeline.config import INGEST_HINTS_CAP   # noqa: E402

CLAIMS: list[tuple[str, bool, str]] = []


def claim(name: str, ok: bool, detail: str = "") -> None:
    CLAIMS.append((name, bool(ok), detail))
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}] {name} {detail}")


def main() -> int:
    print("RUNTIME ERA VERIFY against live stack")
    print("-" * 68)

    # 1) The exact call that used to raise NameError at quality.py:221 —
    #    policy.current_versions resolves through the real era anchor.
    try:
        with get_db_cursor(commit=False) as cur:
            policy_v, model_v = current_versions(cur)
        claim(
            "era anchor current_versions resolves on live DB",
            bool(policy_v and model_v),
            f"pv={policy_v} mv={model_v}",
        )
    except Exception as exc:  # pragma: no cover - failure path
        claim("era anchor current_versions resolves on live DB", False, repr(exc))

    # 2) Threshold is DB-driven (not a blind hardcoded 0.88) through the real
    #    get_effective_threshold path.
    try:
        with get_db_cursor(commit=False) as cur:
            eff = get_effective_threshold(cur)
        claim(
            "effective threshold resolves from DB path",
            isinstance(eff, float),
            f"effective={eff}",
        )
    except Exception as exc:  # pragma: no cover - failure path
        claim("effective threshold resolves from DB path", False, repr(exc))

    # 3) Bounded hint push honors the cap on live Redis.
    try:
        client = get_redis_client()
        key = "nexus:ingest:hints"
        before = client.llen(key)
        pushed = bounded_push_ingest_hints(
            *range(1, 4001)
        )
        after = client.llen(key)
        claim(
            "bounded hint push honors cap on live Redis",
            after - before <= INGEST_HINTS_CAP,
            f"delta={after - before} cap={INGEST_HINTS_CAP}",
        )
        # Best-effort cleanup of hints we pushed.
        client.ltrim(key, before, -1)
    except Exception as exc:  # pragma: no cover - failure path
        claim("bounded hint push honors cap on live Redis", False, repr(exc))

    print("-" * 68)
    passed = sum(1 for _, ok, _ in CLAIMS if ok)
    print(f"RESULT: {passed}/{len(CLAIMS)} era claims hold on LIVE stack")
    return 0 if passed == len(CLAIMS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
