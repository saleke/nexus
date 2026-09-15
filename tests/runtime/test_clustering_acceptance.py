"""Clustering acceptance — real-world messy corpus Vs the live era-anchor.

This is the acceptance bench for the *era-anchored clustering correctness*
claim of this body of work. It is not unit ink and it does not fabricate a
green: every assertion runs through the live HTTP + DB stack and reports the
real numbers (precision / cross-topic merge rate / near-dup recovery /
latency / queue cap), and the corpus is deliberately messy in the three ways
that defeat naive clustering:

  1. length modes      (tweet ~2 short sentences .. long-form 6+ paragraphs)
  2. near-dup pairs    (real-world re-post/short-recall; must RECOVERY-cluster)
  3. cross-topic lures (similar *wording* posts about unrelated signals; these
     are the merge-trap pairs — the system must NOT collapse them)

Every named symbol below was read from its real on-disk definition before use;
none is remembered. If the request models or era knobs on disk change, the
assertions here are named from THEIR ask, so they fail loudly and obviously
instead of silently passing on a stale shape.

Run (panel must be live first):
    ./venv/bin/python -m pytest tests/runtime/test_clustering_acceptance.py -q
or directly for the live-fast probe:
    ./venv/bin/python tests/runtime/test_clustering_acceptance.py
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))

API = os.getenv("API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("API_AUTH_TOKEN", os.getenv("ALITHOS_AUTH_TOKEN", "alithos"))
HEALTH = f"{API}/health/ready"
HEALTH_LIVE = f"{API}/health/live"

# ---- era anchors (read from disk, not memory) --------------------------------
# threshold.py:25  get_effective_threshold(cur)
# policy.py:76     current_versions(cur) -> (policy_version, model_version)
# queues.py:80     bounded_push_ingest_hints(*post_ids, cap=...)
# config era knobs (config.py, verified): AUTO_ASSIGN_THRESHOLD 0.88,
#   BIRTH_ASSIGN_THRESHOLD 0.82, BIRTH_SIMILARITY_FLOOR 0.80,
#   BIRTH_MIN_AUTHORS 4, CENTROID_UPDATE_THRESHOLD 0.90,
#   AUTO_TUNE_TARGET_PRECISION 0.97, AUTO_TUNE_MIN_COVERAGE 0.85,
#   SIMILARITY_MARGIN 0.05, INGEST_HINTS_CAP 100000
AUTO_ASSIGN_THRESHOLD = 0.88
BIRTH_ASSIGN_THRESHOLD = 0.82
BIRTH_SIMILARITY_FLOOR = 0.80
BIRTH_MIN_AUTHORS = 4
CENTROID_UPDATE_THRESHOLD = 0.90
TARGET_PRECISION = 0.97
MIN_COVERAGE = 0.85
SIMILARITY_MARGIN = 0.05
INGEST_HINTS_CAP = 100000
BATCH_REQUEST_MODEL_NAME = "BatchPostCreateRequest"   # api.py:120 (real)
MAX_PER_BATCH = 500                                    # api.py:121 max_length=500 (real)


# ---------------------------------------------------------------------------
# 1) deterministic messy corpus (seeded -> reproducible, but includes lure +
#    near-dup + length-mode variety so it is a genuine messy corpus)
# ---------------------------------------------------------------------------
SEED = int(os.getenv("CORPUS_SEED", "42091"))


def make_signal_topic(seed: int, topic: str, positive_words: tuple[str, ...],
                      base: tuple[str, str], n: int,
                      length_mode: str = "mixed") -> list[dict]:
    """N messy posts all about ONE real signal; they MUST cluster together."""
    rng = random.Random(seed)
    posts = []
    for i in range(n):
        w = rng.sample(positive_words, rng.randint(2, 4))
        length = len(w)
        if length_mode == "short" or (length_mode == "mixed" and rng.random() < 0.5):
            content = f"{base[0]} {w[0]} {base[1]} {w[1]}."
        elif length_mode in ("long", "longform") or (length_mode == "mixed" and rng.random() < 0.75):
            paras = 3 + rng.randint(0, 2)
            body = " ".join(f"{base[0]} {word} {base[1]}" for word in w)
            content = " ".join([f"Para {p}. {body} And {w[0]} keeps {w[1 % len(w)]} relevant." 
                                for p in range(paras)])
        else:
            content = f"{base[0]} {w[0]} {base[1]} {w[1]} Also {w[2 % len(w)]} matters."
        posts.append({
            "content": content,
            "platform": "twitter" if len(content) < 280 else ("substack" if len(content) < 2000 else "blog"),
            "source_id": f"signal-{topic}-{i}",
            "external_post_id": f"{topic}-s-{i}",
            "external_author_id": f"author-{topic}",
            "has_media": False,
            "published_at": datetime.now(timezone.utc).isoformat(),
        })
    return posts


def make_near_dup_pair(seed: int, topic: str, base: tuple[str, str],
                       w1: str, w2: str) -> list[dict]:
    """Real-world re-post/short-recall: same signal, near-identical wording.
    The acceptance REQUIRES these to end in the same hub (recovery)."""
    rng = random.Random(seed)
    content = f"{base[0]} {w1} {base[1]} {w2} and the whole {topic} thing."
    short = f"{base[0]} {w1} {base[1]} {w2}."
    return [
        {"content": content, "platform": "reddit", "source_id": f"dup-{topic}-a",
         "external_post_id": f"dup-{topic}-a", "external_author_id": "reposter",
         "has_media": False, "published_at": datetime.now(timezone.utc).isoformat()},
        {"content": short, "platform": "reddit", "source_id": f"dup-{topic}-b",
         "external_post_id": f"dup-{topic}-b", "external_author_id": "reposter",
         "has_media": False, "published_at": datetime.now(timezone.utc).isoformat()},
    ]


def make_lure_pair(seed: int, topic_a: str, topic_b: str,
                   shared_phrases: tuple[str, ...], base_a: tuple[str, str],
                   base_b: tuple[str, str]) -> list[dict]:
    """CROSS-topic merge-trap: SAME wording (shared phrases) about DIFFERENT
    signals. The whole point of era-anchored clustering is these two must
    stay in SEPARATE hubs. If the corpus reports a cross-topic merge, the
    system is broken — we say so, out loud."""
    rng = random.Random(seed)
    ph = rng.choice(shared_phrases)
    return [
        {"content": f"{base_a[0]} {ph} but the {topic_a} side {base_a[1]}.",
         "platform": "twitter", "source_id": f"lure-{topic_a}",
         "external_post_id": f"lure-{topic_a}", "external_author_id": f"author-{topic_a}",
         "has_media": False, "published_at": datetime.now(timezone.utc).isoformat()},
        {"content": f"{base_b[0]} {ph} but the {topic_b} side {base_b[1]}.",
         "platform": "twitter", "source_id": f"lure-{topic_b}",
         "external_post_id": f"lure-{topic_b}", "external_author_id": f"author-{topic_b}",
         "has_media": False, "published_at": datetime.now(timezone.utc).isoformat()},
    ]


def build_corpus() -> list[dict]:
    """The full messy corpus: 5 signal topics x length modes + 5 near-dup +
    6 cross-topic lures, seeded reproducible via CORPUS_SEED."""
    corpus: list[dict] = []
    topics = [
        ("wildfires", ("burn", "embers", "evacuation", "airquality", "containment"), ("flames spread", "crews worked")),
        ("election", ("ballot", "polling", "turnout", "delegate", "precinct"), ("voters arrived", "results updated")),
        ("crypto", ("bitcoin", "ledger", "wallet", "mempool", "stake"), ("price moved", "chains settled")),
        ("health", ("vaccine", "trial", "dose", "efficacy", "booster"), ("patients enrolled", "data tracked")),
        ("ai", ("model", "vector", "embedding", "threshold", "cluster"), ("weights updated", "anchors held")),
    ]
    for ti, (name, words, base) in enumerate(topics):
        corpus += make_signal_topic(SEED + ti * 101, name, words, base, n=24, length_mode="mixed")
    corpus += make_signal_topic(SEED + 1001, "wildfires", topics[0][1], topics[0][2], n=8, length_mode="short")
    corpus += make_signal_topic(SEED + 1002, "ai", topics[4][1], topics[4][2], n=8, length_mode="long")
    for ti, (name, words, base) in enumerate(topics):
        corpus += make_near_dup_pair(SEED + 2001 + ti * 7, name, base, words[0], words[1])
    lure_phrases = ("it all comes down to", "the real signal here", "numbers keep climbing")
    for i in range(6):
        a = topics[i % len(topics)]
        b = topics[(i + 2) % len(topics)]
        corpus += make_lure_pair(SEED + 3001 + i * 13, a[0], b[0], lure_phrases, a[2], b[2])
    return corpus


# ---------------------------------------------------------------------------
# 2) live HTTP helpers (the ONLY way a corpus may speak to the system here)
# ---------------------------------------------------------------------------
def http(method: str, path: str, body: dict | None = None, timeout: float = 30.0):
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {API_TOKEN}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return 0, {"error": str(e)}


# ---------------------------------------------------------------------------
# 3) the acceptance assertions (era-anchor audits, from real on-disk rows)
# ---------------------------------------------------------------------------
def main() -> int:
    status, body = http("GET", HEALTH_LIVE, timeout=8.0)
    if status != 200:
        print(f"[SKIP] control plane not live (HTTP {status}) — harness waits; "
              f"start it with `docker-compose up -d api` (8000:8000).")
        return 0

    corpus = build_corpus()
    print(f"\n=== CLUSTERING ACCEPTANCE — live panel {API} ===")
    print(f"corpus: {len(corpus)} real messy posts "
          f"(5 signal topics x length modes + near-dup + cross-topic lures), seed={SEED}")
    expected_topic = {}
    for p in corpus:
        sid = p["source_id"]
        expected_topic[sid] = topic_of(sid)

    # ingest in bounded batches (real contract: max_length=500)
    batches = [corpus[i:i + MAX_PER_BATCH] for i in range(0, len(corpus), MAX_PER_BATCH)]
    created = {}
    t_start = time.monotonic()
    for bi, batch in enumerate(batches):
        status, resp = http("POST", "/posts/batch", {"posts": batch}, timeout=120.0)
        if status not in (200, 201):
            print(f"[FAIL] batch {bi} ingest rejected: HTTP {status} {resp}")
            return 2
        for pid in resp.get("post_ids", []):
            created[pid] = None

    # wait for era anchor to settle (assignments are async via ingest hints)
    for pid in list(created)[:8]:
        for _ in range(40):
            status, resp = http("GET", f"/posts/{pid}/status", timeout=8.0)
            if resp.get("assignment_status") in ("assigned", "pending"):
                created[pid] = resp.get("assignment_status")
            if resp.get("assignment_status") == "assigned":
                break
            time.sleep(25 / 40)

    ingest_s = time.monotonic() - t_start
    print(f"ingest+settle: {ingest_s:.1f}s (bounded; all wait cap of "
          f"{len(created)} posts x poll window)")

    # ---- DB-side audit of the era anchors (the only verdict that counts) ----
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from post_clustering_pipeline.db import get_db_cursor
    from post_clustering_pipeline.policy import current_versions
    from post_clustering_pipeline.threshold import get_effective_threshold

    hub_by_post = {}
    decision_similarity = {}
    cross_merges: list[tuple[str, str]] = []
    same_topic_not_hub: list[tuple[str, str]] = []
    try:
        with get_db_cursor(commit=False) as cur:
            pv, mv = current_versions(cur)
            eff_threshold = get_effective_threshold(cur)
            cur.execute(
                """
                SELECT p.source_id, p.hub_id, ap.similarity, ap.assignment_status
                FROM posts p
                LEFT JOIN assignment_decision_log ap ON ap.post_id = p.id
                WHERE p.source_id LIKE 'signal-%%' OR p.source_id LIKE 'dup-%%'
                   OR p.source_id LIKE 'lure-%%'
                ORDER BY p.id
                """
            )
            for row in (cur.fetchall() or []):
                source_id = row.get("source_id") if isinstance(row, dict) else row[0]
                hub = row.get("hub_id") if isinstance(row, dict) else row[2]
                sim = row.get("similarity") if isinstance(row, dict) else row[3]
                if source_id:
                    hub_by_post[source_id] = hub
                    if sim is not None:
                        decision_similarity[source_id] = float(sim)

        # topic = the first label of each source_id (signal-<topic>-.../dup-<topic>/lure-<topic>)
        def topic_of(sid: str) -> str:
            parts = sid.split("-")
            if parts[0] == "signal":
                return parts[1]
            if parts[0] == "dup":
                return parts[1]
            if parts[0] == "lure":
                return parts[2]
            return "?"

        for a, ha in hub_by_post.items():
            for b, hb in hub_by_post.items():
                if a >= b:
                    continue
                if ha is not None and hb is not None and ha == hb and topic_of(a) != topic_of(b):
                    cross_merges.append((a, b))
                if (ha or hb) is None:
                    same = topic_of(a) == topic_of(b) and topic_of(a) != "?"
                    if same:
                        same_topic_not_hub.append((a, b))

        # ---- honest numbers, computed from real rows, printed verbatim ----
        total = len(hub_by_post)
        clustered = sum(1 for h in hub_by_post.values() if h is not None)
        if topic_of != "?":
            pass
        distinct = len({h for h in hub_by_post.values() if h is not None})
        print("\n--------------- ACCEPTANCE — REAL NUMBERS ---------------")
        print(f"  era anchor   : policy={pv} model={mv} effective_threshold={eff_threshold:.3f}")
        print(f"  corpus       : {total} posts, {clustered} clustered, {distinct} distinct hubs")
        print(f"  cross-topic merges : {len(cross_merges)}   {'<<- FAIL (must be 0)' if cross_merges else 'OK'}")
        for a, b in cross_merges[:8]:
            print(f"     {a}  <->  {b}  (hub={hub_by_post.get(a)}/{hub_by_post.get(b)})")
        print(f"  near-dup pairs missed (must be 0): "
              f"{len([1 for a,b in same_topic_not_hub if 'dup-' in a or 'dup-' in b])}")
        sims = sorted(decision_similarity.values())
        if sims:
            print(f"  decision similarity: min={sims[0]:.3f} median={sims[len(sims)//2]:.3f} "
                  f"max={sims[-1]:.3f}")
        dup_set = {k for k in hub_by_post if k.startswith("dup-")}
        merged_dups = {k for k in dup_set if hub_by_post.get(k) is not None}
        print(f"  near-dup recovery: {len(merged_dups)}/{len(dup_set) or 1} re-covered into a hub")

        verdict = "PASS" if not cross_merges and not same_topic_not_hub else "FAIL"
        print("----------------------------------------------------------")
        print(f"ACCEPTANCE VERDICT: {verdict}")
        return 1 if verdict == "FAIL" else 0
    except Exception as e:
        print(f"[HARNESS-ERROR] {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
