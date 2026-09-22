"""Clustering STRESS bench — the hard, end-to-end, real-stack verification.

This is the adversarial correlation to ``test_clustering_acceptance``. It does
NOT fabricate a green: every number is read back from the live PostgreSQL rows
the pipeline wrote, and the corpus is deliberately brutal in every dimension
that has ever bitten this system:

  * 16 real topics x every post kind (short + long-form, exact-dup floods,
    near-dup at the 0.98 fold boundary, edited variants that must NOT fold,
    and a late wave that re-hits live hubs after birth already ran);
  * a confusable actor pair (Apple vs Samsung launch: template-identical
    wording) that the embedder CANNOT tell apart — separation must come from
    the dominant-ORG veto / entity split, not from cosine;
  * cross-topic lure pairs that share literal wording;
  * a 2-author marketing spam flood that PASSES the discourse gate yet must
    never earn a hub (BIRTH_MIN_AUTHORS anti-spam guardrail);
  * two 3-author "weak" topics that must also never earn a hub;
  * pure noise that must be rejected by the discourse gate before embedding.

Determinism contract: the scheduled task *functions* are the real exported
code paths (``drain_ingest_hints`` / ``reconcile_pending_posts`` /
``run_event_birth_scheduled`` / ``fold_hub_reposts_scheduled`` /
``run_hub_merge_scheduled``), invoked in a fixed order with the worker running
but the Beat OFF, so the audit measures pipeline correctness instead of Beat
cadence. Encoding stays inside the worker (single-flight model lock); the
driver only orchestrates.

Runbook (from repo root, DB wiped first):
    export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/clustering_db
    1) offline band oracle (no services needed):
         .venv/bin/python -u tests/runtime/test_clustering_stress.py --validate 2>&1 | tee /tmp/stress-validate.log
    2) wipe DB, start worker + api with Beat OFF, then:
         .venv/bin/python -u tests/runtime/test_clustering_stress.py 2>&1 | tee /tmp/stress-run.log
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

API = os.getenv("API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("API_AUTH_TOKEN", os.getenv("ALITHOS_AUTH_TOKEN", "alithos"))
HEALTH = f"{API}/health/live"

# On-disk era anchors (read through the live package, never remembered).
from post_clustering_pipeline import config as _cfg  # noqa: E402
from post_clustering_pipeline.nlp import content_signature  # noqa: E402

BIRTH_MIN_AUTHORS = _cfg.BIRTH_MIN_AUTHORS
BIRTH_ASSIGN_THRESHOLD = _cfg.BIRTH_ASSIGN_THRESHOLD
BIRTH_SIMILARITY_FLOOR = _cfg.BIRTH_SIMILARITY_FLOOR
BIRTH_COHESION_FLOOR = _cfg.BIRTH_COHESION_FLOOR
AUTO_ASSIGN_THRESHOLD = _cfg.AUTO_ASSIGN_THRESHOLD
MERGE_ENTITY_FLOOR = _cfg.MERGE_ENTITY_FLOOR
MERGE_NO_IDENTITY_SIM = _cfg.MERGE_NO_IDENTITY_SIM
MERGE_ORG_VETO_MIN_ORGS = _cfg.MERGE_ORG_VETO_MIN_ORGS
REPOST_NEAR_DUP_COSINE = _cfg.REPOST_NEAR_DUP_COSINE
INGEST_HINTS_CAP = _cfg.INGEST_HINTS_CAP

SIGNAL_N = 20          # signal posts per topic
BLOG_N = 6             # long-form blog posts per topic
FLOOD_N = 5            # byte-identical copies per topic (must all fold)
NEAR_N = 1             # one near-dup pair per topic (cos >= 0.98 -> fold)
VAR_N = 3              # edited variants per topic (cos < 0.98 -> MUST stay canonical)
LATE_N = 2             # late wave per topic (1 exact copy + 1 variant)
LURE_PAIRS = 6
SPAM_AUTHORS = 2
SPAM_PER_AUTHOR = 20
WEAK_TOPICS = 2
WEAK_AUTHORS = 3
WEAK_POSTS = 8
NOISE_N = 24

SEED = int(os.getenv("CORPUS_SEED", "911"))


@dataclass
class Topic:
    name: str
    anchor: str
    words: tuple[str, ...]
    base: tuple[str, str]   # (lead, tail) — repeated in every post of the topic
    joins: tuple[str, ...]
    brand: str | None = None
    brand_alt: str | None = None   # the confusable rival (apple <-> samsung)


def _topics(sig) -> list[Topic]:
    """All 16 topics. Apple/Samsung deliberately share lead/tail/joins word-for
    word so their hubs are maximally confusable to the embedder."""
    W = ("wind", "fireline", "ash", "smoke", "crews")
    return [
        Topic("wildfires", "Denver", ("flames", "embers", "evacuation", "airquality", "containment"),
              ("Flames tore across the foothills near Denver", "crews widened the containment line"),
              ("as smoke drifted westward", "in hard-hit foothill areas", "while air quality alerts spread", "despite cooler evening air")),
        Topic("river", "Columbus", ("downpour", "levee", "crest", "sandbag", "creek"),
              ("The river crested past the floodwalls in Columbus", "sandbag crews braced the levee"),
              ("as rain kept falling overnight", "near the swollen tributaries", "while the spillway stayed open", "across the low-lying wards")),
        Topic("election", "Ohio", ("ballot", "turnout", "precinct", "delegate", "recount"),
              ("Early ballots flooded precincts across Ohio", "officials kept tallying the returns"),
              ("as turnout climbed", "while precincts kept reporting", "once the polls had closed", "across the tightest counties")),
        Topic("crypto", "Bitcoin", ("ledger", "wallet", "mempool", "staking", "volume"),
              ("Bitcoin punched past its weekly range", "the ledger settled record volume"),
              ("once the mempool cleared", "on record exchange activity", "while leveraged flows surged", "after the halving date")),
        Topic("health", "Cincinnati", ("trial", "dose", "efficacy", "booster", "cohort"),
              ("The patient cohort grew overnight in Cincinnati", "efficacy data kept tracking"),
              ("as recruitment accelerated", "while safety data accumulated", "across the study network", "after the second reading")),
        Topic("ai", "Palo Alto", ("model", "vector", "embedding", "threshold", "centroid"),
              ("Embedding quality jumped at the Palo Alto retrain", "the anchor weights held stable"),
              ("after the nightly retrain", "once the cluster refreshed", "as the new weights settled", "on the updated anchor set")),
        Topic("ev", "Phoenix", ("charger", "battery", "range", "kilowatt", "grid"),
              ("Charging networks expanded across the Phoenix metro", "range anxiety kept fading"),
              ("as the supercharger count grew", "while winter range shrank", "along the interstate corridor", "despite rising kilowatt prices")),
        Topic("housing", "Portland", ("mortgage", "rates", "listing", "appraisal", "closing"),
              ("Listings dried up across the Portland metro", "mortgage rates kept climbing"),
              ("as docs crawled toward closing", "while appraisals slipped", "on the east side corridors", "despite two straight rate cuts")),
        Topic("storm", "Miami", ("hurricane", "gusts", "landfall", "blackout", "outages"),
              ("The storm front slammed into the Miami shore", "powerline crews worked through the night"),
              ("as gusts topped the scale", "while landfall clipped the coast", "once the eye passed", "across the barrier strip")),
        Topic("factory", "Detroit", ("layoff", "assembly", "inventory", "union", "tariff"),
              ("Detroit plants idled another line yesterday", "the union vote swings this week"),
              ("as the tariff wall rose", "while inventory piled up", "across the supplier plants", "despite the latest contract offer")),
        Topic("school", "Austin", ("teacher", "budget", "enrollment", "auditorium", "bond"),
              ("Austin trustees pushed the budget vote back", "teacher pay drives the new contract"),
              ("as enrollment spiked", "while the bond math failed", "across the district campuses", "despite the state allocation")),
        Topic("shipping", "Long Beach", ("container", "berth", "tariff", "schedule", "backlog"),
              ("Long Beach docks stacked empty containers", "tariff deadlines scrambled the schedule"),
              ("as the backlog grew", "while berths sat unused", "on the transpacific route", "despite the overtime deal")),
        Topic("apple", "Apple", ("chip", "iPhone", "Vision", "app", "privacy"),
              ("Apple stepped onto its flagship stage today", "the store stock sells out fast this week"),
              ("as the chip order grew", "while the app store cleared", "on the flagship retail floor", "despite the battery warning"), brand="Apple", brand_alt="Samsung"),
        Topic("samsung", "Samsung", ("Galaxy", "chip", "fold", "display", "chipset"),
              ("Samsung stepped onto its flagship stage today", "the store stock sells out fast this week"),
              ("as the chip order grew", "while the app store cleared", "on the flagship retail floor", "despite the battery warning"), brand="Samsung", brand_alt="Apple"),
        Topic("soccer", "Lakeville", ("goal", "referee", "outfield", "bench", "injury"),
              ("Lakeville Grammar edged the county final", "the roster stays thin for the summer"),
              ("as the referee calls piled up", "while the bench emptied", "on the soggy home pitch", "despite the injured captain")),
        Topic("garden", "Clover Hills", ("compost", "bulb", "bed", "mulch", "tomato"),
              ("The Clover Hills garden club met again", "the compost schedule slipped a week"),
              ("as the bulb order shipped", "while the tomato rows dried", "on the raised beds", "despite the late frost scare")),
    ]


# ---------------------------------------------------------------------------
# 1) deterministic messy corpus (seeded, reproducible, hostile by design)
# ---------------------------------------------------------------------------
def _authors(topic: str, n: int, i: int) -> str:
    return f"author-{topic}-{i % n}"


def _post(content: str, sid: str, topic: str, authors: int, i: int,
          published_at: datetime, platform: str | None = None) -> dict:
    if platform is None:
        platform = "twitter" if len(content) < 280 else ("substack" if len(content) < 2000 else "blog")
    return {
        "content": content,
        "platform": platform,
        "source_id": sid,
        "external_post_id": sid,
        "external_author_id": _authors(topic, authors, i),
        "has_media": False,
        "published_at": published_at.isoformat(),
    }


def _signal_body(rng, t: Topic, short: bool) -> str:
    lead, tail = t.base
    joins = t.joins
    if short:
        w = rng.sample(t.words, rng.randint(2, min(3, len(t.words))))
        return f"{lead}, {w[0]} {w[1]} — {tail}."
    w = rng.sample(t.words, rng.randint(3, min(5, len(t.words))))
    body = " ".join(f"{lead} after {word}; {rng.choice(joins)} {tail}." for word in w)
    return f"{body} Rounding up: {tail} regardless of {w[0]}."


def _blog_body(rng, t: Topic) -> str:
    lead, tail = t.base
    joins = t.joins
    paras = []
    for _ in range(4):
        w = rng.sample(t.words, rng.randint(2, 4))
        sent = " ".join(f"{lead} {word}: {rng.choice(joins)} {tail}." for word in w)
        paras.append(sent + " " + f"{rng.choice(joins)} {tail} while {lead} stays the headline.")
    return "\n\n".join(paras)


def topic_posts(seed: int, t: Topic, authors: int = 8) -> list[dict]:
    """38 posts per topic: 20 signal + 6 blog + 5 exact dup + 1 near-dup pair
    + 3 edited variants + 2 late-wave (exact copy + variant). Weak topics
    (soccer/garden) emit with WEAK_AUTHORS so the birth author guardrail can
    never license a hub for them."""
    rng = random.Random(seed)
    out: list[dict] = []
    now = datetime.now(timezone.utc)
    sig_body = _signal_body(rng, t, short=True)          # the duplicated body
    sig_by_sid: dict[str, str] = {}
    for i in range(SIGNAL_N):
        c = _signal_body(rng, t, short=(i % 5 == 0))
        sid = f"signal-{t.name}-{i + 1}"
        sig_by_sid[sid] = c
        out.append(_post(c, sid, t.name, authors, i, now - timedelta(minutes=rng.randint(0, 240))))
    for i in range(BLOG_N):
        c = _blog_body(rng, t)
        out.append(_post(c, f"blog-{t.name}-{i + 1}", t.name, authors, i, now - timedelta(minutes=rng.randint(0, 240))))
    for j in range(FLOOD_N):
        owner = f"signal-{t.name}-{(j % SIGNAL_N) + 1}"
        out.append(_post(sig_by_sid[owner], f"flood-{t.name}-{j + 1}", t.name, authors, j, now - timedelta(minutes=rng.randint(0, 240))))
    # near-dup pair at BIRTH: the edit strips punctuation only, so the
    # cleansed embedding text is byte-identical while content_sig differs.
    a = sig_body
    b = a.replace(",", " ").replace("—", " ").replace(".", " ").replace(":", " ").replace(";", " ")
    b = " ".join(b.split()) + " as noted"
    out.append(_post(a, f"near-{t.name}-a", t.name, authors, 6, now - timedelta(minutes=rng.randint(0, 240))))
    out.append(_post(b, f"near-{t.name}-b", t.name, authors, 7, now - timedelta(minutes=rng.randint(0, 240))))
    for k in range(VAR_N):
        # heavier edit: synonym swaps + clause shuffle => cos < 0.98. Each of
        # the three variants uses a structurally different sentence so the
        # pairwise variant-vs-variant cosine stays below the fold threshold
        # too (two templated variants with overlapping keyword runs measured
        # >= 0.98 and were wrongly folded onto each other in the live run).
        lead, tail = t.base
        w = rng.sample(t.words, 3)
        jo = rng.choice(t.joins)
        if k == 0:
            v = f"{lead} {tail}, {jo} says {w[0]} while {w[1]} keep moving for {w[2]}."
        elif k == 1:
            v = f"{w[0]} and {w[1]} now anchor the {lead}; {tail} hinges on {w[2]} once {jo}."
        else:
            v = f"The {lead} still turns on {w[0]} — {w[1]} would {jo} before {w[2]}, and {tail}."
        out.append(_post(v, f"var-{t.name}-{k + 1}", t.name, authors, (k + 2) % 8, now - timedelta(minutes=rng.randint(0, 240))))
    # late wave posts (ingested AFTER birth in the live run)
    late_src = sig_by_sid[f"signal-{t.name}-{SIGNAL_N}"]
    out.append(_post(late_src, f"late-{t.name}-a", t.name, authors, 5, now - timedelta(minutes=5)))
    lv = f"{t.base[0]} {t.base[1]} {rng.choice(t.words)} {rng.choice(t.joins)} {t.base[1]}."
    out.append(_post(lv, f"late-{t.name}-b", t.name, authors, 4, now - timedelta(minutes=5)))
    return out


def make_lure(seed: int, a: Topic, b: Topic, k: int, phrase: str) -> list[dict]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    return [
        _post(f"{a.base[0]} {phrase} but the {a.name} side {a.base[1]}.", f"lure-{a.name}-{k}", a.name, 8, 0, now - timedelta(minutes=90)),
        _post(f"{b.base[0]} {phrase} but the {b.name} side {b.base[1]}.", f"lure-{b.name}-{k}", b.name, 8, 1, now - timedelta(minutes=90)),
    ]


def make_spam(seed: int) -> list[dict]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    body = [
        "Join Ashton Falls today for instant bonuses on every shared referral link and the fastest signup in the west.",
        "Ashton Falls pays out fast: stack rewards with each new referral and unlock the premium tier tonight.",
        "Everyone is switching to Ashton Falls bonuses, grab your referral spot and level your earnings all week long.",
    ]
    out = []
    for a in range(SPAM_AUTHORS):
        for i in range(SPAM_PER_AUTHOR):
            c = body[(a + i) % len(body)]
            out.append(_post(c, f"spam-{a}-{i}", "spam", 2, a, now - timedelta(minutes=rng.randint(0, 180))))
    return out


def make_weak(seed: int, t: Topic) -> list[dict]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    out = []
    for a in range(WEAK_AUTHORS):
        for i in range(WEAK_POSTS):
            w = rng.sample(t.words, 2)
            c = f"{t.base[0]} {w[0]} {w[1]} {t.base[1]}."
            out.append(_post(c, f"weak-{t.name}-{a}-{i}", t.name, WEAK_AUTHORS, a, now - timedelta(minutes=rng.randint(0, 200))))
    return out


def make_noise(seed: int) -> list[dict]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    pool = [
        "the wind felt warm and the whole afternoon stretched slow and easy",
        "a quiet morning on the porch with coffee and nothing else to do",
        "the ride back home felt long and nothing good came on the radio",
        "saw some ducks by the pond today totally calm and content",
        "the bread from the corner bakery tastes soft and fresh enough",
        "listening to old songs while the rain taps the window gently",
        "ordered takeout twice this week and cooking felt like too much effort",
        "the garden path needs a sweep before the leaves pile up again",
        "an ordinary tuesday with laundry folding and mild reruns on",
        "the new shoes squeak on the linoleum at work which is funny",
        "sat in the back row at the film and half missed the ending",
        "the espresso machine gurgles every morning at six sharp now",
        "walked the dog around the block a second time just for fun",
        "the library returned my book late and charged a small fine",
        "backyard tomatoes taste better than the store ones honestly",
        "the couch cushion hides the remote again and we laugh it off",
        "the corner bakery ran out of rolls again before dinner was over",
        "the sunset over the garage roof looked orange and very pretty",
        "my cousin visited and we sat on the porch talking all night",
        "the train was ten minutes behind but nobody onboard minded",
        "a slow saturday with pancakes and a crossword from the paper",
        "the hallway light flickers downstairs and we keep meaning to fix it",
        "stayed in bed an extra hour on the weekend without guilt",
        "the street fair had a band and plenty of nachos and it was nice",
    ]
    return [
        _post(p, f"noise-{i}", "noise", 1, 0, now - timedelta(minutes=rng.randint(0, 120)))
        for i, p in enumerate(rng.sample(pool, NOISE_N))
    ]


def build_corpus(seed: int = SEED) -> list[dict]:
    topics = _topics(seed)
    corpus: list[dict] = []
    weak_names = {topics[14].name, topics[15].name}
    for i, t in enumerate(topics):
        corpus += topic_posts(seed + i * 131, t, authors=3 if t.name in weak_names else 8)
    pairs = [(0, 3), (1, 5), (2, 7), (4, 9), (6, 10), (8, 11)]
    phrases = ("it all comes down to", "the real signal here", "numbers keep climbing")
    for k, (ai, bi) in enumerate(pairs):
        corpus += make_lure(seed + 3001 + k * 13, topics[ai], topics[bi], k, phrases[k % len(phrases)])
    corpus += make_spam(seed + 4001)
    for wi, ti in enumerate([14, 15]):
        corpus += make_weak(seed + 5001 + wi * 29, topics[ti])
    corpus += make_noise(seed + 6001)

    # Ingest contract: user_id per author, derived deterministically.
    author_ids: dict[str, int] = {}
    for p in corpus:
        a = p["external_author_id"]
        if a not in author_ids:
            author_ids[a] = 10000 + len(author_ids)
        p["user_id"] = author_ids[a]
    return corpus


# ---------------------------------------------------------------------------
# 2) live HTTP helpers (the ONLY way the corpus speaks to the system)
# ---------------------------------------------------------------------------
def http(method: str, path: str, body: dict | None = None, timeout: float = 60.0):
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


def classify(sid: str) -> tuple[str, str]:
    """(group, topic) for a source_id. groups: signal/blog/flood/near/var/late/
    lure/spam/weak/noise. topic '' for spam/noise."""
    parts = sid.split("-")
    grp = parts[0]
    if grp in ("signal", "blog", "flood", "near", "var", "late", "lure"):
        return grp, parts[1]
    if grp == "weak":
        return grp, parts[1]
    if grp == "spam":
        return "spam", ""
    if grp == "noise":
        return "noise", ""
    return sid, ""


def is_clusterable_intent(sid: str) -> bool:
    g, _ = classify(sid)
    return g in ("signal", "blog", "flood", "near", "var", "late", "lure")


# ---------------------------------------------------------------------------
# 3) offline validation oracle (runs on the REAL embedder, no services)
# ---------------------------------------------------------------------------
def validate(seed: int = SEED) -> int:
    from post_clustering_pipeline.nlp import analyze_discourse_batch, cleanse_text
    from post_clustering_pipeline.models import get_embedding_engine
    import numpy as np

    corpus = build_corpus(seed)
    print(f"\n=== STRESS-VALIDATE — offline oracle on seed {seed} ===")
    print(f"corpus: {len(corpus)} posts, {sum(1 for p in corpus if is_clusterable_intent(p['source_id']))} clusterable-intent, "
          f"{sum(1 for p in corpus if p['source_id'].startswith('spam'))} gate-passing spam, "
          f"{sum(1 for p in corpus if p['source_id'].startswith('weak'))} weak, "
          f"{sum(1 for p in corpus if p['source_id'].startswith('noise'))} noise")

    fails: list[str] = []
    by_sid = {p["source_id"]: p for p in corpus}
    sids = [p["source_id"] for p in corpus]
    verdicts = analyze_discourse_batch([p["content"] for p in corpus], batch_size=64)
    gate = {}
    for sid, (ok, reason, _ents) in zip(sids, verdicts):
        gate[sid] = (ok, reason)
    clusterable_failed = [sid for sid in sids if is_clusterable_intent(sid) and not gate[sid][0]]
    spam_failed = [sid for sid in sids if sid.startswith("spam") and not gate[sid][0]]
    weak_failed = [sid for sid in sids if sid.startswith("weak") and not gate[sid][0]]
    noise_admitted = [sid for sid in sids if sid.startswith("noise") and gate[sid][0]]

    print(f"gate: clusterable admitted {sum(1 for s in sids if is_clusterable_intent(s)) - len(clusterable_failed)}/{sum(1 for s in sids if is_clusterable_intent(s))}, "
          f"spam admitted {len([s for s in sids if s.startswith('spam')]) - len(spam_failed)}/{len([s for s in sids if s.startswith('spam')])}, "
          f"weak admitted {len([s for s in sids if s.startswith('weak')]) - len(weak_failed)}/{len([s for s in sids if s.startswith('weak')])}, "
          f"noise rejected {len([s for s in sids if s.startswith('noise')]) - len(noise_admitted)}/{len([s for s in sids if s.startswith('noise')])}")
    if clusterable_failed:
        fails.append(f"{len(clusterable_failed)} clusterable posts rejected by the gate: {clusterable_failed[:5]}")
    if spam_failed:
        fails.append(f"spam posts failed the gate but spam was DESIGNED to pass (2-author anti-hub test): {spam_failed[:3]}")
    if weak_failed:
        fails.append(f"weak posts failed the gate: {weak_failed[:3]}")
    if noise_admitted:
        fails.append(f"{len(noise_admitted)} noise posts ADMITTED by the gate: {noise_admitted[:3]}")

    # --- embedding bands on CLEANSED text (what the pipeline embeds) ---
    clusterable = [(p["source_id"], cleanse_text(p["content"])) for p in corpus if is_clusterable_intent(p["source_id"])]
    engine = get_embedding_engine()
    t0 = time.monotonic()
    vecs = engine.encode_batch([c for _, c in clusterable], batch_size=48)
    taken = time.monotonic() - t0
    embs = {sid: np.asarray(v, dtype=np.float64) for (sid, _), v in zip(clusterable, vecs)}
    norm = {sid: e / (np.linalg.norm(e) + 1e-9) for sid, e in embs.items()}
    print(f"encoded {len(clusterable)} clusterable texts in {taken:.1f}s ({len(clusterable)/taken:.1f} text/s single-process)")

    topics = {t.name for t in _topics(seed)}
    cloud = {t: [] for t in topics}
    lure_cloud = {t: [] for t in topics}
    for sid, e in norm.items():
        g, t = classify(sid)
        if t and t in cloud:
            cloud[t].append(sid)
        if g == "lure" and t and t in lure_cloud:
            lure_cloud[t].append(sid)

    # intra-topic / cross-topic bands from the SAME texts the live hub sees
    for t, sids_ in theme_sorted(cloud):
        if len(sids_) < 2:
            continue
        sims = np.array([[float(norm[a].dot(norm[b])) for b in sids_] for a in sids_])
        iu = sims[np.triu_indices_from(sims, k=1)]
        print(f"  {t:14s} n={len(sids_):3d} intra mean={iu.mean():.3f} min={iu.min():.3f}")

    all_top = list(topics)
    cross: dict[tuple[str, str], float] = {}
    for i in range(len(all_top)):
        for j in range(i + 1, len(all_top)):
            a, b = all_top[i], all_top[j]
            if not cloud[a] or not cloud[b]:
                continue
            ca = np.mean([norm[s] for s in cloud[a]], axis=0)
            cb = np.mean([norm[s] for s in cloud[b]], axis=0)
            ca /= np.linalg.norm(ca) + 1e-9
            cb /= np.linalg.norm(cb) + 1e-9
            cross[(a, b)] = float(ca.dot(cb))
    worst_other = {k: v for k, v in cross.items() if set(k) != {"apple", "samsung"}}
    worst = max(worst_other.items(), key=lambda kv: kv[1]) if worst_other else (("", ""), 0.0)
    ab = cross.get(("apple", "samsung"), cross.get(("samsung", "apple"), 0.0))
    print(f"  cross-topic centroid sim: max(excl apple×samsung)={worst[1]:.3f} ({worst[0][0]}/{worst[0][1]})")
    print(f"  confusable apple×samsung centroid sim: {ab:.3f} (merge MUST be vetoed, not separated by cosine)")

    for t in topics:
        if not lure_cloud[t]:
            continue
        for lu in lure_cloud[t]:
            tmean = np.mean([norm[s] for s in cloud[t]], axis=0)
            tmean /= np.linalg.norm(tmean) + 1e-9
            print(f"  lure {lu:16s} sim->own topic {float(norm[lu].dot(tmean)):.3f}")

    # exact / near / variant / late boundary checks (the 0.98 razor)
    topics_spec = {t.name: t for t in _topics(seed)}
    near_min = 1.0
    for t in topics:
        near_a = f"near-{t}-a"
        near_b = f"near-{t}-b"
        if near_a in norm and near_b in norm:
            c = float(norm[near_a].dot(norm[near_b]))
            near_min = min(near_min, c)
    var_max = 0.0
    latevar_max = 0.0
    varpair_max = 0.0
    variants_vs_sig = {t: [] for t in topics}
    for t in topics:
        sig_sids = [s for s in norm if classify(s) == ("signal", t)]
        if not sig_sids:
            continue
        for k in range(1, VAR_N + 1):
            v = f"var-{t}-{k}"
            if v in norm:
                worst_var = max(float(norm[v].dot(norm[s])) for s in sig_sids)
                var_max = max(var_max, worst_var)
                variants_vs_sig.setdefault(t, []).append(worst_var)
        lv = f"late-{t}-b"
        if lv in norm:
            latevar_max = max(latevar_max, max(float(norm[lv].dot(norm[s])) for s in sig_sids))
        # pairwise across the topic's variants: two variants must be further
        # apart than the fold threshold too (var-ai-3 folded onto var-ai-1 in
        # the live run because variant-vs-variant was >= 0.98 and unguarded).
        vs_ = [f"var-{t}-{k}" for k in range(1, VAR_N + 1) if f"var-{t}-{k}" in norm]
        for i in range(len(vs_)):
            for j in range(i + 1, len(vs_)):
                varpair_max = max(varpair_max, float(norm[vs_[i]].dot(norm[vs_[j]])))
    exact = []
    for t in topics:
        sig_sids = [s for s in norm if classify(s) == ("signal", t)]
        if not sig_sids:
            continue
        for j in range(1, FLOOD_N + 1):
            f = f"flood-{t}-{j}"
            owner = f"signal-{t}-{((j - 1) % SIGNAL_N) + 1}"
            if f in norm and owner in norm:
                exact.append(float(norm[f].dot(norm[owner])))
    print(f"  exact-dup cos (must be 1.0): min={min(exact) if exact else float('nan'):.4f}")
    print(f"  near-dup pair cos (fold boundary >= {REPOST_NEAR_DUP_COSINE:.2f}): min={near_min:.4f}")
    print(f"  variant cos vs own signal (must be < {REPOST_NEAR_DUP_COSINE:.2f}): max={var_max:.4f}")
    print(f"  variant-vs-variant pairwise cos (must be < {REPOST_NEAR_DUP_COSINE:.2f}): max={varpair_max:.4f}")
    print(f"  late-variant cos vs own signal (must be < {REPOST_NEAR_DUP_COSINE:.2f}): max={latevar_max:.4f}")
    for t, vals in variants_vs_sig.items():
        print(f"    {t:14s} variants vs signal: {', '.join(f'{v:.3f}' for v in vals)}")

    if near_min < REPOST_NEAR_DUP_COSINE:
        fails.append(f"near-dup pairs below fold dot: {near_min:.4f} < 0.98")
    if var_max >= REPOST_NEAR_DUP_COSINE:
        fails.append(f"variants at fold boundary (must NOT fold): max {var_max:.4f} >= 0.98")
    if varpair_max >= REPOST_NEAR_DUP_COSINE:
        fails.append(f"variant-vs-variant at fold boundary: max {varpair_max:.4f} >= 0.98")
    if latevar_max >= REPOST_NEAR_DUP_COSINE:
        fails.append(f"late variants at fold boundary: {latevar_max:.4f} >= 0.98")
    if worst[1] >= MERGE_ENTITY_FLOOR:
        fails.append(f"cross-topic centroid max {worst[1]:.3f} >= MERGE_ENTITY_FLOOR {MERGE_ENTITY_FLOOR} (risk of bogus merge, tune corpus)")
    if ab < MERGE_ENTITY_FLOOR:
        fails.append(f"apple×samsung centroids only {ab:.3f} apart — the ORG veto would never be exercised; corpus too easy")

    # signature collision check: intended same-topic dupes (flood/near/late
    # copying a signal body) are fine; cross-topic or noise dupes are a bug.
    bad_sig: list[str] = []
    sig_map: dict[int, list[str]] = {}
    for p in corpus:
        sig_map.setdefault(content_signature(p["content"]), []).append(p["source_id"])
    for sig, sids_ in sig_map.items():
        if len(sids_) <= 1:
            continue
        kinds = {classify(s)[0] for s in sids_}
        if "noise" in kinds:
            bad_sig.append(f"noise duplicated: {sids_}")
            continue
        tset = {classify(s)[1] for s in sids_ if classify(s)[1]}
        if len(tset) > 1:
            bad_sig.append(f"cross-topic sig {sig}: {sids_}")
    if bad_sig:
        fails.append("signature collisions: " + "; ".join(bad_sig[:5]))

    print("\n--------------- STRESS-VALIDATE VERDICT ---------------")
    if fails:
        for f in fails:
            print("  [FAIL] " + f)
        print("FAIL")
        return 1
    print("PASS — corpus is hard AND the bands sit on the right side of every threshold")
    return 0


def theme_sorted(d: dict):
    return sorted(d.items(), key=lambda kv: kv[0])


# ---------------------------------------------------------------------------
# 4) live audit — the only verdict that counts
# ---------------------------------------------------------------------------
def _db_audit(corpus, timings: dict) -> int:
    from post_clustering_pipeline.db import get_db_cursor, extract_val
    from post_clustering_pipeline.threshold import get_effective_threshold
    from post_clustering_pipeline.policy import current_versions

    sids = [p["source_id"] for p in corpus]
    fails: list[str] = []
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT p.id, p.source_id, p.event_id, p.assignment_status, p.repost_of_id,
                   p.embedding IS NOT NULL AS has_emb, p.content_sig, ap.similarity
            FROM posts p
            LEFT JOIN LATERAL (
                SELECT l.similarity FROM assignment_decision_log l
                WHERE l.post_id = p.id AND l.status NOT IN ('noise')
                ORDER BY l.id DESC LIMIT 1
            ) ap ON true
            WHERE p.source_id = ANY(%s)
            ORDER BY p.id
            """,
            (sids,)
        )
        rows = cur.fetchall() or []
        row_by_sid = {}
        for r in rows:
            sid = extract_val(r, "source_id", 1)
            row_by_sid[sid] = {
                "id": extract_val(r, "id", 0),
                "event_id": extract_val(r, "event_id", 2),
                "status": extract_val(r, "assignment_status", 3),
                "repost_of": extract_val(r, "repost_of_id", 4),
                "has_emb": bool(extract_val(r, "has_emb", 5)),
                "content_sig": extract_val(r, "content_sig", 6),
                "similarity": extract_val(r, "similarity", 7),
            }

        pv, mv = current_versions(cur)
        eff = get_effective_threshold(cur)

    missing = [s for s in sids if s not in row_by_sid]
    if missing:
        fails.append(f"{len(missing)} corpus posts never reached the DB: {missing[:5]}")

    # ---- 1. gate: noise rejected, everything clusterable got embedded ----
    noise_bad = [s for s in sids if s.startswith("noise") and row_by_sid.get(s, {}).get("status") != "noise"]
    if noise_bad:
        fails.append(f"{len(noise_bad)} noise posts not marked noise: {noise_bad[:5]}")
    spam_weak = [s for s in sids if s.startswith(("spam", "weak"))]
    embedded_not_topic = [s for s in sids if is_clusterable_intent(s) and not row_by_sid.get(s, {}).get("has_emb")]
    if embedded_not_topic:
        fails.append(f"{len(embedded_not_topic)} clusterable posts never embedded: {embedded_not_topic[:5]}")

    # ---- 2. spam / weak must never earn a hub ----
    swe_hub = [s for s in spam_weak if row_by_sid.get(s, {}).get("event_id") is not None]
    if swe_hub:
        fails.append(f"{len(swe_hub)} spam/weak posts wrongly in a hub: {swe_hub[:5]}")

    # ---- 3. one live hub per topic, zero cross-topic merges ----
    topic_of_sid = {s: classify(s)[1] for s in sids if classify(s)[1]}
    hub_posts = {s: row_by_sid[s]["event_id"] for s in row_by_sid if row_by_sid[s]["event_id"] is not None}
    # Only count posts whose hub is ACTIVE (merged-away fragments point at the survivor).
    active_hubs = set()
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT id FROM event_hubs WHERE is_active = TRUE;")
        active_hubs = {int(extract_val(r, "id", 0)) for r in (cur.fetchall() or [])}

    live_hub_posts = {s: h for s, h in hub_posts.items() if h in active_hubs}
    hubs_by_topic = {}
    for s, h in live_hub_posts.items():
        t = topic_of_sid.get(s) or "?"
        hubs_by_topic.setdefault(t, set()).add(h)
    multi = {t: hs for t, hs in hubs_by_topic.items() if len(hs) != 1}
    if multi:
        fails.append(f"topics with !=1 live hub: {multi}")
    expected_topics = {classify(s)[1] for s in sids if classify(s)[1]}
    missing_topic_hub = expected_topics - set(hubs_by_topic) - {"soccer", "garden"}
    if missing_topic_hub:
        fails.append(f"topics with no live hub: {missing_topic_hub}")

    # pairwise cross-topic detection via shared hub
    pair_map: dict[int, set[str]] = {}
    for s, h in live_hub_posts.items():
        pair_map.setdefault(h, set()).add(topic_of_sid.get(s) or "?")
    cross = [h for h, ts in pair_map.items() if len(ts) > 1]
    if cross:
        fails.append(f"cross-topic hub merge detected: {[ (h, sorted(pair_map[h])) for h in cross[:5] ]}")

    # per-topic recall: every signal/blog/flood/near/var/lure of a topic must
    # live in that topic's single live hub (100%, no partial clustering).
    for t in expected_topics - {"soccer", "garden"}:
        members = [s for s in sids if classify(s)[1] == t and is_clusterable_intent(s)]
        err = [s for s in members
               if row_by_sid.get(s, {}).get("event_id") not in (hubs_by_topic.get(t) or set())
               or row_by_sid.get(s, {}).get("status") not in ("assigned", "repost")]
        if err:
            fails.append(f"topic {t}: {len(err)}/{len(members)} posts not clustered into its hub: {err[:4]}")

    # no duplicate canonical bodies within any hub (one canonical per sig)
    canon_sig: dict[tuple[int, int], int] = {}
    for r in row_by_sid.values():
        if r["event_id"] is not None and r["repost_of"] is None and r["content_sig"]:
            canon_sig[(r["event_id"], r["content_sig"])] = canon_sig.get((r["event_id"], r["content_sig"]), 0) + 1
    dup_canon = {k: v for k, v in canon_sig.items() if v > 1}
    if dup_canon:
        fails.append(f"duplicate canonical bodies within a hub: {dup_canon}")

    # ---- 4. fold correctness ----
    chains = set()
    by_post = {r["id"]: r for r in row_by_sid.values()}
    for s, r in row_by_sid.items():
        seen = set()
        cur_ro = r["repost_of"]
        while cur_ro is not None:
            if cur_ro in seen:
                chains.add(s)
                break
            seen.add(cur_ro)
            nxt = by_post.get(cur_ro)
            if not nxt:
                break
            cur_ro = nxt["repost_of"]
    if chains:
        fails.append(f"repost chains/cycles: {list(chains)[:5]}")

    def _in_topic(s, t):
        r = row_by_sid.get(s)
        return r and r["event_id"] is not None and classify(s)[1] == t

    weak_topics = {"soccer", "garden"}
    for t in expected_topics - weak_topics:
        flood = [s for s in sids if s.startswith(f"flood-{t}-")]
        if flood and not all(row_by_sid.get(s, {}).get("status") == "repost" for s in flood):
            fails.append(f"non-folded exact flood in {t}: {[s for s in flood if row_by_sid.get(s,{}).get('status') != 'repost'][:3]}")
        near = [s for s in sids if s.startswith(f"near-{t}-")]
        if near and not any(row_by_sid.get(s, {}).get("status") == "repost" for s in near):
            fails.append(f"near-dup pair NOT folded in {t}: {near}")
        for k in range(1, VAR_N + 1):
            v = f"var-{t}-{k}"
            r = row_by_sid.get(v)
            if not _in_topic(v, t):
                fails.append(f"{v} missing from its topic hub")
            elif r and r["status"] == "repost":
                fails.append(f"variant {v} was WRONGLY folded (must stay canonical)")

    # late wave: exact copy folds inline, variant stays canonical in hub.
    # weak topics have no hub by design, so only the canonical topics enforce it.
    for t in expected_topics - weak_topics:
        la = f"late-{t}-a"
        lb = f"late-{t}-b"
        ra = row_by_sid.get(la, {})
        rb = row_by_sid.get(lb, {})
        if ra.get("status") != "repost" or not ra.get("event_id"):
            fails.append(f"late exact copy {la} not folded into its hub (status={ra.get('status')})")
        if not rb.get("event_id") or rb.get("status") == "repost":
            fails.append(f"late variant {lb} not a canonical member of its topic hub (status={rb.get('status')})")

    # ---- 5. hub count integrity: live rows == member_count + repost_count ----
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT eh.id, eh.member_count, eh.repost_count,
                   (SELECT COUNT(*) FROM posts p WHERE p.event_id = eh.id AND p.deleted_at IS NULL) AS live_rows
            FROM event_hubs eh WHERE eh.is_active = TRUE;
            """
        )
        for r in (cur.fetchall() or []):
            eid = extract_val(r, "id", 0)
            mc = int(extract_val(r, "member_count", 1) or 0)
            rc = int(extract_val(r, "repost_count", 2) or 0)
            live = int(extract_val(r, "live_rows", 3) or 0)
            if live != mc + rc:
                fails.append(f"hub {eid}: live_rows={live} != member_count({mc})+repost_count({rc})")

    # ---- 6. outbox parity: post.assigned events == outward canonicals ----
    # A post folded after assignment has its stale post.assigned event marked
    # 'superseded' by the fold sweep (fold.py run_fold_sweep), so active
    # events must equal outward canonicals exactly.
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            "SELECT COUNT(*) FROM integration_outbox WHERE event_type = 'post.assigned' AND delivery_status <> 'superseded';"
        )
        ob_assigned = int(extract_val(cur.fetchone(), "count", 0) or 0)
        cur.execute(
            "SELECT COUNT(*) FROM integration_outbox WHERE event_type = 'post.assigned' AND delivery_status = 'superseded';"
        )
        ob_superseded = int(extract_val(cur.fetchone(), "count", 0) or 0)
        cur.execute(
            "SELECT COUNT(*) FROM posts WHERE event_id IS NOT NULL AND repost_of_id IS NULL AND assignment_status <> 'repost';"
        )
        canon = int(extract_val(cur.fetchone(), "count", 0) or 0)
    if ob_assigned != canon:
        fails.append(f"outbox post.assigned={ob_assigned} != canonical assigned posts {canon} (superseded={ob_superseded})")

    # ---- honest numbers for the report ----
    clustered = sum(1 for r in row_by_sid.values() if r["event_id"] is not None)
    emb = sum(1 for r in row_by_sid.values() if r["has_emb"])
    noise_total = sum(1 for s in sids if s.startswith("noise"))
    spam_weak_total = sum(1 for s in spam_weak)
    print("\n--------------- STRESS AUDIT — REAL NUMBERS ---------------")
    print(f"  era anchor : policy={pv} model={mv} effective_threshold={eff:.3f}")
    print(f"  corpus     : {len(sids)} posts | {len(row_by_sid) - len(missing)} persisted | "
          f"clusterable embedded {emb}/{len([s for s in sids if is_clusterable_intent(s)])} | "
          f"spam/weak {spam_weak_total} (all hubless) | noise {noise_total} (all rejected)")
    print(f"  clustered  : {clustered} posts | {len(active_hubs)} live hubs")
    sims = sorted(r["similarity"] for r in row_by_sid.values() if r["similarity"] is not None and r["status"] not in ("noise",))
    if sims:
        print(f"  decision similarity: min={sims[0]:.3f} median={sims[len(sims)//2]:.3f} max={sims[-1]:.3f} (n={len(sims)})")
    print(f"  timings    : " + " | ".join(f"{k}={v:.1f}s" for k, v in sorted(timings.items())))

    verdict = "PASS" if not fails else "FAIL"
    print("----------------------------------------------------------")
    for f in fails:
        print("  [FAIL] " + f)
    print(f"STRESS VERDICT: {verdict}")
    return 1 if verdict == "FAIL" else 0


# ---------------------------------------------------------------------------
# 5) live driver — deterministic phase order, Beat OFF
# ---------------------------------------------------------------------------
def settle(sids_clusterable: list[str], timeout: float = 1800.0) -> dict:
    from post_clustering_pipeline.tasks import drain_ingest_hints, reconcile_pending_posts
    from post_clustering_pipeline.db import get_db_cursor, extract_val
    t0 = time.monotonic()
    detail: dict | None = None
    loop = 0
    while time.monotonic() - t0 < timeout:
        drain_ingest_hints(max_hints=2000)
        # Backstop STALE/crashed claims only — reconcile dispatches every
        # 'pending' row it sees, so calling it with a bulk batch_limit every
        # loop re-permits the whole remaining wave as ONE giant re-encode task
        # (measured: a 700-row task that re-encoded ~everything in 7.7 min
        # while progress froze). Chip at it rarely and small instead.
        if loop % 6 == 0:
            reconcile_pending_posts(batch_limit=256)
        loop += 1
        time.sleep(5)
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                "SELECT COUNT(*) FROM posts WHERE source_id = ANY(%s) AND embedding IS NULL AND assignment_status <> 'noise';",
                (sids_clusterable,)
            )
            unembedded = int(extract_val(cur.fetchone(), "count", 0) or 0)
            # straggler gate: an embedded clusterable post that has NO event_id
            # and is still sitting in 'pending'/'processing' has not yet been
            # parked into the birth buffer by the stream path. Birth runs only
            # on the buffer, so releasing on unembedded==0 alone could leave a
            # whole topic never seeing a birth pass.
            cur.execute(
                """
                SELECT COUNT(*)
                FROM posts
                WHERE source_id = ANY(%s)
                  AND embedding IS NOT NULL
                  AND event_id IS NULL
                  AND assignment_status IN ('pending', 'processing')
                """,
                (sids_clusterable,)
            )
            stray = int(extract_val(cur.fetchone(), "count", 0) or 0)
            cur.execute(
                "SELECT assignment_status, COUNT(*) FROM posts GROUP BY assignment_status;"
            )
            dist = {extract_val(r, "assignment_status", 0): int(extract_val(r, "count", 1) or 0) for r in (cur.fetchall() or [])}
        detail = {"unembedded": unembedded, "stray_pending": stray, "status_dist": dist}
        if unembedded == 0 and stray == 0:
            return {"elapsed": time.monotonic() - t0, **(detail or {})}
    return {"elapsed": time.monotonic() - t0, **(detail or {})}


def run(seed: int = SEED) -> int:
    status, body = http("GET", "/health/live", timeout=8.0)
    if status != 200:
        print(f"[SKIP] control plane not live (HTTP {status}) — start api+worker with Beat OFF first. {body}")
        return 0

    from post_clustering_pipeline.db import get_db_cursor, extract_val
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT COUNT(*) FROM posts;")
        existing = int(extract_val(cur.fetchone(), "count", 0) or 0)
    if existing:
        print(f"[SKIP] DB not clean ({existing} existing posts). Wipe the DB and restart worker/api before the stress run.")
        return 0

    corpus = build_corpus(seed)
    clust = [p["source_id"] for p in corpus if is_clusterable_intent(p["source_id"])]
    print(f"\n=== CLUSTERING STRESS — live panel {API} ===")
    print(f"corpus: {len(corpus)} posts (seed {seed}) | clusterable-intent {len(clust)} | "
          f"spam {sum(1 for p in corpus if p['source_id'].startswith('spam'))} | "
          f"weak {sum(1 for p in corpus if p['source_id'].startswith('weak'))} | "
          f"noise {sum(1 for p in corpus if p['source_id'].startswith('noise'))}")

    timings: dict[str, float] = {}

    # ---- phase 1: ingest main wave (all topics + lures + spam + weak + noise) ----
    main = [p for p in corpus if not p["source_id"].startswith("late-")]
    late = [p for p in corpus if p["source_id"].startswith("late-")]
    t = time.monotonic()
    for bi in range(0, len(main), 500):
        chunk = main[bi: bi + 500]
        s, resp = http("POST", "/posts/batch", {"posts": chunk}, timeout=120.0)
        if s not in (200, 201):
            print(f"[FAIL] ingest batch {bi} rejected: HTTP {s} {resp}")
            return 2
        if len(resp.get("post_ids", [])) != len(chunk):
            print(f"[FAIL] ingest batch {bi} returned {len(resp.get('post_ids', []))} ids for {len(chunk)} posts")
            return 2
    timings["ingest_main"] = time.monotonic() - t

    # ---- phase 2: settle (worker encodes; all clusterable become embedded
    #               AND parked in the birth buffer, no embedded strays left) ----
    t = time.monotonic()
    st = settle(clust)
    timings["settle"] = st["elapsed"]
    print(f"  settle: {timings['settle']:.1f}s unembedded={st.get('unembedded')} stray_pending={st.get('stray_pending')}")
    if st["unembedded"] != 0 or st.get("stray_pending", 0) != 0:
        print(f"[FAIL] main wave did not settle; status dist {st['status_dist']}")
        return 2

    # ---- phase 3: era birth (clusters + folds at birth + merge fixpoint) ----
    from post_clustering_pipeline.tasks import run_event_birth_scheduled
    t = time.monotonic()
    birth = run_event_birth_scheduled()
    timings["birth_and_merge"] = time.monotonic() - t
    if birth.get("status") != "completed":
        print(f"[FAIL] event birth skipped: {birth}")
        return 2
    print(f"  birth: merged_hubs={birth.get('merged_hubs')} in {timings['birth_and_merge']:.1f}s")

    # ---- phase 4: fold sweep (catches residual exact + near-dup) ----
    from post_clustering_pipeline.tasks import fold_hub_reposts_scheduled
    t = time.monotonic()
    fs1 = fold_hub_reposts_scheduled()
    timings["fold_sweep_1"] = time.monotonic() - t
    print(f"  fold sweep 1: {fs1}")

    # ---- phase 5: late wave (assigns against live hubs, inline exact fold) ----
    t = time.monotonic()
    for bi in range(0, len(late), 500):
        chunk = late[bi: bi + 500]
        s, resp = http("POST", "/posts/batch", {"posts": chunk}, timeout=120.0)
        if s not in (200, 201):
            print(f"[FAIL] late-wave ingest rejected: HTTP {s} {resp}")
            return 2
    st2 = settle(clust)
    timings["late_wave"] = st2["elapsed"]
    if st2["unembedded"] != 0 or st2.get("stray_pending", 0) != 0:
        print(f"[FAIL] late wave unresolved; status dist {st2['status_dist']}")
        return 2
    print(f"  late-wave settle: {timings['late_wave']:.1f}s unembedded={st2.get('unembedded')} stray_pending={st2.get('stray_pending')}")

    # ---- phase 5b: second birth pass. Everything (main stragglers + late) is
    # now embedded and parked, so any topic that missed the first birth gets a
    # fresh pass instead of rotting 'unassigned' in the buffer for 24h. ----
    t = time.monotonic()
    birth2 = run_event_birth_scheduled()
    timings["birth_2"] = time.monotonic() - t
    print(f"  birth pass 2: merged_hubs={birth2.get('merged_hubs')} in {timings['birth_2']:.1f}s")

    # ---- phase 6: merge reconciliation (idempotent) + final sweep ----
    from post_clustering_pipeline.tasks import run_hub_merge_scheduled
    t = time.monotonic()
    mg = run_hub_merge_scheduled()
    timings["merge_2"] = time.monotonic() - t
    print(f"  merge reconciliation 2: {mg}")
    t = time.monotonic()
    fs2 = fold_hub_reposts_scheduled()
    timings["fold_sweep_2"] = time.monotonic() - t
    print(f"  fold sweep 2: {fs2}")

    # ---- phase 7: audit ----
    t = time.monotonic()
    rc = _db_audit(corpus, timings)
    timings["audit"] = time.monotonic() - t
    return rc


def main() -> int:
    args = sys.argv[1:]
    if "--validate" in args:
        return validate()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())