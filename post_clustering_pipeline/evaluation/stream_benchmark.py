"""Stream messy, labeled social posts over time and report live clustering metrics."""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter

import requests

from .db import get_db_connection
from .batch_benchmark import build_posts
from ..jobs.event_birth import run_clustering_pipeline


def metrics(labels: dict[int, str | None]) -> dict[str, float | int]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, event_id FROM posts")
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) AS n FROM event_hubs")
            hubs = cur.fetchone()["n"]
    finally:
        conn.close()
    assigned = [(r["id"], r["event_id"]) for r in rows if r["event_id"] is not None]
    true_assigned = sum(labels.get(pid) is not None for pid, _ in assigned)
    noise_assigned = len(assigned) - true_assigned
    expected = sum(label is not None for label in labels.values())
    return {"posts": len(rows), "assigned": len(assigned), "hubs": hubs,
            "recall": true_assigned / max(expected, 1),
            "precision": true_assigned / max(len(assigned), 1),
            "noise_assigned": noise_assigned}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--events", type=int, default=5)
    p.add_argument("--posts-per-event", type=int, default=20)
    p.add_argument("--noise", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260912)
    args = p.parse_args()
    posts = build_posts(args.events, args.posts_per_event, args.noise, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(posts)
    labels: dict[int, str | None] = {}
    started = time.monotonic()
    sent = 0
    next_report = started
    while sent < len(posts) and time.monotonic() - started < args.duration:
        for index, post in enumerate(posts[sent:sent + args.batch_size], start=sent):
            response = requests.post(f"{args.api}/posts", json={"user_id": post.user_id, "content": post.content, "platform": "benchmark", "source_id": f"stream-{args.seed}", "external_post_id": f"post-{index}"}, timeout=30)
            response.raise_for_status()
            labels[int(response.json()["post_id"])] = post.label
        sent += args.batch_size
        run_clustering_pipeline()
        now = time.monotonic()
        if now >= next_report:
            m = metrics(labels)
            print(f"t={now-started:.1f}s sent={sent} posts={m['posts']} hubs={m['hubs']} assigned={m['assigned']} recall={m['recall']:.1%} precision={m['precision']:.1%} noise_assigned={m['noise_assigned']}", flush=True)
            next_report = now + max(5.0, args.interval * 5)
        time.sleep(args.interval)
    time.sleep(2)
    run_clustering_pipeline()
    m = metrics(labels)
    print(f"FINAL sent={sent} posts={m['posts']} hubs={m['hubs']} assigned={m['assigned']} recall={m['recall']:.1%} precision={m['precision']:.1%} noise_assigned={m['noise_assigned']}", flush=True)


if __name__ == "__main__":
    main()
