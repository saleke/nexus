"""Generate and run a realistic synthetic social-post clustering benchmark.

Requires the API, Celery worker, Redis, PostgreSQL, and pgvector to be running.
Run with: ``python -m post_clustering_pipeline.simulate_social_benchmark``.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import requests

from .config import DATABASE_URL
from .db import get_db_connection


TOPICS = {
    "storm": {
        "entities": ["Hurricane Iris", "Miami", "Florida", "NOAA", "Gulf Coast"],
        "details": ["evacuation orders", "flood warnings", "power outages", "shelters", "wind speeds"],
        "templates": [
            "{entity} is reporting {detail} as Hurricane Iris moves toward the coast.",
            "Residents near {entity} are sharing updates about {detail} tonight.",
            "Emergency crews in {entity} are preparing for {detail} after the latest advisory.",
        ],
    },
    "football": {
        "entities": ["Barcelona", "Real Madrid", "Camp Nou", "La Liga", "Lamine Yamal"],
        "details": ["a late winner", "a red card", "extra time", "the semifinal", "a 3-2 comeback"],
        "templates": [
            "{entity} supporters are celebrating {detail} in tonight's match.",
            "The {entity} match ended with {detail} after a dramatic second half.",
            "Analysts are debating {detail} from the {entity} game.",
        ],
    },
    "markets": {
        "entities": ["Federal Reserve", "Wall Street", "Nasdaq", "Treasury", "Washington"],
        "details": ["a rate cut", "cooling inflation", "bond yields", "market volatility", "new guidance"],
        "templates": [
            "{entity} reacted sharply to {detail} in today's trading session.",
            "Investors are discussing {detail} after the latest {entity} announcement.",
            "Economists expect {detail} to influence {entity} this week.",
        ],
    },
    "launch": {
        "entities": ["SpaceX", "Starship", "NASA", "Starbase", "Texas"],
        "details": ["a successful test flight", "a booster landing", "orbital insertion", "live telemetry", "a launch delay"],
        "templates": [
            "{entity} viewers are watching {detail} during today's mission.",
            "Engineers at {entity} confirmed {detail} during the latest launch attempt.",
            "The {entity} broadcast included updates about {detail}.",
        ],
    },
    "concert": {
        "entities": ["Beyonce", "Renaissance Tour", "London", "Wembley", "Grammy Awards"],
        "details": ["a surprise encore", "sold out tickets", "a new single", "the stage production", "fan reactions"],
        "templates": [
            "Fans in {entity} are posting clips of {detail}.",
            "The {entity} show featured {detail} and drew a huge crowd.",
            "Music reporters are praising {detail} from the {entity} performance.",
        ],
    },
}

NOISE = [
    "I finally reorganized my kitchen shelves and found three missing batteries.",
    "Does anyone know a quiet cafe with reliable Wi-Fi near the train station?",
    "My dog refuses to walk unless it is carrying the blue tennis ball.",
    "The new update made my phone battery last all day, surprisingly.",
    "Looking for recommendations for a beginner-friendly pasta recipe.",
    "Traffic is slow, but at least the sunset over the bridge looks beautiful.",
    "Reminder to drink water and stretch if you have been sitting for hours.",
    "I bought a secondhand bookshelf and spent the afternoon fixing it up.",
]

NOISE_CONTEXT = [
    "before work", "after lunch", "this weekend", "on the bus", "at home",
    "during my break", "this morning", "near the park", "after class", "today",
]


@dataclass(frozen=True)
class SyntheticPost:
    user_id: int
    content: str
    label: str | None


def build_posts(events: int, posts_per_event: int, noise: int, seed: int) -> list[SyntheticPost]:
    rng = random.Random(seed)
    topic_names = list(TOPICS)
    result: list[SyntheticPost] = []
    user_id = 10_000
    for index in range(events):
        topic = TOPICS[topic_names[index % len(topic_names)]]
        label = topic_names[index % len(topic_names)]
        for _ in range(posts_per_event):
            template = rng.choice(topic["templates"])
            content = template.format(entity=rng.choice(topic["entities"]), detail=rng.choice(topic["details"]))
            # Natural variation: hashtags, punctuation, and short context additions.
            if rng.random() < 0.35:
                content += f" #{label}"
            if rng.random() < 0.25:
                content += " Latest update from people on the ground."
            result.append(SyntheticPost(user_id, content, label))
            user_id += 1
    for _ in range(noise):
        content = f"{rng.choice(NOISE)} {rng.choice(NOISE_CONTEXT)}"
        result.append(SyntheticPost(user_id, content, None))
        user_id += 1
    rng.shuffle(result)
    return result


def reset_database() -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE posts, event_hubs, unclustered_posts_buffer RESTART IDENTITY CASCADE;")
        conn.commit()
    finally:
        conn.close()


def run(args: argparse.Namespace) -> None:
    posts = build_posts(args.events, args.posts_per_event, args.noise, args.seed)
    reset_database()
    started = time.monotonic()
    accepted = 0
    labels_by_id: dict[int, str | None] = {}
    for post in posts:
        response = requests.post(
            f"{args.api}/posts",
            json={"user_id": post.user_id, "content": post.content},
            timeout=30,
        )
        response.raise_for_status()
        labels_by_id[int(response.json()["post_id"])] = post.label
        accepted += 1
    time.sleep(args.wait)

    from .cron_event_birth import run_clustering_pipeline

    run_clustering_pipeline()
    elapsed = time.monotonic() - started

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total, COUNT(event_id) AS clustered FROM posts;")
            summary = cur.fetchone()
            total = summary["total"] if isinstance(summary, dict) else summary[0]
            clustered = summary["clustered"] if isinstance(summary, dict) else summary[1]
            cur.execute("SELECT COUNT(*) FROM event_hubs;")
            hub_row = cur.fetchone()
            hubs = hub_row["count"] if isinstance(hub_row, dict) else hub_row[0]
            cur.execute("SELECT event_id, COUNT(*) FROM posts WHERE event_id IS NOT NULL GROUP BY event_id;")
            sizes = [row["count"] if isinstance(row, dict) else row[1] for row in cur.fetchall()]
            cur.execute("SELECT id, event_id FROM posts;")
            assignments = cur.fetchall()
    finally:
        conn.close()

    expected_event_posts = args.events * args.posts_per_event
    clustered_event_posts = 0
    clustered_noise = 0
    for row in assignments:
        post_id = row["id"] if isinstance(row, dict) else row[0]
        event_id = row["event_id"] if isinstance(row, dict) else row[1]
        if event_id is None:
            continue
        if labels_by_id.get(post_id) is None:
            clustered_noise += 1
        else:
            clustered_event_posts += 1
    event_recall = clustered_event_posts / max(expected_event_posts, 1)
    assignment_precision = clustered_event_posts / max(clustered, 1)
    print(f"accepted={accepted} total={total} event_hubs={hubs} elapsed_seconds={elapsed:.2f}")
    print(f"clustered={clustered} ({clustered / max(total, 1):.1%}) expected_event_posts={expected_event_posts}")
    print(f"event_recall={event_recall:.1%} assignment_precision={assignment_precision:.1%}")
    print(f"noise_clustered={clustered_noise} hub_sizes={sorted(sizes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--posts-per-event", type=int, default=30)
    parser.add_argument("--noise", type=int, default=50)
    parser.add_argument("--wait", type=float, default=15, help="Seconds to wait for Celery ingestion")
    parser.add_argument("--seed", type=int, default=20260911)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
