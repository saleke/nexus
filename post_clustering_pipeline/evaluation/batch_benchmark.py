"""Generate and run a realistic synthetic social-post clustering benchmark.

Requires the API, Celery worker, Redis, PostgreSQL, and pgvector to be running.
Run with: ``python -m post_clustering_pipeline.evaluation.batch_benchmark``.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import requests

from ..config import DATABASE_URL
from ..db import get_db_connection


# Realistic, ambiguity-rich corpus. Events are ordered so the first five span
# five unrelated domains (easy baseline) and the second five are deliberate
# rivals of the first five: same domain, overlapping vocabulary, and some posts
# explicitly compare the two actors. A correct clusterer must keep the rivals
# apart while attaching entity-less reports to the right event.
EVENTS = [
    {
        "label": "apple-iphone17",
        "posts": [
            "Apple just wrapped its iPhone 17 keynote in Cupertino and the titanium redesign stole the show.",
            "The iPhone 17 Pro finally got the periscope camera everyone kept asking for.",
            "Watching the Apple event live and the new A19 chip benchmarks look unreal.",
            "Apple says the iPhone 17 ships September 26, with preorders opening Friday.",
            "That iPhone 17 price bump is going to hurt with inflation where it is.",
            "The satellite messaging demo on the iPhone 17 was the highlight for me.",
            "Honestly the iPhone 17 colors are kind of a downgrade this year.",
            "Apple stock popped right after the keynote and analysts are calling it a supercycle.",
            "The keynote ended with one more thing and a September ship date for the new flagship.",
            "Apple's iPhone 17 reveal is already being compared to Samsung's Galaxy S26 event next month.",
        ],
    },
    {
        "label": "fed-rate-decision",
        "posts": [
            "The Federal Reserve held rates steady but signaled two cuts before year end.",
            "Powell's press conference sent Treasury yields tumbling within minutes.",
            "Markets rallied hard after the Fed decision and the S&P closed at a record.",
            "The Fed's dot plot now implies a much softer path than in June.",
            "Rate cut odds jumped to 80 percent right after the FOMC statement dropped.",
            "Bond traders were positioned wrong and got steamrolled by the Fed today.",
            "The Fed minutes confirm they are worried about the labor market cooling.",
            "A surprise dovish pivot sent everything green in the last hour of trading.",
            "The two-year yield had its biggest one-day drop since the pandemic.",
        ],
    },
    {
        "label": "hurricane-iris",
        "posts": [
            "Hurricane Iris strengthened to a Category 4 overnight as it churns toward the Gulf Coast.",
            "Mandatory evacuation orders are up for low-lying parishes ahead of Hurricane Iris.",
            "Hurricane Iris is expected to make landfall near Tampa late Thursday.",
            "Power crews are staging inland before Hurricane Iris knocks the grid out.",
            "Storm surge from Hurricane Iris could reach 12 feet in some bays.",
            "The National Hurricane Center upgraded Iris again and this one is a monster.",
            "The governor declared a state of emergency as shelters opened across the coast.",
            "Gas stations along the interstate ran dry as the coast emptied out.",
        ],
    },
    {
        "label": "barca-champions-league",
        "posts": [
            "Barcelona stunned Atletico in the Champions League with a 90th minute winner.",
            "Lamine Yamal was unplayable as Barcelona advanced past Atletico.",
            "Camp Nou was absolutely electric for Barcelona's comeback win.",
            "Barcelona's high line nearly cost them again in Europe.",
            "That Barcelona red card changed the entire tie against Atletico.",
            "Barcelona fans are singing long into the night after that finish.",
            "The away end went silent when the winner went in off the post.",
            "Barcelona's win sets up a possible El Clasico semifinal against Real Madrid.",
        ],
    },
    {
        "label": "beyonce-tour",
        "posts": [
            "Beyonce opened the Renaissance tour stop with a surprise encore and the crowd lost it.",
            "Beyonce's stage production this tour is on another level entirely.",
            "A sold out Wembley crowd for Beyonce and every phone was up for the finale.",
            "Beyonce brought out a surprise guest halfway through and the internet exploded.",
            "The Renaissance visuals during Beyonce's set deserve their own award.",
            "Fans queued since dawn for Beyonce merch and it sold out by noon.",
            "The lights went down and the whole stadium knew the opener was coming.",
            "Fans are already comparing Beyonce's set to Taylor's stadium run this summer.",
        ],
    },
    {
        "label": "samsung-galaxys26",
        "posts": [
            "Samsung unveiled the Galaxy S26 in Seoul and the rollable concept got gasps.",
            "The Galaxy S26 Ultra zoom lens is apparently a massive leap over last year.",
            "Samsung's Galaxy S26 launch had the strangest stage design I have ever seen.",
            "Galaxy S26 preorders open next week with Korea first as usual.",
            "Those Galaxy S26 battery life claims look a little too good to be true.",
            "Samsung is going all in on on-device AI for the Galaxy S26.",
            "A Galaxy S26 teaser leaked a day before the launch, naturally.",
            "The Galaxy S26 hands-on units were already overheating at the demo booth.",
            "Samsung is timing the Galaxy S26 to undercut Apple's iPhone 17 hype.",
        ],
    },
    {
        "label": "ecb-rate-decision",
        "posts": [
            "The ECB cut its deposit rate by 25 basis points at today's meeting in Frankfurt.",
            "Lagarde pushed back hard on the idea of back-to-back cuts from the ECB.",
            "European bond yields slid after the ECB decision landed.",
            "The ECB's new staff projections show inflation undershooting target in 2027.",
            "Eurozone stocks jumped as the ECB opened the door to more easing.",
            "Traders now price three more ECB cuts by next spring.",
            "A cautious cut in Frankfurt lifted the DAX even as the euro slumped.",
            "The governing council was not unanimous and the hawks made that clear.",
        ],
    },
    {
        "label": "typhoon-kaito",
        "posts": [
            "Typhoon Kaito is bearing down on the Philippines with 160 mph sustained winds.",
            "Manila schools are closed as Typhoon Kaito nears landfall.",
            "Typhoon Kaito triggered landslides in the northern provinces overnight.",
            "Thousands evacuated as Typhoon Kaito strengthens in the Pacific.",
            "Typhoon Kaito is the strongest storm to hit the region in a decade.",
            "Airlines canceled hundreds of flights ahead of Typhoon Kaito.",
            "The archipelago braced as the outer rainbands arrived and ports shut down.",
            "Fishermen were ordered back to shore as the swell built offshore.",
        ],
    },
    {
        "label": "madrid-champions-league",
        "posts": [
            "Real Madrid edged Bayern in extra time to reach the Champions League semis.",
            "Bellingham's late header sent Real Madrid through at the Bernabeu.",
            "Real Madrid were second best for an hour and still found a way.",
            "That Real Madrid penalty shout was waved away and Bayern are furious.",
            "Madrid's bench depth is just unfair on these knockout nights.",
            "Carlo's halftime switch turned the tie for Real Madrid.",
            "The Bernabeu erupted when the fourth official signalled five added minutes.",
            "Real Madrid could meet Barcelona in the semis if both results hold.",
        ],
    },
    {
        "label": "taylor-tour",
        "posts": [
            "Taylor Swift's Eras tour added three more nights after tickets crashed the site.",
            "Taylor brought the whole stadium to tears with the acoustic surprise song.",
            "The Eras tour setlist somehow got longer for the European leg.",
            "Swifties camped overnight for Taylor's merch trucks in Lisbon.",
            "Taylor's Eras staging uses more LED panels than any tour this year.",
            "Taylor swapped in a vault track and the crowd screamed every word.",
            "The friendship bracelets filled entire bins by the exits.",
            "Swifties are arguing Taylor's tour outsold Beyonce's stadium run.",
        ],
    },
]

NOISE = [
    "I finally reorganized my kitchen shelves and found three missing batteries.",
    "Does anyone know a quiet cafe with reliable Wi-Fi near the train station?",
    "My dog refuses to walk unless it is carrying the blue tennis ball.",
    "The new update made my phone battery last all day, surprisingly.",
    "Looking for recommendations for a beginner-friendly pasta recipe.",
    "Traffic is slow, but at least the sunset over the bridge looks beautiful.",
    "Reminder to drink water and stretch if you have been sitting for hours.",
    "I bought a secondhand bookshelf and spent the afternoon fixing it up.",
    "The farmer's market had the most amazing peaches this weekend.",
    "My neighbor's cat has started sleeping on my balcony every afternoon.",
    "I can never remember which recycling bin takes glass jars.",
    "Tried a new board game last night and lost spectacularly three times.",
    "The bus was late again, so I walked the last few stops.",
    "Finally finished that mystery novel and guessed the ending wrong.",
    "My houseplants are somehow thriving despite my total neglect.",
    "There is a woodpecker somewhere in the tree outside my window.",
    "Learning to make sourdough has turned my kitchen into a flour cloud.",
    "The local library added a seed-swap shelf near the entrance.",
    "My headphones died halfway through the podcast on my commute.",
    "I keep finding glitter in the strangest places after the craft fair.",
    "The park has a new set of swings and the kids love them.",
    "Spent an hour picking out a birthday card and left with three.",
    "My bicycle chain came off twice on the way to the hardware store.",
    "The coffee shop across the street changed its roast and it is better now.",
    "A sudden rainstorm caught everyone at the outdoor cinema.",
    "I am slowly getting better at fixing hemmed trousers by hand.",
    "The office plant is technically alive, which is a small victory.",
    "Found a forgotten ten-dollar bill in an old coat pocket.",
    "The river path was crowded with runners this evening.",
    "My grandmother's dumpling recipe is impossible to follow exactly.",
    "The museum's new fossil exhibit opens next Thursday.",
    "I tried to paint the hallway and ended up painting the ceiling too.",
    "The neighborhood squirrel has figured out how to open the bird feeder.",
    "Watched a documentary about deep-sea creatures and could not sleep.",
    "My kettle makes a strange whistle right before it boils.",
    "The train window seat had the best view of the valley.",
    "Started a small balcony garden with herbs and one stubborn tomato.",
    "The bakery sold out of croissants before nine in the morning.",
    "I fixed the squeaky door hinge with a little oil and felt accomplished.",
    "My phone storage is full of photos of clouds and receipts.",
    "The community pool reopens after repairs at the end of the month.",
    "Learning to solve the cube again after forgetting every algorithm.",
    "A fox was trotting down the alley behind the grocery store.",
    "The old clock in the hallway gains two minutes every day.",
    "My friend brought back a tiny jar of sand from her beach trip.",
    "Spent the afternoon sorting photos into albums that will never be opened.",
    "The new bakery's cinnamon rolls are worth the early queue.",
    "I cannot keep track of which light switch controls the porch lamp.",
    "My umbrella turned inside out on the walk to the post office.",
    "The balcony tomato finally produced exactly one small tomato.",
    "We watched the meteor shower from the roof until it got too cold.",
    "The stray cat by the bin has adopted the whole street.",
    "My keyboard has one key that sticks no matter how often I clean it.",
    "The hiking trail had a rope bridge that wobbled more than expected.",
    "I am trying to learn three chords and failing cheerfully.",
    "The corner shop now stocks the hot sauce I liked on holiday.",
    "My winter coat has a pocket I only just discovered.",
    "The pond froze just enough for the ducks to look confused.",
    "Someone left a tiny painted rock on the park bench.",
    "The radio played the same song twice on the school run.",
    "I spent the morning repotting ferns and the afternoon sweeping up.",
]

NOISE_CONTEXT = [
    "before work", "after lunch", "this weekend", "on the bus", "at home",
    "during my break", "this morning", "near the park", "after class", "today",
]

# Hard negatives: personal/goofy posts that borrow each event's vocabulary but
# are not about the event. A good gate must keep these out of the event hubs.
HARD_NOISE = [
    "My phone battery dies so fast I feel like Apple owes me an apology keynote.",
    "The only rate cut I am getting is on my streaming subscription this month.",
    "My umbrella lost its fight with the storm on the walk to work.",
    "Watching the match of egos in my office meeting was the real sport today.",
    "I have had Beyonce stuck in my head all day and no tickets to show for it.",
    "My portfolio is a hurricane and I am just the guy holding a paper umbrella.",
    "That new phone update made my old device feel like a relic.",
    "The forecast says rain and my joints already knew about it.",
    "Our office cup final ended in a draw because nobody could score on the intern.",
    "I bought concert tickets and then remembered I hate crowds.",
    "My team lost the group chat argument and now there is a rematch clause.",
    "I keep getting preorder ads for phones I definitely cannot afford.",
]


@dataclass(frozen=True)
class SyntheticPost:
    user_id: int
    content: str
    label: str | None


# Surface-form jitter: retweets/reposts and social styling mean the same event
# is described in slightly different strings. Kept short so it never changes
# the meaning or the ground-truth label.
VARIATION = [
    "", "", "", "", " #breaking", " #news", " 😳", " via @localhost", " — details to follow",
    " (thread)", " imo", " honestly", " huge if true", " congrats to the team",
]


def build_posts(events: int, posts_per_event: int, noise: int, seed: int) -> list[SyntheticPost]:
    """Build a realistic, ambiguity-rich labeled corpus.

    Each event is a hand-written pool of natural posts about one story. Rival
    events share a domain (two phone launches, two central banks, two storms,
    two clubs, two tours) and some posts explicitly compare the rivals, so the
    embedding space alone cannot separate them - the identity entities must.
    Some event posts are entity-less (they describe the story without naming the
    actor) and must attach by meaning. Noise mixes unrelated posts with hard
    negatives that borrow event vocabulary.
    """
    rng = random.Random(seed)
    if events > len(EVENTS):
        raise ValueError(f"requested {events} events but only {len(EVENTS)} are defined")

    result: list[SyntheticPost] = []
    user_id = 10_000
    for event in EVENTS[:events]:
        pool = event["posts"]
        for _ in range(posts_per_event):
            content = rng.choice(pool) + rng.choice(VARIATION)
            result.append(SyntheticPost(user_id, content, event["label"]))
            user_id += 1

    # Always include the hard negatives; fill the rest with unrelated posts.
    if noise <= len(HARD_NOISE):
        selected = rng.sample(list(HARD_NOISE), noise)
    else:
        need = noise - len(HARD_NOISE)
        if need > len(NOISE):
            raise ValueError(
                f"requested {noise} noise posts but only {len(HARD_NOISE) + len(NOISE)} exist"
            )
        selected = list(HARD_NOISE) + rng.sample(list(NOISE), need)
    for base in selected:
        result.append(SyntheticPost(user_id, f"{base} {rng.choice(NOISE_CONTEXT)}", None))
        user_id += 1

    rng.shuffle(result)
    return result


def run(args: argparse.Namespace) -> None:
    posts = build_posts(args.events, args.posts_per_event, args.noise, args.seed)
    started = time.monotonic()
    accepted = 0
    labels_by_id: dict[int, str | None] = {}
    for index, post in enumerate(posts):
        response = requests.post(
            f"{args.api}/posts",
            json={"user_id": post.user_id, "content": post.content, "platform": "benchmark", "source_id": f"batch-{args.seed}", "external_post_id": f"post-{index}"},
            timeout=30,
        )
        response.raise_for_status()
        labels_by_id[int(response.json()["post_id"])] = post.label
        accepted += 1
    time.sleep(args.wait)

    from ..jobs.event_birth import run_clustering_pipeline
    from ..jobs.merge_hubs import reconcile_hub_merges_to_fixpoint

    # Mirror the production birth path exactly: event_birth births communities
    # and the same call folds template-level fragments into their event before
    # the next assignment wave (see tasks.run_event_birth_scheduled). Measuring
    # only run_clustering_pipeline() scores a transient state that never
    # survives a production cycle.
    run_clustering_pipeline()
    merged = 0 if args.no_reconcile else reconcile_hub_merges_to_fixpoint()
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

    hub_of: dict[int, int | None] = {}
    for row in assignments:
        post_id = row["id"] if isinstance(row, dict) else row[0]
        event_id = row["event_id"] if isinstance(row, dict) else row[1]
        hub_of[post_id] = event_id

    event_posts = [pid for pid, lab in labels_by_id.items() if lab is not None]
    noise_posts = [pid for pid, lab in labels_by_id.items() if lab is None]

    clustered_event_posts = sum(1 for pid in event_posts if hub_of.get(pid) is not None)
    clustered_noise = sum(1 for pid in noise_posts if hub_of.get(pid) is not None)

    # Pairwise clustering quality over ground-truth event posts. This is the
    # metric that actually penalises fragmentation and cross-event merges; the
    # post-level counters above cannot see either.
    tp = fp = fn = 0
    for i in range(len(event_posts)):
        a = event_posts[i]
        for j in range(i + 1, len(event_posts)):
            b = event_posts[j]
            same_label = labels_by_id[a] == labels_by_id[b]
            ha, hb = hub_of.get(a), hub_of.get(b)
            same_hub = ha is not None and ha == hb
            if same_label and same_hub:
                tp += 1
            elif same_hub and not same_label:
                fp += 1
            elif same_label and not same_hub:
                fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Per-event hub fragmentation: a correct clusterer yields one hub per label.
    hubs_by_label: dict[str, set] = {}
    for pid in event_posts:
        label = labels_by_id[pid]
        hubs_by_label.setdefault(label, set())
        hub = hub_of.get(pid)
        if hub is not None:
            hubs_by_label[label].add(hub)
    fragmented = sum(1 for hs in hubs_by_label.values() if len(hs) > 1)
    unclustered = sum(1 for pid in event_posts if hub_of.get(pid) is None)
    hubs_with_noise = sorted({hub_of[pid] for pid in noise_posts if hub_of.get(pid) is not None})

    print(f"accepted={accepted} total={total} event_hubs={hubs} merged={merged} elapsed_seconds={elapsed:.2f}")
    print(f"expected_events={len(hubs_by_label)} expected_event_posts={len(event_posts)}")
    print(f"clustered={clustered} ({clustered / max(total, 1):.1%}) "
          f"post_recall={clustered_event_posts / max(len(event_posts), 1):.1%} "
          f"post_precision={clustered_event_posts / max(clustered, 1):.1%}")
    print(f"pairwise_precision={precision:.3f} pairwise_recall={recall:.3f} pairwise_f1={f1:.3f} "
          f"(tp={tp} fp={fp} fn={fn})")
    print(f"fragmented_events={fragmented} unclustered_event_posts={unclustered} "
          f"noise_clustered={clustered_noise}/{len(noise_posts)} hubs_with_noise={hubs_with_noise}")
    print(f"hub_sizes={sorted(sizes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--posts-per-event", type=int, default=30)
    parser.add_argument("--noise", type=int, default=50)
    parser.add_argument("--wait", type=float, default=15, help="Seconds to wait for Celery ingestion")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--no-reconcile", action="store_true",
                        help="Skip hub reconciliation to inspect the raw post-birth state")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
