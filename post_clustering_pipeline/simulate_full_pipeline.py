import os
import sys
import time
import requests
import psycopg2

API_URL = "http://localhost:8000"
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/clustering_db")

POST_DATA = [
    {"user_id": 101, "content": "Massive AWS cloud outage reported in us-east-1 region today affecting major web services."},
    {"user_id": 102, "content": "AWS server crash is downing multiple popular mobile applications and banking apps globally."},
    {"user_id": 103, "content": "Amazon Web Services confirms network degradation in Northern Virginia datacenter facilities."},
    {"user_id": 104, "content": "Engineers are investigating elevated error rates across S3 and DynamoDB in us-east-1."},
    {"user_id": 105, "content": "Cloudflare and AWS services reporting simultaneous connectivity issues in East Coast datacenters."},
    {"user_id": 201, "content": "FC Barcelona secured a dramatic 3-2 victory over Real Madrid in extra time during El Clasico."},
    {"user_id": 202, "content": "Unbelievable last-minute goal sends Barcelona to Champions League finals past Real Madrid."},
    {"user_id": 203, "content": "Real Madrid eliminated from European tournament after heartbreak loss in Madrid tonight."},
    {"user_id": 204, "content": "Match recap: tactical breakdown of Barcelona 3-2 victory over Madrid in thriller."},
    {"user_id": 205, "content": "Star forward scores hat-trick to seal Champions League semi-final victory for Barcelona."},
    {"user_id": 301, "content": "SpaceX Starship successfully completes orbital insertion and controlled water splashdown."},
    {"user_id": 302, "content": "Historic rocket launch today as Starship reaches space orbit from Starbase Texas platform."},
    {"user_id": 303, "content": "NASA congratulates commercial space partner on successful deep space vehicle flight test."},
    {"user_id": 304, "content": "Super Heavy booster landed safely back on launch pad chopstick arms in engineering marvel."},
    {"user_id": 305, "content": "Live telemetry confirms space capsule achieved target altitude during second stage burn."},
    {"user_id": 401, "content": "Federal Reserve cuts benchmark interest rates by 50 basis points citing cooling inflation metrics."},
    {"user_id": 402, "content": "Stock market rallies sharply following Central Bank rate reduction announcement this afternoon."},
    {"user_id": 403, "content": "Fed Chairman signals further monetary policy easing as inflation drops near target 2 percent."},
    {"user_id": 404, "content": "Mortgage rates expected to fall after central bank slashes short-term borrowing costs."},
    {"user_id": 405, "content": "Tech stocks lead broad market surge following Federal Reserve monetary policy pivot."}
]

def reset_database():
    """Clears all accumulated data before running the pipeline simulation."""
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("TRUNCATE posts, event_hubs, unclustered_posts_buffer RESTART IDENTITY CASCADE;")
        conn.commit()
        cur.close()
        conn.close()
        print("[✓] Database reset complete.")
    except Exception as e:
        print(f"[!] Database reset failed: {e}")

def print_clustering_results():
    """Prints clustered events and unclustered noise posts from the database."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    print("\n" + "=" * 70)
    print(" CLUSTERING RESULTS SUMMARY")
    print("=" * 70)

    cur.execute("""
        SELECT event_id, COUNT(*) 
        FROM posts 
        WHERE event_id IS NOT NULL 
        GROUP BY event_id 
        ORDER BY event_id ASC;
    """)
    clusters = cur.fetchall()

    if clusters:
        print(f"\n[✓] {len(clusters)} Event Hub(s) Created:\n")
        for event_id, count in clusters:
            print(f"--- Event Hub #{event_id} ({count} posts) ---")
            cur.execute("SELECT id, content FROM posts WHERE event_id = %s ORDER BY id ASC;", (event_id,))
            for post_id, content in cur.fetchall():
                print(f"  • [Post #{post_id}] {content}")
    else:
        print("\n[!] No Event Hubs were formed.")

    cur.execute("SELECT id, content FROM posts WHERE event_id IS NULL ORDER BY id ASC;")
    unclustered = cur.fetchall()

    print(f"\n[!] Unclustered Posts / Noise ({len(unclustered)} posts):")
    for post_id, content in unclustered:
        print(f"  • [Post #{post_id}] {content}")

    print("=" * 70 + "\n")
    cur.close()
    conn.close()

def main():
    # 0. Wipe database before each simulation run
    reset_database()

    print("\n--- STEP 1: Ingesting 20 Posts Across 4 Topics ---")
    for idx, post in enumerate(POST_DATA, start=1):
        res = requests.post(f"{API_URL}/posts", json=post)
        print(f"[{idx:02d}/20] Status: {res.status_code} | Res: {res.json()}")

    print("\n--- STEP 2: Waiting 6 seconds for Celery processing ---")
    time.sleep(6)

    print("\n--- STEP 3: Executing DBSCAN Event Birth Cron ---")
    os.system("PYTHONPATH=. python cron_event_birth.py")

    # Display clustering results after birth cron completes
    print_clustering_results()

    print("\n--- STEP 4: Simulating Unlink Feedback ---")
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT id, event_id FROM posts WHERE event_id IS NOT NULL LIMIT 1;")
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            post_id, event_id = row[0], row[1]
            print(f"Targeting active Post #{post_id} linked to Event Hub #{event_id}...")
            res = requests.post(f"{API_URL}/posts/unlink", json={"post_id": post_id, "event_id": event_id})
            print(f"Unlink Status: {res.status_code} | Res: {res.json()}")
        else:
            print("[!] Skipping unlink: No posts were assigned to an event hub during Step 3.")
    except Exception as e:
        print(f"[!] Database lookup failed during Step 4: {e}")

    print("\n--- STEP 5: Executing Monthly LoRA Training Cycle & Hot-Swap ---")
    os.system("PYTHONPATH=. python cron_monthly_learner.py")

if __name__ == "__main__":
    main()