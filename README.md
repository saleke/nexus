# Nexus

> **Real-Time Event Clustering & Intelligence Platform**

Nexus is a high-throughput, fault-tolerant streaming and batch machine learning pipeline designed to ingest high-volume social media streams, generate dense semantic representations, dynamically assign posts to evolving **Event Hubs** (thematic clusters), discover emerging stories via temporal graph community detection, and adapt continuously via active learning feedback loops.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Quickstart Guide](#quickstart-guide)
  - [Docker Compose Deployment (Recommended)](#docker-compose-deployment-recommended)
  - [Local Manual Setup](#local-manual-setup)
- [Configuration Reference](#configuration-reference)
- [API & Integration Specification](#api--integration-specification)
  - [1. Ingestion Endpoints](#1-ingestion-endpoints)
  - [2. Host Event Delivery (Transactional Outbox)](#2-host-event-delivery-transactional-outbox)
  - [3. Human-in-the-Loop Curation & Moderation](#3-human-in-the-loop-curation--moderation)
  - [4. Event Hub Exploration Endpoints](#4-event-hub-exploration-endpoints)
  - [5. Health & Observability](#5-health--observability)
- [Background Jobs & Active Learning](#background-jobs--active-learning)
- [Automated Testing & Benchmarks](#automated-testing--benchmarks)
- [Production Deployment & Operational Best Practices](#production-deployment--operational-best-practices)

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph ClientLayer [Client & Host Systems]
        Client[Host / Ingestion Source]
        Consumer[Event Outbox Consumer]
    end

    subgraph IngestionGateway [Nexus API Gateway :8000]
        API[FastAPI Gateway]
        Pool[Threaded Connection Pool]
        OutboxAPI[Transactional Outbox Service]
    end

    subgraph QueueBroker [Broker & Message Queue]
        RedisQueue[(Redis Queue / Broker)]
    end

    subgraph IngestionWorker [Celery Worker Cluster]
        Worker[Ingestion Worker]
        NLP[Spacy Entity & Noise Filter]
        ModelEngine[Hugging Face MiniLM-L6-v2 + LoRA]
        CentroidCalc[O(1) Bounded Incremental Centroids]
    end

    subgraph StorageLayer [PostgreSQL 16 + pgvector]
        DB[(PostgreSQL 16)]
        PostsTable[posts: HNSW Cosine Index]
        HubsTable[event_hubs: HNSW Cosine Index]
        OutboxTable[integration_outbox: Lease-Locked DLQ]
    end

    subgraph BatchEngine [Batch Intelligence & Active Learning]
        BirthJob[jobs.event_birth: PyTorch Top-K + Louvain Community Detection]
        WeeklyJob[jobs.weekly_threshold: Bayesian GP Threshold Optimization]
        MonthlyJob[jobs.monthly_learner: Contrastive LoRA Fine-Tuning]
        PromoteJob[jobs.promote_model: Production Model Gating]
    end

    Client -->|POST /posts & /posts/batch| API
    API -->|Checkout Connection| Pool
    Pool --> DB
    API -->|Enqueue Task| RedisQueue
    RedisQueue --> Worker
    Worker --> NLP
    NLP --> ModelEngine
    ModelEngine --> CentroidCalc
    CentroidCalc -->|Atomic CAS & Bounded Update| DB

    DB --> BirthJob
    BirthJob -->|Birth New Hubs & Outbox Events| DB

    DB --> OutboxAPI
    Consumer <-->|Pull Leased Events + Ack| OutboxAPI

    DB --> WeeklyJob
    DB --> MonthlyJob
    MonthlyJob --> PromoteJob
```

---

## Key Features

- **Vectorized Inference & Batch Ingestion**: Supports single-item (`POST /posts`) and batch (`POST /posts/batch`) ingestion with multi-item PyTorch forward passes and $L_2$ normalization.
- **$O(1)$ Bounded Incremental Centroids**: Eliminates full-table `AVG(embedding)` scans. Maintains centroids in $O(1)$ time with a bounded rolling window ($N_{\max} = 150$) that prevents semantic fossilization on viral events.
- **Drift-Proof Event Discovery**: Clusters unassigned buffered posts using vectorized PyTorch top-$k$ temporal graphs ($O(N \cdot k)$ memory) and Louvain community detection with a calibrated assignment threshold ($\ge 0.78$).
- **Transactional Outbox with Dead-Letter Queue (DLQ)**: Provides host integration with `FOR UPDATE SKIP LOCKED` lease locking, consumer acknowledgment, and automatic DLQ transitions for poisoned events exceeding retry limits.
- **Self-Healing Concurrency**: Employs atomic Compare-And-Swap (CAS) claiming (`assignment_status = 'processing'`) and an automated reconciliation sweeper (`reconcile_pending_posts`) to eliminate orphaned zombie posts.
- **Process-Safe Connection Pooling**: Transparently detects Celery process forks and instantiates isolated thread pools, preventing PostgreSQL socket corruption across worker subprocesses.
- **Continuous Learning Loop**: Automatically optimizes similarity thresholds weekly using Bayesian Gaussian Process optimization (`skopt.gp_minimize`) and fine-tunes LoRA adapters monthly via contrastive triplet loss on user feedback.

## Product Contract: One-Click Subject Views

Nexus is intentionally selective. It is designed to retain posts that contribute to a meaningful topic, idea, discussion, debate, or real-world event, rather than treating every social post as clusterable content. Casual chatter, selfies, diary-style updates, and other low-information posts are admitted to the pipeline only long enough to be classified as noise and excluded from event hubs.

For each coherent subject, Nexus maintains an **Event Hub** as the canonical destination. The host application does not need to invent a heading, write a summary, or manually assemble related posts. Once a post is assigned, the host can attach the hub URL supplied by the integration event to that post. Opening the link loads the complete subject view in one click: the catalyst post, chronological development, key perspectives, evidence/media, and participating voices. If hubs are later merged, old links resolve through the canonical hub redirect.

## Package Map

The import root is `post_clustering_pipeline`; commands should be run with `python -m post_clustering_pipeline...` or through the declared `nexus-api` entry point.

| Path | Responsibility |
| --- | --- |
| `api.py` | FastAPI ingestion, feedback, integration-outbox, and one-click hub-view endpoints |
| `tasks.py` | Celery ingestion, assignment, reconciliation, scheduled dispatch, and maintenance tasks |
| `nlp.py` | Worth-seeing discourse gate and text cleansing |
| `models.py` | Transformer/LoRA embedding engine; `models/lora_adapter/` contains adapter artifacts only |
| `db.py`, `schema.sql` | Process-safe PostgreSQL pooling and pgvector schema |
| `events.py`, `dispatch.py` | Transactional outbox creation and webhook delivery with retry/DLQ handling |
| `jobs/` | Event birth, hub merging, threshold optimization, learner, and model-promotion jobs |
| `evaluation/` | End-to-end simulation and benchmark utilities |
| `tools/` | Database and model inspection commands |
| `utils/` | Streaming dataset and graph-loading helpers |

---

## Prerequisites

- **Python**: `>= 3.10`
- **PostgreSQL**: with the **[pgvector](https://github.com/pgvector/pgvector)** extension
- **Redis**: `>= 6.2`
- **Docker & Docker Compose** (optional for containerized deployment)

---

## Quickstart Guide

### Docker Compose Deployment (Recommended)

The fastest way to deploy Nexus in an isolated environment is using Docker Compose:

```bash
git clone https://github.com/your-org/nexus.git
cd nexus

# Build and start all services in dependency order
docker compose up --build -d
```

This starts:
1. `postgres`: PostgreSQL 16 with `pgvector` enabled on port `5432`.
2. `redis`: Redis 7 on port `6379`.
3. `migrate`: Automatically initializes `schema.sql`.
4. `api`: Nexus FastAPI gateway running on `http://localhost:8000`.
5. `worker`: Celery worker processing the streaming ingestion pipeline.

To verify service status:
```bash
docker compose ps
curl http://localhost:8000/health/live
```

---

### Local Manual Setup

#### 1. Prepare Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python -m spacy download en_core_web_sm
```

#### 2. Initialize PostgreSQL & pgvector
Ensure PostgreSQL and Redis are running locally, then initialize the database schema:
```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/clustering_db"
export REDIS_URL="redis://localhost:6379/0"

psql "$DATABASE_URL" -f post_clustering_pipeline/schema.sql
```

#### 3. Run Celery Worker
```bash
celery -A post_clustering_pipeline.queues.celery_app worker --loglevel=INFO --pool=threads --concurrency=4
```

#### 4. Run API Gateway
```bash
python -m post_clustering_pipeline
# Or directly with uvicorn:
uvicorn post_clustering_pipeline.api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000` with interactive Swagger docs at `http://localhost:8000/docs`.

---

## Configuration Reference

All settings can be customized via environment variables:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/clustering_db` | PostgreSQL connection string with pgvector. |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker and backend connection URL. |
| `BASE_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face base transformer embedding model. |
| `LORA_ADAPTER_DIR` | `<package_dir>/models/lora_adapter` | Absolute path to active fine-tuned PEFT LoRA adapter. |
| `AUTO_ASSIGN_THRESHOLD` | `0.88` | Cosine similarity threshold for automatic event hub assignment. |
| `CANDIDATE_THRESHOLD` | `0.75` | Floor similarity for tagging posts as `candidate` vs `unassigned`. |
| `CENTROID_UPDATE_THRESHOLD` | `0.90` | Minimum similarity required to trigger an incremental centroid update. |
| `SIMILARITY_MARGIN` | `0.05` | Required margin between top match and runner-up to prevent ambiguity. |
| `BIRTH_ASSIGN_THRESHOLD` | `0.82` | Fallback threshold for unbuffered posts matching active hubs in batch. |
| `CENTROID_MAX_MEMBERS` | `150` | Maximum effective $n$ for bounded incremental centroid moving averages. |
| `BATCH_SIZE` | `32` | Batch size for vectorized PyTorch tensor encoding. |
| `OUTBOX_MAX_ATTEMPTS` | `5` | Delivery retry limit before moving an outbox event to the Dead-Letter Queue. |
| `SWEEPER_INTERVAL_SECONDS` | `60` | Execution frequency for the zombie post reconciliation worker. |

---

## API & Integration Specification

### 1. Ingestion Endpoints

#### Single Post Ingestion
`POST /posts`

```bash
curl -X POST http://localhost:8000/posts \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 101,
    "content": "Massive hurricane warning issued along the coast of Florida today as Category 4 storm approaches.",
    "has_media": false,
    "platform": "twitter",
    "source_id": "weather_feed",
    "external_post_id": "tweet_1829381923"
  }'
```
**Response (`201 Created`):**
```json
{
  "status": "queued",
  "post_id": 1
}
```

#### High-Throughput Batch Ingestion
`POST /posts/batch`

```bash
curl -X POST http://localhost:8000/posts/batch \
  -H "Content-Type: application/json" \
  -d '{
    "posts": [
      {
        "user_id": 101,
        "content": "SpaceX Starship rocket completes second stage orbital burn over Texas.",
        "platform": "twitter",
        "source_id": "space_feed",
        "external_post_id": "post_1001"
      },
      {
        "user_id": 102,
        "content": "NASA congratulates commercial space partner on historic rocket launch achievement.",
        "platform": "twitter",
        "source_id": "space_feed",
        "external_post_id": "post_1002"
      }
    ]
  }'
```
**Response (`201 Created`):**
```json
{
  "status": "queued",
  "post_ids": [2, 3]
}
```

---

### 2. Host Event Delivery (Transactional Outbox)

Nexus uses an outbox pattern with lease-locking (`FOR UPDATE SKIP LOCKED`) so host systems can reliably consume downstream clustering events without message loss.

#### Pulling Available Events
`GET /integration/events?limit=50&consumer=host-worker-1`

**Response (`200 OK`):**
```json
[
  {
    "id": 10,
    "event_key": "10",
    "schema_version": 1,
    "event_type": "post.assigned",
    "post_id": 1,
    "event_id": 4,
    "payload": {
      "post_id": 1,
      "event_id": 4,
      "confidence": 0.9124,
      "status": "assigned"
    },
    "attempts": 1,
    "lease_token": "a8f3b2c1d4e5",
    "created_at": "2026-09-12T14:30:00Z"
  }
]
```

#### Acknowledging Event Processing
`POST /integration/events/{id}/ack?lease_token={lease_token}`

```bash
curl -X POST "http://localhost:8000/integration/events/10/ack?lease_token=a8f3b2c1d4e5"
```
**Response (`200 OK`):**
```json
{
  "id": 10,
  "event_key": "10",
  "delivery_status": "delivered"
}
```
*Note: If the consumer crashes or does not acknowledge within 2 minutes, the lease expires and the event becomes eligible for delivery to another consumer. If an event fails 5 times, it is automatically routed to the Dead-Letter Queue (`delivery_status = 'failed'`).*

---

### 3. Human-in-the-Loop Curation & Moderation

#### Unlink / Detach Post
`POST /posts/unlink`
Removes an incorrectly assigned post, records a `user_removed` log for active learning, and resets status to `candidate`.
```json
{
  "post_id": 1,
  "event_id": 4
}
```

#### Confirm / Reassign Post
`POST /posts/confirm`
Explicitly binds a post to an event hub, records a `user_confirmed` log, and emits a `post.confirmed` outbox event.
```json
{
  "post_id": 1,
  "event_id": 4,
  "actor": "curator_jane"
}
```

#### Soft-Delete Post
`DELETE /posts/{id}`
Tombstones the post (`deleted_at = NOW()`), marks status as `noise`, and emits `post.deleted`.

---

### 4. Event Hub Exploration Endpoints

| Method & Endpoint | Description |
| :--- | :--- |
| `GET /posts/{id}/status` | Get assignment status (`assigned`, `candidate`, `pending`, `noise`), confidence, and hub ID. |
| `GET /events/{id}/anchor` | Retrieve the earliest anchor post that established the event hub. |
| `GET /events/{id}/timeline` | Retrieve all posts in the event hub chronologically (`created_at ASC`). |
| `GET /events/{id}/top` | Retrieve posts sorted by engagement / virality score (`engagement_score DESC`). |
| `GET /events/{id}/media` | Filter posts within the hub that contain media attachments (`has_media = TRUE`). |
| `GET /hubs/{id}/view?limit=50&offset=0` | Unified bounded hub payload; `limit` is capped at 200 and the response includes pagination metadata. |

---

### 5. Health & Observability

- **Liveness Probe**: `GET /health/live` $\rightarrow$ Returns `{"status": "ok"}`.
- **Readiness & DLQ Probe**: `GET /health/ready` $\rightarrow$ Checks database connectivity and reports Dead-Letter Queue poisoned events:
  ```json
  {
    "status": "ready",
    "dlq_failed_events": 0
  }
  ```

---

## Background Jobs & Active Learning

Nexus includes automated jobs to discover new events, tune thresholds, and fine-tune machine learning adapters:

### 1. Temporal Event Birth (`jobs.event_birth`)
Runs on an hourly schedule to partition unclustered buffered posts into 1-hour temporal slices, matches active centroids ($\ge 0.78$), builds a temporal $k$-NN graph using vectorized PyTorch matrix multiplications, and discovers emerging events via Louvain community detection:
```bash
python -m post_clustering_pipeline.jobs.event_birth
```

### 2. Weekly Bayesian Threshold Optimization (`jobs.weekly_threshold`)
Optimizes the runtime `global_similarity_threshold` using Gaussian Process minimization (`gp_minimize`) over the last 7 days of user feedback, penalizing false positives 5:1 against false negatives:
```bash
python -m post_clustering_pipeline.jobs.weekly_threshold
```

### 3. Monthly LoRA Contrastive Learner (`jobs.monthly_learner`)
Fine-tunes the PEFT LoRA adapter on human-in-the-loop unlinking feedback (`user_removed`) using raw text sequence tokenization and Triplet Margin Loss:
```bash
python -m post_clustering_pipeline.jobs.monthly_learner --epochs 2 --batch-size 8 --output-dir ./models/candidate_adapter
```

### 4. Model Quality Gate & Promotion (`jobs.promote_model`)
Ensures a candidate model satisfies strict precision, recall, and false-positive boundaries before activating it in `model_registry`:
```bash
python -m post_clustering_pipeline.jobs.promote_model \
  embedding-v2 \
  ./models/candidate_adapter \
  --precision 0.96 \
  --recall 0.88 \
  --noise-fp 0.01
```

### 5. Model Inspection Probe (`tools.inspect_model`)
Validates model weights, confirms parameter freezing, and checks embedding $L_2$ norm consistency:
```bash
python -m post_clustering_pipeline.tools.inspect_model
```

---

## Automated Testing & Benchmarks

### Running the Test Suite
Nexus includes a comprehensive `pytest` suite testing NLP cleansing, batch tensor embeddings, parameter freezing, bounded centroid arithmetic, temporal data slicing, and graph clustering:

```bash
pytest tests/ -v
```

### Running the Full Pipeline Simulation
A realistic end-to-end benchmark simulating ingestion of 20 multi-topic posts, Celery processing, event birth clustering, and feedback cycles:

```bash
python -m post_clustering_pipeline.evaluation.simulate_full_pipeline
```

---

## Production Deployment & Operational Best Practices

1. **Celery Worker Execution**: Run Celery using `--pool=threads` or `--pool=solo` rather than `prefork` to avoid memory duplication and CUDA fork hazards:
   ```bash
   celery -A post_clustering_pipeline.queues.celery_app worker --loglevel=INFO --pool=threads --concurrency=4
   ```
2. **PostgreSQL Connection Scaling**: Use an external connection pooler such as **PgBouncer** in `transaction` mode to multiplex thousands of client threads across 20–50 PostgreSQL backends.
3. **Automated Sweeper Cron**: Schedule the `reconcile_pending_posts` Celery task or a periodic worker every 60 seconds to guarantee no post remains orphaned due to network blips.
4. **Outbox Maintenance**: Schedule a weekly cron job to purge acknowledged outbox events older than 7 days:
   ```sql
   DELETE FROM integration_outbox WHERE delivery_status = 'delivered' AND delivered_at < NOW() - INTERVAL '7 days';
   ```
