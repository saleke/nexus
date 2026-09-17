# Nexus

> **Real-time event clustering & intelligence platform** — ingests social streams, learns dense semantic embeddings, assigns posts to evolving thematic **Event Hubs**, discovers emerging stories, and closes the loop with human feedback.

Nexus is a high-throughput, fault-tolerant pipeline built around PostgreSQL 16 + `pgvector`, Celery + Redis, and a FastAPI gateway. Its job is to turn a noisy social stream into a curated, queryable, evolvable map of *subjects* — events, debates, products, and stories — and to hand each one to the host application as a single, self-contained link.

## Table of Contents

- [What it Does](#what-it-does)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Control Plane (Admin Panel)](#control-plane-admin-panel)
- [Security](#security)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
  - [Docker Compose (recommended)](#docker-compose-recommended)
  - [Local manual setup](#local-manual-setup)
- [Configuration Reference](#configuration-reference)
- [API & Integration Specification](#api--integration-specification)
- [Background Jobs & Active Learning](#background-jobs--active-learning)
- [Automated Testing](#automated-testing)
- [Operations](#operations)
- [Production Hardening Checklist](#production-hardening-checklist)
- [Package Map](#package-map)

---

## What it Does

Nexus is intentionally selective. It retains posts that contribute to a meaningful subject and treats casual chatter, diary-style updates, and other low-information content as **noise**.

For every coherent subject it maintains an **Event Hub** — the canonical destination for a story or debate. The host application does not need to invent a headline, write a summary, or assemble related posts:

1. A post is ingested and assigned to (or births) an Event Hub.
2. The outbox emits a `post.assigned` event carrying the hub reference.
3. The host attaches the hub URL to the post. Opening it loads a complete **one-click subject view**: catalyst post, chronological timeline, key perspectives, evidence/media, and participating voices.
4. If hubs are later merged, historical links keep working through canonical-hub redirects.

---

## Architecture

```mermaid
flowchart TD
    subgraph ClientLayer [Client & Host Systems]
        Client[Host / Ingestion Source]
        Consumer[Event Outbox Consumer]
        Owner[Operator / Curator]
    end

    subgraph Gateway ["Nexus API Gateway :8000 (FastAPI)"]
        API[Data Plane API]
        Admin[Control Plane /admin]
        Pool[Threaded Connection Pool]
        OutboxAPI[Transactional Outbox Service]
    end

    subgraph Broker [Redis]
        Queue[(Celery broker / hint queue)]
        Ratelimit[(rate-limit counters)]
    end

    subgraph Workers [Celery worker + beat]
        Worker[Ingestion / assignment workers]
        NLP[Spacy entity & noise filter]
        Model[Embedding engine MiniLM-L6-v2 + LoRA]
        Centroid["O(1) bounded incremental centroids"]
        Beat[beat scheduler]
    end

    subgraph Storage [PostgreSQL 16 + pgvector]
        DB[(clustering_db)]
        Posts[posts - HNSW embedding index]
        Hubs[event_hubs - HNSW index]
        Outbox[integration_outbox - lease-locked DLQ]
        Audit[admin_audit_log / decision / feedback tables]
    end

    subgraph Jobs [Intelligence jobs]
        Birth["event_birth: temporal k-NN + Louvain"]
        Autotune["threshold autotune (daily)"]
        Learner["LoRA contrastive learner (monthly)"]
        Promote["model promotion gate"]
    end

    Client -->|POST /posts, /posts/batch<br>Bearer consumer token| API
    Owner -->|HTTPS browser session| Admin
    API --> Pool
    Pool --> DB
    API --> Queue
    Queue --> Worker
    Worker --> NLP --> Model --> Centroid
    Centroid -->|atomic CAS + bounded update| DB
    DB --> Birth --> DB
    DB --> OutboxAPI
    Consumer <-->|pull leased events + ack| OutboxAPI
    DB --> Autotune
    DB --> Learner
    Learner --> Promote
    Beat --> Worker
```

### Data plane vs. control plane

- **Data plane** (`/posts`, `/events`, `/hubs`, `/integration/*`): how a host platform feeds the pipeline and consumes cluster results. Authenticated with a revocable, rate-limited **consumer token**, or the shared `API_AUTH_TOKEN`.
- **Control plane** (`/admin/*`): the operator panel for curation, calibration, admin management, and observability. Authenticated with a short-lived JWT **session cookie** (password + optional TOTP, or Google sign-in).

---

## Key Features

- **High-throughput ingestion** — single-post `POST /posts` and bounded batch `POST /posts/batch` (max 500) with vectorized transformer inference and L2-normalized embeddings.
- **Incremental, bounded centroids** — centroid arithmetic is O(1) with a rolling window (`CENTROID_MAX_MEMBERS=150`), so viral events re-anchor without a full-table `AVG(embedding)` and without semantic fossilization.
- **Drift-proof event discovery** — buffered posts are partitioned into temporal slices, matched against active hubs, and clustered with vectorized top-k temporal graphs plus Louvain community detection.
- **Transactional outbox with DLQ** — `FOR UPDATE SKIP LOCKED` lease locking, consumer acknowledgment, automatic dead-lettering after `OUTBOX_MAX_ATTEMPTS`, and TTL-based garbage collection.
- **Self-healing concurrency** — atomic compare-and-swap claiming plus a reconciliation sweeper removes orphaned "zombie" posts; claim transactions drop `synchronous_commit` safely.
- **Human-in-the-loop learning** — every unlink/confirm decision is logged with the policy/model version in effect, rolled up, and replayed into daily threshold autotuning and monthly LoRA fine-tuning.
- **Entity-aware conflict resolution** — hub merges require similarity plus shared identity evidence; dominant-ORG board splits (e.g. Tesla vs SpaceX) are never force-merged.
- **Structured JSON logging** — API, worker, and beat emit one JSON object per line, with HTTP access lines carrying `path`/`status_code`.

---

## Control Plane (Admin Panel)

The operator panel lives under `/admin` and is served by `admin.py` + `admin_auth.py`. It is built with server-rendered Jinja2 templates, htmx, and a small self-hosted stylesheet.

| Section | Route | Purpose |
|---|---|---|
| Quality desk | `/admin/quality` | Review assignment quality, unlink/confirm posts, pick candidates |
| Decisions | `/admin/decisions` | Human + system decision log with row-level detail |
| Merge desk & log | `/admin/desk`, `/admin/merges` | Inspect and perform hub merges; reopen mistakes |
| Refs & hub graph | `/admin/refs`, `/admin/hub-graph`, `/admin/hub-detail` | Explore hubs and canonical references |
| Calibration | `/admin/calibration` | Tune thresholds, propose/apply/revert revisions |
| Audit | `/admin/audit` | Full admin action audit trail (`admin_audit_log`) |
| Ops | `/admin/ops` | Resume stuck posts, retry DLQ, manage consumer tokens |
| Settings | `/admin/settings` | Security (password/TOTP/username), admins, endpoints, env knobs |
| Network | `/admin/network` | 3-D force-graph visualization of hub relationships |

Admin accounts, invite links, TOTP enrollment, and Google sign-in are all managed from the panel itself. See [Security](#security) for the authentication model.

---

## Security

Security is a first-class concern: a platform that curates the feed is a high-value target. The posture below is enforced by default where it matters, with the production hardening checklist listed at the end.

### Authentication model

| Surface | Credential | Notes |
|---|---|---|
| Data plane | Per-consumer token **or** shared `API_AUTH_TOKEN` | Consumer tokens are revocable, rate-limited, and attributed to every feedback action. |
| Control plane | JWT session cookie (HS256, 12 h) | Issued only after email+password (and TOTP when enrolled) or verified Google sign-in. |
| Public | `/health/*`, `/docs`, login/register/forgot pages | Always available. |

- **Passwords** are salted **PBKDF2-SHA256 with 200k iterations** (`hash_password`/`verify_password`).
- **TOTP** is RFC 6238 (SHA-1, 6 digits, 30 s step) verified with a ±1-step window for Google-Authenticator compatibility.
- **Sessions** are short-lived JWTs signed by a secret persisted in `system_config` (survives restarts). Every request **re-checks the account's `is_active` flag against the database**, so disabling or deleting an admin kills their session immediately — nothing outlives its revocation.
- **Login throttling** — failed attempts are counted per account *and* per client IP over a sliding window (Redis-backed, in-process fallback), enforced by `ratelimit.py`.
- **TOTP at login** — once a second factor is enrolled it is *required* for every sign-in, with a separate rate-limit bucket that cannot be used to lock out the account.
- **Re-authentication for sensitive changes** — changing the password or username, and (re-)enrolling TOTP, requires the *current* password, plus a valid authenticator code when TOTP is enabled. A hijacked session cannot silently swap in an attacker-controlled secret.
- **Google OAuth** validates the returned identity token's `aud` and `iss` claims and enforces `email_verified`.

### Fail-closed production mode

- `NEXUS_ENV=production` + an empty `API_AUTH_TOKEN` **refuses to boot** (startup guard raises). A production deployment cannot accidentally serve unauthenticated traffic.
- In production every non-public request must present a valid credential; the middleware never short-circuits. Development keeps the explicit "dev-open" behavior only when `NEXUS_ENV` is not `production`.

### Application-layer defenses

- **CSRF** — in production, state-changing `/admin` requests authorized by a browser session must carry a matching `Origin` (same host as `Host`, or a `TRUSTED_ORIGINS` entry). Cross-site form posts without an Origin are rejected with 403.
- **Security headers** on every response: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` (plus `frame-ancestors 'none'` via CSP), `Referrer-Policy: no-referrer`, `Strict-Transport-Security` (production), and a `Content-Security-Policy` that allows only self-hosted assets + the pinned three.js CDN with **SRI**.
- **Host validation** — `TRUSTED_HOSTS` (comma-separated) is enforced by Starlette's TrustedHostMiddleware when set, so a crafted `Host` header cannot poison redirects or Origin decisions.
- **Invite links are never logged.** The self-service invite URL is returned in the HTTP response only; it is not written to any file or structured log.
- **All SQL is parameterized**; dynamic `WHERE` builders in admin/audit views are whitelist-only f-string composition over fixed column maps.
- **Jinja2 autoescaping** is on, and templates never use `|safe`/`Markup`/`eval`.

### Secrets hygiene

- `.env` (loaded by `config.py`) is gitignored; compose reads it via `env_file`.
- Docker Compose binds PostgreSQL and Redis to `127.0.0.1` only, and interpolates `POSTGRES_PASSWORD` instead of hardcoding credentials.
- `scripts/backup.sh` prints a **redacted** connection string (password masked).
- The settings "Environment" tab renders secret-valued knobs as `••••••••`, never their value.

### Documented limitations

These are deliberate or incremental and should inform operational decisions:

- The 47 `/admin` routes are `async def` handlers doing blocking db calls on the event loop (fine for single operators; move to `def` if panel concurrency ever matters).
- Consumer-token hashes are SHA-256, not PBKDF2 (callers are pre-authorized trusted host services; the token is high-entropy). Password reset tokens are single-purpose JWTs with a 15-minute TTL.
- `/docs`, `/openapi.json`, and `/redoc` are public in all environments. Restrict at the ingress if your audit policy requires it.
- Redis is unauthenticated by default (localhost-bound in compose). Put it behind a firewall or enable `requirepass` in hostile environments.
- Session cookies use the `Secure` flag in production but are not `__Host-`-prefixed; adopt the prefix in a future release (requires a cookie-name change that invalidates sessions).

---

## Prerequisites

- **Python** ≥ 3.10
- **PostgreSQL** with the **[pgvector](https://github.com/pgvector/pgvector)** extension
- **Redis** ≥ 6.2
- **Docker & Docker Compose** (optional, for containerized deployment)
- A **spaCy** English model: `python -m spacy download en_core_web_sm`

---

## Quickstart

### Docker Compose (recommended)

```bash
git clone https://github.com/your-org/nexus.git
cd nexus

# Optionally set a Postgres password now (interpolated into DATABASE_URL):
# export POSTGRES_PASSWORD=change-me

docker compose up --build -d
```

Starts, in dependency order:

1. `postgres` — pgvector PostgreSQL 16 (`127.0.0.1:5432`)
2. `redis` — Redis 7 (`127.0.0.1:6379`)
3. `migrate` — applies the versioned schema (baseline `schema.sql` + `migrations/`), then exits
4. `api` — FastAPI gateway on `http://localhost:8000`
5. `worker` — Celery ingestion/assignment workers
6. `beat` — Celery beat scheduler (persistent schedule in the `beat_data` volume)

```bash
docker compose ps
curl http://localhost:8000/health/live
```

The control plane is at `http://localhost:8000/admin`. On first boot no owner exists — the login page offers to **create the owner account**.

### Local manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python -m spacy download en_core_web_sm
```

```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/clustering_db"
export REDIS_URL="redis://localhost:6379/0"

python -m post_clustering_pipeline.migrate   # apply baseline + pending migrations
```

```bash
# Worker
nexus-celery-worker -A post_clustering_pipeline.queues.celery_app worker \
  --loglevel=INFO --pool=threads --concurrency=4

# Beat scheduler (background jobs)
nexus-celery-beat -A post_clustering_pipeline.queues.celery_app beat --loglevel=INFO

# API (uses the JSON-logging entry point)
python -m post_clustering_pipeline
# or explicitly with uvicorn:
uvicorn post_clustering_pipeline.api:app --host 0.0.0.0 --port 8000
```

Swagger docs: `http://localhost:8000/docs`. In local development without `API_AUTH_TOKEN`, auth is **dev-open**; the API is intentionally open so the loop works out of the box. Set `NEXUS_ENV=production` (and a token) to flip to fail-closed — see [Security](#security) and [Production Hardening](#production-hardening-checklist).

---

## Configuration Reference

All settings come from environment variables (`.env` at the repo root is loaded automatically; real env vars win).

### Deployment & security

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/clustering_db` | PostgreSQL connection string. |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker/backend + rate-limit store. |
| `NEXUS_ENV` | `development` | `production`/`prod` switches the API to fail-closed auth and enables HSTS. |
| `API_AUTH_TOKEN` | *(empty)* | Shared data-plane bearer token. Empty + production ⇒ the API **refuses to boot**. |
| `SESSION_COOKIE_SECURE` | `true` in production | Set the admin session cookie `Secure` flag. |
| `TRUSTED_HOSTS` | *(empty)* | Comma-separated Host allow-list (TrustedHostMiddleware). Set it behind a proxy. |
| `TRUSTED_ORIGINS` | *(empty)* | Extra origins accepted by the CSRF Origin check. |
| `LOGIN_MAX_ATTEMPTS` | `10` | Failed logins per account before lockout. |
| `LOGIN_IP_MAX_ATTEMPTS` | `30` | Failed logins per client IP before lockout. |
| `LOGIN_LOCK_WINDOW_SECONDS` | `900` | Lockout window (sliding). |
| `GOOGLE_OAUTH_CLIENT_ID` / `_SECRET` / `REDIRECT_BASE` | *(empty)* | Google sign-in. `REDIRECT_BASE` is the public base URL (HTTPS required unless `localhost`). |

### Embedding, clustering & thresholds

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Base embedding model. |
| `LORA_ADAPTER_DIR` | `<package>/models/lora_adapter` | Active fine-tuned PEFT LoRA adapter. |
| `REQUIRE_LORA_ADAPTER` | `false` | Refuse to embed unless the adapter is loadable. |
| `SIMILARITY_MARGIN` | `0.05` | Required margin between top match and runner-up. |
| `AUTO_ASSIGN_THRESHOLD` | `0.62` | Cosine bar for automatic assignment. |
| `CANDIDATE_THRESHOLD` | `0.45` | Floor for `candidate` vs `unassigned`. |
| `CENTROID_UPDATE_THRESHOLD` | `0.68` | Minimum similarity to trigger a centroid update. |
| `BIRTH_ASSIGN_THRESHOLD` | `0.62` | Match threshold when evaluating birth candidates against live hubs. |
| `BIRTH_SIMILARITY_FLOOR` | `0.45` | Lower bound for a new hub cohort. |
| `BIRTH_MIN_AUTHORS` | `4` | Distinct authors required to birth a hub. |
| `BIRTH_COHESION_FLOOR` | `0.45` | Intra-cohort cohesion minimum. |
| `ANCHOR_WEIGHT` | `0.35` | Anchor-post weight for hub identity. |
| `CENTROID_MAX_MEMBERS` | `150` | Bounded centroid window size. |
| `MERGE_SIMILARITY_THRESHOLD` | `0.62` | Similarity bar for hub merges. |
| `MERGE_ENTITY_FLOOR` | `0.45` | Shared identity-overlap bar. |
| `MERGE_NO_IDENTITY_SIM` | `0.72` | Merge bar when neither hub carries identity evidence. |
| `MERGE_ORG_VETO_MIN_ORGS` | `1` | Both hubs carrying disjoint ORGs vetoes a merge. |
| `CANDIDATE_AUTO_RESOLVE_MINUTES` | `10` | Age at which parked candidates are auto-swept (0 disables). |
| `CANDIDATE_AUTO_RESOLVE_SIM` | `0.50` | Confidence bar for the auto-sweep. |

### Processing & reliability

| Variable | Default | Description |
|---|---|---|
| `BATCH_SIZE` | `32` | Embedding batch size. |
| `SWEEPER_INTERVAL_SECONDS` | `60` | Reconcile-sweeper period. |
| `OUTBOX_MAX_ATTEMPTS` | `5` | Delivery attempts before DLQ. |
| `OUTBOX_UNCONSUMED_TTL_DAYS` | `7` | Pending events age to failed after this long unclaimed. |
| `OUTBOX_FAILED_TTL_DAYS` | `30` | Failed/delivered rows are pruned after this. |
| `CLAIM_CHUNK_SIZE` | `128` | Claim transaction chunk (commit cadence). |
| `ASSIGN_CHUNK_SIZE` | `64` | Assignment chunk. |
| `THRESHOLD_CACHE_TTL_SECONDS` | `60` | Process-local global-threshold cache. |
| `BULK_ASSIGN` | `true` | Use the chunked bulk-assignment matcher. |
| `CLAIM_DURABLE` | `false` | Force full durability for derived-state writes. |

### Intelligence & delivery

| Variable | Default | Description |
|---|---|---|
| `ROLLUP_WINDOW_HOURS` | `1` | Feedback-rollup window. |
| `TUNE_WINDOW_DAYS` | `7` | Autotune replay window. |
| `MIN_FEEDBACK_SAMPLES` | `300` | Minimum samples before the autotuner may propose. |
| `AUTO_TUNE_TARGET_PRECISION` | `0.97` | Autotune precision target. |
| `AUTO_TUNE_MIN_COVERAGE` | `0.85` | Autotune coverage floor. |
| `AUTO_APPLY_THRESHOLD` | `false` | Propose-only by default; set true to auto-apply revisions. |
| `EVENT_DELIVERY_MODE` | `polling` | `polling` (pull) or `push` (webhook). |
| `EVENT_WEBHOOK_URL` | *(empty)* | Push destination. |
| `EVENT_WEBHOOK_SIGNING_SECRET` | *(empty)* | HMAC-SHA256 `X-Nexus-Signature` header on pushes. |
| `INGEST_HINTS_KEY` / `_CAP` | `nexus:ingest:hints` / `100000` | Bounded Redis fast-path hint queue. |
| `MODEL_VERSION` / `POLICY_VERSION` | `embedding-v1` / `policy-v1` | Versions stamped into decision/feedback rows. |

---

## API & Integration Specification

All data-plane endpoints honor `Authorization: Bearer <consumer token>` (or the shared `API_AUTH_TOKEN`). Interactive docs: `/docs`.

### 1. Ingestion

#### `POST /posts` — single post

```bash
curl -X POST http://localhost:8000/posts \
  -H "Authorization: Bearer <token>" \
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

```json
{ "status": "queued", "post_id": 1 }
```

#### `POST /posts/batch` — bulk ingestion (max 500)

```json
{
  "posts": [
    { "user_id": 101, "content": "SpaceX Starship completes second stage burn.", "platform": "twitter", "source_id": "space", "external_post_id": "p1" },
    { "user_id": 102, "content": "NASA congratulates commercial partner.", "platform": "twitter", "source_id": "space", "external_post_id": "p2" }
  ]
}
```

```json
{ "status": "queued", "post_ids": [2, 3] }
```

### 2. Host event delivery (transactional outbox)

`GET /integration/events?limit=50&after=0&consumer=host-worker-1` leases the next batch (`FOR UPDATE SKIP LOCKED`), returning events with 2-minute leases; acknowledge each within the lease:

```bash
curl "http://localhost:8000/integration/events/10/ack?lease_token=a8f3b2c1d4e5" \
  -H "Authorization: Bearer <token>"
```

Events exceeding `OUTBOX_MAX_ATTEMPTS` move to the **dead-letter queue** (`delivery_status = 'failed'`); the `/health/ready` probe reports the DLQ count. Unconsumed pending events age out per `OUTBOX_UNCONSUMED_TTL_DAYS`.

### 3. Human-in-the-loop curation & moderation

| Endpoint | Effect |
|---|---|
| `POST /posts/unlink` | Detach a post from a hub (`user_removed` feedback captured), reset to `candidate`. |
| `POST /posts/confirm` | Explicitly bind post→hub (`user_confirmed` feedback), emits `post.confirmed`. |
| `DELETE /posts/{id}` | Tombstone a post (`noise`), emits `post.deleted`; blocks recreate of external ids. |
| `POST /events/merge` | Soft-merge `source_event_id` into `target_event_id` via the canonical guarded path. |

Feedback calls carry an `actor` label (or use the consumer's name automatically), so the learning loop knows who corrected what.

### 4. Hub exploration

| Method & Endpoint | Description |
|---|---|
| `GET /posts/{id}/status` | Assignment status, confidence, hub id. |
| `GET /events/{id}/anchor` | The catalyst post that established the hub. |
| `GET /events/{id}/timeline` | Chronological timeline. |
| `GET /events/{id}/top` | Posts ordered by engagement. |
| `GET /events/{id}/media` | Media-bearing posts. |
| `GET /hubs/{id}/view?limit=50&offset=0` | **One-click subject view**: catalyst, timeline, perspectives, media, voices, pagination. Merged hubs resolve to canonical ids. |

### 5. Health & observability

- `GET /health/live` → `{"status":"ok"}`
- `GET /health/ready` → readiness + `dlq_failed_events` + `stuck_processing` (503 when the DB is unreachable)
- `GET /admin/health/ready` → same, for the panel health pill

---

## Background Jobs & Active Learning

Scheduled via Celery beat; several are also runnable standalone.

| Job | Cadence | Purpose |
|---|---|---|
| `jobs.event_birth` | hourly | Partitions buffered posts into temporal slices, matches live hubs, builds a temporal k-NN graph, runs Louvain to birth new hubs. |
| `quality` rollup + `threshold-autotune-daily` | 5 min / daily | Rolls feedback up; replays human decisions through the versioned calibration path and **proposes** (or applies) a `global_similarity_threshold` revision in `policy_history`. |
| `jobs.monthly_learner` | monthly | Contrastive LoRA fine-tuning on `user_removed` feedback (triplet margin loss). |
| `jobs.promote_model` | as needed | Precision/recall/FP gating before promoting a candidate adapter into the `model_registry`. |
| `tools.inspect_model` | on demand | Weight / freeze / L2-norm sanity probe. |

Run standalone examples:

```bash
python -m post_clustering_pipeline.jobs.event_birth
python -m post_clustering_pipeline.jobs.monthly_learner --epochs 2 --batch-size 8 --output-dir ./models/candidate_adapter
python -m post_clustering_pipeline.jobs.promote_model embedding-v2 ./models/candidate_adapter --precision 0.96 --recall 0.88
python -m post_clustering_pipeline.tools.inspect_model
```

---

## Automated Testing

```bash
pytest tests/ -v            # NLP, embeddings, freezing, centroids, slicing, clustering
```

The unit/behavior suite targets non-DB logic. `tests/runtime/*` exercise a live stack (they write to a real database) and are excluded from the default run — point them at a scratch DB via `DATABASE_URL`.

End-to-end simulation:

```bash
python -m post_clustering_pipeline.evaluation.simulate_full_pipeline
```

---

## Operations

### Schema migrations

Fresh deploys bootstrap with the idempotent `schema.sql` baseline, then `migrations/*.sql` apply in order. Each migration runs in its own transaction; failures roll that file back and abort. Concurrent migrators serialize on a Postgres advisory lock; applied versions are recorded in `schema_migrations`.

```bash
python -m post_clustering_pipeline.migrate            # apply pending
python -m post_clustering_pipeline.migrate --dry-run  # preview
```

Legacy databases without `schema_migrations` are detected: the idempotent baseline is re-applied (no-op) and stamped before pending migrations run.

### Backups

```bash
BACKUP_DIR=/var/backups/nexus RETENTION_DAYS=14 DATABASE_URL=... ./scripts/backup.sh
```

Produces a custom-format `pg_dump` (gzip level 9, **embeddings included**) and prunes dumps older than `RETENTION_DAYS`. Restore:

```bash
pg_restore --clean --if-exists --no-owner -d "$DATABASE_URL" backup.dump
```

### Structured logging

Every process emits one JSON object per line (fields: `ts`, `level`, `logger`, `message`; HTTP access lines add `path`/`status_code`; exceptions add `exc`). Local dev can run components directly for human-readable output; compose services and the `nexus-api` / `nexus-celery-worker` / `nexus-celery-beat` entry points always log JSON (`log.py`).

---

## Production Hardening Checklist

A deployment that curates a platform's feed should take all of these:

1. **Set a strong token**: `NEXUS_ENV=production` **and** `API_AUTH_TOKEN=$(openssl rand -hex 32)` in `.env`. With an empty token the API **refuses to boot** rather than serving open.
2. **Terminate TLS** in front of port 8000; run uvicorn with `--proxy-headers --forwarded-allow-ips=<LB>` so the real client IP (login throttle) and scheme (`Secure` cookie, HSTS) are correct. Verify `GET /admin/login` sends `Strict-Transport-Security`.
3. **Pin trusted hosts**: set `TRUSTED_HOSTS` to the public hostname(s). Behind a proxy also set `TRUSTED_ORIGINS` to any origin the panel is reachable through.
4. **Enroll TOTP** for every admin and use Google sign-in (or a locked-down email+password policy). Login now requires the code automatically.
5. **Keep `.env` out of the image**; set `POSTGRES_PASSWORD` in the environment so compose's localhost-only bindings are not the only protection. Redis should sit behind a firewall (or `requirepass`).
6. **Protect `/docs`, `/openapi.json`** at the ingress if policy requires the schema to stay private.
7. **Schedule backups** (`scripts/backup.sh`) and verify a test restore periodically.
8. **Watch the DLQ**: monitor `GET /health/ready` (`dlq_failed_events`) and retry via the panel's Ops page.
9. **Restrict panel access** to the operator network if possible (the panel additionally accepts the shared bearer token; an admin ingress is defense-in-depth).

---

## Package Map

Import root: `post_clustering_pipeline`.

| Path | Responsibility |
|---|---|
| `api.py` | FastAPI gateway: middleware (auth, CSRF, security headers), ingestion, outbox, hub views |
| `admin.py` / `admin_auth.py` | Control plane routes, owner auth (PBKDF2, TOTP, JWT sessions, Google OAuth) |
| `config.py` | Central env-based configuration (secrets, thresholds, delivery, trust settings) |
| `ratelimit.py` | Redis-first sliding-window login/consumer throttling |
| `log.py` | Structured JSON logging configuration |
| `consumers.py` | Consumer-token hashing, lookup, toggling (`api_consumers`) |
| `tasks.py` / `queues.py` | Celery app, task graph, beat schedule, hint queue |
| `nlp.py` | Discourse gate and text cleansing |
| `models.py` | Transformer/LoRA embedding engine (`models/lora_adapter/` = artifacts) |
| `db.py` / `schema.sql` | Fork-safe connection pooling and pgvector schema |
| `migrate.py` / `migrations/` | Versioned, advisory-locked schema migrations |
| `events.py` / `dispatch.py` | Transactional outbox + webhook delivery |
| `corrections.py` / `merges.py` / `membership.py` / `refs.py` / `policy.py` / `decisions.py` | Feedback, merge orchestration, membership, hub references, versioned calibration, decision logging |
| `jobs/` | event birth, merging, autotune, learner, promotion |
| `audit.py` | Admin action audit trail |
| `evaluation/` / `tools/` / `utils/` | Benchmarks, inspection, dataset helpers |
| `templates/admin/` + `static/` | Admin panel UI (Jinja2 + htmx) |