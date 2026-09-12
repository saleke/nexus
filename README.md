# EventHub Clustering Pipeline

Python package for ingesting posts, embedding them, assigning them to active event hubs, and clustering buffered posts.

## Setup

Install PostgreSQL with the `vector` (pgvector) extension and Redis, then install the package:

```bash
python -m pip install -e .
python -m spacy download en_core_web_sm
psql "$DATABASE_URL" -f post_clustering_pipeline/schema.sql
```

The default database URL is `postgresql://postgres:postgres@localhost:5432/clustering_db`; override it with `DATABASE_URL`. `REDIS_URL` defaults to `redis://localhost:6379/0`.

## Run

```bash
redis-server
celery -A post_clustering_pipeline.queues.celery_app worker --loglevel=INFO
python -m post_clustering_pipeline
```

The API is available at `http://localhost:8000`. Jobs can be run as modules, for example `python -m post_clustering_pipeline.jobs.event_birth`.

## Host event delivery

Choose one delivery mode with an environment variable:

```env
EVENT_DELIVERY_MODE=polling
```

Supported values are `polling`, `long_polling`, and `webhook`. Polling is the default and uses `GET /integration/events` followed by `POST /integration/events/{id}/ack`. The host should store the returned `event_key` and use `after=<last_event_id>` on the next request. Webhook configuration is reserved for the delivery worker and uses `EVENT_WEBHOOK_URL`.

During the build phase, platform identity fields are optional so local test clients can continue sending only `user_id` and `content`. A future versioned production contract can make them required without adding another runtime setting.

For a self-contained deployment, use Docker Compose:

```bash
docker compose up --build
```

The API, worker, Redis, PostgreSQL/pgvector, and schema migration will start in dependency order.
