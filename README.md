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

The API is available at `http://localhost:8000`. Cron modules can be run as modules, for example `python -m post_clustering_pipeline.cron_event_birth`.
