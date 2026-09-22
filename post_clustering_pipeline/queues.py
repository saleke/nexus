from contextlib import contextmanager
from celery import Celery
from celery.schedules import crontab
from celery.signals import setup_logging
import redis

from .config import REDIS_URL, INGEST_HINTS_KEY, INGEST_HINTS_CAP

celery_app = Celery(
    "clustering_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    imports=["post_clustering_pipeline.tasks"],
    # Reliability for a DB/ML workload on a thread-pool worker: never
    # prefetch more than one message per thread (a 4x prefetch lets one
    # saturated worker sit on tasks another worker could run), and only ack a
    # task after it returns so a killed worker re-delivers instead of silently
    # dropping it. All tasks are idempotent (claim/lease/distributed-lock
    # guarded), so re-delivery is safe.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
)


@setup_logging.connect
def _structured_json_logging(**kwargs):
    """Log JSON-only from the worker/beat.

    Celery skips its own logging config whenever a receiver is connected to the
    ``setup_logging`` signal, so this one call is our sole handler: without it
    celery's Logging.setup hijacks the root logger and reinstalls its text
    ColorFormatter. The NullHandler that kombu drops on the 'celery' logger is
    removed by configure_json_logging before celery.boot logs anything.
    """
    from .log import configure_json_logging

    configure_json_logging()

# Automated Beat Schedule
celery_app.conf.beat_schedule = {
    # Fast-path: drain API ingest hints. The row insert is the durable source
    # of truth; this only batches hints sooner than the sweeper below.
    "drain-ingest-hints-every-5s": {
        "task": "post_clustering_pipeline.tasks.drain_ingest_hints",
        "schedule": 5.0,
    },
    # Safety net: returns stuck 'processing' claims to 'pending' and backstops
    # any API hint that was lost (e.g. Redis down). The row insert is the
    # durable source of truth; the Redis task push is only a fast-path hint.
    "reconcile-pending-posts-every-30s": {
        "task": "post_clustering_pipeline.tasks.reconcile_pending_posts",
        "schedule": 30.0,
    },
    "event-birth-hourly": {
        "task": "post_clustering_pipeline.tasks.run_event_birth_scheduled",
        "schedule": crontab(minute=17),
    },
    "hub-merge-reconciliation-every-30m": {
        "task": "post_clustering_pipeline.tasks.run_hub_merge_scheduled",
        "schedule": crontab(minute="5,35"),
    },
    "repost-fold-sweep-every-10m": {
        "task": "post_clustering_pipeline.tasks.fold_hub_reposts_scheduled",
        "schedule": 600.0,
    },
    "prune-delivered-outbox-daily": {
        "task": "post_clustering_pipeline.tasks.prune_delivered_outbox",
        "schedule": crontab(hour=3, minute=45),
    },
    "dispatch-webhooks-every-15s": {
        "task": "post_clustering_pipeline.tasks.dispatch_webhooks_scheduled",
        "schedule": 15.0,
    },
    # Quality loop: roll up the anchored window every few minutes (idempotent
    # because the window is hour-bounded), and replay human feedback into a
    # threshold revision daily. Both honor the proposal workflow.
    "quality-rollup-every-5m": {
        "task": "post_clustering_pipeline.tasks.run_quality_rollup_scheduled",
        "schedule": 300.0,
    },
    "threshold-autotune-daily": {
        "task": "post_clustering_pipeline.tasks.run_threshold_autotune_scheduled",
        "schedule": crontab(hour=4, minute=37),
    },
}


_INGEST_HINTS_CAP_LUA = """
local cap = tonumber(ARGV[1])
if cap > 0 then
    redis.call('LPUSH', KEYS[1], unpack(ARGV, 2))
    redis.call('LTRIM', KEYS[1], 0, cap-1)
else
    redis.call('LPUSH', KEYS[1], unpack(ARGV, 2))
end
return redis.call('LLEN', KEYS[1])
"""[1:]  # <-- when queued via `from .config import INGEST_HINTS_KEY`


def bounded_push_ingest_hints(*post_ids: int, cap: int | None = None) -> int:
    """Atomically append ingest hints with a hard cap; returns queue length.

    The DB row insert is the durable source of truth (posts are never lost even
    if Redis is down). This queue is a latency fast path, so it must stay
    bounded: an unbounded LPUSH list on a busy ingest day is a slowly-leaking
    memory profile attached to a loop that was only ever meant to batch hints
    sooner. ``LTRIM`` in the same script keeps the cap race-free - the cap is
    the maximum and the script is the single writer.
    """
    r = get_redis_client()
    ids = []
    for pid in post_ids:
        try:
            n = int(pid)
        except (TypeError, ValueError):
            continue
        if n > 0:
            ids.append(str(n))
    if not ids:
        return 0
    limit = (cap if cap is not None else INGEST_HINTS_CAP) or 0
    try:
        if limit > 0:
            return int(r.eval(_INGEST_HINTS_CAP_LUA, 1, INGEST_HINTS_KEY, limit, *ids))
        return int(r.lpush(INGEST_HINTS_KEY, *ids))
    except redis.exceptions.ResponseError:
        # Older Redis or scripting disabled: fall back to two calls but still
        # keep the cap (linger is preferable to unbounded in all cases).
        r.lpush(INGEST_HINTS_KEY, *ids)
        if limit > 0:
            r.ltrim(INGEST_HINTS_KEY, 0, limit - 1)
        return int(r.llen(INGEST_HINTS_KEY))


def get_redis_client():
    return redis.Redis.from_url(REDIS_URL)


@contextmanager
def distributed_task_lock(lock_name: str, timeout: int = 3600):
    """Non-blocking Redis distributed lock to prevent scheduled task overlap."""
    r = get_redis_client()
    lock = r.lock(f"nexus:lock:{lock_name}", timeout=timeout, blocking=False)
    acquired = False
    try:
        acquired = lock.acquire()
        yield acquired
    finally:
        if acquired:
            try:
                lock.release()
            except Exception:
                pass


def worker_main() -> None:
    """Console entry point with structured logging (see log.py)."""
    from .log import configure_json_logging
    from celery.bin.celery import celery

    configure_json_logging()
    celery.main()


def beat_main() -> None:
    """Console entry point with structured logging (see log.py)."""
    from .log import configure_json_logging
    from celery.bin.celery import celery

    configure_json_logging()
    celery.main()
