from contextlib import contextmanager
from celery import Celery
from celery.schedules import crontab
import redis

from .config import REDIS_URL

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
    imports=["post_clustering_pipeline.tasks"]
)

# Automated Beat Schedule
celery_app.conf.beat_schedule = {
    "reconcile-pending-posts-every-60s": {
        "task": "post_clustering_pipeline.tasks.reconcile_pending_posts",
        "schedule": 60.0,
    },
    "event-birth-hourly": {
        "task": "post_clustering_pipeline.tasks.run_event_birth_scheduled",
        "schedule": crontab(minute=17),
    },
    "hub-merge-reconciliation-every-30m": {
        "task": "post_clustering_pipeline.tasks.run_hub_merge_scheduled",
        "schedule": crontab(minute="5,35"),
    },
    "prune-delivered-outbox-daily": {
        "task": "post_clustering_pipeline.tasks.prune_delivered_outbox",
        "schedule": crontab(hour=3, minute=45),
    },
    "dispatch-webhooks-every-15s": {
        "task": "post_clustering_pipeline.tasks.dispatch_webhooks_scheduled",
        "schedule": 15.0,
    },
}


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
