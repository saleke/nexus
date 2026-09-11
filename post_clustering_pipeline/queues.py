import sys
import os
from celery import Celery
try:
    from .config import REDIS_URL
except ImportError:
    from config import REDIS_URL

# Append current working directory to Python system path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
