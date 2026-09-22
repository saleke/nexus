"""Beat-schedule completeness + worker reliability config (P5).

Parses ``tasks.py`` for Celery task names instead of importing it: importing
``tasks`` pulls in the embedding model (~70s), which has no place in a unit
test. ``queues`` itself is cheap (celery/redis/config only).
"""
import ast
from pathlib import Path

from post_clustering_pipeline.queues import celery_app

TASKS_PY = Path(__file__).resolve().parents[1] / "post_clustering_pipeline" / "tasks.py"


def _declared_task_names() -> set[str]:
    tree = ast.parse(TASKS_PY.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            if not (isinstance(func, ast.Attribute) and func.attr == "task"):
                continue
            for kw in dec.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    names.add(kw.value.value)
    return names


def test_every_beat_task_is_declared():
    declared = _declared_task_names()
    assert declared, "AST parse found no task names; decorator shape changed?"
    scheduled = {entry["task"] for entry in celery_app.conf.beat_schedule.values()}
    assert scheduled, "beat schedule is empty"
    missing = scheduled - declared
    assert not missing, f"beat schedules undeclared tasks: {sorted(missing)}"


def test_beat_schedule_has_expected_heavy_jobs():
    scheduled = {entry["task"] for entry in celery_app.conf.beat_schedule.values()}
    for task in (
        "post_clustering_pipeline.tasks.run_event_birth_scheduled",
        "post_clustering_pipeline.tasks.run_hub_merge_scheduled",
        "post_clustering_pipeline.tasks.dispatch_webhooks_scheduled",
        "post_clustering_pipeline.tasks.reconcile_pending_posts",
    ):
        assert task in scheduled


def test_worker_reliability_settings():
    conf = celery_app.conf
    assert conf.worker_prefetch_multiplier == 1
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.broker_connection_retry_on_startup is True
