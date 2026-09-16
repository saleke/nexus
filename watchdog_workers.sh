#!/bin/bash
# Respawn the celery worker + beat together. They died silently before and
# nothing restarted them; this keeps the pair alive, logging to:
#   /tmp/opencode/worker.log  /tmp/opencode/beat.log
#   /tmp/opencode/workers-watchdog.log
cd "$(dirname "$0")"
CELERY=.venv/bin/celery

while true; do
    $CELERY -A post_clustering_pipeline.queues.celery_app worker --loglevel=INFO --pool=threads --concurrency=4 >> /tmp/opencode/worker.log 2>&1 &
    WPID=$!
    $CELERY -A post_clustering_pipeline.queues.celery_app beat --loglevel=INFO --schedule=/tmp/alithos-celerybeat.db >> /tmp/opencode/beat.log 2>&1 &
    BPID=$!
    echo "[watchdog] worker=$WPID beat=$BPID up at $(date -u +%FT%TZ)" >> /tmp/opencode/workers-watchdog.log
    wait -n "$WPID" "$BPID"
    echo "[watchdog] one of worker=$WPID beat=$BPID exited at $(date -u +%FT%TZ); restarting pair in 3s" >> /tmp/opencode/workers-watchdog.log
    kill "$WPID" "$BPID" 2>/dev/null
    wait "$WPID" "$BPID" 2>/dev/null
    sleep 3
done