#!/bin/bash
while true; do
    .venv/bin/python -m uvicorn post_clustering_pipeline.api:app --host 0.0.0.0 --port 8000
    code=$?
    echo "[watchdog] uvicorn exited code=$code at $(date -u +%FT%TZ); restarting in 3s" >> /tmp/nexus-admin.log
    sleep 3
done