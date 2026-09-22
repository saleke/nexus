#!/bin/bash
# Keep only the API up. Uses the package entry point (not bare uvicorn) so the
# process gets the same structured JSON logging + log_config=None as
# `python -m post_clustering_pipeline`; stdout/stderr are captured to the log.
cd "$(dirname "$0")"
while true; do
    .venv/bin/python -m post_clustering_pipeline >> /tmp/nexus-admin.log 2>&1
    code=$?
    echo "[watchdog] api exited code=$code at $(date -u +%FT%TZ); restarting in 3s" >> /tmp/nexus-admin.log
    sleep 3
done
