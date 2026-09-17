#!/usr/bin/env bash
# Nexus database backup: structured pg_dump + retention pruning.
#
# Usage: BACKUP_DIR=/var/backups/nexus RETENTION_DAYS=14 ./scripts/backup.sh
# DB target comes from DATABASE_URL (defaults match config.py).
#
# Restore a custom-format dump with:
#   pg_restore --clean --if-exists --no-owner -d "$DATABASE_URL" <file>.dump
set -euo pipefail

DB_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/clustering_db}"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/nexus_${STAMP}.dump"

mkdir -p "$BACKUP_DIR"
REDACTED_URL="$(printf '%s' "$DB_URL" | sed -E 's|(://[^:/@]+:)[^@]+@|\1****@|')"
echo "backing up $REDACTED_URL -> $OUT"
pg_dump --format=custom --compress=9 --no-owner --no-privileges \
    "$DB_URL" > "$OUT"
echo "backup complete: $(du -h "$OUT" | cut -f1)"

# Retention: prune dumps older than RETENTION_DAYS.
find "$BACKUP_DIR" -name 'nexus_*.dump' -mtime +"$RETENTION_DAYS" -print -delete
echo "pruned dumps older than ${RETENTION_DAYS} days"