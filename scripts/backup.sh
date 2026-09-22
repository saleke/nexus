#!/usr/bin/env bash
# Nexus database backup: structured pg_dump + retention pruning.
#
# Usage: BACKUP_DIR=/var/backups/nexus RETENTION_DAYS=14 ./scripts/backup.sh
# DB target comes from DATABASE_URL (defaults match config.py).
# Off-site copy (optional): BACKUP_REMOTE=s3://bucket/prefix | gs://bucket/prefix
#   | user@host:/path. Uploaded with aws/gsutil/rsync respectively; a configured
#   remote with a missing client tool fails loudly rather than silently local-only.
#
# Restore a custom-format dump with:
#   pg_restore --clean --if-exists --no-owner -d "$DATABASE_URL" <file>.dump
set -euo pipefail

DB_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/clustering_db}"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
BACKUP_REMOTE="${BACKUP_REMOTE:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/nexus_${STAMP}.dump"

mkdir -p "$BACKUP_DIR"
REDACTED_URL="$(printf '%s' "$DB_URL" | sed -E 's|(://[^:/@]+:)[^@]+@|\1****@|')"
echo "backing up $REDACTED_URL -> $OUT"
pg_dump --format=custom --compress=9 --no-owner --no-privileges \
    "$DB_URL" > "$OUT"
echo "backup complete: $(du -h "$OUT" | cut -f1)"

# Verify the archive is readable before trusting it (a truncated pg_dump still
# exits 0 when the pipe to the file succeeds; --list fails loudly on corruption).
pg_restore --list "$OUT" >/dev/null
echo "verified dump archive: $(basename "$OUT")"

# Off-site copy (optional): keep a copy off the database host. Runs BEFORE
# retention pruning so the remote retains the pruned-away history.
if [[ -n "$BACKUP_REMOTE" ]]; then
    case "$BACKUP_REMOTE" in
        s3://*)
            command -v aws >/dev/null 2>&1 || { echo "BACKUP_REMOTE set but aws CLI not found" >&2; exit 1; }
            aws s3 cp "$OUT" "${BACKUP_REMOTE%/}/$(basename "$OUT")" ;;
        gs://*)
            command -v gsutil >/dev/null 2>&1 || { echo "BACKUP_REMOTE set but gsutil not found" >&2; exit 1; }
            gsutil cp "$OUT" "${BACKUP_REMOTE%/}/$(basename "$OUT")" ;;
        *:*)
            command -v rsync >/dev/null 2>&1 || { echo "BACKUP_REMOTE set but rsync not found" >&2; exit 1; }
            rsync -a --partial "$OUT" "$BACKUP_REMOTE" ;;
        *)
            echo "unsupported BACKUP_REMOTE scheme: $BACKUP_REMOTE" >&2
            exit 1 ;;
    esac
    echo "off-site copy complete: $BACKUP_REMOTE"
fi

# Retention: prune dumps older than RETENTION_DAYS.
find "$BACKUP_DIR" -name 'nexus_*.dump' -mtime +"$RETENTION_DAYS" -print -delete
echo "pruned dumps older than ${RETENTION_DAYS} days"