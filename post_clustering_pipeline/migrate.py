"""Versioned schema migrations for the Nexus cluster DB.

Fresh deploys bootstrap from schema.sql (baseline), then apply numbered
migrations under migrations/ in order. Live DBs that predate version
bookkeeping get the now-idempotent schema.sql run against them (a no-op on
existing objects) and stamped as the baseline, then pending migrations apply.
Because schema.sql grows in place, a stamped DB minted from an older baseline
is verified against sentinel tables/columns on every run and fails closed
with a convergence hint instead of drifting silently.

Each migration runs in its own transaction; a failure rolls back that file and
aborts the run, so a partial apply is never recorded. Concurrency is guarded
with a Postgres advisory lock so simultaneous migrators cannot interleave.

Usage:
    python -m post_clustering_pipeline.migrate [--db-url URL] [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_BASELINE_VERSION = "0000_schema_sql_baseline"
_LOCK_KEY = 6887001  # hashtext('nexus_schema_migrations') equivalent

# Sentinel objects the current schema.sql baseline is expected to have produced.
# If a stamped DB is missing any of these it was minted from an older baseline;
# the operator must converge it before the stack runs, not discover mid-flight.
_SENTINEL_TABLES = (
    "unclustered_posts_buffer", "clustering_feedback_log", "model_registry",
    "integration_outbox", "assignment_decision_log", "feedback_rollups",
    "policy_history", "admin_audit_log", "admin_users", "api_consumers",
    "hub_merges", "hub_members",
)
_SENTINEL_COLUMNS = {
    "posts": ("embedding", "deleted_at", "post_seq", "event_id"),
    "event_hubs": ("centroid", "seed_post_id", "member_count"),
    "system_config": ("value_text",),
}


def _assert_baseline_objects(conn) -> None:
    """Fail closed when a stamped database lacks baseline objects."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind = 'r'"
        )
        tables = {r[0] for r in cur.fetchall()}
    missing_tables = sorted(t for t in _SENTINEL_TABLES if t not in tables)
    if missing_tables:
        raise RuntimeError(
            "schema_migrations records the baseline, but the database is missing tables: "
            + ", ".join(missing_tables)
            + ". Re-apply the schema.sql baseline to converge this database."
        )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'public'"
        )
        columns: dict[str, set[str]] = {}
        for table, column in cur.fetchall():
            columns.setdefault(table, set()).add(column)
    missing_cols = sorted(
        f"{table}.{column}"
        for table, required in _SENTINEL_COLUMNS.items()
        for column in required
        if column not in columns.get(table, set())
    )
    if missing_cols:
        raise RuntimeError(
            "schema_migrations records the baseline, but the database is missing columns: "
            + ", ".join(missing_cols)
            + ". Re-apply the schema.sql baseline to converge this database."
        )


def _connect(db_url: str):
    return psycopg2.connect(db_url)


def _ensure_bookkeeping(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    TEXT PRIMARY KEY,
                filename   TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    conn.commit()


def _applied_versions(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def _record(conn, version: str, filename: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO schema_migrations (version, filename) VALUES (%s, %s)",
            (version, filename),
        )


def run(db_url: str, dry_run: bool = False) -> list[str]:
    baseline_path = Path(__file__).resolve().parent / "schema.sql"
    migrations = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    applied_this_run: list[str] = []

    with _connect(db_url) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            _ensure_bookkeeping(conn)

            already_stamped = _BASELINE_VERSION in _applied_versions(conn)
            if not already_stamped:
                sql = baseline_path.read_text()
                if dry_run:
                    print(f"[dry-run] would apply baseline {baseline_path.name}")
                else:
                    with conn.cursor() as cur:
                        cur.execute(sql)
                        _record(conn, _BASELINE_VERSION, baseline_path.name)
                    conn.commit()
                applied_this_run.append(baseline_path.name)
            elif not dry_run:
                # Baseline is stamped, but schema.sql grows in place. A database
                # stamped from an older baseline can silently lack later
                # tables/columns and then fail in confusing mid-flight errors.
                # Fail closed here with a clear message instead.
                _assert_baseline_objects(conn)

            pending = []
            for path in migrations:
                version = path.stem
                if version not in _applied_versions(conn):
                    pending.append(path)

            if dry_run:
                for path in pending:
                    print(f"[dry-run] would apply {path.name}")
                if not pending:
                    print("schema up to date")
                return applied_this_run

            for path in pending:
                sql = path.read_text()
                with conn.cursor() as cur:
                    cur.execute(sql)
                    _record(conn, path.stem, path.name)
                conn.commit()
                applied_this_run.append(path.name)
                print(f"applied {path.name}")

            if not applied_this_run:
                print("schema up to date")
        finally:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001 - connection already broken
                pass
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
            conn.commit()

    return applied_this_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Nexus schema migrations")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", ""),
                        help="Postgres DSN (defaults to DATABASE_URL)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list pending migrations without applying")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL", "")
    if not db_url:
        print("error: no database URL (set DATABASE_URL or pass --db-url)", file=sys.stderr)
        return 2

    try:
        run(db_url, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - surface migration failures clearly
        print(f"error: migration failed; nothing committed for that file.\n  {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())