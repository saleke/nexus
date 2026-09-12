import os
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from .config import DATABASE_URL

_pool: ThreadedConnectionPool | None = None
_pool_pid: int | None = None


def get_pool(minconn: int = 2, maxconn: int = 20) -> ThreadedConnectionPool:
    """Return process-safe thread pool, recreating if worker forked."""
    global _pool, _pool_pid
    current_pid = os.getpid()
    if _pool is None or _pool_pid != current_pid:
        _pool = ThreadedConnectionPool(minconn, maxconn, DATABASE_URL, cursor_factory=RealDictCursor)
        _pool_pid = current_pid
    return _pool


def get_db_connection():
    """Direct database connection for standalone scripts / legacy callers."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


@contextmanager
def get_db_cursor(commit: bool = True):
    """Context manager for pooled database operations with automatic commit/rollback."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
