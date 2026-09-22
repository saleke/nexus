import os
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from .config import DATABASE_URL

try:  # pgvector extension integration (optional; text codec is the fallback)
    from pgvector.psycopg2 import register_vector
except ImportError:  # pragma: no cover - exercised only in minimal envs
    register_vector = None

_pool: ThreadedConnectionPool | None = None
_pool_pid: int | None = None

# Psycopg2 C connection objects reject attaching project attributes (raises
# AttributeError on ``conn._pgvector_registered = True``), so track which
# pooled connections already had the vector typecaster bound by connection
# identity instead of mutating the connection object.
_PGVECTOR_REGISTERED: set[int] = set()


def _register_pgvector(conn):
    """Bind the pgvector adapter/typecaster once per pooled connection.

    Lets callers bind numpy arrays directly as vector params and decode bare
    vector columns natively. Harmless (and skipped) when the vector type or
    the package is absent - reads that need text cast explicitly and writes
    fall back to the embed_io text literal.
    """
    if id(conn) in _PGVECTOR_REGISTERED:
        return
    if register_vector is not None:
        try:
            register_vector(conn)
        except Exception:
            pass  # vector type missing in this database; text codec is the fallback
    _PGVECTOR_REGISTERED.add(id(conn))


def extract_val(row, key: str, idx: int):
    """Read a column from a row regardless of cursor dict/tuple style.

    ``RealDictCursor`` rows (the cluster pipeline default) support key lookup;
    legacy plain ``psycopg2`` tuples fall back to positional access.
    """
    if row is None:
        return None
    try:
        return row[key]
    except TypeError:
        return row[idx]


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
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    _register_pgvector(conn)
    return conn


@contextmanager
def get_db_cursor(commit: bool = True, durable: bool = True):
    """Context manager for pooled database operations with commit/rollback.

    ``durable=False`` issues ``SET LOCAL synchronous_commit = off`` for the
    transaction: the commit is acknowledged without a WAL fsync wait. This is
    only safe for state that is *derived* and self-healing - e.g. a claim whose
    gating writes share the same transaction, so a lost commit simply leaves
    the post 'pending' and it is reclaimed later. Contract writes (assignments,
    outboxes, centroids) must keep the default full durability. ``SET LOCAL``
    scopes to the transaction and reverts on commit/rollback, so pooled
    connections cannot leak the weaker durability setting.
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        _register_pgvector(conn)
        with conn.cursor() as cur:
            if not durable:
                cur.execute("SET LOCAL synchronous_commit = off")
            yield cur
        if commit:
            conn.commit()
        else:
            # A read context still opened a transaction on first execute; end it
            # before the connection goes back to the pool, otherwise it is
            # returned "idle in transaction" and pins a snapshot/locks.
            conn.rollback()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)