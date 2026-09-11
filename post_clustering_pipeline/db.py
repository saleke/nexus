import psycopg2
from psycopg2.extras import RealDictCursor
try:
    from .config import DATABASE_URL
except ImportError:
    from config import DATABASE_URL

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn
