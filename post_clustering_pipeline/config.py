import os

# Load .env (project root or CWD) so deployment secrets/credentials can live in
# a gitignored file instead of the process environment. Real env vars always win.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")

GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
GOOGLE_OAUTH_REDIRECT_BASE = os.getenv("GOOGLE_OAUTH_REDIRECT_BASE", "").rstrip("/")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/clustering_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Base model and adapter paths
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_ADAPTER_DIR = os.getenv("LORA_ADAPTER_DIR", os.path.join(PACKAGE_DIR, "models", "lora_adapter"))
REQUIRE_LORA_ADAPTER = os.getenv("REQUIRE_LORA_ADAPTER", "false").lower() in {"1", "true", "yes"}

# LoRA Configuration Settings
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "value"]

# Thresholds & Assignment Policies (Precision-first: Better not to cluster than cluster wrong)
# Calibrated against the measured same-topic MiniLM-L6-v2 cosine scale: intra-topic
# similarities peak around ~0.71 (mean ~0.50) while cross-topic text sits near 0.1
# (exceptions are the intentionally confusable pairs, e.g. Apple vs Samsung phone
# launches, which overlap up to ~0.64 and are handled by the strong-entity
# split/veto instead of a similarity cut). The historical 0.80-0.95 defaults sat
# above the embedder ceiling and froze hub creation entirely (0 hubs / 0% coverage).
SIMILARITY_MARGIN = float(os.getenv("SIMILARITY_MARGIN", "0.05"))
AUTO_ASSIGN_THRESHOLD = float(os.getenv("AUTO_ASSIGN_THRESHOLD", "0.62"))
CANDIDATE_THRESHOLD = float(os.getenv("CANDIDATE_THRESHOLD", "0.45"))
CENTROID_UPDATE_THRESHOLD = float(os.getenv("CENTROID_UPDATE_THRESHOLD", "0.68"))
BIRTH_ASSIGN_THRESHOLD = float(os.getenv("BIRTH_ASSIGN_THRESHOLD", "0.62"))
BIRTH_SIMILARITY_FLOOR = float(os.getenv("BIRTH_SIMILARITY_FLOOR", "0.45"))
BIRTH_MIN_AUTHORS = int(os.getenv("BIRTH_MIN_AUTHORS", "4"))
BIRTH_COHESION_FLOOR = float(os.getenv("BIRTH_COHESION_FLOOR", "0.45"))
ANCHOR_WEIGHT = float(os.getenv("ANCHOR_WEIGHT", "0.35"))
CENTROID_MAX_MEMBERS = int(os.getenv("CENTROID_MAX_MEMBERS", "150"))

# Processing & Reliability Settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SWEEPER_INTERVAL_SECONDS = int(os.getenv("SWEEPER_INTERVAL_SECONDS", "60"))
OUTBOX_MAX_ATTEMPTS = int(os.getenv("OUTBOX_MAX_ATTEMPTS", "5"))

# Chunk sizes bound transaction/fsync cost per batch: commit every chunk so
# partial progress survives crashes, while keeping the number of commits low.
CLAIM_CHUNK_SIZE = int(os.getenv("CLAIM_CHUNK_SIZE", "128"))
ASSIGN_CHUNK_SIZE = int(os.getenv("ASSIGN_CHUNK_SIZE", "64"))

# Process-local TTL cache for the global similarity threshold to avoid a
# SELECT per batch; the weekly job updates it at most once a day.
THRESHOLD_CACHE_TTL_SECONDS = int(os.getenv("THRESHOLD_CACHE_TTL_SECONDS", "60"))

# Bulk assignment matcher: collapses the per-post HNSW + write statements into
# a handful of chunk-level statements (major round-trip reduction). The per-row
# path remains available as a fallback; verify the LATERAL query uses the HNSW
# index with EXPLAIN before trusting the bulk path in production.
BULK_ASSIGN = os.getenv("BULK_ASSIGN", "true").lower() in {"1", "true", "yes", "on"}

# Intelligence layer / control plane
# Rollups run every few minutes but anchor to the hour; a trailing 1h window.
ROLLUP_WINDOW_HOURS = int(os.getenv("ROLLUP_WINDOW_HOURS", "1"))
# Auto-tuning replay window (days) for the threshold estimator.
TUNE_WINDOW_DAYS = int(os.getenv("TUNE_WINDOW_DAYS", "7"))
# Minimum human feedback decisions before the autotuner may propose a change.
MIN_FEEDBACK_SAMPLES = int(os.getenv("MIN_FEEDBACK_SAMPLES", "300"))
AUTO_TUNE_TARGET_PRECISION = float(os.getenv("AUTO_TUNE_TARGET_PRECISION", "0.97"))
AUTO_TUNE_MIN_COVERAGE = float(os.getenv("AUTO_TUNE_MIN_COVERAGE", "0.85"))
# Propose-only by default: the autotuner writes a revision to policy_history
# with status 'proposed' for the owner to approve. Set true to auto-apply.
AUTO_APPLY_THRESHOLD = os.getenv("AUTO_APPLY_THRESHOLD", "false").lower() in {"1", "true", "yes", "on"}

# Derived-state writes (claims, dispatch, reset-stale) commit without an fsync
# wait via SET LOCAL synchronous_commit = off. Safe because each such claim is
# transaction-atomic with its gating result: a lost commit just leaves the post
# 'pending' and it is reclaimed. Contract writes always stay fully durable.
# Set true only to force full durability everywhere as an escape hatch.
CLAIM_DURABLE = os.getenv("CLAIM_DURABLE", "false").lower() in {"1", "true", "yes", "on"}

MODEL_VERSION = os.getenv("MODEL_VERSION", "embedding-v1")
POLICY_VERSION = os.getenv("POLICY_VERSION", "policy-v1")
EVENT_DELIVERY_MODE = os.getenv("EVENT_DELIVERY_MODE", "polling")
EVENT_WEBHOOK_URL = os.getenv("EVENT_WEBHOOK_URL", "")
# Optional HMAC secret for webhook delivery. When set, every POST carries
# ``X-Nexus-Signature: sha256=<hex HMAC of the exact request body>`` so the
# host can verify the payload came from this platform and was not replayed with
# different bytes. Empty string = payloads unsigned (dev only; set in prod).
EVENT_WEBHOOK_SIGNING_SECRET = os.getenv("EVENT_WEBHOOK_SIGNING_SECRET", "")

# Fast-path hint queue between the API and the ingest drain: the API pushes
# post ids here; drain_ingest_hints batches them. The posts row is authoritative.
INGEST_HINTS_KEY = os.getenv("INGEST_HINTS_KEY", "nexus:ingest:hints")
INGEST_HINTS_CAP = int(os.getenv("INGEST_HINTS_CAP", "100000"))
"""Bounded ingest hint queue: the Redis fast-path list is capped at this
many pending ids. The DB insert is the durable source of truth, so a full
hint queue degrades to the 30s pending sweep rather than dropping posts -
it can never silently lose a post, only trade push latency for bounded memory."""
