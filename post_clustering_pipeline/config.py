import os
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")

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
SIMILARITY_MARGIN = float(os.getenv("SIMILARITY_MARGIN", "0.05"))
AUTO_ASSIGN_THRESHOLD = float(os.getenv("AUTO_ASSIGN_THRESHOLD", "0.88"))
CANDIDATE_THRESHOLD = float(os.getenv("CANDIDATE_THRESHOLD", "0.75"))
CENTROID_UPDATE_THRESHOLD = float(os.getenv("CENTROID_UPDATE_THRESHOLD", "0.90"))
BIRTH_ASSIGN_THRESHOLD = float(os.getenv("BIRTH_ASSIGN_THRESHOLD", "0.82"))
BIRTH_SIMILARITY_FLOOR = float(os.getenv("BIRTH_SIMILARITY_FLOOR", "0.80"))
BIRTH_MIN_AUTHORS = int(os.getenv("BIRTH_MIN_AUTHORS", "4"))
BIRTH_COHESION_FLOOR = float(os.getenv("BIRTH_COHESION_FLOOR", "0.82"))
ANCHOR_WEIGHT = float(os.getenv("ANCHOR_WEIGHT", "0.35"))
CENTROID_MAX_MEMBERS = int(os.getenv("CENTROID_MAX_MEMBERS", "150"))

# Processing & Reliability Settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SWEEPER_INTERVAL_SECONDS = int(os.getenv("SWEEPER_INTERVAL_SECONDS", "60"))
OUTBOX_MAX_ATTEMPTS = int(os.getenv("OUTBOX_MAX_ATTEMPTS", "5"))

MODEL_VERSION = os.getenv("MODEL_VERSION", "embedding-v1")
POLICY_VERSION = os.getenv("POLICY_VERSION", "policy-v1")
EVENT_DELIVERY_MODE = os.getenv("EVENT_DELIVERY_MODE", "polling")
EVENT_WEBHOOK_URL = os.getenv("EVENT_WEBHOOK_URL", "")
