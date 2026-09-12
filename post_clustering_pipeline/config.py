import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/clustering_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Base model and adapter paths
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_ADAPTER_DIR = os.getenv("LORA_ADAPTER_DIR", os.path.join(PACKAGE_DIR, "models", "lora_adapter"))

# LoRA Configuration Settings
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "value"]
SIMILARITY_MARGIN = float(os.getenv("SIMILARITY_MARGIN", "0.03"))
AUTO_ASSIGN_THRESHOLD = float(os.getenv("AUTO_ASSIGN_THRESHOLD", "0.86"))
CANDIDATE_THRESHOLD = float(os.getenv("CANDIDATE_THRESHOLD", "0.70"))
CENTROID_UPDATE_THRESHOLD = float(os.getenv("CENTROID_UPDATE_THRESHOLD", "0.90"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "embedding-v1")
POLICY_VERSION = os.getenv("POLICY_VERSION", "policy-v1")
EVENT_DELIVERY_MODE = os.getenv("EVENT_DELIVERY_MODE", "polling")
EVENT_WEBHOOK_URL = os.getenv("EVENT_WEBHOOK_URL", "")
