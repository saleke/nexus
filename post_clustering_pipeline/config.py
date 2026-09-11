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
