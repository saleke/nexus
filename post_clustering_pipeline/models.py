import logging
import threading
from datetime import datetime, timezone

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from .config import BASE_MODEL_NAME, LORA_ADAPTER_DIR, REQUIRE_LORA_ADAPTER, MODEL_VERSION

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(self, base_model_name: str = BASE_MODEL_NAME, adapter_path: str = LORA_ADAPTER_DIR):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base = AutoModel.from_pretrained(base_model_name)
        # RETAIN the base: a later reload_adapter builds a fresh PeftModel over
        # this same graph (never re-downloads, never mutates in place), then
        # rebinds atomically via swap_model under _MODEL_SWAP_LOCK.
        self.base_model = base

        self.adapter_loaded = False
        try:
            self.model = PeftModel.from_pretrained(base, adapter_path)
            self.adapter_loaded = True
        except Exception as exc:
            if REQUIRE_LORA_ADAPTER:
                raise RuntimeError(f"Unable to load required LoRA adapter from {adapter_path}") from exc
            logger.warning("LoRA adapter unavailable; using base model: %s", exc)
            self.model = base

        self.freeze_all()
        self.model.eval()

    def swap_model(self, fresh) -> None:
        """Atomic rebind of the served weights. ``encode_batch``/``encode`` grab
        the same ``_MODEL_SWAP_LOCK`` and resolve ``self.model`` under it, so a
        swap is either fully-before or fully-after an in-flight batch - never
        mid-forward (zero downtime: the old tensor graph stays intact)."""
        with _MODEL_SWAP_LOCK:
            self.model = fresh

    def freeze_all(self):
        """Explicitly freeze all parameters for inference."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Security assertion: ensure base model transformer layers are frozen
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                continue
            if param.requires_grad:
                raise AssertionError(f"CRITICAL SECURITY ERROR: Base parameter {name} is unfrozen!")

    def unfreeze_lora(self):
        """Unfreeze only LoRA adapter parameters for active learning fine-tuning."""
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Batch encode multiple texts efficiently using vectorized PyTorch operations."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                # NOTE: embeddings use the leading 256-token window only.
                # Windowing/tiling was measured and rejected: for realistic
                # long-form recaps the head window is always the best topic
                # representative (0.50/0.27/0.54/0.47 head vs 0.24/0.43 tail
                # on a live corpus), and pooling tricks regressed some posts
                # while failing to recover tail-buried keywords. Topics buried
                # past the window are instead recovered by full-text entity
                # anchoring during assignment/birth (see assignment.py).
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                )
                outputs = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
                pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.extend(pooled.cpu().tolist())

        return all_embeddings

    def encode(self, text: str) -> list[float]:
        """Encode a single text string."""
        return self.encode_batch([text])[0]


_embedding_engine: EmbeddingEngine | None = None


_MODEL_SWAP_LOCK = threading.RLock()

# The LIVE adapter generation the process is CURRENTLY serving. Read from this
# (never from config defaults at era-anchor time) so policy-era filters (#11)
# and the swapped-in encoder always agree on the same generation - a reload mid-
# stream can never leave the era anchor pointing at an adapter the factory is
# no longer serving. Promotion announces by writing here, atomically with the
# weight swap (build-then-swap hold the same lock).
_active_model_version: str = MODEL_VERSION
_last_loaded_at: datetime | None = None
_loaded_adapter_path: str | None = None


def get_active_model_version() -> str:
    """Generation the live encoder currently serves (era anchor source)."""
    return _active_model_version


def reload_adapter(adapter_path: str, *, model_version: str, cur=None) -> bool:
    """Zero-downtime adapter swap: build a fresh PeftModel on the RETAINED
    base, then rebind under an exclusive lock (never mutate in place, so a
    concurrent ``encode_batch`` keeps its resolved tensor graph intact).

    Returns True iff this process actually swapped (another worker that already
    promoted the same version returns False -> no double-work).
    """
    global _active_model_version, _last_loaded_at, _loaded_adapter_path
    with _MODEL_SWAP_LOCK:
        if model_version == _active_model_version and _loaded_adapter_path == adapter_path:
            return False
        engine = get_embedding_engine()
        if engine.base_model is None:
            return False
        try:
            fresh = PeftModel.from_pretrained(engine.base_model, adapter_path)
        except Exception as exc:
            logger.error("adapter swap rejected (0-downtime preserved): %s", exc)
            return False
        fresh.eval()
        for p in fresh.parameters():
            p.requires_grad = False
        _adapter_path = adapter_path
        # Atomic rebind: this process serves the fresh weights now.
        _active_model_version = model_version
        _last_loaded_at = datetime.now(timezone.utc)
        _loaded_adapter_path = adapter_path
        # Swap the engine so subsequent encode_batch calls use the new adapter.
        engine.swap_model(fresh)
        logger.info("adapter swapped: %s -> %s (epoch %s)", _adapter_path, model_version, _last_loaded_at.isoformat())
        return True


def get_embedding_engine() -> EmbeddingEngine:
    """Return the process-local engine, loading model weights on first use."""
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine
