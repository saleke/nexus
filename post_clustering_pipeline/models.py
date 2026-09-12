import logging
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from .config import BASE_MODEL_NAME, LORA_ADAPTER_DIR, REQUIRE_LORA_ADAPTER

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(self, base_model_name: str = BASE_MODEL_NAME, adapter_path: str = LORA_ADAPTER_DIR):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base = AutoModel.from_pretrained(base_model_name)

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


def get_embedding_engine() -> EmbeddingEngine:
    """Return the process-local engine, loading model weights on first use."""
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine
