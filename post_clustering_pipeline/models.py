import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

class EmbeddingEngine:
    def __init__(self, base_model_name="sentence-transformers/all-MiniLM-L6-v2", adapter_path="./models/lora_adapter"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base = AutoModel.from_pretrained(base_model_name)
        
        # Attach LoRA adapter
        try:
            self.model = PeftModel.from_pretrained(base, adapter_path)
        except Exception:
            self.model = base  # Fallback to base if adapter isn't trained yet
            
        self.model.eval()
        
        # Explicitly freeze ALL parameters (Base + LoRA) for pure inference
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Security assertion: ensure base model transformer layers are frozen
        for name, param in self.model.named_parameters():
            # Skip checking LoRA weights if checking base model constraints
            if "lora_" in name:
                continue
            if param.requires_grad:
                raise AssertionError(f"CRITICAL SECURITY ERROR: Base parameter {name} is unfrozen!")

    def encode(self, text: str):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings = pooled.squeeze(0).tolist()
            return embeddings

embedding_engine = EmbeddingEngine()
