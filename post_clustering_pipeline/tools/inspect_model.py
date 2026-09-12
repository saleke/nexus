"""Inspect the active/base embedding model and run embedding sanity probes."""
from __future__ import annotations

import argparse
import json
import os

from ..config import BASE_MODEL_NAME, LORA_ADAPTER_DIR
from ..models import EmbeddingEngine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=BASE_MODEL_NAME)
    p.add_argument("--adapter", default=LORA_ADAPTER_DIR)
    p.add_argument("--json", action="store_true", help="Print machine-readable details")
    args = p.parse_args()
    engine = EmbeddingEngine(args.base_model, args.adapter)
    params = sum(parameter.numel() for parameter in engine.model.parameters())
    trainable = sum(parameter.numel() for parameter in engine.model.parameters() if parameter.requires_grad)
    probes = ["A hurricane warning was issued for Miami.", "The Federal Reserve announced a rate decision.", "I made pasta for dinner."]
    vectors = [engine.encode(text) for text in probes]
    report = {
        "base_model": args.base_model,
        "adapter_path": os.path.abspath(args.adapter),
        "adapter_exists": os.path.isdir(args.adapter),
        "parameters": params,
        "trainable_parameters": trainable,
        "embedding_dimensions": len(vectors[0]),
        "probe_norms": [sum(value * value for value in vector) ** 0.5 for vector in vectors],
        "model_class": type(engine.model).__name__,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return

    adapter_ok = report["adapter_exists"] and report["model_class"] == "PeftModel"
    norms_ok = all(0.98 <= norm <= 1.02 for norm in report["probe_norms"])
    print("Model Check")
    print("===========")
    print(f"Overall status: {'READY' if adapter_ok and norms_ok else 'USING BASE MODEL / CHECK CONFIGURATION'}")
    print(f"Model used: {report['base_model']}")
    print(f"Custom improvement file: {'loaded' if adapter_ok else 'not loaded (base model is being used)'}")
    print(f"Understanding size: {report['embedding_dimensions']} numbers per post")
    print(f"Learning enabled right now: {'no (inference mode)' if report['trainable_parameters'] == 0 else 'yes'}")
    print(f"Basic output check: {'passed' if norms_ok else 'needs attention'}")
    print("\nThis check confirms which model is active and whether it is producing usable results.")


if __name__ == "__main__":
    main()
