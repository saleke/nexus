"""Fine-tune the LoRA adapter using contrastive user feedback with raw text tokenization."""
from __future__ import annotations

import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import psycopg2
from transformers import AutoTokenizer

from ..db import get_db_connection
from ..config import DATABASE_URL, BASE_MODEL_NAME, LORA_ADAPTER_DIR
from ..models import EmbeddingEngine


class ContrastiveFeedbackDataset(Dataset):
    """Dataset extracting raw post text for triplets (anchor, positive, negative)."""

    def __init__(self, db_url: str):
        self.conn = psycopg2.connect(db_url)
        self.cur = self.conn.cursor()

        # Query user-removed feedback with original post content
        self.cur.execute(
            """
            SELECT f.post_id, f.event_id, p.content
            FROM clustering_feedback_log f
            JOIN posts p ON f.post_id = p.id
            WHERE f.feedback_type = 'user_removed'
              AND p.content IS NOT NULL;
            """
        )
        self.unlinks = self.cur.fetchall()

    def __len__(self):
        return len(self.unlinks)

    def __getitem__(self, idx):
        post_id, event_id, anchor_text = self.unlinks[idx]

        # Positive: another post genuinely assigned to this event
        self.cur.execute(
            "SELECT content FROM posts WHERE event_id = %s AND id != %s AND content IS NOT NULL LIMIT 1;",
            (event_id, post_id)
        )
        pos_row = self.cur.fetchone()
        positive_text = pos_row[0] if pos_row else anchor_text

        # Negative: post from another event or candidate noise
        self.cur.execute(
            "SELECT content FROM posts WHERE (event_id IS NULL OR event_id != %s) AND content IS NOT NULL ORDER BY RANDOM() LIMIT 1;",
            (event_id,)
        )
        neg_row = self.cur.fetchone()
        negative_text = neg_row[0] if neg_row else "Unrelated breaking news update across global markets."

        return {
            "anchor": str(anchor_text),
            "positive": str(positive_text),
            "negative": str(negative_text)
        }

    def __del__(self):
        if hasattr(self, "cur") and self.cur:
            self.cur.close()
        if hasattr(self, "conn") and self.conn:
            self.conn.close()


def collate_text_triplets(batch):
    return {
        "anchor": [item["anchor"] for item in batch],
        "positive": [item["positive"] for item in batch],
        "negative": [item["negative"] for item in batch],
    }


def encode_tokenized(model, tokenizer, texts: list[str]) -> torch.Tensor:
    """Run forward pass through transformer on raw texts and return normalized pooled vectors."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
    pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def train_lora_contrastive(
    engine: EmbeddingEngine,
    dataloader: DataLoader,
    output_dir: str,
    epochs: int = 2,
    lr: float = 2e-4
):
    """Fine-tune the LoRA adapter while keeping base transformer weights frozen."""
    engine.unfreeze_lora()
    engine.model.train()

    optimizer = torch.optim.AdamW(
        [p for p in engine.model.parameters() if p.requires_grad],
        lr=lr
    )
    criterion = torch.nn.TripletMarginLoss(margin=0.3, p=2.0)

    total_loss = 0.0
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            anc_out = encode_tokenized(engine.model, engine.tokenizer, batch["anchor"])
            pos_out = encode_tokenized(engine.model, engine.tokenizer, batch["positive"])
            neg_out = encode_tokenized(engine.model, engine.tokenizer, batch["negative"])

            loss = criterion(anc_out, pos_out, neg_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    engine.freeze_all()
    engine.model.eval()

    # Save candidate LoRA weights
    os.makedirs(output_dir, exist_ok=True)
    engine.model.save_pretrained(output_dir)
    print(f"[Learner] LoRA fine-tuning complete. Candidate adapter saved to {output_dir}")
    return total_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output-dir", default="./candidate_adapter")
    args = p.parse_args()

    engine = EmbeddingEngine()
    dataset = ContrastiveFeedbackDataset(DATABASE_URL)
    if len(dataset) < 5:
        print(f"[Learner] Not enough feedback samples ({len(dataset)} found). Skipping training.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_text_triplets)
    train_lora_contrastive(engine, loader, args.output_dir, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
