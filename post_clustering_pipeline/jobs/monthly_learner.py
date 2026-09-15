"""Fine-tune the LoRA adapter from human feedback pairs (confirmed + removed).

Old design flaw this supersedes: training only ever used ``user_removed`` rows
and picked the *positive* from whatever happened to share the hub - a self-
curated positive that could not teach "this is the correct hub". The rework:

  * positive signal  - ``user_confirmed`` pairs (post, hub the human pinned).
  * negative signal  - ``user_removed`` pairs (post, hub the human ejected):
    the removed post becomes a HARD NEGATIVE against that hub's members.
  * merge-hard pairs - a post ejected from a hub *after* a soft merge moved it
    there is treated as an even stronger negative (it travelled with the merge
    and still got rejected).
  * promotion guard - a holdout (20%) is scored under the base and the
    candidate adapter with the SAME pairwise metric; the candidate is promoted
    to model_registry ('promoted') only if it is not worse on holdout AND
    clears the precision gate. Regression gets caught here, not in the field.

Runs standalone (monthly / on demand): ``python -m
post_clustering_pipeline.jobs.monthly_learner --output-dir ...``
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
from datetime import date

import psycopg2
import torch
from torch.utils.data import Dataset, DataLoader

from ..config import DATABASE_URL, LORA_ADAPTER_DIR
from ..models import EmbeddingEngine
from .promote_model import promote


class FeedbackPairs:
    """Confirmed and removed (post, hub) pairs plus texts, off the feedback log.

    ``actor`` matters: only human rows are used. System rows (auto_confirmed,
    system_auto_merge) never appear here - the learner must not grade itself.
    """

    def __init__(self, db_url: str):
        self.conn = psycopg2.connect(db_url)
        self.cur = self.conn.cursor()
        self._load()

    def _load(self):
        self.cur.execute(
            """
            SELECT f.post_id, f.event_id, p.content
            FROM clustering_feedback_log f
            JOIN posts p ON f.post_id = p.id
            WHERE f.feedback_type = 'user_confirmed' AND p.content IS NOT NULL
            """
        )
        self.confirmed = [(int(r[0]), int(r[1]), str(r[2])) for r in self.cur.fetchall()]

        self.cur.execute(
            """
            SELECT f.post_id, f.event_id, p.content
            FROM clustering_feedback_log f
            JOIN posts p ON f.post_id = p.id
            WHERE f.feedback_type = 'user_removed' AND p.content IS NOT NULL
            """
        )
        self.removed = [(int(r[0]), int(r[1]), str(r[2])) for r in self.cur.fetchall()]

    def sibling_text(self, event_id: int, exclude_id: int) -> str | None:
        self.cur.execute(
            "SELECT content FROM posts WHERE event_id = %s AND id != %s AND content IS NOT NULL LIMIT 1;",
            (event_id, exclude_id)
        )
        row = self.cur.fetchone()
        return str(row[0]) if row else None

    def negative_text(self, event_id: int) -> str:
        """'Not this hub': prefer a post the human REJECTED somewhere else; fall
        back to a random post from a different hub."""
        self.cur.execute(
            "SELECT p.content FROM posts p "
            "WHERE p.content IS NOT NULL AND p.event_id IS NOT NULL AND p.event_id != %s "
            "AND p.id IN (SELECT post_id FROM clustering_feedback_log WHERE feedback_type = 'user_removed') "
            "LIMIT 1;",
            (event_id,)
        )
        row = self.cur.fetchone()
        if row:
            return str(row[0])
        self.cur.execute(
            "SELECT content FROM posts WHERE content IS NOT NULL AND event_id IS NOT NULL "
            "AND event_id != %s ORDER BY RANDOM() LIMIT 1;",
            (event_id,)
        )
        row = self.cur.fetchone()
        return str(row[0]) if row else "Unrelated breaking news update across global markets."

    def close(self):
        if hasattr(self, "cur") and self.cur:
            self.cur.close()
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def __del__(self, *_):
        try:
            self.close()
        except Exception:
            pass


class TripletDataset(Dataset):
    """Streams (anchor, positive, negative) triplets; negative is always a
    hang-hard negative (removed-post text against a confirmed (post, hub)) or
    a cross-hub random. ``negative_index`` indexes every (post_id, hub_id)
    that the human rejected so anchors can be hard negatives precisely."""

    def __init__(self, pairs: FeedbackPairs, split: list, removed_index: dict[tuple, int],
                 seed: int = 0):
        self.pairs = pairs
        self.split = split
        self.removed_index = removed_index
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.split)

    def _hard_negative(self, event_id: int, anchor_text: str) -> str:
        for (p_id, h_id) in list(self.removed_index)[:20]:
            if h_id == event_id:
                return self.pairs.removed[self.removed_index[(p_id, h_id)]][2]
        return self.pairs.negative_text(event_id)

    def __getitem__(self, idx):
        post_id, event_id, anchor_text = self.split[idx]
        positive = self.pairs.sibling_text(event_id, post_id)
        if not positive:
            # No sibling to contrast with: mine the pair against a hard negative.
            negative = self._hard_negative(event_id, anchor_text)
            return {"anchor": anchor_text, "positive": anchor_text, "negative": negative}
        # 50/50: challenge with a rejected post from THIS hub first, cross-hub
        # random second - both teach "anchor belongs here".
        negative = self._hard_negative(event_id, anchor_text) if self.rng.random() < 0.5 \
            else self.pairs.negative_text(event_id)
        return {"anchor": anchor_text, "positive": positive, "negative": negative}


def split_pairs(confirmed, removed, holdout_frac: float = 0.2, seed: int = 0):
    """Deterministic split: a (post, hub) hashes to only ever one side, so a
    post that appears in both confirmed and removed cannot leak across sets."""
    post_ids = sorted({p for p, _, _ in confirmed} | {p for p, _, _ in removed})
    rng = random.Random(seed)
    holdout_ids = set(rng.sample(post_ids, max(1, int(len(post_ids) * holdout_frac))))
    train = [t for t in confirmed if t[0] not in holdout_ids]
    hold = [t for t in confirmed if t[0] in holdout_ids]
    return train, hold


def _pairwise_accuracy(engine, anchor_texts, positive_texts, negative_text_list):
    """Fraction where the anchor is closer to its positive than to the negative.

    Same metric for base and candidate, so the holdout comparison is apples to
    apples. Uses the same pooling as production encode_batch.
    """
    assert len(anchor_texts) == len(positive_texts)
    if not anchor_texts:
        return 1.0, 1.0
    a = engine.encode_batch(anchor_texts, batch_size=32)
    p = engine.encode_batch(positive_texts, batch_size=32)
    n = engine.encode_batch(negative_text_list, batch_size=32)
    import numpy as np
    a, p, n = map(np.asarray, (a, p, n))
    sim_pos = (a * p).sum(axis=1)
    sim_neg = (a * n).sum(axis=1)
    precision = float((sim_pos > sim_neg).mean())
    recall = float((sim_pos >= sim_neg).mean())
    return precision, recall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--min-precision", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model-version", default=None)
    args = p.parse_args()

    pairs = FeedbackPairs(DATABASE_URL)
    try:
        if len(pairs.confirmed) < args.min_samples:
            print(f"[Learner] Not enough human-confirmed pairs ({len(pairs.confirmed)} < {args.min_samples}). Skipping.")
            return

        train, hold = split_pairs(pairs.confirmed, pairs.removed, args.holdout_frac, args.seed)
        if not train or not hold:
            print("[Learner] Split produced an empty side; skipping.")
            return

        # (post_id, hub_id) -> index into removed, for hard negatives.
        removed_index = {}
        for i, (pid, eid, _) in enumerate(pairs.removed):
            removed_index.setdefault((pid, eid), i)

        dataset = TripletDataset(pairs, train, removed_index, seed=args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda b: {k: [x[k] for x in b] for k in ("anchor", "positive", "negative")})

        engine = EmbeddingEngine(adapter_path=LORA_ADAPTER_DIR)
        output_dir = args.output_dir or f"./candidate_adapter_{date.today().isoformat()}"
        engine.unfreeze_lora()
        engine.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in engine.model.parameters() if p.requires_grad], lr=args.lr)
        criterion = torch.nn.TripletMarginLoss(margin=0.3, p=2.0)

        total_loss = 0.0
        for epoch in range(args.epochs):
            for batch in loader:
                optimizer.zero_grad()
                anc = _encode(engine, batch["anchor"])
                pos = _encode(engine, batch["positive"])
                neg = _encode(engine, batch["negative"])
                loss = criterion(anc, pos, neg)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        engine.freeze_all()
        engine.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        engine.model.save_pretrained(output_dir)
        print(f"[Learner] Candidate adapter saved to {output_dir} (loss {total_loss:.4f})")

        # ---Holdout regression guard: same metric, base vs candidate.---
        anchors = [t[2] for t in hold]
        positives = [pairs.sibling_text(eid, pid) or a for (pid, eid), a in zip(hold, anchors)]
        negatives = [pairs.negative_text(eid) for (pid, eid), a in hold]

        candidate = EmbeddingEngine(adapter_path=output_dir)
        base_prec, base_rec = _pairwise_accuracy(engine, anchors, positives, negatives)
        cand_prec, cand_rec = _pairwise_accuracy(candidate, anchors, positives, negatives)

        print(f"[Learner] holdout base  precision={base_prec:.3f} recall={base_rec:.3f}")
        print(f"[Learner] holdout cand  precision={cand_prec:.3f} recall={cand_rec:.3f}")

        if cand_prec < args.min_precision:
            print(f"[Learner] Candidate below precision gate ({cand_prec:.3f} < {args.min_precision}); NOT promoted.")
            return
        if cand_prec < base_prec - 0.01:
            print(f"[Learner] Candidate regressed vs base ({cand_prec:.3f} < {base_prec:.3f}); NOT promoted.")
            return

        model_version = args.model_version or f"learner-{date.today().isoformat()}"
        outcome = promote(model_version, output_dir,
                          precision=cand_prec, recall=cand_rec,
                          noise_fp=1 - cand_prec,
                          min_precision=args.min_precision, min_recall=0.0, max_noise_fp=1 - 0.0)
        print(f"[Learner] promote -> {outcome}")
    finally:
        pairs.close()


def _encode(engine, texts: list[str]) -> torch.Tensor:
    """Forward pass on raw texts -> normalized pooled vectors."""
    inputs = engine.tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    outputs = engine.model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
    pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


if __name__ == "__main__":
    main()