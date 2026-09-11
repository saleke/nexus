import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import psycopg2
try:
    from .db import get_db_connection
except ImportError:
    from db import get_db_connection

class ContrastiveFeedbackDataset(Dataset):
    def __init__(self, db_url):
        self.conn = psycopg2.connect(db_url)
        self.cur = self.conn.cursor()
        
        self.cur.execute(
            """
            SELECT f.post_id, f.event_id, p.embedding::text
            FROM clustering_feedback_log f
            JOIN posts p ON f.post_id = p.id
            WHERE f.feedback_type = 'user_removed';
            """
        )
        self.unlinks = self.cur.fetchall()

    def __len__(self):
        return len(self.unlinks)

    def __getitem__(self, idx):
        post_id, event_id, emb_str = self.unlinks[idx]
        anchor_emb = list(map(float, emb_str.strip("[]").split(",")))
        
        self.cur.execute(
            "SELECT embedding::text FROM posts WHERE event_id = %s AND id != %s LIMIT 1;",
            (event_id, post_id)
        )
        pos_row = self.cur.fetchone()
        if pos_row:
            positive_emb = list(map(float, pos_row[0].strip("[]").split(",")))
        else:
            positive_emb = anchor_emb

        self.cur.execute(
            "SELECT embedding::text FROM posts WHERE event_id IS NULL LIMIT 1;"
        )
        neg_row = self.cur.fetchone()
        if neg_row:
            negative_emb = list(map(float, neg_row[0].strip("[]").split(",")))
        else:
            negative_emb = anchor_emb

        return {
            "anchor": torch.tensor(anchor_emb),
            "positive": torch.tensor(positive_emb),
            "negative": torch.tensor(negative_emb)
        }

    def __del__(self):
        self.cur.close()
        self.conn.close()

def train_lora_contrastive(model, dataloader, optimizer, epochs=1):
    model.train()
    criterion = torch.nn.TripletMarginLoss(margin=0.3, p=2.0)
    
    for epoch in range(epochs):
        for batch in dataloader:
            anchor = batch["anchor"]
            positive = batch["positive"]
            negative = batch["negative"]

            optimizer.zero_grad()
            
            anc_out = model(anchor)
            pos_out = model(positive)
            neg_out = model(negative)

            loss = criterion(anc_out, pos_out, neg_out)
            loss.backward()
            optimizer.step()
