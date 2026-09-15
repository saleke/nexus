import psycopg2
import torch
from torch.utils.data import IterableDataset

from ..embed_io import parse_vector_literal

class StreamingPostDataset(IterableDataset):
    def __init__(self, db_url, time_window_hours=12):
        self.db_url = db_url
        self.time_window = time_window_hours

    def __iter__(self):
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT post_id, embedding::text, created_at 
            FROM unclustered_posts_buffer 
            WHERE created_at >= NOW() - INTERVAL '{self.time_window} hours'
            ORDER BY created_at ASC;
            """
        )
        for row in cur:
            post_id, emb_str, created_at = row
            embedding = parse_vector_literal(emb_str)
            yield {
                "post_id": post_id, 
                "embedding": torch.tensor(embedding), 
                "created_at": created_at
            }
        cur.close()
        conn.close()