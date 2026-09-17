-- Deduplicate the HNSW embedding index: an earlier deploy created
-- posts_embedding_idx and a stray posts_embedding_idx1 over the same
-- (embedding, vector_cosine_ops) definition. Keep the named canonical one.
DROP INDEX IF EXISTS posts_embedding_idx1;

-- Panel decisions inspector filters by event and recency.
CREATE INDEX IF NOT EXISTS idx_decision_log_event
    ON assignment_decision_log(event_id, created_at DESC);

-- Per-hub contributor alpha (distinct authors) and per-hub engagement queries.
CREATE INDEX IF NOT EXISTS idx_posts_event_user
    ON posts(event_id, user_id);

-- Learner / feedback-by-post lookups on corrections.
CREATE INDEX IF NOT EXISTS idx_feedback_post
    ON clustering_feedback_log(post_id);

-- Aged-buffer sweep (created_at < NOW() - window) in the event sweeper.
CREATE INDEX IF NOT EXISTS idx_buffer_created
    ON unclustered_posts_buffer(created_at);