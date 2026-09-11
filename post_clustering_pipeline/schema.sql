CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE event_hubs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    centroid vector(384)
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    content TEXT NOT NULL,
    has_media BOOLEAN DEFAULT FALSE,
    engagement_score INT DEFAULT 0,
    ground_truth INT,
    event_id INT REFERENCES event_hubs(id) ON DELETE SET NULL,
    embedding vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE unclustered_posts_buffer (
    post_id INT PRIMARY KEY REFERENCES posts(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE clustering_feedback_log (
    id SERIAL PRIMARY KEY,
    post_id INT REFERENCES posts(id) ON DELETE CASCADE,
    event_id INT REFERENCES event_hubs(id) ON DELETE CASCADE,
    initial_similarity_score NUMERIC(4,3) NOT NULL,
    feedback_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE system_config (
    key VARCHAR(50) PRIMARY KEY,
    value NUMERIC(4,3) NOT NULL
);

INSERT INTO system_config (key, value) VALUES ('global_similarity_threshold', 0.860);

CREATE INDEX ON posts USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_event_hubs_last_updated ON event_hubs(last_updated_at DESC);
CREATE INDEX idx_posts_event_timeline ON posts(event_id, created_at ASC);
CREATE INDEX idx_posts_event_engagement ON posts(event_id, engagement_score DESC);
CREATE INDEX idx_feedback_analysis ON clustering_feedback_log(feedback_type, created_at DESC);

CREATE INDEX ON event_hubs USING hnsw (centroid vector_cosine_ops);
