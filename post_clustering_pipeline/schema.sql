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
    platform VARCHAR(50) NOT NULL DEFAULT 'unknown',
    source_id VARCHAR(200) NOT NULL DEFAULT 'legacy',
    external_post_id VARCHAR(300),
    external_author_id VARCHAR(300),
    published_at TIMESTAMP WITH TIME ZONE,
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    assignment_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    assignment_confidence NUMERIC(5,4),
    assignment_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
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
    actor VARCHAR(100) NOT NULL DEFAULT 'system',
    model_version VARCHAR(100),
    policy_version VARCHAR(100),
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
CREATE INDEX idx_posts_assignment_status ON posts(assignment_status, assignment_updated_at DESC);
CREATE UNIQUE INDEX uq_posts_source_external_id ON posts(source_id, external_post_id) WHERE external_post_id IS NOT NULL;
CREATE INDEX idx_feedback_analysis ON clustering_feedback_log(feedback_type, created_at DESC);

CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(100) NOT NULL UNIQUE,
    adapter_path TEXT NOT NULL,
    validation_precision NUMERIC(5,4),
    validation_recall NUMERIC(5,4),
    noise_false_positive_rate NUMERIC(5,4),
    status VARCHAR(20) NOT NULL DEFAULT 'candidate',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    promoted_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE integration_outbox (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    post_id INT REFERENCES posts(id) ON DELETE CASCADE,
    event_id INT REFERENCES event_hubs(id) ON DELETE CASCADE,
    payload JSONB NOT NULL,
    delivery_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    attempts INT NOT NULL DEFAULT 0,
    schema_version INT NOT NULL DEFAULT 1,
    consumer VARCHAR(200),
    lease_token VARCHAR(100),
    lease_until TIMESTAMP WITH TIME ZONE,
    available_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    delivered_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_outbox_pending ON integration_outbox(delivery_status, available_at, id);
CREATE INDEX idx_outbox_lease ON integration_outbox(lease_until, id);

CREATE INDEX ON event_hubs USING hnsw (centroid vector_cosine_ops);
