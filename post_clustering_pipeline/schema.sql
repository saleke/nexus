CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS event_hubs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    member_count INT NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    discourse_type VARCHAR(50) NOT NULL DEFAULT 'event',
    seed_post_id INT,
    merged_into_id INT REFERENCES event_hubs(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    handle TEXT NOT NULL UNIQUE,
    summary TEXT,
    centroid vector(384)
);

CREATE TABLE IF NOT EXISTS posts (
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
    post_seq BIGINT NOT NULL DEFAULT 0,
    event_id INT REFERENCES event_hubs(id) ON DELETE SET NULL,
    embedding vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    entities text[]
);

CREATE TABLE IF NOT EXISTS unclustered_posts_buffer (
    post_id INT PRIMARY KEY REFERENCES posts(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clustering_feedback_log (
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

CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(50) PRIMARY KEY,
    value NUMERIC(4,3) NOT NULL,
    value_text TEXT
);

INSERT INTO system_config (key, value) VALUES ('global_similarity_threshold', 0.880)
ON CONFLICT (key) DO NOTHING;

-- Idempotent backfill for databases created before value_text existed.
ALTER TABLE system_config ADD COLUMN IF NOT EXISTS value_text TEXT;

-- Named to match the canonical deployed schema so re-applying is a no-op.
CREATE INDEX IF NOT EXISTS posts_embedding_idx ON posts USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_event_hubs_last_updated ON event_hubs(last_updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_event_timeline ON posts(event_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_posts_active_event_timeline
ON posts(event_id, created_at ASC, id ASC)
WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_posts_event_engagement ON posts(event_id, engagement_score DESC);
CREATE INDEX IF NOT EXISTS idx_posts_assignment_status ON posts(assignment_status, assignment_updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_posts_source_external_id ON posts(source_id, external_post_id) WHERE external_post_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_feedback_analysis ON clustering_feedback_log(feedback_type, created_at DESC);

CREATE TABLE IF NOT EXISTS model_registry (
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

CREATE TABLE IF NOT EXISTS integration_outbox (
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
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON integration_outbox(delivery_status, available_at, id);
CREATE INDEX IF NOT EXISTS idx_outbox_lease ON integration_outbox(lease_until, id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_outbox_post_event_type ON integration_outbox(post_id, event_type) WHERE post_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_posts_processing_sweeper ON posts(assignment_status, assignment_updated_at) WHERE assignment_status = 'processing';
CREATE INDEX IF NOT EXISTS idx_event_hubs_redirect ON event_hubs(id, merged_into_id) WHERE is_active = FALSE;

CREATE INDEX IF NOT EXISTS idx_active_event_hubs_hnsw 
ON event_hubs USING hnsw (centroid vector_cosine_ops) 
WHERE is_active = TRUE AND centroid IS NOT NULL;

-- =====================================================================
-- Control plane (de-black-boxing): decision journal, feedback rollups,
-- versioned policy history, admin audit, and per-consumer auth.
-- All statements are idempotent so re-applying schema.sql is safe.
-- =====================================================================

-- Decision journal: one row per final assignment decision (assigned /
-- candidate / unassigned / noise), written in the SAME transaction as the
-- status change. Raw material for precision/coverage estimates, threshold
-- tuning, drift detection, and the owner-facing decisions inspector.
CREATE TABLE IF NOT EXISTS assignment_decision_log (
    id BIGSERIAL PRIMARY KEY,
    post_id INT NOT NULL,
    event_id INT,
    similarity DOUBLE PRECISION,
    runner_similarity DOUBLE PRECISION,
    threshold_used DOUBLE PRECISION,
    margin_budget DOUBLE PRECISION,
    status VARCHAR(20) NOT NULL,
    confidence DOUBLE PRECISION,
    policy_version VARCHAR(64),
    model_version VARCHAR(64),
    reason VARCHAR(200),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_decision_log_post ON assignment_decision_log(post_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_log_window ON assignment_decision_log(status, created_at);
CREATE INDEX IF NOT EXISTS idx_decision_log_post_status ON assignment_decision_log(post_id, status);

-- Periodic per-window quality rollups. 'source' separates human feedback from
-- system-decided counters so the system can never grade itself.
CREATE TABLE IF NOT EXISTS feedback_rollups (
    id BIGSERIAL PRIMARY KEY,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    policy_version VARCHAR(64),
    model_version VARCHAR(64),
    source VARCHAR(20) NOT NULL,
    total_decisions BIGINT NOT NULL DEFAULT 0,
    confirmed BIGINT NOT NULL DEFAULT 0,
    removed BIGINT NOT NULL DEFAULT 0,
    dismissed BIGINT NOT NULL DEFAULT 0,
    precision_value DOUBLE PRECISION,
    coverage_value DOUBLE PRECISION,
    unlink_rate DOUBLE PRECISION,
    drift_index DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (window_start, window_end, policy_version, model_version, source)
);
CREATE INDEX IF NOT EXISTS idx_rollups_recent ON feedback_rollups(created_at DESC);

-- Versioned knob/calibration history. status: applied | proposed | rejected.
-- source: manual | owner | autotune. Every threshold/model/policy change is
-- an explainable, reversible row here.
CREATE TABLE IF NOT EXISTS policy_history (
    id BIGSERIAL PRIMARY KEY,
    policy_version VARCHAR(64) NOT NULL,
    knob VARCHAR(64) NOT NULL,
    old_value DOUBLE PRECISION,
    new_value DOUBLE PRECISION,
    status VARCHAR(20) NOT NULL DEFAULT 'applied',
    source VARCHAR(20) NOT NULL DEFAULT 'manual',
    rationale TEXT,
    actor VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_policy_history_knob ON policy_history(knob, created_at DESC);

-- Every control-plane / panel mutation lands here (who, what, from -> to).
CREATE TABLE IF NOT EXISTS admin_audit_log (
    id BIGSERIAL PRIMARY KEY,
    actor VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(64),
    before_state JSONB,
    after_state JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_admin_audit_recent ON admin_audit_log(created_at DESC);

-- Owner accounts for the control plane (email + PBKDF2 password, optional
-- TOTP). Mirrors the live admin_users definition so fresh docker-compose
-- deployments boot the full plane.
CREATE TABLE IF NOT EXISTS admin_users (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    totp_secret TEXT,
    totp_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    last_password_change_at TIMESTAMPTZ
);

-- Per-consumer API access (feedback + admin routes). Legacy shared-token
-- auth keeps working alongside; consumer rows make signal attribution and
-- severing possible.
CREATE TABLE IF NOT EXISTS api_consumers (
    id SERIAL PRIMARY KEY,
    consumer_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    token_hash VARCHAR(64) NOT NULL,
    rate_limit_per_minute INT NOT NULL DEFAULT 60,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_consumers_active ON api_consumers(is_active);

-- Reversible soft-merges. The snapshot (post ids + source centroid + counts
-- at merge time) makes re-opening deterministic: only snapshot members rebound,
-- fresh arrivals in the target are never disturbed.
CREATE TABLE IF NOT EXISTS hub_merges (
    id SERIAL PRIMARY KEY,
    source_event_id INT NOT NULL,
    target_event_id INT NOT NULL,
    initiated_by VARCHAR(20) NOT NULL DEFAULT 'client',
    status VARCHAR(20) NOT NULL DEFAULT 'merged',
    snapshot_post_ids JSONB NOT NULL,
    snapshot_member_count INT NOT NULL,
    snapshot_centroid_source TEXT,
    opened_note TEXT,
    reopened_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reopened_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_hub_merges_hubs ON hub_merges(source_event_id, target_event_id, status);

-- =====================================================================
-- Hub identity (v3) + first-class membership ledger.
--   * Seed/identity separation: the hub's label lives on event_hubs as
--     title/handle; seed_post_id is lineage only (renamed from anchor_post_id
--     so the old column's identity role is unambiguous).
--   * hub_members: one row per admission; live membership is a closed
--     interval (departed_at NULL = current). Mirrors the single-valued
--     posts.event_id pointer in the same transaction.
-- Re-applying is safe for both fresh installs and live DBs.
-- =====================================================================
DO $$
BEGIN
    -- Column renamed in the canonical DDL; on a live DB this is a no-op
    -- once applied (undefined_column is swallowed), so it stays idempotent.
    ALTER TABLE event_hubs RENAME COLUMN anchor_post_id TO seed_post_id;
EXCEPTION WHEN undefined_column THEN
    NULL;
END $$;

ALTER TABLE event_hubs ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE event_hubs ADD COLUMN IF NOT EXISTS handle TEXT;
ALTER TABLE event_hubs ADD COLUMN IF NOT EXISTS summary TEXT;

CREATE TABLE IF NOT EXISTS hub_members (
    id BIGSERIAL PRIMARY KEY,
    post_id INT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    hub_id INT NOT NULL REFERENCES event_hubs(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL DEFAULT 'member',
    admitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    departed_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_hub_members_live_post
    ON hub_members(post_id) WHERE departed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_hub_members_hub_timeline
    ON hub_members(hub_id, admitted_at, id);
CREATE INDEX IF NOT EXISTS idx_hub_members_post_history
    ON hub_members(post_id, admitted_at DESC);

-- Desk content-search (ILIKE '%...%') is exactly the trigram case: without
-- these, every keystroke in the merge desk scans the whole posts/hubs tables.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_posts_content_trgm
    ON posts USING gin (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_hubs_title_trgm
    ON event_hubs USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_hubs_handle_trgm
    ON event_hubs USING gin (handle gin_trgm_ops);
