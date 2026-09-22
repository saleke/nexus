-- Repost folding: collapse duplicate-content floods within a hub.
--
-- A hub that receives 1k reposts of one body must not present as "1k members":
-- posts carry a content_sig (computed at ingest) and the fold keeps ONE
-- canonical post per (event_id, content_sig), marking the rest as reposts of
-- it. event_hubs.member_count then means DISTINCT content signals, with
-- repost_count carrying the folded copies.

ALTER TABLE event_hubs ADD COLUMN IF NOT EXISTS repost_count INT NOT NULL DEFAULT 0;

ALTER TABLE posts ADD COLUMN IF NOT EXISTS content_sig BIGINT NOT NULL DEFAULT 0;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS repost_of_id INT REFERENCES posts(id) ON DELETE SET NULL;

-- Fold lookup + count recompute: group by (event_id, content_sig) per hub.
CREATE INDEX IF NOT EXISTS idx_posts_hub_sig ON posts(event_id, content_sig);
-- Repost lineage lookups (search/audit).
CREATE INDEX IF NOT EXISTS idx_posts_repost_of ON posts(repost_of_id) WHERE repost_of_id IS NOT NULL;