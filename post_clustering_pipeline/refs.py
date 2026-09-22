"""Human-readable identity layer: content-anchored handles for hubs & posts.

Bare BIGSERIAL ids are machine keys; no human can hold "hub #8712" vs
"hub #9912" in working memory. Every human-facing surface (panel merge desk,
unlink/confirm flows, outbox event payloads, client responses) renders a
*reference block*: the representative content preview (+ type + member count
for hubs, + author for posts), with the id kept as subtext so machines stay
happy too.

References are point-in-time descriptors represented by ids inside them, so
they are stable enough to label a stream but never load-bearing for logic.
"""
from __future__ import annotations

import re

from .db import get_db_cursor, extract_val

PREVIEW_CHARS = 84

# LIKE metacharacters are escaped so the operator's search input is matched
# literally — typing "_" or "%" must not match every row.
_LIKE_ESCAPE = {"\\": "\\\\", "%": "\\%", "_": "\\_"}


def _like_pattern(q: str) -> str:
    escaped = "".join(_LIKE_ESCAPE.get(ch, ch) for ch in q)
    return f"%{escaped}%"


def _preview(text: str | None) -> str | None:
    if not text:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1].rstrip() + "…"


def title_from_seed(cur, seed_post_id: int, fallback: str | None = None) -> str:
    """One-shot display title for a hub from its seed post's content. Identity
    is captured AT BIRTH onto the hub row; the seed's content is never
    re-read to render the hub afterwards, so a deleted/relabeled seed cannot
    silently rename the hub."""
    cur.execute("SELECT content FROM posts WHERE id = %s AND deleted_at IS NULL;", (seed_post_id,))
    r = cur.fetchone()
    return _preview(extract_val(r, "content", 0)) or fallback or f"hub-{seed_post_id}"


def unique_handle(cur, base: str, hub_prefix: int | None = None) -> str:
    """Stable URL-safe handle for a hub, guaranteed unique in event_hubs.

    ``hub_prefix`` (the hub's id once known) doubles as the deterministic
    fallback so collision-free handles exist even for content that slugifies
    to nothing.
    """
    prefix = f"hub-{hub_prefix}" if hub_prefix else "hub"
    s = re.sub(r"[^a-z0-9]+", "-", (base or "").lower()).strip("-")
    s = s[:64].rstrip("-")
    candidate = s or prefix
    n = 2
    while True:
        cur.execute("SELECT 1 FROM event_hubs WHERE handle = %s;", (candidate,))
        if not cur.fetchone():
            return candidate
        candidate = f"{s or prefix}-{n}"
        n += 1


def post_reference(cur, post_id: int) -> dict:
    cur.execute(
        "SELECT id, content, user_id, platform, assignment_status, event_id, created_at "
        "FROM posts WHERE id = %s;",
        (post_id,)
    )
    r = cur.fetchone()
    if not r:
        return {"post_id": post_id, "content_preview": None, "status": "not_found"}
    return {
        "post_id": extract_val(r, "id", 0),
        "content_preview": _preview(extract_val(r, "content", 1)),
        "author_id": extract_val(r, "user_id", 2),
        "platform": extract_val(r, "platform", 3),
        "status": extract_val(r, "assignment_status", 4),
        "event_id": extract_val(r, "event_id", 5),
        "created_at": extract_val(r, "created_at", 6),
    }


def hub_reference(cur, hub_id: int, member_previews: int = 0) -> dict:
    """Handle for a hub: its title/handle (hub-owned identity) plus discourse
    type, member count, and a small member-preview trailer so humans can tell
    "budget debate #3" from "budget debate #4".

    Identity lives ON the hub row - the label no longer walks back to a member
    post's content, so a hub survives its seed post leaving/deleting, and
    per-surface renders are a single cheap row read. ``seed_post_id`` is kept
    as lineage only.
    """
    cur.execute(
        "SELECT id, status, discourse_type, member_count, repost_count, seed_post_id, title, handle, summary, created_at, last_updated_at "
        "FROM event_hubs WHERE id = %s;",
        (hub_id,)
    )
    r = cur.fetchone()
    if not r:
        return {"event_id": hub_id, "label": f"hub-{hub_id}", "status": "not_found"}

    title = extract_val(r, "title", 6) or f"hub-{hub_id}"
    label = title if len(title) <= PREVIEW_CHARS else title[: PREVIEW_CHARS - 1].rstrip() + "…"

    trailer = []
    if member_previews > 0:
        cur.execute(
            "SELECT content FROM posts WHERE event_id = %s AND deleted_at IS NULL AND repost_of_id IS NULL "
            "ORDER BY created_at ASC LIMIT %s;",
            (hub_id, member_previews + 1)
        )
        for m in (cur.fetchall() or []):
            preview = _preview(extract_val(m, "content", 0))
            if preview:
                trailer.append(preview)

    label = label or f"hub-{hub_id}"
    return {
        "event_id": hub_id,
        "label": label,
        "handle": extract_val(r, "handle", 7),
        "title": title,
        "summary": extract_val(r, "summary", 8),
        "discourse_type": extract_val(r, "discourse_type", 2) or "event",
        "member_count": int(extract_val(r, "member_count", 3) or 0),
        "repost_count": int(extract_val(r, "repost_count", 4) or 0),
        "status": extract_val(r, "status", 1) or "active",
        "anchor_preview": label,
        "seed_post_id": extract_val(r, "seed_post_id", 5),
        "member_previews": trailer[:member_previews],
        "created_at": extract_val(r, "created_at", 9),
        "last_updated_at": extract_val(r, "last_updated_at", 10),
    }


def search_refs(cur, q: str, limit: int = 20, kind: str | None = None) -> dict:
    """Find hubs and posts by bare id (exact) or by content / anchor text
    (ILIKE). Lets a human paste ANY fragment they remember and get labeled
    cards, instead of typing ids they cannot know.

    ``kind`` narrows the search to one side ('hub' | 'post'); None searches
    both. User input is matched LITERALLY (LIKE metacharacters escaped), so a
    query of "_" or "%" cannot match everything and look broken. Hub identity
    now lives on the hub row, so title/handle are matched along with seed and
    member content.
    """
    limit = max(1, min(limit, 100))
    q = (q or "").strip()
    out = {"hubs": [], "posts": []}

    if q.isdigit():
        if kind in (None, "hub"):
            hub = hub_reference(cur, int(q))
            if hub.get("status") != "not_found":
                out["hubs"].append(hub)
        if kind in (None, "post"):
            post = post_reference(cur, int(q))
            if post.get("status") != "not_found":
                out["posts"].append(post)
        return out

    if q:
        like = _like_pattern(q)
        if kind in (None, "hub"):
            cur.execute(
                """
                SELECT h.id
                FROM event_hubs h
                LEFT JOIN posts a ON a.id = h.seed_post_id
                WHERE h.is_active = TRUE AND (
                    h.id::text = %s
                    OR h.title ILIKE %s
                    OR h.handle ILIKE %s
                    OR a.content ILIKE %s
                    OR EXISTS (SELECT 1 FROM posts p WHERE p.event_id = h.id AND p.content ILIKE %s)
                )
                ORDER BY h.member_count DESC
                LIMIT %s;
                """,
                (q, like, like, like, like, limit)
            )
            for hr in (cur.fetchall() or []):
                out["hubs"].append(hub_reference(cur, extract_val(hr, "id", 0)))
        if kind in (None, "post"):
            cur.execute(
                """
                SELECT id FROM posts
                WHERE deleted_at IS NULL AND content ILIKE %s
                ORDER BY created_at DESC LIMIT %s;
                """,
                (like, limit)
            )
            for pr in (cur.fetchall() or []):
                out["posts"].append(post_reference(cur, extract_val(pr, "id", 0)))
    return out


def resolve_for_human(cur, hub_id: int) -> dict:
    """Canonical hub resolved to its reference block, so callers never send a
    stale/redirected hub id to a human - they get the effective hub's handle."""
    from .merges import resolve_canonical_hub

    canonical, redirected = resolve_canonical_hub(cur, hub_id)
    ref = hub_reference(cur, canonical)
    ref["was_redirected"] = redirected
    ref["requested_event_id"] = hub_id
    return ref


def with_hub_label(state: str, hub_ref: dict | None) -> str:
    """Human label for an event/hub with graceful fallback for dead refs."""
    if not hub_ref or not hub_ref.get("label"):
        return state
    return f"{state}: {hub_ref['label']}"