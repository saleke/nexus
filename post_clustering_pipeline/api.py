import uuid
import hmac
import os
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datetime import datetime

from .db import get_db_cursor
from .queues import bounded_push_ingest_hints as queues_bounded_push_ingest_hints
from .config import (
    OUTBOX_MAX_ATTEMPTS, API_AUTH_TOKEN,
    PACKAGE_DIR,
)
from .events import write_outbox
from .membership import close_membership
from .refs import hub_reference
from .nlp import content_signature
from .merges import client_merge, resolve_canonical_hub
from .corrections import unlink_post as unlink_correction, confirm_post as confirm_correction, CorrectionError
from .consumers import authenticate_consumer, rate_limit_check
from .admin import router as admin_router

from . import config as _cfg

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Fail-stop guard: a production deployment without the shared API token is
    almost certainly a misconfiguration that would expose the feed's control
    plane. Refuse to serve instead of serving open."""
    if _cfg.PRODUCTION and not API_AUTH_TOKEN:
        raise RuntimeError(
            "NEXUS_ENV=production requires API_AUTH_TOKEN. Refusing to serve "
            "unauthenticated traffic. Set a strong shared token, e.g. "
            "`openssl rand -hex 32`, and restart."
        )
    yield


app = FastAPI(title="Nexus Event Clustering Platform API", lifespan=_lifespan)
app.include_router(admin_router)
app.mount("/admin/static", StaticFiles(directory=os.path.join(PACKAGE_DIR, "static")), name="admin_static")

# Defense-in-depth response headers. The admin panel needs inline scripts/styles,
# so script-src/style-src include 'unsafe-inline'; CSP still blocks external
# script injection, data exfiltration images, plugin objects and clickjacking.
# cdn.jsdelivr.net serves the /docs + /redoc assets, and the admin panel loads
# three.js from cdnjs.cloudflare.com; fonts.googleapis.com/gstatic.com and the
# fastapi.tiangolo.com favicon are needed by the auto-generated docs pages only.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' https://fonts.gstatic.com; connect-src 'self'; media-src 'self'; "
    "object-src 'none'; frame-ancestors 'none'; form-action 'self'; base-uri 'self'"
)

_missing_token_warned = False


def _same_origin(request: Request, origin: str) -> bool:
    """True when ``origin`` matches the request's own host (default ports
    normalized) or an explicitly trusted origin."""
    def _norm(value: str) -> str:
        value = value.strip().rstrip("/")
        try:
            parts = urlsplit(value)
            host = (parts.hostname or "").lower()
            port = parts.port
            if port is None:
                port = 443 if parts.scheme.lower() == "https" else 80
        except ValueError:
            return value.lower()
        if port in (80, 443):
            return f"{parts.scheme.lower()}://{host}"
        return f"{parts.scheme.lower()}://{host}:{port}"

    host = request.headers.get("host", "")
    allowed = {_norm(o) for o in _cfg.TRUSTED_ORIGINS}
    allowed.add(_norm(f"https://{host}"))
    allowed.add(_norm(f"http://{host}"))
    return _norm(origin) in allowed


def _csrf_check(request: Request) -> bool:
    """Origin check for production /admin state-changing requests authorized by
    a browser session. API-token (non-browser) callers bypass it; going through
    a browser without an Origin header is a rejected cross-site form."""
    methods = ("POST", "PUT", "PATCH", "DELETE")
    if _cfg.PRODUCTION and request.method in methods:
        origin = request.headers.get("origin", "")
        if not origin or not _same_origin(request, origin):
            return False
    return True

# Fast-path hint queue: the API only pushes ids here (LPUSH). The row insert is
# the durable source of truth; a redis push failure or a lost hint is recovered
# by the 30s reconcile sweeper, which re-claims and dispatches pending posts.


@app.middleware("http")
async def service_authentication(request: Request, call_next):
    """Authenticate every non-public request.

    Two surfaces, two credentials:
      * owner plane (/admin/*): the shared API_AUTH_TOKEN only - the panel must
        additionally sit behind the operator's admin ingress / MFA upstream.
      * data plane (everything else): the shared token OR a per-consumer token
        (revocable, rate-limited on mutating + integration routes, and every
        feedback action is attributed to the consumer that made it).

    If API_AUTH_TOKEN is unset AND NEXUS_ENV is not production, the middleware
    stays open for local development (documented behavior). With
    NEXUS_ENV=production the API is fail-closed: a missing token never
    short-circuits auth, and every request must present the shared token or
    (on /admin) a valid panel session / consumer credential.
    """
    public_paths = {"/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc", "/admin/health/ready"}
    admin_public_paths = {"/admin/login", "/admin/forgot", "/admin/reset", "/admin/register",
                          "/admin/login/google", "/admin/auth/google/callback"}
    path = request.url.path
    if path in public_paths or path.startswith("/admin/static") or path in admin_public_paths or path.startswith("/admin/invite/"):
        return await call_next(request)

    shared_configured = bool(API_AUTH_TOKEN)
    supplied = request.headers.get("authorization", "")
    shared_ok = shared_configured and hmac.compare_digest(supplied, f"Bearer {API_AUTH_TOKEN}")

    global _missing_token_warned
    if _cfg.PRODUCTION and not shared_configured and not _missing_token_warned:
        _missing_token_warned = True
        log.warning(
            "NEXUS_ENV=production but API_AUTH_TOKEN is not set - fail-closed: "
            "all requests will be denied except public endpoints and valid panel sessions"
        )

    if path.startswith("/admin"):
        if shared_ok:
            return await call_next(request)
        # Panel-first: a valid browser session unlocks the owner plane.
        from .admin_auth import verify_session
        if verify_session(request.cookies.get("nexus_admin_session")):
            if not _csrf_check(request):
                return JSONResponse(status_code=403,
                                    content={"detail": "cross-site request rejected"})
            return await call_next(request)
        if not shared_configured and not _cfg.PRODUCTION:
            return await call_next(request)  # dev-open
        return await _admin_denied(request)

    # Data plane: shared token (or dev-open) short-circuits.
    if shared_ok or (not shared_configured and not _cfg.PRODUCTION):
        return await call_next(request)

    consumer = None
    if supplied.startswith("Bearer "):
        consumer = authenticate_consumer(supplied[len("Bearer "):].strip())
    if consumer is None:
        return JSONResponse(status_code=401, content={"detail": "invalid credentials"},
                            headers={"WWW-Authenticate": "Bearer"})
    request.state.consumer = consumer
    if _rate_limited_path(path, request.method):
        allowed, _ = rate_limit_check(str(consumer["consumer_id"]),
                                      int(consumer["rate_limit_per_minute"]))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded for this consumer"},
                headers={"Retry-After": "1",
                         "X-Nexus-RateLimit-Limit": str(consumer["rate_limit_per_minute"]),
                         "X-Nexus-RateLimit-Remaining": "0"})
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Attach browser hardening headers to every response."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if _cfg.PRODUCTION:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    response.headers.setdefault("Content-Security-Policy", _CSP)
    return response


if _cfg.TRUSTED_HOSTS:
    # Registered last so it is the outermost middleware: the Host header is
    # validated before any auth/CSRF/security-header logic runs.
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_cfg.TRUSTED_HOSTS)


async def _admin_denied(request: Request):
    """Session missing/expired: give browsers a friendly redirect and let
    HTMX partials follow ``HX-Redirect``; plain API callers get a 401 JSON."""
    hx = request.headers.get("hx-request") == "true"
    if hx:
        return JSONResponse(status_code=401, content={"detail": "authentication required"},
                            headers={"HX-Redirect": "/admin/login", "WWW-Authenticate": "Bearer"})
    accepts_html = "text/html" in request.headers.get("accept", "")
    if accepts_html and request.method == "GET":
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/admin/login", status_code=status.HTTP_303_SEE_OTHER)
    return JSONResponse(status_code=401, content={"detail": "authentication required"},
                        headers={"WWW-Authenticate": "Bearer"})


def _rate_limited_path(path: str, method: str) -> bool:
    """Consumer tokens are throttled where they can move load: the delivery
    poll/ack loop and every data-plane write (ingest, corrections, merges,
    deletes). Plain reads (status, hub views) stay open for token holders."""
    if path.startswith("/integration/events"):
        return True
    return method in ("POST", "PUT", "PATCH", "DELETE")


def _consumer_actor(request: Request, fallback: str = "user") -> str:
    """Attribute a feedback action to the consumer that performed it; without
    a consumer token, fall back to the caller-supplied actor label."""
    c = getattr(request.state, "consumer", None)
    if c and c.get("name"):
        return c["name"]
    return fallback


def run():
    import uvicorn

    from .log import configure_json_logging

    configure_json_logging()
    uvicorn.run("post_clustering_pipeline.api:app", host="0.0.0.0", port=8000, log_config=None)


# --- Request Models ---
class PostCreateRequest(BaseModel):
    user_id: int
    # Bounded to match the storage columns: an unbounded content/platform/etc.
    # either exhausts memory or overflows a VARCHAR and surfaces as a 500.
    content: str = Field(max_length=100_000)
    has_media: bool = False
    platform: str = Field(default="unknown", max_length=50)
    source_id: str = Field(default="legacy", max_length=200)
    external_post_id: str | None = Field(default=None, max_length=300)
    external_author_id: str | None = Field(default=None, max_length=300)
    published_at: datetime | None = None


class BatchPostCreateRequest(BaseModel):
    # Bound request size so one transaction cannot monopolize the database.
    posts: list[PostCreateRequest] = Field(default_factory=list, max_length=500)


class FeedbackRemoveRequest(BaseModel):
    post_id: int
    event_id: int


class FeedbackConfirmRequest(BaseModel):
    post_id: int
    event_id: int
    actor: str = Field(default="user", max_length=200)


class EventMergeRequest(BaseModel):
    source_event_id: int
    target_event_id: int
    actor: str = Field(default="user", max_length=200)
    note: str | None = Field(default=None, max_length=2000)


# --- Helper Function for Cursors ---
def extract_field(row, dict_key: str, index: int = 0):
    """Safely extract values whether using RealDictCursor or standard tuple cursor."""
    if not row:
        return None
    if isinstance(row, dict):
        return row.get(dict_key)
    if isinstance(row, (list, tuple)):
        return row[index]
    return row


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT COUNT(*) AS dlq_count FROM integration_outbox WHERE delivery_status = 'failed'")
            row = cur.fetchone()
            dlq_count = extract_field(row, "dlq_count", 0) or 0
            cur.execute(
                """
                SELECT COUNT(*) AS stuck_processing FROM posts
                WHERE assignment_status = 'processing'
                  AND assignment_updated_at < NOW() - INTERVAL '5 minutes';
                """
            )
            row = cur.fetchone()
            stuck_processing = extract_field(row, "stuck_processing", 0) or 0
        return {
            "status": "ready" if stuck_processing == 0 else "degraded",
            "dlq_failed_events": dlq_count,
            "stuck_processing": stuck_processing,
        }
    except Exception as exc:
        # Generic to callers (this endpoint is unauthenticated): the real
        # exception may carry DSN/host details. Log it for the operator.
        print(f"\n[HEALTH ERROR] readiness probe failed: {exc}\n")
        raise HTTPException(status_code=503, detail="database unavailable")


@app.get("/posts/{post_id}/status")
def get_post_status(post_id: int):
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, event_id, assignment_status, assignment_confidence, assignment_updated_at FROM posts WHERE id = %s",
            (post_id,)
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")
    return row


@app.post("/posts", status_code=status.HTTP_201_CREATED)
def create_post(post: PostCreateRequest):
    try:
        with get_db_cursor() as cur:
            if post.external_post_id:
                cur.execute(
                    """INSERT INTO posts
                    (user_id, content, has_media, platform, source_id, external_post_id,
                     external_author_id, published_at, assignment_status, content_sig)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
                    ON CONFLICT (source_id, external_post_id) WHERE external_post_id IS NOT NULL
                    DO UPDATE SET content = EXCLUDED.content, has_media = EXCLUDED.has_media,
                                  content_sig = EXCLUDED.content_sig,
                                  published_at = COALESCE(EXCLUDED.published_at, posts.published_at)
                    WHERE posts.deleted_at IS NULL
                    RETURNING id;""",
                    (post.user_id, post.content, post.has_media, post.platform, post.source_id,
                     post.external_post_id, post.external_author_id, post.published_at,
                     content_signature(post.content))
                )
                row = cur.fetchone()
                post_id = extract_field(row, "id", 0)
                if post_id is None:
                    # The conflict matched a tombstone (deleted_at set): the
                    # DO UPDATE ... WHERE filtered it out, so no row came back.
                    # Fail closed - a deleted post must never be resurrected.
                    raise HTTPException(
                        status_code=409,
                        detail="This platform post was deleted and cannot be recreated"
                    )
            else:
                cur.execute(
                    """INSERT INTO posts
                    (user_id, content, has_media, platform, source_id, external_post_id,
                     external_author_id, published_at, assignment_status, content_sig)
                    VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, 'pending', %s)
                    RETURNING id;""",
                    (post.user_id, post.content, post.has_media, post.platform,
                     post.external_author_id, post.published_at, content_signature(post.content))
                )
                row = cur.fetchone()
                post_id = extract_field(row, "id", 0)
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[API ERROR] Failed to create post: {e}\n")
        raise HTTPException(status_code=500, detail="could not create post")

    try:
        queues_bounded_push_ingest_hints(post_id)
    except Exception as e:
        print(f"\n[REDIS ERROR] Failed to push ingest hint: {e}\n")

    return {"status": "queued", "post_id": post_id}


@app.post("/posts/batch", status_code=status.HTTP_201_CREATED)
def create_posts_batch(payload: BatchPostCreateRequest):
    """High-throughput multi-post submission endpoint."""
    if not payload.posts:
        return {"status": "queued", "post_ids": []}

    created_posts: list[tuple[int, str]] = []
    try:
        with get_db_cursor() as cur:
            for post in payload.posts:
                if post.external_post_id:
                    cur.execute(
                        """INSERT INTO posts
                        (user_id, content, has_media, platform, source_id, external_post_id,
                         external_author_id, published_at, assignment_status, content_sig)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
                        ON CONFLICT (source_id, external_post_id) WHERE external_post_id IS NOT NULL
                        DO UPDATE SET content = EXCLUDED.content, has_media = EXCLUDED.has_media,
                                      content_sig = EXCLUDED.content_sig,
                                      published_at = COALESCE(EXCLUDED.published_at, posts.published_at)
                        WHERE posts.deleted_at IS NULL
                        RETURNING id;""",
                        (post.user_id, post.content, post.has_media, post.platform, post.source_id,
                         post.external_post_id, post.external_author_id, post.published_at,
                         content_signature(post.content))
                    )
                    row = cur.fetchone()
                    pid = extract_field(row, "id", 0)
                    if not pid:
                        continue  # Tombstone conflict -> skip, never resurrect
                else:
                    cur.execute(
                        """INSERT INTO posts
                        (user_id, content, has_media, platform, source_id, external_post_id,
                         external_author_id, published_at, assignment_status, content_sig)
                        VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, 'pending', %s)
                        RETURNING id;""",
                        (post.user_id, post.content, post.has_media, post.platform,
                         post.external_author_id, post.published_at, content_signature(post.content))
                    )
                    row = cur.fetchone()
                    pid = extract_field(row, "id", 0)
                if pid:
                    created_posts.append((pid, post.content))
    except Exception as e:
        print(f"\n[API ERROR] Failed in batch post ingestion: {e}\n")
        raise HTTPException(status_code=500, detail="could not create posts")

    try:
        if created_posts:
            queues_bounded_push_ingest_hints(*[str(pid) for pid, _ in created_posts])
    except Exception as e:
        print(f"\n[REDIS ERROR] Failed to push ingest hints: {e}\n")

    return {"status": "queued", "post_ids": [pid for pid, _ in created_posts]}


@app.delete("/posts/{post_id}")
def delete_post(post_id: int):
    with get_db_cursor() as cur:
        # One statement: lock the post, tombstone it, and decrement its
        # *pre-update* hub - RETURNING yields post-update values, so the old
        # event_id must be captured by a FOR UPDATE read before the UPDATE.
        # A delete racing with a confirm/assignment can therefore never
        # double-decrement or leave a stale hub balance, and a re-delete sees
        # event_id already NULL and decrements nothing.
        cur.execute(
            """
            WITH prior AS (
                SELECT id, event_id FROM posts WHERE id = %s FOR UPDATE
            ),
            deleted AS (
                UPDATE posts
                SET deleted_at = COALESCE(deleted_at, NOW()),
                    assignment_status = 'noise',
                    event_id = NULL,
                    assignment_updated_at = NOW()
                WHERE id = (SELECT id FROM prior LIMIT 1)
                RETURNING id
            ),
            hub_drop AS (
                UPDATE event_hubs h
                SET member_count = GREATEST(h.member_count - 1, 0),
                    last_updated_at = NOW()
                FROM prior p, deleted d
                WHERE h.id = p.event_id
                  AND p.event_id IS NOT NULL
                  AND d.id = p.id
            )
            SELECT p.id, p.event_id FROM prior p, deleted d WHERE d.id = p.id
            """,
            (post_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Post not found")
        old_event = extract_field(row, "event_id", 1)

        if old_event is not None:
            close_membership(cur, post_id)
            write_outbox(cur, "post.deleted", post_id, None, {
                "status": "deleted", "post_id": post_id, "previous_event_id": old_event
            })
            # Reparent surviving reposts: deleting a canonical must never leave
            # children reposting a tombstone (which the exact-fold keeper would
            # otherwise adopt as the body's canonical). The earliest live child
            # is promoted to canonical in the same hub and the rest repoint to
            # it; counts are then true from live membership.
            from .membership import record_memberships
            from .fold import sync_hub_counts
            cur.execute(
                "SELECT id FROM posts WHERE repost_of_id = %s AND deleted_at IS NULL ORDER BY id;",
                (post_id,)
            )
            child_ids = [extract_val(r, "id", 0) for r in (cur.fetchall() or [])]
            if child_ids:
                new_canonical = int(child_ids[0])
                cur.execute(
                    "UPDATE posts SET repost_of_id = NULL, assignment_status = 'assigned', assignment_updated_at = NOW() WHERE id = %s;",
                    (new_canonical,)
                )
                if len(child_ids) > 1:
                    cur.execute(
                        "UPDATE posts SET repost_of_id = %s WHERE id = ANY(%s::int[]) AND id <> %s AND deleted_at IS NULL;",
                        (new_canonical, [int(c) for c in child_ids], new_canonical)
                    )
                record_memberships(cur, [(new_canonical, old_event, "member")])
                sync_hub_counts(cur, [old_event])
        else:
            write_outbox(cur, "post.deleted", post_id, None, {"status": "deleted", "post_id": post_id})
        cur.execute("DELETE FROM unclustered_posts_buffer WHERE post_id = %s;", (post_id,))
        from .policy import current_versions
        from .decisions import log_decision
        pv, mv = current_versions(cur)
        log_decision(cur, post_id=post_id, event_id=None, similarity=None, runner_similarity=None,
                     threshold_used=None, margin_budget=None, status="noise", confidence=None,
                     policy_version=pv, model_version=mv, reason="manual_delete")
    return {"status": "deleted", "post_id": post_id}


@app.post("/posts/unlink", status_code=status.HTTP_200_OK)
def unlink_post(payload: FeedbackRemoveRequest, request: Request):
    """Human-corrected removal: the post goes back to 'candidate' (gradeable),
    the feedback is tagged with the policy/model in effect, and the outbox
    event carries content references so the operator's client can label the
    change without a follow-up fetch."""
    try:
        actor = _consumer_actor(request, "user")
        with get_db_cursor() as cur:
            return unlink_correction(cur, payload.post_id, payload.event_id, actor=actor)
    except CorrectionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/posts/confirm", status_code=status.HTTP_200_OK)
def confirm_post(payload: FeedbackConfirmRequest, request: Request):
    actor = _consumer_actor(request, payload.actor)
    try:
        with get_db_cursor() as cur:
            return confirm_correction(cur, payload.post_id, payload.event_id, actor=actor)
    except CorrectionError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/events/merge", status_code=status.HTTP_200_OK)
def merge_events(payload: EventMergeRequest, request: Request):
    """Soft-merge ``source_event_id`` into ``target_event_id``.

    Uses the SAME guarded path as the internal auto-detect job: canonical
    resolution, advisory-lock serialization, member snapshot, centroid
    rebalance. The response and the ``event.merged`` payload carry the two
    hubs' content references so humans see WHICH hubs were merged, not ids.
    """
    actor = _consumer_actor(request, payload.actor)
    try:
        result = client_merge(payload.source_event_id, payload.target_event_id, actor)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "merged", **result, "actor": actor, "note": payload.note}


@app.get("/integration/events")
def integration_events(limit: int = 100, after: int = 0, consumer: str = "default"):
    """Pull interface with lease locking and dead-letter queue (DLQ) protection."""
    limit = max(1, min(limit, 500))
    token = uuid.uuid4().hex

    with get_db_cursor() as cur:
        # Move poisoned records exceeding max retries to Dead Letter Queue (failed)
        cur.execute(
            """
            UPDATE integration_outbox 
            SET delivery_status = 'failed', last_error = 'Max delivery attempts exceeded'
            WHERE attempts >= %s AND delivery_status IN ('pending', 'leased')
              AND (lease_until IS NULL OR lease_until < NOW());
            """,
            (OUTBOX_MAX_ATTEMPTS,)
        )

        cur.execute(
            """
            WITH claimed AS (
                SELECT id FROM integration_outbox
                WHERE id > %s AND available_at <= NOW()
                  AND (delivery_status = 'pending' OR (delivery_status = 'leased' AND lease_until < NOW()))
                ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED
            ),
            leased AS (
                UPDATE integration_outbox o SET delivery_status = 'leased', consumer = %s,
                    lease_token = %s, lease_until = NOW() + INTERVAL '2 minutes', attempts = attempts + 1
                FROM claimed c WHERE o.id = c.id
                RETURNING o.id, o.id::text AS event_key, o.schema_version, o.event_type, o.post_id,
                    o.event_id, o.payload, o.attempts, o.lease_token, o.created_at
            )
            SELECT * FROM leased ORDER BY id
            """,
            (after, limit, consumer, token)
        )
        rows = cur.fetchall()

    return rows


@app.post("/integration/events/{outbox_id}/ack")
def acknowledge_integration_event(outbox_id: int, lease_token: str | None = None):
    with get_db_cursor() as cur:
        # Ownership rule: an event with an ACTIVE lease (leased and
        # lease_until in the future) may only be acked by the consumer holding
        # its lease_token. Unleased or expired-lease events may be acked by
        # anyone. This prevents consumers from acking events they never
        # leased (previously any caller could ack a leased row by passing no
        # token).
        cur.execute(
            """
            UPDATE integration_outbox
            SET delivery_status = 'delivered', delivered_at = NOW(), lease_until = NULL
            WHERE id = %s AND delivery_status IN ('pending', 'leased')
              AND (
                    lease_until IS NULL OR lease_until <= NOW()
                    OR (lease_until > NOW() AND lease_token = %s AND %s IS NOT NULL)
                  )
            RETURNING id, id::text AS event_key, delivery_status
            """,
            (outbox_id, lease_token, lease_token)
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                "SELECT id, id::text AS event_key, delivery_status, lease_until FROM integration_outbox WHERE id = %s",
                (outbox_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Integration event not found")
            if extract_field(row, "delivery_status", 2) in ("pending", "leased"):
                raise HTTPException(
                    status_code=409,
                    detail="Integration event is under a lease that this consumer does not own",
                )

    return row


@app.get("/hubs/{hub_id}/view")
@app.get("/events/{hub_id}/view")
def get_hub_view(hub_id: int, limit: int = 50, offset: int = 0):
    """Unified 1-click portal view for a story, debate, topic, or event."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with get_db_cursor(commit=False) as cur:
        canonical_id, was_redirected = resolve_canonical_hub(cur, hub_id)
        cur.execute(
            """
            SELECT id, status, discourse_type, member_count, seed_post_id, created_at, last_updated_at
            FROM event_hubs
            WHERE id = %s;
            """,
            (canonical_id,)
        )
        hub_row = cur.fetchone()
        if not hub_row:
            raise HTTPException(status_code=404, detail="Event Hub not found")

        anchor_pid = extract_field(hub_row, "seed_post_id", 4)
        status_val = extract_field(hub_row, "status", 1) or "active"
        discourse_type = extract_field(hub_row, "discourse_type", 2) or "event"

        # 1. Fetch catalyst anchor post
        catalyst = None
        if anchor_pid:
            cur.execute(
                "SELECT id, user_id, content, has_media, engagement_score, created_at FROM posts WHERE id = %s AND deleted_at IS NULL;",
                (anchor_pid,)
            )
            catalyst = cur.fetchone()

        if not catalyst:
            cur.execute(
                "SELECT id, user_id, content, has_media, engagement_score, created_at FROM posts WHERE event_id = %s AND repost_of_id IS NULL AND deleted_at IS NULL ORDER BY created_at ASC LIMIT 1;",
                (canonical_id,)
            )
            catalyst = cur.fetchone()

        # 2. Fetch chronological timeline (canonical posts only - folded reposts
        #    are surfaced via their repost_count / unique_voices, not as
        #    duplicate feed entries).
        cur.execute(
            """
            SELECT id, user_id, content, has_media, engagement_score, created_at
            FROM posts
            WHERE event_id = %s AND deleted_at IS NULL AND repost_of_id IS NULL
            ORDER BY created_at ASC, id ASC
            LIMIT %s OFFSET %s;
            """,
            (canonical_id, limit + 1, offset)
        )
        timeline_rows = cur.fetchall() or []
        has_more = len(timeline_rows) > limit
        timeline = timeline_rows[:limit]
        cur.execute(
            "SELECT COUNT(*) AS total_posts FROM posts WHERE event_id = %s AND deleted_at IS NULL AND repost_of_id IS NULL;",
            (canonical_id,)
        )
        count_row = cur.fetchone()
        total_posts = int(extract_field(count_row, "total_posts", 0) or 0)
        cur.execute(
            "SELECT COUNT(*) AS reposts FROM posts WHERE event_id = %s AND deleted_at IS NULL AND repost_of_id IS NOT NULL;",
            (canonical_id,)
        )
        repost_row = cur.fetchone()
        repost_count = int(extract_field(repost_row, "reposts", 0) or 0)

        # 3. Fetch key perspectives (top distinct viewpoints / high engagement posts)
        catalyst_id = extract_field(catalyst, "id", 0) if catalyst else -1
        cur.execute(
            """
            SELECT id, user_id, content, has_media, engagement_score, created_at
            FROM posts
            WHERE event_id = %s AND id != %s AND deleted_at IS NULL AND repost_of_id IS NULL
            ORDER BY engagement_score DESC, created_at ASC
            LIMIT %s;
            """,
            (canonical_id, catalyst_id, min(limit, 20))
        )
        perspectives = cur.fetchall() or []

        # 4. Fetch evidence media (unique media posts only)
        cur.execute(
            """
            SELECT id, user_id, content, created_at
            FROM posts
            WHERE event_id = %s AND has_media = TRUE AND deleted_at IS NULL AND repost_of_id IS NULL
            ORDER BY created_at ASC, id ASC
            LIMIT %s;
            """,
            (canonical_id, min(limit, 100))
        )
        media = cur.fetchall() or []

        # 5. Count unique voices (all members, canonical + reposts - a reshared
        #    post from a new author is still a distinct voice).
        cur.execute(
            "SELECT COUNT(DISTINCT user_id) AS unique_voices FROM posts WHERE event_id = %s AND deleted_at IS NULL;",
            (canonical_id,)
        )
        uv_row = cur.fetchone()
        unique_voices = extract_field(uv_row, "unique_voices", 0) or len(timeline)

        # The hub reference must be built while the cursor is still open - the
        # return dict below is evaluated after the context manager has closed it.
        reference = hub_reference(cur, canonical_id)

    return {
        "hub_id": canonical_id,
        "requested_hub_id": hub_id,
        "was_redirected": was_redirected,
        "status": status_val,
        "discourse_type": discourse_type,
        "reference": reference,
        "metrics": {
            "total_posts": total_posts,
            "reposts": repost_count,
            "member_count": extract_field(hub_row, "member_count", 3),
            "unique_voices": unique_voices,
            "created_at": extract_field(hub_row, "created_at", 5),
            "last_updated_at": extract_field(hub_row, "last_updated_at", 6),
        },
        "pagination": {
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
        },
        "catalyst_anchor": catalyst,
        "key_perspectives": perspectives,
        "timeline": timeline,
        "evidence_media": media,
    }


@app.get("/events/{event_id}/anchor")
def get_anchor_post(event_id: int):
    with get_db_cursor(commit=False) as cur:
        canonical_id, _ = resolve_canonical_hub(cur, event_id)
        cur.execute("SELECT id, user_id, content, has_media, engagement_score, created_at FROM posts WHERE event_id = %s AND repost_of_id IS NULL AND deleted_at IS NULL ORDER BY created_at ASC LIMIT 1;", (canonical_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Event hub standard anchor post not found")
    return row


@app.get("/events/{event_id}/media")
def get_event_media(event_id: int):
    with get_db_cursor(commit=False) as cur:
        canonical_id, _ = resolve_canonical_hub(cur, event_id)
        cur.execute(
            "SELECT id, content, created_at FROM posts WHERE event_id = %s AND has_media = TRUE AND repost_of_id IS NULL AND deleted_at IS NULL ORDER BY created_at ASC;",
            (canonical_id,)
        )
        rows = cur.fetchall()
    return rows


@app.get("/events/{event_id}/timeline")
def get_event_timeline(event_id: int):
    with get_db_cursor(commit=False) as cur:
        canonical_id, _ = resolve_canonical_hub(cur, event_id)
        cur.execute("SELECT id, user_id, content, has_media, engagement_score, created_at FROM posts WHERE event_id = %s AND repost_of_id IS NULL ORDER BY created_at ASC;", (canonical_id,))
        rows = cur.fetchall()
    return rows


@app.get("/events/{event_id}/top")
def get_event_top_discussion(event_id: int):
    with get_db_cursor(commit=False) as cur:
        canonical_id, _ = resolve_canonical_hub(cur, event_id)
        cur.execute("SELECT id, user_id, content, has_media, engagement_score, created_at FROM posts WHERE event_id = %s AND repost_of_id IS NULL ORDER BY engagement_score DESC;", (canonical_id,))
        rows = cur.fetchall()
    return rows
