"""Owner control plane (FastAPI router, mounted under /admin).

Control plane != data plane: every mutation here is versioned, validated,
dry-runnable or estimated, warned, audited, and reversible. The panel is
server-rendered Jinja2 fragments driven by HTMX so the operator gets a
stateful app without a JS framework.

Every human-facing surface renders hubs/posts by their CONTENT references
(see ``refs.py``) - the merge desk, decisions inspector, and calibration
all anchor to what people can actually read and recognize.
"""
from __future__ import annotations

import hmac
import os
import secrets
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from .config import PACKAGE_DIR, API_AUTH_TOKEN
from .db import get_db_cursor, extract_val
from .policy import (
    KNOB_RANGES, KNOB_WARNINGS, current_versions, apply_policy_change,
    propose_policy_change,
)
from .refs import search_refs, hub_reference, post_reference
from .merges import client_merge, reopen_merge
from .audit import log_admin_action
from .consumers import hash_token

router = APIRouter(prefix="/admin")
templates = Jinja2Templates(directory=os.path.join(PACKAGE_DIR, "templates", "admin"))
TEMPLATE_METADATA = {"type": "text/html", "Cache-Control": "no-store"}

DEFAULT_ACTOR = "panel"

PRECISION_LABELS = {
    "confident_assign": ("ok", "confident assign"),
    "margin_fail": ("warn", "margin fail"),
    "below_threshold": ("warn", "below threshold"),
    "below_candidate_floor": ("neutral", "no candidate"),
    "discourse_gate_noise": ("neutral", "discourse gate"),
    "entity_conflict": ("danger", "entity conflict veto"),
    "event_birth_community": ("ok", "born hub member"),
    "event_birth_fallback": ("ok", "birth fallback assign"),
    "manual_delete": ("danger", "deleted"),
    "aged_unclustered_noise": ("muted", "aged unclustered"),
}
STATUS_LABELS = {
    "assigned": ("ok", "assigned"),
    "candidate": ("warn", "candidate"),
    "unassigned": ("neutral", "unassigned"),
    "noise": ("muted", "noise"),
}


def _ctx(request: Request, **extra) -> dict:
    from .config import MODEL_VERSION
    session = _current_admin(request)
    path = request.url.path
    for key, prefix in (("quality", "/admin/quality"), ("decisions", "/admin/decisions"),
                        ("desk", "/admin/desk"), ("merges", "/admin/merges"),
                        ("calibration", "/admin/calibration"), ("ops", "/admin/ops"),
                        ("audit", "/admin/audit"), ("settings", "/admin/settings")):
        if path.startswith(prefix):
            nav_active = key
            break
    else:
        nav_active = ""
    ctx = {
        "request": request,
        "model_version": MODEL_VERSION,
        "api_token_configured": bool(API_AUTH_TOKEN),
        "session_email": session["email"] if session else None,
        "nav_active": nav_active,
    }
    ctx.update(extra)
    return ctx


def _render_admins_fragment(request: Request, search: str = "", status: str = "",
                            limit: int = 25, offset: int = 0):
    from .admin_auth import list_admins, admin_stats
    current_admin = _current_admin(request)
    admins, total = list_admins(search=search, status=status, limit=limit, offset=offset)
    return templates.TemplateResponse(request, "admins_panel.html",
        _ctx(request, admins=admins, total=total, admin_stats=admin_stats(),
             search=search, status=status,
             my_admin_id=(current_admin or {}).get("admin_id", 0),
             has_more=(offset + len(admins)) < total,
             next_offset=offset + len(admins), offset=offset, limit=limit),
        headers=TEMPLATE_METADATA)


@router.get("/settings/admins", include_in_schema=False)
async def panel_settings_admins(request: Request, q: str = "", status: str = "",
                                limit: int = 25, offset: int = 0, partial: int = 0):
    """Admin-list fragment: search + status filter + load-more pagination.
    `partial=1` returns just the table rows (for the load-more row)."""
    from .admin_auth import list_admins, admin_stats
    current_admin = _current_admin(request)
    search = q.strip()
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    admins, total = list_admins(search=search, status=status, limit=limit, offset=offset)
    ctx = _ctx(request, admins=admins, total=total, admin_stats=admin_stats(),
               search=search, status=status,
               my_admin_id=(current_admin or {}).get("admin_id", 0),
               has_more=(offset + len(admins)) < total,
               next_offset=offset + len(admins), offset=offset, limit=limit)
    if partial:
        return templates.TemplateResponse(request, "admin_rows.html", ctx,
                                          headers=TEMPLATE_METADATA)
    return templates.TemplateResponse(request, "admins_panel.html", ctx,
                                      headers=TEMPLATE_METADATA)


# ---------------------------------------------------------------------------
# Quality / observatory
# ---------------------------------------------------------------------------

@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def panel_index(request: Request):
    return await panel_quality(request)


@router.get("/health/ready", include_in_schema=False)
async def panel_health_ready():
    """Owner-panel alias of the public /health/ready probe (returns an HTML pill)."""
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT COUNT(*) AS dlq_count FROM integration_outbox WHERE delivery_status = 'failed'")
        row = cur.fetchone()
        dlq_count = extract_val(row, "dlq_count", 0) or 0
        cur.execute(
            """
            SELECT COUNT(*) AS stuck_processing FROM posts
            WHERE assignment_status = 'processing'
              AND assignment_updated_at < NOW() - INTERVAL '5 minutes';
            """
        )
        row = cur.fetchone()
        stuck_processing = extract_val(row, "stuck_processing", 0) or 0
    if stuck_processing == 0 and dlq_count == 0:
        cls, label = "ok", "ready"
    elif stuck_processing == 0:
        cls, label = "warn", "dlq active"
    else:
        cls, label = "danger", "degraded"
    return Response(
        content=(
            f'<span class="dot {cls} blink" aria-hidden="true"></span>'
            f'<span class="health-status {cls}">{label}</span>'
            '<span class="health-detail">'
            f'<b>{(stuck_processing or 0)}</b> stuck&nbsp;&middot;&nbsp;'
            f'<b>{dlq_count}</b> dlq'
            '</span>'
        ),
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Owner authentication (email + password, TOTP recovery)
# ---------------------------------------------------------------------------

def _session_cookie(session_token: str) -> dict:
    from . import config as cfg
    return {
        "key": "nexus_admin_session",
        "value": session_token,
        "httponly": True,
        "samesite": "lax",
        "secure": cfg.SESSION_COOKIE_SECURE,
        "path": "/admin",
        "max_age": 12 * 3600,
    }


def _current_admin(request: Request) -> dict | None:
    from .admin_auth import verify_session
    return verify_session(request.cookies.get("nexus_admin_session"))


@router.get("/login", include_in_schema=False)
async def panel_login(request: Request):
    from .admin_auth import admin_count
    from . import config as cfg
    google_errors = {
        "google-oauth-not-configured": "Sign in with Google is not configured on the server "
                                       "(set GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET).",
        "google-denied": "You cancelled the Google sign-in; no changes were made.",
        "google-invalid-state": "Google sign-in failed its state check; try again.",
        "google-token-failed": "Google could not exchange the login code; try again.",
        "google-verify-failed": "Google could not verify your identity; try again.",
        "google-email-unverified": "That Google account has no verified email; try another account.",
        "google-no-account": "No admin account matches that Google email. Ask an existing admin to invite you "
                             "before signing in with Google.",
        "account-disabled": "account disabled - contact an active admin",
    }
    error = google_errors.get(request.query_params.get("error", ""), None)
    return templates.TemplateResponse(request, "login.html",
        _ctx(request, error=error, must_register=admin_count() == 0,
             google_enabled=bool(cfg.GOOGLE_OAUTH_CLIENT_ID and cfg.GOOGLE_OAUTH_CLIENT_SECRET)),
        headers=TEMPLATE_METADATA)


@router.post("/login", include_in_schema=False)
async def panel_login_submit(request: Request):
    from .admin_auth import (find_admin_by_email, verify_password, issue_session, touch_login,
                             admin_count, valid_totp, _rate_limited_totp, _note_totp_failure)
    from .ratelimit import blocked, note_failure, reset
    from fastapi.responses import RedirectResponse
    from . import config as cfg
    form = await request.form()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "")
    client_ip = request.client.host if request.client else "unknown"
    window = cfg.LOGIN_LOCK_WINDOW_SECONDS
    if blocked("login", email, cfg.LOGIN_MAX_ATTEMPTS, window) or \
            blocked("login-ip", client_ip, cfg.LOGIN_IP_MAX_ATTEMPTS, window):
        return templates.TemplateResponse(request, "login.html",
            _ctx(request, error="too many failed attempts - wait a few minutes",
                 must_register=admin_count() == 0),
            headers=TEMPLATE_METADATA)
    admin = find_admin_by_email(email)
    if not admin or not verify_password(password, admin["password_hash"]):
        note_failure("login", email, window)
        note_failure("login-ip", client_ip, window)
        return templates.TemplateResponse(request, "login.html",
            _ctx(request, error="invalid email or password", must_register=admin_count() == 0),
            headers=TEMPLATE_METADATA)
    reset("login", email)
    reset("login-ip", client_ip)
    if not admin["is_active"]:
        return templates.TemplateResponse(request, "login.html",
            _ctx(request, error="account disabled - contact an active admin", must_register=False),
            headers=TEMPLATE_METADATA)
    if admin["totp_enabled"]:
        code = (form.get("totp") or "").strip()
        tkey = f"login-totp:{admin['id']}"
        if _rate_limited_totp(tkey):
            return templates.TemplateResponse(request, "login.html",
                _ctx(request, error="too many attempts - wait a few minutes",
                     must_register=False), headers=TEMPLATE_METADATA)
        if not valid_totp(admin["totp_secret"], code):
            _note_totp_failure(tkey)
            return templates.TemplateResponse(request, "login.html",
                _ctx(request, error="invalid Authenticator code - try again",
                     must_register=False), headers=TEMPLATE_METADATA)
    token = issue_session(admin["email"], admin["id"])
    touch_login(admin["id"])
    resp = RedirectResponse(url="/admin/", status_code=303)
    resp.set_cookie(**_session_cookie(token))
    return resp


@router.post("/logout", include_in_schema=False)
async def panel_logout(request: Request):
    from fastapi.responses import RedirectResponse
    resp = RedirectResponse(url="/admin/login", status_code=303)
    resp.delete_cookie("nexus_admin_session", path="/admin")
    return resp


@router.get("/register", include_in_schema=False)
async def panel_register(request: Request):
    from .admin_auth import admin_count
    if admin_count() > 0:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(request, "register.html",
        _ctx(request, error=None), headers=TEMPLATE_METADATA)


@router.post("/register", include_in_schema=False)
async def panel_register_submit(request: Request):
    from .admin_auth import admin_count, create_admin, issue_session, touch_login
    from fastapi.responses import RedirectResponse
    if admin_count() > 0:
        return RedirectResponse(url="/admin/login", status_code=303)
    form = await request.form()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "")
    if len(password) < 10:
        return templates.TemplateResponse(request, "register.html",
            _ctx(request, error="password must be at least 10 characters"),
            headers=TEMPLATE_METADATA)
    try:
        admin = create_admin(email, password)
    except Exception:
        return templates.TemplateResponse(request, "register.html",
            _ctx(request, error="could not create account (email may already exist)"),
            headers=TEMPLATE_METADATA)
    token = issue_session(admin["email"], admin["id"])
    touch_login(admin["id"])
    resp = RedirectResponse(url="/admin/", status_code=303)
    resp.set_cookie(**_session_cookie(token))
    return resp


# ---------------------------------------------------------------------------
# Sign in with Google (OAuth 2.0). Google also handles the forgotten-password
# recovery flow on its side; this panel just maps a verified Google account to
# an existing admin_users row. Requires GOOGLE_OAUTH_CLIENT_ID & _SECRET.
# ---------------------------------------------------------------------------

@router.get("/login/google", include_in_schema=False)
async def panel_login_google(request: Request):
    import secrets
    from fastapi.responses import RedirectResponse
    from . import config as cfg
    if not cfg.GOOGLE_OAUTH_CLIENT_ID or not cfg.GOOGLE_OAUTH_CLIENT_SECRET:
        return RedirectResponse(url="/admin/login?error=google-oauth-not-configured", status_code=303)
    redirect_base = cfg.GOOGLE_OAUTH_REDIRECT_BASE or str(request.base_url).rstrip("/")
    state = secrets.token_urlsafe(24)
    resp = RedirectResponse(
        url="https://accounts.google.com/o/oauth2/v2/auth"
            f"?client_id={_urlenc(cfg.GOOGLE_OAUTH_CLIENT_ID)}"
            f"&redirect_uri={_urlenc(redirect_base + '/admin/auth/google/callback')}"
            "&response_type=code"
            "&scope=openid%20email%20profile"
            f"&state={_urlenc(state)}"
            "&access_type=online",
        status_code=status.HTTP_303_SEE_OTHER)
    resp.set_cookie(key="nexus_google_oauth_state", value=state, httponly=True,
                    samesite="lax", secure=cfg.SESSION_COOKIE_SECURE, path="/admin", max_age=600)
    return resp


@router.get("/auth/google/callback", include_in_schema=False)
async def panel_google_callback(request: Request):
    from fastapi.responses import RedirectResponse
    import requests as _req
    from . import config as cfg
    from .admin_auth import find_admin_by_email, issue_session, touch_login
    if not cfg.GOOGLE_OAUTH_CLIENT_ID:
        return RedirectResponse(url="/admin/login?error=google-oauth-not-configured", status_code=303)
    error = request.query_params.get("error")
    if error:
        return RedirectResponse(url="/admin/login?error=google-denied", status_code=303)
    code = request.query_params.get("code", "")
    state = request.query_params.get("state", "")
    expected_state = request.cookies.get("nexus_google_oauth_state", "")
    if not code or not state or not expected_state or not hmac.compare_digest(state, expected_state):
        return RedirectResponse(url="/admin/login?error=google-invalid-state", status_code=303)
    redirect_base = cfg.GOOGLE_OAUTH_REDIRECT_BASE or str(request.base_url).rstrip("/")
    token_resp = _req.post("https://oauth2.googleapis.com/token", data={
        "code": code,
        "client_id": cfg.GOOGLE_OAUTH_CLIENT_ID,
        "client_secret": cfg.GOOGLE_OAUTH_CLIENT_SECRET,
        "redirect_uri": redirect_base + "/admin/auth/google/callback",
        "grant_type": "authorization_code",
    }, timeout=15)
    if token_resp.status_code != 200:
        return RedirectResponse(url="/admin/login?error=google-token-failed", status_code=303)
    id_token = token_resp.json().get("id_token", "")
    info_resp = _req.get(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"id_token": id_token}, timeout=15)
    if info_resp.status_code != 200:
        return RedirectResponse(url="/admin/login?error=google-verify-failed", status_code=303)
    info = info_resp.json()
    # Validate the identity token's audience and issuer: the token was minted
    # for THIS client_id by Google's accounts.issuer, not for an attacker's app.
    if info.get("aud") != cfg.GOOGLE_OAUTH_CLIENT_ID or info.get("iss") not in (
            "accounts.google.com", "https://accounts.google.com"):
        return RedirectResponse(url="/admin/login?error=google-invalid-state", status_code=303)
    email = (info.get("email") or "").strip().lower()
    verified = info.get("email_verified") in (True, "true")
    if not verified or not email:
        return RedirectResponse(url="/admin/login?error=google-email-unverified", status_code=303)
    admin = find_admin_by_email(email)
    if not admin:
        return RedirectResponse(url="/admin/login?error=google-no-account", status_code=303)
    if not admin["is_active"]:
        return RedirectResponse(url="/admin/login?error=account-disabled", status_code=303)
    token = issue_session(admin["email"], admin["id"])
    touch_login(admin["id"])
    resp = RedirectResponse(url="/admin/", status_code=303)
    resp.delete_cookie("nexus_google_oauth_state", path="/admin")
    resp.set_cookie(**_session_cookie(token))
    return resp


def _urlenc(value: str) -> str:
    from urllib.parse import quote
    return quote(value, safe="")


@router.get("/forgot", include_in_schema=False)
async def panel_forgot(request: Request):
    return templates.TemplateResponse(request, "forgot.html",
        _ctx(request, step="email", email="", error=None),
        headers=TEMPLATE_METADATA)


@router.post("/forgot", include_in_schema=False)
async def panel_forgot_submit(request: Request):
    from .admin_auth import (find_admin_by_email, valid_totp, issue_reset_token,
                             _rate_limited_totp, _note_totp_failure)
    from fastapi.responses import RedirectResponse
    form = await request.form()
    email = form.get("email", "").strip().lower()
    code = (form.get("code") or "").strip()
    admin = find_admin_by_email(email)
    if not admin:
        # Do not leak which emails exist; same generic step-1 outcome.
        return templates.TemplateResponse(request, "forgot.html",
            _ctx(request, step="email", email="",
                 error="if that email exists, we sent a reset request"),
            headers=TEMPLATE_METADATA)
    if not admin["totp_secret"] or not admin["totp_enabled"]:
        return templates.TemplateResponse(request, "forgot.html",
            _ctx(request, step="email", email=email,
                 error="this account has no Authenticator enrolled; contact the operator"),
            headers=TEMPLATE_METADATA)
    if not code:
        return templates.TemplateResponse(request, "forgot.html",
            _ctx(request, step="otp", email=email, error=None),
            headers=TEMPLATE_METADATA)
    try:
        key = f"forgot:{admin['id']}"
        if _rate_limited_totp(key):
            return templates.TemplateResponse(request, "forgot.html",
                _ctx(request, step="otp", email=email,
                     error="too many attempts - wait a few minutes"),
                headers=TEMPLATE_METADATA)
        if not valid_totp(admin["totp_secret"], code):
            _note_totp_failure(key)
            return templates.TemplateResponse(request, "forgot.html",
                _ctx(request, step="otp", email=email, error="invalid or expired code"),
                headers=TEMPLATE_METADATA)
    except Exception:
        return templates.TemplateResponse(request, "forgot.html",
            _ctx(request, step="otp", email=email, error="could not verify code"),
            headers=TEMPLATE_METADATA)
    token = issue_reset_token(admin["email"], admin["id"])
    return templates.TemplateResponse(request, "reset.html",
        _ctx(request, token=token, error=None), headers=TEMPLATE_METADATA)


@router.post("/reset", include_in_schema=False)
async def panel_reset_submit(request: Request):
    from .admin_auth import verify_reset_token, update_password
    from fastapi.responses import RedirectResponse
    form = await request.form()
    token = form.get("token", "")
    password = form.get("password", "")
    confirm = form.get("confirm", "")
    claim = verify_reset_token(token)
    if not claim:
        return templates.TemplateResponse(request, "reset.html",
            _ctx(request, token=token, error="reset link invalid or expired"),
            headers=TEMPLATE_METADATA)
    if password != confirm:
        return templates.TemplateResponse(request, "reset.html",
            _ctx(request, token=token, error="passwords do not match"),
            headers=TEMPLATE_METADATA)
    if len(password) < 10:
        return templates.TemplateResponse(request, "reset.html",
            _ctx(request, token=token, error="password must be at least 10 characters"),
            headers=TEMPLATE_METADATA)
    update_password(claim["admin_id"], password)
    from .audit import log_admin_action
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, claim["email"], "auth.password.reset", "admin_users",
                         claim["admin_id"], {}, {})
    resp = RedirectResponse(url="/admin/login", status_code=303)
    return resp


@router.post("/settings/enroll-totp", include_in_schema=False)
async def panel_enroll_totp(request: Request):
    """Generate a new TOTP secret for the signed-in owner and show the QR + otpauth URI.

    Re-authorizes the caller first: when TOTP is already enrolled, the CURRENT
    Authenticator code must be supplied before a replacement can be issued, so
    a hijacked session cannot lock the real owner out by swapping in an
    attacker-controlled secret."""
    from .admin_auth import (generate_totp_secret, set_totp, totp_uri,
                             find_admin_by_email, reauthorize)
    from .audit import log_admin_action
    import base64, io
    import qrcode
    claim = _current_admin(request)
    if not claim:
        raise HTTPException(status_code=401, detail="not signed in")
    form = await request.form()
    admin = find_admin_by_email(claim["email"])
    err = reauthorize(admin, form.get("current_password") or "",
                      (form.get("totp") or "").strip())
    if err:
        raise HTTPException(status_code=400, detail=err)
    secret = generate_totp_secret()
    set_totp(claim["admin_id"], secret, enabled=False)
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, claim["email"], "auth.totp.enroll", "admin_users",
                         claim["admin_id"], {}, {})
    uri = totp_uri(claim["email"], secret)
    qr = qrcode.QRCode(border=2)
    qr.add_data(uri)
    qr.make(fit=True)
    buf = io.BytesIO()
    qr.make_image(fill_color="#e6edf3", back_color="#0d1117").save(buf, format="PNG")
    qr_data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return templates.TemplateResponse(request, "security_totp_enroll.html",
        _ctx(request, secret=secret, uri=uri, qr_data_uri=qr_data_uri),
        headers=TEMPLATE_METADATA)


@router.post("/settings/confirm-totp", include_in_schema=False)
async def panel_confirm_totp(request: Request):
    """Verify a first-time TOTP code; only then is the secret enabled."""
    from .admin_auth import (find_admin_by_email, valid_totp, set_totp,
                             _rate_limited_totp, _note_totp_failure)
    from .audit import log_admin_action
    claim = _current_admin(request)
    if not claim:
        raise HTTPException(status_code=401, detail="not signed in")
    form = await request.form()
    code = (form.get("code") or "").strip()
    admin = find_admin_by_email(claim["email"])
    if not admin or not admin["totp_secret"]:
        raise HTTPException(status_code=400, detail="no TOTP secret pending enrollment")
    key = f"enroll:{claim['admin_id']}"
    if _rate_limited_totp(key):
        raise HTTPException(status_code=429, detail="too many attempts")
    if not valid_totp(admin["totp_secret"], code):
        _note_totp_failure(key)
        raise HTTPException(status_code=400, detail="invalid code - try again")
    set_totp(claim["admin_id"], admin["totp_secret"], enabled=True)
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, claim["email"], "auth.totp.confirm", "admin_users",
                         claim["admin_id"], {}, {})
    return templates.TemplateResponse(request, "ops_action.html",
        _ctx(request, message="Authenticator enrolled - recovery is now available"),
        headers=TEMPLATE_METADATA)


@router.post("/settings/change-password", include_in_schema=False)
async def panel_change_password(request: Request):
    from .admin_auth import update_password, find_admin_by_email, reauthorize
    from .audit import log_admin_action
    claim = _current_admin(request)
    if not claim:
        raise HTTPException(status_code=401, detail="not signed in")
    form = await request.form()
    password = form.get("password", "")
    confirm = form.get("confirm", "")
    if len(password) < 10:
        raise HTTPException(status_code=400, detail="password must be at least 10 characters")
    if password != confirm:
        raise HTTPException(status_code=400, detail="passwords do not match")
    admin = find_admin_by_email(claim["email"])
    err = reauthorize(admin, form.get("current_password") or "",
                      (form.get("totp") or "").strip())
    if err:
        raise HTTPException(status_code=400, detail=err)
    update_password(claim["admin_id"], password)
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, claim["email"], "auth.password.change", "admin_users",
                         claim["admin_id"], {}, {})
    return templates.TemplateResponse(request, "ops_action.html",
        _ctx(request, message="password updated"), headers=TEMPLATE_METADATA)


@router.post("/settings/change-username", include_in_schema=False)
async def panel_change_username(request: Request):
    from .admin_auth import update_email, find_admin_by_email, reauthorize
    from .audit import log_admin_action
    claim = _current_admin(request)
    if not claim:
        raise HTTPException(status_code=401, detail="not signed in")
    form = await request.form()
    email = (form.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="a valid email is required")
    if find_admin_by_email(email):
        raise HTTPException(status_code=409, detail="that email is already in use")
    admin = find_admin_by_email(claim["email"])
    err = reauthorize(admin, form.get("current_password") or "",
                      (form.get("totp") or "").strip())
    if err:
        raise HTTPException(status_code=400, detail=err)
    old_email = claim["email"]
    update_email(claim["admin_id"], email)
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, old_email, "auth.email.change", "admin_users",
                         claim["admin_id"], {"old": old_email, "new": email}, {})
    return templates.TemplateResponse(request, "ops_action.html",
        _ctx(request, message=f"username changed to {email} - use it to log in from now on"),
        headers=TEMPLATE_METADATA)


@router.post("/settings/invite", include_in_schema=False)
async def panel_invite_admin(request: Request):
    """Existing admin invites a new admin. No email is sent; the route returns a
    copyable self-service link (guest picks their own password) that the
    operator delivers over the channel they prefer. The invite is never written
    to a log file - the link appears only in this HTTP response."""
    from .admin_auth import create_admin, issue_invite_token
    from .audit import log_admin_action
    import secrets
    claim = _current_admin(request)
    if not claim:
        raise HTTPException(status_code=401, detail="not signed in")
    form = await request.form()
    email = (form.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="a valid email is required")
    try:
        # Placeholder password; the invitee sets the real one via the link.
        created = create_admin(email, secrets.token_urlsafe(18))
    except Exception:
        raise HTTPException(status_code=409, detail="that email is already an admin")
    invite = issue_invite_token(created["email"], created["id"])
    base = str(request.base_url).rstrip("/")
    link = f"{base}/admin/invite/{invite}"
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, claim["email"], "admin.invite", "admin_users",
                         created["id"], {"email": email}, {})
    return templates.TemplateResponse(request, "invite_result.html",
        _ctx(request, email=email, invite_link=link),
        headers=TEMPLATE_METADATA)


@router.get("/invite/{token}", include_in_schema=False)
async def panel_invite_accept(request: Request, token: str):
    """Guest opens the invite link: verifies and shows a set-password form."""
    from .admin_auth import verify_invite_token
    claim_invite = verify_invite_token(token)
    if not claim_invite:
        return templates.TemplateResponse(request, "reset.html",
            _ctx(request, token=token, error="invite link invalid or expired"),
            headers=TEMPLATE_METADATA)
    return templates.TemplateResponse(request, "invite.html",
        _ctx(request, token=token, email=claim_invite["email"], error=None),
        headers=TEMPLATE_METADATA)


@router.post("/invite/{token}", include_in_schema=False)
async def panel_invite_accept_submit(request: Request, token: str):
    """Guest sets their own password and becomes an active admin."""
    from .admin_auth import verify_invite_token, update_password
    from .audit import log_admin_action
    from fastapi.responses import RedirectResponse
    claim_invite = verify_invite_token(token)
    if not claim_invite:
        return templates.TemplateResponse(request, "reset.html",
            _ctx(request, token=token, error="invite link invalid or expired"),
            headers=TEMPLATE_METADATA)
    form = await request.form()
    password = form.get("password", "")
    confirm = form.get("confirm", "")
    if password != confirm:
        return templates.TemplateResponse(request, "invite.html",
            _ctx(request, token=token, email=claim_invite["email"],
                 error="passwords do not match"), headers=TEMPLATE_METADATA)
    if len(password) < 10:
        return templates.TemplateResponse(request, "invite.html",
            _ctx(request, token=token, email=claim_invite["email"],
                 error="password must be at least 10 characters"),
            headers=TEMPLATE_METADATA)
    update_password(claim_invite["admin_id"], password)
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, claim_invite["email"], "admin.invite.accept", "admin_users",
                         claim_invite["admin_id"], {}, {})
    return RedirectResponse(url="/admin/login?invited=1", status_code=303)


@router.post("/settings/admins/{admin_id}/active", include_in_schema=False)
async def panel_set_admin_active(request: Request, admin_id: int):
    """Deactivate/reactivate an admin. Sign-in for deactivated accounts is blocked."""
    from .admin_auth import set_admin_active
    from .audit import log_admin_action
    claim = _current_admin(request)
    if not claim:
        raise HTTPException(status_code=401, detail="not signed in")
    form = await request.form()
    active = (form.get("active") or "").strip().lower() in ("1", "true", "on", "yes")
    if admin_id == claim["admin_id"] and not active:
        raise HTTPException(status_code=400, detail="you cannot deactivate your own account")
    set_admin_active(admin_id, active)
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, claim["email"], "admin.active.set", "admin_users",
                         admin_id, {"active": active}, {})
    return _render_admins_fragment(request,
        search=(form.get("q") or "").strip(), status=(form.get("status") or "").strip(),
        limit=25, offset=int(form.get("offset") or 0))


@router.post("/settings/admins/{admin_id}/delete", include_in_schema=False)
async def panel_delete_admin(request: Request, admin_id: int):
    """Permanently delete an admin account (cannot delete yourself, and the
    last remaining admin cannot be deleted)."""
    from .admin_auth import delete_admin, admin_count
    from .audit import log_admin_action
    claim = _current_admin(request)
    if not claim:
        raise HTTPException(status_code=401, detail="not signed in")
    if admin_id == claim["admin_id"]:
        raise HTTPException(status_code=400, detail="you cannot delete your own account")
    if admin_count() <= 1:
        raise HTTPException(status_code=400, detail="cannot delete the last admin")
    if not delete_admin(admin_id):
        raise HTTPException(status_code=404, detail="admin not found")
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, claim["email"], "admin.delete", "admin_users",
                         admin_id, {}, {})
    form = await request.form()
    return _render_admins_fragment(request,
        search=(form.get("q") or "").strip(), status=(form.get("status") or "").strip(),
        limit=25, offset=int(form.get("offset") or 0))


@router.get("/quality")
async def panel_quality(request: Request, windows: int = 14, source: str = "",
                        sort: str = "window_desc", partial: int = 0):
    with get_db_cursor(commit=False) as cur:
        pv, mv = current_versions(cur)
        cur.execute(
            "SELECT value FROM system_config WHERE key = 'global_similarity_threshold';"
        )
        row = cur.fetchone()
        threshold = float(extract_val(row, "value", 0) or 0.88)

        cur.execute(
            "SELECT DISTINCT source FROM feedback_rollups ORDER BY 1"
        )
        sources = [extract_val(r, "source", 0) for r in (cur.fetchall() or [])]

        order = {
            "window_desc": "r.window_end DESC", "window_asc": "r.window_end ASC",
            "total_desc": "r.total_decisions DESC NULLS LAST",
            "total_asc": "r.total_decisions ASC NULLS LAST",
            "precision_desc": "r.precision_value DESC NULLS LAST",
            "precision_asc": "r.precision_value ASC NULLS LAST",
            "coverage_desc": "r.coverage_value DESC NULLS LAST",
            "coverage_asc": "r.coverage_value ASC NULLS LAST",
            "unlink_desc": "r.unlink_rate DESC NULLS LAST",
            "unlink_asc": "r.unlink_rate ASC NULLS LAST",
            "drift_desc": "r.drift_index DESC NULLS LAST",
            "drift_asc": "r.drift_index ASC NULLS LAST",
        }.get(sort, "r.window_end DESC")

        if source:
            cur.execute(
                f"""
                SELECT r.* FROM (
                    SELECT DISTINCT ON (window_end) window_start, window_end
                    FROM feedback_rollups
                    WHERE source = %s
                    ORDER BY window_end DESC
                    LIMIT %s
                ) w
                JOIN feedback_rollups r USING (window_start, window_end)
                WHERE r.source = %s
                ORDER BY {order};
                """,
                (source, windows, source)
            )
        else:
            cur.execute(
                f"""
                SELECT r.* FROM (
                    SELECT DISTINCT ON (window_end) window_start, window_end
                    FROM feedback_rollups
                    ORDER BY window_end DESC
                    LIMIT %s
                ) w
                JOIN feedback_rollups r USING (window_start, window_end)
                ORDER BY {order};
                """,
                (windows,)
            )
        rollups = cur.fetchall() or []

        def _f(row, key, idx, default=None):
            val = extract_val(row, key, idx)
            return val if val is not None else default

        t_total = sum(int(_f(r, "total_decisions", 4) or 0) for r in rollups)
        t_confirmed = sum(int(_f(r, "confirmed", 5) or 0) for r in rollups)
        t_removed = sum(int(_f(r, "removed", 6) or 0) for r in rollups)
        t_dismissed = sum(int(_f(r, "dismissed", 7) or 0) for r in rollups)
        n = len(rollups)
        def _avg(rows, key, idx):
            vals = [float(v) for r in rows if (v := _f(r, key, idx)) is not None]
            return (sum(vals) / len(vals)) if vals else None
        a_precision = _avg(rollups, "precision_value", 8)
        a_coverage = _avg(rollups, "coverage_value", 9)
        a_unlink = _avg(rollups, "unlink_rate", 10)
        a_drift = _avg(rollups, "drift_index", 11)

        cur.execute(
            """
            SELECT policy_version, knob, old_value, new_value, status, source, actor, rationale, created_at
            FROM policy_history ORDER BY id DESC LIMIT 10;
            """
        )
        history = cur.fetchall() or []
        cur.execute(
            "SELECT COUNT(*) AS total FROM integration_outbox WHERE delivery_status = 'failed';"
        )
        dlq = int(extract_val(cur.fetchone(), "total", 0) or 0)
        cur.execute(
            """
            SELECT COUNT(*) AS stuck FROM posts
            WHERE assignment_status = 'processing'
              AND assignment_updated_at < NOW() - INTERVAL '5 minutes';
            """
        )
        stuck = int(extract_val(cur.fetchone(), "stuck", 0) or 0)

    ctx = _ctx(request, pv=pv, mv=mv, threshold=threshold,
               rollups=rollups, history=history, dlq=dlq, stuck=stuck,
               sources=sources, source=source, sort=sort,
               s_total=t_total, s_confirmed=t_confirmed, s_removed=t_removed,
               s_dismissed=t_dismissed, s_precision=a_precision, s_coverage=a_coverage,
               s_unlink=a_unlink, s_drift=a_drift, window_sel=windows)
    if partial or request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(request, "quality_rollups.html", ctx,
                                          headers=TEMPLATE_METADATA)
    return templates.TemplateResponse(request,
        "quality.html", ctx, headers=TEMPLATE_METADATA)


@router.get("/decisions")
async def panel_decisions(request: Request, post_id: str | None = None,
                          hub_id: str | None = None, limit: int = 100,
                          window: str = "24h", status: str = "",
                          sort: str = "time_desc", offset: int = 0, partial: int = 0):
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    now = _dt.now(_tz.utc)
    windows = {
        "1h": now - _td(hours=1), "6h": now - _td(hours=6),
        "24h": now - _td(hours=24), "7d": now - _td(days=7),
        "30d": now - _td(days=30), "all": None,
    }
    if status not in STATUS_LABELS and status != "":
        status = ""
    order = {
        "time_desc": "d.created_at DESC", "time_asc": "d.created_at ASC",
        "sim_desc": "d.similarity DESC NULLS LAST", "sim_asc": "d.similarity ASC NULLS LAST",
        "threshold_desc": "d.threshold_used DESC NULLS LAST", "threshold_asc": "d.threshold_used ASC NULLS LAST",
    }.get(sort, "d.created_at DESC")

    pid: int | None = None
    hid: int | None = None
    if post_id is not None and str(post_id).strip().isdigit():
        pid = int(post_id)
    if hub_id is not None and str(hub_id).strip().isdigit():
        hid = int(hub_id)

    with get_db_cursor(commit=False) as cur:
        where = []
        args: list = []
        cutoff = windows.get(window)
        if cutoff is not None:
            where.append("d.created_at >= %s")
            args.append(cutoff)
        if pid is not None:
            where.append("d.post_id = %s")
            args.append(pid)
        if hid is not None:
            where.append("d.event_id = %s")
            args.append(hid)
        if status:
            where.append("d.status = %s")
            args.append(status)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        stats = {"total": 0, "assigned": 0, "candidate": 0, "unassigned": 0, "noise": 0}
        cur.execute(f"SELECT d.status, count(*) AS n FROM assignment_decision_log d {where_sql} GROUP BY d.status", args)
        for srow in cur.fetchall() or []:
            s = extract_val(srow, "status", 0); n = extract_val(srow, "n", 1)
            stats["total"] += n
            if s in stats:
                stats[s] = n

        fetch_args = args + [min(limit, 100), max(offset, 0)]
        cur.execute(
            f"""
            SELECT d.post_id, d.event_id, d.similarity, d.threshold_used, d.margin_budget,
                   d.status, d.confidence, d.policy_version, d.model_version, d.reason, d.created_at
            FROM assignment_decision_log d
            {where_sql}
            ORDER BY {order}
            LIMIT %s OFFSET %s;
            """,
            fetch_args
        )
        rows = cur.fetchall() or []
        decided = []
        for r in rows:
            pid = extract_val(r, "post_id", 0)
            decided.append({
                "post_id": pid,
                "event_id": extract_val(r, "event_id", 1),
                "similarity": extract_val(r, "similarity", 2),
                "threshold_used": extract_val(r, "threshold_used", 3),
                "margin_budget": extract_val(r, "margin_budget", 4),
                "status": extract_val(r, "status", 5),
                "status_label": STATUS_LABELS.get(extract_val(r, "status", 5), (None, extract_val(r, "status", 5)))[0],
                "status_text": STATUS_LABELS.get(extract_val(r, "status", 5), (None, extract_val(r, "status", 5)))[1],
                "confidence": extract_val(r, "confidence", 6),
                "policy_version": extract_val(r, "policy_version", 7),
                "model_version": extract_val(r, "model_version", 8),
                "reason": extract_val(r, "reason", 9),
                "reason_label": PRECISION_LABELS.get(extract_val(r, "reason", 9), (None, extract_val(r, "reason", 9)))[0],
                "reason_text": PRECISION_LABELS.get(extract_val(r, "reason", 9), (None, extract_val(r, "reason", 9)))[1],
                "created_at": extract_val(r, "created_at", 10),
                "post": post_reference(cur, pid),
                "hub": hub_reference(cur, extract_val(r, "event_id", 1)) if extract_val(r, "event_id", 1) else None,
            })
        has_more = len(decided) == min(limit, 100)
        next_offset = offset + len(decided)

    window_labels = {
        "1h": "last hour", "6h": "last 6 hours", "24h": "last 24 hours",
        "7d": "last 7 days", "30d": "last 30 days", "all": "all time",
    }
    ctx = _ctx(request, decided=decided, post_id=post_id, hub_id=hub_id,
               window=window, window_label=window_labels.get(window, "24h"),
               status=status, sort=sort, stats=stats, has_more=has_more,
               next_offset=next_offset, offset=offset, partial=bool(partial))
    if partial:
        return templates.TemplateResponse(request, "decisions_rows.html", ctx,
                                          headers=TEMPLATE_METADATA)
    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(request, "decisions_body.html", ctx,
                                          headers=TEMPLATE_METADATA)
    return templates.TemplateResponse(request, "decisions.html", ctx,
                                      headers=TEMPLATE_METADATA)


# ---------------------------------------------------------------------------
# Refs: the operator never types ids, they search content
# ---------------------------------------------------------------------------

@router.get("/refs")
async def refs_fragment(request: Request, q: str = "", kind: str = "hub", limit: int = 8,
                        slot: str = ""):
    """HTMX search fragment: labeled hub/post cards for the merge desk and the
    unlink/confirm pickers. ``slot=confirm`` renders buttons that write into the
    confirm form's slots. ALSO serves JSON for programmatic callers when the
    client requests application/json."""
    if slot not in ("", "confirm"):
        raise HTTPException(status_code=400, detail="invalid slot")
    kind = (kind or "").strip().lower()
    if kind not in ("hub", "post"):
        kind = "hub"
    with get_db_cursor(commit=False) as cur:
        found = search_refs(cur, q, limit=limit, kind=kind)
    if request.headers.get("hx-request") or not request.headers.get("accept", "").startswith("application/json"):
        return templates.TemplateResponse(request,
            "refs.html", _ctx(request, found=found, q=q, kind=kind, slot=slot),
            headers=TEMPLATE_METADATA)
    return found


# A soft ceiling on how many hub centroids we compare pairwise per render. The
# graph keeps degrading gracefully beyond this, but the desk/quality network is
# a monitoring lens, not a warehouse query.
_GRAPH_TYPE_COLORS = {
    "event": "#58a6ff",
    "debate": "#d29922",
    "analysis": "#3fb950",
    "narrative": "#bc8cff",
    "social": "#f778ba",
    "alert": "#f85149",
}
_GRAPH_CACHE: dict = {"at": 0.0, "payload": None}

_CALLSIGN_TICKERS = {
    "aws": "AWS", "amazon": "AWS", "fed": "FED", "federal": "FED",
    "microsoft": "MSFT", "apple": "AAPL", "google": "GOOG", "meta": "META",
    "openai": "OPENAI", "nvidia": "NVDA", "tesla": "TSLA", "spacex": "SPACEX",
    "barcelona": "BAR", "fc": "BAR", "nasa": "NASA", "nato": "NATO",
    "opec": "OPEC", "un": "UN", "eu": "EU", "who": "WHO", "fedex": "FDX",
    "barclays": "BARC", "musk": "MUSK", "trump": "TRUMP", "putin": "PUTIN",
    "zelensky": "ZEL", "netflix": "NFLX", "spotify": "SPOT", "uber": "UBER",
}

# slug parts that carry no memorable meaning for a short callsign
_CALLSIGN_STOP = frozenset({
    "a", "an", "the", "in", "on", "at", "by", "of", "to", "for", "and", "or",
    "not", "its", "it", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "with", "over", "under", "as", "from",
    "into", "upon", "amid", "after", "before", "has", "have", "had", "can",
    "could", "will", "would", "shall", "should", "may", "might", "us", "new",
    "major", "minor", "report", "reports", "reported", "say", "says", "said",
    "seeing", "today", "yesterday", "early", "late", "expected", "continues",
    "continued", "start", "starts", "began", "begin", "plans", "plan",
    "sees", "watch", "aftermath", "weeks", "month", "year", "another",
    "first", "second", "third", "one", "two", "three", "every", "any", "all",
    "due", "ahead", "again", "enough", "more", "among", "against", "during",
    "set", "warns", "warn", "jumps", "jump", "soars", "soar", "rises", "rise",
    "falls", "fall", "fell", "drops", "drop", "double", "triple", "back",
    "stage", "while", "via", "using", "used", "use", "make", "makes", "made",
    "take", "takes", "took", "secured", "dramatic", "victory", "completes",
})


def _callsign(handle: str, title: str, event_id: int) -> str:
    """Short operational codename for a hub: [TICKER-]WORD pairs so the network
    reads as recognizable glyphs (AWS-CLOUD, FED-RESERVE, BAR-REAL) instead of
    a wall of description. Determinstic, dedup handled by the caller."""
    text = (handle or title or "").lower().replace("_", "-")
    toks = [t for t in text.split("-") if t]
    content = [t for t in toks if t not in _CALLSIGN_STOP] or toks
    root = _CALLSIGN_TICKERS.get(content[0], "")
    if not root:
        root = content[0][:6].upper() or f"HUB{event_id}"
    suffix = ""
    if len(root) < 6:
        for t in content[1:]:
            if len(t) <= 7 and not set(t) <= set("0123456789-_."):
                suffix = t.upper()
                break
    code = f"{root}-{suffix}" if suffix else root
    return code[:17]
_GRAPH_CACHE_TTL = 30.0


@router.get("/hub-graph")
async def hub_graph(request: Request):
    """Live hub network for the desk/quality visualizations: every active hub as
    a node (short callsign + label + member count + discourse-type color) plus
    three kinds of real edges that chain the hubs into a network:

      birth    hubs that entered the stream next to each other (timeline chain)
      similar  hubs whose centroids sit close together (semantic proximity)
      voice    hubs that SHARE ACTIVE USERS - genuine community overlap

    JSON only; consumed by the canvas force simulation."""
    from .embed_io import parse_vector_literal

    now = time.time()
    cached = _GRAPH_CACHE
    if cached["payload"] is not None and now - cached["at"] < _GRAPH_CACHE_TTL:
        return cached["payload"]

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, title, handle, member_count, discourse_type, centroid::text,
                   created_at
            FROM event_hubs
            WHERE is_active = TRUE AND centroid IS NOT NULL
            ORDER BY created_at ASC, id ASC;
            """
        )
        rows = cur.fetchall() or []

    nodes = []
    vectors = []
    used_codes = set()
    for r in rows:
        hub_id = int(extract_val(r, "id", 0))
        dtype = (extract_val(r, "discourse_type", 4) or "event").strip().lower()
        title = extract_val(r, "title", 1) or f"hub #{hub_id}"
        handle = extract_val(r, "handle", 2) or ""
        code = _callsign(handle, title, hub_id)
        if code in used_codes:
            code = f"{code}-{hub_id}"
        used_codes.add(code)
        nodes.append({
            "id": hub_id,
            "label": title[:64],
            "code": code,
            "members": int(extract_val(r, "member_count", 3) or 1),
            "type": dtype if dtype else "event",
        })
        vectors.append(parse_vector_literal(extract_val(r, "centroid", 5)))

    # how much raw stream the data-source core is digesting right now. This
    # feeds the globe's radius: more content tapped -> the sphere swells -> it
    # outgrows the reserved window and parts must be rotated into view.
    volume = 0
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT count(*) FROM posts
                WHERE event_id IS NOT NULL AND deleted_at IS NULL;
                """
            )
            row = cur.fetchone()
            if row:
                volume = int(extract_val(row, "count", 0))
    except Exception:
        volume = len(nodes)

    edges = []
    # ordered pair -> edge; strong kinds win when two relationships share a pair
    by_pair: dict = {}

    def add_edge(source, target, kind, weight):
        pair = (source, target) if source < target else (target, source)
        if pair in by_pair:
            return
        by_pair[pair] = {"source": source, "target": target, "kind": kind,
                         "weight": round(max(0.05, min(float(weight), 1.0)), 3)}

    birth_ids = [n["id"] for n in nodes]
    for i in range(len(birth_ids) - 1):
        add_edge(birth_ids[i], birth_ids[i + 1], "birth", 1.0)

    if len(vectors) >= 2:
        try:
            mat = np.vstack([v / (np.linalg.norm(v) + 1e-9) for v in vectors])
            sim = cosine_similarity(mat)
        except Exception:
            sim = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                w = float(sim[i][j])
                if w >= 0.18:
                    add_edge(nodes[i]["id"], nodes[j]["id"], "similar", w)

    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                WITH multi AS (
                    SELECT user_id, array_agg(DISTINCT event_id) AS hubs
                    FROM posts
                    WHERE event_id IS NOT NULL AND deleted_at IS NULL
                    GROUP BY user_id
                )
                SELECT hubs FROM multi WHERE array_length(hubs, 1) > 1;
                """
            )
            voice_count = {}  # (a, b) -> number of shared active users
            for r in (cur.fetchall() or []):
                hub_ids = r.get("hubs") or []
                for a in range(len(hub_ids)):
                    for b in range(a + 1, len(hub_ids)):
                        pair = tuple(sorted((hub_ids[a], hub_ids[b])))
                        voice_count[pair] = voice_count.get(pair, 0) + 1
            for (a, b), cnt in sorted(voice_count.items()):
                add_edge(a, b, "voice", min(cnt, 8))
    except Exception:
        # shared-voice probing is best-effort; the graph still renders
        pass

    payload = {"nodes": nodes, "edges": list(by_pair.values()),
               "volume": int(volume),
               "generated_at": round(time.time(), 3)}
    _GRAPH_CACHE["at"] = now
    _GRAPH_CACHE["payload"] = payload
    return payload


@router.get("/hub-detail")
async def hub_detail(request: Request):
    """Tiny on-click detail for a globe unit - never the full hub. What a
    human reaching for a node actually wants: the id, the callsign/name, and a
    few member post names. Kept deliberately light (see refs.hub_reference)."""
    raw = request.query_params.get("id")
    try:
        hub_id = int(raw or "")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="hub id required")
    with get_db_cursor(commit=False) as cur:
        ref = hub_reference(cur, hub_id, member_previews=6)
    if ref.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="hub not found")
    code = _callsign(ref.get("handle"), ref.get("title"), hub_id)
    previews = [p for p in (ref.get("member_previews") or []) if p]
    return {
        "id": hub_id,
        "code": code,
        "label": ref.get("label") or f"hub-{hub_id}",
        "type": ref.get("discourse_type") or "event",
        "members": int(ref.get("member_count") or 0),
        "posts": [{"name": p} for p in previews[:6]],
    }


@router.post("/merge")
async def panel_merge(request: Request):
    """Merge desk action. The form posts the two hubs picked from the search
    cards (hidden inputs rendered by /admin/pick); the response announces
    WHICH hubs were merged (references). Tagged as a PANEL (human) action so
    rollups never misattribute it to the system."""
    form = await request.form()
    source = int(form.get("source_event_id", 0) or 0)
    target = int(form.get("target_event_id", 0) or 0)
    if source == target or source == 0 or target == 0:
        raise HTTPException(status_code=400, detail="choose two distinct hubs")
    try:
        result = client_merge(source, target, DEFAULT_ACTOR, initiated_by="panel")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return templates.TemplateResponse(request,
        "merge_result.html", _ctx(request, result=result, error=None),
        headers=TEMPLATE_METADATA)


@router.post("/unlink")
async def panel_unlink(request: Request):
    form = await request.form()
    post_id = int(form.get("post_id", 0) or 0)
    event_id = int(form.get("event_id", 0) or 0)
    if not post_id or not event_id:
        raise HTTPException(status_code=400, detail="select a post first")
    from .corrections import unlink_post as unlink_correction, CorrectionError
    try:
        with get_db_cursor() as cur:
            result = unlink_correction(cur, post_id, event_id, actor="panel-user", detailed=True)
    except CorrectionError as e:
        return templates.TemplateResponse(request,
            "unlink_result.html", _ctx(request, result=None, error=str(e)),
            headers=TEMPLATE_METADATA)
    return templates.TemplateResponse(request,
        "unlink_result.html", _ctx(request, result=result, error=None),
        headers=TEMPLATE_METADATA)


@router.post("/confirm")
async def panel_confirm(request: Request):
    """Pin a candidate post into a chosen hub as a human-confirmed assignment.
    Mirror of the unlink flow: pick by content via search cards, then submit."""
    form = await request.form()
    post_id = int(form.get("post_id", 0) or 0)
    event_id = int(form.get("event_id", 0) or 0)
    if not post_id or not event_id:
        raise HTTPException(status_code=400, detail="select a post and a hub first")
    from .corrections import confirm_post as confirm_correction, CorrectionError
    try:
        with get_db_cursor() as cur:
            result = confirm_correction(cur, post_id, event_id, actor="panel-user", detailed=True)
    except CorrectionError as e:
        return templates.TemplateResponse(request,
            "confirm_result.html", _ctx(request, result=None, error=str(e)),
            headers=TEMPLATE_METADATA)
    return templates.TemplateResponse(request,
        "confirm_result.html", _ctx(request, result=result, error=None),
        headers=TEMPLATE_METADATA)


# --- Merge-desk pickers: hidden-input fragments rendered INTO the desk forms.
# The operator selects by CONTENT (search -> labeled card -> pick); the hidden
# inputs carry the ids to the merge/unlink handlers. No bare typing required.
# Each pick response ALSO clears the search picker it was chosen from (OOB),
# so the remaining options disappear once one is selected.
_PICKER_IDS = {
    "source-picker", "target-picker",
    "post-picker", "confirm-target-picker", "confirm-post-picker",
}
@router.post("/pick")
async def panel_pick(request: Request):
    form = await request.form()
    role = form.get("role", "source")
    if role not in ("source", "target"):
        raise HTTPException(status_code=400, detail="role must be source or target")
    name = form.get("name") or f"{role}_event_id"
    if name not in ("source_event_id", "target_event_id", "event_id"):
        raise HTTPException(status_code=400, detail="invalid slot name")
    picker = form.get("picker", "")
    if picker not in _PICKER_IDS:
        raise HTTPException(status_code=400, detail="invalid picker")
    try:
        event_id = int(form.get("event_id", 0) or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid event_id")
    with get_db_cursor(commit=False) as cur:
        hub = hub_reference(cur, event_id)
        if hub.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="hub not found")
    return templates.TemplateResponse(request,
        "picked.html", _ctx(request, name=name, value=event_id, hub=hub, picker=picker),
        headers=TEMPLATE_METADATA)


@router.post("/pick-post")
async def panel_pick_post(request: Request):
    form = await request.form()
    try:
        post_id = int(form.get("post_id", 0) or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid post_id")
    name = form.get("name") or "post_id"
    if name != "post_id":
        raise HTTPException(status_code=400, detail="invalid slot name")
    picker = form.get("picker", "")
    if picker not in _PICKER_IDS:
        raise HTTPException(status_code=400, detail="invalid picker")
    with get_db_cursor(commit=False) as cur:
        post = post_reference(cur, post_id)
        if post.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="post not found")
    return templates.TemplateResponse(request,
        "picked_post.html", _ctx(request, post=post, picker=picker), headers=TEMPLATE_METADATA)


@router.get("/pick-clear")
async def panel_pick_clear(request: Request):
    return Response("", media_type="text/html")


@router.post("/merge/{merge_id}/reopen")
async def panel_reopen(request: Request, merge_id: int):
    form = await request.form()
    note = form.get("note") or None
    with get_db_cursor(commit=True) as cur:
        try:
            result = reopen_merge(cur, merge_id, DEFAULT_ACTOR, note)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    return templates.TemplateResponse(request,
        "reopen_result.html", _ctx(request, result=result, error=None),
        headers=TEMPLATE_METADATA)


@router.get("/desk")
async def panel_desk(request: Request):
    """Merge desk: search-anchored hub pickers, no bare ids. The operator pastes
    any remembered fragment -> labeled cards -> selects source/target -> merge.
    The desk also hosts the unlink/confirm correction flow for a searched post."""
    return templates.TemplateResponse(request,"desk.html", _ctx(request, compact=True),
                                      headers=TEMPLATE_METADATA)


@router.get("/merges")
async def panel_merges(request: Request, limit: int = 50, status: str = "", q: str = ""):
    status = (status or "").strip().lower()
    q = (q or "").strip()
    where = []
    params: list = []
    if status in ("merged", "reopened"):
        where.append("hm.status = %s")
        params.append(status)
    if q:
        qpat = f"%{q}%"
        where.append(
            "(hs.title ILIKE %s OR ht.title ILIKE %s "
            "OR hm.initiated_by ILIKE %s OR hm.opened_note ILIKE %s)"
        )
        params += [qpat, qpat, qpat, qpat]
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            f"""
            SELECT hm.id, hm.source_event_id, hm.target_event_id, hm.initiated_by, hm.status,
                   hm.snapshot_member_count, hm.opened_note, hm.reopened_note, hm.created_at, hm.reopened_at
            FROM hub_merges hm
            LEFT JOIN event_hubs hs ON hs.id = hm.source_event_id
            LEFT JOIN event_hubs ht ON ht.id = hm.target_event_id
            {where_sql}
            ORDER BY hm.id DESC LIMIT %s;
            """,
            (*params, min(limit, 200))
        )
        merges = []
        for r in (cur.fetchall() or []):
            sid = int(extract_val(r, "source_event_id", 1))
            tid = int(extract_val(r, "target_event_id", 2))
            merges.append({
                "id": extract_val(r, "id", 0),
                "source": hub_reference(cur, sid),
                "target": hub_reference(cur, tid),
                "initiated_by": extract_val(r, "initiated_by", 3),
                "status": extract_val(r, "status", 4),
                "moved": extract_val(r, "snapshot_member_count", 5),
                "opened_note": extract_val(r, "opened_note", 6),
                "reopened_note": extract_val(r, "reopened_note", 7),
                "created_at": extract_val(r, "created_at", 8),
                "reopened_at": extract_val(r, "reopened_at", 9),
            })
        # The screen only renders the most recent rows; the headline aggregates
        # must count the whole table, not the paginated slice.
        cur.execute(
            """
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE status = 'merged') AS merged,
                   COUNT(*) FILTER (WHERE status = 'reopened') AS reopened,
                   COALESCE(SUM(snapshot_member_count) FILTER (WHERE status = 'merged'), 0) AS moved
            FROM hub_merges;
            """
        )
        agg = cur.fetchone()
        total_all = int(extract_val(agg, "total", 0) or 0)
        merged_all = int(extract_val(agg, "merged", 1) or 0)
        reopened_all = int(extract_val(agg, "reopened", 2) or 0)
        moved_all = int(extract_val(agg, "moved", 3) or 0)
    return templates.TemplateResponse(request,
        "merges.html", _ctx(request, merges=merges,
                             total_merges=total_all,
                             merged_count=merged_all,
                             reopened_count=reopened_all,
                             moved_total=moved_all,
                             f_status=status, f_q=q),
        headers=TEMPLATE_METADATA)


# ---------------------------------------------------------------------------
# Calibration (with explicit warnings on performance-affecting changes)
# ---------------------------------------------------------------------------

def _calibration_history(cur) -> list:
    cur.execute(
        "SELECT id, policy_version, knob, old_value, new_value, status, source, actor, rationale, created_at "
        "FROM policy_history ORDER BY id DESC LIMIT 30;"
    )
    return cur.fetchall() or []


@router.get("/calibration")
async def panel_calibration(request: Request):
    with get_db_cursor(commit=False) as cur:
        pv, mv = current_versions(cur)
        cur.execute("SELECT value FROM system_config WHERE key = 'global_similarity_threshold';")
        row = cur.fetchone()
        threshold = float(extract_val(row, "value", 0) or 0.88)
        history = _calibration_history(cur)
    knobs = [{
        "name": k,
        "lo": v[0], "hi": v[1], "decimals": v[2],
        "range": KNOB_RANGES[k],
        "warning": KNOB_WARNINGS.get(k),
        "current": threshold if k == "global_similarity_threshold" else None,
    } for k, v in KNOB_RANGES.items()]
    return templates.TemplateResponse(request,
        "calibration.html", _ctx(request, pv=pv, mv=mv, threshold=threshold,
                                 knobs=knobs, history=history),
        headers=TEMPLATE_METADATA)


@router.post("/calibration/apply")
async def panel_calibration_apply(request: Request):
    form = await request.form()
    knob = form.get("knob", "")
    try:
        new_value = float(form.get("new_value", ""))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid value")
    rationale = form.get("rationale") or None
    with get_db_cursor(commit=True) as cur:
        try:
            result = apply_policy_change(cur, knob, new_value, "panel", DEFAULT_ACTOR, rationale)
            log_admin_action(cur, DEFAULT_ACTOR, "policy.apply", "system_config", knob,
                             {"knob": knob}, {"new_value": result["new_value"],
                                              "policy_version": result["policy_version"]})
        except ValueError as e:
            # Invalid value / no-op (already current): render inline instead of
            # surfacing as an error page, and still refresh the history+knob.
            return templates.TemplateResponse(request,
                "calibration_result.html", _ctx(request, result=None, assessed=None,
                                                 history=_calibration_history(cur), error=str(e)),
                headers=TEMPLATE_METADATA)
        assessed = result.get("impact_estimate")
        history = _calibration_history(cur)
    return templates.TemplateResponse(request,
        "calibration_result.html", _ctx(request, result=result, assessed=assessed, history=history, error=None),
        headers=TEMPLATE_METADATA)


@router.post("/calibration/propose")
async def panel_calibration_propose(request: Request):
    form = await request.form()
    knob = form.get("knob", "")
    try:
        new_value = float(form.get("new_value", ""))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="invalid value")
    rationale = form.get("rationale") or None
    with get_db_cursor(commit=True) as cur:
        try:
            result = propose_policy_change(cur, knob, new_value, "panel", DEFAULT_ACTOR, rationale)
        except ValueError as e:
            return templates.TemplateResponse(request,
                "calibration_result.html", _ctx(request, result=None, assessed=None,
                                                 history=_calibration_history(cur), error=str(e)),
                headers=TEMPLATE_METADATA)
        assessed = result.get("impact_estimate")
        history = _calibration_history(cur)
    return templates.TemplateResponse(request,
        "calibration_result.html", _ctx(request, result=result, assessed=assessed, history=history, error=None),
        headers=TEMPLATE_METADATA)


@router.post("/calibration/{history_id}/revert")
async def panel_calibration_revert(request: Request, history_id: int):
    form = await request.form()
    knob = form.get("knob")
    if not knob:
        raise HTTPException(status_code=400, detail="knob required")
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "SELECT old_value FROM policy_history WHERE id = %s AND status = 'applied';",
            (history_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="applied change not found")
        old_value = float(extract_val(row, "old_value", 0))
        try:
            result = apply_policy_change(cur, knob, old_value, "panel-revert", DEFAULT_ACTOR,
                                         f"revert of policy_history#{history_id}")
        except ValueError as e:
            # e.g. the target value is already current (already reverted) - a
            # no-op revert must not pollute the history either.
            return templates.TemplateResponse(request,
                "calibration_result.html", _ctx(request, result=None, assessed=None,
                                                 history=_calibration_history(cur), error=str(e)),
                headers=TEMPLATE_METADATA)
        log_admin_action(cur, DEFAULT_ACTOR, "policy.revert", "policy_history", history_id,
                         {"knob": knob, "old_value": old_value}, {"new_value": result["new_value"]})
        assessed = result.get("impact_estimate")
        history = _calibration_history(cur)
    return templates.TemplateResponse(request,
        "calibration_result.html", _ctx(request, result=result, assessed=assessed, history=history, error=None),
        headers=TEMPLATE_METADATA)


# ---------------------------------------------------------------------------
# Ops: queue health + recovery actions
# ---------------------------------------------------------------------------

@router.get("/ops")
async def panel_ops(request: Request):
    from redis import RedisError
    from .queues import get_redis_client
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT COUNT(*) AS total FROM integration_outbox WHERE delivery_status = 'failed';")
        dlq = int(extract_val(cur.fetchone(), "total", 0) or 0)
        cur.execute(
            "SELECT COUNT(*) AS leased FROM integration_outbox WHERE delivery_status = 'leased' AND lease_until >= NOW();"
        )
        leased = int(extract_val(cur.fetchone(), "leased", 0) or 0)
        hb_hour = datetime.now(timezone.utc) - timedelta(hours=2)
        cur.execute("SELECT MAX(window_end) AS last_rollup FROM feedback_rollups WHERE source = 'human';")
        last_rollup = extract_val(cur.fetchone(), "last_rollup", 0)
        rollup_fresh = last_rollup is not None and last_rollup >= hb_hour
        cur.execute(
            """
            SELECT COUNT(*) AS stuck FROM posts
            WHERE assignment_status = 'processing'
              AND assignment_updated_at < NOW() - INTERVAL '5 minutes';
            """
        )
        stuck = int(extract_val(cur.fetchone(), "stuck", 0) or 0)
        cur.execute(
            "SELECT COUNT(*) AS pending FROM posts WHERE assignment_status = 'pending' "
            "AND assignment_updated_at < NOW() - INTERVAL '15 minutes';"
        )
        stalled_pending = int(extract_val(cur.fetchone(), "pending", 0) or 0)
        cur.execute(
            "SELECT consumer_id, is_active, rate_limit_per_minute, last_used_at FROM api_consumers ORDER BY id;"
        )
        consumers = cur.fetchall() or []
    redis_state = "unknown"
    queue_len = None
    r = get_redis_client()
    try:
        queue_len = r.llen("celery")
        redis_state = "ok"
    except RedisError:
        redis_state = "down"
    return templates.TemplateResponse(request,
        "ops.html", _ctx(request,
                         dlq=dlq, leased=leased, stuck=stuck, stalled_pending=stalled_pending,
                         last_rollup=last_rollup, rollup_fresh=rollup_fresh,
                         redis_state=redis_state, queue_len=queue_len, consumers=consumers),
        headers=TEMPLATE_METADATA)


@router.post("/ops/resume-stuck")
async def panel_resume_stuck(request: Request):
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE posts SET assignment_status = 'pending', assignment_updated_at = NOW()
            WHERE assignment_status = 'processing'
              AND assignment_updated_at < NOW() - INTERVAL '5 minutes'
            RETURNING id;
            """
        )
        n = len(cur.fetchall() or [])
        log_admin_action(cur, DEFAULT_ACTOR, "ops.resume_stuck", "posts", None, {"count": n}, {})
    return templates.TemplateResponse(request,"ops_action.html",
                                      _ctx(request, message=f"resumed {n} stuck posts to pending"),
                                      headers=TEMPLATE_METADATA)


@router.post("/ops/retry-dlq")
async def panel_retry_dlq(request: Request):
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE integration_outbox
            SET delivery_status = 'pending', attempts = 0, available_at = NOW(), last_error = NULL
            WHERE delivery_status = 'failed'
            RETURNING id;
            """
        )
        n = len(cur.fetchall() or [])
        log_admin_action(cur, DEFAULT_ACTOR, "ops.retry_dlq", "integration_outbox", None, {"count": n}, {})
    return templates.TemplateResponse(request,"ops_action.html",
                                      _ctx(request, message=f"re-queued {n} DLQ events for dispatch"),
                                      headers=TEMPLATE_METADATA)


# ---------------------------------------------------------------------------
# Consumers (per-party pull credentials)
# ---------------------------------------------------------------------------

@router.post("/consumers")
async def panel_create_consumer(request: Request):
    form = await request.form()
    name = form.get("name", "").strip()
    limit = int(form.get("rate_limit_per_minute", 60) or 60)
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    consumer_id = secrets.token_hex(8)
    token = secrets.token_urlsafe(32)
    token_hash = hash_token(token)
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO api_consumers (consumer_id, name, token_hash, rate_limit_per_minute)
            VALUES (%s, %s, %s, %s) RETURNING id;
            """,
            (consumer_id, name, token_hash, limit)
        )
        cid = extract_val(cur.fetchone(), "id", 0)
        log_admin_action(cur, DEFAULT_ACTOR, "consumer.create", "api_consumers", cid,
                         {"consumer_id": consumer_id}, {"name": name})
    return templates.TemplateResponse(request,
        "consumer_created.html",
        _ctx(request, consumer_id=consumer_id, token=token, name=name),
        headers=TEMPLATE_METADATA)


@router.post("/consumers/{consumer_id}/toggle")
async def panel_toggle_consumer(request: Request, consumer_id: str):
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE api_consumers SET is_active = NOT is_active WHERE consumer_id = %s RETURNING is_active;",
            (consumer_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="consumer not found")
        active = extract_val(row, "is_active", 0)
    return templates.TemplateResponse(request,
        "ops_action.html",
        _ctx(request, message=f"consumer {consumer_id} {'enabled' if active else 'disabled'}"),
        headers=TEMPLATE_METADATA)


@router.post("/consumers/{consumer_id}/delete")
async def panel_delete_consumer(request: Request, consumer_id: str):
    from .consumers import delete_consumer
    from .audit import log_admin_action
    if not delete_consumer(consumer_id):
        raise HTTPException(status_code=404, detail="consumer not found")
    with get_db_cursor(commit=True) as cur:
        log_admin_action(cur, DEFAULT_ACTOR, "consumer.delete", "api_consumers", None,
                         {"consumer_id": consumer_id}, {})
    return Response(content="", status_code=200)


# ---------------------------------------------------------------------------
# Audit + settings inventory
# ---------------------------------------------------------------------------

@router.get("/audit")
async def panel_audit(request: Request, actor: str = "", domain: str = "",
                      entity: str = "", window: str = "", q: str = "",
                      limit: int = 0, offset: int = 0, frag: str = ""):
    """Audit log: scannable entries with filters, paging, and a per-entry diff.

    Snapshots are stored as JSONB and used to be rendered raw, which read as a
    data dump. Each entry is now reduced to a labelled summary line and expands
    on demand into a field level diff of what actually changed.

    ``frag=results`` swaps the table region (filter submits) and ``frag=rows``
    appends the next page (load more).
    """
    from . import audit_view as av

    window = window or av.DEFAULT_WINDOW
    limit = limit or av.DEFAULT_LIMIT
    with get_db_cursor(commit=False) as cur:
        entries, total = av.fetch_entries(
            cur, actor=actor, domain=domain, entity=entity,
            window=window, query=q, limit=limit, offset=offset)
        stats = av.audit_stats(cur)
        options = av.filter_options(cur)

    from urllib.parse import urlencode

    loaded = offset + len(entries)
    # The "load more" sentinel reuses the active filters; building the query
    # here keeps the template free of string assembly.
    carry = {k: v for k, v in (("actor", actor), ("domain", domain),
                               ("entity", entity), ("window", window),
                               ("q", q.strip()), ("limit", limit)) if v}
    more_url = "/admin/audit?" + urlencode({**carry, "frag": "rows", "offset": loaded})
    ctx = _ctx(request,
               entries=entries, total=total, stats=stats, options=options,
               actor=actor, domain=domain, entity=entity, window=window,
               window_label=av.window_label(window), query=q,
               windows=av.AUDIT_WINDOWS, limit=limit, offset=offset,
               has_more=loaded < total, next_offset=loaded, more_url=more_url,
               has_filters=bool(actor or domain or entity or q.strip()
                                or window != av.DEFAULT_WINDOW))
    template = {"rows": "audit_rows.html",
                "results": "audit_results.html"}.get(frag, "audit.html")
    return templates.TemplateResponse(request, template, ctx, headers=TEMPLATE_METADATA)


@router.get("/settings")
async def panel_settings(request: Request):
    from . import config as cfg
    import re as _re
    secret_hint = _re.compile(r"(TOKEN|SECRET|PASSWORD|KEY|HASH)", _re.IGNORECASE)
    env_knobs = []
    # Group the config into the sections an operator actually thinks about.
    _SECTIONS = [
        ("auth", "Authentication", r"^(API_AUTH_TOKEN|GOOGLE_OAUTH_)"),
        ("infra", "Infrastructure", r"^(DATABASE_URL|REDIS_URL|PACKAGE_DIR)"),
        ("model", "Model", r"^(BASE_MODEL_NAME|REQUIRE_LORA_ADAPTER|LORA_|MODEL_VERSION|POLICY_VERSION)"),
        ("thresholds", "Assignment thresholds", r"^(SIMILARITY_MARGIN|AUTO_ASSIGN_THRESHOLD|CANDIDATE_THRESHOLD|CENTROID_UPDATE_THRESHOLD|BIRTH_|ANCHOR_WEIGHT|CENTROID_MAX_MEMBERS)"),
        ("processing", "Processing & reliability", r"^(BATCH_SIZE|SWEEPER_|OUTBOX_|CLAIM_CHUNK_SIZE|ASSIGN_CHUNK_SIZE|THRESHOLD_CACHE_TTL|BULK_ASSIGN|CLAIM_DURABLE)"),
        ("intelligence", "Rollups & autotuning", r"^(ROLLUP_WINDOW_HOURS|TUNE_WINDOW_DAYS|MIN_FEEDBACK_SAMPLES|AUTO_TUNE_|AUTO_APPLY_THRESHOLD)"),
        ("delivery", "Event delivery", r"^(EVENT_DELIVERY_MODE|EVENT_WEBHOOK_URL|EVENT_WEBHOOK_SIGNING_SECRET)"),
        ("ingest", "Ingest hints", r"^(INGEST_HINTS_)"),
    ]
    for name in sorted(dir(cfg)):
        if not name.isupper():
            continue
        val = getattr(cfg, name)
        if not isinstance(val, (int, float, str, bool)):
            continue
        sensitive = bool(secret_hint.search(name))
        env_set = name in os.environ
        section = next((s for s in _SECTIONS if _re.match(s[2], name)), ("other", "Other", ""))
        env_knobs.append({
            "name": name,
            "section": section[0],
            "section_label": section[1],
            "value": "********" if sensitive else val,
            "sensitive": sensitive,
            "env_set": env_set,
            "kind": "bool" if isinstance(val, bool) else
                    ("int" if isinstance(val, int) else
                     ("float" if isinstance(val, float) else "str")),
        })
    env_sections = []
    for code, label, _pat in _SECTIONS:
        items = [k for k in env_knobs if k["section"] == code]
        if items:
            env_sections.append({"code": code, "label": label, "knobs": items})
    sec_counts = {"secrets": sum(1 for k in env_knobs if k["sensitive"]),
                  "env_set": sum(1 for k in env_knobs if k["env_set"]),
                  "total": len(env_knobs)}
    current_admin = _current_admin(request)
    totp_state = {"email": "", "totp_enabled": False}
    if current_admin:
        from .admin_auth import find_admin_by_email
        admin_row = find_admin_by_email(current_admin["email"])
        if admin_row:
            totp_state = {"email": admin_row["email"], "totp_enabled": admin_row["totp_enabled"]}
    from .admin_auth import list_admins, admin_stats
    admins, total_admins = list_admins(limit=25, offset=0)
    try:
        from .consumers import list_consumers  # type: ignore
        consumers = list_consumers()
    except Exception:
        consumers = []
    c_active = sum(1 for c in consumers if c.get("is_active"))
    c_tokens = len(consumers)
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT value FROM system_config WHERE key = 'global_similarity_threshold';")
        row = cur.fetchone()
        threshold = float(extract_val(row, "value", 0) or 0.88)
    knobs = [{
        "name": k,
        "lo": v[0], "hi": v[1], "decimals": v[2],
        "warning": KNOB_WARNINGS.get(k),
        "current": threshold if k == "global_similarity_threshold" else None,
    } for k, v in KNOB_RANGES.items()]
    for _k in knobs:
        if _k["current"] is not None:
            _k["current_disp"] = f"{{0:.{_k['decimals']}f}}".format(_k["current"])
    return templates.TemplateResponse(request,
        "settings.html", _ctx(request,
                              env_sections=env_sections, env_knobs=env_knobs,
                              env_counts=sec_counts,
                              knob_ranges=KNOB_RANGES, knobs=knobs,
                              warnings=KNOB_WARNINGS, current_admin=totp_state,
                              admins=admins, total=total_admins, admin_stats=admin_stats(),
                              search="", status="",
                              has_more=25 < total_admins, next_offset=25, offset=0, limit=25,
                              my_admin_id=(current_admin or {}).get("admin_id", 0),
                              consumers=consumers, c_active=c_active, c_tokens=c_tokens,
                              delivery={
                                  "mode": cfg.EVENT_DELIVERY_MODE,
                                  "url": cfg.EVENT_WEBHOOK_URL or "",
                                  "signing": bool(cfg.EVENT_WEBHOOK_SIGNING_SECRET),
                              }),
        headers=TEMPLATE_METADATA)
