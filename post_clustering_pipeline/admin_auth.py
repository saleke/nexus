"""Owner authentication for the control plane.

Email + password login with RFC 6238 TOTP (Google Authenticator compatible)
recovery. Sessions are short-lived JWT (HS256) cookies whose signing secret is
generated once and persisted in ``system_config`` so restarting the API does
not invalidate everyone's session.

Security posture:
  * Passwords are salted PBKDF2-SHA256 (200k iterations) - stdlib only.
  * TOTP secrets are base32 and stateless-verified with a +-1 step window.
  * Forgot-password requires a valid TOTP code: possession of the enrolled
    Authenticator secret is the proof of identity; a code can only be redeemed
    once (rotated on use).
  * Failed login TOTP attempts are rate-limited in-process.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from datetime import datetime, timedelta, timezone

import jwt

from .db import get_db_cursor, extract_val
from .config import DATABASE_URL

COOKIE_NAME = "nexus_admin_session"
SESSION_TTL_HOURS = 12
FORGOT_TTL_MINUTES = 15
INVITE_TTL_DAYS = 7
PBKDF2_ITERATIONS = 200_000
TOTP_STEP_SECONDS = 30
TOTP_DIGITS = 6
_TOTP_FAIL_BUCKET: dict[str, list[float]] = {}


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters_b64, salt_b64, hash_b64 = stored.split("$")
        iters = int(iters_b64)
        salt = base64.b64decode(salt_b64)
        digest = base64.b64decode(hash_b64)
    except (ValueError, TypeError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(candidate, digest)


# ---------------------------------------------------------------------------
# TOTP (RFC 6238, SHA1 / 6 digits / 30s step - Google Authenticator)
# ---------------------------------------------------------------------------

def generate_totp_secret() -> str:
    """Return a new random base32 secret (32 chars, the Authy/GA default length)."""
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _totp_counter(time_s: float, step: int = TOTP_STEP_SECONDS) -> int:
    return int(time_s / step)


def totp_code(secret_b32: str, time_s: float | None = None, step: int = TOTP_STEP_SECONDS) -> str:
    key = base64.b32decode(secret_b32.upper() + "=" * ((8 - len(secret_b32) % 8) % 8))
    msg = struct.pack(">Q", _totp_counter(time_s if time_s is not None else time.time(), step))
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** TOTP_DIGITS)
    return str(code).zfill(TOTP_DIGITS)


def totp_uri(email: str, secret_b32: str, issuer: str = "Nexus") -> str:
    from urllib.parse import quote
    return "otpauth://totp/{issuer}:{email}?secret={secret}&issuer={issuer}".format(
        issuer=quote(issuer), email=quote(email), secret=secret_b32)


def valid_totp(secret_b32: str, code: str, window: int = 1) -> bool:
    """Verify a TOTP code against the current 30s step (and +-window)."""
    code = (code or "").strip()
    if not code.isdigit() or len(code) != TOTP_DIGITS:
        return False
    now = time.time()
    for offset in range(-window, window + 1):
        if hmac.compare_digest(totp_code(secret_b32, now + offset * TOTP_STEP_SECONDS), code):
            return True
    return False


def _rate_limited_totp(key: str, limit: int = 5, window_s: int = 300) -> bool:
    """Return True when the key has exceeded TOTP failure attempts."""
    now = time.time()
    bucket = [t for t in _TOTP_FAIL_BUCKET.get(key, []) if now - t < window_s]
    if len(bucket) >= limit:
        _TOTP_FAIL_BUCKET[key] = bucket
        return True
    return False


def _note_totp_failure(key: str):
    _TOTP_FAIL_BUCKET.setdefault(key, []).append(time.time())
    _TOTP_FAIL_BUCKET[key] = [t for t in _TOTP_FAIL_BUCKET[key] if time.time() - t < 600]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _session_secret() -> str:
    with get_db_cursor(commit=True) as cur:
        cur.execute("SELECT value_text FROM system_config WHERE key = 'admin_session_secret'")
        row = cur.fetchone()
        stored = extract_val(row, "value_text", 0)
        if stored:
            return stored
        # Bootstrap race: first writer wins. A concurrent caller's INSERT hits
        # the primary key and does nothing (or repairs a NULL/empty value), and
        # the re-read below returns the committed winner, so both callers agree
        # on ONE secret. Previously a second caller overwrote the stored secret
        # and silently invalidated every session issued by the first.
        secret = secrets.token_urlsafe(48)
        cur.execute(
            "INSERT INTO system_config (key, value, value_text) VALUES ('admin_session_secret', 0, %s) "
            "ON CONFLICT (key) DO UPDATE SET value_text = EXCLUDED.value_text "
            "WHERE system_config.value_text IS NULL OR system_config.value_text = '';",
            (secret,)
        )
        cur.execute("SELECT value_text FROM system_config WHERE key = 'admin_session_secret'")
        row = cur.fetchone()
        return extract_val(row, "value_text", 0)


# ---------------------------------------------------------------------------
# Single-use grants for reset / invite tokens
# ---------------------------------------------------------------------------
# Reset and invite links are stateless JWTs, so without a server-side record a
# captured link stays redeemable for its whole TTL. Each issuance writes the
# token's ``jti`` into system_config under a per-(purpose, admin) key; redeeming
# atomically deletes that key, so a token can be used at most once and issuing a
# newer link supersedes the old one.

def _grant_key(purpose: str, admin_id: int) -> str:
    return f"token:{purpose}:{int(admin_id)}"


def _store_grant(purpose: str, admin_id: int, jti: str) -> None:
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO system_config (key, value, value_text) VALUES (%s, 0, %s) "
            "ON CONFLICT (key) DO UPDATE SET value_text = EXCLUDED.value_text;",
            (_grant_key(purpose, admin_id), jti),
        )


def _grant_active(purpose: str, admin_id: int, jti: str) -> bool:
    if not jti:
        return False
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT value_text FROM system_config WHERE key = %s;",
                    (_grant_key(purpose, admin_id),))
        row = cur.fetchone()
    return extract_val(row, "value_text", 0) == jti


def claim_grant(purpose: str, admin_id: int, jti: str) -> bool:
    """Atomically consume a single-use grant.

    Returns True for exactly one caller: the DELETE only matches while the
    stored jti still equals this token's, so a replayed or concurrently
    redeemed token finds nothing to consume. Call after input validation and
    immediately before applying the privileged change."""
    if not jti:
        return False
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "DELETE FROM system_config WHERE key = %s AND value_text = %s;",
            (_grant_key(purpose, admin_id), jti),
        )
        return cur.rowcount > 0


def issue_session(email: str, admin_id: int) -> str:
    payload = {
        "sub": str(admin_id),
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time() + SESSION_TTL_HOURS * 3600),
    }
    return jwt.encode(payload, _session_secret(), algorithm="HS256")


def verify_session(token: str) -> dict | None:
    """Verify a session JWT AND that the account still exists and is active.

    Re-checking against admin_users on every request means a session that was
    not signed out explicitly still dies the moment the account is disabled or
    deleted - it cannot outlive its revocation window."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, _session_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    try:
        admin = find_admin_by_id(int(payload.get("sub", 0)))
    except Exception:
        return None
    if not admin or not admin["is_active"]:
        return None
    return {
        "admin_id": admin["id"],
        "email": admin["email"],
    }


def reauthorize(admin: dict | None, password: str, totp_code: str = "") -> str | None:
    """Re-verify current credentials before a privileged change.

    Used for changing a password/username or re-enrolling TOTP: the caller must
    re-prove possession of the account (and the enrolled Authenticator, when
    TOTP is on). Returns None when the re-auth succeeds, or a human-readable
    error message otherwise. TOTP failures share the in-process rate-limit
    bucket used by the other TOTP checks."""
    if not admin:
        return "account not found"
    if not verify_password(password or "", admin["password_hash"]):
        return "current password is incorrect"
    if admin["totp_enabled"]:
        key = f"reauth:{admin['id']}"
        if _rate_limited_totp(key):
            return "too many attempts - wait a few minutes"
        if not valid_totp(admin["totp_secret"], totp_code):
            _note_totp_failure(key)
            return "invalid Authenticator code"
    return None


# ---------------------------------------------------------------------------
# Admin user queries
# ---------------------------------------------------------------------------

def find_admin_by_email(email: str) -> dict | None:
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, email, password_hash, totp_secret, totp_enabled, is_active "
            "FROM admin_users WHERE email = %s;",
            (email.lower().strip(),)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": extract_val(row, "id", 0),
        "email": extract_val(row, "email", 1),
        "password_hash": extract_val(row, "password_hash", 2),
        "totp_secret": extract_val(row, "totp_secret", 3),
        "totp_enabled": bool(extract_val(row, "totp_enabled", 4)),
        "is_active": bool(extract_val(row, "is_active", 5)),
    }


def find_admin_by_id(admin_id: int) -> dict | None:
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, email, password_hash, totp_secret, totp_enabled, is_active "
            "FROM admin_users WHERE id = %s;",
            (admin_id,)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": extract_val(row, "id", 0),
        "email": extract_val(row, "email", 1),
        "password_hash": extract_val(row, "password_hash", 2),
        "totp_secret": extract_val(row, "totp_secret", 3),
        "totp_enabled": bool(extract_val(row, "totp_enabled", 4)),
        "is_active": bool(extract_val(row, "is_active", 5)),
    }


def update_email(admin_id: int, email: str):
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE admin_users SET email = %s WHERE id = %s;",
            (email.lower().strip(), admin_id)
        )


def admin_count() -> int:
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM admin_users;")
        row = cur.fetchone()
    return int(extract_val(row, "n", 0) or 0)


def create_admin(email: str, password: str, totp_secret: str | None = None) -> dict:
    email = email.lower().strip()
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO admin_users (email, password_hash, totp_secret, totp_enabled)
            VALUES (%s, %s, %s, %s)
            RETURNING id, email, totp_secret;
            """,
            (email, hash_password(password), totp_secret, bool(totp_secret))
        )
        row = cur.fetchone()
    return {
        "id": extract_val(row, "id", 0),
        "email": extract_val(row, "email", 1),
        "totp_secret": extract_val(row, "totp_secret", 2),
    }


def list_admins(search: str = "", status: str = "", limit: int | None = None,
                offset: int = 0) -> list[dict]:
    """List admins, optionally filtered by email search and status.
    Returns (rows, total) when `limit` is set (paged), else just rows."""
    where = []
    args: list = []
    if search:
        where.append("email ILIKE %s")
        args.append(f"%{search}%")
    if status == "active":
        where.append("is_active = true")
    elif status == "disabled":
        where.append("is_active = false")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    total: int | None = None
    with get_db_cursor(commit=False) as cur:
        if limit is not None:
            cur.execute(f"SELECT COUNT(*) AS n FROM admin_users{where_sql};", args)
            total = int(extract_val(cur.fetchone(), "n", 0) or 0)
        cur.execute(
            f"""
            SELECT id, email, totp_enabled, is_active, created_at, last_login_at
            FROM admin_users{where_sql} ORDER BY id ASC
            {("LIMIT %s OFFSET %s" if limit is not None else "")};
            """,
            args + ([limit, offset] if limit is not None else [])
        )
        rows = cur.fetchall()
    result = [
        {
            "id": extract_val(r, "id", 0),
            "email": extract_val(r, "email", 1),
            "totp_enabled": bool(extract_val(r, "totp_enabled", 2)),
            "is_active": bool(extract_val(r, "is_active", 3)),
            "created_at": extract_val(r, "created_at", 4),
            "last_login_at": extract_val(r, "last_login_at", 5),
        }
        for r in rows
    ]
    return (result, total) if limit is not None else result


def admin_stats() -> dict:
    """Counts over the whole admin_users table, for the Admins summary row."""
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE is_active) AS active,
                   COUNT(*) FILTER (WHERE is_active = false) AS disabled,
                   COUNT(*) FILTER (WHERE totp_enabled) AS otp
            FROM admin_users;
            """
        )
        r = cur.fetchone()
        return {
            "total": int(extract_val(r, "total", 0) or 0),
            "active": int(extract_val(r, "active", 1) or 0),
            "disabled": int(extract_val(r, "disabled", 2) or 0),
            "otp": int(extract_val(r, "otp", 3) or 0),
        }


def set_admin_active(admin_id: int, active: bool):
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE admin_users SET is_active = %s WHERE id = %s;",
            (active, admin_id)
        )


def delete_admin(admin_id: int) -> bool:
    """Permanently delete an admin account. Returns False if the id is unknown."""
    with get_db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM admin_users WHERE id = %s;", (admin_id,))
        return cur.rowcount > 0


def update_password(admin_id: int, password: str):
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE admin_users SET password_hash = %s, last_password_change_at = NOW() WHERE id = %s;",
            (hash_password(password), admin_id)
        )


def set_totp(admin_id: int, secret_b32: str, enabled: bool = True):
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE admin_users SET totp_secret = %s, totp_enabled = %s WHERE id = %s;",
            (secret_b32, enabled, admin_id)
        )


def touch_login(admin_id: int):
    with get_db_cursor(commit=True) as cur:
        cur.execute("UPDATE admin_users SET last_login_at = NOW() WHERE id = %s;", (admin_id,))


# ---------------------------------------------------------------------------
# Password-reset tokens (JWT, short-lived, single-purpose)
# ---------------------------------------------------------------------------

def issue_reset_token(email: str, admin_id: int) -> str:
    jti = secrets.token_urlsafe(32)
    payload = {
        "sub": str(admin_id),
        "email": email,
        "purpose": "password_reset",
        "jti": jti,
        "iat": int(time.time()),
        "exp": int(time.time() + FORGOT_TTL_MINUTES * 60),
    }
    token = jwt.encode(payload, _session_secret(), algorithm="HS256")
    _store_grant("password_reset", admin_id, jti)
    return token


def verify_reset_token(token: str) -> dict | None:
    if not token:
        return None
    try:
        payload = jwt.decode(token, _session_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if payload.get("purpose") != "password_reset":
        return None
    admin_id = int(payload.get("sub", 0))
    jti = payload.get("jti", "")
    if not _grant_active("password_reset", admin_id, jti):
        return None  # already redeemed, or superseded by a newer link
    return {"admin_id": admin_id, "email": payload.get("email", ""), "jti": jti}


def claim_reset_token(token: str) -> dict | None:
    """Verify and atomically consume a reset token (single-use)."""
    claim = verify_reset_token(token)
    if not claim or not claim_grant("password_reset", claim["admin_id"], claim["jti"]):
        return None
    return claim


# ---------------------------------------------------------------------------
# Invite tokens (JWT, long-lived, let a guest choose their own password)
# ---------------------------------------------------------------------------

def issue_invite_token(email: str, admin_id: int) -> str:
    jti = secrets.token_urlsafe(32)
    payload = {
        "sub": str(admin_id),
        "email": email,
        "purpose": "admin_invite",
        "jti": jti,
        "iat": int(time.time()),
        "exp": int(time.time() + INVITE_TTL_DAYS * 86400),
    }
    token = jwt.encode(payload, _session_secret(), algorithm="HS256")
    _store_grant("admin_invite", admin_id, jti)
    return token


def verify_invite_token(token: str) -> dict | None:
    if not token:
        return None
    try:
        payload = jwt.decode(token, _session_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if payload.get("purpose") != "admin_invite":
        return None
    admin_id = int(payload.get("sub", 0))
    jti = payload.get("jti", "")
    if not _grant_active("admin_invite", admin_id, jti):
        return None  # already redeemed, or superseded by a newer invite
    return {"admin_id": admin_id, "email": payload.get("email", ""), "jti": jti}


def claim_invite_token(token: str) -> dict | None:
    """Verify and atomically consume an invite token (single-use)."""
    claim = verify_invite_token(token)
    if not claim or not claim_grant("admin_invite", claim["admin_id"], claim["jti"]):
        return None
    return claim