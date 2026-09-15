"""Read model for the owner-panel audit log page.

``admin_audit_log`` stores a before/after JSONB snapshot per action. Rendered
raw, those snapshots are an unreadable dump, so this module turns each row into
something an operator can scan:

* a plain-language label for the action and the entity it touched,
* a field level diff of the snapshot pair, listing only what actually changed,
* the counts and filter choices the page needs to paginate and narrow down.

Only reads live here. Writing audit rows stays in :mod:`audit`.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .db import extract_val

# Window presets offered by the filter bar: code, label, lookback hours
# (None means no lower bound).
AUDIT_WINDOWS = (
    ("24h", "last 24 hours", 24),
    ("7d", "last 7 days", 24 * 7),
    ("30d", "last 30 days", 24 * 30),
    ("all", "all time", None),
)
DEFAULT_WINDOW = "7d"

ADDED = "added"
REMOVED = "removed"
CHANGED = "changed"

DEFAULT_LIMIT = 50
MAX_LIMIT = 200

# Plain language for the action codes written by the control plane. Unknown
# codes fall back to the raw code so a new action is never invisible.
ACTION_LABELS = {
    "auth.totp.enroll": "Authenticator enrollment started",
    "auth.totp.confirm": "Authenticator enrolled",
    "auth.password.change": "Password changed",
    "auth.password.reset": "Password reset by recovery",
    "auth.email.change": "Username changed",
    "admin.invite": "Admin invited",
    "admin.invite.accept": "Invite accepted",
    "admin.active.set": "Admin access changed",
    "admin.delete": "Admin deleted",
    "consumer.create": "Consumer token created",
    "consumer.delete": "Consumer token revoked",
    "hub.merge": "Hubs merged",
    "hub.merge.reopen": "Merge reopened",
    "ops.resume_stuck": "Stuck posts resumed",
    "ops.retry_dlq": "Dead letter retried",
    "policy.apply": "Policy change applied",
    "policy.revert": "Policy change reverted",
}

DOMAIN_LABELS = {
    "auth": "Authentication",
    "admin": "Admins",
    "consumer": "Consumer tokens",
    "hub": "Event hubs",
    "ops": "Operations",
    "policy": "Policy",
}

ENTITY_LABELS = {
    "admin_users": "admin account",
    "api_consumers": "consumer token",
    "hub_merges": "merge record",
    "integration_outbox": "outbox event",
    "policy_history": "policy revision",
    "posts": "post",
    "system_config": "system config",
}


def window_cutoff(code: str) -> datetime | None:
    """Lower bound for a window preset, or None for all time."""
    for key, _label, hours in AUDIT_WINDOWS:
        if key == code:
            return None if hours is None else datetime.now(timezone.utc) - timedelta(hours=hours)
    return window_cutoff(DEFAULT_WINDOW)


def window_label(code: str) -> str:
    for key, label, _hours in AUDIT_WINDOWS:
        if key == code:
            return label
    return window_label(DEFAULT_WINDOW)


def action_domain(action: str) -> str:
    """``auth.password.change`` -> ``auth``."""
    return (action or "").split(".", 1)[0] or "other"


def action_label(action: str) -> str:
    return ACTION_LABELS.get(action or "", action or "unknown action")


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain or "other")


def entity_label(entity_type: str | None) -> str:
    if not entity_type:
        return "unscoped"
    return ENTITY_LABELS.get(entity_type, entity_type.replace("_", " "))


def as_mapping(value) -> dict:
    """Normalise a JSONB column to a dict.

    psycopg2 hands jsonb back as a dict, but text snapshots and NULLs show up
    in older rows and in tests, so tolerate those too.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray, str)):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _display(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def diff_states(before, after) -> list[dict]:
    """Field level diff between two snapshots.

    An audit row records a change, so unchanged fields are omitted: repeating
    identical values would bury the one line the operator came to read.
    """
    before_map, after_map = as_mapping(before), as_mapping(after)
    fields = list(before_map)
    fields += [name for name in after_map if name not in before_map]
    changes = []
    for field in fields:
        had_before, has_after = field in before_map, field in after_map
        old, new = before_map.get(field), after_map.get(field)
        if had_before and has_after:
            if old == new:
                continue
            kind = CHANGED
        elif has_after:
            kind = ADDED
        else:
            kind = REMOVED
        changes.append({
            "field": field,
            "before": _display(old) if had_before else None,
            "after": _display(new) if has_after else None,
            "kind": kind,
        })
    return changes


def change_tally(changes: list[dict]) -> dict:
    return {
        "added": sum(1 for c in changes if c["kind"] == ADDED),
        "removed": sum(1 for c in changes if c["kind"] == REMOVED),
        "changed": sum(1 for c in changes if c["kind"] == CHANGED),
        "total": len(changes),
    }


def _entry(row) -> dict:
    action = extract_val(row, "action", 2) or ""
    before = extract_val(row, "before_state", 5) or {}
    after = extract_val(row, "after_state", 6) or {}
    changes = diff_states(before, after)
    domain = action_domain(action)
    return {
        "id": extract_val(row, "id", 0),
        "actor": extract_val(row, "actor", 1) or "",
        "action": action,
        "action_label": action_label(action),
        "domain": domain,
        "domain_label": domain_label(domain),
        "entity_type": extract_val(row, "entity_type", 3),
        "entity_label": entity_label(extract_val(row, "entity_type", 3)),
        "entity_id": extract_val(row, "entity_id", 4),
        "created_at": extract_val(row, "created_at", 7),
        "changes": changes,
        "tally": change_tally(changes),
        "has_snapshot": bool(as_mapping(before) or as_mapping(after)),
        "before_json": json.dumps(as_mapping(before), ensure_ascii=False, indent=2, sort_keys=True),
        "after_json": json.dumps(as_mapping(after), ensure_ascii=False, indent=2, sort_keys=True),
    }


def _filters(actor, domain, entity, window, query):
    where, args = [], []
    cutoff = window_cutoff(window)
    if cutoff is not None:
        where.append("created_at >= %s")
        args.append(cutoff)
    if actor:
        where.append("actor = %s")
        args.append(actor)
    if domain:
        where.append("split_part(action, '.', 1) = %s")
        args.append(domain)
    if entity:
        where.append("entity_type = %s")
        args.append(entity)
    query = (query or "").strip()
    if query:
        like = f"%{query}%"
        where.append(
            "(action ILIKE %s OR actor ILIKE %s OR coalesce(entity_id, '') ILIKE %s"
            " OR before_state::text ILIKE %s OR after_state::text ILIKE %s)"
        )
        args.extend([like] * 5)
    return ("WHERE " + " AND ".join(where)) if where else "", args


def fetch_entries(cur, actor: str = "", domain: str = "", entity: str = "",
                  window: str = DEFAULT_WINDOW, query: str = "",
                  limit: int = DEFAULT_LIMIT, offset: int = 0) -> tuple[list[dict], int]:
    """Matching entries (newest first) plus the total match count."""
    where_sql, args = _filters(actor, domain, entity, window, query)
    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    offset = max(0, int(offset or 0))
    cur.execute(f"SELECT COUNT(*) AS n FROM admin_audit_log {where_sql};", args)
    total = int(extract_val(cur.fetchone(), "n", 0) or 0)
    cur.execute(
        f"""
        SELECT id, actor, action, entity_type, entity_id, before_state, after_state, created_at
        FROM admin_audit_log
        {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT %s OFFSET %s;
        """,
        args + [limit, offset],
    )
    return [_entry(row) for row in (cur.fetchall() or [])], total


def audit_stats(cur) -> dict:
    """Headline counters for the page's metric row."""
    cur.execute(
        "SELECT COUNT(*) AS total, COUNT(DISTINCT actor) AS actors,"
        " COUNT(DISTINCT entity_type) AS entities FROM admin_audit_log;"
    )
    row = cur.fetchone()
    cur.execute("SELECT COUNT(*) AS recent FROM admin_audit_log"
                " WHERE created_at >= NOW() - INTERVAL '24 hours';")
    return {
        "total": int(extract_val(row, "total", 0) or 0),
        "actors": int(extract_val(row, "actors", 1) or 0),
        "entities": int(extract_val(row, "entities", 2) or 0),
        "recent": int(extract_val(cur.fetchone(), "recent", 0) or 0),
    }


def filter_options(cur) -> dict:
    """Distinct values that populate the filter selects."""
    cur.execute("SELECT DISTINCT actor FROM admin_audit_log ORDER BY 1;")
    actors = [extract_val(r, "actor", 0) for r in (cur.fetchall() or [])]
    cur.execute("SELECT DISTINCT split_part(action, '.', 1) AS domain"
                " FROM admin_audit_log ORDER BY 1;")
    domains = [{"code": extract_val(r, "domain", 0),
                "label": domain_label(extract_val(r, "domain", 0))}
               for r in (cur.fetchall() or [])]
    cur.execute("SELECT DISTINCT entity_type FROM admin_audit_log"
                " WHERE entity_type IS NOT NULL ORDER BY 1;")
    entities = [{"code": extract_val(r, "entity_type", 0),
                 "label": entity_label(extract_val(r, "entity_type", 0))}
                for r in (cur.fetchall() or [])]
    return {"actors": actors, "domains": domains, "entities": entities}
