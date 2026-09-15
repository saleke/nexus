"""Admin/control-plane audit trail.

Every calibration or intervention (threshold changes, model promotion, merge
reopens, consumer revocation) writes an immutable intent row so the owner can
answer "who changed what, and what was it before?".
"""
from __future__ import annotations

import json


def log_admin_action(cur, actor: str, action: str, entity_type: str | None = None,
                     entity_id: str | int | None = None,
                     before: dict | None = None, after: dict | None = None) -> None:
    cur.execute(
        """
        INSERT INTO admin_audit_log (actor, action, entity_type, entity_id, before_state, after_state)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb);
        """,
        (actor, action, entity_type, str(entity_id) if entity_id is not None else None,
         json.dumps(before or {}), json.dumps(after or {}))
    )