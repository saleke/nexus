"""Versioned calibration: knobs, policy versioning, and approved changes.

Every threshold/model/policy mutation is an explainable, reversible row in
``policy_history``. The current policy version is derived deterministically
from the base version + number of applied changes, so it needs no extra
storage and never drifts from what the decision journal actually used.
"""
from __future__ import annotations

from .config import POLICY_VERSION, MODEL_VERSION
from .db import extract_val

# Knob validation: name -> (min, max, decimals). Only knobs listed here may be
# mutated through the control plane; anything else is rejected at apply time.
# decimals is BOTH the step and the persistence precision: system_config.value
# is NUMERIC(4,3), so a 4-decimal write would overflow. Keep it in lockstep.
KNOB_RANGES: dict[str, tuple[float, float, int]] = {
    "global_similarity_threshold": (0.500, 0.999, 3),
}

# Explicit warnings surfaced on ANY change to a performance/safety-affecting
# setting. The operator must see why the change matters and what it touches
# before confirming; the API returns these in the response body and the panel
# renders them as an inline warning card (not dismissible by policy).
KNOB_WARNINGS: dict[str, dict] = {
    "global_similarity_threshold": {
        "severity": "performance",
        "warning": (
            "Changing the assignment threshold alters how many posts get "
            "clustered into hubs: lower values cluster more posts but raise "
            "cross-topic merge risk; higher values are more precise but "
            "produce larger candidate pools and fewer hub births."
        ),
        "affected": [
            "post.assigned / post.candidate event volume",
            "hub formation density and precision-vs-recall balance",
            "adapter fine-tune data distribution",
            "downstream notification volume",
        ],
        "impact_estimator": True,
    },
}


def estimate_threshold_impact(cur, new_value: float) -> dict:
    """Approximate blast radius from the decision journal.

    Counts posts on the current threshold that would flip status under the
    proposed one. Margin conditions are ignored (the journal stores similarities
    but not the per-row runner), so this is an upper-bound estimate used for
    the confirmation screen, never a commit-time guarantee.
    """
    cur.execute(
        """
        SELECT COUNT(*) FILTER (WHERE status = 'assigned') AS assigned_now,
               COUNT(*) FILTER (WHERE status = 'assigned'
                                AND similarity IS NOT NULL AND similarity < %s) AS would_flip_down,
               COUNT(*) FILTER (WHERE status = 'candidate'
                                AND similarity IS NOT NULL AND similarity >= %s) AS would_flip_up
        FROM assignment_decision_log
        WHERE status IN ('assigned', 'candidate');
        """,
        (round(float(new_value), _persist_precision("global_similarity_threshold")),
         round(float(new_value), _persist_precision("global_similarity_threshold")))
    )
    r = cur.fetchone()
    return {
        "estimate": True,
        "known_decisions": int(extract_val(r, "assigned_now", 0) or 0),
        "would_flip_up": int(extract_val(r, "would_flip_up", 2) or 0),
        "would_flip_down": int(extract_val(r, "would_flip_down", 1) or 0),
        "margin_not_considered": True,
    }


def current_versions(cur) -> tuple[str, str]:
    """Resolve the policy and model versions actually in effect.

    policy_version: base POLICY_VERSION (policy-v1) unless a control-plane
    change was applied, in which case it becomes ``policy-v{1 + applied}``.
    model_version: the promoted model from the registry, else the config base.
    """
    cur.execute("SELECT COUNT(*) AS c FROM policy_history WHERE status = 'applied';")
    row = cur.fetchone()
    applied = int(extract_val(row, "c", 0) or 0)
    policy_version = POLICY_VERSION if applied == 0 else f"policy-v{applied + 1}"

    cur.execute("SELECT model_version FROM model_registry WHERE status = 'promoted' ORDER BY promoted_at DESC NULLS LAST, id DESC LIMIT 1;")
    row = cur.fetchone()
    model_version = extract_val(row, "model_version", 0) or MODEL_VERSION
    return policy_version, model_version


def validate_knob(knob: str, value: float) -> tuple[bool, str]:
    spec = KNOB_RANGES.get(knob)
    if spec is None:
        return False, f"knob '{knob}' is not calibratable"
    lo, hi, decimals = spec
    rounded = round(float(value), decimals)
    if not (lo <= rounded <= hi):
        return False, f"value must be within [{lo}, {hi}]"
    return True, ""


def _persist_precision(knob: str) -> int:
    """Persistence decimals must match the storage column (system_config.value
    is NUMERIC(4,3) => 3). Round once here, everywhere."""
    spec = KNOB_RANGES.get(knob)
    return spec[2] if spec else 3


def apply_policy_change(cur, knob: str, new_value: float, source: str, actor: str | None,
                        rationale: str | None = None) -> dict:
    """Apply a validated calibration change inside the caller's transaction.

    Must be invoked with an open, committing transaction (the caller owns
    commit/rollback). Records old->new in ``policy_history`` so the change is
    versioned, explainable, and reversible.
    """
    ok, err = validate_knob(knob, new_value)
    if not ok:
        raise ValueError(err)
    decimals = _persist_precision(knob)
    new_value = round(new_value, decimals)

    cur.execute("SELECT value FROM system_config WHERE key = %s FOR UPDATE;", (knob,))
    row = cur.fetchone()
    old_value = extract_val(row, "value", 0)
    old_value = float(old_value) if old_value is not None else None

    # Reject no-ops: re-applying the current value is not a change and must
    # not mint a new policy_history row (multi-clicks / resubmits / same-value
    # re-assertions all funnel here). A versioned history should only ever
    # contain real transitions.
    if old_value is not None and round(old_value, decimals) == new_value:
        raise ValueError(f"{knob} is already {new_value} — nothing to apply")

    cur.execute("SELECT COUNT(*) AS c FROM policy_history WHERE status = 'applied';")
    applied = int(extract_val(cur.fetchone(), "c", 0) or 0)
    new_version = POLICY_VERSION if applied == 0 else f"policy-v{applied + 1}"

    # Harden against a missing row: a blind UPDATE would silently affect 0 rows
    # while the history row still claims a change happened. Upsert instead.
    cur.execute(
        """
        INSERT INTO system_config (key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        (knob, new_value)
    )
    cur.execute(
        """
        INSERT INTO policy_history (policy_version, knob, old_value, new_value, status, source, rationale, actor)
        VALUES (%s, %s, %s, %s, 'applied', %s, %s, %s)
        RETURNING id, policy_version;
        """,
        (new_version, knob, old_value, new_value, source, rationale, actor)
    )
    hist_row = cur.fetchone()
    result = {
        "policy_history_id": extract_val(hist_row, "id", 0),
        "policy_version": extract_val(hist_row, "policy_version", 1),
        "knob": knob,
        "old_value": old_value,
        "new_value": new_value,
        "source": source,
        "actor": actor,
    }
    warning = KNOB_WARNINGS.get(knob)
    if warning:
        result["warning"] = warning
        if warning.get("impact_estimator"):
            result["impact_estimate"] = estimate_threshold_impact(cur, new_value)
    return result


def propose_policy_change(cur, knob: str, new_value: float, source: str, actor: str | None,
                          rationale: str | None) -> dict:
    """Insert a non-applied proposal (owner approves later)."""
    ok, err = validate_knob(knob, new_value)
    if not ok:
        raise ValueError(err)
    decimals = _persist_precision(knob)
    new_value = round(new_value, decimals)
    cur.execute("SELECT value FROM system_config WHERE key = %s;", (knob,))
    row = cur.fetchone()
    cur_val = extract_val(row, "value", 0)
    if cur_val is not None and round(float(cur_val), decimals) == new_value:
        raise ValueError(f"{knob} is already {new_value} — proposal would be a no-op")
    cur.execute(
        """
        INSERT INTO policy_history (policy_version, knob, new_value, status, source, rationale, actor)
        VALUES (%s, %s, %s, 'proposed', %s, %s, %s)
        RETURNING id, policy_version;
        """,
        ("pending", knob, new_value, source, rationale, actor)
    )
    hist_row = cur.fetchone()
    result = {
        "policy_history_id": extract_val(hist_row, "id", 0),
        "policy_version": extract_val(hist_row, "policy_version", 1),
        "knob": knob,
        "new_value": new_value,
        "source": source,
    }
    warning = KNOB_WARNINGS.get(knob)
    if warning:
        result["warning"] = warning
        if warning.get("impact_estimator"):
            result["impact_estimate"] = estimate_threshold_impact(cur, new_value)
    return result