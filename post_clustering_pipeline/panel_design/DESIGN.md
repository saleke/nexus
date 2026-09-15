# DESIGN.md — Nexus Control Plane

The owner-facing control panel for the Nexus clustering system, plus the
service-layer rules that make it trustworthy. Served by FastAPI + Jinja2, driven
by HTMX (vendored at `static/htmx.min.js` — no CDN dependency).

## Why this design exists

Humans cannot hold dynamically-assigned BIGSERIAL ids for hubs and posts in
working memory. Every surface here is **content-anchored**: hubs and posts are
picked, shown, and confirmed by their **representative content**, and ids are
rendered only as subtext (`id 123`, `.id-hint`). Nobody on this panel ever types
a bare id to act — they type *something they remember* (a phrase, a user, an id)
into search, and pick from labeled cards.

The panel is **control plane, not data plane**. The clustering pipeline
(assignment, encoding, birth, aggregate, dispatch) keeps running regardless of
panel state. The panel only ever:
  * reads lifecycle surfaces (quality, decisions, ops, audit),
  * mutates the three things an operator can safely touch — threshold policy,
    hub membership corrections (merge/unlink/confirm), and access credentials —
    every mutation versioned, warned, estimated, audited, and reversible.

## Non-negotiable rules (the service layer)

1. **One merge implementation.** Client API, panel desk, and the autodetect job
   all go through `merges.apply_merge_soft` (canonical redirect resolution →
   `pg_advisory_xact_lock` on the ordered pair → member snapshot → centroid
   rebalance → `hub_merges` row → outbox event). `jobs/merge_hubs.py` only
   *detects* pairs; it never folds anything itself.
2. **Corrections are scoped and canonical.** `corrections.unlink_post` refuses
   (409) if the post already left the stated hub (a concurrent merge moved it).
   Confirm/unlink resolve merges first — a stale hub id never receives a write.
3. **Human decisions are never counted as system behavior.** Rollups separate
   `source = human` vs `system`. Panel merges tag `initiated_by=panel` →
   feedback `owner_merge`; only the autodetect job writes `system_auto_merge`.
   The autotuner requires ≥300 human-graded decisions and proposes (never
   auto-applies) by default.
4. **Warnings travel with the change.** Any performance/safety-affecting setting
   carries a `KNOB_WARNINGS` entry; apply/propose return it, the panel renders it
   as a non-dismissible warning card, and a blast-radius estimate
   (`estimate_threshold_impact`) is shown before commit. Restart-only knobs are
   listed read-only with a badge — an editable panel value that decays on restart
   would be a lie.
5. **Every mutation is reversible + audited.** Threshold changes go to
   `policy_history` (status applied/proposed) with revert; merges snapshot
   post-ids + source centroid into `hub_merges` for deterministic reopen (only
   snapshot members still on the target rebound); panel actions land in
   `admin_audit_log`. Secrets never render on /admin/settings (redacted).
6. **Panel HTML stays a thin fragment layer.** Data-plane logic lives in
   service modules (`merges`, `corrections`, `policy`, `quality`, `decisions`);
   templates render. Tests target the services; templates are structure-light.

## Identity layer (`refs.py`)

| Handle      | Rendered as                                                       | id shown as |
|-------------|-------------------------------------------------------------------|-------------|
| Hub         | anchor content preview (or latest member) + `discourse_type · N members` + up to 2 member-sample previews | `id <n>` subtext |
| Post        | content preview + author + status (+ current hub)                 | `id <n>` subtext |
| Search      | `/admin/refs?q=` exact-id match **or** ILIKE over content/anchor  | —           |

Pickers write hidden inputs (`{source_event_id,target_event_id}` / `post_id+event_id`)
into the desk forms; outbox events and API responses embed the same reference
blocks so client apps can label changes without a follow-up fetch.

## Design tokens (mirror of `static/panel.css` — keep both in sync)

```
bg         #0d1117   surface   #161b22   surface-2  #1c2128      border    #30363d
text       #e6edf3   muted     #8b949e   accent     #58a6ff
ok         #3fb950   warn      #d29922   danger     #f85149
radius     8px / 5px (sm)      spacing   4px grid (8 / 16 / 32)
type       14px/1.5 sans; 11px uppercase table heads; mono for ids/code
```

Badge semantics: `ok`=good/active/fresh, `warn`=intervention-eligible/at-risk/stale,
`danger`=destructive/needs reconsideration, `muted`=inert/restart-only.

## Pages & behaviors (interactions are HTMX fragments)

| Page  | Reads                                            | Writes (warned) |
|-------|--------------------------------------------------|-----------------|
| Quality | rollups by version+source, threshold, DLQ/stuck, policy history | — |
| Decisions | decision journal (+references+reason badges), filter by post/hub | — |
| Desk | hub/post search cards (content anchors) | merge (into target), unlink post |
| Merge log | `hub_merges` with source/target labels | reopen merge |
| Calibration | current threshold, version chips, history | apply/propose/revert threshold |
| Ops | redis, queue, DLQ, leases, stuck, rollup freshness, consumers | resume stuck, retry DLQ, consumer CRUD |
| Audit | `admin_audit_log` | — |
| Settings | system_config + env knobs (secrets redacted, restart-only badges) | — |

The operator flow for every write is: **search → recognize content → select →
confirm → read the announcement card (which mentions *what* changed in content
terms + the warning/estimate) → revert/undo available downstream**.

## Edge cases handled (and where)

- Concurrent merge races → advisory lock (merges.py) + post-move 409 (corrections).
- Reopen after members were unlinked → only `assigned` snapshot posts still on
  the target rebound; source count = actual rebound (merges.py).
- Stale hub id in a client call → canonical resolve + `was_redirected` in response.
- Threshold set above/below range or decimals beyond storage → validation + round
  to `NUMERIC(4,3)` precision (policy.py).
- Self-grading → decision journal stores the versions that produced each status,
  rollups separate sources, autotune uses only human `user_confirmed/removed`.
- Lost/poisoned deliveries → lease/ACK/DLQ with stuck-`processing` sweep shown in Ops.
- Panel misconfig → panel is behind admin ingress (bearer middleware today);
  static assets exempted; upgrades documented separately.