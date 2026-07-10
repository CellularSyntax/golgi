# Authentication & Audit

golgi is multi-user and keeps a tamper-evident **audit trail** of every meaningful action. Auth and
audit are **per-user-local** (a single workstation/server), not a multi-tenant cloud service.

Source: `golgi/auth/` — `models.py` (DB), `session.py` (session), `decorators.py` (gating + logging),
`audit.py` (the audit writer).

---

## Users & sessions

Users live in a local **SQLite** database (`auth.db`), with email, optional username, hashed password,
profile fields (name, country, institution, position), and an avatar. The GUI's **Sign in** dialog
authenticates against this DB; the session is held process-side. Projects carry an owner and a
**shared-with** list, so the welcome screen shows you only your own and shared projects.

In **headless** mode (the [Python API](Python-API) / CLI) golgi synthesises a local `headless` user so
the auth-gated pipeline drivers run without interactive login — the audit writer still records the run.

## Gating & logging decorators

Pipeline actions are wrapped with two decorators:

- **`@log_action("name")`** — emits an audit row (with truncated arg capture) recording success or
  failure.
- **`@gated("name")`** — refuses to run when no user is logged in, surfaces the login prompt, and
  audits the blocked attempt.

Audited actions include `load_geometry`, `build_mesh`, `solve_fem`, `solve_fiber_paths`, `sweep`,
`export_study`, `import_study`, `segment_uct`, `export_figure`, `bulk_export`, and `generate_report`.

## The audit log

`audit.py` is a **flight recorder**: actions push events onto a bounded in-memory queue that a daemon
thread batches into the database. It's robust by design — if the DB write fails, events fall back to an
append-only `audit_fallback.jsonl`, and that file is drained back into the DB on the next successful
write. Each event records timestamp, user, action, a JSON payload, the project directory, and a status
(`success` / `failure` / `info` / `blocked`).

You can review the trail in the **Activity** tab of the Project Details dialog (GUI). When a study is
exported, a **project-scoped audit excerpt** travels in the [bundle](Reproducible-Study-Bundles); on
import, those events are replayed under the importing user, tagged with the original provenance — so a
shared study carries its history with it.

---

### See also
[Reproducible Study Bundles](Reproducible-Study-Bundles) · [GUI Walkthrough](GUI-Walkthrough) ·
[Headless / HPC](Headless-and-HPC) · [Architecture](Architecture)
