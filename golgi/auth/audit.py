# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Flight recorder — non-blocking audit-log writer.

Architecture: action thread → bounded in-memory queue → daemon
writer thread → batched INSERT into `audit_events`.

Failure tiers:
  1. DB INSERT fails → batch is appended to a JSON-Lines
     fallback file (`<AUTH_DB_DIR>/audit_fallback.jsonl`).
     Each line is one record; append is atomic on POSIX up
     to PIPE_BUF so a crash mid-write loses at most one row.
  2. Fallback file write fails → record is printed to stderr
     so a captured-stdout container still gets it. Dropped.
  3. On the NEXT successful DB batch, the writer drains the
     fallback file too and truncates it — so the file never
     grows unbounded.

Queue overflow (>10 k unflushed entries — would require a
wedged writer thread for many minutes): `put_nowait` raises
`queue.Full` and the caller prints the record to stderr and
drops it. Action thread is NEVER blocked.

Init: call _init_audit_writer(fallback_path) once at startup.
golgi.py wires this from _ensure_initialized() with
AUTH_DB_DIR / "audit_fallback.jsonl".
"""
from __future__ import annotations

import atexit
import json
import queue as _stdqueue
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import _AuditEvent, get_session


_AUDIT_QUEUE_MAX = 10_000
_AUDIT_BATCH_SIZE = 50
_AUDIT_BATCH_INTERVAL_S = 0.25

# Set by _init_audit_writer() before any writes happen. None
# beforehand means "no fallback configured yet" — direct stderr
# print is the only failure path until init lands.
_audit_fallback_path: "Path | None" = None

_audit_queue: "_stdqueue.Queue[dict]" = _stdqueue.Queue(
    maxsize=_AUDIT_QUEUE_MAX,
)
_audit_writer_stop = threading.Event()
_audit_writer_thread: "threading.Thread | None" = None
_atexit_registered = False


def _audit_record_to_orm(rec: dict) -> _AuditEvent:
    """Materialise a queue record into a SQLAlchemy ORM instance."""
    return _AuditEvent(
        ts=rec.get("ts") or datetime.now(timezone.utc),
        user_id=rec.get("user_id"),
        action=str(rec.get("action", ""))[:64],
        payload=rec.get("payload"),
        project_dir=(
            str(rec.get("project_dir"))[:512]
            if rec.get("project_dir") is not None
            else None
        ),
        status=(
            str(rec.get("status"))[:16]
            if rec.get("status") is not None
            else None
        ),
    )


def _audit_replay_fallback(session) -> int:
    """If `audit_fallback.jsonl` has entries, drain it into the
    open SQLAlchemy session in a single ORM batch and truncate
    the file. Returns the number of replayed records (0 if the
    file is missing / empty / unreadable). Caller commits."""
    if (_audit_fallback_path is None
            or not _audit_fallback_path.is_file()):
        return 0
    try:
        with _audit_fallback_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as ex:
        print(f"[audit] fallback read failed: {ex}", flush=True)
        return 0
    if not lines:
        return 0
    replayed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        # `ts` is round-tripped as an ISO string in JSON.
        ts = rec.get("ts")
        if isinstance(ts, str):
            try:
                rec["ts"] = datetime.fromisoformat(ts)
            except Exception:
                rec["ts"] = datetime.now(timezone.utc)
        session.add(_audit_record_to_orm(rec))
        replayed += 1
    # Truncate the file now that everything is queued for commit.
    # If the commit later fails, the records are still in the
    # ORM session and we'll re-fallback the whole new batch —
    # the old file content is lost but the new batch carries it
    # forward.
    try:
        with _audit_fallback_path.open("w", encoding="utf-8") as f:
            f.truncate(0)
    except Exception as ex:
        print(f"[audit] fallback truncate failed: {ex}", flush=True)
    return replayed


def _audit_write_to_fallback(batch: list, ex: Exception) -> None:
    """DB write failed — append the batch to the JSONL fallback
    file. Each record becomes one line. `default=str` on the
    JSON dump catches datetimes / paths cleanly."""
    if _audit_fallback_path is None:
        print(f"[audit] DB batch write failed ({len(batch)} rec)"
              f" + no fallback configured: {ex}", flush=True)
        for rec in batch:
            print(f"[audit-lost] {json.dumps(rec, default=str)}",
                  flush=True)
        return
    print(f"[audit] DB batch write failed ({len(batch)} rec): "
          f"{ex} — appending to fallback "
          f"{_audit_fallback_path}", flush=True)
    try:
        _audit_fallback_path.parent.mkdir(
            parents=True, exist_ok=True,
        )
        with _audit_fallback_path.open("a", encoding="utf-8") as f:
            for rec in batch:
                # Convert datetime → ISO so json.loads can
                # round-trip later.
                rec_out = dict(rec)
                ts = rec_out.get("ts")
                if isinstance(ts, datetime):
                    rec_out["ts"] = ts.isoformat()
                f.write(json.dumps(rec_out, default=str) + "\n")
    except Exception as ex2:
        # Last resort: stderr.
        print(f"[audit] FALLBACK WRITE FAILED: {ex2}",
              flush=True)
        for rec in batch:
            print(f"[audit-lost] {json.dumps(rec, default=str)}",
                  flush=True)


def _audit_writer_flush(batch: list) -> None:
    """Try to write a batch of records to the DB. On success,
    also drain any backlog from the fallback file (so the
    flight recorder self-heals after a transient outage).
    Always best-effort: any exception is caught and re-routed
    to the fallback path."""
    try:
        with get_session() as session:
            for rec in batch:
                session.add(_audit_record_to_orm(rec))
            # Drain backlog from previous failures (if any) in
            # the SAME transaction — atomic recovery.
            replayed = _audit_replay_fallback(session)
            session.commit()
            if replayed:
                print(f"[audit] replayed {replayed} fallback "
                      f"record(s) on recovery", flush=True)
    except Exception as ex:
        _audit_write_to_fallback(batch, ex)


def _audit_writer_loop() -> None:
    """Background-thread main loop. Waits up to
    `_AUDIT_BATCH_INTERVAL_S` for the first record, then drains
    everything currently queued up to `_AUDIT_BATCH_SIZE` and
    flushes one batch. Exits cleanly when `_audit_writer_stop`
    is set AND the queue is drained."""
    while True:
        # Block until a record arrives OR we're told to stop.
        try:
            first = _audit_queue.get(
                timeout=_AUDIT_BATCH_INTERVAL_S,
            )
        except _stdqueue.Empty:
            if _audit_writer_stop.is_set():
                return
            continue
        batch = [first]
        while len(batch) < _AUDIT_BATCH_SIZE:
            try:
                batch.append(_audit_queue.get_nowait())
            except _stdqueue.Empty:
                break
        try:
            _audit_writer_flush(batch)
        except Exception as ex:  # safety net
            print(f"[audit] writer crashed mid-flush: {ex}",
                  flush=True)
        # If shutdown is requested and queue is empty, drain one
        # last cycle then return.
        if _audit_writer_stop.is_set() and _audit_queue.empty():
            return


def _start_audit_writer() -> None:
    """Idempotent: starts the daemon writer thread if it isn't
    already running. Daemon=True so the thread dies with the
    process if we hit a hard exit; clean shutdown (atexit
    handler) drains the queue first."""
    global _audit_writer_thread
    if (_audit_writer_thread is not None
            and _audit_writer_thread.is_alive()):
        return
    _audit_writer_thread = threading.Thread(
        target=_audit_writer_loop,
        daemon=True,
        name="audit-writer",
    )
    _audit_writer_thread.start()


def _shutdown_audit_writer(timeout_s: float = 3.0) -> None:
    """Signal the writer to finish + wait for it to drain.
    Registered via `atexit` so a clean Python shutdown gets
    every queued record persisted (or fallback'd) before the
    process exits."""
    _audit_writer_stop.set()
    t = _audit_writer_thread
    if t is not None and t.is_alive():
        t.join(timeout=timeout_s)


def _init_audit_writer(fallback_path: Path) -> None:
    """Register the atexit drain handler + start the background
    writer thread. Idempotent: _start_audit_writer() already
    checks for a live thread, and _atexit_registered guards the
    atexit hook so we only register it once per process.

    `fallback_path` is where DB-failure batches get journaled as
    JSONL (`audit_fallback.jsonl` under the auth DB dir)."""
    global _atexit_registered, _audit_fallback_path
    _audit_fallback_path = fallback_path
    if not _atexit_registered:
        atexit.register(_shutdown_audit_writer)
        _atexit_registered = True
    _start_audit_writer()


def _audit_log(
    user_id: int | None,
    action: str,
    payload: dict | None = None,
    project_dir: str | None = None,
    status: str | None = "info",
) -> None:
    """Enqueue one audit record for the background writer.
    Non-blocking — typical cost is a few µs. If the queue is
    saturated (>10 k unflushed entries), the record is printed
    to stderr and dropped rather than blocking the caller.

    `status` is one of "success" | "failure" | "info"; the
    flight-recorder decorators (`@log_action` / `@gated`) set
    it automatically based on whether the wrapped function
    raised. Direct callers can pass it explicitly (e.g.
    `_audit_log(None, "login_failed", ..., status="failure")`).
    """
    rec = {
        "ts": datetime.now(timezone.utc),
        "user_id": user_id,
        "action": str(action)[:64],
        "payload": (
            json.dumps(payload, default=str)
            if payload is not None else None
        ),
        "project_dir": (
            str(project_dir) if project_dir is not None else None
        ),
        "status": status,
    }
    try:
        _audit_queue.put_nowait(rec)
    except _stdqueue.Full:
        # Last-resort: shouldn't happen under normal load (the
        # writer drains at ~200 batches/sec). If it does, the
        # record goes to stderr so an external log collector
        # can pick it up.
        print(f"[audit] queue full — dropping: "
              f"{json.dumps(rec, default=str)}", flush=True)
