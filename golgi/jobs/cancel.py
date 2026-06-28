# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cooperative cancellation for jobs spawned by golgi.

CancelToken holds two pieces of state:
  * a `requested` flag flipped by the UI's Cancel button
  * an optional handle to the running subprocess (so the flag
    flip can immediately SIGTERM it instead of waiting for the
    runner's next poll)

JobRunner implementations:
  * call `token.arm(proc)` right after subprocess.Popen
  * call `token.was_requested()` between work units
  * call `token.clear()` when the job finishes / fails

UI:
  * `token.request()` from the Cancel button handler

Replaces the per-build_app `_cancellation = {"proc": None,
"requested": False}` dict + `_register_subprocess` /
`_clear_subprocess` / `_was_cancelled` helpers in step 4.1 of
migration.md. Not yet wired into golgi.py — that lands in 4.2.
"""
from __future__ import annotations


class CancelToken:
    """One token per logical job. Reuse across jobs by calling
    clear() between runs; or just create a fresh one per job."""

    def __init__(self) -> None:
        self._requested: bool = False
        self._proc = None  # subprocess.Popen | None

    def arm(self, proc) -> None:
        """Register the running subprocess so a subsequent
        `request()` can terminate it. Also resets the requested
        flag (a fresh job starts fresh)."""
        self._proc = proc
        self._requested = False

    def request(self) -> None:
        """User clicked Cancel. Set the flag + SIGTERM the
        registered process (if any), and schedule a hard-kill
        fallback on the event loop in case the child ignores
        SIGTERM. Idempotent."""
        import asyncio
        import subprocess
        self._requested = True
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
        except Exception:                                    # noqa: BLE001
            pass
        # 2 s hard-kill fallback — run in an executor so we don't
        # block the calling loop. `asyncio.get_event_loop()` is
        # safe here because `request()` is invoked from a click
        # handler running on the trame server loop.
        try:
            loop = asyncio.get_event_loop()

            def _hard_kill():
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:                        # noqa: BLE001
                        pass
                except Exception:                            # noqa: BLE001
                    pass

            loop.run_in_executor(None, _hard_kill)
        except Exception:                                    # noqa: BLE001
            pass

    def was_requested(self) -> bool:
        return self._requested

    def clear(self) -> None:
        """Reset to the post-init state. Call after the job has
        finished (success or failure) so the token is ready for
        the next job."""
        self._requested = False
        self._proc = None
