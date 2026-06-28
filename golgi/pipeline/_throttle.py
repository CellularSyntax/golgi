# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Flush throttle — coalesce rapid state pushes during noisy
subprocess output.

Without this, each TetGen / FENiCSx / fiber-paths stdout line
schedules its own `state.flush()`. On a heavy mesh build that
prints 50+ lines/sec, the WS layer ends up with 50+ broadcast
tasks per second. Heavy WS write pressure can lead the browser-
side wslink client to consider the connection unhealthy and
trigger an `onclose` (visible as "Connection closed" overlay
+ a TypeError in the cleanup chain — observed during 70M-tet
TetGen builds with the tab partially backgrounded).

Pattern: trailing-edge debounce on `state.flush()` only — the
underlying state mutation still happens on every line, so the
next flush picks up the latest values. The user sees the live
tail update at ~`min_interval` cadence instead of per-line.

Default `min_interval=0.5` is conservative — chosen after a
150ms default still produced disconnects during 70M-tet
TetGen runs. 500ms means ~2 WS pushes/sec during the noisiest
phases, which is still snappy for a "live log tail" UI but a
lot easier on the browser's message queue. Bump down toward
0.15 if log tail lag becomes noticeable on less noisy ops.

Used by the _on_line callbacks in pipeline/mesh.py, fem.py,
fibers.py."""
from __future__ import annotations

import asyncio
import time
from typing import Any


class FlushThrottle:
    """Trailing-edge debounced wrapper around `state.flush()`.

    Call `tick()` from the main asyncio thread (i.e. from inside
    a function dispatched via `loop.call_soon_threadsafe(...)`).
    Each tick either flushes immediately (if enough time has
    elapsed since the last flush) or arms a single deferred
    flush via `loop.call_later`. The trailing-edge flush picks
    up whatever state mutations the caller made between ticks,
    so the visible client state is never older than
    `min_interval` seconds."""

    __slots__ = (
        "loop", "state", "min_interval",
        "_last_flush", "_pending",
    )

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        state: Any,
        min_interval: float = 0.5,
    ) -> None:
        self.loop = loop
        self.state = state
        self.min_interval = float(min_interval)
        self._last_flush: float = 0.0
        self._pending: bool = False

    def tick(self) -> None:
        """Request a flush. Always run from the asyncio thread."""
        now = time.monotonic()
        elapsed = now - self._last_flush
        if elapsed >= self.min_interval:
            self.state.flush()
            self._last_flush = now
            self._pending = False
        elif not self._pending:
            self._pending = True
            delay = max(0.0, self.min_interval - elapsed)
            self.loop.call_later(delay, self._fire_pending)

    def _fire_pending(self) -> None:
        self.state.flush()
        self._last_flush = time.monotonic()
        self._pending = False

    def force_flush(self) -> None:
        """Bypass the throttle — flush immediately. Use at end-
        of-job to make sure the final state lands without waiting
        for the next tick."""
        self.state.flush()
        self._last_flush = time.monotonic()
        self._pending = False
