# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""InProcessRunner — JobRunner that invokes a callable directly.

Used for compute that's not (yet) a subprocess. Today: the
single-fiber + population fiber-sim paths in golgi.py call
pyfibers/axonml directly inside the action coroutine via
loop.run_in_executor. Wrapping them in this runner lets the
pipeline driver use the same JobRunner.run() shape as the
subprocess-based pipelines, so swapping to a remote runner
later only changes the runner instantiation.

Cancel is cooperative: the wrapped callable receives the cancel
token and is expected to check `cancel.was_requested()` at safe
points (between fibers, between time steps, etc.). The runner
itself can't SIGKILL an in-process call.

Step 4.1: skeleton only. Wiring lands in 4.6 / 4.7.
"""
from __future__ import annotations

from typing import Callable

from .cancel import CancelToken
from .protocol import JobOutputs, JobRequest


class InProcessRunner:
    """Wraps a callable `fn(req, on_line, cancel) -> JobOutputs`."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def run(
        self,
        req: JobRequest,
        on_line: Callable[[str], None],
        cancel: CancelToken,
    ) -> JobOutputs:
        try:
            result = self.fn(req, on_line, cancel)
        except Exception as ex:
            on_line(f"[job] in-process call raised: "
                    f"{type(ex).__name__}: {ex}")
            return JobOutputs(return_code=1)
        if isinstance(result, JobOutputs):
            return result
        # Caller returned something else; wrap as success with
        # the raw value under "result". Pipeline drivers can
        # decide how to interpret it.
        return JobOutputs(return_code=0, outputs={"result": result})
