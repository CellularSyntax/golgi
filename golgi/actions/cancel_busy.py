# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cancel-busy-dialog action handlers — three handlers gated on the
shared `CancelToken` instance the build_app constructs once per
session.

W1.8a (step 1/5 of the do_* handler extraction) — originally took a
dict + did the SIGTERM + 2 s hard-kill inline. W1.8c refactor:
swapped to `golgi.jobs.cancel.CancelToken`; the token's `request()`
method now owns both the flag flip and the hard-kill fallback, so
`do_confirm_cancel` simplifies to one call + the state cleanup.

The async handler whose work is being cancelled detects this via
`token.was_requested()` (poll site lives in the pipeline modules)
and skips post-subprocess output loading.
"""
from __future__ import annotations

from typing import Callable

from golgi.jobs.cancel import CancelToken


def register(
    state,
    *,
    cancel_token: CancelToken,
) -> dict[str, Callable]:
    """Wire request / dismiss / confirm cancel handlers."""

    def do_confirm_cancel():
        # The CancelToken's request() sets the requested flag,
        # SIGTERMs the registered subprocess, and schedules the
        # 2 s hard-kill fallback. Everything OS-level lives there;
        # this handler just owns the busy-overlay teardown.
        cancel_token.request()
        # Batch every state mutation in one atomic push: closing
        # the confirm dialog + lowering the busy overlay together.
        # Without the batch, the confirm dialog can lag behind the
        # busy-overlay teardown and the user sees a stuck dialog
        # with no working buttons.
        with state:
            state.show_cancel_dialog = False
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.busy_cancel_no_confirm = False

    def do_request_cancel():
        if not state.busy:
            return
        # Single-click-cancel mode — handlers running in-process
        # restartable work (e.g. SAM2 video propagation) set
        # `state.busy_cancel_no_confirm = True` on entry so the
        # busy-lightbox Cancel button fires immediately. The
        # confirm sub-dialog can render UNDER an already-open
        # modal like segment-µCT, leaving Cancel effectively
        # un-clickable; and for in-process work there's no
        # subprocess to kill so the extra confirm step buys no
        # safety. Skip it.
        if bool(getattr(state, "busy_cancel_no_confirm", False)):
            do_confirm_cancel()
            return
        state.show_cancel_dialog = True

    def do_dismiss_cancel():
        state.show_cancel_dialog = False

    return {
        "do_request_cancel": do_request_cancel,
        "do_dismiss_cancel": do_dismiss_cancel,
        "do_confirm_cancel": do_confirm_cancel,
    }
