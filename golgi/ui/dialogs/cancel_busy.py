# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Confirm-cancel dialog for any in-progress mesh/FEM/fiber subprocess."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    do_dismiss_cancel: Callable,
    do_confirm_cancel: Callable,
) -> None:
    with v3.VDialog(
        v_model=("show_cancel_dialog",),
        max_width=420,
        persistent=True,
        # `golgi-dialog-above-busy` class hooks the CSS rule
        # that pushes this dialog above .golgi-overlay (which
        # itself sits above all other dialogs). Without it the
        # confirm dialog would be hidden under the busy
        # lightbox the user is trying to cancel.
        class_="golgi-dialog-above-busy",
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Cancel operation?",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Any in-progress work will be discarded. "
                    "Cached results from previous steps stay "
                    "intact — you can resume the cancelled "
                    "step later.",
                    classes="golgi-dialog-body",
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Keep running",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=do_dismiss_cancel,
                )
                html.Button(
                    "Cancel operation",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=do_confirm_cancel,
                )
