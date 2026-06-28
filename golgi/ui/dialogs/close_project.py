# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Close-project confirmation dialog."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    do_cancel_close: Callable,
    do_confirm_close: Callable,
) -> None:
    with v3.VDialog(
        v_model=("show_close_dialog",),
        max_width=420,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Close project?",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Your work is autosaved after every major "
                    "step, and a final save runs before "
                    "closing. You can reopen this project "
                    "anytime from the welcome screen.",
                    classes="golgi-dialog-body",
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Keep open",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=do_cancel_close,
                )
                html.Button(
                    "Close",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=do_confirm_close,
                )
