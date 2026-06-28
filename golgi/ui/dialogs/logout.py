# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Sign-out confirmation dialog. Persistent (scrim click ignored)
so a stray click can't accidentally tear down the workspace."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    do_dismiss_logout_dialog: Callable,
    do_confirm_logout: Callable,
) -> None:
    """Build the logout-confirmation VDialog inside the current
    VAppLayout context."""
    with v3.VDialog(
        v_model=("show_logout_dialog",),
        max_width=440,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Sign out?",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Signing out will also close the open "
                    "project. A final save runs as part of the "
                    "close, so your work is preserved — you "
                    "can reopen the project after signing back "
                    "in.",
                    classes="golgi-dialog-body",
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=do_dismiss_logout_dialog,
                )
                html.Button(
                    "Sign out",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=do_confirm_logout,
                )
