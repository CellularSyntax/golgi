# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Delete-project confirmation dialog."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    do_cancel_delete: Callable,
    do_confirm_delete: Callable,
) -> None:
    with v3.VDialog(
        v_model=("show_delete_dialog",),
        max_width=440,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Delete project?",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "This permanently removes the project "
                    "folder “{{ delete_project_name }}” and "
                    "all of its cached results "
                    "(mesh, FEM, fibers, thumbnail). This "
                    "cannot be undone.",
                    classes="golgi-dialog-body",
                )
                html.Div(
                    "{{ delete_error }}",
                    v_show=("delete_error",),
                    style=("color: #e24b4a; font-size: 12px; "
                            "margin-top: 10px;"),
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=do_cancel_delete,
                )
                html.Button(
                    "Delete",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=do_confirm_delete,
                )
