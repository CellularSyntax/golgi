# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""New-project dialog (text field + Cancel / Create)."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    do_cancel_new_project: Callable,
    do_create_and_open_project: Callable,
) -> None:
    with v3.VDialog(
        v_model=("show_new_project_dialog",),
        max_width=440,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "New project",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Give the project a name. You can import "
                    "a nerve geometry from the Import drawer "
                    "once the workspace opens.",
                    classes="golgi-dialog-body mb-4",
                )
                v3.VTextField(
                    v_model=("new_project_name",),
                    label="project name",
                    autofocus=True,
                    density="compact",
                    hide_details=True,
                    classes="mb-2",
                    keydown_enter=do_create_and_open_project,
                )
                html.Div(
                    "{{ new_project_error }}",
                    v_show=("new_project_error",),
                    style=("color: #e24b4a; font-size: 12px; "
                            "margin-top: 6px;"),
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=do_cancel_new_project,
                )
                html.Button(
                    "Create",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=do_create_and_open_project,
                )
