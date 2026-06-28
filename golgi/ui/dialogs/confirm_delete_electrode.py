# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Confirm-delete-electrode dialog. Driven by confirm_delete_eid;
both buttons use inline-JS click expressions (no Python callbacks)."""
from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render() -> None:
    with v3.VDialog(
        v_model=("show_confirm_delete_dialog",),
        max_width=420,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Delete electrode?",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "“{{ confirm_delete_name }}” will be removed "
                    "from the design. This cannot be undone.",
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
                    click=(
                        "show_confirm_delete_dialog = false; "
                        "confirm_delete_eid = ''; "
                        "confirm_delete_name = ''"
                    ),
                )
                html.Button(
                    "Delete",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=(
                        "remove_design_request = "
                        "  confirm_delete_eid; "
                        "show_confirm_delete_dialog = false; "
                        "confirm_delete_eid = ''; "
                        "confirm_delete_name = ''"
                    ),
                )
