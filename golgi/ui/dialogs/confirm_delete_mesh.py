# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Confirm-delete-mesh dialog. Mirrors confirm_delete_electrode:
driven by `confirm_delete_mesh_eid` + `confirm_delete_mesh_name`;
both buttons use inline-JS click expressions (no Python callbacks),
and confirm sets `remove_mesh_request` to the target eid which an
app.py watcher picks up and routes to do_delete_mesh.
"""
from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render() -> None:
    with v3.VDialog(
        v_model=("show_confirm_delete_mesh_dialog",),
        max_width=440,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Delete mesh?",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "The built nerve.msh + cached PLC/TetGen "
                    "files for "
                    "“{{ confirm_delete_mesh_name }}” will be "
                    "removed. Any cached FEM solves for this "
                    "design are deleted too. This cannot be "
                    "undone.",
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
                        "show_confirm_delete_mesh_dialog = false; "
                        "confirm_delete_mesh_eid = ''; "
                        "confirm_delete_mesh_name = ''"
                    ),
                )
                html.Button(
                    "Delete",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=(
                        "remove_mesh_request = "
                        "  confirm_delete_mesh_eid; "
                        "show_confirm_delete_mesh_dialog = false; "
                        "confirm_delete_mesh_eid = ''; "
                        "confirm_delete_mesh_name = ''"
                    ),
                )
