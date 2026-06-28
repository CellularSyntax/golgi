# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Confirm-remove-geometry dialog (single boolean toggle)."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(*, do_remove_geometry: Callable) -> None:
    with v3.VDialog(
        v_model=("show_confirm_remove_geometry_dialog",),
        max_width=460,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Remove loaded geometry?",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "The current nerve surface, every "
                    "derived artefact (mesh, FEM result, "
                    "fiber trajectories, population) and "
                    "the source file on disk will be "
                    "deleted. Cuff, electrode and "
                    "conductivity settings are kept. "
                    "This cannot be undone.",
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
                        "show_confirm_remove_geometry_dialog"
                        " = false"
                    ),
                )
                html.Button(
                    "Remove",
                    type="button",
                    classes=(
                        "golgi-btn-primary golgi-btn-sm"
                    ),
                    click=(
                        do_remove_geometry,
                        "[]",
                    ),
                )
