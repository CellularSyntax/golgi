# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Export Study progress dialog (F2.2).

Auto-opens when an export starts and stays up until the user
dismisses it. The Download anchor lives here so the navbar's
File → Export study entry produces visible UI even when the
project detail dialog isn't open.

The same `study_export_pending_*` state slot is shared with the
project-detail dialog's status strip — both surfaces stay in
sync (the strip is the persistent in-context indicator; this
dialog is the auto-popped notification)."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html, vuetify3 as v3


def render(*, do_close: Callable) -> None:
    with v3.VDialog(
        v_model=("show_export_study_dialog",),
        max_width=520,
        persistent=False,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Export study",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Packing every project file (configs · mesh "
                    "· FEM outputs · fibers · sims · sweep "
                    "results · audit excerpt) into a single "
                    "self-contained `.zip`.",
                    classes="golgi-dialog-body mb-3",
                )

                # Linear progress bar — striped + filled per the
                # overall percent the action handler computes from
                # the bundle's stage callbacks. Visible while the
                # export is busy.
                with html.Div(
                    v_show=("study_export_pending_busy",),
                    style="margin-bottom: 12px;",
                ):
                    v3.VProgressLinear(
                        model_value=(
                            "study_export_pending_progress",
                        ),
                        color="#e24b4a",
                        height=10,
                        striped=True,
                        classes="mb-2",
                    )
                    html.Div(
                        "{{ study_export_pending_status }}",
                        style=(
                            "font-size: 12px; color: #444;"
                        ),
                    )

                # Status line after completion.
                html.Div(
                    "{{ study_export_pending_status }}",
                    v_show=(
                        "!study_export_pending_busy "
                        "&& study_export_pending_status "
                        "&& !study_export_pending_error",
                    ),
                    style=(
                        "font-size: 12px; "
                        "color: #146e3a; "
                        "margin-bottom: 6px;"
                    ),
                )
                # Error line.
                html.Div(
                    "{{ study_export_pending_error }}",
                    v_show=("study_export_pending_error",),
                    style=(
                        "font-size: 12px; "
                        "color: #c0392b; "
                        "margin-bottom: 6px;"
                    ),
                )

            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Close",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    disabled=("study_export_pending_busy",),
                    click=do_close,
                )
                with html.A(
                    href=("study_export_pending_data_uri",),
                    download=("study_export_pending_filename",),
                    classes=(
                        "golgi-btn-primary golgi-btn-sm"
                    ),
                    style="margin-left: 8px;",
                    v_show=("study_export_pending_data_uri",),
                ):
                    html.I(
                        classes="mdi mdi-folder-zip-outline",
                        style="font-size: 16px;",
                    )
                    html.Span("Download study .zip")
