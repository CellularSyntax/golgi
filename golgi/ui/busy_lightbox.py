# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Busy lightbox — full-viewport modal overlay shown while a
long-running operation is in flight (mesh build, FEM solve,
fiber sim, population sweep, …). Shows the wave loader, the
current `busy_msg`, a live tail of `busy_log`, and a Cancel
button that opens the cancel-confirmation sub-dialog.

Mounted at the VAppLayout level (sibling to VMain), NOT inside
VMain's flex column — that column has its own scroll context
and child flex constraints that can clip a `position: fixed`
dimmer in the Single fiber / Population analysis tabs. Call
this directly inside the `with VAppLayout(...)` block."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html


def render(*, do_request_cancel: Callable) -> None:
    with html.Div(
        v_show=("busy",),
        classes="golgi-overlay",
    ):
        with html.Div(classes="golgi-lightbox"):
            # Wave loader. Black-on-white wave → SVG filter
            # `#loader-recolor` recolours to magma red. Variant
            # class (v1..v5) picks the animation;
            # loader_variants.js cycles it every 10 s.
            html.Div(classes="loader v5")
            html.Div(
                "{{ busy_msg }}…",
                classes="golgi-loader-text",
            )
            # Live tail of the long-running subprocess log
            # (last N lines). Hidden when empty so short ops
            # (load / fit / build) don't show a blank box.
            html.Pre(
                "{{ busy_log }}",
                v_show=("busy_log",),
                classes="golgi-lightbox-log",
            )
            # Cancel button — opens a confirmation sub-dialog
            # that, on confirm, terminates any active
            # subprocess and lowers busy.
            with html.Div(
                classes="golgi-lightbox-actions",
            ):
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary "
                        "golgi-btn-sm"
                    ),
                    click=do_request_cancel,
                )
