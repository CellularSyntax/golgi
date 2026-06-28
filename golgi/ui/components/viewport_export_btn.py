# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""3D viewport export button (F2.3.a follow-up).

Floating mdi-camera button anchored to a PyVista viewport. Clicks
trigger `do_export_viewport_screenshot(viewport_id)`; once the
handler populates `state.export_pending_*`, the popover's Download
anchor becomes active and the browser saves the PNG.

Smaller and simpler than the per-figure popover — viewport
screenshots are PNG only (PyVista's framebuffer is raster), so
there's no Format / Preset selector. Single "Capture" CTA + the
Download anchor that appears after capture.

Anchored bottom-right so it doesn't overlap the eye-icon FAB
(top-right) already on the viewport."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html, vuetify3 as v3


def render(
    viewport_id: str,
    *,
    do_export_viewport_screenshot: Callable,
    label: str = "Viewport screenshot",
    style: str = (
        "position: absolute; "
        "bottom: 16px; right: 16px; "
        "z-index: 6;"
    ),
) -> None:
    """Mount the camera button. Place inside any positioned
    container that already wraps the PyVista viewport — the
    workspace's `.golgi-viewport` div is the typical parent."""
    fig_id_expr = f"'viewport.{viewport_id}'"
    with html.Div(
        classes="golgi-viewport-export-btn",
        style=style,
    ):
        with v3.VBtn(
            icon=True,
            size="small",
            variant="tonal",
            density="compact",
            title=f"Save {label} as PNG",
        ):
            v3.VIcon("mdi-camera-outline", size="20")
            with v3.VMenu(
                activator="parent",
                close_on_content_click=False,
                location="bottom end",
                offset="8",
                min_width="260",
            ):
                with v3.VCard(classes="pa-3"):
                    html.Div(
                        f"Capture · {label}",
                        style=(
                            "font-size: 11px; "
                            "letter-spacing: 0.04em; "
                            "color: #888a90; "
                            "text-transform: uppercase; "
                            "margin-bottom: 8px;"
                        ),
                    )
                    html.Div(
                        "PNG screenshot at the current viewport "
                        "resolution + background. Resize the "
                        "browser window first if you need a "
                        "larger frame.",
                        style=(
                            "font-size: 11px; "
                            "color: #555; "
                            "margin-bottom: 10px; "
                            "line-height: 1.4;"
                        ),
                    )
                    html.Button(
                        "Capture",
                        type="button",
                        classes=(
                            "golgi-btn-primary "
                            "golgi-btn-sm "
                            "golgi-btn-block"
                        ),
                        style="margin-bottom: 8px;",
                        disabled=(
                            "export_pending_busy "
                            f"&& export_pending_fig_id === "
                            f"{fig_id_expr}",
                        ),
                        click=(
                            do_export_viewport_screenshot,
                            f"['{viewport_id}']",
                        ),
                    )
                    html.Div(
                        "Capturing…",
                        v_show=(
                            "export_pending_busy "
                            f"&& export_pending_fig_id === "
                            f"{fig_id_expr}",
                        ),
                        style=(
                            "font-size: 11px; "
                            "color: #555; "
                            "margin-bottom: 6px;"
                        ),
                    )
                    html.Div(
                        "{{ export_pending_error }}",
                        v_show=(
                            "export_pending_error "
                            f"&& export_pending_fig_id === "
                            f"{fig_id_expr}",
                        ),
                        style=(
                            "font-size: 11px; "
                            "color: #c0392b; "
                            "margin-bottom: 6px; "
                            "white-space: pre-wrap;"
                        ),
                    )
                    with html.A(
                        href=("export_pending_data_uri",),
                        download=("export_pending_filename",),
                        classes=(
                            "golgi-btn-secondary "
                            "golgi-btn-sm "
                            "golgi-btn-block"
                        ),
                        v_show=(
                            "export_pending_data_uri "
                            f"&& export_pending_fig_id === "
                            f"{fig_id_expr}",
                        ),
                    ):
                        html.I(
                            classes=(
                                "mdi mdi-file-download-outline"
                            ),
                            style="font-size: 16px;",
                        )
                        html.Span("Download PNG")
