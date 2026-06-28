# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Reusable per-panel figure-export button (F2.3.a).

`render(fig_id, do_export_single_figure, ...)` mounts a tiny
mdi-download icon button anchored inside whatever container it
sits in (typically a figure tile). On click the button activates a
Vuetify VMenu popover:

  ┌───────────────────────────────┐
  │  Export figure                │
  │  Format:   [PDF ▾]            │
  │  Preset:   [paper-300 ▾]      │
  │                  [Generate ▶] │
  │  (after generate)             │
  │                  [⬇ Download] │
  └───────────────────────────────┘

The Format + Preset bind to the GLOBAL `export_default_format` /
`export_default_preset` — set once and it sticks for every panel,
which is what the user wants when bulk-exporting figures for a paper.

The Generate button calls `do_export_single_figure(fig_id)`; the
handler writes `state.export_pending_*` once the render is done. The
Download button is a plain <a> with `download` attr, v_show'd only
when `export_pending_fig_id === this fig_id AND
export_pending_data_uri !== ''` — so opening the popover on panel A
while panel B's export is pending doesn't leak A's anchor to B.

Intentionally rendered with absolute positioning + a tiny footprint
so it sits in the figure tile's top-right corner without taking
layout space (clarifying answer #2 from the F2.3 setup)."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html, vuetify3 as v3


def render(
    fig_id: str,
    *,
    do_export_single_figure: Callable,
) -> None:
    """Mount the export button. Place INSIDE a positioned parent
    (typically the figure tile div) so the absolute positioning
    anchors to the right place. The parent should have
    `position: relative` — `golgi-fiber-tile` already does."""
    # Activator: small mdi-download icon button. tonal+small so it
    # blends with the Plotly toolbar without competing.
    with html.Div(
        classes="golgi-figure-export-btn",
        style=(
            "position: absolute; "
            "top: 6px; right: 6px; "
            "z-index: 5;"
        ),
    ):
        with v3.VBtn(
            icon=True,
            size="x-small",
            variant="tonal",
            density="compact",
            title=f"Export {fig_id} as PDF / SVG / PNG",
        ):
            v3.VIcon("mdi-download", size="18")
            # The popover.
            with v3.VMenu(
                activator="parent",
                close_on_content_click=False,
                location="bottom end",
                offset="8",
                min_width="280",
            ):
                with v3.VCard(classes="pa-3"):
                    html.Div(
                        f"Export figure · {fig_id}",
                        style=(
                            "font-size: 11px; "
                            "letter-spacing: 0.04em; "
                            "color: #888a90; "
                            "text-transform: uppercase; "
                            "margin-bottom: 8px;"
                        ),
                    )
                    # items lists live on state (see
                    # state_defaults/exports.py) because VSelect
                    # evaluates `items=` as a Vue expression — a
                    # Python literal here becomes a string in the
                    # template and the dropdown shows "No data".
                    v3.VSelect(
                        v_model=("export_default_format",),
                        items=("export_format_items",),
                        item_title="title",
                        item_value="value",
                        label="Format",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        classes="mb-2",
                    )
                    v3.VSelect(
                        v_model=("export_default_preset",),
                        items=("export_preset_items",),
                        item_title="title",
                        item_value="value",
                        label="Preset",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        classes="mb-3",
                    )
                    # Generate button — golgi-btn-primary for the
                    # active CTA. Shows a spinner while busy. The
                    # click handler passes fig_id as the first arg
                    # so one handler can route every panel.
                    html.Button(
                        "Generate",
                        type="button",
                        classes=(
                            "golgi-btn-primary "
                            "golgi-btn-sm "
                            "golgi-btn-block"
                        ),
                        style="margin-bottom: 8px;",
                        disabled=(
                            "export_pending_busy "
                            f"&& export_pending_fig_id === '{fig_id}'",
                        ),
                        click=(
                            do_export_single_figure,
                            f"['{fig_id}']",
                        ),
                    )
                    # Inline busy / error line.
                    html.Div(
                        "Rendering…",
                        v_show=(
                            "export_pending_busy "
                            f"&& export_pending_fig_id === '{fig_id}'",
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
                            f"&& export_pending_fig_id === '{fig_id}'",
                        ),
                        style=(
                            "font-size: 11px; "
                            "color: #c0392b; "
                            "margin-bottom: 6px; "
                            "white-space: pre-wrap;"
                        ),
                    )
                    # Download anchor — only shows when THIS figure's
                    # export is ready. Same styling family as the
                    # Sweep tab's CSV download anchors.
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
                            f"&& export_pending_fig_id === '{fig_id}'",
                        ),
                    ):
                        html.I(
                            classes="mdi mdi-file-download-outline",
                            style="font-size: 16px;",
                        )
                        html.Span("Download")
