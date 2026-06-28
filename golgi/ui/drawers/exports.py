# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Bulk Exports drawer (F2.3.b).

Right-side drawer mirroring the look of Mesh / Fibers / Conductivities:
  - Format + Preset dropdowns at the top (shared with the per-panel
    popover via state.export_default_*).
  - Two action buttons: "Select all available" / "Clear".
  - Categorised checkbox list iterated from
    `state.exports_registry_meta` (populated at build_app startup).
  - "Export N figures → ZIP" CTA at the bottom.
  - Progress lines + Download anchor that appear after the export
    completes — separate state slot
    (`state.bulk_export_pending_*`) from the per-figure popover so
    the two flows don't collide.

For v1 the checkbox list shows EVERY registered figure regardless
of current availability — the "Select all available" button does a
server-side filter to populate only the figures whose source data
exists right now. Unavailable figures the user checks manually
fail per-figure in the bulk run (logged into the progress feed)
but don't break the rest.
"""
from __future__ import annotations

from typing import Callable

from trame.widgets import html, vuetify3 as v3


def render(
    *,
    do_bulk_export: Callable,
    do_bulk_export_select_all_available: Callable,
    do_bulk_export_clear: Callable,
) -> None:
    with v3.VNavigationDrawer(
        v_model=("show_exports",),
        location="right", width=480,
        elevation=8,
    ):
        with v3.VContainer(classes="pa-4"):
            # Title row.
            with html.Div(
                classes=(
                    "d-flex align-center "
                    "justify-space-between mb-3"
                ),
            ):
                html.H3(
                    "Bulk export",
                    classes="golgi-drawer-title",
                )
                v3.VBtn(
                    icon="mdi-close",
                    variant="text",
                    density="compact",
                    size="small",
                    click="show_exports = false",
                )
            html.Div(
                "Pick figures, pick a preset, click Export. "
                "All selected figures are bundled into a ZIP "
                "with a MANIFEST.json — same render pipeline "
                "as the per-panel popover.",
                style=(
                    "font-size: 11px; color: #555; "
                    "line-height: 1.4; margin-bottom: 12px;"
                ),
            )

            # ---- Format + preset ----
            html.Div(
                "FORMAT",
                style=(
                    "font-size: 10px; letter-spacing: 0.04em; "
                    "color: #888a90; text-transform: uppercase; "
                    "margin-bottom: 4px;"
                ),
            )
            v3.VSelect(
                v_model=("export_default_format",),
                items=("export_format_items",),
                item_title="title",
                item_value="value",
                density="compact",
                hide_details=True,
                variant="outlined",
                classes="mb-2",
            )
            html.Div(
                "PRESET",
                style=(
                    "font-size: 10px; letter-spacing: 0.04em; "
                    "color: #888a90; text-transform: uppercase; "
                    "margin-bottom: 4px;"
                ),
            )
            v3.VSelect(
                v_model=("export_default_preset",),
                items=("export_preset_items",),
                item_title="title",
                item_value="value",
                density="compact",
                hide_details=True,
                variant="outlined",
                classes="mb-3",
            )

            # ---- Quick-action row ----
            with html.Div(
                classes="d-flex align-center mb-2",
                style="gap: 8px;",
            ):
                html.Button(
                    "Select all available",
                    type="button",
                    classes=(
                        "golgi-btn-secondary "
                        "golgi-btn-sm"
                    ),
                    click=do_bulk_export_select_all_available,
                )
                html.Button(
                    "Clear",
                    type="button",
                    classes=(
                        "golgi-btn-secondary "
                        "golgi-btn-sm"
                    ),
                    click=do_bulk_export_clear,
                )
                html.Span(
                    "{{ exports_selected_fig_ids.length }} "
                    "selected · "
                    "{{ exports_registry_meta.length }} total",
                    style=(
                        "margin-left: auto; "
                        "font-size: 11px; "
                        "color: #555;"
                    ),
                )

            # ---- Categorised checkbox list ----
            # Flat single-pass iteration over the pre-grouped data
            # `state.exports_registry_grouped = [{category, items}]`.
            # The nested `<template v-for>` + `.filter()` pattern v1
            # used lost rows for some categories on first paint —
            # Vue 3 nested template v-fors need an explicit :key on
            # each level for stable reactivity. Pre-grouping the
            # data + binding :key directly side-steps that.
            with html.Div(
                classes="golgi-exports-list",
                style=(
                    "max-height: 380px; "
                    "overflow-y: auto; "
                    "border: 1px solid #ececef; "
                    "border-radius: 6px; "
                    "padding: 8px 12px; "
                    "background: #fafafa; "
                    "margin-bottom: 12px;"
                ),
            ):
                with html.Div(
                    v_for="g in exports_registry_grouped",
                    key="g.category",
                ):
                    html.Div(
                        "{{ g.category }}",
                        style=(
                            "font-size: 10px; "
                            "letter-spacing: 0.04em; "
                            "color: #888a90; "
                            "text-transform: uppercase; "
                            "margin: 10px 0 4px 0; "
                            "font-weight: 600;"
                        ),
                    )
                    with html.Div(
                        v_for="m in g.items",
                        key="m.id",
                    ):
                        v3.VCheckbox(
                            v_model=("exports_selected_fig_ids",),
                            value=("m.id",),
                            label=("m.title + ' (' + m.id + ')'",),
                            density="compact",
                            hide_details=True,
                            color="primary",
                            classes="golgi-export-row",
                        )

            # ---- Status / progress / error ----
            html.Div(
                "{{ bulk_export_pending_status }}",
                v_show=(
                    "bulk_export_pending_status",
                ),
                style=(
                    "font-size: 11px; "
                    "color: #146e3a; "
                    "margin-bottom: 6px;"
                ),
            )
            html.Pre(
                "{{ bulk_export_progress }}",
                v_show=("bulk_export_progress",),
                style=(
                    "font-size: 10px; "
                    "color: #444; "
                    "background: #f6f6f7; "
                    "padding: 6px 10px; "
                    "border-radius: 4px; "
                    "margin-bottom: 8px; "
                    "font-family: monospace; "
                    "white-space: pre-wrap; "
                    "max-height: 140px; "
                    "overflow-y: auto;"
                ),
            )
            html.Div(
                "{{ bulk_export_pending_error }}",
                v_show=("bulk_export_pending_error",),
                style=(
                    "font-size: 11px; "
                    "color: #c0392b; "
                    "margin-bottom: 8px;"
                ),
            )

            # ---- CTA: Export to ZIP ----
            html.Button(
                "Export to ZIP",
                type="button",
                classes=(
                    "golgi-btn-primary "
                    "golgi-btn-sm "
                    "golgi-btn-block"
                ),
                disabled=(
                    "bulk_export_pending_busy "
                    "|| !exports_selected_fig_ids.length",
                ),
                click=do_bulk_export,
                style="margin-bottom: 6px;",
            )
            # Download anchor — appears after the bulk export
            # finishes. Same golgi-btn-secondary look as the
            # per-figure popover's Download.
            with html.A(
                href=("bulk_export_pending_data_uri",),
                download=("bulk_export_pending_filename",),
                classes=(
                    "golgi-btn-secondary "
                    "golgi-btn-sm "
                    "golgi-btn-block"
                ),
                v_show=("bulk_export_pending_data_uri",),
            ):
                html.I(
                    classes="mdi mdi-folder-zip-outline",
                    style="font-size: 16px;",
                )
                html.Span("Download ZIP")
