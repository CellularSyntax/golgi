# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Per-figure export defaults (F2.3.a).

Two global pickers (format + preset) used by every per-panel
export button — keep the popover UI simple and let the user set a
sensible default once. Plus a SHARED "pending export" slot so the
browser-download anchor can read the active export's bytes + name
without coupling each panel to its own state var triple.

The pending slot uses a fig_id discriminator: the popover's
Download anchor only un-hides when `export_pending_fig_id ===
this_panel.fig_id`, so opening the popover on panel A while
panel B's export is still pending doesn't leak A's anchor to B.
"""
from __future__ import annotations


def register(state) -> None:
    # ---- Global defaults (popover dropdowns bind to these) ----
    # "pdf" picked over "svg" because PDF is the most-asked-for
    # journal format and renders consistently across viewers. Users
    # change once and it sticks for the session — full per-figure
    # presets land in the bulk Exports tab (F2.3.b).
    state.export_default_format = "pdf"      # "pdf" | "svg" | "png"
    # Default to match-ui so the export looks like the on-screen
    # figure (just at 2× resolution). paper-300 / paper-600 are
    # available for users who specifically want journal-spec
    # dimensions and 8-pt fonts.
    state.export_default_preset = "match-ui"
    # ^ key into golgi.figures.export.PRESETS — paper-300 is the
    # canonical journal preset (3.4 × 2.6 in, viridis-cb palette,
    # 8 pt sans). screen / paper-600 / paper-svg also available.

    # ---- Pending export slot ----
    # Filled by the do_export_single_figure handler once the render
    # completes. The popover's Download anchor binds href +
    # download to these vars; v_show'd by fig_id discriminator.
    state.export_pending_fig_id = ""        # which spec we built
    state.export_pending_data_uri = ""      # data:<mime>;base64,…
    state.export_pending_filename = ""      # suggested file name
    state.export_pending_busy = False       # spinner on the Generate button
    state.export_pending_error = ""         # surfaced under the popover

    # ---- Dropdown item lists ----
    # Pushed into state so VSelect (which evaluates `items=` as a
    # Vue expression in the trame template sandbox) can read them
    # by name. Passing a Python list directly to VSelect's items
    # prop ends up as a string in Vue and renders "No data
    # available" — the items must be a state-backed array.
    state.export_format_items = [
        {"title": "PDF (vector)", "value": "pdf"},
        {"title": "SVG (vector)", "value": "svg"},
        {"title": "PNG (raster)", "value": "png"},
    ]
    # ---- Bulk export drawer (F2.3.b) ----
    # Array of fig_ids that are checked in the Exports drawer. Each
    # VCheckbox binds via `value="fig.id"` + `v-model` on this
    # array, so toggling a box adds/removes the id automatically.
    state.exports_selected_fig_ids = []
    # Toggles
    state.exports_include_csvs = False        # CSV alongside each fig
    # Pending bulk-export ZIP slot. Separate from the single-figure
    # slot so opening a per-panel popover during a bulk export
    # doesn't leak the bulk's filename to the popover.
    state.bulk_export_pending_data_uri = ""
    state.bulk_export_pending_filename = ""
    state.bulk_export_pending_busy = False
    state.bulk_export_pending_status = ""    # short status line
    state.bulk_export_pending_error = ""
    state.bulk_export_progress = ""          # rolling progress log

    # Registry meta for the bulk-export drawer's v-for. Each entry
    # carries {id, title, category}; populated by build_app right
    # after `_actions.figure_export.register` so the dropdown
    # always reflects the current registry without a refresh.
    state.exports_registry_meta = []
    # Distinct categories in display order — drawn from the
    # registry meta in build_app. The drawer renders one section
    # header per category, then iterates the meta filtered by
    # `e.category === '<cat>'`.
    state.exports_registry_categories = []
    # Pre-grouped variant — [{category, items: [{id,title}]}] —
    # populated alongside the flat list in build_app. The drawer
    # iterates this single source so the inner v-for doesn't need
    # a runtime .filter().
    state.exports_registry_grouped = []

    # ---- Generate Report dialog (F2.3.c) ----
    state.show_generate_report_dialog = False
    state.report_section_electrode = True
    state.report_section_mesh = True
    state.report_section_fibers = True
    state.report_section_fem = True
    state.report_section_single_fiber = True
    state.report_section_population = True
    state.report_section_sweep = True
    state.report_pending_data_uri = ""
    state.report_pending_filename = ""
    state.report_pending_busy = False
    state.report_pending_status = ""
    state.report_pending_error = ""

    state.export_preset_items = [
        {
            "title": (
                "match-ui (PDF, keeps on-screen layout @ 2× res)"
            ),
            "value": "match-ui",
        },
        {
            "title": (
                "match-ui-png (PNG, keeps on-screen layout @ 2× res)"
            ),
            "value": "match-ui-png",
        },
        {
            "title": "paper-300 (PDF, 3.4×2.6 in, 8 pt, viridis-cb)",
            "value": "paper-300",
        },
        {
            "title": "paper-600 (PNG @ 600 DPI, same layout)",
            "value": "paper-600",
        },
        {
            "title": "paper-svg (SVG, paper layout)",
            "value": "paper-svg",
        },
        {
            "title": "screen (PNG @ 120 DPI, default palette)",
            "value": "screen",
        },
    ]
