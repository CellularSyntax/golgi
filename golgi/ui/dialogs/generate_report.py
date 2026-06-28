# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Generate Report dialog (F2.3.c).

Modal launched from the navbar's Report entry. Section checkboxes
let the user pick which sections to include in the multi-page PDF;
auto-included sections (cover, TOC, conductivity, reproducibility,
bibliography, audit) are not toggleable.

A note above the checkboxes reminds the user to set up the
workspace viewport they want captured BEFORE clicking Generate
— v1 reuses one snapshot for every 3D-render section. v2 will
swap in multi-variant programmatic renders from
golgi.figures.render3d.

Generate → busy spinner → Download anchor activates.
"""
from __future__ import annotations

from typing import Callable

from trame.widgets import html, vuetify3 as v3


def render(
    *,
    do_generate_report: Callable,
    do_close_generate_report_dialog: Callable,
) -> None:
    with v3.VDialog(
        v_model=("show_generate_report_dialog",),
        max_width=620,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Generate report",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Builds a multi-page PDF combining the figures + "
                    "sim configs you select below, plus an auto "
                    "appendix with conductivity table, "
                    "reproducibility hashes, bibliography, and "
                    "audit excerpt.",
                    classes="golgi-dialog-body mb-3",
                )

                html.Div(
                    "VIEWPORT NOTE",
                    style=(
                        "font-size: 10px; "
                        "letter-spacing: 0.04em; "
                        "color: #888a90; "
                        "text-transform: uppercase; "
                        "margin-bottom: 4px;"
                    ),
                )
                html.Div(
                    "v1 captures the workspace 3D viewport ONCE "
                    "and reuses that snapshot for every 3D-render "
                    "section (Electrode / Mesh / Fibers / FEM). "
                    "Toggle visibility + position the camera in "
                    "the workspace before clicking Generate — "
                    "multi-variant region renders land in a "
                    "follow-up commit.",
                    style=(
                        "font-size: 11px; "
                        "color: #555; "
                        "line-height: 1.4; "
                        "margin-bottom: 12px;"
                    ),
                )

                html.Div(
                    "SECTIONS TO INCLUDE",
                    style=(
                        "font-size: 10px; "
                        "letter-spacing: 0.04em; "
                        "color: #888a90; "
                        "text-transform: uppercase; "
                        "margin-bottom: 4px;"
                    ),
                )
                v3.VCheckbox(
                    v_model=("report_section_electrode",),
                    label="Electrode design (3D snapshot)",
                    density="compact",
                    hide_details=True,
                    color="primary",
                )
                v3.VCheckbox(
                    v_model=("report_section_mesh",),
                    label="Mesh results (3D + tet-quality histogram)",
                    density="compact",
                    hide_details=True,
                    color="primary",
                )
                v3.VCheckbox(
                    v_model=("report_section_fibers",),
                    label=(
                        "Fiber trajectories (3D + branch summary)"
                    ),
                    density="compact",
                    hide_details=True,
                    color="primary",
                )
                v3.VCheckbox(
                    v_model=("report_section_fem",),
                    label=(
                        "FEM results (3D + axis + slice + AF)"
                    ),
                    density="compact",
                    hide_details=True,
                    color="primary",
                )
                v3.VCheckbox(
                    v_model=("report_section_single_fiber",),
                    label=(
                        "Single-fiber simulation "
                        "(pulse + propagation + waterfall + config)"
                    ),
                    density="compact",
                    hide_details=True,
                    color="primary",
                )
                v3.VCheckbox(
                    v_model=("report_section_population",),
                    label=(
                        "Population simulation "
                        "(KDE + xsec + activation + config)"
                    ),
                    density="compact",
                    hide_details=True,
                    color="primary",
                )
                v3.VCheckbox(
                    v_model=("report_section_sweep",),
                    label=(
                        "Sweep (recruitment + threshold + heatmap)"
                    ),
                    density="compact",
                    hide_details=True,
                    color="primary",
                )

                html.Div(
                    "Always included: cover · TOC · conductivities "
                    "(table + σ(f) plot) · reproducibility "
                    "appendix · bibliography · audit excerpt.",
                    style=(
                        "font-size: 11px; "
                        "color: #888a90; "
                        "margin-top: 8px; "
                        "font-style: italic;"
                    ),
                )

                # Status / error / pending download
                html.Div(
                    "{{ report_pending_status }}",
                    v_show=("report_pending_status",),
                    style=(
                        "margin-top: 14px; "
                        "font-size: 11px; "
                        "color: #146e3a;"
                    ),
                )
                html.Div(
                    "{{ report_pending_error }}",
                    v_show=("report_pending_error",),
                    style=(
                        "margin-top: 6px; "
                        "font-size: 11px; "
                        "color: #c0392b;"
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
                    click=do_close_generate_report_dialog,
                )
                # The Download anchor only appears AFTER a
                # successful generation. Same data-URI pattern as
                # the per-figure popover + bulk-exports drawer.
                with html.A(
                    href=("report_pending_data_uri",),
                    download=("report_pending_filename",),
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    v_show=("report_pending_data_uri",),
                    style="margin-right: 8px;",
                ):
                    html.I(
                        classes="mdi mdi-file-pdf-box",
                        style="font-size: 16px;",
                    )
                    html.Span("Download PDF")
                html.Button(
                    "Generate",
                    type="button",
                    classes=(
                        "golgi-btn-primary golgi-btn-sm"
                    ),
                    disabled=("report_pending_busy",),
                    click=do_generate_report,
                )
