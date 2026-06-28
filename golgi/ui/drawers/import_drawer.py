# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Import drawer — Source file picker, unit-scaling preset +
factor, optional file upload, Load / Remove buttons, geom
summary readout, and triangle-quality histogram + colour
toggle (the last two only after a nerve is loaded)."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

try:                                                # pragma: no cover
    from trame.widgets import plotly as twp
except ImportError:                                 # pragma: no cover
    twp = None


def render(
    *,
    do_load_geometry: Callable,
    export_btn: Callable | None = None,
) -> None:
    with v3.VNavigationDrawer(
        v_model=("show_import",),
        location="right", width=420,
        elevation=8,
    ):
        with v3.VContainer(classes="pa-4"):
            # Title row: H3 + info-icon tooltip + close button.
            with html.Div(
                classes=(
                    "d-flex align-center "
                    "justify-space-between mb-3"
                ),
            ):
                with html.Div(
                    classes="d-flex align-center",
                ):
                    html.H3(
                        "Import nerve geometry",
                        classes="text-h6 mb-0",
                    )
                    with v3.VTooltip(
                        location="bottom",
                        max_width=380,
                    ):
                        with v3.Template(
                            v_slot_activator=(
                                "{ props }",
                            ),
                        ):
                            with v3.VBtn(
                                v_bind="props",
                                icon=True,
                                size="small",
                                variant="text",
                                density="compact",
                                classes="ml-1",
                            ):
                                v3.VIcon(
                                    "mdi-information-outline",
                                    size="18",
                                    color="grey-darken-1",
                                )
                        html.Span(
                            "Load a closed nerve surface "
                            "from disk (STL / NAS / OBJ) "
                            "or upload one. The file is "
                            "scaled to metres, run "
                            "through a global PCA, and "
                            "shown in the viewport. The "
                            "summary readout below "
                            "reports triangle counts, "
                            "watertightness and "
                            "per-triangle quality."
                        )
                v3.VBtn(
                    icon="mdi-close", size="small",
                    variant="text",
                    click="show_import = false",
                )

            # ---- Source file ----
            with html.Div(
                classes="d-flex align-center",
                style="gap: 2px;",
            ):
                html.Span(
                    "Source file",
                    style=(
                        "font-size: 11px; color: #555;"
                    ),
                )
                with v3.VTooltip(
                    location="top", max_width=320,
                ):
                    with v3.Template(
                        v_slot_activator=("{ props }",),
                    ):
                        with v3.VBtn(
                            v_bind="props",
                            icon=True,
                            size="x-small",
                            variant="text",
                            density="compact",
                        ):
                            v3.VIcon(
                                "mdi-information-outline",
                                size="14",
                                color="grey-darken-1",
                            )
                    html.Span(
                        "Pick a surface mesh visible to "
                        "the app — files in the global "
                        "data/ folder plus anything you "
                        "have uploaded into this project. "
                        "STL, NAS, OBJ are recognised."
                    )
            v3.VSelect(
                v_model=("selected_file",),
                items=("data_files",),
                density="compact", hide_details=True,
                variant="outlined",
                classes="mb-3 mt-1",
            )

            # ---- Unit scaling ----
            with html.Div(
                classes="d-flex align-center",
                style="gap: 2px;",
            ):
                html.Span(
                    "Unit scaling preset",
                    style=(
                        "font-size: 11px; color: #555;"
                    ),
                )
                with v3.VTooltip(
                    location="top", max_width=320,
                ):
                    with v3.Template(
                        v_slot_activator=("{ props }",),
                    ):
                        with v3.VBtn(
                            v_bind="props",
                            icon=True,
                            size="x-small",
                            variant="text",
                            density="compact",
                        ):
                            v3.VIcon(
                                "mdi-information-outline",
                                size="14",
                                color="grey-darken-1",
                            )
                    html.Span(
                        "Multiplier that converts the "
                        "source file's coordinate units "
                        "into metres. Picking a preset "
                        "fills the numeric value below; "
                        "choose Custom to type any "
                        "factor (e.g. 5.0e-4 for a 0.5 mm "
                        "unit). Everything downstream "
                        "assumes SI."
                    )
            v3.VSelect(
                v_model=("scale_preset",),
                items=([
                    "mm → m (×1e-3)",
                    "µm → m (×1e-6)",
                    "m → m (×1)",
                    "cm → m (×1e-2)",
                    "custom",
                ],),
                density="compact", hide_details=True,
                variant="outlined",
                classes="mb-2 mt-1",
            )
            v3.VTextField(
                v_model=("scale_factor",),
                label="Scale factor (source units → m)",
                type="number", step="0.000001",
                density="compact", hide_details=True,
                variant="outlined",
                classes="mb-3",
            )

            # ---- Or upload a file ----
            v3.VFileInput(
                v_model=("upload_file",),
                label="Or upload a new file",
                density="compact", hide_details=True,
                variant="outlined",
                show_size=True,
                classes="mb-2",
            )
            html.Div(
                "{{ upload_info }}",
                style="color:#666;font-size:11px;",
                classes="mb-3",
            )
            html.Button(
                "▶ Load geometry",
                type="button",
                classes=(
                    "golgi-btn-primary golgi-btn-block mb-2"
                ),
                click=do_load_geometry,
            )
            # ---- Remove geometry (gated on has_geometry) ----
            # Destructive: clears the in-memory geometry, every
            # downstream artefact (mesh / FEM / fibers /
            # population) and the cached files on disk. User
            # parameter preferences are preserved. Opens a
            # confirmation dialog first.
            html.Button(
                "✕ Remove geometry",
                type="button",
                v_show=("has_geometry",),
                classes=(
                    "golgi-btn-secondary "
                    "golgi-btn-block mb-3"
                ),
                click=(
                    "show_confirm_remove_geometry_dialog "
                    "= true"
                ),
            )
            # Stats / quality readout (auto-populated by the
            # load worker).
            html.Pre(
                "{{ geom_summary }}",
                style=("color:#333; font-size:11px; "
                        "font-family: ui-monospace, "
                        "Menlo, Consolas, monospace; "
                        "background:#f7f7f8; "
                        "border:1px solid #e6e6e8; "
                        "border-radius:6px; "
                        "padding:8px; "
                        "white-space:pre-wrap; "
                        "margin:0;"),
            )
            # Triangle-quality histogram + colour toggle —
            # only visible once a nerve is loaded.
            with html.Div(v_show=("has_geometry",),
                             classes="mt-3"):
                with html.Div(
                    classes="d-flex align-center",
                    style=(
                        "gap: 2px; margin-bottom: 4px;"
                    ),
                ):
                    html.Span(
                        "Triangle quality "
                        "(radius ratio)",
                        style=(
                            "font-size: 11px; color: #555;"
                        ),
                    )
                    with v3.VTooltip(
                        location="top", max_width=320,
                    ):
                        with v3.Template(
                            v_slot_activator=(
                                "{ props }",
                            ),
                        ):
                            with v3.VBtn(
                                v_bind="props",
                                icon=True,
                                size="x-small",
                                variant="text",
                                density="compact",
                            ):
                                v3.VIcon(
                                    "mdi-information-outline",
                                    size="14",
                                    color="grey-darken-1",
                                )
                        html.Span(
                            "Per-triangle "
                            "inscribed-to-circumscribed "
                            "circle ratio (q_radius). 1 = "
                            "equilateral, 0 = degenerate. "
                            "Low-quality tris on the "
                            "input surface produce "
                            "low-quality tets after TetGen "
                            "and tend to break the FEM "
                            "solve."
                        )
                # Plotly histogram — interactive replacement for the
                # legacy matplotlib PNG. Same RdYlGn colouring.
                with html.Div(
                    style=("width: 100%; max-width: 100%; "
                            "height: 220px; display: block; "
                            "border: 1px solid #e6e6e8; "
                            "border-radius: 6px; "
                            "background: white; "
                            "position: relative;"),
                ):
                    if export_btn is not None:
                        export_btn("mesh.surface_quality_hist")
                    if twp is not None:
                        twp.Figure(
                            state_variable_name=(
                                "quality_hist_figure"
                            ),
                            display_logo=False,
                            display_mode_bar=True,
                        )
                v3.VCheckbox(
                    v_model=("show_quality_color",),
                    label="Colour nerve by triangle quality",
                    density="compact", hide_details=True,
                    color="primary", classes="mt-2",
                )
