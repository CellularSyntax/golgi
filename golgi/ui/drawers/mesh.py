# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Mesh drawer — TetGen multi-domain build parameters.

As of F3.2-M2.1c the nerve-level params (decim_target_k, use_epi,
epi_thickness_um, lc_endo_um, lc_epi_um, lc_muscle_um, muscle
pads + offsets) all live in the import stepper (Steps 2 + 4).
This drawer now only hosts the per-design CUFF mesh-size knobs
(saline / silicone / contact) plus the design selector + Build
button and the post-build stats panel."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

try:                                                # pragma: no cover
    from trame.widgets import plotly as twp
except ImportError:                                 # pragma: no cover
    twp = None

from ._helpers import param_row_with_info


def render(
    *,
    do_build_mesh: Callable,
    export_btn: Callable | None = None,
) -> None:
    with v3.VNavigationDrawer(
        v_model=("show_mesh",),
        location="right", width=480,
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
                        "Mesh build",
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
                            "Build a per-design TetGen mesh. "
                            "Nerve / epineurium / muscle params "
                            "are set in the import stepper; "
                            "only the cuff-shell mesh sizes "
                            "and the design selector live here."
                        )
                v3.VBtn(
                    icon="mdi-close", size="small",
                    variant="text",
                    click="show_mesh = false",
                )

            # ---- Tissue region mesh sizes ----
            # F3.2-M3: nerve-tissue mesh sizes moved here from the
            # import stepper. They're shared across designs (the
            # nerve mesh is the same for every cuff), and grouping
            # them with the cuff sizes lets the user tune all six
            # tag-level edge lengths from one place.
            v3.VSwitch(
                v_model=("use_gmsh_mesher",),
                label=(
                    "gmsh mesher (conformal, robust; auto-falls back to "
                    "TetGen for true-3D nerves or on error)"
                ),
                density="compact", hide_details=True, color="primary",
                classes="mb-1",
            )
            html.H4(
                "Tissue region mesh sizes",
                classes="text-subtitle-2 mt-1 mb-1",
                style=(
                    "color: #888a90; "
                    "letter-spacing: 0.04em; "
                    "text-transform: uppercase; "
                    "font-size: 10px;"
                ),
            )
            param_row_with_info(
                "lc_endo_um",
                "Endoneurium",
                "Characteristic edge length inside the "
                "endoneurium (tag 1). Drives the FEM "
                "solve cost — finer = better Vₑ "
                "resolution.",
                "µm", 10, 50, 500,
            )
            param_row_with_info(
                "lc_epi_um",
                "Epineurium",
                "Edge length inside the epineurium "
                "shell (tag 5). Should be ≤ shell "
                "thickness so at least one tet fits "
                "radially. Only used when the "
                "epineurium shell is enabled in the "
                "import stepper.",
                "µm", 5, 25, 500,
            )
            param_row_with_info(
                "lc_muscle_um",
                "Muscle",
                "Characteristic edge length inside the "
                "muscle bbox (tag 4). Usually the "
                "coarsest region — far from the fibers "
                "but dominates tet count by volume.",
                "µm", 100, 200, 3000,
            )
            param_row_with_info(
                "lc_scar_um",
                "Scar / connective tissue",
                "Characteristic edge length inside the "
                "scar / connective tissue shell (tag 7). "
                "Should be ≤ the scar layer thickness so "
                "at least one tet fits radially. Only "
                "used when a design has the scar shell "
                "enabled in the design drawer.",
                "µm", 5, 25, 500,
            )

            # ---- Cuff region mesh sizes ----
            # Cuff shell + gold contacts — these exist only when
            # a design is placed, so they live with the build
            # step.
            html.H4(
                "Cuff region mesh sizes",
                classes="text-subtitle-2 mt-4 mb-1",
                style=(
                    "color: #888a90; "
                    "letter-spacing: 0.04em; "
                    "text-transform: uppercase; "
                    "font-size: 10px;"
                ),
            )
            param_row_with_info(
                "lc_saline_um",
                "Saline",
                "Characteristic edge length inside the "
                "saline region (tag 2) between the "
                "nerve and the cuff inner wall.",
                "µm", 10, 50, 500,
            )
            param_row_with_info(
                "lc_silicone_um",
                "Silicone",
                "Characteristic edge length inside the "
                "silicone cuff shell (tag 3).",
                "µm", 20, 100, 800,
            )
            param_row_with_info(
                "lc_contact_um",
                "Contacts",
                "Characteristic edge length inside the "
                "gold contacts (tag 6). Should be the "
                "finest region so the current density "
                "at the metal-saline interface is "
                "well-resolved.",
                "µm", 10, 30, 300,
            )

            # F3.2a: which designs to mesh. Each design owns its
            # own multi-domain mesh (the cuff silicone is baked
            # into the mesh, and the cuff moves/rotates per
            # design). Multi-select so the user can queue several
            # in a single run; an empty selection falls back to
            # the currently-active design.
            html.H4(
                "Designs to mesh",
                classes="text-subtitle-2 mt-4 mb-1",
            )
            html.Div(
                "Pick which cuff designs to build meshes for. "
                "TetGen runs once per design in sequence — each "
                "mesh lands in <out>/designs/<eid>/. Empty list "
                "→ just the active design.",
                style=(
                    "font-size: 10px; color: #888a90; "
                    "margin-bottom: 6px; line-height: 1.4;"
                ),
            )
            v3.VSelect(
                v_model=("mesh_design_selection",),
                items=("designs",),
                item_title="name",
                item_value="eid",
                multiple=True,
                chips=True,
                closable_chips=True,
                density="compact",
                hide_details=True,
                variant="outlined",
                classes="mb-1",
            )
            html.Div(
                "Will build "
                "{{ (mesh_design_selection || []).length "
                "    || 1 }} mesh(es).",
                style=(
                    "font-size: 10px; color: #888; "
                    "margin-bottom: 8px;"
                ),
            )

            html.Button(
                "▶ Build mesh (TetGen)",
                type="button",
                classes=(
                    "golgi-btn-primary golgi-btn-block "
                    "mt-2 mb-3"
                ),
                disabled=(
                    "!(designs && designs.length > 0)",
                ),
                click=do_build_mesh,
            )

            # ---- Built meshes (delete per design) ----
            # One row per design that currently has a built
            # mesh. Styled to mirror the electrode-designs list
            # in cuff_electrodes.py (white card with thin border,
            # name + electrode-type label stacked vertically,
            # round ✕ delete on the right). Clicking ✕ opens the
            # confirm_delete_mesh dialog; the dialog's Delete
            # button posts the eid into `remove_mesh_request`,
            # which watchers/cuff.py routes to `do_delete_mesh`
            # (clears <project>/designs/<eid>/ + fem/<eid>/ +
            # sims/<eid>/, flips design.has_mesh false, and
            # strips this design's region actors from the
            # plotter).
            #
            # Hidden when no design has a built mesh so the
            # drawer stays compact pre-build.
            with html.Div(
                v_show=(
                    "(designs || []).some("
                    "  d => d && d.has_mesh)",
                ),
                classes="mb-3",
            ):
                html.H4(
                    "Built meshes",
                    classes="text-subtitle-2 mt-2 mb-1",
                    style=(
                        "color: #888a90; "
                        "letter-spacing: 0.04em; "
                        "text-transform: uppercase; "
                        "font-size: 10px;"
                    ),
                )
                with html.Div(
                    style=(
                        "display: flex; "
                        "flex-direction: column; gap: 4px;"
                    ),
                ):
                    with html.Div(
                        v_for=(
                            "design in (designs || [])"
                            ".filter(d => d && d.has_mesh)"
                        ),
                        key="design.eid",
                        style=(
                            "padding: 8px 10px; "
                            "border-radius: 8px; "
                            "background: #ffffff; "
                            "border: 1px solid #e3e3e6; "
                            "border-left: 5px solid "
                            "  transparent; "
                            "display: flex; "
                            "flex-direction: row; "
                            "flex-wrap: nowrap; "
                            "align-items: center; "
                            "gap: 8px;"
                        ),
                    ):
                        with html.Div(
                            style=(
                                "flex: 1 1 auto; "
                                "min-width: 0; "
                                "display: flex; "
                                "flex-direction: column; "
                                "gap: 2px; "
                                "overflow: hidden;"
                            ),
                        ):
                            html.Span(
                                "{{ design.name }}",
                                style=(
                                    "font-size: 12px; "
                                    "font-weight: 500; "
                                    "color: #1f2024; "
                                    "white-space: nowrap; "
                                    "overflow: hidden; "
                                    "text-overflow: ellipsis;"
                                ),
                            )
                            html.Span(
                                "nerve.msh + cached PLC / "
                                "TetGen artefacts",
                                style=(
                                    "font-size: 10px; "
                                    "color: #888a90; "
                                    "white-space: nowrap; "
                                    "overflow: hidden; "
                                    "text-overflow: ellipsis;"
                                ),
                            )
                        html.Button(
                            "✕",
                            type="button",
                            classes=(
                                "golgi-btn-secondary "
                                "golgi-btn-round "
                                "golgi-btn-sm"
                            ),
                            title="Delete mesh",
                            style="flex: 0 0 auto;",
                            click=(
                                "$event.stopPropagation(); "
                                "confirm_delete_mesh_eid = "
                                "  design.eid; "
                                "confirm_delete_mesh_name = "
                                "  design.name; "
                                "show_confirm_delete_mesh_"
                                "dialog = true"
                            ),
                        )

            # ---- after a build: per-design stats panels +
            # combined quality histogram ----
            # F3.2-M2.1e: the histogram is one tall plotly
            # figure with N rows (one subplot per built
            # design) — `mesh_quality_hist_figure` is
            # rebuilt by the pipeline's end-of-batch hook.
            # Stats HTML is iterated separately so each
            # design's per-tag table sits with its name.
            with html.Div(v_show=("has_mesh",)):
                # Per-design stats tables, stacked.
                with html.Div(
                    v_for=(
                        "panel in (designs_mesh_panels || [])",
                    ),
                    key="panel.eid",
                    classes="mb-3",
                ):
                    html.H4(
                        "{{ panel.name }}",
                        classes=(
                            "text-subtitle-2 mt-2 mb-1"
                        ),
                        style=(
                            "color: #1f2024; "
                            "font-size: 13px;"
                        ),
                    )
                    html.Div(
                        "Mesh statistics",
                        style=(
                            "font-size: 10px; color: #888; "
                            "margin-bottom: 4px; "
                            "letter-spacing: 0.03em; "
                            "text-transform: uppercase;"
                        ),
                    )
                    html.Div(
                        v_html=("panel.stats_html",),
                        style=(
                            "background:#fafafa; "
                            "border:1px solid #e6e6e8; "
                            "border-radius:6px; "
                            "padding:10px; "
                            "margin-bottom:8px;"
                        ),
                    )
                # Combined tet-quality histogram, one row per
                # design. Height scales with the count so
                # extra meshes don't crush each row's bars.
                html.Div(
                    "Tet shape-quality "
                    "(6√2·V / max_edge³)",
                    style=(
                        "font-size: 11px; color: #555; "
                        "margin-top: 12px; "
                        "margin-bottom: 6px; "
                        "letter-spacing: 0.03em; "
                        "text-transform: uppercase;"
                    ),
                )
                # Static 480 px fits ~3 design rows comfortably.
                # For more designs the inner plotly subplot
                # rows compress but stay legible; scrolling the
                # drawer reveals the rest of the panel column.
                with html.Div(
                    style=(
                        "width: 100%; max-width: 100%; "
                        "height: 480px; display: block; "
                        "border: 1px solid #e6e6e8; "
                        "border-radius: 6px; "
                        "background: white; "
                        "position: relative;"
                    ),
                ):
                    if export_btn is not None:
                        export_btn("mesh.tet_quality_hist")
                    if twp is not None:
                        twp.Figure(
                            state_variable_name=(
                                "mesh_quality_hist_figure"
                            ),
                            display_logo=False,
                            display_mode_bar=True,
                        )

            # (mesh log lives in the lightbox during a build,
            # no need to duplicate it here)
