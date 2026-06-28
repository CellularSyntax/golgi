# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fibers (trajectories) drawer — streamline integration knobs,
cap detection / branch-clustering params, Generate CTA, and a
post-build branch-summary table with inline rename + failure
banner."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from ._helpers import param_row_with_info, slider_row


def render(
    *,
    do_generate_fibers: Callable,
    do_start_branch_rename: Callable,
    do_apply_branch_rename: Callable,
    do_cancel_branch_rename: Callable,
) -> None:
    with v3.VNavigationDrawer(
        v_model=("show_fibers",),
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
                        "Fiber trajectories",
                        classes="text-h6 mb-0",
                    )
                    with v3.VTooltip(
                        location="bottom",
                        max_width=360,
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
                            "Streamlines through the "
                            "endoneurium subdomain "
                            "(Laplace −∇φ → RK4 "
                            "integration). Requires a "
                            "built TetGen mesh."
                        )
                v3.VBtn(
                    icon="mdi-close", size="small",
                    variant="text",
                    click="show_fibers = false",
                )

            # Status banner — also surfaces "no mesh yet"
            # message that gates the Generate button.
            html.Div(
                "{{ fiber_status }}",
                style=("font-size: 11px; color: #444; "
                        "background: #f6f6f7; padding: 6px 10px; "
                        "border-radius: 4px; "
                        "margin-bottom: 12px; "
                        "font-family: monospace;"),
            )

            # ---- Streamline integration ----
            html.H4(
                "Streamlines",
                classes="text-subtitle-2 mt-2 mb-1",
                style=(
                    "color: #888a90; "
                    "letter-spacing: 0.04em; "
                    "text-transform: uppercase; "
                    "font-size: 10px;"
                ),
            )
            slider_row(
                "n_fibers",
                "Number of fiber seeds",
                1, 500, 1, "toFixed(0)",
            )
            slider_row(
                "fiber_max_steps",
                "Max integration steps (RK4)",
                1000, 50000, 500, "toFixed(0)",
            )
            v3.VSelect(
                v_model=("fiber_seed_end",),
                items=([
                    "trunk (low z)",
                    "branched (high z)",
                ],),
                label="Seed cap end",
                density="compact", hide_details=True,
                classes="mt-2 mb-3",
            )

            # ---- Cap detection / branch clustering ----
            # Knobs the solver uses to find the cap facets at
            # each end of the nerve and cluster them into
            # individual branches. Defaults match historical
            # hard-coded values so changing nothing reproduces
            # prior behaviour.
            html.H4(
                "Cap detection",
                classes="text-subtitle-2 mt-3 mb-1",
                style=(
                    "color: #888a90; "
                    "letter-spacing: 0.04em; "
                    "text-transform: uppercase; "
                    "font-size: 10px;"
                ),
            )
            param_row_with_info(
                "fiber_cluster_eps_mm",
                "Cluster radius",
                "DBSCAN xy-radius used to group adjacent "
                "cap facets into one cluster. Two branches "
                "whose end caps sit closer than this in "
                "the xy-plane merge into a single cluster. "
                "Decrease for tightly-spaced branches; "
                "increase if a real cap is being split.",
                "mm", 0.1, 0.1, 20.0,
            )
            param_row_with_info(
                "fiber_cap_band_pct",
                "Cap z-band",
                "Width of the z-band at each end of the "
                "nerve (as % of total nerve length) within "
                "which axial-normal facets are considered "
                "candidate cap facets. Narrower bands "
                "reject mid-trunk artefacts; wider bands "
                "tolerate tilted caps.",
                "%", 1.0, 1.0, 40.0,
            )
            param_row_with_info(
                "fiber_min_rel_size_pct",
                "Min cluster size",
                "Drop any cluster whose facet count is "
                "below this fraction of the largest "
                "cluster at the same end. Filters out "
                "surface artefacts (small lateral kinks "
                "that happen to have axial-aligned "
                "normals) before they get reported as "
                "spurious branches.",
                "%", 1.0, 0.0, 90.0,
            )
            param_row_with_info(
                "fiber_axial_normal_thresh",
                "Axial normal threshold",
                "Minimum |n·ẑ| (intrinsic frame) for a "
                "boundary facet to be considered "
                "cap-like. 0 = include all facets; 1 = "
                "only perfectly axial facets. Lowering it "
                "captures tilted caps but admits more "
                "lateral-wall noise.",
                "", 0.05, 0.0, 1.0,
            )

            html.Button(
                "▶ Generate trajectories",
                type="button",
                classes=(
                    "golgi-btn-primary "
                    "golgi-btn-block mb-3 mt-3"
                ),
                disabled=("!has_geometry",),
                click=do_generate_fibers,
            )

            # ---- Branch summary (with inline rename) ----
            with html.Div(
                v_show=("has_fibers && !fiber_failed",),
            ):
                html.Div(
                    "Branch summary",
                    style=("font-size: 11px; color: #555; "
                            "margin-top: 16px; "
                            "margin-bottom: 4px; "
                            "letter-spacing: 0.03em; "
                            "text-transform: uppercase;"),
                )
                html.Div(
                    "Trajectory length statistics by branch. "
                    "Mean, min, max, std are in millimetres.",
                    style=("font-size: 10px; "
                            "color: #888a90; "
                            "margin-bottom: 8px; "
                            "line-height: 1.4;"),
                )
                with html.Div(
                    classes="golgi-branch-summary",
                ):
                    # ---- Header rows (static) ----
                    with html.Div(
                        classes=(
                            "golgi-branch-summary-row "
                            "is-header"
                        ),
                    ):
                        html.Span(
                            "",
                            classes="golgi-bs-name",
                        )
                        html.Span(
                            "Fibers",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "Mean",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "Min",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "Max",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "Std",
                            classes="golgi-bs-num",
                        )
                    with html.Div(
                        classes=(
                            "golgi-branch-summary-row "
                            "is-subheader"
                        ),
                    ):
                        html.Span(
                            "",
                            classes="golgi-bs-name",
                        )
                        html.Span(
                            "",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "mm",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "mm",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "mm",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "mm",
                            classes="golgi-bs-num",
                        )
                    # ---- Data rows (v_for over summary) ----
                    with html.Div(
                        v_for=(
                            "row in fiber_branch_summary",
                        ),
                        key="row.idx",
                        classes=(
                            "['golgi-branch-summary-row', "
                            "row.idx === -1 "
                            "? 'is-overall' "
                            ": 'is-branch']",
                        ),
                    ):
                        # ---- Name column (display / edit) ----
                        # Two-mode: display (swatch + name +
                        # pencil) vs edit (text field + ✓ /
                        # ✕). Mode keyed on
                        # branch_rename_active === row.idx.
                        with html.Div(
                            classes="golgi-bs-name",
                        ):
                            # Display mode. `row.editable`
                            # gates the pencil so the Overall
                            # row stays read-only.
                            with html.Div(
                                v_show=(
                                    "branch_rename_active "
                                    "!== row.idx",
                                ),
                                classes=(
                                    "golgi-bs-name-display"
                                ),
                            ):
                                html.Div(
                                    v_show=("row.color",),
                                    classes=(
                                        "golgi-bs-swatch"
                                    ),
                                    style=(
                                        "'background: ' "
                                        "+ row.color",
                                    ),
                                )
                                html.Span(
                                    "{{ row.label }}",
                                    classes=(
                                        "golgi-bs-label"
                                    ),
                                )
                                html.Button(
                                    "✎",
                                    type="button",
                                    v_show=("row.editable",),
                                    classes=(
                                        "golgi-bs-edit-btn"
                                    ),
                                    title="Rename branch",
                                    click=(
                                        do_start_branch_rename,
                                        "[row.idx]",
                                    ),
                                )
                            # Edit mode.
                            with html.Div(
                                v_show=(
                                    "branch_rename_active "
                                    "=== row.idx",
                                ),
                                classes=(
                                    "golgi-bs-name-edit"
                                ),
                            ):
                                v3.VTextField(
                                    v_model=(
                                        "branch_rename_value",
                                    ),
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    autofocus=True,
                                    keydown_enter=(
                                        do_apply_branch_rename
                                    ),
                                    keydown_escape=(
                                        do_cancel_branch_rename
                                    ),
                                    style=(
                                        "flex: 1 1 auto; "
                                        "min-width: 0;"
                                    ),
                                )
                                html.Button(
                                    "✓",
                                    type="button",
                                    classes=(
                                        "golgi-bs-save-btn"
                                    ),
                                    title="Save",
                                    click=(
                                        do_apply_branch_rename
                                    ),
                                )
                                html.Button(
                                    "✕",
                                    type="button",
                                    classes=(
                                        "golgi-bs-cancel-btn"
                                    ),
                                    title="Cancel",
                                    click=(
                                        do_cancel_branch_rename
                                    ),
                                )
                        # ---- Stats columns ----
                        html.Span(
                            "{{ row.n_fibers }}",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "{{ row.mean_mm.toFixed(1) }}",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "{{ row.min_mm.toFixed(1) }}",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "{{ row.max_mm.toFixed(1) }}",
                            classes="golgi-bs-num",
                        )
                        html.Span(
                            "{{ row.std_mm.toFixed(1) }}",
                            classes="golgi-bs-num",
                        )
            # ---- failure: error banner + log tail ----
            with html.Div(v_show=("fiber_failed",)):
                html.Div(
                    "{{ fiber_status }}",
                    style=("color: #b8336a; font-size: 12px; "
                            "font-weight: 600; "
                            "margin-bottom: 6px;"),
                )
                html.Pre(
                    "{{ fiber_log }}",
                    style=("font-family: ui-monospace,"
                            "Menlo,Consolas,monospace; "
                            "font-size: 10px; color:#222; "
                            "background:#f4f4f4; padding:8px; "
                            "max-height:300px; overflow:auto;"
                            "white-space: pre-wrap;"),
                )
