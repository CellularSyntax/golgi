# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Top VAppBar — logo, project pill, File / Electrodes / Mesh /
Materials / Fibers / Simulate tab buttons, and the right-side
user avatar dropdown. v_model on the VAppBar slides it in/out
based on `has_active_project`, so the welcome view gets the full
viewport."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    logo_url: str,
    do_close_all_tabs: Callable,
    do_save_project: Callable,
    do_show_close_dialog: Callable,
    do_open_profile_dialog: Callable,
    do_open_generate_report_dialog: Callable | None = None,
    do_open_import_study_dialog: Callable | None = None,
    do_export_study: Callable | None = None,
    do_open_segment_uct_dialog: Callable | None = None,
    do_open_bundle_import_dialog: Callable | None = None,
) -> None:
    """Build the navbar inside the surrounding VAppLayout.

    State references (show_*, current_user_*, has_active_project
    …) are wired by the existing state_defaults / watchers
    modules — no Python imports needed; the strings are JS-side
    bindings evaluated by Vuetify."""
    with v3.VAppBar(
        v_model=("has_active_project",),
        density="compact",
        elevation=1,
        color="#fafafa",
        style="color: #1f2024; border-bottom: 1px solid #e6e6e6;",
    ):
        # Logo doubles as a "back to workspace" affordance —
        # clicking it closes every open drawer / analysis tab
        # and drops the viewport back to its full-screen 3-D
        # render. Cursor + tooltip-style title surface this.
        html.Img(
            src=logo_url,
            alt="GOLGI",
            title="Back to 3D view",
            classes="ml-4 mr-6 golgi-navbar-logo",
            style=("height: 32px; "
                    "width: auto; "
                    "display: block; "
                    "cursor: pointer;"),
            click=do_close_all_tabs,
        )
        # ----- Project name + last-saved chip -----
        with html.Div(
            v_show=("has_active_project",),
            classes="golgi-navbar-project",
        ):
            html.Button(
                "{{ current_project_name }}",
                type="button",
                title="Edit project details",
                classes="golgi-navbar-project-name",
                click=(
                    "detail_project = projects_list.find("
                    "  p => p.dir === current_project_dir"
                    ") || {"
                    "  name: current_project_name, "
                    "  dir: current_project_dir,"
                    "  labels: [],"
                    "  thumbnail_data_uri: ''"
                    "}; "
                    "detail_dialog_source = 'navbar'; "
                    "show_detail_dialog = true"
                ),
            )

        # File menu — Import / Save / Close project.
        with v3.VBtn(
            "File",
            prepend_icon="mdi-folder-outline",
            append_icon="mdi-menu-down",
            variant="text",
            style="color: #1f2024;",
            classes=(
                "[{'golgi-tab-active': show_import_stepper}]",
            ),
            v_show=("has_active_project",),
        ):
            with v3.VMenu(
                v_model=("show_file_menu",),
                activator="parent",
                close_on_content_click=True,
                location="bottom start",
                offset="8",
            ):
                with v3.VList(
                    density="compact",
                    classes="golgi-file-menu",
                ):
                    v3.VListItem(
                        title="Import Nerve",
                        prepend_icon="mdi-folder-open",
                        click=(
                            "show_import_stepper = "
                            "!show_import_stepper"
                        ),
                        classes=(
                            "[{'golgi-file-menu-active': "
                            "show_import_stepper}]",
                        ),
                    )
                    # V1 Phase A.3b — temporary entry point for
                    # the µCT segmentation dialog. Phase D will
                    # move this into the Import Nerve stepper as
                    # a third nerve-source tile (Analytical / A1
                    # Atlas / µCT segment); until then a top-
                    # level File menu entry gives the user
                    # something clickable to validate the
                    # dialog UX.
                    if do_open_segment_uct_dialog is not None:
                        v3.VListItem(
                            title="Segment µCT slice…",
                            prepend_icon="mdi-image-filter-center-focus",
                            click=do_open_segment_uct_dialog,
                            classes=(
                                "[{'golgi-file-menu-active': "
                                "show_segment_uct_dialog}]",
                            ),
                        )
                    # M47 — sibling entry for "I already have
                    # the masks" workflow. Imports a 4-TIFF
                    # bundle (slide + 3 masks), extrudes a
                    # nerve volume directly without SAM2.
                    if do_open_bundle_import_dialog is not None:
                        v3.VListItem(
                            title=(
                                "Import histology bundle…"
                            ),
                            prepend_icon=(
                                "mdi-folder-image"
                            ),
                            click=(
                                do_open_bundle_import_dialog
                            ),
                            classes=(
                                "[{'golgi-file-menu-active': "
                                "show_bundle_import_dialog}]",
                            ),
                        )
                    v3.VListItem(
                        title="Save",
                        prepend_icon="mdi-content-save-outline",
                        click=do_save_project,
                    )
                    v3.VDivider()
                    # F2.2 — study export / import. Sit
                    # between Save and Close so they share the
                    # File-menu's "project-scoped action" half.
                    # Items are hidden (rather than disabled)
                    # when their handler isn't wired so legacy
                    # builds without F2.2 still get a clean
                    # menu.
                    if do_export_study is not None:
                        v3.VListItem(
                            title="Export study…",
                            prepend_icon="mdi-package-up",
                            click=do_export_study,
                        )
                    if do_open_import_study_dialog is not None:
                        v3.VListItem(
                            title="Import study…",
                            prepend_icon="mdi-package-down",
                            click=do_open_import_study_dialog,
                        )
                    v3.VDivider()
                    v3.VListItem(
                        title="Close project",
                        prepend_icon="mdi-folder-remove",
                        click=do_show_close_dialog,
                    )
        # F3.2-M2.1b — Designs is gated on `has_fibers`. The
        # stepper owns the load → endo → fibers → muscle setup;
        # designs can only be placed once trajectories exist
        # (so the user can preview the fibers that the cuff will
        # eventually stimulate). Tooltip explains the gate.
        with v3.VTooltip(
            location="bottom",
            disabled=("has_fibers",),
        ):
            with v3.Template(v_slot_activator=("{ props }",)):
                with html.Div(v_bind="props"):
                    v3.VBtn(
                        "Designs",
                        prepend_icon="mdi-lightning-bolt",
                        variant="text",
                        click="show_cuff = !show_cuff",
                        style="color: #1f2024;",
                        classes=(
                            "[{'golgi-tab-active': show_cuff}]",
                        ),
                        v_show=("has_active_project",),
                        disabled=("!has_fibers",),
                    )
            html.Span(
                "Complete the import stepper first "
                "(File → Import Nerve → Load + Fibers)."
            )
        # F3.2-M2.1b — Mesh is gated on at least one design
        # existing. The Mesh drawer used to host nerve-level
        # tessellation knobs (now in stepper Step 2/4) — what
        # remains is per-design build affordances (M2.1c will
        # finish the drawer cleanup; gating happens here).
        with v3.VTooltip(
            location="bottom",
            disabled=("designs && designs.length > 0",),
        ):
            with v3.Template(v_slot_activator=("{ props }",)):
                with html.Div(v_bind="props"):
                    v3.VBtn(
                        "Mesh",
                        prepend_icon="mdi-grid",
                        variant="text",
                        click="show_mesh = !show_mesh",
                        style="color: #1f2024;",
                        classes=(
                            "[{'golgi-tab-active': show_mesh}]",
                        ),
                        v_show=("has_active_project",),
                        disabled=(
                            "!(designs && designs.length > 0)",
                        ),
                    )
            html.Span(
                "Add at least one cuff design first."
            )
        # F3.2-M2.1d — Materials enabled once a mesh exists.
        # Materials are per-region σ values; they only make
        # sense after the regions themselves have been meshed.
        with v3.VTooltip(
            location="bottom",
            disabled=("has_mesh",),
        ):
            with v3.Template(v_slot_activator=("{ props }",)):
                with html.Div(v_bind="props"):
                    v3.VBtn(
                        "Materials",
                        prepend_icon="mdi-test-tube-empty",
                        variant="text",
                        click="show_sigma = !show_sigma",
                        style="color: #1f2024;",
                        classes=(
                            "[{'golgi-tab-active': show_sigma}]",
                        ),
                        v_show=("has_active_project",),
                        disabled=("!has_mesh",),
                    )
            html.Span(
                "Build a mesh first."
            )
        # F3.2-M2.1b — Fiber Trajectories tab dropped. Trajectory
        # generation, params, and the branch-rename + summary UI
        # all live in the import stepper now (Step 3). Re-open
        # the stepper via FILE → Import Nerve to revisit.

        # Simulate umbrella menu — FEM / single fiber / population.
        # F3.2-M2.1d — disabled until materials have been
        # confirmed (`sigma_committed` flips true when the user
        # clicks Update in the Materials drawer). Children:
        #   - FEM: `!sigma_committed` (same as umbrella).
        #   - Single fiber / Population / Sweep: `!has_fem`.
        #   - Compare: needs >=2 FEM-solved configs.
        with v3.VTooltip(
            location="bottom",
            disabled=("sigma_committed",),
        ):
            with v3.Template(v_slot_activator=("{ props }",)):
                with html.Div(v_bind="props"):
                    with v3.VBtn(
                        "Simulate",
                        prepend_icon="mdi-flash-outline",
                        append_icon="mdi-menu-down",
                        variant="text",
                        style="color: #1f2024;",
                        classes=(
                            "[{'golgi-tab-active': "
                            "show_solve || show_fiber "
                            "|| show_pop}]",
                        ),
                        v_show=("has_active_project",),
                        disabled=("!sigma_committed",),
                    ):
                        with v3.VMenu(
                            v_model=("show_sim_menu",),
                            activator="parent",
                            close_on_content_click=True,
                            location="bottom start",
                            offset="8",
                        ):
                            with v3.VList(
                                density="compact",
                                classes="golgi-sim-menu",
                            ):
                                v3.VListItem(
                                    title=(
                                        "Extracellular field (FEM)"
                                    ),
                                    prepend_icon=(
                                        "mdi-function-variant"
                                    ),
                                    click=(
                                        "show_solve = !show_solve"
                                    ),
                                    disabled=(
                                        "!sigma_committed",
                                    ),
                                    classes=(
                                        "[{'golgi-sim-menu-active': "
                                        "show_solve}]",
                                    ),
                                )
                                v3.VListItem(
                                    title="Single fiber",
                                    prepend_icon=(
                                        "mdi-chart-bell-curve"
                                    ),
                                    click=(
                                        "show_fiber = !show_fiber"
                                    ),
                                    disabled=("!has_fem",),
                                    classes=(
                                        "[{'golgi-sim-menu-active': "
                                        "show_fiber}]",
                                    ),
                                )
                                v3.VListItem(
                                    title="Fiber population",
                                    prepend_icon=(
                                        "mdi-chart-scatter-plot"
                                    ),
                                    click=(
                                        "show_pop = !show_pop"
                                    ),
                                    disabled=("!has_fem",),
                                    classes=(
                                        "[{'golgi-sim-menu-active': "
                                        "show_pop}]",
                                    ),
                                )
                                # F2.1.c — Parameter sweep /
                                # threshold finder.
                                v3.VListItem(
                                    title=(
                                        "Sweep (recruitment / "
                                        "threshold)"
                                    ),
                                    prepend_icon=(
                                        "mdi-chart-line-variant"
                                    ),
                                    click=(
                                        "show_sweep = !show_sweep"
                                    ),
                                    disabled=("!has_fem",),
                                    classes=(
                                        "[{'golgi-sim-menu-active': "
                                        "show_sweep}]",
                                    ),
                                )
                                # F3.2e — Compare view: overlay
                                # solved configs side-by-side
                                # (Vₑ axes, slice heatmaps).
                                # Disabled until ≥2 configs have
                                # FEM outputs on disk.
                                v3.VListItem(
                                    title=(
                                        "Compare configurations"
                                    ),
                                    prepend_icon="mdi-compare",
                                    click=(
                                        "show_compare = "
                                        "!show_compare"
                                    ),
                                    disabled=(
                                        "!(fem_configs "
                                        "  && fem_configs.length "
                                        ">= 2)",
                                    ),
                                    classes=(
                                        "[{'golgi-sim-menu-active': "
                                        "show_compare}]",
                                    ),
                                )
            html.Span(
                "Open Materials and click Update first."
            )

        # F2.3.b/c — unified Export umbrella menu. Two children:
        # "Export figures" (toggles the Bulk Exports drawer) and
        # "Export reports" (opens the Generate Report dialog).
        # Top-level entry stays active while EITHER child is open
        # so the user can see at-a-glance where they are in the
        # export workflow.
        # F3.2-M2.1d — disabled until at least one nerve-level
        # simulation has run successfully (single-fiber sim →
        # `has_fiber_sim`; population sim → `pop_sim_done`).
        # Until then there's no real output to export anyway.
        with v3.VTooltip(
            location="bottom",
            disabled=(
                "has_fiber_sim || pop_sim_done",
            ),
        ):
          with v3.Template(v_slot_activator=("{ props }",)):
            with html.Div(v_bind="props"):
              with v3.VBtn(
                "Export",
                prepend_icon="mdi-download",
                append_icon="mdi-menu-down",
                variant="text",
                style="color: #1f2024;",
                classes=(
                    "[{'golgi-tab-active': show_exports "
                    "|| show_generate_report_dialog}]",
                ),
                v_show=("has_active_project",),
                disabled=(
                    "!(has_fiber_sim || pop_sim_done)",
                ),
              ):
                with v3.VMenu(
                    v_model=("show_export_menu",),
                    activator="parent",
                    close_on_content_click=True,
                    location="bottom start",
                    offset="8",
                ):
                    with v3.VList(
                        density="compact",
                        classes="golgi-export-menu",
                    ):
                        v3.VListItem(
                            title="Export figures",
                            prepend_icon=(
                                "mdi-download-multiple"
                            ),
                            click=(
                                "show_exports = !show_exports"
                            ),
                            classes=(
                                "[{'golgi-sim-menu-active': "
                                "show_exports}]",
                            ),
                        )
                        if (
                            do_open_generate_report_dialog
                            is not None
                        ):
                            v3.VListItem(
                                title="Export report",
                                prepend_icon=(
                                    "mdi-file-document-outline"
                                ),
                                click=(
                                    do_open_generate_report_dialog
                                ),
                                classes=(
                                    "[{'golgi-sim-menu-active': "
                                    "show_generate_report_dialog}]",
                                ),
                            )
          html.Span(
              "Run at least one nerve simulation "
              "(Single fiber / Population) first."
          )

        v3.VSpacer()

        html.Span(
            "saved {{ current_project_modified }}",
            classes="golgi-navbar-project-saved",
            style="margin-right: 12px;",
            v_show=(
                "has_active_project "
                "&& current_project_modified",
            ),
        )

        # ----- User avatar chip + dropdown -----
        with html.Button(
            type="button",
            classes="golgi-navbar-userchip",
            style="margin-right: 12px;",
            v_if=(
                "authenticated && "
                "view_mode === 'workspace'",
            ),
            title=(
                "current_user_username "
                "|| current_user_email "
                "|| 'User'",
            ),
        ):
            html.Span(
                "{{ current_user_username "
                "|| current_user_email "
                "|| 'user' }}",
                classes=(
                    "golgi-navbar-userchip-name"
                ),
            )
            html.Img(
                src=("current_user_avatar",),
                v_show=("current_user_avatar",),
                alt="user avatar",
                classes="golgi-navbar-userchip-img",
            )
            with v3.VMenu(
                v_model=("show_user_menu",),
                activator="parent",
                close_on_content_click=False,
                location="bottom end",
                offset="8",
            ):
                with html.Div(classes="golgi-user-menu-card"):
                    with html.Div(
                        classes="golgi-user-menu-header",
                    ):
                        html.Img(
                            src=("current_user_avatar",),
                            v_show=("current_user_avatar",),
                            alt="",
                            classes=(
                                "golgi-user-menu-avatar-img"
                            ),
                        )
                        html.Div(
                            "{{ ((current_user_first_name "
                            "|| '') + ' ' + "
                            "(current_user_last_name || ''))"
                            ".trim() "
                            "|| current_user_username "
                            "|| current_user_email }}",
                            classes=(
                                "golgi-user-menu-name"
                            ),
                        )
                        html.Div(
                            "{{ current_user_email }}",
                            classes=(
                                "golgi-user-menu-email"
                            ),
                        )
                    with html.Div(
                        classes="golgi-user-menu-items",
                    ):
                        with html.Button(
                            type="button",
                            classes=(
                                "golgi-user-menu-item"
                            ),
                            click=do_open_profile_dialog,
                        ):
                            v3.VIcon(
                                "mdi-account-cog",
                                size="20",
                                classes=(
                                    "golgi-user-menu-icon"
                                ),
                            )
                            html.Span("Profile Settings")
                    with html.Div(
                        classes=(
                            "golgi-user-menu-footer"
                        ),
                    ):
                        html.Button(
                            "Sign out",
                            type="button",
                            classes=(
                                "golgi-user-menu-signout"
                            ),
                            click=(
                                "show_user_menu = false; "
                                "show_logout_dialog = true"
                            ),
                        )
