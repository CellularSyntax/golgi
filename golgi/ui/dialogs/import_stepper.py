# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""F3.2-M2.1 — Import-nerve stepper wizard.

A modal v-stepper that walks the user through the nerve-level
setup: load the nerve, set endoneurium / epineurium params,
generate fiber trajectories, set muscle bbox. Replaces the four
separate sidebar drawers that used to host these steps so the
flow is enforced top-to-bottom.

Bound to state.show_import_stepper (dialog open/close) and
state.import_stepper_step (current step 1-4). Each step pre-
populates from current state vars, so re-opening the dialog
acts as an edit mode for any of the four panels.

Step actions invoke headless functions (`do_load_geometry`,
`do_generate_fibers`) directly; the rest of each step just
binds form controls to existing state vars. The mesh-build
step is NOT in this wizard — meshing happens per-design in
the Mesh tab after designs are placed.
"""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

try:                                                # pragma: no cover
    from trame.widgets import plotly as twp
except ImportError:                                 # pragma: no cover
    twp = None

from ..drawers._helpers import param_row_with_info, slider_row


def _step_header_label(num: int, label: str) -> str:
    return f"{num}. {label}"


def _step_heading_with_info(title: str, info: str) -> None:
    """In-body step heading: H4 title + info-icon tooltip carrying
    the descriptive text. Mirrors the same pattern used by the
    Mesh-drawer title row so the visual language stays consistent
    across the app."""
    with html.Div(
        classes="d-flex align-center mb-3",
        style="gap: 2px;",
    ):
        html.H4(
            title,
            classes="text-subtitle-1 mb-0",
            style="color: #1f2024; font-weight: 600;",
        )
        with v3.VTooltip(
            location="bottom",
            max_width=380,
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
                    classes="ml-1",
                ):
                    v3.VIcon(
                        "mdi-information-outline",
                        size="16",
                        color="grey-darken-1",
                    )
            html.Span(info)


def _advanced_toggle(model: str, label: str = "Advanced") -> None:
    """A right-aligned chevron-prefixed toggle row used to open /
    close the "Advanced" subsection inside a stepper step. Style
    matches `golgi-legend-supheader` so the visual cue ("things
    fold here") reads the same as the legend."""
    with html.Div(
        classes="d-flex align-center mt-4 mb-2",
        style=(
            "cursor: pointer; user-select: none; "
            "color: #888a90; letter-spacing: 0.04em; "
            "text-transform: uppercase; font-size: 10px;"
        ),
        click=f"{model} = !{model}",
    ):
        html.Span(
            "{{ " + model + " ? '▾' : '▸' }}",
            style="margin-right: 6px;",
        )
        html.Span(label)


def render(
    *,
    do_stepper_next: Callable,
    do_stepper_action: Callable,
    do_start_branch_rename: Callable | None = None,
    do_apply_branch_rename: Callable | None = None,
    do_cancel_branch_rename: Callable | None = None,
    export_btn: Callable | None = None,
    do_select_source_stl: Callable | None = None,
    do_select_source_uct_bundle: Callable | None = None,
    do_select_source_histo_bundle: Callable | None = None,
    do_delete_source_file: Callable | None = None,
    do_delete_epi_file: Callable | None = None,
    do_delete_uct_bundle: Callable | None = None,
    do_delete_histo_bundle: Callable | None = None,
) -> None:
    """Render the import-nerve stepper dialog. Wired via
    `show_import_stepper` state var; the navbar's FILE > Import
    Nerve item flips that flag.

    The single `do_stepper_next` callback drives every step's
    primary action: it runs the relevant async (load nerve /
    generate fibers) and advances the step on success. The
    global busy lightbox covers the dialog while either action
    is in flight, so no inline spinner needed — but we still
    disable the dialog buttons via `:disabled="busy"` to prevent
    double-clicks before the lightbox mounts."""
    with v3.VDialog(
        v_model=("show_import_stepper",),
        max_width=650,
        # F3.2-M3: dock the wizard to the LEFT of the viewport
        # and use a transparent scrim. The user walks through the
        # steps while the live 3D scene stays visible on the
        # right — load nerve → nerve appears; tweak muscle pads
        # → bbox preview updates; generate fibers → tubes mount.
        # See `.golgi-stepper-dialog` rules in golgi.css.
        scrim="transparent",
        content_class="golgi-stepper-dialog",
        # `persistent=True` — never close on outside click or Esc.
        # The wizard has its own close affordances (the X icon in
        # the title bar + the "Done" button on step 4), and the
        # transparent scrim + click-through CSS make outside
        # clicks reach the viewport instead, so an accidental
        # backdrop click would otherwise dismiss the wizard
        # without the user realising.
        persistent=True,
        scrollable=True,
    ):
        with v3.VCard():
            # Title bar with close button.
            with v3.VCardTitle(
                classes=(
                    "d-flex align-center "
                    "justify-space-between pa-4"
                ),
            ):
                html.Span(
                    "Set up nerve",
                    classes="text-h6",
                )
                v3.VBtn(
                    icon="mdi-close",
                    size="small",
                    variant="text",
                    disabled=("busy",),
                    click=(
                        "show_import_stepper = false"
                    ),
                )
            with v3.VCardText(classes="pa-0"):
                with v3.VStepper(
                    v_model=("import_stepper_step",),
                    flat=True,
                    hide_actions=True,
                    # editable=True lets users click any header
                    # to revisit a step (useful for tweaking
                    # params on a second pass); freezes while
                    # an action is in flight so the user can't
                    # mutate state mid-build.
                    editable=("!busy",),
                ):
                    # ---- Header (step pills) ----
                    with v3.VStepperHeader():
                        v3.VStepperItem(
                            title=_step_header_label(
                                1, "Load nerve",
                            ),
                            value="1",
                            complete=("has_geometry",),
                            editable=("!busy",),
                        )
                        v3.VDivider()
                        v3.VStepperItem(
                            title=_step_header_label(
                                2, "Endoneurium",
                            ),
                            value="2",
                            # Has no boolean completion gate
                            # (params always have defaults); just
                            # show it as the user steps through.
                            # Disabled in bundle mode — the
                            # bundle already carries epi +
                            # fascicle surfaces, so the
                            # inward-offset workflow this step
                            # drives doesn't apply. (We also
                            # short-circuit do_stepper_next so
                            # the user can't land here, but
                            # greying the header makes the skip
                            # visible.)
                            editable=(
                                "!busy && "
                                "import_source_type "
                                "!== 'uct_bundle' "
                                "&& import_source_type "
                                "!== 'histo_bundle'",
                            ),
                        )
                        v3.VDivider()
                        v3.VStepperItem(
                            title=_step_header_label(
                                3, "Fibers",
                            ),
                            value="3",
                            complete=("has_fibers",),
                            editable=("!busy",),
                        )
                        v3.VDivider()
                        v3.VStepperItem(
                            title=_step_header_label(
                                4, "Muscle",
                            ),
                            value="4",
                            editable=("!busy",),
                        )

                    # F3.2-M3: the wizard is docked to the LEFT
                    # edge with a transparent scrim (see the
                    # `.golgi-stepper-dialog` CSS rules), so the
                    # main 3D viewport stays fully visible +
                    # interactive on the right while the user
                    # walks through the steps. No embedded mini
                    # viewport — one less VTK view to keep in
                    # sync, and the user gets the full-resolution
                    # workspace plotter for free.

                    # ---- Window (step content) ----
                    with v3.VStepperWindow():
                        # ===== Step 1: Load nerve =====
                        with v3.VStepperWindowItem(value="1"):
                            with html.Div(classes="pa-4"):
                                _step_heading_with_info(
                                    "Load nerve",
                                    "Pick a closed nerve surface "
                                    "(STL / NAS / OBJ) and the "
                                    "scale factor to convert its "
                                    "units to metres, OR pick a "
                                    "Golgi µCT reconstruction "
                                    "bundle saved by the Segment-"
                                    "µCT dialog (epi + per-"
                                    "fascicle endoneurium "
                                    "surfaces, mm-scaled).",
                                )
                                # ---- Source-type tile picker ----
                                # Two clickable tiles set
                                # `import_source_type` directly via
                                # an inline Vue expression so the
                                # selection flip is synchronous and
                                # the downstream v_show panels swap
                                # without a server round-trip. The
                                # bundle tile is greyed out (and the
                                # click is a no-op) when no bundles
                                # exist in the current project — the
                                # `:disabled` analogue is the inline
                                # check on `uct_bundle_items.length`.
                                with html.Div(
                                    classes="d-flex flex-row mb-3",
                                    style="gap: 8px;",
                                ):
                                    with html.Div(
                                        # M47 fix — JS-string-concat
                                        # :style binding (the previous
                                        # version) did not re-render on
                                        # `import_source_type` change in
                                        # this Vuetify-3 / trame-client
                                        # combo: the expression
                                        # evaluated at mount only,
                                        # tracking no deps so Vue never
                                        # re-ran it. Vue 3 OBJECT-syntax
                                        # :style is the supported
                                        # reactive form; each key's
                                        # expression is its own
                                        # tracked dep. Static layout
                                        # bits move to `classes=` /
                                        # plain `style=` so only the
                                        # state-dependent props are in
                                        # the reactive binding.
                                        classes=(
                                            "d-flex align-center "
                                            "tile-source-pick"
                                        ),
                                        style=(
                                            "{"
                                            "border: '2px solid '"
                                            " + (import_source_type"
                                            " === 'stl' "
                                            "? '#1976d2' : '#ccc'),"
                                            "background:"
                                            " import_source_type"
                                            " === 'stl' "
                                            "? '#e3f2fd' : 'white'"
                                            "}",
                                        ),
                                        click=(
                                            do_select_source_stl
                                        ),
                                    ):
                                        v3.VIcon(
                                            "mdi-file-outline",
                                            size="22",
                                            color="primary",
                                        )
                                        with html.Div():
                                            html.Div(
                                                "STL surface",
                                                style=(
                                                    "font-weight: "
                                                    "600; "
                                                    "font-size: "
                                                    "13px;"
                                                ),
                                            )
                                            html.Div(
                                                "Single closed "
                                                "mesh from disk "
                                                "or upload",
                                                style=(
                                                    "font-size: "
                                                    "10px; "
                                                    "color: #666;"
                                                ),
                                            )
                                    with html.Div(
                                        # M47 fix — same object-syntax
                                        # :style as the STL tile above
                                        # (see that comment for why
                                        # the previous string-concat
                                        # version did not re-render).
                                        # Opacity + cursor are also in
                                        # the reactive binding because
                                        # they depend on
                                        # uct_bundle_items.length.
                                        classes=(
                                            "d-flex align-center "
                                            "tile-source-pick"
                                        ),
                                        style=(
                                            "{"
                                            "border: '2px solid '"
                                            " + (import_source_type"
                                            " === 'uct_bundle' "
                                            "? '#1976d2' : '#ccc'),"
                                            "background:"
                                            " import_source_type"
                                            " === 'uct_bundle' "
                                            "? '#e3f2fd' : 'white',"
                                            "opacity:"
                                            " uct_bundle_items"
                                            ".length === 0 "
                                            "? 0.45 : 1,"
                                            "cursor:"
                                            " uct_bundle_items"
                                            ".length === 0 "
                                            "? 'not-allowed'"
                                            " : 'pointer'"
                                            "}",
                                        ),
                                        click=(
                                            do_select_source_uct_bundle
                                        ),
                                    ):
                                        v3.VIcon(
                                            "mdi-image-multiple-"
                                            "outline",
                                            size="22",
                                            color="primary",
                                        )
                                        with html.Div():
                                            html.Div(
                                                "Golgi µCT bundle",
                                                style=(
                                                    "font-weight: "
                                                    "600; "
                                                    "font-size: "
                                                    "13px;"
                                                ),
                                            )
                                            html.Div(
                                                "{{ "
                                                "uct_bundle_items"
                                                ".length }} bundle"
                                                "{{ "
                                                "uct_bundle_items"
                                                ".length === 1 "
                                                " ? '' : 's' }} "
                                                "available",
                                                style=(
                                                    "font-size: "
                                                    "10px; "
                                                    "color: #666;"
                                                ),
                                            )
                                    # M47 — third tile: histology
                                    # bundle. Source dir is
                                    # <project>/histology/nerve_3d/
                                    # (separate from µCT bundles).
                                    # Same Vue 3 object-syntax
                                    # :style as the other two tiles.
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center "
                                            "tile-source-pick"
                                        ),
                                        style=(
                                            "{"
                                            "border: '2px solid '"
                                            " + (import_source_type"
                                            " === 'histo_bundle' "
                                            "? '#1976d2' : '#ccc'),"
                                            "background:"
                                            " import_source_type"
                                            " === 'histo_bundle' "
                                            "? '#e3f2fd' : 'white',"
                                            "opacity:"
                                            " histo_bundle_items"
                                            ".length === 0 "
                                            "? 0.45 : 1,"
                                            "cursor:"
                                            " histo_bundle_items"
                                            ".length === 0 "
                                            "? 'not-allowed'"
                                            " : 'pointer'"
                                            "}",
                                        ),
                                        click=(
                                            do_select_source_histo_bundle
                                        ),
                                    ):
                                        v3.VIcon(
                                            "mdi-microscope",
                                            size="22",
                                            color="primary",
                                        )
                                        with html.Div():
                                            html.Div(
                                                "Histology bundle",
                                                style=(
                                                    "font-weight: "
                                                    "600; "
                                                    "font-size: "
                                                    "13px;"
                                                ),
                                            )
                                            html.Div(
                                                "{{ "
                                                "histo_bundle_items"
                                                ".length }} "
                                                "bundle{{ "
                                                "histo_bundle_items"
                                                ".length === 1 "
                                                " ? '' : 's' }} "
                                                "available",
                                                style=(
                                                    "font-size: "
                                                    "10px; "
                                                    "color: #666;"
                                                ),
                                            )

                                # ---- µCT bundle picker (shown
                                # only when bundle tile is active).
                                # M47 fix — v_show binds to a single
                                # boolean state var (computed by the
                                # @state.change watcher on
                                # `import_source_type`), not a complex
                                # JS expression. The expression form
                                # was not reactive in this build.
                                with html.Div(
                                    v_show=(
                                        "show_picker_uct_bundle",
                                    ),
                                ):
                                    html.Div(
                                        "Bundle",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555;"
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center "
                                            "mb-3 mt-1"
                                        ),
                                        style="gap: 6px;",
                                    ):
                                        v3.VSelect(
                                            v_model=(
                                                "selected_uct"
                                                "_bundle",
                                            ),
                                            items=(
                                                "uct_bundle_items",
                                            ),
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style="flex: 1;",
                                        )
                                        if (do_delete_uct_bundle
                                                is not None):
                                            v3.VBtn(
                                                icon=(
                                                    "mdi-delete-"
                                                    "outline"
                                                ),
                                                size="small",
                                                variant="text",
                                                color="grey-darken-1",
                                                disabled=(
                                                    "!selected_uct"
                                                    "_bundle",
                                                ),
                                                click=(
                                                    do_delete_uct_bundle
                                                ),
                                            )
                                    # Summary card for the picked
                                    # bundle — pulls the .summary
                                    # field of the matched item out
                                    # of `uct_bundle_items`.
                                    html.Div(
                                        "{{ "
                                        "(uct_bundle_items.find("
                                        "b => b.value === "
                                        "selected_uct_bundle) || "
                                        "{}).summary || "
                                        "'Pick a bundle above.'"
                                        " }}",
                                        style=(
                                            "background: #f7f8fa; "
                                            "border: 1px solid "
                                            "#e6e6e8; "
                                            "border-radius: 4px; "
                                            "padding: 8px 10px; "
                                            "font-size: 11px; "
                                            "color: #555;"
                                        ),
                                    )

                                # ---- Histology bundle picker (M47).
                                # Shown only when the histology tile
                                # is active. Lists `<project>/
                                # histology/nerve_3d/*/manifest.json`.
                                # Same single-boolean v_show fix as
                                # the µCT picker above.
                                with html.Div(
                                    v_show=(
                                        "show_picker_histo_bundle",
                                    ),
                                ):
                                    html.Div(
                                        "Bundle",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555;"
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center "
                                            "mb-3 mt-1"
                                        ),
                                        style="gap: 6px;",
                                    ):
                                        v3.VSelect(
                                            v_model=(
                                                "selected_histo"
                                                "_bundle",
                                            ),
                                            items=(
                                                "histo_bundle"
                                                "_items",
                                            ),
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style="flex: 1;",
                                        )
                                        if (do_delete_histo_bundle
                                                is not None):
                                            v3.VBtn(
                                                icon=(
                                                    "mdi-delete-"
                                                    "outline"
                                                ),
                                                size="small",
                                                variant="text",
                                                color="grey-darken-1",
                                                disabled=(
                                                    "!selected"
                                                    "_histo_bundle",
                                                ),
                                                click=(
                                                    do_delete_histo_bundle
                                                ),
                                            )
                                    html.Div(
                                        "{{ "
                                        "(histo_bundle_items.find("
                                        "b => b.value === "
                                        "selected_histo_bundle) || "
                                        "{}).summary || "
                                        "'Pick a bundle above.'"
                                        " }}",
                                        style=(
                                            "background: #f7f8fa; "
                                            "border: 1px solid "
                                            "#e6e6e8; "
                                            "border-radius: 4px; "
                                            "padding: 8px 10px; "
                                            "font-size: 11px; "
                                            "color: #555;"
                                        ),
                                    )

                                # ---- STL flow (existing pickers)
                                # ---- Pick an existing file ----
                                # M47 fix — single-boolean v_show.
                                with html.Div(
                                    v_show=(
                                        "show_picker_stl",
                                    ),
                                ):
                                    html.Div(
                                        "Source file",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555;"
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center "
                                            "mb-3 mt-1"
                                        ),
                                        style="gap: 6px;",
                                    ):
                                        v3.VSelect(
                                            v_model=(
                                                "selected_file",
                                            ),
                                            items=("data_files",),
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style="flex: 1;",
                                        )
                                        # Per-item delete: removes the
                                        # selected uploaded surface from
                                        # disk (bundled examples are
                                        # refused server-side).
                                        if (do_delete_source_file
                                                is not None):
                                            v3.VBtn(
                                                icon=(
                                                    "mdi-delete-"
                                                    "outline"
                                                ),
                                                size="small",
                                                variant="text",
                                                color="grey-darken-1",
                                                disabled=(
                                                    "!selected_file",
                                                ),
                                                click=(
                                                    do_delete_source_file
                                                ),
                                            )
                                # All STL-flow knobs (upload, OR
                                # divider, scaling, decimation)
                                # are gated together on the source-
                                # type tile. The bundle path needs
                                # none of them — coords land in mm,
                                # scale is metadata-set, and decim
                                # is owned by the segmentation
                                # marching-cubes step, not the
                                # importer.
                                with html.Div(
                                    v_show=("show_picker_stl",),
                                ):
                                    # ---- OR divider ----
                                    # Horizontal line + centred "OR"
                                    # chip so the alternative-upload
                                    # affordance reads as a peer of the
                                    # picker, not a buried fallback.
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center "
                                            "my-3"
                                        ),
                                    ):
                                        html.Div(
                                            style=(
                                                "flex: 1 1 auto; "
                                                "border-top: 1px "
                                                "solid #e3e3e6;"
                                            ),
                                        )
                                        html.Span(
                                            "OR",
                                            style=(
                                                "padding: 0 12px; "
                                                "color: #888a90; "
                                                "font-size: 11px; "
                                                "letter-spacing: "
                                                "0.08em; "
                                                "font-weight: 600;"
                                            ),
                                        )
                                        html.Div(
                                            style=(
                                                "flex: 1 1 auto; "
                                                "border-top: 1px "
                                                "solid #e3e3e6;"
                                            ),
                                        )
                                    # ---- Upload a new file ----
                                    v3.VFileInput(
                                        v_model=("upload_file",),
                                        label=(
                                            "Upload a new file"
                                        ),
                                        prepend_icon=(
                                            "mdi-upload"
                                        ),
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        show_size=True,
                                        classes="mb-1",
                                    )
                                    html.Div(
                                        "{{ upload_info }}",
                                        v_show=("upload_info",),
                                        style=(
                                            "color: #666; "
                                            "font-size: 11px; "
                                            "margin-top: 4px;"
                                        ),
                                        classes="mb-3",
                                    )
                                    # ---- Epineurium surface (optional)
                                    # Supply a SEPARATE outer surface
                                    # to build a real multi-region
                                    # nerve: the surface picked above
                                    # becomes the endoneurium and this
                                    # one the epineurium hull. golgi
                                    # then assembles the same multi-
                                    # domain (uct_bundle) geometry the
                                    # µCT / histology bundles produce —
                                    # exactly what fig 8's human
                                    # selectivity model needs (epi +
                                    # endo, no inward-offset shell).
                                    # Leave empty for a single-region
                                    # nerve (the legacy inward-offset
                                    # epi shell in step 2 still applies).
                                    v3.VDivider(classes="my-4")
                                    html.H4(
                                        "Epineurium surface (optional)",
                                        classes=(
                                            "text-subtitle-2 mb-2"
                                        ),
                                        style=(
                                            "color: #888a90; "
                                            "letter-spacing: 0.04em; "
                                            "text-transform: "
                                            "uppercase; "
                                            "font-size: 10px;"
                                        ),
                                    )
                                    html.Div(
                                        "Add an outer surface to build "
                                        "a real epi + endo model "
                                        "(the surface above is then "
                                        "treated as the endoneurium). "
                                        "Leave empty for a single-"
                                        "region nerve.",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; "
                                            "margin-bottom: 6px;"
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center "
                                            "mb-2"
                                        ),
                                        style="gap: 6px;",
                                    ):
                                        v3.VSelect(
                                            v_model=(
                                                "selected_epi_file",
                                            ),
                                            items=("data_files",),
                                            label=(
                                                "epineurium surface"
                                            ),
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            clearable=True,
                                            style="flex: 1;",
                                        )
                                        if (do_delete_epi_file
                                                is not None):
                                            v3.VBtn(
                                                icon=(
                                                    "mdi-delete-"
                                                    "outline"
                                                ),
                                                size="small",
                                                variant="text",
                                                color="grey-darken-1",
                                                disabled=(
                                                    "!selected_epi"
                                                    "_file",
                                                ),
                                                click=(
                                                    do_delete_epi_file
                                                ),
                                            )
                                    v3.VFileInput(
                                        v_model=("epi_upload_file",),
                                        label=(
                                            "or upload an epineurium "
                                            "surface"
                                        ),
                                        prepend_icon="mdi-upload",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        show_size=True,
                                        classes="mb-1",
                                    )
                                    # ---- Section break: file pickers
                                    # are done, now nerve-prep knobs.
                                    v3.VDivider(
                                        classes="my-4",
                                    )
                                    # ---- Scaling subsection ----
                                    # Sub-heading style mirrors the
                                    # "Epineurium shell" header in
                                    # Step 2 + the cuff-region header
                                    # in the Mesh drawer so the
                                    # visual language stays consistent.
                                    html.H4(
                                        "Scaling",
                                        classes=(
                                            "text-subtitle-2 mb-2"
                                        ),
                                        style=(
                                            "color: #888a90; "
                                            "letter-spacing: 0.04em; "
                                            "text-transform: "
                                            "uppercase; "
                                            "font-size: 10px;"
                                        ),
                                    )
                                    html.Div(
                                        "Unit scaling preset",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555;"
                                        ),
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
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        classes="mb-2 mt-1",
                                    )
                                    v3.VTextField(
                                        v_model=("scale_factor",),
                                        label=(
                                            "Scale factor "
                                            "(source units → m)"
                                        ),
                                        type="number",
                                        step="0.000001",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    # ---- Decimation subsection ----
                                    # F3.2-M3: moved here from Step 2
                                    # — it's a property of the
                                    # imported surface (TetGen input
                                    # complexity), not of the
                                    # endoneurium tessellation.
                                    html.H4(
                                        "Decimation",
                                        classes=(
                                            "text-subtitle-2 "
                                            "mt-3 mb-2"
                                        ),
                                        style=(
                                            "color: #888a90; "
                                            "letter-spacing: 0.04em; "
                                            "text-transform: "
                                            "uppercase; "
                                            "font-size: 10px;"
                                        ),
                                    )
                                    param_row_with_info(
                                        "decim_target_k",
                                        "Decimation target",
                                        "Target triangle count (in "
                                        "thousands) after decimating "
                                        "the imported nerve surface. "
                                        "Lower values give TetGen a "
                                        "simpler PLC; 50–80k is a "
                                        "good range for vagus nerve.",
                                        "k tris", 5, 5, 200,
                                    )
                                # Action runs from the Next button
                                # in the footer — no inline trigger
                                # so the user only ever drives the
                                # flow from one place.
                                html.Pre(
                                    "{{ geom_summary }}",
                                    v_show=("geom_summary",),
                                    style=(
                                        "color: #333; "
                                        "font-size: 11px; "
                                        "font-family: "
                                        "ui-monospace, Menlo, "
                                        "Consolas, monospace; "
                                        "background: #f7f7f8; "
                                        "border: 1px solid "
                                        "#e6e6e8; "
                                        "border-radius: 6px; "
                                        "padding: 8px; "
                                        "white-space: pre-wrap; "
                                        "margin-top: 8px;"
                                    ),
                                )
                                # Clearer unload affordance — when a
                                # nerve is already loaded, a one-click
                                # "start over" that opens the
                                # remove-geometry confirm dialog (the
                                # same destructive unload as the Import
                                # drawer's Remove button; clears the
                                # nerve + every derived artefact).
                                html.Button(
                                    "✕ Remove loaded nerve",
                                    type="button",
                                    v_show=("has_geometry",),
                                    classes=(
                                        "golgi-btn-secondary "
                                        "golgi-btn-sm mt-3"
                                    ),
                                    click=(
                                        "show_confirm_remove"
                                        "_geometry_dialog = true"
                                    ),
                                )
                                # ---- Surface quality (post-load)
                                # F3.2-M3: surface triangle quality
                                # histogram, built by
                                # `do_load_geometry` into
                                # `state.quality_hist_figure`.
                                # Shown only after a successful
                                # load so the user can sanity-
                                # check the imported mesh before
                                # moving on. Same RdYlGn colour
                                # semantics as the post-TetGen
                                # histogram in the Mesh drawer.
                                with html.Div(
                                    v_show=("has_geometry",),
                                ):
                                    html.H4(
                                        "Surface mesh quality",
                                        classes=(
                                            "text-subtitle-2 "
                                            "mt-4 mb-2"
                                        ),
                                        style=(
                                            "color: #888a90; "
                                            "letter-spacing: "
                                            "0.04em; "
                                            "text-transform: "
                                            "uppercase; "
                                            "font-size: 10px;"
                                        ),
                                    )
                                    with html.Div(
                                        style=(
                                            "width: 100%; "
                                            "max-width: 100%; "
                                            "height: 220px; "
                                            "display: block; "
                                            "border: 1px solid "
                                            "#e6e6e8; "
                                            "border-radius: 6px; "
                                            "background: white; "
                                            "position: relative;"
                                        ),
                                    ):
                                        if (export_btn is not None
                                                and twp is not None):
                                            export_btn(
                                                "mesh."
                                                "surface_quality"
                                                "_hist"
                                            )
                                        if twp is not None:
                                            twp.Figure(
                                                state_variable_name=(
                                                    "quality_hist_"
                                                    "figure"
                                                ),
                                                display_logo=False,
                                                display_mode_bar=(
                                                    True
                                                ),
                                            )

                        # ===== Step 2: Endoneurium =====
                        with v3.VStepperWindowItem(value="2"):
                            with html.Div(classes="pa-4"):
                                _step_heading_with_info(
                                    "Endoneurium",
                                    "Optional inward-offset "
                                    "epineurium shell around the "
                                    "endoneurium. The volume "
                                    "between the two surfaces is "
                                    "tagged as epineurium (tag 5) "
                                    "and gets its own σ at solve "
                                    "time. Mesh-size knobs for "
                                    "endoneurium + epineurium live "
                                    "in the Mesh drawer.",
                                )
                                # ---- STL flow: inward-offset
                                # epi-shell generator. Hidden when a
                                # µCT bundle is loaded — the bundle
                                # already carries an explicit epi
                                # surface (epi.stl) plus per-
                                # fascicle endoneurium volumes, so
                                # any inward-offset would just
                                # duplicate / fight the bundle's
                                # actual geometry.
                                # Read-only note when an explicit
                                # epineurium surface was supplied in
                                # step 1: the inward-offset shell is
                                # disabled (the epi is real, not
                                # offset-derived). Dedicated boolean
                                # state var because compound v_show
                                # expressions don't reliably
                                # re-evaluate in this trame / Vuetify
                                # build (see M47).
                                with html.Div(
                                    v_show=("show_stl_epi_note",),
                                    style=(
                                        "background: #f1f8e9; "
                                        "border: 1px solid "
                                        "#c5e1a5; "
                                        "border-radius: 6px; "
                                        "padding: 12px; "
                                        "font-size: 12px; "
                                        "color: #555;"
                                    ),
                                ):
                                    html.Div(
                                        "Epineurium taken from the "
                                        "imported surface — the "
                                        "inward-offset shell is "
                                        "disabled. The nerve is "
                                        "built as a multi-region "
                                        "epi + endo model.",
                                    )
                                with html.Div(
                                    v_show=("show_stl_offset",),
                                ):
                                    v3.VCheckbox(
                                        v_model=("use_epi",),
                                        label=(
                                            "Generate inward-offset "
                                            "epineurium shell"
                                        ),
                                        density="compact",
                                        hide_details=True,
                                        color="primary",
                                        classes="mb-1",
                                    )
                                    with html.Div(
                                        v_show=("use_epi",),
                                    ):
                                        param_row_with_info(
                                            "epi_thickness_um",
                                            "Shell thickness",
                                            "Inward radial offset "
                                            "from the nerve surface. "
                                            "The volume between the "
                                            "two surfaces is tagged "
                                            "as epineurium (tag 5).",
                                            "µm", 5, 10, 500,
                                        )

                                # ---- Bundle flow: read-only
                                # summary card. The bundle's
                                # `epi.stl` is the outer hull (tag
                                # 5) and each fascicle.stl is a
                                # subdomain (tag 1, same σ_endo
                                # for all for now — per-fascicle
                                # σ assignment is planned later).
                                # No knobs here; the geometry is
                                # baked in upstream.
                                with html.Div(
                                    v_show=(
                                        "show_picker_uct_bundle "
                                        "|| "
                                        "show_picker_histo_bundle",
                                    ),
                                    style=(
                                        "background: #f1f8e9; "
                                        "border: 1px solid "
                                        "#c5e1a5; "
                                        "border-radius: 6px; "
                                        "padding: 12px;"
                                    ),
                                ):
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center"
                                        ),
                                        style=(
                                            "gap: 8px; "
                                            "margin-bottom: 6px;"
                                        ),
                                    ):
                                        v3.VIcon(
                                            "mdi-check-circle",
                                            color="success",
                                            size="20",
                                        )
                                        html.Span(
                                            "Epineurium from "
                                            "µCT bundle",
                                            style=(
                                                "font-weight: 600; "
                                                "font-size: 13px;"
                                            ),
                                        )
                                    html.Div(
                                        "Outer hull from "
                                        "<code>epi.stl</code> · "
                                        "{{ uct_bundle_n_fasc }} "
                                        "fascicle endoneurium "
                                        "volume"
                                        "{{ uct_bundle_n_fasc "
                                        "=== 1 ? '' : 's' }}.",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555;"
                                        ),
                                    )
                                    html.Div(
                                        "All fascicles share "
                                        "σ_endo at solve time; "
                                        "per-fascicle σ is a "
                                        "future enhancement.",
                                        style=(
                                            "font-size: 10px; "
                                            "color: #888; "
                                            "margin-top: 6px;"
                                        ),
                                    )

                        # ===== Step 3: Fiber trajectories =====
                        with v3.VStepperWindowItem(value="3"):
                            with html.Div(classes="pa-4"):
                                _step_heading_with_info(
                                    "Fibers",
                                    "Generate fiber trajectories "
                                    "along the nerve from one "
                                    "Laplace solve. The same "
                                    "trajectories are shared "
                                    "across all designs — fibers "
                                    "are a property of the "
                                    "imported nerve, not of any "
                                    "cuff.",
                                )
                                html.Div(
                                    "{{ fiber_status }}",
                                    v_show=("fiber_status",),
                                    style=(
                                        "font-size: 11px; "
                                        "color: #444; "
                                        "background: #f6f6f7; "
                                        "padding: 6px 10px; "
                                        "border-radius: 4px; "
                                        "margin-bottom: 12px; "
                                        "font-family: monospace;"
                                    ),
                                )
                                # ---- Always-visible knobs ----
                                # Number of seeds, auto-detect-
                                # branches toggle, method picker.
                                # Everything else hides behind
                                # Advanced.
                                slider_row(
                                    "n_fibers",
                                    "Number of fiber seeds",
                                    1, 500, 1, "toFixed(0)",
                                )
                                v3.VCheckbox(
                                    v_model=(
                                        "fiber_auto_detect_"
                                        "branches"
                                    ),
                                    label=(
                                        "Auto-detect branches"
                                    ),
                                    density="compact",
                                    hide_details=True,
                                    color="primary",
                                    classes="mt-2 mb-2",
                                )
                                html.Div(
                                    "Generation method",
                                    style=(
                                        "font-size: 11px; "
                                        "color: #555; "
                                        "margin-top: 8px;"
                                    ),
                                )
                                # Algorithmic 1/2 are placeholders
                                # — flagged via the `props` key so
                                # Vuetify greys them out + ignores
                                # selection. The `subtitle` text
                                # explains they're not wired yet.
                                v3.VSelect(
                                    v_model=("fiber_method",),
                                    items=([
                                        {
                                            "value": "streamlines",
                                            "title": "Streamlines",
                                            "subtitle": (
                                                "Laplace + RK4 "
                                                "integration"
                                            ),
                                            "props": {
                                                "disabled": False,
                                            },
                                        },
                                        {
                                            "value": "axial",
                                            "title": (
                                                "Axial extrude"
                                            ),
                                            "subtitle": (
                                                "Straight lines "
                                                "along fascicle "
                                                "PCA axis "
                                                "(µCT bundle "
                                                "only)"
                                            ),
                                            "props": {
                                                "disabled": False,
                                            },
                                        },
                                        {
                                            "value": (
                                                "algorithmic_2"
                                            ),
                                            "title": (
                                                "Algorithmic 2"
                                            ),
                                            "subtitle": (
                                                "Coming soon"
                                            ),
                                            "props": {
                                                "disabled": True,
                                            },
                                        },
                                    ],),
                                    item_value="value",
                                    item_title="title",
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    classes="mt-1 mb-2",
                                )
                                # ---- Advanced (collapsed) ----
                                # RK4 step cap + the four branch-
                                # detection knobs. The branch
                                # knobs themselves are gated on
                                # `fiber_auto_detect_branches` —
                                # turning detection off hides them
                                # entirely (single-bundle mode
                                # never reads them).
                                _advanced_toggle(
                                    "stepper_fiber_advanced_open"
                                )
                                with html.Div(
                                    v_show=(
                                        "stepper_fiber_"
                                        "advanced_open",
                                    ),
                                ):
                                    slider_row(
                                        "fiber_max_steps",
                                        "Max integration steps "
                                        "(RK4)",
                                        1000, 50000, 500,
                                        "toFixed(0)",
                                    )
                                    v3.VSelect(
                                        v_model=(
                                            "fiber_seed_end",
                                        ),
                                        items=([
                                            "trunk (low z)",
                                            "branched (high z)",
                                        ],),
                                        label="Seed cap end",
                                        density="compact",
                                        hide_details=True,
                                        classes="mt-2 mb-3",
                                    )
                                    with html.Div(
                                        v_show=(
                                            "fiber_auto_detect_"
                                            "branches",
                                        ),
                                    ):
                                        html.H4(
                                            "Cap detection",
                                            classes=(
                                                "text-subtitle-2 "
                                                "mt-3 mb-1"
                                            ),
                                            style=(
                                                "color: #888a90; "
                                                "letter-spacing: "
                                                "0.04em; "
                                                "text-transform: "
                                                "uppercase; "
                                                "font-size: 10px;"
                                            ),
                                        )
                                        param_row_with_info(
                                            "fiber_cluster_eps_mm",
                                            "Cluster radius",
                                            "DBSCAN xy-radius "
                                            "used to group "
                                            "adjacent cap facets "
                                            "into one cluster.",
                                            "mm", 0.1, 0.1, 20.0,
                                        )
                                        param_row_with_info(
                                            "fiber_cap_band_pct",
                                            "Cap z-band",
                                            "Width of the z-band "
                                            "at each end (% of "
                                            "nerve length) "
                                            "within which axial-"
                                            "normal facets are "
                                            "considered "
                                            "candidate caps.",
                                            "%", 1.0, 1.0, 40.0,
                                        )
                                        param_row_with_info(
                                            "fiber_min_rel_size_"
                                            "pct",
                                            "Min cluster size",
                                            "Drop clusters whose "
                                            "facet count is "
                                            "below this fraction "
                                            "of the largest "
                                            "cluster at the same "
                                            "end.",
                                            "%", 1.0, 0.0, 90.0,
                                        )
                                        param_row_with_info(
                                            "fiber_axial_normal_"
                                            "thresh",
                                            "Axial normal "
                                            "threshold",
                                            "Minimum |n·ẑ| "
                                            "(intrinsic frame) "
                                            "for a boundary "
                                            "facet to count as "
                                            "cap-like.",
                                            "", 0.05, 0.0, 1.0,
                                        )
                                # Action runs from the Next button
                                # in the footer — same single-driver
                                # rule as step 1's Load.

                                # ---- Branch summary + inline
                                # rename ---- F3.2-M2.1b: lifted
                                # from the now-defunct Fibers
                                # drawer. Visible once fibers have
                                # been generated. Each row shows
                                # n_fibers / mean / min / max / std
                                # length in mm; the pencil edits
                                # the branch's display name (used
                                # in the legend + analysis tabs).
                                with html.Div(
                                    v_show=(
                                        "has_fibers "
                                        "&& !fiber_failed",
                                    ),
                                    classes="mt-4",
                                ):
                                    html.Div(
                                        "Branch summary",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; "
                                            "margin-top: 16px; "
                                            "margin-bottom: 4px; "
                                            "letter-spacing: "
                                            "0.03em; "
                                            "text-transform: "
                                            "uppercase;"
                                        ),
                                    )
                                    html.Div(
                                        "Trajectory length "
                                        "statistics by branch. "
                                        "Mean, min, max, std "
                                        "are in millimetres.",
                                        style=(
                                            "font-size: 10px; "
                                            "color: #888a90; "
                                            "margin-bottom: 8px; "
                                            "line-height: 1.4;"
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "golgi-branch-summary"
                                        ),
                                    ):
                                        with html.Div(
                                            classes=(
                                                "golgi-branch-"
                                                "summary-row "
                                                "is-header"
                                            ),
                                        ):
                                            html.Span(
                                                "", classes=(
                                                    "golgi-bs-"
                                                    "name"
                                                ),
                                            )
                                            for _h in (
                                                "Fibers", "Mean",
                                                "Min", "Max",
                                                "Std",
                                            ):
                                                html.Span(
                                                    _h,
                                                    classes=(
                                                        "golgi-"
                                                        "bs-num"
                                                    ),
                                                )
                                        with html.Div(
                                            classes=(
                                                "golgi-branch-"
                                                "summary-row "
                                                "is-subheader"
                                            ),
                                        ):
                                            html.Span(
                                                "", classes=(
                                                    "golgi-bs-"
                                                    "name"
                                                ),
                                            )
                                            for _u in (
                                                "", "mm", "mm",
                                                "mm", "mm",
                                            ):
                                                html.Span(
                                                    _u,
                                                    classes=(
                                                        "golgi-"
                                                        "bs-num"
                                                    ),
                                                )
                                        with html.Div(
                                            v_for=(
                                                "row in "
                                                "fiber_branch_"
                                                "summary",
                                            ),
                                            key="row.idx",
                                            classes=(
                                                "['golgi-"
                                                "branch-summary-"
                                                "row', "
                                                "row.idx === -1 "
                                                "? 'is-overall' "
                                                ": 'is-branch']",
                                            ),
                                        ):
                                            with html.Div(
                                                classes=(
                                                    "golgi-bs-"
                                                    "name"
                                                ),
                                            ):
                                                with html.Div(
                                                    v_show=(
                                                        "branch_"
                                                        "rename_"
                                                        "active "
                                                        "!== "
                                                        "row.idx",
                                                    ),
                                                    classes=(
                                                        "golgi-"
                                                        "bs-"
                                                        "name-"
                                                        "display"
                                                    ),
                                                ):
                                                    html.Div(
                                                        v_show=(
                                                            "row."
                                                            "color",
                                                        ),
                                                        classes=(
                                                            "golgi"
                                                            "-bs-"
                                                            "swatch"
                                                        ),
                                                        style=(
                                                            "'"
                                                            "back"
                                                            "ground"
                                                            ": ' "
                                                            "+ row"
                                                            ".color",
                                                        ),
                                                    )
                                                    html.Span(
                                                        "{{ row."
                                                        "label "
                                                        "}}",
                                                        classes=(
                                                            "golgi"
                                                            "-bs-"
                                                            "label"
                                                        ),
                                                    )
                                                    if (
                                                        do_start_branch_rename
                                                        is not None
                                                    ):
                                                        html.Button(
                                                            "✎",
                                                            type="button",
                                                            v_show=(
                                                                "row.editable",
                                                            ),
                                                            classes=(
                                                                "golgi-bs-edit-btn"
                                                            ),
                                                            title=(
                                                                "Rename branch"
                                                            ),
                                                            click=(
                                                                do_start_branch_rename,
                                                                "[row.idx]",
                                                            ),
                                                        )
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
                                                    if (
                                                        do_apply_branch_rename
                                                        is not None
                                                    ):
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
                                                    if (
                                                        do_cancel_branch_rename
                                                        is not None
                                                    ):
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

                        # ===== Step 4: Muscle cylinder =====
                        with v3.VStepperWindowItem(value="4"):
                            with html.Div(classes="pa-4"):
                                _step_heading_with_info(
                                    "Muscle",
                                    "Surrounding muscle volume "
                                    "(used as the FEM far-field "
                                    "boundary). The bbox is "
                                    "auto-fit to the nerve in "
                                    "PCA frame; tweak the pads "
                                    "and translation to match "
                                    "your anatomical context. "
                                    "Muscle mesh size lives in "
                                    "the Mesh drawer.",
                                )
                                param_row_with_info(
                                    "muscle_radial_pad_mm",
                                    "Radial padding",
                                    "Padding (mm) added to the "
                                    "nerve's max radius to set "
                                    "the muscle cylinder radius.",
                                    "mm", 0.5, 2, 100,
                                )
                                param_row_with_info(
                                    "muscle_axial_pad_mm",
                                    "Axial padding",
                                    "Padding (mm) added at each "
                                    "z-end of the nerve to set "
                                    "the cylinder length.",
                                    "mm", 2, 5, 300,
                                )
                                # ---- Advanced (collapsed) ----
                                # Bbox translation knobs — rarely
                                # needed; default 0 fits everything
                                # for the canonical PCA-aligned
                                # nerve case.
                                _advanced_toggle(
                                    "stepper_muscle_advanced_open"
                                )
                                with html.Div(
                                    v_show=(
                                        "stepper_muscle_"
                                        "advanced_open",
                                    ),
                                ):
                                    param_row_with_info(
                                        "muscle_dx_mm",
                                        "Δx offset",
                                        "Translate the bbox "
                                        "along x without "
                                        "resizing it.",
                                        "mm", 0.5, -100, 100,
                                    )
                                    param_row_with_info(
                                        "muscle_dy_mm",
                                        "Δy offset",
                                        "Translate the bbox "
                                        "along y without "
                                        "resizing it.",
                                        "mm", 0.5, -100, 100,
                                    )
                                    param_row_with_info(
                                        "muscle_dz_mm",
                                        "Δz offset",
                                        "Translate the bbox "
                                        "along z (nerve axis) "
                                        "without resizing it.",
                                        "mm", 0.5, -200, 200,
                                    )

            # ---- Action bar ----
            # F3.2-M3 — two buttons per step:
            #   * Action (left)   — runs the step's underlying
            #     action (load nerve / set focus / generate
            #     fibers / commit muscle). Does NOT advance the
            #     step. Label changes with the active step.
            #   * Continue (right) — advances to the next step
            #     (or closes the wizard on step 4). Gated on the
            #     step's action having completed: Step 1 needs
            #     has_geometry, Step 3 needs has_fibers. Steps 2
            #     and 4 enable Continue immediately.
            # Back stays as a no-op step-rewind. All buttons
            # disabled while busy so a double-click can't queue
            # two actions.
            with v3.VCardActions(classes="pa-4"):
                html.Button(
                    "← Back",
                    type="button",
                    v_show=(
                        "import_stepper_step !== '1'",
                    ),
                    disabled=("busy",),
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    # Bundle mode: jump 3 → 1 so step 2 is
                    # also skipped on the way back (the
                    # Endoneurium step is inert for bundles,
                    # since the bundle already carries epi
                    # + fascicle surfaces). Anywhere else,
                    # the normal step-1 decrement applies.
                    click=(
                        "import_stepper_step = ("
                        "(import_source_type === 'uct_bundle' "
                        "|| import_source_type === 'histo_bundle'"
                        ") && import_stepper_step === '3'"
                        ") ? '1' : String("
                        "  Math.max(1, "
                        "  parseInt(import_stepper_step) - 1)"
                        ")"
                    ),
                )
                v3.VSpacer()
                html.Button(
                    (
                        "{{ "
                        "import_stepper_step === '1' ? "
                        "'▶ Load nerve' : "
                        "import_stepper_step === '2' ? "
                        "'▶ Generate endoneurium' : "
                        "import_stepper_step === '3' ? "
                        "'▶ Generate fibers' : "
                        "'▶ Generate muscle'"
                        " }}"
                    ),
                    type="button",
                    # Step 1 needs a selected file (STL flow)
                    # OR a selected bundle (Golgi µCT flow);
                    # Step 3 needs has_geometry; Steps 2 + 4
                    # always enabled. The bundle branch was
                    # gating on `!selected_file` only, which
                    # left the button stuck disabled when the
                    # user landed on Step 1 with a bundle
                    # pre-selected (e.g. via the "Done → Import
                    # wizard" handoff from the Segment dialog).
                    # M47 — load_nerve_blocked is a server-
                    # computed boolean (see app.py
                    # `_recompute_load_nerve_blocked`) that
                    # tracks whether the current source-type
                    # has a selection. Complex ternary
                    # expressions like the previous
                    # `import_source_type === 'uct_bundle'
                    # ? !selected_uct_bundle : !selected_file`
                    # didn't reactively re-evaluate here for
                    # the same reason picker v_show didn't.
                    disabled=(
                        "busy || ("
                        "import_stepper_step === '1' && "
                        "load_nerve_blocked"
                        ") || ("
                        "import_stepper_step === '3' && "
                        "!has_geometry"
                        ")",
                    ),
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm "
                        "mr-2"
                    ),
                    click=do_stepper_action,
                )
                html.Button(
                    (
                        "{{ "
                        "import_stepper_step === '4' ? "
                        "'Done' : "
                        "'Continue →'"
                        " }}"
                    ),
                    type="button",
                    # Continue is the primary advance button.
                    # Gate on the step's action having
                    # completed so the user can't skip past
                    # without explicitly clicking the action.
                    # Step 4's Done always enables (it just
                    # closes the wizard).
                    disabled=(
                        "busy || ("
                        "import_stepper_step === '1' && "
                        "!has_geometry"
                        ") || ("
                        "import_stepper_step === '3' && "
                        "!has_fibers"
                        ")",
                    ),
                    classes=(
                        "golgi-btn-primary golgi-btn-sm"
                    ),
                    click=do_stepper_next,
                )
