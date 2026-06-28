# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cuff & Electrodes drawer — multi-electrode list (add /
select / rename / refit / delete), per-electrode contact
configuration (bipolar / tripolar / ring-array / helical /
DUKE preset), cuff geometry, frame & placement controls, and
per-contact polarity picker."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from ._helpers import slider_row


def render(
    *,
    duke_electrode_type: str,
    electrode_types: list,
    edit_icon_url: str,
    do_add_design: Callable,
    do_save_rename_eid: Callable,
    do_cancel_rename_eid: Callable,
    do_open_cuff_designer: Callable,
) -> None:
    with v3.VNavigationDrawer(
        v_model=("show_cuff",),
        location="right", width=460,
        elevation=8,
    ):
        with v3.VContainer(classes="pa-4"):
            # Title row: H3 + info-icon tooltip + close button.
            with html.Div(
                classes=(
                    "d-flex align-center "
                    "justify-space-between mb-2"
                ),
            ):
                with html.Div(
                    classes="d-flex align-center",
                ):
                    html.H3(
                        "Cuff designs",
                        classes="text-h6 mb-0",
                    )
                    with v3.VTooltip(
                        location="bottom",
                        max_width=400,
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
                            "Each design is a physical nerve "
                            "cuff at a user-chosen axial offset "
                            "+ rotation, with a chosen contact "
                            "hardware (bipolar / tripolar / "
                            "ring-array / helical / DUKE "
                            "preset). Each design gets its own "
                            "FEM mesh. Per-contact "
                            "anode/cathode/off polarity drives "
                            "the FEM source terms downstream."
                        )
                with html.Div(
                    classes="d-flex align-center",
                    style="gap: 6px;",
                ):
                    v3.VBtn(
                        icon="mdi-close", size="small",
                        variant="text",
                        click="show_cuff = false",
                    )

            # --- Multi-design list ---
            with html.Div(
                classes=(
                    "d-flex align-center "
                    "justify-space-between mt-2 mb-1"
                ),
            ):
                html.H4(
                    "Designs",
                    classes="text-subtitle-2 mb-0",
                )
                with html.Div(
                    classes="d-flex align-center",
                    style="gap: 6px;",
                ):
                    html.Button(
                        "+ Sweep",
                        type="button",
                        title=(
                            "Clone the selected design into a "
                            "batch varying Z and/or rotation"
                        ),
                        classes=(
                            "golgi-btn-secondary "
                            "golgi-btn-sm"
                        ),
                        disabled=("!selected_design_id",),
                        click=(
                            "show_sweep_designs_dialog = true"
                        ),
                    )
                    html.Button(
                        "+ Add",
                        type="button",
                        classes=(
                            "golgi-btn-secondary "
                            "golgi-btn-sm"
                        ),
                        click=do_add_design,
                    )
            # Stack of electrode tiles — one row per entry in
            # state.designs. Click selects, pencil enters
            # rename mode, ✕ removes. Selected one gets the red
            # accent border. Empty-state hint when the list is
            # empty.
            with html.Div(
                classes="mb-3",
                style="display: flex; flex-direction: column; "
                        "gap: 4px;",
            ):
                with html.Div(
                    v_for="design in designs",
                    key="design.eid",
                    # Tuple (note trailing comma) → trame emits
                    # `:style="..."` rather than `style="..."`,
                    # so the conditional actually runs. Without
                    # the comma the parens just group a single
                    # string and trame writes a static attribute
                    # the browser can't parse as conditional CSS.
                    style=(
                        "'padding: 8px 10px; "
                        "border-radius: 8px; "
                        "cursor: pointer; "
                        "display: flex; "
                        "flex-direction: row; "
                        "flex-wrap: nowrap; "
                        "align-items: center; "
                        "gap: 8px; '"
                        " + (design.eid === "
                        "      selected_design_id "
                        "    ? 'background: #ffd6d6; "
                        "       border: 1px solid #e24b4a; "
                        "       border-left: 5px solid "
                        "         #e24b4a; "
                        "       box-shadow: 0 1px 6px "
                        "         rgba(226,75,74,0.25);' "
                        "    : 'background: #ffffff; "
                        "       border: 1px solid #e3e3e6; "
                        "       border-left: 5px solid "
                        "         transparent;')",
                    ),
                    click=(
                        "rename_eid_active === design.eid "
                        "  ? null "
                        "  : (selected_design_id = "
                        "      design.eid)"
                    ),
                ):
                    # Display mode (name + edit pencil + type
                    # label + per-row refit + delete).
                    with html.Div(
                        v_show=(
                            "rename_eid_active !== design.eid",
                        ),
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
                            # For DUKE electrodes show the
                            # specific preset (e.g.
                            # "LivaNova3000_v2") so the row
                            # reads as a design identifier at
                            # a glance.
                            "{{ design.electrode_type === "
                            "  '" + duke_electrode_type + "' "
                            "  ? (design.duke_preset || "
                            "      'DUKE (no preset)') "
                            "  : design.electrode_type }}",
                            style=(
                                "font-size: 10px; "
                                "color: #888a90; "
                                "white-space: nowrap; "
                                "overflow: hidden; "
                                "text-overflow: ellipsis;"
                            ),
                        )
                    with html.Button(
                        type="button",
                        v_show=(
                            "rename_eid_active !== design.eid",
                        ),
                        classes="golgi-detail-edit-btn",
                        title="Rename",
                        style="flex: 0 0 auto;",
                        click=(
                            "$event.stopPropagation(); "
                            "rename_eid_active = design.eid; "
                            "rename_eid_value = design.name"
                        ),
                    ):
                        html.Img(
                            src=edit_icon_url,
                            alt="",
                            classes="golgi-detail-edit-icon",
                        )
                    html.Button(
                        "Refit",
                        type="button",
                        classes=(
                            "golgi-btn-primary golgi-btn-sm"
                        ),
                        title="Refit cuff at this electrode",
                        v_show=(
                            "rename_eid_active !== design.eid",
                        ),
                        style="flex: 0 0 auto;",
                        click=(
                            "$event.stopPropagation(); "
                            "refit_design_request = "
                            "  design.eid"
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
                        title="Remove",
                        v_show=(
                            "rename_eid_active !== design.eid",
                        ),
                        style="flex: 0 0 auto;",
                        click=(
                            "$event.stopPropagation(); "
                            "confirm_delete_eid = design.eid; "
                            "confirm_delete_name = design.name; "
                            "show_confirm_delete_dialog = true"
                        ),
                    )
                    # Edit mode — text input + ✓ save + ✕ cancel.
                    v3.VTextField(
                        v_model=("rename_eid_value",),
                        v_show=(
                            "rename_eid_active === design.eid",
                        ),
                        density="compact",
                        hide_details=True,
                        autofocus=True,
                        variant="outlined",
                        keydown_enter=do_save_rename_eid,
                        keydown_escape=do_cancel_rename_eid,
                        style=(
                            "flex: 1 1 auto; "
                            "font-size: 12px;"
                        ),
                        click=("$event.stopPropagation()",),
                    )
                    html.Button(
                        "✓",
                        type="button",
                        v_show=(
                            "rename_eid_active === design.eid",
                        ),
                        classes="golgi-detail-edit-confirm",
                        title="Save",
                        click=(
                            "$event.stopPropagation()"
                        ),
                        mouseup=do_save_rename_eid,
                    )
                    html.Button(
                        "✕",
                        type="button",
                        v_show=(
                            "rename_eid_active === design.eid",
                        ),
                        classes="golgi-detail-edit-cancel",
                        title="Cancel",
                        click=(
                            "$event.stopPropagation()"
                        ),
                        mouseup=do_cancel_rename_eid,
                    )
                # Empty-state hint when no designs exist.
                html.Div(
                    "No designs yet — click + Add to "
                    "place one.",
                    v_show=("!designs.length",),
                    style=(
                        "color: #888a90; "
                        "font-size: 11px; "
                        "padding: 12px 4px; "
                        "text-align: center; "
                        "border: 1px dashed #e3e3e6; "
                        "border-radius: 8px;"
                    ),
                )

            # --- Per-electrode params (only when one is
            # selected). Edits the CURRENTLY SELECTED electrode
            # via mirror state vars; watchers auto-save back
            # into the electrodes[] dict.
            with html.Div(v_show=("selected_design_id",)):
                with html.H3(
                    classes="golgi-elec-edit-heading",
                ):
                    html.Span(
                        "Configuring "
                        "{{ (designs.find("
                        "  e => e.eid === "
                        "    selected_design_id) "
                        "  || {}).name "
                        "  || selected_design_id }}",
                    )
                    html.Span(
                        "{{ selected_design_id }}",
                        classes="golgi-elec-edit-eid",
                    )
                v3.VSelect(
                    v_model=("electrode_type",),
                    items=(electrode_types,),
                    label="Contact configuration",
                    density="compact", hide_details=True,
                    classes="mb-2",
                )

                # Bipolar
                with html.Div(
                    v_show=(
                        "electrode_type === "
                        "'bipolar ring-pair'",
                    ),
                ):
                    slider_row(
                        "bipolar_axial_sep_mm",
                        "Ring separation (mm)",
                        0.1, 100.0, 0.1,
                        info=(
                            "Axial distance between the "
                            "centres of the two rings "
                            "(anode and cathode)."
                        ),
                    )
                    slider_row(
                        "bipolar_ring_width_mm",
                        "Ring width (mm)",
                        0.05, 10.0, 0.05, "toFixed(2)",
                        info=(
                            "Axial extent of each ring "
                            "contact along the nerve."
                        ),
                    )

                # Tripolar
                with html.Div(
                    v_show=(
                        "electrode_type === "
                        "'tripolar (anode-cathode-anode)'",
                    ),
                ):
                    slider_row(
                        "tripolar_axial_sep_mm",
                        "Cathode–anode separation (mm)",
                        0.1, 50.0, 0.1,
                        info=(
                            "Axial distance between the "
                            "central cathode ring and "
                            "each flanking anode ring."
                        ),
                    )
                    slider_row(
                        "tripolar_ring_width_mm",
                        "Ring width (mm)",
                        0.05, 10.0, 0.05, "toFixed(2)",
                        info=(
                            "Axial extent of each ring "
                            "contact along the nerve."
                        ),
                    )

                # Ring-array NxM
                with html.Div(
                    v_show=(
                        "electrode_type === "
                        "'ring-array (NxM)'",
                    ),
                ):
                    slider_row(
                        "array_n_rows",
                        "Axial rows",
                        1, 12, 1, "toFixed(0)",
                        info=(
                            "Number of ring rows arrayed "
                            "along the nerve axis."
                        ),
                    )
                    slider_row(
                        "array_n_cols",
                        "Circumferential columns",
                        2, 24, 1, "toFixed(0)",
                        info=(
                            "Number of contact patches "
                            "around the nerve "
                            "circumference per row."
                        ),
                    )
                    slider_row(
                        "array_row_sep_mm",
                        "Row separation (mm)",
                        0.1, 50.0, 0.1,
                        info=(
                            "Axial distance between "
                            "successive rows."
                        ),
                    )
                    slider_row(
                        "array_contact_w_mm",
                        "Contact axial width (mm)",
                        0.05, 10.0, 0.05, "toFixed(2)",
                        info=(
                            "Axial extent of each "
                            "rectangular contact patch."
                        ),
                    )
                    slider_row(
                        "array_contact_phi_deg",
                        "Contact angular extent (°)",
                        5, 360, 1, "toFixed(0)",
                        info=(
                            "Angular sweep covered by "
                            "each contact patch around "
                            "the nerve circumference."
                        ),
                    )

                # Helical
                with html.Div(
                    v_show=(
                        "electrode_type === "
                        "'helical (Livanova-style)'",
                    ),
                ):
                    slider_row(
                        "helix_n_bands",
                        "Spiral bands",
                        1, 12, 1, "toFixed(0)",
                        info=(
                            "Number of helical contact "
                            "bands wrapping the nerve."
                        ),
                    )
                    slider_row(
                        "helix_pitch_mm",
                        "Axial pitch (mm per turn)",
                        0.5, 100.0, 0.5,
                        info=(
                            "Axial distance the helix "
                            "advances along the nerve "
                            "in one full 360° turn. "
                            "Smaller pitch = tighter "
                            "spiral."
                        ),
                    )
                    slider_row(
                        "helix_dphi_deg",
                        "Band arc width (°)",
                        5, 720, 5, "toFixed(0)",
                        info=(
                            "Total angular sweep covered "
                            "by each band along its "
                            "helical path. Larger sweep "
                            "= more nerve circumference "
                            "engaged."
                        ),
                    )

                # LIFE — longitudinal intrafascicular array
                # (N axial contacts per wire × M parallel
                # wires). Parameters mirror ring-array's
                # NxM shape but the "perpendicular" axis is
                # transverse position inside the nerve
                # rather than angular position around it.
                with html.Div(
                    v_show=(
                        "electrode_type === "
                        "'LIFE (longitudinal "
                        "intrafascicular)'",
                    ),
                ):
                    slider_row(
                        "life_n_rows",
                        "Axial contacts per wire",
                        1, 16, 1, "toFixed(0)",
                        info=(
                            "Number of exposed contact "
                            "bands along each LIFE wire."
                        ),
                    )
                    slider_row(
                        "life_n_cols",
                        "Number of wires",
                        1, 8, 1, "toFixed(0)",
                        info=(
                            "Number of parallel LIFE "
                            "filaments inserted along a "
                            "chord across the nerve. "
                            "1 = classic single-wire LIFE."
                        ),
                    )
                    slider_row(
                        "life_row_sep_mm",
                        "Contact axial spacing (mm)",
                        0.1, 20.0, 0.1,
                        info=(
                            "Centre-to-centre spacing "
                            "between contact bands on the "
                            "same wire."
                        ),
                    )
                    slider_row(
                        "life_col_sep_mm",
                        "Wire transverse spacing (mm)",
                        0.05, 5.0, 0.05, "toFixed(2)",
                        info=(
                            "Centre-to-centre transverse "
                            "spacing between adjacent "
                            "wires along the chord."
                        ),
                    )
                    slider_row(
                        "life_contact_length_mm",
                        "Contact length (mm)",
                        0.05, 5.0, 0.05, "toFixed(2)",
                        info=(
                            "Axial extent of each exposed "
                            "contact band on the wire."
                        ),
                    )
                    slider_row(
                        "life_diameter_um",
                        "Wire diameter (µm)",
                        25, 500, 5, "toFixed(0)",
                        info=(
                            "Diameter of each LIFE filament. "
                            "Typical tfLIFE is ~75 µm."
                        ),
                    )
                    slider_row(
                        "life_chord_phi_deg",
                        "Wire-array chord angle (°)",
                        -90, 90, 1, "toFixed(0)",
                        info=(
                            "Orientation of the wire array "
                            "in the cuff transverse plane. "
                            "0° = wires arrayed along +x; "
                            "auto-fit picks this from "
                            "fascicle PCA when available."
                        ),
                    )
                    slider_row(
                        "life_target_fascicle_idx",
                        "Target fascicle index "
                        "(-1 = nerve centroid)",
                        -1, 32, 1, "toFixed(0)",
                        info=(
                            "Auto-fit lands the array "
                            "centre at this fascicle's "
                            "centroid (0-indexed by "
                            "bundle order). -1 uses the "
                            "whole-nerve centroid — the "
                            "only valid choice on "
                            "monofascicular / STL "
                            "imports."
                        ),
                    )

                # TIME — transverse intrafascicular array
                # (N axial rows × M transverse columns of
                # contacts on the ribbon's front face).
                with html.Div(
                    v_show=(
                        "electrode_type === "
                        "'TIME (transverse "
                        "intrafascicular)'",
                    ),
                ):
                    slider_row(
                        "time_n_rows",
                        "Axial rows",
                        1, 8, 1, "toFixed(0)",
                        info=(
                            "Number of contact rows along "
                            "the nerve axis. 1 = classic "
                            "single-row TIME."
                        ),
                    )
                    slider_row(
                        "time_n_cols",
                        "Contacts per row",
                        1, 32, 1, "toFixed(0)",
                        info=(
                            "Number of contacts arrayed "
                            "transversely along the ribbon "
                            "in each axial row."
                        ),
                    )
                    slider_row(
                        "time_row_sep_mm",
                        "Row axial spacing (mm)",
                        0.05, 10.0, 0.05, "toFixed(2)",
                        info=(
                            "Centre-to-centre axial spacing "
                            "between successive rows."
                        ),
                    )
                    slider_row(
                        "time_col_sep_mm",
                        "Contact pitch (mm)",
                        0.05, 2.0, 0.005, "toFixed(3)",
                        info=(
                            "Centre-to-centre transverse "
                            "spacing along the ribbon. "
                            "Typical TIME pitch ~230 µm."
                        ),
                    )
                    slider_row(
                        "time_contact_w_mm",
                        "Contact axial width (mm)",
                        0.02, 2.0, 0.005, "toFixed(3)",
                        info=(
                            "Axial extent of each contact "
                            "on the ribbon."
                        ),
                    )
                    slider_row(
                        "time_ribbon_width_mm",
                        "Ribbon transverse length (mm)",
                        0.5, 10.0, 0.1,
                        info=(
                            "Total length of the ribbon "
                            "in the chord direction. Should "
                            "exceed the nerve diameter so "
                            "the ribbon threads through "
                            "fully."
                        ),
                    )
                    slider_row(
                        "time_ribbon_thickness_um",
                        "Ribbon thickness (µm)",
                        20, 500, 5, "toFixed(0)",
                        info=(
                            "Out-of-plane thickness of the "
                            "ribbon. Rendered as a thin "
                            "translucent strip; FEM treats "
                            "contacts as thin conductors."
                        ),
                    )
                    slider_row(
                        "time_chord_phi_deg",
                        "Ribbon chord angle (°)",
                        -90, 90, 1, "toFixed(0)",
                        info=(
                            "Orientation of the ribbon's "
                            "long axis in the cuff "
                            "transverse plane. Auto-fit "
                            "picks this from fascicle PCA "
                            "when ≥ 2 fascicles are present."
                        ),
                    )

                # DUKE Cuff designer — replaces the param
                # sliders with a single "Open designer"
                # button that opens the ASCENT designer
                # dialog scoped to the selected electrode.
                with html.Div(
                    v_show=(
                        f"electrode_type === "
                        f"'{duke_electrode_type}'",
                    ),
                ):
                    html.Div(
                        "Use the designer to pick an ASCENT "
                        "preset (LivaNova / MultiContact) "
                        "and tweak its parameters. The "
                        "design is stored per-electrode.",
                        style=(
                            "font-size: 11px; "
                            "color: #6c6c70; "
                            "margin-bottom: 8px; "
                            "line-height: 1.4;"
                        ),
                    )
                    html.Div(
                        "Current preset: "
                        "{{ designs.find(e => e.eid === "
                        "selected_design_id)?.duke_preset "
                        "|| '(none chosen)' }}",
                        style=(
                            "font-size: 11px; "
                            "color: #888a90; "
                            "margin-bottom: 12px; "
                            "font-family: monospace;"
                        ),
                    )
                    html.Button(
                        "Open electrode designer",
                        type="button",
                        classes=(
                            "golgi-btn-primary golgi-btn-sm "
                            "golgi-btn-block"
                        ),
                        click=do_open_cuff_designer,
                    )

                # --- Cuff geometry ---
                # Standard cuff geometry sliders apply to the
                # built-in primitive types only. DUKE cuffs
                # carry their own geometry inside the preset
                # so we hide this block when DUKE is selected.
                html.H4(
                    "Cuff geometry",
                    v_show=(
                        f"electrode_type !== "
                        f"'{duke_electrode_type}'",
                    ),
                    classes="text-subtitle-2 mt-4 mb-1",
                    style=(
                        "color: #888a90; "
                        "letter-spacing: 0.04em; "
                        "text-transform: uppercase; "
                        "font-size: 10px;"
                    ),
                )
                with html.Div(
                    v_show=(
                        f"electrode_type !== "
                        f"'{duke_electrode_type}'",
                    ),
                ):
                    slider_row(
                        "L_cuff_mm",
                        "Cuff length (mm)",
                        1, 100, 0.1,
                        info=(
                            "Axial length of the silicone "
                            "cuff shell along the nerve."
                        ),
                    )
                    slider_row(
                        "cuff_clearance_mm",
                        "Inner radius clearance (mm)",
                        0.01, 10.0, 0.01, "toFixed(2)",
                        info=(
                            "Radial gap between the nerve "
                            "outer surface and the cuff "
                            "inner wall. The gap is "
                            "filled with saline (tag 2) "
                            "when the saline infill is "
                            "enabled."
                        ),
                    )
                    slider_row(
                        "cuff_wall_mm",
                        "Cuff wall thickness (mm)",
                        0.01, 10.0, 0.01, "toFixed(2)",
                        info=(
                            "Radial thickness of the "
                            "silicone shell (tag 3) "
                            "between the inner saline "
                            "cavity and the outer "
                            "muscle-facing surface."
                        ),
                    )
                    v3.VCheckbox(
                        v_model=("show_saline",),
                        label="Generate saline infill",
                        density="compact", hide_details=True,
                        color="primary", classes="mt-2",
                    )
                    # F3.2-M3 — per-design scar / connective
                    # tissue layer. When enabled, the PLC
                    # builder inserts a scar cylinder at
                    # R_scar = R_ci − scar_thickness; the
                    # PLC carves a tag-7 region between the
                    # nerve outer surface and this cylinder,
                    # and saline auto-fills the remaining
                    # gap from R_scar out to R_ci. Bigger
                    # thickness ⇒ thinner saline annulus.
                    v3.VCheckbox(
                        v_model=("use_scar",),
                        label=(
                            "Generate scar / "
                            "connective tissue shell"
                        ),
                        density="compact", hide_details=True,
                        color="primary", classes="mt-2",
                    )
                    with html.Div(
                        v_show=("use_scar",),
                    ):
                        slider_row(
                            "scar_thickness_um",
                            "Scar layer thickness (µm)",
                            10, 2000, 10, "toFixed(0)",
                            info=(
                                "Radial thickness of the scar "
                                "shell, measured outward from "
                                "the nerve surface. Bigger "
                                "value = thicker scar layer. "
                                "R_scar = r_nerve + thickness; "
                                "saline auto-fills the "
                                "remaining annular gap up to "
                                "the cuff inner wall (R_ci). "
                                "Auto-clamped so R_scar never "
                                "exceeds R_ci (any excess is "
                                "silently truncated at mesh "
                                "time)."
                            ),
                        )

                # --- Cuff frame & placement (all types) ---
                html.H4(
                    "Cuff frame & placement",
                    classes="text-subtitle-2 mt-4 mb-1",
                    style=(
                        "color: #888a90; "
                        "letter-spacing: 0.04em; "
                        "text-transform: uppercase; "
                        "font-size: 10px;"
                    ),
                )
                # Anchor end — label + (i) tooltip + VSelect.
                with html.Div(
                    classes="d-flex align-center mt-2",
                    style="gap: 2px;",
                ):
                    html.Span(
                        "Anchor end",
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
                            "Reference point used as the "
                            "z-origin for the axial "
                            "offset. 'trunk' / 'branched' "
                            "use the auto-detected nerve "
                            "ends; 'centroid' uses the "
                            "nerve midpoint."
                        )
                v3.VSelect(
                    v_model=("cuff_anchor",),
                    items=(["trunk", "branched", "centroid"],),
                    density="compact", hide_details=True,
                    variant="outlined",
                    classes="mb-1 mt-1",
                )
                slider_row(
                    "cuff_offset_mm",
                    "Axial offset from anchor (mm)",
                    -200, 200, 0.1,
                    info=(
                        "Signed distance from the anchor "
                        "end along the nerve axis. "
                        "Positive moves toward the "
                        "opposite end."
                    ),
                )
                # F3.2-M3: fine-placement knobs (transverse
                # shift + intrinsic Euler rotations + local PCA
                # radius) tucked behind an Advanced toggle. The
                # default cuff fit with just an axial offset is
                # all most users need; the rest live here for
                # off-axis / non-coaxial experiments. Toggle
                # state is ephemeral — not persisted, opens
                # fresh-closed each session.
                with html.Div(
                    classes="d-flex align-center mt-3 mb-2",
                    style=(
                        "cursor: pointer; user-select: none; "
                        "color: #888a90; letter-spacing: 0.04em; "
                        "text-transform: uppercase; "
                        "font-size: 10px;"
                    ),
                    click=(
                        "cuff_placement_advanced_open = "
                        "!cuff_placement_advanced_open"
                    ),
                ):
                    html.Span(
                        "{{ cuff_placement_advanced_open "
                        "? '▾' : '▸' }}",
                        style="margin-right: 6px;",
                    )
                    html.Span("Advanced")
                with html.Div(
                    v_show=("cuff_placement_advanced_open",),
                ):
                    slider_row(
                        "cuff_dx_mm",
                        "Transverse Δx (mm)",
                        -100, 100, 0.05, "toFixed(2)",
                        info=(
                            "Shift the cuff centre "
                            "perpendicular to the local "
                            "nerve axis along the x direction "
                            "(cuff frame)."
                        ),
                    )
                    slider_row(
                        "cuff_dy_mm",
                        "Transverse Δy (mm)",
                        -100, 100, 0.05, "toFixed(2)",
                        info=(
                            "Shift the cuff centre "
                            "perpendicular to the local "
                            "nerve axis along the y direction "
                            "(cuff frame)."
                        ),
                    )
                    # F3.2a: intrinsic Euler rotations in the
                    # cuff's own frame. Applied as Rx → Ry → Rz
                    # so the user can pitch, then yaw, then
                    # twist around the cuff's own axis.
                    # Defaults are 0° so existing designs stay
                    # aligned with the nerve trajectory.
                    slider_row(
                        "cuff_rot_x_deg",
                        "Pitch — rotate around local x (°)",
                        -90, 90, 1, "toFixed(0)",
                        info=(
                            "Tilt the cuff off the nerve axis "
                            "around its local x. Use small "
                            "angles to study off-axis "
                            "stimulation; large tilts make "
                            "the cuff non-coaxial with the "
                            "nerve."
                        ),
                    )
                    slider_row(
                        "cuff_rot_y_deg",
                        "Yaw — rotate around local y (°)",
                        -90, 90, 1, "toFixed(0)",
                        info=(
                            "Tilt the cuff off the nerve axis "
                            "around its local y. Combine with "
                            "pitch for two-axis "
                            "off-alignment."
                        ),
                    )
                    slider_row(
                        "cuff_rot_z_deg",
                        "Twist — rotate around cuff axis (°)",
                        -180, 180, 1, "toFixed(0)",
                        info=(
                            "Rotate the contact pattern "
                            "around the local longitudinal "
                            "axis of the cuff. Useful for "
                            "sweeping which fascicles sit "
                            "under each contact without "
                            "physically moving the cuff."
                        ),
                    )
                    slider_row(
                        "local_pca_radius_mm",
                        "Local PCA radius (mm)",
                        1, 100, 0.5,
                        info=(
                            "Neighbourhood radius used to "
                            "compute a local nerve frame at "
                            "the cuff position. Larger = "
                            "smoother frame (averages out "
                            "small curvature); smaller = "
                            "follows local curvature more "
                            "tightly."
                        ),
                    )

                # --- Contact polarity (per-contact picker) ---
                html.H4(
                    "Contact polarity",
                    v_show=("contact_count > 0",),
                    classes="text-subtitle-2 mt-4 mb-1",
                    style=(
                        "color: #888a90; "
                        "letter-spacing: 0.04em; "
                        "text-transform: uppercase; "
                        "font-size: 10px;"
                    ),
                )

                # M1: quick-preset dropdown. Lets the user pick
                # a canonical N-polar config in one click instead
                # of clicking through each contact. The preset
                # writes contact_polarities + (optionally)
                # contact_current_fractions in one go.
                with html.Div(
                    v_show=("contact_count > 0",),
                    classes="d-flex align-center",
                    style=(
                        "gap: 8px; "
                        "margin-bottom: 8px;"
                    ),
                ):
                    html.Span(
                        "Quick preset",
                        style=(
                            "font-size: 11px; "
                            "color: #555; "
                            "min-width: 88px;"
                        ),
                    )
                    v3.VSelect(
                        v_model=("contact_preset",),
                        items=([
                            {"title": "—", "value": ""},
                            {
                                "title": (
                                    "Bipolar "
                                    "(1 anode + 1 cathode)"
                                ),
                                "value": "bipolar",
                            },
                            {
                                "title": (
                                    "Tripolar guard "
                                    "(longitudinal "
                                    "A-C-A)"
                                ),
                                "value": "tripolar_long",
                            },
                            {
                                "title": (
                                    "Tripolar guard "
                                    "(transverse "
                                    "A-C-A)"
                                ),
                                "value": "tripolar_trans",
                            },
                            {
                                "title": (
                                    "Quadripolar guarded "
                                    "bipole "
                                    "(A-C-C-A, 50/50)"
                                ),
                                "value": "quadripolar",
                            },
                            {
                                "title": (
                                    "All cathode "
                                    "(monopolar, "
                                    "remote ground)"
                                ),
                                "value": "monopolar",
                            },
                        ],),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        update_modelValue=(
                            "trigger('do_apply_polarity_preset', "
                            "[$event])"
                        ),
                        style="flex: 1 1 auto;",
                    )

                # M1: sum-check chips. One chip per polarity
                # group present in the current assignment;
                # `color` is 'success' (green) when the fractions
                # sum correctly OR all-implicit equal split, or
                # 'warning' (amber) when they don't.
                with html.Div(
                    v_show=("contact_polarity_sums.length > 0",),
                    classes="d-flex flex-wrap",
                    style=(
                        "gap: 6px; "
                        "margin-bottom: 10px;"
                    ),
                ):
                    with html.Template(
                        v_for=("g in contact_polarity_sums",),
                        key="g.label",
                    ):
                        v3.VChip(
                            "{{ g.label }} ({{ g.n_total }}) · "
                            "{{ g.hint }}",
                            density="compact",
                            size="small",
                            variant="tonal",
                            color=("g.color",),
                        )
                with html.Div(
                    v_show=("contact_count > 0",),
                    style=(
                        "display: flex; "
                        "flex-direction: column; gap: 6px;"
                    ),
                ):
                    with html.Div(
                        v_for=(
                            "(pol, idx) in contact_polarities"
                        ),
                        key="idx",
                        style=(
                            "display: flex; "
                            "align-items: center; "
                            "gap: 8px;"
                        ),
                    ):
                        # Coloured marker next to the dropdown
                        # showing the CURRENT polarity (matches
                        # the contact's tint in the 3-D view).
                        # Same palette as the dropdown item
                        # dots: cathode blue, anode red, ground
                        # neutral grey, off near-white.
                        html.Div(
                            style=(
                                "'width: 14px; height: 14px; "
                                "border-radius: 50%; "
                                "border: 1px solid "
                                "  rgba(0,0,0,0.25); "
                                "flex: 0 0 auto; "
                                "background: ' "
                                "+ (pol === 'anode' "
                                "    ? '#e02a32' "
                                "    : (pol === 'cathode' "
                                "        ? '#2f7fe6' "
                                "        : (pol === 'ground' "
                                "            ? '#888888' "
                                "            : '#dddddd')))",
                            ),
                        )
                        html.Span(
                            "Contact {{ idx + 1 }}",
                            style=(
                                "font-size: 12px; "
                                "min-width: 70px;"
                            ),
                        )
                        with v3.VSelect(
                            model_value=("pol",),
                            # M1: items carry a `color` field
                            # so the slot template can render a
                            # CSS-styled dot instead of a
                            # Unicode-emoji circle (the emoji
                            # renders inconsistently across
                            # platforms + can't be sized to
                            # match the rest of the UI).
                            items=([
                                {"title": "Off",
                                 "value": "off",
                                 "color": "#dddddd"},
                                {"title": "Cathode",
                                 "value": "cathode",
                                 "color": "#2f7fe6"},
                                {"title": "Anode",
                                 "value": "anode",
                                 "color": "#e02a32"},
                                {"title": "Ground",
                                 "value": "ground",
                                 "color": "#888888"},
                            ],),
                            item_title="title",
                            item_value="value",
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            update_modelValue=(
                                "contact_polarities = "
                                "  contact_polarities.map("
                                "    (p, i) => i === idx "
                                "      ? $event : p)"
                            ),
                            style="flex: 1 1 200px;",
                        ):
                            # `selection` slot — the chip-style
                            # preview of the currently-selected
                            # value (shown in the VSelect's
                            # input row when closed). CSS dot
                            # + title text, same shape as the
                            # item-list rows below.
                            with v3.Template(
                                v_slot_selection=(
                                    "{ item }",
                                ),
                            ):
                                html.Span(
                                    classes=(
                                        "golgi-fiber-chip-dot"
                                    ),
                                    style=(
                                        "'background:' "
                                        "+ (item.raw.color "
                                        "   || '#888')",
                                    ),
                                )
                                html.Span(
                                    "{{ item.raw.title }}",
                                )
                            # `item` slot — each row in the
                            # dropdown menu. Uses VListItem so
                            # Vuetify still gives us hover +
                            # active-state styling; the
                            # `prepend` slot inside it carries
                            # the CSS dot.
                            with v3.Template(
                                v_slot_item=(
                                    "{ props, item }",
                                ),
                            ):
                                with v3.VListItem(
                                    v_bind="props",
                                ):
                                    with v3.Template(
                                        v_slot_prepend=True,
                                    ):
                                        html.Span(
                                            classes=(
                                                "golgi-fiber"
                                                "-chip-dot"
                                            ),
                                            style=(
                                                "'background:' "
                                                "+ (item.raw"
                                                ".color "
                                                "   || '#888')",
                                            ),
                                        )
                        # M1: per-contact current-fraction
                        # input. Visible only when ≥ 2 contacts
                        # share this polarity (otherwise the
                        # fraction is implicitly 1.0). Empty
                        # value = "equal share within group"
                        # (the FEM driver fills 1/N at solve
                        # time).
                        v3.VTextField(
                            model_value=(
                                "contact_current_fractions[idx]",
                            ),
                            type="number",
                            placeholder="auto",
                            step="0.05",
                            min="0",
                            max="1",
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            suffix="frac",
                            update_modelValue=(
                                "contact_current_fractions = "
                                "  contact_current_fractions.map("
                                "    (f, i) => i === idx "
                                "      ? ("
                                "         $event === '' "
                                "         || $event === null "
                                "         || $event === undefined "
                                "          ? null "
                                "          : Number($event)) "
                                "      : f)"
                            ),
                            v_show=(
                                "(pol === 'anode' "
                                "  && contact_polarities"
                                "       .filter(p => p === 'anode')"
                                "       .length >= 2) "
                                "|| (pol === 'cathode' "
                                "  && contact_polarities"
                                "       .filter(p => p === 'cathode')"
                                "       .length >= 2)",
                            ),
                            style="flex: 0 0 110px;",
                        )
                        # R1.1: Recording-montage badge. Shows
                        # when this contact is bound to a
                        # montage in the active config. The
                        # whole pill is the montage colour with
                        # white "Label ±" text. Hidden when the
                        # contact has no montage binding (the
                        # common case).
                        with html.Div(
                            v_show=(
                                "contact_montage_map[idx]",
                            ),
                            style=(
                                "'flex: 0 0 auto; "
                                "padding: 2px 8px; "
                                "border-radius: 8px; "
                                "font-size: 11px; "
                                "color: white; "
                                "font-weight: 500; "
                                "background: ' + "
                                "(contact_montage_map[idx]"
                                "  ?.color || '#888')",
                            ),
                        ):
                            html.Span(
                                "{{ contact_montage_map[idx]"
                                "    ?.label || '' }} "
                                "{{ contact_montage_map[idx]"
                                "    ?.pole || '' }}",
                            )

                # --- F3.2b: Contact configurations ---
                # A "config" is a named snapshot of the current
                # polarity + current-fraction pattern. Multiple
                # configs can share one design — same physical
                # cuff, different wirings → no remesh, just a
                # different FEM solve. The list below shows every
                # config bound to the currently-selected design;
                # clicking one loads its polarities back into the
                # picker above. "+ Save current" snapshots the
                # picker into a new entry; per-row icons let you
                # rename / overwrite / delete.
                html.H4(
                    "Configurations",
                    v_show=("contact_count > 0",),
                    classes="text-subtitle-2 mt-5 mb-1",
                    style=(
                        "color: #888a90; "
                        "letter-spacing: 0.04em; "
                        "text-transform: uppercase; "
                        "font-size: 10px;"
                    ),
                )
                with html.Div(
                    v_show=("contact_count > 0",),
                    style=(
                        "display: flex; "
                        "flex-direction: column; "
                        "gap: 4px; margin-bottom: 8px;"
                    ),
                ):
                    # One row per config bound to the selected
                    # design. Active config gets a red accent;
                    # click selects, ✎ renames inline, ⟲ overwrites
                    # with current polarities, ✕ deletes.
                    with html.Div(
                        v_for=(
                            "cfg in (configs || [])"
                            "  .filter(c => c.design_id"
                            "    === selected_design_id)"
                        ),
                        key="cfg.cid",
                        style=(
                            "'padding: 6px 10px; "
                            "border-radius: 6px; "
                            "cursor: pointer; "
                            "display: flex; "
                            "align-items: center; "
                            "gap: 6px; '"
                            " + (cfg.cid === selected_config_id "
                            "    ? 'background: #ffeeee; "
                            "       border: 1px solid #e24b4a; "
                            "       border-left: 4px solid "
                            "         #e24b4a;' "
                            "    : 'background: #ffffff; "
                            "       border: 1px solid #e3e3e6; "
                            "       border-left: 4px solid "
                            "         transparent;')",
                        ),
                        click=(
                            "trigger('do_config_select', "
                            "[cfg.cid])"
                        ),
                    ):
                        # Inline rename: when rename_cfg_cid_active
                        # matches this row, swap the label for a
                        # text field; ENTER commits, ESC cancels.
                        html.Span(
                            "{{ cfg.name }}",
                            v_show=(
                                "rename_cfg_cid_active "
                                "!== cfg.cid",
                            ),
                            style=(
                                "flex: 1 1 auto; "
                                "min-width: 0; "
                                "font-size: 12px; "
                                "color: #1f2024; "
                                "white-space: nowrap; "
                                "overflow: hidden; "
                                "text-overflow: ellipsis;"
                            ),
                        )
                        v3.VTextField(
                            v_model=("rename_cfg_value",),
                            v_show=(
                                "rename_cfg_cid_active "
                                "=== cfg.cid",
                            ),
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            autofocus=True,
                            style="flex: 1 1 auto;",
                            keydown_enter=(
                                "trigger('do_config_rename', "
                                "[{cid: cfg.cid, "
                                "  name: rename_cfg_value}]); "
                                "rename_cfg_cid_active = ''"
                            ),
                            keydown_escape=(
                                "rename_cfg_cid_active = ''"
                            ),
                            blur=(
                                "trigger('do_config_rename', "
                                "[{cid: cfg.cid, "
                                "  name: rename_cfg_value}]); "
                                "rename_cfg_cid_active = ''"
                            ),
                        )
                        # ✎ Rename
                        html.Button(
                            "✎",
                            type="button",
                            title="Rename",
                            classes=(
                                "golgi-btn-secondary "
                                "golgi-btn-round "
                                "golgi-btn-sm"
                            ),
                            style="flex: 0 0 auto;",
                            v_show=(
                                "rename_cfg_cid_active "
                                "!== cfg.cid",
                            ),
                            click=(
                                "$event.stopPropagation(); "
                                "rename_cfg_value = cfg.name; "
                                "rename_cfg_cid_active = cfg.cid"
                            ),
                        )
                        # ⟲ Overwrite with current polarities
                        html.Button(
                            "⟲",
                            type="button",
                            title=(
                                "Overwrite with current "
                                "polarities"
                            ),
                            classes=(
                                "golgi-btn-secondary "
                                "golgi-btn-round "
                                "golgi-btn-sm"
                            ),
                            style="flex: 0 0 auto;",
                            v_show=(
                                "rename_cfg_cid_active "
                                "!== cfg.cid",
                            ),
                            click=(
                                "$event.stopPropagation(); "
                                "trigger('do_config_save_current',"
                                " [cfg.cid])"
                            ),
                        )
                        # ✕ Delete
                        html.Button(
                            "✕",
                            type="button",
                            title="Delete config",
                            classes=(
                                "golgi-btn-secondary "
                                "golgi-btn-round "
                                "golgi-btn-sm"
                            ),
                            style="flex: 0 0 auto;",
                            v_show=(
                                "rename_cfg_cid_active "
                                "!== cfg.cid",
                            ),
                            click=(
                                "$event.stopPropagation(); "
                                "trigger('do_config_delete', "
                                "[cfg.cid])"
                            ),
                        )
                # "+ Save current as new config" row
                with html.Div(
                    v_show=("contact_count > 0",),
                    classes="d-flex align-center",
                    style="gap: 6px;",
                ):
                    v3.VTextField(
                        v_model=("new_config_name",),
                        placeholder="New config name…",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        style="flex: 1 1 auto;",
                        keydown_enter=(
                            "trigger('do_config_save_as_new', "
                            "[new_config_name]); "
                            "new_config_name = ''"
                        ),
                    )
                    html.Button(
                        "+ Save current",
                        type="button",
                        classes=(
                            "golgi-btn-primary "
                            "golgi-btn-sm"
                        ),
                        click=(
                            "trigger('do_config_save_as_new', "
                            "[new_config_name]); "
                            "new_config_name = ''"
                        ),
                    )

                # --- R1.1: Recording montages ---
                # Bipolar recording-electrode montages bound to
                # the active config. Each montage names two
                # contacts (+, −) on the cuff. The compute side
                # (R1.2) writes per-contact lead-field files;
                # the analysis tab (R1.4/R1.5) sums per-fiber
                # transmembrane currents against those lead
                # fields to produce the cNAP trace.
                html.H4(
                    "Recording montages",
                    v_show=("contact_count > 0",),
                    classes="text-subtitle-2 mt-5 mb-1",
                    style=(
                        "color: #888a90; "
                        "letter-spacing: 0.04em; "
                        "text-transform: uppercase; "
                        "font-size: 10px;"
                    ),
                )
                with html.Div(
                    v_show=("contact_count > 0",),
                    style=(
                        "display: flex; "
                        "flex-direction: column; "
                        "gap: 4px; margin-bottom: 8px;"
                    ),
                ):
                    # Empty-state hint.
                    html.Div(
                        "No recording montages — click "
                        "+ Add to define a bipolar pair.",
                        v_show=(
                            "(recording_montages || [])"
                            "  .length === 0",
                        ),
                        style=(
                            "font-size: 11px; color: #777; "
                            "padding: 6px 10px; "
                            "font-style: italic;"
                        ),
                    )
                    # One row per existing montage.
                    with html.Div(
                        v_for=(
                            "m in (recording_montages || [])"
                        ),
                        key="m.mid",
                        style=(
                            "padding: 6px 10px; "
                            "border-radius: 6px; "
                            "background: #ffffff; "
                            "border: 1px solid #e3e3e6; "
                            "display: flex; "
                            "align-items: center; gap: 8px;"
                        ),
                    ):
                        # Color dot.
                        html.Div(
                            style=(
                                "'width: 12px; height: 12px; "
                                "border-radius: 50%; "
                                "flex: 0 0 auto; "
                                "background: ' "
                                "+ (m.color || '#888')",
                            ),
                        )
                        # Label + contact-pair summary.
                        with html.Div(
                            style=(
                                "flex: 1 1 auto; "
                                "min-width: 0;"
                            ),
                        ):
                            html.Div(
                                "{{ m.label }}",
                                style=(
                                    "font-size: 12px; "
                                    "font-weight: 500;"
                                ),
                            )
                            html.Div(
                                "{{ m.kind || 'bipolar' }} "
                                "· C{{ m.plus_contact + 1 }}"
                                " (+) — "
                                "C{{ m.minus_contact + 1 }}"
                                " (−)",
                                style=(
                                    "font-size: 10px; "
                                    "color: #777;"
                                ),
                            )
                        v3.VBtn(
                            icon="mdi-pencil-outline",
                            size="small",
                            variant="text",
                            density="compact",
                            click=(
                                "trigger("
                                "'do_montage_open_edit', "
                                "[m.mid])"
                            ),
                        )
                        v3.VBtn(
                            icon="mdi-trash-can-outline",
                            size="small",
                            variant="text",
                            density="compact",
                            click=(
                                "trigger("
                                "'do_montage_delete', "
                                "[m.mid])"
                            ),
                        )
                # + Add montage button (hidden while editor open).
                with html.Div(
                    v_show=(
                        "(contact_count > 0) "
                        "&& (!show_montage_editor)",
                    ),
                    style=(
                        "display: flex; gap: 8px; "
                        "margin-bottom: 8px;"
                    ),
                ):
                    html.Button(
                        "+ Add montage",
                        type="button",
                        classes=(
                            "golgi-btn-primary "
                            "golgi-btn-sm"
                        ),
                        click=(
                            "trigger('do_montage_open_add')"
                        ),
                    )
                # Inline editor — opens above as a small form.
                with html.Div(
                    v_show=("show_montage_editor",),
                    style=(
                        "margin-bottom: 8px; "
                        "padding: 10px; "
                        "background: #f7f7fa; "
                        "border: 1px solid #d8d8df; "
                        "border-radius: 6px; "
                        "display: flex; "
                        "flex-direction: column; "
                        "gap: 8px;"
                    ),
                ):
                    html.Div(
                        "{{ editing_montage_id "
                        "  ? 'Edit montage' "
                        "  : 'New montage' }}",
                        style=(
                            "font-size: 11px; "
                            "color: #555; "
                            "text-transform: uppercase; "
                            "letter-spacing: 0.04em;"
                        ),
                    )
                    v3.VTextField(
                        v_model=("montage_form_label",),
                        label="Label",
                        placeholder="Rec A",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                    v3.VSelect(
                        v_model=("montage_form_plus",),
                        label="+ contact",
                        items=(
                            "(contact_polarities || [])"
                            "  .map((p, i) => ({"
                            "    title: 'C' + (i + 1) "
                            "      + (p === 'off' "
                            "         ? '' "
                            "         : ' (' + p + ')'), "
                            "    value: i"
                            "  }))",
                        ),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                    v3.VSelect(
                        v_model=("montage_form_minus",),
                        label="− contact",
                        items=(
                            "(contact_polarities || [])"
                            "  .map((p, i) => ({"
                            "    title: 'C' + (i + 1) "
                            "      + (p === 'off' "
                            "         ? '' "
                            "         : ' (' + p + ')'), "
                            "    value: i"
                            "  }))",
                        ),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                    html.Div(
                        "{{ montage_form_error }}",
                        v_show=("montage_form_error",),
                        style=(
                            "color: #c0392b; "
                            "font-size: 11px;"
                        ),
                    )
                    with html.Div(
                        style="display: flex; gap: 8px;",
                    ):
                        html.Button(
                            "Save",
                            type="button",
                            classes=(
                                "golgi-btn-primary "
                                "golgi-btn-sm"
                            ),
                            click=(
                                "trigger('do_montage_save', "
                                "[{"
                                "  label: "
                                "    montage_form_label, "
                                "  plus_contact: "
                                "    montage_form_plus, "
                                "  minus_contact: "
                                "    montage_form_minus"
                                "}])"
                            ),
                        )
                        html.Button(
                            "Cancel",
                            type="button",
                            classes=(
                                "golgi-btn-secondary "
                                "golgi-btn-sm"
                            ),
                            click=(
                                "trigger("
                                "'do_montage_cancel_edit'"
                                ")"
                            ),
                        )

                # --- F3.2b: contact sweep generators ---
                # Each button creates a batch of configs for the
                # currently-selected design. Algorithmic
                # generators (bipolar adjacent / all pairs /
                # tripolar / monopolar) fire instantly. Random
                # and manual open dialogs for parameter entry.
                html.H4(
                    "Sweep generators",
                    v_show=("contact_count > 0",),
                    classes="text-subtitle-2 mt-4 mb-1",
                    style=(
                        "color: #888a90; "
                        "letter-spacing: 0.04em; "
                        "text-transform: uppercase; "
                        "font-size: 10px;"
                    ),
                )
                html.Div(
                    "Each generator appends configs to the list "
                    "above for the currently-selected design. "
                    "Existing configs are kept.",
                    v_show=("contact_count > 0",),
                    style=(
                        "font-size: 10px; color: #888a90; "
                        "margin-bottom: 6px; line-height: 1.4;"
                    ),
                )
                with html.Div(
                    v_show=("contact_count > 0",),
                    style=(
                        "display: grid; "
                        "grid-template-columns: 1fr 1fr; "
                        "gap: 6px; margin-bottom: 8px;"
                    ),
                ):
                    html.Button(
                        "Bipolar adjacent",
                        type="button",
                        classes=(
                            "golgi-btn-secondary golgi-btn-sm"
                        ),
                        title=(
                            "N-1 configs: each adjacent contact "
                            "pair as cathode↓ / anode↑"
                        ),
                        click=(
                            "trigger("
                            "  'do_sweep_bipolar_adjacent', "
                            "  [])"
                        ),
                    )
                    html.Button(
                        "All bipolar pairs",
                        type="button",
                        classes=(
                            "golgi-btn-secondary golgi-btn-sm"
                        ),
                        title=(
                            "N×(N-1)/2 configs: every (i<j) pair "
                            "as cathode↓ / anode↑"
                        ),
                        click=(
                            "trigger("
                            "  'do_sweep_bipolar_all_pairs', "
                            "  [])"
                        ),
                    )
                    html.Button(
                        "Tripolar (axial)",
                        type="button",
                        classes=(
                            "golgi-btn-secondary golgi-btn-sm"
                        ),
                        title=(
                            "N-2 configs: centre cathode + "
                            "flanking anodes (0.5/0.5)"
                        ),
                        click=(
                            "trigger("
                            "  'do_sweep_tripolar_axial', "
                            "  [])"
                        ),
                    )
                    html.Button(
                        "Monopolar each",
                        type="button",
                        classes=(
                            "golgi-btn-secondary golgi-btn-sm"
                        ),
                        title=(
                            "N configs: each contact as the "
                            "only cathode; others wired to "
                            "ground"
                        ),
                        click=(
                            "trigger("
                            "  'do_sweep_monopolar_each', "
                            "  [])"
                        ),
                    )
                    html.Button(
                        "Random draws…",
                        type="button",
                        classes=(
                            "golgi-btn-secondary golgi-btn-sm"
                        ),
                        title=(
                            "Random polarity assignments with "
                            "user-chosen N + cathode/anode counts"
                        ),
                        click=(
                            "show_sweep_random_dialog = true"
                        ),
                    )
                    html.Button(
                        "Manual pairs…",
                        type="button",
                        classes=(
                            "golgi-btn-secondary golgi-btn-sm"
                        ),
                        title=(
                            "Enter a list of (cathode, anode) "
                            "pairs by hand — one config per "
                            "pair"
                        ),
                        click=(
                            "show_sweep_manual_dialog = true"
                        ),
                    )
