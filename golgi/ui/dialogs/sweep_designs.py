# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Design sweep wizard (F3.2b).

Clones the currently-selected design into a batch of new designs
that vary along Z translation (cuff_offset_mm), rotation around
the cuff axis (cuff_rot_z_deg), or both in a grid. Each new
design becomes a real state.designs entry (and gets its own mesh
when the user picks it in the Mesh tab's multi-select).
"""
from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render() -> None:
    with v3.VDialog(
        v_model=("show_sweep_designs_dialog",),
        max_width=560,
        persistent=False,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Design sweep — clone the selected cuff",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Generates new designs cloned from "
                    "{{ (designs || [])"
                    "   .find(d => d.eid === selected_design_id)"
                    "   ?.name || '(none)' }}. "
                    "Each clone gets a new id and one Default "
                    "config; the parent stays unchanged. Use the "
                    "Mesh tab's multi-select to mesh the ones "
                    "you want.",
                    classes="golgi-dialog-body mb-3",
                )
                # Axis radio.
                html.Div(
                    "Sweep over:",
                    style=(
                        "font-size: 11px; color: #555; "
                        "margin-bottom: 4px;"
                    ),
                )
                with v3.VRadioGroup(
                    v_model=("sweep_design_axis",),
                    inline=True,
                    density="compact",
                    hide_details=True,
                    classes="mb-3",
                ):
                    v3.VRadio(label="Z translation", value="z")
                    v3.VRadio(label="Twist (rot_z)", value="rot_z")
                    v3.VRadio(
                        label="Grid (Z × rot_z)", value="grid",
                    )
                    # F3.2-M3 — scar thickness sweep. Single
                    # axis; force-enables use_scar on every
                    # clone so the scar shell renders + meshes.
                    v3.VRadio(
                        label="Scar thickness",
                        value="scar",
                    )
                # Z params — shown when axis is z or grid.
                with html.Div(
                    v_show=(
                        "sweep_design_axis === 'z' "
                        "|| sweep_design_axis === 'grid'",
                    ),
                    style="margin-bottom: 12px;",
                ):
                    html.Div(
                        "Z translation (cuff_offset_mm):",
                        style=(
                            "font-size: 11px; color: #555; "
                            "margin-bottom: 4px;"
                        ),
                    )
                    with html.Div(
                        style=(
                            "display: grid; "
                            "grid-template-columns: "
                            "  1fr 1fr 1fr; "
                            "gap: 8px;"
                        ),
                    ):
                        v3.VTextField(
                            label="Start (mm)",
                            v_model_number=(
                                "sweep_design_z_start_mm",
                            ),
                            type="number",
                            step=0.5,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                        v3.VTextField(
                            label="End (mm)",
                            v_model_number=(
                                "sweep_design_z_end_mm",
                            ),
                            type="number",
                            step=0.5,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                        v3.VTextField(
                            label="Steps (N)",
                            v_model_number=(
                                "sweep_design_z_steps",
                            ),
                            type="number",
                            min=1, max=100, step=1,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                # rot_z params — shown when axis is rot_z or grid.
                with html.Div(
                    v_show=(
                        "sweep_design_axis === 'rot_z' "
                        "|| sweep_design_axis === 'grid'",
                    ),
                    style="margin-bottom: 12px;",
                ):
                    html.Div(
                        "Twist around cuff axis "
                        "(cuff_rot_z_deg):",
                        style=(
                            "font-size: 11px; color: #555; "
                            "margin-bottom: 4px;"
                        ),
                    )
                    with html.Div(
                        style=(
                            "display: grid; "
                            "grid-template-columns: "
                            "  1fr 1fr 1fr; "
                            "gap: 8px;"
                        ),
                    ):
                        v3.VTextField(
                            label="Start (°)",
                            v_model_number=(
                                "sweep_design_rot_start_deg",
                            ),
                            type="number",
                            step=5,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                        v3.VTextField(
                            label="End (°)",
                            v_model_number=(
                                "sweep_design_rot_end_deg",
                            ),
                            type="number",
                            step=5,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                        v3.VTextField(
                            label="Steps (N)",
                            v_model_number=(
                                "sweep_design_rot_steps",
                            ),
                            type="number",
                            min=1, max=100, step=1,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                # F3.2-M3 — scar thickness params, shown only on
                # the scar axis. Each step gets its own design
                # clone with use_scar=True + scar_thickness_um
                # set to the linspace value; saline auto-fills
                # the remaining cuff bore.
                with html.Div(
                    v_show=(
                        "sweep_design_axis === 'scar'",
                    ),
                    style="margin-bottom: 12px;",
                ):
                    html.Div(
                        "Scar layer thickness "
                        "(scar_thickness_um):",
                        style=(
                            "font-size: 11px; color: #555; "
                            "margin-bottom: 4px;"
                        ),
                    )
                    with html.Div(
                        style=(
                            "display: grid; "
                            "grid-template-columns: "
                            "  1fr 1fr 1fr; "
                            "gap: 8px;"
                        ),
                    ):
                        v3.VTextField(
                            label="Start (µm)",
                            v_model_number=(
                                "sweep_design_scar_start_um",
                            ),
                            type="number",
                            step=10,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                        v3.VTextField(
                            label="End (µm)",
                            v_model_number=(
                                "sweep_design_scar_end_um",
                            ),
                            type="number",
                            step=10,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                        v3.VTextField(
                            label="Steps (N)",
                            v_model_number=(
                                "sweep_design_scar_steps",
                            ),
                            type="number",
                            min=1, max=100, step=1,
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                        )
                    html.Div(
                        "use_scar is auto-enabled on every "
                        "clone; values that exceed the cuff "
                        "clearance gap are silently clamped "
                        "at mesh time.",
                        style=(
                            "font-size: 10px; color: #888; "
                            "margin-top: 6px; "
                            "line-height: 1.4;"
                        ),
                    )
                # Name prefix.
                v3.VTextField(
                    label="Name prefix",
                    v_model=("sweep_design_name_prefix",),
                    placeholder="Sweep",
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    style="margin-bottom: 8px;",
                )
                # Live count summary.
                html.Div(
                    "Will generate "
                    "{{ sweep_design_axis === 'z' "
                    "    ? sweep_design_z_steps "
                    "  : (sweep_design_axis === 'rot_z' "
                    "      ? sweep_design_rot_steps "
                    "  : (sweep_design_axis === 'scar' "
                    "      ? sweep_design_scar_steps "
                    "      : (sweep_design_z_steps "
                    "         * sweep_design_rot_steps))) "
                    "}} new design(s).",
                    style=(
                        "font-size: 11px; color: #146e3a; "
                        "font-weight: 500; "
                        "background: #eaf6ee; "
                        "border-radius: 4px; "
                        "padding: 6px 10px;"
                    ),
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click="show_sweep_designs_dialog = false",
                )
                html.Button(
                    "Generate",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    disabled=("!selected_design_id",),
                    click="trigger('do_sweep_designs_run', [])",
                )
