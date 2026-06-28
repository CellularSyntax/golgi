# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Electrode designer dialog — ASCENT cuff preset picker +
slider rows + a live offscreen pyvista plotter (pl_cuff)."""
from __future__ import annotations

from typing import Callable

import cuff_designer

from pyvista.trame.ui import plotter_ui
from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    pl_cuff,
    ctrl,
    do_close_cuff_designer: Callable,
    do_apply_cuff_design: Callable,
    do_export_viewport_screenshot: Callable | None = None,
) -> None:
    """Build the electrode-designer VDialog. `pl_cuff` is the
    second offscreen pv.Plotter; `ctrl` is server.controller —
    we set view_cuff_update / view_cuff_reset_camera on it so
    the rebuild-cuff-preview helpers in build_app can drive
    the WebGL view.

    `do_export_viewport_screenshot` (optional) — when supplied the
    Live preview panel gets a floating camera button (F2.3.a) so
    the user can save the cuff design as a PNG without opening the
    workspace view."""
    with v3.VDialog(
        v_model=("show_cuff_designer_dialog",),
        max_width=1400,
        eager=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Electrode designer",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Pick an ASCENT cuff preset, adjust the "
                    "key geometric parameters with the sliders, "
                    "and inspect the live preview on the right. "
                    "Apply mounts the design in the workspace "
                    "viewport at the current cuff frame.",
                    classes="golgi-dialog-body mb-3",
                )
                with html.Div(
                    style=(
                        "display: grid; "
                        "grid-template-columns: "
                        "  minmax(0, 460px) 1fr; "
                        "gap: 24px;"
                    ),
                ):
                    # ---- Left column: preset + slider rows ----
                    with html.Div():
                        v3.VAutocomplete(
                            v_model=("cuff_preset_name",),
                            items=("cuff_preset_items",),
                            item_title="title",
                            item_value="value",
                            label="ASCENT cuff preset",
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            classes="mb-3",
                        )
                        html.Div(
                            "{{ cuff_designer_status }}",
                            style=(
                                "font-size: 11px; "
                                "color: #6c6c70; "
                                "margin-bottom: 12px;"
                            ),
                        )
                        # Two sets of slider rows — one per
                        # preset code — wrapped in v_show so
                        # only the rows for the active preset
                        # render. Each row: label + slider +
                        # numeric + unit suffix.
                        for _code, _params in (
                            cuff_designer
                                .DESIGNER_VISIBLE_PARAMS.items()
                        ):
                            with html.Div(
                                v_show=(
                                    f"cuff_preset_code === "
                                    f"'{_code}'"
                                ),
                            ):
                                for _vp in _params:
                                    _key = (
                                        f"cuff_p_{_vp['name']}"
                                    )
                                    with html.Div(
                                        style=(
                                            "margin-bottom: "
                                            "  10px;"
                                        ),
                                    ):
                                        html.Div(
                                            _vp["label"],
                                            style=(
                                                "font-size: "
                                                "  11px; "
                                                "font-weight: "
                                                "  500; "
                                                "color: "
                                                "  #1f2024; "
                                                "margin-bottom: "
                                                "  2px;"
                                            ),
                                        )
                                        with html.Div(
                                            classes=(
                                                "d-flex "
                                                "align-center"
                                            ),
                                            style=(
                                                "gap: 12px;"
                                            ),
                                        ):
                                            v3.VSlider(
                                                v_model=(_key,),
                                                min=_vp["min"],
                                                max=_vp["max"],
                                                step=_vp[
                                                    "step"
                                                ],
                                                density=(
                                                    "compact"
                                                ),
                                                hide_details=(
                                                    True
                                                ),
                                                color="primary",
                                                thumb_label=(
                                                    False
                                                ),
                                                style=(
                                                    "flex: 1 1 "
                                                    "  auto;"
                                                ),
                                            )
                                            v3.VTextField(
                                                v_model_number=(
                                                    _key,
                                                ),
                                                suffix=(
                                                    _vp["unit"]
                                                ),
                                                type="number",
                                                step=_vp[
                                                    "step"
                                                ],
                                                density=(
                                                    "compact"
                                                ),
                                                hide_details=(
                                                    True
                                                ),
                                                variant=(
                                                    "outlined"
                                                ),
                                                style=(
                                                    "flex: 0 0 "
                                                    "120px; "
                                                    "font-family"
                                                    ": monospace"
                                                    "; "
                                                    "font-size:"
                                                    " 12px;"
                                                ),
                                            )

                        # Fallback when the loaded preset has
                        # no slider entry — only happens if a
                        # new preset code lands without a
                        # matching DESIGNER_VISIBLE_PARAMS row.
                        html.Div(
                            "(no editable parameters for "
                            "this preset)",
                            v_show=(
                                "cuff_preset_code && "
                                "!Object.keys("
                                "  {LN:1, MCT:1}"
                                ").includes(cuff_preset_code)"
                            ),
                            style=(
                                "color: #888a90; "
                                "font-size: 12px; "
                                "padding: 16px 0;"
                            ),
                        )

                    # ---- Right column: live 3-D plotter ----
                    with html.Div():
                        html.Div(
                            "Live preview",
                            style=(
                                "font-size: 11px; "
                                "font-weight: 600; "
                                "color: #888a90; "
                                "text-transform: uppercase; "
                                "letter-spacing: 0.04em; "
                                "margin-bottom: 8px;"
                            ),
                        )
                        with html.Div(
                            style=(
                                "width: 100%; "
                                "height: 520px; "
                                "border: 1px solid #e3e3e6; "
                                "border-radius: 8px; "
                                "overflow: hidden; "
                                "background: white; "
                                "position: relative;"
                            ),
                        ):
                            # Second pyvista plotter view —
                            # the designer's own interactive
                            # canvas. plotter_ui is mounted
                            # eagerly via VDialog(eager=True)
                            # so its WebGL view is ready the
                            # first time the dialog opens.
                            view_cuff = plotter_ui(
                                pl_cuff,
                                interactive_ratio=1,
                                mode="trame",
                                default_server_rendering=False,
                            )
                            ctrl.view_cuff_update = (
                                view_cuff.update
                            )
                            ctrl.view_cuff_reset_camera = (
                                view_cuff.reset_camera
                            )
                            # F2.3.a — floating camera button.
                            # Same UX as the workspace viewport:
                            # bottom-right anchor, triggers a
                            # PNG screenshot of pl_cuff via the
                            # do_export_viewport_screenshot
                            # handler.
                            if do_export_viewport_screenshot is not None:
                                from golgi.ui.components import (
                                    viewport_export_btn,
                                )
                                viewport_export_btn.render(
                                    viewport_id="cuff_designer",
                                    do_export_viewport_screenshot=(
                                        do_export_viewport_screenshot
                                    ),
                                    label="cuff design preview",
                                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=do_close_cuff_designer,
                )
                html.Button(
                    "Apply",
                    type="button",
                    classes=(
                        "golgi-btn-primary golgi-btn-sm"
                    ),
                    click=do_apply_cuff_design,
                )
