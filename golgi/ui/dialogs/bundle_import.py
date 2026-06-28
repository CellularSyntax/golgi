# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Histology bundle import dialog.

Compact modal: directory path, scale-bar length picker,
extrusion thickness, "Detect files" + "Import" buttons. Auto-
detected file roles are shown read-only so the user can verify
before committing to the reconstruct.
"""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


_SCALE_PRESETS = [
    {"title": "500 µm", "value": 500.0},
    {"title": "1 mm",   "value": 1000.0},
    {"title": "2 mm",   "value": 2000.0},
    {"title": "5 mm",   "value": 5000.0},
]


def render(
    *,
    do_close_bundle_import_dialog: Callable,
    do_detect_bundle_files: Callable,
    do_run_bundle_import: Callable,
) -> None:
    with v3.VDialog(
        v_model=("show_bundle_import_dialog",),
        max_width=620,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Import histology bundle",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Upload the bundle's files — the slide image "
                    "+ NerveMask + FascMask + ScaleMask (open the "
                    "bundle folder and select all of them). Files "
                    "are auto-detected by filename suffix and the "
                    "mask images skip SAM2 entirely, feeding "
                    "straight into the single-slice polygon "
                    "extrude.",
                    classes="golgi-dialog-body mb-4",
                )

                # --- Bundle file upload (multi-select) -------
                # The user opens the bundle folder and selects all
                # of its files; the bundle_upload_files watcher
                # (actions/bundle_import.py) writes them to a temp
                # dir and auto-detects roles. Replaces the old
                # "paste a directory path" text field + Detect
                # button — re-detection on a scale-bar change still
                # works because the temp dir persists in
                # bundle_dir_path.
                v3.VFileInput(
                    v_model=("bundle_upload_files",),
                    label="upload bundle files",
                    placeholder=(
                        "select every file in the bundle folder"
                    ),
                    prepend_icon="mdi-folder-upload",
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    multiple=True,
                    show_size=True,
                    clearable=True,
                    classes="mb-3",
                )

                # --- Detection error ------------------------
                html.Div(
                    "{{ bundle_detect_error }}",
                    v_show=("bundle_detect_error",),
                    style=(
                        "color: #e24b4a; font-size: 12px; "
                        "margin-bottom: 8px;"
                    ),
                )

                # --- Detected file roles --------------------
                with html.Div(
                    v_show=("bundle_slide_path",),
                    style=(
                        "background: #f5f7fa; "
                        "border-radius: 6px; "
                        "padding: 10px 12px; "
                        "margin-bottom: 12px; "
                        "font-size: 12px; line-height: 1.55;"
                    ),
                ):
                    html.Div(
                        "Detected files",
                        style=(
                            "font-weight: 600; color: #333; "
                            "margin-bottom: 4px;"
                        ),
                    )
                    html.Div(
                        "slide · {{ bundle_slide_path."
                        "split('/').slice(-1)[0] }}",
                        style="color: #555;",
                    )
                    html.Div(
                        "nerve mask · {{ bundle_nerve_path."
                        "split('/').slice(-1)[0] }}",
                        style="color: #555;",
                    )
                    html.Div(
                        "fascicle mask · {{ bundle_fasc_path."
                        "split('/').slice(-1)[0] }}",
                        style="color: #555;",
                    )
                    html.Div(
                        "scale mask · {{ bundle_scale_path."
                        "split('/').slice(-1)[0] }}",
                        style="color: #555;",
                    )

                # --- Scale + thickness --------------------
                with html.Div(
                    classes="d-flex align-center mb-3",
                    style="gap: 12px;",
                ):
                    v3.VSelect(
                        v_model=("bundle_scale_bar_um",),
                        items=("bundle_scale_preset_items",),
                        label="scale bar length",
                        item_title="title",
                        item_value="value",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        update_modelValue=(
                            do_detect_bundle_files
                        ),
                        style="flex: 1;",
                    )
                    v3.VTextField(
                        v_model=("bundle_thickness_mm",),
                        label="thickness (mm)",
                        type="number",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        style="flex: 1;",
                    )

                # --- Nerve cross-section deform (applied at import) ----
                v3.VSelect(
                    v_model=("nerve_deform",),
                    items=(
                        "nerve_deform_items",
                        [{"title": "Round (area-preserving, Duke-style)",
                          "value": "round"},
                         {"title": "None (keep real segmented shape)",
                          "value": "none"}],
                    ),
                    label="nerve cross-section deform",
                    item_title="title", item_value="value",
                    density="compact", hide_details=True,
                    variant="outlined", classes="mt-2",
                )

                # --- Derived pixel pitch readout ----------
                html.Div(
                    "Derived pixel pitch: "
                    "{{ bundle_pixel_pitch_um.toFixed(3) }} "
                    "µm/px",
                    v_show=(
                        "bundle_pixel_pitch_um > 0",
                    ),
                    style=(
                        "font-size: 12px; color: #2e7d32; "
                        "margin-bottom: 12px;"
                    ),
                )

                # --- Status / log line --------------------
                html.Div(
                    "{{ bundle_status }}",
                    v_show=("bundle_status",),
                    style=(
                        "font-size: 12px; color: #555; "
                        "margin-bottom: 4px;"
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
                    click=do_close_bundle_import_dialog,
                )
                html.Button(
                    "Import & extrude",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=do_run_bundle_import,
                    disabled=(
                        "!bundle_slide_path "
                        "|| !(bundle_thickness_mm > 0)",
                    ),
                )
