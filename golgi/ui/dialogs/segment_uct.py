# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""V1 Phase A.3 — µCT segmentation dialog.

Full-screen modal. Click-to-select-proposal UX: user runs
"everything mode" once, the dialog renders all candidate masks
as coloured overlays on the slice preview, and the user assigns
each proposal a label (background / epi / fascicle / discard)
via VChip buttons in the side panel. Re-labelling re-renders the
preview server-side and pushes a fresh PNG data URL.

State vars the dialog binds to (registered in build_app):
  show_segment_uct_dialog  : bool   — modal v-model
  uct_file_path            : str    — absolute path the user types
                                       or drag-drops in
  uct_voxel_size_um        : float  — pixel pitch; default 1.0;
                                       used by Phase B for the
                                       pixel→mm extrusion step
  uct_stack_loaded         : bool   — gates slice scrubber + Run
  uct_stack_info_html      : str    — '{n_frames} × {h}×{w} {dtype}'
                                       summary line under the path
  uct_slice_idx            : int    — current slice in the stack
  uct_slice_max            : int    — n_frames - 1 (slider max)
  uct_overlay_url          : str    — data:image/png;base64,... URL
                                       the <img> renders
  uct_segmenter_name       : str    — 'MedSAM2' / 'stub'
  uct_segmenter_warning    : str    — non-empty when MedSAM2 fell
                                       back to stub; surfaced as a
                                       VAlert
  uct_proposals_meta       : list   — [{idx,area_px,bbox_str,label}]
  uct_label_counts         : dict   — {'epi': n, 'fascicle': n, ...}
                                       for the count chips in the
                                       header
  uct_busy                 : bool   — disables Run + Save while a
                                       solve is in flight
  uct_status               : str    — bottom-of-dialog log line

Action handler signatures (registered in build_app):
  do_open_segment_uct_dialog()
  do_close_segment_uct_dialog()
  do_load_uct_stack()
  do_change_uct_slice()       — reads state.uct_slice_idx
  do_run_uct_segmentation()
  do_label_uct_proposal(idx, label)
  do_save_uct_segmentation()  — Phase A.4 wires persistence
"""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

# Embedded pyvista viewport for the Step-3 preview. Same
# `plotter_ui` widget the cuff-designer dialog uses, so the
# two embedded plotters share a consistent feel.
from pyvista.trame.ui import plotter_ui


def render(
    *,
    do_close_segment_uct_dialog: Callable,
    do_run_uct_segmentation: Callable,
    do_clear_uct_stack: Callable,
    do_label_uct_proposal: Callable,
    do_generate_epi: Callable,
    do_refine_masks: Callable,
    do_save_uct_segmentation: Callable,
    do_finalize_segmentation: Callable,
    do_toggle_keyframe: Callable,
    do_propagate_from_keyframes: Callable,
    do_recon_next: Callable,
    do_recon_back: Callable,
    do_run_reconstruction: Callable,
    do_run_reconstruction_preview: Callable,
    do_finish_recon: Callable,
    pl_uct_recon=None,
    ctrl=None,
    plotly_module=None,
) -> None:
    """Render the dialog.

    Note: `do_load_uct_stack` is intentionally NOT a render
    kwarg anymore — the upload callback chain
    (golgi_uct_upload.js → /api/uct/upload → on_upload_complete
    in app.py) drives the load, and the dialog has no manual
    "Load" button left. The handler still exists as an action
    (used by the restore-on-open path + the upload callback)
    but isn't reachable from the UI directly.

    The dialog is a 2-step stepper:
      Step 1 — Segment + label slices (per-slice 2D masks).
      Step 2 — Reconstruct 3D nerve volume (single-slice extrude
               or marching-cubes across a range). Saves STLs into
               `<project>/uct/nerve_3d/<timestamp>/`.
    `state.uct_step` (1 or 2) gates which body + CTA render.
    Step transitions go through `do_recon_next` / `do_recon_back`
    so the action layer can prefill defaults (Z-spacing from
    voxel metadata, slice range from annotated coverage)."""
    with v3.VDialog(
        v_model=("show_segment_uct_dialog",),
        # 70 vw × 80 vh modal. Sized to leave the wizard
        # context visible behind on a normal-DPI screen, while
        # still being tall enough for the Step-2 image editor
        # to be usable without aggressive scrolling. Removed
        # the fullscreen + bottom-transition combo — that read
        # as a takeover panel, not a dialog. `width` controls
        # the VDialog's outer box; the VCard inside uses
        # height: 80 vh so the inner flex-column hits the
        # bottom-CTA bar without scroll.
        width="70vw",
        max_width="70vw",
        persistent=True,
    ):
        with v3.VCard(
            classes="d-flex flex-column",
            style="height: 80vh;",
        ):
            # ---- Top toolbar ----
            with v3.VToolbar(
                density="compact",
                color="white",
                classes="border-b",
                style="flex: 0 0 auto;",
            ):
                v3.VToolbarTitle(
                    "Segment µCT slice → extrude as nerve",
                    classes="text-h6",
                )
                with v3.VChip(
                    v_show=("uct_segmenter_name",),
                    size="x-small",
                    variant="tonal",
                    color="primary",
                    classes="ml-3",
                ):
                    html.Span(
                        "engine: {{ uct_segmenter_name }}",
                    )
                v3.VSpacer()
                v3.VBtn(
                    icon="mdi-close",
                    variant="text",
                    click=do_close_segment_uct_dialog,
                )

            # ---- VStepper header ----
            # Three-step pipeline:
            #   "1" Upload  → "2" Segment  → "3" Reconstruct 3D
            #
            # We use VStepper purely for the visual header — the
            # body content is gated via v_show on each existing
            # block below, rather than a VStepperWindow. The
            # image preview is shared between Upload (drag-drop
            # landing zone) and Segment (full editor) and pushing
            # it into per-window mounts would force re-binding
            # the JS mouse / wheel handlers on every transition.
            #
            # Completion gates:
            #   Step 1 done when `uct_stack_loaded`.
            #   Step 2 done when ≥ 1 proposal exists.
            #   Step 3 is terminal; no completion mark.
            # Editable gates protect against jumping forward
            # past prerequisites (e.g. clicking "Segment" with
            # no stack loaded would land on an empty preview).
            # VStepper grows to fill the rest of the 80 vh
            # VCard so its WindowItems have a bounded height.
            # `flex: 0 0 auto` (the original) made VStepper
            # size to its content, which let the Segment-step
            # right-panel chip list push the whole stepper
            # below the 80 vh fold whenever proposals existed.
            # `min-height: 0` + `overflow: hidden` keep child
            # content from leaking out the bottom.
            with v3.VStepper(
                v_model=("uct_step",),
                flat=True,
                hide_actions=True,
                editable=("!uct_busy",),
                style=(
                    "flex: 1 1 auto; "
                    "min-height: 0; "
                    "overflow: hidden; "
                    "display: flex; "
                    "flex-direction: column;"
                ),
            ):
                with v3.VStepperHeader():
                    v3.VStepperItem(
                        title="Upload",
                        value="1",
                        complete=("uct_stack_loaded",),
                        editable=("!uct_busy",),
                    )
                    v3.VDivider()
                    v3.VStepperItem(
                        title="Segment",
                        value="2",
                        complete=(
                            "uct_proposals_meta && "
                            "uct_proposals_meta.length > 0",
                        ),
                        editable=(
                            "uct_stack_loaded && !uct_busy",
                        ),
                    )
                    v3.VDivider()
                    v3.VStepperItem(
                        title="Reconstruct 3D",
                        value="3",
                        editable=(
                            "uct_proposals_meta && "
                            "uct_proposals_meta.length > 0 && "
                            "!uct_busy",
                        ),
                    )

                # ---- VStepperWindow ----
                # VStepperWindow + VStepperWindowItem are what
                # actually swap visible content per step — only
                # the WindowItem whose `value` matches `uct_step`
                # is mounted in the DOM, so we don't end up with
                # all three step bodies stacked on top of each
                # other (the bug the user flagged in the screen-
                # shot). VStepperHeader above just drives the v-
                # model; without VStepperWindow it's purely
                # decorative.
                with v3.VStepperWindow(
                    style=(
                        "flex: 1 1 auto; "
                        "min-height: 0; "
                        "overflow: hidden; "
                        "display: flex; "
                        "flex-direction: column;"
                    ),
                ):
                    # =====================================
                    # Step 1: Upload
                    # =====================================
                    # Centred drop-zone + file picker. Once a
                    # stack lands (do_load_uct_stack), the
                    # action layer auto-advances to "2".
                    with v3.VStepperWindowItem(
                        value="1",
                        style=(
                            "flex: 1 1 auto; "
                            "min-height: 0; "
                            "display: flex; "
                            "flex-direction: column; "
                            "overflow: hidden;"
                        ),
                    ):
                        with html.Div(
                            classes="d-flex flex-row",
                            style=(
                                "flex: 1 1 auto; "
                                "min-height: 0; "
                                "height: 100%;"
                            ),
                        ):
                            # Drop zone (left side).
                            with html.Div(
                                id="golgi-uct-crop-panel",
                                style=(
                                    "flex: 1 1 auto; "
                                    "background: #1f2024; "
                                    "display: flex; "
                                    "align-items: center; "
                                    "justify-content: center; "
                                    "padding: 16px; "
                                    "min-width: 0; "
                                    "position: relative; "
                                    "user-select: none;"
                                ),
                                raw_attrs=[
                                    '@dragover.prevent='
                                    '"uct_drag_active = true"',
                                    '@dragleave.prevent='
                                    '"uct_drag_active = false"',
                                    '@drop.prevent='
                                    '"uct_drag_active = false;'
                                    ' window.golgi_uct_upload('
                                    '$event.dataTransfer'
                                    '.files)"',
                                ],
                            ):
                                html.Div(
                                    "Drop your µCT stack here, "
                                    "or pick one from the "
                                    "panel on the right.",
                                    style=(
                                        "color: #888; "
                                        "font-size: 14px; "
                                        "font-style: italic; "
                                        "max-width: 360px; "
                                        "text-align: center;"
                                    ),
                                )
                                # Drag-overlay (visible while a
                                # file is being dragged over).
                                with html.Div(
                                    v_show=("uct_drag_active",),
                                    style=(
                                        "position: absolute; "
                                        "inset: 8px; "
                                        "border: 3px dashed "
                                        "#e24b4a; "
                                        "border-radius: 12px; "
                                        "background: rgba("
                                        "226, 75, 74, 0.18); "
                                        "display: flex; "
                                        "align-items: center; "
                                        "justify-content: "
                                        "center; "
                                        "color: white; "
                                        "font-size: 20px; "
                                        "font-weight: 600; "
                                        "letter-spacing: "
                                        "0.04em; "
                                        "pointer-events: "
                                        "none; "
                                        "z-index: 10;"
                                    ),
                                ):
                                    v3.VIcon(
                                        "mdi-tray-arrow-down",
                                        size="48",
                                        classes="mr-3",
                                    )
                                    html.Span(
                                        "Drop file to upload",
                                    )
                                # Busy / progress overlay —
                                # visible during the upload XHR
                                # AND during the post-upload
                                # load_stack / DICOM compression
                                # phase. The right-panel progress
                                # bar is easy to miss when the
                                # user's eyes are on the drop
                                # zone; this overlay surfaces the
                                # SAME state vars (uct_upload_*
                                # for upload, busy_msg / busy_log
                                # for post-upload work) right
                                # where the user just dropped
                                # the file. Higher z-index than
                                # the drag overlay so a
                                # mid-upload re-drag doesn't
                                # cover it.
                                with html.Div(
                                    # Gate on upload OR a true
                                    # post-upload load. Critical
                                    # to NOT show this for
                                    # ordinary editing actions
                                    # (segmentation run, crop
                                    # change, slice scroll) —
                                    # those use state.busy /
                                    # state.uct_busy too but the
                                    # user needs the brush /
                                    # paint / label tools to
                                    # remain interactive while
                                    # the global busy lightbox
                                    # is visible elsewhere. The
                                    # `uct_busy` flag is set by
                                    # do_load_uct_stack (initial
                                    # file load) AND by every
                                    # other long action — we
                                    # only want this overlay for
                                    # the FIRST one. Switching
                                    # to a dedicated
                                    # `uct_upload_load_busy`
                                    # flag would be cleaner;
                                    # for now bound to upload
                                    # only, which is the case
                                    # where the user has zero
                                    # other feedback.
                                    v_show=("uct_uploading",),
                                    style=(
                                        "position: absolute; "
                                        "inset: 8px; "
                                        "border-radius: 12px; "
                                        "background: rgba("
                                        "20, 22, 26, 0.92); "
                                        "display: flex; "
                                        "flex-direction: "
                                        "column; "
                                        "align-items: center; "
                                        "justify-content: "
                                        "center; "
                                        "padding: 24px; "
                                        "color: #f5f6fa; "
                                        "z-index: 20; "
                                        # Critical: never
                                        # intercept events
                                        # underneath. When the
                                        # overlay is visible
                                        # the user shouldn't be
                                        # interacting anyway,
                                        # and when it's
                                        # display:none from
                                        # v_show it shouldn't
                                        # matter — but
                                        # belt+suspenders so a
                                        # stale v_show=true
                                        # doesn't lock out the
                                        # crop / brush / label
                                        # tools.
                                        "pointer-events: "
                                        "none; "
                                        "gap: 14px;"
                                    ),
                                ):
                                    # Spinner + headline status.
                                    with html.Div(
                                        style=(
                                            "display: flex; "
                                            "align-items: "
                                            "center; "
                                            "gap: 14px;"
                                        ),
                                    ):
                                        v3.VProgressCircular(
                                            indeterminate=True,
                                            size=42,
                                            width=4,
                                            color="#e24b4a",
                                        )
                                        with html.Div(
                                            style=(
                                                "display: "
                                                "flex; "
                                                "flex-"
                                                "direction: "
                                                "column;"
                                            ),
                                        ):
                                            html.Div(
                                                # Prefer
                                                # uct_upload_-
                                                # status during
                                                # the upload
                                                # phase, fall
                                                # through to
                                                # busy_msg
                                                # (set by
                                                # do_load_uct_-
                                                # stack) when
                                                # the upload's
                                                # done.
                                                "{{ "
                                                "uct_uploading"
                                                " ? "
                                                "(uct_upload_"
                                                "status || "
                                                "'Uploading…')"
                                                " : "
                                                "(busy_msg || "
                                                "'Preparing "
                                                "stack…') }}",
                                                style=(
                                                    "font-"
                                                    "size: "
                                                    "16px; "
                                                    "font-"
                                                    "weight: "
                                                    "600; "
                                                    "color: "
                                                    "#fff;"
                                                ),
                                            )
                                            # Sub-line: live
                                            # tail of busy_log
                                            # (per-step
                                            # messages from
                                            # do_load_uct_stack,
                                            # compress_dicom_-
                                            # series_to_nifti's
                                            # on_log).
                                            html.Div(
                                                "{{ busy_log "
                                                "}}",
                                                v_show=(
                                                    "!uct_"
                                                    "uploading "
                                                    "&& "
                                                    "busy_log",
                                                ),
                                                style=(
                                                    "font-"
                                                    "size: "
                                                    "11px; "
                                                    "color: "
                                                    "#bbb; "
                                                    "margin-"
                                                    "top: "
                                                    "4px; "
                                                    "max-"
                                                    "width: "
                                                    "360px; "
                                                    "white-"
                                                    "space: "
                                                    "nowrap; "
                                                    "overflow:"
                                                    " hidden; "
                                                    "text-"
                                                    "overflow:"
                                                    " "
                                                    "ellipsis;"
                                                ),
                                            )
                                    # Progress bar — only
                                    # meaningful during the
                                    # XHR upload phase
                                    # (post-upload work is
                                    # indeterminate, the
                                    # spinner above carries
                                    # that case).
                                    with html.Div(
                                        v_show=(
                                            "uct_uploading",
                                        ),
                                        style=(
                                            "width: 70%; "
                                            "max-width: "
                                            "440px;"
                                        ),
                                    ):
                                        v3.VProgressLinear(
                                            model_value=(
                                                "uct_upload_"
                                                "progress",
                                            ),
                                            color="#e24b4a",
                                            height=8,
                                            striped=True,
                                            rounded=True,
                                        )
                            # Right: file picker + upload UI.
                            with html.Div(
                                style=(
                                    "flex: 0 0 380px; "
                                    "border-left: 1px solid "
                                    "#e6e6e8; "
                                    "overflow-y: auto; "
                                    "padding: 16px; "
                                    "background: #fafafa;"
                                ),
                            ):
                                # Backend warning, hoisted from
                                # the legacy section.
                                v3.VAlert(
                                    v_show=(
                                        "uct_segmenter_warning",
                                    ),
                                    text=(
                                        "uct_segmenter_warning",
                                    ),
                                    type="info",
                                    density="compact",
                                    variant="tonal",
                                    classes="mb-3",
                                    prepend_icon=(
                                        "mdi-information-"
                                        "outline"
                                    ),
                                )
                                html.H4(
                                    "Upload image stack",
                                    classes=(
                                        "text-subtitle-2 mb-1"
                                    ),
                                )
                                v3.VFileInput(
                                    v_model=(
                                        "uct_file_input",
                                    ),
                                    label=(
                                        "Pick file or DICOM "
                                        "series (TIFF / DICOM / "
                                        "NIfTI / NRRD / "
                                        "MetaImage / Analyze / "
                                        "JPEG2000 / PNG / JPEG)"
                                    ),
                                    accept=(
                                        ".tif,.tiff,.dcm,"
                                        ".dicom,.nii,.nii.gz,"
                                        ".nrrd,.nhdr,.mha,.mhd,"
                                        ".img,.hdr,.jp2,.j2k,"
                                        ".png,.jpg,.jpeg,.bmp"
                                    ),
                                    # `multiple=True` lets the
                                    # user select N .dcm files
                                    # in the system picker. The
                                    # uploader bundles them into
                                    # one multipart POST; the
                                    # server writes them into a
                                    # shared series subdir which
                                    # load_stack auto-detects
                                    # as DICOM.
                                    multiple=True,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    prepend_icon=(
                                        "mdi-image-multiple-"
                                        "outline"
                                    ),
                                    disabled=(
                                        "uct_uploading "
                                        "|| uct_busy",
                                    ),
                                    classes="mb-2",
                                    raw_attrs=[
                                        '@update:modelValue='
                                        '"window'
                                        '.golgi_uct_upload('
                                        '$event)"',
                                    ],
                                )
                                html.Div(
                                    "Streams the file straight "
                                    "to disk via "
                                    "/api/uct/upload (no "
                                    "WebSocket buffer cap → "
                                    "multi-GB OK). Saved under "
                                    "<code>&lt;project&gt;/uct"
                                    "/uploads/</code>.",
                                    style=(
                                        "font-size: 10px; "
                                        "color: #888; "
                                        "margin-bottom: 6px;"
                                    ),
                                )
                                with html.Div(
                                    v_show=("uct_uploading",),
                                    style=(
                                        "margin-bottom: 12px;"
                                    ),
                                ):
                                    v3.VProgressLinear(
                                        model_value=(
                                            "uct_upload_"
                                            "progress",
                                        ),
                                        color="#e24b4a",
                                        height=10,
                                        striped=True,
                                        classes="mb-2",
                                    )
                                    html.Div(
                                        "{{ uct_upload_status "
                                        "}}",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #444;"
                                        ),
                                    )
                                v3.VAlert(
                                    v_show=(
                                        "uct_upload_error",
                                    ),
                                    text=(
                                        "uct_upload_error",
                                    ),
                                    type="error",
                                    density="compact",
                                    variant="tonal",
                                    classes="mb-2",
                                )
                                html.Div(
                                    "Tip: you can also drag "
                                    "and drop the file onto "
                                    "the dark zone on the "
                                    "left.",
                                    style=(
                                        "font-size: 10px; "
                                        "color: #888; "
                                        "font-style: italic; "
                                        "margin-bottom: 8px;"
                                    ),
                                )
                                html.Div(
                                    v_html=(
                                        "uct_stack_info_html",
                                    ),
                                    v_show=("uct_stack_loaded",),
                                    style=(
                                        "font-size: 11px; "
                                        "color: #555; "
                                        "font-family: "
                                        "ui-monospace,Menlo,"
                                        "Consolas,monospace; "
                                        "background: #eee; "
                                        "padding: 4px 8px; "
                                        "border-radius: 4px; "
                                        "margin-top: 8px;"
                                    ),
                                )

                    # =====================================
                    # Step 2: Segment
                    # =====================================
                    # Holds the legacy 2-column body (image
                    # editor + sections 2–5 + Step-2 CTA bar).
                    # Existing widgets stay indented inside
                    # this WindowItem; the now-redundant
                    # v_show conditions on uct_step are kept
                    # because they're harmless when the
                    # surrounding WindowItem already gates
                    # visibility.
                    with v3.VStepperWindowItem(
                        value="2",
                        style=(
                            "flex: 1 1 auto; "
                            "min-height: 0; "
                            "display: flex; "
                            "flex-direction: column; "
                            "overflow: hidden;"
                        ),
                    ):
                        # ---- Steps 1 + 2 body: 2-column layout ----
                        # The image preview (left column) + right controls
                        # column are shared between Upload and Segment so
                        # the user can drag-drop a file from either step.
                        # Content of the right column is gated per-step via
                        # nested v_show wrappers further down.
                        # 2-column body. Explicit max-height
                        # via calc(80vh − ~190px) so neither
                        # column relies on the flex chain
                        # propagating a definite height — that
                        # chain was breaking when Vuetify's
                        # VStepperWindow/Item internal CSS
                        # interleaved, and the right panel
                        # stopped scrolling even though it had
                        # overflow-y: auto.
                        with html.Div(
                            v_show=(
                                "uct_step === '1' || uct_step === '2'",
                            ),
                            classes="d-flex flex-row",
                            style=(
                                "flex: 1 1 auto; "
                                "min-height: 0; "
                                "max-height: calc(80vh - 190px); "
                                "overflow: hidden;"
                            ),
                        ):
                            # ---- Left: image preview + drop zone + crop ----
                            # The container is also the cropper's anchor
                            # (id="golgi-uct-crop-panel" so the JS
                            # findImg() walker finds it without selectors).
                            # Three event channels stacked on this div:
                            #   * @dragover / @dragleave / @drop — file
                            #     upload (HTML5 drag-and-drop)
                            #   * @mousedown.left / @mousemove / @mouseup
                            #     — drag-to-crop via window.golgi_uct_crop_*
                            #   * @wheel.prevent — slice scrubber (mouse-
                            #     wheel / touchpad scroll)
                            # Sized 0 1 55 % — narrower than the
                            # legacy full-width, since you said
                            # the image doesn't need to be huge.
                            with html.Div(
                                id="golgi-uct-crop-panel",
                                style=(
                                    "flex: 0 1 55%; "
                                    "max-width: 55%; "
                                    "background: #1f2024; "
                                    "display: flex; "
                                    "align-items: center; "
                                    "justify-content: center; "
                                    "padding: 16px; "
                                    "min-width: 0; "
                                    "min-height: 0; "
                                    "position: relative; "
                                    "user-select: none;"
                                ),
                                raw_attrs=[
                                    '@dragover.prevent='
                                    '"uct_drag_active = true"',
                                    '@dragleave.prevent='
                                    '"uct_drag_active = false"',
                                    '@drop.prevent='
                                    '"uct_drag_active = false; '
                                    'window.golgi_uct_upload('
                                    '$event.dataTransfer.files)"',
                                    # Mouse crop. The JS handles the preview
                                    # rectangle (DOM only); Vue inline
                                    # expression on mouseup owns the state
                                    # push so the server-side @state.change
                                    # watcher fires reliably (a previous
                                    # revision had the JS call
                                    # trame.state.update directly and the
                                    # server-side watcher didn't always
                                    # see it).
                                    #
                                    # mousedown receives the CURRENT crop
                                    # via uct_crop_x_range / uct_crop_y_range
                                    # so the JS can convert screen pixels
                                    # straight to full-image pixels — the
                                    # displayed <img> shows the cropped
                                    # slice mapped onto the available
                                    # screen area, so screen px → image px
                                    # is a linear interpolation across the
                                    # current crop window.
                                    # Both crop + zoom ranges go to the JS
                                    # (so screenToImage picks zoom-if-set
                                    # else crop) AND the current tool mode
                                    # (so the JS can branch crop/zoom rect
                                    # vs paint/erase stroke without
                                    # re-reading state).
                                    # Single @mousedown / @mouseup for
                                    # all gestures. golgi_uct_crop_start
                                    # dispatches by modifier + button:
                                    #   Alt held OR middle-button → pan
                                    #   left-button (no Alt)       → use
                                    #     uct_tool_mode (crop / zoom /
                                    #     paint / erase / label)
                                    # Right-button drag was tried earlier
                                    # but Vue's `.right` modifier and the
                                    # contextmenu event interfered with
                                    # each other across browsers; Alt-
                                    # drag is the user-preferred gesture
                                    # and works reliably everywhere. The
                                    # cursor flips to "grab" while Alt
                                    # is held and "grabbing" while the
                                    # pan-drag is in flight — see the
                                    # document-level keydown / keyup
                                    # listeners in app.py.
                                    '@mousedown.prevent='
                                    '"window.golgi_uct_crop_start('
                                    '$event, uct_crop_x_range, '
                                    'uct_crop_y_range, '
                                    'uct_zoom_x_range, '
                                    'uct_zoom_y_range, uct_tool_mode)"',
                                    # Mousemove drives two things:
                                    # (1) the crop/zoom drag preview rect
                                    # (only active while a drag is in
                                    # flight) and (2) the brush cursor
                                    # indicator in paint/erase modes.
                                    # Brush cursor also needs the active
                                    # view so its on-screen size matches
                                    # the affected image-pixel radius.
                                    '@mousemove='
                                    '"window.golgi_uct_crop_move($event); '
                                    'window.golgi_uct_brush_cursor('
                                    '$event, uct_tool_mode, '
                                    'uct_brush_radius, uct_crop_x_range, '
                                    'uct_crop_y_range, uct_zoom_x_range, '
                                    'uct_zoom_y_range)"',
                                    # Mouseup dispatches by uct_tool_mode +
                                    # gesture type:
                                    #   crop  + drag  → set crop range
                                    #   crop  + click → cycle proposal label
                                    #   zoom  + drag  → set display zoom
                                    #   zoom  + click → no-op
                                    #   paint + (any) → push paint payload
                                    #   erase + (any) → push erase payload
                                    # Date.now() in the timestamp slot makes
                                    # the watcher re-fire on repeated clicks
                                    # at the same coords.
                                    # Dispatch by gesture type (JS already
                                    # branched on tool mode at mousedown):
                                    #   "crop"   → uct_crop_*  or zoom_*
                                    #   "click"  → label_payload (label
                                    #              mode only)
                                    #   "stroke" → uct_paint_payload (flat
                                    #              [is_paint, ts, x0,y0,
                                    #              x1,y1, ...])
                                    '@mouseup.prevent='
                                    '"window._uct_r = window.golgi_uct_'
                                    'crop_end($event); '
                                    'if (!window._uct_r) { } '
                                    'else if (window._uct_r.type === '
                                    '\'stroke\') { '
                                    '  uct_paint_payload = '
                                    'window._uct_r.flat; '
                                    '} else if (window._uct_r.type === '
                                    '\'crop\') { '
                                    '  if (uct_tool_mode === \'zoom\') { '
                                    '    uct_zoom_x_range = '
                                    '[window._uct_r.x0, window._uct_r.x1]; '
                                    '    uct_zoom_y_range = '
                                    '[window._uct_r.y0, window._uct_r.y1]; '
                                    '  } else if (uct_tool_mode === \'crop\') { '
                                    '    uct_crop_x_range = '
                                    '[window._uct_r.x0, window._uct_r.x1]; '
                                    '    uct_crop_y_range = '
                                    '[window._uct_r.y0, window._uct_r.y1]; '
                                    '  } '
                                    '} else if (window._uct_r.type === '
                                    '\'pan\') { '
                                    '  window.golgi_uct_apply_pan('
                                    'window._uct_r.dx, window._uct_r.dy); '
                                    '} else if (window._uct_r.type === '
                                    '\'click\') { '
                                    '  if (uct_tool_mode === \'label\') { '
                                    '    uct_click_payload = '
                                    '[window._uct_r.x, window._uct_r.y, '
                                    'Date.now()]; '
                                    '  } '
                                    '}"',
                                    # Mousewheel scrubs slices with
                                    # accumulated-delta throttling. Native
                                    # wheel events fire ~10× per touchpad
                                    # gesture so 1-event-per-slice felt
                                    # like jumping. The JS accumulates
                                    # deltaY and only emits a step (= ±1
                                    # slice idx) once |Σ deltaY| ≥ 100,
                                    # which is one notch detent on a
                                    # discrete wheel.
                                    '@wheel.prevent='
                                    '"window._uct_w = window.golgi_uct_'
                                    'wheel_step($event, uct_slice_idx, '
                                    'uct_slice_max); '
                                    'if (window._uct_w !== null) { '
                                    'uct_slice_idx = window._uct_w; '
                                    '}"',
                                    # Double-click anywhere on the image
                                    # resets BOTH crop and zoom back to
                                    # the full-image view. Routed
                                    # through a JS function so the
                                    # state update goes via
                                    # window.trame.state.update (which
                                    # is the path we know fires
                                    # reliably for big-int / array
                                    # patches). The previous inline
                                    # `uct_crop_x_range = [0, 0]; …`
                                    # version sometimes didn't trigger
                                    # the watcher — Vue inline
                                    # multi-statement reactivity is a
                                    # known sharp edge on array
                                    # assignments.
                                    '@dblclick.prevent='
                                    '"window.golgi_uct_reset_crop()"',
                                ],
                            ):
                                # Empty state — instructs the user before
                                # they've picked a file.
                                html.Div(
                                    "Drop file here, or pick one from the "
                                    "right panel.",
                                    v_show=("!uct_stack_loaded",),
                                    style=("color: #888; "
                                            "font-size: 14px; "
                                            "font-style: italic;"),
                                )
                                # width/height 100% + object-fit:contain
                                # forces the image to fill the available
                                # area while preserving aspect ratio, so a
                                # small zoomed PNG (say 400×250 of native)
                                # actually magnifies to the full panel
                                # instead of just shrinking. image-rendering
                                # 'pixelated' keeps the magnification sharp
                                # at pixel boundaries when zoomed deep.
                                html.Img(
                                    v_show=("uct_stack_loaded",),
                                    src=("uct_overlay_url",),
                                    style=(
                                        "width: 100%; "
                                        "height: 100%; "
                                        "object-fit: contain; "
                                        "background: black; "
                                        "image-rendering: pixelated;"
                                    ),
                                    # @load fires once per src change
                                    # (i.e., every slice / crop / zoom),
                                    # which is exactly when the
                                    # displayed bbox may have changed.
                                    # The data-* bindings let the
                                    # MutationObserver in
                                    # golgi_uct_scalebar_update re-run
                                    # the layout whenever the user
                                    # edits the voxel-size field or
                                    # the slice's native dimensions
                                    # change (e.g., a new crop).
                                    raw_attrs=[
                                        '@load='
                                        '"window.golgi_uct_scalebar'
                                        '_update($event.target, '
                                        'uct_voxel_size_um, '
                                        'uct_image_orig_w)"',
                                        ':data-voxel-um='
                                        '"uct_voxel_size_um"',
                                        ':data-orig-w='
                                        '"uct_image_orig_w"',
                                    ],
                                )
                                # 1 mm physical-scale bar overlay.
                                # Width / position is computed by
                                # `window.golgi_uct_scalebar_update`
                                # using the IMG's actual
                                # getBoundingClientRect — that's the
                                # only way to get the post-object-fit-
                                # contain letterboxed image bbox
                                # without re-implementing object-fit
                                # in CSS. The JS also installs a
                                # ResizeObserver on the panel so the
                                # bar resizes when the user resizes
                                # the dialog / window. Visibility is
                                # controlled by JS (`display: none`
                                # when voxel_um == 0 or no slice
                                # loaded) instead of v_show, so the
                                # same JS path owns both the geometry
                                # and the show/hide logic.
                                with html.Div(
                                    classes="golgi-uct-scalebar",
                                    style=(
                                        "position: absolute; "
                                        "display: none; "
                                        "flex-direction: column; "
                                        # Bar sits flush-right at
                                        # the displayed-image's
                                        # bottom-right corner, so
                                        # right-align its contents
                                        # (the bar + the "1 mm"
                                        # label).
                                        "align-items: flex-end; "
                                        "gap: 2px; "
                                        "pointer-events: none; "
                                        "z-index: 5;"
                                    ),
                                ):
                                    # White tick. Black drop-shadow
                                    # gives readable contrast on the
                                    # light µCT background without
                                    # needing a chip behind it.
                                    html.Div(
                                        classes=(
                                            "golgi-uct-scalebar-tick"
                                        ),
                                        style=(
                                            "height: 3px; "
                                            "background: white; "
                                            "border-radius: 1px; "
                                            "filter: drop-shadow("
                                            "  0 0 1.5px "
                                            "  rgba(0,0,0,0.9));"
                                        ),
                                    )
                                    html.Div(
                                        "1 mm",
                                        style=(
                                            "color: white; "
                                            "font-size: 11px; "
                                            "font-weight: 700; "
                                            "letter-spacing: "
                                            "  0.03em; "
                                            "line-height: 1; "
                                            "font-family: "
                                            "  ui-monospace, "
                                            "  Menlo, "
                                            "  Consolas, "
                                            "  monospace; "
                                            "text-shadow: "
                                            "  0 0 2px "
                                            "    rgba(0,0,0,0.95), "
                                            "  0 0 2px "
                                            "    rgba(0,0,0,0.95);"
                                        ),
                                    )
                                # Drop-zone overlay — visible while dragging
                                # a file over the panel. pointer-events:none
                                # so it doesn't intercept the drop event
                                # itself (which would land on the overlay
                                # and not the parent's @drop handler).
                                with html.Div(
                                    v_show=("uct_drag_active",),
                                    style=(
                                        "position: absolute; "
                                        "inset: 8px; "
                                        "border: 3px dashed #e24b4a; "
                                        "border-radius: 12px; "
                                        "background: "
                                        "  rgba(226, 75, 74, 0.18); "
                                        "display: flex; "
                                        "align-items: center; "
                                        "justify-content: center; "
                                        "color: white; "
                                        "font-size: 20px; "
                                        "font-weight: 600; "
                                        "letter-spacing: 0.04em; "
                                        "pointer-events: none; "
                                        "z-index: 10;"
                                    ),
                                ):
                                    v3.VIcon(
                                        "mdi-tray-arrow-down",
                                        size="48",
                                        classes="mr-3",
                                    )
                                    html.Span("Drop file to upload")

                            # ---- Right: controls panel ----
                            # `flex: 1 1 auto` so the panel
                            # expands to fill whatever width the
                            # image panel (`0 0 55%`) leaves —
                            # no more dead whitespace on the
                            # right edge of the dialog. Width
                            # ends up roughly 45 % of the 70 vw
                            # dialog, which gives the FASCICLE /
                            # BG / EPI / NONE chip rows enough
                            # room to breathe. Vertical sizing
                            # still relies on `max-height: 100%`
                            # + `overflow-y: auto` so the chip
                            # list scrolls when it exceeds the
                            # 80 vh dialog (the bug from the
                            # previous round).
                            with html.Div(
                                style=(
                                    "flex: 1 1 auto; "
                                    "min-width: 0; "
                                    "border-left: 1px solid #e6e6e8; "
                                    "overflow-y: auto; "
                                    "overflow-x: hidden; "
                                    "max-height: 100%; "
                                    "min-height: 0; "
                                    "padding: 16px; "
                                    "background: #fafafa;"
                                ),
                            ):
                                # Segmenter-backend fallback warning. Only
                                # shown when MedSAM2 wasn't available.
                                v3.VAlert(
                                    v_show=("uct_segmenter_warning",),
                                    text=("uct_segmenter_warning",),
                                    type="info",
                                    density="compact",
                                    variant="tonal",
                                    classes="mb-3",
                                    prepend_icon="mdi-information-outline",
                                )

                                # ---- 1. File loader (visible in Step 1
                                # only; the user returns to Step 1 to upload
                                # a different file at any time). v_show is
                                # added per-widget below rather than on a
                                # wrapping div — keeps the indentation of
                                # the existing section unchanged so the diff
                                # stays surgical.
                                html.H4(
                                    "1. Upload image stack",
                                    v_show=("uct_step === '1'",),
                                    classes="text-subtitle-2 mb-1",
                                )
                                # Browser → server file picker. The XHR
                                # uploader (window.golgi_uct_upload, served
                                # via golgi_uct_upload.js) bypasses trame's
                                # WS so multi-GB stacks stream straight to
                                # disk in 64 kB chunks. Server-side route
                                # writes into <project>/uct/uploads/ and
                                # the on_upload_complete callback fires
                                # do_load_uct_stack automatically.
                                v3.VFileInput(
                                    v_model=("uct_file_input",),
                                    v_show=("uct_step === '1'",),
                                    label=(
                                        "Pick file or DICOM series "
                                        "(TIFF / DICOM / NIfTI / "
                                        "NRRD / MetaImage / Analyze / "
                                        "JPEG2000 / PNG / JPEG)"
                                    ),
                                    accept=(
                                        ".tif,.tiff,.dcm,.dicom,.nii,"
                                        ".nii.gz,.nrrd,.nhdr,.mha,.mhd,"
                                        ".img,.hdr,.jp2,.j2k,.png,.jpg,"
                                        ".jpeg,.bmp"
                                    ),
                                    multiple=True,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    prepend_icon="mdi-image-multiple-outline",
                                    disabled=("uct_uploading || uct_busy",),
                                    classes="mb-2",
                                    raw_attrs=[
                                        '@update:modelValue='
                                        '"window.golgi_uct_upload($event)"',
                                    ],
                                )
                                html.Div(
                                    "Streams the file straight to disk via "
                                    "/api/uct/upload (no WebSocket buffer "
                                    "cap → multi-GB OK). Saved under your "
                                    "project's uct/uploads/ folder.",
                                    v_show=("uct_step === '1'",),
                                    style=("font-size: 10px; color: #888; "
                                            "margin-bottom: 6px;"),
                                )
                                # Upload progress bar — matches the F2.2
                                # import-study style (golgi red + striped +
                                # height 10). Determinate model_value
                                # because the JS XHR uploader pushes
                                # bytes-on-the-wire progress, unlike the
                                # study upload which can only do
                                # indeterminate. Cleared by the JS on
                                # success/failure.
                                with html.Div(
                                    v_show=(
                                        "uct_uploading && uct_step === '1'",
                                    ),
                                    style="margin-bottom: 12px;",
                                ):
                                    v3.VProgressLinear(
                                        model_value=(
                                            "uct_upload_progress",
                                        ),
                                        color="#e24b4a",
                                        height=10,
                                        striped=True,
                                        classes="mb-2",
                                    )
                                    html.Div(
                                        "{{ uct_upload_status }}",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #444;"
                                        ),
                                    )
                                v3.VAlert(
                                    v_show=(
                                        "uct_upload_error "
                                        "&& uct_step === '1'",
                                    ),
                                    text=("uct_upload_error",),
                                    type="error",
                                    density="compact",
                                    variant="tonal",
                                    classes="mb-2",
                                )
                                # Hint: also-drag-to-upload affordance.
                                html.Div(
                                    "Tip: you can also drag and drop the "
                                    "file straight onto the dark preview "
                                    "area on the left.",
                                    v_show=("uct_step === '1'",),
                                    style=("font-size: 10px; "
                                            "color: #888; "
                                            "font-style: italic; "
                                            "margin-bottom: 8px;"),
                                )
                                # Stack metadata line + Clear button
                                # — visible after load. Flex row so the
                                # chip and the "Clear" affordance stay on
                                # the same line; the chip expands to take
                                # available width.
                                with html.Div(
                                    classes=(
                                        "d-flex align-center"
                                    ),
                                    v_show=("uct_stack_loaded",),
                                    style=(
                                        "gap: 8px; "
                                        "margin-bottom: 12px;"
                                    ),
                                ):
                                    html.Div(
                                        v_html=(
                                            "uct_stack_info_html",
                                        ),
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; "
                                            "font-family: "
                                            "  ui-monospace,"
                                            "  Menlo,Consolas,"
                                            "  monospace; "
                                            "background: #eee; "
                                            "padding: 4px 8px; "
                                            "border-radius: 4px; "
                                            "flex: 1 1 auto;"
                                        ),
                                    )
                                    v3.VBtn(
                                        "Clear",
                                        prepend_icon=(
                                            "mdi-close-circle-"
                                            "outline"
                                        ),
                                        variant="text",
                                        density="compact",
                                        color="error",
                                        disabled=(
                                            "uct_busy "
                                            "|| uct_propagation_"
                                            "busy "
                                            "|| uct_uploading",
                                        ),
                                        click=do_clear_uct_stack,
                                    )

                                # ---- 2. Voxel size + slice scrubber ----
                                with html.Div(v_show=(
                                    "uct_stack_loaded "
                                    "&& uct_step === '2'",
                                )):
                                    html.H4(
                                        "2. Slice + voxel size",
                                        classes=(
                                            "text-subtitle-2 mt-2 mb-1"
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center"
                                        ),
                                        style="gap: 8px;",
                                    ):
                                        html.Span(
                                            "voxel µm",
                                            style=("font-size: 11px; "
                                                    "color: #555;"),
                                        )
                                        v3.VTextField(
                                            v_model_number=(
                                                "uct_voxel_size_um",
                                            ),
                                            type="number",
                                            step="0.1",
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style="max-width: 110px;",
                                        )
                                    html.Div(
                                        "Pixel pitch (µm). Used to scale the "
                                        "extruded nerve to mm in Phase B.",
                                        style=("font-size: 10px; "
                                                "color: #888; "
                                                "margin: 2px 0 8px 0;"),
                                    )
                                    # Slice scrubber. v-slider's @end event
                                    # fires when the user releases the thumb
                                    # → re-render slice. We don't bind
                                    # @update:modelValue so dragging doesn't
                                    # spam re-renders on every pixel.
                                    with html.Div(
                                        classes="d-flex align-center",
                                        style="gap: 8px;",
                                    ):
                                        html.Span(
                                            "slice",
                                            style=("font-size: 11px; "
                                                    "color: #555;"),
                                        )
                                        v3.VSlider(
                                            v_model=("uct_slice_idx",),
                                            min=0, step=1,
                                            max=("uct_slice_max",),
                                            density="compact",
                                            hide_details=True,
                                            thumb_label=True,
                                            style="flex: 1 1 auto;",
                                            disabled=("uct_busy",),
                                        )

                                # ---- 3. Tools ----
                                # Toolbar: crop / zoom / paint / erase. The
                                # selected mode flips uct_tool_mode and the
                                # image-panel's @mousedown / @mouseup
                                # expressions dispatch accordingly. Crop is
                                # persistent across slice scrolls so the
                                # user can scrub the stack while keeping
                                # the same view window + segmentation.
                                with html.Div(v_show=(
                                    "uct_stack_loaded "
                                    "&& uct_step === '2'",
                                )):
                                    html.H4(
                                        "3. Tools",
                                        classes=(
                                            "text-subtitle-2 mt-3 mb-1"
                                        ),
                                    )
                                    # 4-button mode picker. VBtnToggle drives
                                    # the single-select with the active
                                    # button getting the "flat" filled
                                    # variant. We use VBtn instead of v_for
                                    # so each icon can be inlined directly
                                    # — only 4 buttons, no real value in
                                    # iterating.
                                    # `mandatory=False` lets the legend
                                    # pills push uct_tool_mode to 'label'
                                    # without forcing one of the 4 visible
                                    # toolbar buttons to stay selected.
                                    # When tool_mode === 'label' none of
                                    # the 4 buttons show as active.
                                    with v3.VBtnToggle(
                                        v_model=("uct_tool_mode",),
                                        mandatory=False,
                                        density="compact",
                                        color="primary",
                                        divided=True,
                                        classes="mb-2",
                                        style="width: 100%;",
                                    ):
                                        v3.VBtn(
                                            "Crop",
                                            value="crop",
                                            prepend_icon="mdi-crop",
                                            size="small",
                                            style="flex: 1 1 auto;",
                                        )
                                        v3.VBtn(
                                            "Zoom",
                                            value="zoom",
                                            prepend_icon=(
                                                "mdi-magnify-plus-outline"
                                            ),
                                            size="small",
                                            style="flex: 1 1 auto;",
                                        )
                                        v3.VBtn(
                                            "Paint",
                                            value="paint",
                                            prepend_icon="mdi-brush",
                                            size="small",
                                            style="flex: 1 1 auto;",
                                        )
                                        v3.VBtn(
                                            "Erase",
                                            value="erase",
                                            prepend_icon="mdi-eraser",
                                            size="small",
                                            style="flex: 1 1 auto;",
                                        )
                                    # Mode-specific help line.
                                    html.Div(
                                        "{{ "
                                        "uct_tool_mode === 'crop' ? "
                                        "  'Drag to set crop. Click a "
                                        "label pill below to switch to "
                                        "relabel mode.' : "
                                        "uct_tool_mode === 'zoom' ? "
                                        "  'Drag to magnify. Double-click "
                                        "to reset.' : "
                                        "uct_tool_mode === 'paint' ? "
                                        "  'Click to paint a brush circle "
                                        "of the paint label. Strokes grow "
                                        "the largest existing mask of that "
                                        "label, or create a new one.' : "
                                        "uct_tool_mode === 'erase' ? "
                                        "  'Click to erase a brush circle "
                                        "from any mask under the pointer.' "
                                        ": uct_tool_mode === 'label' ? "
                                        "  'Click on a mask to assign the "
                                        "active pill\\'s label. Switch "
                                        "back to Crop / Zoom in the "
                                        "toolbar when done.' "
                                        ": 'Pick a tool above, or click a "
                                        "label pill below to start "
                                        "labelling.' "
                                        "}}",
                                        style=("font-size: 10px; "
                                                "color: #888; "
                                                "margin-bottom: 6px;"),
                                    )
                                    # Paint/erase secondary controls — only
                                    # shown in those modes.
                                    with html.Div(
                                        v_show=(
                                            "uct_tool_mode === 'paint' "
                                            "|| uct_tool_mode === 'erase'",
                                        ),
                                        classes="mb-2",
                                        style=(
                                            "border: 1px solid #e6e6e8; "
                                            "border-radius: 6px; "
                                            "padding: 8px; "
                                            "background: #fafafa;"
                                        ),
                                    ):
                                        # Target label picker — only matters
                                        # for paint mode; in erase mode the
                                        # brush hits any mask under the
                                        # cursor regardless.
                                        with html.Div(
                                            v_show=(
                                                "uct_tool_mode === 'paint'",
                                            ),
                                            classes="mb-2",
                                        ):
                                            v3.VSelect(
                                                v_model=("uct_paint_label",),
                                                items=(
                                                    "uct_paint_label_items",
                                                ),
                                                item_title="title",
                                                item_value="value",
                                                label="paint label",
                                                density="compact",
                                                hide_details=True,
                                                variant="outlined",
                                            )
                                        with html.Div(
                                            classes="d-flex align-center",
                                            style="gap: 8px;",
                                        ):
                                            html.Span(
                                                "brush",
                                                style=(
                                                    "font-size: 11px; "
                                                    "color: #555; "
                                                    "min-width: 36px;"
                                                ),
                                            )
                                            v3.VSlider(
                                                v_model=(
                                                    "uct_brush_radius",
                                                ),
                                                min=2, max=64, step=1,
                                                density="compact",
                                                hide_details=True,
                                                thumb_label=True,
                                                style="flex: 1 1 auto;",
                                            )
                                            html.Span(
                                                "{{ uct_brush_radius }} "
                                                "px",
                                                style=(
                                                    "font-size: 10px; "
                                                    "color: #888; "
                                                    "font-family: "
                                                    "ui-monospace,Menlo,"
                                                    "Consolas,monospace; "
                                                    "min-width: 44px;"
                                                ),
                                            )
                                    # Crop / zoom range readout + Reset.
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center "
                                            "justify-space-between"
                                        ),
                                        style="gap: 8px;",
                                    ):
                                        html.Div(
                                            # Show the active window
                                            # depending on mode.
                                            "{{ uct_tool_mode === 'zoom' "
                                            "&& uct_zoom_x_range[1] > "
                                            "uct_zoom_x_range[0] ? "
                                            "  'zoom · x: ' + "
                                            "uct_zoom_x_range[0] + ' → ' + "
                                            "uct_zoom_x_range[1] + ' · y: ' + "
                                            "uct_zoom_y_range[0] + ' → ' + "
                                            "uct_zoom_y_range[1] + ' px' : "
                                            "  'crop · x: ' + "
                                            "uct_crop_x_range[0] + ' → ' + "
                                            "uct_crop_x_range[1] + ' · y: ' + "
                                            "uct_crop_y_range[0] + ' → ' + "
                                            "uct_crop_y_range[1] + ' px' "
                                            "}}",
                                            style=(
                                                "font-size: 10px; "
                                                "color: #888; "
                                                "font-family: ui-monospace,"
                                                "Menlo,Consolas,monospace;"
                                            ),
                                        )
                                        # Reset button — drops zoom first if
                                        # one is active, else resets crop to
                                        # full extent.
                                        html.Button(
                                            "{{ uct_tool_mode === 'zoom' "
                                            "|| uct_zoom_x_range[1] > "
                                            "uct_zoom_x_range[0] "
                                            "  ? 'Reset zoom' "
                                            "  : 'Reset crop' }}",
                                            type="button",
                                            classes=(
                                                "golgi-btn-secondary "
                                                "golgi-btn-sm"
                                            ),
                                            click=(
                                                "if (uct_zoom_x_range[1] > "
                                                "uct_zoom_x_range[0]) { "
                                                "  uct_zoom_x_range = "
                                                "[0, 0]; "
                                                "  uct_zoom_y_range = "
                                                "[0, 0]; "
                                                "} else { "
                                                "  uct_crop_x_range = "
                                                "[0, uct_crop_max_x]; "
                                                "  uct_crop_y_range = "
                                                "[0, uct_crop_max_y]; "
                                                "}"
                                            ),
                                        )

                                # ---- 4. Segment ----
                                with html.Div(v_show=(
                                    "uct_stack_loaded "
                                    "&& uct_step === '2'",
                                )):
                                    html.H4(
                                        "4. Segment",
                                        classes=(
                                            "text-subtitle-2 mt-3 mb-1"
                                        ),
                                    )
                                    # Scope picker — current slice only vs
                                    # entire stack. Stack-wide can take
                                    # 1-3 h on CPU SAM2 for ~64 slices;
                                    # current-only is the fast iteration
                                    # path.
                                    with v3.VBtnToggle(
                                        v_model=("uct_segment_scope",),
                                        mandatory=True,
                                        density="compact",
                                        color="primary",
                                        divided=True,
                                        classes="mb-2",
                                        style="width: 100%;",
                                    ):
                                        v3.VBtn(
                                            "Current slice",
                                            value="current",
                                            prepend_icon=(
                                                "mdi-image-outline"
                                            ),
                                            size="small",
                                            style="flex: 1 1 auto;",
                                        )
                                        v3.VBtn(
                                            "All slices",
                                            value="all",
                                            prepend_icon=(
                                                "mdi-image-multiple-"
                                                "outline"
                                            ),
                                            size="small",
                                            style="flex: 1 1 auto;",
                                        )
                                    # Sweep step — only meaningful in
                                    # "all slices" scope. Lets the user
                                    # segment a sparse subset (every Nth
                                    # frame) and rely on Step-3's ZOH
                                    # fill to interpolate the gaps when
                                    # building the 3D nerve.
                                    with html.Div(
                                        v_show=(
                                            "uct_segment_scope "
                                            "=== 'all'",
                                        ),
                                        classes="d-flex align-center",
                                        style=(
                                            "gap: 8px; "
                                            "margin-bottom: 6px;"
                                        ),
                                    ):
                                        html.Span(
                                            "every Nth slice",
                                            style=(
                                                "font-size: 11px; "
                                                "color: #555; "
                                                "min-width: 110px;"
                                            ),
                                        )
                                        v3.VTextField(
                                            v_model_number=(
                                                "uct_segment_step",
                                            ),
                                            type="number",
                                            min=1,
                                            step=1,
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style="max-width: 90px;",
                                        )
                                        html.Div(
                                            "(1 = every slice; 5 = "
                                            "every 5th; gaps are "
                                            "ZOH-filled at 3D-"
                                            "reconstruct time)",
                                            style=(
                                                "font-size: 10px; "
                                                "color: #888;"
                                            ),
                                        )
                                    # Backend picker — auto / MedSAM2 /
                                    # vanilla SAM2 / Otsu stub. The
                                    # _on_uct_backend_change watcher drops
                                    # the cached segmenter so the next
                                    # Segment click rebuilds with the new
                                    # backend.
                                    v3.VSelect(
                                        v_model=("uct_backend_choice",),
                                        items=("uct_backend_items",),
                                        item_title="title",
                                        item_value="value",
                                        label="engine",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        classes="mb-2",
                                        disabled=("uct_busy",),
                                    )
                                    # CLAHE pre-processing toggle.
                                    v3.VCheckbox(
                                        v_model=("uct_clahe",),
                                        label=(
                                            "Apply CLAHE contrast "
                                            "enhancement before segmenting"
                                        ),
                                        density="compact",
                                        hide_details=True,
                                        color="primary",
                                        classes="mb-1",
                                    )
                                    html.Div(
                                        "Adaptive histogram equalisation "
                                        "lifts low local contrast (clip 2.0, "
                                        "tile 8x8). Helps when MedSAM2 "
                                        "returns 0 masks or misses faint "
                                        "fascicles.",
                                        style=("font-size: 10px; "
                                                "color: #888; "
                                                "margin-bottom: 8px;"),
                                    )
                                    with html.Button(
                                        type="button",
                                        classes=(
                                            "golgi-cta-wrapper "
                                            "golgi-cta-wrapper-block mb-2"
                                        ),
                                        disabled=("uct_busy",),
                                        click=do_run_uct_segmentation,
                                    ):
                                        html.Span(
                                            classes="golgi-cta-spinner",
                                        )
                                        with html.Span(
                                            classes="golgi-cta-inner",
                                        ):
                                            html.Span("▶ Segment")
                                    html.Div(
                                        "MedSAM2 (or the stub fallback) "
                                        "proposes all distinct objects in "
                                        "the cropped view. Assign each "
                                        "blob a label below.",
                                        style=("font-size: 10px; "
                                                "color: #888; "
                                                "margin-bottom: 8px;"),
                                    )

                                    # 2D mask cleanup APPLIED AT
                                    # SEGMENT TIME. Each proposal
                                    # returned by the segmenter (or by
                                    # SAM2 video propagation in 4b
                                    # below) is run through
                                    # cleanup_2d_mask before being
                                    # added to the per-slice cache —
                                    # so the overlay the user sees
                                    # already reflects these knobs.
                                    # Drops small foreground speckles
                                    # (false positives), fills small
                                    # holes inside the foreground
                                    # (false negatives), and seals
                                    # thin gaps via morphological
                                    # closing. Both pixel knobs at
                                    # source-image resolution; 0
                                    # disables that direction.
                                    # Typical values: 30-100 px for
                                    # either knob on a 1024² µCT
                                    # slice — small enough to leave
                                    # thin fascicle arms intact, big
                                    # enough to wipe 1-10 px noise.
                                    html.Div(
                                        "2D mask cleanup",
                                        classes=(
                                            "text-subtitle-2 mt-3 "
                                            "mb-1"
                                        ),
                                        style=(
                                            "color: #888a90; "
                                            "letter-spacing: 0.04em; "
                                            "text-transform: "
                                            "uppercase; "
                                            "font-size: 10px;"
                                        ),
                                    )
                                    with html.Div(
                                        classes="d-flex align-center",
                                        style=(
                                            "gap: 8px; "
                                            "margin-bottom: 4px;"
                                        ),
                                    ):
                                        html.Div(
                                            "Drop foreground "
                                            "speckles < N px (0 = off)",
                                            style=(
                                                "font-size: 12px; "
                                                "min-width: 240px;"
                                            ),
                                        )
                                        v3.VTextField(
                                            v_model=(
                                                "uct_recon_clean"
                                                "_min_component_px",
                                            ),
                                            type="number",
                                            min=0,
                                            step=10,
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style=(
                                                "max-width: 140px;"
                                            ),
                                        )
                                    with html.Div(
                                        classes="d-flex align-center",
                                        style=(
                                            "gap: 8px; "
                                            "margin-bottom: 4px;"
                                        ),
                                    ):
                                        html.Div(
                                            "Fill holes < N px "
                                            "inside foreground "
                                            "(0 = off)",
                                            style=(
                                                "font-size: 12px; "
                                                "min-width: 240px;"
                                            ),
                                        )
                                        v3.VTextField(
                                            v_model=(
                                                "uct_recon_clean"
                                                "_min_hole_px",
                                            ),
                                            type="number",
                                            min=0,
                                            step=10,
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style=(
                                                "max-width: 140px;"
                                            ),
                                        )
                                    with html.Div(
                                        classes="d-flex align-center",
                                        style=(
                                            "gap: 8px; "
                                            "margin-bottom: 4px;"
                                        ),
                                    ):
                                        html.Div(
                                            "Close gaps with radius "
                                            "N px (bridges 2N-wide "
                                            "gaps; 0 = off)",
                                            style=(
                                                "font-size: 12px; "
                                                "min-width: 240px;"
                                            ),
                                        )
                                        v3.VTextField(
                                            v_model=(
                                                "uct_recon_clean"
                                                "_closing_radius_px",
                                            ),
                                            type="number",
                                            min=0,
                                            step=1,
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style=(
                                                "max-width: 140px;"
                                            ),
                                        )
                                    html.Div(
                                        "Applied per proposal at "
                                        "segment time. 8-connectivity "
                                        "component analysis (no morph "
                                        "kernels — preserves thin "
                                        "structures). The closing "
                                        "knob uses a (2r+1)² all-ones "
                                        "structuring element applied "
                                        "last; typical r is 1-3 px. "
                                        "Re-run Segment after changing "
                                        "these to refresh the overlay.",
                                        style=(
                                            "font-size: 10px; "
                                            "color: #888; "
                                            "margin-top: 2px; "
                                            "margin-bottom: 8px;"
                                        ),
                                    )

                                # ---- 4b. SAM2 video / keyframe propagation
                                # Opt-in workflow: mark 1-5 slices as
                                # keyframes, then propagate via SAM2's
                                # video predictor. The per-slice
                                # workflow stays the default; this
                                # section sits alongside it.
                                with html.Div(
                                    classes="mt-4",
                                    v_show=("uct_step === '2'",),
                                ):
                                    html.H4(
                                        "4b. Propagate from keyframes",
                                        classes=(
                                            "text-subtitle-2 mt-2 mb-1"
                                        ),
                                    )
                                    # Two stacked rows: Mark/Unmark
                                    # toggle for the current slice +
                                    # Propagate button. Gated on
                                    # uct_sam2_video_available so users
                                    # without the SAM2 install see a
                                    # disabled state + hint text.
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center mb-2"
                                        ),
                                        style="gap: 8px;",
                                    ):
                                        with v3.VBtn(
                                            variant="outlined",
                                            density="compact",
                                            color="primary",
                                            disabled=(
                                                "!uct_stack_loaded "
                                                "|| uct_busy "
                                                "|| uct_propagation_"
                                                "busy "
                                                "|| !uct_sam2_video_"
                                                "available",
                                            ),
                                            click=do_toggle_keyframe,
                                            classes="flex-grow-1",
                                        ):
                                            # VBtn's `text` prop
                                            # only accepts a static
                                            # string — for dynamic
                                            # text we render the
                                            # Vue mustache in a
                                            # child span. Same
                                            # pattern as the
                                            # existing scope-picker
                                            # buttons.
                                            html.Span(
                                                "{{ uct_keyframe_"
                                                "slices.includes("
                                                "uct_slice_idx) ? "
                                                "'Unmark keyframe' "
                                                ": 'Mark keyframe' "
                                                "}}",
                                            )
                                    # Keyframe count chip — shows the
                                    # current set so the user knows
                                    # what they've anchored. Empty
                                    # text means no keyframes yet.
                                    html.Div(
                                        "{{ uct_keyframe_summary "
                                        "|| 'No keyframes yet — "
                                        "mark at least one slice "
                                        "to enable Propagate.' }}",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; "
                                            "margin-bottom: 8px;"
                                        ),
                                    )
                                    with html.Button(
                                        type="button",
                                        classes=(
                                            "golgi-cta-wrapper "
                                            "golgi-cta-wrapper-"
                                            "block mb-2"
                                        ),
                                        disabled=(
                                            "uct_busy "
                                            "|| uct_propagation_"
                                            "busy "
                                            "|| !uct_sam2_video_"
                                            "available "
                                            "|| !uct_keyframe_"
                                            "slices.length",
                                        ),
                                        click=(
                                            do_propagate_from_keyframes
                                        ),
                                    ):
                                        html.Span(
                                            classes=(
                                                "golgi-cta-spinner"
                                            ),
                                        )
                                        with html.Span(
                                            classes=(
                                                "golgi-cta-inner"
                                            ),
                                        ):
                                            html.Span(
                                                "▶ Propagate "
                                                "from keyframes"
                                            )
                                    # Availability hint — shown
                                    # only when SAM2 video can't
                                    # run (no sam2 install or no
                                    # checkpoint). Tells the user
                                    # exactly what's missing so
                                    # they can fix it.
                                    html.Div(
                                        "{{ uct_sam2_video_reason "
                                        "}}",
                                        v_show=(
                                            "!uct_sam2_video_"
                                            "available",
                                        ),
                                        style=(
                                            "font-size: 10px; "
                                            "color: #c25048; "
                                            "margin-bottom: 8px;"
                                        ),
                                    )
                                    html.Div(
                                        "SAM2 treats the z-stack as "
                                        "video frames. Mark a few "
                                        "well-labelled slices as "
                                        "keyframes; propagation "
                                        "fills the rest forward + "
                                        "backward. Re-mark / re-"
                                        "propagate to refine.",
                                        v_show=(
                                            "uct_sam2_video_"
                                            "available",
                                        ),
                                        style=(
                                            "font-size: 10px; "
                                            "color: #888; "
                                            "margin-bottom: 8px;"
                                        ),
                                    )

                                # ---- 5. Proposal list (label assignment) ----
                                with html.Div(
                                    v_show=(
                                        "uct_proposals_meta && "
                                        "uct_proposals_meta.length "
                                        "&& uct_step === '2'",
                                    ),
                                ):
                                    html.H4(
                                        "5. Assign labels",
                                        classes=(
                                            "text-subtitle-2 mt-3 mb-1"
                                        ),
                                    )
                                    # Count chips for the assigned labels —
                                    # at-a-glance "how many of each".
                                    # Active-stamp pills — clickable.
                                    # Selecting a pill sets state.uct_active
                                    # _label; subsequent clicks on a mask
                                    # (in crop / zoom mode) assign that
                                    # label. The active pill renders
                                    # `flat` (filled); inactive pills are
                                    # `outlined`. Counts shown as part of
                                    # each pill's label.
                                    with html.Div(
                                        classes="d-flex flex-wrap mb-2",
                                        style="gap: 6px;",
                                    ):
                                        # Clicking a pill enters "label"
                                        # tool mode AND sets the active
                                        # label in one go. The pill shows
                                        # filled (`flat` variant) only
                                        # when both tool_mode === 'label'
                                        # AND it matches the active
                                        # label — so the user can see at
                                        # a glance whether they're in
                                        # relabel mode or just remember
                                        # what the last pill was.
                                        with html.Template(
                                            v_for=(
                                                "pill in "
                                                "uct_active_label_items"
                                            ),
                                            key="pill.value",
                                        ):
                                            v3.VChip(
                                                "{{ pill.title }}: "
                                                "{{ uct_label_counts["
                                                "pill.value] || 0 }}",
                                                size="small",
                                                color=("pill.color",),
                                                variant=(
                                                    "uct_tool_mode === "
                                                    "'label' && "
                                                    "uct_active_label === "
                                                    "pill.value ? 'flat' "
                                                    ": 'outlined'",
                                                ),
                                                click=(
                                                    "uct_tool_mode = "
                                                    "'label'; "
                                                    "uct_active_label = "
                                                    "pill.value"
                                                ),
                                                style=(
                                                    "cursor: pointer; "
                                                    "font-weight: 500;"
                                                ),
                                            )
                                        v3.VChip(
                                            "epi: "
                                            "{{ uct_label_counts.epi "
                                            "  || 0 }}",
                                            v_show=(
                                                "uct_label_counts.epi",
                                            ),
                                            size="small",
                                            color="success",
                                            variant="tonal",
                                        )
                                    html.Div(
                                        "Pick the active label above, then "
                                        "click on a coloured mask in the "
                                        "preview to assign it. Epineurium "
                                        "= (full slice − background) − "
                                        "fascicles, derived via the "
                                        "Generate button.",
                                        style=("font-size: 10px; "
                                                "color: #888; "
                                                "font-style: italic; "
                                                "margin-bottom: 6px;"),
                                    )
                                    # Per-proposal row: meta + label toggle
                                    # buttons. The button group fires
                                    # `do_label_uct_proposal(idx, label)`
                                    # on click; the server re-renders the
                                    # overlay PNG to reflect the new colour.
                                    with html.Div(
                                        v_for=(
                                            "p in uct_proposals_meta"
                                        ),
                                        key="p.idx",
                                        style=(
                                            "border: 1px solid #e6e6e8; "
                                            "border-radius: 6px; "
                                            "padding: 6px 8px; "
                                            "background: white; "
                                            "margin-bottom: 6px;"
                                        ),
                                    ):
                                        # Per-proposal header row: colour
                                        # swatch + id + area + bbox. The
                                        # swatch matches the on-image tint
                                        # for this proposal (when it's
                                        # unlabelled), so the user can map
                                        # each chip to the highlighted
                                        # blob at a glance. Once labelled,
                                        # the on-image tint switches to the
                                        # canonical class colour, but the
                                        # swatch on the chip keeps showing
                                        # the per-index identity.
                                        with html.Div(
                                            classes="d-flex align-center",
                                            style=(
                                                "gap: 6px; "
                                                "margin-bottom: 4px;"
                                            ),
                                        ):
                                            # Trailing comma → tuple → :style
                                            # binding so p.color_hex evaluates
                                            # per row (same trick as the I1
                                            # access-impedance chips). See
                                            # cuff_electrodes.py L142-148 for
                                            # the same pattern.
                                            html.Div(
                                                style=(
                                                    "'width: 12px; "
                                                    "height: 12px; "
                                                    "border-radius: 3px; "
                                                    "border: 1px solid "
                                                    "rgba(0,0,0,0.25); "
                                                    "flex: 0 0 auto; "
                                                    "background: ' + "
                                                    "p.color_hex",
                                                ),
                                            )
                                            html.Div(
                                                (
                                                    "#{{ p.idx }} · "
                                                    "{{ p.area_px."
                                                    "toLocaleString() }} px"
                                                    " · {{ p.bbox_str }}"
                                                ),
                                                style=(
                                                    "font-size: 10px; "
                                                    "color: #555; "
                                                    "font-family: "
                                                    "ui-monospace,Menlo,"
                                                    "Consolas,monospace;"
                                                ),
                                            )
                                        with html.Div(
                                            classes="d-flex",
                                            style="gap: 4px;",
                                        ):
                                            # Four label buttons mirroring
                                            # the four active-stamp pills
                                            # above: Fascicle / Background
                                            # / Epineurium / None. Clicking
                                            # a button assigns that label
                                            # directly to this proposal
                                            # without changing tool mode
                                            # or active label (useful for
                                            # keyboard / power-user flows).
                                            v3.VBtn(
                                                "fasc.",
                                                size="x-small",
                                                color="primary",
                                                style="flex: 1 1 auto;",
                                                variant=(
                                                    "p.label === 'fascicle'"
                                                    "  ? 'flat' "
                                                    "  : 'outlined'",
                                                ),
                                                click=(
                                                    do_label_uct_proposal,
                                                    "[p.idx, 'fascicle']",
                                                ),
                                            )
                                            v3.VBtn(
                                                "bg",
                                                size="x-small",
                                                color="error",
                                                style="flex: 1 1 auto;",
                                                variant=(
                                                    "p.label === 'background'"
                                                    "  ? 'flat' "
                                                    "  : 'outlined'",
                                                ),
                                                click=(
                                                    do_label_uct_proposal,
                                                    "[p.idx, 'background']",
                                                ),
                                            )
                                            v3.VBtn(
                                                "epi",
                                                size="x-small",
                                                color="success",
                                                style="flex: 1 1 auto;",
                                                variant=(
                                                    "p.label === 'epi'"
                                                    "  ? 'flat' "
                                                    "  : 'outlined'",
                                                ),
                                                click=(
                                                    do_label_uct_proposal,
                                                    "[p.idx, 'epi']",
                                                ),
                                            )
                                            v3.VBtn(
                                                "none",
                                                size="x-small",
                                                color="grey",
                                                style="flex: 1 1 auto;",
                                                variant=(
                                                    "p.label === 'unlabeled'"
                                                    "  ? 'flat' "
                                                    "  : 'outlined'",
                                                ),
                                                click=(
                                                    do_label_uct_proposal,
                                                    "[p.idx, 'unlabeled']",
                                                ),
                                            )

                        # ---- Step 2 (Segment) bottom CTA — Status +
                        #      Refine + Epi + Save + Next → ----
                        with html.Div(
                            v_show=("uct_step === '2'",),
                            classes="d-flex align-center",
                            style=(
                                "flex: 0 0 auto; "
                                "border-top: 1px solid #e6e6e8; "
                                "padding: 12px 16px; "
                                "background: white; "
                                "gap: 8px;"
                            ),
                        ):
                            html.Div(
                                "{{ uct_status }}",
                                style=("flex: 1 1 auto; "
                                        "font-size: 11px; "
                                        "color: #555;"),
                            )
                            # Finalize: runs the cleanup pipeline
                            # (refine masks → derive epineurium →
                            # save segmentation) in one click.
                            # Replaces the legacy three-button row
                            # (Refine + Generate epi + Save) — most
                            # of the time the user wants all three
                            # in that order, and there's no value
                            # in re-running them independently
                            # mid-segmentation.
                            html.Button(
                                "✓ Finalize",
                                type="button",
                                classes=(
                                    "golgi-btn-secondary"
                                ),
                                style=(
                                    "background: #4caf50; "
                                    "color: white; "
                                    "border-color: #43a047;"
                                ),
                                title=(
                                    "Runs the end-of-segmentation "
                                    "pipeline: (1) refine masks "
                                    "(fill holes, close gaps, "
                                    "smooth, drop unlabelled), "
                                    "(2) derive epineurium = "
                                    "slice − background − "
                                    "fascicles, (3) save "
                                    "segmentation.json + the "
                                    "labelled mask PNG under "
                                    "<project>/uct/. Safe to "
                                    "re-run."
                                ),
                                disabled=(
                                    "!uct_proposals_meta || "
                                    "!uct_proposals_meta.length || "
                                    "uct_busy",
                                ),
                                click=do_finalize_segmentation,
                            )
                            # Next → Step 3 (3D reconstruction).
                            # Disabled until at least one slice
                            # has an epi or fascicle proposal;
                            # do_recon_next refuses on the server
                            # side too with a status message.
                            html.Button(
                                "Next",
                                type="button",
                                classes=(
                                    "golgi-btn-primary"
                                ),
                                style=(
                                    "background: #1976d2; "
                                    "color: white; "
                                    "border-color: #1565c0;"
                                ),
                                title=(
                                    "Continue to 3D reconstruction. "
                                    "Enabled after Finalize has been "
                                    "run on the current set of "
                                    "labelled slices."
                                ),
                                disabled=(
                                    "!uct_proposals_meta || "
                                    "!uct_proposals_meta.length || "
                                    "!uct_step2_finalized || "
                                    "uct_busy",
                                ),
                                click=do_recon_next,
                            )

                    # =====================================
                    # Step 3: Reconstruct 3D nerve
                    # =====================================
                    # Preview viewport + reconstruction controls.
                    # All content goes inside this WindowItem so
                    # nothing leaks into Step 1/2 views. The
                    # outer flex-column wrapper is preserved
                    # from the legacy layout (viewport at top,
                    # controls below) — the WindowItem just
                    # gates whether it mounts at all.
                    with v3.VStepperWindowItem(
                        value="3",
                        style=(
                            "flex: 1 1 auto; "
                            "min-height: 0; "
                            "display: flex; "
                            "flex-direction: column; "
                            "overflow: hidden;"
                        ),
                    ):
                        # ---- Step 3 body ----
                        # Single full-width panel (no per-slice
                        # preview here — the user toggles back
                        # to Step 2 to inspect labels). Controls
                        # are stacked top-to-bottom:
                        #   - Mode radio (single / multi)
                        #   - Mode-specific parameters
                        #   - Smoothing / refinement
                        #   - Annotation-coverage readout
                        #   - Generated-files list
                        # Step-3 body. Pinning `max-height` to
                        # `calc(80vh − 190px)` (same trick as
                        # Step-2's 2-column body) guarantees the
                        # body has a definite height regardless
                        # of the VStepperWindow flex-chain — the
                        # `flex: 1 1 auto + overflow-y: auto`
                        # combo alone wasn't enough on Step 3,
                        # so the controls overflowed past the
                        # CTA bar and you couldn't reach the
                        # "Generate 3D nerve" / "Done" buttons.
                        with html.Div(
                            v_show=("uct_step === '3'",),
                            classes="d-flex flex-column",
                            style=(
                                "flex: 1 1 auto; "
                                "min-height: 0; "
                                "max-height: calc(80vh - 190px); "
                                "overflow-y: auto; "
                                "padding: 20px 24px; "
                                "background: #fafafa;"
                            ),
                        ):
                            # ---- 3D preview viewport + histogram ----
                            # Side-by-side row: 65 % wide viewport
                            # (gives the prism a sensible aspect
                            # ratio) + 35 % wide quality histogram
                            # next to it. Both fixed at 360 px tall.
                            with html.Div(
                                classes="d-flex flex-row",
                                style=(
                                    "flex: 0 0 auto; "
                                    "gap: 12px; "
                                    "margin-bottom: 8px;"
                                ),
                            ):
                              # 3D preview viewport (65 % of row).
                              # Embedded PyVista plotter — rotate /
                              # pan / zoom like the main workspace
                              # viewport. Same `plotter_ui` widget
                              # the cuff-designer uses; populated
                              # by `_update_recon_viewport` in
                              # app.py on every Preview / Generate.
                              with html.Div(
                                style=(
                                    "flex: 0 0 65%; "
                                    "height: 360px; "
                                    "background: #1f2024; "
                                    "border-radius: 6px; "
                                    "position: relative; "
                                    "overflow: hidden;"
                                ),
                              ):
                                if pl_uct_recon is not None:
                                    view_uct_recon = plotter_ui(
                                        pl_uct_recon,
                                        interactive_ratio=1,
                                        mode="trame",
                                        default_server_rendering=(
                                            False
                                        ),
                                    )
                                    if ctrl is not None:
                                        ctrl.view_uct_recon_update = (
                                            view_uct_recon.update
                                        )
                                        ctrl.view_uct_recon_reset_camera = (
                                            view_uct_recon.reset_camera
                                        )
                                # Empty-state overlay — only when
                                # there are no actors yet.
                                html.Div(
                                    "Click ▶ Preview to render the 3D "
                                    "geometry, or ▶ Generate 3D nerve to "
                                    "save the .stl files and refresh "
                                    "the view.",
                                    v_show=(
                                        "!uct_recon_mesh_items.length",
                                    ),
                                    style=(
                                        "position: absolute; "
                                        "inset: 0; "
                                        "color: #888; "
                                        "font-size: 13px; "
                                        "max-width: 360px; "
                                        "margin: auto; "
                                        "text-align: center; "
                                        "font-style: italic; "
                                        "padding: 0 16px; "
                                        "display: flex; "
                                        "align-items: center; "
                                        "justify-content: center; "
                                        "pointer-events: none;"
                                    ),
                                )

                              # ---- Quality histogram (35 % of row).
                              # Plotly figure built by
                              # _build_quality_histogram_figure
                              # from the union of per-mesh
                              # triangle-quality arrays (Heron
                              # radius-ratio). Same RdYlGn colour
                              # semantics as the nerve-importer
                              # histogram.
                              with html.Div(
                                v_show=(
                                    "uct_recon_mesh_items.length",
                                ),
                                style=(
                                    "flex: 0 0 35%; "
                                    # Bumped from 360 → 520 px to
                                    # give each per-surface subplot
                                    # enough vertical room when the
                                    # reconstruct produces 3+ meshes.
                                    # Inner overflow:auto lets the
                                    # panel scroll cleanly when even
                                    # 520 isn't enough (10+ fascicles
                                    # in single-slice mode).
                                    "height: 520px; "
                                    "background: white; "
                                    "border: 1px solid #e6e6e8; "
                                    "border-radius: 6px; "
                                    "padding: 8px; "
                                    "display: flex; "
                                    "flex-direction: column; "
                                    "overflow: hidden;"
                                ),
                              ):
                                html.Div(
                                    "Mesh quality (per surface)",
                                    style=(
                                        "font-size: 11px; "
                                        "color: #555; "
                                        "margin-bottom: 4px; "
                                        "letter-spacing: 0.04em; "
                                        "text-transform: uppercase; "
                                        "flex: 0 0 auto;"
                                    ),
                                )
                                with html.Div(
                                    style=(
                                        "flex: 1 1 auto; "
                                        "width: 100%; "
                                        "min-height: 0;"
                                    ),
                                ):
                                    if plotly_module is not None:
                                        plotly_module.Figure(
                                            state_variable_name=(
                                                "uct_recon_quality"
                                                "_hist_figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=False,
                                        )

                            # ---- Viewport toolbar ----
                            # Edges + quality colormap toggles +
                            # camera reset. Watchers in app.py
                            # rebuild actors via the meshes cache
                            # so flipping a toggle is fast (no
                            # marching-cubes re-run).
                            with html.Div(
                                classes="d-flex align-center",
                                style=(
                                    "gap: 12px; "
                                    "margin-bottom: 12px; "
                                    "padding: 4px 8px; "
                                    "background: white; "
                                    "border: 1px solid #e6e6e8; "
                                    "border-radius: 4px;"
                                ),
                            ):
                                v3.VCheckbox(
                                    v_model=(
                                        "uct_recon_show_edges",
                                    ),
                                    label="Show edges",
                                    density="compact",
                                    hide_details=True,
                                )
                                v3.VCheckbox(
                                    v_model=(
                                        "uct_recon_color_by_quality",
                                    ),
                                    label=(
                                        "Colour by mesh quality"
                                    ),
                                    density="compact",
                                    hide_details=True,
                                )
                                v3.VSpacer()
                                v3.VBtn(
                                    "Reset camera",
                                    size="x-small",
                                    variant="text",
                                    prepend_icon=(
                                        "mdi-crosshairs-gps"
                                    ),
                                    click=(
                                        ctrl.view_uct_recon_reset_camera
                                        if ctrl is not None else None
                                    ),
                                )

                            # ---- Legend: per-mesh visibility ----
                            # One chip per generated surface.
                            # Clicking the checkbox toggles
                            # `uct_recon_mesh_items[i].visible`;
                            # the watcher in app.py flips the
                            # actor's visibility on pl_uct_recon.
                            html.Div(
                                "Visible structures",
                                style=(
                                    "font-size: 11px; "
                                    "color: #555; "
                                    "margin-bottom: 4px; "
                                    "letter-spacing: 0.04em; "
                                    "text-transform: uppercase;"
                                ),
                                v_show=(
                                    "uct_recon_mesh_items.length",
                                ),
                            )
                            with html.Div(
                                v_show=(
                                    "uct_recon_mesh_items.length",
                                ),
                                classes=(
                                    "d-flex flex-wrap"
                                ),
                                style=(
                                    "gap: 6px; "
                                    "margin-bottom: 12px;"
                                ),
                            ):
                                with html.Div(
                                    v_for=(
                                        "(m, i) in "
                                        "uct_recon_mesh_items"
                                    ),
                                    key="m.name",
                                    classes="d-flex align-center",
                                    style=(
                                        "gap: 6px; "
                                        "padding: 2px 8px; "
                                        "background: white; "
                                        "border: 1px solid "
                                        "#e6e6e8; "
                                        "border-radius: 12px; "
                                        "font-size: 11px;"
                                    ),
                                ):
                                    v3.VCheckbox(
                                        model_value=("m.visible",),
                                        density="compact",
                                        hide_details=True,
                                        classes="ma-0 pa-0",
                                        style=(
                                            "transform: scale(0.7); "
                                            "transform-origin: "
                                            "left center;"
                                        ),
                                        raw_attrs=[
                                            '@update:modelValue='
                                            '"uct_recon_mesh_items '
                                            '= uct_recon_mesh_items'
                                            '.map((mm, ii) => ii '
                                            '=== i ? '
                                            '{...mm, visible: '
                                            '$event} : mm)"',
                                        ],
                                    )
                                    html.Div(
                                        style=(
                                            "'width: 10px; "
                                            "height: 10px; "
                                            "border-radius: 2px; "
                                            "background: ' "
                                            "+ m.color + ';'",
                                        ),
                                    )
                                    html.Span(
                                        "{{ m.name }} "
                                        "({{ m.n_tris."
                                        "toLocaleString() }} tris)",
                                    )

                            # Histogram now lives next to the
                            # viewport at the top of the panel
                            # (see "Quality histogram (35 % of
                            # row)" above) — no second copy.

                            # Section header.
                            html.H4(
                                "Reconstruct 3D nerve volume",
                                classes="text-subtitle-1 mb-1",
                            )
                            html.Div(
                                "Generate one or more .stl surfaces "
                                "from the annotated slices. Files are "
                                "saved into "
                                "<code>&lt;project&gt;/uct/nerve_3d/&lt;"
                                "timestamp&gt;/</code> and become "
                                "available in the Import wizard.",
                                style=(
                                    "font-size: 11px; color: #666; "
                                    "margin-bottom: 16px;"
                                ),
                            )

                            # ---- Mode picker ----
                            html.H4(
                                "Mode",
                                classes="text-subtitle-2 mt-2 mb-1",
                            )
                            with html.Div(
                                classes="d-flex flex-row",
                                style="gap: 8px; margin-bottom: 12px;",
                            ):
                                # Single-slice tile.
                                with html.Div(
                                    classes="d-flex align-center",
                                    style=(
                                        "'border: 2px solid ' + "
                                        "(uct_recon_mode === 'single' "
                                        "  ? '#1976d2' : '#ccc') + "
                                        "'; background: ' + "
                                        "(uct_recon_mode === 'single' "
                                        "  ? '#e3f2fd' : 'white') + "
                                        "'; border-radius: 8px; "
                                        "padding: 12px; "
                                        "cursor: pointer; "
                                        "flex: 1 1 0; gap: 8px;'",
                                    ),
                                    raw_attrs=[
                                        '@click='
                                        '"uct_recon_mode = \'single\'"',
                                    ],
                                ):
                                    v3.VIcon(
                                        "mdi-layers-outline",
                                        size="24",
                                        color="primary",
                                    )
                                    with html.Div():
                                        html.Div(
                                            "Single-slice extrusion",
                                            style=(
                                                "font-weight: 600; "
                                                "font-size: 13px;"
                                            ),
                                        )
                                        html.Div(
                                            "Sweep one slice up by a fixed "
                                            "thickness. Fast; ignores the "
                                            "rest of the stack.",
                                            style=(
                                                "font-size: 10px; "
                                                "color: #666;"
                                            ),
                                        )
                                # Multi-slice tile.
                                with html.Div(
                                    classes="d-flex align-center",
                                    style=(
                                        "'border: 2px solid ' + "
                                        "(uct_recon_mode === 'multi' "
                                        "  ? '#1976d2' : '#ccc') + "
                                        "'; background: ' + "
                                        "(uct_recon_mode === 'multi' "
                                        "  ? '#e3f2fd' : 'white') + "
                                        "'; border-radius: 8px; "
                                        "padding: 12px; "
                                        "cursor: pointer; "
                                        "flex: 1 1 0; gap: 8px;'",
                                    ),
                                    raw_attrs=[
                                        '@click='
                                        '"uct_recon_mode = \'multi\'"',
                                    ],
                                ):
                                    v3.VIcon(
                                        "mdi-cube-outline",
                                        size="24",
                                        color="primary",
                                    )
                                    with html.Div():
                                        html.Div(
                                            "Multi-slice marching cubes",
                                            style=(
                                                "font-weight: 600; "
                                                "font-size: 13px;"
                                            ),
                                        )
                                        html.Div(
                                            "Stack annotated slices into a "
                                            "3D volume, ZOH-fill gaps, "
                                            "isosurface to STL.",
                                            style=(
                                                "font-size: 10px; "
                                                "color: #666;"
                                            ),
                                        )

                            # ---- Single-slice params ----
                            with html.Div(
                                v_show=("uct_recon_mode === 'single'",),
                            ):
                                html.H4(
                                    "Single-slice parameters",
                                    classes="text-subtitle-2 mt-3 mb-1",
                                )
                                with html.Div(
                                    classes="d-flex align-center",
                                    style="gap: 8px; margin-bottom: 6px;",
                                ):
                                    html.Span(
                                        "annotated slice",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; min-width: 90px;"
                                        ),
                                    )
                                    # Annotated-only slice picker. Items
                                    # come from `uct_recon_annotated_items`
                                    # (built by `_refresh_recon_coverage`)
                                    # so the user can't pick an empty slice
                                    # by mistake. Default selection is set
                                    # in `do_recon_next` to the first
                                    # annotated slice or the current one.
                                    v3.VSelect(
                                        v_model=(
                                            "uct_recon_single_slice_idx",
                                        ),
                                        items=(
                                            "uct_recon_annotated_items",
                                        ),
                                        item_value="value",
                                        item_title="title",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 260px;",
                                        no_data_text=(
                                            "No annotated slices yet — "
                                            "label fascicle or epi on at "
                                            "least one slice in Step 2."
                                        ),
                                    )
                                with html.Div(
                                    classes="d-flex align-center",
                                    style="gap: 8px;",
                                ):
                                    html.Span(
                                        "thickness (mm)",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; min-width: 90px;"
                                        ),
                                    )
                                    v3.VTextField(
                                        v_model_number=(
                                            "uct_recon_thickness_mm",
                                        ),
                                        type="number",
                                        step="0.1",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 110px;",
                                    )

                            # ---- Multi-slice params ----
                            with html.Div(
                                v_show=("uct_recon_mode === 'multi'",),
                            ):
                                html.H4(
                                    "Multi-slice parameters",
                                    classes="text-subtitle-2 mt-3 mb-1",
                                )
                                with html.Div(
                                    classes="d-flex align-center",
                                    style="gap: 8px; margin-bottom: 6px;",
                                ):
                                    html.Span(
                                        "slice start",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; min-width: 90px;"
                                        ),
                                    )
                                    v3.VTextField(
                                        v_model_number=(
                                            "uct_recon_slice_start",
                                        ),
                                        type="number",
                                        step="1",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 110px;",
                                    )
                                with html.Div(
                                    classes="d-flex align-center",
                                    style="gap: 8px; margin-bottom: 6px;",
                                ):
                                    html.Span(
                                        "slice end",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; min-width: 90px;"
                                        ),
                                    )
                                    v3.VTextField(
                                        v_model_number=(
                                            "uct_recon_slice_end",
                                        ),
                                        type="number",
                                        step="1",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 110px;",
                                    )
                                with html.Div(
                                    classes="d-flex align-center",
                                    style="gap: 8px; margin-bottom: 6px;",
                                ):
                                    html.Span(
                                        "Z spacing (mm)",
                                        style=(
                                            "font-size: 11px; "
                                            "color: #555; min-width: 90px;"
                                        ),
                                    )
                                    v3.VTextField(
                                        v_model_number=(
                                            "uct_recon_voxel_z_mm",
                                        ),
                                        type="number",
                                        step="0.001",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 130px;",
                                    )
                                    html.Span(
                                        "(pre-filled from stack metadata "
                                        "if available)",
                                        style=(
                                            "font-size: 10px; "
                                            "color: #888;"
                                        ),
                                    )

                            # ---- Smoothing (shared by both modes) ----
                            html.H4(
                                "Smoothing",
                                classes="text-subtitle-2 mt-3 mb-1",
                            )
                            with html.Div(
                                classes="d-flex align-center",
                                style="gap: 8px;",
                            ):
                                v3.VCheckbox(
                                    v_model=("uct_recon_smooth",),
                                    label=(
                                        "Gaussian-smooth volume before "
                                        "marching cubes"
                                    ),
                                    density="compact",
                                    hide_details=True,
                                )
                                html.Span(
                                    "σ",
                                    style=(
                                        "font-size: 11px; color: #555;"
                                    ),
                                )
                                v3.VTextField(
                                    v_model_number=(
                                        "uct_recon_smooth_sigma",
                                    ),
                                    type="number",
                                    step="0.1",
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style="max-width: 90px;",
                                    disabled=("!uct_recon_smooth",),
                                )

                            # ---- Mesh refinement ----
                            # NO decimation in this pipeline —
                            # the MC output is already at the
                            # right resolution for the chosen
                            # voxel size. Refinement is:
                            #   1. drop tiny disconnected
                            #      components (speck removal)
                            #   2. Taubin smooth
                            #   3. pymeshfix repair
                            #      (genus-preserving)
                            #   4. trimesh defensive pass
                            #   5. optimesh CVT relaxation
                            #      (the histogram-mover)
                            # Optional power-user knob below:
                            # pyacvd isotropic remesh with an
                            # explicit target edge length, so
                            # small fascicles get few vertices
                            # and the epi shell gets many, both
                            # at uniform local resolution.
                            html.H4(
                                "Mesh refinement",
                                classes="text-subtitle-2 mt-3 mb-1",
                            )
                            v3.VCheckbox(
                                v_model=("uct_recon_refine",),
                                label=(
                                    "Run mesh cleanup "
                                    "(drop specks → Taubin → "
                                    "pymeshfix → optimesh CVT)"
                                ),
                                density="compact",
                                hide_details=True,
                            )
                            # M27 — legacy "Isotropic remesh"
                            # checkbox + edge length field
                            # removed. Both are now surfaced via
                            # the "Surface size control"
                            # combobox below (selecting
                            # "Isotropic remesh" exposes the
                            # edge-length input alongside the
                            # mode). The underlying state vars
                            # `uct_recon_remesh` +
                            # `uct_recon_edge_len_um` are still
                            # used by the action dispatcher
                            # (size_mode='isotropic' forces
                            # remesh on regardless of the bool).
                            # optimesh CVT relaxation. Opt-in
                            # because it can segfault on shell
                            # meshes; even when stable it adds
                            # noticeable runtime. The genus-
                            # guard inside `_maybe_optimesh`
                            # auto-skips epi-style shells, so
                            # enabling this only buys CVT for
                            # the simply-connected fascicles.
                            v3.VCheckbox(
                                v_model=(
                                    "uct_recon_use_optimesh",
                                ),
                                label=(
                                    "Aggressive triangulation "
                                    "relaxation (optimesh CVT)"
                                ),
                                density="compact",
                                hide_details=True,
                                disabled=("!uct_recon_refine",),
                            )
                            html.Div(
                                "Pushes fascicle triangles "
                                "toward near-equilateral. Auto-"
                                "skipped on the epi shell (genus "
                                "&gt; 0) because optimesh has "
                                "been observed to segfault on "
                                "high-genus inputs.",
                                style=(
                                    "font-size: 10px; "
                                    "color: #888; "
                                    "margin-top: 2px;"
                                ),
                            )

                            # 3D volume cleanup section.
                            # Per-slice 2D cleanup runs at
                            # Segment time (see the cleanup
                            # block under Step 2 / Segment);
                            # this 3D pass runs after the stack
                            # is assembled, BEFORE marching
                            # cubes. Catches Z-direction
                            # speckle / voids that 2D per-slice
                            # can't see (a streak speckle
                            # living in 3-4 adjacent slices but
                            # nowhere else; a background column
                            # punched through the middle of a
                            # fascicle).
                            html.Div(
                                "3D volume cleanup",
                                classes=(
                                    "text-subtitle-2 mt-3 "
                                    "mb-1"
                                ),
                                style=(
                                    "color: #888a90; "
                                    "letter-spacing: 0.04em; "
                                    "text-transform: "
                                    "uppercase; "
                                    "font-size: 10px;"
                                ),
                            )
                            with html.Div(
                                classes="d-flex align-center",
                                style=(
                                    "gap: 8px; "
                                    "margin-bottom: 4px;"
                                ),
                            ):
                                html.Div(
                                    "Drop 3D speckles < N "
                                    "voxels (0 = off)",
                                    style=(
                                        "font-size: 12px; "
                                        "min-width: 240px;"
                                    ),
                                )
                                v3.VTextField(
                                    v_model=(
                                        "uct_recon_clean_3d"
                                        "_min_component_vox",
                                    ),
                                    type="number",
                                    min=0,
                                    step=100,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style=(
                                        "max-width: 140px;"
                                    ),
                                )
                            with html.Div(
                                classes="d-flex align-center",
                                style=(
                                    "gap: 8px; "
                                    "margin-bottom: 4px;"
                                ),
                            ):
                                html.Div(
                                    "Fill 3D holes < N "
                                    "voxels inside "
                                    "foreground (0 = off)",
                                    style=(
                                        "font-size: 12px; "
                                        "min-width: 240px;"
                                    ),
                                )
                                v3.VTextField(
                                    v_model=(
                                        "uct_recon_clean_3d"
                                        "_min_hole_vox",
                                    ),
                                    type="number",
                                    min=0,
                                    step=100,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style=(
                                        "max-width: 140px;"
                                    ),
                                )
                            html.Div(
                                "26-connectivity components. "
                                "Knobs in VOXELS — typically "
                                "larger than the 2D-pixel "
                                "equivalents because the "
                                "same artefact spans several "
                                "slices in Z (a 50 px "
                                "slice-speckle that survives "
                                "3 slices = 150 voxels).",
                                style=(
                                    "font-size: 10px; "
                                    "color: #888; "
                                    "margin-top: 2px;"
                                ),
                            )

                            # M24 — Fascicle inward-offset.
                            # Before marching cubes, erode the
                            # epi volume by N voxels and AND
                            # the fascicle volume against it so
                            # fascicle isosurfaces land cleanly
                            # inside the epi after smoothing.
                            # Critical for TetGen — without this
                            # fascicles "straddle" the epi by
                            # sub-µm to few-µm in some places,
                            # producing PLC failures TetGen
                            # can't classify. 0 disables.
                            html.Div(
                                "Fascicle nesting",
                                classes=(
                                    "text-subtitle-2 mt-3 "
                                    "mb-1"
                                ),
                                style=(
                                    "color: #888a90; "
                                    "letter-spacing: 0.04em; "
                                    "text-transform: "
                                    "uppercase; "
                                    "font-size: 10px;"
                                ),
                            )
                            with html.Div(
                                classes="d-flex align-center",
                                style=(
                                    "gap: 8px; "
                                    "margin-bottom: 4px;"
                                ),
                            ):
                                html.Div(
                                    "Inset fascicles N voxels "
                                    "inside epi (0 = off)",
                                    style=(
                                        "font-size: 12px; "
                                        "min-width: 240px;"
                                    ),
                                )
                                v3.VTextField(
                                    v_model=(
                                        "uct_recon_fasc"
                                        "_inset_vox",
                                    ),
                                    type="number",
                                    min=0,
                                    step=1,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style=(
                                        "max-width: 140px;"
                                    ),
                                )
                            html.Div(
                                "Eroding the epi by N voxels "
                                "and intersecting fascicles "
                                "against it guarantees a "
                                "N-voxel margin between each "
                                "fascicle's isosurface and "
                                "the epi's. Default 2 voxels "
                                "matches the smoothing kernel "
                                "(σ=1) so fascicles don't "
                                "straddle the epi after "
                                "marching cubes. Bump to 3-4 "
                                "if you still see touching/ "
                                "straddling in the "
                                "inter-surface diagnostics.",
                                style=(
                                    "font-size: 10px; "
                                    "color: #888; "
                                    "margin-top: 2px;"
                                ),
                            )

                            # M27 — Surface size control. The
                            # combobox picks the mode; the
                            # sub-control underneath swaps
                            # based on the selection.
                            html.Div(
                                "Surface size control",
                                classes=(
                                    "text-subtitle-2 mt-3 "
                                    "mb-1"
                                ),
                                style=(
                                    "color: #888a90; "
                                    "letter-spacing: 0.04em; "
                                    "text-transform: "
                                    "uppercase; "
                                    "font-size: 10px;"
                                ),
                            )
                            v3.VSelect(
                                v_model=(
                                    "uct_recon_size_mode",
                                ),
                                items=(
                                    "uct_recon_size_mode_items",
                                ),
                                item_title="title",
                                item_value="value",
                                density="compact",
                                hide_details=True,
                                variant="outlined",
                                classes="mb-1",
                            )
                            # Sub-control: keep fraction (mode='fraction')
                            with html.Div(
                                v_show=(
                                    "uct_recon_size_mode "
                                    "=== 'fraction'",
                                ),
                                classes=(
                                    "d-flex align-center mt-2"
                                ),
                                style="gap: 8px;",
                            ):
                                html.Div(
                                    "Keep fraction of tris "
                                    "per surface (0 = drop "
                                    "all, 1 = keep all)",
                                    style=(
                                        "font-size: 12px; "
                                        "min-width: 240px;"
                                    ),
                                )
                                v3.VTextField(
                                    v_model=(
                                        "uct_recon_decimate"
                                        "_fraction",
                                    ),
                                    type="number",
                                    min=0.01,
                                    max=1.0,
                                    step=0.05,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style="max-width: 140px;",
                                )
                            # Sub-control: target tri count (mode='target_tris')
                            with html.Div(
                                v_show=(
                                    "uct_recon_size_mode "
                                    "=== 'target_tris'",
                                ),
                                classes=(
                                    "d-flex align-center mt-2"
                                ),
                                style="gap: 8px;",
                            ):
                                html.Div(
                                    "Decimate each surface to "
                                    "≤ N tris",
                                    style=(
                                        "font-size: 12px; "
                                        "min-width: 240px;"
                                    ),
                                )
                                v3.VTextField(
                                    v_model=(
                                        "uct_recon_decimate"
                                        "_target_tris",
                                    ),
                                    type="number",
                                    min=0,
                                    step=1000,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style="max-width: 140px;",
                                )
                            # Sub-control: edge length (mode='isotropic')
                            with html.Div(
                                v_show=(
                                    "uct_recon_size_mode "
                                    "=== 'isotropic'",
                                ),
                                classes=(
                                    "d-flex align-center mt-2"
                                ),
                                style="gap: 8px;",
                            ):
                                html.Div(
                                    "Target edge length (µm)",
                                    style=(
                                        "font-size: 12px; "
                                        "min-width: 240px;"
                                    ),
                                )
                                v3.VTextField(
                                    v_model=(
                                        "uct_recon_edge_len_um",
                                    ),
                                    type="number",
                                    min=5,
                                    step=10,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style="max-width: 140px;",
                                )
                            html.Div(
                                "Fraction is recommended when "
                                "surfaces vary in size — a "
                                "uniform target-tris cap "
                                "crushes the epi (488 k → 20 k) "
                                "while barely touching the "
                                "fascicles, producing huge "
                                "edge-length disparities. "
                                "Isotropic remesh is the most "
                                "expensive but produces the "
                                "cleanest uniform-edge mesh.",
                                style=(
                                    "font-size: 10px; "
                                    "color: #888; "
                                    "margin-top: 4px;"
                                ),
                            )

                            # ---- Annotation coverage readout ----
                            html.H4(
                                "Annotation coverage",
                                classes="text-subtitle-2 mt-3 mb-1",
                            )
                            html.Div(
                                "{{ uct_recon_coverage_msg }}",
                                style=(
                                    "font-size: 11px; color: #555;"
                                ),
                            )
                            html.Div(
                                "Annotated slices: "
                                "{{ uct_recon_annotated.join(', ') "
                                "|| 'none' }}",
                                style=(
                                    "font-size: 11px; "
                                    "color: #555; "
                                    "font-family: ui-monospace,Menlo,"
                                    "Consolas,monospace; "
                                    "margin-top: 4px;"
                                ),
                            )

                            # ---- Generated files ----
                            html.H4(
                                "Generated files",
                                classes="text-subtitle-2 mt-3 mb-1",
                                v_show=("uct_recon_files.length > 0",),
                            )
                            with html.Div(
                                v_show=("uct_recon_files.length > 0",),
                                style=(
                                    "background: white; "
                                    "border: 1px solid #e6e6e8; "
                                    "border-radius: 4px; "
                                    "padding: 8px; "
                                    "font-size: 11px; "
                                    "font-family: ui-monospace,Menlo,"
                                    "Consolas,monospace;"
                                ),
                            ):
                                with html.Div(
                                    v_for=(
                                        "f in uct_recon_files",
                                    ),
                                ):
                                    html.Span(
                                        "{{ f.name }}",
                                        style=(
                                            "font-weight: 600;"
                                        ),
                                    )
                                    html.Span(
                                        " — {{ f.path }}",
                                        style="color: #888;",
                                    )

                        # ---- Step 3 (Reconstruct) bottom CTA — Status +
                        #      Back + Generate 3D ----
                        with html.Div(
                            v_show=("uct_step === '3'",),
                            classes="d-flex align-center",
                            style=(
                                "flex: 0 0 auto; "
                                "border-top: 1px solid #e6e6e8; "
                                "padding: 12px 16px; "
                                "background: white; "
                                "gap: 8px;"
                            ),
                        ):
                            # Back floated left, white-on-grey-
                            # border styling so it reads as a
                            # tertiary control (the right-side
                            # blue / green buttons are the
                            # primary actions).
                            html.Button(
                                "← Back",
                                type="button",
                                classes="golgi-btn-secondary",
                                style=(
                                    "background: white; "
                                    "color: #444; "
                                    "border: 1px solid #ccc;"
                                ),
                                click=do_recon_back,
                            )
                            html.Div(
                                "{{ uct_recon_status }}",
                                style=(
                                    "flex: 1 1 auto; "
                                    "font-size: 11px; "
                                    "color: #555;"
                                ),
                            )
                            # Reconstruct: writes STLs + manifest
                            # into <project>/uct/nerve_3d/<ts>/
                            # and refreshes the viewport. User
                            # can tweak params + re-run freely —
                            # each click creates a new
                            # timestamped bundle dir.
                            html.Button(
                                "▶ Reconstruct",
                                type="button",
                                classes="golgi-btn-primary",
                                style=(
                                    "background: #1976d2; "
                                    "color: white; "
                                    "border-color: #1565c0;"
                                ),
                                title=(
                                    "Builds the 3D meshes, writes .stl + "
                                    "manifest.json into "
                                    "<project>/uct/nerve_3d/<timestamp>/, "
                                    "and refreshes the viewport above. "
                                    "Re-run as needed to iterate on "
                                    "parameters; each run creates a new "
                                    "bundle directory."
                                ),
                                disabled=("uct_busy",),
                                click=do_run_reconstruction,
                            )
                            # Done: closes the Segment-µCT dialog
                            # and opens the Import-Nerve stepper
                            # with the just-generated bundle pre-
                            # selected as the source. Disabled
                            # until the user has run Generate at
                            # least once (so there's a bundle to
                            # hand over).
                            html.Button(
                                "✓ Done — open in Import wizard",
                                type="button",
                                classes="golgi-btn-primary",
                                style=(
                                    "background: #2e7d32; "
                                    "color: white; "
                                    "border-color: #1b5e20;"
                                ),
                                title=(
                                    "Closes this dialog and opens the "
                                    "nerve-import wizard with the most "
                                    "recently-generated bundle pre-"
                                    "selected. From there you can step "
                                    "through fiber-trajectory + muscle "
                                    "configuration (epi generation is "
                                    "skipped — the bundle already "
                                    "contains epi + fascicle surfaces)."
                                ),
                                disabled=(
                                    "uct_busy "
                                    "|| !uct_last_bundle_id",
                                ),
                                click=do_finish_recon,
                            )
