# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Import Study dialog (F2.2 Phase 2).

Three-stage flow:
  1. User picks a .zip via the VFileInput. As soon as bytes
     arrive (`@change` on the v-model), the action handler peeks
     the manifest and stashes a one-line summary so the user can
     confirm what they're about to import.
  2. Optionally clicks "Reproduction Run" (Phase 3) to verify the
     bundle's DAG hashes — same dialog, surfaces a pass/fail line.
  3. Clicks "Import" to unpack into a fresh project dir under
     PROJECTS_ROOT and open it.

Cancel + Close are wired separately so the user can back out at
any stage. The dialog stays open across the busy lightbox during
the actual unpack so the post-import status is visible inline."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html, vuetify3 as v3


def render(
    *,
    do_import_study_close: Callable,
    do_import_study_run: Callable,
    do_import_study_upload: Callable | None = None,
    do_import_study_check_only: Callable | None = None,
    do_import_study_load_from_disk: Callable | None = None,
) -> None:
    """Render the Import Study dialog.

    `do_import_study_upload` is no longer wired to VFileInput's
    @change attr (that DOM event doesn't fire reliably across
    trame builds). The real entry point is the state.change
    watcher build_app installs on `study_import_upload` — the
    v-model populating IS the "user picked a file" signal."""
    with v3.VDialog(
        v_model=("show_import_study_dialog",),
        max_width=620,
        persistent=True,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Import study",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Upload a `.zip` exported via "
                    "File → Export → Export study. The "
                    "bundle's geometry, mesh, FEM outputs, "
                    "fiber populations + sim caches will be "
                    "unpacked into a fresh project under your "
                    "workspace.",
                    classes="golgi-dialog-body mb-3",
                )

                # File picker — VFileInput v-model is a list of
                # {name, size, content}. The state.change watcher
                # on `study_import_upload` (wired in build_app)
                # fires once the bytes arrive at the server and
                # runs the manifest peek.
                # File picker — change event runs a custom JS
                # uploader (window.golgi_study_upload) that POSTs
                # the file to /api/study/upload via XHR. Bypasses
                # trame's WebSocket entirely so:
                #   * works for arbitrary bundle sizes (no
                #     msgpack ArrayBuffer cap)
                #   * shows real upload progress via
                #     xhr.upload.onprogress
                #   * doesn't block the WS doing anything else
                # The v_model field stays for Vuetify's internal
                # state (so the picker shows the filename), but
                # we don't read it server-side — the uploaded
                # path arrives via the /api endpoint's callback.
                v3.VFileInput(
                    v_model=("study_import_upload",),
                    label="Pick study .zip",
                    density="compact",
                    hide_details=True,
                    accept=".zip",
                    prepend_icon="mdi-folder-zip-outline",
                    disabled=(
                        "study_import_pending_busy "
                        "|| study_import_uploading",
                    ),
                    raw_attrs=[
                        # Vuetify v-file-input emits the File[]
                        # array on @update:modelValue; pass it
                        # straight to the JS uploader.
                        '@update:modelValue='
                        '"window.golgi_study_upload($event)"',
                    ],
                    classes="mb-3",
                )

                # Indeterminate progress while the browser → server
                # upload is in flight (the gap between file-pick
                # and the state.change watcher firing). Trame's
                # WS upload doesn't expose a per-chunk progress
                # signal, so the bar is intentionally striped +
                # indeterminate here.
                v3.VProgressLinear(
                    v_show=("study_import_uploading",),
                    indeterminate=True,
                    color="#e24b4a",
                    height=10,
                    striped=True,
                    classes="mb-3",
                )

                # Determinate progress during the server-side
                # unpack — bar fills 0-100 as files are extracted.
                with html.Div(
                    v_show=("study_import_pending_busy",),
                    style="margin-bottom: 12px;",
                ):
                    v3.VProgressLinear(
                        model_value=(
                            "study_import_pending_progress",
                        ),
                        color="#e24b4a",
                        height=10,
                        striped=True,
                        classes="mb-2",
                    )
                    html.Div(
                        "{{ study_import_pending_status }}",
                        style=(
                            "font-size: 11px; color: #444;"
                        ),
                    )

                # Manifest peek summary — multi-line monospace
                # block so file names + stage list read tidily.
                html.Pre(
                    "{{ study_import_manifest_summary }}",
                    v_show=(
                        "study_import_manifest_summary",
                    ),
                    style=(
                        "font-size: 11px; "
                        "color: #333; "
                        "background: #f6f6f7; "
                        "padding: 8px 10px; "
                        "border-radius: 4px; "
                        "white-space: pre-wrap; "
                        "font-family: monospace; "
                        "margin-bottom: 12px;"
                    ),
                )

                # Verbose progress log (shown during the unpack).
                # Same content as the busy lightbox tail, but
                # in-dialog so the user doesn't have to look at
                # the global overlay.
                html.Pre(
                    "{{ study_import_pending_progress_log }}",
                    v_show=(
                        "study_import_pending_progress_log "
                        "&& study_import_pending_busy",
                    ),
                    style=(
                        "font-size: 10px; "
                        "color: #444; "
                        "background: #f6f6f7; "
                        "padding: 6px 10px; "
                        "border-radius: 4px; "
                        "margin-bottom: 8px; "
                        "font-family: monospace; "
                        "max-height: 140px; "
                        "overflow-y: auto;"
                    ),
                )

                # Final-status line (e.g. "✓ Imported into X")
                # rendered ONLY after the busy spinner clears so
                # it doesn't duplicate the in-progress status.
                html.Div(
                    "{{ study_import_pending_status }}",
                    v_show=(
                        "!study_import_pending_busy "
                        "&& study_import_pending_status "
                        "&& !study_import_pending_error",
                    ),
                    style=(
                        "font-size: 11px; "
                        "color: #146e3a; "
                        "margin-bottom: 6px;"
                    ),
                )
                html.Div(
                    "{{ study_import_pending_error }}",
                    v_show=("study_import_pending_error",),
                    style=(
                        "font-size: 11px; "
                        "color: #c0392b; "
                        "margin-bottom: 6px;"
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
                    click=do_import_study_close,
                )
                if do_import_study_check_only is not None:
                    html.Button(
                        "Reproduction Run",
                        type="button",
                        classes=(
                            "golgi-btn-secondary golgi-btn-sm"
                        ),
                        style="margin-right: 8px;",
                        disabled=(
                            "!study_import_ready "
                            "|| study_import_pending_busy",
                        ),
                        click=do_import_study_check_only,
                    )
                html.Button(
                    "Import",
                    type="button",
                    classes=(
                        "golgi-btn-primary golgi-btn-sm"
                    ),
                    disabled=(
                        "!study_import_ready "
                        "|| study_import_pending_busy",
                    ),
                    click=do_import_study_run,
                )
