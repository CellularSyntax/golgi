# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Study-bundle UI state defaults (F2.2).

Two flows — export + import — each with its own pending slot.
The export slot pushes the bundle bytes as a base64 data URI for
browser download; the import slot stores the just-uploaded bundle
bytes between "peek manifest" and "run import" so the dialog can
show the user the bundle's metadata before committing."""
from __future__ import annotations


def register(state) -> None:
    # ---- Export ----
    state.study_export_pending_busy = False
    state.study_export_pending_data_uri = ""
    state.study_export_pending_filename = ""
    state.study_export_pending_status = ""
    state.study_export_pending_error = ""
    # Overall progress 0-100 — driven by the export callback's
    # (stage, fraction) emissions, mapped to a single 0-100 axis
    # so the dialog's VProgressLinear has a smooth fill.
    state.study_export_pending_progress = 0
    # Dedicated progress dialog — auto-opens when the export
    # starts so the navbar entry produces visible UI even when
    # no project-detail dialog is up.
    state.show_export_study_dialog = False

    # ---- Import dialog flow ----
    state.show_import_study_dialog = False
    state.study_import_pending_busy = False
    state.study_import_pending_status = ""
    state.study_import_pending_error = ""
    # Rolling text log of the last N progress lines — shown in
    # the dialog under the progress bar for verbose feedback.
    state.study_import_pending_progress_log = ""
    # VFileInput v_model — populated by trame's file-upload
    # binding when the user picks a .zip. Set to None after
    # the import completes so the next dialog open starts
    # clean.
    state.study_import_upload = None
    # Base64-encoded bytes of the uploaded zip, stashed between
    # the "peek manifest" step (which sets it) and the
    # "reproduction run" step (which consumes it). Plain bytes
    # don't survive state serialisation cleanly; base64 round-
    # trips fine.
    state.study_import_upload_bytes_b64 = ""
    # Manifest peek summary — multi-line string shown above the
    # Reproduction Run CTA so the user knows what they're about
    # to import.
    state.study_import_manifest_summary = ""
    state.study_import_ready = False
    # Overall progress 0-100 — driven by the import callback's
    # (stage, fraction) emissions. Indeterminate during the
    # initial upload phase (browser → server), determinate
    # during the server-side unpack.
    state.study_import_pending_progress = 0
    # True while the browser → server upload is in flight (the
    # window between the user picking a file and the
    # state.change watcher firing). The dialog shows an
    # indeterminate VProgressLinear during this gap.
    state.study_import_uploading = False

    # Alternative ingest path for big bundles: server reads the
    # .zip directly from a path on disk (typed/pasted by the
    # user). Bypasses the browser-side msgpack/ArrayBuffer cap
    # that kills the VFileInput WS path for files larger than
    # ~100 MB.
    state.study_import_path_on_disk = ""
    # When the user picked the disk-path flow, this holds the
    # resolved Path string for the subsequent import step (so
    # we don't have to slurp 1.5 GB of bytes into state). When
    # the user picked the WS upload flow, this stays empty and
    # the run step reads bytes from study_import_upload_bytes_b64.
    state.study_import_resolved_path = ""
