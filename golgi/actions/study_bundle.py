# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Study-bundle action handlers (F2.2 Phase 1+2).

Three handlers wire the project-detail dialog + navbar Import
Nerve menu to `golgi.projects.bundle`:

  - do_export_study           pack the active project into a .zip,
                              push as a data URI for browser
                              download.
  - do_import_study_open      open the upload dialog. The actual
                              file-upload state-binding (the
                              VFileInput) lives in the dialog;
                              this just flips the show_* flag.
  - do_import_study_upload    server-side: receive the uploaded
                              bytes, unzip, register the new
                              project, fire audit + open it.

All long ops raise the global busy lightbox and push progress
into `state.busy_msg + busy_log` so the user sees activity
through the (potentially multi-minute) export of a big study.
"""
from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import json
import traceback
from pathlib import Path
from typing import Callable

from golgi.projects import bundle as _bundle


def _safe_filename(name: str) -> str:
    """Project name → filesystem-safe `_`-joined slug. Drops
    everything outside [A-Za-z0-9._-]. Empty input falls back to
    'project'."""
    cleaned = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in (name or "")
    ).strip("_")
    return cleaned or "project"


def register(
    state,
    *,
    pipeline_ctx,
    projects_root: Path,
    downloads_dir: Path | None = None,
    downloads_endpoint: str = "_downloads",
    do_open_project: Callable | None = None,
    refresh_projects_list: Callable | None = None,
) -> dict[str, Callable]:
    """Wire the study-bundle handlers. Returns the do_* bag for
    build_app to splat into the navbar / dialog click handlers.

    `projects_root` is the root under which new project dirs land
    on import (PROJECTS_ROOT in app.py). `do_open_project(pdir)`
    is called after a successful import so the user lands in the
    just-imported project. `refresh_projects_list` triggers the
    welcome-view tile rebuild so the imported project appears
    without a page reload.

    `downloads_dir` is the on-disk path registered as a trame
    static endpoint (`server.serve[downloads_endpoint]`) so the
    export handler can stream large bundles via plain HTTP
    instead of a base64 data URI. Without it, the handler falls
    back to the data-URI path (which stalls for big bundles)."""

    # ---- EXPORT ----

    async def do_export_study(*args) -> None:
        """Bundle a project + push the .zip as a data URI.

        First positional arg (when present) is the project dir to
        export — the project-detail dialog's "Export study"
        button passes `detail_project.dir`, which lets the user
        export ANY visible project (open or closed) from the
        welcome view. With no arg, falls back to the active
        project's dir from `state.current_project_dir`."""
        # Resolve project dir — explicit arg wins.
        project_dir_arg = ""
        if args:
            a = args[0]
            if isinstance(a, (str, bytes)):
                project_dir_arg = str(a)
            elif isinstance(a, dict):
                project_dir_arg = str(a.get("dir", "") or "")
            elif a is not None:
                project_dir_arg = str(a)
        if project_dir_arg:
            project_dir = (
                Path(project_dir_arg).expanduser().resolve()
            )
        elif bool(getattr(state, "has_active_project", False)):
            cdir = str(
                getattr(state, "current_project_dir", "") or "",
            )
            if not cdir:
                with state:
                    state.study_export_pending_error = (
                        "No active project dir on state — "
                        "open a project first."
                    )
                state.flush()
                return
            project_dir = Path(cdir).expanduser().resolve()
        else:
            with state:
                state.study_export_pending_error = (
                    "No project to export — open one or use "
                    "the project detail dialog's Export Study "
                    "button on the welcome view."
                )
            state.flush()
            return
        if not project_dir.is_dir():
            with state:
                state.study_export_pending_error = (
                    f"project dir not found: {project_dir}"
                )
            state.flush()
            return
        # Project name comes from the on-disk manifest so the
        # bundle filename + cover read correctly for closed
        # projects too (state.current_project_name is empty
        # before a project is open).
        try:
            import json as _json
            pj = _json.loads(
                (project_dir / "project.json").read_text(
                    encoding="utf-8",
                ),
            )
            project_name = str(
                pj.get("name", project_dir.name),
            )
        except Exception:                                # noqa: BLE001
            project_name = (
                str(getattr(state, "current_project_name", ""))
                or project_dir.name
            )
        user_id = (
            int(state.current_user_id)
            if getattr(state, "current_user_id", None) else None
        )
        user_label = (
            f"{state.current_user_first_name} "
            f"{state.current_user_last_name}".strip()
            or str(getattr(state, "current_user_username", ""))
            or str(getattr(state, "current_user_email", ""))
        )

        # Raise the busy lightbox + reset the pending slot +
        # auto-open the dedicated Export Study dialog so the
        # progress + Download anchor are always visible (the
        # project-detail status strip is only visible when THAT
        # dialog is open).
        with state:
            state.study_export_pending_busy = True
            state.study_export_pending_data_uri = ""
            state.study_export_pending_filename = ""
            state.study_export_pending_error = ""
            state.study_export_pending_status = (
                "Scanning project files…"
            )
            state.study_export_pending_progress = 0
            state.show_export_study_dialog = True
            state.busy = True
            state.busy_msg = "Exporting study bundle…"
            state.busy_log = ""
        state.flush()

        progress_lines: list[str] = []

        # Map per-stage progress onto a single 0-100 overall
        # axis. `scan` and `manifest` are tiny — `files` dominates
        # so it gets most of the bar. Stages outside this map
        # default to no overall update.
        STAGE_WEIGHTS = {
            "scan":     (0, 5),
            "files":    (5, 95),
            "manifest": (95, 100),
        }

        def _emit(stage: str, frac: float) -> None:
            line = (
                f"  {stage:10s} {int(frac * 100):3d}%"
            )
            progress_lines.append(line)
            tail = progress_lines[-10:]
            # Compute the overall percent based on the stage
            # weights — clamped to [0, 100] in case a caller
            # emits frac>1 or <0.
            lo, hi = STAGE_WEIGHTS.get(stage, (0, 100))
            overall = int(round(
                lo + max(0.0, min(1.0, float(frac))) * (hi - lo),
            ))
            try:
                with state:
                    state.busy_log = "\n".join(tail)
                    state.study_export_pending_status = (
                        f"Exporting · {stage} "
                        f"({int(frac * 100)}%)"
                    )
                    state.study_export_pending_progress = overall
            except Exception:                            # noqa: BLE001
                pass
            state.flush()

        loop = asyncio.get_event_loop()
        try:
            blob = await loop.run_in_executor(
                None,
                lambda: _bundle.export_study(
                    project_dir,
                    exported_by_user=user_label,
                    exported_by_user_id=user_id,
                    on_progress=_emit,
                ),
            )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[study-export] failed: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )
            traceback.print_exc()
            with state:
                state.study_export_pending_busy = False
                state.study_export_pending_status = ""
                state.study_export_pending_error = (
                    f"export failed: "
                    f"{type(ex).__name__}: {ex}"
                )
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
            state.flush()
            return

        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"golgi_study_{_safe_filename(project_name)}_"
            f"{stamp}.zip"
        )
        # Prefer HTTP-route delivery: write the bundle to the
        # trame static-serve dir + set the anchor href to the
        # public URL. The browser streams the file via plain
        # HTTP, no base64 inflation, no DOM pressure, instant
        # save dialog. Falls back to a data URI for tiny bundles
        # (<8 MB) or when no downloads_dir was configured.
        anchor_href = ""
        try:
            import secrets
            if (downloads_dir is not None
                    and downloads_dir.is_dir()):
                token = secrets.token_hex(8)
                out_path = downloads_dir / (
                    f"{token}_{filename}"
                )
                out_path.write_bytes(blob)
                anchor_href = (
                    f"/{downloads_endpoint}/"
                    f"{out_path.name}"
                )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[study-export] downloads-dir write failed, "
                f"falling back to data URI: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )
        if not anchor_href:
            # Fallback — same base64 data URI as v1. Slow for
            # big bundles but works without a static endpoint.
            b64 = base64.b64encode(blob).decode("ascii")
            anchor_href = f"data:application/zip;base64,{b64}"

        with state:
            state.study_export_pending_busy = False
            state.study_export_pending_data_uri = anchor_href
            state.study_export_pending_filename = filename
            state.study_export_pending_status = (
                f"✓ Bundle ready · "
                f"{len(blob) / (1024 * 1024):.2f} MB"
            )
            state.study_export_pending_progress = 100
            state.study_export_pending_error = ""
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
        state.flush()

        # Audit.
        try:
            from golgi.auth.audit import _audit_log
            _audit_log(
                user_id=user_id,
                action="study_exported",
                payload={
                    "filename": filename,
                    "bytes": len(blob),
                    "project_name": project_name,
                },
                project_dir=str(project_dir),
                status="success",
            )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[study-export] audit failed: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )

        print(
            f"[study-export] ready · "
            f"{len(blob) / 1024:.0f} KB · {filename}",
            flush=True,
        )

    # ---- IMPORT ----

    def do_export_study_dialog_close(*_args) -> None:
        """Close the Export Study progress dialog. Safe to call
        while an export is in flight — only flips the show flag,
        the busy lightbox + the action coroutine keep running."""
        with state:
            state.show_export_study_dialog = False
        state.flush()

    def do_import_study_open(*_args) -> None:
        """Open the Import Study dialog. Resets the pending slot
        so a previous import's error/state doesn't leak in."""
        with state:
            state.show_import_study_dialog = True
            state.study_import_pending_busy = False
            state.study_import_pending_error = ""
            state.study_import_pending_status = ""
            state.study_import_pending_progress = 0
            state.study_import_uploading = False
            # The VFileInput's v-model goes here.
            state.study_import_upload = None
            state.study_import_upload_bytes_b64 = ""
            state.study_import_path_on_disk = ""
            state.study_import_resolved_path = ""
            state.study_import_manifest_summary = ""
            state.study_import_ready = False
        state.flush()

    def do_import_study_close(*_args) -> None:
        """Close the dialog. Long-running import is NOT
        interruptible from here — Cancel is wired separately
        below via clear_subprocess on the pipeline_ctx."""
        with state:
            state.show_import_study_dialog = False
        state.flush()

    def _ingest_uploaded_bundle(upload) -> None:
        """Process the VFileInput's v-model payload. Called from
        the state.change watcher that build_app installs on
        `study_import_upload` (the v-model target). Synchronous —
        manifest peek + file-bytes stash, no I/O. Async unpack
        is a separate handler (`do_import_study_run`).

        Trame's VFileInput hands us either a list of dicts
        ({name, size, content}) or a single dict; content is
        bytes (raw upload) or a base64 str (WS-serialised).
        Normalises both shapes."""
        # Diagnostic — confirms whether the state.change watcher
        # actually fired AND what shape trame is handing us.
        # Stays in for now so any future "file pick does
        # nothing" report has obvious terminal output to share.
        print(
            f"[study-import] _ingest_uploaded_bundle fired · "
            f"upload type={type(upload).__name__} · "
            f"truthy={bool(upload)}",
            flush=True,
        )
        if upload is not None:
            try:
                if isinstance(upload, list) and upload:
                    e = upload[0]
                    print(
                        f"[study-import]   first entry keys: "
                        f"{list(e.keys()) if isinstance(e, dict) else type(e).__name__}"
                        + (
                            f", name={e.get('name')!r}, "
                            f"size={e.get('size')}, "
                            f"content type="
                            f"{type(e.get('content')).__name__}, "
                            f"content len="
                            f"{len(e.get('content') or b'')}"
                            if isinstance(e, dict) else ""
                        ),
                        flush=True,
                    )
                elif isinstance(upload, dict):
                    print(
                        f"[study-import]   dict keys: "
                        f"{list(upload.keys())}",
                        flush=True,
                    )
            except Exception as _ex:                     # noqa: BLE001
                print(
                    f"[study-import]   diag print failed: "
                    f"{_ex}",
                    flush=True,
                )
        # Once we enter this function, the upload has arrived
        # at the server. Flip uploading=False so the dialog's
        # indeterminate bar collapses.
        if not upload:
            with state:
                state.study_import_uploading = False
                state.study_import_manifest_summary = ""
                state.study_import_ready = False
            state.flush()
            return
        if isinstance(upload, list):
            upload = upload[0] if upload else None
        if upload is None:
            with state:
                state.study_import_pending_error = (
                    "Upload payload empty."
                )
                state.study_import_uploading = False
            state.flush()
            return
        content = upload.get("content") if isinstance(upload, dict) else upload
        if isinstance(content, str):
            try:
                content = base64.b64decode(content)
            except Exception:                            # noqa: BLE001
                pass
        if not isinstance(content, (bytes, bytearray)):
            with state:
                state.study_import_pending_error = (
                    f"unsupported upload payload "
                    f"type {type(content).__name__}"
                )
                state.study_import_uploading = False
            state.flush()
            return
        # Stash the bytes for the subsequent reproduction-run
        # step. Clear the path source so we don't have two
        # ambiguous handles waiting for the run step.
        state.study_import_upload_bytes_b64 = base64.b64encode(
            bytes(content),
        ).decode("ascii")
        state.study_import_resolved_path = ""
        # Peek the manifest.
        try:
            manifest = _bundle.read_manifest(bytes(content))
        except Exception as ex:                          # noqa: BLE001
            with state:
                state.study_import_pending_error = (
                    f"not a valid study bundle: "
                    f"{type(ex).__name__}: {ex}"
                )
                state.study_import_manifest_summary = ""
                state.study_import_ready = False
                state.study_import_uploading = False
            state.flush()
            return
        proj = manifest.get("project", {}) or {}
        size_mb = len(content) / (1024 * 1024)
        summary = (
            f"Bundle from {manifest.get('exported_by', '?')} "
            f"at {manifest.get('exported_at', '?')}\n"
            f"Project: {proj.get('name', '?')}\n"
            f"Size:    {size_mb:.2f} MB · "
            f"{len(manifest.get('files', []))} files\n"
            f"Stages:  "
            + ", ".join(
                f"{s['stage']}{'' if s['present'] else '✗'}"
                for s in manifest.get("dag", [])
            )
        )
        with state:
            state.study_import_manifest_summary = summary
            state.study_import_pending_error = ""
            state.study_import_ready = True
            state.study_import_uploading = False
        state.flush()

    def do_import_study_load_from_disk(*_args) -> None:
        """Read the bundle's MANIFEST from a path on the server's
        local filesystem. NEVER slurps the full file into memory
        — `bundle.read_manifest(path)` reads only the central
        directory + the one manifest entry, so 1.5 GB bundles
        cost the same as 1 KB bundles for the peek.

        IMPORTANT: this path uses the SERVER's filesystem. When
        GOLGI runs locally (desktop), that's the user's
        machine. When deployed on a remote server, the .zip
        must already be on that server (uploaded via SCP / NFS
        mount / etc.). The browser file picker DOES upload
        across the network but is capped by msgpack /
        ArrayBuffer at ~100 MB — bigger bundles need this
        disk-path route + a way to get the file onto the
        server."""
        raw = str(
            getattr(state, "study_import_path_on_disk", "") or "",
        ).strip()
        if not raw:
            with state:
                state.study_import_pending_error = (
                    "Type or paste the path to the .zip on disk."
                )
                state.study_import_ready = False
                state.study_import_manifest_summary = ""
            state.flush()
            return
        # Strip surrounding quotes (drag-and-drop on macOS often
        # quotes the path) + expand ~.
        if raw.startswith(("'", '"')) and raw.endswith(("'", '"')):
            raw = raw[1:-1]
        p = Path(raw).expanduser()
        if not p.is_file():
            with state:
                state.study_import_pending_error = (
                    f"file not found: {p}"
                )
                state.study_import_ready = False
                state.study_import_manifest_summary = ""
            state.flush()
            return
        with state:
            state.study_import_uploading = True
            state.study_import_pending_error = ""
        state.flush()
        # Peek the manifest WITHOUT loading the body.
        try:
            manifest = _bundle.read_manifest(p)
        except Exception as ex:                          # noqa: BLE001
            with state:
                state.study_import_uploading = False
                state.study_import_pending_error = (
                    f"not a valid study bundle: "
                    f"{type(ex).__name__}: {ex}"
                )
                state.study_import_ready = False
                state.study_import_manifest_summary = ""
            state.flush()
            return
        # Stash the resolved path for the subsequent import-run
        # step. Empty the bytes slot — only one of the two
        # sources is active at a time.
        proj = manifest.get("project", {}) or {}
        size_mb = p.stat().st_size / (1024 * 1024)
        summary = (
            f"Bundle from {manifest.get('exported_by', '?')} "
            f"at {manifest.get('exported_at', '?')}\n"
            f"Project: {proj.get('name', '?')}\n"
            f"Size:    {size_mb:.2f} MB · "
            f"{len(manifest.get('files', []))} files\n"
            f"Stages:  "
            + ", ".join(
                f"{s['stage']}{'' if s['present'] else '✗'}"
                for s in manifest.get("dag", [])
            )
            + f"\nSource:  {p}"
        )
        with state:
            state.study_import_resolved_path = str(p)
            state.study_import_upload_bytes_b64 = ""
            state.study_import_manifest_summary = summary
            state.study_import_pending_error = ""
            state.study_import_ready = True
            state.study_import_uploading = False
        state.flush()

    async def do_import_study_upload(*_args) -> None:
        """Legacy wrapper kept for the dialog's `change=` attr
        if anything still wires it. The real entry point is the
        state.change watcher build_app installs on
        `study_import_upload` — the @change event on VFileInput
        doesn't fire reliably across trame builds, but the
        v-model state change always does."""
        _ingest_uploaded_bundle(state.study_import_upload)

    async def do_import_study_run(*_args) -> None:
        """Unpack the previously-uploaded bundle into a fresh
        project dir under PROJECTS_ROOT, then open it.

        Picks the bundle source in priority order:
          1. `state.study_import_resolved_path` — path on the
             server's disk (set by do_import_study_load_from_disk).
             Streams the zip directly, no full-bundle RAM cost.
          2. `state.study_import_upload_bytes_b64` — bytes from
             a small WS upload (set by _ingest_uploaded_bundle).
        Only one is active at a time; the dialog-open + each
        ingest path clears the other to avoid ambiguity."""
        resolved_path = str(
            getattr(state, "study_import_resolved_path", "")
            or "",
        )
        bundle_source: "bytes | Path | None" = None
        if resolved_path and Path(resolved_path).is_file():
            bundle_source = Path(resolved_path)
        else:
            b64 = getattr(
                state, "study_import_upload_bytes_b64", "",
            )
            if not b64:
                with state:
                    state.study_import_pending_error = (
                        "No bundle source — pick a .zip OR "
                        "paste a path first."
                    )
                state.flush()
                return
            try:
                bundle_source = base64.b64decode(b64)
            except Exception as ex:                      # noqa: BLE001
                with state:
                    state.study_import_pending_error = (
                        f"upload decode failed: "
                        f"{type(ex).__name__}: {ex}"
                    )
                state.flush()
                return
        # Peek manifest for the suggested target dir name. Cheap
        # for both source types — manifest is < 100 KB.
        try:
            manifest = _bundle.read_manifest(bundle_source)
        except Exception as ex:                          # noqa: BLE001
            with state:
                state.study_import_pending_error = (
                    f"manifest read failed: "
                    f"{type(ex).__name__}: {ex}"
                )
            state.flush()
            return
        # Pick a unique target dir. `<projects_root>/<slug>` —
        # append `_N` if a project of that name already exists.
        slug = _safe_filename(
            manifest.get("project", {}).get("name", "imported"),
        )
        target = projects_root / slug
        n = 1
        while target.exists():
            n += 1
            target = projects_root / f"{slug}_{n}"

        owner_user_id = (
            int(state.current_user_id)
            if getattr(state, "current_user_id", None) else None
        )

        # Busy lightbox.
        with state:
            state.study_import_pending_busy = True
            state.study_import_pending_status = (
                f"Unpacking into {target.name}…"
            )
            state.study_import_pending_error = ""
            state.study_import_pending_progress = 0
            state.busy = True
            state.busy_msg = "Importing study bundle…"
            state.busy_log = ""
        state.flush()

        progress_lines: list[str] = []
        # Map per-stage progress onto a single 0-100 axis. Mirrors
        # the export-side stage weights so the dialog's bar fills
        # consistently in both directions.
        STAGE_WEIGHTS = {
            "scan":     (0, 5),
            "files":    (5, 95),
            "manifest": (95, 100),
        }

        def _emit(stage: str, frac: float) -> None:
            line = (
                f"  {stage:10s} {int(frac * 100):3d}%"
            )
            progress_lines.append(line)
            tail = progress_lines[-10:]
            lo, hi = STAGE_WEIGHTS.get(stage, (0, 100))
            overall = int(round(
                lo + max(0.0, min(1.0, float(frac))) * (hi - lo),
            ))
            try:
                with state:
                    state.busy_log = "\n".join(tail)
                    state.study_import_pending_progress_log = (
                        "\n".join(tail)
                    )
                    state.study_import_pending_progress = overall
                    state.study_import_pending_status = (
                        f"Importing · {stage} "
                        f"({int(frac * 100)}%)"
                    )
            except Exception:                            # noqa: BLE001
                pass
            state.flush()

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda src=bundle_source: _bundle.import_study(
                    src,
                    target,
                    owner_user_id=owner_user_id,
                    on_progress=_emit,
                ),
            )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[study-import] failed: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )
            traceback.print_exc()
            with state:
                state.study_import_pending_busy = False
                state.study_import_pending_status = ""
                state.study_import_pending_error = (
                    f"import failed: "
                    f"{type(ex).__name__}: {ex}"
                )
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
            state.flush()
            return

        # Refresh the welcome-view tiles + audit-log the import.
        try:
            from golgi.auth.audit import _audit_log
            _audit_log(
                user_id=owner_user_id,
                action="study_imported",
                payload={
                    "from_exporter": manifest.get(
                        "exported_by", "?",
                    ),
                    "from_at": manifest.get(
                        "exported_at", "?",
                    ),
                    "target": str(target),
                    "n_files": len(
                        manifest.get("files", []),
                    ),
                },
                project_dir=str(target),
                status="success",
            )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[study-import] audit log failed: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )

        if refresh_projects_list is not None:
            try:
                refresh_projects_list()
            except Exception as ex:                      # noqa: BLE001
                print(
                    f"[study-import] refresh failed: "
                    f"{type(ex).__name__}: {ex}",
                    flush=True,
                )

        with state:
            state.study_import_pending_busy = False
            state.study_import_pending_status = (
                f"✓ Imported into {target.name}"
            )
            state.study_import_pending_error = ""
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.show_import_study_dialog = False
        state.flush()
        print(
            f"[study-import] ✓ imported into {target}",
            flush=True,
        )

        # Open the imported project so the user lands in it.
        # do_open_project takes a path string positionally.
        if do_open_project is not None:
            try:
                await do_open_project(str(target))
            except Exception as ex:                      # noqa: BLE001
                print(
                    f"[study-import] open failed: "
                    f"{type(ex).__name__}: {ex}",
                    flush=True,
                )

    return {
        "do_export_study": do_export_study,
        "do_export_study_dialog_close": (
            do_export_study_dialog_close
        ),
        "do_import_study_open": do_import_study_open,
        "do_import_study_close": do_import_study_close,
        "do_import_study_upload": do_import_study_upload,
        "do_import_study_load_from_disk": (
            do_import_study_load_from_disk
        ),
        "do_import_study_run": do_import_study_run,
        # Exposed for build_app's @state.change("study_import_upload")
        # watcher — VFileInput's @change DOM event doesn't fire
        # reliably across trame builds, but the v-model state
        # change does.
        "ingest_uploaded_bundle": _ingest_uploaded_bundle,
    }
