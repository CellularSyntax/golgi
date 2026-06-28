# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""HTTP POST route for µCT / medical-image uploads (V1 Phase A.6).

The Segment-µCT dialog needs to accept files that routinely run
100s of MB to several GB (multi-page TIFF stacks, DICOM series
zipped up, full NIfTI volumes). Same WebSocket caps apply as for
F2.2 study bundles — Vuetify's VFileInput v-model rides through
wslink's reassembly buffer + the browser msgpack ArrayBuffer
ceiling, both of which torpedo big uploads.

The fix is identical to `projects/upload_route.py`: register a
plain `multipart/form-data` POST route on the underlying aiohttp
app, stream the body to disk in 64 kB chunks, return a JSON
payload with the on-disk path. The Trame dialog's JS side
posts to this route via XHR (so the user sees real progress)
and feeds the returned path back into the action handler
through a state-change trigger.

Difference from study uploads: this route writes into the
ACTIVE project's `uct/uploads/` subdirectory rather than a
shared downloads_dir, so the uploaded image is colocated with
the segmentation it produces. The project lookup happens at
upload time (not registration time) so switching projects mid-
session "just works"."""
from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import Callable


def register(
    server,
    *,
    get_active_project_dir: Callable[[], Path],
    on_upload_complete: Callable[[str], None] | None = None,
    on_upload_progress: (
        Callable[[int, str, int, int], None] | None
    ) = None,
    route_path: str = "/api/uct/upload",
) -> None:
    """Hook the µCT upload route onto trame's aiohttp app via the
    on_server_bind controller event.

    `get_active_project_dir()` returns the live ActiveProject's
    out_dir; we mkdir the `uct/uploads/` subtree under it on
    every upload, so a clean project switch just changes the
    destination, no re-registration needed.

    `on_upload_complete(server_path)` is dispatched onto the
    main asyncio loop AFTER the response goes out so the segment
    dialog can fire the do_load_uct_stack handler without
    requiring a JS round-trip.

    `on_upload_progress(file_idx, file_name, file_bytes,
    total_bytes_so_far)` (optional) is dispatched onto the main
    asyncio loop AFTER each file part has been written to disk.
    Lets the caller push a per-file status line into state for
    visual progress — most useful on multi-file DICOM series
    uploads where the JS xhr.upload.onprogress only reports
    cumulative byte progress, not file boundaries."""
    from aiohttp import web

    async def _handle_upload(request: "web.Request"):
        """Stream multipart body → file under <project>/uct/
        uploads/, return JSON {path, size, name}."""
        token = secrets.token_hex(6)
        out_path: "Path | None" = None
        size_total = 0
        # Resolve the destination dir at upload time so it
        # tracks project switches. Bail out if no project is
        # active — the dialog gates the upload button on
        # `has_active_project`, but defensive 4xx here too.
        try:
            project_dir = Path(get_active_project_dir())
        except Exception as ex:                          # noqa: BLE001
            return web.json_response(
                {"error": (
                    f"no active project: "
                    f"{type(ex).__name__}: {ex}"
                )},
                status=409,
            )
        if not project_dir.exists():
            return web.json_response(
                {"error": (
                    f"active project dir does not exist: "
                    f"{project_dir}"
                )},
                status=409,
            )
        uct_dir = project_dir / "uct" / "uploads"
        uct_dir.mkdir(parents=True, exist_ok=True)
        try:
            reader = await request.multipart()
        except Exception as ex:                          # noqa: BLE001
            return web.json_response(
                {"error": (
                    f"not a multipart body: "
                    f"{type(ex).__name__}: {ex}"
                )},
                status=400,
            )
        # Collect all `file` parts. One file = legacy single-
        # stack flow (TIFF / NIfTI / NRRD / ...); N files =
        # DICOM series (the user dropped N .dcm files), in
        # which case we write them into a fresh series subdir
        # and return that subdir path. `load_stack` already
        # auto-detects a DICOM directory via _is_dicom_dir, so
        # the loader needs no changes — just hand it the dir.
        written: list[Path] = []
        series_dir: "Path | None" = None
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name != "file":
                continue
            filename = (
                part.filename or f"upload_{token}.bin"
            )
            safe_name = (
                "".join(
                    c if c.isalnum() or c in "._-" else "_"
                    for c in Path(filename).name
                ).strip("_")
                or "upload"
            )
            # If this is the second-or-later file part, we're
            # in series-upload mode. Lazily create the subdir
            # on first multi-part detection so single-file
            # uploads land directly in uct_dir as before.
            if len(written) == 0:
                target = uct_dir / f"{token}_{safe_name}"
            else:
                if series_dir is None:
                    # Promote the previously-written single
                    # file into a series subdir alongside the
                    # rest. Use the FIRST file's base name
                    # (stripped of its safe-name token) as the
                    # series dir name so it's recognisable.
                    series_dir = uct_dir / f"{token}_series"
                    series_dir.mkdir(parents=True, exist_ok=True)
                    first = written[0]
                    # Move the first file's stem (drop the
                    # leading "<token>_") inside the series
                    # dir so all parts live together.
                    new_first = (
                        series_dir / first.name[len(token) + 1:]
                    )
                    first.replace(new_first)
                    written[0] = new_first
                target = series_dir / safe_name
            file_bytes = 0
            with open(target, "wb") as f:
                while True:
                    chunk = await part.read_chunk(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    file_bytes += len(chunk)
                    size_total += len(chunk)
            written.append(target)
            if on_upload_progress is not None:
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(
                        on_upload_progress,
                        int(len(written)),
                        str(target.name),
                        int(file_bytes),
                        int(size_total),
                    )
                except Exception as ex:                      # noqa: BLE001
                    print(
                        f"[uct-upload] on_upload_progress "
                        f"dispatch failed: "
                        f"{type(ex).__name__}: {ex}",
                        flush=True,
                    )
        if not written:
            return web.json_response(
                {"error": "no `file` part in multipart body"},
                status=400,
            )
        # Decide what path to return + dispatch to the action
        # layer. Single file → file path; ≥2 files → series dir.
        if series_dir is not None:
            out_path = series_dir
            print(
                f"[uct-upload] series · {len(written)} files · "
                f"{size_total / (1024 * 1024):.2f} MB → "
                f"{out_path}",
                flush=True,
            )
            # NOTE: compression (DICOM → volume.nii.gz)
            # happens DOWNSTREAM in do_load_uct_stack, NOT
            # here. Doing it inline blocks the aiohttp
            # response until the conversion finishes (10-60s
            # for a sheep VN scan) without any user-visible
            # status — the browser shows the upload XHR as
            # "still in flight" with no feedback. Moving it
            # into the action layer lets us drive the
            # dialog's busy lightbox + busy_log with explicit
            # progress messages.
        else:
            out_path = written[0]
            print(
                f"[uct-upload] {size_total / (1024 * 1024):.2f} "
                f"MB → {out_path}",
                flush=True,
            )
        if on_upload_complete is not None:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    on_upload_complete, str(out_path),
                )
            except Exception as ex:                      # noqa: BLE001
                print(
                    f"[uct-upload] on_upload_complete "
                    f"scheduling failed: "
                    f"{type(ex).__name__}: {ex}",
                    flush=True,
                )
        return web.json_response({
            "path": str(out_path),
            "size": size_total,
            "name": out_path.name,
            "n_files": len(written),
            "series": series_dir is not None,
        })

    @server.controller.on_server_bind.add
    def _add_uct_route(wslink_server):
        """Register the POST route before aiohttp starts
        listening. on_server_bind is the only window — once
        runner.setup() runs, the router is locked."""
        app = wslink_server.app
        app.router.add_post(route_path, _handle_upload)
        try:
            app._client_max_size = 1024 ** 4  # 1 TB ceiling
        except Exception:                                # noqa: BLE001
            pass
        print(
            f"[uct-upload] POST {route_path} registered "
            f"on the aiohttp app",
            flush=True,
        )
