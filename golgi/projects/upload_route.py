# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""HTTP POST route for study-bundle uploads (F2.2).

The WebSocket path that VFileInput's v-model rides through has
TWO hard caps that kill big uploads:

  * Server: wslink's MAX_MSG_SIZE (bumped to 512 MB in app.py,
    but reassembly buffers still cost RAM)
  * Client: browser msgpack throws RangeError when it can't
    allocate a single ArrayBuffer big enough for the file
    (Chrome caps at ~2 GB practically, often less)

The fix is to bypass the WS entirely for bundle uploads: a
plain `multipart/form-data` POST to a custom aiohttp route. The
browser streams the file body in TCP-friendly chunks, the
server writes it straight to disk via aiohttp's
`MultipartReader.read_chunk` loop — neither side has to hold
the whole bundle in memory at once. This is the same pattern
every HTML-form file upload has used since 1997; we just
needed to wire it past trame's WS-first plumbing.

Returns JSON `{path, size, name}` on success — the path is the
on-disk location of the uploaded zip, which the client then
hands back through a server trigger so the existing manifest-
peek + import flow picks it up unchanged."""
from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import Callable


def register(
    server,
    *,
    downloads_dir: Path,
    on_upload_complete: Callable[[str], None] | None = None,
) -> None:
    """Hook into trame's `on_server_bind` controller event to
    register the POST route on the underlying aiohttp app
    BEFORE it starts listening.

    `downloads_dir` is the dir where uploaded zips land — same
    dir used for the bundle DOWNLOAD path so cleanup logic
    sweeps both ways. `on_upload_complete(server_path)` is
    invoked on the main asyncio loop after a successful upload
    so the action handler can fire the manifest peek without
    requiring a JS roundtrip."""
    from aiohttp import web

    async def _handle_upload(request: "web.Request"):
        """Stream a multipart POST body to a fresh file under
        downloads_dir + return its path."""
        token = secrets.token_hex(8)
        out_path: "Path | None" = None
        size_total = 0
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
        while True:
            part = await reader.next()
            if part is None:
                break
            # We expect a single "file" part; ignore anything
            # else (form-fields, etc.).
            if part.name != "file":
                continue
            filename = (
                part.filename
                or f"upload_{token}.zip"
            )
            # Sanitise the filename to a safe basename so a
            # client can't traverse out of downloads_dir.
            safe_name = (
                "".join(
                    c if c.isalnum() or c in "._-" else "_"
                    for c in Path(filename).name
                ).strip("_")
                or "upload"
            )
            out_path = downloads_dir / (
                f"{token}_{safe_name}"
            )
            # Streamed write — 64 KB at a time so a 10 GB
            # upload costs ~64 KB resident on the server.
            with out_path.open("wb") as f:
                while True:
                    chunk = await part.read_chunk(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    size_total += len(chunk)
            break
        if out_path is None:
            return web.json_response(
                {"error": "no `file` part in multipart body"},
                status=400,
            )
        print(
            f"[study-upload] received {size_total / (1024 * 1024):.2f} MB "
            f"→ {out_path}",
            flush=True,
        )
        # Fire the action handler on the asyncio main loop. We
        # schedule rather than await so the HTTP response goes
        # out immediately + the manifest peek runs in the
        # background (the next state.flush pushes the result
        # to the dialog).
        if on_upload_complete is not None:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    on_upload_complete, str(out_path),
                )
            except Exception as ex:                      # noqa: BLE001
                print(
                    f"[study-upload] on_upload_complete "
                    f"scheduling failed: "
                    f"{type(ex).__name__}: {ex}",
                    flush=True,
                )
        return web.json_response({
            "path": str(out_path),
            "size": size_total,
            "name": out_path.name,
        })

    @server.controller.on_server_bind.add
    def _add_route(wslink_server):
        """`on_server_bind` runs after wslink creates the aiohttp
        app but BEFORE it starts listening — the only window we
        can add routes in without poking the live router (which
        is locked once `runner.setup()` has run)."""
        app = wslink_server.app
        app.router.add_post(
            "/api/study/upload", _handle_upload,
        )
        # No client-max-size cap — aiohttp's default is 1 MB
        # which would torpedo the whole point. Bump to the
        # process's available memory (None = unbounded).
        try:
            app._client_max_size = 1024 ** 4  # 1 TB ceiling
        except Exception:                                # noqa: BLE001
            pass
        print(
            "[study-upload] POST /api/study/upload registered "
            "on the aiohttp app",
            flush=True,
        )
