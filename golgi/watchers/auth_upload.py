# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Auth + upload + open-project watchers.

Avatar upload bridges for the login + profile dialogs, the
generic file-upload sink, and the project-tile click bridge that
turns a state-var write into an async do_open_project call.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable


def register(
    state,
    *,
    ingest_avatar_upload: Callable,
    list_data_files: Callable,
    do_open_project: Callable,
    active_upload_dir: Callable,
) -> None:
    """Wire the 4 auth-upload-open watchers. `active_upload_dir`
    is a 0-arg callable returning the current
    ActiveProject.upload_dir Path."""

    @state.change("auth_image_file")
    def _on_auth_image_file(**_kwargs):
        ingest_avatar_upload(
            state.auth_image_file,
            error_var="auth_error",
            target_var="auth_image_data_uri",
        )

    @state.change("profile_image_file")
    def _on_profile_image_file(**_kwargs):
        ingest_avatar_upload(
            state.profile_image_file,
            error_var="profile_error",
            target_var="profile_image_data_uri",
            also_clear_remove="profile_remove_image",
        )

    @state.change("upload_file")
    def _on_upload(**_kwargs):
        info = state.upload_file
        if info is None:
            return
        # Vuetify VFileInput returns a list of dicts {name,
        # content, ...}
        if isinstance(info, list):
            entries = info
        else:
            entries = [info]
        for entry in entries:
            if entry is None or "name" not in entry:
                continue
            name = entry["name"]
            content = entry.get("content")
            if not content:
                continue
            target = Path(active_upload_dir()) / name
            with open(target, "wb") as fh:
                fh.write(content)
            state.upload_info = f"saved {name}"
        state.data_files = list_data_files()

    @state.change("open_project_request")
    def _on_open_project_request(open_project_request, **_kwargs):
        if not open_project_request:
            return
        target = str(open_project_request)
        # Reset before launching the task — if the user mashes a
        # tile twice while one open is in flight, we don't want
        # to double-fire.
        state.open_project_request = ""
        try:
            asyncio.create_task(do_open_project(target))
        except RuntimeError:
            # No running loop (very early startup) — fall back
            # to a synchronous best-effort. Shouldn't happen
            # since the welcome view requires a live server.
            pass
