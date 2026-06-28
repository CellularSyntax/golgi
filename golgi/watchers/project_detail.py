# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Project-detail dialog watchers — open-refresh + label removal.

Extracted from build_app in step W1.7d of FEATURES.md (the last two
inline watchers — completes Phase 5.2's migration goal of moving all
@state.change handlers out of build_app).

Two watchers:
- `show_detail_dialog`, `detail_project` (multi-key): refresh briefs
  whenever the dialog opens OR the target project changes while open.
  Listens to BOTH keys to fix the "need to open a few times before
  stages/activity populate" bug — the inline-JS opener writes both
  keys in sequence and a single-key watcher could fire against a
  stale value.
- `remove_label_request`: pops a label off the project's labels list
  and persists it; the dialog re-reads on the next refresh.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def register(
    state,
    *,
    refresh_detail_briefs: Callable[[], None],
    persist_labels: Callable[[Path, list[str]], None],
) -> None:
    """Wire the two project-detail-dialog watchers.

    `refresh_detail_briefs`: closure that re-reads the project dir
        and re-populates state.detail_status_rows + detail_activity_*.
    `persist_labels`: closure that writes the labels JSON for a
        project directory.
    """

    @state.change("show_detail_dialog", "detail_project")
    def _on_detail_dialog_open(**_kwargs):
        if bool(state.show_detail_dialog):
            refresh_detail_briefs()

    @state.change("remove_label_request")
    def _on_remove_label_request(remove_label_request, **_kwargs):
        if not remove_label_request:
            return
        label = str(remove_label_request)
        state.remove_label_request = ""
        proj = state.detail_project
        if not proj:
            return
        pdir = Path(proj.get("dir", ""))
        if not pdir.is_dir():
            return
        current = list(proj.get("labels", []))
        if label not in current:
            return
        current.remove(label)
        try:
            persist_labels(pdir, current)
        except Exception as ex:                              # noqa: BLE001
            print(
                f"[project] remove label failed: {ex}",
                flush=True,
            )
