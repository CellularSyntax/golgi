# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cuff-designer dialog watchers — preset picker + per-param sliders.

Extracted from build_app in step W1.7c of FEATURES.md.

Two watchers:
- `cuff_preset_name`: when the user picks a preset from the dropdown,
  refresh the visible-state mirror + rebuild the offscreen preview.
- `cuff_p_<name>` multi-key: when any of the per-preset visible
  parameter sliders moves, rebuild the preview. Guard-skipped while
  the populator is busy filling the mirror (so the chain of
  state-writes triggered by populate_* doesn't kick off N preview
  rebuilds).

Both watchers are gated on `state.show_cuff_designer_dialog` so they
no-op when the dialog is closed (the cuff-preview pl is hidden then,
so a rebuild would be wasted work + might race the rendering loop).
"""
from __future__ import annotations

from typing import Callable, Iterable


def register(
    state,
    *,
    cuff_visible_names: Iterable[str],
    populate_cuff_visible_state: Callable[[], None],
    rebuild_cuff_preview: Callable[[], None],
    cuff_designer_guard: dict,
) -> None:
    """Wire the cuff-designer dialog's preset + per-param watchers.

    `cuff_visible_names` is the live list of `cuff_p_<name>` keys
    to listen on — built from the loaded cuff_designer presets at
    module load (golgi/app.py:_CUFF_ALL_VISIBLE_NAMES).
    `cuff_designer_guard["populating"]` is set to True by
    populate_cuff_visible_state while it's writing state — the
    per-param watcher must NOT rebuild during that window.
    """

    @state.change("cuff_preset_name")
    def _on_cuff_preset_change(**_kwargs):
        if not state.show_cuff_designer_dialog:
            return
        populate_cuff_visible_state()
        rebuild_cuff_preview()

    @state.change(*[f"cuff_p_{_n}" for _n in cuff_visible_names])
    def _on_cuff_visible_param_change(**_kwargs):
        if cuff_designer_guard["populating"]:
            return
        if not state.show_cuff_designer_dialog:
            return
        rebuild_cuff_preview()
