# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Conductivity-tab watchers — σ value + preset → value pair."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def register(
    state,
    *,
    default_sigma: dict,
    sigma_match_label: Callable,
    sigma_preset_lookup: Callable,
    active_project_out_dir: Callable,
) -> None:
    """Wire σ value watcher (persists + syncs preset dropdowns)
    + per-preset → value watcher (one per tissue)."""

    @state.change(*default_sigma.keys())
    def _on_sigma_change(**_kwargs):
        # Persist every edit to conductivities.json so subsequent
        # sims pick up the user's values without a manual save
        # step.
        cfg = {k: float(state[k]) for k in default_sigma}
        try:
            (Path(active_project_out_dir())
             / "conductivities.json").write_text(
                json.dumps(cfg, indent=2),
            )
        except Exception:
            pass
        # Clear the "✓ Conductivities updated" chip from the
        # Update button.
        if state.sigma_update_status:
            state.sigma_update_status = ""
        # Sync each preset dropdown to reflect the current value.
        for _k in default_sigma:
            preset_key = f"{_k}_preset"
            new_label = sigma_match_label(_k, float(state[_k]))
            if state[preset_key] != new_label:
                state[preset_key] = new_label

    # Per-tissue preset → value watcher. Picking a preset from
    # the dropdown sets the corresponding σ. We bind one watcher
    # per key (rather than a single shared one) because the
    # @state.change kwargs don't tell us which key fired in the
    # multi-key form — and we need to know to do the lookup.
    def _make_preset_watcher(sigma_key: str):
        preset_key = f"{sigma_key}_preset"

        @state.change(preset_key)
        def _on_preset_change(**_kwargs):
            label = str(state[preset_key])
            val = sigma_preset_lookup(sigma_key, label)
            if val is None:
                return  # "Custom value" or unknown → no-op
            if float(state[sigma_key]) != float(val):
                state[sigma_key] = float(val)
        return _on_preset_change

    for _k in default_sigma:
        _make_preset_watcher(_k)
