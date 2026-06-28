# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Import-drawer watchers — scale-factor preset picker."""
from __future__ import annotations


def register(state) -> None:
    """Wire the scale_preset → scale_factor mapping for the import
    drawer's mm / µm / m / cm dropdown. 'custom' leaves
    scale_factor untouched so the user's typed value sticks."""

    @state.change("scale_preset")
    def _on_scale_preset_change(**_kwargs):
        _presets = {
            "mm → m (×1e-3)": 1.0e-3,
            "µm → m (×1e-6)": 1.0e-6,
            "m → m (×1)": 1.0,
            "cm → m (×1e-2)": 1.0e-2,
        }
        if state.scale_preset in _presets:
            state.scale_factor = _presets[state.scale_preset]
        # 'custom' → leave scale_factor untouched so the user's
        # typed value sticks.
