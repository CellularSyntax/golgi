# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Import drawer state defaults — file picker + scale presets."""
from __future__ import annotations

from typing import Callable


def register(state, *, list_data_files: Callable) -> None:
    """Seed import-drawer state defaults.

    `list_data_files` is the module-level helper in golgi.py
    that scans HERE/data/ for STL/NAS/OBJ files."""
    state.data_files = list_data_files()
    state.selected_file = (
        state.data_files[0] if state.data_files else None
    )
    state.scale_preset = "mm → m (×1e-3)"
    state.scale_factor = 1.0e-3   # arbitrary multiplier, source → m
    state.upload_info = ""
    # Optional explicit epineurium surface (STL flow). When set, the
    # picked source file is treated as the endoneurium and a multi-
    # region epi+endo (uct_bundle) nerve is built — see
    # do_load_geometry + _on_import_source_type_change. The two
    # show_stl_* booleans gate the step-2 offset generator vs. the
    # "epi from imported surface" note (dedicated state vars because
    # compound v_show expressions are unreliable in this build).
    state.selected_epi_file = ""
    state.epi_upload_file = None
    state.stl_has_epi = False
    state.show_stl_offset = True
    state.show_stl_epi_note = False
