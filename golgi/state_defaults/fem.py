# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""FEM-solve panel state defaults (Phase B / §9)."""
from __future__ import annotations


def register(state) -> None:
    state.has_fem = False             # set when an FEM solve completes
    state.I_stim_mA = 1.0             # total cathodic stim current
    state.fem_slice_z_idx = 20        # 0..N_SLICES-1, default = mid
    state.fem_field = "Ve"            # "Ve" or "Emag"
    state.fem_axis_b64 = ""           # axis-line plot PNG data URI
    state.fem_slice_b64 = ""          # slice heatmap plot PNG data URI
    # §10 activation-function plot. Sliders mirror nerve_studio:
    # `fem_fiber_sel` highlights one fiber on top of the per-
    # branch ribbon; `fem_sg_window` is the AF smoothing window
    # (σ ≈ window/6 samples for the light Gaussian pre-smooth — see
    # golgi.figures.fem._activation_function; wider = smoother AF).
    state.fem_fiber_sel = 0           # index into the surviving fiber list
    state.fem_sg_window = 9           # AF Gaussian smoothing window (≥ 5)
    state.fem_af_b64 = ""             # legacy AF PNG (kept for now)
    # Plotly figure state for the 2×2 Solve tab. Each tile gets
    # its own state key so the AF slider can update ONLY the AF
    # figure without touching the others.
    state.fem_slice_figure = {"data": [], "layout": {}}
    state.fem_axis_figure = {"data": [], "layout": {}}
    state.fem_af_figure = {"data": [], "layout": {}}
    # Horizontal Vₑ colourbar (plasma cmap) that sits over the
    # 3D viewport so the colour-coded endo/epi surfaces + the
    # Vₑ-on-fibers tubes have a shared scale legend.
    state.fem_ve_cbar_b64 = ""
    state.fem_status = "No FEM run yet."
    state.fem_failed = False
    state.fem_log = ""                # full log on failure
    # Solver preset selector (Step 7.1b). Drives the --preset
    # CLI flag passed to solve_nerve.py. "Balanced" reproduces
    # the pre-7.1 hard-coded PETSc options.
    state.fem_preset = "Balanced"     # "Quick" | "Balanced" | "HPC"
    state.fem_preset_options = [
        "Quick", "Balanced", "HPC",
    ]
    # F3.1: per-design FEM layout. `fem_designs` is the list the
    # analysis drawer reads to populate its design switcher;
    # `active_design_id` is the currently-loaded design (drives
    # which design's outputs feed the plots + the 3D viewport
    # overlays). Both stay empty / "default" until run_fem_solve
    # populates them.
    state.fem_designs = []
    state.active_design_id = ""
