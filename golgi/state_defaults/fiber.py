# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Single-fiber simulation tab state defaults."""
from __future__ import annotations


def register(
    state,
    *,
    fiber_diameter_config: dict,
    fiber_diameter_default: dict,
    tab10_palette: "list | tuple",
) -> None:
    """Seed the fiber-tab state defaults.

    `fiber_diameter_config` is FIBER_MODEL_DIAMETER_CONFIG from
    golgi.py; `fiber_diameter_default` is
    _FIBER_MODEL_DIAMETER_DEFAULT; `tab10_palette` is
    TAB10_PALETTE. Passed in by build_app so this module stays
    free of golgi.py imports.
    """
    # Tab10 palette mirrored into state so client-side inline-JS
    # expressions (Population panel's "+ Add fiber type", chip
    # slots) can index into it without hardcoding the colour
    # list in each handler. Kept in sync with the module-level
    # `TAB10_PALETTE` constant.
    state.fiber_tab10_palette = list(tab10_palette)
    # `fiber_sel_idx` is the SINGLE fiber whose plots are
    # currently displayed in the output tiles. After a multi-
    # fiber run, the user switches this via the result picker to
    # view a different fiber's V_m / waterfall plots.
    state.fiber_sel_idx = 0
    # `fiber_sel_indices` is the LIST of fibers the user picked
    # in the VCombobox to simulate. Each entry is a fiber index
    # (int).
    state.fiber_sel_indices = []
    # Active branch tab inside the VCombobox dropdown. Initial
    # value is set to the first branch's idx by
    # `_refresh_fiber_sel_items` once the geom is loaded.
    state.fiber_sel_tab = "0"
    # Dropdown items for the fiber-selector VCombobox. Rebuilt
    # by `_refresh_fiber_sel_items()` after every fiber
    # generation / restore / branch reclassification.
    state.fiber_sel_items = []
    # Per-fiber sim results meta.
    state.fiber_sim_results_meta = []
    state.fiber_backend = "pyfibers"      # "pyfibers" | "axonml"
    state.fiber_model = "MRG_INTERPOLATION"
    state.fiber_diameter_um = 5.7
    # Per-model diameter range — pushed to state so the
    # client-side slider + input pair can bind their min/max
    # /step directly. Updated by the `_on_fiber_model_change`
    # watcher whenever fiber_model flips.
    _diam_cfg = fiber_diameter_config.get(
        "MRG_INTERPOLATION", fiber_diameter_default,
    )
    state.fiber_diameter_min = float(_diam_cfg["min"])
    state.fiber_diameter_max = float(_diam_cfg["max"])
    state.fiber_diameter_step = float(_diam_cfg["step"])
    # `fiber_diameter_ticks` is the list of permitted discrete
    # values for MRG_DISCRETE — empty list for continuous models.
    state.fiber_diameter_ticks = []
    # Pulse type + monophasic widgets
    state.fiber_pulse_type = "monophasic"   # "monophasic" | "biphasic"
    state.fiber_mono_polarity = "cathodic"  # "cathodic" | "anodic"
    state.fiber_mono_amp_mA = 1.0
    state.fiber_mono_pw_us = 1000.0
    # Biphasic widgets
    state.fiber_bi_order = "cathodic-first"  # "cathodic-first" | "anodic-first"
    state.fiber_bi_charge_balanced = False
    state.fiber_bi_phase1_amp_mA = 1.0
    state.fiber_bi_phase1_pw_us = 1000.0
    state.fiber_bi_gap_us = 0.0
    state.fiber_bi_phase2_amp_mA = 1.0
    state.fiber_bi_phase2_pw_us = 1000.0
    # Timing
    state.fiber_onset_ms = 1.0
    state.fiber_tstop_ms = 8.0
    # Sim status / progress / output mirrors. The actual sim
    # arrays live on `geom.fiber_sim_data` (heavy numpy state);
    # these state vars are just the strings + small numbers the
    # client needs for the UI labels + the Plotly figures.
    state.fiber_sim_status = "No simulation run yet."
    state.fiber_sim_log = ""
    state.fiber_sim_busy = False
    state.fiber_sim_failed = False
    state.fiber_sim_summary = ""
    # F3.2-M2.1d — set true the first time a single-fiber sim
    # succeeds; mirrored from pipeline/fiber_sim.py. Used by
    # the EXPORT navbar gate (any nerve sim done → enable).
    state.has_fiber_sim = False
    state.fiber_pulse_figure = {"data": [], "layout": {}}
    state.fiber_propagation_figure = {"data": [], "layout": {}}
    state.fiber_waterfall_figure = {"data": [], "layout": {}}
    state.fiber_stim_figure = {"data": [], "layout": {}}
