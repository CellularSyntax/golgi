# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Population tab state defaults (per-branch fiber-type mixture)."""
from __future__ import annotations

from . import pop_presets as _pop_presets


def register(state) -> None:
    # Per-branch dynamic list of fiber-type rows. Each row is a
    # dict {id, model, mean_um, std_um, frac}; keys of
    # `pop_branch_types` are STRINGS (str(branch_idx)) because
    # Vue templates serialise int keys to strings. Empty list
    # for a branch = "no types assigned" (fibers in that branch
    # stay untyped on generate).
    state.pop_branch_types = {}
    # UI metadata about each detected branch (refreshed after
    # every fiber-generation step). Each entry:
    #   {"idx": int, "label": "Branch X", "n_fibers": int,
    #    "color": "#hex"}.
    state.pop_branches_meta = []
    state.pop_seed = 42
    state.pop_status = "No population generated yet."
    state.pop_busy = False
    state.pop_failed = False
    # True once a generate has succeeded — gates the Run button
    # and tells the viewport to recolour by type.
    state.pop_generated = False
    # Map of model-name → hex colour, built on generate (legacy
    # per-model colour scheme). The canonical lookup is now
    # `pop_row_meta` / `pop_row_colors` (keyed by row.id) so two
    # rows with the same model still get distinct visual
    # identities.
    state.pop_type_colors = {}
    # row-id → hex colour. Drives every per-row visual: chip
    # dots, KDE traces, 3-D actor tints, per-row subplot outlines.
    state.pop_row_colors = {}
    # row-id → {"name", "model", "color", "branch", "mean_um",
    # "std_um"}. Built on generate so the KDE + result picker +
    # any future per-row UI can look everything up without
    # re-walking pop_branch_types.
    state.pop_row_meta = {}
    # Per-row visibility, keyed by row.id → bool. Drives the
    # Population-tab legend toggles AND the per-row colouring in
    # `_set_fiber_groups` (population branch): when a row's
    # entry is False, the row's fibers render grey instead of
    # their tab10 colour.
    state.pop_row_visible = {}
    # Plotly KDE figure (per-branch subplots + overall).
    state.pop_kde_figure = {"data": [], "layout": {}}
    # Population-sim outputs — populated by do_pop_run_sim.
    state.pop_sim_done = False
    # Result-picker meta for the cross-section + heatmap tiles.
    state.pop_sim_results_meta = []
    # `pop_view_idx` = fiber currently displayed in the heatmap
    # + waterfall tiles.
    state.pop_view_idx = 0
    # Sorted list of fiber indices whose sim fired at least one
    # AP — drives the activated-vs-quiescent split in the cross-
    # section overview.
    state.pop_activated_set = []
    state.pop_xsec_figure = {"data": [], "layout": {}}
    # New cross-section at cuff centre — gated on pop_generated
    # (NOT pop_sim_done) so the user sees the population layout
    # before running the sim.
    state.pop_xsec_cuff_figure = {"data": [], "layout": {}}
    state.pop_propagation_figure = {"data": [], "layout": {}}
    state.pop_waterfall_figure = {"data": [], "layout": {}}
    # ---- F1.1: curated fiber-population presets ----
    # `pop_preset_choice` is the registry key of the currently-
    # selected preset (or "" for "none"). Mutating it does NOT
    # rewrite `pop_branch_types` — that only happens on
    # do_pop_apply_preset. The watcher in app.py rebuilds the
    # preview figure live as the dropdown changes.
    state.pop_preset_choice = ""
    state.pop_preset_items = _pop_presets.preset_dropdown_items()
    state.pop_preset_meta = _pop_presets.preset_meta("")
    state.pop_preset_preview_figure = (
        _pop_presets.build_preview_kde_figure("")
    )
