# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Sweep-tab state defaults (F2.1.c).

The Sweep sub-tab in the Analysis section. Two modes, gated on
state.sweep_mode:
  - "recruitment": sweep amplitudes; populate
      state.sweep_recruitment_figure + state.sweep_heatmap_figure.
  - "threshold": per-fiber bisection; populate
      state.sweep_threshold_figure (+ also the heatmap if amps are
      provided as a separate "preview" axis).

Three figure state vars hold Plotly dicts directly (same pattern
as state.pop_kde_figure etc.). Empty-figure placeholders surface
the "Run a sweep to see…" message via figures/util.py.
"""
from __future__ import annotations


def register(state) -> None:
    # ---- Mode ----
    # "recruitment" | "threshold". Drives which one-click button
    # the UI shows as primary + which figures populate.
    state.sweep_mode = "recruitment"

    # ---- Model source ----
    # "population": per-fiber model + backend from the Population
    #   tab's assignment (geom.fiber_pop_types[sel] +
    #   state.pop_row_meta[row]["backend"]). Falls back to Single-
    #   fiber tab values per-field when a fiber has no population
    #   row assignment.
    # "single_fiber": one model + one backend for every fiber,
    #   straight from state.fiber_model + state.fiber_backend.
    # Default = "population" because the Population tab's per-row
    # model picker is otherwise dead UI for the sweep.
    state.sweep_model_source = "population"

    # ---- Recruitment-mode axis ----
    # Closed range [start, stop] with n_points samples, lin/log
    # spacing. Defaults match the canonical VNS recruitment window.
    state.sweep_amp_min_mA = 0.05
    state.sweep_amp_max_mA = 2.0
    state.sweep_amp_n_points = 25
    state.sweep_amp_spacing = "log"   # "lin" | "log"

    # ---- Threshold-mode bisect bounds ----
    state.sweep_bisect_lo_mA = 0.01
    state.sweep_bisect_hi_mA = 5.0
    state.sweep_bisect_tol_uA = 10.0

    # ---- Fiber filters (None = no filter) ----
    # Vuetify VSelect treats numbers as integers OK; for branch
    # filter we use -1 as the "All branches" sentinel so a single
    # int field can carry both states cleanly.
    state.sweep_branch_filter = -1     # -1 = all branches
    state.sweep_type_filter = ""       # "" = all types

    # ---- UI controls ----
    state.sweep_show_advanced = False
    state.sweep_busy = False
    state.sweep_failed = False
    state.sweep_status = "No sweep run yet."
    state.sweep_progress = ""          # short progress line for the busy
                                        # lightbox / inline status

    # ---- Result state ----
    # Set to True after a successful sweep so figures + CSV
    # download buttons unhide. Stays True on project reopen if
    # F2.1.d restores a cached sweep.
    state.sweep_has_result = False
    state.sweep_result_summary = ""    # short text under the figures
                                        # describing what was swept

    # Three Plotly figure dicts — driven by the figure builders in
    # golgi/figures/recruitment.py. Empty placeholders by default
    # so the trame widget can render them without conditional
    # logic on the caller side.
    state.sweep_recruitment_figure = {"data": [], "layout": {}}
    state.sweep_threshold_figure = {"data": [], "layout": {}}
    state.sweep_heatmap_figure = {"data": [], "layout": {}}

    # ---- F2.1.d: cache + browser-download payloads ----
    # The sha + the on-disk paths exist for reproducibility (the
    # disk cache is still written for project-reopen restore +
    # F2.2-style study-bundle export). Browser downloads use the
    # eagerly-built `data:...;base64,…` URIs below + a friendly
    # filename. Empty until the first sweep completes (or until
    # auto-restore reads them from the .npz cache).
    state.sweep_cache_sha = ""

    # Per-figure CSV download payloads. Each pair is a
    # base64-encoded data URI + the suggested filename. The
    # Download buttons in the UI use an inline JS click handler
    # that creates a hidden <a download="..." href="...">, clicks
    # it, removes it — pure client-side, no second WS round-trip.
    state.sweep_recruitment_csv_data_uri = ""
    state.sweep_recruitment_csv_filename = ""
    state.sweep_threshold_csv_data_uri = ""
    state.sweep_threshold_csv_filename = ""
    state.sweep_heatmap_csv_data_uri = ""
    state.sweep_heatmap_csv_filename = ""

    # Full SweepResult binary cache (.npz) — same eager-push
    # download pattern as the CSVs. The bytes are base64 in the
    # data URI; for a typical population sweep this is tens of KB.
    state.sweep_npz_data_uri = ""
    state.sweep_npz_filename = ""
