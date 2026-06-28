# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fiber-tab + Population-tab + branch-name watchers.

Includes the pulse-design watcher set + the fiber-model /
diameter / backend snap-and-sync watchers + the pop-view + pop-
row-visible + fiber-selection + branch-name watchers + the
active_analysis tab swap.

Pulse-params + pulse-preview helpers move here too because they
were closures in build_app — now they're free functions on
state. `fiber_pulse_params` is also wired into the pipeline
helpers SimpleNamespace from build_app so the pipeline drivers
(fiber_sim / pop_sim) keep using the same function.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from golgi.figures.fiber import (
    _build_fiber_propagation_figure,
    _build_fiber_pulse_figure,
    _build_fiber_waterfall_figure,
)


def _snap_to_nearest(value: float, choices: list) -> float:
    if not choices:
        return value
    return min(choices, key=lambda v: abs(v - value))


def fiber_pulse_params(state, *, effective_anod_pw_ms: Callable) -> dict:
    """Resolve the current pulse-design state vars into the
    common (cath_amp, cath_pw_ms, gap_ms, anod_amp, anod_pw_ms,
    anode_first, kind) tuple. Used by both the live preview AND
    the sim itself.

    `effective_anod_pw_ms` is _fiber_effective_anod_pw_ms from
    golgi.py (module-level)."""
    t0 = float(state.fiber_onset_ms)
    tstop = float(state.fiber_tstop_ms)
    is_bi = (str(state.fiber_pulse_type) == "biphasic")
    if not is_bi:
        mono_amp = float(state.fiber_mono_amp_mA)
        mono_pw_ms = float(state.fiber_mono_pw_us) * 1e-3
        if str(state.fiber_mono_polarity) == "cathodic":
            cath_amp, cath_pw_ms = mono_amp, mono_pw_ms
            anod_amp, anod_pw_ms = 0.0, 0.0
            anode_first = False
        else:
            cath_amp, cath_pw_ms = 0.0, 0.0
            anod_amp, anod_pw_ms = mono_amp, mono_pw_ms
            anode_first = True
        gap_ms = 0.0
        kind = ("monophasic anodic" if anode_first
                else "monophasic cathodic")
    else:
        gap_ms = float(state.fiber_bi_gap_us) * 1e-3
        ph1_amp = float(state.fiber_bi_phase1_amp_mA)
        ph1_pw_ms = float(state.fiber_bi_phase1_pw_us) * 1e-3
        ph2_amp = float(state.fiber_bi_phase2_amp_mA)
        ph2_pw_ms_user = (
            float(state.fiber_bi_phase2_pw_us) * 1e-3
        )
        ph2_pw_ms_eff = effective_anod_pw_ms(
            ph1_amp, ph1_pw_ms, ph2_amp, ph2_pw_ms_user,
            bool(state.fiber_bi_charge_balanced),
        )
        if str(state.fiber_bi_order) == "cathodic-first":
            cath_amp, cath_pw_ms = ph1_amp, ph1_pw_ms
            anod_amp, anod_pw_ms = ph2_amp, ph2_pw_ms_eff
            anode_first = False
        else:
            anod_amp, anod_pw_ms = ph1_amp, ph1_pw_ms
            cath_amp, cath_pw_ms = ph2_amp, ph2_pw_ms_eff
            anode_first = True
        kind = ("biphasic anode-first" if anode_first
                else "biphasic cathode-first")
    return {
        "t0": t0, "tstop": tstop,
        "cath_amp_mA": cath_amp, "cath_pw_ms": cath_pw_ms,
        "gap_ms": gap_ms,
        "anod_amp_mA": anod_amp, "anod_pw_ms": anod_pw_ms,
        "anode_first": anode_first, "kind": kind,
    }


def register(
    state,
    *,
    geom,
    fiber_diameter_config: dict,
    fiber_diameter_default: dict,
    max_fiber_branches: int,
    request_render: Callable,
    rebuild_scene_state: Callable,
    refresh_fiber_sel_items: Callable,
    branch_name: Callable,
    build_pulse_waveform: Callable,
    effective_anod_pw_ms: Callable,
) -> Callable:
    """Wire fiber-tab + pop-tab + branch-name watchers.

    Returns the `_fiber_pulse_params` closure (bound to this
    state + effective_anod_pw_ms) so build_app can wire it into
    the pipeline helpers SimpleNamespace."""

    def _fiber_pulse_params() -> dict:
        return fiber_pulse_params(
            state, effective_anod_pw_ms=effective_anod_pw_ms,
        )

    def _refresh_fiber_pulse_preview() -> None:
        """Rebuild the pulse-preview Plotly figure from the
        current pulse-design state vars."""
        try:
            p = _fiber_pulse_params()
            dt_ms = 0.01
            n_t = int(round(p["tstop"] / dt_ms)) + 1
            t_grid = np.arange(n_t, dtype=np.float64) * dt_ms
            wave = build_pulse_waveform(
                t_grid, p["t0"],
                p["cath_amp_mA"], p["cath_pw_ms"], p["gap_ms"],
                p["anod_amp_mA"], p["anod_pw_ms"],
                p["anode_first"],
            )
            state.fiber_pulse_figure = (
                _build_fiber_pulse_figure(
                    t_grid, wave,
                    title=f"Designed pulse  ·  {p['kind']}",
                )
            )
        except Exception as ex:
            print(f"[fiber] pulse preview failed: {ex}",
                  flush=True)

    # ---- Population result-picker ----
    # W1.7a: F1.1 curated-population-preset preview refresh.
    # Live-refresh the preview tile + citation/notes box as the
    # user scans presets in the dropdown. No mutation of
    # pop_branch_types — that only happens on the explicit Apply
    # button (do_pop_apply_preset).
    @state.change("pop_preset_choice")
    def _on_pop_preset_choice(pop_preset_choice, **_):
        from golgi.state_defaults import pop_presets as _pp
        name = str(pop_preset_choice or "")
        state.pop_preset_meta = _pp.preset_meta(name)
        state.pop_preset_preview_figure = (
            _pp.build_preview_kde_figure(name)
        )

    @state.change("pop_view_idx")
    def _on_pop_view_change(**_kwargs):
        if not geom.fiber_pop_sim_results:
            return
        try:
            vi = int(state.pop_view_idx)
        except (TypeError, ValueError):
            return
        sim_data = geom.fiber_pop_sim_results.get(vi)
        if sim_data is None:
            return
        state.pop_propagation_figure = (
            _build_fiber_propagation_figure(sim_data)
        )
        state.pop_waterfall_figure = (
            _build_fiber_waterfall_figure(sim_data)
        )

    # ---- Pulse-design watcher (13 keys) ----
    @state.change(
        "fiber_pulse_type",
        "fiber_mono_polarity", "fiber_mono_amp_mA",
        "fiber_mono_pw_us",
        "fiber_bi_order", "fiber_bi_charge_balanced",
        "fiber_bi_phase1_amp_mA", "fiber_bi_phase1_pw_us",
        "fiber_bi_gap_us",
        "fiber_bi_phase2_amp_mA", "fiber_bi_phase2_pw_us",
        "fiber_onset_ms", "fiber_tstop_ms",
    )
    def _on_fiber_pulse_change(**_kwargs):
        _refresh_fiber_pulse_preview()

    # ---- Backend snap ----
    @state.change("fiber_backend")
    def _on_fiber_backend_change(**_kwargs):
        if str(state.fiber_backend) == "axonml":
            state.fiber_model = "MRG_INTERPOLATION"

    # ---- Model + diameter sync ----
    @state.change("fiber_model")
    def _on_fiber_model_change(**_kwargs):
        cfg = fiber_diameter_config.get(
            str(state.fiber_model),
            fiber_diameter_default,
        )
        state.fiber_diameter_min = float(cfg["min"])
        state.fiber_diameter_max = float(cfg["max"])
        state.fiber_diameter_step = float(cfg["step"])
        ticks = list(cfg.get("ticks") or [])
        state.fiber_diameter_ticks = ticks
        # Snap or clamp current diameter into the new range.
        current = float(state.fiber_diameter_um)
        if ticks:
            current = _snap_to_nearest(current, ticks)
        else:
            current = max(float(cfg["min"]),
                          min(float(cfg["max"]), current))
        state.fiber_diameter_um = float(current)

    @state.change("fiber_diameter_um")
    def _on_fiber_diameter_change(**_kwargs):
        ticks = list(state.fiber_diameter_ticks or [])
        if not ticks:
            return
        try:
            current = float(state.fiber_diameter_um)
        except (TypeError, ValueError):
            return
        snapped = _snap_to_nearest(current, ticks)
        if abs(snapped - current) > 1e-3:
            state.fiber_diameter_um = float(snapped)

    # ---- Pop legend toggle ----
    @state.change("pop_row_visible")
    def _on_pop_row_visible_change(**_kwargs):
        rebuild_scene_state()
        request_render()

    # ---- Fiber selection ----
    @state.change("fiber_sel_idx", "fiber_sel_indices")
    def _on_fiber_sel_change(**_kwargs):
        n_paths = (
            len(geom.fiber_paths_raw)
            if geom.fiber_paths_raw is not None
            else 0
        )
        cleaned: list = []
        for v in (state.fiber_sel_indices or []):
            try:
                vi = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= vi < n_paths and vi not in cleaned:
                cleaned.append(vi)
        if cleaned != list(state.fiber_sel_indices or []):
            state.fiber_sel_indices = cleaned
            # Early-return to avoid double-render.
            return
        request_render()
        # Swap output-tile figures to the currently-viewed
        # fiber's stored sim_data.
        if not geom.fiber_sim_results:
            return
        view_i = int(state.fiber_sel_idx)
        sim_data = geom.fiber_sim_results.get(view_i)
        if sim_data is None:
            return
        geom.fiber_sim_data = sim_data
        state.fiber_propagation_figure = (
            _build_fiber_propagation_figure(sim_data)
        )
        state.fiber_waterfall_figure = (
            _build_fiber_waterfall_figure(sim_data)
        )

    # ---- Analysis-tab swap ----
    @state.change("active_analysis")
    def _on_active_analysis_change(**_kwargs):
        if (str(state.active_analysis) == "solve"
                and state.has_fem
                and not bool(state.show_field_lines)):
            state.show_field_lines = True
        request_render()

    # ---- Branch-name renames flow into every label location ----
    @state.change(*[
        f"fiber_branch_name_{_i}"
        for _i in range(max_fiber_branches)
    ])
    def _on_fiber_branch_names_change(**_kwargs):
        if not state.has_fibers:
            return
        refresh_fiber_sel_items()
        # Patch fiber_branch_summary labels in place.
        summary = list(state.fiber_branch_summary or [])
        if summary:
            new_summary = []
            for entry in summary:
                row = dict(entry)
                idx = int(row.get("idx", -1))
                if idx >= 0:
                    row["label"] = branch_name(idx)
                new_summary.append(row)
            state.fiber_branch_summary = new_summary
        # Patch pop_branches_meta labels in place.
        meta = list(state.pop_branches_meta or [])
        if meta:
            new_meta = []
            for entry in meta:
                row = dict(entry)
                row["label"] = branch_name(int(row.get("idx", 0)))
                new_meta.append(row)
            state.pop_branches_meta = new_meta

    return _fiber_pulse_params
