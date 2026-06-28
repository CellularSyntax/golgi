# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""FEM-panel watchers — slice slider + AF param sliders.

Both refresh the Plotly figures without re-running the solve.
"""
from __future__ import annotations

from golgi.figures.fem import _build_fem_af_figure


def register(
    state,
    *,
    geom,
    elec_sync_guard: dict,
    refresh_fem_plots,
) -> None:
    """Wire fem_slice_z_idx/fem_field + fem_fiber_sel/fem_sg_window
    watchers."""

    @state.change("fem_slice_z_idx", "fem_field")
    def _on_fem_slice_change(**_kwargs):
        """Re-render only the slice heatmap when the user moves
        the z slider or flips the Vₑ ↔ |E| field toggle.
        Doesn't re-run the FEM solve."""
        if elec_sync_guard["loading"]:
            return
        if not state.has_fem or geom.fem_slice is None:
            return
        try:
            refresh_fem_plots(slice_only=True)
        except Exception:
            pass

    @state.change("fem_fiber_sel", "fem_sg_window")
    def _on_fem_af_param_change(**_kwargs):
        """Re-render the §10 AF plot when the user moves the
        select-fiber slider or the SG-window slider. Cheap —
        just resamples and re-fits Savgol on the cached per-
        fiber Vₑ; doesn't re-run the FEM solve."""
        if elec_sync_guard["loading"]:
            return
        if (not state.has_fem
                or geom.fiber_paths_Ve is None):
            return
        _af_paths = (geom.fiber_paths_for_Ve
                     if geom.fiber_paths_for_Ve is not None
                     else geom.fiber_paths_raw)
        if _af_paths is None:
            return
        try:
            state.fem_af_figure = _build_fem_af_figure(
                paths_Ve=geom.fiber_paths_Ve,
                paths_raw=_af_paths,
                branch_idx=geom.fiber_branch_idx,
                sel_fiber=int(state.fem_fiber_sel),
                sg_window=int(state.fem_sg_window),
            )
        except Exception:
            pass
