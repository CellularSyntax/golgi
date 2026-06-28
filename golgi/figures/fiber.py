# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Single-fiber simulation figure builders (pulse, propagation, waterfall)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .util import (
    _FIBER_AXIS_TICK_FONT,
    _FIBER_AXIS_TITLE_FONT,
    _FLARE_COLORSCALE,
    _plotly_placeholder,
)

if TYPE_CHECKING:                                  # pragma: no cover
    from .export import FigureExportPreset


def _maybe_apply_preset(
    fig: dict, preset: "FigureExportPreset | None",
) -> dict:
    """Apply an export preset to a Plotly figure dict if one was
    passed. Centralised so the per-builder boilerplate stays a single
    line."""
    if preset is None:
        return fig
    from .export import apply_preset_to_plotly_fig
    return apply_preset_to_plotly_fig(fig, preset)


def _build_fiber_pulse_figure(
    t_grid_ms, wave_mA,
    title: str = "Designed stim pulse",
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Plotly figure for the designed stim waveform. Step (hv)
    line plot — exactly the preview the user designs against in
    the Fiber tab + the post-sim 'applied stim' panel."""
    t = np.asarray(t_grid_ms, dtype=np.float64).tolist()
    w = np.asarray(wave_mA, dtype=np.float64).tolist()
    return _maybe_apply_preset({
        "data": [
            {
                "type": "scatter",
                "x": t,
                "y": w,
                "mode": "lines",
                "line": {"color": "#0d3b66", "width": 1.8,
                          "shape": "hv"},
                "name": "I_stim",
                "hovertemplate": (
                    "t = %{x:.3f} ms<br>"
                    "I = %{y:.3f} mA<extra></extra>"
                ),
            },
        ],
        "layout": {
            "title": {
                "text": title,
                "font": {"size": 12},
                "x": 0.5, "xanchor": "center",
            },
            "xaxis": {
                "title": {
                    "text": "time  (ms)",
                    "font": _FIBER_AXIS_TITLE_FONT,
                },
                "tickfont": _FIBER_AXIS_TICK_FONT,
                "showgrid": True, "gridcolor": "#e5e5ea",
            },
            "yaxis": {
                "title": {
                    "text": "stim current  (mA)",
                    "font": _FIBER_AXIS_TITLE_FONT,
                },
                "tickfont": _FIBER_AXIS_TICK_FONT,
                "showgrid": True, "gridcolor": "#e5e5ea",
                "zeroline": True,
                "zerolinecolor": "#bbbbc4",
            },
            "margin": {"l": 60, "r": 20, "t": 36, "b": 48},
            "paper_bgcolor": "white",
            "plot_bgcolor": "rgba(248,249,251,0.6)",
        },
    }, preset)


def _build_fiber_propagation_figure(
    sim_data: dict | None,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Plotly heatmap of Vm(t, node) + per-node spike markers.
    Stim node (where the static V_e is most negative) gets a tiny
    arrow annotation."""
    if not sim_data:
        return _plotly_placeholder(
            "Run a single-fiber simulation to see the V<sub>m</sub> "
            "propagation heatmap."
        )
    vm = np.asarray(sim_data["vm"], dtype=np.float64)
    t = np.asarray(sim_data["t"], dtype=np.float64).tolist()
    node_z_um = np.asarray(sim_data["node_z_um"], dtype=np.float64)
    node_z_mm = (node_z_um * 1e-3).tolist()
    spike_t = np.asarray(sim_data["spike_t"], dtype=np.float64)
    Ve_at_nodes = np.asarray(
        sim_data["Ve_at_nodes_mV"], dtype=np.float64,
    )
    stim_node = int(np.argmin(Ve_at_nodes)) if Ve_at_nodes.size else 0
    # Heatmap z is (n_nodes, n_t). Plotly expects rows = y values
    # (node index), cols = x values (time). The flare colormap is
    # sequential (light → dark red/purple); we set zmin/zmax to
    # the full Vm range so the AP wavefront (V_m peak ≈ +40 mV)
    # paints as dark purple against the lighter resting baseline
    # (V_m ≈ -80 mV).
    traces = [{
        "type": "heatmap",
        "x": t,
        "y": node_z_mm,
        "z": vm.tolist(),
        "colorscale": _FLARE_COLORSCALE,
        "zmin": -90.0, "zmax": 50.0,
        "colorbar": {
            "title": {
                "text": "V<sub>m</sub>  (mV)",
                "font": _FIBER_AXIS_TITLE_FONT,
                "side": "right",
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "thickness": 12, "len": 0.78,
        },
        "zsmooth": "best",
        "hovertemplate": (
            "t = %{x:.2f} ms<br>"
            "z = %{y:.2f} mm<br>"
            "V<sub>m</sub> = %{z:.1f} mV<extra></extra>"
        ),
        "name": "Vm",
        "showlegend": False,
    }]
    # Spike markers — one dot per node that fired a real AP.
    # White dot, black halo so it reads on dark flare background.
    sx, sy = [], []
    for i_node, ts in enumerate(spike_t):
        if not np.isfinite(ts):
            continue
        sx.append(float(ts))
        sy.append(float(node_z_um[i_node] * 1e-3))
    if sx:
        traces.append({
            "type": "scatter",
            "x": sx, "y": sy,
            "mode": "markers",
            "marker": {
                "color": "#ffffff", "size": 5,
                "line": {"color": "#000000", "width": 0.7},
            },
            "name": "AP",
            "hovertemplate": (
                "spike t = %{x:.2f} ms<br>"
                "node z = %{y:.2f} mm<extra></extra>"
            ),
            "showlegend": False,
        })
    layout = {
        "title": {
            "text": (
                f"V<sub>m</sub> propagation  ·  "
                f"{sim_data.get('n_real', 0)}/"
                f"{sim_data.get('n_nodes', 0)} nodes fired"
            ),
            "font": {"size": 12},
            "x": 0.5, "xanchor": "center",
        },
        "xaxis": {
            "title": {
                "text": "time  (ms)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "yaxis": {
            "title": {
                "text": "node position along fiber  (mm)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        # Stim-node arrow + label drawn in BLACK so it remains
        # readable both on the flare-dark side of the heatmap
        # AND on the flare-light side (red was invisible on the
        # equally-red flare).
        "annotations": [{
            "x": float(sim_data.get("stim_onset_ms", 0.0)),
            "y": float(node_z_um[stim_node] * 1e-3),
            "ax": -40, "ay": 0,
            "text": "stim node",
            "showarrow": True,
            "arrowhead": 2,
            "arrowsize": 1.0,
            "arrowwidth": 1.6,
            "arrowcolor": "#000000",
            "font": {"size": 11, "color": "#000000",
                     "family": "system-ui, sans-serif"},
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#000000",
            "borderwidth": 0.8,
            "borderpad": 3,
        }],
        "margin": {"l": 70, "r": 20, "t": 36, "b": 48},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
    }
    return _maybe_apply_preset({"data": traces, "layout": layout}, preset)


def _build_fiber_waterfall_figure(
    sim_data: dict | None,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Waterfall of Vm(t) at ~15 evenly-spaced nodes. Each trace
    is offset vertically so adjacent nodes don't overlap."""
    if not sim_data:
        return _plotly_placeholder(
            "Run a single-fiber simulation to see the V<sub>m</sub> "
            "waterfall traces."
        )
    vm = np.asarray(sim_data["vm"], dtype=np.float64)
    t = np.asarray(sim_data["t"], dtype=np.float64)
    node_z_mm = (np.asarray(sim_data["node_z_um"]) * 1e-3)
    n_nodes = int(vm.shape[0])
    n_traces = min(15, n_nodes)
    idxs = np.linspace(0, n_nodes - 1, n_traces).astype(int)
    # Offset = 50 mV per trace (AP peak is ~+40 mV, baseline ~-80
    # mV, so 50 mV gap keeps adjacent traces clearly separated).
    offset_per = 50.0
    traces: list = []
    yticks: list = []
    ytext: list = []
    for k, i_node in enumerate(idxs):
        y_offset = k * offset_per
        traces.append({
            "type": "scatter",
            "x": t.tolist(),
            "y": (vm[i_node] + y_offset).tolist(),
            "mode": "lines",
            "line": {
                "color": "#0d3b66",
                "width": 1.0,
            },
            "name": f"node {int(i_node)}",
            "hovertemplate": (
                f"node {int(i_node)} @ z = "
                f"{float(node_z_mm[i_node]):.2f} mm<br>"
                f"t = %{{x:.2f}} ms<br>"
                f"V<sub>m</sub> = %{{y:.1f}} mV<extra></extra>"
            ),
            "showlegend": False,
        })
        yticks.append(y_offset)
        ytext.append(f"{float(node_z_mm[i_node]):.1f}")
    layout = {
        "title": {
            "text": "V<sub>m</sub> waterfall (sampled nodes)",
            "font": {"size": 12},
            "x": 0.5, "xanchor": "center",
        },
        "xaxis": {
            "title": {
                "text": "time  (ms)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "yaxis": {
            "title": {
                "text": "node position along fiber  (mm)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "tickvals": yticks,
            "ticktext": ytext,
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "margin": {"l": 70, "r": 20, "t": 36, "b": 48},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
    }
    return _maybe_apply_preset({"data": traces, "layout": layout}, preset)
