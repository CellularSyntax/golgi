# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cole-Cole σ(f) plot — interactive Plotly version (F2.3.a follow-up).

Replaces the matplotlib PNG renderer with a Plotly dict so the
Conductivities dialog renders an interactive log-log curve + the
new per-panel export button picks it up via the registry.

Public API:
    _build_cole_cole_figure(...) -> {"data": [...], "layout": {...}}

The legacy `_render_cole_cole_plot` helper (matplotlib → data URI)
is removed — no callers remain. Watcher writes the dict to
`state.cc_plot_figure`; the dialog renders via twp.Figure.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .export import apply_preset_to_plotly_fig

if TYPE_CHECKING:                                  # pragma: no cover
    from .export import FigureExportPreset


def _maybe_apply_preset(
    fig: dict, preset: "FigureExportPreset | None",
) -> dict:
    if preset is not None:
        apply_preset_to_plotly_fig(fig, preset)
    return fig


def _build_cole_cole_figure(
    eps_inf: float,
    sigma_ionic: float,
    dispersions: list[tuple[float, float, float]],
    f_marker_hz: float,
    *,
    sigma_fn: Callable[
        [float, float, float, list[tuple[float, float, float]]],
        float,
    ],
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Build a log-log σ(f) Plotly figure for the supplied Cole-Cole
    parameters. Adds a vertical dashed marker at f_marker_hz + a
    dot+label at the corresponding σ value (matches the legacy
    matplotlib layout).

    `sigma_fn` is injected (kwarg-only) so this module stays
    decoupled from the cole_cole evaluator in golgi.py — callers
    pass `sigma_fn=cole_cole_sigma`.

    Returns a `{data, layout}` dict ready to push into
    `state.cc_plot_figure`."""
    # 1 Hz to 1 GHz covers Gabriel's full 4-term span. Logspace at
    # ~240 pts keeps the curve smooth without bloating the JSON.
    freqs = np.logspace(0, 9, 240)
    sigmas = np.empty_like(freqs)
    for i, f in enumerate(freqs):
        sigmas[i] = sigma_fn(
            float(f), eps_inf, sigma_ionic, dispersions,
        )

    traces: list[dict] = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": freqs.tolist(),
            "y": sigmas.tolist(),
            "line": {"color": "#e24b4a", "width": 2.0},
            "name": "σ(f)",
            "hovertemplate": (
                "f = %{x:.3g} Hz<br>"
                "σ = %{y:.4g} S/m<extra></extra>"
            ),
        },
    ]

    shapes: list[dict] = []
    annotations: list[dict] = []
    # Marker line + dot + label at f_marker — only when it falls in
    # the visible range AND σ is positive (log axis can't render 0).
    try:
        s_marker = sigma_fn(
            float(f_marker_hz), eps_inf, sigma_ionic, dispersions,
        )
    except Exception:                                # noqa: BLE001
        s_marker = 0.0
    if f_marker_hz > 0 and s_marker > 0:
        shapes.append({
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": float(f_marker_hz),
            "x1": float(f_marker_hz),
            "y0": 0.0, "y1": 1.0,
            "line": {
                "color": "#1f2024",
                "width": 1.0,
                "dash": "dash",
            },
            "opacity": 0.55,
        })
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [float(f_marker_hz)],
            "y": [float(s_marker)],
            "marker": {
                "size": 9,
                "color": "#1f2024",
                "line": {"color": "white", "width": 1.6},
            },
            "name": "f",
            "showlegend": False,
            "hovertemplate": (
                f"f = {f_marker_hz:.3g} Hz<br>"
                f"σ = {s_marker:.4g} S/m"
                "<extra></extra>"
            ),
        })
        annotations.append({
            "x": float(f_marker_hz),
            "y": float(s_marker),
            "xref": "x", "yref": "y",
            "text": f"{s_marker:.4g} S/m",
            "showarrow": False,
            "xshift": 10, "yshift": -10,
            "font": {"size": 11, "color": "#1f2024"},
        })

    fig: dict = {
        "data": traces,
        "layout": {
            "xaxis": {
                "type": "log",
                "title": {
                    "text": "frequency [Hz]",
                    "font": {"size": 11},
                },
                "tickfont": {"size": 10},
                "range": [0.0, 9.0],   # log10(1) to log10(1e9)
                "showgrid": True,
                "gridcolor": "#e6e6e8",
            },
            "yaxis": {
                "type": "log",
                "title": {
                    "text": "σ [S/m]",
                    "font": {"size": 11},
                },
                "tickfont": {"size": 10},
                "showgrid": True,
                "gridcolor": "#e6e6e8",
            },
            "shapes": shapes,
            "annotations": annotations,
            "margin": {"l": 60, "r": 20, "t": 16, "b": 50},
            "showlegend": False,
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "hovermode": "closest",
        },
    }
    return _maybe_apply_preset(fig, preset)
