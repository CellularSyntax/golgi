# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""R1.4 — Plotly figure builders for cNAP recording traces.

Two figures:
  * Single-fiber: one line per active montage (or just the
    currently-selected montage).
  * Population: total trace + per-fiber-type decomposition.

All figures use the same time base as the underlying fiber sim
(typically dt=0.005 ms). Trace amplitudes are reported in µV
(phi is stored as Volts in geom.cnap_*, rescaled here for the
plot).

Convention: returns a plotly figure as a {data, layout} dict, the
shape trame's `twp.Figure` consumes. Empty/placeholder cases
return an empty figure dict, never None — keeps the panel's
v_show logic simple.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Stable colour ordering for fiber-type decomposition. Keys are
# the type labels that show up on `state.pop_branch_types`;
# unknown labels fall through to a fallback palette.
_FIBER_TYPE_COLORS: dict[str, str] = {
    # MRG / myelinated.
    "MRG_INTERPOLATION": "#1f77b4",
    "MRG": "#1f77b4",
    "A-alpha": "#1f77b4",
    "A-beta": "#ff7f0e",
    "A-delta": "#2ca02c",
    # Unmyelinated.
    "Tigerholm": "#d62728",
    "Sundt": "#d62728",
    "C": "#d62728",
    # Other myelinated / generic.
    "B": "#9467bd",
}
_FALLBACK_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
)


def _empty_fig(message: str = "") -> dict:
    """Empty Plotly figure with a centred annotation."""
    layout: dict = {
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "margin": {"l": 30, "r": 20, "t": 30, "b": 30},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fafafa",
    }
    if message:
        layout["annotations"] = [{
            "text": message,
            "showarrow": False,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "font": {"size": 13, "color": "#777"},
        }]
    return {"data": [], "layout": layout}


def _color_for(label: str, idx: int) -> str:
    if label in _FIBER_TYPE_COLORS:
        return _FIBER_TYPE_COLORS[label]
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def _common_layout(
    *, title: str, x_range: Optional[tuple] = None,
) -> dict:
    layout: dict = {
        "title": {
            "text": title,
            "font": {"size": 13},
            "x": 0.02, "xanchor": "left",
        },
        "xaxis": {
            "title": "Time (ms)",
            "showgrid": True, "gridcolor": "#e6e6e6",
            "zeroline": False,
        },
        "yaxis": {
            "title": "φ (µV)",
            "showgrid": True, "gridcolor": "#e6e6e6",
            "zeroline": True, "zerolinecolor": "#bbb",
        },
        "margin": {"l": 60, "r": 20, "t": 40, "b": 50},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#ffffff",
        "showlegend": True,
        "legend": {
            "orientation": "v",
            "x": 1.0, "xanchor": "right",
            "y": 1.0, "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.7)",
        },
        "hovermode": "x unified",
    }
    if x_range is not None:
        layout["xaxis"]["range"] = list(x_range)
    return layout


# ---------------------------------------------------------------
# Single-fiber cNAP figure
# ---------------------------------------------------------------


def build_fiber_cnap_figure(
    *,
    cnap_by_montage: dict,
    montage_meta: list,
    active_mid: str,
    fiber_label: str = "",
) -> dict:
    """One line: the cNAP this fiber alone would produce at the
    *active* montage. Switching the dropdown to a different
    montage just swaps which entry in `cnap_by_montage` is
    rendered — no re-sim.

    Args:
      cnap_by_montage: {mid: {"t_ms": (n_t,), "phi_V": (n_t,)}}
        from compute_cnap_single, one entry per montage that the
        active config has on it.
      montage_meta: list of montage dicts (from
        state.recording_montages) so we can render the label +
        colour matching the cuff drawer.
      active_mid: which mid to render. If empty or missing, picks
        the first one in cnap_by_montage.
      fiber_label: e.g. "Branch 0 · Fiber 142 · 8.7 µm Aα".

    Returns: plotly figure dict.
    """
    if not cnap_by_montage:
        return _empty_fig(
            "No recording montages — add one in the cuff "
            "drawer and re-run the FEM solve."
        )
    mid = str(active_mid or "")
    if mid not in cnap_by_montage:
        mid = next(iter(cnap_by_montage))
    entry = cnap_by_montage[mid]
    t_ms = np.asarray(entry.get("t_ms", []), dtype=np.float64)
    phi_uV = np.asarray(entry.get("phi_V", []), dtype=np.float64) * 1.0e6
    # Defensive NaN clean — plotly renders NaN as gaps, which
    # can make a real trace look empty if the upstream
    # interpolation left holes.
    phi_uV = np.nan_to_num(phi_uV, nan=0.0)
    label = mid
    color = "#22c55e"
    for m in montage_meta or []:
        if str(m.get("mid", "")) == mid:
            label = str(m.get("label", mid))
            color = str(m.get("color") or color)
            break

    peak_uV = float(np.max(np.abs(phi_uV))) if phi_uV.size else 0.0
    title = f"Single-fiber · {label} · peak {peak_uV:.2f} µV"
    if fiber_label:
        title = f"{title} · {fiber_label}"
    trace = {
        "type": "scatter",
        "mode": "lines",
        "x": t_ms.tolist(),
        "y": phi_uV.tolist(),
        "line": {"color": color, "width": 2},
        "name": label,
        "hovertemplate": "%{x:.3f} ms<br>%{y:.3f} µV<extra></extra>",
    }
    return {
        "data": [trace],
        "layout": _common_layout(title=title),
    }


# ---------------------------------------------------------------
# Population cNAP figure
# ---------------------------------------------------------------


def build_pop_cnap_figure(
    *,
    cnap_by_montage: dict,
    montage_meta: list,
    active_mid: str,
    decompose_by_type: bool = True,
    peak_latency_lines: bool = True,
) -> dict:
    """Total population cNAP at the active montage, optionally
    stacked with per-fiber-type contributions.

    `cnap_by_montage[mid]` shape:
        {
          "t_ms": (n_t,),
          "phi_total_V": (n_t,),
          "phi_by_type": {label: (n_t,) V}   (optional),
          "peak_latencies_ms": {label: float}  (optional),
        }
    """
    if not cnap_by_montage:
        return _empty_fig(
            "No recording montages — add one in the cuff "
            "drawer and re-run the FEM solve, then re-run "
            "the population sim."
        )
    mid = str(active_mid or "")
    if mid not in cnap_by_montage:
        mid = next(iter(cnap_by_montage))
    entry = cnap_by_montage[mid]
    t_ms = np.asarray(entry.get("t_ms", []), dtype=np.float64)
    phi_total_uV = (
        np.asarray(entry.get("phi_total_V", []), dtype=np.float64)
        * 1.0e6
    )
    label = mid
    color = "#222"
    for m in montage_meta or []:
        if str(m.get("mid", "")) == mid:
            label = str(m.get("label", mid))
            color = str(m.get("color") or color)
            break

    data: list[dict] = []

    # Per-type stack (thin lines beneath the total).
    if decompose_by_type:
        by_type = dict(entry.get("phi_by_type", {}) or {})
        for i, (type_label, phi_arr) in enumerate(
            sorted(by_type.items()),
        ):
            phi_uV = np.asarray(phi_arr, dtype=np.float64) * 1.0e6
            if phi_uV.size != t_ms.size:
                continue
            type_color = _color_for(type_label, i)
            data.append({
                "type": "scatter",
                "mode": "lines",
                "x": t_ms.tolist(),
                "y": phi_uV.tolist(),
                "line": {"color": type_color, "width": 1.4},
                "name": type_label,
                "opacity": 0.85,
                "hovertemplate": (
                    f"{type_label}<br>%{{x:.3f}} ms<br>"
                    "%{y:.3f} µV<extra></extra>"
                ),
            })

    # Total trace on top (thick).
    data.append({
        "type": "scatter",
        "mode": "lines",
        "x": t_ms.tolist(),
        "y": phi_total_uV.tolist(),
        "line": {"color": color, "width": 2.5},
        "name": f"{label} (total)",
        "hovertemplate": (
            "Total<br>%{x:.3f} ms<br>%{y:.3f} µV<extra></extra>"
        ),
    })

    layout = _common_layout(title=f"Population cNAP · {label}")

    # Peak-latency annotation lines.
    if peak_latency_lines:
        peaks = dict(entry.get("peak_latencies_ms", {}) or {})
        shapes = []
        annots = []
        y_min = float(min(0.0, np.nanmin(phi_total_uV)
                          if phi_total_uV.size else 0.0))
        y_max = float(max(0.0, np.nanmax(phi_total_uV)
                          if phi_total_uV.size else 1.0))
        y_span = max(y_max - y_min, 1.0)
        for i, (tlabel, t_peak) in enumerate(sorted(peaks.items())):
            if not np.isfinite(t_peak):
                continue
            type_color = _color_for(tlabel, i)
            shapes.append({
                "type": "line",
                "x0": float(t_peak), "x1": float(t_peak),
                "xref": "x",
                "y0": 0.0, "y1": 1.0, "yref": "paper",
                "line": {
                    "color": type_color,
                    "width": 1, "dash": "dot",
                },
            })
            annots.append({
                "x": float(t_peak),
                "y": y_max + 0.06 * y_span,
                "xref": "x", "yref": "y",
                "text": f"{tlabel}<br>{t_peak:.2f} ms",
                "showarrow": False,
                "font": {"size": 10, "color": type_color},
                "align": "center",
            })
        if shapes:
            layout["shapes"] = shapes
        if annots:
            layout["annotations"] = annots

    return {"data": data, "layout": layout}
