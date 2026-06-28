# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Sweep-result figure builders + CSV exporters (F2.1.b).

Three figures consume a `golgi.jobs.schemas.SweepResult`:

1. `build_recruitment_curve_figure(result, preset=None)`
   % activated vs I_stim, ribbon mean + 95 % CI per branch, with
   per-fiber-type dotted-line overlay (when the SweepResult carries
   fiber_type_labels). Requires `result.activated` (recruitment mode).

2. `build_threshold_scatter_figure(result, preset=None)`
   threshold (µA) vs fiber diameter (µm), point colour by branch,
   marker symbol by fiber type. NaN thresholds (no activation in
   range) plot as upward-triangle markers at the y-axis top.
   Requires `result.thresholds_uA` (threshold mode).

3. `build_activation_heatmap_figure(result, preset=None)`
   Per-fiber × per-amplitude heatmap; cell value = activated.
   Fibers sorted by branch + diameter so type clusters are visible.
   Requires `result.activated` (recruitment mode).

Each builder returns the standard `{data, layout}` Plotly dict +
respects the F1.2 publication-preset kwarg.

CSV exporters (one per figure) emit the underlying numeric series
in a reviewer-friendly tabular format; F2.1.c's per-figure
"Download data" buttons call them.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

from golgi.jobs.schemas import SweepResult

from .util import (
    _FIBER_AXIS_TICK_FONT,
    _FIBER_AXIS_TITLE_FONT,
    _hex_to_rgba,
    _plotly_placeholder,
)

if TYPE_CHECKING:                                  # pragma: no cover
    from .export import FigureExportPreset


# Branch palette — same hues used by the FEM ribbon plot
# (figures/fem.py:_RIBBON_BRANCH_COLOURS) so the user sees one
# canonical "Branch 0 = navy" across every sweep + FEM tile.
_BRANCH_COLOURS: tuple[str, ...] = (
    "#0d3b66", "#f95738", "#3a9d23", "#ee964b",
    "#7f5af0", "#16c172", "#e63946",
)

# Plotly marker symbols for fiber-type discrimination on the
# threshold scatter (when SweepResult carries fiber_type_labels).
# Cycled per unique type label encountered.
_TYPE_SYMBOLS: tuple[str, ...] = (
    "circle", "square", "diamond", "cross", "x",
    "triangle-up", "star", "hexagon",
)


def _maybe_apply_preset(fig: dict, preset) -> dict:
    if preset is None:
        return fig
    from .export import apply_preset_to_plotly_fig
    return apply_preset_to_plotly_fig(fig, preset)


def _branch_colour(b: int) -> str:
    return _BRANCH_COLOURS[int(b) % len(_BRANCH_COLOURS)]


def _percent_ci(p_arr: np.ndarray) -> tuple[float, float, float]:
    """Mean + 95 %  Wald confidence interval for a binary array
    (Bernoulli proportions, Wald approximation). Returns
    (mean_pct, lo_pct, hi_pct). Small-n safeguard: if n < 5 the
    CI collapses to the mean."""
    n = int(p_arr.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    p = float(p_arr.mean())
    if n < 5:
        return p * 100.0, p * 100.0, p * 100.0
    se = float(np.sqrt(max(p * (1 - p), 0.0) / n))
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return p * 100.0, lo * 100.0, hi * 100.0


# ---------------------------------------------------------------------------
# 1. Recruitment curve
# ---------------------------------------------------------------------------


def build_recruitment_curve_figure(
    result: SweepResult,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """% activated vs I_stim per branch (ribbon = mean ± 95 % CI).
    Optional per-fiber-type dotted overlay when result carries
    fiber_type_labels."""
    if (result.activated is None
            or result.request.amplitudes_mA is None
            or len(result.request.amplitudes_mA) == 0):
        return _plotly_placeholder(
            "Run an amplitude sweep to see the recruitment curve."
        )
    amps = np.asarray(
        result.request.amplitudes_mA, dtype=np.float64,
    )
    act = np.asarray(result.activated, dtype=np.bool_)  # (n_fibers, n_amps)
    bidx = np.asarray(result.fiber_branch_idx, dtype=np.int32)
    type_labels = list(result.fiber_type_labels or [])

    traces: list[dict] = []

    # Overall ribbon (across all fibers) — neutral dashed black,
    # underlays the per-branch curves so the user can read both.
    if act.shape[0] >= 1:
        means = np.array([
            float(act[:, ai].mean()) * 100.0
            for ai in range(act.shape[1])
        ])
        traces.append({
            "type": "scatter",
            "x": amps.tolist(),
            "y": means.tolist(),
            "mode": "lines",
            "line": {"color": "#1f2024", "width": 1.8,
                       "dash": "dot"},
            "name": "all fibers (mean)",
            "hovertemplate": (
                "I = %{x:.3f} mA<br>%{y:.1f} %<extra></extra>"
            ),
        })

    # Per-branch ribbons.
    unique_branches = sorted({int(b) for b in bidx.tolist()})
    for b in unique_branches:
        mask = (bidx == b)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        mean_pct = []
        lo_pct = []
        hi_pct = []
        for ai in range(act.shape[1]):
            m, lo, hi = _percent_ci(act[mask, ai])
            mean_pct.append(m)
            lo_pct.append(lo)
            hi_pct.append(hi)
        col = _branch_colour(b)
        # Ribbon (upper bound).
        traces.append({
            "type": "scatter",
            "x": amps.tolist(),
            "y": hi_pct,
            "mode": "lines",
            "line": {"width": 0.0, "color": col},
            "showlegend": False,
            "hoverinfo": "skip",
            "name": f"branch {b} hi",
        })
        # Ribbon (lower bound — fills against the trace above).
        traces.append({
            "type": "scatter",
            "x": amps.tolist(),
            "y": lo_pct,
            "mode": "lines",
            "line": {"width": 0.0, "color": col},
            "fill": "tonexty",
            "fillcolor": _hex_to_rgba(col, 0.18),
            "showlegend": False,
            "hoverinfo": "skip",
            "name": f"branch {b} lo",
        })
        # Mean line on top.
        traces.append({
            "type": "scatter",
            "x": amps.tolist(),
            "y": mean_pct,
            "mode": "lines+markers",
            "line": {"color": col, "width": 2.2},
            "marker": {"color": col, "size": 5},
            "name": f"Branch {b}  (n={n_b})",
            "hovertemplate": (
                f"Branch {b}<br>I = %{{x:.3f}} mA"
                "<br>%{y:.1f} %<extra></extra>"
            ),
        })

    # Per-fiber-type dotted overlay (when type labels present).
    unique_types = [
        t for t in sorted({tl for tl in type_labels if tl})
    ]
    for ti, tlabel in enumerate(unique_types):
        mask = np.array([
            (lbl == tlabel) for lbl in type_labels
        ], dtype=np.bool_)
        n_t = int(mask.sum())
        if n_t == 0:
            continue
        mean_pct = np.array([
            float(act[mask, ai].mean()) * 100.0
            for ai in range(act.shape[1])
        ])
        # Pick a hue distinct from the branch palette by stepping
        # through the same colour list at a non-trivial offset.
        col = _BRANCH_COLOURS[
            (len(unique_branches) + ti) % len(_BRANCH_COLOURS)
        ]
        traces.append({
            "type": "scatter",
            "x": amps.tolist(),
            "y": mean_pct.tolist(),
            "mode": "lines",
            "line": {"color": col, "width": 1.4, "dash": "dash"},
            "name": f"{tlabel}  (n={n_t})",
            "hovertemplate": (
                f"{tlabel}<br>I = %{{x:.3f}} mA"
                "<br>%{y:.1f} %<extra></extra>"
            ),
        })

    layout = {
        "title": {
            "text": (
                f"Recruitment curve  ·  "
                f"{int(act.shape[0])} fibers, "
                f"{int(act.shape[1])} amplitudes  ·  "
                f"{result.elapsed_s:.1f} s"
            ),
            "font": {"size": 12},
            "x": 0.5, "xanchor": "center",
        },
        "xaxis": {
            "title": {
                "text": "stim current  (mA)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
            "type": (
                "log" if (amps.max() / max(amps.min(), 1e-9)
                          >= 100.0) else "linear"
            ),
        },
        "yaxis": {
            "title": {
                "text": "fibers activated  (%)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
            "range": [-2, 102],
        },
        "margin": {"l": 60, "r": 20, "t": 36, "b": 48},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
        "legend": {
            "orientation": "h", "x": 0.0, "y": -0.22,
            "font": {"size": 10, "color": "#4a4a52"},
        },
        "hovermode": "x unified",
    }
    return _maybe_apply_preset({"data": traces, "layout": layout},
                                  preset)


def recruitment_to_csv(result: SweepResult) -> str:
    """One row per (amplitude × branch); columns: I_mA, branch,
    n_fibers, n_activated, percent_activated, ci_lo_pct, ci_hi_pct.
    Reviewer-friendly long format."""
    if result.activated is None:
        return "I_mA,branch,n_fibers,n_activated,pct,ci_lo_pct,ci_hi_pct\n"
    amps = np.asarray(
        result.request.amplitudes_mA, dtype=np.float64,
    )
    act = np.asarray(result.activated, dtype=np.bool_)
    bidx = np.asarray(result.fiber_branch_idx, dtype=np.int32)
    out = io.StringIO()
    out.write(
        "I_mA,branch,n_fibers,n_activated,pct,ci_lo_pct,ci_hi_pct\n"
    )
    unique_branches = sorted({int(b) for b in bidx.tolist()})
    # "Overall" row first for each amplitude.
    for ai, amp in enumerate(amps):
        n_total = int(act.shape[0])
        n_act = int(act[:, ai].sum())
        m, lo, hi = _percent_ci(act[:, ai])
        out.write(
            f"{amp:.6g},all,{n_total},{n_act},"
            f"{m:.4f},{lo:.4f},{hi:.4f}\n"
        )
        for b in unique_branches:
            mask = (bidx == b)
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            n_ab = int(act[mask, ai].sum())
            m, lo, hi = _percent_ci(act[mask, ai])
            out.write(
                f"{amp:.6g},{b},{n_b},{n_ab},"
                f"{m:.4f},{lo:.4f},{hi:.4f}\n"
            )
    return out.getvalue()


# ---------------------------------------------------------------------------
# 2. Threshold-vs-diameter scatter
# ---------------------------------------------------------------------------


def build_threshold_scatter_figure(
    result: SweepResult,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Per-fiber threshold (µA) vs diameter (µm). Colour = branch;
    marker symbol = fiber type. NaN thresholds (no activation in
    bisect range) plot as upward triangles at the top of the
    y-axis range."""
    if result.thresholds_uA is None:
        return _plotly_placeholder(
            "Run threshold-finder mode to see the threshold-vs-"
            "diameter scatter."
        )
    thr = np.asarray(result.thresholds_uA, dtype=np.float64)
    dia = np.asarray(result.fiber_diameters_um, dtype=np.float64)
    bidx = np.asarray(result.fiber_branch_idx, dtype=np.int32)
    type_labels = list(result.fiber_type_labels or [])
    unique_types = sorted({tl for tl in type_labels if tl})

    # Y-axis range: span of finite thresholds; NaN markers go above
    # the max (at y = ymax * 1.15) with an "no activation" hover.
    finite = thr[np.isfinite(thr)]
    if finite.size > 0:
        ymin = float(finite.min())
        ymax = float(finite.max())
    else:
        ymin, ymax = 0.0, 1000.0  # arbitrary fallback
    y_oob = ymax * 1.15 if ymax > 0 else 1.0

    traces: list[dict] = []
    unique_branches = sorted({int(b) for b in bidx.tolist()})
    for b in unique_branches:
        mask_b = (bidx == b)
        for ti, tlabel in enumerate(unique_types or [""]):
            if unique_types:
                mask_t = np.array(
                    [(tl == tlabel) for tl in type_labels],
                    dtype=np.bool_,
                )
                mask = mask_b & mask_t
                tag = f"Branch {b} · {tlabel}"
                sym = _TYPE_SYMBOLS[ti % len(_TYPE_SYMBOLS)]
            else:
                mask = mask_b
                tag = f"Branch {b}"
                sym = "circle"
            n = int(mask.sum())
            if n == 0:
                continue
            finite_mask = mask & np.isfinite(thr)
            nan_mask = mask & ~np.isfinite(thr)
            col = _branch_colour(b)
            # Activated points.
            if finite_mask.any():
                traces.append({
                    "type": "scatter",
                    "x": dia[finite_mask].tolist(),
                    "y": thr[finite_mask].tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": col, "size": 8,
                        "symbol": sym,
                        "line": {"color": "#1f2024",
                                  "width": 0.7},
                    },
                    "name": f"{tag} ({int(finite_mask.sum())})",
                    "hovertemplate": (
                        f"{tag}<br>"
                        "d = %{x:.2f} µm<br>"
                        "thr = %{y:.1f} µA<extra></extra>"
                    ),
                })
            # No-activation triangles at y_oob.
            if nan_mask.any():
                traces.append({
                    "type": "scatter",
                    "x": dia[nan_mask].tolist(),
                    "y": [y_oob] * int(nan_mask.sum()),
                    "mode": "markers",
                    "marker": {
                        "color": col, "size": 7,
                        "symbol": "triangle-up-open",
                        "line": {"color": col, "width": 1.4},
                    },
                    "name": (
                        f"{tag} · no act."
                        f"  ({int(nan_mask.sum())})"
                    ),
                    "hovertemplate": (
                        f"{tag}<br>d = %{{x:.2f}} µm<br>"
                        "no activation in range<extra></extra>"
                    ),
                    "showlegend": True,
                })

    layout = {
        "title": {
            "text": (
                f"Activation threshold vs diameter  ·  "
                f"{int(thr.size)} fibers  ·  "
                f"{int(np.isfinite(thr).sum())} activated  ·  "
                f"{result.elapsed_s:.1f} s"
            ),
            "font": {"size": 12},
            "x": 0.5, "xanchor": "center",
        },
        "xaxis": {
            "title": {
                "text": "fiber diameter  (µm)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "yaxis": {
            "title": {
                "text": "threshold current  (µA)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "margin": {"l": 70, "r": 20, "t": 36, "b": 48},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
        "legend": {
            "orientation": "v",
            "font": {"size": 10, "color": "#4a4a52"},
            "x": 1.02, "y": 1.0,
        },
    }
    return _maybe_apply_preset({"data": traces, "layout": layout},
                                  preset)


def threshold_to_csv(result: SweepResult) -> str:
    """Per-fiber rows: fiber_idx, branch, type, diameter_um,
    threshold_uA, bisect_iters. NaN written as 'NaN'."""
    if result.thresholds_uA is None:
        return ("fiber_idx,branch,fiber_type,diameter_um,"
                "threshold_uA,bisect_iters\n")
    out = io.StringIO()
    out.write("fiber_idx,branch,fiber_type,diameter_um,"
              "threshold_uA,bisect_iters\n")
    types = list(result.fiber_type_labels or [])
    iters = (np.asarray(result.bisect_iters, dtype=np.int32)
             if result.bisect_iters is not None
             else np.zeros(len(result.fiber_indices), dtype=np.int32))
    for fi in range(len(result.fiber_indices)):
        thr = float(result.thresholds_uA[fi])
        thr_str = "NaN" if not np.isfinite(thr) else f"{thr:.3f}"
        tlabel = (types[fi] if fi < len(types) else "")
        out.write(
            f"{int(result.fiber_indices[fi])},"
            f"{int(result.fiber_branch_idx[fi])},"
            f"{tlabel},"
            f"{float(result.fiber_diameters_um[fi]):.4f},"
            f"{thr_str},"
            f"{int(iters[fi])}\n"
        )
    return out.getvalue()


# ---------------------------------------------------------------------------
# 3. Per-fiber activation heatmap
# ---------------------------------------------------------------------------


def build_activation_heatmap_figure(
    result: SweepResult,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Heatmap of fiber × amplitude activation. Fibers sorted by
    (branch, diameter) so visual clusters track functional groups.
    Cell colour: silent (light) vs activated (dark). Hover reveals
    fiber index, branch, type, diameter."""
    if (result.activated is None
            or result.request.amplitudes_mA is None
            or len(result.request.amplitudes_mA) == 0):
        return _plotly_placeholder(
            "Run an amplitude sweep to see the per-fiber "
            "activation heatmap."
        )
    amps = np.asarray(
        result.request.amplitudes_mA, dtype=np.float64,
    )
    act = np.asarray(result.activated, dtype=np.bool_).astype(
        np.int8,
    )
    bidx = np.asarray(result.fiber_branch_idx, dtype=np.int32)
    dia = np.asarray(result.fiber_diameters_um, dtype=np.float64)
    fidx = np.asarray(result.fiber_indices, dtype=np.int64)
    type_labels = list(result.fiber_type_labels or [])

    # Sort fibers by (branch asc, diameter asc) so clusters group.
    order = np.lexsort((dia, bidx))
    act_sorted = act[order]
    bidx_sorted = bidx[order]
    dia_sorted = dia[order]
    fidx_sorted = fidx[order]
    types_sorted = [
        (type_labels[i] if i < len(type_labels) else "")
        for i in order.tolist()
    ]

    # Y-axis ticks: show every Nth fiber label so dense grids stay
    # readable. Compute step from total fibers.
    n = int(act_sorted.shape[0])
    step = max(1, n // 12)
    tick_idx = list(range(0, n, step))
    tick_text = [
        f"f{int(fidx_sorted[i])} · b{int(bidx_sorted[i])}"
        + (f" · {types_sorted[i]}" if types_sorted[i] else "")
        for i in tick_idx
    ]

    # Hover text per cell: build a 2-D matrix of strings.
    hover = []
    for i in range(n):
        row = []
        for ai in range(act_sorted.shape[1]):
            row.append(
                f"fiber {int(fidx_sorted[i])} · "
                f"branch {int(bidx_sorted[i])}"
                + (f" · {types_sorted[i]}"
                   if types_sorted[i] else "")
                + f"<br>d = {float(dia_sorted[i]):.2f} µm"
                + f"<br>I = {float(amps[ai]):.3f} mA"
                + f"<br>{'activated' if act_sorted[i, ai] else 'silent'}"
            )
        hover.append(row)

    traces = [{
        "type": "heatmap",
        "x": amps.tolist(),
        "y": list(range(n)),
        "z": act_sorted.tolist(),
        "colorscale": [
            [0.0, "#f4f4f6"],   # silent: very light grey
            [1.0, "#1f2024"],   # activated: dark
        ],
        "zmin": 0, "zmax": 1,
        "showscale": False,
        "hoverinfo": "text",
        "text": hover,
        "xgap": 0.5, "ygap": 0.5,
    }]
    layout = {
        "title": {
            "text": (
                f"Per-fiber activation map  ·  "
                f"{n} fibers (sorted by branch + diameter)  ·  "
                f"{int(act_sorted.shape[1])} amplitudes"
            ),
            "font": {"size": 12},
            "x": 0.5, "xanchor": "center",
        },
        "xaxis": {
            "title": {
                "text": "stim current  (mA)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": False,
        },
        "yaxis": {
            "title": {
                "text": "fiber  (sorted)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "tickmode": "array",
            "tickvals": tick_idx,
            "ticktext": tick_text,
            "showgrid": False,
            "autorange": "reversed",
        },
        "margin": {"l": 110, "r": 20, "t": 36, "b": 48},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
    }
    return _maybe_apply_preset({"data": traces, "layout": layout},
                                  preset)


def activation_heatmap_to_csv(result: SweepResult) -> str:
    """Wide-format CSV: rows = fibers (sorted), columns =
    amplitudes. Header: fiber_idx, branch, fiber_type, diameter_um,
    then one column per amplitude `I_X.XXX_mA` carrying 0/1."""
    if result.activated is None:
        return "fiber_idx,branch,fiber_type,diameter_um\n"
    amps = np.asarray(
        result.request.amplitudes_mA, dtype=np.float64,
    )
    act = np.asarray(result.activated, dtype=np.bool_).astype(int)
    bidx = np.asarray(result.fiber_branch_idx, dtype=np.int32)
    dia = np.asarray(result.fiber_diameters_um, dtype=np.float64)
    fidx = np.asarray(result.fiber_indices, dtype=np.int64)
    type_labels = list(result.fiber_type_labels or [])
    order = np.lexsort((dia, bidx))

    out = io.StringIO()
    headers = ["fiber_idx", "branch", "fiber_type", "diameter_um"]
    headers += [f"I_{a:.4f}_mA" for a in amps]
    out.write(",".join(headers) + "\n")
    for ii in order.tolist():
        cells = [
            str(int(fidx[ii])),
            str(int(bidx[ii])),
            type_labels[ii] if ii < len(type_labels) else "",
            f"{float(dia[ii]):.4f}",
        ]
        cells += [str(int(act[ii, ai]))
                   for ai in range(act.shape[1])]
        out.write(",".join(cells) + "\n")
    return out.getvalue()
