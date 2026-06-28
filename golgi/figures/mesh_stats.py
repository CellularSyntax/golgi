# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Mesh-quality histogram + per-tag stats table renderers.

Two surfaces consume the histogram:
  * Import drawer — surface triangle quality after STL loading
    (`state.quality_hist_figure`, axis label "q_radius_ratio").
  * Mesh drawer — tetrahedral element quality after TetGen
    (`state.mesh_quality_hist_figure`, axis label "tet quality
    6√2·V / max_edge³").

Both are now Plotly bars with a RdYlGn colour-coded marker.color
mapped per bin (red = degenerate, green = near-equilateral), so the
per-panel F2.3.a export button + the bulk Exports tab pick them up
via the registry like every other Plotly figure."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .export import apply_preset_to_plotly_fig

if TYPE_CHECKING:                                  # pragma: no cover
    from .export import FigureExportPreset


# Canonical 11-stop matplotlib RdYlGn colorscale, written out
# as a Plotly `[position, "rgb(...)"]` list so all the quality
# histograms (import-nerve, mesh drawer, µCT-reconstruct) render
# identically regardless of which plotly.js version ships with
# the host trame install. Plotly's named "RdYlGn" scale has
# shipped with different interpolation tables in the past, so
# pinning the stops here is the only way to guarantee:
#   * q = 0   → dark red    (degenerate triangle)
#   * q ≈ 0.5 → pale yellow (neutral)
#   * q = 1   → dark green  (near-equilateral)
# Stops are from Cynthia Brewer's `RdYlGn` diverging palette
# (colorbrewer2.org).
_RDYLGN_STOPS = [
    [0.0,  "rgb(165, 0, 38)"],
    [0.1,  "rgb(215, 48, 39)"],
    [0.2,  "rgb(244, 109, 67)"],
    [0.3,  "rgb(253, 174, 97)"],
    [0.4,  "rgb(254, 224, 139)"],
    [0.5,  "rgb(255, 255, 191)"],
    [0.6,  "rgb(217, 239, 139)"],
    [0.7,  "rgb(166, 217, 106)"],
    [0.8,  "rgb(102, 189, 99)"],
    [0.9,  "rgb(26, 152, 80)"],
    [1.0,  "rgb(0, 104, 55)"],
]


def _maybe_apply_preset(
    fig: dict, preset: "FigureExportPreset | None",
) -> dict:
    if preset is not None:
        apply_preset_to_plotly_fig(fig, preset)
    return fig


def _build_quality_histogram_figure(
    q: np.ndarray,
    nbins: int = 24,
    *,
    x_label: str = "q_radius_ratio",
    y_label: str = "# elements",
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Build a Plotly bar histogram of quality scores with bins
    coloured along RdYlGn by their centre value. Same visual
    semantics as the old matplotlib renderer (degenerate = red,
    near-equilateral = green); returns a `{data, layout}` dict
    ready to push into a state variable + render via twp.Figure."""
    counts, edges = np.histogram(q, bins=nbins, range=(0.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])
    bar_width = float((edges[1] - edges[0]) * 0.96)
    n_total = int(q.size)
    median_q = float(np.median(q)) if n_total else 0.0
    min_q = float(np.min(q)) if n_total else 0.0
    fig: dict = {
        "data": [
            {
                "type": "bar",
                "x": centres.tolist(),
                "y": counts.tolist(),
                "width": [bar_width] * len(centres),
                "marker": {
                    # Pinned 11-stop RdYlGn (red @ degenerate,
                    # green @ near-equilateral) so all quality
                    # histograms in the app render identically.
                    "color": centres.tolist(),
                    "colorscale": _RDYLGN_STOPS,
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "line": {"color": "#222222", "width": 0.3},
                    "showscale": False,
                },
                "hovertemplate": (
                    "q ≈ %{x:.3f}<br>"
                    "# = %{y}<extra></extra>"
                ),
                "name": "quality",
            },
        ],
        "layout": {
            "xaxis": {
                "title": {"text": x_label, "font": {"size": 10}},
                "range": [0.0, 1.0],
                "tickfont": {"size": 9},
                "showgrid": True,
                "gridcolor": "#ececef",
            },
            "yaxis": {
                "title": {"text": y_label, "font": {"size": 10}},
                "tickfont": {"size": 9},
                "showgrid": True,
                "gridcolor": "#ececef",
            },
            "margin": {"l": 50, "r": 12, "t": 28, "b": 40},
            "bargap": 0.04,
            "showlegend": False,
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "annotations": [
                {
                    "xref": "paper", "yref": "paper",
                    "x": 1.0, "y": 1.04,
                    "xanchor": "right", "yanchor": "bottom",
                    "text": (
                        f"n = {n_total:,} · median = "
                        f"{median_q:.3f} · min = {min_q:.3f}"
                    ),
                    "showarrow": False,
                    "font": {
                        "size": 10, "color": "#555555",
                    },
                },
            ],
        },
    }
    return _maybe_apply_preset(fig, preset)


def _build_combined_quality_histogram_figure(
    panels: list,
    *,
    x_label: str = "tet quality (6√2·V / max_edge³)",
    y_label: str = "# elements",
) -> dict:
    """Stack one quality-histogram subplot per design vertically.
    `panels` is a list of dicts with keys {name, q}. Each row gets
    its own bar trace coloured along RdYlGn + a header annotation
    naming the design and reporting n / median / min. Returns a
    plotly `{data, layout}` dict.

    F3.2-M2.1e — the Mesh drawer used to host a single histogram
    of the active design's mesh. With per-design meshing, we
    want every built design visible at once."""
    if not panels:
        return {"data": [], "layout": {}}

    n = len(panels)
    nbins = 24
    bin_edges = np.linspace(0.0, 1.0, nbins + 1)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_width = float((bin_edges[1] - bin_edges[0]) * 0.96)

    data: list[dict] = []
    annotations: list[dict] = []
    layout: dict = {
        "showlegend": False,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        # Vertical spacing between subplots; small enough that
        # 5+ designs fit reasonably in the drawer.
        "margin": {"l": 50, "r": 12, "t": 20, "b": 36},
    }
    for i, p in enumerate(panels, start=1):
        # `or []` would trip numpy's "ambiguous truth value of
        # array" — use an explicit None check.
        _q_in = p.get("q")
        if _q_in is None:
            q = np.zeros(0, dtype=np.float64)
        else:
            q = np.asarray(_q_in, dtype=np.float64)
        n_total = int(q.size)
        if n_total == 0:
            counts = np.zeros(nbins, dtype=np.int64)
            median_q = 0.0
            min_q = 0.0
        else:
            counts, _ = np.histogram(
                q, bins=nbins, range=(0.0, 1.0),
            )
            median_q = float(np.median(q))
            min_q = float(np.min(q))
        xaxis_key = "x" if i == 1 else f"x{i}"
        yaxis_key = "y" if i == 1 else f"y{i}"
        # Each row occupies an equal vertical slice of the figure,
        # leaving room for the bottom row's x-axis labels.
        y_top = 1.0 - (i - 1) / n
        y_bot = 1.0 - i / n
        y_pad = 0.04 / n
        layout[f"xaxis{i if i > 1 else ''}"] = {
            "title": (
                {"text": x_label, "font": {"size": 10}}
                if i == n else
                {"text": "", "font": {"size": 10}}
            ),
            "range": [0.0, 1.0],
            "tickfont": {"size": 8},
            "showgrid": True,
            "gridcolor": "#ececef",
            "anchor": yaxis_key,
            "domain": [0.0, 1.0],
        }
        layout[f"yaxis{i if i > 1 else ''}"] = {
            "title": {"text": y_label, "font": {"size": 9}},
            "tickfont": {"size": 8},
            "showgrid": True,
            "gridcolor": "#ececef",
            "anchor": xaxis_key,
            "domain": [y_bot + y_pad, y_top - y_pad],
        }
        data.append({
            "type": "bar",
            "x": centres.tolist(),
            "y": counts.tolist(),
            "width": [bar_width] * len(centres),
            "marker": {
                "color": centres.tolist(),
                "colorscale": "RdYlGn",
                "cmin": 0.0,
                "cmax": 1.0,
                "line": {"color": "#222222", "width": 0.3},
                "showscale": False,
            },
            "hovertemplate": (
                "q ≈ %{x:.3f}<br># = %{y}<extra></extra>"
            ),
            "xaxis": xaxis_key,
            "yaxis": yaxis_key,
            "name": str(p.get("name", "")),
        })
        # Per-row header — design name + summary stats anchored
        # to the top of that subplot's domain.
        annotations.append({
            "xref": "paper", "yref": "paper",
            "x": 0.0, "y": y_top,
            "xanchor": "left", "yanchor": "bottom",
            "text": (
                f"<b>{p.get('name', '')}</b>"
                f"  ·  n = {n_total:,}"
                f"  ·  median = {median_q:.3f}"
                f"  ·  min = {min_q:.3f}"
            ),
            "showarrow": False,
            "font": {"size": 10, "color": "#1f2024"},
        })
    layout["annotations"] = annotations
    return {"data": data, "layout": layout}


def _compute_mesh_stats_html(nodes: np.ndarray,
                               elems: np.ndarray,
                               tags: np.ndarray,
                               q_tet: np.ndarray,
                               *,
                               defaults_by_tag: dict) -> str:
    """HTML stats table for the built multi-domain mesh.
    Reports per-tag tet count, points-used count, and quality
    summary (min / median / mean) in a compact panel matching the
    fiber-stats look.

    `defaults_by_tag` is injected (kwarg-only) — the live `DEFAULTS`
    dict (tag → {label, color}) still lives in golgi.py, so callers
    pass `defaults_by_tag=DEFAULTS`."""
    if elems is None or len(elems) == 0:
        return ("<div style='color:#666;font-size:11px;'>"
                "no mesh data</div>")
    utags = sorted({int(t) for t in np.unique(tags)})
    rows: list[tuple] = [
        ("Overall", "—",
         int(len(nodes)), int(len(elems)),
         float(q_tet.min()),
         float(np.median(q_tet)),
         float(q_tet.mean())),
    ]
    for t in utags:
        mask = (tags == t)
        if not mask.any():
            continue
        sub_tets = elems[mask]
        n_t = int(mask.sum())
        n_p = int(np.unique(sub_tets.ravel()).size)
        sub_q = q_tet[mask]
        label = (defaults_by_tag[t]["label"] if t in defaults_by_tag
                  else f"tag {t}")
        rgb = (defaults_by_tag[t]["color"] if t in defaults_by_tag
                else (0.6, 0.6, 0.6))
        rgb_css = (
            f"rgb({int(rgb[0]*255)},"
            f"{int(rgb[1]*255)},"
            f"{int(rgb[2]*255)})"
        )
        rows.append((
            label, rgb_css,
            n_p, n_t,
            float(sub_q.min()),
            float(np.median(sub_q)),
            float(sub_q.mean()),
        ))
    # Render
    head = (
        "<tr style='border-bottom:1px solid #d6d6da;'>"
        "<th style='text-align:left;padding:3px 8px 3px 0;'></th>"
        "<th style='text-align:right;padding:3px 6px;'>pts</th>"
        "<th style='text-align:right;padding:3px 6px;'>tets</th>"
        "<th style='text-align:right;padding:3px 6px;'>q min</th>"
        "<th style='text-align:right;padding:3px 6px;'>q med</th>"
        "<th style='text-align:right;padding:3px 0 3px 6px;'>q μ</th>"
        "</tr>"
    )
    tr_html: list[str] = []
    for name, rgb_css, n_p, n_t, qmin, qmed, qmean in rows:
        if name == "Overall":
            badge = (
                "<span style='font-weight:600;"
                "color:#1f2024;'>Overall</span>"
            )
        else:
            badge = (
                f"<span style='display:inline-block;"
                f"width:10px;height:10px;border-radius:2px;"
                f"background:{rgb_css};margin-right:6px;"
                f"vertical-align:middle;'></span>"
                f"<span style='color:#1f2024;'>{name}</span>"
            )
        tr_html.append(
            f"<tr>"
            f"<td style='padding:3px 8px 3px 0;'>{badge}</td>"
            f"<td style='padding:3px 6px;text-align:right;'>"
            f"{n_p:,}</td>"
            f"<td style='padding:3px 6px;text-align:right;'>"
            f"{n_t:,}</td>"
            f"<td style='padding:3px 6px;text-align:right;'>"
            f"{qmin:.3f}</td>"
            f"<td style='padding:3px 6px;text-align:right;'>"
            f"{qmed:.3f}</td>"
            f"<td style='padding:3px 0 3px 6px;text-align:right;'>"
            f"{qmean:.3f}</td>"
            f"</tr>"
        )
    return (
        "<table style='border-collapse:collapse;font-size:12px;"
        "width:100%;font-family:-apple-system,BlinkMacSystemFont,"
        "\"Segoe UI\",sans-serif;'>"
        f"{head}{''.join(tr_html)}"
        "</table>"
    )
