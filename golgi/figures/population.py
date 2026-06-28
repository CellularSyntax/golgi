# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Population-tab figure builders (KDE traces, cross-sections)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .util import (
    _FIBER_AXIS_TICK_FONT,
    _FIBER_AXIS_TITLE_FONT,
    _hex_to_rgba,
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


def _build_pop_kde_figure(
    bidx: np.ndarray,
    pop_rows: np.ndarray,
    pop_diams: np.ndarray,
    branches_meta: list,
    row_meta: dict,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Per-branch + overall + per-row KDE of fiber diameters.

    Layout:
      Top row: N branch subplots + 1 overall subplot. Each
               subplot stacks one filled KDE curve per ROW
               whose fibers appear in that slot, coloured by
               the row's tab10 colour.
      Bottom row (only if there's ≥1 named row): one subplot
               per named row, showing just that row's fibers.
               Subplot title = row.name.

    `row_meta` is the {row_id → {name, color, model, ...}} map
    built by `do_pop_generate`. Plain plotly dicts (no
    plotly.graph_objects) so trame-plotly can render directly
    via state_variable_name."""
    if (pop_rows is None or pop_diams is None
            or len(branches_meta) == 0):
        return _plotly_placeholder(
            "Generate a population to see the diameter "
            "distribution by named subpopulation."
        )
    pop_diams = np.asarray(pop_diams, dtype=np.float64)
    bidx = np.asarray(bidx, dtype=np.int32)
    valid = (pop_diams > 0) & np.array(
        [bool(r) for r in pop_rows], dtype=bool,
    )
    if not valid.any():
        return _plotly_placeholder(
            "Generate a population to see the diameter "
            "distribution by named subpopulation."
        )
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return _plotly_placeholder(
            "Install `scipy` to render the KDE figure."
        )
    # Order rows by branch then by add-order so the per-row
    # subplots in the bottom strip read in a sensible left-to-
    # right sequence (Branch 0's rows first, then Branch 1's,
    # etc.).
    row_ids_ordered = sorted(
        row_meta.keys(),
        key=lambda rid: (
            int(row_meta[rid].get("branch", -1)),
            rid,
        ),
    )

    d_lo = max(0.0, float(pop_diams[valid].min()) - 1.0)
    d_hi = float(pop_diams[valid].max()) + 1.0
    if d_hi - d_lo < 0.5:
        d_hi = d_lo + 0.5
    xs = np.linspace(d_lo, d_hi, 200)
    xs_list = xs.tolist()

    # ---- Subplot grid layout ---------------------------------
    n_branches = len(branches_meta)
    top_n = n_branches + 1            # branch subplots + Overall
    bottom_n = len(row_ids_ordered)   # per-row subplots
    has_bottom = bottom_n > 0
    gap_x = 0.025
    # Vertical split: top row gets a fixed fraction, bottom the
    # rest. When there's no bottom strip, top takes the full
    # plot area.
    if has_bottom:
        top_y_lo, top_y_hi = 0.58, 1.0
        bot_y_lo, bot_y_hi = 0.0, 0.42
    else:
        top_y_lo, top_y_hi = 0.0, 1.0
        bot_y_lo, bot_y_hi = 0.0, 0.0

    def _row_domains(n: int) -> list[tuple[float, float]]:
        if n <= 0:
            return []
        span = (1.0 - gap_x * (n - 1)) / n
        return [
            (i * (span + gap_x), i * (span + gap_x) + span)
            for i in range(n)
        ]

    top_domains = _row_domains(top_n)
    bot_domains = _row_domains(bottom_n)

    # Plotly axes: 1-indexed but the first uses no suffix.
    # We assign top axes 1..top_n and bottom axes
    # top_n+1..top_n+bottom_n. Each subplot has one x-axis +
    # one y-axis, sharing the same index suffix.

    def _ax_suf(idx_1based: int) -> str:
        return "" if idx_1based == 1 else str(idx_1based)

    traces: list = []
    legend_seen: set = set()

    def _add_curves_for(
        slot_mask: np.ndarray, ax_idx: int,
        only_row_id: str | None = None,
    ) -> None:
        diams = pop_diams[slot_mask]
        rids = pop_rows[slot_mask]
        if diams.size == 0:
            return
        # Iterate rows in deterministic add-order so the stacks
        # match across subplots.
        candidate_rids = (
            [only_row_id] if only_row_id is not None
            else [r for r in row_ids_ordered
                  if r in set(rids.tolist())]
        )
        for rid in candidate_rids:
            rmask = rids == rid
            r_diams = diams[rmask]
            if r_diams.size < 2:
                continue
            try:
                kde = gaussian_kde(r_diams)
                ys = kde(xs).tolist()
            except Exception:
                continue
            meta = row_meta.get(rid, {})
            color = str(meta.get("color", "#666"))
            name = str(meta.get("name", "?"))
            ax_suf = _ax_suf(ax_idx)
            show_legend = (
                only_row_id is None and rid not in legend_seen
            )
            if show_legend:
                legend_seen.add(rid)
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": xs_list,
                "y": ys,
                "name": name,
                "legendgroup": rid,
                "showlegend": show_legend,
                "fill": "tozeroy",
                "fillcolor": _hex_to_rgba(color, 0.25),
                "line": {"color": color, "width": 2},
                "xaxis": f"x{ax_suf}",
                "yaxis": f"y{ax_suf}",
                "hovertemplate": (
                    f"{name}<br>d = %{{x:.2f}} µm"
                    "<br>density = %{y:.3f}<extra></extra>"
                ),
            })

    # Top row: per-branch + overall.
    for col, meta in enumerate(branches_meta, start=1):
        b = int(meta["idx"])
        slot = (bidx == b) & valid
        _add_curves_for(slot, col)
    _add_curves_for(valid, top_n)

    # Bottom row: per-named-row.
    for i, rid in enumerate(row_ids_ordered):
        slot = (pop_rows == rid) & valid
        _add_curves_for(slot, top_n + 1 + i, only_row_id=rid)

    # ---- Layout ----------------------------------------------
    # Plotly figure height — needs to grow when the bottom
    # strip is present so the per-named-row subplots have
    # enough room for their tick labels + the legend below,
    # otherwise they overflow into whatever's below the KDE
    # tile (the pulse designer in the Population panel).
    total_height = 560 if has_bottom else 280
    layout: dict = {
        "height": total_height,
        "margin": {"l": 56, "r": 16, "t": 44, "b": 64},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "x": 0.5, "y": -0.18,
            "xanchor": "center",
            "font": _FIBER_AXIS_TICK_FONT,
        },
        "annotations": [],
    }
    top_titles = (
        [m["label"] for m in branches_meta] + ["Overall"]
    )
    bot_titles = [
        str(row_meta[rid].get("name", "?"))
        for rid in row_ids_ordered
    ]

    def _add_axis(idx_1based: int,
                  x_lo: float, x_hi: float,
                  y_lo: float, y_hi: float,
                  title_text: str,
                  show_y_title: bool) -> None:
        ax_suf = _ax_suf(idx_1based)
        layout[f"xaxis{ax_suf}"] = {
            "domain": [x_lo, x_hi],
            "anchor": f"y{ax_suf}",
            "title": {
                "text": "diameter (µm)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
            "range": [d_lo, d_hi],
        }
        layout[f"yaxis{ax_suf}"] = {
            "domain": [y_lo, y_hi],
            "anchor": f"x{ax_suf}",
            "title": (
                {"text": "density",
                  "font": _FIBER_AXIS_TITLE_FONT}
                if show_y_title else None
            ),
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
            "rangemode": "tozero",
        }
        # Centred title annotation above the subplot.
        layout["annotations"].append({
            "text": title_text,
            "x": (x_lo + x_hi) / 2.0,
            "y": y_hi + 0.02,
            "xref": "paper", "yref": "paper",
            "xanchor": "center", "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 12, "color": "#1f2024"},
        })

    for i, (lo, hi) in enumerate(top_domains):
        _add_axis(
            i + 1, lo, hi, top_y_lo, top_y_hi,
            top_titles[i], show_y_title=(i == 0),
        )
    for i, (lo, hi) in enumerate(bot_domains):
        _add_axis(
            top_n + 1 + i, lo, hi, bot_y_lo, bot_y_hi,
            bot_titles[i], show_y_title=(i == 0),
        )
    return _maybe_apply_preset({"data": traces, "layout": layout}, preset)


def _build_pop_xsec_at_cuff_figure(
    paths_display: list,
    bidx: np.ndarray | None,
    pop_rows: np.ndarray | None,
    pop_diams: np.ndarray | None,
    row_meta: dict,
    nerve_pts_cuff_m: np.ndarray | None,
    z_band_mm: float = 0.5,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Cross-section at the cuff centre (z = 0 in cuff frame).
    Drives the new top tile in the Population panel. Layers:
      * Nerve outline — convex hull of the nerve points within
        a thin ±`z_band_mm` band of z=0, drawn as a light grey
        polygon so the dots have a contextual frame.
      * Per-branch convex hull — black dotted polygon around
        each branch's fiber positions at z=0 (skipped if a
        branch has < 3 fibers).
      * Fiber dots — one marker per fiber at its trajectory's
        nearest-to-z=0 (x, y), coloured by the user's named
        subpopulation (`row_meta[pop_rows[i]].color`), sized by
        the sampled diameter.

    Returns a plain plotly dict — no plotly.graph_objects so the
    trame-plotly widget can render it via state_variable_name.
    """
    if (not paths_display or pop_rows is None
            or bidx is None):
        return _plotly_placeholder(
            "Generate a population to see the cross-section "
            "at the cuff centre."
        )
    pop_rows = np.asarray(pop_rows, dtype=object)
    pop_diams = (
        np.asarray(pop_diams, dtype=np.float64)
        if pop_diams is not None
        else np.zeros(len(paths_display), dtype=np.float64)
    )
    bidx = np.asarray(bidx, dtype=np.int32)
    n_fibers = len(paths_display)

    xs: list[float] = []
    ys: list[float] = []
    colors: list[str] = []
    sizes: list[float] = []
    hovers: list[str] = []
    branch_pts: dict[int, list[tuple[float, float]]] = {}

    for fi, path in enumerate(paths_display):
        try:
            p = np.asarray(path, dtype=np.float64)
        except Exception:
            continue
        if p.ndim != 2 or p.shape[0] == 0:
            continue
        p_mm = p * 1000.0  # m → mm (cuff frame)
        ki = int(np.argmin(np.abs(p_mm[:, 2])))
        x_mm = float(p_mm[ki, 0])
        y_mm = float(p_mm[ki, 1])
        rid = str(pop_rows[fi]) if fi < pop_rows.size else ""
        meta = row_meta.get(rid, {})
        color = str(meta.get("color", "#d0d3d8"))
        name = str(meta.get("name", "—"))
        d_um = (
            float(pop_diams[fi])
            if fi < pop_diams.size else 0.0
        )
        marker_size = max(5.0, min(d_um * 1.4, 18.0))
        xs.append(x_mm)
        ys.append(y_mm)
        colors.append(color)
        sizes.append(marker_size)
        hovers.append(
            f"Fiber {fi}<br>type: {name}"
            f"<br>d = {d_um:.2f} µm"
        )
        b = int(bidx[fi]) if fi < bidx.size else -1
        if b >= 0:
            branch_pts.setdefault(b, []).append((x_mm, y_mm))

    traces: list = []

    # 1) Nerve outline at z=0 — convex hull of cuff-frame
    # points whose z is within ±z_band_mm of 0. Light grey
    # fill so the colour dots sit inside a recognisable shape.
    if (nerve_pts_cuff_m is not None
            and nerve_pts_cuff_m.size > 0):
        try:
            pts_mm = (
                np.asarray(nerve_pts_cuff_m, dtype=np.float64)
                * 1000.0
            )
            zmask = np.abs(pts_mm[:, 2]) <= float(z_band_mm)
            if zmask.sum() >= 3:
                slice_xy = pts_mm[zmask][:, :2]
                from scipy.spatial import ConvexHull
                hull = ConvexHull(slice_xy)
                hpts = slice_xy[hull.vertices]
                hx = list(hpts[:, 0]) + [float(hpts[0, 0])]
                hy = list(hpts[:, 1]) + [float(hpts[0, 1])]
                traces.append({
                    "type": "scatter",
                    "mode": "lines",
                    "x": hx, "y": hy,
                    "line": {
                        "color": "#1f2024", "width": 1,
                    },
                    "fill": "toself",
                    "fillcolor": "rgba(31,32,36,0.04)",
                    "name": "Nerve outline",
                    "hoverinfo": "skip",
                    "showlegend": True,
                })
        except Exception as _ex:
            print(
                f"[pop_xsec] nerve hull failed: {_ex}",
                flush=True,
            )

    # 2) Per-branch convex hull — thin dotted black polygon.
    # Skip branches with fewer than 3 points (hull undefined).
    try:
        from scipy.spatial import ConvexHull
        for b in sorted(branch_pts.keys()):
            pts = branch_pts[b]
            if len(pts) < 3:
                continue
            arr = np.asarray(pts, dtype=np.float64)
            try:
                hull = ConvexHull(arr)
            except Exception:
                continue
            hpts = arr[hull.vertices]
            hx = list(hpts[:, 0]) + [float(hpts[0, 0])]
            hy = list(hpts[:, 1]) + [float(hpts[0, 1])]
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": hx, "y": hy,
                "line": {
                    "color": "#000000",
                    "width": 1.4,
                    "dash": "dot",
                },
                "name": f"Branch {b}",
                "hoverinfo": "skip",
                "showlegend": True,
            })
    except Exception as _ex:
        print(
            f"[pop_xsec] branch hulls failed: {_ex}",
            flush=True,
        )

    # 3) Fiber dots (in last so they paint over the hulls).
    if xs:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": xs, "y": ys,
            "marker": {
                "color": colors,
                "size": sizes,
                "line": {"width": 0.4, "color": "#1f2024"},
            },
            "text": hovers,
            "hovertemplate": "%{text}<extra></extra>",
            "name": "Fibers",
            "showlegend": False,
        })

    layout: dict = {
        "height": 380,
        "margin": {"l": 56, "r": 16, "t": 28, "b": 56},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
        "xaxis": {
            "title": {
                "text": "x (mm) — cuff frame",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
            "scaleanchor": "y",
            "scaleratio": 1,
            "zeroline": True,
            "zerolinecolor": "#d0d3d8",
        },
        "yaxis": {
            "title": {
                "text": "y (mm)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
            "zeroline": True,
            "zerolinecolor": "#d0d3d8",
        },
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "x": 0.5, "y": -0.22,
            "xanchor": "center",
            "font": _FIBER_AXIS_TICK_FONT,
        },
    }
    return _maybe_apply_preset({"data": traces, "layout": layout}, preset)


def _build_pop_xsec_figure(
    paths_display: list,
    pop_rows: np.ndarray | None,
    pop_diams: np.ndarray | None,
    row_meta: dict,
    activated_set: set,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Cross-section overview: each fiber rendered as ONE
    point at (centroid_x, centroid_y) in the display frame
    (mm). Activated fibers (≥1 AP fired) get their row's
    tab10 colour; un-activated fibers stay light grey. Marker
    diameter scaled by the sampled fiber diameter so the user
    can see the population's distribution at a glance.

    Two traces — activated and un-activated — so toggling
    them in the legend is cheap. Plain plotly dicts; no
    plotly.graph_objects."""
    if (paths_display is None or len(paths_display) == 0
            or pop_rows is None):
        return _plotly_placeholder(
            "Run the population simulation to see the "
            "cross-section overview."
        )
    pop_rows = np.asarray(pop_rows, dtype=object)
    if pop_diams is None:
        pop_diams_f = np.ones(len(paths_display), dtype=np.float64)
    else:
        pop_diams_f = np.asarray(pop_diams, dtype=np.float64)
    act = {
        "x": [], "y": [], "color": [], "size": [], "text": [],
    }
    qui = {
        "x": [], "y": [], "color": [], "size": [], "text": [],
    }
    for fi, path in enumerate(paths_display):
        rid = str(pop_rows[fi]) if fi < len(pop_rows) else ""
        if not rid:
            continue
        meta = row_meta.get(rid, {})
        try:
            p_mm = np.asarray(path, dtype=np.float64) * 1000.0
        except Exception:
            continue
        if p_mm.size == 0:
            continue
        cx, cy = float(p_mm[:, 0].mean()), float(p_mm[:, 1].mean())
        d_um = (float(pop_diams_f[fi])
                if fi < pop_diams_f.size else 0.0)
        marker_size = max(5.0, min(d_um * 1.6, 22.0))
        is_act = int(fi) in activated_set
        bucket = act if is_act else qui
        bucket["x"].append(cx)
        bucket["y"].append(cy)
        bucket["color"].append(
            str(meta.get("color", "#666"))
            if is_act else "#d0d3d8"
        )
        bucket["size"].append(marker_size)
        bucket["text"].append(
            f"Fiber {fi}<br>"
            f"{meta.get('name', '?')} ({meta.get('model', '?')})"
            f"<br>d = {d_um:.2f} µm"
            f"<br>{'activated' if is_act else 'quiescent'}"
        )
    traces: list = []
    if qui["x"]:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": qui["x"], "y": qui["y"],
            "name": "quiescent",
            "marker": {
                "size": qui["size"],
                "color": qui["color"],
                "line": {
                    "width": 0.5,
                    "color": "rgba(0,0,0,0.20)",
                },
                "opacity": 0.7,
            },
            "text": qui["text"],
            "hovertemplate": "%{text}<extra></extra>",
        })
    if act["x"]:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": act["x"], "y": act["y"],
            "name": "activated",
            "marker": {
                "size": act["size"],
                "color": act["color"],
                "line": {
                    "width": 0.8,
                    "color": "rgba(0,0,0,0.45)",
                },
                "opacity": 1.0,
            },
            "text": act["text"],
            "hovertemplate": "%{text}<extra></extra>",
        })
    if not traces:
        return _plotly_placeholder(
            "Run the population simulation to see the "
            "cross-section overview."
        )
    layout = {
        "height": 380,
        "margin": {"l": 60, "r": 16, "t": 36, "b": 56},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "x": 0.5, "y": -0.18,
            "xanchor": "center",
            "font": _FIBER_AXIS_TICK_FONT,
        },
        "xaxis": {
            "title": {
                "text": "x  (mm)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
            "zeroline": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        "yaxis": {
            "title": {
                "text": "y  (mm)",
                "font": _FIBER_AXIS_TITLE_FONT,
            },
            "tickfont": _FIBER_AXIS_TICK_FONT,
            "showgrid": True, "gridcolor": "#e5e5ea",
            "zeroline": False,
        },
    }
    return _maybe_apply_preset({"data": traces, "layout": layout}, preset)
