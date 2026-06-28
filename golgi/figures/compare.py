# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Compare-view figure builders (F3.2e).

Reads per-config FEM outputs from `<out>/configs/<cid>/` (the
F3.2c layout) and produces overlay figures across N selected
configs:

  * `build_compare_axis_figure(out_dir, cids, configs_meta)`
      — Vₑ along the cuff axis, one line per config + a shared
        z-axis. Source: `<config_dir>/axis_line.npz`.
  * `build_compare_slice_grid(out_dir, cids, configs_meta, z_idx)`
      — Vₑ slice heatmaps at a chosen z-index, one subplot per
        config in a responsive grid. Source: `<config_dir>/
        slice_volume.npz` (the full 3D volume baked at solve
        time; we just pick the slice).

Both return Plotly figure dicts. Empty / missing inputs → a
plotly placeholder with the user-facing reason.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .util import _plotly_placeholder as _placeholder


def _maybe_apply_preset(fig: dict, preset) -> dict:
    """Lazy-import the export-preset shim so this module stays
    cheap to load when nobody's exporting. Mirrors the same
    helper in figures/recruitment.py."""
    if preset is None:
        return fig
    from .export import apply_preset_to_plotly_fig
    return apply_preset_to_plotly_fig(fig, preset)


# Distinct line colours for up to ~10 configs. The N+1-th wraps
# around. Matches the existing palette in figures/fem.py to keep
# the look consistent.
_CONFIG_COLOURS = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#e377c2", "#7f7f7f", "#bcbd22", "#8c564b",
)


def _config_dir(out_dir: Path, cid: str) -> Path:
    """Resolve a config's solve-output dir. Mirrors
    `pipeline.fem_layout.config_dir` but kept local so the
    figures module doesn't take a hard dep on the pipeline
    package."""
    out_dir = Path(out_dir)
    p = out_dir / "configs" / cid
    if p.is_dir():
        return p
    # Legacy flat fallback — only matters for projects that ran
    # FEM before F3.2c (axis_line.npz at project root).
    if cid == "default" and (out_dir / "axis_line.npz").is_file():
        return out_dir
    return p


def build_compare_axis_figure(
    out_dir: Path,
    cids: list[str],
    configs_meta: list[dict],
    *,
    preset=None,
) -> dict:
    """Per-config Vₑ-along-axis overlay. One trace per cid; the
    trace label combines the parent design name + the config
    name so identically-named configs on different designs are
    distinguishable in the legend."""
    if not cids:
        return _placeholder(
            "Pick at least one config in the Compare tab "
            "multi-select to overlay V<sub>e</sub> along the "
            "cuff axis."
        )
    meta_by_id = {m.get("id", ""): m for m in (configs_meta or [])}
    traces = []
    plotted = 0
    for i, cid in enumerate(cids):
        d = _config_dir(out_dir, cid)
        axis_path = d / "axis_line.npz"
        if not axis_path.is_file():
            continue
        try:
            data = np.load(axis_path, allow_pickle=True)
            z_m = np.asarray(data["z"], dtype=np.float64)
            ve = np.asarray(data["Ve"], dtype=np.float64)
        except Exception:                                # noqa: BLE001
            continue
        if z_m.size == 0 or z_m.size != ve.size:
            continue
        meta = meta_by_id.get(cid, {})
        cfg_name = str(meta.get("name", cid))
        design_name = str(meta.get("design_name", ""))
        label = (
            f"{design_name} · {cfg_name}"
            if design_name else cfg_name
        )
        colour = _CONFIG_COLOURS[i % len(_CONFIG_COLOURS)]
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": label,
            "x": (z_m * 1.0e3).tolist(),       # m → mm
            "y": (ve * 1.0e3).tolist(),        # V → mV
            "line": {"color": colour, "width": 2},
            "hovertemplate": (
                f"<b>{label}</b><br>"
                "z = %{x:.2f} mm<br>"
                "V<sub>e</sub> = %{y:.3f} mV<extra></extra>"
            ),
        })
        plotted += 1

    if plotted == 0:
        return _placeholder(
            "None of the picked configs have a solved "
            "axis_line.npz. Run an FEM solve for them first."
        )

    layout = {
        "title": {
            "text": (
                f"V<sub>e</sub> along the cuff axis — "
                f"{plotted} config(s)"
            ),
        },
        "xaxis": {
            "title": {"text": "z along cuff axis (mm)"},
            "zeroline": True,
            "showgrid": True,
            "gridcolor": "#e8e8ec",
        },
        "yaxis": {
            "title": {"text": "V<sub>e</sub> (mV)"},
            "zeroline": True,
            "showgrid": True,
            "gridcolor": "#e8e8ec",
        },
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "y": -0.15,
            "x": 0.5,
            "xanchor": "center",
        },
        "margin": {"l": 60, "r": 20, "t": 50, "b": 60},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
    }
    return _maybe_apply_preset(
        {"data": traces, "layout": layout}, preset,
    )


def build_compare_slice_grid(
    out_dir: Path,
    cids: list[str],
    configs_meta: list[dict],
    z_idx: int,
    *,
    preset=None,
) -> dict:
    """Vₑ slice heatmap at z-index `z_idx`, one subplot per cid
    in a responsive grid. Subplots share x/y axes + a common
    colour scale so absolute magnitudes are comparable across
    configs."""
    if not cids:
        return _placeholder(
            "Pick at least one config in the Compare tab "
            "multi-select to see slice heatmaps side-by-side."
        )
    meta_by_id = {m.get("id", ""): m for m in (configs_meta or [])}
    # Pre-read every picked config's slice. Skip configs whose
    # slice_volume.npz is missing — those tiles get omitted.
    slices = []
    for cid in cids:
        d = _config_dir(out_dir, cid)
        sp = d / "slice_volume.npz"
        if not sp.is_file():
            continue
        try:
            data = np.load(sp, allow_pickle=True)
            xx = np.asarray(data["x"], dtype=np.float64)
            yy = np.asarray(data["y"], dtype=np.float64)
            zz = np.asarray(data["z"], dtype=np.float64)
            ve_vol = np.asarray(data["Ve"], dtype=np.float64)
        except Exception:                                # noqa: BLE001
            continue
        if ve_vol.ndim != 3 or ve_vol.size == 0:
            continue
        z_i = max(0, min(int(z_idx), ve_vol.shape[0] - 1))
        meta = meta_by_id.get(cid, {})
        cfg_name = str(meta.get("name", cid))
        design_name = str(meta.get("design_name", ""))
        label = (
            f"{design_name} · {cfg_name}"
            if design_name else cfg_name
        )
        slices.append({
            "cid": cid,
            "label": label,
            "x_mm": (xx * 1.0e3).tolist(),
            "y_mm": (yy * 1.0e3).tolist(),
            "z_mm": float(zz[z_i] * 1.0e3),
            "ve_mV": (ve_vol[z_i] * 1.0e3),
        })
    if not slices:
        return _placeholder(
            "None of the picked configs have a solved "
            "slice_volume.npz."
        )
    # Share a single colour scale across all subplots — pick the
    # 2nd/98th percentile across every picked slice so a single
    # outlier config doesn't blow up the range.
    all_vals = np.concatenate(
        [np.asarray(s["ve_mV"]).ravel() for s in slices],
    )
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size > 0:
        vmin = float(np.percentile(all_vals, 2.0))
        vmax = float(np.percentile(all_vals, 98.0))
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = -1.0, 1.0
    n = len(slices)
    # Grid: 2 cols × ceil(n/2) rows for compactness.
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    traces = []
    annotations = []
    for k, s in enumerate(slices):
        row = (k // cols) + 1
        col = (k % cols) + 1
        x_axis = f"x{k + 1}" if k > 0 else "x"
        y_axis = f"y{k + 1}" if k > 0 else "y"
        traces.append({
            "type": "heatmap",
            "x": s["x_mm"],
            "y": s["y_mm"],
            "z": s["ve_mV"].tolist(),
            "colorscale": "RdBu_r",
            "zmin": vmin,
            "zmax": vmax,
            "showscale": k == 0,
            "colorbar": (
                {"title": "V<sub>e</sub> (mV)"}
                if k == 0 else None
            ),
            "xaxis": x_axis,
            "yaxis": y_axis,
            "hovertemplate": (
                f"<b>{s['label']}</b><br>"
                "x = %{x:.2f} mm<br>y = %{y:.2f} mm<br>"
                "V<sub>e</sub> = %{z:.3f} mV<extra></extra>"
            ),
        })
        annotations.append({
            "text": s["label"],
            "showarrow": False,
            "xref": "paper", "yref": "paper",
            "x": (col - 0.5) / cols,
            "y": 1.0 - (row - 1) / rows,
            "xanchor": "center", "yanchor": "bottom",
            "font": {"size": 12, "color": "#1f2024"},
        })
    layout = {
        "title": {
            "text": (
                f"V<sub>e</sub> slice grid · z = "
                f"{slices[0]['z_mm']:.2f} mm · "
                f"{n} config(s)"
            ),
        },
        "annotations": annotations,
        "margin": {"l": 50, "r": 20, "t": 60, "b": 40},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "grid": {
            "rows": rows,
            "columns": cols,
            "pattern": "independent",
        },
    }
    # Per-subplot axes — equal aspect ratio so the slices aren't
    # squashed by the grid layout.
    for k in range(n):
        x_key = f"xaxis{k + 1}" if k > 0 else "xaxis"
        y_key = f"yaxis{k + 1}" if k > 0 else "yaxis"
        layout[x_key] = {
            "title": "x (mm)" if k >= n - cols else "",
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": y_key,
            "scaleratio": 1.0,
        }
        layout[y_key] = {
            "title": "y (mm)" if k % cols == 0 else "",
            "showgrid": False,
            "zeroline": False,
        }
    return _maybe_apply_preset(
        {"data": traces, "layout": layout}, preset,
    )
