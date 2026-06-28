# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""FEM-result figure builders + the matplotlib Vₑ colourbar PNG.

Mix of two output types:
  • plotly-figure dicts (consumed by trame.widgets.plotly.Figure)
  • base64 PNG data URIs (consumed by html.Img)

No live plotly / trame imports — the dicts are serialized to JSON
by the trame widget at render time, so this module can also be
called from a headless worker process.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .util import _fig_to_data_uri, _hex_to_rgba, _plotly_placeholder

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


# Per-branch ribbon palette — same colours as nerve_studio §9b/§10
# so the two tools stay visually consistent. First two cover the
# trunk / branch-1 split for vagal nerve work; remaining colours kick
# in for multi-branch dissections.
_RIBBON_BRANCH_COLOURS: tuple[str, ...] = (
    "#0d3b66", "#f95738", "#3a9d23", "#ee964b",
    "#7f5af0", "#16c172", "#e63946",
)


def _activation_function(ve_uni, s_uni, sg_window):
    """AF(s) = ∂²Vₑ/∂s² computed the SAME way as the headless render
    (scripts/render_components.py `_save_ve_curves`): a LIGHT Gaussian
    pre-smooth followed by a double `np.gradient`, instead of a wide
    Savitzky-Golay 2nd-derivative.

    The old SG filter (default window 31 ≈ 6 mm — wider than the 5.4 mm cuff)
    low-passed away the real multi-row-cuff structure: the TWIN AF peaks that
    come from the three axial contact rows (z = -1.35, 0, +1.35 mm) of the Duke
    cuff blur into a single hump. The light Gaussian (σ ≈ sg_window/6 samples)
    preserves that structure, so the UI activation function now matches the
    headless figure instead of over-smoothing.

    `ve_uni` is Vₑ in volts on the uniform arc-length grid `s_uni` (metres);
    returns AF in V/m². The outer ~2σ samples are NaN'd as a gradient edge
    guard (kept out of the population mean / outlier stats)."""
    from scipy.ndimage import gaussian_filter1d
    sigma = max(0.5, float(sg_window) / 6.0)   # window ≈ ±3σ; default 9 → σ≈1.5
    ve_sm = gaussian_filter1d(np.asarray(ve_uni, dtype=np.float64),
                              sigma, mode="nearest")
    af = np.gradient(np.gradient(ve_sm, s_uni), s_uni)
    trim = max(2, int(round(2.0 * sigma)))
    if 2 * trim < len(af):
        af[:trim] = np.nan
        af[-trim:] = np.nan
    return af


def _render_ve_colorbar_png(
    v_lo: float,
    v_hi: float,
    cmap: str = "plasma",
    label: str = "Vₑ  (mV)",
    *,
    preset: "FigureExportPreset | None" = None,
) -> str:
    """Render a thin horizontal colourbar PNG (~480×54) that
    matches the plasma cmap + clim used for the FEM Vₑ overlays
    on the nerve surface + fiber tubes. Returned as a base64
    PNG data URI for direct embedding in the viewport HTML —
    same delivery channel as every other golgi plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    if not (v_hi > v_lo):
        v_hi = v_lo + 1.0
    # Taller figure (0.65 → 0.95) so the layout has three
    # vertical bands with breathing room: title on top, the
    # colour gradient in the middle, tick labels on the bottom.
    # The earlier 0.65-inch figure put the title at y=0.97 and
    # the bar top at y=0.85 — visually close enough that the
    # "Vₑ (mV)" baseline crashed into the orange end of the
    # gradient on most screens. Splitting the bands lets each
    # one own its slice.
    fig = plt.figure(figsize=(5.2, 0.95), dpi=120)
    # Colourbar axis sits in the MIDDLE band. Title rendered
    # above (y=0.92), tick labels appear below the bar in the
    # bottom band.
    ax = fig.add_axes([0.06, 0.38, 0.88, 0.22])
    sm = ScalarMappable(
        norm=Normalize(vmin=v_lo, vmax=v_hi), cmap=cmap,
    )
    sm.set_array([])
    cb = fig.colorbar(
        sm, cax=ax, orientation="horizontal",
        extend="neither",
    )
    cb.outline.set_linewidth(0.0)
    # Dark tick + title colours so the PNG sits cleanly on the
    # solid-white viewport background.
    ax.tick_params(
        labelsize=9, colors="#1f2024", length=3,
        pad=1, labelcolor="#1f2024",
    )
    fig.text(
        0.5, 0.92, label,
        ha="center", va="top",
        fontsize=10, color="#1f2024",
        family="-apple-system",
    )
    # Transparent background so the underlying viewport gradient
    # bleeds through and the colourbar reads as an overlay
    # rather than a card.
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    return _fig_to_data_uri(fig, preset=preset)


# ---------------------------------------------------------------------------
# Interactive FEM plots — Plotly figure builders.
# ---------------------------------------------------------------------------
# These replaced the previous matplotlib PNG renderers so the §9/§10 tiles
# can pan, zoom, toggle traces, hover for values, and export PNG/SVG via
# Plotly's built-in modebar. Each builder returns a dict shaped as
# {"data": [...], "layout": {...}} that we hand straight to the
# trame.widgets.plotly.Figure widget. Empty / unavailable input yields a
# small "data not available" placeholder figure rather than failing.
_EMPTY_PLOTLY_FIG: dict = {"data": [], "layout": {}}


def _build_fem_axis_figure(
    paths_Ve: list | None,
    paths_Ez: list | None,
    paths_raw: list | None,
    branch_idx: np.ndarray | None,
    I_stim_mA: float,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Per-branch Vₑ + E_z ribbon figure (mean ± 1σ) along fiber
    arc-length s. Returns a Plotly figure dict with two stacked
    subplots and shared x-axis."""
    have = (
        paths_Ve is not None and paths_raw is not None
        and len(paths_Ve) > 0 and len(paths_raw) == len(paths_Ve)
    )
    if not have:
        return _plotly_placeholder(
            "Run a FEM solve with fiber trajectories to see "
            "V<sub>e</sub>(s) and E<sub>z</sub>(s) ribbons.",
        )
    S_RES = 200
    per_branch: dict[int, list] = {}
    for fi, path in enumerate(paths_raw):
        p = np.asarray(path, dtype=np.float64)
        if p.shape[0] < 5:
            continue
        ve = np.asarray(paths_Ve[fi], dtype=np.float64)
        if ve.shape[0] != p.shape[0]:
            continue
        ez = (np.asarray(paths_Ez[fi], dtype=np.float64)
              if (paths_Ez is not None
                  and fi < len(paths_Ez)
                  and len(paths_Ez[fi]) == p.shape[0])
              else None)
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(ds)])
        s_max = float(s_cum[-1])
        if s_max <= 0:
            continue
        finite = np.isfinite(ve)
        if finite.sum() < 5:
            continue
        s_uni = np.linspace(0.0, s_max, S_RES)
        ve_uni = np.interp(
            s_uni, s_cum[finite], ve[finite],
        ) * 1.0e3
        if ez is not None and np.isfinite(ez).sum() >= 5:
            fin_e = np.isfinite(ez)
            ez_uni = np.interp(s_uni, s_cum[fin_e], ez[fin_e])
        else:
            ez_uni = np.full(S_RES, np.nan)
        bi = (int(branch_idx[fi])
              if branch_idx is not None
              and fi < len(branch_idx) else 0)
        per_branch.setdefault(bi, []).append(
            (s_uni * 1.0e3, ve_uni, ez_uni),
        )

    if not per_branch:
        return _plotly_placeholder(
            "No fibers had enough valid V<sub>e</sub> samples."
        )

    traces: list = []
    legend_seen_in_subplot = {1: set(), 2: set()}
    for bi in sorted(per_branch.keys()):
        samples = per_branch[bi]
        colour = _RIBBON_BRANCH_COLOURS[
            bi % len(_RIBBON_BRANCH_COLOURS)
        ]
        s_max = max(s.max() for s, _, _ in samples)
        s_shared = np.linspace(0.0, s_max, S_RES)
        ve_mat = np.full((len(samples), S_RES), np.nan)
        ez_mat = np.full((len(samples), S_RES), np.nan)
        for i, (s, ve, ez) in enumerate(samples):
            mask = s_shared <= s.max()
            ve_mat[i, mask] = np.interp(s_shared[mask], s, ve)
            if np.any(np.isfinite(ez)):
                ez_mat[i, mask] = np.interp(s_shared[mask], s, ez)
        ve_mean = np.nanmean(ve_mat, axis=0)
        ve_std = np.nanstd(ve_mat, axis=0)
        ez_mean = np.nanmean(ez_mat, axis=0)
        ez_std = np.nanstd(ez_mat, axis=0)
        label = f"Branch {bi}  ({len(samples)} fibers)"
        rgba_fill = _hex_to_rgba(colour, alpha=0.20)
        # Vₑ subplot — ribbon (mean+std then mean-std reversed to
        # close the polygon for `fill='toself'`).
        x_band = np.concatenate([s_shared, s_shared[::-1]])
        y_ve_band = np.concatenate([
            ve_mean + ve_std, (ve_mean - ve_std)[::-1],
        ])
        traces.append(dict(
            type="scatter",
            x=x_band.tolist(),
            y=y_ve_band.tolist(),
            fill="toself",
            fillcolor=rgba_fill,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x", yaxis="y",
        ))
        traces.append(dict(
            type="scatter",
            x=s_shared.tolist(),
            y=ve_mean.tolist(),
            mode="lines",
            name=label,
            line=dict(color=colour, width=2.2),
            legendgroup=f"br{bi}",
            xaxis="x", yaxis="y",
            hovertemplate=(
                "s = %{x:.2f} mm<br>V<sub>e</sub> = "
                "%{y:.3f} mV<extra>" + label + "</extra>"
            ),
        ))
        # E_z subplot (only when Ez data was finite)
        if np.any(np.isfinite(ez_mat)):
            y_ez_band = np.concatenate([
                ez_mean + ez_std, (ez_mean - ez_std)[::-1],
            ])
            traces.append(dict(
                type="scatter",
                x=x_band.tolist(),
                y=y_ez_band.tolist(),
                fill="toself",
                fillcolor=rgba_fill,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                xaxis="x2", yaxis="y2",
            ))
            traces.append(dict(
                type="scatter",
                x=s_shared.tolist(),
                y=ez_mean.tolist(),
                mode="lines",
                name=label,
                line=dict(color=colour, width=2.2),
                legendgroup=f"br{bi}",
                showlegend=False,
                xaxis="x2", yaxis="y2",
                hovertemplate=(
                    "s = %{x:.2f} mm<br>E<sub>z</sub> = "
                    "%{y:.2f} V/m<extra>" + label + "</extra>"
                ),
            ))

    layout = {
        "title": {
            "text": (
                f"V<sub>e</sub> and E<sub>z</sub> along fiber "
                f"arc-length s — mean ± 1σ per branch  "
                f"(I<sub>stim</sub> = {I_stim_mA:.2f} mA)"
            ),
            "font": {"size": 12},
            "x": 0.5, "xanchor": "center",
        },
        "grid": {"rows": 2, "columns": 1, "pattern": "independent"},
        "xaxis": {
            "anchor": "y", "domain": [0.0, 1.0],
            "showgrid": True, "gridcolor": "#e5e5ea",
            "title": "", "matches": "x2",
        },
        "yaxis": {
            "anchor": "x", "domain": [0.55, 1.0],
            "title": "V<sub>e</sub>  (mV)",
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "xaxis2": {
            "anchor": "y2", "domain": [0.0, 1.0],
            "showgrid": True, "gridcolor": "#e5e5ea",
            "title": "arc-length s  (mm)",
        },
        "yaxis2": {
            "anchor": "x2", "domain": [0.0, 0.45],
            "title": "E<sub>z</sub>  (V/m)",
            "showgrid": True, "gridcolor": "#e5e5ea",
            "zeroline": True, "zerolinecolor": "#9aa0a6",
        },
        "legend": {
            "x": 1.0, "y": 1.0, "xanchor": "right",
            "yanchor": "top", "bgcolor": "rgba(255,255,255,0.80)",
            "bordercolor": "#dddddd", "borderwidth": 1,
            "font": {"size": 10},
        },
        "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
        "hovermode": "x unified",
    }
    return _maybe_apply_preset({"data": traces, "layout": layout}, preset)


def _build_fem_slice_figure(
    slice_data: dict,
    L_cuff_m: float = 0.0,
    R_ci_m: float = 0.0,
    R_co_m: float = 0.0,
    pts_cuff: np.ndarray | None = None,
    boundary_raw: np.ndarray | None = None,
    muscle_R_m: float = 0.0,
    electrode_patches: list | None = None,
    init_z_idx: int = 0,
    upsample: int = 3,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Interactive Plotly slice heatmap with a built-in z-slider.
    Each z station ships as its own (interpolated) heatmap trace
    with `visible=False`; the slider's `steps` array flips one
    trace visible at a time so scrubbing is instant client-side
    (no round-trip to the server)."""
    if slice_data is None:
        return _plotly_placeholder(
            "Run a FEM solve to see the V<sub>e</sub> slice "
            "heatmap.",
        )
    x_arr = np.asarray(slice_data["x"], dtype=np.float64)
    y_arr = np.asarray(slice_data["y"], dtype=np.float64)
    z_arr = np.asarray(slice_data["z"], dtype=np.float64)
    Ve = np.asarray(slice_data["Ve"], dtype=np.float64)
    nz = int(Ve.shape[0])
    if nz == 0 or x_arr.size < 2 or y_arr.size < 2:
        return _plotly_placeholder("Slice volume too small.")

    # Bicubic upsample each Ve slice for display so the heatmap
    # reads as a smooth field instead of a chunky grid. Scipy
    # zoom with order=3 = cubic spline; falls back to order=1
    # (bilinear) if scipy isn't available (it always is here).
    Ve_disp = Ve
    x_disp = x_arr
    y_disp = y_arr
    if upsample > 1:
        try:
            from scipy.ndimage import zoom as _zoom
            Ve_disp = _zoom(
                Ve, zoom=(1, upsample, upsample),
                order=3, mode="nearest",
            )
            x_disp = np.linspace(
                x_arr[0], x_arr[-1], Ve_disp.shape[2],
            )
            y_disp = np.linspace(
                y_arr[0], y_arr[-1], Ve_disp.shape[1],
            )
        except Exception:
            pass
    Ve_disp_mV = Ve_disp * 1.0e3

    # GLOBAL colour scale across the whole volume so colours stay
    # comparable as the user scrubs through z.
    #
    # `np.isfinite` (not `~np.isnan`) — `sample()` in solve_nerve
    # can return ±inf at facets on the active-contact boundary
    # (delta-source singularity). Letting ±inf through to
    # `abs_all.max()` collapses v_max to inf, which Plotly
    # silently rejects: layout renders (title, axes, dashed
    # outline, slider all fine) but the heatmap trace + colorbar
    # never draw. That was the "empty slice heatmap" symptom.
    abs_all = np.abs(
        Ve_disp_mV[np.isfinite(Ve_disp_mV)],
    )
    if abs_all.size and abs_all.max() > 0:
        nz_arr = abs_all[abs_all > 0.001 * abs_all.max()]
        v_max = float(
            np.percentile(nz_arr, 99.5)
            if nz_arr.size > 100 else abs_all.max()
        )
    else:
        v_max = 1.0
    v_max = max(v_max, 0.01)
    # Final defensive guard — even after the percentile clip a
    # malformed slice volume could leak inf into v_max via a
    # length-2 array path. Floor the value.
    if not np.isfinite(v_max) or v_max <= 0:
        v_max = 1.0

    # Snap the initial slider position to the slice closest to
    # z = 0 (cuff plane). For asymmetric nerve geometries the
    # state-saved `init_z_idx` (default 20) lands on a z that's
    # past the muscle bath — Ve is NaN there and the heatmap
    # cells all render transparent, which is the "empty
    # heatmap" the user kept seeing. The cuff plane is where
    # the FEM field is densest, so it's always the right place
    # to start.
    init_z_idx = int(np.argmin(np.abs(z_arr)))

    def _slice_tris_at_z(pts, tris, z0):
        z = pts[:, 2]
        tri_z = z[tris]
        sd = tri_z - z0
        edges = np.array([[0, 1], [1, 2], [2, 0]])
        sd_a = sd[:, edges[:, 0]]
        sd_b = sd[:, edges[:, 1]]
        crosses = sd_a * sd_b < 0
        t = sd_a / (sd_a - sd_b + 1e-30)
        pa = pts[tris[:, edges[:, 0]]]
        pb = pts[tris[:, edges[:, 1]]]
        intersect = pa + t[..., None] * (pb - pa)
        has_two = crosses.sum(axis=1) >= 2
        segs = []
        for ti in np.where(has_two)[0]:
            ei = np.where(crosses[ti])[0]
            if len(ei) >= 2:
                segs.append((
                    intersect[ti, ei[0], :2].tolist(),
                    intersect[ti, ei[1], :2].tolist(),
                ))
        return segs

    # ----- Pre-compute per-slice arrays (lists) -----
    # NaN values in JSON are non-standard; replace with None so
    # Plotly treats them as missing (renders the cell as
    # transparent). The 41-trace visibility-toggle pattern we
    # had before silently failed to render any heatmap; this
    # version uses ONE heatmap trace + a `restyle` slider that
    # swaps its `z` array on each step, which is Plotly's
    # documented pattern for "scrub through N images".
    def _nan_to_none(arr2d: np.ndarray) -> list:
        out = []
        for row in arr2d:
            out_row = []
            for v in row:
                out_row.append(
                    None if (
                        not np.isfinite(v)
                    ) else float(v)
                )
            out.append(out_row)
        return out

    z_slices_list = [
        _nan_to_none(Ve_disp_mV[zi]) for zi in range(nz)
    ]
    x_axis_mm = (x_disp * 1.0e3).tolist()
    y_axis_mm = (y_disp * 1.0e3).tolist()

    # Nerve cross-section outlines per z (each is a 2-list:
    # [seg_x, seg_y] in mm with None separators between segs).
    nerve_xy_per_z: list = []
    if pts_cuff is not None and boundary_raw is not None:
        _pts_arr = np.asarray(pts_cuff, dtype=np.float64)
        _tris_arr = np.asarray(boundary_raw, dtype=np.int64)
        for zi in range(nz):
            seg_x: list = []
            seg_y: list = []
            try:
                segs = _slice_tris_at_z(
                    _pts_arr, _tris_arr, float(z_arr[zi]),
                )
                for (a, b) in segs:
                    seg_x.extend([
                        a[0] * 1.0e3, b[0] * 1.0e3, None,
                    ])
                    seg_y.extend([
                        a[1] * 1.0e3, b[1] * 1.0e3, None,
                    ])
            except Exception:
                pass
            nerve_xy_per_z.append((seg_x, seg_y))
    else:
        nerve_xy_per_z = [([], []) for _ in range(nz)]

    # Exactly TWO traces. Trace 0 = heatmap; trace 1 = nerve
    # outline. Both are updated together by the slider's
    # `restyle`. The muscle outline that used to be a third
    # trace is now a layout shape (below) — keeping it out of
    # the trace list means the restyle's per-trace value-lists
    # cleanly target only the two updating traces, and we
    # avoid the silent "trace 2 got its z/x/y overwritten with
    # None and disappeared" behaviour we just hit.
    traces: list = []
    traces.append(dict(
        type="heatmap",
        z=z_slices_list[init_z_idx],
        x=x_axis_mm,
        y=y_axis_mm,
        colorscale="magma",
        zmin=-v_max, zmax=v_max,
        colorbar=dict(
            title="V<sub>e</sub>  (mV)",
            thickness=12, len=0.78,
        ),
        zsmooth="best",
        connectgaps=False,
        hoverongaps=False,
        hovertemplate=(
            "x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>"
            "V<sub>e</sub>=%{z:.3f} mV<extra></extra>"
        ),
        name="Vₑ",
        showlegend=False,
    ))
    init_seg_x, init_seg_y = nerve_xy_per_z[init_z_idx]
    traces.append(dict(
        type="scatter",
        x=init_seg_x, y=init_seg_y,
        mode="lines",
        line=dict(color="#f5f5f7", width=1.6),
        hoverinfo="skip", showlegend=False,
        name="nerve",
    ))

    # Static layout shapes (don't participate in slider restyle).
    # Muscle outer outline at every z.
    layout_shapes: list = []
    if muscle_R_m > 0:
        layout_shapes.append({
            "type": "circle",
            "xref": "x", "yref": "y",
            "x0": -muscle_R_m * 1.0e3,
            "y0": -muscle_R_m * 1.0e3,
            "x1": +muscle_R_m * 1.0e3,
            "y1": +muscle_R_m * 1.0e3,
            "line": {
                "color": "#888a90",
                "width": 1.2,
                "dash": "dash",
            },
        })

    # Slider steps: per-step `restyle` swaps trace-0 z and
    # trace-1 x/y; `relayout` updates the title.
    steps: list = []
    for zi in range(nz):
        steps.append(dict(
            method="update",
            args=[
                {
                    # Per-trace value-list. Length 2 = updates
                    # apply to traces [0, 1] in order.
                    "z": [z_slices_list[zi], None],
                    "x": [x_axis_mm, nerve_xy_per_z[zi][0]],
                    "y": [y_axis_mm, nerve_xy_per_z[zi][1]],
                },
                {
                    "title": {
                        "text": (
                            f"V<sub>e</sub> @ z = "
                            f"{z_arr[zi] * 1.0e3:+.2f} mm  "
                            f"(slice {zi + 1}/{nz})"
                        ),
                        "x": 0.5, "xanchor": "center",
                    },
                },
            ],
            label=f"{z_arr[zi] * 1.0e3:+.1f}",
        ))

    layout = {
        "title": {
            "text": (
                f"V<sub>e</sub> @ z = "
                f"{z_arr[init_z_idx] * 1.0e3:+.2f} mm  "
                f"(slice {init_z_idx + 1}/{nz})"
            ),
            "font": {"size": 12},
            "x": 0.5, "xanchor": "center",
        },
        "shapes": layout_shapes,
        "xaxis": {
            "title": "x  (mm)",
            "scaleanchor": "y", "scaleratio": 1,
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "yaxis": {
            "title": "y  (mm)",
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "sliders": [{
            "active": init_z_idx,
            "currentvalue": {
                "prefix": "z = ",
                "suffix": " mm",
                "font": {"size": 11},
            },
            "pad": {"t": 30, "b": 4},
            "len": 0.92,
            "x": 0.05, "y": -0.04,
            "steps": steps,
        }],
        "margin": {"l": 55, "r": 20, "t": 50, "b": 95},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
    }
    return _maybe_apply_preset({"data": traces, "layout": layout}, preset)



def _build_fem_af_figure(
    paths_Ve: list | None,
    paths_raw: list | None,
    branch_idx: np.ndarray | None,
    sel_fiber: int = 0,
    sg_window: int = 31,
    *,
    preset: "FigureExportPreset | None" = None,
) -> dict:
    """Plotly version of the §10 activation-function plot.
    Two stacked subplots (Vₑ and AF), per-branch ribbons, the
    selected fiber overlaid as a dashed black highlight."""
    have = (
        paths_Ve is not None and paths_raw is not None
        and len(paths_Ve) > 0 and len(paths_raw) == len(paths_Ve)
    )
    if not have:
        return _plotly_placeholder(
            "Run a FEM solve with fiber trajectories to see "
            "the activation function.",
        )
    per_fiber: list[dict] = []
    for fi, path in enumerate(paths_raw):
        p = np.asarray(path, dtype=np.float64)
        if p.shape[0] < 5:
            continue
        ve = np.asarray(paths_Ve[fi], dtype=np.float64)
        if ve.shape[0] != p.shape[0]:
            continue
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        finite = np.isfinite(ve)
        if finite.sum() < 5:
            continue
        s_f = s[finite]
        ve_f = ve[finite]
        if s_f[-1] <= 0:
            continue
        s_uni = np.linspace(s_f[0], s_f[-1], len(s_f))
        ve_uni = np.interp(s_uni, s_f, ve_f)
        af = _activation_function(ve_uni, s_uni, sg_window)
        bi = (int(branch_idx[fi])
              if branch_idx is not None
              and fi < len(branch_idx) else 0)
        per_fiber.append({
            "i": fi, "branch": bi,
            "s": s_uni, "Ve": ve_uni, "AF": af,
        })

    if not per_fiber:
        return _plotly_placeholder(
            "No fibers had enough valid V<sub>e</sub> samples.",
        )

    # Short-fiber filter
    if len(per_fiber) >= 5:
        _spans = np.array([
            float(r["s"][-1] - r["s"][0]) for r in per_fiber
        ])
        _med_span = float(np.median(_spans))
        _thr = 0.5 * _med_span
        per_fiber = [
            r for r, sp in zip(per_fiber, _spans)
            if sp >= _thr
        ]
    # Outlier filter (Tukey IQR)
    if len(per_fiber) >= 5:
        peaks = np.array([
            float(np.nanmax(np.abs(r["AF"]))) for r in per_fiber
        ])
        q1, q3 = (float(p)
                  for p in np.percentile(peaks, [25.0, 75.0]))
        iqr = max(q3 - q1, 1e-12)
        thr = q3 + 3.0 * iqr
        per_fiber = [
            r for r, pk in zip(per_fiber, peaks)
            if pk <= thr and np.isfinite(pk)
        ]
    if not per_fiber:
        return _plotly_placeholder(
            "Short-fiber + outlier filters dropped every fiber.",
        )

    S_RES = 300
    per_branch: dict[int, list] = {}
    for r in per_fiber:
        per_branch.setdefault(r["branch"], []).append(r)
    traces: list = []
    for bi in sorted(per_branch.keys()):
        samples = per_branch[bi]
        colour = _RIBBON_BRANCH_COLOURS[
            bi % len(_RIBBON_BRANCH_COLOURS)
        ]
        rgba_fill = _hex_to_rgba(colour, alpha=0.20)
        s_max = max(
            float((r["s"] - r["s"].min()).max())
            for r in samples
        )
        s_shared = np.linspace(0.0, s_max, S_RES)
        ve_mat = np.full((len(samples), S_RES), np.nan)
        af_mat = np.full((len(samples), S_RES), np.nan)
        for i, r in enumerate(samples):
            s_loc = r["s"] - r["s"].min()
            mask = s_shared <= s_loc.max()
            ve_mat[i, mask] = np.interp(
                s_shared[mask], s_loc, r["Ve"] * 1.0e3,
            )
            af_mat[i, mask] = np.interp(
                s_shared[mask], s_loc, r["AF"],
            )
        ve_mean = np.nanmean(ve_mat, axis=0)
        ve_std = np.nanstd(ve_mat, axis=0)
        af_mean = np.nanmean(af_mat, axis=0)
        af_std = np.nanstd(af_mat, axis=0)
        s_mm = s_shared * 1.0e3
        label = f"Branch {bi}  ({len(samples)} fibers)"
        x_band = np.concatenate([s_mm, s_mm[::-1]])
        # Vₑ ribbon
        traces.append(dict(
            type="scatter",
            x=x_band.tolist(),
            y=np.concatenate([
                ve_mean + ve_std, (ve_mean - ve_std)[::-1],
            ]).tolist(),
            fill="toself", fillcolor=rgba_fill,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
            xaxis="x", yaxis="y",
        ))
        traces.append(dict(
            type="scatter",
            x=s_mm.tolist(),
            y=ve_mean.tolist(),
            mode="lines", name=label,
            line=dict(color=colour, width=2.2),
            legendgroup=f"br{bi}",
            xaxis="x", yaxis="y",
            hovertemplate=(
                "s=%{x:.2f} mm<br>V<sub>e</sub>=%{y:.3f} mV"
                "<extra>" + label + "</extra>"
            ),
        ))
        # AF ribbon
        traces.append(dict(
            type="scatter",
            x=x_band.tolist(),
            y=np.concatenate([
                af_mean + af_std, (af_mean - af_std)[::-1],
            ]).tolist(),
            fill="toself", fillcolor=rgba_fill,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
            xaxis="x2", yaxis="y2",
        ))
        traces.append(dict(
            type="scatter",
            x=s_mm.tolist(),
            y=af_mean.tolist(),
            mode="lines", name=label,
            line=dict(color=colour, width=2.2),
            legendgroup=f"br{bi}",
            showlegend=False,
            xaxis="x2", yaxis="y2",
            hovertemplate=(
                "s=%{x:.2f} mm<br>AF=%{y:.1f} V/m²"
                "<extra>" + label + "</extra>"
            ),
        ))
    # Selected fiber overlay
    sel_i = int(np.clip(sel_fiber, 0, max(len(per_fiber) - 1, 0)))
    sel = per_fiber[sel_i]
    s_sel = (sel["s"] - sel["s"].min()) * 1.0e3
    traces.append(dict(
        type="scatter",
        x=s_sel.tolist(),
        y=(sel["Ve"] * 1.0e3).tolist(),
        mode="lines",
        line=dict(color="black", width=1.4, dash="dash"),
        name=f"selected (fiber {sel['i']})",
        legendgroup="selected",
        xaxis="x", yaxis="y",
    ))
    traces.append(dict(
        type="scatter",
        x=s_sel.tolist(),
        y=sel["AF"].tolist(),
        mode="lines",
        line=dict(color="black", width=1.4, dash="dash"),
        name=f"selected (fiber {sel['i']})",
        legendgroup="selected",
        showlegend=False,
        xaxis="x2", yaxis="y2",
    ))

    layout = {
        "title": {
            "text": (
                f"V<sub>e</sub> and activation function — "
                f"mean ± 1σ across {len(per_fiber)} fibers, "
                f"per branch"
            ),
            "font": {"size": 12},
            "x": 0.5, "xanchor": "center",
        },
        "grid": {"rows": 2, "columns": 1, "pattern": "independent"},
        "xaxis": {
            "anchor": "y", "domain": [0.0, 1.0],
            "showgrid": True, "gridcolor": "#e5e5ea",
            "matches": "x2",
        },
        "yaxis": {
            "anchor": "x", "domain": [0.55, 1.0],
            "title": "V<sub>e</sub>  (mV)",
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "xaxis2": {
            "anchor": "y2", "domain": [0.0, 1.0],
            "showgrid": True, "gridcolor": "#e5e5ea",
            "title": "arc-length s  (mm)",
        },
        "yaxis2": {
            "anchor": "x2", "domain": [0.0, 0.45],
            "title": "AF = ∂²V<sub>e</sub>/∂s²  (V/m²)",
            "showgrid": True, "gridcolor": "#e5e5ea",
            "zeroline": True, "zerolinecolor": "#9aa0a6",
        },
        "legend": {
            "x": 1.0, "y": 1.0, "xanchor": "right",
            "yanchor": "top", "bgcolor": "rgba(255,255,255,0.80)",
            "bordercolor": "#dddddd", "borderwidth": 1,
            "font": {"size": 10},
        },
        "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
        "hovermode": "x unified",
    }
    return _maybe_apply_preset({"data": traces, "layout": layout}, preset)


def _render_fem_axis_plot(
    paths_Ve: list | None,
    paths_Ez: list | None,
    paths_raw: list | None,
    branch_idx: np.ndarray | None,
    I_stim_mA: float,
    axis_fallback: dict | None = None,
    *,
    preset: "FigureExportPreset | None" = None,
) -> str:
    """Per-branch ribbon plot of Vₑ(s) and E_z(s) along fiber
    arc-length s. Mirrors nerve_studio §9b: each fiber is
    resampled onto a shared s-grid per branch, then mean ± 1σ
    is drawn. Falls back to the legacy single-axis Vₑ(z) /
    E_z(z) line plot when per-fiber data is missing (e.g. FEM
    solved without trajectories)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have_fibers = (
        paths_Ve is not None and paths_raw is not None
        and len(paths_Ve) > 0 and len(paths_raw) == len(paths_Ve)
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(9.5, 5.4), dpi=110, sharex=True,
    )

    if have_fibers:
        # Per-fiber resample onto a uniform arc-length grid so
        # means align horizontally inside each branch. We use
        # 200 stations per fiber (same as §9b) — enough to keep
        # the mean curve smooth without bloating the PNG.
        S_RES = 200
        per_branch: dict[int, list] = {}
        for fi, path in enumerate(paths_raw):
            p = np.asarray(path, dtype=np.float64)
            if p.shape[0] < 5:
                continue
            ve = np.asarray(paths_Ve[fi], dtype=np.float64)
            if ve.shape[0] != p.shape[0]:
                continue
            ez = (np.asarray(paths_Ez[fi], dtype=np.float64)
                  if (paths_Ez is not None
                      and fi < len(paths_Ez)
                      and len(paths_Ez[fi]) == p.shape[0])
                  else None)
            ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
            s_cum = np.concatenate([[0.0], np.cumsum(ds)])
            s_max = float(s_cum[-1])
            if s_max <= 0:
                continue
            finite = np.isfinite(ve)
            if finite.sum() < 5:
                continue
            s_uni = np.linspace(0.0, s_max, S_RES)
            ve_uni = np.interp(
                s_uni, s_cum[finite], ve[finite],
            ) * 1.0e3  # V → mV
            if ez is not None:
                fin_e = np.isfinite(ez)
                if fin_e.sum() >= 5:
                    ez_uni = np.interp(
                        s_uni, s_cum[fin_e], ez[fin_e],
                    )
                else:
                    ez_uni = np.full(S_RES, np.nan)
            else:
                ez_uni = np.full(S_RES, np.nan)
            bi = (int(branch_idx[fi])
                  if branch_idx is not None
                  and fi < len(branch_idx) else 0)
            per_branch.setdefault(bi, []).append(
                (s_uni * 1.0e3, ve_uni, ez_uni),
            )

        for bi in sorted(per_branch.keys()):
            samples = per_branch[bi]
            colour = _RIBBON_BRANCH_COLOURS[
                bi % len(_RIBBON_BRANCH_COLOURS)
            ]
            s_max = max(s.max() for s, _, _ in samples)
            s_shared = np.linspace(0.0, s_max, S_RES)
            ve_mat = np.full((len(samples), S_RES), np.nan)
            ez_mat = np.full((len(samples), S_RES), np.nan)
            for i, (s, ve, ez) in enumerate(samples):
                mask = s_shared <= s.max()
                ve_mat[i, mask] = np.interp(
                    s_shared[mask], s, ve,
                )
                if np.any(np.isfinite(ez)):
                    ez_mat[i, mask] = np.interp(
                        s_shared[mask], s, ez,
                    )
            ve_mean = np.nanmean(ve_mat, axis=0)
            ve_std = np.nanstd(ve_mat, axis=0)
            ez_mean = np.nanmean(ez_mat, axis=0)
            ez_std = np.nanstd(ez_mat, axis=0)
            label = f"Branch {bi}  ({len(samples)} fibers)"
            axes[0].fill_between(
                s_shared, ve_mean - ve_std, ve_mean + ve_std,
                color=colour, alpha=0.20, linewidth=0,
            )
            axes[0].plot(
                s_shared, ve_mean, color=colour, lw=2.0,
                label=label,
            )
            if np.any(np.isfinite(ez_mat)):
                axes[1].fill_between(
                    s_shared, ez_mean - ez_std,
                    ez_mean + ez_std,
                    color=colour, alpha=0.20, linewidth=0,
                )
                axes[1].plot(
                    s_shared, ez_mean, color=colour, lw=2.0,
                    label=label,
                )

        axes[0].set_ylabel("V$_e$  (mV)", fontsize=10)
        axes[0].set_title(
            f"V$_e$ and E$_z$ along fiber arc-length s "
            f"— mean ± 1σ per branch  (I$_{{stim}}$ = "
            f"{I_stim_mA:.2f} mA)",
            fontsize=11,
        )
        axes[0].grid(True, alpha=0.30)
        axes[0].legend(loc="best", fontsize=9, framealpha=0.85)
        axes[1].set_xlabel("arc-length s  (mm)", fontsize=10)
        axes[1].set_ylabel("E$_z$  (V/m)", fontsize=10)
        axes[1].axhline(0, color="0.3", lw=0.6)
        axes[1].grid(True, alpha=0.30)
        axes[1].legend(loc="best", fontsize=9, framealpha=0.85)
        # Disable matplotlib's auto-offset Y-axis notation —
        # otherwise a nearly-constant Vₑ renders as
        # "1e-5 - 5.05046e1" which is unreadable.
        for _ax in axes:
            try:
                _ax.ticklabel_format(
                    axis="y", useOffset=False, style="plain",
                )
            except Exception:
                pass
    else:
        # Fallback path — no fiber trajectories yet. Plot Vₑ(z)
        # and E_z(z) along the cuff centerline so the user can
        # at least sanity-check the FEM solve.
        axis = axis_fallback or {}
        if not axis or "z" not in axis:
            for ax in axes:
                ax.text(
                    0.5, 0.5,
                    "No FEM axis data — run a solve first.",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    color="#888a90", fontsize=11,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            fig.tight_layout(pad=0.6)
            return _fig_to_data_uri(fig, preset=preset)
        z_mm = np.asarray(axis["z"]) * 1.0e3
        ve_mv = np.asarray(axis["Ve"]) * 1.0e3
        ez_vpm = np.asarray(axis["Ez"])
        has_elec = (
            "elec_z"
            in (axis.files if hasattr(axis, "files") else axis)
        )
        elec_z_mm = (
            np.asarray(axis["elec_z"]) * 1.0e3 if has_elec
            else np.array([])
        )
        axes[0].plot(z_mm, ve_mv, color="#0d3b66", lw=1.6)
        for ez in elec_z_mm:
            axes[0].axvline(
                ez, color="#cc4778", lw=0.8, alpha=0.6,
            )
        axes[0].set_ylabel("V$_e$  (mV)", fontsize=10)
        axes[0].set_title(
            f"FEM extracellular potential — cuff centerline  "
            f"(I$_{{stim}}$ = {I_stim_mA:.2f} mA)  "
            f"[run fibers for per-branch ribbons]",
            fontsize=11,
        )
        axes[0].grid(True, alpha=0.30)
        axes[1].plot(z_mm, ez_vpm, color="#f95738", lw=1.6)
        for ez in elec_z_mm:
            axes[1].axvline(
                ez, color="#cc4778", lw=0.8, alpha=0.6,
            )
        axes[1].axhline(0, color="0.3", lw=0.6)
        axes[1].set_xlabel("z  (mm)", fontsize=10)
        axes[1].set_ylabel("E$_z$  (V/m)", fontsize=10)
        axes[1].grid(True, alpha=0.30)

    fig.tight_layout(pad=0.6)
    return _fig_to_data_uri(fig, preset=preset)


def _slice_tris_at_z(pts_m: np.ndarray,
                        tris: np.ndarray,
                        z0_m: float) -> list:
    """Fully-vectorised triangle/plane slicing for the nerve
    cross-section overlay. Returns a list of [(x1,y1),(x2,y2)]
    line segments (in metres) where the plane z = z0 cuts the
    triangulation. Ported verbatim from nerve_studio §9."""
    z = pts_m[:, 2]
    tri_z = z[tris]
    sd = tri_z - z0_m
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    sd_a = sd[:, edges[:, 0]]
    sd_b = sd[:, edges[:, 1]]
    crosses = sd_a * sd_b < 0
    t = sd_a / (sd_a - sd_b + 1e-30)
    pa = pts_m[tris[:, edges[:, 0]]]
    pb = pts_m[tris[:, edges[:, 1]]]
    intersect = pa + t[..., None] * (pb - pa)
    has_two = crosses.sum(axis=1) >= 2
    segs: list = []
    for ti in np.where(has_two)[0]:
        ei = np.where(crosses[ti])[0]
        if len(ei) >= 2:
            segs.append((
                intersect[ti, ei[0], :2].tolist(),
                intersect[ti, ei[1], :2].tolist(),
            ))
    return segs


def _render_fem_slice_plot(slice_data: dict,
                              z_idx: int,
                              L_cuff_m: float = 0.0,
                              R_ci_m: float = 0.0,
                              R_co_m: float = 0.0,
                              pts_cuff: np.ndarray | None = None,
                              boundary_raw: np.ndarray | None = None,
                              muscle_R_m: float = 0.0,
                              electrode_patches: list | None = None,
                              field: str = "Ve",
                              *,
                              preset: "FigureExportPreset | None" = None,
                              ) -> str:
    """Two-panel Vₑ | |E|+E_{xy} quiver heatmap at the chosen z
    station, with the nerve cross-section + cuff inner/outer
    rings + electrode arc markers overlaid in each panel.
    Mirrors nerve_studio §9. `field` is kept for API compat but
    no longer changes the layout — both panels are always drawn
    so the user gets the full §9 view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    x_arr = np.asarray(slice_data["x"])
    y_arr = np.asarray(slice_data["y"])
    z_arr = np.asarray(slice_data["z"])
    z_idx = int(np.clip(z_idx, 0, len(z_arr) - 1))
    x_mm = x_arr * 1.0e3
    y_mm = y_arr * 1.0e3
    X, Y = np.meshgrid(x_mm, y_mm, indexing="xy")
    Ve_mV = np.asarray(slice_data["Ve"])[z_idx] * 1.0e3
    Ex = np.asarray(slice_data["Ex"])[z_idx]
    Ey = np.asarray(slice_data["Ey"])[z_idx]
    Ez = np.asarray(slice_data["Ez"])[z_idx]
    E_mag = np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
    zc_m = float(z_arr[z_idx])
    zc_mm = zc_m * 1.0e3

    # Global colour limits across the whole slice volume so the
    # scale stays comparable when the user scrubs the z slider.
    # The bulk muscle far from the cuff sits near zero, so we
    # mask everything below 0.1 % of |max| before percentile to
    # avoid the colour scale being dragged by that zero-bulk.
    Ve_all = np.asarray(slice_data["Ve"]) * 1.0e3
    abs_all = np.abs(Ve_all[~np.isnan(Ve_all)])
    if abs_all.size > 0 and abs_all.max() > 0:
        nz = abs_all[abs_all > 0.001 * abs_all.max()]
        v_max = float(np.percentile(nz, 99.5)
                       if nz.size > 100 else abs_all.max())
    else:
        v_max = 1.0
    v_max = max(v_max, 0.01)

    Ex_all = np.asarray(slice_data["Ex"])
    Ey_all = np.asarray(slice_data["Ey"])
    Ez_all = np.asarray(slice_data["Ez"])
    E_all = np.sqrt(Ex_all ** 2 + Ey_all ** 2 + Ez_all ** 2)
    E_flat = E_all[~np.isnan(E_all)]
    if E_flat.size > 0 and E_flat.max() > 0:
        nz = E_flat[E_flat > 0.001 * E_flat.max()]
        e_max = float(np.percentile(nz, 99.5)
                       if nz.size > 100 else E_flat.max())
    else:
        e_max = 1.0
    e_max = max(e_max, 1.0e-6)

    # Nerve cross-section at this z (if geometry was supplied).
    nerve_segs_mm: list = []
    if pts_cuff is not None and boundary_raw is not None:
        try:
            segs_m = _slice_tris_at_z(
                np.asarray(pts_cuff, dtype=np.float64),
                np.asarray(boundary_raw, dtype=np.int64),
                zc_m,
            )
            nerve_segs_mm = [
                ((a[0] * 1.0e3, a[1] * 1.0e3),
                 (b[0] * 1.0e3, b[1] * 1.0e3))
                for a, b in segs_m
            ]
        except Exception:
            nerve_segs_mm = []

    def _draw_overlays(ax, line_colour: str) -> None:
        if nerve_segs_mm:
            ax.add_collection(LineCollection(
                nerve_segs_mm,
                colors=line_colour, linewidths=0.8, zorder=10,
            ))
        # Cuff rings (only when the slice is inside the cuff
        # axial window — otherwise drawing them would be a lie).
        if L_cuff_m > 0 and abs(zc_m) < L_cuff_m / 2:
            phi = np.linspace(0.0, 2.0 * np.pi, 128)
            for R_m in (R_ci_m, R_co_m):
                if R_m <= 0:
                    continue
                ax.plot(
                    R_m * np.cos(phi) * 1.0e3,
                    R_m * np.sin(phi) * 1.0e3,
                    color=line_colour, lw=0.8, zorder=10,
                )
            # Electrode patch arcs at r = R_ci_m, but only if
            # this z station lies inside the patch's axial span.
            for p in (electrode_patches or []):
                if p.get("type") != "axial":
                    continue
                p_z = float(p.get("z", 0.0))
                p_dz = float(p.get("dz", 0.0))
                if not (p_z - p_dz / 2
                        <= zc_m <= p_z + p_dz / 2):
                    continue
                p_phi = float(p.get("phi", 0.0))
                p_dphi = float(p.get("dphi", 2 * np.pi))
                arc = np.linspace(
                    p_phi - p_dphi / 2,
                    p_phi + p_dphi / 2, 48,
                )
                col = ("#e74c3c"
                       if p.get("role") == "active"
                       else "#3498db")
                ax.plot(
                    R_ci_m * np.cos(arc) * 1.0e3,
                    R_ci_m * np.sin(arc) * 1.0e3,
                    color=col, lw=3.0, zorder=11,
                )
        # Muscle outer cylinder (always shown, dashed).
        if muscle_R_m > 0:
            phi = np.linspace(0.0, 2.0 * np.pi, 128)
            ax.plot(
                muscle_R_m * np.cos(phi) * 1.0e3,
                muscle_R_m * np.sin(phi) * 1.0e3,
                color=line_colour, lw=0.8,
                linestyle="--", zorder=10,
            )

    fig, axs = plt.subplots(
        1, 2, figsize=(11.0, 4.6), dpi=110,
        constrained_layout=True,
    )

    pcm0 = axs[0].pcolormesh(
        X, Y, Ve_mV, cmap="RdBu_r",
        vmin=-v_max, vmax=v_max, shading="auto",
    )
    _draw_overlays(axs[0], line_colour="black")
    axs[0].set_aspect("equal")
    axs[0].set_title(
        f"V$_e$  @  z = {zc_mm:+.2f} mm "
        f"(slice {z_idx + 1}/{len(z_arr)})",
        fontsize=11,
    )
    axs[0].set_xlabel("x  (mm)", fontsize=10)
    axs[0].set_ylabel("y  (mm)", fontsize=10)
    axs[0].tick_params(labelsize=9)
    fig.colorbar(
        pcm0, ax=axs[0], label="V$_e$  (mV)",
        extend="both", shrink=0.85,
    )

    pcm1 = axs[1].pcolormesh(
        X, Y, E_mag, cmap="viridis",
        vmin=0.0, vmax=e_max, shading="auto",
    )
    _draw_overlays(axs[1], line_colour="white")
    # In-plane E quiver — ~18 arrows across the wider axis,
    # white so they stand out against the dark viridis low end.
    step = max(1, X.shape[0] // 18)
    axs[1].quiver(
        X[::step, ::step], Y[::step, ::step],
        Ex[::step, ::step], Ey[::step, ::step],
        color="white", width=0.004, alpha=0.7,
    )
    axs[1].set_aspect("equal")
    axs[1].set_title(
        f"|E| + E$_{{xy}}$  @  z = {zc_mm:+.2f} mm",
        fontsize=11,
    )
    axs[1].set_xlabel("x  (mm)", fontsize=10)
    axs[1].set_ylabel("y  (mm)", fontsize=10)
    axs[1].tick_params(labelsize=9)
    fig.colorbar(
        pcm1, ax=axs[1], label="|E|  (V/m)",
        extend="max", shrink=0.85,
    )
    return _fig_to_data_uri(fig, preset=preset)


def _render_fem_af_plot(
    paths_Ve: list | None,
    paths_raw: list | None,
    branch_idx: np.ndarray | None,
    sel_fiber: int = 0,
    sg_window: int = 31,
    *,
    preset: "FigureExportPreset | None" = None,
) -> str:
    """§10 activation function plot. Sample Vₑ on each fiber,
    compute AF(s)=∂²Vₑ/∂s² via a light Gaussian smooth + double
    gradient (see _activation_function — matches the headless
    figure), draw Vₑ(s) and AF(s) as mean ± 1σ ribbons per branch
    with the selected fiber overlaid as a dashed black highlight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 1, figsize=(9.5, 5.4), dpi=110, sharex=True,
    )

    have = (
        paths_Ve is not None and paths_raw is not None
        and len(paths_Ve) > 0 and len(paths_raw) == len(paths_Ve)
    )
    if not have:
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "Run a FEM solve with fiber trajectories to see "
                "the activation function.",
                ha="center", va="center",
                transform=ax.transAxes,
                color="#888a90", fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.tight_layout(pad=0.6)
        return _fig_to_data_uri(fig, preset=preset)

    # Per-fiber: arc-length, Vₑ in mV, AF in V/m². AF uses the
    # same light-Gaussian + double-gradient scheme as the headless
    # figure (see _activation_function); its outer ~2σ edge samples
    # are NaN'd to stop boundary artefacts polluting the mean.
    per_fiber: list[dict] = []
    for fi, path in enumerate(paths_raw):
        p = np.asarray(path, dtype=np.float64)
        if p.shape[0] < 5:
            continue
        ve = np.asarray(paths_Ve[fi], dtype=np.float64)
        if ve.shape[0] != p.shape[0]:
            continue
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        finite = np.isfinite(ve)
        if finite.sum() < 5:
            continue
        s_f = s[finite]
        ve_f = ve[finite]
        if s_f[-1] <= 0:
            continue
        s_uni = np.linspace(s_f[0], s_f[-1], len(s_f))
        ve_uni = np.interp(s_uni, s_f, ve_f)
        af = _activation_function(ve_uni, s_uni, sg_window)
        bi = (int(branch_idx[fi])
              if branch_idx is not None
              and fi < len(branch_idx) else 0)
        per_fiber.append({
            "i": fi, "branch": bi,
            "s": s_uni, "Ve": ve_uni, "AF": af,
        })

    if not per_fiber:
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "No fibers had enough valid Vₑ samples for the "
                "activation function.",
                ha="center", va="center",
                transform=ax.transAxes,
                color="#888a90", fontsize=11,
            )
        fig.tight_layout(pad=0.6)
        return _fig_to_data_uri(fig, preset=preset)

    # Short-fiber filter (matches nerve_studio §10): drop any
    # fiber whose arc-length span is less than half the median.
    # Without this, a handful of very short branch streamlines
    # produce step-shaped means as they drop out of the
    # population window, which is what the user was seeing on
    # the Branch 0 (5 fibers) curve.
    if len(per_fiber) >= 5:
        _spans = np.array([
            float(r["s"][-1] - r["s"][0]) for r in per_fiber
        ])
        _med_span = float(np.median(_spans))
        _span_thr = 0.5 * _med_span
        per_fiber = [
            r for r, sp in zip(per_fiber, _spans)
            if sp >= _span_thr
        ]
        if not per_fiber:
            for ax in axes:
                ax.text(
                    0.5, 0.5,
                    "Short-fiber filter dropped every fiber — "
                    "check fiber generation.",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    color="#888a90", fontsize=11,
                )
            fig.tight_layout(pad=0.6)
            return _fig_to_data_uri(fig, preset=preset)

    # Outlier filter (Tukey IQR rule on |AF|_max).
    if len(per_fiber) >= 5:
        peaks = np.array([
            float(np.nanmax(np.abs(r["AF"]))) for r in per_fiber
        ])
        q1, q3 = (float(p)
                  for p in np.percentile(peaks, [25.0, 75.0]))
        iqr = max(q3 - q1, 1e-12)
        thr = q3 + 3.0 * iqr
        per_fiber = [
            r for r, pk in zip(per_fiber, peaks)
            if pk <= thr and np.isfinite(pk)
        ]
    if not per_fiber:
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "All fiber AFs were outliers — check the FEM "
                "solve / mesh quality.",
                ha="center", va="center",
                transform=ax.transAxes,
                color="#888a90", fontsize=11,
            )
        fig.tight_layout(pad=0.6)
        return _fig_to_data_uri(fig, preset=preset)

    # Per-branch ribbons on a shared s grid.
    S_RES = 300
    per_branch: dict[int, list] = {}
    for r in per_fiber:
        per_branch.setdefault(r["branch"], []).append(r)
    for bi in sorted(per_branch.keys()):
        samples = per_branch[bi]
        colour = _RIBBON_BRANCH_COLOURS[
            bi % len(_RIBBON_BRANCH_COLOURS)
        ]
        s_max = max(
            float((r["s"] - r["s"].min()).max()) for r in samples
        )
        s_shared = np.linspace(0.0, s_max, S_RES)
        ve_mat = np.full((len(samples), S_RES), np.nan)
        af_mat = np.full((len(samples), S_RES), np.nan)
        for i, r in enumerate(samples):
            s_loc = r["s"] - r["s"].min()
            mask = s_shared <= s_loc.max()
            ve_mat[i, mask] = np.interp(
                s_shared[mask], s_loc, r["Ve"] * 1.0e3,
            )
            af_mat[i, mask] = np.interp(
                s_shared[mask], s_loc, r["AF"],
            )
        ve_mean = np.nanmean(ve_mat, axis=0)
        ve_std = np.nanstd(ve_mat, axis=0)
        af_mean = np.nanmean(af_mat, axis=0)
        af_std = np.nanstd(af_mat, axis=0)
        s_mm = s_shared * 1.0e3
        label = f"Branch {bi}  ({len(samples)} fibers)"
        axes[0].fill_between(
            s_mm, ve_mean - ve_std, ve_mean + ve_std,
            color=colour, alpha=0.20, linewidth=0,
        )
        axes[0].plot(
            s_mm, ve_mean, color=colour, lw=2.0, label=label,
        )
        axes[1].fill_between(
            s_mm, af_mean - af_std, af_mean + af_std,
            color=colour, alpha=0.20, linewidth=0,
        )
        axes[1].plot(
            s_mm, af_mean, color=colour, lw=2.0, label=label,
        )

    # Selected fiber overlay — sel_fiber indexes into per_fiber
    # (the survivors after the outlier filter), clamped to range.
    sel_i = int(np.clip(sel_fiber, 0, max(len(per_fiber) - 1, 0)))
    sel = per_fiber[sel_i]
    s_sel_mm = (sel["s"] - sel["s"].min()) * 1.0e3
    axes[0].plot(
        s_sel_mm, sel["Ve"] * 1.0e3,
        color="black", lw=1.3, alpha=0.85, linestyle="--",
        label=f"selected (fiber {sel['i']})",
    )
    axes[1].plot(
        s_sel_mm, sel["AF"],
        color="black", lw=1.3, alpha=0.85, linestyle="--",
        label=f"selected (fiber {sel['i']})",
    )

    axes[0].set_ylabel("V$_e$  (mV)", fontsize=10)
    axes[0].set_title(
        f"V$_e$ and activation function along fiber arc-length "
        f"s — mean ± 1σ across {len(per_fiber)} fibers, "
        f"per branch",
        fontsize=11,
    )
    axes[0].grid(True, alpha=0.30)
    axes[0].legend(loc="best", fontsize=8, framealpha=0.85, ncol=2)
    axes[1].axhline(0, color="0.3", lw=0.6)
    axes[1].set_xlabel("arc-length s  (mm)", fontsize=10)
    axes[1].set_ylabel(
        r"AF = $\partial^2 V_e/\partial s^2$  (V/m$^2$)",
        fontsize=10,
    )
    axes[1].grid(True, alpha=0.30)
    axes[1].legend(loc="best", fontsize=8, framealpha=0.85, ncol=2)
    # Suppress matplotlib's automatic offset notation
    # ("1e-5 - 5.05046e1" etc.) so the user reads absolute Vₑ /
    # AF values directly off the axis.
    for _ax in axes:
        try:
            _ax.ticklabel_format(
                axis="y", useOffset=False, style="plain",
            )
        except Exception:
            pass
    fig.tight_layout(pad=0.6)
    return _fig_to_data_uri(fig, preset=preset)
