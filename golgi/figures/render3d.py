# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Off-screen 3D capture for Generate Report + bulk Exports
(F2.3.c Phase 2).

Phase 1 originally shipped only `capture_live_viewport(plotter)` —
a single screenshot reused for every "3D" report section. That
gave the user the same picture on every page, which defeats the
point of having multiple variants in the FEATURES.md spec.

This module replaces the single-snapshot path with a real off-
screen render pipeline:

  * `_build_offscreen_plotter()` opens a fresh `pv.Plotter` with
    `off_screen=True`, white background, anti-aliasing on. The
    plotter is unrelated to the workspace `pl` — toggling actors
    on it never touches the user's view.
  * Tiny actor adders (`_add_region`, `_add_fibers`, `_add_cuff`)
    take the existing `geom.region_surfaces_viz` polydata + the
    `region_defaults` style dict (passed in from build_app so
    this module stays free of app.py imports) and add the
    relevant pieces to an offscreen plotter.
  * `_frame_camera(plotter, bbox, view)` sets camera presets:
    `iso` (default; az=35, el=20 framed to bbox), `cuff_zoom`
    (same iso but framed to the cuff bbox), `cross_section`
    (xz view through the cuff centre).
  * `render_variant(variant_id, geom, *, region_defaults, …)`
    builds the offscreen plotter, adds the actors that the
    variant config asks for, frames the camera, and returns
    PNG bytes.

The variant catalogue (`VARIANTS`) matches the F2.3 spec — twelve
visibility combinations across Electrode / Mesh / Fibers / FEM
sections, plus the two zoom variants requested in the most recent
feedback (full-nerve overview + cuff-zoom iso)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Plotter / actor primitives
# ---------------------------------------------------------------------------


def _build_offscreen_plotter(size=(1280, 960), bg="white"):
    """Fresh off-screen PyVista plotter that mirrors the live
    workspace plotter's lighting + AA setup, so the renders that
    land in the report read as the same scene the user sees in
    the app.

    We delegate to `golgi.scene.renderer.build_plotter` (the same
    factory that builds the live `pl` + the cuff-designer's
    `pl_cuff`) so the 3-point cinematic lighting + SSAA + white
    background match exactly. Window size is overridden so PDF
    embeds can be tuned independently of the live viewport's
    1400×900."""
    try:
        from golgi.scene.renderer import build_plotter
        p = build_plotter(bg_color=bg)
        # build_plotter hard-codes 1400×900; resize for our use.
        try:
            p.window_size = tuple(size)
        except Exception:                                # noqa: BLE001
            pass
        return p
    except Exception as ex:                              # noqa: BLE001
        # Fallback path — keeps the module importable even when
        # the scene renderer isn't available (e.g. unit tests).
        print(
            f"[render3d] build_plotter unavailable, falling "
            f"back to defaults: {type(ex).__name__}: {ex}",
            flush=True,
        )
        import pyvista as pv
        p = pv.Plotter(off_screen=True, window_size=tuple(size))
        p.set_background(bg)
        try:
            p.enable_anti_aliasing("ssaa")
        except Exception:                                # noqa: BLE001
            try:
                p.enable_anti_aliasing()
            except Exception:                            # noqa: BLE001
                pass
        return p


def _polydata_with_normals(poly):
    """Compute point normals so phong shading reads as smooth.
    Mirrors `_add_phong_mesh` in app.py — the live scene uses
    the SAME flags so the offscreen render matches. Returns the
    original polydata on failure (better an un-shaded actor than
    no actor)."""
    try:
        return poly.compute_normals(
            cell_normals=False, point_normals=True,
            consistent_normals=True, auto_orient_normals=True,
            non_manifold_traversal=False,
        )
    except Exception:                                    # noqa: BLE001
        return poly


def _add_region(
    plotter, polydata_mm, *, style, show_edges=False,
    opacity_override=None,
):
    """Add a mesh region polydata with the supplied phong style
    dict. `style` matches the schema of golgi.app DEFAULTS values:
    {color, opacity, ambient, diffuse, specular, specular_power}.
    Polydata is expected in mm (same as `geom.region_surfaces_viz`).

    Computes point normals first so the phong shader has the
    inputs the live scene's `_add_phong_mesh` relies on —
    without that step the surface reads flat instead of smooth.

    `show_edges=True` overlays the mesh wireframe in dark grey so
    the user can see the triangulation density. Used by the mesh
    quality variants.

    `opacity_override` forces a specific opacity (useful when the
    default style value would hide the actor — e.g. tag 4 muscle
    at α=0.2 is invisible without it)."""
    try:
        surf = _polydata_with_normals(polydata_mm)
        return plotter.add_mesh(
            surf,
            color=style.get("color", (0.6, 0.6, 0.6)),
            opacity=(
                float(opacity_override)
                if opacity_override is not None
                else float(style.get("opacity", 1.0))
            ),
            pbr=False,
            ambient=float(style.get("ambient", 0.3)),
            diffuse=float(style.get("diffuse", 0.7)),
            specular=float(style.get("specular", 0.2)),
            specular_power=float(
                style.get("specular_power", 10.0),
            ),
            smooth_shading=True,
            show_edges=bool(show_edges),
            edge_color="#333333" if show_edges else None,
            line_width=0.4 if show_edges else 1.0,
        )
    except Exception as ex:                              # noqa: BLE001
        print(
            f"[render3d] add_region failed: "
            f"{type(ex).__name__}: {ex}",
            flush=True,
        )
        return None


def _add_region_colored_by_quality(plotter, polydata_full):
    """Add a region polydata coloured by per-cell `q_tet` (mesh
    triangle quality), RdYlGn cmap clipped to [0, 1]. Mesh edges
    are overlaid in dark grey so the user can see the boundary
    triangulation density.

    Requires the FULL (non-decimated) polydata since `q_tet` lives
    in `cell_data` and decimation drops cell scalars."""
    if "q_tet" not in polydata_full.cell_data:
        return None
    try:
        # Skip normals computation here — RdYlGn quality colouring
        # uses per-CELL data; computing point normals would force
        # a flat shading hack that obscures the per-cell colours.
        return plotter.add_mesh(
            polydata_full,
            scalars="q_tet",
            cmap="RdYlGn",
            clim=(0.0, 1.0),
            opacity=0.92,
            smooth_shading=False,
            show_edges=True,
            edge_color="#333333",
            line_width=0.4,
            show_scalar_bar=False,
        )
    except Exception as ex:                              # noqa: BLE001
        print(
            f"[render3d] add_region_colored_by_quality failed: "
            f"{type(ex).__name__}: {ex}",
            flush=True,
        )
        return None


def _add_region_colored_by_ve(
    plotter, polydata_full, ve_array_mV, *, clim_mV=None,
):
    """Add a region polydata coloured by a per-point Ve scalar
    (already in mV). Uses the plasma cmap to match the live scene.
    `clim_mV` is a (low, high) pair; falls back to the 1-99
    percentile of the input array when None.

    `polydata_full` must be the non-decimated polydata so the
    scalar length matches the point count."""
    try:
        ve = np.asarray(ve_array_mV, dtype=np.float32).copy()
        good = np.isfinite(ve)
        if good.any():
            ve[~good] = float(np.median(ve[good]))
        else:
            ve[:] = 0.0
        if clim_mV is None:
            lo = float(np.percentile(ve, 1.0))
            hi = float(np.percentile(ve, 99.0))
            if hi - lo < 1e-6:
                hi = lo + 1e-6
            clim_mV = (lo, hi)
        surf = _polydata_with_normals(polydata_full.copy())
        surf.point_data["Ve"] = ve
        surf.GetPointData().SetActiveScalars("Ve")
        return plotter.add_mesh(
            surf,
            scalars="Ve",
            cmap="plasma",
            clim=clim_mV,
            opacity=1.0,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=False,
        )
    except Exception as ex:                              # noqa: BLE001
        print(
            f"[render3d] add_region_colored_by_ve failed: "
            f"{type(ex).__name__}: {ex}",
            flush=True,
        )
        return None


def _add_fibers_colored_by_ve(
    plotter, fiber_paths_raw, fiber_paths_Ve, *,
    radius_mm=0.025, clim_mV=None,
):
    """Render fiber trajectories as a single tube actor coloured
    by per-point Ve (V → mV conversion inside). Matches the live
    scene's `render_fibers_by_branch` Ve-overlay branch."""
    if (not fiber_paths_raw
            or not fiber_paths_Ve
            or len(fiber_paths_Ve) != len(fiber_paths_raw)):
        return None
    import pyvista as pv
    pts_chunks: list[np.ndarray] = []
    cells_chunks: list[np.ndarray] = []
    ve_chunks: list[np.ndarray] = []
    offset = 0
    for p, ve in zip(fiber_paths_raw, fiber_paths_Ve):
        p_mm = np.asarray(p, dtype=np.float64) * 1000.0
        n = int(p_mm.shape[0])
        if n < 2 or len(ve) != n:
            continue
        pts_chunks.append(p_mm)
        cells_chunks.append(
            np.concatenate([[n], np.arange(n) + offset]),
        )
        ve_chunks.append(np.asarray(ve, dtype=np.float32))
        offset += n
    if not pts_chunks:
        return None
    ve_pts = np.concatenate(ve_chunks).astype(np.float32)
    good = np.isfinite(ve_pts)
    if good.any():
        ve_pts[~good] = float(np.median(ve_pts[good]))
    else:
        ve_pts[:] = 0.0
    poly = pv.PolyData(
        np.vstack(pts_chunks).astype(np.float64),
        lines=np.concatenate(cells_chunks).astype(np.int64),
    )
    poly.point_data["Ve"] = ve_pts
    poly.GetPointData().SetActiveScalars("Ve")
    try:
        tube = poly.tube(
            radius=radius_mm, n_sides=10, capping=False,
        )
    except Exception as ex:                              # noqa: BLE001
        print(
            f"[render3d] fiber tube build failed: "
            f"{type(ex).__name__}: {ex}",
            flush=True,
        )
        return None
    tube.point_data["Ve"] = (
        tube.point_data["Ve"].astype(np.float32) * 1.0e3  # V → mV
    )
    tube.GetPointData().SetActiveScalars("Ve")
    if clim_mV is None:
        lo = float(np.percentile(tube.point_data["Ve"], 1.0))
        hi = float(np.percentile(tube.point_data["Ve"], 99.0))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        clim_mV = (lo, hi)
    return plotter.add_mesh(
        tube,
        scalars="Ve",
        cmap="plasma",
        clim=clim_mV,
        opacity=1.0,
        smooth_shading=True,
        show_edges=False,
        show_scalar_bar=False,
    )


def _add_field_streamlines(plotter, field_lines_poly):
    """Render the pre-computed E-field streamline polydata
    (`geom.field_lines_poly`) as thin tubes coloured by `E_mag`.
    Returns None if the polydata is missing or doesn't carry the
    E_mag scalar."""
    if field_lines_poly is None:
        return None
    try:
        tube = field_lines_poly.tube(
            radius=0.03, n_sides=8, capping=False,
        )
    except Exception as ex:                              # noqa: BLE001
        print(
            f"[render3d] field-line tube build failed: "
            f"{type(ex).__name__}: {ex}",
            flush=True,
        )
        return None
    scalars = "E_mag" if "E_mag" in tube.point_data else None
    return plotter.add_mesh(
        tube,
        scalars=scalars,
        cmap="viridis" if scalars else None,
        color=None if scalars else "#0080ff",
        opacity=0.95,
        smooth_shading=True,
        show_edges=False,
        show_scalar_bar=False,
    )


def _add_fibers(
    plotter,
    fiber_paths_raw,
    branch_idx,
    branch_palette,
    *,
    radius_mm=0.025,
):
    """Render fiber trajectories as thin tubes coloured by branch.
    `fiber_paths_raw` is the workspace's per-fiber path list in
    metres; we convert to mm to match the region polydata."""
    if not fiber_paths_raw:
        return
    import pyvista as pv
    n_palette = max(1, len(branch_palette))
    for i, path in enumerate(fiber_paths_raw):
        path_mm = np.asarray(path, dtype=np.float64) * 1000.0
        if path_mm.shape[0] < 2:
            continue
        if branch_idx is not None and i < len(branch_idx):
            b = int(branch_idx[i])
        else:
            b = 0
        color = branch_palette[b % n_palette]
        try:
            line = pv.lines_from_points(path_mm)
            tube = line.tube(radius=radius_mm, n_sides=8, capping=False)
            plotter.add_mesh(
                tube,
                color=color,
                opacity=1.0,
                smooth_shading=True,
                show_edges=False,
            )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[render3d] fiber {i} skipped: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )


def _add_cuff(
    plotter,
    region_surfaces_viz,
    *,
    silicone_style,
    gold_style,
    saline_style=None,
    include_saline=False,
):
    """Add the cuff actors. The cuff body lives at tag 3 (silicone)
    and the contacts at tag TAG_GOLD (6) in `region_surfaces_viz`.
    `include_saline` toggles the surrounding saline shell (tag 2)
    — useful for FEM-zoom variants where the saline is part of the
    visible domain."""
    if not region_surfaces_viz:
        return
    if include_saline and saline_style is not None:
        saline = region_surfaces_viz.get(2)
        if saline is not None:
            _add_region(plotter, saline, style=saline_style)
    silicone = region_surfaces_viz.get(3)
    if silicone is not None:
        _add_region(plotter, silicone, style=silicone_style)
    gold = region_surfaces_viz.get(6)
    if gold is not None:
        _add_region(plotter, gold, style=gold_style)


# ---------------------------------------------------------------------------
# Camera framing
# ---------------------------------------------------------------------------


def _nerve_bbox_mm(geom) -> "tuple[float, ...] | None":
    """Bounding box of the full nerve in mm. Tries the raw nerve
    points first (pre-mesh path), then the mesh nodes (post-mesh
    path). Returns (xmin, xmax, ymin, ymax, zmin, zmax) or None
    when no geometry is loaded."""
    try:
        pts = getattr(geom, "nerve", {}).get("pts_raw")
        if pts is not None and len(pts) > 0:
            pts_mm = np.asarray(pts, dtype=np.float64) * 1000.0
            return (
                float(pts_mm[:, 0].min()),
                float(pts_mm[:, 0].max()),
                float(pts_mm[:, 1].min()),
                float(pts_mm[:, 1].max()),
                float(pts_mm[:, 2].min()),
                float(pts_mm[:, 2].max()),
            )
    except Exception:                                    # noqa: BLE001
        pass
    try:
        nodes = getattr(geom, "mesh_nodes", None)
        if nodes is not None and len(nodes) > 0:
            n = np.asarray(nodes, dtype=np.float64)
            # mesh nodes are stored in mm directly in this codebase.
            return (
                float(n[:, 0].min()), float(n[:, 0].max()),
                float(n[:, 1].min()), float(n[:, 1].max()),
                float(n[:, 2].min()), float(n[:, 2].max()),
            )
    except Exception:                                    # noqa: BLE001
        pass
    return None


def _cuff_bbox_mm(geom) -> "tuple[float, ...] | None":
    """Bounding box of the cuff region (tag 3 silicone + tag 6
    gold contacts) in mm. Falls back to the gold contacts alone
    when silicone isn't present yet."""
    rs = getattr(geom, "region_surfaces_viz", None) or {}
    bbs: list[tuple[float, ...]] = []
    for tag in (3, 6):
        poly = rs.get(tag)
        if poly is not None:
            try:
                b = poly.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
                bbs.append(tuple(b))
            except Exception:                            # noqa: BLE001
                continue
    if not bbs:
        return None
    arr = np.asarray(bbs, dtype=np.float64)
    return (
        float(arr[:, 0].min()), float(arr[:, 1].max()),
        float(arr[:, 2].min()), float(arr[:, 3].max()),
        float(arr[:, 4].min()), float(arr[:, 5].max()),
    )


def _frame_camera(
    plotter,
    bbox: "tuple[float, ...] | None",
    *,
    view: str = "iso",
    azimuth: float = 35.0,
    elevation: float = 20.0,
    zoom: float = 1.0,
    pad: float = 0.12,
) -> None:
    """Set the offscreen plotter's camera to a named preset. `bbox`
    in mm is used to frame the view; pad widens the framing so the
    geometry doesn't kiss the image edges.

    Views:
      * "iso" — azimuth/elevation rotation from the +z (default)
        camera position, framed to `bbox`.
      * "side" — xz plane (look from +y).
      * "top" — xy plane (look from +z).
      * "cross_section" — yz plane (look from +x), used for the
        through-cuff cross-section variant.
    """
    if bbox is None:
        # No geometry → reset_camera does the best it can.
        plotter.reset_camera()
        return
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    if view == "side":
        plotter.view_xz()
    elif view == "top":
        plotter.view_xy()
    elif view == "cross_section":
        plotter.view_yz()
    else:  # "iso"
        plotter.view_isometric()
    plotter.reset_camera(
        bounds=(
            xmin - pad * (xmax - xmin),
            xmax + pad * (xmax - xmin),
            ymin - pad * (ymax - ymin),
            ymax + pad * (ymax - ymin),
            zmin - pad * (zmax - zmin),
            zmax + pad * (zmax - zmin),
        ),
    )
    if view == "iso":
        try:
            plotter.camera.azimuth = float(azimuth)
            plotter.camera.elevation = float(elevation)
        except Exception:                                # noqa: BLE001
            pass
    if zoom != 1.0:
        try:
            plotter.camera.zoom(float(zoom))
        except Exception:                                # noqa: BLE001
            pass


def _screenshot_png(plotter) -> bytes:
    """Render the offscreen plotter to PNG bytes via mpl.image
    (already a project dep — saves pulling in PIL just for the
    encode). Closes the plotter afterwards so VTK releases the
    framebuffer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as _mpli
    img = plotter.screenshot(
        return_img=True,
        transparent_background=False,
    )
    buf = io.BytesIO()
    _mpli.imsave(buf, img, format="png")
    try:
        plotter.close()
    except Exception:                                    # noqa: BLE001
        pass
    return buf.getvalue()


def _png_to_mpl_image(png_bytes: bytes):
    """Decode a PNG byte-string into an (H, W, C) numpy array via
    matplotlib's PNG reader. Used by report.py to embed snapshots
    into a matplotlib PDF page (`ax.imshow`)."""
    import matplotlib.image
    return matplotlib.image.imread(io.BytesIO(png_bytes))


def capture_live_viewport(plotter) -> bytes:
    """Snapshot the live workspace PyVista plotter as PNG bytes.
    Same helper signature as Phase 1 — kept so callers that DO
    want the user's current view (e.g. a "use current camera"
    override) can still reach it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image
    img = plotter.screenshot(
        return_img=True,
        transparent_background=False,
    )
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, img, format="png")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VariantConfig:
    """One off-screen render variant.

    `regions`  iterable of region tags to add (1 endo, 2 saline,
               3 silicone, 4 muscle, 5 epineurium, 6 gold contacts).
               TAG_GOLD = 6 is handled separately so the cuff body
               + gold pair correctly with the cuff_zoom view.
    `fibers`   if True, draw fiber tubes via _add_fibers.
    `view`     camera view preset (passed to _frame_camera).
    `bbox`     "nerve" or "cuff" — which bbox to frame against.

    Overlay / scalar-rendering flags:
    `show_edges`         draw triangulation edges on every region
                         actor (used by mesh quality variants).
    `color_by_quality`   colour endo/epi/muscle by per-cell
                         `q_tet` scalar via RdYlGn cmap. Falls
                         back to solid colour when q_tet is
                         missing on a particular region.
    `ve_on_regions`      tags (subset of `regions`) that should be
                         coloured by Ve point-scalar instead of
                         their solid styling. Tag 1 (endo) uses
                         `geom.nerve_surface_Ve` directly; tag 5
                         (epi) borrows it via nearest-vertex
                         lookup.
    `ve_on_fibers`       fibers as a single tube actor coloured by
                         per-point Ve from `geom.fiber_paths_Ve`.
    `show_field_lines`   render `geom.field_lines_poly` as tubes
                         coloured by |E|.
    `use_full_polydata`  prefer `geom.region_surfaces` over the
                         decimated `geom.region_surfaces_viz` —
                         needed when q_tet or Ve scalars must
                         match the polydata's vertex / cell count.
    `caption`            short caption embedded under the figure
                         in the report (per-variant context, e.g.
                         "muscle hidden" / "Ve plasma overlay").
    """
    title: str
    regions: tuple[int, ...] = ()
    fibers: bool = False
    view: str = "iso"
    bbox: str = "nerve"
    show_edges: bool = False
    color_by_quality: bool = False
    ve_on_regions: tuple[int, ...] = ()
    ve_on_fibers: bool = False
    show_field_lines: bool = False
    use_full_polydata: bool = False
    caption: str = ""


# Section ID prefix matches the FEATURES.md F2.3 spec's
# `render3d.<name>` naming so future readers can map docs → code.
#
# Tag legend: 1 endo · 2 saline · 3 silicone (cuff body) · 4 muscle
#             · 5 epineurium · 6 gold contacts (TAG_GOLD).
VARIANTS: dict[str, _VariantConfig] = {
    # ---- Electrode-only (cuff dry; in nerve; with saline infill) ----
    "render3d.electrode_geom": _VariantConfig(
        title="Electrode geometry · dry",
        regions=(3, 6),       # silicone + gold contacts
        view="iso", bbox="cuff",
        caption="Cuff silicone + gold contacts only.",
    ),
    "render3d.electrode_polar": _VariantConfig(
        title="Electrode · anode/cathode coloured",
        regions=(3, 6),
        view="iso", bbox="cuff",
        caption=(
            "v1: same as Electrode geometry — per-contact "
            "polarity colouring lands in a follow-up commit."
        ),
    ),
    "render3d.electrode_with_saline": _VariantConfig(
        title="Electrode · with saline infill",
        regions=(2, 3, 6),    # saline + silicone + gold
        view="iso", bbox="cuff",
        caption=(
            "Cuff with the surrounding saline bath included — "
            "shows the FEM electrolyte domain around the contacts."
        ),
    ),
    "render3d.electrode_in_nerve": _VariantConfig(
        title="Electrode · on epineurium",
        regions=(5, 3, 6),    # epi outline + silicone + gold
        view="iso", bbox="cuff",
        caption=(
            "Cuff overlaid on the epineurium so the placement "
            "relative to the nerve is visible."
        ),
    ),
    # ---- Mesh (regions + quality-coloured + per-region wireframe) ----
    "render3d.mesh_all_regions": _VariantConfig(
        title="Mesh · all regions",
        regions=(4, 5, 1, 3, 6),    # outer → inner blend
        view="iso", bbox="nerve",
        caption="All meshed regions, default styling.",
    ),
    "render3d.mesh_quality_all": _VariantConfig(
        title="Mesh · quality (RdYlGn, with edges)",
        regions=(4, 5, 1, 3, 6),
        view="iso", bbox="nerve",
        show_edges=True,
        color_by_quality=True,
        use_full_polydata=True,    # q_tet lives on the full polydata
        caption=(
            "Per-cell q_tet scalar (RdYlGn) with the boundary "
            "triangulation edges overlaid."
        ),
    ),
    "render3d.mesh_muscle": _VariantConfig(
        title="Mesh · muscle only (with edges)",
        regions=(4,), view="iso", bbox="nerve",
        show_edges=True,
        use_full_polydata=True,
    ),
    "render3d.mesh_endo": _VariantConfig(
        title="Mesh · endoneurium only (with edges)",
        regions=(1,), view="iso", bbox="nerve",
        show_edges=True,
        use_full_polydata=True,
    ),
    "render3d.mesh_epi": _VariantConfig(
        title="Mesh · epineurium only (with edges)",
        regions=(5,), view="iso", bbox="nerve",
        show_edges=True,
        use_full_polydata=True,
    ),
    "render3d.mesh_cuff": _VariantConfig(
        title="Mesh · cuff only (with edges)",
        regions=(3, 6), view="iso", bbox="cuff",
        show_edges=True,
        use_full_polydata=True,
    ),
    # ---- Fibers ----
    "render3d.fibers_epi": _VariantConfig(
        title="Fibers · with epineurium @ α=0.5",
        regions=(5,),
        fibers=True, view="iso", bbox="nerve",
    ),
    "render3d.fibers_in_nerve": _VariantConfig(
        title="Fibers · with endoneurium + epineurium",
        regions=(1, 5),
        fibers=True, view="iso", bbox="nerve",
        caption=(
            "Fibers inside endoneurium with translucent "
            "epineurium for context."
        ),
    ),
    # ---- Geometry overviews (no overlays) ----
    "render3d.geometry_full": _VariantConfig(
        title="Geometry · full (tissues + cuff + fibers)",
        regions=(4, 5, 1, 3, 6),
        fibers=True, view="iso", bbox="nerve",
        caption=(
            "Full domain: muscle + epi + endo + cuff body + "
            "contacts + fiber trajectories. No overlays."
        ),
    ),
    "render3d.geometry_no_muscle": _VariantConfig(
        title="Geometry · no muscle",
        regions=(5, 1, 3, 6),
        fibers=True, view="iso", bbox="nerve",
        caption=(
            "Geometry without the muscle envelope so the "
            "nerve + cuff relationship is clearer."
        ),
    ),
    # ---- FEM Ve overlays (on endo, epi, fibers, all) ----
    "render3d.ve_on_endo": _VariantConfig(
        title="FEM · V_e on endoneurium",
        regions=(1,), view="iso", bbox="nerve",
        ve_on_regions=(1,),
        use_full_polydata=True,
        caption=(
            "Endoneurium coloured by V_e (plasma cmap). Requires "
            "a completed FEM solve."
        ),
    ),
    "render3d.ve_on_epi": _VariantConfig(
        title="FEM · V_e on epineurium",
        regions=(5,), view="iso", bbox="nerve",
        ve_on_regions=(5,),
        use_full_polydata=True,
        caption=(
            "Epineurium coloured by V_e (nearest-vertex sample "
            "from the endo Ve array)."
        ),
    ),
    "render3d.ve_on_fibers": _VariantConfig(
        title="FEM · V_e on fibers",
        regions=(5,),
        ve_on_fibers=True, view="iso", bbox="nerve",
        caption=(
            "Fiber trajectories coloured by per-point V_e — the "
            "scalar each axon actually sees along its length."
        ),
    ),
    "render3d.ve_on_all": _VariantConfig(
        title="FEM · V_e on endo + epi + fibers",
        regions=(1, 5),
        ve_on_regions=(1, 5),
        ve_on_fibers=True,
        use_full_polydata=True,
        view="iso", bbox="nerve",
        caption=(
            "All three Ve-aware actors with a shared plasma "
            "colourbar — endo / epi surfaces + fiber tubes."
        ),
    ),
    # ---- FEM visibility combos (kept from v1, kept fiber overlay) ----
    "render3d.fem_full": _VariantConfig(
        title="FEM · all regions visible",
        regions=(4, 5, 1, 3, 6),
        fibers=True, view="iso", bbox="nerve",
    ),
    "render3d.fem_no_muscle": _VariantConfig(
        title="FEM · muscle hidden",
        regions=(5, 1, 3, 6),
        fibers=True, view="iso", bbox="nerve",
    ),
    "render3d.fem_no_epi": _VariantConfig(
        title="FEM · epineurium hidden",
        regions=(4, 1, 3, 6),
        fibers=True, view="iso", bbox="cuff",
    ),
    "render3d.fem_no_endo": _VariantConfig(
        title="FEM · endoneurium hidden",
        regions=(4, 5, 3, 6),
        fibers=True, view="iso", bbox="cuff",
    ),
    # ---- E-field streamlines ----
    "render3d.field_streamlines": _VariantConfig(
        title="FEM · E-field streamlines",
        regions=(5, 3, 6),    # translucent epi + cuff for context
        show_field_lines=True,
        view="iso", bbox="nerve",
        caption=(
            "E-field streamlines integrated through the FEM slice "
            "volume, coloured by |E|. Cuff + epi rendered for "
            "spatial context."
        ),
    ),
    "render3d.field_streamlines_cuff_zoom": _VariantConfig(
        title="FEM · E-field streamlines (cuff zoom)",
        regions=(3, 6),
        show_field_lines=True,
        view="iso", bbox="cuff",
        caption=(
            "Same streamlines, zoomed into the cuff region — "
            "shows where the current actually concentrates."
        ),
    ),
    # ---- Cuff-zoom Ve overlays (the most-asked-for variants) ----
    # Three Ve targets (epi / endo / fibers) × three context
    # combos (with cuff + streamlines · no cuff + streamlines ·
    # no cuff, no streamlines) = nine pages worth of "what does
    # the field look like AT the cuff?". Each frames the camera
    # to the cuff bbox so the relevant geometry is large.
    #
    # — with cuff + streamlines —
    "render3d.cuff_zoom_ve_epi_with_cuff_streamlines": _VariantConfig(
        title="Cuff zoom · V_e on epi + cuff + streamlines",
        regions=(5, 3, 6),
        ve_on_regions=(5,),
        show_field_lines=True,
        use_full_polydata=True,
        view="iso", bbox="cuff",
        caption=(
            "Epineurium coloured by V_e, cuff body + contacts "
            "visible, E-field streamlines overlaid. Camera "
            "framed to the cuff bbox."
        ),
    ),
    "render3d.cuff_zoom_ve_endo_with_cuff_streamlines": _VariantConfig(
        title="Cuff zoom · V_e on endo + cuff + streamlines",
        regions=(1, 3, 6),
        ve_on_regions=(1,),
        show_field_lines=True,
        use_full_polydata=True,
        view="iso", bbox="cuff",
        caption=(
            "Endoneurium coloured by V_e, cuff body + contacts "
            "visible, E-field streamlines overlaid."
        ),
    ),
    "render3d.cuff_zoom_ve_fibers_with_cuff_streamlines": _VariantConfig(
        title="Cuff zoom · V_e on fibers + cuff + streamlines",
        regions=(3, 6),
        ve_on_fibers=True,
        show_field_lines=True,
        view="iso", bbox="cuff",
        caption=(
            "Fiber trajectories coloured by V_e, cuff body + "
            "contacts visible, E-field streamlines overlaid."
        ),
    ),
    # — no cuff, with streamlines —
    "render3d.cuff_zoom_ve_epi_streamlines_no_cuff": _VariantConfig(
        title="Cuff zoom · V_e on epi + streamlines (no cuff)",
        regions=(5,),
        ve_on_regions=(5,),
        show_field_lines=True,
        use_full_polydata=True,
        view="iso", bbox="cuff",
        caption=(
            "Same view as above with the cuff hidden — easier to "
            "read the field on the epineurium without specular "
            "highlights from the silicone."
        ),
    ),
    "render3d.cuff_zoom_ve_endo_streamlines_no_cuff": _VariantConfig(
        title="Cuff zoom · V_e on endo + streamlines (no cuff)",
        regions=(1,),
        ve_on_regions=(1,),
        show_field_lines=True,
        use_full_polydata=True,
        view="iso", bbox="cuff",
        caption=(
            "Endo V_e + streamlines, cuff hidden — exposes the "
            "field inside the nerve directly."
        ),
    ),
    "render3d.cuff_zoom_ve_fibers_streamlines_no_cuff": _VariantConfig(
        title="Cuff zoom · V_e on fibers + streamlines (no cuff)",
        regions=(),
        ve_on_fibers=True,
        show_field_lines=True,
        view="iso", bbox="cuff",
        caption=(
            "Fiber-tube V_e + streamlines without the cuff — "
            "isolates the per-fiber field along each axon."
        ),
    ),
    # — no cuff, no streamlines (Ve only) —
    "render3d.cuff_zoom_ve_epi_only": _VariantConfig(
        title="Cuff zoom · V_e on epi (no cuff, no streamlines)",
        regions=(5,),
        ve_on_regions=(5,),
        use_full_polydata=True,
        view="iso", bbox="cuff",
        caption=(
            "Pure epineurium V_e map — no cuff actor, no "
            "streamlines, nothing competing with the colour map."
        ),
    ),
    "render3d.cuff_zoom_ve_endo_only": _VariantConfig(
        title="Cuff zoom · V_e on endo (no cuff, no streamlines)",
        regions=(1,),
        ve_on_regions=(1,),
        use_full_polydata=True,
        view="iso", bbox="cuff",
        caption=(
            "Pure endoneurium V_e map at the cuff."
        ),
    ),
    "render3d.cuff_zoom_ve_fibers_only": _VariantConfig(
        title="Cuff zoom · V_e on fibers (no cuff, no streamlines)",
        regions=(),
        ve_on_fibers=True,
        view="iso", bbox="cuff",
        caption=(
            "Pure fiber V_e map at the cuff."
        ),
    ),
    # ---- Zoom + cross-section ----
    "render3d.cuff_zoom_iso": _VariantConfig(
        title="Cuff region · iso zoom",
        regions=(5, 1, 3, 6),
        fibers=True, view="iso", bbox="cuff",
    ),
    "render3d.cuff_cross_section": _VariantConfig(
        title="Cuff region · cross-section",
        regions=(1, 3, 6),
        fibers=False, view="cross_section", bbox="cuff",
    ),
}


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


def _resolve_endo_ve_for_epi(geom, epi_polydata):
    """Sample the endo Ve array onto an epi polydata's points via
    nearest-vertex KDTree lookup. Matches the live scene's "Ve on
    nerve surface" branch for tag 5. Returns the mV-scaled array
    (length == epi.n_points), or None when the endo Ve / endo
    surface aren't available."""
    nerve_surface_Ve = getattr(geom, "nerve_surface_Ve", None)
    region_surfaces = getattr(geom, "region_surfaces", None) or {}
    if nerve_surface_Ve is None or 1 not in region_surfaces:
        return None
    endo_surf = region_surfaces[1]
    if len(nerve_surface_Ve) != endo_surf.n_points:
        return None
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(
            np.asarray(endo_surf.points, dtype=np.float64),
        )
        _, nn = tree.query(
            np.asarray(epi_polydata.points, dtype=np.float64),
            k=1,
        )
        ve = np.asarray(
            nerve_surface_Ve, dtype=np.float32,
        )[nn]
        return ve * 1.0e3  # V → mV
    except Exception as ex:                              # noqa: BLE001
        print(
            f"[render3d] epi Ve resample failed: "
            f"{type(ex).__name__}: {ex}",
            flush=True,
        )
        return None


def render_variant(
    variant_id: str,
    geom,
    *,
    region_defaults: dict,
    gold_style: dict,
    branch_palette,
    size: tuple[int, int] = (1280, 960),
) -> bytes:
    """Render one of the registered variants to PNG bytes. Returns
    an empty bytes object when the geometry inputs the variant
    needs are missing (e.g. mesh-region variants on a pre-mesh
    project, Ve variants on a pre-solve project). Caller is
    expected to surface a placeholder page in that case rather
    than crash.

    `region_defaults` maps tag → style dict (DEFAULTS in app.py);
    `gold_style` is the GOLD_STYLE constant; `branch_palette` is
    the BRANCH_PALETTE list. Passed in so this module stays free
    of app.py imports."""
    cfg = VARIANTS.get(variant_id)
    if cfg is None:
        raise ValueError(
            f"unknown render3d variant '{variant_id}' — "
            f"valid ids: {sorted(VARIANTS.keys())}",
        )
    # Pick the polydata source: full (region_surfaces) when we need
    # cell scalars like q_tet or point scalars like Ve, decimated
    # otherwise. Falling back to the available one when the
    # preferred isn't there.
    rs_viz = getattr(geom, "region_surfaces_viz", None) or {}
    rs_full = getattr(geom, "region_surfaces", None) or {}
    if cfg.use_full_polydata:
        rs_primary = rs_full or rs_viz
    else:
        rs_primary = rs_viz or rs_full

    needs_regions = bool(cfg.regions)
    has_regions = bool(rs_primary)
    needs_fibers = cfg.fibers or cfg.ve_on_fibers
    has_fibers = bool(
        getattr(geom, "fiber_paths_raw", None) is not None
        and len(geom.fiber_paths_raw) > 0,
    )
    needs_field_lines = cfg.show_field_lines
    has_field_lines = (
        getattr(geom, "field_lines_poly", None) is not None
    )
    needs_ve = cfg.ve_on_fibers or bool(cfg.ve_on_regions)
    has_surface_ve = (
        getattr(geom, "nerve_surface_Ve", None) is not None
    )
    has_fiber_ve = (
        getattr(geom, "fiber_paths_Ve", None) is not None
        and len(getattr(geom, "fiber_paths_Ve", []) or [])
        == (len(geom.fiber_paths_raw or [])
            if geom.fiber_paths_raw is not None else 0)
    )

    # Availability gates — fail fast with empty bytes so the
    # caller can drop a placeholder page.
    if needs_regions and not has_regions:
        return b""
    if needs_fibers and not has_fibers:
        if not has_regions:
            return b""
    if needs_field_lines and not has_field_lines:
        if not has_regions:
            return b""
    if cfg.ve_on_regions and not has_surface_ve:
        # Fall back to solid rendering on those regions; not fatal.
        pass
    if cfg.ve_on_fibers and not has_fiber_ve:
        if not has_regions:
            return b""

    plotter = _build_offscreen_plotter(size)

    # ---- Regions ----
    # Honour outer→inner ordering for nice blending with transparent
    # outer regions (muscle / epi).
    for tag in cfg.regions:
        poly = rs_primary.get(tag)
        if poly is None:
            # Try the other source if the primary doesn't have it.
            poly = (rs_full if rs_primary is rs_viz else rs_viz).get(tag)
        if poly is None:
            continue
        style = region_defaults.get(tag)
        if style is None and tag == 6:
            style = gold_style
        if style is None:
            continue

        # 1) Ve on this region wins over quality colouring (matches
        #    the live scene's precedence).
        if (cfg.ve_on_regions
                and tag in cfg.ve_on_regions
                and has_surface_ve):
            # Endo: use nerve_surface_Ve directly.
            # Epi: sample via KDTree from the endo Ve.
            if tag == 1:
                surface_Ve = getattr(geom, "nerve_surface_Ve", None)
                ve_mV = (
                    np.asarray(surface_Ve, dtype=np.float32) * 1e3
                    if surface_Ve is not None else None
                )
            elif tag == 5:
                ve_mV = _resolve_endo_ve_for_epi(geom, poly)
            else:
                ve_mV = None
            if ve_mV is not None and len(ve_mV) == poly.n_points:
                _add_region_colored_by_ve(
                    plotter, poly, ve_mV,
                )
                continue
            # Length mismatch / missing → fall through to solid.

        # 2) Quality colouring (RdYlGn with edges).
        if cfg.color_by_quality:
            actor = _add_region_colored_by_quality(plotter, poly)
            if actor is not None:
                continue
            # q_tet missing → fall through to solid + edges.

        # 3) Solid styling. Force a minimum opacity for muscle/epi
        #    in cuff-zoom variants so they don't render invisible
        #    when the user pulls in tight.
        opacity_override = None
        if cfg.bbox == "cuff" and tag in (4,):
            # Cuff zoom + muscle = effectively invisible at α=0.2;
            # bump to 0.06 so it reads as a faint context shell.
            opacity_override = 0.06
        _add_region(
            plotter, poly, style=style,
            show_edges=cfg.show_edges,
            opacity_override=opacity_override,
        )

    # ---- Fibers ----
    if cfg.ve_on_fibers and has_fibers and has_fiber_ve:
        _add_fibers_colored_by_ve(
            plotter,
            geom.fiber_paths_raw,
            geom.fiber_paths_Ve,
        )
    elif cfg.fibers and has_fibers:
        _add_fibers(
            plotter,
            geom.fiber_paths_raw,
            getattr(geom, "fiber_branch_idx", None),
            branch_palette,
        )

    # ---- E-field streamlines ----
    if cfg.show_field_lines and has_field_lines:
        _add_field_streamlines(
            plotter,
            geom.field_lines_poly,
        )

    # ---- Camera ----
    bbox = (
        _cuff_bbox_mm(geom) if cfg.bbox == "cuff"
        else _nerve_bbox_mm(geom)
    )
    _frame_camera(plotter, bbox, view=cfg.view)
    return _screenshot_png(plotter)
