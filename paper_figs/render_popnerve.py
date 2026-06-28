# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cel-shaded 3D render of a Duke population nerve (multifascicular) + 12-contact cuff
+ fiber population, in the dog-VNS / electrode-gallery style. Reuses render_components'
geometry (mask rebuild -> per-fascicle prisms, telescoped reveal, analytic cuff +
contact pads) but renders with flat fills + dark silhouettes instead of PBR.

Usage: render(sample_dir, out_png).  Used for the Fig 5 population nerves (swine, human)."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import px_per_mm, write_ppmm  # noqa: E402
from render_components import (  # noqa: E402
    _nerve_from_masks, _slice_poly, _sheath_prism, _offset_loop, _clip_capped, _cuff,
    _contact_pads, _derive_mask_dir, _read_mm, STYLE, FIBER_COLOR, EPS_MM)

SILICONE = "#b7c0cc"   # light gray cuff (matches the comic style of fig2a/fig7a, not glassy blue)
GOLD = "#f1c62f"
INK = "#1b1d22"
SIL = dict(color=INK, line_width=2.5)
ENDO, EPI, SALINE = STYLE["endo"][1], STYLE["epi"][1], STYLE["surf_saline"][1]


def _flat(p, m, color, opacity, sil=True, smooth=True):
    p.add_mesh(m, color=color, opacity=opacity, smooth_shading=smooth, ambient=0.46,
               diffuse=0.66, specular=0.0, show_scalar_bar=False,
               silhouette=SIL if sil else None)


def _ink(p, m, lw=2.0, fa=18):
    e = m.extract_feature_edges(feature_angle=fa, boundary_edges=True,
                                non_manifold_edges=False, feature_edges=True,
                                manifold_edges=False)
    if e.n_cells:
        p.add_mesh(e, color=INK, line_width=lw, render_lines_as_tubes=True, show_scalar_bar=False)


def _gold(p, m):
    p.add_mesh(m, color=GOLD, opacity=1.0, smooth_shading=True, ambient=0.6, diffuse=0.55,
               specular=0.06, show_scalar_bar=False, silhouette=SIL)


def render(sample, out_png, fiber_tube_um=30.0, fiber_stride=4,
           fiber_stick=0.5, endo_stick=2.7):
    sample = Path(sample)
    mask_dir = _derive_mask_dir(sample)
    epi_pv, fasc_pv = _nerve_from_masks(mask_dir)
    zmax = max(abs(epi_pv.bounds[4]), abs(epi_pv.bounds[5]))
    near = 1.0
    endo_cut = near * (zmax - fiber_stick)
    epi_cut = near * (zmax - fiber_stick - endo_stick)
    epi_outline = _slice_poly(epi_pv)
    fasc_polys = [pp for pp in (_slice_poly(fp) for fp in fasc_pv) if pp is not None]

    WS = (5400, 1420)          # high-res panel-a render (SSAA supersamples ~2x internally;
                               # larger windows overflow the GPU framebuffer with depth peeling)
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=WS)
    p.background_color = "white"
    p.enable_depth_peeling(60, occlusion_ratio=0.0)
    p.remove_all_lights()
    for pos, inten in [((7, 11, 13), 0.62), ((-9, 4, 7), 0.34), ((0, -7, -11), 0.30), ((-4, 9, -6), 0.26)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 0), color="#ffffff", intensity=inten)
        lt.positional = False
        p.add_light(lt)

    # endoneurium fascicles (telescoped, flat purple)
    for fp in fasc_pv:
        m = _clip_capped(fp, endo_cut, near)
        _flat(p, m, ENDO, 0.85, smooth=False)
    # fiber population, contained inside the fascicles
    fib_f = sample / "fibers.vtp"
    if fib_f.exists():
        fib = _read_mm(str(fib_f))
        r_tube = fiber_tube_um / 1000.0
        endo_surf = fasc_pv[0].copy()
        for m in fasc_pv[1:]:
            endo_surf = endo_surf.merge(m)
        ctr = np.asarray(fib.cell_centers().points)
        probe = pv.PolyData(np.column_stack([ctr[:, 0], ctr[:, 1], np.zeros(len(ctr))]))
        dist = np.asarray(probe.compute_implicit_distance(
            endo_surf.triangulate())["implicit_distance"])
        keep = np.where(np.abs(dist) > 1.2 * r_tube)[0]
        fib = fib.extract_cells(keep).extract_surface()
        if fiber_stride > 1:
            fib = fib.extract_cells(np.arange(0, fib.n_cells, fiber_stride)).extract_surface()
        _flat(p, fib.tube(radius=r_tube, n_sides=16, capping=True), FIBER_COLOR, 1.0, sil=False)
    # epineurium sheath (telescoped shorter so the fascicles emerge)
    zlo = epi_pv.bounds[4]
    holes = [_offset_loop(h, EPS_MM) for h in fasc_polys]
    epi_m = _sheath_prism(epi_outline, holes, zlo, epi_cut)
    _flat(p, epi_m, EPI, 0.30)
    # cuff: analytic saline annulus + silicone tube + gold contact pads
    saline, silicone, _R = _cuff(epi_pv, epi_outline)
    _flat(p, saline, SALINE, 0.20, sil=False)
    _flat(p, silicone, SILICONE, 0.34); _ink(p, silicone, lw=1.8)
    for pad, role in _contact_pads(sample, 0.12):
        _gold(p, pad)

    # flat cel-shaded look (matches fig2a/fig7a): NO SSAO ambient-occlusion shading,
    # which otherwise gives a glassy/realistic appearance inconsistent with the
    # comic renders used elsewhere.
    p.enable_anti_aliasing("ssaa")
    # flat side-on view: nerve long axis (z) spans the image horizontally -> very wide landscape
    p.view_vector((0.92, 0.28, 0.12), viewup=(0, 1, 0))
    p.reset_camera(); p.camera.zoom(2.4)      # fill the frame (was 1.3 -> ~1/3 fill, wasted pixels)
    ppmm = px_per_mm(p, (0.0, 0.0, 0.0))      # nerve centred at origin (mm)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out_png), window_size=WS)
    write_ppmm(out_png, ppmm)
    p.close()
    from PIL import Image
    im = Image.open(out_png).convert("RGB"); a = np.asarray(im)
    ys, xs = np.where((a < 248).any(2))
    if len(xs):
        pad = 18
        im.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                 min(im.width, int(xs.max()) + pad), min(im.height, int(ys.max()) + pad))).save(out_png)
    print(f"  wrote {out_png}  ({len(fasc_pv)} fascicles, {ppmm:.1f} px/mm)")


def main():
    D = ROOT / "results_golgi/duke_meshes"
    O = ROOT / "paper_figs/out/renders/popnerve"
    render(D / "sub-4_sam-3", O / "swine.png")
    render(D / "human_sub-50_sam-2", O / "human.png")
    print("done")


if __name__ == "__main__":
    main()
