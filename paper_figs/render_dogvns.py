# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Comic / cel-shaded device illustration of the dog-VNS validation setup
(Fig 8d companion). Reads the ACTUAL FEM run (dogvns_ascent/nerve_none).

Style: flat (non-PBR) shading + thin dark silhouette contour lines, like a clean
schematic / comic panel. The nerve (epineurium sheath + endoneurium core) keeps
its REAL meshed elliptical shape; everything else is built from clean analytic
primitives: two silicone collars (Disc-extruded tubes), two gold ring contacts,
saline annuli. The nerve is telescoped at +z (endo emerges past epi, the centroid
fiber past endo) so the layering reads at a glance."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "scripts"))
from render_components import (_read_mm, _clip_capped, _sheath_prism, _slice_poly,  # noqa: E402
                               _offset_loop, EPS_MM, STYLE, FIBER_COLOR, CONTACT_COLOR)

NERVE = ROOT / "paper_figs/out/_intermediate/dogvns_ascent/nerve_none"
OUT = ROOT / "paper_figs/out/renders/dogvns_render.png"
# telescoping window (mm): endo emerges past epi at +z; fiber past endo
BOT, TOP_EPI, TOP_ENDO, FIB_TOP = -7.5, 5.8, 8.2, 9.5
SIL = dict(color="#121419", line_width=6.0)          # comic contour lines
INK = "#121419"


def _ink_edges(p, mesh, lw=3.2, fa=18):
    """Crisp dark feature/rim lines on a hard-edged analytic part (comic edges)."""
    e = mesh.extract_feature_edges(feature_angle=fa, boundary_edges=True,
                                   non_manifold_edges=False, feature_edges=True,
                                   manifold_edges=False)
    if e.n_cells:
        p.add_mesh(e, color=INK, line_width=lw, render_lines_as_tubes=True,
                   show_scalar_bar=False)


def centroid_fiber_mm():
    """The fiber nearest the fascicle centroid (the thresholded centroid fiber)."""
    d = np.load(NERVE / "nerve_paths_fibers.npz", allow_pickle=True)
    flat = np.asarray(d["paths_flat"], float) * 1000.0
    lens = np.asarray(d["path_lengths"], int)
    off = np.concatenate([[0], np.cumsum(lens)])
    mids = np.array([flat[off[i]:off[i + 1]][lens[i] // 2, :2] for i in range(len(lens))])
    c = mids.mean(0)
    bic = int(np.argmin(np.hypot(mids[:, 0] - c[0], mids[:, 1] - c[1])))
    return flat[off[bic]:off[bic + 1]]


def _slab(m, lo, hi):
    """Clip a closed shell to z in [lo, hi], capping both cut faces flat."""
    return _clip_capped(_clip_capped(m, hi, 1.0), lo, -1.0)


def _ring(R, thk, z0, z1, n=220):
    """Clean analytic annular band (a contact or a silicone collar): a washer
    (inner R, outer R+thk) extruded over the axial span [z0, z1]."""
    disc = pv.Disc(center=(0, 0, z0), inner=R, outer=R + thk, normal=(0, 0, 1),
                   r_res=2, c_res=n)
    return disc.extrude((0, 0, z1 - z0), capping=True)


def _flat(p, mesh, color, opacity, sil=True):
    """Flat (comic) fill: high ambient, no specular highlight, dark contour."""
    p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True,
               ambient=0.46, diffuse=0.66, specular=0.0, show_scalar_bar=False,
               silhouette=SIL if sil else None)


def main():
    W, H = 2600, 1500
    # ---- meshed nerve (real elliptical shape), telescoped ----
    endo = _slab(_read_mm(str(NERVE / "surf_endo.vtp")), BOT, TOP_ENDO)
    epi_raw = _read_mm(str(NERVE / "surf_epi.vtp"))
    epi = _slab(epi_raw, BOT, TOP_EPI)
    epi_outline = _slice_poly(epi_raw)

    # ---- analytic cuff geometry from the real silicone extents ----
    sil_raw = _read_mm(str(NERVE / "surf_silicone.vtp"))
    pr = np.asarray(sil_raw.points)
    rr = np.hypot(pr[:, 0], pr[:, 1])
    R_ci, R_co = float(rr.min()), float(rr.max())
    zc = pr[:, 2]
    bands = [(float(zc[zc > 0].min()), float(zc[zc > 0].max())),
             (float(zc[zc < 0].min()), float(zc[zc < 0].max()))]
    th = np.linspace(0, 2 * np.pi, 220, endpoint=False)
    circle = np.column_stack([(R_ci - EPS_MM) * np.cos(th), (R_ci - EPS_MM) * np.sin(th)])
    hole = _offset_loop(epi_outline, EPS_MM)
    saline = [_sheath_prism(circle, [hole], z0, z1) for z0, z1 in bands]
    silicone = [_ring(R_ci, R_co - R_ci, z0, z1) for z0, z1 in bands]
    cfg = json.loads((NERVE / "electrode_config.json").read_text())
    contacts = [_ring(c["R"] * 1e3, 0.14, c["z"] * 1e3 - c["dz"] * 1e3 / 2,
                      c["z"] * 1e3 + c["dz"] * 1e3 / 2) for c in cfg["patches"]]

    cf = centroid_fiber_mm()
    cf = cf[(cf[:, 2] >= BOT) & (cf[:, 2] <= FIB_TOP)]
    fib = pv.lines_from_points(cf).tube(radius=0.085, n_sides=20, capping=True)

    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=(W, H))
    p.background_color = "white"
    p.enable_depth_peeling(60, occlusion_ratio=0.0)
    p.remove_all_lights()
    for pos, inten, col in [((7, 11, 13), 0.62, "#ffffff"), ((-9, 4, 7), 0.34, "#ffffff"),
                            ((0, -7, -11), 0.30, "#ffffff"), ((-4, 9, -6), 0.26, "#ffffff")]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 0), color=col, intensity=inten)
        lt.positional = False
        p.add_light(lt)

    # endoneurium (purple core; emerges past epi at +z) + centroid fiber
    _flat(p, endo, STYLE["endo"][1], 0.86)
    _flat(p, fib, FIBER_COLOR, 1.0)
    # epineurium sheath (light translucent rose so the purple endo shows through)
    _flat(p, epi, STYLE["epi"][1], 0.32)
    # saline infill (faint cyan, no contour to keep it clean)
    for s in saline:
        _flat(p, s, STYLE["surf_saline"][1], 0.22, sil=False)
    # silicone collars (translucent blue glass, flat) — two separated collars
    for s in silicone:
        _flat(p, s, STYLE["surf_silicone"][1], 0.26)
        _ink_edges(p, s, lw=2.6)
    # gold ring contacts (clean flat gold, solid, dark contour rims)
    for c in contacts:
        _flat(p, c, CONTACT_COLOR, 1.0)
        _ink_edges(p, c, lw=3.4)

    p.enable_anti_aliasing("ssaa")
    p.view_vector((0.62, 0.5, 0.86), viewup=(0, 1, 0))
    p.reset_camera()
    p.camera.zoom(1.55)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(OUT), window_size=(W, H))
    p.close()
    # tight-crop the white margin so it sits cleanly when embedded
    from PIL import Image
    im = Image.open(OUT).convert("RGB")
    a = np.asarray(im)
    ys, xs = np.where((a < 248).any(2))
    if len(xs):
        pad = 26
        im.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                 min(im.width, int(xs.max()) + pad),
                 min(im.height, int(ys.max()) + pad))).save(OUT)
    print(f"wrote {OUT}  R_ci={R_ci:.2f} R_co={R_co:.2f} bands={[(round(a,2),round(b,2)) for a,b in bands]}")


if __name__ == "__main__":
    main()
