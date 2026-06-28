# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cel-shaded renders of the two Bucksot electrode setups (the actual FEM geometry):
the 5-fascicle rabbit-sciatic nerve in a 270-deg gapped bipolar cuff, with the
conductor gap up ("circumferential") vs rotated 180 deg ("inverted"). Same comic
style as the other setup renders. Writes bucksot_circ.png / bucksot_inverted.png.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from render_electrodes import _flat, EPI, SILICONE, GOLD, SIL  # noqa: E402
from fig_bucksot import FCOLORS  # noqa: E402  (same per-fascicle palette as the recruitment curves)
WORK = ROOT / "paper_figs/out/_intermediate/bucksot"
OUTDIR = ROOT / "paper_figs/out/renders"
ZCLIP = 7.6        # mm; show the cuff region + a little nerve protruding
ROW, C_R, C_W, C_SPAN = 5.0, 1.62, 1.5, 275.0   # contacts at z=+/-5 mm, R=1.62, 1.5 mm wide, 275 deg
PHI_C = {"circ": 0.0, "inverted": np.pi}         # conductor angular CENTRE (gap opposite)


def load(d, name):
    m = pv.read(d / f"{name}.vtp"); m.points = m.points * 1000.0
    # invert=False keeps the part INSIDE the box (this PyVista keeps OUTSIDE when True)
    return m.clip_box([-99, 99, -99, 99, -ZCLIP, ZCLIP], invert=False)


def arc_band(phi_c, zc):
    """Clean 275-deg conductor band (gap opposite phi_c) as a single-sided quad sheet
    (built as PolyData directly -> no coincident faces, so no z-fight speckle)."""
    half = np.radians(C_SPAN / 2.0)
    phi = np.linspace(phi_c - half, phi_c + half, 100)
    z = np.linspace(zc - C_W / 2, zc + C_W / 2, 6)
    nphi, nz = len(phi), len(z)
    P, Z = np.meshgrid(phi, z, indexing="ij")
    pts = np.column_stack([C_R * np.cos(P).ravel(), C_R * np.sin(P).ravel(), Z.ravel()])
    faces = []
    for i in range(nphi - 1):
        for j in range(nz - 1):
            a = i * nz + j; b = (i + 1) * nz + j; c = (i + 1) * nz + j + 1; e = i * nz + j + 1
            faces.append([4, a, b, c, e])
    return pv.PolyData(pts, np.hstack(faces).astype(np.int64))


def color_fascicles(p, endo, d):
    """Split the endoneurium into the 5 fascicles and colour each with its
    recruitment-curve colour (matched by centroid to fascicles.json id == branch_idx)."""
    fj = json.loads((d / "fascicles.json").read_text())["fascicles"]
    cents = {int(f["id"]): np.array(f["centroid_xy_m"]) * 1000 for f in fj}
    bodies = endo.split_bodies()
    for body in bodies:
        b = body.extract_surface()
        cxy = b.points[:, :2].mean(0)
        fid = min(cents, key=lambda i: np.linalg.norm(cents[i] - cxy))
        _flat(p, b, FCOLORS[fid % len(FCOLORS)], 0.62)
    return len(bodies)


def render(orient, out_png):
    d = WORK / orient
    epi, endo = load(d, "surf_epi"), load(d, "surf_endo")
    # clean analytic cuff wall (the faceted FEM silicone speckles the gold behind it)
    sil = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=2.62,
                      height=2 * ZCLIP * 0.92, resolution=140, capping=False)
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=(1700, 1500))
    p.background_color = "white"
    p.enable_depth_peeling(60, occlusion_ratio=0.0)
    p.remove_all_lights()
    for pos, inten in [((7, 11, 13), 0.62), ((-9, 4, 7), 0.34), ((0, -7, -11), 0.30),
                       ((-4, 9, -6), 0.26)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 0), color="#ffffff", intensity=inten)
        lt.positional = False
        p.add_light(lt)
    _flat(p, sil, SILICONE, 0.12)                 # faint clean cuff wall (consistent SIL silhouette)
    _flat(p, epi, EPI, 0.18)                       # translucent epineurium
    n_f = color_fascicles(p, endo, d)             # 5 fascicles, coloured per recruitment curve
    for zc in (-ROW, ROW):                         # two 275-deg contacts (gap shows the orientation)
        band = arc_band(PHI_C[orient], zc).compute_normals(auto_orient_normals=True)
        p.add_mesh(band, color=GOLD, opacity=1.0, smooth_shading=True, ambient=0.6,
                   diffuse=0.55, specular=0.12, show_scalar_bar=False, silhouette=SIL)
    p.enable_anti_aliasing("ssaa")
    p.view_vector((0.6, 0.42, 0.68), viewup=(0, 1, 0))    # 3/4 view; contact bands wrap the nerve, gap on the curve
    p.reset_camera(); p.camera.zoom(1.12)                 # modest zoom so the whole nerve+cuff stays in frame
    out_png.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out_png), window_size=(1700, 1500))
    p.close()
    from PIL import Image
    im = Image.open(out_png).convert("RGB"); a = np.asarray(im)
    ys, xs = np.where((a < 248).any(2))
    if len(xs):
        pad = 16
        im.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                 min(im.width, int(xs.max()) + pad), min(im.height, int(ys.max()) + pad))).save(out_png)
    print(f"wrote {out_png.name}")


def main():
    for orient in ("circ", "inverted"):
        render(orient, OUTDIR / f"bucksot_{orient}.png")


if __name__ == "__main__":
    main()
