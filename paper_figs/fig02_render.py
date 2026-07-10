# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 3 (modeling setup) renders, PyVista. Produces transparent cropped PNGs:
  _fig3_model  : nerve (endo+epi) in 12-contact cuff (translucent silicone +
                 gold contacts) with fascicular fibers
  _fig3_mesh   : clipped multi-region tetrahedral mesh colored by material
Shared by fig02_setup.py (which adds the Ve / activating-function plots).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pyvista as pv
from PIL import Image
import sys

pv.OFF_SCREEN = True
ROOT = Path(__import__("os").environ.get("GOLGI_PAPER_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import TMP   # noqa: E402
ND = ROOT / "results_golgi/duke_meshes/sub-4_sam-3"

MAT = {1: ("endoneurium", "#fb9a4b"), 2: ("saline", "#a6cee3"),
       3: ("silicone (cuff)", "#d9d9d9"), 4: ("muscle", "#fb9a99"),
       5: ("epineurium", "#fdd870")}
CONTACT_COL = "#525252"


def _poly(vtp):
    s = pv.read(vtp); return pv.PolyData(s.points * 1e3, s.faces)


def fibers(n_show=120, color="fascicle"):
    f = np.load(ND / "nerve_paths_fibers.npz", allow_pickle=True)
    flat, lens = f["paths_flat"] * 1e3, f["path_lengths"]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    br = f["branch_idx"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(lens), size=min(n_show, len(lens)), replace=False)
    pts, lines, scal, base = [], [], [], 0
    for i in idx:
        p = flat[off[i]:off[i + 1]]
        if len(p) < 3:
            continue
        pts.append(p); lines.append(np.concatenate([[len(p)], np.arange(base, base + len(p))]))
        scal += [int(br[i]) % 20] * len(p); base += len(p)
    poly = pv.PolyData(np.vstack(pts)); poly.lines = np.hstack(lines)
    poly["s"] = np.array(scal, float)
    return poly.tube(radius=0.025, n_sides=6)


def _cam(pl, bounds, az=22, el=16):
    span = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    c = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
    axis = int(np.argmax(span)); cr = [i for i in range(3) if i != axis]
    curv = cr[int(np.argmax(span[cr]))]; depth = cr[1] if curv == cr[0] else cr[0]
    e = np.eye(3); d = 2.2 * span[axis]
    pos = c + e[depth] * d + e[axis] * 0.30 * d + e[curv] * 0.22 * d
    pl.camera_position = [tuple(pos), tuple(c), tuple(e[curv])]
    pl.enable_parallel_projection(); pl.camera.zoom(1.5)


def _save(pl, key):
    img = pl.screenshot(transparent_background=True, return_img=True); pl.close()
    im = Image.fromarray(img); bb = im.getbbox()
    if bb:
        im = im.crop(bb)
    p = TMP / f"_fig3_{key}.png"; im.save(p)
    print(f"  {key}: {im.size}")
    return p


def render_model():
    pl = pv.Plotter(off_screen=True, window_size=(1500, 850), border=False)
    pl.set_background("white")
    sil = _poly(ND / "surf_silicone.vtp")
    pl.add_mesh(sil, color=MAT[3][1], opacity=0.10, smooth_shading=True)
    pl.add_mesh(_poly(ND / "surf_epi.vtp"), color=MAT[5][1], opacity=0.12, smooth_shading=True)
    pl.add_mesh(_poly(ND / "surf_endo.vtp"), color=MAT[1][1], opacity=0.22, smooth_shading=True)
    pl.add_mesh(fibers(), scalars="s", cmap="tab20", show_scalar_bar=False, smooth_shading=True)
    for i in range(12):
        f = ND / f"surf_contact_{i:02d}.vtp"
        if f.exists():
            pl.add_mesh(_poly(f), color=CONTACT_COL, opacity=0.97, smooth_shading=True,
                        specular=0.6, specular_power=20)
    pl.enable_depth_peeling(14)
    _cam(pl, sil.bounds)
    return _save(pl, "model")


def render_mesh():
    """Longitudinal section of the multi-region mesh (x-z plane through the
    nerve axis): shows the cuff, its 3 contact rows, nerve, saline, silicone
    and muscle along the nerve length. Cells colored by material."""
    m = pv.read(ND / "mesh_clip.vtu")
    m.points = m.points * 1e3                       # mm
    sl = m.slice(normal="y", origin=(0.0, 0.0, 0.0))   # x-z plane (passes phi=0/180 contacts)
    sl = sl.clip_box([-2.6, 2.6, -0.5, 0.5, -6.0, 6.0], invert=False)
    pl = pv.Plotter(off_screen=True, window_size=(1500, 650), border=False)
    pl.set_background("white")
    reg = sl["region"]
    for tag, (name, col) in MAT.items():
        ids = np.where(reg == tag)[0]
        if len(ids):
            pl.add_mesh(sl.extract_cells(ids), color=col, show_edges=True,
                        edge_color="black", line_width=0.4, show_scalar_bar=False)
    ids = np.where(reg >= 100)[0]
    if len(ids):
        pl.add_mesh(sl.extract_cells(ids), color=CONTACT_COL, show_edges=True,
                    edge_color="black", line_width=0.4, show_scalar_bar=False)
    # z horizontal (nerve axis), x vertical, looking along y
    pl.camera_position = [(0, -60, 0), (0, 0, 0), (1, 0, 0)]
    pl.enable_parallel_projection(); pl.reset_camera(); pl.camera.zoom(1.32)
    return _save(pl, "mesh")


if __name__ == "__main__":
    print("rendering fig3 components ...")
    render_model()
    render_mesh()
