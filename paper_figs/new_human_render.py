# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Presentation renders of the NEW full-length human multi-region mesh (4x5 cuff).
Three panels in the fig3 style + a region legend:
  a  full multi-region mesh (muscle sheath + nerve + cuff + contacts)
  b  nerve + electrode close-up, NO muscle (epi translucent, endo fascicles, cuff, contacts)
  c  longitudinal cross-section of the mesh in the cuff window (+/- a few mm), colored by region
Reads the snapshot mesh paper_figs/out/render_mesh/nerve_off15.msh (gmsh:physical tags 1-5).
Contacts (no mesh tag in the bundle path) are drawn from the 4x5 design layout on the cuff wall.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pyvista as pv
from PIL import Image

pv.OFF_SCREEN = True
ROOT = Path(__file__).parent.parent
MSH = ROOT / "paper_figs/out/render_mesh/nerve_off15.msh"
OUTD = ROOT / "paper_figs/out/figures/png"
OUTD.mkdir(parents=True, exist_ok=True)

MAT = {1: ("endoneurium", "#cf98b4"), 2: ("saline", "#a6cee3"),
       3: ("silicone (cuff)", "#d9d9d9"), 4: ("muscle", "#fb9a99"),
       5: ("epineurium", "#9cc183")}   # sage green: contrasts the #cf98b4 endoneurium
CONTACT_COL = "#2f2f2f"
NERVE_EDGE = "#a07e90"   # light mauve mesh edges: visible but let the flat #cf98b4 show
R_CI = 1.95
ROWS_Z = [-1.5, -0.5, 0.5, 1.5]
N_COL, PHI, W_AX = 5, 30.0, 0.6
FOCAL = (1.5, 0.4, -4.5)   # nerve center (mm)


def load():
    m = pv.read(MSH); m.points = m.points * 1e3
    m.cell_data["region"] = np.asarray(m.cell_data["gmsh:physical"])
    return m


def region_surf(m, tag):
    return m.extract_cells(np.where(m.cell_data["region"] == tag)[0]).extract_surface()


def region_surf_dec(m, tag, target=11000):
    """Decimated region surface for the 3D renders: the true endo boundary is ~77k faces,
    so its edges merge into a dark mass and hide the #cf98b4 color. A coarser surface shows
    a clean visible wireframe with the colour between triangles (the true fine mesh is in c)."""
    s = region_surf(m, tag)
    if s.n_cells > target:
        try:
            s = s.triangulate().decimate(1.0 - target / s.n_cells)
        except Exception:
            pass
    return s


def contact_polys(r=R_CI):
    out = []
    for z0 in ROWS_Z:
        for k in range(N_COL):
            c0 = k * 360.0 / N_COL
            th = np.radians(np.linspace(c0 - PHI / 2, c0 + PHI / 2, 12))
            P, F = [], []
            for z in (z0 - W_AX / 2, z0 + W_AX / 2):
                for t in th:
                    P.append([r * np.cos(t), r * np.sin(t), z])
            nth = len(th)
            for j in range(nth - 1):
                F += [4, j, j + 1, nth + j + 1, nth + j]
            out.append(pv.PolyData(np.array(P), np.array(F)))
    return out


def add_contacts(pl, r=R_CI):
    for c in contact_polys(r):
        pl.add_mesh(c, color=CONTACT_COL, smooth_shading=True, specular=0.5, specular_power=18)


def add_sheath(pl, surf, color, edge, fill_op=0.06, edge_op=0.9, lw=0.6):
    """Two passes: a FAINT volume fill + a PROMINENT wireframe, so the mesh edges read
    clearly while the low-opacity fill lets the inner structures show through."""
    pl.add_mesh(surf, color=color, opacity=fill_op, lighting=False)
    pl.add_mesh(surf, style="wireframe", color=edge, line_width=lw, opacity=edge_op, lighting=False)


def setcam(pl, pos, up=(1, 0, 0), zoom=1.4, focal=FOCAL):
    pl.camera_position = [pos, focal, up]      # up=x -> z horizontal
    pl.enable_parallel_projection(); pl.reset_camera(); pl.camera.zoom(zoom)


def setcam_horizontal(pl, focal=FOCAL, dist=70, zoom=1.5, tilt=0.16):
    """Nerve long axis (z) horizontal. `tilt` adds a small downward/oblique view so the
    translucent muscle block reads as a 3-D sheath (tilt=0 -> perfectly flat side view).
    view-up = +x keeps z horizontal on screen."""
    cx, cy, cz = focal
    pos = (cx + dist * tilt * 0.7, cy - dist, cz + dist * tilt)
    pl.camera_position = [pos, focal, (1, 0, 0)]
    pl.enable_parallel_projection(); pl.reset_camera(); pl.camera.zoom(zoom)


def crop(pl, key):
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(transparent_background=True, return_img=True); pl.close()
    im = Image.fromarray(img); bb = im.getbbox()
    if bb:
        im = im.crop(bb)
    p = OUTD / f"new_human_mesh_{key}.png"; im.save(p, dpi=(500, 500))
    print(f"  {key}: {im.size}px @500dpi -> {p.name}")


def _add_all_meshes(pl, m, zclip=None, mus_op=0.08, epi_op=0.16, endo_lw=0.15):
    """All region meshes at ORIGINAL resolution with visible edges: muscle (sheath),
    epineurium, saline+silicone cuff, endoneurium fascicles, platinum contacts."""
    def rs(tag):
        s = region_surf(m, tag)
        return s.clip_box(zclip, invert=False) if zclip else s
    pl.add_mesh(rs(4), color=MAT[4][1], opacity=mus_op, show_edges=True,
                edge_color="#d6a3a3", line_width=0.3)                    # muscle sheath
    pl.add_mesh(rs(5), color=MAT[5][1], opacity=epi_op, show_edges=True,
                edge_color="#5f7d45", line_width=0.18)                   # epineurium
    pl.add_mesh(rs(2), color=MAT[2][1], opacity=0.30, smooth_shading=True)   # saline
    pl.add_mesh(rs(3), color=MAT[3][1], opacity=0.28, smooth_shading=True)   # silicone cuff
    pl.add_mesh(rs(1), color=MAT[1][1], opacity=0.99, show_edges=True,
                edge_color=NERVE_EDGE, line_width=endo_lw)               # endo, full-res
    add_contacts(pl)
    pl.enable_depth_peeling(20)


def render_a(m):
    """Entire nerve length, perfectly horizontal, original-resolution mesh, all regions."""
    pl = pv.Plotter(off_screen=True, window_size=(6500, 1700), border=False)
    pl.set_background("white")
    # muscle clipped to a sheath snug around the nerve (the FEM block is much larger)
    mus = region_surf(m, 4).clip_box([-6, 9, -6, 8, -23, 14], invert=False)
    add_sheath(pl, mus, MAT[4][1], "#b85f5f", fill_op=0.04, edge_op=0.6, lw=0.6)
    add_sheath(pl, region_surf(m, 5), MAT[5][1], "#3f5a2a", fill_op=0.05, edge_op=0.95, lw=0.6)
    pl.add_mesh(region_surf(m, 2), color=MAT[2][1], opacity=0.30, smooth_shading=True)
    pl.add_mesh(region_surf(m, 3), color=MAT[3][1], opacity=0.26, smooth_shading=True)
    pl.add_mesh(region_surf(m, 1), color=MAT[1][1], opacity=1.0, show_edges=True,
                edge_color=NERVE_EDGE, line_width=0.15, lighting=False)
    add_contacts(pl)
    pl.enable_depth_peeling(20)
    setcam_horizontal(pl, focal=(1.5, 0.4, -4.5), dist=80, zoom=2.0)
    crop(pl, "a_full")


def render_b(m):
    """Same, zoomed and centered on the electrode."""
    pl = pv.Plotter(off_screen=True, window_size=(4400, 2500), border=False)
    pl.set_background("white")
    zc = [-9, 9, -9, 9, -9, 9]
    add_sheath(pl, region_surf(m, 4).clip_box([-6, 9, -6, 8, -9, 9], invert=False),
               MAT[4][1], "#b85f5f", fill_op=0.04, edge_op=0.55, lw=0.55)
    add_sheath(pl, region_surf(m, 5).clip_box(zc, invert=False), MAT[5][1], "#3f5a2a",
               fill_op=0.05, edge_op=0.95, lw=0.6)
    pl.add_mesh(region_surf(m, 2), color=MAT[2][1], opacity=0.30, smooth_shading=True)
    pl.add_mesh(region_surf(m, 3), color=MAT[3][1], opacity=0.12, smooth_shading=True)
    pl.add_mesh(region_surf(m, 1).clip_box(zc, invert=False), color=MAT[1][1],
                opacity=1.0, show_edges=True, edge_color=NERVE_EDGE, line_width=0.25, lighting=False)
    add_contacts(pl)
    pl.enable_depth_peeling(20)
    setcam_horizontal(pl, focal=(1.2, 0.3, 0.0), dist=22, zoom=2.7)   # centered on electrode
    crop(pl, "b_nerve_electrode")


def render_c(m):
    sl = m.slice(normal="y", origin=(0, 0, 0)).clip_box([-3.6, 3.6, -1, 1, -7.5, 7.5], invert=False)
    pl = pv.Plotter(off_screen=True, window_size=(5000, 2150), border=False)
    pl.set_background("white")
    reg = np.asarray(sl.cell_data["region"])
    for tag, (name, col) in MAT.items():
        ids = np.where(reg == tag)[0]
        if len(ids):
            pl.add_mesh(sl.extract_cells(ids), color=col, show_edges=True,
                        edge_color="black", line_width=0.3, show_scalar_bar=False)
    angs = [k * 360.0 / N_COL for k in range(N_COL)]
    for z0 in ROWS_Z:
        for xs, hit in [(+1, 0.0), (-1, 180.0)]:
            if min(abs(((a - hit + 180) % 360) - 180) for a in angs) > PHI / 2 + 6:
                continue
            r0, r1 = sorted([xs * (R_CI - 0.22), xs * (R_CI + 0.22)])
            pl.add_mesh(pv.Box([r0, r1, -0.25, 0.25, z0 - W_AX / 2 - 0.05, z0 + W_AX / 2 + 0.05]),
                        color=CONTACT_COL)
    pl.camera_position = [(0, -50, 0), (0, 0, 0), (1, 0, 0)]
    pl.enable_parallel_projection(); pl.reset_camera(); pl.camera.zoom(1.4)
    crop(pl, "c_xsection")


def make_legend():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    items = [(MAT[1][0], MAT[1][1]), (MAT[2][0], MAT[2][1]), (MAT[3][0], MAT[3][1]),
             (MAT[5][0], MAT[5][1]), (MAT[4][0], MAT[4][1]), ("platinum contact", CONTACT_COL)]
    h = [Patch(facecolor=c, edgecolor="#444", label=n) for n, c in items]
    fig = plt.figure(figsize=(9, 0.7)); fig.legend(handles=h, ncol=6, loc="center",
                                                    frameon=False, fontsize=11)
    p = OUTD / "new_human_mesh_legend.png"; fig.savefig(p, dpi=400, transparent=True,
                                                        bbox_inches="tight"); plt.close()
    print(f"  legend -> {p.name}")


if __name__ == "__main__":
    m = load()
    print("rendering new-human mesh figures ...")
    render_a(m); render_b(m); render_c(m); make_legend()
