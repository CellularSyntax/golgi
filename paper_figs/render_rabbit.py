# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cel-shaded render of the real-3D rabbit cervical vagus for Fig 6a — same comic
style as the dog-VNS / electrode-gallery renders, but with the curved fiber
trajectories kept and colored by destination branch (SCB = orange, trunk = blue).

Everything is in the raw micro-CT frame (mm): the nerve surface and the 468 fiber
paths live there. Each fiber is assigned to the trunk-continuation or SCB branch by
its nearest distal cap (nerve_paths_caps.json). The ring-array cuff is placed on the
nerve centerline 40 mm from the trunk end (cuff_offset_mm) along the local tangent."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import px_per_mm, write_ppmm  # noqa: E402
RAB = ROOT / "paper_figs/rabbit_out"
GUI = ROOT / "paper_figs/out/_intermediate/rabbit_gui_study"
OUT = ROOT / "paper_figs/out/renders/rabbit_render.png"

SCB, TRUNK = "#e6550d", "#1f77b4"     # branch 1 = SCB (orange), branch 0 = trunk (blue)
NERVE = "#e3a7c6"                      # rose epineurium
SILICONE = "#b7c0cc"                  # neutral grey cuff (won't clash with blue trunk fibers)
GOLD = "#f1c62f"
INK = "#1b1d22"
SIL = dict(color=INK, line_width=3.0)


def _flat(p, mesh, color, opacity, sil=True):
    p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, ambient=0.46,
               diffuse=0.66, specular=0.0, show_scalar_bar=False,
               silhouette=SIL if sil else None)


def load_surface():
    s = np.load(RAB / "nerve_only_surface.npz", allow_pickle=True)
    pts = s["pts_raw"] * 1e3
    tris = s["tris"].astype(np.int64)
    faces = np.hstack([np.full((len(tris), 1), 3, np.int64), tris]).ravel()
    return pv.PolyData(pts, faces)


def load_fibers_by_branch():
    f = np.load(RAB / "nerve_paths_fibers.npz", allow_pickle=True)
    flat, lens = f["paths_flat"] * 1e3, f["path_lengths"].astype(int)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    caps = json.loads((RAB / "nerve_paths_caps.json").read_text())
    bc = np.asarray(caps["branch_cap_centroids_m"]) * 1e3            # [2,3] mm (0=trunk cont, 1=SCB)
    groups = {0: [], 1: []}
    for i in range(len(lens)):
        p = flat[off[i]:off[i + 1]]
        if len(p) < 3:
            continue
        distal = p[np.argmax(p[:, 2])]                              # branched end = high z
        b = int(np.argmin(np.linalg.norm(bc - distal, axis=1)))
        groups[b].append(p)
    return groups


def fiber_tube(fibers, radius=0.016, n=8):   # 10x scale fix (was 0.16)
    pts, lines, base = [], [], 0
    for p in fibers:
        k = len(p); pts.append(p)
        lines.append(np.concatenate([[k], np.arange(base, base + k)])); base += k
    poly = pv.PolyData(np.vstack(pts)); poly.lines = np.hstack(lines)
    return poly.tube(radius=radius, n_sides=n)


def centerline(surf, nbin=240):
    z = surf.points[:, 2]
    edges = np.linspace(z.min(), z.max(), nbin)
    cen = []
    for i in range(nbin - 1):
        m = (z >= edges[i]) & (z < edges[i + 1])
        if m.sum() > 8:
            q = surf.points[m]
            cen.append([q[:, 0].mean(), q[:, 1].mean(), 0.5 * (edges[i] + edges[i + 1])])
    return np.asarray(cen)


def cuff_meshes(center, axis, R_ci=0.3616, R_co=0.4616, L=1.0):   # 10x scale fix
    a = axis / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, ref); u /= np.linalg.norm(u); v = np.cross(a, u)
    M = np.column_stack([u, v, a])

    def tow(mesh):
        mesh.points = center + mesh.points @ M.T
        return mesh
    sil = pv.Disc(center=(0, 0, -L / 2), inner=R_ci, outer=R_co, normal=(0, 0, 1),
                  r_res=2, c_res=160).extrude((0, 0, L), capping=True)
    pats = []
    ec = json.loads((GUI / "configs/cfg_02/electrode_config.json").read_text())
    for pc in ec["patches"]:
        zc, dz = float(pc["z"]) * 1e3, float(pc["dz"]) * 1e3
        phi, dphi = float(pc["phi"]), float(pc["dphi"])
        ph = np.linspace(phi - dphi / 2, phi + dphi / 2, 16)
        zz = np.linspace(zc - dz / 2, zc + dz / 2, 4)
        P, Z = np.meshgrid(ph, zz)
        sg = pv.StructuredGrid(R_ci * np.cos(P), R_ci * np.sin(P), Z).extract_surface()
        pats.append(sg)
    return tow(sil), [tow(pp) for pp in pats]


def main():
    surf = load_surface()
    if surf.n_cells > 180000:
        surf = surf.decimate(1 - 180000 / surf.n_cells).extract_surface()
    groups = load_fibers_by_branch()
    cl = centerline(surf)
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    ic = int(np.searchsorted(arc, 4.0)); ic = min(max(ic, 1), len(cl) - 2)   # 10x scale fix (was 40)
    # refit the cuff to the LOCAL nerve (centroid + PCA principal axis of the surface
    # patch around the cuff) so it aligns with the nerve's local axis, like golgi's refit
    near = np.linalg.norm(surf.points - cl[ic], axis=1) < 0.6   # 10x scale fix (was 6.0)
    q = surf.points[near]; ccen = q.mean(0); qc = q - ccen
    cax = np.linalg.svd(qc, full_matrices=False)[2][0]
    if cax[2] < 0:
        cax = -cax
    r_loc = float(np.linalg.norm(qc - np.outer(qc @ cax, cax), axis=1).max())
    sil, pats = cuff_meshes(ccen, cax)
    print(f"fibers: trunk={len(groups[0])} SCB={len(groups[1])} | cuff @ {ccen.round(1)} "
          f"axis {cax.round(2)} | local nerve r~{r_loc:.2f}mm (R_ci 3.62)")

    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=(2800, 1000))
    p.background_color = "white"
    p.enable_depth_peeling(40, occlusion_ratio=0.0)
    p.remove_all_lights()
    for pos, inten in [((6, 12, 10), 0.6), ((-8, 5, 7), 0.34), ((0, -8, 9), 0.3), ((2, 6, -10), 0.26)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 0), color="#ffffff", intensity=inten)
        lt.positional = False
        p.add_light(lt)
    # translucent nerve + branch-colored fiber tubes + cuff
    _flat(p, surf, NERVE, 0.22)
    _flat(p, fiber_tube(groups[0]), TRUNK, 1.0, sil=False)
    _flat(p, fiber_tube(groups[1]), SCB, 1.0, sil=False)
    _flat(p, sil, SILICONE, 0.34)
    for pp in pats:
        p.add_mesh(pp, color=GOLD, opacity=1.0, smooth_shading=True, ambient=0.55,
                   diffuse=0.55, specular=0.06, show_scalar_bar=False, silhouette=SIL)

    # frame: long (z) axis horizontal, look along the thinnest dim with a slight tilt
    b = surf.bounds
    span = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    axis = int(np.argmax(span)); cr = [i for i in range(3) if i != axis]
    curv = cr[int(np.argmax(span[cr]))]; depth = cr[1] if curv == cr[0] else cr[0]
    e = np.eye(3); dist = 1.8 * span[axis]
    pos = center + e[depth] * dist + e[axis] * 0.04 * dist + e[curv] * 0.10 * dist
    p.camera_position = [tuple(pos), tuple(center), tuple(e[curv])]
    p.enable_anti_aliasing("ssaa")
    p.camera.zoom(1.85)
    ppmm = px_per_mm(p, tuple(center))         # nerve centre (mm)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(OUT), window_size=(2800, 1000))
    write_ppmm(OUT, ppmm)
    p.close()
    from PIL import Image
    im = Image.open(OUT).convert("RGB"); a = np.asarray(im)
    ys, xs = np.where((a < 248).any(2))
    if len(xs):
        pad, pad_bot = 16, 52        # extra bottom margin leaves white space for the scale bar
        im.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                 min(im.width, int(xs.max()) + pad), min(im.height, int(ys.max()) + pad_bot))).save(OUT)
    print(f"wrote {OUT}  ({ppmm:.1f} px/mm)")


if __name__ == "__main__":
    main()
