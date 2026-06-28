# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cel-shaded render of the NEW real-3D human cervical vagus for fig8 panel a, matching the
fig5-8 render style AND the rabbit fig7a small-multiples layout:
  * new_human_setup.png            — the BEST cuff only, full detail (4x5 array + tripole), big.
  * new_human_setup_thumb_<tag>.png — one tiny render per swept position (nerve + cuff at that z),
                                     same camera, so the cuff visibly steps along the nerve.
New STLs + new fibers are co-registered in the trajectory (mm) frame; each cuff is refit to the
local nerve (centroid + PCA) and placed at that position's FEM lead-field |Ve|-peak z. Images
are mirrored (branch on the RIGHT) to match fig5-8.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
from pathlib import Path
import numpy as np
import pyvista as pv
from PIL import Image

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import px_per_mm, write_ppmm   # noqa: E402

NHM = ROOT / "data/new_human_meshes"
EPI = NHM / "EPINERIUM_Epinerium_cleaned_aligned_masks_mm_SMOOTHED_LONGER_original_duplicate_duplicate_wrapped_duplicate.stl"
ENDO = NHM / "ENDO_dec80_fixed.stl"
FB = ROOT / "paper_figs/new_human3d_out/nerve_paths_branch.npz"
DD = ROOT / "paper_figs/out/data"
OUT = ROOT / "paper_figs/out/renders/new_human_setup.png"

BEST_TAG = os.environ.get("BEST_TAG", "off15_4x5")
BEST_CFG = os.environ.get("BEST_CFG", "long_tripole")
# (sweep tag, distance-from-branch mm). 30 mm (tag 4x5) dropped (intermixed cranial trunk).
POSITIONS = [("off15_4x5", 20), ("off22_4x5", 13), ("off27_4x5", 8)]
ROWS_Z = np.array([-1.5, -0.5, 0.5, 1.5]); N_COL = 5
DZ, DPHI = 0.6, np.radians(30.0)
L_CUFF, CLEAR, WALL_VIS = 5.0, 0.4, 0.25

SCB, TRUNK = "#e6550d", "#1f77b4"
NERVE, ENDO_C = "#e3a7c6", "#dcb3cd"
SILICONE, GOLD, INK = "#b7c0cc", "#f1c62f", "#1b1d22"
CATH, ANODE = "#2ca02c", "#7b3294"
SIL = dict(color=INK, line_width=3.0)
BEST_META = DD / "new_human_tripole_sweep" / BEST_TAG / "meta.json"


def _flat(p, mesh, color, opacity, sil=True):
    p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, ambient=0.46,
               diffuse=0.66, specular=0.0, show_scalar_bar=False,
               silhouette=SIL if sil else None)


def load_surface(path, cap=180000):
    s = pv.read(str(path)).extract_surface().triangulate()
    if s.n_cells > cap:
        s = s.decimate(1 - cap / s.n_cells).extract_surface()
    return s


def load_fibers():
    f = np.load(FB, allow_pickle=True)
    flat, lens, br = f["paths_flat"] * 1e3, f["path_lengths"].astype(int), f["branch_idx"]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    groups = {0: [], 1: []}
    for i in range(len(lens)):
        p = flat[off[i]:off[i + 1]]
        if len(p) >= 3:
            groups[int(br[i])].append(p)
    return groups, flat, lens, off


def fiber_tube(fibers, radius=0.045, n=8):
    pts, lines, base = [], [], 0
    for p in fibers:
        k = len(p); pts.append(p)
        lines.append(np.concatenate([[k], np.arange(base, base + k)])); base += k
    poly = pv.PolyData(np.vstack(pts)); poly.lines = np.hstack(lines)
    return poly.tube(radius=radius, n_sides=n)


def centerline(surf, axis=2, nbin=260):
    t = surf.points[:, axis]
    edges = np.linspace(t.min(), t.max(), nbin); cen = []
    for i in range(nbin - 1):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() > 8:
            cen.append(surf.points[m].mean(0))
    return np.asarray(cen)


def cuff_z_of(tag, flat_traj, off):
    """Median trajectory-z of the per-fiber lead-field |Ve| peak (same fiber order)."""
    lf = np.load(DD / f"new_human_branch_{tag}/paths_Ve.npz", allow_pickle=True)
    Vt = np.abs(lf["Ve_mat"]).max(1)
    fz = flat_traj[:, 2]
    zs = [fz[off[i] + int(np.argmax(Vt[off[i]:off[i + 1]]))] for i in range(len(off) - 1)]
    return float(np.median(zs))


def frame_at(surf, cl, cuff_z, ball=4.0):
    ic = int(np.argmin(np.abs(cl[:, 2] - cuff_z))); ic = min(max(ic, 1), len(cl) - 2)
    near = np.linalg.norm(surf.points - cl[ic], axis=1) < ball
    q = surf.points[near] if near.sum() >= 8 else surf.points
    ccen = q.mean(0); qc = q - ccen
    cax = np.linalg.svd(qc, full_matrices=False)[2][0]
    if cax[2] < 0:
        cax = -cax
    r_loc = float(np.percentile(np.linalg.norm(qc - np.outer(qc @ cax, cax), axis=1), 99))
    return ccen, cax, r_loc


def basis(axis):
    a = axis / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, ref); u /= np.linalg.norm(u); v = np.cross(a, u)
    return np.column_stack([u, v, a])


def silicone_cyl(center, axis, R_ci, R_co, L):
    M = basis(axis)
    s = pv.Disc(center=(0, 0, -L / 2), inner=R_ci, outer=R_co, normal=(0, 0, 1),
                r_res=2, c_res=160).extrude((0, 0, L), capping=True)
    s.points = center + s.points @ M.T
    return s


def contacts_4x5(center, axis, R_ci):
    M = basis(axis); out = []
    for zc in ROWS_Z:
        for k in range(N_COL):
            phi = k * 2 * np.pi / N_COL
            ph = np.linspace(phi - DPHI / 2, phi + DPHI / 2, 16)
            zz = np.linspace(zc - DZ / 2, zc + DZ / 2, 4)
            P, Z = np.meshgrid(ph, zz)
            sg = pv.StructuredGrid(R_ci * np.cos(P), R_ci * np.sin(P), Z).extract_surface()
            sg.points = center + sg.points @ M.T
            out.append(sg)
    return out


def add_cuff(p, frame, cath, anodes, show_tripole):
    ccen, cax, r_loc = frame
    R_ci = r_loc + CLEAR; R_co = R_ci + WALL_VIS
    p.add_mesh(silicone_cyl(ccen, cax, R_ci, R_co, L_CUFF), color=SILICONE, opacity=0.40,
               smooth_shading=True, ambient=0.46, diffuse=0.66, specular=0.0,
               show_scalar_bar=False, silhouette=SIL)
    for i, c in enumerate(contacts_4x5(ccen, cax, R_ci)):
        col = (CATH if i == cath else ANODE if i in anodes else GOLD) if show_tripole else GOLD
        p.add_mesh(c, color=col, opacity=1.0, smooth_shading=True, ambient=0.55,
                   diffuse=0.55, specular=0.06, show_scalar_bar=False, silhouette=SIL)


def scene(surf, endo, tube0, tube1, wsize):
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=wsize)
    p.background_color = "white"
    p.enable_depth_peeling(40, occlusion_ratio=0.0)
    p.remove_all_lights()
    for pos, inten in [((6, 12, 10), 0.6), ((-8, 5, 7), 0.34), ((0, -8, 9), 0.3), ((2, 6, -10), 0.26)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 0), color="#ffffff", intensity=inten)
        lt.positional = False; p.add_light(lt)
    _flat(p, surf, NERVE, 0.11)
    _flat(p, endo, ENDO_C, 0.18, sil=False)
    _flat(p, tube0, TRUNK, 1.0, sil=False)
    _flat(p, tube1, SCB, 1.0, sil=False)
    return p


def set_cam(p, surf):
    b = surf.bounds
    span = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    axL = int(np.argmax(span)); cr = [i for i in range(3) if i != axL]
    curv = cr[int(np.argmax(span[cr]))]; depth = cr[1] if curv == cr[0] else cr[0]
    e = np.eye(3); cdist = 1.8 * span[axL]
    pos = center + e[depth] * cdist + e[axL] * 0.04 * cdist + e[curv] * 0.10 * cdist
    p.camera_position = [tuple(pos), tuple(center), tuple(e[curv])]
    p.enable_anti_aliasing("ssaa"); p.camera.zoom(1.85)


def crop_save(p, out_path, wsize, flip=True):
    # FLIP L-R so the SCB bifurcation is on the RIGHT, matching the rabbit fig7a (branch on the
    # right, trunk on the left). This human nerve renders natively with the branch on the LEFT,
    # so the flip is needed for fig7/fig8 to read the same way.
    p.screenshot(str(out_path), window_size=wsize, transparent_background=True)
    im = Image.open(out_path).convert("RGBA"); a = np.asarray(im)
    ys, xs = np.where(a[:, :, 3] > 8)
    if len(xs):
        pad = 16
        im = im.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                      min(im.width, int(xs.max()) + pad), min(im.height, int(ys.max()) + pad)))
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(out_path)


def make_parts(ppmm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    od = OUT.parent
    items = [(NERVE, "epineurium"), (ENDO_C, "endoneurium"), (SCB, "SCB axons"), (TRUNK, "trunk axons"),
             (SILICONE, "cuff (silicone)"), (GOLD, "contact"), (CATH, "cathode"), (ANODE, "anode (guard)")]
    h = [Patch(facecolor=c, edgecolor="#333", label=l) for c, l in items]
    fig = plt.figure(figsize=(12, 0.55))
    fig.legend(handles=h, ncol=len(items), loc="center", frameon=False, fontsize=12)
    fig.savefig(od / "new_human_setup_legend.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    barmm = 10.0; barpx = barmm * ppmm
    fig = plt.figure(figsize=((barpx + 40) / 100.0, 0.7), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, barpx + 40); ax.set_ylim(0, 70)
    ax.plot([20, 20 + barpx], [24, 24], "k-", lw=5, solid_capstyle="butt")
    ax.text(20 + barpx / 2, 30, f"{barmm:g} mm", ha="center", va="bottom", fontsize=13)
    fig.savefig(od / "new_human_setup_scalebar.png", dpi=100, transparent=True)
    plt.close(fig)
    print(f"  parts: legend + scalebar (10 mm @ {ppmm:.1f} px/mm) -> {od}", flush=True)


def main():
    surf = load_surface(EPI); endo = load_surface(ENDO)
    groups, flat_traj, lens, off = load_fibers()
    tube0, tube1 = fiber_tube(groups[0]), fiber_tube(groups[1])
    cl = centerline(surf)
    frames = {tag: frame_at(surf, cl, cuff_z_of(tag, flat_traj, off)) for tag, _ in POSITIONS}
    bm = json.loads(BEST_META.read_text())
    cath = bm["cathode"]
    anodes = set(bm["long_anodes"] if BEST_CFG == "long_tripole" else
                 bm["trans_anodes"] if BEST_CFG == "trans_tripole" else [])
    print(f"fibers trunk={len(groups[0])} SCB={len(groups[1])}; best={BEST_TAG} cathode c{cath}", flush=True)

    # ---- MAIN: best cuff only, full detail, big ----
    p = scene(surf, endo, tube0, tube1, (3000, 1450)); set_cam(p, surf)
    add_cuff(p, frames[BEST_TAG], cath, anodes, show_tripole=True)
    ppmm = px_per_mm(p, tuple(np.array(surf.center)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    crop_save(p, OUT, (3000, 1050)); write_ppmm(OUT, ppmm); p.close()
    print(f"wrote {OUT} ({ppmm:.1f} px/mm) [best cuff, big]", flush=True)
    make_parts(ppmm)

    # ---- THUMBS: one per position, same camera, small ----
    for tag, dist in POSITIONS:
        pt = scene(surf, endo, tube0, tube1, (1500, 720)); set_cam(pt, surf)
        add_cuff(pt, frames[tag], cath, anodes, show_tripole=(tag == BEST_TAG))
        out = OUT.parent / f"new_human_setup_thumb_{tag}.png"
        crop_save(pt, out, (1500, 520)); pt.close()
        print(f"  thumb {tag}: {dist} mm from branch{' BEST' if tag == BEST_TAG else ''} -> {out.name}",
              flush=True)


if __name__ == "__main__":
    main()
