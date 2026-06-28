# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cel-shaded render of the rabbit cervical vagus for the rabbit selectivity figure (fig7
panel a), matching the fig5-8 render style. Produces SMALL-MULTIPLES panel a:
  * rabbit_setup.png            — the BEST cuff only, full detail (4x5 array + tripole), big.
  * rabbit_setup_thumb_<tag>.png — one tiny render per swept position (nerve + cuff at that z),
                                   same camera, so the cuff visibly steps along the nerve.
The 3 mm cuff vs 1 mm position spacing makes 4 cuffs overlap in one frame, so instead of
ghosting/blobbing them we show each position in its own thumbnail. Everything is in the
rabbit_out import frame (metres); surface, fibers and FEM lead fields share it.

env: BEST_TAG (default off3_4x5), BEST_CFG (long_tripole|trans_tripole|mono).
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

RO = ROOT / "paper_figs/rabbit_out"
SURF = RO / "nerve_only_surface.npz"
DD = ROOT / "paper_figs/out/data"
SW = DD / "rabbit_tripole_sweep"
OUT = ROOT / "paper_figs/out/renders/rabbit_setup.png"

BEST_TAG = os.environ.get("BEST_TAG", "off3_4x5")
BEST_CFG = os.environ.get("BEST_CFG", "long_tripole")
# (sweep tag, cuff offset mm from proximal trunk end, distance-from-branch mm = 8 - offset)
POSITIONS = [("off3_4x5", 3, 5), ("off4_4x5", 4, 4), ("off5_4x5", 5, 3), ("off6_4x5", 6, 2)]
N_COL = 5
ROW_SEP, CONTACT_W, PHI_DEG = 0.6, 0.4, 30.0
ROWS_Z = (np.arange(4) - 1.5) * ROW_SEP
DZ, DPHI = CONTACT_W, np.radians(PHI_DEG)
L_CUFF, WALL_VIS, CLEAR = 3.0, 0.2, 0.15

SCB, TRUNK = "#e6550d", "#1f77b4"
NERVE = "#e3a7c6"
SILICONE, GOLD, INK = "#b7c0cc", "#f1c62f", "#1b1d22"
CATH, ANODE = "#2ca02c", "#7b3294"
SIL = dict(color=INK, line_width=2.5)
BEST_META = SW / BEST_TAG / "meta.json"


def _flat(p, mesh, color, opacity, sil=True):
    p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, ambient=0.46,
               diffuse=0.66, specular=0.0, show_scalar_bar=False,
               silhouette=SIL if sil else None)


def _faces(tris):
    return np.hstack([np.full((len(tris), 1), 3, np.int64), tris]).ravel()


def load_surface(cap=160000):
    d = np.load(SURF, allow_pickle=True)
    s = pv.PolyData(np.asarray(d["pts_raw"], float) * 1e3,
                    _faces(np.asarray(d["tris"], np.int64))).extract_surface().triangulate()
    if s.n_cells > cap:
        s = s.decimate(1 - cap / s.n_cells).extract_surface()
    return s


def load_fibers():
    g = np.load(RO / "nerve_paths_fibers.npz", allow_pickle=True)
    flat = np.asarray(g["paths_flat"], float) * 1e3                        # mm, original frame
    lens = np.asarray(g["path_lengths"], int)
    br = np.asarray(np.load(DD / f"rabbit_branch_{BEST_TAG}/paths_Ve.npz",
                            allow_pickle=True)["branch_idx"], int)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    groups = {0: [], 1: []}
    for i in range(len(lens)):
        p = flat[off[i]:off[i + 1]]
        if len(p) >= 3:
            groups[int(br[i])].append(p)
    return groups


def fiber_tube(fibers, radius=0.018, n=8):
    pts, lines, base = [], [], 0
    for p in fibers:
        k = len(p); pts.append(p)
        lines.append(np.concatenate([[k], np.arange(base, base + k)])); base += k
    poly = pv.PolyData(np.vstack(pts)); poly.lines = np.hstack(lines)
    return poly.tube(radius=radius, n_sides=n)


def centerline(surf, axis=2, nbin=120):
    t = surf.points[:, axis]
    edges = np.linspace(t.min(), t.max(), nbin); cen = []
    for i in range(nbin - 1):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() > 6:
            cen.append(surf.points[m].mean(0))
    return np.asarray(cen)


def frame_at(surf, cl, cuff_z, ball=1.2):
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


def contacts_4x5(center, axis, R_ci, phi0=0.0):
    M = basis(axis); out = []
    for zc in ROWS_Z:
        for k in range(N_COL):
            phi = phi0 + k * 2 * np.pi / N_COL
            ph = np.linspace(phi - DPHI / 2, phi + DPHI / 2, 16)
            zz = np.linspace(zc - DZ / 2, zc + DZ / 2, 4)
            P, Z = np.meshgrid(ph, zz)
            sg = pv.StructuredGrid(R_ci * np.cos(P), R_ci * np.sin(P), Z).extract_surface()
            sg.points = center + sg.points @ M.T
            out.append(sg)
    return out


def add_cuff(p, frame, cath, anodes, show_tripole):
    """Add a solid cuff (silicone + 4x5 contacts) at `frame`; colour the tripole if asked."""
    ccen, cax, r_loc = frame
    R_ci = r_loc + CLEAR; R_co = R_ci + WALL_VIS
    p.add_mesh(silicone_cyl(ccen, cax, R_ci, R_co, L_CUFF), color=SILICONE, opacity=0.40,
               smooth_shading=True, ambient=0.46, diffuse=0.66, specular=0.0,
               show_scalar_bar=False, silhouette=SIL)
    for i, c in enumerate(contacts_4x5(ccen, cax, R_ci)):
        col = (CATH if i == cath else ANODE if i in anodes else GOLD) if show_tripole else GOLD
        p.add_mesh(c, color=col, opacity=1.0, smooth_shading=True, ambient=0.55,
                   diffuse=0.55, specular=0.06, show_scalar_bar=False, silhouette=SIL)


def scene(surf, tube0, tube1, wsize):
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=wsize)
    p.background_color = "white"
    p.enable_depth_peeling(40, occlusion_ratio=0.0)
    p.remove_all_lights()
    for pos, inten in [((6, 12, 10), 0.6), ((-8, 5, 7), 0.34), ((0, -8, 9), 0.3), ((2, 6, -10), 0.26)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 0), color="#ffffff", intensity=inten)
        lt.positional = False; p.add_light(lt)
    _flat(p, surf, NERVE, 0.12)
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
    return center


def crop_save(p, out_path, wsize):
    p.screenshot(str(out_path), window_size=wsize, transparent_background=True)
    im = Image.open(out_path).convert("RGBA"); a = np.asarray(im)
    ys, xs = np.where(a[:, :, 3] > 8)
    if len(xs):
        pad = 14
        im = im.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                      min(im.width, int(xs.max()) + pad), min(im.height, int(ys.max()) + pad)))
    im.save(out_path)


def make_parts(ppmm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    od = OUT.parent
    items = [(NERVE, "epineurium"), (SCB, "SCB axons"), (TRUNK, "trunk axons"),
             (SILICONE, "cuff (silicone)"), (GOLD, "contact"), (CATH, "cathode"),
             (ANODE, "anode (guard)")]
    h = [Patch(facecolor=c, edgecolor="#333", label=l) for c, l in items]
    fig = plt.figure(figsize=(11, 0.55))
    fig.legend(handles=h, ncol=len(items), loc="center", frameon=False, fontsize=12)
    fig.savefig(od / "rabbit_setup_legend.png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    barmm = 2.0; barpx = barmm * ppmm
    fig = plt.figure(figsize=((barpx + 40) / 100.0, 0.7), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, barpx + 40); ax.set_ylim(0, 70)
    ax.plot([20, 20 + barpx], [24, 24], "k-", lw=5, solid_capstyle="butt")
    ax.text(20 + barpx / 2, 30, f"{barmm:g} mm", ha="center", va="bottom", fontsize=13)
    fig.savefig(od / "rabbit_setup_scalebar.png", dpi=100, transparent=True)
    plt.close(fig)
    print(f"  parts: legend + scalebar (2 mm @ {ppmm:.1f} px/mm) -> {od}", flush=True)


def main():
    surf = load_surface()
    groups = load_fibers()
    tube0, tube1 = fiber_tube(groups[0]), fiber_tube(groups[1])
    cl = centerline(surf)
    s_cl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
    frames = {tag: frame_at(surf, cl, float(cl[int(np.argmin(np.abs(s_cl - off))), 2]))
              for tag, off, _ in POSITIONS}
    bm = json.loads(BEST_META.read_text())
    cath = bm["cathode"]
    anodes = set(bm["long_anodes"] if BEST_CFG == "long_tripole" else
                 bm["trans_anodes"] if BEST_CFG == "trans_tripole" else [])
    print(f"fibers trunk={len(groups[0])} SCB={len(groups[1])}; best={BEST_TAG} cathode c{cath}", flush=True)

    # ---- MAIN: best cuff only, full detail, big ----
    p = scene(surf, tube0, tube1, (3000, 1450)); set_cam(p, surf)
    add_cuff(p, frames[BEST_TAG], cath, anodes, show_tripole=True)
    ppmm = px_per_mm(p, tuple(np.array(surf.center)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    crop_save(p, OUT, (3000, 1050)); write_ppmm(OUT, ppmm); p.close()
    print(f"wrote {OUT} ({ppmm:.1f} px/mm) [best cuff, big]", flush=True)
    make_parts(ppmm)

    # ---- THUMBS: one per position, same camera, small ----
    for tag, offset, dist in POSITIONS:
        pt = scene(surf, tube0, tube1, (1500, 720)); set_cam(pt, surf)
        add_cuff(pt, frames[tag], cath, anodes, show_tripole=(tag == BEST_TAG))
        out = OUT.parent / f"rabbit_setup_thumb_{tag}.png"
        crop_save(pt, out, (1500, 520)); pt.close()
        print(f"  thumb {tag}: offset {offset}mm ({dist} mm from branch)"
              f"{' BEST' if tag == BEST_TAG else ''} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
