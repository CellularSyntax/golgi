# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""NEW human real-3D cervical vagus full-length branch-resolved trajectories — identical
method to human3d_traj.py (fig3), pointed at new_human3d_out (the new cleaned endo).

Re-integrates RK4 streamlines from the stored Laplace field (nerve_paths.vtu): seed at the
single cranial trunk cap, flow +E toward the caudal branch caps, and on wall exit clamp the
point back to the nearest interior vertex (so paths slide along the curved nerve instead of
dying). Full-length filter + branch clustering. Writes new_human3d_out/nerve_paths_branch.npz.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path(__file__).parent.parent
OUT = ROOT / "paper_figs/new_human3d_out"
PREVIEW = ROOT / "paper_figs/out/_intermediate/new_human3d_traj_preview.png"

# --- knobs (same as fig3 human3d_traj.py) ------------------------------------
N_SEED = 600          # candidate seeds across the trunk cap
STEP_UM = 60.0        # RK4 step
MAX_STEPS = 1400      # 1400 * 60 um = 84 mm arc-length ceiling (>> 33.5 mm length)
SEED_BAND_MM = 0.6    # trunk-cap seed slab thickness
RIM_PULL = 0.18       # pull seeds toward the trunk-cap centroid (xy of long axis)
COVER_MIN = 0.92      # keep fibers spanning >= this fraction of the nerve length
SCB_COL, TRUNK_COL = "#e6550d", "#1f77b4"


def main():
    grid = pv.read(OUT / "nerve_paths.vtu")
    caps = json.loads((OUT / "nerve_paths_caps.json").read_text())
    pts_v = np.asarray(grid.points, float)
    E = np.asarray(grid["E"], float)
    tets = grid.cells_dict[pv.CellType.TETRA].astype(np.int64)

    trunk_c = np.asarray(caps["trunk_cap_centroid_m"], float)
    branch_c = np.asarray(caps["branch_cap_centroids_m"], float)
    span = pts_v.max(0) - pts_v.min(0); ax = int(np.argmax(span))
    zlo, zhi = pts_v[:, ax].min(), pts_v[:, ax].max(); L = zhi - zlo
    trunk_hi = trunk_c[ax] > 0.5 * (zlo + zhi)
    print(f"nerve length {L*1e3:.1f} mm (axis {ax}); trunk at "
          f"{'high' if trunk_hi else 'low'}-z; {len(branch_c)} branch caps")

    from scipy.spatial import cKDTree
    vtree = cKDTree(pts_v)
    edge = float(np.median(np.linalg.norm(pts_v[tets[:, 1]] - pts_v[tets[:, 0]], axis=1)))
    OUTSIDE = 2.5 * edge
    print(f"mesh edge ~{edge*1e6:.0f} um; outside threshold {OUTSIDE*1e6:.0f} um")

    def eval_E(x, k=6):
        x = np.ascontiguousarray(np.atleast_2d(x))
        d, vi = vtree.query(x, k=k)
        w = 1.0 / np.maximum(d, 1e-9); w /= w.sum(1, keepdims=True)
        Ei = (w[..., None] * E[vi]).sum(1)
        return Ei, d[:, 0] < OUTSIDE

    def unit(v):
        m = np.linalg.norm(v, axis=1, keepdims=True)
        return np.where(m > 1e-30, v / np.maximum(m, 1e-30), 0.0)

    band = SEED_BAND_MM * 1e-3
    seed_mask = (pts_v[:, ax] > zhi - band) if trunk_hi else (pts_v[:, ax] < zlo + band)
    seeds = pts_v[seed_mask].copy()
    perp = [i for i in range(3) if i != ax]
    seeds[:, perp] += RIM_PULL * (trunk_c[perp][None, :] - seeds[:, perp])
    seeds[:, ax] -= np.sign(trunk_c[ax] - 0.5 * (zlo + zhi)) * 0.3e-3
    if len(seeds) > N_SEED:
        seeds = seeds[np.linspace(0, len(seeds) - 1, N_SEED).astype(int)]
    print(f"seeding {len(seeds)} streamlines at the trunk cap")

    step = STEP_UM * 1e-6
    x = seeds.copy(); N = len(x)
    paths = [[x[i].copy()] for i in range(N)]
    active = np.ones(N, bool); stuck = np.zeros(N, int)
    for it in range(MAX_STEPS):
        if not active.any():
            break
        idx = np.where(active)[0]; xa = x[idx]
        k1, v1 = eval_E(xa)
        bad = ~v1
        if bad.any():
            snap = vtree.query(xa[bad])[1]
            xa[bad] = pts_v[snap] + 0.0
            k1[bad], _ = eval_E(xa[bad])
            stuck[idx[bad]] += 1
        k1 = unit(k1)
        k2 = unit(eval_E(xa + 0.5 * step * k1)[0]); k2 = np.where(np.linalg.norm(k2, axis=1, keepdims=True) > 0.5, k2, k1)
        k3 = unit(eval_E(xa + 0.5 * step * k2)[0]); k3 = np.where(np.linalg.norm(k3, axis=1, keepdims=True) > 0.5, k3, k2)
        k4 = unit(eval_E(xa + step * k3)[0]); k4 = np.where(np.linalg.norm(k4, axis=1, keepdims=True) > 0.5, k4, k3)
        xa = xa + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        for li, gi in enumerate(idx):
            if (np.linalg.norm(k1[li]) < 0.5) or stuck[gi] > 40:
                active[gi] = False; continue
            x[gi] = xa[li]; paths[gi].append(x[gi].copy())
            arrived = (x[gi, ax] < zlo + 0.4e-3) if trunk_hi else (x[gi, ax] > zhi - 0.4e-3)
            if arrived:
                active[gi] = False
        if (it + 1) % 300 == 0:
            print(f"  step {it+1}: {int(active.sum())} active")

    paths = [np.asarray(p) for p in paths if len(p) >= 5]
    cov = np.array([(p[:, ax].max() - p[:, ax].min()) / L for p in paths])
    full = [p for p, c in zip(paths, cov) if c >= COVER_MIN]
    print(f"{len(paths)} paths produced; {len(full)} span >= {COVER_MIN*100:.0f}% (full-length)")

    branch_idx = []
    for p in full:
        caud = p[np.argmin(p[:, ax])] if trunk_hi else p[np.argmax(p[:, ax])]
        branch_idx.append(int(np.argmin(np.linalg.norm(branch_c - caud, axis=1))))
    branch_idx = np.array(branch_idx)
    for b in range(len(branch_c)):
        print(f"  branch {b}: {(branch_idx == b).sum()} fibers "
              f"(cap area {caps['branch_cap_areas_m2'][b]*1e6:.2f} mm^2)")

    flat = np.vstack(full); lens = np.array([len(p) for p in full])
    np.savez_compressed(OUT / "nerve_paths_branch.npz", paths_flat=flat, path_lengths=lens,
                        branch_idx=branch_idx, step_m=step, long_axis=ax)
    print(f"wrote {OUT/'nerve_paths_branch.npz'} ({len(full)} fibers)")
    _preview(full, branch_idx)


def _preview(fibers, branch_idx):
    surf = np.load(OUT / "nerve_only_surface.npz", allow_pickle=True)
    pts = surf["pts_raw"]; tris = surf["tris"].astype(np.int64)
    faces = np.hstack([np.full((len(tris), 1), 3, np.int64), tris]).ravel()
    nerve = pv.PolyData(pts, faces)
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=(2600, 900))
    p.background_color = "white"
    p.add_mesh(nerve, color="#e3a7c6", opacity=0.18, smooth_shading=True, show_scalar_bar=False)

    def tube(group):
        if not group:
            return None
        ptsl, lines, base = [], [], 0
        for f in group:
            ptsl.append(f); lines.append(np.concatenate([[len(f)], np.arange(base, base + len(f))]))
            base += len(f)
        poly = pv.PolyData(np.vstack(ptsl)); poly.lines = np.hstack(lines)
        return poly.tube(radius=80e-6, n_sides=6)
    for b, col in [(1, SCB_COL), (0, TRUNK_COL)]:
        t = tube([f for f, bi in zip(fibers, branch_idx) if bi == b])
        if t is not None:
            p.add_mesh(t, color=col, show_scalar_bar=False)
    b = nerve.bounds; span = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    axis = int(np.argmax(span)); cr = [i for i in range(3) if i != axis]
    depth = cr[int(np.argmin(span[cr]))]; e = np.eye(3)
    p.camera_position = [tuple(c + e[depth] * 2.0 * span[axis]), tuple(c), tuple(e[axis])]
    p.enable_anti_aliasing("ssaa"); p.camera.zoom(1.5)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(PREVIEW)); p.close()
    print(f"wrote {PREVIEW}")


if __name__ == "__main__":
    main()
