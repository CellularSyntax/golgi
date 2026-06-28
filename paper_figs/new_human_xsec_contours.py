# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Precompute the epineurium + endoneurium (multi-lobe fascicle) cross-section contours at
the best-position cuff plane, in the SAME mesh frame as the figure's fiber xy, so the
new-human selectivity cross-section (panel d) can draw the real nerve boundaries behind the
activation map. Rigidly maps the new STLs from the trajectory frame to the FEM mesh frame
via the fibers (present in both, 1:1), slices at the cuff plane, returns one loop per lobe.
Cached -> new_human_tripole_sweep/<tag>/xsec_contours.npz (slicing the 384k endo is slow).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
NHM = ROOT / "data/new_human_meshes"
EPI = NHM / "EPINERIUM_Epinerium_cleaned_aligned_masks_mm_SMOOTHED_LONGER_original_duplicate_duplicate_wrapped_duplicate.stl"
ENDO = NHM / "ENDONERIUM_masks_ns_2w_processed_reduced_2w_reduced2.stl"   # full-res for clean lobes
TRAJ = ROOT / "paper_figs/new_human3d_out/nerve_paths_branch.npz"
SW = ROOT / "paper_figs/out/data/new_human_tripole_sweep"
BEST_COL = 1   # long_tripole column, for the cuff-plane z


def kabsch(A, B):
    """Rigid R,t mapping A->B (corresponding rows), reflection-guarded."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cA, cB


def ordered_loops(sl, min_pts):
    """Order each connected slice contour by WALKING its line segments (a clean closed-loop
    boundary order), instead of angular-sorting around a centroid (which self-intersects for
    non-convex lobes -> 'weird paths')."""
    from collections import defaultdict
    pts = np.asarray(sl.points)
    if sl.n_lines == 0:
        return []
    segs = np.asarray(sl.lines).reshape(-1, 3)[:, 1:]      # (nseg, 2) point-id pairs
    adj = defaultdict(list)
    for a, b in segs:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    seen, loops = set(), []
    for s in list(adj):
        if s in seen:
            continue
        loop, cur, prev = [], s, -1
        while cur not in seen:
            seen.add(cur); loop.append(cur)
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        if len(loop) >= min_pts:
            loops.append(pts[np.asarray(loop)])
    return loops


def loops(stl_path, R, cA, cB, cz_m, ctr_mm, min_pts):
    m = pv.read(str(stl_path))
    m.points = ((np.asarray(m.points) * 1e-3 - cA) @ R.T + cB)   # m, mesh frame
    sl = m.slice(normal="z", origin=(0.0, 0.0, cz_m))
    if sl.n_points == 0:
        return []
    out = []
    for loop in ordered_loops(sl, min_pts):
        p = loop[:, :2] * 1e3 - ctr_mm                   # mm, centered
        # drop spurious slivers (tiny enclosed area via shoelace)
        area = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
        if area > 1e-3:
            out.append(p)
    return out


def compute(tag):
    pv_ = np.load(SW / tag / "paths_Ve.npz", allow_pickle=True)
    clf = np.asarray(pv_["paths_flat"], float)            # m, MESH frame
    Ve = np.abs(pv_["Ve_mat"][:, BEST_COL]); lens = pv_["path_lengths"]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    flat_mm = clf * 1e3
    pk_z = np.array([flat_mm[off[i]:off[i + 1]][np.argmax(Ve[off[i]:off[i + 1]]), 2]
                     for i in range(len(lens))])
    cz_mm = float(np.median(pk_z)); cz_m = cz_mm / 1e3
    ctr_mm = flat_mm[:, :2].mean(0)

    tr = np.asarray(np.load(TRAJ, allow_pickle=True)["paths_flat"], float)   # m, TRAJ frame
    assert len(tr) == len(clf), f"point count mismatch {len(tr)} vs {len(clf)}"
    idx = np.linspace(0, len(tr) - 1, min(len(tr), 8000)).astype(int)
    R, cA, cB = kabsch(tr[idx], clf[idx])
    resid = np.linalg.norm(((tr[idx] - cA) @ R.T + cB) - clf[idx], axis=1).mean() * 1e3
    print(f"[{tag}] kabsch residual {resid:.3f} mm; cuff plane z={cz_mm:.2f} mm (mesh)")

    epi = loops(EPI, R, cA, cB, cz_m, ctr_mm, 8)
    endo = loops(ENDO, R, cA, cB, cz_m, ctr_mm, 6)
    epi_main = max(epi, key=len) if epi else np.zeros((0, 2))
    np.savez(SW / tag / "xsec_contours.npz",
             epi=epi_main, n_endo=len(endo),
             **{f"endo{i}": e for i, e in enumerate(endo)})
    print(f"[{tag}] epi loop {len(epi_main)} pts; {len(endo)} endo lobes "
          f"-> {SW/tag/'xsec_contours.npz'}")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["off15_4x5"]
    for t in tags:
        compute(t)
