# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Precompute the rabbit nerve cross-section contour at the cuff plane, in the SAME (mesh)
frame as the figure's fiber xy, so fig7's selectivity cross-section (panel d) can draw the
real nerve boundary behind the activation map. RABBIT analogue of new_human_xsec_contours.py.

Frame note: the per-position FEM RE-CENTRES the nerve on its cuff (cuff at z~0), so paths_Ve
is in a rigidly-shifted "mesh" frame while the nerve surface (nerve_only_surface.npz) is in
the original rabbit_out import frame. The rabbit_out streamline fibers exist 1:1 in BOTH frames
(same 321,711 points), so a Kabsch fit maps original -> mesh; apply it to the surface, slice at
the mesh-frame cuff plane, and the contour lands exactly under the figure's fibers.

The common-trunk rabbit nerve is a single bundle (SCB and trunk fascicles intermingled, not
anatomically separated), so there is ONE outer loop (epi = nerve boundary) and NO multi-lobe
endoneurium — the SCB/trunk separation is shown by the fibre colours, not a fascicle contour.
Cached -> rabbit_tripole_sweep/<tag>/xsec_contours.npz (epi loop + n_endo=0).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path(__file__).parent.parent
RO = ROOT / "paper_figs/rabbit_out"
SURF = RO / "nerve_only_surface.npz"
ORIGFIB = RO / "nerve_paths_fibers.npz"          # original-frame fibres (same pts as paths_Ve)
SW = ROOT / "paper_figs/out/data/rabbit_tripole_sweep"
BEST_COL = 1   # long_tripole column, for the cuff-plane z (matches the figure's panel d)


def kabsch(A, B):
    """Rigid R,t mapping A->B (corresponding rows), reflection-guarded."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cA, cB


def ordered_loops(sl, min_pts):
    """Order each connected slice contour by WALKING its line segments (clean closed loop)."""
    from collections import defaultdict
    pts = np.asarray(sl.points)
    if sl.n_lines == 0:
        return []
    segs = np.asarray(sl.lines).reshape(-1, 3)[:, 1:]
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


def _faces(tris):
    return np.hstack([np.full((len(tris), 1), 3, np.int64), tris]).ravel()


def compute(tag, best_col=BEST_COL):
    pv_ = np.load(SW / tag / "paths_Ve.npz", allow_pickle=True)
    clf = np.asarray(pv_["paths_flat"], float)            # m, MESH frame
    col = min(best_col, pv_["Ve_mat"].shape[1] - 1)
    Ve = np.abs(pv_["Ve_mat"][:, col]); lens = pv_["path_lengths"]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    flat_mm = clf * 1e3
    pk_z = np.array([flat_mm[off[i]:off[i + 1]][np.argmax(Ve[off[i]:off[i + 1]]), 2]
                     for i in range(len(lens))])
    cz_mm = float(np.median(pk_z)); cz_m = cz_mm / 1e3
    ctr_mm = flat_mm[:, :2].mean(0)                        # mesh-frame fibre centroid

    tr = np.asarray(np.load(ORIGFIB, allow_pickle=True)["paths_flat"], float)   # m, ORIGINAL frame
    assert len(tr) == len(clf), f"point count mismatch {len(tr)} vs {len(clf)}"
    idx = np.linspace(0, len(tr) - 1, min(len(tr), 8000)).astype(int)
    R, cA, cB = kabsch(tr[idx], clf[idx])                  # original -> mesh
    resid = np.linalg.norm(((tr[idx] - cA) @ R.T + cB) - clf[idx], axis=1).mean() * 1e3

    d = np.load(SURF, allow_pickle=True)
    m = pv.PolyData(np.asarray(d["pts_raw"], float), _faces(np.asarray(d["tris"], np.int64)))
    m = m.clean().triangulate()
    m.points = ((np.asarray(m.points) - cA) @ R.T + cB)    # original -> mesh frame (m)
    sl = m.slice(normal="z", origin=(0.0, 0.0, cz_m))
    epi = np.zeros((0, 2))
    if sl.n_points:
        cand = []
        for loop in ordered_loops(sl, 8):
            p = loop[:, :2] * 1e3 - ctr_mm                 # mm, centred on mesh fibre centroid
            area = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
            if area > 1e-3:
                cand.append(p)
        if cand:
            epi = max(cand, key=len)

    (SW / tag).mkdir(parents=True, exist_ok=True)
    np.savez(SW / tag / "xsec_contours.npz", epi=epi, n_endo=0)
    print(f"[{tag}] kabsch residual {resid:.3f} mm; cuff plane z={cz_mm:.2f} mm (mesh); "
          f"epi loop {len(epi)} pts, maxabs {np.abs(epi).max() if len(epi) else 0:.2f} mm "
          f"(single bundle, no endo lobes) -> {SW/tag/'xsec_contours.npz'}")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["off3_4x5"]
    for t in tags:
        compute(t)
