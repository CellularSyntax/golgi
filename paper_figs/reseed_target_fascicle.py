# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Targeted re-seed of the fig5/6 TARGET fascicle so its per-class selectivity
bar (panel f) and threshold-vs-diameter (panel g) have enough fibers per class.

The duke fiber set seeds area-proportionally, so a small peripheral target
fascicle gets only ~11-23 of 700 fibers — far too few for a 5-class split. Here
we densify ONLY the target fascicle: we take the duke straight-extruded fiber
trajectories (already in the mesh frame), pick a representative target fiber as a
template, and translate copies of it to many new (x,y) seed points sampled inside
the convex hull of the existing target fibers (so every new fiber stays inside
the fascicle). The full fiber set (all existing + dense target) is written so
solve_nerve_ci can re-sample the per-contact CONTACT-IMPEDANCE lead field along
them on the cached mesh — no re-mesh, no re-running the 12-contact selectivity.

usage:  python reseed_target_fascicle.py <duke_dir_name> <target_fascicle_id> [n_new]
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent


def main():
    nd = sys.argv[1]
    kt = int(sys.argv[2])
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    src = ROOT / "results_golgi/duke_meshes" / nd
    hd = ROOT / "paper_figs/out/_intermediate" / f"reseed_{nd}"
    hd.mkdir(parents=True, exist_ok=True)
    for f in ["nerve.msh", "mesh_config.json", "electrode_config.json"]:
        shutil.copy(src / f, hd / f)

    d = np.load(src / "nerve_paths_fibers.npz", allow_pickle=True)
    flat = np.asarray(d["paths_flat"], float)
    lens = np.asarray(d["path_lengths"], np.int64)
    br = np.asarray(d["branch_idx"], np.int64)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)

    tgt = np.where(br == kt)[0]
    if len(tgt) < 4:
        raise SystemExit(f"target fascicle {kt} has only {len(tgt)} fibers — cannot hull")
    centers = np.array([flat[off[i]:off[i + 1], :2].mean(0) for i in tgt])
    # representative template fiber = median length, central-most
    ci = np.argmin(np.linalg.norm(centers - centers.mean(0), axis=1))
    ref_i = tgt[ci]
    ref = flat[off[ref_i]:off[ref_i + 1]].copy()
    ref_c = ref[:, :2].mean(0)

    from scipy.spatial import ConvexHull, Delaunay
    hull = ConvexHull(centers)
    dela = Delaunay(centers[hull.vertices])
    lo, hi = centers.min(0), centers.max(0)
    rng = np.random.default_rng(0)
    new = []
    tries = 0
    while len(new) < n_new and tries < n_new * 200:
        tries += 1
        p = rng.uniform(lo, hi)
        if dela.find_simplex(p) >= 0:
            nf = ref.copy()
            nf[:, 0] += p[0] - ref_c[0]
            nf[:, 1] += p[1] - ref_c[1]
            new.append(nf)

    all_flat = [flat] + new
    all_lens = list(lens) + [len(nf) for nf in new]
    all_br = list(br) + [kt] * len(new)
    new_flat = np.vstack(all_flat)
    np.savez(hd / "nerve_paths_fibers.npz", paths_flat=new_flat,
             path_lengths=np.asarray(all_lens, np.int64),
             branch_idx=np.asarray(all_br, np.int64))
    n_tgt_new = int((np.asarray(all_br) == kt).sum())
    print(f"reseed {nd}: target fasc {kt} {len(tgt)} -> {n_tgt_new} fibers "
          f"(+{len(new)} new); total {len(all_lens)} fibers; "
          f"hull xy extent {(hi - lo).round(5)} m -> {hd}")


if __name__ == "__main__":
    main()
