# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Stage the NEW cleaned human endoneurium for golgi's quasi-static real-3D trajectory
generator (same method as fig3 / human3d_prep.py). Uses the decimated+repaired endo so the
streamlines live in the SAME geometry as the FEM endo region. STL is mm -> m via 1e-3.

Writes new_human3d_out/{nerve_only_surface.npz, nerve_paths_seed_config.json}. Then run:
  FIBER_OUT_DIR=paper_figs/new_human3d_out python golgi/compute/solve_fiber_paths_nerve.py
followed by new_human3d_traj.py (RK4 reintegration with wall-clamp).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import meshio

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
STL = ROOT / "data/new_human_meshes/ENDO_dec80_fixed.stl"   # same endo the FEM mesh uses
OUT = ROOT / "paper_figs/new_human3d_out"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 1e-3   # mm -> m


def main():
    m = meshio.read(str(STL))
    pts = m.points.astype(np.float64)
    tris = m.cells_dict["triangle"].astype(np.int64)
    used = np.unique(tris)
    remap = -np.ones(len(pts), np.int64); remap[used] = np.arange(len(used))
    spts_m = pts[used] * SCALE
    stris = remap[tris]
    span_mm = (spts_m.max(0) - spts_m.min(0)) * 1e3
    print(f"new human endo: {len(spts_m):,} pts / {len(stris):,} tris | "
          f"span (mm) {span_mm.round(2)} -> ~{span_mm.max():.0f}mm long, "
          f"~{np.sort(span_mm)[:2].mean():.1f}mm wide")
    np.savez_compressed(OUT / "nerve_only_surface.npz",
                        pts_raw=spts_m, pts_cuff=spts_m, tris=stris)
    cfg = dict(n_seeds=500, step_um=100.0, seed_end="low", max_steps=50000,
               cluster_eps_m=5e-4)
    (OUT / "nerve_paths_seed_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {OUT/'nerve_only_surface.npz'} + seed config")


if __name__ == "__main__":
    main()
