# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Stage the rabbit nerve (.nas tetra volume) for golgi's quasi-static
trajectory generator: extract the boundary surface and write
nerve_only_surface.npz + seed config into FIBER_OUT_DIR.

Scale: .nas units -> metres via factor 1e-4 (1 unit = 0.1 mm), confirmed by
the user. Rabbit nerve is ~22x19x135 mm (true physical scale).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import meshio

ROOT = Path(__file__).parent.parent
NAS = ROOT / "data/Reduced_Reduced_Smoothed_Wrapped_mask_from_object 3_wrapped.nas"
OUT = ROOT / "paper_figs/rabbit_out"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 1e-4   # .nas units -> metres (user-confirmed)


def main():
    t = time.time()
    m = meshio.read(str(NAS))
    tets = m.cells_dict["tetra"].astype(np.int64)
    pts = m.points.astype(np.float64)
    print(f"read {len(pts):,} pts / {len(tets):,} tets in {time.time()-t:.1f}s")

    # boundary faces = triangles that appear in exactly one tet
    F = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                        tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]], 0)
    Fs = np.sort(F, 1)
    uniq, idx, cnt = np.unique(Fs, axis=0, return_index=True, return_counts=True)
    bnd = F[idx[cnt == 1]]                      # keep original winding
    print(f"boundary tris: {len(bnd):,}")

    used = np.unique(bnd)
    remap = -np.ones(len(pts), np.int64); remap[used] = np.arange(len(used))
    spts_m = pts[used] * SCALE                  # true metres
    stris = remap[bnd]

    span_mm = (spts_m.max(0) - spts_m.min(0)) * 1e3
    print(f"scaled nerve span (mm): {span_mm.round(2)}  "
          f"-> ~{span_mm.max():.0f}mm long, ~{np.sort(span_mm)[:2].mean():.1f}mm wide")

    np.savez_compressed(OUT / "nerve_only_surface.npz",
                        pts_raw=spts_m, pts_cuff=spts_m, tris=stris)
    cfg = dict(n_seeds=500, step_um=200.0, seed_end="low", max_steps=50000)
    (OUT / "nerve_paths_seed_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {OUT/'nerve_only_surface.npz'} ({len(spts_m):,} pts, "
          f"{len(stris):,} tris) + seed config")


if __name__ == "__main__":
    main()
