# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 6 step 1 — rabbit real-3D nerve, DUKE 3x4 cuff on the proximal trunk,
via golgi's headless Study pipeline. Produces a project dir with per-contact
FEM lead fields (paths_Ve.npz) + auto-detected branch labels (SCB vs trunk),
which fig5_thresholds.py then consumes for branch-selective threshold sweeps.

Run after the Fig-4 recruit run frees the CPU (mesh+FEM are heavy).
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
NAS = ROOT / "data/Reduced_Reduced_Smoothed_Wrapped_mask_from_object 3_wrapped.nas"
PROJ = ROOT / "paper_figs/out/_intermediate/rabbit_project"
CUFF_OFFSET_MM = float(os.environ.get("CUFF_OFFSET_MM", "30.0"))   # along trunk from proximal end (matches GUI)


def main():
    import golgi
    if PROJ.exists():
        shutil.rmtree(PROJ)
    s = golgi.Study.create(PROJ)
    info = s.import_nerve(NAS, scale_factor=1e-4)
    print("[1] imported:", {k: info[k] for k in list(info)[:6]})

    s.set_mesh(use_epi=(os.environ.get("USE_EPI", "1") == "1"), epi_thickness_um=50,
               decim_target_k=int(os.environ.get("DECIM_K", "15")))
    s.set_electrodes([{
        "eid": "elec_01", "name": "ring-array 3x4 @ trunk",
        "cuff_offset_mm": CUFF_OFFSET_MM,
        "electrode_type": "ring-array (NxM)",        # analytical axial patches = same as extruded nerves
        "array_n_rows": 3, "array_n_cols": 4,        # 3 axial x 4 circumferential = 12 contacts
    }])
    # Refit the cuff to the local nerve axis — the GUI "refit" step that aligns
    # the electrode/field frame with the nerve+fiber frame (skipping it leaves
    # the fibers in the raw frame and the FEM field unsampled along them).
    from golgi.scene.cuff_fit import refit_design_geometry
    ok = refit_design_geometry("elec_01", geom=s._geom, state=s._state)
    print(f"[2] electrode set (cuff @ {CUFF_OFFSET_MM} mm) + refit -> {ok}; meshing ...")
    msh = s.run_mesh()
    print("[3] meshed:", msh if isinstance(msh, (int, str)) else type(msh).__name__)

    s.set_fiber_seed(n_fibers=400, fiber_seed_end="trunk (low z)",
                     fiber_auto_detect_branches=True, fiber_method="streamlines",
                     fiber_max_steps=10000, fiber_cluster_eps_mm=2.0,
                     fiber_cap_band_pct=15.0, fiber_min_rel_size_pct=20.0,
                     fiber_axial_normal_thresh=0.70)
    fibers = s.run_fibers()
    print(f"[4] fibers: {fibers.get('n_paths')} paths, "
          f"{fibers.get('n_branches')} branches")

    s.run_fem()
    print("[5] FEM solved")

    # Threshold sweep -> per-fiber thresholds + branch labels (SCB vs trunk)
    from golgi.jobs.schemas import SweepRequest
    res = s.run_sweep(SweepRequest(mode="threshold", bisect_lo_mA=0.01,
                                   bisect_hi_mA=5.0, bisect_tol_uA=20.0))
    thr = np.asarray(res.thresholds_uA, float)
    bidx = np.asarray(res.fiber_branch_idx, int)
    pos = bidx[bidx >= 0]
    print(f"[6] sweep: {int(np.isfinite(thr).sum())}/{len(thr)} activated; "
          f"branches {np.unique(pos)} counts {np.bincount(pos).tolist() if pos.size else []}")

    fb = np.load(PROJ / "nerve_paths_fibers.npz", allow_pickle=True)
    out = ROOT / "paper_figs/out/data/rabbit_branch.npz"
    np.savez_compressed(out, thr_uA=thr, branch_idx=bidx,
                        paths_flat=fb["paths_flat"], path_lengths=fb["path_lengths"])
    print(f"[7] wrote {out}")


if __name__ == "__main__":
    main()
