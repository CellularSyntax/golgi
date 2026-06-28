# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Reciprocity FEM on the NEW full-length multi-region bundle mesh (new_human_mesh.py must
run first with GOLGI_FASCICLE_FULL_LENGTH=1). Re-attaches the bundle geom, re-sets the
ring-array + refit, injects the real-3D branch-clustered streamline fibers (new_human3d_out/
nerve_paths_branch.npz — fig3 method), and runs golgi's per-contact lead-field solve sampled
along them -> new_human_branch/paths_Ve.npz (Ve_mat[N, n_contacts] + branch labels).
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from new_human_mesh import attach_geom, design_dict, EID, PROJ   # noqa: E402

BRANCH = ROOT / "paper_figs/new_human3d_out/nerve_paths_branch.npz"
# array/position-tagged output dir so successive layouts/offsets don't clobber each other.
# SWEEP_TAG (set by new_human_sweep.py) wins; else fall back to the array size.
_TAG = os.environ.get("SWEEP_TAG") or f"{os.environ.get('ARR_ROWS', '2')}x{os.environ.get('ARR_COLS', '4')}"
OUT = ROOT / f"paper_figs/out/data/new_human_branch_{_TAG}"
RECDIR = PROJ / "designs" / EID / "recording"


def _load_fibers():
    d = np.load(BRANCH, allow_pickle=True)
    flat, lens, bidx = d["paths_flat"], d["path_lengths"], d["branch_idx"]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    return [flat[off[i]:off[i + 1]] for i in range(len(lens))], np.asarray(bidx, int)


def main():
    import golgi
    from golgi.scene.cuff_fit import refit_design_geometry
    s = golgi.Study.open(PROJ)
    attach_geom(s)
    s.set_electrodes([design_dict()])
    refit_design_geometry(EID, geom=s._geom, state=s._state)
    design = next(d for d in s._state.designs if d.get("eid") == EID)
    nc = int(design.get("array_n_rows", 2)) * int(design.get("array_n_cols", 4))

    fibers, bidx = _load_fibers()
    s._geom.fiber_paths_raw = fibers
    s._geom.msh_path = str(PROJ / "designs" / EID / "nerve.msh")
    s._state.has_mesh = True
    s._state.emit_impedance = False
    montages = [{"mid": f"rec{i}", "label": f"rec{i}", "kind": "bipolar",
                 "plus_contact": 2 * i, "minus_contact": 2 * i + 1}
                for i in range(max(1, nc // 2))]
    cfgs = list(s._state.configs)
    for c in cfgs:
        if c.get("design_id") == EID:
            c["recording_montages"] = montages
    s._state.configs = cfgs
    from golgi.conductivity.perineurium import perineurium_thickness_um, fascicle_diameter_um
    import math
    dfasc_um = fascicle_diameter_um(area_um2=math.pi * (600.0 ** 2))   # new endo lobe ~0.6mm r
    peri_thk_m = perineurium_thickness_um("human", dfasc_um) * 1e-6
    s.set_mesh(perineurium_ci=True, peri_thk_m=peri_thk_m, perineurium_species="human")
    print(f"[fem] perineurium CI: thk {peri_thk_m*1e6:.1f}um", flush=True)
    print(f"[fem] {len(fibers)} fibers ({int((bidx==0).sum())} br0 / "
          f"{int((bidx==1).sum())} br1), {nc} contacts; reciprocity FEM ...", flush=True)
    res = s.run_fem()
    print(f"[fem] run_fem -> {res}", flush=True)

    cols, flat, lens = {}, None, None
    for f in sorted(RECDIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if "Ve_flat" not in d.files:
            continue
        cols[int(d["contact_id"])] = np.asarray(d["Ve_flat"], float)
        flat = np.asarray(d["paths_flat"], float); lens = np.asarray(d["path_lengths"], np.int64)
    if not cols:
        raise SystemExit(f"no recording npz in {RECDIR}")
    cids = sorted(cols)
    Ve_mat = -np.column_stack([cols[c] for c in cids])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "paths_Ve.npz", paths_flat=flat, path_lengths=lens,
                        branch_idx=bidx, Ve_mat=Ve_mat, contact_ids=np.asarray(cids, np.int64),
                        units="V_per_A", inject_A=1.0)
    print(f"[assemble] {len(cids)} contacts; Ve_mat {Ve_mat.shape}; "
          f"NaN frac {np.mean(~np.isfinite(Ve_mat)):.3f}; "
          f"range [{np.nanmin(Ve_mat):.3g},{np.nanmax(Ve_mat):.3g}] -> wrote {OUT/'paths_Ve.npz'}")


if __name__ == "__main__":
    main()
