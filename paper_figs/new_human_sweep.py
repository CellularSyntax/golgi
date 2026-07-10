# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cuff-POSITION selectivity sweep on the NEW full-length human model. The cuff anchor is
'trunk', so cuff_offset_mm = distance FROM the trunk end; branch-side distance ≈ 35 - offset.
The epineurium is a single trunk the whole length (SCB + trunk-continuation are internal endo
branches), so the 4x5 cuff wraps at any position. Move it from the cranial trunk toward the
caudal endo bifurcation and at each position re-mesh (full-length endo) + reciprocity FEM along
the real-3D branch-resolved streamlines, then plot best-contact SCB/trunk vs distance-to-branch.

Baseline (offset 5.4 ≈ 29.5 mm from the branch) reused from new_human_branch_4x5.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
PY = sys.executable
D = ROOT / "paper_figs/out/data"

ARR = dict(ARR_ROWS="4", ARR_COLS="5", ARR_ROW_SEP="1.0", ARR_CONTACT_W="0.6",
           ARR_CONTACT_PHI="30.0", L_CUFF="5.0")
MESH_ENV = dict(GOLGI_PLC_CDT="1", GOLGI_TETGEN_SWITCHES="pzAaS150000",
                GOLGI_TETGEN_EPSILON="1e-6", GOLGI_FASCICLE_FULL_LENGTH="1")
# (cuff_offset_mm, approx distance-from-branch mm) — new positions toward the bifurcation
NEW = [(15.0, 19.9), (22.0, 12.9), (27.0, 7.9), (30.0, 4.9)]


def run(script, env, tag, timeout):
    e = dict(os.environ); e.update(ARR); e.update(env)
    with open(f"/tmp/sweep_{tag}.log", "w") as log:
        try:
            r = subprocess.run([PY, "-u", "paper_figs/" + script], env=e, cwd=str(ROOT),
                               stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-9", "-f", "tetgen"], capture_output=True)
            subprocess.run(["pkill", "-9", "-f", script], capture_output=True)
            return False


def best_selectivity(tag):
    d = np.load(D / f"new_human_branch_{tag}/paths_Ve.npz", allow_pickle=True)
    Ve = np.abs(d["Ve_mat"]); bidx = d["branch_idx"]; lens = d["path_lengths"]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    fp = np.empty((len(lens), Ve.shape[1]))
    for i in range(len(lens)):
        fp[i] = Ve[off[i]:off[i + 1]].max(0)
    b0 = bidx == 0; b1 = bidx == 1
    ratios = np.array([fp[b1, c].mean() / fp[b0, c].mean() for c in range(Ve.shape[1])])
    return float(ratios.max()), int(ratios.argmax()), float(ratios.min())


def main():
    results = []   # (dist_from_branch_mm, best_ratio, best_contact, min_ratio, offset)
    r, c, lo = best_selectivity("4x5")            # baseline offset 5.4 ≈ 29.5 mm from branch
    results.append((29.5, r, c, lo, 5.4))
    print(f"baseline offset 5.4 (≈29.5mm from branch): best SCB/trunk {r:.2f} @ c{c}", flush=True)
    for off, dist in NEW:
        tag = f"off{off:g}_4x5"
        print(f"\n=== offset {off} (≈{dist}mm from branch) ===", flush=True)
        env = dict(MESH_ENV); env["CUFF_OFFSET"] = f"{off}"
        if not run("new_human_mesh.py", env, f"mesh_off{off:g}", 600):
            print(f"  off{off}: MESH FAILED — skip (/tmp/sweep_mesh_off{off:g}.log)", flush=True); continue
        env2 = dict(env); env2["SWEEP_TAG"] = tag
        if not run("new_human_fem.py", env2, f"fem_off{off:g}", 2000):
            print(f"  off{off}: FEM FAILED — skip (/tmp/sweep_fem_off{off:g}.log)", flush=True); continue
        r, c, lo = best_selectivity(tag)
        results.append((dist, r, c, lo, off))
        print(f"  off{off} (≈{dist}mm from branch): best SCB/trunk {r:.2f} @ c{c} (min {lo:.2f})", flush=True)
    results.sort(reverse=True)   # far -> near the branch
    a = np.array([(d, r, c, lo, o) for d, r, c, lo, o in results])
    np.savez(D / "new_human_selectivity_sweep.npz", results=a,
             cols="dist_from_branch_mm,best_ratio,best_contact,min_ratio,offset_mm")
    print("\n=== SWEEP RESULT: best-contact SCB/trunk vs distance from bifurcation ===")
    for dist, r, c, lo, o in results:
        print(f"  {dist:5.1f} mm from branch (off {o:g}): best {r:.2f} @ c{int(c)}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5.4, 4))
    plt.plot(a[:, 0], a[:, 1], 'o-', color='#e6550d', lw=2, ms=8)
    plt.axhline(1.0, ls='--', c='gray', lw=0.8)
    plt.xlabel("cuff distance from endo bifurcation (mm)\n← nearer the branch")
    plt.ylabel("best-contact SCB / trunk selectivity")
    plt.title("Selectivity vs cuff position\n(4×5, full-length epi+endo, real-3D fibers)")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    out = ROOT / "paper_figs/out/figures/png/new_human_selectivity_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
