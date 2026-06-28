# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Re-sample the rabbit fiber population from a RABBIT cervical-vagus distribution
(C-dominated, small myelinated, capped ~6 um so internodes stay below the sub-mm cuff)
instead of the pig preset, and re-sweep the steered thresholds. Fixes the inverted
recruitment-by-class (panel c) where pig-sized 8-11 um fibers failed to recruit because
their internode exceeded the tiny cuff. Old thr backed up as rabbit_steer_thr.npz.pigpop.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import shutil
import numpy as np
import multiprocessing as mp

ROOT = "/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/paper_figs")
from fig5_thresholds import _worker, _load   # noqa: E402

D = ROOT + "/paper_figs/out/data"
NPZ = D + "/rabbit_branch_steer/paths_Ve.npz"
THR = D + "/rabbit_steer_thr.npz"


def sample_rabbit(n, rng):
    # rabbit cervical vagus: C-dominated; small myelinated; few A-beta, capped at 6 um
    groups = [("SUNDT", 0.45, 0.9, 0.3, 0.4, 1.6),                   # C (unmyelinated)
              ("SMALL_MRG_INTERPOLATION", 0.30, 2.1, 0.5, 1.5, 3.0),  # B
              ("SMALL_MRG_INTERPOLATION", 0.17, 3.8, 0.7, 3.0, 5.0),  # A-delta
              ("MRG_INTERPOLATION", 0.08, 5.4, 0.4, 5.0, 6.0)]        # A-beta (few, capped)
    counts = (np.array([g[1] for g in groups]) * n).round().astype(int)
    counts[-1] = n - counts[:-1].sum()
    diam = np.zeros(n); model = np.empty(n, object); k = 0
    for (mdl, _, mean, std, lo, hi), c in zip(groups, counts):
        for _ in range(int(c)):
            diam[k] = float(np.clip(rng.normal(mean, std), lo, hi)); model[k] = mdl; k += 1
    p = rng.permutation(n)
    return diam[p], model[p].astype(str)


def main():
    from fig5_population import classify, UNMYEL
    old = np.load(THR, allow_pickle=True)
    fidx = old["fiber_idx"].astype(int); branch = old["branch_idx"]; xy = old["xy_cuff_mm"]
    rng = np.random.default_rng(1)
    diam, model = sample_rabbit(len(fidx), rng)
    typ = classify(diam, np.isin(model, list(UNMYEL)))

    fibers, cids, ncon = _load(NPZ)
    nall = len(fibers)
    diam_full = np.full(nall, 3.0); model_full = np.array(["SMALL_MRG_INTERPOLATION"] * nall, object)
    for k, gi in enumerate(fidx):
        diam_full[gi] = diam[k]; model_full[gi] = model[k]
    nproc = max(1, min(8, mp.cpu_count() - 1))
    chunks = [list(c) for c in np.array_split(np.asarray(fidx), nproc) if len(c)]
    args = [(ch, NPZ, diam_full, model_full, 2.0, None, 0.0, 0.1, 3.0, 5.0) for ch in chunks]
    print(f"re-sweeping {len(fidx)} rabbit fibers (rabbit population) on {nproc} procs ...", flush=True)
    with mp.Pool(nproc) as pool:
        res = pool.map(_worker, args)
    by = {}
    for idxs, out in res:
        for r, gi in enumerate(idxs):
            by[int(gi)] = out[r]
    thr = np.array([by[int(gi)] for gi in fidx])

    if not os.path.exists(THR + ".pigpop"):
        shutil.copy(THR, THR + ".pigpop")
    np.savez_compressed(THR, thr_uA=thr, fiber_idx=fidx, branch_idx=branch, diameter_um=diam,
                        model=model, type_label=typ.astype(str), xy_cuff_mm=xy, contact_ids=cids)
    print("per-class (size order should be A-beta easiest -> C hardest):")
    for cl in ["Aα", "Aβ", "Aδ", "B", "C"]:
        m = typ == cl
        if m.any():
            mn = np.nanmin(thr[m], axis=1)
            print(f"  {cl} n={int(m.sum())} d{diam[m].mean():.1f}um: c0 med {np.nanmedian(thr[m][:,0]):.0f} "
                  f"min-across med {np.nanmedian(mn):.0f} uA, recruited {np.mean(np.isfinite(mn))*100:.0f}%")


if __name__ == "__main__":
    main()
