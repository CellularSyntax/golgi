# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Realistic rabbit-population thresholds on the BEST tripole position, for fig7 panels b-g.
Uses the RABBIT cervical-vagus distribution (sample_rabbit from rabbit_repop_resweep:
C-dominated 45%, B 30%, A-delta 17%, A-beta 8% capped at 6um) -- NOT a pig/human preset
(pig 8-11um fibers don't recruit on the sub-mm rabbit cuff; the same lesson fixed the old
fig7 panel c). Writes thr_pop.npz (100us) + thr_pop_pw300.npz (300us, clinical) at the best
position in the EXACT fig5_thresholds format so fig07_rabbit_selectivity.py reads them like
fig8 reads the human pop files.

env: BEST_TAG (default off5_4x5).
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import json
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from fig5_thresholds import _worker, _load           # noqa: E402
from fig5_population import classify, UNMYEL          # noqa: E402
from rabbit_repop_resweep import sample_rabbit        # noqa: E402  (validated rabbit distribution)

SW = ROOT / "paper_figs/out/data/rabbit_tripole_sweep"
BEST_TAG = os.environ.get("BEST_TAG", "off3_4x5")
TRUNC_MM = 5.0; HI_MA = 50.0; TOL = 20.0; TSTOP = 3.0    # match the controlled 10um sweep
# C-fibers (~45% of the rabbit pop) are slow in NEURON; SUBSAMPLE>0 runs a balanced subset
# (still plenty per class) to keep the realistic-pop sweep tractable. 0 = all 468 fibers.
SUBSAMPLE = int(os.environ.get("SUBSAMPLE", "0"))


def run(pw_ms, outname):
    npz = str(SW / BEST_TAG / "paths_Ve.npz")
    fibers, cids, ncon = _load(npz, None, TRUNC_MM * 1e3)
    nall = len(fibers)
    branch = np.array([fibers[i][2] for i in range(nall)], int)
    xy = np.array([fibers[i][3] for i in range(nall)], float)
    rng = np.random.default_rng(1)
    diam, model = sample_rabbit(nall, rng)               # full-length, global-indexed by _worker
    typ = classify(diam, np.isin(model, list(UNMYEL)))
    if SUBSAMPLE and SUBSAMPLE < nall:                    # balanced SCB/trunk subsample for speed
        rs = np.random.default_rng(0); nh = SUBSAMPLE // 2
        si = np.where(branch == 1)[0]; ti = np.where(branch == 0)[0]
        sel = np.sort(np.concatenate([rs.choice(si, min(nh, len(si)), replace=False),
                                      rs.choice(ti, min(nh, len(ti)), replace=False)]))
    else:
        sel = np.arange(nall)
    nproc = max(1, min(8, mp.cpu_count() - 1))
    chunks = [list(c) for c in np.array_split(sel, nproc * 3) if len(c)]
    args = [(ch, npz, diam, model, TOL, None, TRUNC_MM * 1e3, pw_ms, TSTOP, HI_MA) for ch in chunks]
    print(f"[{BEST_TAG}] pw {pw_ms*1e3:.0f}us: {len(sel)}/{nall} fibers x {ncon} patterns "
          f"on {nproc} procs ...", flush=True)
    thr = np.full((nall, ncon), np.nan)
    with mp.get_context("spawn").Pool(nproc) as pool:
        for idxs, block in pool.imap_unordered(_worker, args):
            thr[idxs] = block
    np.savez_compressed(SW / BEST_TAG / outname, thr_uA=thr[sel], fiber_idx=sel,
                        branch_idx=branch[sel], diameter_um=diam[sel], model=model[sel].astype(str),
                        type_label=typ[sel].astype(str), xy_cuff_mm=xy[sel], contact_ids=cids,
                        meta=json.dumps(dict(pop="rabbit", pw_ms=pw_ms, hi_mA=HI_MA,
                                             trunc_mm=TRUNC_MM, n_fibers=int(len(sel)))))
    comp = {t: int((typ[sel] == t).sum()) for t in np.unique(typ[sel])}
    print(f"[{BEST_TAG}] wrote {outname}: pop {comp}; recruited(<= {HI_MA}mA) "
          f"{np.mean(np.isfinite(thr[sel]).any(1))*100:.0f}% -> {SW/BEST_TAG/outname}", flush=True)


def main():
    run(0.3, "thr_pop_pw300.npz")    # clinical 300 us — feeds the MAIN figure (do first)
    run(0.1, "thr_pop.npz")          # 100 us — for the supplementary pulse-width panel


if __name__ == "__main__":
    main()
