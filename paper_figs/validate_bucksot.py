# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Validation — reproduce ASCENT's Bucksot 2019 multi-fascicle recruitment example
(Musselman 2021 reproducing Bucksot et al. 2019, PLoS ONE 14:e0215191).

Bucksot's multi-fascicle case: a rabbit-sciatic-sized nerve (~3.02 mm) with FIVE
fascicles in a circumferential bipolar cuff, 100 MRG axons per fascicle (diameters
~ N(8.85, 3.1) um, >=2 um), biphasic 0.1 ms/phase. The FEM voltage is scaled
linearly to current; per-axon activation gives % fibers activated vs current (mA),
per fascicle. Bucksot's point: a circumferential contact recruits the whole nerve
fairly uniformly, and rotating the contact 180 deg ("inverted") barely changes the
aggregate curve, even though individual fascicles (near vs far from the conductor)
shift — fascicles near the contact have lower thresholds.

golgi side: the SAME ASCENT segmentation masks (n.tif/i.tif = 5 fascicles) run
through golgi's Duke extrude -> CI-FEM pipeline with a new 270-deg gapped bipolar
cuff (bucksot-bipolar), at two orientations (phi = 0 "circumferential", phi = pi
"inverted"); then golgi's pyfibers/NEURON MRG path gives per-axon thresholds.

Stages:  --mesh (masks + extrude + CI-FEM, both orientations)
         --sweep (per-axon thresholds, both orientations)
         (default) build the figure from saved thresholds.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, shutil, argparse, time, multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import DATA   # noqa: E402

ASCENT = ROOT / "reference_code/ascent-master/examples/results/bucksot_2019/circumferential"
WORK = ROOT / "paper_figs/out/_intermediate/bucksot"
MASKS = WORK / "masks"
SCALE_BAR_UM = 1000.0                 # bucksot s.tif scale bar = 1000 um (shrinkage 0)
N_FIBERS = 500                        # ~100 / fascicle (5 fascicles), as in Bucksot
PW_MS = 0.1                           # 0.1 ms / phase
DIAM_MEAN, DIAM_SD, DIAM_LO, DIAM_HI = 8.85, 3.1, 2.0, 16.0   # rabbit sciatic A-fibers
ORIENT = {"circ": 0.0, "inverted": np.pi}


def setup_masks():
    if MASKS.exists():
        shutil.rmtree(MASKS)
    MASKS.mkdir(parents=True)
    for src, dst in [("n.tif", "bk_NerveMask.tif"), ("i.tif", "bk_FascMask.tif"),
                     ("s.tif", "bk_ScaleMask.tif")]:
        shutil.copy(ASCENT / src, MASKS / dst)
    print(f"[bucksot] masks -> {sorted(p.name for p in MASKS.glob('*.tif'))}", flush=True)


def build(orient):
    import scripts.batch_mesh_duke as bmd
    bmd.N_FIBERS = N_FIBERS
    bmd.SCALE_BAR_UM = SCALE_BAR_UM
    bmd.DEFORM = "none"               # keep the real 5-fascicle shape
    bmd.MUS_RAD_PAD_M = 10.0e-3       # rabbit ambient ~40 mm dia
    bmd.MUS_AX_PAD_M = 12.0e-3
    bmd.BUCKSOT_PHI_RAD = ORIENT[orient]
    out = WORK / orient
    if out.exists():
        shutil.rmtree(out)
    msg = bmd.process(MASKS, out, species="swine", cuff="bucksot-bipolar")
    print(f"[bucksot:{orient}] phi={ORIENT[orient]:.2f} -> {msg}", flush=True)


# ---- threshold sweep ----
def _load(orient):
    from fig5_thresholds import _taper   # noqa
    d = WORK / orient
    pv = np.load(d / "paths_Ve.npz", allow_pickle=True)
    flat, Ve, lens = pv["paths_flat"], pv["Ve_mat"], pv["path_lengths"].astype(int)
    cids = list(pv["contact_ids"]); bidx = pv["branch_idx"].astype(int)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    # bipolar drive: contact id 0 = cathode, id 1 = anode (see bucksot-bipolar cuff)
    ec = json.loads((d / "electrode_config.json").read_text())
    role = {int(p["id"]): p["role"] for p in ec["patches"]}
    cath = [cids.index(c) for c in cids if role[int(c)] == "cathode"]
    anod = [cids.index(c) for c in cids if role[int(c)] == "anode"]
    fibers = []
    for i in range(len(lens)):
        xyz = flat[off[i]:off[i + 1]]
        s_um = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]) * 1e6
        Vb = (Ve[off[i]:off[i + 1]][:, cath].mean(1) - Ve[off[i]:off[i + 1]][:, anod].mean(1))
        fibers.append((s_um, Vb.astype(np.float64) * _taper(s_um)))
    return fibers, bidx


def _worker(args):
    idxs, payload, diam, pw = args
    from golgi.pipeline.fiber_sim import FiberSimJobRequest
    from golgi.pipeline.sweep import _bisect_threshold, _scaled_pulse_params
    from fig5_thresholds import _bp, PULSE
    H = SimpleNamespace(build_pulse_breakpoints=_bp,
                        MYELINATED_MODELS=["MRG_INTERPOLATION", "SMALL_MRG_INTERPOLATION"],
                        UNMYELINATED_MODELS=["SUNDT", "TIGERHOLM"])
    out = np.full(len(idxs), np.nan)
    for r, i in enumerate(idxs):
        s_um, Vb = payload[i]
        pp = _scaled_pulse_params(dict(PULSE, cath_pw_ms=pw, tstop=max(3.0, pw + 2.0)), 1.0)
        req = FiberSimJobRequest(sel=0, s_um=s_um, Ve_mV=Vb, diameter_um=float(diam[i]),
                                 length_um=float(s_um.max()), pulse_params=pp,
                                 backend="pyfibers", model_name="MRG_INTERPOLATION", helpers=H)
        try:
            th, _ = _bisect_threshold(req, 0.005, 30.0, 5.0)   # uA return; 5-30000 uA range
            out[r] = th / 1e3                                   # -> mA
        except Exception:
            out[r] = np.nan
    return idxs, out


def sweep(orient, workers=8, seed=0):
    fibers, bidx = _load(orient)
    n = len(fibers)
    rng = np.random.default_rng(seed)
    diam = np.clip(rng.normal(DIAM_MEAN, DIAM_SD, n), DIAM_LO, DIAM_HI)
    print(f"[bucksot:{orient}] {n} fibers across {len(np.unique(bidx))} fascicles; "
          f"diam {diam.mean():.1f}+/-{diam.std():.1f} um", flush=True)
    chunks = np.array_split(np.arange(n), workers * 3)
    tasks = [(list(ch), fibers, diam, PW_MS) for ch in chunks if len(ch)]
    thr = np.full(n, np.nan); t0 = time.perf_counter(); done = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for idxs, block in pool.imap_unordered(_worker, tasks):
            thr[idxs] = block; done += len(idxs)
            print(f"  {done}/{n}  {time.perf_counter()-t0:5.1f}s", flush=True)
    np.savez(WORK / f"thr_{orient}.npz", thr_mA=thr, diam_um=diam, fascicle=bidx)
    fin = np.isfinite(thr)
    print(f"[bucksot:{orient}] finite {fin.sum()}/{n}; thr median {np.nanmedian(thr):.3f} mA "
          f"[{np.nanmin(thr):.3f},{np.nanmax(thr):.3f}]; wrote thr_{orient}.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.mesh:
        setup_masks()
        for o in ORIENT:
            build(o)
    if a.sweep:
        for o in ORIENT:
            sweep(o, a.workers)
    if not a.mesh and not a.sweep:
        print("nothing to do; pass --mesh and/or --sweep (figure: fig_bucksot.py)")


if __name__ == "__main__":
    main()
