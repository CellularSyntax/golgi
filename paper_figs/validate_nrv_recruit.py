# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Validation V5 (native FEM) — NRV LIFE recruitment on golgi's OWN reciprocity
FEM lead field. Consumes the FEM Ve computed by validate_nrv_fem.py (160 straight
MRG fibers in a monofascicular cat-tibial-like nerve, single intrafascicular LIFE,
monopolar cathode, outer-muscle ground), then golgi's exact pyfibers/NEURON MRG
path gives per-fiber thresholds vs stimulus current (50 / 20 us cathodic) and vs
cathodic pulse duration (7 / 8 / 9 uA). We report the two metrics NRV compared
to in-vivo: the recruitment-rate ratio (50->20 us) and the 7->9 uA increment,
against Nannini & Horch 1991 / Yoshida & Horch 1993 (in-vivo) and Couppey 2024
(NRV, Fig 9).

Unlike validate_nrv.py (analytic intrafascicular line source), the extracellular
potential here is golgi's real FEM solution in the anisotropic endoneurium — the
LIFE active site geometry, perineurium and nerve/muscle boundaries are all in the
mesh. Writes paper_figs/out/data/validate_nrv.json.

Run after validate_nrv_fem.py --mesh --fem has produced the lead field:
    python paper_figs/validate_nrv_recruit.py --smoke 2     # calibrate (serial)
    python paper_figs/validate_nrv_recruit.py --workers 8   # full sweep
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time, argparse, multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from fig5_thresholds import _bp, PULSE                # noqa: E402
from validate_nrv import recruit, rate, REF, PWS_MS   # noqa: E402  (reuse metrics machinery)
from io_paths import DATA                             # noqa: E402

FEM = ROOT / "paper_figs/out/_intermediate/nrv_life/study/configs/cfg_01/paths_Ve.npz"
DIAM = ROOT / "paper_figs/out/_intermediate/nrv_life/fiber_diam.npy"
VE_SCALE = 1e3        # FEM Ve_flat is volts per unit (1 mA) drive -> mV per mA


def load_fem():
    """FEM Ve per fiber as (n_fibers, n_pts) in mV per mA, + arc-length s (um) and
    MRG diameters. Fibers are straight & co-sampled (all path_lengths equal)."""
    d = np.load(FEM, allow_pickle=True)
    pl = d["path_lengths"].astype(int)
    assert len(np.unique(pl)) == 1, f"non-uniform fiber sampling {np.unique(pl)}"
    n, npts = len(pl), int(pl[0])
    Ve = d["Ve_flat"].reshape(n, npts).astype(float) * VE_SCALE
    pf = d["paths_flat"].reshape(n, npts, 3).astype(float)
    s_um = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pf[0], axis=0), axis=1))]) * 1e6
    diam = np.load(DIAM).astype(float)
    return Ve, s_um, diam


def _thr(Vi, s_um, diam_i, pw_ms):
    from golgi.pipeline.fiber_sim import FiberSimJobRequest
    from golgi.pipeline.sweep import _bisect_threshold
    H = SimpleNamespace(build_pulse_breakpoints=_bp,
                        MYELINATED_MODELS=["MRG_INTERPOLATION", "SMALL_MRG_INTERPOLATION"],
                        UNMYELINATED_MODELS=["SUNDT"])
    pp = dict(PULSE, cath_pw_ms=pw_ms, tstop=max(3.0, pw_ms + 2.0))
    req = FiberSimJobRequest(sel=0, s_um=s_um, Ve_mV=Vi, diameter_um=float(diam_i),
                             length_um=float(s_um.max()), pulse_params=pp,
                             backend="pyfibers", model_name="MRG_INTERPOLATION", helpers=H)
    th, _ = _bisect_threshold(req, 0.0005, 0.5, 0.2)   # 0.5 uA tol, 0.5-500 uA range; returns mA
    return th


def _worker(args):
    idxs, Ve, s_um, diam, pw_ms = args
    out = np.full((len(idxs), len(pw_ms)), np.nan)
    for r, i in enumerate(idxs):
        for c, pw in enumerate(pw_ms):
            try:
                out[r, c] = _thr(Ve[i], s_um, diam[i], pw)
            except Exception:
                out[r, c] = np.nan
    return idxs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0, help="serial calibration on N fibers")
    a = ap.parse_args()
    Ve, s_um, diam = load_fem()
    n = len(diam)
    print(f"[nrv-fem] {n} MRG fibers, diam {diam.mean():.1f}+/-{diam.std():.1f} um "
          f"[{diam.min():.1f},{diam.max():.1f}]; fiber len {s_um.max()/1e3:.1f} mm; "
          f"|Ve|peak {np.abs(Ve).max():.0f} mV/mA; {len(PWS_MS)} pulse widths", flush=True)

    if a.smoke:
        # serial calibration on the strongest-coupled fibers (largest peak |Ve|);
        # thresholds should land in single-uA..tens-of-uA, matching NRV/in-vivo
        idx = np.argsort(-np.abs(Ve).max(1))[:a.smoke]
        for i in idx:
            for pw in (0.02, 0.05, 0.2):
                t0 = time.perf_counter()
                th = _thr(Ve[i], s_um, diam[i], pw)        # already uA
                print(f"  fiber {i:3d} d={diam[i]:4.1f}um peak|Ve|={np.abs(Ve[i]).max():.0f} "
                      f"pw={pw}ms -> thr={th:7.2f} uA  ({time.perf_counter()-t0:.1f}s)", flush=True)
        return

    chunks = np.array_split(np.arange(n), a.workers * 2)
    tasks = [(list(ch), Ve, s_um, diam, PWS_MS) for ch in chunks if len(ch)]
    thr = np.full((n, len(PWS_MS)), np.nan); t0 = time.perf_counter(); done = 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for idxs, block in pool.imap_unordered(_worker, tasks):
            thr[idxs] = block; done += len(idxs)           # _bisect_threshold returns uA
            print(f"  {done}/{n} fibers  {time.perf_counter()-t0:5.1f}s", flush=True)

    # save the FULL threshold matrix so all curves/metrics are recomputable offline
    np.savez(DATA / "validate_nrv_thr.npz", thr=thr, diam=diam, pws=np.array(PWS_MS))
    i20, i50 = PWS_MS.index(0.02), PWS_MS.index(0.05)
    # window must span BOTH curves (20 us thresholds are highest) into their 80% range
    hi = float(np.nanpercentile(thr[:, i20], 92)) * 1.1
    amps = np.linspace(0, max(hi, 120.0), 400)
    rc20, rc50 = recruit(thr[:, i20], amps), recruit(thr[:, i50], amps)
    rr = rate(rc50, amps) / rate(rc20, amps)               # 50->20 us rate ratio (scaling-robust)
    dur = np.array([p * 1e3 for p in PWS_MS])              # us
    res = dict(source="native_fem", n_fibers=int(n), amps_uA=amps.tolist(),
               rec_50us=rc50.tolist(), rec_20us=rc20.tolist(), dur_us=dur.tolist(),
               thr_50us_uA=thr[:, i50].tolist(), thr_20us_uA=thr[:, i20].tolist(),
               diam_um=diam.tolist(), rate_ratio=float(rr), ref=REF)
    (DATA / "validate_nrv.json").write_text(json.dumps(res, indent=2))
    fin = np.isfinite(thr)
    print(f"[nrv-fem] thr finite {fin.sum()}/{thr.size}; 50us median "
          f"{np.nanmedian(thr[:, i50]):.0f} uA, 20us median {np.nanmedian(thr[:, i20]):.0f} uA", flush=True)
    print(f"[nrv-fem] strength-duration rate ratio (50->20us) golgi {rr:.2f} | "
          f"in-vivo {REF['rate_ratio_invivo']} (NRV {REF['rate_ratio_nrv']}); "
          f"wrote validate_nrv.json + validate_nrv_thr.npz", flush=True)


if __name__ == "__main__":
    main()
