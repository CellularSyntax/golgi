# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Validation V5 — replicate the NRV in-vivo benchmark (Couppey et al. 2024,
Fig 9): cat tibial nerve, intrafascicular LIFE electrode, myelinated (MRG) axons,
recruitment vs stimulus current (50 / 20 us cathodic) and recruitment vs cathodic
pulse duration (7 / 8 / 9 uA), against in-vivo data of Nannini & Horch 1991 (NH)
and Yoshida & Horch 1993 (YH).

golgi side: the LIFE active site is modeled as the canonical intrafascicular line
current source in the anisotropic endoneurium (the standard model for a LIFE),
and per-fiber thresholds use golgi's exact pyfibers/NEURON MRG path. We report
the recruitment curves and the two metrics NRV compared: the recruitment-rate
ratio (50->20 us) and the 7->9 uA recruitment increment.

Writes paper_figs/out/data/validate_nrv.json.
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
from fig5_thresholds import _bp, _taper, PULSE   # noqa: E402
from io_paths import DATA   # noqa: E402

SIG_T, SIG_L = 0.1667, 0.5714      # endoneurium transverse / longitudinal conductivity (S/m)
FASC_R_UM = 275.0                  # 550 um fascicle diameter
L_ACTIVE_M = 1.0e-3                # LIFE active-site length
FIB_LEN_UM = 30_000.0
# in-vivo / NRV-reported reference metrics (Couppey 2024, Fig 9 text)
REF = {"rate_ratio_invivo": 2.1, "rate_ratio_nrv": 2.4,
       "incr79_invivo": 0.37, "incr79_nrv": 0.48}
# PWs (ms) needed for both panels: 0.02/0.05 (vs-current) + a spread (vs-duration)
PWS_MS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def life_Ve_mV(d_um, s_um, I_ref_mA=1.0, n_src=21):
    """Anisotropic line-source potential (mV per I_ref) along a straight fiber at
    transverse distance d from a LIFE active site of length L_ACTIVE_M."""
    z = (s_um - s_um.mean()) * 1e-6                       # m, centered on the active site
    d2 = (d_um * 1e-6) ** 2
    src = np.linspace(-L_ACTIVE_M / 2, L_ACTIVE_M / 2, n_src)
    V = np.zeros_like(z)
    for sz in src:
        V += 1.0 / np.maximum(np.sqrt(d2 / SIG_T + (z - sz) ** 2 / SIG_L), 1e-9)
    V *= (I_ref_mA * 1e-3 / n_src) / (4 * np.pi * np.sqrt(SIG_T ** 2 * SIG_L))
    return V * 1e3                                        # V -> mV


def sample_pop(n, seed=0):
    """n myelinated fibers: random transverse distance d (uniform area in the
    fascicle) + MRG diameter from a broad cat-tibial-like distribution (1-16 um)."""
    rng = np.random.default_rng(seed)
    d_um = FASC_R_UM * np.sqrt(rng.uniform(0, 1, n))      # uniform-area radius
    diam = np.clip(rng.normal(10.0, 3.2, n), 2.0, 16.0)   # large-myelinated-dominated (GM motor)
    return d_um, diam


def _worker(args):
    idxs, d_um, diam, pw_ms = args
    from golgi.pipeline.fiber_sim import FiberSimJobRequest
    from golgi.pipeline.sweep import _bisect_threshold
    H = SimpleNamespace(build_pulse_breakpoints=_bp,
                        MYELINATED_MODELS=["MRG_INTERPOLATION", "SMALL_MRG_INTERPOLATION"],
                        UNMYELINATED_MODELS=["SUNDT"])
    s_um = np.linspace(0.0, FIB_LEN_UM, 601)
    out = np.full((len(idxs), len(pw_ms)), np.nan)        # threshold current (mA) per fiber per PW
    for r, i in enumerate(idxs):
        Ve = life_Ve_mV(d_um[i], s_um) * _taper(s_um)
        for c, pw in enumerate(pw_ms):
            pp = dict(PULSE, cath_pw_ms=pw, tstop=max(3.0, pw + 2.0))
            req = FiberSimJobRequest(sel=0, s_um=s_um, Ve_mV=Ve, diameter_um=float(diam[i]),
                                     length_um=float(s_um.max()), pulse_params=pp,
                                     backend="pyfibers", model_name="MRG_INTERPOLATION", helpers=H)
            try:
                th, _ = _bisect_threshold(req, 0.0005, 0.5, 0.2)   # 0.5 uA tol, 0.5-500 uA range
            except Exception:
                th = np.nan
            out[r, c] = th
    return idxs, out


def recruit(thr_uA, amps_uA):
    t = thr_uA[np.isfinite(thr_uA)]
    return np.array([(t <= a).mean() if t.size else 0.0 for a in amps_uA])


def rate(curve, amps):
    """recruitment rate = slope between 20% and 80% recruitment (per uA)."""
    lo = np.interp(0.2, curve, amps); hi = np.interp(0.8, curve, amps)
    return 0.6 / max(hi - lo, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fibers", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d_um, diam = sample_pop(a.n_fibers)
    print(f"[nrv] {a.n_fibers} MRG fibers, d in [{d_um.min():.0f},{d_um.max():.0f}] um, "
          f"diam {diam.mean():.1f}+/-{diam.std():.1f} um; {len(PWS_MS)} pulse widths", flush=True)
    chunks = np.array_split(np.arange(a.n_fibers), a.workers * 2)
    tasks = [(list(ch), d_um, diam, PWS_MS) for ch in chunks if len(ch)]
    thr = np.full((a.n_fibers, len(PWS_MS)), np.nan); t0 = time.perf_counter(); done = 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for idxs, block in pool.imap_unordered(_worker, tasks):
            thr[idxs] = block * 1e3; done += len(idxs)     # mA -> uA
            print(f"  {done}/{a.n_fibers} fibers  {time.perf_counter()-t0:5.1f}s", flush=True)
    i20, i50 = PWS_MS.index(0.02), PWS_MS.index(0.05)
    amps = np.linspace(0, 60, 200)
    rc20, rc50 = recruit(thr[:, i20], amps), recruit(thr[:, i50], amps)
    rr = rate(rc50, amps) / rate(rc20, amps)               # 50->20 us rate ratio
    # recruitment vs duration at 7/8/9 uA
    dur = np.array([p * 1e3 for p in PWS_MS])              # us
    rec_byI = {I: np.array([(thr[:, c][np.isfinite(thr[:, c])] <= I).mean() for c in range(len(PWS_MS))])
               for I in (7, 8, 9)}
    # 7->9 uA increment (mean over durations >= 200 us, the YH range)
    msk = dur >= 200
    incr79 = float(np.mean(rec_byI[9][msk] - rec_byI[7][msk]))
    res = dict(n_fibers=a.n_fibers, amps_uA=amps.tolist(), rec_50us=rc50.tolist(), rec_20us=rc20.tolist(),
               dur_us=dur.tolist(), rec_by_current={str(k): v.tolist() for k, v in rec_byI.items()},
               rate_ratio=float(rr), incr79=incr79, ref=REF)
    (DATA / "validate_nrv.json").write_text(json.dumps(res, indent=2))
    print(f"[nrv] rate ratio (50->20us) golgi {rr:.2f} | in-vivo {REF['rate_ratio_invivo']} "
          f"(NRV {REF['rate_ratio_nrv']})", flush=True)
    print(f"[nrv] 7->9uA recruitment increment golgi {incr79:.2f} | in-vivo {REF['incr79_invivo']} "
          f"(NRV {REF['incr79_nrv']}); wrote validate_nrv.json", flush=True)


if __name__ == "__main__":
    main()
