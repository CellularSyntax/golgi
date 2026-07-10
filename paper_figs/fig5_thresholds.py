# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 5 engine: per-(contact, fiber) activation-threshold matrix for a
golgi nerve, golgi's EXACT pyfibers/NEURON threshold path, parallelised.

Two population modes:
  --diam 10            fixed-diameter (controlled spatial selectivity)
  --pop cervical_vagus_pig|cervical_vagus_human
                       realistic myelinated (A-alpha/beta/delta + B from the
                       species vagus preset) + --n-c C-fibers (SUNDT) for the
                       C-sparing demonstration. Per-fiber (diameter, model).

Outputs .npz: thr_uA [n_fibers x n_contacts] (NaN = not recruited <= hi),
branch_idx, diameter_um, model[], type_label[], xy_cuff_mm, contact_ids, meta.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time, argparse, multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__import__("os").environ.get("GOLGI_PAPER_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

PULSE = dict(t0=0.1, tstop=3.0, cath_amp_mA=1.0, cath_pw_ms=0.1, gap_ms=0.0,
             anod_amp_mA=0.0, anod_pw_ms=0.0, anode_first=False, kind="monophasic")
LO_MA, HI_MA = 0.005, 10.0
UNMYEL = {"SUNDT", "TIGERHOLM", "RATTAY", "SCHILD94", "SCHILD97"}
CLIP = {"MRG_INTERPOLATION": (2.0, 16.0), "SMALL_MRG_INTERPOLATION": (1.5, 5.0)}


def _bp(t0_ms, cath_amp_mA, cath_pw_ms, gap_ms, anod_amp_mA, anod_pw_ms,
        anode_first, tstop_ms):
    t1_hi = t0_ms + float(cath_pw_ms)
    return (np.array([0.0, float(t0_ms), t1_hi, float(tstop_ms)]),
            np.array([0.0, float(cath_amp_mA), 0.0, 0.0]))


def _load(npz_path, contact_subset=None, trunc_um=0.0):
    d = np.load(npz_path, allow_pickle=True)
    flat, Ve, lens = d["paths_flat"], d["Ve_mat"], d["path_lengths"]
    branch, cids = d["branch_idx"], d["contact_ids"]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    csel = list(range(Ve.shape[1])) if contact_subset is None else contact_subset
    fibers = []
    for i in range(len(lens)):
        a, b = off[i], off[i + 1]
        xyz = flat[a:b]
        ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        s_um = np.concatenate([[0.0], np.cumsum(ds)]) * 1e6
        Vi = Ve[a:b][:, csel].astype(np.float64)
        xy = xyz[:, 0:2].mean(axis=0) * 1e3
        if trunc_um > 0:                       # keep window around the cuff (|Ve| peak), where AP initiates
            pk = float(s_um[int(np.argmax(np.abs(Vi).max(axis=1)))])
            keep = np.abs(s_um - pk) <= trunc_um
            if keep.sum() >= 5:
                s_um = s_um[keep] - s_um[keep].min(); Vi = Vi[keep]
        fibers.append((s_um, Vi, int(branch[i]), xy))
    return fibers, np.asarray(cids)[csel], len(csel)


def sample_pop(preset_name, n_total, n_c, rng):
    from golgi.state_defaults.pop_presets import POP_PRESETS
    rows = POP_PRESETS[preset_name].templates[0].rows
    myel = [r for r in rows if r.model not in UNMYEL]
    crow = [r for r in rows if r.model in UNMYEL][0]
    n_c = min(n_c, n_total); n_my = n_total - n_c
    w = np.array([r.frac for r in myel], float); w /= w.sum()
    diam = np.zeros(n_total); model = np.empty(n_total, object); typ = np.empty(n_total, object)
    sel = rng.choice(len(myel), size=n_my, p=w)
    for k in range(n_my):
        r = myel[sel[k]]; lo, hi = CLIP.get(r.model, (1.0, 16.0))
        diam[k] = float(np.clip(rng.normal(r.mean_um, r.std_um), lo, hi))
        model[k] = r.model
        typ[k] = r.name.split("(")[0].split("/")[0].strip()[:7]
    for k in range(n_my, n_total):
        diam[k] = float(np.clip(rng.normal(crow.mean_um, crow.std_um), 0.25, 2.0))
        model[k] = crow.model; typ[k] = "C"
    p = rng.permutation(n_total)
    return diam[p], model[p], typ[p]


def _taper(s_um, frac=0.2):
    """Cosine taper to 0 over the outer `frac` of the fiber at each end. Driving
    the extracellular field (and its gradient) to zero at the sealed terminals
    removes the virtual-electrode 'end excitation' that otherwise produces
    spuriously low thresholds for long pulses. The contact sits mid-fiber, so the
    central activation region is untouched."""
    s = np.asarray(s_um, float); s = s - s.min(); L = s.max()
    if L <= 0:
        return np.ones_like(s)
    a = max(frac * L, 1e-9); w = np.ones_like(s)
    lo = s < a; hi = s > (L - a)
    w[lo] = 0.5 * (1 - np.cos(np.pi * s[lo] / a))
    w[hi] = 0.5 * (1 - np.cos(np.pi * (L - s[hi]) / a))
    return w


def _worker(args):
    idxs, npz_path, diam, model, tol_uA, csel, trunc_um, pw_ms, tstop_ms, hi_mA = args
    from golgi.pipeline.fiber_sim import FiberSimJobRequest
    from golgi.pipeline.sweep import _bisect_threshold
    H = SimpleNamespace(build_pulse_breakpoints=_bp,
                        MYELINATED_MODELS=list(CLIP), UNMYELINATED_MODELS=list(UNMYEL))
    fibers, _, ncon = _load(npz_path, csel, trunc_um)
    pp = dict(PULSE, cath_pw_ms=pw_ms, tstop=tstop_ms)
    out = np.full((len(idxs), ncon), np.nan)
    for r, i in enumerate(idxs):
        s_um, Vi, br, xy = fibers[i]
        w = _taper(s_um)
        for c in range(ncon):
            req = FiberSimJobRequest(
                sel=0, s_um=s_um, Ve_mV=Vi[:, c] * w, diameter_um=float(diam[i]),
                length_um=float(s_um.max()), pulse_params=pp,
                backend="pyfibers", model_name=str(model[i]), helpers=H)
            try:
                th, _ = _bisect_threshold(req, LO_MA, hi_mA, tol_uA)
            except Exception:
                th = np.nan
            out[r, c] = th
    return idxs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nerve_dir")
    ap.add_argument("--n-fibers", type=int, default=700)
    ap.add_argument("--diam", default="10")
    ap.add_argument("--pop", default="")          # preset name -> realistic population
    ap.add_argument("--n-c", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--tol-uA", type=float, default=20.0)
    ap.add_argument("--contacts", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trunc-mm", type=float, default=0.0,
                    help="keep only +/- this axial window per fiber (0 = full)")
    ap.add_argument("--pw-ms", type=float, default=0.1, help="cathodic pulse width (ms)")
    ap.add_argument("--hi-mA", type=float, default=10.0, help="bisection ceiling (mA)")
    ap.add_argument("--tstop-ms", type=float, default=3.0, help="sim duration (ms)")
    a = ap.parse_args()

    nd = Path(a.nerve_dir); npz = nd / "paths_Ve.npz"
    csel = None if not a.contacts else [int(x) for x in a.contacts.split(",")]
    trunc_um = a.trunc_mm * 1e3
    fibers, cids, ncon = _load(npz, csel, trunc_um)
    nfib_total = len(fibers)
    rng = np.random.default_rng(a.seed)
    sub = np.sort(rng.choice(nfib_total, size=min(a.n_fibers, nfib_total), replace=False))
    branch = np.array([fibers[i][2] for i in range(nfib_total)])
    xy = np.array([fibers[i][3] for i in range(nfib_total)])

    dd = np.full(nfib_total, 10.0)
    mm = np.array(["MRG_INTERPOLATION"] * nfib_total, object)
    tt = np.array([""] * nfib_total, object)
    if a.pop:
        ds, ms, ts = sample_pop(a.pop, len(sub), a.n_c, rng)
        dd[sub] = ds; mm[sub] = ms; tt[sub] = ts
        comp = {t: int((ts == t).sum()) for t in sorted(set(ts))}
        print(f"population {a.pop}: {comp}", flush=True)
    else:
        dd[:] = float(a.diam)

    print(f"nerve={nd.name} fibers={len(sub)}/{nfib_total} contacts={ncon} "
          f"pop={a.pop or a.diam} tol={a.tol_uA} workers={a.workers}", flush=True)
    print(f"pulse: {a.pw_ms}ms monophasic | ceiling {a.hi_mA}mA | tstop {a.tstop_ms}ms", flush=True)
    chunks = np.array_split(sub, a.workers * 3)
    tasks = [(list(ch), str(npz), dd, mm, a.tol_uA, csel, trunc_um, a.pw_ms, a.tstop_ms, a.hi_mA)
             for ch in chunks if len(ch)]
    t0 = time.perf_counter(); thr = np.full((nfib_total, ncon), np.nan); done = 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for idxs, block in pool.imap_unordered(_worker, tasks):
            thr[idxs] = block; done += len(idxs)
            el = time.perf_counter() - t0
            print(f"  {done}/{len(sub)} fibers  {el:6.1f}s ({el/max(done,1):.2f}s/fiber)", flush=True)
    el = time.perf_counter() - t0

    np.savez_compressed(
        a.out, thr_uA=thr[sub], fiber_idx=sub, branch_idx=branch[sub],
        diameter_um=dd[sub], model=mm[sub].astype(str), type_label=tt[sub].astype(str),
        xy_cuff_mm=xy[sub], contact_ids=cids,
        meta=json.dumps(dict(nerve=nd.name, n_fibers=int(len(sub)), n_contacts=int(ncon),
                             pop=a.pop or None, diam=a.diam, n_c=a.n_c if a.pop else 0,
                             tol_uA=a.tol_uA, pw_ms=a.pw_ms, hi_mA=a.hi_mA,
                             tstop_ms=a.tstop_ms, elapsed_s=el)))
    rec = np.isfinite(thr[sub])
    print(f"done {el:.1f}s | recruited(<= {a.hi_mA}mA): "
          f"{rec.any(axis=1).mean()*100:.0f}% of fibers by >=1 contact -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
