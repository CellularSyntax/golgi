# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Validation V1+V2 — single-fiber biophysics vs literature, using golgi's EXACT
pyfibers/NEURON path (FiberSimJobRequest + _bisect_threshold), with a synthetic
monopolar point-source extracellular field on a straight MRG fiber.

  CV vs diameter   — conduction velocity of myelinated fibers vs the Hursh (1939)
                     ~6 m/s per um rule and the McIntyre 2002 (MRG) values.
  Strength-duration — threshold vs pulse width for a 10 um fiber; fit the
                     Weiss/Lapicque law to extract rheobase + chronaxie, compare
                     to the literature range for large myelinated fibers.

Writes paper_figs/out/data/validate_fiber.json.
Run AFTER the rabbit FEM frees the CPU (single-fiber, but keep it clean).
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import DATA   # noqa: E402
from fig5_thresholds import _bp, _taper, PULSE   # noqa: E402

# Standard MRG outer diameters (McIntyre, Richardson & Grill 2002). The McIntyre
# reference conduction velocity is computed here from PyFibers' discrete MRG model
# (the faithful McIntyre-2002 implementation) through the SAME extracellular pipeline
# as golgi's interpolated MRG, so the two are directly comparable. McIntyre's CV-vs-
# diameter is two-regime (~4.6 m/s/um for small fibers, ~5.7 for large), NOT a single
# ~5.8 ratio; computing it avoids the error of hard-coding linearised values.
CV_DIAMS = [5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 16.0]
SD_PWS_MS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]
SD_DIAM = 10.0
SIGMA = 0.2           # S/m, representative tissue conductivity for the point source
R0_MM = 1.0           # electrode-to-fiber distance for the monopole


def point_source_Ve_mV(s_um, r0_mm=R0_MM, sigma=SIGMA):
    """Cathodic monopole at mid-fiber: Ve (mV) per 1 mA, V=I/(4 pi sigma r)."""
    zc = 0.5 * (s_um.min() + s_um.max())
    r_m = np.sqrt((r0_mm * 1e-3) ** 2 + ((s_um - zc) * 1e-6) ** 2)
    return -(1e-3) / (4.0 * np.pi * sigma * r_m) * 1e3      # cathodic, mV per mA


def _H():
    return SimpleNamespace(build_pulse_breakpoints=_bp,
                           MYELINATED_MODELS=["MRG_INTERPOLATION", "SMALL_MRG_INTERPOLATION",
                                              "MRG_DISCRETE"],
                           UNMYELINATED_MODELS=["SUNDT"])


def _req(s_um, Ve_mV, diam, amp_mA, pw_ms, tstop, model="MRG_INTERPOLATION"):
    from golgi.pipeline.fiber_sim import FiberSimJobRequest
    from golgi.pipeline.sweep import _scaled_pulse_params
    pp = _scaled_pulse_params(dict(PULSE, cath_pw_ms=pw_ms, tstop=tstop), amp_mA)
    return FiberSimJobRequest(sel=0, s_um=s_um, Ve_mV=Ve_mV,
                              diameter_um=float(diam), length_um=float(s_um.max()),
                              pulse_params=pp, backend="pyfibers",
                              model_name=model, helpers=_H())


def threshold_uA(s_um, Ve_mV, diam, pw_ms, tstop, hi_mA=15.0, model="MRG_INTERPOLATION"):
    from golgi.pipeline.sweep import _bisect_threshold
    th, _ = _bisect_threshold(_req(s_um, Ve_mV, diam, 1.0, pw_ms, tstop, model),
                              0.005, hi_mA, 5.0)
    return th


def cv_from(sd):
    st = np.asarray(sd["spike_t"], float); z = np.asarray(sd["node_z_um"], float) / 1e3
    fin = np.isfinite(st)
    if fin.sum() < 5:
        return np.nan
    zf, tf = z[fin], st[fin]; k = int(np.argmin(tf))
    right, left = zf > zf[k], zf < zf[k]
    side = right if right.sum() >= left.sum() else left
    if side.sum() < 3:
        return np.nan
    return abs(np.polyfit(tf[side], zf[side], 1)[0])         # mm/ms = m/s


def _cv_for(s_um, Ve, d, tstop, model):
    """Threshold then propagate at 1.3x threshold; return conduction velocity (m/s)."""
    from golgi.pipeline.fiber_sim import _do_one_fiber
    th = threshold_uA(s_um, Ve, d, pw_ms=0.1, tstop=tstop, model=model)
    if not np.isfinite(th):
        return None
    amp = 1.3 * th / 1e3
    sd = _do_one_fiber(_req(s_um, Ve, d, amp, 0.1, tstop, model),
                       on_line=None, cancel=None).outputs["sim_data"]
    return float(cv_from(sd))


def run_cv():
    L_um = 70_000.0                                          # 70 mm, long enough for slow fibers
    s_um = np.linspace(0.0, L_um, 1401)
    Ve = point_source_Ve_mV(s_um) * _taper(s_um)
    out = []
    for d in CV_DIAMS:
        tstop = 12.0                                         # covers the slowest here (~24 m/s)
        cv = _cv_for(s_um, Ve, d, tstop, "MRG_INTERPOLATION")     # golgi (interpolated MRG)
        mrg = _cv_for(s_um, Ve, d, tstop, "MRG_DISCRETE")        # McIntyre 2002 (discrete MRG)
        out.append(dict(diam=d, cv=cv, hursh=6.0 * d, mrg_ref=mrg))
        print(f"  d={d:>4} um: golgi CV={cv if cv is None else round(cv,1)} m/s | "
              f"MRG (McIntyre) {mrg if mrg is None else round(mrg,1)} | Hursh 6d={6*d:4.0f}",
              flush=True)
    return out


def run_sd():
    L_um = 40_000.0
    s_um = np.linspace(0.0, L_um, 801)
    Ve = point_source_Ve_mV(s_um) * _taper(s_um)
    pts = []
    for pw in SD_PWS_MS:
        th = threshold_uA(s_um, Ve, SD_DIAM, pw_ms=pw, tstop=max(3.0, pw + 2.0))
        pts.append(dict(pw_ms=pw, th_mA=(th / 1e3 if np.isfinite(th) else None)))
        print(f"  PW={pw:>4} ms: I_th={th/1e3 if np.isfinite(th) else float('nan'):.4f} mA", flush=True)
    # Weiss fit: I_th = I_rheo * (1 + chronaxie / PW)  ->  I_th = a + b/PW (a=I_rheo, b=I_rheo*chronaxie)
    pw = np.array([p["pw_ms"] for p in pts if p["th_mA"]])
    it = np.array([p["th_mA"] for p in pts if p["th_mA"]])
    A = np.vstack([np.ones_like(pw), 1.0 / pw]).T
    a, b = np.linalg.lstsq(A, it, rcond=None)[0]
    rheo, chron = float(a), float(b / a)
    print(f"  Weiss fit: rheobase={rheo:.4f} mA, chronaxie={chron*1e3:.1f} us", flush=True)
    return dict(diam=SD_DIAM, points=pts, rheobase_mA=rheo, chronaxie_ms=chron)


def main():
    print("[validate_fiber] CV vs diameter ...", flush=True)
    cv = run_cv()
    print("[validate_fiber] strength-duration ...", flush=True)
    sd = run_sd()
    # Hursh slope fit (CV = k * diam) over the valid points
    dv = np.array([r["diam"] for r in cv if r["cv"]])
    cvv = np.array([r["cv"] for r in cv if r["cv"]])
    k = float(np.linalg.lstsq(dv[:, None], cvv, rcond=None)[0][0]) if len(dv) else None
    res = dict(cv_diameter=cv, hursh_slope_fit=k, strength_duration=sd,
               point_source=dict(sigma_S_per_m=SIGMA, r0_mm=R0_MM))
    (DATA / "validate_fiber.json").write_text(json.dumps(res, indent=2))
    print(f"[validate_fiber] CV-diam slope = {k:.2f} m/s/um (Hursh ~6); "
          f"chronaxie = {sd['chronaxie_ms']*1e3:.0f} us; wrote validate_fiber.json", flush=True)


if __name__ == "__main__":
    main()
