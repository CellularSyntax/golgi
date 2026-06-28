# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Dog-VNS validation, NATIVE rebuild — replicate ASCENT's yoo_2013 in-vivo dog
cervical VNS (Yoo 2013) faithfully with golgi's own pipeline: a monofascicular
3.4 mm endoneurium + 197 um perineurium contact-impedance sheet (sigma=1/1149 S/m,
ASCENT's value) inside a LivaNova-style HELICAL bipolar cuff, solved with the
native reciprocity FEM. Then A/B/C activation thresholds for a fiber of each type
at the fascicle centroid (MRG 7.8/3.6/2.1 um + Tigerholm 1.0 um, 300 us monophasic
cathodic) vs the in-vivo ENG ranges.

Stages:  --mesh   --fem (mesh+FEM+lead field)
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, shutil, argparse
from pathlib import Path
import numpy as np

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT))
PROJ = ROOT / "paper_figs/out/_intermediate/dogvns_native"
EID = "livanova"
# ASCENT yoo_2013 morphology (sample.json: areas -> equiv diameters; a,b are FULL axes,
# so area = pi*(a/2)*(b/2)). Endoneurium (fascicle inner) ~1.70 mm, perineurium annulus
# ~98 um, nerve (epineurium) ~3.27 mm: a SMALL fascicle inside a thick epineurium. The
# LivaNova cuff is expandable ("expandable": true) -> the helix opens to wrap the 3.27 mm
# nerve (it is NOT compressed into the 2 mm nominal bore).
ENDO_D_M = 1.70e-3                          # fascicle (endoneurium) equiv diameter
NERVE_D_M = 3.27e-3                         # nerve (epineurium) equiv diameter
EPI_THK_M = (NERVE_D_M - ENDO_D_M) / 2      # epineurium shell (~785 um)
PERI_THK_M = 98e-6                          # perineurium CI-sheet thickness (annulus)
NERVE_R_M = NERVE_D_M / 2                   # imported surface = epineurium outer
NERVE_L_M = 28.0e-3
L_CUFF_MM = 20.0                            # helix 2 bands -> sep = 0.3*L_cuff = 6 mm
SIGMA_PERI = 1.0 / 1149.0                   # ASCENT (Weerasuriya @ 1 Hz)


def make_nerve():
    import pyvista as pv
    PROJ.mkdir(parents=True, exist_ok=True)
    line = pv.Line((0, 0, 0), (0, 0, NERVE_L_M), resolution=90)
    cyl = line.tube(radius=NERVE_R_M, n_sides=64, capping=True).triangulate()
    stl = PROJ / "nerve.stl"; cyl.save(str(stl))
    return stl


def recon():
    import golgi
    if (PROJ / "study").exists():
        shutil.rmtree(PROJ / "study")
    stl = make_nerve()
    s = golgi.Study.create(PROJ / "study")
    info = s.import_nerve(stl, scale_factor=1.0)
    print(f"[recon] import: bbox_mm={tuple(round(x,2) for x in info['bbox_mm'])} "
          f"watertight={info['watertight']}", flush=True)
    s.set_mesh(use_epi=True, epi_thickness_um=EPI_THK_M * 1e6, decim_target_k=40)
    # perineurium contact-impedance sheet at the endo<->epi interface (ASCENT BC)
    s._state.perineurium_ci = True
    s._state.peri_thk_m = PERI_THK_M
    s._state.sigma_peri = SIGMA_PERI
    s._state.cuff_anchor = "trunk (low z)"
    s._state.L_cuff_mm = L_CUFF_MM
    s.set_electrodes([{"eid": EID, "name": "LivaNova", "cuff_offset_mm": NERVE_L_M / 2 * 1e3,
                       "L_cuff_mm": L_CUFF_MM, "cuff_anchor": "trunk (low z)",
                       "electrode_type": "helical (Livanova-style)",
                       "helix_n_bands": 2, "helix_pitch_mm": 2.1, "helix_dphi_deg": 320.0}])
    from golgi.scene.cuff_fit import refit_design_geometry
    ok = refit_design_geometry(EID, geom=s._geom, state=s._state)
    print(f"[recon] LivaNova helix set + refit -> {ok}; peri_ci=True Rs={PERI_THK_M/SIGMA_PERI:.4f} ohm*m^2",
          flush=True)
    return s


# type, diameter (um), fiber model, in-vivo ENG range (mA, Yoo 2013 / ASCENT Fig 5B), bisection (lo,hi mA)
FIBER_TYPES = [
    ("A (7.8 um)",      7.8, "MRG_INTERPOLATION", (0.1, 0.6),   (0.0002, 3.0)),
    ("fast B (3.6 um)", 3.6, "MRG_INTERPOLATION", (1.0, 3.0),   (0.0005, 8.0)),
    ("slow B (2.1 um)", 2.1, "MRG_INTERPOLATION", (2.5, 6.0),   (0.001, 20.0)),
    ("C (1.0 um)",      1.0, "TIGERHOLM",         (10.0, 25.0), (0.01, 100.0)),
]


def recruit():
    import json
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / "paper_figs"))
    from fig5_thresholds import _bp, _taper, PULSE
    from io_paths import DATA
    from golgi.pipeline.fiber_sim import FiberSimJobRequest
    from golgi.pipeline.sweep import _bisect_threshold, _scaled_pulse_params
    rec = PROJ / "study" / "designs" / EID / "recording"
    d0 = np.load(rec / "V_e_rec_0.npz"); d1 = np.load(rec / "V_e_rec_1.npz")
    flat = d0["paths_flat"]; lens = d0["path_lengths"].astype(int)
    Ve0 = d0["Ve_flat"].astype(float); Ve1 = d1["Ve_flat"].astype(float)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    i = 0                                     # centroid fiber (xy[0] = [0,0])
    seg = flat[off[i]:off[i + 1]]
    s_um = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]) * 1e6
    Ve_bip = -(Ve0[off[i]:off[i + 1]] - Ve1[off[i]:off[i + 1]])   # cathodic bipolar, mV per 1 mA
    print(f"[recruit] centroid fiber: {len(s_um)} pts, span {s_um.max()/1e3:.1f} mm, "
          f"Ve_bip peak {np.abs(Ve_bip).max():.3g} mV/mA", flush=True)
    H = SimpleNamespace(build_pulse_breakpoints=_bp,
                        MYELINATED_MODELS=["MRG_INTERPOLATION", "SMALL_MRG_INTERPOLATION"],
                        UNMYELINATED_MODELS=["SUNDT", "TIGERHOLM"])
    out = []
    for name, diam, model, invivo, (lo, hi) in FIBER_TYPES:
        Ve = Ve_bip * _taper(s_um)
        pp = _scaled_pulse_params(dict(PULSE, cath_pw_ms=0.3, tstop=6.0), 1.0)
        req = FiberSimJobRequest(sel=0, s_um=s_um, Ve_mV=Ve, diameter_um=diam,
                                 length_um=float(s_um.max()), pulse_params=pp,
                                 backend="pyfibers", model_name=model, helpers=H)
        try:
            th_uA, _ = _bisect_threshold(req, lo, hi, 8.0)   # lo,hi in mA; returns uA
            th = (th_uA / 1e3) if np.isfinite(th_uA) else None   # -> mA
        except Exception:
            th = None
        inb = (th is not None and invivo[0] <= th <= invivo[1])
        out.append({"type": name, "diam_um": diam, "thr_mA": th, "invivo_mA": list(invivo), "in_band": inb})
        ts = ("%.3f" % th) if th is not None else "NA"
        print(f"  {name:16s} thr={ts:>8} mA   in-vivo {invivo}   {'IN-RANGE' if inb else 'out'}", flush=True)
    res = {"pulse_width_ms": 0.3, "cuff": "expandable LivaNova helical (native FEM)",
           "nerve": "monofascicular 1.7 mm fascicle in 3.27 mm nerve + perineurium CI sheet",
           "thresholds": out}
    (DATA / "validate_dogvns_native.json").write_text(json.dumps(res, indent=2))
    print("[recruit] wrote validate_dogvns_native.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", action="store_true")
    ap.add_argument("--fem", action="store_true")
    ap.add_argument("--recruit", action="store_true")
    a = ap.parse_args()
    if a.recruit:
        recruit(); return
    s = recon()
    print("[mesh] building native mesh ...", flush=True)
    msh = s.run_mesh()
    print(f"[mesh] -> {msh if isinstance(msh,(dict,str)) else type(msh).__name__}", flush=True)
    mpath = PROJ / "study" / "designs" / EID / "nerve.msh"
    print(f"[mesh] nerve.msh exists={mpath.is_file()} size={mpath.stat().st_size if mpath.is_file() else 0}", flush=True)
    if a.mesh and not a.fem:
        return
    # straight fibers spanning the nerve at the fascicle centroid + a small spread,
    # so the centroid fiber of each type is well-defined
    z = np.linspace(0.5e-3, NERVE_L_M - 0.5e-3, 501)
    R_ENDO = ENDO_D_M / 2
    rng = np.random.default_rng(0)
    rr = R_ENDO * np.sqrt(rng.uniform(0, 1, 40)); th = rng.uniform(0, 2 * np.pi, 40)
    xy = np.column_stack([rr * np.cos(th), rr * np.sin(th)]); xy[0] = [0.0, 0.0]   # idx0 = centroid
    fibers = [np.column_stack([np.full_like(z, x), np.full_like(z, y), z]) for x, y in xy]
    s._geom.fiber_paths_raw = fibers
    s._geom.msh_path = str(mpath); s._state.has_mesh = True; s._state.emit_impedance = False
    cfgs = list(s._state.configs)
    for c in cfgs:
        if c.get("design_id") == EID:
            c["recording_montages"] = [{"mid": "biphelix", "label": "biphelix", "kind": "bipolar",
                                        "plus_contact": 0, "minus_contact": 1}]
    s._state.configs = cfgs
    np.save(PROJ / "fiber_xy.npy", xy)
    print("[fem] running native LivaNova reciprocity FEM ...", flush=True)
    s.run_fem()
    rec = PROJ / "study" / "designs" / EID / "recording"
    print(f"[fem] recording files: {sorted(p.name for p in rec.glob('*.npz')) if rec.exists() else 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
