# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Rabbit position-sweep FEM via the FULL-LENGTH STL-BUNDLE pipeline (same path as
new_human_mesh/new_human_fem) — NOT cropped windows. The whole nerve is meshed at every cuff
offset: EPI = the rabbit_out nerve surface (outer, full-length); ENDO = an inward offset of it
(the fascicle inside the epi shell). build_bundle_geom-style geom + FASCICLE_FULL_LENGTH means
the cuff window only cuts the conduction annuli OUTSIDE the full-length epi (so TetGen no longer
crashes on a cropped thin nerve). Shared rabbit_out streamline fibers (468, 13.5mm CORRECT
scale) are injected, then golgi's reciprocity FEM gives the 20-contact lead fields ->
rabbit_branch_{tag}/paths_Ve.npz.

env: CUFF_OFFSET_MM, ARR_ROWS=4, ARR_COLS=5, SWEEP_TAG, ENDO_OFFSET_UM, DECIM_PTS.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("GOLGI_FASCICLE_FULL_LENGTH", "1")
os.environ.setdefault("GOLGI_PLC_CDT", "1")
os.environ.setdefault("GOLGI_TETGEN_SWITCHES", "pzAaS150000")
os.environ.setdefault("GOLGI_TETGEN_EPSILON", "1e-6")
# tiny 0.34mm rabbit nerve: shrink the seam-snap + weld tolerances (defaults 50/8µm are tuned
# for mm-radius nerves and over-merge the rabbit into degenerate tris → TetGen recovery fails)
os.environ.setdefault("GOLGI_PLC_SEAM_UM", "5")
os.environ.setdefault("GOLGI_PLC_WELD_UM", "2")
import sys, json, shutil, math
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT))
# golgi spawns TetGen in a SUBPROCESS (compute/tetgen_runner.py) that inherits os.environ but
# NOT the parent's runtime sys.path.insert → it must find golgi via PYTHONPATH or it dies with
# "No module named 'golgi'" and run_mesh silently returns no nerve.msh. Make it self-contained.
os.environ["PYTHONPATH"] = str(ROOT) + (os.pathsep + os.environ["PYTHONPATH"]
                                        if os.environ.get("PYTHONPATH") else "")
RO = ROOT / "paper_figs/rabbit_out"                  # nerve surface + streamline fibers (13.5mm)
SURF = RO / "nerve_only_surface.npz"
OFFSET = float(os.environ.get("CUFF_OFFSET_MM", "8.0"))
NR = int(os.environ.get("ARR_ROWS", "4")); NC = int(os.environ.get("ARR_COLS", "5"))
TAG = os.environ.get("SWEEP_TAG") or f"off{OFFSET:g}_{NR}x{NC}"
PROJ = ROOT / f"paper_figs/out/_intermediate/rabbit_sweep_{TAG}"
OUT = ROOT / f"paper_figs/out/data/rabbit_branch_{TAG}"
EID = "elec_01"
ENDO_OFFSET_UM = float(os.environ.get("ENDO_OFFSET_UM", "150"))  # epi/perineurium shell; 70µm too
DECIM_PTS = int(os.environ.get("DECIM_PTS", "28000"))            # thin on 0.34mm nerve → TetGen SI


def _faces(tris):
    return np.hstack([np.full((len(tris), 1), 3, np.int64), tris]).ravel()


def _repair(v, f):
    """pymeshfix → watertight, self-intersection-free (same as the human endo prep)."""
    import pymeshfix
    mf = pymeshfix.MeshFix(v, f)
    mf.repair(joincomp=False, remove_smallest_components=False)
    return np.asarray(mf.mesh.points, float), mf.mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def _epi_endo():
    """EPI = rabbit nerve surface (outer); ENDO = inward offset by ENDO_OFFSET_UM (fascicle).
    Both decimated + pymeshfix-repaired (watertight, no self-intersections) so the bundle PLC
    feeds TetGen cleanly. Returns verts(m), faces."""
    d = np.load(SURF, allow_pickle=True)
    epi_v = np.asarray(d["pts_raw"], float); tris = np.asarray(d["tris"], np.int64)
    surf = pv.PolyData(epi_v, _faces(tris)).clean().triangulate()
    if surf.n_points > DECIM_PTS:
        surf = surf.decimate(1.0 - DECIM_PTS / surf.n_points).clean().triangulate()
    surf = surf.compute_normals(point_normals=True, cell_normals=False,
                                auto_orient_normals=True, consistent_normals=True)
    epi_v = np.asarray(surf.points, float)
    epi_f = surf.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    n = np.asarray(surf.point_data["Normals"], float)
    _epi_out_um = float(os.environ.get("EPI_OUTWARD_UM", "0"))
    if _epi_out_um > 0:
        # OUTWARD collar: endo = full nerve surface (the fascicle, perineurium at
        # its boundary); epi = nerve offset OUTWARD. Avoids the inward-offset
        # self-intersection that collapses the epi region on the thin curved
        # rabbit; the perineurium contact-impedance sheet sits at endo↔epi = the
        # nerve surface (the monofascicular fascicle boundary).
        endo_v0 = epi_v.copy(); endo_f0 = epi_f.copy()
        epi_v0 = epi_v + (_epi_out_um * 1e-6) * n
        epi_v, epi_f = _repair(epi_v0, epi_f)
        endo_v, endo_f = _repair(endo_v0, endo_f0)
        print(f"[geom] OUTWARD collar: endo {len(endo_v)} pts (full nerve) / "
              f"epi {len(epi_v)} pts (+{_epi_out_um:g}µm), pymeshfix", flush=True)
    else:
        endo_v0 = epi_v - (ENDO_OFFSET_UM * 1e-6) * n          # shrink inward → fascicle
        endo_f0 = epi_f.copy()                                  # same topology before repair
        epi_v, epi_f = _repair(epi_v, epi_f)                    # clean both watertight (no SI)
        endo_v, endo_f = _repair(endo_v0, endo_f0)
        print(f"[geom] epi {len(epi_v)} pts / endo {len(endo_v)} pts (offset {ENDO_OFFSET_UM:g}µm, pymeshfix)",
              flush=True)
    return epi_v, epi_f, endo_v, endo_f


SINGLE_REGION = os.environ.get("SINGLE_REGION", "1") == "1"   # one nerve region (no endo/epi split)


def build_bundle_geom_rabbit():
    epi_v, epi_f, endo_v, endo_f = _epi_endo()
    if SINGLE_REGION:
        fasc = []                                            # no fascicle → nerve = one conductor
        nendo = 0
    else:
        fasc = [dict(verts_m=endo_v, faces=endo_f, stl_path="rabbit_endo")]
        nendo = len(endo_v)
    nerve = dict(pts_raw=epi_v, tets_raw=None, boundary_raw=epi_f,
                 source_file=str(SURF), kind="uct_bundle",
                 bundle=dict(epi=dict(verts_m=epi_v, faces=epi_f, stl_path="rabbit_epi"),
                             fascicles=fasc, voxel_xy_mm=0.01, voxel_z_mm=0.01,
                             manifest={}, bundle_id="rabbit"))
    return nerve, (len(epi_v), nendo)


def attach_geom_rabbit(s):
    from golgi.app import global_pca, _surface_quality
    nerve, (nepi, nendo) = build_bundle_geom_rabbit()
    centroid, R_global = global_pca(nerve["pts_raw"])
    try:
        q, _ = _surface_quality(nerve["pts_raw"], nerve["boundary_raw"])
    except Exception:
        q = None
    g = s._geom
    g.nerve = nerve; g.centroid = centroid; g.R_global = R_global
    g.nerve_q = q; g.nerve_poly = None; g._fit_locked = False
    g._R_local_cached = None; g._R_ci_cached = None
    print(f"[geom] rabbit bundle: epi {nepi} + endo {nendo} verts; centroid {centroid.round(4)}",
          flush=True)
    return centroid


def _design():
    return dict(eid=EID, name=f"ring-array {NR}x{NC} @ {OFFSET:g}mm",
                electrode_type="ring-array (NxM)",
                cuff_offset_mm=OFFSET, L_cuff_mm=float(os.environ.get("L_CUFF", "3.0")),
                array_n_rows=NR, array_n_cols=NC,
                array_row_sep_mm=float(os.environ.get("ARR_ROW_SEP", "0.6")),
                array_contact_w_mm=float(os.environ.get("ARR_CONTACT_W", "0.4")),
                array_contact_phi_deg=float(os.environ.get("ARR_CONTACT_PHI", "30.0")),
                cuff_clearance_mm=float(os.environ.get("CUFF_CLEAR", "0.15")), cuff_wall_mm=0.5)


def _load_fibers_and_branch():
    fz = np.load(RO / "nerve_paths_fibers.npz", allow_pickle=True)
    flat = np.asarray(fz["paths_flat"], float); lens = np.asarray(fz["path_lengths"], np.int64)
    out, off = [], 0
    for L in lens:
        out.append(flat[off:off + int(L)]); off += int(L)
    caps = json.loads((RO / "nerve_paths_caps.json").read_text())
    cap_c = np.asarray(caps["branch_cap_centroids_m"], float)
    cap_a = np.asarray(caps["branch_cap_areas_m2"], float)
    trunk_c = np.asarray(caps["trunk_cap_centroid_m"], float)
    ax = 2
    trunk_hi = trunk_c[ax] > np.mean([p[:, ax].mean() for p in out])
    scb_cap = int(np.argmin(cap_a))                            # smaller cap = SCB
    bidx = np.empty(len(out), np.int64)
    for i, p in enumerate(out):
        bend = p[np.argmin(p[:, ax])] if trunk_hi else p[np.argmax(p[:, ax])]
        bidx[i] = 1 if int(np.argmin(np.linalg.norm(cap_c - bend, axis=1))) == scb_cap else 0
    return out, lens, bidx, cap_a


def main():
    import golgi
    from golgi.scene.cuff_fit import refit_design_geometry
    from golgi.conductivity.perineurium import perineurium_thickness_um, fascicle_diameter_um
    if PROJ.exists():
        shutil.rmtree(PROJ)
    s = golgi.Study.create(PROJ)
    attach_geom_rabbit(s)
    if SINGLE_REGION:
        s.set_mesh(use_epi=False, decim_target_k=int(os.environ.get("DECIM_K", "20")))
        print("[mesh] SINGLE-REGION nerve (no endo/epi fascicle, no perineurium CI)", flush=True)
    else:
        try:
            dfasc_um = fascicle_diameter_um(area_um2=math.pi * (380.0 ** 2))
            peri_thk_m = perineurium_thickness_um("pig", dfasc_um) * 1e-6
        except Exception:
            peri_thk_m = 3.0e-6
        s.set_mesh(use_epi=True, decim_target_k=int(os.environ.get("DECIM_K", "20")),
                   perineurium_ci=True, peri_thk_m=peri_thk_m, perineurium_species="pig")
    # CRITICAL for the tiny 13.5mm rabbit: shrink the muscle far-field block (default 80/20mm
    # gives a ±89mm block → huge scale disparity vs the 0.4mm nerve → TetGen recovery fails).
    s._state.muscle_axial_pad_mm = float(os.environ.get("MUS_AX_PAD", "3.0"))
    s._state.muscle_radial_pad_mm = float(os.environ.get("MUS_RAD_PAD", "2.0"))
    # CRITICAL scale fix (the constant the rabbit never patched): the DEFAULT mesh edge
    # lengths (lc_saline=300, lc_epi=250, lc_muscle=3000 µm) are COARSER than the rabbit's
    # thin shells (saline gap 150 µm, epi shell = ENDO_OFFSET 150 µm, muscle pad 2 mm) → TetGen
    # cannot fit a tet across them, so epi (5) + muscle (4) collapse and saline floods their
    # volume (the 865k-tet balloon). Scale every lc to a fraction of its shell so each region
    # keeps ≥3 tets radially. (The human used the same override loop in new_human_mesh, but its
    # mm-scale shells already fit the defaults so it never showed up there.)
    _lc = dict(lc_endo_um=60.0, lc_epi_um=40.0, lc_saline_um=50.0,
               lc_silicone_um=120.0, lc_muscle_um=500.0, lc_contact_um=40.0)
    for _k, _v in _lc.items():
        setattr(s._state, _k, float(os.environ.get(_k.upper(), _v)))
    print("[mesh] rabbit-scale lc (µm): "
          + ", ".join(f"{k}={getattr(s._state, k):g}" for k in _lc), flush=True)
    s.set_electrodes([_design()])
    ok = refit_design_geometry(EID, geom=s._geom, state=s._state)
    print(f"[mesh] cuff {NR}x{NC} @ {OFFSET}mm + refit -> {ok}; FULL-LENGTH bundle meshing ...",
          flush=True)
    msh = s.run_mesh()
    print(f"[mesh] -> {msh if isinstance(msh, (int, str)) else type(msh).__name__}", flush=True)

    # region-tag census: confirm epi(5) + muscle(4) survive (they need lc < shell thickness)
    try:
        import gmsh
        from collections import Counter
        gmsh.initialize(); gmsh.open(str(PROJ / "designs" / EID / "nerve.msh"))
        reg = Counter()
        for dim, t in gmsh.model.getPhysicalGroups(3):
            nm = (gmsh.model.getPhysicalName(dim, t) or str(t)).split("_")[0]
            for e in gmsh.model.getEntitiesForPhysicalGroup(dim, t):
                for typ, tg in zip(*gmsh.model.mesh.getElements(dim, e)[:2]):
                    if typ == 4:
                        reg[nm] += len(tg)
        gmsh.finalize()
        miss = {"endo", "epi", "saline", "silicone", "muscle"} - set(reg)
        print(f"[REGIONS] tets {sum(reg.values()):,}: {dict(reg)} | "
              f"{'ALL 5 PRESENT' if not miss else f'MISSING {miss}'}", flush=True)
    except Exception as _e:
        print(f"[REGIONS] census skipped: {_e}", flush=True)

    fibers, lens, bidx, cap_a = _load_fibers_and_branch()
    s._geom.fiber_paths_raw = fibers
    s._geom.msh_path = str(PROJ / "designs" / EID / "nerve.msh")
    s._state.has_mesh = True
    s._state.emit_impedance = False
    design = next(d for d in s._state.designs if d.get("eid") == EID)
    if not getattr(s._geom, "R_ci", None):
        s._geom._R_ci_cached = float(design.get("R_ci_m"))
        s._geom._R_co_cached = float(design.get("R_co_m"))
    nc = NR * NC
    montages = [{"mid": f"rec{i}", "label": f"rec{i}", "kind": "bipolar",
                 "plus_contact": 2 * i, "minus_contact": 2 * i + 1} for i in range(nc // 2)]
    cfgs = list(s._state.configs)
    for c in cfgs:
        if c.get("design_id") == EID:
            c["recording_montages"] = montages
    s._state.configs = cfgs
    print(f"[fem] {len(fibers)} fibers (SCB[1]={int((bidx==1).sum())} trunk[0]={int((bidx==0).sum())}), "
          f"{nc} contacts; reciprocity FEM ...", flush=True)
    res = s.run_fem()
    print(f"[fem] run_fem -> {res}", flush=True)

    RECDIR = PROJ / "designs" / EID / "recording"
    cols, flat, plens = {}, None, None
    for f in sorted(RECDIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if "Ve_flat" not in d.files:
            continue
        cols[int(d["contact_id"])] = np.asarray(d["Ve_flat"], float)
        flat = np.asarray(d["paths_flat"], float); plens = np.asarray(d["path_lengths"], np.int64)
    if not cols:
        raise SystemExit(f"no recording npz in {RECDIR}")
    cids = sorted(cols)
    Ve_mat = -np.column_stack([cols[c] for c in cids])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "paths_Ve.npz", paths_flat=flat, path_lengths=plens,
                        branch_idx=bidx, Ve_mat=Ve_mat, contact_ids=np.asarray(cids, np.int64),
                        units="V_per_A", inject_A=1.0)
    print(f"[done] {len(cids)} contacts; Ve_mat {Ve_mat.shape}; NaN {np.mean(~np.isfinite(Ve_mat)):.3f}; "
          f"range [{np.nanmin(Ve_mat):.3g},{np.nanmax(Ve_mat):.3g}] -> {OUT/'paths_Ve.npz'}", flush=True)


if __name__ == "__main__":
    main()
