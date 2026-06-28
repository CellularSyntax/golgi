# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Multi-region human nerve mesh from REAL surfaces: epineurium (outer) + endoneurium
(inner fascicle), the proper anatomy. Builds golgi's "uct_bundle" geom directly from the
two repaired STLs (epi = outer hull / nerve, endo = one inner fascicle) — no synthetic
epi offset, which is exactly the construction golgi uses to avoid the TetGen self-
intersection choke. Cuff = ring-array on the straight cranial trunk (z=28), refit to the
EPI. Same recipe as the GUI, plus the real epi region.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import shutil
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT))
PROJ = ROOT / "paper_figs/out/_intermediate/human_bundle_project"
EPI = ROOT / "data/EPINERIUM_repaired.stl"
# smoothed+decimated single-lobe fascicular envelope for the position SWEEP (meshes at any
# cuff offset; the real multi-lobe tree self-intersects with the cuff away from z28). fig8
# panel-d still draws the real multi-lobe fascicles (sliced directly from the real STL).
ENDO = ROOT / ("data/ENDO_envelope_smooth.stl" if os.environ.get("USE_ENVELOPE") == "1"
               else "data/ENDONERIUM_repaired.stl")
EID = "elec_01"


def design_dict():
    # 2x4 ring array: the epi+endo bundle meshes cleanly with 2 axial rows (3 rows self-
    # intersect the multi-region assembly past TetGen recovery). Circumferential guarded
    # tripole over the SCB cluster, like the rabbit. Perineurium CI added at solve time.
    off = float(os.environ.get("CUFF_OFFSET", "28"))
    nr = int(os.environ.get("ARR_ROWS", "2"))
    nc = int(os.environ.get("ARR_COLS", "4"))
    return dict(
        eid=EID, name=f"ring-array {nr}x{nc} @ trunk z{off:g}",
        electrode_type="ring-array (NxM)",
        cuff_offset_mm=off, L_cuff_mm=float(os.environ.get("L_CUFF", "5")),
        array_n_rows=nr, array_n_cols=nc,
        # row-spacing / contact size env-overridable for the contact-density
        # study (denser arrays need smaller spacing to fit the cuff window).
        array_row_sep_mm=float(os.environ.get("ARR_ROW_SEP", "3.0")),
        array_contact_w_mm=float(os.environ.get("ARR_CONTACT_W", "0.6")),
        array_contact_phi_deg=float(os.environ.get("ARR_CONTACT_PHI", "60.0")),
        cuff_clearance_mm=float(os.environ.get("CUFF_CLEAR", "0.2")), cuff_wall_mm=1.0)


def _vf(path):
    m = pv.read(str(path)).extract_surface().triangulate()
    v = np.asarray(m.points, dtype=np.float64)
    f = m.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    return v, f


def build_bundle_geom():
    """golgi 'uct_bundle' nerve dict: epi outer hull + endo inner fascicle (verts in m).
    USE_SINGLE_REGION=1 omits the fascicle -> homogeneous nerve. The multi-region fascicle
    cap planes are the irreducible self-intersection source toward the branch; dropping them
    lets the cuff mesh at any axial position for the placement sweep (Fig 8 keeps the full
    epi+endo model at the operating point)."""
    epi_v, epi_f = _vf(EPI); endo_v, endo_f = _vf(ENDO)
    epi_v_m, endo_v_m = epi_v * 1e-3, endo_v * 1e-3
    # USE_ENDO_AS_NERVE: model the nerve as the SMOOTH endoneurium envelope
    # only (drop the raw epineurium hull). The raw epi cross-section is
    # non-convex with tiny clip-ring edges that break the cuff-window cap
    # triangulation off-trunk; the smoothed endo envelope clips to a clean,
    # near-convex ring that meshes at any cuff position.
    if os.environ.get("USE_ENDO_AS_NERVE") == "1":
        epi_v_m, epi_f = endo_v_m, endo_f
        EPI_SRC = ENDO
    else:
        EPI_SRC = EPI
    single = (os.environ.get("USE_SINGLE_REGION") == "1"
              or os.environ.get("USE_ENDO_AS_NERVE") == "1")
    fasc = [] if single else [dict(verts_m=endo_v_m, faces=endo_f, stl_path=ENDO)]
    nerve = dict(
        pts_raw=epi_v_m, tets_raw=None, boundary_raw=epi_f,        # OUTER hull = epi only
        source_file=str(EPI_SRC), kind="uct_bundle",
        bundle=dict(
            epi=dict(verts_m=epi_v_m, faces=epi_f, stl_path=EPI_SRC),
            fascicles=fasc,
            voxel_xy_mm=0.01, voxel_z_mm=0.01, manifest={}, bundle_id="human"),
    )
    return nerve, (len(epi_v_m), 0 if single else len(endo_v_m))


def attach_geom(s):
    from golgi.app import global_pca, _surface_quality
    nerve, (nepi, nendo) = build_bundle_geom()
    centroid, R_global = global_pca(nerve["pts_raw"])
    try:
        q, _ = _surface_quality(nerve["pts_raw"], nerve["boundary_raw"])
    except Exception:
        q = None
    g = s._geom
    g.nerve = nerve; g.centroid = centroid; g.R_global = R_global
    g.nerve_q = q; g.nerve_poly = None; g._fit_locked = False
    g._R_local_cached = None; g._R_ci_cached = None
    print(f"bundle geom: epi {nepi} verts + endo {nendo} verts; centroid {centroid.round(4)}")
    return centroid


def main():
    import golgi
    from golgi.scene.cuff_fit import refit_design_geometry
    if PROJ.exists():
        shutil.rmtree(PROJ)
    s = golgi.Study.create(PROJ)
    attach_geom(s)
    # perineurium contact-impedance at the endo<->epi interface, set at MESH time so it is
    # persisted in mesh_config.json (the solver reads PERI_CI from the persisted config).
    # It is conductivity metadata (Rs = thk/sigma_peri), not geometry -> the PLC / TetGen
    # recovery are unaffected, so the 2x4 bundle still meshes.
    from golgi.conductivity.perineurium import perineurium_thickness_um, fascicle_diameter_um
    import math
    dfasc_um = fascicle_diameter_um(area_um2=math.pi * (950.0 ** 2))   # endo ~0.95 mm radius
    peri_thk_m = perineurium_thickness_um("human", dfasc_um) * 1e-6
    s.set_mesh(use_epi=(os.environ.get("USE_EPI", "1") == "1"),
               decim_target_k=int(os.environ.get("DECIM_K", "15")),
               perineurium_ci=True, peri_thk_m=peri_thk_m, perineurium_species="human")
    # per-region mesh sizes (µm) — env override. Off-trunk, the flat cuff-window cap planes
    # cut the curved nerve obliquely into thin cap facets in the endo/epi; refining those fine
    # regions makes TetGen non-terminate. Coarsening the nerve (endo/epi) while keeping the
    # silicone wall fine sidesteps the choke yet still resolves the contacts.
    for _k in ("lc_endo_um", "lc_epi_um", "lc_silicone_um", "lc_saline_um", "lc_muscle_um"):
        _ev = os.environ.get(_k.upper())
        if _ev:
            setattr(s._state, _k, float(_ev))
            print(f"  {_k} = {_ev}")
    print(f"perineurium CI: thk {peri_thk_m*1e6:.1f}um Rs {peri_thk_m*1149:.4f} ohm.m2")
    s.set_electrodes([design_dict()])
    ok = refit_design_geometry(EID, geom=s._geom, state=s._state)
    d2 = next(d for d in s._state.designs if d.get("eid") == EID)
    print(f"refit -> {ok}; R_ci {float(d2.get('R_ci_m'))*1e3:.3f} mm offset {d2.get('cuff_offset_mm')} "
          f"L {d2.get('L_cuff_mm')} array {d2.get('array_n_rows')}x{d2.get('array_n_cols')}")
    s.run_mesh()
    nmsh = PROJ / "designs" / EID / "nerve.msh"
    print(f"[MESH OK] {nmsh.stat().st_size/1e6:.0f} MB" if nmsh.exists() else "[NO MESH] TetGen failed")


if __name__ == "__main__":
    main()
