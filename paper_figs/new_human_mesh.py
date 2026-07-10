# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Multi-compartment mesh of the NEW cleaned human nerve surfaces (data/new_human_meshes):
proper epineurium (outer) + endoneurium (inner, real multi-lobe) + one ring-array cuff at a
straight position (global z~18mm = PCA offset ~5.4mm) + muscle block. This is the PROPER
epi+endo multi-region model (NOT the endo-as-nerve workaround) — the test of whether the
new, COMSOL-clean surfaces tetrahedralize as a multi-domain PLC in golgi/TetGen.

Window note: within the 5mm cuff band the endo is a SINGLE connected piece (lobes merge
across the band) so one endo seed suffices; the hard part is the epi end-cap (annulus with
~6 endo-lobe holes) — handled by the gmsh constrained-cap path (GOLGI_PLC_CDT=1).
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import shutil
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
PROJ = ROOT / "paper_figs/out/_intermediate/new_human_project"
D = ROOT / "data/new_human_meshes"
EPI = D / "EPINERIUM_Epinerium_cleaned_aligned_masks_mm_SMOOTHED_LONGER_original_duplicate_duplicate_wrapped_duplicate.stl"
# pre-decimated + pymeshfix-repaired endo (384k->77k faces, watertight, 6 lobes preserved)
ENDO = D / "ENDO_dec80_fixed.stl"
EID = "elec_01"


def _vf(path):
    m = pv.read(str(path)).extract_surface().triangulate()
    v = np.asarray(m.points, dtype=np.float64)
    f = m.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    return v, f


def build_bundle_geom():
    epi_v, epi_f = _vf(EPI); endo_v, endo_f = _vf(ENDO)
    epi_v_m, endo_v_m = epi_v * 1e-3, endo_v * 1e-3
    fasc = [dict(verts_m=endo_v_m, faces=endo_f, stl_path=str(ENDO))]
    nerve = dict(
        pts_raw=epi_v_m, tets_raw=None, boundary_raw=epi_f,
        source_file=str(EPI), kind="uct_bundle",
        bundle=dict(
            epi=dict(verts_m=epi_v_m, faces=epi_f, stl_path=str(EPI)),
            fascicles=fasc,
            voxel_xy_mm=0.01, voxel_z_mm=0.01, manifest={}, bundle_id="new_human"),
    )
    return nerve, (len(epi_v_m), len(endo_v_m))


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


def design_dict():
    off = float(os.environ.get("CUFF_OFFSET", "5.4"))   # PCA mm == global z~18
    nr = int(os.environ.get("ARR_ROWS", "2"))
    nc = int(os.environ.get("ARR_COLS", "4"))
    return dict(
        eid=EID, name=f"ring-array {nr}x{nc} @ z{off:g}",
        electrode_type="ring-array (NxM)",
        cuff_offset_mm=off, L_cuff_mm=float(os.environ.get("L_CUFF", "5")),
        array_n_rows=nr, array_n_cols=nc,
        array_row_sep_mm=float(os.environ.get("ARR_ROW_SEP", "3.0")),
        array_contact_w_mm=float(os.environ.get("ARR_CONTACT_W", "0.6")),
        array_contact_phi_deg=float(os.environ.get("ARR_CONTACT_PHI", "60.0")),
        cuff_clearance_mm=float(os.environ.get("CUFF_CLEAR", "0.2")), cuff_wall_mm=1.0)


def main():
    import golgi
    from golgi.scene.cuff_fit import refit_design_geometry
    if PROJ.exists():
        shutil.rmtree(PROJ)
    s = golgi.Study.create(PROJ)
    attach_geom(s)
    from golgi.conductivity.perineurium import perineurium_thickness_um, fascicle_diameter_um
    import math
    dfasc_um = fascicle_diameter_um(area_um2=math.pi * (600.0 ** 2))   # endo lobe ~0.6mm r
    peri_thk_m = perineurium_thickness_um("human", dfasc_um) * 1e-6
    s.set_mesh(use_epi=True, decim_target_k=int(os.environ.get("DECIM_K", "20")),
               perineurium_ci=True, peri_thk_m=peri_thk_m, perineurium_species="human")
    for _k in ("lc_endo_um", "lc_epi_um", "lc_silicone_um", "lc_saline_um", "lc_muscle_um"):
        _ev = os.environ.get(_k.upper())
        if _ev:
            setattr(s._state, _k, float(_ev)); print(f"  {_k} = {_ev}")
    # muscle far-field block — default pads are huge (axial 80mm / radial 20mm → ±100mm
    # block, ~half the tets). Trim to a tight far-field that still encloses the full-length
    # nerve and keeps the ground BC several mm beyond the cuff.
    s._state.muscle_axial_pad_mm = float(os.environ.get("MUS_AXIAL_PAD_MM", "15"))
    s._state.muscle_radial_pad_mm = float(os.environ.get("MUS_RADIAL_PAD_MM", "12"))
    print(f"  muscle pad: axial {s._state.muscle_axial_pad_mm} / "
          f"radial {s._state.muscle_radial_pad_mm} mm")
    print(f"perineurium CI: thk {peri_thk_m*1e6:.1f}um")
    s.set_electrodes([design_dict()])
    ok = refit_design_geometry(EID, geom=s._geom, state=s._state)
    d2 = next(d for d in s._state.designs if d.get("eid") == EID)
    print(f"refit -> {ok}; R_ci {float(d2.get('R_ci_m'))*1e3:.3f} mm offset {d2.get('cuff_offset_mm')} "
          f"L {d2.get('L_cuff_mm')} array {d2.get('array_n_rows')}x{d2.get('array_n_cols')}")
    s.run_mesh()
    nmsh = PROJ / "designs" / EID / "nerve.msh"
    if nmsh.exists():
        print(f"[MESH OK] {nmsh.stat().st_size/1e6:.0f} MB")
        # region tag census
        import gmsh
        gmsh.initialize(); gmsh.open(str(nmsh))
        from collections import Counter
        pg = gmsh.model.getPhysicalGroups(3); reg = Counter()
        for d, t in pg:
            nm = gmsh.model.getPhysicalName(d, t); n = 0
            for e in gmsh.model.getEntitiesForPhysicalGroup(d, t):
                for typ, tg in zip(*gmsh.model.mesh.getElements(d, e)[:2]):
                    if typ == 4: n += len(tg)
            reg[nm or str(t)] = n
        nt = sum(reg.values())
        gmsh.finalize()
        print(f"[REGIONS] total tets {nt:,}: {dict(reg)}")
        names = set(reg)
        need = {"endo", "epi", "saline", "silicone", "muscle"}
        miss = need - {n.split('_')[0] for n in names}
        print(f"[CHECK] regions present: {sorted({n.split('_')[0] for n in names})}; "
              f"{'ALL 5 PRESENT' if not miss else f'MISSING {miss}'}")
    else:
        print("[NO MESH] TetGen failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
