# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Validation V5 (native FEM) — NRV Fig 9 LIFE benchmark with golgi's real
intrafascicular FEM. Synthetic monofascicular cat-tibial-like nerve + a single
LIFE (25 um wire, 1 mm active site) inside the fascicle; golgi's native reciprocity
FEM gives the LIFE lead field along straight MRG fibers, then NEURON thresholds
give recruitment vs current (50/20 us) and vs pulse duration (7/8/9 uA), compared
to Nannini & Horch 1991 and Yoshida & Horch 1993.

Stages:  --mesh (geometry + mesh)   --fem (mesh + FEM + lead field)
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, shutil, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
PROJ = ROOT / "paper_figs/out/_intermediate/nrv_life"
EID = "life"
FASC_R_M = 0.275e-3          # 550 um fascicle
NERVE_R_M = 0.325e-3         # + 50 um epi shell
NERVE_L_M = 30.0e-3
L_CUFF_MM = 5.0              # LIFE meshing-region length (caps at +/- 2.5 mm, well inside)
N_FIBERS = 160


def make_nerve():
    import pyvista as pv
    PROJ.mkdir(parents=True, exist_ok=True)
    # finely-tessellated tube (rings along z) so the cuff-cap clip extracts clean
    # loops (pv.Cylinder has a single ring of full-length side facets => degenerate)
    line = pv.Line((0, 0, 0), (0, 0, NERVE_L_M), resolution=80)
    cyl = line.tube(radius=NERVE_R_M, n_sides=48, capping=True).triangulate()
    # A perfectly axisymmetric cylinder is DEGENERATE for the multi-domain
    # mesher: every ring of side vertices is exactly coplanar and exactly
    # coaxial with the (concentric) saline/silicone/muscle shells, so the
    # cuff-window clip + annular caps produce coincident, zero-area facets
    # that stall gmsh OCC and make TetGen's PLC self-intersect. A tiny
    # deterministic radial wobble (~3 % ≈ ±10 µm) breaks the symmetry — this
    # is exactly why real, irregular nerves mesh cleanly, and it is
    # physically negligible for the intrafascicular LIFE lead field. Seeded
    # (no Date/random) → byte-reproducible. Disable with NRV_PERTURB=0.
    if os.environ.get("NRV_PERTURB", "1") == "1":
        P = np.asarray(cyl.points, dtype=float).copy()
        r = np.hypot(P[:, 0], P[:, 1])
        side = r > 1e-9                       # leave the axial cap centres
        th = np.arctan2(P[side, 1], P[side, 0])
        zz = P[side, 2] / NERVE_L_M
        scale = (1.0 + 0.03 * np.sin(3.0 * th + 17.0 * zz)
                 + 0.015 * np.cos(5.0 * th - 11.0 * zz))
        rr = r[side] * scale
        P[side, 0] = rr * np.cos(th)
        P[side, 1] = rr * np.sin(th)
        cyl.points = P
    stl = PROJ / "nerve.stl"; cyl.save(str(stl))
    return stl


def straight_fibers(seed=0):
    """N straight myelinated fibers, random (x,y) uniform-area in the fascicle,
    running the nerve length; raw frame (same as the imported cylinder)."""
    rng = np.random.default_rng(seed)
    r = FASC_R_M * np.sqrt(rng.uniform(0, 1, N_FIBERS))
    th = rng.uniform(0, 2 * np.pi, N_FIBERS)
    xy = np.column_stack([r * np.cos(th), r * np.sin(th)])
    z = np.linspace(0.3e-3, NERVE_L_M - 0.3e-3, 1201)   # fine sampling to resolve the active-site edges
    fibers = [np.column_stack([np.full_like(z, x), np.full_like(z, y), z]) for x, y in xy]
    diam = np.clip(rng.normal(10.0, 3.2, N_FIBERS), 2.0, 16.0)
    return fibers, xy, diam


def recon():
    import golgi
    if (PROJ / "study").exists():
        shutil.rmtree(PROJ / "study")
    stl = make_nerve()
    s = golgi.Study.create(PROJ / "study")
    info = s.import_nerve(stl, scale_factor=1.0)
    print(f"[recon] import: bbox_mm={tuple(round(x,2) for x in info['bbox_mm'])} "
          f"watertight={info['watertight']}", flush=True)
    # FINE near-field: a 25 µm intrafascicular wire sets a sharp, edge-dominated
    # longitudinal activating function (-> rheobase). Refine the contact + endoneurium
    # so the active-site edges are resolved (coarse mesh over-smooths -> rheobase too high).
    # use_epi env-overridable: the inward-offset epi shell on this thin
    # synthetic cylinder self-intersects and stalls the conformal mesher,
    # and a monofascicular LIFE nerve carries no epineurium anyway — set
    # NRV_USE_EPI=0 for a clean single-region PLC (intrafascicular field
    # near the wire is unchanged).
    _use_epi = os.environ.get("NRV_USE_EPI", "1") == "1"
    _lc_contact = float(os.environ.get("NRV_LC_CONTACT", "6.0"))
    _lc_endo = float(os.environ.get("NRV_LC_ENDO", "30.0"))
    _mesh_kw = dict(use_epi=_use_epi, epi_thickness_um=50, decim_target_k=80,
                    lc_contact_um=_lc_contact, lc_endo_um=_lc_endo,
                    lc_epi_um=50.0)
    # NRV_MUSCLE_PAD shrinks the surrounding-muscle radial pad. The default
    # ~20 mm bulk gives a 30:1 nerve:muscle scale disparity that the current
    # multi-domain mesher cannot tetrahedralize (PLC self-intersections /
    # runaway Steiner insertion). A small pad bounds the mesh and is
    # physically fine for an intrafascicular LIFE (near-wire-field-dominated).
    if os.environ.get("NRV_MUSCLE_PAD"):
        _mesh_kw["muscle_radial_pad_mm"] = float(os.environ["NRV_MUSCLE_PAD"])
    s.set_mesh(**_mesh_kw)
    # A LIFE is intrafascicular — there is no cuff. Mesh the nerve directly
    # in a single homogeneous bath (endo + saline), no concentric cuff
    # shells. This is the physically-correct domain AND avoids the
    # axisymmetric-cylinder degeneracy that stalls the multi-domain cuff
    # mesher. Disable with NRV_BARE_BATH=0 to use the legacy cuff path.
    s._state.mesh_bare_bath = os.environ.get("NRV_BARE_BATH", "1") == "1"
    if s._state.mesh_bare_bath:
        # Tight homogeneous bath — the intrafascicular LIFE field is
        # near-wire-dominated, so a few mm of bath beyond the nerve is
        # ample for the ground BC and keeps the mesh small.
        s._state.muscle_radial_pad_mm = float(os.environ.get("NRV_BATH_R_MM", "5"))
        s._state.muscle_axial_pad_mm = float(os.environ.get("NRV_BATH_Z_MM", "5"))
        # Coarse bath: the field is smooth away from the intrafascicular
        # wire, so the saline bath needs only a coarse tet size — TetGen
        # uses a uniform per-region size (no grading), so a fine bath
        # would explode the mesh for no accuracy gain.
        s._state.lc_saline_um = float(os.environ.get("NRV_BATH_LC", "1000"))
    s._state.cuff_anchor = "trunk (low z)"
    s._state.L_cuff_mm = L_CUFF_MM
    s.set_electrodes([{"eid": EID, "name": "LIFE", "cuff_offset_mm": NERVE_L_M / 2 * 1e3,
                       "L_cuff_mm": L_CUFF_MM, "cuff_anchor": "trunk (low z)",
                       "electrode_type": "LIFE (longitudinal intrafascicular)",
                       "life_n_rows": 2, "life_n_cols": 1, "life_row_sep_mm": 4.0,
                       "life_contact_length_mm": 1.0,
                       "life_diameter_um": 25.0, "life_x_mm": 0.0, "life_y_mm": 0.0,
                       "life_target_fascicle_idx": -1}])
    from golgi.scene.cuff_fit import refit_design_geometry, find_cuff_origin_pca
    ok = refit_design_geometry(EID, geom=s._geom, state=s._state)
    pca = (s._geom.nerve["pts_raw"] - s._geom.centroid) @ s._geom.R_global
    org = find_cuff_origin_pca(pca, s._state.cuff_anchor, NERVE_L_M / 2 * 1e3, 0.0, 0.0)
    print(f"[recon] LIFE refit -> {ok}; pca z-range mm [{pca[:,2].min()*1e3:.1f},{pca[:,2].max()*1e3:.1f}] "
          f"cuff origin pca z = {org[2]*1e3:.2f} mm; L_cuff={L_CUFF_MM} (caps at +/-{L_CUFF_MM/2} mm)", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", action="store_true")
    ap.add_argument("--fem", action="store_true")
    a = ap.parse_args()
    s = recon()
    print("[mesh] building native mesh ...", flush=True)
    msh = s.run_mesh()
    print(f"[mesh] -> {msh if isinstance(msh,(dict,str)) else type(msh).__name__}", flush=True)
    mpath = PROJ / "study" / "designs" / EID / "nerve.msh"
    print(f"[mesh] nerve.msh exists={mpath.is_file()} size={mpath.stat().st_size if mpath.is_file() else 0}", flush=True)
    if a.mesh and not a.fem:
        return
    # fibers + recording montage for the LIFE lead field
    fibers, xy, diam = straight_fibers()
    s._geom.fiber_paths_raw = fibers
    s._geom.msh_path = str(mpath); s._state.has_mesh = True; s._state.emit_impedance = False
    cfgs = list(s._state.configs)
    for c in cfgs:
        if c.get("design_id") == EID:
            c["recording_montages"] = [{"mid": "life", "label": "life", "kind": "monopolar",
                                        "plus_contact": 0, "minus_contact": 0}]
    s._state.configs = cfgs
    np.save(PROJ / "fiber_xy.npy", xy); np.save(PROJ / "fiber_diam.npy", diam)
    print("[fem] running native LIFE reciprocity FEM ...", flush=True)
    s.run_fem()
    rec = PROJ / "study" / "designs" / EID / "recording"
    print(f"[fem] recording files: {sorted(p.name for p in rec.glob('*.npz')) if rec.exists() else 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
