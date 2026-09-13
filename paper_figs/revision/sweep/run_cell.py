"""One cell of the R2.2 cuff-fit sensitivity study, on the PROPER geometry.

Builds the real human cervical-vagus cross-section (epineurium outline + 5
fascicles, from the reproduction bundle) into an extruded multi-domain model
using golgi's own `extrude_single_slice`, optionally applying golgi's own
area-preserving round (the Duke/ASCENT deform mode), then runs golgi's
headless pipeline: mesh -> fibers -> FEM -> threshold sweep.

  python r22_sweep/run_cell.py --deform none --clearance 0.15 --tag u015
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = "/Users/admin/Desktop/DATA/Uni/2026/Projects/golgi_revision"
XSEC = ("comsol_data/comsol_handover/models/M4_human_sub-47_sam-2/"
        "nerve_xsec.json")
THICKNESS_MM = 20.0      # SPEC: nerve extrusion length, z in [-10, +10] mm
L_CUFF_MM = 5.4          # SPEC
PITCH_MM = 0.005         # 5 um/px rasterisation


def _load_slice(deform: str):
    """Real outline + fascicle polygons in mm; optionally area-preserving round."""
    sys.path.insert(0, REPO)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "golgi_deform", f"{REPO}/golgi/segmentation/deform.py")
    gd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gd)

    j = json.loads(Path(XSEC).read_text())
    outline = np.asarray(j["nerve_outline_xy_um"], float) / 1e3
    fascs = [np.asarray(f["polygon_xy_um"], float) / 1e3 for f in j["fascicles"]]
    det = 1.0
    if deform == "round":
        M, c = gd.nerve_round_affine(outline)
        det = float(np.linalg.det(M))

        def _r(p):
            return gd.apply_round_xy(
                np.column_stack([p, np.zeros(len(p))]), M, c)[:, :2]
        outline = _r(outline)
        fascs = [_r(f) for f in fascs]
    return outline, fascs, det


def _rasterise(outline, fascs, pitch_mm):
    from skimage.draw import polygon as sk_polygon
    margin = 0.10
    lo = outline.min(0) - margin
    hi = outline.max(0) + margin
    n = np.ceil((hi - lo) / pitch_mm).astype(int) + 1
    epi = np.zeros((int(n[1]), int(n[0])), bool)
    fas = np.zeros_like(epi)

    def fill(mask, poly):
        c = (poly[:, 0] - lo[0]) / pitch_mm
        r = (poly[:, 1] - lo[1]) / pitch_mm
        rr, cc = sk_polygon(r, c, shape=mask.shape)
        mask[rr, cc] = True

    fill(epi, outline)
    for f in fascs:
        fill(fas, f)
    fas &= epi                      # fascicles live inside the nerve
    return epi, fas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deform", choices=("none", "round"), required=True)
    ap.add_argument("--clearance", type=float, required=True)   # mm
    ap.add_argument("--fibers", type=int, default=24)
    ap.add_argument("--min-rel-size-pct", type=float, default=0.3)
    ap.add_argument("--lc-scale", type=float, default=2.0)
    ap.add_argument("--lc-muscle", type=float, default=3000.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="r22_sweep/results")
    args = ap.parse_args(argv)

    sys.path.insert(0, REPO)
    import neuron
    mech = Path("mech/MOD").resolve()
    if mech.exists():
        neuron.load_mechanisms(str(mech))

    import trimesh
    import golgi
    from golgi.segmentation.reconstruct3d import extrude_single_slice
    from golgi.jobs.schemas import SweepRequest

    t = {}
    outline, fascs, det = _load_slice(args.deform)
    epi_mask, fasc_mask = _rasterise(outline, fascs, PITCH_MM)
    print(f"[geom] deform={args.deform} det(M)={det:.6f} "
          f"mask={epi_mask.shape} fascicles={len(fascs)}", flush=True)

    t0 = time.time()
    meshes = extrude_single_slice(
        epi_mask, fasc_mask,
        voxel_xy_mm=PITCH_MM, thickness_mm=THICKNESS_MM,
    )
    t["extrude_s"] = time.time() - t0
    print(f"[geom] extruded {len(meshes)} meshes in {t['extrude_s']:.0f}s "
          f"({[m.name for m in meshes]})", flush=True)
    epi_mesh, fasc_meshes = meshes[0], meshes[1:]

    work = Path("r22_sweep/work") / args.tag
    work.mkdir(parents=True, exist_ok=True)
    stl = work / "epi.stl"
    trimesh.Trimesh(vertices=np.asarray(epi_mesh.verts, float),
                    faces=np.asarray(epi_mesh.faces, np.int64)).export(stl)

    project = Path("r22_sweep/projects") / args.tag
    if project.exists():
        shutil.rmtree(project)
    s = golgi.Study.create(project)
    info = s.import_nerve(stl.resolve(), scale_factor=1.0e-3)
    print(f"[study] imported epi: {info['n_pts']} pts {info['n_tris']} tris",
          flush=True)

    # inject the multi-domain bundle (same structure golgi's import action
    # builds): epi surface + fascicle surfaces, vertices in metres
    s._geom.nerve["bundle"] = {
        "epi": {"verts_m": np.asarray(epi_mesh.verts, float) * 1e-3,
                "faces": np.asarray(epi_mesh.faces, np.int64)},
        "fascicles": [
            {"verts_m": np.asarray(m.verts, float) * 1e-3,
             "faces": np.asarray(m.faces, np.int64)}
            for m in fasc_meshes
        ],
    }
    print(f"[study] bundle injected: 1 epi + {len(fasc_meshes)} fascicles",
          flush=True)

    k = float(args.lc_scale)
    s.set_mesh(
        use_epi=True, epi_thickness_um=50,
        lc_endo_um=200 * k, lc_epi_um=150 * k, lc_muscle_um=args.lc_muscle,
        lc_saline_um=150 * k, lc_silicone_um=300 * k, lc_contact_um=100 * k,
        lc_scar_um=150 * k,
        muscle_radial_pad_mm=8.0, muscle_axial_pad_mm=10.0,
    )
    s.set_electrodes([{
        "eid": "elec_01", "name": "12-contact cuff",
        "electrode_type": "ring-array (NxM)",
        "cuff_offset_mm": THICKNESS_MM / 2.0,
        "L_cuff_mm": L_CUFF_MM, "array_n_rows": 3, "array_n_cols": 4,
        "cuff_clearance_mm": float(args.clearance),
    }])

    t0 = time.time(); s.run_mesh(); t["mesh_s"] = time.time() - t0
    print(f"[study] meshed in {t['mesh_s']:.0f}s", flush=True)
    t0 = time.time()
    s.set_fiber_seed(n_fibers=int(args.fibers), fiber_auto_detect_branches=False,
                     fiber_min_rel_size_pct=float(args.min_rel_size_pct))
    s.run_fibers(); t["fibers_s"] = time.time() - t0
    t0 = time.time(); s.run_fem(); t["fem_s"] = time.time() - t0
    print(f"[study] fibers {t['fibers_s']:.0f}s, FEM {t['fem_s']:.0f}s",
          flush=True)

    t0 = time.time()
    res = s.run_sweep(SweepRequest(
        mode="threshold", bisect_lo_mA=0.01, bisect_hi_mA=10.0,
        bisect_tol_uA=10.0, model_source="single_fiber",
    ))
    t["sweep_s"] = time.time() - t0

    thr = [float(v) for v in np.asarray(res.thresholds_uA).ravel()]
    fasc_id = None
    for attr in ("fascicle_idx", "branch_idx", "fiber_fascicle"):
        v = getattr(res, attr, None)
        if v is not None:
            fasc_id = [int(x) for x in np.asarray(v).ravel()]
            break
    fiber_xy = []
    for p in sorted(project.rglob("paths_Ve.npz")):
        z = np.load(p, allow_pickle=True)
        if "paths_flat" in z and "path_lengths" in z:
            flat = np.asarray(z["paths_flat"], float)
            lens = np.asarray(z["path_lengths"], int)
            off = 0
            for L in lens:
                seg = flat[off:off + L]; off += L
                fiber_xy.append((seg[np.argmin(np.abs(seg[:, 2])), :2] * 1e3).tolist())
            break

    R_bore_mm = None
    for p in project.rglob("mesh_config.json"):
        try:
            R_bore_mm = float(json.loads(p.read_text())["R_cuff_inner"]) * 1e3
            break
        except Exception:
            pass

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rec = dict(tag=args.tag, deform=args.deform, det_M=det,
               clearance_mm=args.clearance, R_bore_mm=R_bore_mm,
               n_fascicles=len(fasc_meshes), n_fibers=int(args.fibers),
               thresholds_uA=thr, fascicle_id=fasc_id, fiber_xy_mm=fiber_xy, timings_s=t,
               lc_scale=args.lc_scale, lc_muscle_um=args.lc_muscle)
    (outdir / f"{args.tag}.json").write_text(json.dumps(rec, indent=1))
    act = sorted(v for v in thr if v > 0)
    med = act[len(act) // 2] if act else float("nan")
    print(f"[{args.tag}] deform={args.deform} clearance={args.clearance} "
          f"R_bore={R_bore_mm} activated={len(act)}/{len(thr)} "
          f"median={med:.1f} uA | mesh={t['mesh_s']:.0f}s fem={t['fem_s']:.0f}s "
          f"sweep={t['sweep_s']:.0f}s", flush=True)
    s.close()
    return 0 if act else 1


if __name__ == "__main__":
    sys.exit(main())
