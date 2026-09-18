"""Probe just the meshing step so mesh settings can be tuned fast.

  python r22_sweep/mesh_probe.py --stl r22_sweep/nerve_undeformed.stl \
      --clearance 0.15 --lc-scale 2.0 --pad-radial 3 --pad-axial 5 --tag p1
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO = "/Users/admin/Desktop/DATA/Uni/2026/Projects/golgi_revision"
NERVE_LENGTH_MM = 20.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--clearance", type=float, default=0.15)
    ap.add_argument("--lc-scale", type=float, default=1.0)
    ap.add_argument("--lc-muscle", type=float, default=1500.0)  # SPEC: ~1.5 mm bulk
    ap.add_argument("--pad-radial", type=float, default=8.0)    # SPEC: 8 mm
    ap.add_argument("--pad-axial", type=float, default=10.0)    # SPEC: 10 mm
    ap.add_argument("--length", type=float, default=NERVE_LENGTH_MM)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, REPO)
    import golgi

    project = Path("r22_sweep/probe") / args.tag
    if project.exists():
        shutil.rmtree(project)
    s = golgi.Study.create(project)
    info = s.import_nerve(Path(args.stl).resolve(), scale_factor=1.0e-3)
    print(f"imported: {info['n_pts']} pts {info['n_tris']} tris bbox={info['bbox_mm']}",
          flush=True)

    k = float(args.lc_scale)
    s.set_mesh(
        use_epi=True, epi_thickness_um=50,
        lc_endo_um=200 * k, lc_epi_um=150 * k, lc_muscle_um=args.lc_muscle,
        lc_saline_um=150 * k, lc_silicone_um=300 * k, lc_contact_um=100 * k,
        lc_scar_um=150 * k,
        muscle_radial_pad_mm=args.pad_radial,
        muscle_axial_pad_mm=args.pad_axial,
    )
    s.set_electrodes([{
        "eid": "elec_01", "name": "12-contact cuff",
        "electrode_type": "ring-array (NxM)",
        "cuff_offset_mm": args.length / 2.0,
        "L_cuff_mm": 5.4, "array_n_rows": 3, "array_n_cols": 4,
        "cuff_clearance_mm": float(args.clearance),
    }])
    print("meshing ...", flush=True)
    t0 = time.time()
    out = s.run_mesh()
    dt = time.time() - t0
    print(f"MESH OK in {dt:.0f}s -> {out}", flush=True)
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
