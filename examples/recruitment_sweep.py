"""End-to-end headless pipeline demo: nerve → mesh → electrodes → FEM →
fibers → threshold sweep → bundle export.

By default this runs on a **synthetic cylindrical nerve** generated on the fly,
so it needs no external data:

    # Synthetic nerve (no data needed):
    python examples/recruitment_sweep.py

    # Your own geometry:
    python examples/recruitment_sweep.py --nerve /path/to/nerve.stl --project /tmp/study

All five compute methods are exercised (import_nerve, run_mesh, run_fem,
run_fibers, run_sweep). Outputs (the project directory and the `*_study.zip`
bundle) are written under the current working directory unless --project is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import golgi

NERVE_LENGTH_MM = 20.0


def make_synthetic_nerve(path: Path, *, radius_mm: float = 1.0,
                         length_mm: float = NERVE_LENGTH_MM,
                         sections: int = 72, ring_dz_mm: float = 1.0) -> Path:
    """Write a watertight round cylindrical endoneurium surface — a simple
    monofascicular synthetic nerve — so the demo needs no external data.

    The tube is built as a stack of cross-section rings every ``ring_dz_mm``
    along z (not just two end caps). Intermediate rings matter: the cuff fitter
    samples nerve vertices *near the cuff plane*, so a bare two-ring cylinder
    (vertices only at the ends) has nothing to fit to. This mirrors how the
    COMSOL-validation cylinders are built.
    """
    import numpy as np
    import trimesh
    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    circle = np.column_stack([radius_mm * np.cos(theta),
                              radius_mm * np.sin(theta)])          # (M, 2) mm
    nz = max(2, int(round(length_mm / ring_dz_mm)) + 1)
    zs = np.linspace(-length_mm / 2.0, length_mm / 2.0, nz)
    M = sections
    verts = np.vstack([np.column_stack([circle, np.full(M, z)]) for z in zs])
    ic_lo, ic_hi = len(verts), len(verts) + 1
    verts = np.vstack([verts, [0.0, 0.0, zs[0]], [0.0, 0.0, zs[-1]]])
    faces = []
    for k in range(nz - 1):                                       # side walls
        a0, b0 = k * M, (k + 1) * M
        for i in range(M):
            j = (i + 1) % M
            faces += [[a0 + i, a0 + j, b0 + j], [a0 + i, b0 + j, b0 + i]]
    top0 = (nz - 1) * M
    for i in range(M):                                            # end-cap fans
        j = (i + 1) % M
        faces += [[ic_lo, j, i], [ic_hi, top0 + i, top0 + j]]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, np.int64))
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description="golgi headless recruitment-sweep demo.")
    ap.add_argument(
        "--nerve", type=Path, default=None,
        help="Endoneurium surface (STL/NAS/OBJ). Default: a synthetic "
             "cylindrical nerve generated on the fly.")
    ap.add_argument(
        "--project", type=Path,
        default=Path.cwd() / "golgi_demo_recruitment_sweep",
        help="Project directory (default: ./golgi_demo_recruitment_sweep).")
    args = ap.parse_args(argv)

    # ---- 1. Project setup ----
    # Idempotent: a prior run's directory is wiped and rebuilt.
    project_dir = args.project
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir)
    s = golgi.Study.create(project_dir)
    print(f"[1] created project: {s.project_dir}")

    # ---- 2. Nerve geometry ----
    if args.nerve is not None:
        stl_path = args.nerve
        print(f"[2] nerve geometry: {stl_path} (user-provided)")
    else:
        stl_path = project_dir / "synthetic_nerve.stl"
        make_synthetic_nerve(stl_path)
        print(f"[2] nerve geometry: {stl_path.name} (synthetic cylinder)")
    info = s.import_nerve(stl_path)
    print(
        f"    imported {info['n_pts']:,} pts, {info['n_tris']:,} tris, "
        f"bbox {info['bbox_mm']}")

    # ---- 3. Mesh parameters ----
    s.set_mesh(
        use_epi=True,
        epi_thickness_um=50,
        lc_endo_um=200,
        lc_epi_um=150,
        lc_muscle_um=1000,
        lc_saline_um=150,
        lc_silicone_um=300,
        lc_contact_um=100,
        lc_scar_um=150,
        muscle_radial_pad_mm=5,
        muscle_axial_pad_mm=10,
    )

    # ---- 4. Electrodes ----
    # A bipolar ring pair around the nerve, centred along its length.
    # cuff_offset_mm is measured from the nerve's low-z end, so half the
    # nerve length centres the cuff on the trunk (a cuff at the end cap would
    # leave no nerve on one side → a degenerate field).
    s.set_electrodes([
        {
            "eid": "elec_01",
            "name": "Bipolar cuff",
            "cuff_offset_mm": NERVE_LENGTH_MM / 2.0,
            "electrode_type": "bipolar ring-pair",
        },
    ])
    print("[3] set 1 electrode")

    s.run_mesh()
    print("[4] meshed")

    # ---- 5. Fibers (trajectories traced through the nerve) ----
    # Trace fibers BEFORE the FEM solve: run_fem samples each contact's lead
    # field onto the existing fiber paths (that per-fiber potential is what the
    # threshold sweep needs). Fibers first → the field lands on them.
    s.set_fiber_seed(n_fibers=50, fiber_auto_detect_branches=True)
    s.run_fibers()
    print("[5] fiber trajectories generated")

    # ---- 6. FEM (per-contact lead fields, sampled onto the fibers) ----
    s.run_fem()
    print("[6] FEM solved")

    # ---- 7. Amplitude sweep + threshold finder ----
    from golgi.jobs.schemas import SweepRequest
    result = s.run_sweep(SweepRequest(
        mode="threshold",
        bisect_lo_mA=0.01,
        bisect_hi_mA=10.0,
        bisect_tol_uA=10.0,
        # No population step in this minimal demo — sweep every generated
        # fiber at the single-fiber default diameter/model instead.
        model_source="single_fiber",
    ))
    n_act = int((result.thresholds_uA > 0).sum())
    n_tot = int(result.thresholds_uA.size)
    print(f"[7] swept thresholds: {n_act}/{n_tot} fibers activated (≤10 mA)")

    # ---- 8. Bundle export ----
    out_zip = project_dir.parent / f"{project_dir.name}_study.zip"
    s.export_bundle(out_zip)
    print(f"[8] wrote study bundle: {out_zip}")

    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
