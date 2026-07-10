"""F4.1 example — full headless pipeline: load nerve → mesh →
electrodes → FEM → fibers → threshold sweep → bundle export.

Run from a venv with golgi installed:

    python examples/recruitment_sweep.py

All five compute methods are wired as of F4.1 Phase D
(import_nerve, run_mesh, run_fibers, run_fem, run_sweep). The
script runs end-to-end on the bundled sample nerve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import golgi


def main() -> int:
    # ---- 1. Project setup ----
    # Create the project in the current working directory (where you
    # ran this script) so its outputs are easy to find. Idempotent:
    # a prior run's directory is wiped and rebuilt from scratch.
    project_dir = Path.cwd() / "golgi_demo_recruitment_sweep"
    if project_dir.exists():
        # Wipe and recreate for a clean run.
        import shutil
        shutil.rmtree(project_dir)
    s = golgi.Study.create(project_dir)
    print(f"[1] created project: {s.project_dir}")

    # ---- 2. Import nerve ----
    # Default to the bundled sample geometry. Override with your
    # own STL/NAS/OBJ via the `GOLGI_DEMO_STL` env var.
    import os
    stl_path = Path(os.environ.get(
        "GOLGI_DEMO_STL",
        "data/ENDONERIUM_masks_ns_2w_processed_reduced_2w_reduced2.stl",
    ))
    info = s.import_nerve(stl_path)
    print(
        f"[2] imported nerve: {stl_path.name} "
        f"({info['n_pts']:,} pts, {info['n_tris']:,} tris, "
        f"bbox {info['bbox_mm']})"
    )

    # ---- 3. Mesh parameters + run mesh ----
    s.set_mesh(
        use_epi=True,
        epi_thickness_um=50,
        decim_target_k=60,
        lc_endo_um=200,
        lc_epi_um=150,
        lc_muscle_um=1000,
        lc_saline_um=150,
        lc_silicone_um=300,
        lc_contact_um=100,
        lc_scar_um=150,
        muscle_radial_pad_mm=20,
        muscle_axial_pad_mm=80,
    )

    # ---- 4. Electrodes ----
    # Bipolar ring pair, centred 5 mm along the nerve. The full
    # per-design schema is in `golgi.app.DEFAULT_CUFF` +
    # `DEFAULT_ELECTRODE`; only the cuff offset is overridden
    # here, the rest take their defaults.
    s.set_electrodes([
        {
            "eid": "elec_01",
            "name": "Bipolar @ 5 mm",
            "cuff_offset_mm": 5.0,
            "electrode_type": "bipolar ring-pair",
        },
    ])
    print("[4] set 1 electrode")

    s.run_mesh()
    print("[3] meshed designs")

    # ---- 5. FEM ----
    s.run_fem()
    print("[5] FEM solved for every config")

    # ---- 6. Fibers ----
    s.set_fiber_seed(
        n_fibers=50,
        fiber_auto_detect_branches=True,
    )
    s.run_fibers()
    print("[6] fiber trajectories generated")

    # ---- 7. Amplitude sweep + threshold finder ----
    from golgi.jobs.schemas import SweepRequest
    sweep_req = SweepRequest(
        mode="threshold",
        bisect_lo_mA=0.01,
        bisect_hi_mA=2.0,
        bisect_tol_uA=10.0,
    )
    result = s.run_sweep(sweep_req)
    print(
        f"[7] swept thresholds: "
        f"{int((result.thresholds_uA > 0).sum())} fibers "
        f"activated"
    )

    # ---- 8. Bundle export ----
    out_zip = project_dir.parent / (
        f"{project_dir.name}_study.zip"
    )
    s.export_bundle(out_zip)
    print(f"[8] wrote study bundle: {out_zip}")

    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
