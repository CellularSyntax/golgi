# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""End-to-end headless example: image-to-recruitment in one script.

Drives the full golgi pipeline through the headless ``golgi.Study`` API —
the same drivers the GUI uses — with no browser:

    import_nerve → set_mesh → set_electrodes → run_mesh → run_fem
                 → run_fibers → run_sweep (recruitment) → export_bundle

It writes a recruitment-curve PNG and a reproducible, integrity-hashed
study bundle to the project directory.

Usage
-----
    # Run on a built-in synthetic nerve (no data needed):
    python examples/recruitment_sweep.py

    # Run on your own nerve surface (STL / NAS / OBJ, units = mm):
    python examples/recruitment_sweep.py --nerve /path/to/nerve.stl

    # Choose where the project is written:
    python examples/recruitment_sweep.py --project /tmp/my_vagus_study

Requirements
------------
The compute stages need the FEniCSx solver stack (dolfinx, gmsh,
tetgen/wildmeshing, pyfibers). See the Installation page of the wiki:
https://github.com/CellularSyntax/golgi/wiki/Installation
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

import golgi
from golgi.jobs.schemas import SweepRequest


def build_synthetic_nerve(path: Path) -> Path:
    """Write a clean, watertight ~2 mm × 30 mm capsule nerve as STL.

    A capped cylinder with gentle multi-frequency radial bumps: watertight
    by construction and *slightly* irregular, so the multi-domain cuff PLC
    meshes reliably (a perfectly concentric cylinder produces coincident
    facets TetGen's exact predicates reject). Good enough to exercise the
    whole pipeline; pass ``--nerve`` to use a real geometry instead.
    """
    import pyvista as pv

    cyl = pv.Cylinder(
        center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        radius=1.0, height=30.0, resolution=64, capping=True,
    ).triangulate().subdivide(2)
    try:
        cyl = cyl.smooth_taubin(n_iter=40, pass_band=0.1)
    except AttributeError:                                # older pyvista
        cyl = cyl.smooth(n_iter=60, relaxation_factor=0.04)
    cyl = cyl.clean().triangulate()
    pts = cyl.points.copy()
    rad = np.hypot(pts[:, 0], pts[:, 1])
    th = np.arctan2(pts[:, 1], pts[:, 0])
    m = rad > 1e-6
    bump = (0.06 * np.sin(3.0 * th + 0.7 * pts[:, 2])
            + 0.04 * np.cos(5.0 * th - 0.4 * pts[:, 2]))
    new_r = rad[m] * (1.0 + bump[m])
    pts[m, 0] = new_r * np.cos(th[m])
    pts[m, 1] = new_r * np.sin(th[m])
    cyl.points = pts
    cyl = cyl.clean().triangulate()
    path.parent.mkdir(parents=True, exist_ok=True)
    cyl.save(str(path))
    return path


def plot_recruitment(result, out_png: Path) -> None:
    """Plot population recruitment (fraction activated) vs amplitude."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amps = np.asarray(result.request.amplitudes_mA, dtype=float)
    activated = np.asarray(result.activated)              # (n_fibers, n_amps)
    recruited = activated.mean(axis=0)                    # fraction in [0, 1]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(amps, 100.0 * recruited, "-o", lw=2, ms=4, color="#c1272d")
    ax.set_xlabel("stimulus amplitude (mA)")
    ax.set_ylabel("fibers recruited (%)")
    ax.set_title("Recruitment curve")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nerve", type=Path, default=None,
                    help="Nerve surface (STL/NAS/OBJ, mm). Default: synthetic.")
    ap.add_argument("--project", type=Path, default=Path("./vagus_study"),
                    help="Project directory to create (default: ./vagus_study).")
    ap.add_argument("--fibers", type=int, default=20,
                    help="Number of fibers to seed (default: 20).")
    args = ap.parse_args()

    project = args.project.expanduser().resolve()
    if project.exists():
        shutil.rmtree(project)                           # fresh run each time

    nerve = args.nerve
    if nerve is None:
        nerve = build_synthetic_nerve(project.parent / "synthetic_nerve.stl")
        print(f"[example] using synthetic nerve: {nerve}")

    s = golgi.Study.create(project)

    # 1) Load the nerve surface (PCA frame + surface quality + topology).
    info = s.import_nerve(nerve)
    print(f"[example] nerve: {info['n_pts']} pts, {info['n_tris']} tris, "
          f"watertight={info['watertight']}")

    # 2) Mesh + electrode design (a centred bipolar ring-pair cuff).
    s.set_mesh(
        use_epi=False, decim_target_k=60,
        lc_endo_um=300, lc_muscle_um=1500,
        lc_saline_um=300, lc_silicone_um=400, lc_contact_um=200,
        muscle_radial_pad_mm=6, muscle_axial_pad_mm=8,
    )
    s.set_electrodes([{
        "eid": "elec_01",
        "name": "Bipolar (centred)",
        "cuff_anchor": "centroid",
        "cuff_offset_mm": 0.0,
        "L_cuff_mm": 6.0,
        "cuff_clearance_mm": 1.2,
        "cuff_wall_mm": 1.2,
        "electrode_type": "bipolar ring-pair",
    }])

    # 3) Build the multi-region TetGen mesh.
    meshes = s.run_mesh()
    print(f"[example] meshed {len(meshes)} design(s)")

    # 4) Solve the anisotropic extracellular field (per-contact lead fields).
    fem = s.run_fem()
    print(f"[example] solved FEM for {len(fem)} config(s)")

    # 5) Generate curved 3-D fiber trajectories on the nerve.
    s.set_fiber_seed(n_fibers=args.fibers, fiber_auto_detect_branches=True)
    fibers = s.run_fibers()
    print(f"[example] {fibers['n_paths']} fibers across "
          f"{fibers['n_branches']} branch(es)")

    # 6) Sweep amplitude → recruitment curve (NEURON/PyFibers backend).
    result = s.run_sweep(SweepRequest(
        mode="recruitment",
        amplitudes_mA=list(np.linspace(0.1, 2.0, 20)),
        backend="pyfibers",
        model_name="MRG_INTERPOLATION",
    ))
    out_png = project / "recruitment_curve.png"
    plot_recruitment(result, out_png)
    print(f"[example] wrote recruitment curve → {out_png}")

    # 7) Export a reproducible, integrity-hashed study bundle.
    bundle = s.export_bundle(project.parent / "vagus_study.golgi")
    print(f"[example] exported study bundle → {bundle}")
    print("[example] verify it any time with:  golgi replay "
          f"{bundle}")

    s.close()


if __name__ == "__main__":
    main()
