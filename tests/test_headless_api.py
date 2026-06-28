# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Headless `Study` API tests (F4.1).

Three tiers, fastest first:

  1. ``test_compute_*`` — pure source/attribute guards. No imports beyond
     ``golgi.api``, so they run anywhere (no solver stack needed). They pin
     the fact that the compute methods are *wired to the pipeline drivers*,
     not stubbed — so the old "F4.1 Phase B/C raises NotImplementedError"
     status can never silently come back.

  2. ``test_study_lifecycle`` — needs ``golgi.app`` importable (it pulls the
     Trame/VTK stack). Auto-skips otherwise. Exercises project create +
     parameter setters + inspectors + the no-clobber guard, with no solver.

  3. ``test_end_to_end_pipeline`` — the real
     ``import_nerve → run_mesh → run_fem → run_fibers → run_sweep`` chain on a
     sample nerve. Auto-skips unless the FEniCSx solver stack (dolfinx, gmsh,
     tetgen/wildmeshing, pyfibers) **and** a sample geometry are present. This
     is the reproducibility verification the platform paper leans on.

Run just the fast guards anywhere::

    pytest tests/test_headless_api.py -m "not integration"

Run the full chain on a box with the solver stack::

    pytest tests/test_headless_api.py -m integration
    # or point it at your own geometry:
    GOLGI_TEST_NERVE=/path/to/nerve.stl pytest tests/test_headless_api.py -m integration
"""
from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path

import pytest

from golgi.api import Study

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Tier 1 — source / attribute guards (always run, no heavy deps)
# ---------------------------------------------------------------------------
# method -> distinctive token(s) proving it dispatches to its pipeline driver.
# (any() match, so a single rename in the driver layer trips the test loudly.)
_WIRED = {
    "import_nerve": ("load_nerve_file",),
    "run_mesh": ("run_mesh_build",),
    "run_fem": ("run_fem_solve",),
    "run_fibers": ("run_generate_fibers",),
    "run_sweep": ("_pipeline_sweep", "save_sweep"),
}


@pytest.mark.parametrize("method", sorted(_WIRED))
def test_compute_method_present(method):
    assert callable(getattr(Study, method, None)), f"Study.{method} is missing"


@pytest.mark.parametrize("method,tokens", sorted(_WIRED.items()))
def test_compute_method_not_stubbed(method, tokens):
    src = inspect.getsource(getattr(Study, method))
    assert "NotImplementedError" not in src, (
        f"Study.{method} raises NotImplementedError — the F4.1 Phase B/C "
        f"stub has regressed; it must dispatch to the pipeline driver."
    )
    assert any(tok in src for tok in tokens), (
        f"Study.{method} no longer references its pipeline driver "
        f"{tokens!r}; the headless wiring may be broken."
    )


# ---------------------------------------------------------------------------
# Tier 2 — lifecycle (needs golgi.app; auto-skips if its stack is absent)
# ---------------------------------------------------------------------------
def test_study_lifecycle(tmp_path):
    pytest.importorskip(
        "golgi.app", reason="golgi.app (Trame/VTK stack) not importable here"
    )

    proj = tmp_path / "proj"
    with Study.create(proj) as s:
        assert s.project_dir.is_dir()
        assert s.user  # default 'headless'
        # setters just forward to the state shim; should not raise.
        s.set_mesh(use_epi=True, epi_thickness_um=50, lc_endo_um=300)
        s.set_fiber_seed(n_fibers=8, fiber_auto_detect_branches=True)
        s.set_electrodes([{"eid": "elec_01", "name": "test cuff"}])
        # inspectors return lists on a fresh project (no cuffs persisted yet).
        assert isinstance(s.list_designs(), list)
        assert isinstance(s.list_configs(), list)

    # create() refuses to clobber a non-empty directory.
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "marker.txt").write_text("not empty")
    with pytest.raises(FileExistsError):
        Study.create(busy)

    # open() on a missing directory raises.
    with pytest.raises(FileNotFoundError):
        Study.open(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Tier 3 — end-to-end pipeline (needs the FEniCSx solver stack + a sample)
# ---------------------------------------------------------------------------
_SOLVER_MODS = ("dolfinx", "gmsh", "pyfibers")


def _missing_solver() -> list[str]:
    miss = [m for m in _SOLVER_MODS if importlib.util.find_spec(m) is None]
    # the mesher can be either backend
    if (importlib.util.find_spec("tetgen") is None
            and importlib.util.find_spec("wildmeshing") is None):
        miss.append("tetgen/wildmeshing")
    return miss


def _nerve_stl(path: Path) -> Path:
    """A clean, watertight capsule nerve (radius 1.5 mm, length 25 mm).

    A cylinder with hemispherical end caps, built as a surface of
    revolution. This shape is ideal for the pipeline: it is smooth
    everywhere (the cap↔body junction is tangent-continuous, so the
    µCT-tuned Taubin/clean preprocessing has no sharp rim to
    self-intersect and collapse — what a raw cylinder hits), and it has
    a long *constant-radius* trunk so the cuff fits a clean annulus and
    TetGen doesn't choke on a degenerate cross-section — what the
    tapering spindle hits. Deterministic + small ⇒ fast, reliable mesh.
    The test exercises API plumbing, not meshing robustness on
    pathological µCT geometry. Override with a real sample via
    GOLGI_TEST_NERVE.
    """
    import numpy as np
    pv = pytest.importorskip("pyvista", reason="pyvista needed to build fixture")
    # Dense capped cylinder (watertight by construction), rounded rim
    # (shrink-free Taubin) so the pipeline's Taubin has no sharp edge,
    # then add gentle multi-frequency radial bumps so the surface is
    # *irregular* like a real nerve. Perfect concentric symmetry produces
    # exact-coincidence/coplanar facets that TetGen's exact predicates
    # reject in the multi-domain cuff PLC; a slightly irregular tube
    # avoids that — which is why real nerves mesh and a perfect cylinder
    # doesn't. Thin (1 mm) + long so the cuff annulus stays fat.
    cyl = pv.Cylinder(
        center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        radius=1.0, height=30.0, resolution=64, capping=True,
    ).triangulate().subdivide(2)
    try:
        cyl = cyl.smooth_taubin(n_iter=40, pass_band=0.1)
    except AttributeError:                              # older pyvista
        cyl = cyl.smooth(n_iter=60, relaxation_factor=0.04,
                         feature_smoothing=False, boundary_smoothing=True)
    cyl = cyl.clean().triangulate()
    pts = cyl.points.copy()
    rad = np.hypot(pts[:, 0], pts[:, 1])
    th = np.arctan2(pts[:, 1], pts[:, 0])
    m = rad > 1e-6
    bump = (0.06 * np.sin(3.0 * th + 0.7 * pts[:, 2])
            + 0.04 * np.cos(5.0 * th - 0.4 * pts[:, 2]))   # ~±10 %
    new_r = rad[m] * (1.0 + bump[m])
    pts[m, 0] = new_r * np.cos(th[m])
    pts[m, 1] = new_r * np.sin(th[m])
    cyl.points = pts
    cyl = cyl.clean().triangulate()
    cyl.save(str(path))
    return path


@pytest.fixture
def nerve_stl(tmp_path) -> Path:
    env = os.environ.get("GOLGI_TEST_NERVE") or os.environ.get("GOLGI_DEMO_STL")
    if env and Path(env).is_file():
        return Path(env)
    return _nerve_stl(tmp_path / "cylinder_nerve.stl")


@pytest.mark.integration
@pytest.mark.skipif(
    bool(_missing_solver()),
    reason=f"FEniCSx solver stack missing: {_missing_solver()}",
)
def test_end_to_end_pipeline(tmp_path, nerve_stl):
    """import_nerve → run_mesh → run_fem → run_fibers → run_sweep.

    Mirrors examples/recruitment_sweep.py with a small fiber count for
    speed. Asserts each stage produced the artifacts its return contract
    promises. Uses a synthetic cylinder nerve by default (set
    GOLGI_TEST_NERVE to run against a real geometry).
    """
    s = Study.create(tmp_path / "e2e")

    info = s.import_nerve(nerve_stl)
    assert info["n_pts"] > 0 and info["n_tris"] > 0

    s.set_mesh(
        use_epi=False, decim_target_k=60,          # simplest PLC for the smoke
        lc_endo_um=300, lc_muscle_um=1500,
        lc_saline_um=300, lc_silicone_um=400, lc_contact_um=200,
        muscle_radial_pad_mm=6, muscle_axial_pad_mm=8,
    )
    s.set_electrodes([{
        "eid": "elec_01",
        "name": "Bipolar (centred)",
        # Anchor at the PCA centroid (not the default "trunk" low-z end)
        # so the cuff window straddles the nerve middle and both cap
        # planes land inside the geometry.
        "cuff_anchor": "centroid",
        "cuff_offset_mm": 0.0,
        "L_cuff_mm": 6.0,                 # short cuff → less seam, fits trunk
        "cuff_clearance_mm": 1.2,         # fat saline annulus (no SI on a
        "cuff_wall_mm": 1.2,              #   ~1 mm nerve)  + fat silicone wall
        "electrode_type": "bipolar ring-pair",
    }])

    meshes = s.run_mesh()
    assert meshes, "run_mesh produced no per-design .msh"
    for path in meshes.values():
        assert Path(path).is_file(), f"missing mesh file: {path}"

    fem = s.run_fem()
    assert fem, "run_fem produced no per-config output dir"
    for d in fem.values():
        assert Path(d).is_dir(), f"missing FEM output dir: {d}"

    s.set_fiber_seed(n_fibers=8, fiber_auto_detect_branches=True)
    fibers = s.run_fibers()
    assert fibers["n_paths"] > 0, "run_fibers generated no trajectories"

    from golgi.jobs.schemas import SweepRequest
    result = s.run_sweep(SweepRequest(
        mode="threshold",
        bisect_lo_mA=0.01, bisect_hi_mA=2.0, bisect_tol_uA=10.0,
    ))
    assert hasattr(result, "thresholds_uA")
    assert len(result.thresholds_uA) > 0

    s.close()
