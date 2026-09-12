# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end regression test against recorded reference results.

Runs the reference study shipped as ``examples/recruitment_sweep.py``
(synthetic monofascicular nerve, bipolar ring cuff, 12 fibers, NEURON/MRG
threshold bisection) through the headless API and compares the outcome
with ``tests/reference/e2e_synthetic_reference.json``, which was produced
with the release environment (see the ``environment`` block in that file).

What is asserted
  * every stage produces its artifacts (mesh, lead fields, fiber paths,
    thresholds, bundle);
  * all 12 fibers reach threshold below the 10 mA bisection ceiling;
  * the median activation threshold lies within ``rel_tol`` of the recorded
    value (mesh generation and the direct solver are deterministic for a
    given library build, but element counts and therefore thresholds may
    move slightly between platforms/compilers — the tolerance covers that);
  * the exported bundle verifies with ``golgi replay``.

Requires the full solver stack; auto-skips without it. Run explicitly with

    pytest -m integration tests/test_e2e_reference.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).parent / "reference" / "e2e_synthetic_reference.json"
_SOLVER_MODS = ("dolfinx", "gmsh", "pyfibers", "tetgen")


def _missing() -> list[str]:
    return [m for m in _SOLVER_MODS if importlib.util.find_spec(m) is None]


@pytest.mark.integration
@pytest.mark.skipif(bool(_missing()), reason=f"solver stack missing: {_missing()}")
@pytest.mark.skipif(not REF.is_file(), reason="reference results file missing")
def test_reference_study_reproduces_recorded_thresholds(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from recruitment_sweep import NERVE_LENGTH_MM, make_synthetic_nerve  # noqa: E402
    import golgi
    from golgi.jobs.schemas import SweepRequest
    from golgi.projects import replay as _replay

    ref = json.loads(REF.read_text())
    n_fibers = int(ref["n_fibers"])

    proj = tmp_path / "ref_study"
    s = golgi.Study.create(proj)
    stl = make_synthetic_nerve(proj / "synthetic_nerve.stl")
    info = s.import_nerve(stl)
    assert info["n_tris"] > 0 and info.get("watertight", True)

    s.set_mesh(**ref["mesh_params"])
    s.set_electrodes([{"eid": "elec_01", "name": "Bipolar cuff",
                       "cuff_offset_mm": NERVE_LENGTH_MM / 2.0,
                       "electrode_type": "bipolar ring-pair"}])
    meshes = s.run_mesh()
    assert meshes and all(p.is_file() for p in meshes.values())

    s.set_fiber_seed(n_fibers=n_fibers, fiber_auto_detect_branches=True)
    fib = s.run_fibers()
    assert fib["n_paths"] == n_fibers

    fem = s.run_fem()
    assert fem and all(d.is_dir() for d in fem.values())

    res = s.run_sweep(SweepRequest(mode="threshold", **ref["sweep_params"]))
    thr = np.asarray(res.thresholds_uA, dtype=float)
    assert thr.size == n_fibers
    assert np.all(np.isfinite(thr)) and np.all(thr > 0), thr
    assert np.all(thr < ref["sweep_params"]["bisect_hi_mA"] * 1e3)

    med = float(np.median(thr))
    ref_med = float(ref["thresholds_uA"]["median"])
    rel = abs(med - ref_med) / ref_med
    assert rel <= ref["rel_tol"], (
        f"median threshold {med:.1f} µA deviates {rel:.1%} from the "
        f"recorded {ref_med:.1f} µA (tolerance {ref['rel_tol']:.0%})")
    lo, hi = ref["thresholds_uA"]["min"], ref["thresholds_uA"]["max"]
    assert thr.min() >= lo * (1 - ref["rel_tol"]) - ref["sweep_params"]["bisect_tol_uA"]
    assert thr.max() <= hi * (1 + ref["rel_tol"]) + ref["sweep_params"]["bisect_tol_uA"]

    out_zip = tmp_path / "ref_study.golgi.zip"
    s.export_bundle(out_zip)
    s.close()
    report = _replay.replay_study(out_zip, check_only=True)
    assert report.ok and report.n_files_mismatched == 0
