# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Package the fig4–fig8 nerve simulations as hashed, replay-verified
golgi study bundles for the Zenodo deposit (cited from the manuscript).

Each bundle is a self-contained, integrity-hashed golgi project zip:
geometry/mesh + per-config FEM lead-fields + fiber population + the
per-contact recordings, plus a SHA-256 MANIFEST. A reviewer imports it
with `golgi.Study.import_bundle(...)` (or the GUI File → Import study)
and `golgi replay` re-verifies it byte-for-byte.

The figure pipelines wrote their artifacts in three shapes:
  * "direct"   — already a golgi project dir (designs/ + configs/); we
                 just ensure a project.json and export it.
  * "assemble" — a bespoke flat / per-config layout; we stage a clean
                 project via FILE SYMLINKS (disk is tight, no copies)
                 into the standard designs/ + configs/ layout, then
                 export. The exporter resolves symlinks and hashes the
                 real bytes, so the bundle is a normal standalone zip.

fig4b (NRV LIFE) has no saved FEM/recordings on disk → it must be
re-run (paper_figs/validate_nrv_fem.py) before it can be bundled; it is
listed here as SKIP so the gap is explicit rather than silent.

Usage:
    python paper_figs/make_study_bundles.py [fig_id ...]
        no args  → all available specs
        fig_id   → only those (e.g. fig07_rabbit fig08_human_scb)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(
    "/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/"
    "Fenics_tests"
)
sys.path.insert(0, str(ROOT))

from golgi.projects import bundle as _bundle      # noqa: E402
from golgi.projects.replay import replay_study     # noqa: E402

INTER = ROOT / "paper_figs/out/_intermediate"
DEPOSIT = ROOT / "paper_figs/out/study_bundles"
STAGE = DEPOSIT / "_staging"

_NOW = _dt.datetime.now().replace(microsecond=0).isoformat()

# Files that belong with the per-DESIGN folder (geometry + fibers +
# the PLC/tetgen payload); everything else flat goes to the config.
_DESIGN_NAMES = {
    "nerve.msh", "nerve_paths_fibers.npz", "nerve_surface_pts.npz",
    "current_plc.vtp", "current_tetgen.npz",
    "current_tetgen_payload.json", "nerve_paths_caps.json",
}


def _symlink(dest: Path, src: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(src.resolve())


def _write_project_json(pdir: Path, name: str, source_file: str = "") -> None:
    """Write a minimal-but-valid project.json so the bundle lists +
    loads as a golgi project (the heavy params live in mesh_config.json
    / configs/<id>/electrode_config.json on disk)."""
    pj = pdir / "project.json"
    existing = {}
    if pj.is_file():
        try:
            existing = json.loads(pj.read_text())
        except Exception:                                # noqa: BLE001
            existing = {}
    payload = {
        "version": 1,
        "name": name,
        "created": existing.get("created", _NOW),
        "last_modified": _NOW,
        "source_file": existing.get("source_file", source_file),
        "labels": existing.get("labels", []),
        "ui_state": existing.get("ui_state", {}),
    }
    pj.write_text(json.dumps(payload, indent=2))


# --- assemble builders (flat / per-config → standard layout) ----------
def _assemble_flat_single(src: Path, stage: Path) -> None:
    """One nerve, one FEM config laid out flat in `src` (fig5 swine)."""
    eid, cid = "elec_01", "cfg_01"
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if p.name in _DESIGN_NAMES:
            _symlink(stage / "designs" / eid / p.name, p)
        else:
            _symlink(stage / "configs" / cid / p.name, p)
    mc = src / "mesh_config.json"
    if mc.is_file():
        _symlink(stage / "mesh_config.json", mc)


def _assemble_bucksot(src: Path, stage: Path) -> None:
    """Bucksot: one nerve, two montage configs (circ + inverted), each
    in its own subdir, plus root thr_*.npz threshold sweeps + masks."""
    eid = "elec_01"
    # geometry from the circ config (both configs share the nerve)
    msh = src / "circ" / "nerve.msh"
    if msh.is_file():
        _symlink(stage / "designs" / eid / "nerve.msh", msh)
    for cid in ("circ", "inverted"):
        cdir = src / cid
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.iterdir()):
            if p.is_file() and p.name != "nerve.msh":
                _symlink(stage / "configs" / cid / p.name, p)
    # threshold sweeps → sweep_*.npz so the bundler's glob picks them up
    for thr in sorted(src.glob("thr_*.npz")):
        _symlink(stage / f"sweep_{thr.name}", thr)
    # source histology masks
    masks = src / "masks"
    if masks.is_dir():
        for p in sorted(masks.iterdir()):
            if p.is_file():
                _symlink(stage / "source" / p.name, p)


# --- specs ------------------------------------------------------------
SPECS = [
    dict(id="fig04a_dogvns_validation", name="fig4a — dog cervical VNS validation",
         mode="direct", src=INTER / "dogvns_native" / "study"),
    dict(id="fig04b_nrv_life_validation", name="fig4b — NRV LIFE validation",
         mode="skip", src=INTER / "nrv_life" / "study",
         reason="synthetic LIFE study not re-meshable on this pipeline (TetGen "
                "ignores the sizing field → runaway; gmsh stalls). Figure is "
                "reproducible from cached data (fig04b_nrv_life_figure_data/)."),
    dict(id="fig04c_bucksot_validation", name="fig4c — Bucksot multifascicular cuff validation",
         mode="assemble", src=INTER / "bucksot", build=_assemble_bucksot),
    dict(id="fig05_swine_cervical_vagus", name="fig5 — swine cervical vagus selectivity",
         mode="assemble", src=INTER / "reseed_sub-4_sam-3", build=_assemble_flat_single),
    dict(id="fig06_human_cervical_vagus", name="fig6 — human cervical vagus selectivity",
         mode="direct", src=INTER / "human_bundle_project"),
    dict(id="fig07_rabbit_branching", name="fig7 — rabbit branching vagus (cardiac branch)",
         mode="direct", src=INTER / "rabbit_gui_study"),
    dict(id="fig08_human_scb_branching", name="fig8 — human cervical vagus branch selectivity",
         mode="direct", src=INTER / "new_human_project"),
]


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Per-bundle activation-threshold fiber classes. Each bundle reproduces
# its own validation's fiber types; the cached sweep assigns these
# (diameter µm, MRG/small-MRG model) round-robin across the bundle's
# cached fibers and runs one threshold bisection per fiber, so the GUI's
# threshold scatter shows the per-class thresholds on import.
#
# dog-VNS: A 7.8 / fast-B 3.6 / slow-B 2.1 µm MRG + small 1.0 µm
# (validate_dogvns_native.py). Other bundles fall back to _DEFAULT_-
# CLASSES until their validation classes are wired in — extract from the
# matching fig*_thresholds / validate_* script.
_DEFAULT_CLASSES = [
    {"diam_um": 10.0, "model": "MRG_INTERPOLATION"},
    {"diam_um": 5.7, "model": "MRG_INTERPOLATION"},
    {"diam_um": 2.0, "model": "SMALL_MRG_INTERPOLATION"},
]
# Per-bundle activation-threshold fiber populations. A value is EITHER a
# list of discrete {diam_um, model} classes assigned round-robin, OR a
# {"preset": <POP_PRESETS name>, "n_c_frac": f} that samples the paper's
# curated species vagus population (golgi.state_defaults.pop_presets)
# exactly like fig5_thresholds.sample_pop.
#
# 4a dog-VNS uses the validation's discrete A/B/B/C classes. The
# selectivity bundles (5/6/7/8) draw the realistic species distribution.
# NOTE: rabbit (7) + Bucksot (4c) fall back to the rat preset — confirm
# against each figure's actual --pop choice before the final deposit.
FIBER_CLASSES: dict[str, object] = {
    "fig04a_dogvns_validation": [
        {"diam_um": 7.8, "model": "MRG_INTERPOLATION"},
        {"diam_um": 3.6, "model": "MRG_INTERPOLATION"},
        {"diam_um": 2.1, "model": "MRG_INTERPOLATION"},
        # C fibre — unmyelinated Tigerholm (MRG-family models reject
        # diameters < 1.011 µm).
        {"diam_um": 1.0, "model": "TIGERHOLM"},
    ],
    "fig04c_bucksot_validation": {"preset": "cervical_vagus_rat",
                                   "n_c_frac": 0.0},
    "fig05_swine_cervical_vagus": {"preset": "cervical_vagus_pig",
                                   "n_c_frac": 0.25},
    "fig06_human_cervical_vagus": {"preset": "cervical_vagus_human",
                                   "n_c_frac": 0.25},
    "fig07_rabbit_branching": {"preset": "cervical_vagus_rat",
                               "n_c_frac": 0.25},
    "fig08_human_scb_branching": {"preset": "cervical_vagus_human",
                                  "n_c_frac": 0.25},
}

# MRG-family models reject diameters below these floors; unmyelinated
# models take the small end. Mirrors fig5_thresholds.CLIP.
_CLIP = {"MRG_INTERPOLATION": (2.0, 16.0),
         "SMALL_MRG_INTERPOLATION": (1.5, 5.0)}
_UNMYEL = {"SUNDT", "TIGERHOLM", "RATTAY", "SCHILD94", "SCHILD97"}


def _population_for(spec, n):
    """Return (diameters_um, models) arrays of length n for a bundle's
    FIBER_CLASSES entry — either round-robin discrete classes or a
    sampled species preset (fixed seed for reproducible bundles)."""
    import numpy as np
    if isinstance(spec, list):
        diam = np.array([spec[i % len(spec)]["diam_um"]
                         for i in range(n)], float)
        model = [spec[i % len(spec)]["model"] for i in range(n)]
        return diam, model
    # preset spec
    from golgi.state_defaults.pop_presets import POP_PRESETS
    rows = POP_PRESETS[spec["preset"]].templates[0].rows
    myel = [r for r in rows if r.model not in _UNMYEL]
    crows = [r for r in rows if r.model in _UNMYEL]
    rng = np.random.default_rng(1)
    n_c = int(round(n * float(spec.get("n_c_frac", 0.0)))) if crows else 0
    n_c = min(n_c, n)
    n_my = n - n_c
    diam = np.zeros(n)
    model = [""] * n
    w = np.array([r.frac for r in myel], float)
    w /= w.sum()
    sel = rng.choice(len(myel), size=n_my, p=w)
    for k in range(n_my):
        r = myel[sel[k]]
        lo, hi = _CLIP.get(r.model, (1.0, 16.0))
        diam[k] = float(np.clip(rng.normal(r.mean_um, r.std_um), lo, hi))
        model[k] = r.model
    for k in range(n_my, n):
        cr = crows[0]
        diam[k] = float(np.clip(rng.normal(cr.mean_um, cr.std_um),
                                0.25, 2.0))
        model[k] = cr.model
    p = rng.permutation(n)
    return diam[p], [model[i] for i in p]


def _ensure_threshold_sweep(proj: Path, fid: str) -> str:
    """Compute + cache a per-class activation-threshold sweep on `proj`
    so the exported bundle surfaces thresholds in the GUI's Sweep tab on
    import (via <project>/sweeps/). Idempotent: skips if a sweep cache
    already exists. Needs NEURON/pyfibers — returns a status string; on
    any failure (no NEURON, no fibers/FEM on disk) it logs and returns
    the reason so the bundle still exports, just without thresholds."""
    if (proj / "sweeps" / "latest.txt").is_file() or list(
        proj.glob("sweeps/sweep_*.npz")
    ):
        return "cached (already present)"
    try:
        import numpy as np
        import golgi
        from golgi.jobs.schemas import SweepRequest
    except Exception as ex:                                  # noqa: BLE001
        return f"skip: golgi import failed ({ex})"
    spec = FIBER_CLASSES.get(fid, _DEFAULT_CLASSES)
    try:
        s = golgi.Study.open(proj)
    except Exception as ex:                                  # noqa: BLE001
        return f"skip: could not open project ({ex})"
    try:
        info = s.load_cached_geometry()
        n = int(info.get("n_fibers") or 0)
        if not n or not info.get("n_ve_fibers"):
            return (
                f"skip: no cached fibers/field "
                f"(fibers={n}, Ve={info.get('n_ve_fibers')})"
            )
        diams, models = _population_for(spec, n)
        s._geom.fiber_pop_diameters_um = np.asarray(diams, dtype=np.float64)
        s._geom.fiber_pop_types = list(models)
        s.run_sweep(SweepRequest(
            mode="threshold",
            bisect_lo_mA=0.001,
            bisect_hi_mA=20.0,
            bisect_tol_uA=10.0,
            model_source="population",
        ))
        if isinstance(spec, dict):
            desc = f"preset {spec['preset']}"
        else:
            desc = "classes " + "/".join(
                f"{c['diam_um']:g}" for c in spec) + "µm"
        dflt = " (DEFAULT — not wired)" if fid not in FIBER_CLASSES else ""
        return (
            f"ok ({n} fibers, {desc}, "
            f"{len(set(models))} model(s)){dflt}"
        )
    except Exception as ex:                                  # noqa: BLE001
        return f"skip: sweep failed ({type(ex).__name__}: {ex})"
    finally:
        try:
            s.close()
        except Exception:                                    # noqa: BLE001
            pass


def _export_one(spec: dict) -> dict | None:
    fid, name, mode = spec["id"], spec["name"], spec["mode"]
    if mode == "skip":
        print(f"  SKIP {fid}: {spec['reason']}")
        return {"id": fid, "name": name, "status": "skipped",
                "reason": spec["reason"]}
    src = spec["src"]
    if not src.exists():
        print(f"  SKIP {fid}: source missing {src}")
        return {"id": fid, "name": name, "status": "missing", "reason": str(src)}

    if mode == "direct":
        proj = src
        _write_project_json(proj, name)
    else:  # assemble
        proj = STAGE / fid
        if proj.exists():
            shutil.rmtree(proj)
        proj.mkdir(parents=True, exist_ok=True)
        spec["build"](src, proj)
        _write_project_json(proj, name)

    # Cache an activation-threshold sweep so the bundle shows thresholds
    # in the GUI on import (needs NEURON; skips cleanly if unavailable).
    _sw = _ensure_threshold_sweep(proj, fid)
    print(f"    threshold sweep: {_sw}", flush=True)

    out_zip = DEPOSIT / f"{fid}.golgi.zip"
    print(f"  export {fid} ← {proj.relative_to(ROOT)} …", flush=True)
    blob = _bundle.export_study(proj, exported_by_user="golgi-paper")
    out_zip.write_bytes(blob)
    del blob

    rep = replay_study(out_zip, check_only=True)
    sz = out_zip.stat().st_size
    zsha = _sha256_file(out_zip)
    ok = bool(rep.ok) and rep.n_files_mismatched == 0 and rep.n_files_missing == 0
    print(f"    {'OK ' if ok else 'FAIL'} {sz/1e6:.1f} MB · "
          f"{rep.n_files_matched}/{rep.n_files_total} files verified · "
          f"sha256 {zsha[:16]}…", flush=True)

    # clean up symlink staging (tiny, but keep the deposit dir clean)
    if mode == "assemble" and proj.exists():
        shutil.rmtree(proj)

    return {"id": fid, "name": name, "status": "ok" if ok else "verify_failed",
            "zip": out_zip.name, "size_bytes": sz, "sha256": zsha,
            "n_files": rep.n_files_total, "n_verified": rep.n_files_matched}


def main() -> None:
    want = set(sys.argv[1:])
    # Initialise auth/audit DB so the bundle's audit excerpt populates
    # (otherwise export prints a non-fatal "Auth DB not initialised").
    try:
        from golgi.app import _ensure_initialized
        _ensure_initialized()
    except Exception:                                    # noqa: BLE001
        pass
    DEPOSIT.mkdir(parents=True, exist_ok=True)
    specs = [s for s in SPECS if not want or s["id"] in want]
    results = []
    for spec in specs:
        print(f"[{spec['id']}]")
        try:
            results.append(_export_one(spec))
        except Exception as ex:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"id": spec["id"], "status": "error", "reason": str(ex)})

    # summary + checksums for the Zenodo deposit
    (DEPOSIT / "BUNDLES.json").write_text(json.dumps(results, indent=2))
    lines = [f"{r['sha256']}  {r['zip']}" for r in results
             if r.get("status") == "ok"]
    (DEPOSIT / "CHECKSUMS.sha256").write_text("\n".join(lines) + ("\n" if lines else ""))

    print("\n=== SUMMARY ===")
    for r in results:
        extra = (f"{r.get('size_bytes',0)/1e6:6.1f} MB  "
                 f"{r.get('n_verified','?')}/{r.get('n_files','?')}"
                 if r.get("status") == "ok" else r.get("reason", ""))
        print(f"  {r['status']:14s} {r['id']:34s} {extra}")
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n{n_ok}/{len(results)} bundles written + replay-verified → "
          f"{DEPOSIT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
