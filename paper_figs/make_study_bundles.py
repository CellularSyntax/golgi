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


# Activation thresholds are bundled from the figures' STORED sweep
# outputs, so a bundle shows the EXACT published numbers on import — no
# recomputation, no re-sampling. Each fig*_thresholds.py already wrote a
# per-fibre threshold .npz (thr_uA[n_fibers×n_contacts], fiber_idx,
# diameter_um, model, type_label, branch_idx) computed on the SAME fibers
# the bundle carries; we reconstitute it into the bundle's sweeps/ cache.
# A single-config bundle shows ONE contact column (the "slice" of the
# figure's position×config sweep). STORED_THR maps bundle id → (npz under
# out/data, contact-column index).
# NOTE: rabbit (12 contacts) + human-SCB tripole (4) — col 0 is a
# placeholder; set it to the contact the bundle's cfg_01 actually solves
# before the final deposit. 4a dog-VNS + 4c Bucksot use validation-table
# formats (not per-fibre) and are handled separately / left uncached.
# col is an int contact column, "best" (pick the config the figure
# highlights = argmax SCB−trunk selectivity, TARGET branch 1), or "json"
# (the dog-VNS validation table: 4 per-class thresholds).
STORED_THR: dict[str, tuple[str, object]] = {
    "fig04a_dogvns_validation": ("data/validate_dogvns_native.json",
                                 "json"),
    "fig04c_bucksot_validation": (
        "_intermediate/bucksot/thr_circ.npz", "bucksot"),
    "fig05_swine_cervical_vagus": ("data/thr_pop_swine.npz", 0),
    "fig06_human_cervical_vagus": ("data/thr_pop_human.npz", 0),
    "fig07_rabbit_branching": ("data/rabbit_branch_thr.npz", "best"),
    "fig08_human_scb_branching": ("data/new_human_tripole_thr.npz",
                                  "best"),
}
OUT_DIR = ROOT / "paper_figs" / "out"
_SCB_BRANCH = 1                                    # TARGET in the figs


def _best_config_col(thr2d, branch) -> int:
    """Pick the config column the selectivity figures highlight: the one
    maximising (SCB% − trunk%) over amplitude, exactly like
    new_human_selectivity_fig.analyse (iop = argmax(Ron − Roff))."""
    import numpy as np
    fin = thr2d[np.isfinite(thr2d)]
    if fin.size == 0:
        return 0
    amps = np.logspace(np.log10(max(fin.min(), 50.0)),
                       np.log10(fin.max()), 160)
    on = np.asarray(branch) == _SCB_BRANCH
    off = ~on
    best_j, best_si = 0, -1e18
    for j in range(thr2d.shape[1]):
        col = thr2d[:, j]
        con, cof = col[on], col[off]
        Ron = np.array([np.mean(np.isfinite(con) & (con <= a))
                        for a in amps])
        Roff = np.array([np.mean(np.isfinite(cof) & (cof <= a))
                         for a in amps])
        si = float(np.max(Ron - Roff))
        if si > best_si:
            best_si, best_j = si, j
    return best_j


def _ingest_stored_thresholds(proj: Path, fid: str) -> str:
    """Reconstitute the figure's STORED per-fibre thresholds into the
    bundle's sweeps/ cache so the GUI shows the exact published numbers.
    Idempotent; no NEURON, no recomputation."""
    if (proj / "sweeps" / "latest.txt").is_file():
        return "cached (already present)"
    ent = STORED_THR.get(fid)
    if ent is None:
        return "no stored-threshold mapping (validation-table figure)"
    fname, col = ent
    src = OUT_DIR / fname
    if not src.is_file():
        return f"skip: stored file missing ({src.name})"
    try:
        import numpy as np
        from golgi.jobs.schemas import SweepRequest, SweepResult
        from golgi.projects import sweep_cache as _swc
        from golgi.pipeline import fem_layout as _fl
    except Exception as ex:                                  # noqa: BLE001
        return f"skip: import failed ({ex})"
    col_desc = str(col)
    if col == "json":
        # dog-VNS validation table → one point per fibre class.
        import json as _json
        try:
            rows = _json.loads(src.read_text())["thresholds"]
        except Exception as ex:                              # noqa: BLE001
            return f"skip: could not read {fname} ({ex})"
        thr = np.array([float(r["thr_mA"]) * 1000.0 for r in rows])
        diam = np.array([float(r["diam_um"]) for r in rows])
        types = [str(r.get("type", "")) for r in rows]
        fidx = np.arange(len(rows), dtype=np.int64)
        bidx = np.zeros(len(rows), dtype=np.int32)
    elif col == "bucksot":
        # Bucksot multifascicular validation: thr in mA, per-fibre with
        # a `fascicle` index (used as branch), no explicit fiber_idx /
        # model. One config's slice (circ); the bundle also carries the
        # inverted config's FEM.
        try:
            d = np.load(src, allow_pickle=True)
            thr = np.asarray(d["thr_mA"], dtype=np.float64) * 1000.0
            diam = np.asarray(d["diam_um"], dtype=np.float64)
            bidx = np.asarray(d["fascicle"], dtype=np.int32) \
                if "fascicle" in d.files else np.zeros(len(thr), np.int32)
            fidx = np.arange(len(thr), dtype=np.int64)
            types = []
        except Exception as ex:                              # noqa: BLE001
            return f"skip: could not read {fname} ({ex})"
    else:
        try:
            d = np.load(src, allow_pickle=True)
            thr2d = np.asarray(d["thr_uA"], dtype=np.float64)
            fidx = np.asarray(d["fiber_idx"], dtype=np.int64)
            diam = np.asarray(d["diameter_um"], dtype=np.float64)
            bidx = np.asarray(
                d["branch_idx"] if "branch_idx" in d.files
                else np.zeros(len(fidx)), dtype=np.int32,
            )
            types = ([str(x) for x in d["type_label"]]
                     if "type_label" in d.files else [])
            if thr2d.ndim == 2:
                j = (_best_config_col(thr2d, bidx) if col == "best"
                     else min(int(col), thr2d.shape[1] - 1))
                thr = thr2d[:, j]
                col_desc = f"col {j}" + (" (best SI)" if col == "best"
                                         else "")
            else:
                thr = thr2d
                col_desc = "col 0"
        except Exception as ex:                              # noqa: BLE001
            return f"skip: could not read {fname} ({ex})"
    # Fiber-count alignment against the bundle's cached fibers — refuse
    # to write a sweep indexed against a different fiber set.
    fpath = None
    for sub in ("designs", "configs"):
        root = proj / sub
        if root.is_dir():
            for dd in sorted(root.iterdir()):
                if (dd / "nerve_paths_fibers.npz").is_file():
                    fpath = dd / "nerve_paths_fibers.npz"
                    break
        if fpath is not None:
            break
    if fpath is None and (proj / "nerve_paths_fibers.npz").is_file():
        fpath = proj / "nerve_paths_fibers.npz"
    if fpath is None:
        return "skip: bundle has no fibers"
    n_bundle = int(len(np.load(fpath)["path_lengths"]))
    if col == "json":
        # A handful of representative validation points (dog-VNS A/B/C)
        # mapped onto the first fibres — a subset, not the whole set.
        if int(fidx.max()) >= n_bundle:
            return (f"skip: validation points ({int(fidx.max()) + 1}) "
                    f"exceed bundle fibers ({n_bundle})")
    elif len(fidx) != n_bundle:
        # Per-fibre population sweeps must align 1:1 with the bundle's
        # fibres — a count mismatch means the thresholds were computed
        # on a DIFFERENT fibre generation, so refuse rather than write
        # results indexed against the wrong fibres.
        return (f"skip: fiber-set mismatch — stored {len(fidx)} fibres, "
                f"bundle {n_bundle} (different fibre generation)")
    # Save project-wide (cid=None) so the bare `sweep_<sha>.npz` +
    # global latest.txt are what the GUI's load_latest() reads on
    # import. A cid-tagged file is written under a `latest_<cid>.txt`
    # that the project-wide loader doesn't consult.
    result = SweepResult(
        request=SweepRequest(mode="threshold"),
        fiber_indices=fidx,
        fiber_diameters_um=diam,
        fiber_branch_idx=bidx,
        fiber_type_labels=types,
        thresholds_uA=thr,
        elapsed_s=0.0,
        n_sims_total=int(len(fidx)),
    )
    try:
        _swc.save_sweep(result, proj, write_csvs=True, cid=None)
    except Exception as ex:                                  # noqa: BLE001
        return f"skip: cache write failed ({ex})"
    n_act = int(np.isfinite(thr).sum())
    return (f"ok ({len(fidx)} fibers from {fname} [{col_desc}], "
            f"{n_act} recruited ≤hi)")


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
    _sw = _ingest_stored_thresholds(proj, fid)
    print(f"    threshold cache: {_sw}", flush=True)

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
