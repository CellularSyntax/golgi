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
