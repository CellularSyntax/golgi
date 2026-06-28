# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Per-design disk-layout helpers (F3.2a).

In F3.2a each electrode design owns its own physical cuff
placement AND its own multi-domain FEM mesh — the prior F3.1
assumption of a single shared `nerve.msh` was wrong, because the
mesh embeds the cuff silicone shell as a region and that shell
moves / resizes per design.

Disk layout per project:

  <out>/
    project.json                       manifest
    conductivities.json                σ values (shared)
    source/<orig>.{stl,obj,nas,...}    original geometry import
    nerve_paths_fibers_raw.npz         raw nerve-frame fibers
    nerve_paths_caps.json              fiber-segment caps
    thumbnail.png                      project tile
    designs/<eid>/                     per-electrode-design folder
      manifest.json                    physical placement +
                                       cuff dims + electrode type
      nerve.msh                        multi-domain TetGen mesh
      mesh_config.json                 σ + slice extents snapshot
      electrode_config.json            patch geometry + polarity
      nerve_surface_pts.npz            cuff-frame endo vertices
      nerve_paths_fibers.npz           cuff-frame fiber paths
      axis_line.npz                    FEM outputs ↓
      slice_volume.npz
      paths_Ve.npz
      nerve_surface_Ve.npz
      Ve.{xdmf,h5}
      E.{xdmf,h5}
      fem_results.npz                  (legacy single-axis dump)
      fem_results.npy
    sims/<eid>/                        per-design sim caches
      fiber_sim_results.pkl
      pop_state.pkl
    sweeps/                            sweep caches (not yet per-
                                       design; see F3.2 plan note)

Back-compat: legacy single-cuff projects that pre-date F3.2a
keep the FEM outputs at the project root. `enumerate_designs`
synthesises a single `default` design that points back at the
root via `design_dir`, so they keep rendering without a
migration step.

Public surface:

  * `design_dir(out_dir, design_id)` — canonical path to a
    design's folder. Returns `<out>/designs/<id>/` for new
    layouts; falls back to `out_dir` for the legacy `default`
    design when a pre-F3.2a flat layout is detected.

  * `enumerate_designs(out_dir)` — walk the designs/ subdir (or
    detect a legacy flat layout). Each entry: { id, name,
    n_patches, I_stim_mA, has_mesh, has_fem, sha256 }.

  * `write_design_manifest(out_dir, design_id, manifest)` —
    persist a design's manifest.json.

  * `read_design_manifest(out_dir, design_id)` — read it back.

  * `sim_dir(out_dir, design_id)` — per-design sim cache dir.

  * `safe_design_id(value)` — sanitise an electrode id / display
    name into a POSIX-safe folder name.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# FEM output files that live INSIDE a design folder.
DESIGN_FEM_OUTPUT_FILES: tuple[str, ...] = (
    "axis_line.npz",
    "slice_volume.npz",
    "paths_Ve.npz",
    "nerve_surface_Ve.npz",
    "Ve.xdmf",
    "Ve.h5",
    "E.xdmf",
    "E.h5",
    "fem_results.npz",
    "fem_results.npy",
)

# Mesh-tier files that ALSO live inside the design folder (each
# design has its own multi-domain mesh).
DESIGN_MESH_FILES: tuple[str, ...] = (
    "nerve.msh",
    "nerve_surface_pts.npz",
    "nerve_paths_fibers.npz",
)

# Everything that should be hashed when fingerprinting a design.
DESIGN_OUTPUT_FILES: tuple[str, ...] = (
    DESIGN_MESH_FILES + DESIGN_FEM_OUTPUT_FILES
)


def safe_design_id(value: str) -> str:
    """Sanitise a string to a POSIX-friendly folder name.
    Lowercased, non-alphanumeric → `_`, leading/trailing `_`
    stripped. Empty input falls back to `design`."""
    cleaned = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in (value or "")
    ).strip("_").lower()
    return cleaned or "design"


def has_legacy_flat_layout(out_dir: Path) -> bool:
    """True iff a pre-F3.2a project layout is present at the
    project root (nerve.msh OR paths_Ve.npz at the root). Used to
    decide whether to enumerate as a single `default` design."""
    out_dir = Path(out_dir)
    return (
        (out_dir / "nerve.msh").is_file()
        or (out_dir / "paths_Ve.npz").is_file()
    )


def design_dir(out_dir: Path, design_id: str) -> Path:
    """Return the directory holding a design's mesh + FEM outputs.

    For a new F3.2a+ layout: `<out_dir>/designs/<design_id>/`.
    For a legacy flat layout AND design_id == 'default': return
    `out_dir` itself so consumer code doesn't have to special-
    case missing subdirs. Any other design_id always points into
    the new subdir layout (even if it doesn't exist yet)."""
    out_dir = Path(out_dir)
    new_dir = out_dir / "designs" / design_id
    if new_dir.is_dir():
        return new_dir
    if design_id == "default" and has_legacy_flat_layout(out_dir):
        return out_dir
    return new_dir


def config_dir(out_dir: Path, config_id: str) -> Path:
    """Return the directory holding a config's FEM outputs (F3.2c).
    Configs are children of designs: one cuff hardware (= one
    mesh under designs/<eid>/) can carry many polarity wirings,
    each of which gets its own FEM solve under
    `<out_dir>/configs/<config_id>/`.

    For a legacy flat layout AND config_id == 'default': return
    `out_dir` itself so pre-F3.2c projects keep rendering."""
    out_dir = Path(out_dir)
    new_dir = out_dir / "configs" / config_id
    if new_dir.is_dir():
        return new_dir
    if config_id == "default" and has_legacy_flat_layout(out_dir):
        return out_dir
    return new_dir


def write_config_manifest(
    out_dir: Path,
    config_id: str,
    manifest: dict,
) -> Path:
    """Persist a config's manifest.json under
    `<out>/configs/<config_id>/`. Creates the directory if
    needed. Returns the written path."""
    out_dir = Path(out_dir)
    d = out_dir / "configs" / config_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    p.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return p


def read_config_manifest(
    out_dir: Path,
    config_id: str,
) -> dict:
    """Read a config's manifest.json. Returns an empty dict when
    the file is missing or unparseable."""
    out_dir = Path(out_dir)
    p = out_dir / "configs" / config_id / "manifest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def enumerate_configs(out_dir: Path) -> list[dict]:
    """Return the list of solved contact-config FEM outputs under
    `<out>/configs/<cid>/`. Each entry surfaces the manifest
    fields needed by the analysis chip switcher: id, name,
    design_id, design_name, n_patches, I_stim_mA, has_fem (does
    paths_Ve.npz exist?), sha256.

    Empty list when no configs/ subdir is present — caller falls
    through to the F3.2a per-design layout or the legacy flat
    root for back-compat."""
    out_dir = Path(out_dir)
    configs_root = out_dir / "configs"
    out: list[dict] = []
    if not configs_root.is_dir():
        return out
    for sub in sorted(p for p in configs_root.iterdir()
                      if p.is_dir()):
        m = read_config_manifest(out_dir, sub.name)
        out.append({
            "id": sub.name,
            "name": str(m.get("name", sub.name)),
            "design_id": str(m.get("design_id", "")),
            "design_name": str(m.get("design_name", "")),
            "n_patches": int(m.get("n_patches", 0) or 0),
            "I_stim_mA": float(
                m.get("I_stim_mA", 0.0) or 0.0,
            ),
            "has_fem": (sub / "paths_Ve.npz").is_file(),
            "sha256": str(m.get("sha256", "")),
        })
    return out


def recording_dir(out_dir: Path, design_id: str) -> Path:
    """R1.2 — return the recording-basis directory for a design.
    Per-contact lead fields (V_e_rec_<contact_id>.npz) and the
    cache manifest live here. Because the lead field depends only
    on geometry + σ (not on stim wiring), it's per-DESIGN, not
    per-config — shared across every config bound to the design.

    Path: `<out_dir>/designs/<design_id>/recording/`. For the
    legacy flat layout (design_id == 'default' + project-root
    nerve.msh), returns `<out_dir>/recording/`."""
    out_dir = Path(out_dir)
    if design_id == "default" and has_legacy_flat_layout(out_dir):
        return out_dir / "recording"
    return out_dir / "designs" / design_id / "recording"


def sim_dir(out_dir: Path, key: str) -> Path:
    """Return the per-key directory for simulation outputs
    (`fiber_sim_results.pkl`, `pop_state.pkl`, future per-key
    sweep caches). The `key` is a CONFIG cid in the F3.2c
    layout (one sim cache per polarity wiring) — pre-F3.2c
    callers passed a design eid; both reuse this same routing
    because the on-disk shape is the same.

      * New layout: `<out_dir>/sims/<key>/` — created lazily
        when something is written to it.
      * Legacy fallback: when `key == "default"` AND a legacy
        `fiber_sim_results.pkl` or `pop_state.pkl` lives at
        the project root, return `out_dir` itself. Keeps
        pre-F3.2a projects readable without migration."""
    out_dir = Path(out_dir)
    new_dir = out_dir / "sims" / key
    if new_dir.is_dir():
        return new_dir
    if key == "default":
        if (
            (out_dir / "fiber_sim_results.pkl").is_file()
            or (out_dir / "pop_state.pkl").is_file()
        ):
            return out_dir
    return new_dir


def write_design_manifest(
    out_dir: Path,
    design_id: str,
    manifest: dict,
) -> Path:
    """Persist a design's manifest.json under
    `<out>/designs/<design_id>/`. Creates the directory if
    needed. Returns the written path."""
    out_dir = Path(out_dir)
    d = out_dir / "designs" / design_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    p.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return p


def read_design_manifest(
    out_dir: Path,
    design_id: str,
) -> dict:
    """Read a design's manifest.json. Returns an empty dict when
    the file is missing or unparseable."""
    out_dir = Path(out_dir)
    p = out_dir / "designs" / design_id / "manifest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def _legacy_default_manifest(out_dir: Path) -> dict[str, Any]:
    """Synthesise a manifest for a legacy flat-layout project so
    the rest of the system can treat it as a single design."""
    out_dir = Path(out_dir)
    ec_path = out_dir / "electrode_config.json"
    try:
        ec = (
            json.loads(ec_path.read_text(encoding="utf-8"))
            if ec_path.is_file() else {}
        )
    except Exception:                                    # noqa: BLE001
        ec = {}
    return {
        "id": "default",
        "name": str(ec.get("name", "default")),
        "n_patches": len(ec.get("patches", []) or []),
        "I_stim_mA": float(ec.get("I_stim", 0.0)) * 1e3,
        "has_mesh": (out_dir / "nerve.msh").is_file(),
        "has_fem": (out_dir / "paths_Ve.npz").is_file(),
        "sha256": "",
    }


def enumerate_designs(out_dir: Path) -> list[dict[str, Any]]:
    """Return the list of physical-electrode designs for the
    project under `out_dir`. Each entry has the keys: id, name,
    n_patches, I_stim_mA, has_mesh, has_fem, sha256.

    Order of resolution:
      1. `<out>/designs/<id>/manifest.json` × N (F3.2a+ layout)
      2. Legacy flat layout — single `default` design pointed
         at the project root.
      3. Nothing → empty list."""
    out_dir = Path(out_dir)
    designs_root = out_dir / "designs"
    out: list[dict[str, Any]] = []
    if designs_root.is_dir():
        for sub in sorted(p for p in designs_root.iterdir()
                          if p.is_dir()):
            m = read_design_manifest(out_dir, sub.name)
            # Tolerate missing manifest — surface what we can
            # from the on-disk artefacts so the user can at least
            # see "this design folder exists" in the picker.
            entry = {
                "id": sub.name,
                "name": str(m.get("name", sub.name)),
                "n_patches": int(m.get("n_patches", 0) or 0),
                "I_stim_mA": float(m.get("I_stim_mA", 0.0) or 0.0),
                "has_mesh": (sub / "nerve.msh").is_file(),
                "has_fem": (sub / "paths_Ve.npz").is_file(),
                "sha256": "",
            }
            out.append(entry)
        if out:
            return out
    # Legacy fallback.
    if has_legacy_flat_layout(out_dir):
        return [_legacy_default_manifest(out_dir)]
    return []


def design_sha256(design_dir_path: Path) -> str:
    """Compute a stable sha256 over a design's output files
    (mesh + FEM). Used to fingerprint a solve so a follow-up
    replay can detect divergence stage-by-stage."""
    if not design_dir_path.is_dir():
        return ""
    h = hashlib.sha256()
    for name in DESIGN_OUTPUT_FILES:
        p = design_dir_path / name
        if not p.is_file():
            continue
        try:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except Exception:                                # noqa: BLE001
            continue
    return h.hexdigest()


# ---------------------------------------------------------------
# F3.1 legacy compatibility shims — these names were referenced
# by the now-defunct "shared mesh, per-config FEM" prototype.
# Kept as thin wrappers so existing imports don't break while
# the rest of the codebase migrates to the new helpers.
# ---------------------------------------------------------------

def fem_design_dir(out_dir: Path, design_id: str) -> Path:
    """Deprecated alias for `design_dir` (F3.1 → F3.2a rename)."""
    return design_dir(out_dir, design_id)


def sim_design_dir(out_dir: Path, design_id: str) -> Path:
    """Deprecated alias for `sim_dir` (F3.1 → F3.2a rename)."""
    return sim_dir(out_dir, design_id)
