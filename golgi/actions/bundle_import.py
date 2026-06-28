# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Histology bundle import — actions.

Lightweight UI flow that wraps `segmentation/bundle_import.py`:
  1. User types / browses to a directory containing the 4-TIFF
     bundle (slide + NerveMask + FascMask + ScaleMask).
  2. `do_detect_bundle_files` auto-groups the files by role.
  3. User picks scale-bar length + extrusion thickness.
  4. `do_run_bundle_import` loads, calibrates, extrudes, writes
     STLs + manifest into `<project>/uct/nerve_3d/<ts>/` in the
     same layout the µCT recon path uses — the existing import
     wizard then picks it up unchanged.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import traceback
from pathlib import Path
from typing import Callable

from golgi.segmentation import bundle_import as _bi
from golgi.segmentation import reconstruct3d as _r3d


_BUNDLE_KIND = "golgi-uct-nerve"


def register(
    state, *,
    get_active_project_dir: Callable[[], str | None] | None = None,
    do_open_import_stepper: Callable[[str], None] | None = None,
) -> dict[str, Callable]:
    """Register all dialog + run actions. Returns a dict of
    {action_name: callable}.

    `get_active_project_dir` returns the active project's
    output directory (the same value `segment_uct.register`
    receives); the bundle output STLs get written into
    `<project>/uct/nerve_3d/<ts>/`. We accept a callable, not
    a path, because the active project can change between
    dialog opens.

    `do_open_import_stepper` is the handler that opens the
    existing nerve-import drawer; if provided, the dialog's
    "Import & extrude" button opens it once the STLs are on
    disk so the user can immediately load the new bundle.
    """

    def _reset_detect_state() -> None:
        state.bundle_slide_path = ""
        state.bundle_nerve_path = ""
        state.bundle_fasc_path = ""
        state.bundle_scale_path = ""
        state.bundle_detect_error = ""
        state.bundle_pixel_pitch_um = 0.0
        state.bundle_status = ""
        state.bundle_dir_path = ""
        state.bundle_upload_files = None

    def do_open_bundle_import_dialog(*_args) -> None:
        with state:
            state.show_bundle_import_dialog = True
            _reset_detect_state()
            # Sensible defaults — 10× histology with a 1 mm
            # scale bar is the most common case.
            if not getattr(state, "bundle_dir_path", ""):
                state.bundle_dir_path = ""
            if not getattr(
                state, "bundle_scale_bar_um", 0.0,
            ):
                state.bundle_scale_bar_um = 1000.0
            if not getattr(
                state, "bundle_thickness_mm", 0.0,
            ):
                state.bundle_thickness_mm = 10.0

    def do_close_bundle_import_dialog(*_args) -> None:
        with state:
            state.show_bundle_import_dialog = False

    def do_detect_bundle_files(*_args) -> None:
        """Walk `bundle_dir_path` for TIFFs and auto-assign
        each one to its role. Populates the four
        `bundle_<role>_path` state vars + computes pixel pitch
        on the spot (so the user sees the derived value before
        committing to reconstruct)."""
        with state:
            state.bundle_detect_error = ""
            state.bundle_pixel_pitch_um = 0.0
        raw = (
            getattr(state, "bundle_dir_path", "") or ""
        ).strip()
        if not raw:
            with state:
                state.bundle_detect_error = (
                    "Set a directory path first."
                )
            return
        d = Path(raw).expanduser()
        if not d.is_dir():
            with state:
                state.bundle_detect_error = (
                    f"Not a directory: {d}"
                )
            return
        # All files (masks + slide + scale .mat); from_files tolerates the
        # extras in human packages and captures the ScaleLength.mat.
        files = sorted(p for p in d.iterdir() if p.is_file())
        n_masks = sum(1 for p in files
                      if p.suffix.lower() in (".tif", ".tiff"))
        if n_masks < 3:
            with state:
                state.bundle_detect_error = (
                    f"Need the mask TIFFs in {d.name}/, found "
                    f"{n_masks}.")
            return
        try:
            roles = _bi.BundleRoles.from_files(files)
        except ValueError as ex:
            with state:
                state.bundle_detect_error = str(ex)
            return
        # Try the scale calibration right away so the user can
        # eyeball the derived pixel pitch before committing. Human
        # packages carry the bar length in ScaleLength.mat.
        scale_um = _bi._scale_bar_um_from_mat(
            roles.scale_mat,
            float(getattr(state, "bundle_scale_bar_um", 1000.0) or 1000.0),
        )
        try:
            scale_mask = _bi._load_bool_mask(roles.scale_mask)
            calib = _bi.calibrate_pixel_pitch(
                scale_mask, scale_bar_length_um=scale_um,
            )
            pitch_um = calib.pixel_pitch_um
            n_bar_px = calib.measured_bar_length_px
            aspect = calib.aspect_ratio
        except Exception as ex:                       # noqa: BLE001
            with state:
                state.bundle_detect_error = (
                    f"Scale-bar calibration failed: {ex}"
                )
            return
        with state:
            state.bundle_slide_path = str(roles.slide)
            state.bundle_nerve_path = str(roles.nerve_mask)
            state.bundle_fasc_path = str(roles.fasc_mask)
            state.bundle_scale_path = str(roles.scale_mask)
            state.bundle_pixel_pitch_um = float(pitch_um)
            state.bundle_status = (
                f"Detected · scale bar {n_bar_px} px "
                f"(aspect {aspect:.1f}:1) · "
                f"pitch {pitch_um:.3f} µm/px"
            )

    @state.change("bundle_upload_files")
    def _on_bundle_upload(**_kw) -> None:
        """Write the user-selected bundle files into a temp dir and
        auto-detect their roles. Replaces the old paste-a-path flow:
        the user opens the bundle folder and multi-selects every file
        (slide + masks + ScaleLength.mat for human packages). Vuetify's
        VFileInput hands us a list of {name, content} dicts."""
        info = getattr(state, "bundle_upload_files", None)
        if not info:
            return
        entries = info if isinstance(info, list) else [info]
        project_dir = (
            get_active_project_dir()
            if get_active_project_dir is not None
            else None
        )
        if not project_dir:
            with state:
                state.bundle_detect_error = (
                    "No active project. Create or open a "
                    "project first."
                )
            return
        # Unique temp dir per upload so re-uploads don't collide; the
        # detect step + scale re-detect read straight back from here.
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        tmp = Path(project_dir) / "histology" / "_uploads" / ts
        tmp.mkdir(parents=True, exist_ok=True)
        n = 0
        for entry in entries:
            if not entry or "name" not in entry:
                continue
            content = entry.get("content")
            if not content:
                continue
            try:
                with open(tmp / entry["name"], "wb") as fh:
                    fh.write(content)
                n += 1
            except OSError as ex:                       # noqa: PERF203
                with state:
                    state.bundle_detect_error = (
                        f"Could not save {entry['name']}: {ex}"
                    )
                return
        if n == 0:
            return
        with state:
            state.bundle_dir_path = str(tmp)
        do_detect_bundle_files()

    async def do_run_bundle_import(*_args) -> None:
        """Load bundle, extrude meshes, write STLs + manifest
        into the project's uct/nerve_3d/<ts>/ directory. After
        success, the existing import wizard can pick up the new
        bundle directly.
        """
        with state:
            state.bundle_detect_error = ""
        if not (
            getattr(state, "bundle_nerve_path", "")
            and getattr(state, "bundle_fasc_path", "")
            and getattr(state, "bundle_scale_path", "")
            and getattr(state, "bundle_slide_path", "")
        ):
            with state:
                state.bundle_detect_error = (
                    "Detect the bundle files first."
                )
            return
        try:
            scale_um = float(
                getattr(state, "bundle_scale_bar_um", 1000.0)
                or 1000.0,
            )
            thickness_mm = float(
                getattr(state, "bundle_thickness_mm", 10.0)
                or 10.0,
            )
        except (TypeError, ValueError):
            with state:
                state.bundle_detect_error = (
                    "Scale and thickness must be numbers."
                )
            return
        if scale_um <= 0 or thickness_mm <= 0:
            with state:
                state.bundle_detect_error = (
                    "Scale and thickness must be > 0."
                )
            return
        project_dir = (
            get_active_project_dir()
            if get_active_project_dir is not None
            else None
        )
        if not project_dir:
            with state:
                state.bundle_detect_error = (
                    "No active project. Create or open a "
                    "project first."
                )
            return
        loop = asyncio.get_running_loop()
        with state:
            state.busy = True
            state.busy_msg = "Importing histology bundle…"
            state.busy_log = "Reading TIFFs…"
        state.flush()
        await asyncio.sleep(0)

        def _do_work() -> dict:
            # Re-detect from the bundle folder so human packages keep their
            # peri masks + ScaleLength.mat (a 4-path reconstruction would
            # drop them). The folder is the parent of the nerve mask.
            bdir = Path(state.bundle_nerve_path).parent
            roles = _bi.BundleRoles.from_files(
                sorted(p for p in bdir.iterdir() if p.is_file()))
            bundle = _bi.load_bundle(
                roles, scale_bar_length_um=scale_um,
            )
            meshes = _bi.extrude_bundle(
                bundle, thickness_mm=thickness_mm,
            )
            # Nerve cross-section deform, applied HERE at import so the
            # geometry written to STL (and everything downstream — render,
            # cuff fit, fibers, mesh) is consistent. "round" = area-
            # preserving circularization of the nerve + its fascicles
            # (same affine); "none" = keep the real segmented shape.
            deform = str(getattr(state, "nerve_deform", "round")).lower()
            if deform == "round":
                from golgi.segmentation.deform import round_mesh_list
                round_mesh_list(meshes)
            # M47 — histology bundles land in their OWN
            # `<project>/histology/nerve_3d/<ts>/` directory,
            # NOT alongside µCT bundles. The import wizard
            # has a dedicated "Histology bundle" tile (third
            # tile, separate from "Golgi µCT bundle") that
            # lists this subdir, so the two ingestion paths
            # never collide.
            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = (
                Path(project_dir) / "histology" / "nerve_3d"
                / ts
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            files: list[str] = []
            for m in meshes:
                stl_path = out_dir / f"{m.name}.stl"
                _r3d.write_binary_stl(m, stl_path)
                files.append(stl_path.name)
            n_fasc = sum(
                1 for fn in files
                if fn.startswith("fascicle_")
                or fn == "endoneurium.stl"
            )
            manifest = {
                "kind": _BUNDLE_KIND,
                "schema": "v1",
                "mode": "single",
                "n_fascicles": int(n_fasc),
                "voxel_xy_mm": (
                    bundle.calibration.pixel_pitch_um / 1000.0
                ),
                "voxel_z_mm": None,
                "thickness_mm": float(thickness_mm),
                "smooth_sigma": None,
                # Bundle-import provenance — distinguishes
                # these reconstructions from SAM2-driven µCT
                # reconstructions when reopening a project.
                "source": "histology_bundle",
                "histology_bundle": {
                    "slide": str(roles.slide) if roles.slide else None,
                    "nerve_mask": str(roles.nerve_mask),
                    "fasc_mask": str(roles.fasc_mask),
                    "scale_mask": str(roles.scale_mask),
                    "scale_bar_um": float(
                        bundle.calibration.scale_bar_length_um),
                    "measured_bar_length_px": int(
                        bundle.calibration
                        .measured_bar_length_px
                    ),
                    "bundle_stem": bundle.bundle_stem,
                    # human packages: measured perineurium (PeriExt−PeriInner)
                    "species": "human" if bundle.human else "swine",
                    "peri_inner_mask": (str(roles.peri_inner)
                                        if roles.peri_inner else None),
                    "peri_ext_mask": (str(roles.peri_ext)
                                      if roles.peri_ext else None),
                    "measured_peri_thk_um": bundle.peri_thk_um,
                },
                "slice_idx": None,
                "slice_range": None,
                "annotated_slices": [],
                "files": files,
            }
            with open(out_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            return {
                "out_dir": out_dir,
                # Bundle id = timestamp subdir name, the same key
                # the import wizard's `uct_bundle_items` uses
                # (via list_bundles' "id" field).
                "bundle_id": out_dir.name,
                "n_meshes": len(meshes),
                "n_fasc": n_fasc,
                "pitch_um": bundle.calibration.pixel_pitch_um,
            }

        try:
            result = await loop.run_in_executor(None, _do_work)
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.busy = False
                state.bundle_detect_error = (
                    f"{type(ex).__name__}: {ex}"
                )
                state.bundle_status = ""
            return

        with state:
            state.busy = False
            state.bundle_status = (
                f"Wrote {result['n_meshes']} STLs to "
                f"{result['out_dir'].name} · "
                f"{result['n_fasc']} fascicle"
                f"{'s' if result['n_fasc'] != 1 else ''} · "
                f"pitch {result['pitch_um']:.3f} µm/px"
            )
            state.show_bundle_import_dialog = False
        if do_open_import_stepper is not None:
            do_open_import_stepper(result["bundle_id"])

    return {
        "do_open_bundle_import_dialog": (
            do_open_bundle_import_dialog
        ),
        "do_close_bundle_import_dialog": (
            do_close_bundle_import_dialog
        ),
        "do_detect_bundle_files": do_detect_bundle_files,
        "do_run_bundle_import": do_run_bundle_import,
    }
