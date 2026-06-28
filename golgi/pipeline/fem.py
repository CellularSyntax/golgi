# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""FEM-solve pipeline driver.

Owns the do_solve_fem flow:
  1. Build mesh_config.json (σ values + axis/slice extents)
  2. Build electrode_config.json (per-patch role/geometry, with
     DUKE-conductor patch synthesis for designer cuffs)
  3. Write nerve_surface_pts.npz (cuff-frame endo vertices that
     solve_nerve.py samples Vₑ at)
  4. Ensure fiber paths are in cuff frame (raw→cuff transform
     via the shared _frames helper) so per-fiber Vₑ sampling
     hits the FEM mesh
  5. Spawn solve_nerve.py via FEMRunner; heartbeat on idle so
     the lightbox stays alive during long solves
  6. Load axis_line.npz + slice_volume.npz back into geom;
     split paths_Ve.npz into per-fiber arrays
  7. _refresh_fem_plots; auto-enable the Vₑ / field-line
     overlays; autosave with thumbnail

PipelineContext carries the closure hooks (stamp_user_line,
register_subprocess, was_cancelled, clear_subprocess, autosave,
safe_update, scene.request_render) + helpers SimpleNamespace
(bundling the module-level callables that still live in
golgi.py: build_electrode_patches_dicts, transform_to_cuff_frame,
_cuff_ns_extras, _ensure_polarities, _refresh_fem_plots,
DUKE_ELECTRODE_TYPE, DEFAULT_ELECTRODE, _CUFF_PRESETS, cuff_designer,
active_project).
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from golgi.jobs import CancelToken, JobRequest, LocalSubprocessRunner
from golgi.jobs.schemas import (
    ElectrodeConfig, ElectrodePatch, MeshConfig,
    RecordingMontage,
)
from ._throttle import FlushThrottle
from .context import PipelineContext
from .fem_layout import (
    config_dir as config_dir_fn,
    design_dir as design_dir_fn,
    design_sha256,
    safe_design_id,
    write_config_manifest,
    write_design_manifest,
)

_SOLVE_NERVE_PATH = (
    Path(__file__).resolve().parent.parent
    / "compute" / "solve_nerve.py"
)


@dataclass
class FEMJobRequest(JobRequest):
    """solve_nerve.py reads its config via env vars + files in
    cwd. No payload file.

      solve_out_dir — SOLVE_OUT_DIR (writes go here +
                      mesh_config.json and electrode_config.json
                      are read from here)
      mesh_input_dir — SOLVE_SHARED_DIR (nerve.msh + cuff-frame
                       fibers + endo surface read from here). When
                       None, falls back to solve_out_dir (F3.2a
                       single-design flow).
      preset       — passed as `--preset X` on the script CLI.
                     None → script picks from MeshConfig, then
                     its own DEFAULT_PRESET ("Balanced").
      mpi_cores    — when > 1, the runner spawns `mpirun -n N`
                     instead of plain `python`. When == 1 the
                     plain-Python path is kept so a dev env
                     without mpirun still works."""
    solve_out_dir: Path
    cwd: Path
    mesh_input_dir: Path | None = None
    preset: str | None = None
    mpi_cores: int = 1
    # I1 Phase A — when True, solve_nerve.py runs an additional N
    # Dirichlet dual-solves (one per contact: V=1 on that contact's
    # facets, V=0 on the outer muscle bbox boundary) and integrates
    # σ·∇V·n over each contact's surface to compute DC impedance.
    # Per-pair impedance is derived from the user's configured
    # anode/cathode pairs in contact_polarities. Output:
    # `<solve_out_dir>/impedance.json`. ~N+M extra solves per FEM
    # run; for typical 2-4 contact cuffs the overhead is seconds.
    emit_impedance: bool = True
    # R1.2 — when True AND electrode_config.json carries non-empty
    # recording_montages, solve_nerve.py runs one Laplace solve per
    # unique contact referenced by a montage: 1 A injected at the
    # contact (Neumann) + V=0 on the outer muscle bbox (Dirichlet),
    # all other contacts insulating. Output: per-contact
    # V_e_rec_<id>.npz + manifest.json under
    # `<mesh_input_dir>/recording/`. The per-design dir is reused
    # because the lead field is independent of stim wiring — the
    # same basis serves every config bound to this design. Cached
    # by (mesh mtime+size, σ-per-tag, patch geometry).
    emit_recording: bool = False


class FEMRunner(LocalSubprocessRunner):
    """LocalSubprocessRunner specialised for solve_nerve.py.
    No payload file; env var + cwd + CLI flags."""

    def _build_env(self, req: FEMJobRequest) -> dict:
        env = {"SOLVE_OUT_DIR": str(req.solve_out_dir)}
        if req.mesh_input_dir is not None:
            env["SOLVE_SHARED_DIR"] = str(req.mesh_input_dir)
        # I1 Phase A — solve_nerve.py reads this env var to gate
        # the per-contact impedance dual-solve loop.
        env["GOLGI_EMIT_IMPEDANCE"] = (
            "1" if req.emit_impedance else "0"
        )
        # R1.2 — gates the per-contact reciprocity solves that
        # produce the recording lead fields under
        # `<mesh_input_dir>/recording/`.
        env["GOLGI_EMIT_RECORDING"] = (
            "1" if req.emit_recording else "0"
        )
        # macOS mpich + libfabric tries to use the en0 NIC for OFI
        # transport even on single-rank, no-network runs and
        # crashes in MPI_Finalize when it can't find it ("OFI call
        # tsenddata failed (default nic=en0: No such file or
        # directory)"). The solve completes; only the process exit
        # is dirty. Pin the provider to tcp so finalize doesn't
        # touch libfabric's networking layer. Caller can override
        # via GOLGI_FI_PROVIDER if they're on a real cluster.
        import os as _os
        env.setdefault(
            "FI_PROVIDER",
            _os.environ.get("GOLGI_FI_PROVIDER", "tcp"),
        )
        return env

    def _build_cwd(self, req: FEMJobRequest) -> Path:
        return req.cwd

    def _build_argv(self, req: FEMJobRequest) -> list[str]:
        """Spawn under `mpirun -n N` when N > 1; plain Python
        otherwise. The mpirun binary can be overridden via the
        `GOLGI_MPIRUN` env var (default `mpirun`) so cluster
        environments can swap in `mpiexec` or `srun`. Appends
        `--preset` and `--cores` to the script's argv."""
        import os
        import sys
        cores = max(1, int(req.mpi_cores))
        if cores > 1:
            mpi_bin = os.environ.get("GOLGI_MPIRUN", "mpirun")
            prefix = [mpi_bin, "-n", str(cores)]
        else:
            prefix = []
        argv = prefix + [sys.executable, "-u", str(self.script_path)]
        if req.preset:
            argv += ["--preset", str(req.preset)]
        argv += ["--cores", str(cores)]
        return argv


async def run_fem_solve(ctx: PipelineContext) -> None:
    """Full FEM-solve driver. See module docstring for stages."""
    state = ctx.state
    geom = ctx.geom
    H = ctx.helpers

    if not state.has_mesh or geom.msh_path is None:
        state.fem_status = "Build a mesh first."
        return
    state.fem_failed = False
    state.fem_log = ""
    state.busy_log = ""
    state.busy = True
    state.busy_msg = "Solving FEM"
    state.flush()

    loop = asyncio.get_event_loop()
    log_lines: list[str] = []
    out_dir = Path(H.active_project().out_dir)
    # Trailing-edge debounce on state.flush() to cap WS traffic
    # during dense solver output (ksp_view + ksp_monitor bursts).
    throttle = FlushThrottle(loop=loop, state=state)

    def _on_line(line: str):
        line = ctx.stamp_user_line(line)
        print(f"[fem] {line}", flush=True)
        line = line[:220]
        log_lines.append(line)
        tail = "\n".join(log_lines[-14:])

        def _push():
            state.busy_log = tail
            throttle.tick()
        try:
            loop.call_soon_threadsafe(_push)
        except RuntimeError:
            pass

    def _surface_failure(msg: str) -> None:
        state.fem_log = "\n".join(log_lines[-80:])
        state.fem_failed = True
        state.fem_status = msg

    try:
        # ---- 1. mesh_config.json ----
        # F3.2c fix: build a PER-DESIGN MeshConfig inside the
        # per-config loop further down. The previous version
        # built mesh_cfg ONCE from geom.R_ci / state.L_cuff_mm,
        # which only matched whichever design was actively
        # loaded — every other config's solve got the wrong
        # R_cuff_inner and solve_nerve's ±5% radial facet filter
        # then rejected every cuff-wall facet → "No facets
        # matched any patch".
        #
        # The active-design fallback values below are still
        # written to the project root as a back-compat snapshot
        # but the per-config code in the loop below uses each
        # design's own R_ci_m / R_co_m / L_cuff_mm.
        L_cuff_m = float(state.L_cuff_mm) * 1e-3
        R_ci_m = float(geom.R_ci or 0.0)
        R_co_m = float(geom.R_co or 0.0)
        pts = geom.pts_cuff
        z_lo = (float(pts[:, 2].min())
                if pts is not None else -L_cuff_m * 2.5)
        z_hi = (float(pts[:, 2].max())
                if pts is not None else +L_cuff_m * 2.5)
        r_max = (float(np.linalg.norm(pts[:, :2], axis=1).max())
                  if pts is not None else R_co_m * 1.5)
        slice_xy_half_m = max(
            R_co_m + float(state.muscle_radial_pad_mm) * 1e-3,
            r_max * 1.05,
        )
        _preset = str(state.fem_preset or "Balanced")

        # F3.2 — per-batch canonical-frame inputs (one-shot per
        # solve). Each design's on-disk mesh is in ITS OWN cuff-
        # local frame, so per-design fiber paths + surface points
        # must also be in that design's cuff-local before
        # solve_nerve.py samples Vₑ at them. The transforms are
        # derived inside the loop from each parent design's M_D
        # and cuff_origin_D_pca; we only need the PCA points +
        # the anchor origin to compute design_offset_canon.
        pts_pca_all = None
        anchor_origin_pca = None
        if (geom.nerve is not None
                and geom.centroid is not None
                and geom.R_global is not None):
            from golgi.scene.cuff_fit import (
                anchor_origin_pca_for_designs,
            )
            pts_pca_all = (
                (geom.nerve["pts_raw"] - geom.centroid)
                @ geom.R_global
            )
            anchor_origin_pca = anchor_origin_pca_for_designs(
                pts_pca_all, list(state.designs or []),
                state.cuff_anchor,
            )

        def _build_mesh_cfg(
            design: dict,
        ) -> MeshConfig:
            """Per-design MeshConfig. R_cuff_inner / R_co /
            L_cuff come from the design dict (refit values);
            axis-sampling extents and slice xy-half are derived
            from L_cuff with a 5×-margin so axis_line covers the
            full nerve segment around the cuff. σ values are
            global. solver_preset is global too.

            F3.2 per-design-local: each design's on-disk mesh is
            in its OWN cuff-local frame (cuff axis-aligned at
            origin), so solve_nerve.py needs NO cuff transform —
            the legacy single-cuff path is correct. The
            cuff_offset_m / cuff_R_flat MeshConfig fields are
            therefore unused; we leave them None so consumers
            fall back to identity."""
            d_L_m = float(
                design.get("L_cuff_mm", state.L_cuff_mm),
            ) * 1e-3
            d_R_ci = float(design.get("R_ci_m") or R_ci_m)
            d_R_co = float(design.get("R_co_m") or R_co_m)
            d_slice_xy = max(
                d_R_co
                + float(state.muscle_radial_pad_mm) * 1e-3,
                d_R_co * 2.0,
            )
            return MeshConfig(
                mode="imported",
                R_cuff_inner=d_R_ci,
                L_cuff=d_L_m,
                axis_z_lo_m=-d_L_m * 2.5,
                axis_z_hi_m=+d_L_m * 2.5,
                slice_xy_half_m=d_slice_xy,
                sigma_endo=float(state.sigma_endo),
                sigma_saline=float(state.sigma_saline),
                sigma_silicone=float(state.sigma_silicone),
                sigma_muscle=float(state.sigma_muscle),
                sigma_epi=float(state.sigma_epi),
                sigma_contact=float(state.sigma_contact),
                sigma_scar=float(state.sigma_scar),
                perineurium_ci=bool(getattr(state, "perineurium_ci", False)),
                peri_thk_m=getattr(state, "peri_thk_m", None),
                perineurium_species=getattr(state, "perineurium_species", None),
                sigma_peri=getattr(state, "sigma_peri", None),
                solver_preset=_preset,
            )

        # Project-root snapshot (back-compat: pre-F3.2c flat
        # layout reads mesh_config.json from project root).
        mesh_cfg = MeshConfig(
            mode="imported",
            R_cuff_inner=R_ci_m,
            L_cuff=L_cuff_m,
            axis_z_lo_m=z_lo,
            axis_z_hi_m=z_hi,
            slice_xy_half_m=slice_xy_half_m,
            sigma_endo=float(state.sigma_endo),
            sigma_saline=float(state.sigma_saline),
            sigma_silicone=float(state.sigma_silicone),
            sigma_muscle=float(state.sigma_muscle),
            sigma_epi=float(state.sigma_epi),
            sigma_contact=float(state.sigma_contact),
            sigma_scar=float(state.sigma_scar),
            perineurium_ci=bool(getattr(state, "perineurium_ci", False)),
            peri_thk_m=getattr(state, "peri_thk_m", None),
            perineurium_species=getattr(state, "perineurium_species", None),
            sigma_peri=getattr(state, "sigma_peri", None),
            solver_preset=_preset,
        )
        (out_dir / "mesh_config.json").write_text(
            json.dumps(mesh_cfg.serialize(), indent=2),
        )

        # ---- 2. electrode_config.json ----
        # Patch enumeration:
        # * For parametric electrode types the existing helper
        #   `build_electrode_patches_dicts` returns axial /
        #   helical descriptors that solve_nerve.py matches to
        #   inner-wall facets.
        # * For the DUKE-designer cuff, conductors are arbitrary
        #   3D meshes. _duke_conductor_patches approximates each
        #   by its z-bbox + φ-bbox on the cuff inner wall,
        #   emitting an `axial` descriptor.
        def _duke_conductor_patches(elec: dict) -> list[dict]:
            preset_name = str(
                elec.get("duke_preset", "") or "",
            )
            preset = H._CUFF_PRESETS.get(preset_name, None)
            if preset is None:
                return []
            overrides_si = {
                str(k): float(v)
                for k, v in (
                    elec.get("duke_overrides", {}) or {}
                ).items()
            }
            try:
                # F3.2c fix: render the DUKE preset against THIS
                # design's refit R_ci, not whatever's currently
                # loaded into geom. Without this, sweep clones
                # whose R_ci differs from the loaded mesh's R_ci
                # get contact patches whose (z, phi) bounds don't
                # match the cuff inner wall facets → "No facets
                # matched any patch" failures at FEM time.
                _elec_R_ci = elec.get("R_ci_m")
                parts = H.cuff_designer.render_design(
                    preset,
                    param_overrides=overrides_si,
                    ns_extras=H.cuff_ns_extras(
                        r_nerve_m=(
                            float(_elec_R_ci)
                            if _elec_R_ci else None
                        ),
                    ),
                )
            except Exception as ex:
                _on_line(
                    f"⚠ cuff_designer render failed for "
                    f"{elec.get('eid', '')}: {ex}",
                )
                return []
            pols = H.ensure_polarities(elec)
            fractions = list(
                elec.get("contact_current_fractions") or [],
            )
            out: list[dict] = []
            cond_idx = 0
            for _inst, _sub, mesh, role in parts:
                if role != "conductor":
                    continue
                pol = (pols[cond_idx]
                       if cond_idx < len(pols) else "off")
                this_frac: Optional[float] = (
                    float(fractions[cond_idx])
                    if (cond_idx < len(fractions)
                        and fractions[cond_idx] is not None)
                    else None
                )
                cond_idx += 1
                if pol == "off":
                    continue
                # M1: emit the polarity as-is into the patch
                # `role` field; solve_nerve.py recognises
                # "anode" / "cathode" / "ground" directly +
                # uses the per-patch current_fraction to weight
                # multi-contact configurations.
                patch_role = pol
                pts_m = np.asarray(
                    mesh.points, dtype=np.float64,
                )
                if pts_m.size == 0:
                    continue
                z_min = float(pts_m[:, 2].min())
                z_max = float(pts_m[:, 2].max())
                z_center = 0.5 * (z_min + z_max)
                dz = max(z_max - z_min, 1.0e-6)
                xy_cen = pts_m[:, :2].mean(axis=0)
                if float(np.linalg.norm(xy_cen)) < 1.0e-9:
                    phi_center = 0.0
                    dphi = 2.0 * math.pi
                else:
                    phi_c = math.atan2(
                        float(xy_cen[1]), float(xy_cen[0]),
                    )
                    phi_pts = np.arctan2(
                        pts_m[:, 1], pts_m[:, 0],
                    )
                    rel = (
                        (phi_pts - phi_c + math.pi)
                        % (2.0 * math.pi) - math.pi
                    )
                    rel_min = float(rel.min())
                    rel_max = float(rel.max())
                    phi_center = phi_c + 0.5 * (
                        rel_min + rel_max
                    )
                    dphi = max(rel_max - rel_min, 1.0e-6)
                    if dphi > 2.0 * math.pi - 1.0e-6:
                        dphi = 2.0 * math.pi
                patch_dict: dict = {
                    "id": len(out),
                    "type": "axial",
                    "role": patch_role,
                    "z": float(z_center),
                    "dz": float(dz),
                    "phi": float(phi_center),
                    "dphi": float(dphi),
                }
                if this_frac is not None:
                    patch_dict["current_fraction"] = this_frac
                out.append(patch_dict)
            return out

        def _build_patches_for_elec(
            elec: dict | None,
        ) -> tuple[str, list[dict]]:
            """Return (electrode_type_name, patches) for one
            electrode dict. Mirrors the legacy single-target path:
            DUKE designer cuffs use the _duke_conductor_patches
            synthesiser; parametric kinds use
            build_electrode_patches_dicts. With `elec=None` returns
            an empty list (caller treats as a hard failure)."""
            if elec is None:
                return str(state.electrode_type), []
            cfg_name_ = str(elec.get(
                "electrode_type", state.electrode_type,
            ))
            if cfg_name_ == H.DUKE_ELECTRODE_TYPE:
                return cfg_name_, _duke_conductor_patches(elec)
            _elec_L_m = float(elec.get(
                "L_cuff_mm", state.L_cuff_mm,
            )) * 1e-3
            _elec_R_ci = float(
                elec.get("R_ci_m") or R_ci_m,
            )
            _elec_cfg = {
                k: elec.get(k, H.DEFAULT_ELECTRODE[k])
                for k in H.DEFAULT_ELECTRODE
                if k != "electrode_type"
            }
            return cfg_name_, H.build_electrode_patches_dicts(
                _elec_L_m, _elec_R_ci,
                kind=cfg_name_, cfg=_elec_cfg,
            )

        # F3.2c — enumerate the CONFIGS we'll solve. Each config
        # is a polarity/current-fraction snapshot bound to one
        # design; the design owns the mesh, the config owns the
        # solve. Source of truth:
        #   1. state.solve_config_selection (multi-select in the
        #      Solve tab) — list of cids to solve.
        #   2. fallback: the currently-active config
        #      (state.selected_config_id).
        # Each (config, design) pair becomes one solve_nerve.py
        # invocation reading the design's mesh and writing into
        # the config's own subdir.
        all_configs = list(state.configs or [])
        all_designs = list(state.designs or [])
        designs_by_eid = {
            str(d.get("eid", "")): d for d in all_designs
        }
        picked_cids: list[str] = [
            str(c) for c in
            (state.solve_config_selection or [])
        ]
        if not picked_cids:
            sel_cid = str(state.selected_config_id or "")
            if sel_cid:
                picked_cids = [sel_cid]
        if not picked_cids:
            _surface_failure(
                "⚠ no config selected to solve. Open the Designs "
                "drawer, save at least one configuration, and "
                "either select it or tick it in the Solve tab's "
                "multi-select.",
            )
            return
        configs_plan: list[tuple[str, dict, dict]] = []
        for cid in picked_cids:
            cfg = next(
                (c for c in all_configs if c.get("cid") == cid),
                None,
            )
            if cfg is None:
                _on_line(
                    f"⚠ config '{cid}' not found in "
                    f"state.configs — skipping"
                )
                continue
            parent_eid = str(cfg.get("design_id", ""))
            parent = designs_by_eid.get(parent_eid)
            if parent is None:
                _on_line(
                    f"⚠ config '{cid}' references unknown "
                    f"design '{parent_eid}' — skipping"
                )
                continue
            configs_plan.append((cid, cfg, parent))
        if not configs_plan:
            _surface_failure(
                "⚠ none of the selected configs can be solved "
                "(missing parents?) — see log"
            )
            return

        # ---- 3. Run solve_nerve.py once per design (F3.2a) ----
        # Each electrode in state.designs has its own multi-domain
        # mesh under <out>/designs/<eid>/nerve.msh (built by the
        # mesh pipeline). For each design we now also write the
        # solve's per-design inputs into the SAME subdir:
        #   electrode_config.json — patches + polarities
        #   mesh_config.json      — σ values + slice extents
        #   nerve_surface_pts.npz — endoneurium vertices, in this
        #                           design's local cuff frame
        #   nerve_paths_fibers.npz — fiber polylines, in this
        #                            design's local cuff frame
        # solve_nerve.py then reads everything from SOLVE_OUT_DIR
        # with no SOLVE_SHARED_DIR — the F3.1 "shared mesh"
        # assumption is gone in F3.2a.
        runner = FEMRunner(_SOLVE_NERVE_PATH)
        # Raw nerve-frame fiber paths used as the source for the
        # per-design transforms below. `geom.fiber_paths_raw` is
        # populated by the fiber-generation pipeline; under F3.2a
        # we never mutate it in-place (the old
        # ensure_fibers_in_cuff_frame helper was non-idempotent
        # across designs).
        _raw_fibers = (
            list(geom.fiber_paths_raw or [])
            if geom.fiber_paths_raw is not None else []
        )

        # MPI core count comes from an env var rather than the
        # UI so the deployment owner controls how aggressively
        # to parallelise (laptop = 1, beefy workstation = 8+).
        # Default 1 = plain-Python path, no mpirun dependency.
        import os as _os
        try:
            _cores = max(1, int(_os.environ.get(
                "GOLGI_FEM_CORES", "1",
            )))
        except ValueError:
            _cores = 1

        async def _heartbeat():
            import time as _time
            t0 = _time.time()
            while True:
                await asyncio.sleep(8)
                _on_line(
                    f"# … still solving "
                    f"({_time.time() - t0:.0f}s)"
                )

        # Cache of (design_eid → bool) — did we already write
        # this design's cuff-frame fiber + surface .npz files?
        # Configs that share a design also share these inputs,
        # so we only write them once per parent design.
        _shared_inputs_done: set[str] = set()
        configs_meta: list[dict] = []
        total_configs = len(configs_plan)
        for _idx, (cid, cfg, parent) in enumerate(configs_plan):
            parent_eid = str(parent.get("eid", ""))
            disp_name = str(cfg.get("name") or cid)
            parent_name = str(parent.get("name") or parent_eid)
            # Live per-config progress in the busy lightbox. The
            # heading reads "Solving config 2/5 · Default · Cuff
            # A" so the user can tell at a glance how far through
            # the batch we are. Heartbeat still streams elapsed
            # time into busy_log under the heading.
            state.busy_msg = (
                f"Solving config "
                f"{_idx + 1}/{total_configs} · "
                f"{disp_name} · {parent_name}"
            )
            state.flush()
            d_dir = design_dir_fn(out_dir, parent_eid)
            d_dir.mkdir(parents=True, exist_ok=True)
            # The design must have been meshed first — without
            # nerve.msh there's nothing to solve on.
            if not (d_dir / "nerve.msh").is_file():
                _on_line(
                    f"⚠ config '{cid}' ({disp_name}): parent "
                    f"design '{parent_eid}' has no nerve.msh "
                    f"in {d_dir} — mesh the design first"
                )
                continue

            # Build an `elec_for_solve` dict: parent design's
            # geometry + this config's polarities/fractions/
            # I_stim. The patch builders read these fields, so
            # this is what swaps the wiring without re-meshing.
            cfg_pols = list(cfg.get("contact_polarities") or [])
            cfg_fracs = list(
                cfg.get("contact_current_fractions") or [],
            )
            elec_for_solve = dict(parent)
            elec_for_solve["contact_polarities"] = cfg_pols
            elec_for_solve["contact_current_fractions"] = (
                cfg_fracs
            )
            cfg_name, patches = _build_patches_for_elec(
                elec_for_solve,
            )
            n_pat = len(patches)
            if not patches:
                _on_line(
                    f"⚠ config '{cid}' ({disp_name}): no "
                    f"electrode patches generated — at least "
                    f"one contact must have a polarity set "
                    f"(anode/cathode/ground). Skipping."
                )
                continue
            I_stim = float(
                cfg.get("I_stim_mA", state.I_stim_mA) or 0.0,
            )

            c_dir = (
                out_dir / "configs" / safe_design_id(cid)
            )
            c_dir.mkdir(parents=True, exist_ok=True)
            # R1.2 — surface this config's recording_montages into
            # electrode_config.json. compute/solve_nerve.py reads
            # them back and, when GOLGI_EMIT_RECORDING=1, runs one
            # reciprocity solve per unique contact referenced.
            _cfg_montages = list(
                cfg.get("recording_montages") or [],
            )
            _rec_montages = [
                RecordingMontage.deserialize(m)
                for m in _cfg_montages
                if isinstance(m, dict)
            ]
            elec_cfg = ElectrodeConfig(
                name=cfg_name,
                I_stim=I_stim * 1e-3,
                patches=[
                    ElectrodePatch.deserialize(p)
                    if isinstance(p, dict)
                    else p
                    for p in patches
                ],
                recording_montages=_rec_montages,
            )
            (c_dir / "electrode_config.json").write_text(
                json.dumps(elec_cfg.serialize(), indent=2),
            )
            # F3.2c fix: per-config mesh_config carries THIS
            # config's parent design's R_ci / R_co / L_cuff —
            # NOT the active design's. Without this, solve_nerve
            # rejects every cuff-wall facet by its ±5% radial
            # filter against the wrong R.
            _per_design_mesh_cfg = _build_mesh_cfg(parent)
            (c_dir / "mesh_config.json").write_text(
                json.dumps(
                    _per_design_mesh_cfg.serialize(),
                    indent=2,
                ),
            )

            # F3.2: nerve_paths_fibers.npz lives alongside the
            # MESH (in d_dir) — shared across every config bound
            # to the same parent design. Write it once per parent
            # design, in THIS DESIGN'S cuff-local frame (the
            # frame the on-disk nerve.msh is in). solve_nerve.py
            # samples Vₑ at these points directly in that frame.
            #
            # nerve_surface_pts.npz is NOT written here; mesh.py
            # writes it at mesh-build time from the freshly-built
            # region surface (the only source that always matches
            # that design's nerve.msh regardless of which design
            # is currently focused).
            if (parent_eid not in _shared_inputs_done
                    and _raw_fibers
                    and pts_pca_all is not None
                    and anchor_origin_pca is not None):
                from golgi.scene.cuff_fit import (
                    _design_M, find_cuff_origin_pca,
                )
                _M_parent = _design_M(parent)
                _cuff_origin_parent_pca = find_cuff_origin_pca(
                    pts_pca_all, state.cuff_anchor,
                    float(parent.get("cuff_offset_mm", 0.0)),
                    float(parent.get("cuff_dx_mm", 0.0)),
                    float(parent.get("cuff_dy_mm", 0.0)),
                )
                # raw → parent's cuff-local (matches the on-disk
                # nerve.msh frame written by mesh.py for this
                # parent design):
                #   p_pca = (p_raw - centroid) @ R_global
                #   p_local = (p_pca - cuff_origin_parent_pca) @ M_parent
                _centroid = np.asarray(
                    geom.centroid, dtype=np.float64,
                )
                _R_global = np.asarray(
                    geom.R_global, dtype=np.float64,
                )
                transformed = []
                for p in _raw_fibers:
                    p_arr = np.asarray(p, dtype=np.float64)
                    if p_arr.size == 0:
                        continue
                    p_pca = (p_arr - _centroid) @ _R_global
                    p_local = (
                        (p_pca - _cuff_origin_parent_pca)
                        @ _M_parent
                    )
                    transformed.append(p_local)
                if transformed:
                    flat = np.vstack(transformed)
                    lens = np.array(
                        [len(p) for p in transformed],
                        dtype=np.int64,
                    )
                    np.savez(
                        d_dir / "nerve_paths_fibers.npz",
                        paths_flat=flat,
                        path_lengths=lens,
                        frame_is_cuff=np.int8(1),
                    )
                _shared_inputs_done.add(parent_eid)
            elif parent_eid not in _shared_inputs_done:
                # Mark done so we don't keep retrying when the
                # nerve isn't loaded.
                _shared_inputs_done.add(parent_eid)
            _on_line(
                f"# [{_idx + 1}/{len(configs_plan)}] config "
                f"'{cid}' ({disp_name}) on design "
                f"'{parent_name}': {n_pat} patches "
                f"→ {c_dir}"
            )

            tok = CancelToken()
            _orig_arm = tok.arm

            def _arm_and_forward(proc, _orig=_orig_arm):
                _orig(proc)
                try:
                    ctx.register_subprocess(proc)
                except Exception:
                    pass
            tok.arm = _arm_and_forward  # type: ignore[method-assign]

            req = FEMJobRequest(
                solve_out_dir=c_dir,
                mesh_input_dir=d_dir,
                cwd=H.script_cwd,
                preset=_preset,
                mpi_cores=_cores,
                emit_impedance=bool(
                    getattr(state, "emit_impedance", True),
                ),
                # R1.2 — emit reciprocity solves whenever this
                # config has any recording montages. Solver-side
                # caching ensures unchanged contacts are skipped,
                # so re-running with the same wiring is cheap.
                emit_recording=bool(_rec_montages),
            )
            _on_line(
                f"# launching solve_nerve.py "
                f"(SOLVE_OUT_DIR={c_dir}, "
                f"SOLVE_SHARED_DIR={d_dir})"
            )
            # Record the solve start time so we can tell THIS
            # run's outputs apart from a prior session's by
            # mtime. We deliberately do NOT wipe outputs pre-
            # solve: if this run fails, the user's previously-
            # solved outputs should stay intact.
            import time as _time
            _solve_start = _time.time() - 1.0  # 1s clock slack
            hb = asyncio.create_task(_heartbeat())
            try:
                outputs = await loop.run_in_executor(
                    None,
                    lambda: runner.run(req, _on_line, tok),
                )
            finally:
                hb.cancel()
                try:
                    await hb
                except asyncio.CancelledError:
                    pass
            rc = outputs.return_code
            if ctx.was_cancelled():
                _on_line("⚠ FEM solve cancelled by user")
                state.fem_status = "FEM solve cancelled."
                _surface_failure(
                    "⚠ FEM solve cancelled — see log"
                )
                return
            # Validate by checking that the expected outputs
            # were written DURING THIS RUN (mtime ≥ solve start
            # time). We don't wipe outputs pre-solve, so files
            # left over from a previous successful solve could
            # exist on disk; mtime distinguishes them from a
            # fresh write.
            #
            # mpich on macOS sometimes crashes in MPI_Finalize
            # AFTER a successful solve (libfabric OFI nic
            # lookup) → rc != 0 even though every output IS
            # fresh. The mtime check handles that gracefully.
            def _is_fresh(p):
                if not p.is_file():
                    return False
                try:
                    return p.stat().st_mtime >= _solve_start
                except Exception:                       # noqa: BLE001
                    return False

            outputs_fresh = (
                _is_fresh(c_dir / "axis_line.npz")
                and _is_fresh(c_dir / "slice_volume.npz")
                and _is_fresh(c_dir / "paths_Ve.npz")
            )
            if rc != 0:
                if outputs_fresh:
                    _on_line(
                        f"⚠ subprocess exited with code {rc} "
                        f"for config '{cid}' but all expected "
                        f"outputs are fresh on disk — "
                        f"proceeding (likely MPI_Finalize "
                        f"fallout, not a solve failure)"
                    )
                else:
                    # Real failure for THIS config — log loudly,
                    # but don't abort the batch. The other picked
                    # configs may still succeed; the user can
                    # come back and investigate this one
                    # separately. State only flips to "failed"
                    # if NO config succeeds (checked below).
                    # Any pre-existing outputs from a previous
                    # successful session stay intact — this run
                    # didn't touch them.
                    _on_line(
                        f"⚠ subprocess exited with code {rc} "
                        f"for config '{cid}' — see log for the "
                        f"underlying error. Continuing to "
                        f"next config; any previously-solved "
                        f"outputs for this config are "
                        f"untouched."
                    )
                    continue

            _new_meta = {
                "id": cid,
                "name": disp_name,
                "design_id": parent_eid,
                "design_name": parent_name,
                "n_patches": int(n_pat),
                "I_stim_mA": float(I_stim),
                "sha256": design_sha256(c_dir),
            }
            configs_meta.append(_new_meta)
            # F3.2c fix: publish each config to state.fem_configs
            # AS IT COMPLETES (not after the entire batch
            # finishes). Otherwise the analysis-chip switcher
            # only learns about new configs at the very end,
            # which made multi-config batches look stuck on
            # "one result" mid-run. Also persist the manifest
            # immediately so a crash mid-batch doesn't lose the
            # already-solved configs.
            write_config_manifest(out_dir, cid, _new_meta)
            # Merge with any pre-existing fem_configs from a
            # prior batch so reruns are additive, not destructive
            # — replacing an entry that has the same cid.
            _existing = [
                c for c in (state.fem_configs or [])
                if c.get("id") != cid
            ]
            state.fem_configs = _existing + [_new_meta]
            state.flush()
            _on_line(
                f"  ✓ config '{cid}' published to "
                f"fem_configs ({len(configs_meta)} solved "
                f"this batch, {len(state.fem_configs)} total "
                f"on disk)"
            )

        if not configs_meta:
            _surface_failure(
                "⚠ no FEM solves produced output. Check the "
                "log for per-config skip reasons (missing mesh, "
                "no patches, etc.)."
            )
            return
        # Pick which config is "active" (drives plots + 3D
        # overlays). Preserve the prior active if it survived;
        # otherwise default to the first newly-solved one.
        _prev_active = str(
            getattr(state, "active_config_id", "") or "",
        )
        _known_cids = {m["id"] for m in configs_meta}
        active_cid = (
            _prev_active
            if _prev_active in _known_cids
            else configs_meta[0]["id"]
        )
        state.active_config_id = active_cid
        # active_design_id mirrors whichever design the active
        # config lives on — keeps the design-side selectors in
        # sync.
        active_parent_eid = next(
            (m["design_id"] for m in configs_meta
             if m["id"] == active_cid),
            "",
        )
        if active_parent_eid:
            state.active_design_id = active_parent_eid
        active_design_dir = config_dir_fn(out_dir, active_cid)
        _on_line(
            f"# wrote {len(configs_meta)} config manifest(s); "
            f"active = '{active_cid}' (on design "
            f"'{active_parent_eid}')"
        )

        # ---- 4. Load + cache the .npz outputs ----
        axis_path = active_design_dir / "axis_line.npz"
        slice_path = active_design_dir / "slice_volume.npz"
        if not axis_path.exists() or not slice_path.exists():
            _surface_failure(
                "⚠ FEM outputs missing — see log"
            )
            return
        geom.fem_axis = np.load(axis_path, allow_pickle=True)
        geom.fem_slice = np.load(slice_path, allow_pickle=True)
        # paths_Ve.npz → per-fiber Vₑ for the Ve-on-fibers overlay
        paths_ve_path = active_design_dir / "paths_Ve.npz"
        if paths_ve_path.exists():
            try:
                pvz = np.load(
                    paths_ve_path, allow_pickle=True,
                )
                pv_lens = np.asarray(pvz["path_lengths"])
                pv_Ve = np.asarray(pvz["Ve_flat"])
                pv_flat = (
                    np.asarray(pvz["paths_flat"])
                    if "paths_flat" in pvz.files else None
                )
                pv_Ez = (
                    np.asarray(pvz["Ez_flat"])
                    if "Ez_flat" in pvz.files else None
                )
                paths_Ve: list = []
                paths_Ez: list = []
                paths_xyz: list = []
                _off = 0
                for L in pv_lens:
                    _n = int(L)
                    paths_Ve.append(
                        pv_Ve[_off:_off + _n].copy(),
                    )
                    if pv_Ez is not None:
                        paths_Ez.append(
                            pv_Ez[_off:_off + _n].copy(),
                        )
                    if pv_flat is not None:
                        paths_xyz.append(
                            pv_flat[_off:_off + _n].copy(),
                        )
                    _off += _n
                geom.fiber_paths_Ve = paths_Ve
                geom.fiber_paths_Ez = (
                    paths_Ez if pv_Ez is not None else None
                )
                geom.fiber_paths_for_Ve = (
                    paths_xyz if pv_flat is not None else None
                )
                _on_line(
                    f"# loaded paths_Ve.npz "
                    f"({len(paths_Ve)} fibers, "
                    f"{pv_Ve.size:,} pts; "
                    f"Ez {'present' if pv_Ez is not None else 'missing'}, "
                    f"paths_flat "
                    f"{'present' if pv_flat is not None else 'missing'})"
                )
            except Exception as ex:
                _on_line(
                    f"  ⚠ paths_Ve.npz load failed: {ex}"
                )
                geom.fiber_paths_Ve = None
                geom.fiber_paths_Ez = None
                geom.fiber_paths_for_Ve = None
        else:
            geom.fiber_paths_Ve = None
            geom.fiber_paths_Ez = None
            geom.fiber_paths_for_Ve = None

        # Surface Vₑ (one value per region_surfaces[1] vertex).
        nsv_path = active_design_dir / "nerve_surface_Ve.npz"
        if nsv_path.exists():
            try:
                nsvd = np.load(nsv_path, allow_pickle=True)
                geom.nerve_surface_Ve = np.asarray(
                    nsvd["Ve"], dtype=np.float64,
                )
                _on_line(
                    f"# loaded nerve_surface_Ve.npz "
                    f"({geom.nerve_surface_Ve.size:,} pts)"
                )
            except Exception as ex:
                _on_line(
                    f"  ⚠ nerve_surface_Ve.npz load failed: "
                    f"{ex}"
                )
                geom.nerve_surface_Ve = None
        else:
            geom.nerve_surface_Ve = None
        # Stash the stim current for the plot title.
        geom.fem_axis_extra = {
            "I_stim_mA": float(state.I_stim_mA),
        }
        _on_line(
            f"# loaded axis_line.npz "
            f"({len(geom.fem_axis['z'])} pts) + "
            f"slice_volume.npz "
            f"({geom.fem_slice['Ve'].shape[0]} slices)"
        )

        # ---- 5. Render plots ----
        H.refresh_fem_plots()
        state.has_fem = True
        # I1 Phase A — load impedance.json if solve_nerve.py
        # emitted it. Surface to state so the Compare-panel
        # impedance tile + figure builders can read it. Layout:
        # state.fem_impedance = {
        #   cid: {schema, frequency_hz, per_contact, per_pair},
        #   ...
        # }
        try:
            import json as _json
            _imp_path = active_design_dir / "impedance.json"
            if _imp_path.is_file():
                _imp_data = _json.loads(
                    _imp_path.read_text(),
                )
                _all_imp = dict(
                    getattr(state, "fem_impedance", {}) or {},
                )
                _all_imp[str(active_cid)] = _imp_data
                state.fem_impedance = _all_imp
                _on_line(
                    f"# loaded impedance.json "
                    f"({len(_imp_data.get('per_contact', []))} "
                    f"contacts, "
                    f"{len(_imp_data.get('per_pair', []))} pairs)"
                )
        except Exception as _ex:                         # noqa: BLE001
            _on_line(
                f"# WARN: impedance load failed: "
                f"{type(_ex).__name__}: {_ex}"
            )
        _n_configs = len(configs_meta)
        _config_chip = (
            f" across {_n_configs} configs "
            f"(active: '{active_cid}')"
            if _n_configs > 1 else ""
        )
        state.fem_status = (
            f"✓ FEM solved — {len(geom.fem_axis['z'])} axis "
            f"pts, {geom.fem_slice['Ve'].shape[0]} slices"
            f"{_config_chip}"
        )
        # Auto-enable the Ve overlays now that fresh FEM output
        # is available — otherwise the user has to hunt for the
        # legend toggles after every solve.
        if (geom.fiber_paths_Ve is not None
                and len(geom.fiber_paths_Ve) > 0):
            if not bool(state.show_ve_fibers):
                state.show_ve_fibers = True
            elif geom.fiber_paths_raw is not None:
                ctx.scene.request_render()
        if (geom.nerve_surface_Ve is not None
                and geom.nerve_surface_Ve.size > 0):
            if not bool(state.show_ve_surface):
                state.show_ve_surface = True
            elif state.has_mesh and geom.msh_path is not None:
                ctx.scene.request_render()
        # E-field streamlines — same auto-enable pattern.
        if not bool(state.show_field_lines):
            state.show_field_lines = True
        else:
            ctx.scene.request_render()
        # Autosave: FEM outputs are on disk; capture a thumbnail
        # so the project tile reflects the latest stage reached.
        ctx.autosave(stage="fem", capture_thumb=True)
    except Exception as ex:
        if ctx.was_cancelled():
            _on_line("⚠ FEM solve cancelled by user")
            state.fem_status = "FEM solve cancelled."
        else:
            _on_line(f"⚠ {type(ex).__name__}: {ex}")
            _surface_failure(
                f"⚠ {type(ex).__name__}: {ex} — see log"
            )
    finally:
        ctx.clear_subprocess()
        state.busy = False
        state.busy_log = ""
        state.flush()
        ctx.safe_update()
