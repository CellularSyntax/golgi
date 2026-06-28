# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""GeometryState — the mutable container holding everything we've
computed so far for the active project.

Pure data class. Read + written from many places across golgi.py
(compute pipeline parse-back, UI reset, project restore, ...).
The scene tier in later sub-steps will route mutations through
Scene.update_* methods for fields that drive rendering; for now,
direct attribute access stays.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GeometryState:
    """Mutable container holding everything we've computed so far.
    Each step adds more fields."""
    nerve: dict | None = None              # raw STL/NAS load
    centroid: np.ndarray | None = None     # global-PCA centroid
    R_global: np.ndarray | None = None     # global-PCA rotation
    cuff_origin_pca: np.ndarray | None = None  # in PCA frame
    R_local: np.ndarray | None = None      # local-PCA rotation
    pts_cuff: np.ndarray | None = None     # nerve in cuff frame
    R_ci: float | None = None              # auto-sized in m
    R_co: float | None = None              # in m
    msh_path: Path | None = None           # built mesh on disk
    # Cached built-mesh node / element arrays (so the colour-by-
    # quality overlay can re-render without re-reading the .msh).
    mesh_nodes: np.ndarray | None = None
    mesh_elems: np.ndarray | None = None
    mesh_tags: np.ndarray | None = None
    mesh_q: np.ndarray | None = None       # per-tet shape quality
    # Pre-computed per-region surface PolyData (one entry per tag).
    # Boundary extraction via vtkGeometryFilter is done ONCE in the
    # build executor; render_built_mesh just mounts these as actors.
    # Without this cache, a 22 M-tet mesh would block the asyncio
    # loop for minutes while np.unique chews through 88 M faces.
    #
    # SINGLE-SLOT fields below mirror the CURRENTLY-SELECTED
    # design's mesh data — kept for back-compat with legacy code
    # paths (FEM driver bbox calcs, project bundle export, etc.).
    # The MULTI-DESIGN authoritative source is `designs_meshes`
    # (added in F3.2-M1): every solved design's mesh + region
    # surfaces live there simultaneously so the scene pipeline can
    # render all designs at once with per-design visibility toggles.
    region_surfaces: dict | None = None    # tag → pv.PolyData (mm)
    # Decimated copies of region_surfaces for viewport rendering.
    # Built once after _extract_region_surfaces_mm so the heavy
    # decimation cost doesn't run on every actor refresh. None
    # until the mesh has been (re)built or restored.
    region_surfaces_viz: dict | None = None
    # F3.2-M1: per-design mesh storage. Maps each design's eid to
    # a dict mirroring the single-slot fields above:
    #   {"mesh_nodes": ndarray, "mesh_elems": ndarray,
    #    "mesh_tags":  ndarray, "mesh_q":    ndarray | None,
    #    "region_surfaces":     dict[int, pv.PolyData],
    #    "region_surfaces_viz": dict[int, pv.PolyData] | None,
    #    "msh_path":   Path,    "R_ci":  float, "R_co": float}
    # Populated incrementally by pipeline/mesh.py as each design's
    # mesh build completes, AND wholesale by app.py's mesh-restore
    # on project open (which now loads every design's nerve.msh
    # from disk, not just the active one). The scene pipeline's
    # _set_region_groups iterates this dict and mounts a per-
    # design region actor `region_<eid>_<tag>` for every (eid, tag)
    # pair. None means "no per-design data yet" — fall back to
    # single-slot fields when present.
    designs_meshes: dict | None = None
    # ---- FEM solve outputs (Phase B / §9) ----
    # In-memory mirrors of axis_line.npz + slice_volume.npz so the
    # slice slider can re-render without re-reading the disk.
    fem_axis: dict | None = None          # {s, z, Ve, Ez, ...}
    fem_slice: dict | None = None         # {x, y, z, Ve, Emag, nerve_mask, ...}
    # Per-fiber Ve sampled from the FEM solve (paths_Ve.npz). List of
    # (N_i,) arrays matching geom.fiber_paths_raw, one per fiber.
    fiber_paths_Ve: list | None = None
    # Per-fiber Ez along the same paths (from paths_Ve.npz `Ez_flat`).
    # Same shape as fiber_paths_Ve. Used by the §9 ribbon plot.
    fiber_paths_Ez: list | None = None
    # Per-fiber trajectory coordinates ACTUALLY USED when solve_nerve
    # sampled Ve. solve_nerve.py copies `paths_flat` into paths_Ve.npz
    # alongside Ve_flat / Ez_flat, so this dict guarantees the (path,
    # Ve, Ez) triple stays consistent — even if the user regenerated
    # fibers after the FEM solve (in which case `fiber_paths_raw`
    # from nerve_paths_fibers.npz is stale w.r.t. the cached Ve).
    fiber_paths_for_Ve: list | None = None
    # True once `fiber_paths_raw` has been rewritten in cuff frame —
    # either at solve time via `_ensure_fibers_in_cuff_frame` (which
    # also rewrites nerve_paths_fibers.npz on disk), or because the
    # cached file already carries the `frame_is_cuff` flag. Drives
    # `_render_fibers_current_frame` to skip the raw→cuff transform
    # when paths are already there.
    fibers_in_cuff_frame: bool = False
    # Shared Vₑ colour-scale limits (mV) used across the endo
    # surface, the epi surface, and the per-fiber tube overlay so
    # the plasma cmap maps the SAME value to the SAME colour
    # everywhere — and so the horizontal colourbar legend below
    # the viewport applies to all three. Recomputed in
    # _refresh_fem_plots after each solve from a percentile clip
    # of the nerve_surface_Ve array (the canonical full-volume
    # sample). None until a FEM solve produces output.
    ve_clim_mV: tuple[float, float] | None = None
    # Cached 3D electric-field streamlines (pv.PolyData) computed
    # once per FEM solve from slice_volume.npz. Built lazily the
    # first time the user toggles `show_field_lines` on so the
    # streamline integration (a few seconds for a typical mesh)
    # doesn't run unless the feature is actually used.
    field_lines_poly: object | None = None
    # Per-vertex Ve sampled at the endoneurium surface (nerve_surface_
    # Ve.npz). 1-D array matching region_surfaces[1].points indexing.
    nerve_surface_Ve: np.ndarray | None = None
    _needs_camera_reset: bool = False      # one-shot client-cam fit
    # Direct reference to the vtkPolyData backing the nerve actor.
    # Stashed by render_raw_nerve every time it (re)mounts the
    # nerve, so the in-place point-update path can write straight
    # to the polydata without going through
    # `pl.actors['nerve'].GetMapper().GetInput()` — that round-
    # trip returns None during the first tick after add_mesh on
    # some pyvista/trame versions, and forcing a full remount
    # to work around it brings back the "two nerves" race
    # (pyvista's pl.remove_actor doesn't always evict the prior
    # vtkActor from the underlying VTK renderer on the same tick).
    nerve_poly: object | None = None
    # Cached "rigid cuff" fit. After a full refit (anchor change /
    # cuff-geometry slider) these hold the locked R_local + R_ci.
    # Position-only changes (offset/dx/dy) skip refitting and reuse
    # the cache so the cuff translates rigidly instead of resizing.
    _fit_locked: bool = False
    _R_local_cached: np.ndarray | None = None
    _R_ci_cached: float | None = None
    # Quality scalars for the nerve boundary surface (one value
    # per triangle in `nerve["boundary_raw"]`). Computed once at
    # load and reused for both the topology readout and the
    # color-by-quality render mode.
    nerve_q: np.ndarray | None = None
    # Generated fiber trajectories in RAW frame (list of (N_i, 3)
    # polylines in metres). Cached here so a later cuff fit can
    # re-render them transformed into cuff frame without having
    # to regenerate.
    fiber_paths_raw: list | None = None
    # Per-fiber branch index after kNN clustering against
    # nerve_paths_caps.json. fiber_branch_idx[k] ∈ [0, n_branches).
    fiber_branch_idx: np.ndarray | None = None
    fiber_n_branches: int = 0
    # Last single-fiber sim result (set by do_run_fiber_sim).
    # Schema matches the §12 fiber_sim_data dict from nerve_studio:
    # vm, t, node_z_um, spike_t, vm_peak, n_real, n_thresh, n_nodes,
    # model, diameter, stim_*, wave_t, Ve_at_nodes_mV, fiber_index,
    # source_label. Stays in-memory only — fiber-sim outputs are
    # cheap to reproduce so we don't persist them to disk.
    fiber_sim_data: dict | None = None
    fiber_sim_summary: str = ""
    # Multi-fiber sim results from a do_run_fiber_sim batch. Map
    # of fiber-index → sim_data dict (same schema as
    # fiber_sim_data). `fiber_sim_data` mirrors whichever entry
    # is currently being viewed in the output tiles — switching
    # via the result-picker just rewrites the figure state from
    # this map. Reset by project close / re-generate.
    fiber_sim_results: dict | None = None
    # Population-tab assignments (set by do_pop_generate). All
    # three arrays are length N = number of fibers, indexed
    # parallel to fiber_paths_raw / fiber_branch_idx.
    # fiber_pop_types[k]   = membrane-model string for fiber k
    #                       (e.g. "MRG_INTERPOLATION", "SUNDT").
    #                       "" (empty) for unassigned fibers.
    # fiber_pop_rows[k]    = row-id (hex string) of the
    #                       fiber-type row that fiber k was
    #                       sampled from. Drives per-row 3-D
    #                       coloring + per-row KDE traces.
    # fiber_pop_diameters_um[k] = sampled diameter in µm.
    # Reset to None whenever the user re-generates fibers (since
    # the population mapping is then stale).
    fiber_pop_types: np.ndarray | None = None
    fiber_pop_rows: np.ndarray | None = None
    fiber_pop_diameters_um: np.ndarray | None = None
    # Per-fiber sim results from a population run. Map of
    # fiber-index → sim_data dict (same schema as
    # fiber_sim_data / fiber_sim_results). Re-used by the
    # per-fiber heatmap + waterfall tiles via the population
    # result-picker. Reset on project close / re-generate.
    fiber_pop_sim_results: dict | None = None
