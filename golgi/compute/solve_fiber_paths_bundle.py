# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Per-fascicle Laplace + streamlines solver for µCT-bundle imports.

Pipeline driver (`golgi/pipeline/fibers.py`) writes one cleaned
surface npz per fascicle plus a `fascicle_manifest.json` index;
this subprocess iterates over the manifest, builds a small tet
mesh per fascicle, solves the same Neumann-current Laplace BVP
as the whole-nerve solver (`solve_fiber_paths_nerve.py`), and
RK4-integrates streamlines from the seed cap to the drain cap.

Why per-fascicle (vs. one solve over the epi shell):
  - Epi shell of a sheep VN µCT bundle tetrahedralises to ~1.8 M
    tets, which OOM-kills the Laplace assembly + L2 projection on
    a 16 GB machine (the subprocess that this script replaces
    exited with -9 / SIGKILL after ~5 min).
  - Each fascicle is ~5–15 % of the epi volume → ~100–250 k tets,
    well within memory. The fascicle-by-fascicle loop solves a
    sequence of small problems and accumulates the paths.
  - Fascicles are by construction the right physiological seed
    region for endoneural fibers — no need for the post-hoc
    fascicle filter the whole-nerve path used.

Seed budget: the requested `n_fibers` is split proportionally to
each fascicle's seed-end cap area (computed on the SURFACE before
tet-meshing, so the split happens before any expensive work).
Failure mode: a fascicle whose solve raises (TetGen rejects the
PLC, Laplace doesn't converge, etc.) is logged and skipped — the
other fascicles still produce paths. A run that yielded zero
paths writes no npz and returns rc=1 so the driver surfaces the
failure.

Output contract (consumed by `pipeline/fibers.py`):
  nerve_paths_fibers.npz:
      paths_flat:    (sum(L_i), 3) float64, all paths concatenated
      path_lengths:  (n_paths,) int64
      branch_idx:    (n_paths,) int64, fascicle index per path
      step_m:        scalar step size in metres
      seed_end:      "low" or "high"
  nerve_paths_caps.json:
      branched_end:  the opposite end from `seed_end`
      n_branch_caps: number of fascicles that produced ≥ 1 path
      branch_cap_centroids_m: per-fascicle drain-cap centroids
                              (in raw frame; same frame as the
                              fascicle surfaces the driver wrote)
The `branch_idx` field lets the driver skip the kNN-against-
caps-json reclassification entirely: each path's fascicle index
is authoritative.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import (
    bb_tree, compute_colliding_cells, compute_collisions_points,
)
from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
OUT = Path(os.environ.get("FIBER_OUT_DIR", str(HERE / "results")))

comm = MPI.COMM_WORLD
rank = comm.rank


def _say(msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _build_nerve_only_mesh(pts, tris, lc_target, tmp_msh_path):
    """Build a clean single-domain tet mesh from a closed surface
    triangulation `(pts, tris)`. Mirrors the helper in
    `solve_fiber_paths_nerve.py` so the bundle solver stays
    independent of the legacy script. Writes a temporary .msh22
    next to OUT and loads it via dolfinx io.gmsh.
    """
    import pyvista as pv
    import tetgen as _tetgen
    from dolfinx import io as _dio

    pts = np.asarray(pts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    _say(f"    PLC: {len(pts):,} pts / {len(tris):,} tris")

    _faces = np.empty(len(tris) * 4, dtype=np.int64)
    _faces[0::4] = 3
    _faces[1::4] = tris[:, 0]
    _faces[2::4] = tris[:, 1]
    _faces[3::4] = tris[:, 2]
    plc = pv.PolyData(pts, _faces)

    _max_vol = lc_target ** 3 / (6.0 * np.sqrt(2.0))
    _say(f"    target lc = {lc_target*1e6:.0f} µm  →  "
         f"max tet volume = {_max_vol:.3e} m³")

    t = _tetgen.TetGen(plc)
    _say("    running TetGen on fascicle surface (single domain) ...")
    _result = t.tetrahedralize(
        maxvolume=_max_vol,
        epsilon=1.0e-6,
        collinear_ang_tol=178.0,
        facet_separate_ang_tol=178.0,
        verbose=1,
    )
    _nodes, _elems = _result[0], _result[1]
    _say(f"    TetGen done: {len(_nodes):,} pts, {len(_elems):,} tets")

    _nodes_arr = np.asarray(_nodes, dtype=np.float64)
    _elems_arr = np.asarray(_elems, dtype=np.int64)
    with open(tmp_msh_path, "w") as _f:
        _f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        _f.write(
            "$PhysicalNames\n1\n3 1 \"fascicle\"\n$EndPhysicalNames\n"
        )
        _f.write(f"$Nodes\n{len(_nodes_arr)}\n")
        for _i, _p in enumerate(_nodes_arr):
            _f.write(
                f"{_i+1} {_p[0]:.9g} {_p[1]:.9g} {_p[2]:.9g}\n"
            )
        _f.write("$EndNodes\n")
        _f.write(f"$Elements\n{len(_elems_arr)}\n")
        for _i, _e in enumerate(_elems_arr):
            _f.write(
                f"{_i+1} 4 2 1 1 "
                f"{_e[0]+1} {_e[1]+1} {_e[2]+1} {_e[3]+1}\n"
            )
        _f.write("$EndElements\n")
    _say(f"    wrote {tmp_msh_path.name}; loading into dolfinx ...")
    mesh_data = _dio.gmsh.read_from_msh(
        str(tmp_msh_path), comm, gdim=3,
    )
    return mesh_data.mesh


def _cluster_facets_by_xy(facet_idx, mids, eps_m):
    """Group facets into connected components by xy-proximity.
    Returns list of (facet_indices_subset, midpoints_subset) per
    cluster with ≥ 5 facets (filters noise). Same algorithm as
    the whole-nerve solver — kept here so the bundle script is
    self-contained.
    """
    if len(facet_idx) == 0:
        return []
    _tree = cKDTree(mids[:, :2])
    _pairs = _tree.query_pairs(r=eps_m, output_type="ndarray")
    _n = len(facet_idx)
    if len(_pairs) == 0:
        # No pairs within eps → each facet is its own cluster.
        # For a clean fascicle that's just one cluster per end,
        # so fall through with the whole input as one cluster.
        return [(facet_idx, mids)]
    _row = np.concatenate([_pairs[:, 0], _pairs[:, 1]])
    _col = np.concatenate([_pairs[:, 1], _pairs[:, 0]])
    _data = np.ones(len(_row), dtype=np.int32)
    _adj = csr_matrix((_data, (_row, _col)), shape=(_n, _n))
    _, _labels = connected_components(_adj, directed=False)
    _clusters = []
    for _c in range(int(_labels.max()) + 1):
        _m = _labels == _c
        if int(_m.sum()) >= 5:
            _clusters.append((facet_idx[_m], mids[_m]))
    return _clusters


def _estimate_cap_area_on_surface(
    pts: np.ndarray, tris: np.ndarray,
    seed_end: str,
    axial_normal_thresh: float,
    cap_band_frac: float,
) -> float:
    """Estimate the total seed-end cap area from the SURFACE mesh
    (before tet-meshing). Used to apportion `n_seeds` across
    fascicles proportionally to cap area. Cheap — pure numpy on a
    few-thousand-triangle surface — so we can compute the budget
    upfront before paying for any tet meshing.

    Algorithm:
      * PCA on the surface points to find the fascicle's intrinsic
        long axis.
      * Per-triangle: unit normal, signed projection onto the axis.
      * Axial triangles (|n·axis| > threshold) that lie in the
        seed-end z-band (within `cap_band_frac` of the appropriate
        extreme) form the seed-end cap; sum their area.
    """
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    _, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    cross_mag = np.linalg.norm(cross, axis=1)
    area = cross_mag * 0.5
    safe = cross_mag > 1.0e-30
    n_unit = np.zeros_like(cross)
    n_unit[safe] = cross[safe] / cross_mag[safe, None]
    nz = n_unit @ axis
    tri_centers = (v0 + v1 + v2) / 3.0
    z = (tri_centers - centroid) @ axis
    z_min, z_max = float(z.min()), float(z.max())
    band = cap_band_frac * (z_max - z_min)
    axial = np.abs(nz) > axial_normal_thresh
    if seed_end == "low":
        mask = axial & (z < z_min + band)
    else:
        mask = axial & (z > z_max - band)
    return float(area[mask].sum())


def _solve_one_fascicle(
    pts_raw: np.ndarray,
    tris: np.ndarray,
    n_seeds: int,
    seed_cfg,
    lc_target: float,
    fasc_idx: int,
    tmp_msh_path: Path,
):
    """Build tet mesh → detect caps → Laplace + L2-project E →
    seed at one cap, drain at the other → RK4 streamlines.

    Returns (paths_raw, drain_centroid_m) on success;
    raises on failure (caller catches and skips this fascicle).
    """
    cluster_eps_m = seed_cfg.cluster_eps_m
    cap_band_frac = seed_cfg.cap_band_frac
    min_rel_size = seed_cfg.min_rel_size
    axial_normal_thresh = seed_cfg.axial_normal_thresh
    seed_end = seed_cfg.seed_end
    step_um = seed_cfg.step_um
    max_steps_int = seed_cfg.max_steps
    step_m = step_um * 1.0e-6

    _say(f"  [fasc {fasc_idx}] meshing surface "
         f"({len(pts_raw):,} pts / {len(tris):,} tris) ...")
    submesh = _build_nerve_only_mesh(
        pts_raw, tris, lc_target, tmp_msh_path,
    )
    tdim = submesh.topology.dim
    fdim = tdim - 1
    n_sub_cells = submesh.topology.index_map(tdim).size_local
    n_sub_pts = submesh.geometry.x.shape[0]
    _say(f"  [fasc {fasc_idx}] tet mesh: "
         f"{n_sub_cells:,} cells, {n_sub_pts:,} points")

    submesh.topology.create_connectivity(fdim, tdim)
    submesh.topology.create_connectivity(fdim, 0)
    boundary_facets = mesh.exterior_facet_indices(submesh.topology)
    facet_mids = mesh.compute_midpoints(submesh, fdim, boundary_facets)

    # Intrinsic-frame cap detection (PCA on boundary points → align
    # long axis with +z, then axial-normal + z-band filter).
    _all_pts = submesh.geometry.x
    _centroid_mesh = _all_pts.mean(axis=0)
    _centered = _all_pts - _centroid_mesh
    _cov = np.cov(_centered, rowvar=False)
    _eigvals, _eigvecs = np.linalg.eigh(_cov)
    _principal = _eigvecs[:, -1]
    _target = np.array([0.0, 0.0, 1.0])
    _v = np.cross(_principal, _target)
    _s = np.linalg.norm(_v)
    _c = float(np.dot(_principal, _target))
    if _s < 1.0e-8:
        _R_intr = np.eye(3) if _c > 0 else -np.eye(3)
    else:
        _K = np.array([[0, -_v[2], _v[1]],
                        [_v[2], 0, -_v[0]],
                        [-_v[1], _v[0], 0]])
        _R_intr = (np.eye(3) + _K
                   + _K @ _K * ((1.0 - _c) / (_s * _s)))

    mids_intrinsic = (facet_mids - _centroid_mesh) @ _R_intr.T
    z_min_intr = float(mids_intrinsic[:, 2].min())
    z_max_intr = float(mids_intrinsic[:, 2].max())

    # Per-facet normals → axial mask.
    _fv = submesh.topology.connectivity(fdim, 0)
    _verts_geom = submesh.geometry.x
    _facet_normals = np.zeros(
        (len(boundary_facets), 3), dtype=np.float64,
    )
    for _fi, _facet in enumerate(boundary_facets):
        _vs = _fv.links(int(_facet))
        if len(_vs) < 3:
            continue
        _v0 = _verts_geom[_vs[0]]
        _v1 = _verts_geom[_vs[1]]
        _v2 = _verts_geom[_vs[2]]
        _nv = np.cross(_v1 - _v0, _v2 - _v0)
        _nm = np.linalg.norm(_nv)
        if _nm > 1.0e-30:
            _facet_normals[_fi] = _nv / _nm
    _facet_normals_intr = _facet_normals @ _R_intr.T
    _facet_nz_intr = _facet_normals_intr[:, 2]
    _axial_mask = np.abs(_facet_nz_intr) > axial_normal_thresh
    _intr_mid_z = mids_intrinsic[:, 2]

    _z_span_intr = z_max_intr - z_min_intr
    _cap_band = cap_band_frac * _z_span_intr
    _bot_mask = _axial_mask & (_intr_mid_z < z_min_intr + _cap_band)
    _top_mask = _axial_mask & (_intr_mid_z > z_max_intr - _cap_band)
    _bot_facets = boundary_facets[_bot_mask]
    _top_facets = boundary_facets[_top_mask]
    _bot_mids = mids_intrinsic[_bot_mask]
    _top_mids = mids_intrinsic[_top_mask]
    _say(f"  [fasc {fasc_idx}] cap facets: "
         f"{len(_bot_facets)} low-z, {len(_top_facets)} high-z")

    caps_lo = _cluster_facets_by_xy(
        _bot_facets, _bot_mids, eps_m=cluster_eps_m,
    )
    caps_hi = _cluster_facets_by_xy(
        _top_facets, _top_mids, eps_m=cluster_eps_m,
    )

    # Drop clusters too far from the extreme (saddle / mid-trunk
    # axial artefacts). Less of a concern for clean fascicle tubes
    # but kept for parity with the whole-nerve solver.
    def _filter_near_extreme(clusters, extreme_z, side):
        kept = []
        for _fs, _mids in clusters:
            _mean_z = float(_mids[:, 2].mean())
            if side == "lo":
                _ok = _mean_z < extreme_z + _cap_band
            else:
                _ok = _mean_z > extreme_z - _cap_band
            if _ok:
                kept.append((_fs, _mids))
        return kept

    caps_lo = _filter_near_extreme(caps_lo, z_min_intr, "lo")
    caps_hi = _filter_near_extreme(caps_hi, z_max_intr, "hi")

    def _filter_relative_size(clusters):
        if len(clusters) <= 1:
            return clusters
        _max_n = max(len(_fs) for _fs, _ in clusters)
        return [
            (_fs, _mids) for _fs, _mids in clusters
            if len(_fs) / max(_max_n, 1) >= min_rel_size
        ]

    caps_lo = _filter_relative_size(caps_lo)
    caps_hi = _filter_relative_size(caps_hi)
    _say(f"  [fasc {fasc_idx}] caps after filtering: "
         f"low-z={len(caps_lo)}, high-z={len(caps_hi)}")

    if len(caps_lo) == 0 or len(caps_hi) == 0:
        raise RuntimeError(
            f"fascicle {fasc_idx}: cap topology unworkable — "
            f"got {len(caps_lo)} low + {len(caps_hi)} high"
        )

    # Per-fascicle trunk vs branched: pick smaller-count side as
    # "single", other as "multi". For a clean fascicle both are 1
    # cap, so the split is symmetric — seed_end selects which side
    # is the source.
    if len(caps_lo) <= len(caps_hi):
        single_caps, multi_caps = caps_lo, caps_hi
        single_label, multi_label = "low-z", "high-z"
    else:
        single_caps, multi_caps = caps_hi, caps_lo
        single_label, multi_label = "high-z", "low-z"

    # Facet markers + Neumann BCs.
    _n_total_facets = (
        submesh.topology.index_map(fdim).size_local
        + submesh.topology.index_map(fdim).num_ghosts
    )
    _markers = np.zeros(_n_total_facets, dtype=np.int32)
    for _fs, _ in single_caps:
        _markers[_fs] = 10
    for _i, (_fs, _) in enumerate(multi_caps):
        _markers[_fs] = 11 + _i
    _marked_idx = np.where(_markers > 0)[0].astype(np.int32)
    _marked_vals = _markers[_marked_idx]
    _order = np.argsort(_marked_idx)
    _marked_idx = _marked_idx[_order]
    _marked_vals = _marked_vals[_order]
    facet_tags = mesh.meshtags(
        submesh, fdim, _marked_idx, _marked_vals,
    )
    ds = ufl.Measure(
        "ds", domain=submesh, subdomain_data=facet_tags,
    )

    V = fem.functionspace(submesh, ("Lagrange", 1))
    one = fem.Constant(submesh, 1.0)
    A_trunk = float(fem.assemble_scalar(fem.form(one * ds(10))))
    A_branches = [
        float(fem.assemble_scalar(fem.form(one * ds(11 + _i))))
        for _i in range(len(multi_caps))
    ]
    _say(f"  [fasc {fasc_idx}] cap areas (mm²): "
         f"trunk={A_trunk*1e6:.2f}, "
         f"drains={[f'{_A*1e6:.2f}' for _A in A_branches]}")

    I_total = 1.0
    _N_branches = len(multi_caps)
    _J_trunk = +I_total / A_trunk
    _J_branches = [
        -I_total / (_N_branches * _A) for _A in A_branches
    ]

    phi = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(phi), ufl.grad(w)) * ufl.dx
    L = _J_trunk * w * ds(10)
    for _i, _J in enumerate(_J_branches):
        L = L + _J * w * ds(11 + _i)

    # Pin φ = 0 at one interior vertex (centroid-nearest) to fix
    # the pure-Neumann additive-constant ambiguity.
    _pin_pt = int(np.argmin(
        np.linalg.norm(
            submesh.geometry.x - _centroid_mesh, axis=1,
        )
    ))
    bc_pin = fem.dirichletbc(
        0.0, np.array([_pin_pt], dtype=np.int32), V,
    )

    phi_h = fem.Function(V, name="phi")
    _say(f"  [fasc {fasc_idx}] solving Laplace ...")
    problem = LinearProblem(
        a, L, bcs=[bc_pin], u=phi_h,
        petsc_options_prefix=f"fasc{fasc_idx}_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1.0e-10,
        },
    )
    problem.solve()

    # L2-project -∇φ onto P1 vector field.
    _say(f"  [fasc {fasc_idx}] L2-projecting E = -∇φ ...")
    W = fem.functionspace(submesh, ("Lagrange", 1, (3,)))
    E_trial = ufl.TrialFunction(W)
    E_test = ufl.TestFunction(W)
    _a_E = ufl.dot(E_trial, E_test) * ufl.dx
    _L_E = ufl.dot(-ufl.grad(phi_h), E_test) * ufl.dx
    E_field = fem.Function(W, name="E")
    _problem_E = LinearProblem(
        _a_E, _L_E, u=E_field,
        petsc_options_prefix=f"fasc{fasc_idx}_E_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1.0e-8,
        },
    )
    _problem_E.solve()

    # Pick seed cluster(s) based on seed_end.
    submesh.topology.create_connectivity(fdim, 0)
    _fv_conn = submesh.topology.connectivity(fdim, 0)
    if seed_end == "low":
        _seed_facet_lists = caps_lo
    else:
        _seed_facet_lists = caps_hi
    if not _seed_facet_lists:
        # Fall back to all axial-normal facets at this end.
        _mask = _bot_mask if seed_end == "low" else _top_mask
        _seed_facets_flat = boundary_facets[_mask]
        _clusters_for_seeds = [(
            _seed_facets_flat,
            np.zeros((len(_seed_facets_flat), 3)),
        )]
    else:
        _clusters_for_seeds = _seed_facet_lists

    _cluster_seed_xyz: list[np.ndarray] = []
    _cluster_centroids_xy: list[np.ndarray] = []
    for _fs, _ in _clusters_for_seeds:
        _vset = set()
        for _f in _fs:
            for _v in _fv_conn.links(int(_f)):
                _vset.add(int(_v))
        if not _vset:
            continue
        _vtx = np.array(sorted(_vset), dtype=np.int64)
        _xyz = np.asarray(
            submesh.geometry.x[_vtx], dtype=np.float64,
        )
        _cluster_seed_xyz.append(_xyz)
        _cluster_centroids_xy.append(
            _xyz[:, :2].mean(axis=0),
        )

    _RIM_PULL = 0.15
    _seed_xyz_list = []
    for _xyz, _cxy in zip(
        _cluster_seed_xyz, _cluster_centroids_xy,
    ):
        _xy_pulled = _xyz[:, :2] + _RIM_PULL * (
            _cxy[None, :] - _xyz[:, :2]
        )
        _seed_xyz_list.append(
            np.column_stack([_xy_pulled, _xyz[:, 2]]),
        )
    _seed_xyz = (
        np.vstack(_seed_xyz_list)
        if _seed_xyz_list else np.zeros((0, 3))
    )
    if len(_seed_xyz) == 0:
        raise RuntimeError(
            f"fascicle {fasc_idx}: zero seed candidates after "
            f"cap selection"
        )

    if len(_seed_xyz) > n_seeds:
        _idx = np.linspace(
            0, len(_seed_xyz) - 1, n_seeds,
        ).astype(int)
        _seed_xyz = _seed_xyz[_idx]
    _say(f"  [fasc {fasc_idx}] {len(_seed_xyz):,} seed vertices "
         f"(requested {n_seeds})")

    # bb_tree + batched RK4.
    _say(f"  [fasc {fasc_idx}] building bb_tree ...")
    _tree = bb_tree(submesh, tdim)

    def _batch_locate(points):
        _pts = np.ascontiguousarray(points, dtype=np.float64)
        _cand = compute_collisions_points(_tree, _pts)
        _coll = compute_colliding_cells(submesh, _cand, _pts)
        _cells = np.zeros(len(_pts), dtype=np.int32)
        _valid = np.zeros(len(_pts), dtype=bool)
        for _i in range(len(_pts)):
            _cs = _coll.links(_i)
            if len(_cs) > 0:
                _cells[_i] = int(_cs[0])
                _valid[_i] = True
        return _cells, _valid

    def _batch_eval_E(points):
        _cells, _valid = _batch_locate(points)
        _E_all = np.zeros((len(points), 3), dtype=np.float64)
        if _valid.any():
            _pts_in = np.ascontiguousarray(
                points[_valid], dtype=np.float64,
            )
            _cells_in = np.ascontiguousarray(
                _cells[_valid], dtype=np.int32,
            )
            _vals = E_field.eval(_pts_in, _cells_in)
            _E_all[_valid] = _vals
        return _E_all

    def _normalised(E_arr, eps=1.0e-30):
        _mag = np.linalg.norm(E_arr, axis=1, keepdims=True)
        return np.where(
            _mag > eps, E_arr / np.maximum(_mag, eps), 0.0,
        )

    def _fill_invalid(k_new, k_fallback):
        _mag = np.linalg.norm(k_new, axis=1, keepdims=True)
        return np.where(_mag > 0.5, k_new, k_fallback)

    # Sign: from trunk seeds, follow E (+1); from branched seeds,
    # go against E (-1). For per-fascicle, the seed is on whichever
    # end was selected — `single_label` is the side with fewer
    # clusters. Match the legacy solver's logic:
    if seed_end == single_label.split("-")[0]:
        _sign = +1.0
    else:
        _sign = -1.0

    _say(f"  [fasc {fasc_idx}] integrating "
         f"{len(_seed_xyz):,} streamlines, step={step_um:.0f} µm, "
         f"max_steps={max_steps_int}")
    _N = len(_seed_xyz)
    _x = _seed_xyz.copy()
    _active = np.ones(_N, dtype=bool)
    _paths_local: list[list] = [
        [_x[_i].copy()] for _i in range(_N)
    ]

    for _step_i in range(max_steps_int):
        if not _active.any():
            break
        _x_act = _x[_active]
        _k1 = _normalised(_batch_eval_E(_x_act))
        _mag1 = np.linalg.norm(_k1, axis=1)
        _new_dead_local = _mag1 < 0.5

        _k2 = _fill_invalid(_normalised(_batch_eval_E(
            _x_act + _sign * 0.5 * step_m * _k1,
        )), _k1)
        _k3 = _fill_invalid(_normalised(_batch_eval_E(
            _x_act + _sign * 0.5 * step_m * _k2,
        )), _k2)
        _k4 = _fill_invalid(_normalised(_batch_eval_E(
            _x_act + _sign * step_m * _k3,
        )), _k3)

        _dx = _sign * (step_m / 6.0) * (
            _k1 + 2 * _k2 + 2 * _k3 + _k4
        )
        _x_act_new = _x_act + _dx

        _local_to_global = np.where(_active)[0]
        for _li, _gi in enumerate(_local_to_global):
            if _new_dead_local[_li]:
                _active[_gi] = False
                continue
            _x[_gi] = _x_act_new[_li]
            _paths_local[_gi].append(_x[_gi].copy())

        if (_step_i + 1) % 500 == 0:
            _say(f"    [fasc {fasc_idx}] step "
                 f"{_step_i+1}/{max_steps_int}, "
                 f"{int(_active.sum())} active")

    paths_raw = [
        np.asarray(_p, dtype=np.float64)
        for _p in _paths_local if len(_p) >= 5
    ]
    _say(f"  [fasc {fasc_idx}] produced {len(paths_raw):,} paths")

    # Drain centroid in raw frame: average midpoint of the
    # opposite-end cap facets, transformed back from intrinsic →
    # mesh frame. The mesh frame == raw frame here (pts_cuff =
    # pts_raw written by the driver).
    if seed_end == "low":
        _drain_mids_intr = (
            _top_mids if len(caps_hi) else _bot_mids
        )
    else:
        _drain_mids_intr = (
            _bot_mids if len(caps_lo) else _top_mids
        )
    if len(_drain_mids_intr) == 0:
        _drain_centroid_m = np.zeros(3, dtype=np.float64)
    else:
        _drain_intr_c = _drain_mids_intr.mean(axis=0)
        _drain_centroid_m = (
            _drain_intr_c @ _R_intr + _centroid_mesh
        )

    return paths_raw, _drain_centroid_m


def main():
    from golgi.jobs.schemas import FiberSeedConfig as _FiberSeedConfig

    _SEED_CFG_PATH = OUT / "nerve_paths_seed_config.json"
    if _SEED_CFG_PATH.exists():
        try:
            seed_cfg = _FiberSeedConfig.deserialize(
                json.loads(_SEED_CFG_PATH.read_text()),
            )
        except Exception:
            seed_cfg = _FiberSeedConfig()
    else:
        seed_cfg = _FiberSeedConfig()

    _say(f"per-fascicle solver: cap detection "
         f"eps={seed_cfg.cluster_eps_m*1e3:.2f} mm, "
         f"z-band={seed_cfg.cap_band_frac*100:.1f}%, "
         f"|n_z|>{seed_cfg.axial_normal_thresh:.2f}; "
         f"step={seed_cfg.step_um:.0f} µm, "
         f"max_steps={seed_cfg.max_steps}")

    _manifest_path = OUT / "fascicle_manifest.json"
    if not _manifest_path.exists():
        _say(f"ERROR: fascicle_manifest.json not found at "
             f"{_manifest_path}")
        raise SystemExit(2)
    manifest = json.loads(_manifest_path.read_text())
    fascicles = manifest["fascicles"]
    lc_target = float(manifest.get("lc_target", 2.0e-4))
    _say(f"manifest: {len(fascicles)} fascicle(s), "
         f"target lc = {lc_target*1e6:.0f} µm")

    # --- Pre-pass: estimate cap area on each surface for the
    # proportional seed split. Cheap (no tet-meshing yet). ---
    cap_areas: list[float] = []
    surfaces: list[tuple] = []
    for fasc in fascicles:
        _npz = OUT / fasc["npz"]
        _d = np.load(_npz)
        pts = np.asarray(_d["pts_raw"], dtype=np.float64)
        tris = np.asarray(_d["tris"], dtype=np.int64)
        surfaces.append((pts, tris))
        a = _estimate_cap_area_on_surface(
            pts, tris,
            seed_end=seed_cfg.seed_end,
            axial_normal_thresh=seed_cfg.axial_normal_thresh,
            cap_band_frac=seed_cfg.cap_band_frac,
        )
        cap_areas.append(a)
        _say(f"  fascicle {fasc.get('idx', '?')}: "
             f"surface cap area est. = {a*1e6:.3f} mm² "
             f"({len(pts):,} pts, {len(tris):,} tris)")

    cap_areas_arr = np.asarray(cap_areas, dtype=np.float64)
    total = float(cap_areas_arr.sum())
    if total <= 0.0:
        # Fall back to equal split.
        per = max(1, seed_cfg.n_seeds // max(1, len(fascicles)))
        n_per = [per] * len(fascicles)
        _say(f"  cap-area total = 0 — falling back to equal "
             f"split: {per} seeds per fascicle")
    else:
        weights = cap_areas_arr / total
        n_per = [
            max(1, int(round(seed_cfg.n_seeds * float(w))))
            for w in weights
        ]
        _say(f"  proportional seed split (by surface cap area): "
             f"{n_per} (total={sum(n_per)}, "
             f"requested={seed_cfg.n_seeds})")

    # --- Solve loop ---
    all_paths: list[np.ndarray] = []
    all_branch_idx: list[int] = []
    all_drain_centroids: list[list] = []
    _tmp_msh_path = OUT / "_fasc_tmp.msh"

    for fi, ((pts, tris), n_seeds_this, fasc) in enumerate(
        zip(surfaces, n_per, fascicles),
    ):
        _label_idx = int(fasc.get("idx", fi))
        _say(f"\n=== fascicle {fi+1}/{len(fascicles)} "
             f"(manifest idx={_label_idx}, "
             f"seeds={n_seeds_this}) ===")
        try:
            paths, drain_c = _solve_one_fascicle(
                pts, tris, n_seeds_this, seed_cfg,
                lc_target, _label_idx, _tmp_msh_path,
            )
        except Exception as ex:           # noqa: BLE001
            _say(f"  ⚠ fascicle {_label_idx} solve FAILED "
                 f"({type(ex).__name__}: {ex}); skipping")
            gc.collect()
            continue
        for p in paths:
            all_paths.append(p)
            all_branch_idx.append(_label_idx)
        all_drain_centroids.append(drain_c.tolist())
        gc.collect()

    if not all_paths:
        _say("\n⚠ no fascicle produced any paths")
        raise SystemExit(1)

    # --- Write outputs ---
    if rank == 0:
        _flat = np.vstack(all_paths)
        _lens = np.array(
            [len(p) for p in all_paths], dtype=np.int64,
        )
        _branch = np.array(all_branch_idx, dtype=np.int64)
        np.savez(
            OUT / "nerve_paths_fibers.npz",
            paths_flat=_flat,
            path_lengths=_lens,
            branch_idx=_branch,
            step_m=np.float64(seed_cfg.step_um * 1.0e-6),
            seed_end=np.array([seed_cfg.seed_end]),
        )
        _say(f"\nwrote {len(all_paths)} paths from "
             f"{len(all_drain_centroids)}/{len(fascicles)} "
             f"fascicles ({_flat.shape[0]:,} total pts)")

        caps_info = {
            "trunk_end": seed_cfg.seed_end,
            "branched_end": (
                "high" if seed_cfg.seed_end == "low" else "low"
            ),
            "n_branch_caps": len(all_drain_centroids),
            "branch_cap_centroids_m": all_drain_centroids,
        }
        (OUT / "nerve_paths_caps.json").write_text(
            json.dumps(caps_info, indent=2),
        )
        _say("wrote nerve_paths_caps.json")

    _say("done")


if __name__ == "__main__":
    main()
