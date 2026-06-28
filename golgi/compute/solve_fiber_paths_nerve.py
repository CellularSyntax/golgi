# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Laplace on the endoneurium with Neumann current BCs at the caps.

The right physics for fiber trajectories:
  - Domain: ENDONEURIUM only (tag 1) — drop epi (tag 5).
  - σ = 1 (uniform; only the field topology matters for streamlines).
  - Lateral nerve surface: -σ ∂φ/∂n = 0  (perfect insulator).
  - Caps: Neumann CURRENT injection / drain (NOT Dirichlet voltage).
    Dirichlet at the caps forces them to be equipotential planes →
    streamlines enter/exit perpendicular to the cap → they all funnel
    through the centerline. Neumann lets the cap potential vary so
    streamlines retain their lateral identity through the nerve.
  - The branched end has TWO (or more) disconnected caps. We
    auto-detect them via spatial clustering and split the total drain
    current equally across the branch caps.
  - Pin φ=0 at one interior point to fix the otherwise-free additive
    constant of pure-Neumann problems.

Streamlines of −∇φ then follow the natural current flow from trunk
through the bifurcation into the branches.

Outputs:
    results/nerve_paths.vtu         — submesh + phi + E (pyvista-readable)
    results/nerve_paths_caps.json   — auto-detected cap metadata
    results/nerve_paths_field.xdmf  — for ParaView (optional)
"""
import json
import os
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
# Allow the caller (e.g. golgi.py) to redirect the I/O directory
# without editing this file. Defaults to the original results/.
OUT = Path(os.environ.get("FIBER_OUT_DIR", str(HERE / "results")))

comm = MPI.COMM_WORLD
rank = comm.rank


def _say(msg):
    if rank == 0:
        print(msg, flush=True)


def _rigid_procrustes(src, dst):
    """Best-fit rigid transform (R, t) from src→dst with corresponding rows.

    Returns R (3,3) rotation matrix and t (3,) translation such that
        dst ≈ src @ R.T + t
    in least-squares sense (no scaling, no reflection — orthogonal R).
    Use this to map any raw-frame point to cuff frame once we have the
    two corresponding nerve-vertex point clouds.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    # Guard against an improper-rotation (reflection) result
    _det = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, 1.0 if _det > 0 else -1.0])
    R = Vt.T @ D @ U.T
    t = dst_c - src_c @ R.T
    return R, t


def _build_nerve_only_mesh(pts, tris, lc_target):
    """Build a clean single-domain tet mesh from a closed surface
    triangulation `(pts, tris)`. Uses TetGen on the surface PLC
    directly — the SAME tool nerve_studio already feeds this surface
    into for the multi-domain build, so it's guaranteed to accept it.

    Returns a dolfinx Mesh.

    Rationale: gmsh's classifySurfaces → createGeometry path needs a
    NURBS-parameterizable boundary and routinely fails ('Wrong
    topology of boundary mesh for parametrization') on the .nas
    nerve mesh (275 k pts, 424 k tris). TetGen instead takes the
    surface as a discrete PLC and tetrahedralizes it — no NURBS, no
    parametrization step, exactly what nerve_studio's Section 5
    already does successfully.
    """
    import pyvista as pv
    import tetgen as _tetgen

    pts = np.asarray(pts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    _say(f"    PLC: {len(pts):,} pts / {len(tris):,} tris")

    # pyvista expects faces as flat [3, v0, v1, v2, 3, v0, v1, v2, ...]
    _faces = np.empty(len(tris) * 4, dtype=np.int64)
    _faces[0::4] = 3
    _faces[1::4] = tris[:, 0]
    _faces[2::4] = tris[:, 1]
    _faces[3::4] = tris[:, 2]
    plc = pv.PolyData(pts, _faces)

    # Cap the tet volume at the same ideal-regular-tet volume that
    # corresponds to `lc_target` edge length. nerve_studio does the
    # same thing for the endo region:  V = h³ / (6√2)
    _max_vol = lc_target ** 3 / (6.0 * np.sqrt(2.0))
    _say(f"    target lc = {lc_target*1e6:.0f} µm  →  "
         f"max tet volume = {_max_vol:.3e} m³")

    t = _tetgen.TetGen(plc)
    # Single domain — no add_region needed; TetGen fills the entire
    # PLC interior. The kwargs match the relaxed-predicate setup that
    # nerve_studio's Section 5 already uses successfully on this same
    # surface (epsilon + ang_tol allow near-degenerate triangles).
    _say("    running TetGen on nerve surface (single domain) ...")
    _result = t.tetrahedralize(
        maxvolume=_max_vol,
        epsilon=1.0e-6,
        collinear_ang_tol=178.0,
        facet_separate_ang_tol=178.0,
        verbose=1,
    )
    _nodes, _elems = _result[0], _result[1]
    _say(f"    TetGen done: {len(_nodes):,} pts, {len(_elems):,} tets")

    # Hand the resulting mesh to dolfinx via a tiny .msh22 round-trip
    # — same hand-written format nerve_studio uses in Section 5 so we
    # know the dolfinx io.gmsh reader is happy with it.
    _tmp_msh = OUT / "_nerve_only_tmp.msh"
    _nodes_arr = np.asarray(_nodes, dtype=np.float64)
    _elems_arr = np.asarray(_elems, dtype=np.int64)
    with open(_tmp_msh, "w") as _f:
        _f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        _f.write("$PhysicalNames\n1\n3 1 \"nerve\"\n$EndPhysicalNames\n")
        _f.write(f"$Nodes\n{len(_nodes_arr)}\n")
        for _i, _p in enumerate(_nodes_arr):
            _f.write(f"{_i+1} {_p[0]:.9g} {_p[1]:.9g} "
                     f"{_p[2]:.9g}\n")
        _f.write("$EndNodes\n")
        _f.write(f"$Elements\n{len(_elems_arr)}\n")
        for _i, _e in enumerate(_elems_arr):
            _f.write(f"{_i+1} 4 2 1 1 "
                     f"{_e[0]+1} {_e[1]+1} {_e[2]+1} {_e[3]+1}\n")
        _f.write("$EndElements\n")
    _say(f"    wrote {_tmp_msh.name}; loading into dolfinx ...")
    mesh_data = io.gmsh.read_from_msh(str(_tmp_msh), comm, gdim=3)
    return mesh_data.mesh


def _cluster_facets_by_xy(facet_idx, mids, eps_m=0.002):
    """Group facets into connected components by xy-proximity (KD-tree
    + scipy.sparse connected_components). eps_m in metres; default 2 mm
    is well below typical branch separation (~5–20 mm for VN) but well
    above per-facet edge length, so a single cap is one component.

    Returns list of (facet_indices_subset, midpoints_subset) per cluster
    with ≥ 5 facets (filters noise)."""
    if len(facet_idx) == 0:
        return []
    _tree = cKDTree(mids[:, :2])
    _pairs = _tree.query_pairs(r=eps_m, output_type="ndarray")
    _n = len(facet_idx)
    if len(_pairs) == 0:
        return []
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


def main():
    # =====================================================================
    # Read the seed-config JSON early so the cap-detection clustering
    # knobs are available before they get used (~300 lines below).
    # Defaults match the historical hard-coded values so callers that
    # don't supply the keys reproduce prior behaviour.
    # =====================================================================
    # Step 6.2c: parse via the typed FiberSeedConfig dataclass shared
    # with the pipeline driver (golgi/jobs/schemas.py) so a field
    # rename breaks at deserialize instead of silently degrading
    # to the default. The same `_seed_cfg` is reused ~450 lines
    # below for the streamline-integration knobs.
    from golgi.jobs.schemas import FiberSeedConfig as _FiberSeedConfig
    _SEED_CFG_PATH = OUT / "nerve_paths_seed_config.json"
    if _SEED_CFG_PATH.exists():
        try:
            _seed_cfg = _FiberSeedConfig.deserialize(
                json.loads(_SEED_CFG_PATH.read_text()),
            )
        except Exception:
            _seed_cfg = _FiberSeedConfig()
    else:
        _seed_cfg = _FiberSeedConfig()
    cluster_eps_m = _seed_cfg.cluster_eps_m
    cap_band_frac = _seed_cfg.cap_band_frac
    min_rel_size = _seed_cfg.min_rel_size
    axial_normal_thresh = _seed_cfg.axial_normal_thresh
    _say(f"  cap detection: eps={cluster_eps_m*1e3:.2f} mm, "
         f"z-band={cap_band_frac*100:.1f}%, "
         f"min rel size={min_rel_size*100:.1f}%, "
         f"|n_z|>{axial_normal_thresh:.2f}")

    # =====================================================================
    # Mesh source: prefer a clean nerve-only tet mesh built from the
    # imported surface in RAW frame. That way the trajectory solver
    # never sees the cuff geometry — cap detection, BCs and streamline
    # integration are 100 % decoupled from cuff position. Trajectories
    # come out in raw frame and we transform them to cuff frame at
    # the end using the rigid (R, t) we recover from the corresponding
    # nerve vertices that nerve_studio also wrote to the npz.
    # =====================================================================
    NERVE_ONLY_NPZ = OUT / "nerve_only_surface.npz"
    USE_NERVE_ONLY = NERVE_ONLY_NPZ.exists()
    raw_to_cuff_R = None
    raw_to_cuff_t = None

    if USE_NERVE_ONLY:
        _say("nerve_only_surface.npz found — building a clean "
             "nerve-only tet mesh from imported surface (RAW frame, "
             "INDEPENDENT of cuff position)")
        _d = np.load(NERVE_ONLY_NPZ, allow_pickle=True)
        _pts_raw_in = np.asarray(_d["pts_raw"], dtype=np.float64)
        _pts_cuff_disk = np.asarray(_d["pts_cuff"], dtype=np.float64)
        _tris_in = np.asarray(_d["tris"], dtype=np.int64)
        raw_to_cuff_R, raw_to_cuff_t = _rigid_procrustes(
            _pts_raw_in, _pts_cuff_disk
        )
        _residual = np.linalg.norm(
            (_pts_raw_in @ raw_to_cuff_R.T + raw_to_cuff_t)
            - _pts_cuff_disk,
            axis=1,
        )
        _say(f"  raw→cuff Procrustes residual: "
             f"max {_residual.max()*1e6:.1f} µm, "
             f"rms {np.sqrt((_residual**2).mean())*1e6:.1f} µm "
             f"(should be ≪ mesh size)")
        # Target mesh size — same as nerve_studio's nerve interior
        _lc_endo = 2.0e-4
        _mc_path = OUT / "mesh_config.json"
        if _mc_path.exists():
            try:
                _mc = json.loads(_mc_path.read_text())
                _lc_endo = float(_mc.get("lc_nerve", _lc_endo))
            except Exception:
                pass
        _say(f"  target lc = {_lc_endo*1e6:.0f} µm "
             f"(matches mesh_config.json lc_nerve if present)")
        _say(f"  meshing surface: {len(_pts_raw_in):,} pts / "
             f"{len(_tris_in):,} bnd tris ...")
        submesh = _build_nerve_only_mesh(
            _pts_raw_in, _tris_in, _lc_endo
        )
        tdim = submesh.topology.dim
        fdim = tdim - 1
        n_sub_cells = submesh.topology.index_map(tdim).size_local
        n_sub_pts = submesh.geometry.x.shape[0]
        _say(f"  nerve-only mesh: {n_sub_cells:,} cells, "
             f"{n_sub_pts:,} points")
    else:
        _say("nerve_only_surface.npz not found — falling back to "
             "extracting the endo submesh from nerve.msh (legacy "
             "path; coupled to cuff position)")
        _say("loading multi-domain mesh ...")
        mesh_data = io.gmsh.read_from_msh(
            str(OUT / "nerve.msh"), comm, gdim=3
        )
        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags
        tdim = domain.topology.dim
        fdim = tdim - 1

        if cell_tags is None or len(cell_tags.values) == 0:
            raise RuntimeError(
                "Mesh has no cell_tags — was it built by nerve_studio?"
            )

        # Endoneurium only (tag 1) — drop epi (tag 5).
        nerve_cells = np.where(
            cell_tags.values == 1
        )[0].astype(np.int32)
        _say(f"  endo cells (tag 1): {len(nerve_cells):,} of "
             f"{len(cell_tags.values):,} total")
        if len(nerve_cells) == 0:
            raise RuntimeError("No cells with tag 1 found.")

        _say("  creating endo submesh ...")
        _sub_result = mesh.create_submesh(domain, tdim, nerve_cells)
        submesh = _sub_result[0]
        n_sub_cells = submesh.topology.index_map(tdim).size_local
        n_sub_pts = submesh.geometry.x.shape[0]
        _say(f"  submesh: {n_sub_cells:,} cells, {n_sub_pts:,} points")

    # Boundary facets + midpoints (in MESH frame — which is nerve_studio's
    # cuff frame: aligned with the LOCAL nerve axis at the cuff site, NOT
    # the nerve's intrinsic principal axis).
    submesh.topology.create_connectivity(fdim, tdim)
    submesh.topology.create_connectivity(fdim, 0)
    boundary_facets = mesh.exterior_facet_indices(submesh.topology)
    facet_mids = mesh.compute_midpoints(submesh, fdim, boundary_facets)
    z_min_b = float(facet_mids[:, 2].min())
    z_max_b = float(facet_mids[:, 2].max())
    _say(f"  boundary: {len(boundary_facets):,} facets, mesh-frame "
         f"z range {z_min_b*1e3:+.2f}..{z_max_b*1e3:+.2f} mm")

    # === Intrinsic-frame cap detection ===
    # The mesh is in nerve_studio's CUFF FRAME. Cap detection that
    # relies on z-extremes is meaningful only when the nerve's
    # principal axis IS the +z axis — which is true only at one cuff
    # position (where the local PCA happens to align with the
    # global one). At any other cuff offset, the local PCA rotates
    # the entire nerve, so the cuff-frame z-extremes no longer
    # correspond to the physical endcaps and branches get filtered
    # out as "not at extreme". Fix: compute a GLOBAL PCA on the
    # nerve points themselves, do cap detection in that intrinsic
    # frame, and leave the FEM BC application to use the original
    # facet indices (those are frame-independent). The intrinsic
    # frame is rotation-invariant under any rigid transform of
    # the input mesh.
    _say("  computing intrinsic PCA on nerve boundary points ...")
    _all_pts = submesh.geometry.x
    _centroid = _all_pts.mean(axis=0)
    _centered = _all_pts - _centroid
    _cov = np.cov(_centered, rowvar=False)
    _eigvals, _eigvecs = np.linalg.eigh(_cov)
    _principal = _eigvecs[:, -1]   # largest-variance direction
    _target = np.array([0.0, 0.0, 1.0])
    _v = np.cross(_principal, _target)
    _s = np.linalg.norm(_v)
    _c = float(np.dot(_principal, _target))
    if _s < 1e-8:
        _R_intr = np.eye(3) if _c > 0 else -np.eye(3)
    else:
        _K = np.array([[0, -_v[2], _v[1]],
                       [_v[2], 0, -_v[0]],
                       [-_v[1], _v[0], 0]])
        _R_intr = np.eye(3) + _K + _K @ _K * ((1.0 - _c) / (_s * _s))
    _say(f"    intrinsic principal axis (mesh frame): "
         f"({_principal[0]:+.3f}, {_principal[1]:+.3f}, "
         f"{_principal[2]:+.3f})")
    # Rotate facet midpoints into the intrinsic frame
    mids_intrinsic = (facet_mids - _centroid) @ _R_intr.T
    z_min_intr = float(mids_intrinsic[:, 2].min())
    z_max_intr = float(mids_intrinsic[:, 2].max())
    _say(f"    intrinsic-frame z range "
         f"{z_min_intr*1e3:+.2f}..{z_max_intr*1e3:+.2f} mm "
         f"(length {(z_max_intr-z_min_intr)*1e3:.1f} mm)")

    # Identify cap facets via NORMAL DIRECTION in the INTRINSIC frame.
    # A cap facet has normal aligned with the intrinsic ±z axis; a
    # lateral wall facet has normal perpendicular to it. Using
    # |n_z_intr| > 0.7 robustly excludes lateral facets regardless of
    # how the cuff is rotated relative to the nerve.
    _say("  computing facet normals to identify cap facets ...")
    _fv = submesh.topology.connectivity(fdim, 0)
    _verts_geom = submesh.geometry.x
    _facet_normals = np.zeros((len(boundary_facets), 3), dtype=np.float64)
    for _fi, _facet in enumerate(boundary_facets):
        _vs = _fv.links(int(_facet))
        if len(_vs) < 3:
            continue
        _v0 = _verts_geom[_vs[0]]
        _v1 = _verts_geom[_vs[1]]
        _v2 = _verts_geom[_vs[2]]
        _n_vec = np.cross(_v1 - _v0, _v2 - _v0)
        _n_norm = np.linalg.norm(_n_vec)
        if _n_norm > 1.0e-30:
            _facet_normals[_fi] = _n_vec / _n_norm
    # Rotate the per-facet normal vectors into the intrinsic frame.
    _facet_normals_intr = _facet_normals @ _R_intr.T
    _facet_nz_intr = _facet_normals_intr[:, 2]

    _axial_mask = np.abs(_facet_nz_intr) > axial_normal_thresh
    _intr_mid_z = mids_intrinsic[:, 2]
    # Pre-filter axial-normal facets to a tight Z-BAND near each
    # extreme BEFORE clustering. The old split-at-midpoint approach
    # let mid-trunk axial features (bifurcation saddle, branch
    # stubs, surface kinks aligned with the principal axis) into
    # the low-z / high-z pools. On a complex VN STL those mid-trunk
    # axial facets are at small xy (near the trunk axis), which is
    # the SAME xy region the actual end caps occupy — so DBSCAN's
    # xy-only clustering merges them into one cluster, the centroid
    # ends up at mid-nerve, and `_filter_near_extreme` then drops
    # the whole cluster including the real cap facets.
    # Fix: keep only facets within a tight z-band of each extreme
    # in the first place. 15 % of nerve length is wide enough to
    # capture a cap that's slightly tilted but narrow enough to
    # exclude any saddle / mid-trunk feature.
    _z_span_intr = z_max_intr - z_min_intr
    _cap_band = cap_band_frac * _z_span_intr
    _bot_mask = _axial_mask & (_intr_mid_z < z_min_intr + _cap_band)
    _top_mask = _axial_mask & (_intr_mid_z > z_max_intr - _cap_band)
    _bot_facets = boundary_facets[_bot_mask]
    _top_facets = boundary_facets[_top_mask]
    _bot_mids = mids_intrinsic[_bot_mask]    # USE INTRINSIC COORDS
    _top_mids = mids_intrinsic[_top_mask]    # for clustering & filtering
    _say(f"  axial-normal cap facets (intrinsic frame, pre-band): "
         f"{len(_bot_facets)} low-z (band z < "
         f"{(z_min_intr + _cap_band)*1e3:+.1f} mm), "
         f"{len(_top_facets)} high-z (band z > "
         f"{(z_max_intr - _cap_band)*1e3:+.1f} mm) "
         f"of {len(boundary_facets):,} total boundary")

    # Cluster each end's cap facets in xy (in the INTRINSIC frame —
    # so each branch's cap is a compact cluster regardless of cuff
    # rotation).
    caps_lo = _cluster_facets_by_xy(
        _bot_facets, _bot_mids, eps_m=cluster_eps_m,
    )
    caps_hi = _cluster_facets_by_xy(
        _top_facets, _top_mids, eps_m=cluster_eps_m,
    )
    _say(f"  raw clusters: low-z={len(caps_lo)}, high-z={len(caps_hi)}")
    # From here on, the per-cluster `_mids` arrays are in INTRINSIC
    # coordinates. `_filter_near_extreme` compares mean z to
    # z_min_intr / z_max_intr below — that's consistent.
    z_min_b = z_min_intr
    z_max_b = z_max_intr

    # Post-clustering safety net: drop spurious clusters whose
    # centroid still doesn't sit near the extreme. With the pre-band
    # filter above (15 % of nerve length) this rarely fires, but
    # protects against a cluster that happens to straddle the band
    # edge.
    _z_extreme_band = cap_band_frac * (z_max_b - z_min_b)
    def _filter_near_extreme(clusters, extreme_z, side):
        """Keep clusters whose mean z is within band of `extreme_z`.
        side = 'lo' (mean z < extreme + band) or 'hi' (mean z > extreme - band)."""
        kept = []
        for _fs, _mids in clusters:
            _mean_z = float(_mids[:, 2].mean())
            if side == "lo":
                _ok = _mean_z < extreme_z + _z_extreme_band
            else:
                _ok = _mean_z > extreme_z - _z_extreme_band
            if _ok:
                kept.append((_fs, _mids))
            else:
                _say(f"    DROPPED {side}-z cluster at z="
                     f"{_mean_z*1e3:+.1f} mm "
                     f"({len(_fs)} facets) — too far from extreme "
                     f"z={extreme_z*1e3:+.1f} mm "
                     f"(threshold ±{_z_extreme_band*1e3:.1f} mm); "
                     f"likely a saddle / bifurcation artefact.")
        return kept

    # === Diagnostic C: detailed per-cluster info BEFORE filtering ===
    # Show centroid + size of every raw DBSCAN cluster at each end so
    # we can see whether branches are being correctly identified or
    # merged. DBSCAN eps_m=2 mm can merge two tightly-spaced branch
    # caps into one cluster on small bifurcations.
    def _log_clusters(_clusters, _label):
        for _i, (_fs, _mids) in enumerate(_clusters):
            _cxyz = _mids.mean(axis=0)
            _r = float(np.linalg.norm(
                _mids[:, :2] - _cxyz[:2], axis=1
            ).max())
            _say(f"    [diag C] raw {_label} cluster {_i}: "
                 f"{len(_fs)} facets, centroid (x,y,z) = "
                 f"({_cxyz[0]*1e3:+.2f}, {_cxyz[1]*1e3:+.2f}, "
                 f"{_cxyz[2]*1e3:+.2f}) mm, "
                 f"xy spread r_max = {_r*1e3:.2f} mm")
    _log_clusters(caps_lo, "low-z")
    _log_clusters(caps_hi, "high-z")
    _say(f"  [diag C] DBSCAN eps_m = {cluster_eps_m*1e3:.2f} mm — "
         f"if two branch caps sit within {cluster_eps_m*1e3:.2f} mm "
         f"in xy they get merged into one cluster. "
         f"Look at the centroids above: do you expect ≥ 2 caps at "
         f"one end of YOUR nerve?")

    caps_lo = _filter_near_extreme(caps_lo, z_min_b, "lo")
    caps_hi = _filter_near_extreme(caps_hi, z_max_b, "hi")

    # Relative-size filter: drop clusters that are tiny next to the
    # dominant cluster at the same end. Real branch caps are within a
    # factor of a few in facet count; a spurious cluster (lateral-wall
    # kink that happens to have an axial-normal patch, leftover
    # boundary noise from .nas surface defects, …) is usually 10-100×
    # smaller. This stops the cap detector from reporting an extra
    # "branch" that's really just a surface artefact.
    _MIN_REL_SIZE = min_rel_size
    def _filter_relative_size(clusters, label):
        if len(clusters) <= 1:
            return clusters
        _max_n = max(len(_fs) for _fs, _ in clusters)
        kept = []
        for _fs, _mids in clusters:
            _ratio = len(_fs) / max(_max_n, 1)
            if _ratio >= _MIN_REL_SIZE:
                kept.append((_fs, _mids))
            else:
                _cxyz = _mids.mean(axis=0)
                _say(f"    DROPPED {label} cluster at "
                     f"(x,y,z)=({_cxyz[0]*1e3:+.2f}, "
                     f"{_cxyz[1]*1e3:+.2f}, {_cxyz[2]*1e3:+.2f}) mm "
                     f"— only {len(_fs)} facets "
                     f"({_ratio*100:.0f}% of dominant cap "
                     f"= {_max_n} facets); likely a surface "
                     f"artefact, not a real branch cap.")
        return kept
    caps_lo = _filter_relative_size(caps_lo, "low-z")
    caps_hi = _filter_relative_size(caps_hi, "high-z")

    _say(f"  low-z end:  {len(caps_lo)} cap(s) after filtering")
    _say(f"  high-z end: {len(caps_hi)} cap(s) after filtering")

    # Decide which end is the trunk (single cap) vs branched (multi cap)
    if len(caps_lo) <= len(caps_hi):
        single_caps, multi_caps = caps_lo, caps_hi
        single_label, multi_label = "low-z", "high-z"
    else:
        single_caps, multi_caps = caps_hi, caps_lo
        single_label, multi_label = "high-z", "low-z"
    if len(single_caps) == 0 or len(multi_caps) == 0:
        raise RuntimeError(
            f"Cap topology unworkable — got "
            f"{len(caps_lo)} low caps + {len(caps_hi)} high caps. "
            f"Expected 1 + N (trunk + N branches)."
        )
    _say(f"  trunk = {single_label} end ({len(single_caps)} cap), "
         f"branches = {multi_label} end ({len(multi_caps)} cap{'s' if len(multi_caps)!=1 else ''})")

    # Build facet markers for ds():
    #   10 = trunk cap (current injection)
    #   11, 12, ... = branch caps (current drain)
    _n_total_facets = (submesh.topology.index_map(fdim).size_local
                        + submesh.topology.index_map(fdim).num_ghosts)
    _markers = np.zeros(_n_total_facets, dtype=np.int32)
    for _fs, _ in single_caps:
        _markers[_fs] = 10
    for _i, (_fs, _) in enumerate(multi_caps):
        _markers[_fs] = 11 + _i
    _marked_idx = np.where(_markers > 0)[0].astype(np.int32)
    _marked_vals = _markers[_marked_idx]
    # Sort by facet index for meshtags (required)
    _order = np.argsort(_marked_idx)
    _marked_idx = _marked_idx[_order]
    _marked_vals = _marked_vals[_order]
    facet_tags = mesh.meshtags(
        submesh, fdim, _marked_idx, _marked_vals,
    )
    ds = ufl.Measure("ds", domain=submesh, subdomain_data=facet_tags)

    # Function space + cap areas
    V = fem.functionspace(submesh, ("Lagrange", 1))
    one = fem.Constant(submesh, 1.0)
    A_trunk = float(fem.assemble_scalar(fem.form(one * ds(10))))
    A_branches = [
        float(fem.assemble_scalar(fem.form(one * ds(11 + _i))))
        for _i in range(len(multi_caps))
    ]
    _say(f"  trunk cap area = {A_trunk*1e6:.2f} mm²")
    for _i, _A in enumerate(A_branches):
        _say(f"  branch cap {_i} area = {_A*1e6:.2f} mm²")

    # Variational form with Neumann RHS.
    # weak form: ∫ σ∇φ·∇w dx = ∫_∂Ω J_n_inward * w ds
    # σ = 1, J_n_inward > 0 = current INTO domain.
    I_total = 1.0  # arbitrary; only the field shape matters
    _N_branches = len(multi_caps)
    _J_trunk = +I_total / A_trunk
    _J_branches = [-I_total / (_N_branches * _A) for _A in A_branches]
    _say(f"  J_n on trunk:    +{_J_trunk:.3e} A/m² (inject)")
    for _i, _J in enumerate(_J_branches):
        _say(f"  J_n on branch {_i}: {_J:.3e} A/m² (drain)")

    phi = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(phi), ufl.grad(w)) * ufl.dx
    L = _J_trunk * w * ds(10)
    for _i, _J in enumerate(_J_branches):
        L = L + _J * w * ds(11 + _i)

    # Pin φ = 0 at one interior vertex (near centroid) to fix the
    # additive-constant ambiguity of pure-Neumann problems.
    _centroid = submesh.geometry.x.mean(axis=0)
    _pin_pt = int(np.argmin(
        np.linalg.norm(submesh.geometry.x - _centroid, axis=1)
    ))
    bc_pin = fem.dirichletbc(
        0.0, np.array([_pin_pt], dtype=np.int32), V,
    )
    _say(f"  pinned φ=0 at vertex #{_pin_pt} (≈ centroid)")

    phi_h = fem.Function(V, name="phi")
    _say("  solving Laplace with Neumann current BCs ...")
    problem = LinearProblem(
        a, L, bcs=[bc_pin], u=phi_h,
        petsc_options_prefix="nerve_paths_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1.0e-10,
        },
    )
    problem.solve()
    _say(f"  solved (φ range {float(phi_h.x.array.min()):.3e}.."
         f"{float(phi_h.x.array.max()):.3e})")

    # L2-project -∇φ onto P1 (smooth point-data vector field for
    # pyvista streamline integration — no DG0 boundary averaging bias).
    _say("  L2-projecting -∇φ onto P1 ...")
    W = fem.functionspace(submesh, ("Lagrange", 1, (3,)))
    E_trial = ufl.TrialFunction(W)
    E_test = ufl.TestFunction(W)
    _a_E = ufl.dot(E_trial, E_test) * ufl.dx
    _L_E = ufl.dot(-ufl.grad(phi_h), E_test) * ufl.dx
    E_field = fem.Function(W, name="E")
    _problem_E = LinearProblem(
        _a_E, _L_E, u=E_field,
        petsc_options_prefix="nerve_paths_E_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1.0e-8,
        },
    )
    _problem_E.solve()
    _say("  E projected (P1 vector)")

    # ====================================================================
    # Batched RK4 streamline integration via dolfinx point evaluation.
    # This replaces pyvista's vtkStreamTracer which kept stalling on the
    # branches due to adaptive step shrinking + cell-length unit issues.
    # With dolfinx's bb_tree + Function.eval we get robust per-tet
    # barycentric evaluation, and a fixed-step RK4 traverses the full
    # nerve geometry without needing to think about step units.
    # ====================================================================
    from dolfinx.geometry import (
        bb_tree, compute_colliding_cells, compute_collisions_points,
    )

    # Streamline-integration knobs from the same FiberSeedConfig
    # parsed at the top of this block — no need to re-read the file.
    n_seeds = _seed_cfg.n_seeds
    seed_end = _seed_cfg.seed_end  # "low" or "high"
    step_um = _seed_cfg.step_um
    max_steps_int = _seed_cfg.max_steps
    _say(f"  streamline seed cfg: n_seeds={n_seeds}, "
         f"seed_end={seed_end!r}, step={step_um:.0f} µm, "
         f"max_steps={max_steps_int}")
    step_m = step_um * 1.0e-6

    # Pick seed vertices. Match the seed_end to the auto-detected
    # trunk / branched ends. Get the boundary facets at that end's caps,
    # gather their vertices, then subsample to n_seeds.
    submesh.topology.create_connectivity(fdim, 0)
    _fv_conn = submesh.topology.connectivity(fdim, 0)
    if seed_end == "low":
        _seed_facet_lists = (caps_lo if single_label == "low-z"
                              else caps_lo)
    else:
        _seed_facet_lists = (caps_hi if single_label == "high-z"
                              else caps_hi)
    # caps_lo / caps_hi may be []; if so just take all boundary facets
    # with axial normals at that z-end.
    if not _seed_facet_lists:
        _say("  WARN: requested seed end has no detected caps; falling "
             "back to all axial-normal facets at that z-extreme")
        _mask = _bot_mask if seed_end == "low" else _top_mask
        _seed_facets_flat = boundary_facets[_mask]
    else:
        _seed_facets_flat = np.concatenate(
            [_fs for _fs, _ in _seed_facet_lists]
        )

    # Per-cluster vertex gathering — track which cap each seed came
    # from so we can pull rim vertices inward toward THAT cap's xy
    # centroid (not the global one — branched ends have multiple caps
    # with different centroids).
    _cluster_seed_xyz = []  # one (k, 3) array per cluster
    _cluster_centroids_xy = []  # one (2,) array per cluster (in xy)
    if _seed_facet_lists:
        _clusters_for_seeds = _seed_facet_lists
    else:
        # Fallback path with a single synthetic cluster covering all
        # axial-normal facets at the chosen z-end.
        _clusters_for_seeds = [(
            _seed_facets_flat,
            np.zeros((len(_seed_facets_flat), 3)),
        )]
    for _fs, _ in _clusters_for_seeds:
        _vset = set()
        for _f in _fs:
            for _v in _fv_conn.links(int(_f)):
                _vset.add(int(_v))
        if not _vset:
            continue
        _vtx = np.array(sorted(_vset), dtype=np.int64)
        _xyz = np.asarray(
            submesh.geometry.x[_vtx], dtype=np.float64
        )
        _cluster_seed_xyz.append(_xyz)
        _cluster_centroids_xy.append(_xyz[:, :2].mean(axis=0))

    # Pull each seed inward toward its OWN cap centroid (xy only). The
    # cap vertex right at the rim sits where the cap meets the lateral
    # wall — the local field direction is ambiguous there, the first
    # RK4 step jumps into the saline-side region where σ is 5× higher,
    # and V_e along the resulting streamline gets a kink that makes
    # the activation function (second derivative) explode to 10×+ the
    # population peak. Moving 15 % of the way from rim → cap centroid
    # in xy lands the seed solidly inside the cap, away from the
    # singularity. The z-coordinate is unchanged so we stay on the
    # cap plane.
    _RIM_PULL = 0.15
    _seed_xyz_list = []
    for _xyz, _cxy in zip(
        _cluster_seed_xyz, _cluster_centroids_xy
    ):
        _xy_pulled = _xyz[:, :2] + _RIM_PULL * (
            _cxy[None, :] - _xyz[:, :2]
        )
        _seed_xyz_list.append(
            np.column_stack([_xy_pulled, _xyz[:, 2]])
        )
    _seed_xyz = (np.vstack(_seed_xyz_list)
                  if _seed_xyz_list else np.zeros((0, 3)))
    _say(f"  {len(_seed_xyz):,} candidate seed vertices on "
         f"{seed_end!r} end (from {len(_seed_facets_flat):,} cap "
         f"facets, axial-normal; pulled {_RIM_PULL*100:.0f}% toward "
         f"cap centroid in xy to avoid rim singularities)")

    # V1 — µCT-bundle: when a fascicle sidecar npz is present,
    # filter cap-end seed candidates to only those inside at
    # least one fascicle endoneurium surface. fibers.py writes
    # the sidecar from `geom.nerve["bundle"]["fascicles"]` so
    # the verts are already in raw frame and units (m). Falls
    # back to the unfiltered set if filtering would leave zero
    # seeds — better to seed in the inter-fascicle epi region
    # than to fail the build outright when a fascicle's verts
    # don't reach the seeding cap.
    _FASC_NPZ = OUT / "nerve_only_fascicle_surfaces.npz"
    if _FASC_NPZ.exists() and len(_seed_xyz):
        try:
            import pyvista as _pv
            _fasc = np.load(_FASC_NPZ)
            _fv = np.asarray(_fasc["verts"], dtype=np.float64)
            _ff = np.asarray(_fasc["faces"], dtype=np.int64)
            _foff = np.asarray(_fasc["offsets"], dtype=np.int64)
            _say(
                f"  fascicle filter: {len(_foff)} surface(s) "
                f"loaded; testing {len(_seed_xyz):,} candidates"
            )
            _probe = _pv.PolyData(
                np.ascontiguousarray(_seed_xyz),
            )
            _inside_any = np.zeros(len(_seed_xyz), dtype=bool)
            for _row in _foff:
                v_lo, v_hi, t_lo, t_hi = (
                    int(_row[0]), int(_row[1]),
                    int(_row[2]), int(_row[3]),
                )
                # Triangles in the concat buffer already reference
                # the global vert buffer; pass it as-is rather
                # than slicing + re-indexing.
                _tris = _ff[t_lo:t_hi]
                n_t = int(_tris.shape[0])
                _flat = np.empty(n_t * 4, dtype=np.int64)
                _flat[0::4] = 3
                _flat[1::4] = _tris[:, 0]
                _flat[2::4] = _tris[:, 1]
                _flat[3::4] = _tris[:, 2]
                _surf = _pv.PolyData(_fv, _flat)
                _sel = _probe.select_enclosed_points(
                    _surf, check_surface=False,
                    tolerance=1.0e-9,
                )
                _inside_any |= np.asarray(
                    _sel["SelectedPoints"], dtype=bool,
                )
            _n_kept = int(_inside_any.sum())
            if _n_kept > 0:
                _seed_xyz = _seed_xyz[_inside_any]
                _say(
                    f"  fascicle filter: kept {_n_kept:,} / "
                    f"{len(_inside_any):,} cap-end candidates "
                    "(inside at least one fascicle)"
                )
            else:
                _say(
                    "  fascicle filter: ZERO cap-end candidates "
                    "land inside any fascicle — keeping "
                    "unfiltered set so the build doesn't fail "
                    "(check that fascicles extend to the seed "
                    "cap)"
                )
        except Exception as _ex:                  # noqa: BLE001
            _say(
                f"  fascicle filter failed ({_ex}); using "
                "unfiltered seed set"
            )

    # Sub-sample to n_seeds. Default: deterministic even-spacing. If
    # FIBER_SEED_RNG is set, draw a random subset with that seed instead —
    # used for seed-count convergence/robustness testing (multiple
    # realizations at fixed n_seeds).
    if len(_seed_xyz) > n_seeds:
        _rng_env = os.environ.get("FIBER_SEED_RNG")
        if _rng_env is not None and _rng_env != "":
            _idx = np.sort(np.random.default_rng(int(_rng_env)).choice(
                len(_seed_xyz), size=n_seeds, replace=False))
        else:
            _idx = np.linspace(0, len(_seed_xyz)-1, n_seeds).astype(int)
        _seed_xyz = _seed_xyz[_idx]
    _say(f"  using {len(_seed_xyz):,} seeds"
         + (f" (rng={os.environ.get('FIBER_SEED_RNG')})"
            if os.environ.get("FIBER_SEED_RNG") else ""))

    # Set up bb_tree once for the whole integration
    _say("  building bb_tree ...")
    _tree = bb_tree(submesh, tdim)
    _say("  bb_tree built")

    def _batch_locate(points):
        """Find first containing cell for each of N points (-1 if none).
        Returns (cells: (N,), valid_mask: (N,))."""
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
        """Evaluate the L2-projected E field at N points. Returns
        (N, 3); rows for out-of-domain points are zero."""
        _cells, _valid = _batch_locate(points)
        _E_all = np.zeros((len(points), 3), dtype=np.float64)
        if _valid.any():
            _pts_in = np.ascontiguousarray(
                points[_valid], dtype=np.float64
            )
            _cells_in = np.ascontiguousarray(
                _cells[_valid], dtype=np.int32
            )
            _vals = E_field.eval(_pts_in, _cells_in)
            _E_all[_valid] = _vals
        return _E_all

    def _normalised(E_arr, eps=1.0e-30):
        """Row-wise unit-direction vectors. Zero rows stay zero."""
        _mag = np.linalg.norm(E_arr, axis=1, keepdims=True)
        return np.where(_mag > eps, E_arr / np.maximum(_mag, eps), 0.0)

    # Decide integration direction: with our Neumann BCs, current flows
    # from the trunk into the domain, so −∇φ (= E) points away from the
    # trunk toward the branches. From trunk seeds we want to FOLLOW E
    # (sign=+1). From branch seeds we want to go AGAINST E (sign=−1).
    if seed_end == single_label.split("-")[0]:  # "low-z" → "low"
        # trunk seed
        _sign = +1.0
        _say(f"  trunk seed → integrating FORWARD along E "
             f"(toward branches)")
    else:
        _sign = -1.0
        _say(f"  branch seed → integrating BACKWARD against E "
             f"(toward trunk)")

    # Fixed-step batched RK4 ------------------------------------------
    _say(f"  integrating {len(_seed_xyz):,} streamlines, "
         f"RK4 step = {step_um:.0f} µm, max_steps = {max_steps_int} "
         f"(≤ {step_m * max_steps_int * 1e3:.0f} mm arc length)")
    _N = len(_seed_xyz)
    _x = _seed_xyz.copy()
    _active = np.ones(_N, dtype=bool)
    _paths = [[_x[_i].copy()] for _i in range(_N)]
    # Optional clamp-on-exit (env FIBER_CLAMP_ON_EXIT=1): instead of killing a
    # streamline whose point steps outside the (thin / curved) nerve, snap it back to
    # the nearest interior vertex — the insulating wall makes E tangential there, so
    # the path slides along the nerve and still reaches the far cap. Kills only after
    # FIBER_MAX_STUCK consecutive failures. Greatly increases full-length yield on
    # thin curved nerves (e.g. the human cervical vagus). Off by default → existing
    # nerves reproduce byte-for-byte.
    _CLAMP = os.environ.get("FIBER_CLAMP_ON_EXIT", "0") == "1"
    _MAX_STUCK = int(os.environ.get("FIBER_MAX_STUCK", "40"))
    if _CLAMP:
        from scipy.spatial import cKDTree as _cKDTree
        _verts_xyz = np.asarray(submesh.geometry.x, dtype=np.float64)
        _vtree = _cKDTree(_verts_xyz)
        _stuck = np.zeros(_N, dtype=int)
        _say(f"  clamp-on-exit ENABLED (snap strays to nearest vertex, "
             f"max_stuck={_MAX_STUCK})")

    import time as _time
    _t_int = _time.time()
    def _fill_invalid(k_new, k_fallback):
        """Where |k_new| is small (point fell outside the mesh or in a
        ~zero region), fall back to k_fallback instead of zero.
        Prevents intermediate RK4 stages from killing streamlines."""
        _mag = np.linalg.norm(k_new, axis=1, keepdims=True)
        _valid = _mag > 0.5
        return np.where(_valid, k_new, k_fallback)

    _step_count = 0
    for _step_i in range(max_steps_int):
        if not _active.any():
            break
        _x_act = _x[_active]
        # k1 is the PRIMARY eval at current position. If it's bad here, this
        # streamline has stopped flowing — kill it (or, with clamp, snap inside).
        _k1 = _normalised(_batch_eval_E(_x_act))
        _mag1 = np.linalg.norm(_k1, axis=1)
        if _CLAMP:
            _lg = np.where(_active)[0]
            _bad = _mag1 < 0.5
            if _bad.any():
                _snap = _vtree.query(_x_act[_bad])[1]
                _x_act[_bad] = _verts_xyz[_snap]
                _x[_lg[_bad]] = _x_act[_bad]
                _stuck[_lg[_bad]] += 1
                _k1 = _normalised(_batch_eval_E(_x_act))
                _mag1 = np.linalg.norm(_k1, axis=1)
            _stuck[_lg[~_bad]] = 0
            _new_dead_local = (_mag1 < 0.5) | (_stuck[_lg] > _MAX_STUCK)
        else:
            _new_dead_local = _mag1 < 0.5

        # k2, k3, k4: intermediate evaluations. If they fall outside
        # the mesh (e.g., a tentative midpoint stepped through a thin
        # branch wall), fall back to the previous good direction
        # rather than killing the streamline.
        _k2_raw = _normalised(_batch_eval_E(
            _x_act + _sign * 0.5 * step_m * _k1
        ))
        _k2 = _fill_invalid(_k2_raw, _k1)
        _k3_raw = _normalised(_batch_eval_E(
            _x_act + _sign * 0.5 * step_m * _k2
        ))
        _k3 = _fill_invalid(_k3_raw, _k2)
        _k4_raw = _normalised(_batch_eval_E(
            _x_act + _sign * step_m * _k3
        ))
        _k4 = _fill_invalid(_k4_raw, _k3)

        # Combined RK4 step
        _dx = _sign * (step_m / 6.0) * (_k1 + 2*_k2 + 2*_k3 + _k4)
        _x_act_new = _x_act + _dx

        # Local indices → global indices
        _local_to_global = np.where(_active)[0]
        for _li, _gi in enumerate(_local_to_global):
            if _new_dead_local[_li]:
                _active[_gi] = False
                continue
            _x[_gi] = _x_act_new[_li]
            _paths[_gi].append(_x[_gi].copy())

        _step_count += 1
        if (_step_i + 1) % 500 == 0:
            _say(f"    step {_step_i+1}/{max_steps_int}, "
                 f"{int(_active.sum())} active, "
                 f"{_time.time()-_t_int:.1f}s elapsed")

    _say(f"  integration done: {_step_count} steps in "
         f"{_time.time()-_t_int:.1f}s, "
         f"{int(_active.sum())}/{_N} reached max_steps without "
         f"stopping")

    # Filter out paths shorter than 5 points (≈1 mm at 200 µm step) —
    # those are degenerate / immediately-killed seeds.
    fiber_paths = [
        np.asarray(_p, dtype=np.float64) for _p in _paths
        if len(_p) >= 5
    ]
    _say(f"  produced {len(fiber_paths):,} fiber paths "
         f"(avg length: "
         f"{np.mean([len(_p) for _p in fiber_paths]):.0f} pts ≈ "
         f"{np.mean([len(_p) for _p in fiber_paths]) * step_um * 1e-3:.1f} mm each)")

    # Transform trajectories from raw frame into cuff frame using the
    # rigid (R, t) we recovered earlier. nerve_studio downstream code
    # expects everything in cuff frame.
    if raw_to_cuff_R is not None:
        _say("  applying raw→cuff rigid transform to trajectories")
        fiber_paths = [
            (_p @ raw_to_cuff_R.T + raw_to_cuff_t)
            for _p in fiber_paths
        ]

    # Save fiber paths as npz with concat layout
    if rank == 0:
        _flat = np.vstack(fiber_paths)
        _lens = np.array([len(_p) for _p in fiber_paths], dtype=np.int64)
        np.savez(
            OUT / "nerve_paths_fibers.npz",
            paths_flat=_flat,
            path_lengths=_lens,
            step_m=np.float64(step_m),
            seed_end=np.array([seed_end]),
            sign=np.float64(_sign),
        )
        _say(f"  wrote {OUT / 'nerve_paths_fibers.npz'} "
             f"({len(fiber_paths)} paths, "
             f"{_flat.shape[0]:,} total pts)")

    # Helper: bring an array of intrinsic-frame midpoints back to
    # cuff frame (intrinsic → submesh → cuff). For the legacy path,
    # submesh frame IS cuff frame so the final step is a no-op.
    def _intr_mids_to_cuff(_mids_intr):
        _sub = _mids_intr @ _R_intr + _centroid
        if raw_to_cuff_R is not None:
            return _sub @ raw_to_cuff_R.T + raw_to_cuff_t
        return _sub

    # Write VTU
    if rank == 0:
        import meshio
        pts = np.asarray(submesh.geometry.x, dtype=np.float64)
        submesh.topology.create_connectivity(tdim, 0)
        conn = submesh.topology.connectivity(tdim, 0)
        cells_arr = conn.array.reshape(-1, 4).astype(np.int64)
        phi_vals = np.asarray(phi_h.x.array, dtype=np.float64)
        E_vals = np.asarray(E_field.x.array, dtype=np.float64).reshape(-1, 3)
        # Bring the mesh into cuff frame so the VTU lines up with the
        # cuff-frame trajectories — rotate both the vertex coordinates
        # and the vector E values.
        if raw_to_cuff_R is not None:
            pts = pts @ raw_to_cuff_R.T + raw_to_cuff_t
            E_vals = E_vals @ raw_to_cuff_R.T
        m_out = meshio.Mesh(
            points=pts,
            cells=[("tetra", cells_arr)],
            point_data={"phi": phi_vals, "E": E_vals},
        )
        m_out.write(str(OUT / "nerve_paths.vtu"))
        _say(f"  wrote {OUT / 'nerve_paths.vtu'} "
             f"({len(pts):,} pts, {len(cells_arr):,} tets)")

        # Cap metadata — useful for the streamline cell to know which
        # end is the trunk (where to seed). Centroids are dumped in
        # CUFF FRAME so they line up with the trajectory endpoints
        # (which nerve_studio uses for branch classification).
        _trunk_mids_intr = (_bot_mids if single_label == "low-z"
                              else _top_mids)
        _trunk_centroid_cuff = _intr_mids_to_cuff(
            _trunk_mids_intr.mean(axis=0, keepdims=True)
        )[0]
        _branch_centroids_cuff = [
            _intr_mids_to_cuff(_m.mean(axis=0, keepdims=True))[0]
            for _, _m in multi_caps
        ]
        _cap_info = {
            "trunk_end": single_label,
            "branched_end": multi_label,
            "n_branch_caps": len(multi_caps),
            "trunk_cap_centroid_m": _trunk_centroid_cuff.tolist(),
            "branch_cap_centroids_m": [
                _c.tolist() for _c in _branch_centroids_cuff
            ],
            "trunk_cap_area_m2": A_trunk,
            "branch_cap_areas_m2": A_branches,
        }
        with open(OUT / "nerve_paths_caps.json", "w") as _fp:
            json.dump(_cap_info, _fp, indent=2)
        _say(f"  wrote cap metadata to nerve_paths_caps.json")

    try:
        with io.XDMFFile(comm, str(OUT / "nerve_paths_field.xdmf"), "w") as _f:
            _f.write_mesh(submesh)
            _f.write_function(phi_h)
            _f.write_function(E_field)
    except Exception as _e:
        _say(f"  (XDMF write failed: {_e!r}; .vtu still written)")

    _say("done")


if __name__ == "__main__":
    main()
