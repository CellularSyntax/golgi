# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""PLC (planar straight line graph) assembly for the multi-domain
nerve + cuff + muscle volume mesh.

Extracted from `golgi/app.py` in step W1.3 of FEATURES.md.

Direct port of nerve_studio.py § 5. The naive "stack capped cylinders
on top of the full nerve surface" approach is replaced by the proper
nerve_studio scheme:
  - clip the nerve into below / middle / above pieces at safe z-planes
  - extract the cut-edge polylines from each clipped piece (so caps
    share byte-identical vertices with the clipped nerve at the seam)
  - build the cuff / silicone / muscle cylinders as discrete
    lateral strips with explicit top/bottom ring indices
  - triangulate annular caps with mapbox_earcut (multi-hole = trunk
    + branches → multiple inner loops)
  - global vertex deduplication on a fine quantized grid + drop
    degenerate (collapsed) triangles

Public surface (single entry point):
  - `assemble_multi_domain_plc(...)` → (plc_pv: pv.PolyData, seeds: dict)

Internal helpers (kept with underscore prefix; the only callers are
within this module + a unit test): `_triangulate_polygon_xy`,
`_triangulate_annulus_xy`, `_triangulate_annulus_xy_multi`,
`_build_cylinder_lateral`, `_open_boundary_polylines`, `_signed_area`,
`_orient`, `_count_self_intersections`,
`_surgical_remove_intersections`, `_preprocess_nerve_surface`,
`_assemble_plc`.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv


def _triangulate_polygon_xy(poly_xy: np.ndarray) -> np.ndarray:
    """Earcut a simple CCW polygon in xy. Returns triangle indices."""
    import mapbox_earcut as ec
    _verts = np.ascontiguousarray(poly_xy, dtype=np.float64)
    _rings = np.array([len(poly_xy)], dtype=np.uint32)
    return np.asarray(
        ec.triangulate_float64(_verts, _rings), dtype=np.int64,
    ).reshape(-1, 3)


def _earcut_annulus_xy_multi(outer_xy: np.ndarray,
                              hole_xys: list,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """[earcut fallback] One CCW outer ring + N CW inner holes →
    (pts2d, tris). Valid but low quality (long slivers); used only when
    the quality path below is unavailable or errors."""
    import mapbox_earcut as ec
    _outer = np.ascontiguousarray(outer_xy, dtype=np.float64)
    _holes = [np.ascontiguousarray(h, dtype=np.float64)
                for h in hole_xys]
    _comb = np.ascontiguousarray(
        np.vstack([_outer] + _holes), dtype=np.float64,
    )
    _rings = [len(_outer)]
    for _h in _holes:
        _rings.append(_rings[-1] + len(_h))
    _rings_arr = np.array(_rings, dtype=np.uint32)
    return _comb, np.asarray(
        ec.triangulate_float64(_comb, _rings_arr),
        dtype=np.int64,
    ).reshape(-1, 3)


def _gmsh_triangulate_xy(outer_xy, hole_xys=(), target_h=None):
    """Robust constrained 2-D triangulation of a CCW outer ring with N inner
    holes, via gmsh (already a golgi dependency — see compute/gmsh_mesher).

    This is the cuff-window cap triangulator. The legacy unconstrained
    `scipy.Delaunay` + centroid-filter (`_quality_triangulate_xy`) silently
    DROPS the ring vertices that sit in the concavities of a non-convex
    hole (the nerve cross-section at an off-trunk / curved cuff), leaving
    T-junctions and cap↔nerve-wall overlap → TetGen self-intersections →
    epsilon-merge collapses the seam → the epineurium region floods into
    the muscle (region 4 disappears). gmsh meshes the planar surface with
    the rings as constraints, so EVERY ring vertex is preserved on the cap
    boundary (verified conforming) with quality interior triangles (no
    slivers, no hole-filling). Boundary curves are transfinite (2 nodes) so
    gmsh keeps the exact input ring vertices instead of resampling them.
    Returns (pts2d (N,2), tris (M,3))."""
    import gmsh

    def _dedup_ring(R, tol):
        # Drop ring vertices that make sub-`tol` edges — the oblique clip
        # leaves µm-scale segments that, fed to gmsh as transfinite (2-node)
        # boundary curves, conflict with the size field and hang the 2-D
        # mesher. Keeps the ring closed and ordered.
        R = np.asarray(R, dtype=np.float64)[:, :2]
        if len(R) < 4:
            return R
        keep = [0]
        for k in range(1, len(R)):
            if np.linalg.norm(R[k] - R[keep[-1]]) > tol:
                keep.append(k)
        out = R[keep]
        while len(out) > 3 and np.linalg.norm(out[-1] - out[0]) <= tol:
            out = out[:-1]
        return out

    outer = _dedup_ring(outer_xy, 8.0e-6)
    holes = [_dedup_ring(h, 8.0e-6) for h in (hole_xys or [])]
    holes = [h for h in holes if len(h) >= 3]
    if target_h is None:
        _e = np.linalg.norm(
            np.diff(np.vstack([outer, outer[:1]]), axis=0), axis=1)
        target_h = float(np.median(_e)) if _e.size else 1.0e-3
    target_h = max(float(target_h), 1.0e-9)
    _started = gmsh.isInitialized()
    if not _started:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("golgi_cap")
        geo = gmsh.model.geo
        loop_tags = []
        pt_id = 1
        ln_id = 1
        for ri, R in enumerate(([outer] + holes)):
            n = len(R)
            pids = []
            for k in range(n):
                geo.addPoint(float(R[k, 0]), float(R[k, 1]), 0.0,
                             target_h, pt_id)
                pids.append(pt_id)
                pt_id += 1
            lids = []
            for k in range(n):
                geo.addLine(pids[k], pids[(k + 1) % n], ln_id)
                geo.mesh.setTransfiniteCurve(ln_id, 2)  # keep exact verts
                lids.append(ln_id)
                ln_id += 1
            geo.addCurveLoop(lids, ri + 1)
            loop_tags.append(ri + 1)
        geo.addPlaneSurface(loop_tags, 1)
        geo.synchronize()
        # CRUCIAL: ignore the per-point sizes — the nerve ring boundary is
        # very fine (~60 µm) and, left on, gmsh propagates that inward,
        # giving a dense cap (~3 k tris) that drives TetGen's volume
        # refinement into a runaway. Disable size-from-points/curvature so
        # the cap INTERIOR stays coarse (~outer-ring spacing) while the
        # transfinite boundary still keeps every exact ring vertex.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_h * 0.7)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_h * 2.0)
        gmsh.option.setNumber("Mesh.Algorithm", 5)        # Delaunay
        gmsh.model.mesh.generate(2)
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        ncoords = ncoords.reshape(-1, 3)
        tag2idx = {int(t): i for i, t in enumerate(ntags)}
        etypes, _etags, enodes = gmsh.model.mesh.getElements(2)
        tris = np.zeros((0, 3), dtype=np.int64)
        for et, en in zip(etypes, enodes):
            if int(et) == 2:                              # 3-node triangle
                conn = en.reshape(-1, 3)
                tris = np.array(
                    [[tag2idx[int(v)] for v in row] for row in conn],
                    dtype=np.int64)
        gmsh.model.remove()
        return ncoords[:, :2], tris
    finally:
        if not _started:
            gmsh.finalize()


def _cdt_annulus_xy_multi(outer_xy, hole_xys=()):
    """Constrained Delaunay triangulation of a CCW outer ring with N inner
    holes, dependency-free (scipy Delaunay + interior Steiner points for
    quality, then constraint-edge recovery + a Delaunay flip pass).

    `_quality_triangulate_xy`'s unconstrained `scipy.Delaunay` +
    centroid-filter silently DROPS the ring vertices that sit in
    concavities of a non-convex hole (the nerve cross-section at an
    off-trunk / curved cuff). Those dropped verts leave T-junctions and
    cap↔nerve-wall overlap that TetGen reports as self-intersections; once
    its epsilon vertex-merge collapses the near-coincident seam triangle, a
    hole opens and the epineurium region floods into the muscle (region 4
    disappears) — only off the straight trunk, where the cross-section is
    non-convex. This routine keeps the Steiner points (good angles) but
    then RECOVERS every missing ring edge by flipping the interior edges it
    crosses, so all ring vertices stay on the cap boundary. After recovery
    the centroid-filter is reliable (no triangle straddles a constraint).
    Falls back to earcut (valid but slivery) if recovery can't complete.
    Returns (pts2d (N,2), tris (M,3))."""
    from scipy.spatial import Delaunay, cKDTree
    from matplotlib.path import Path as _Path
    from collections import defaultdict, deque
    outer = np.ascontiguousarray(np.asarray(outer_xy, dtype=np.float64)[:, :2])
    holes = [np.ascontiguousarray(np.asarray(h, dtype=np.float64)[:, :2])
             for h in (hole_xys or [])]
    rings = [outer] + holes

    # Merge near-coincident ring vertices (clip artifacts < 1 µm) so the
    # triangulation can't make zero-width slivers, remapping the per-ring
    # indices to a shared, deduped vertex array.
    _raw = np.vstack(rings)
    _tree0 = cKDTree(_raw)
    _remap = np.arange(len(_raw))
    for _i in range(len(_raw)):
        if _remap[_i] != _i:
            continue
        for _j in _tree0.query_ball_point(_raw[_i], 1.0e-6):
            if _j > _i and _remap[_j] == _j:
                _remap[_j] = _i
    _uniq, _inv = np.unique(_remap, return_inverse=True)
    bnd = _raw[_uniq]
    n_bnd = len(bnd)
    _old2new = np.empty(len(_raw), dtype=np.int64)
    _old2new = _inv  # maps raw row -> bnd row

    # Constraint edges = consecutive vertex pairs on each ring (deduped).
    cons = set()
    s = 0
    for r in rings:
        rl = len(r)
        for k in range(rl):
            a = int(_old2new[s + k])
            b = int(_old2new[s + (k + 1) % rl])
            if a != b:
                cons.add((min(a, b), max(a, b)))
        s += rl

    # Ring vertices only — the cuff-window annuli have ring spacing finer
    # than lc already, so interior Steiner points aren't needed (and they
    # cost a slow, sliver-prone constraint recovery). The constrained
    # Delaunay of the two rings is a clean triangle strip.
    pts = bnd
    outer_path = _Path(outer)
    hole_paths = [_Path(hh) for hh in holes]

    try:
        tris = Delaunay(pts).simplices.tolist()
    except Exception:                                      # noqa: BLE001
        return _earcut_annulus_xy_multi(outer, holes)

    def _ori(a, b, c):
        return ((pts[b, 0] - pts[a, 0]) * (pts[c, 1] - pts[a, 1])
                - (pts[b, 1] - pts[a, 1]) * (pts[c, 0] - pts[a, 0]))

    def _incirc(a, b, c, d):
        ax, ay = pts[a, 0] - pts[d, 0], pts[a, 1] - pts[d, 1]
        bx, by = pts[b, 0] - pts[d, 0], pts[b, 1] - pts[d, 1]
        cx, cy = pts[c, 0] - pts[d, 0], pts[c, 1] - pts[d, 1]
        return ((ax * ax + ay * ay) * (bx * cy - by * cx)
                - (bx * bx + by * by) * (ax * cy - ay * cx)
                + (cx * cx + cy * cy) * (ax * by - ay * bx))

    def _seg_x(a, b, u, v):
        def _cr(o, x, y):
            return ((pts[x, 0] - pts[o, 0]) * (pts[y, 1] - pts[o, 1])
                    - (pts[x, 1] - pts[o, 1]) * (pts[y, 0] - pts[o, 0]))
        return (_cr(u, v, a) * _cr(u, v, b) < 0
                and _cr(a, b, u) * _cr(a, b, v) < 0)

    tris = [t if _ori(*t) >= 0 else [t[0], t[2], t[1]] for t in tris]

    def _adj():
        e = defaultdict(list)
        for ti, t in enumerate(tris):
            if t is None:
                continue
            a, b, c = t
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
                e[(min(u, v), max(u, v))].append((ti, w))
        return e

    e2t = _adj()

    def _flip(ek):
        lst = e2t.get(ek)
        if not lst or len(lst) != 2:
            return False
        (t1, w1), (t2, w2) = lst
        a, b = ek
        if _ori(a, b, w1) * _ori(a, b, w2) >= 0:
            return False
        if _ori(w1, w2, a) * _ori(w1, w2, b) >= 0:    # non-convex quad
            return False
        n1 = [w1, w2, a]
        n2 = [w2, w1, b]
        if _ori(*n1) < 0:
            n1 = [n1[0], n1[2], n1[1]]
        if _ori(*n2) < 0:
            n2 = [n2[0], n2[2], n2[1]]
        tris[t1] = n1
        tris[t2] = n2
        return True

    # Recover each missing constraint edge by flipping the interior edges
    # that cross it.
    for ek in list(cons):
        if ek in e2t:
            continue
        ca, cb = ek
        guard = 0
        while ek not in e2t and guard < 4 * n_bnd + 200:
            guard += 1
            moved = False
            for ee in list(e2t.keys()):
                if ee in cons:
                    continue
                u, v = ee
                if u in (ca, cb) or v in (ca, cb):
                    continue
                if _seg_x(ca, cb, u, v) and _flip(ee):
                    e2t = _adj()
                    moved = True
                    break
            if not moved:
                break

    if any(ek not in e2t for ek in cons):
        return _earcut_annulus_xy_multi(outer, holes)

    # Delaunay flip pass for quality (never crossing a constraint).
    q = deque(k for k in e2t if k not in cons and len(e2t[k]) == 2)
    inq = set(q)
    nf = 0
    max_flip = 50 * len(tris) + 1000
    while q and nf < max_flip:
        ek = q.popleft()
        inq.discard(ek)
        if ek in cons:
            continue
        lst = e2t.get(ek)
        if not lst or len(lst) != 2:
            continue
        (t1, w1), (t2, w2) = lst
        a, b = ek
        if _ori(a, b, w1) * _ori(a, b, w2) >= 0:
            continue
        if _ori(w1, w2, a) * _ori(w1, w2, b) >= 0:
            continue
        if _incirc(a, b, w1, w2) <= 1e-24:
            continue
        if not _flip(ek):
            continue
        e2t = _adj()
        nf += 1
        for ne in (
            (min(w1, a), max(w1, a)), (min(a, w2), max(a, w2)),
            (min(w2, b), max(w2, b)), (min(b, w1), max(b, w1)),
        ):
            if ne not in cons and ne not in inq and len(e2t.get(ne, [])) == 2:
                q.append(ne)
                inq.add(ne)

    # Centroid filter (reliable now: no triangle straddles a constraint).
    tris = [t for t in tris if t is not None]
    cent = np.array([pts[t].mean(0) for t in tris])
    keep = outer_path.contains_points(cent)
    for hp in hole_paths:
        keep &= ~hp.contains_points(cent)
    out = [t for t, k in zip(tris, keep)
           if k and abs(_ori(*t)) > 1e-20]     # drop exact-degenerate tris
    if not out:
        return _earcut_annulus_xy_multi(outer, holes)
    return pts, np.asarray(out, dtype=np.int64)


def _quality_triangulate_xy(outer_xy, hole_xys=(), *, target_h=None):
    """Quality 2D triangulation of a CCW polygon (optionally with inner
    holes), preserving ALL input boundary vertices exactly so the shared
    rings stay conforming with the cylinder pieces after the assembly
    dedup. Interior Steiner points are sprinkled on an offset (hex-like)
    grid at ~`target_h` spacing and the union is Delaunay-triangulated —
    Delaunay maximises the minimum angle, so the output has none of the
    near-zero-area slivers `mapbox_earcut` produces (those degenerate
    facets are what trip TetGen's `recoversubfaces` boundary recovery).
    Triangles whose centroid falls outside the region (or inside a hole)
    are dropped. Returns (pts2d (N,2), tris (M,3)). Falls back to earcut
    on any failure so the PLC build never hard-stops.
    """
    outer = np.asarray(outer_xy, dtype=np.float64)[:, :2]
    holes = [np.asarray(h, dtype=np.float64)[:, :2]
             for h in (hole_xys or [])]
    try:
        from scipy.spatial import Delaunay, cKDTree
        from matplotlib.path import Path as _Path
        bnd = np.vstack([outer] + holes)
        if target_h is None:
            _e = np.linalg.norm(
                np.diff(np.vstack([outer, outer[:1]]), axis=0), axis=1,
            )
            target_h = float(np.median(_e)) if _e.size else 0.0
        h = max(float(target_h), 1e-9)
        lo = bnd.min(0)
        hi = bnd.max(0)
        xs = np.arange(lo[0] + 0.5 * h, hi[0], h)
        ys = np.arange(lo[1] + 0.5 * h, hi[1], 0.866 * h)
        if xs.size and ys.size:
            gx, gy = np.meshgrid(xs, ys)
            gx = gx.copy()
            gx[1::2] += 0.5 * h                # hex offset alt rows
            grid = np.column_stack([gx.ravel(), gy.ravel()])
        else:
            grid = np.empty((0, 2))
        outer_path = _Path(outer)
        hole_paths = [_Path(hh) for hh in holes]
        if grid.shape[0]:
            inside = outer_path.contains_points(grid)
            for hp in hole_paths:
                inside &= ~hp.contains_points(grid)
            grid = grid[inside]
            if grid.shape[0]:                  # not too close to boundary
                d, _ = cKDTree(bnd).query(grid, k=1)
                grid = grid[d > 0.5 * h]
        pts = np.vstack([bnd, grid]) if grid.shape[0] else bnd
        simp = Delaunay(pts).simplices
        cent = pts[simp].mean(axis=1)
        keep = outer_path.contains_points(cent)
        for hp in hole_paths:
            keep &= ~hp.contains_points(cent)
        simp = simp[keep]
        if simp.shape[0] == 0:
            raise RuntimeError("quality triangulation produced no tris")
        return pts, simp.astype(np.int64)
    except Exception:                                      # noqa: BLE001
        if holes:
            return _earcut_annulus_xy_multi(outer, holes)
        return outer, _triangulate_polygon_xy(outer)


def _triangulate_annulus_xy_multi(outer_xy: np.ndarray,
                                    hole_xys: list,
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """One CCW outer ring + N CW inner holes → (pts2d, tris). Now routed
    through the quality (Delaunay) triangulator so cuff caps don't carry
    earcut slivers into the PLC; preserves the input ring vertices so the
    caps stay conforming with the cylinders. Critical for cuff caps when
    the nerve cross-section has multiple disconnected loops (trunk +
    branch).

    With holes present (cuff-window caps over a non-convex nerve cross-
    section), routes through the constrained Delaunay (`_cdt_annulus_xy_
    multi`) so EVERY ring vertex stays on the cap boundary — the
    unconstrained `_quality_triangulate_xy` drops ring verts in concavities
    at off-trunk cuff positions, which is what makes the muscle region
    merge into the epineurium. Gated by GOLGI_PLC_CDT (default on); set to
    "0" to fall back to the legacy unconstrained path."""
    import os as _os
    _holes = list(hole_xys)
    if _holes and _os.environ.get("GOLGI_PLC_CDT", "1") != "0":
        # gmsh constrained mesher: conforming + sliver-free (primary).
        try:
            _p, _t = _gmsh_triangulate_xy(outer_xy, _holes)
            if len(_t):
                return _p, _t
        except Exception:                                  # noqa: BLE001
            pass
        # Dependency-free constrained-Delaunay recovery (fallback).
        try:
            return _cdt_annulus_xy_multi(outer_xy, _holes)
        except Exception:                                  # noqa: BLE001
            pass
    return _quality_triangulate_xy(outer_xy, _holes)


def _triangulate_annulus_xy(outer_xy: np.ndarray,
                              hole_xy: np.ndarray,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Single-hole convenience wrapper."""
    return _triangulate_annulus_xy_multi(outer_xy, [hole_xy])


def _build_cylinder_lateral(R: float, z_lo: float, z_hi: float,
                              n_circ: int = 72, n_axial: int = 1,
                              x_c: float = 0.0,
                              y_c: float = 0.0,
                              ) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """Lateral surface of an axis-aligned cylinder, tessellated as a
    rectangular strip. Returns (pts, tris, bottom_ring_idx,
    top_ring_idx). The ring index arrays let downstream cap code
    reuse the EXACT cylinder vertices on the seam — no near-duplicates
    for TetGen to choke on."""
    _theta = 2 * np.pi * np.arange(n_circ) / n_circ
    _pts: list = []
    for _k in range(n_axial + 1):
        _z = z_lo + (z_hi - z_lo) * _k / n_axial
        for _t in _theta:
            _pts.append([
                x_c + R * np.cos(_t),
                y_c + R * np.sin(_t),
                _z,
            ])
    _pts = np.asarray(_pts, dtype=np.float64)
    _tris: list = []
    for _k in range(n_axial):
        _r0 = _k * n_circ
        _r1 = (_k + 1) * n_circ
        for _i in range(n_circ):
            _j = (_i + 1) % n_circ
            _tris.append([_r0 + _i, _r0 + _j, _r1 + _j])
            _tris.append([_r0 + _i, _r1 + _j, _r1 + _i])
    _tris = np.asarray(_tris, dtype=np.int64)
    _bot = np.arange(n_circ, dtype=np.int64)
    _top = np.arange(n_axial * n_circ,
                      (n_axial + 1) * n_circ, dtype=np.int64)
    return _pts, _tris, _bot, _top


def _open_boundary_polylines(pd: "pv.PolyData") -> list:
    """Extract open-boundary loops of a clipped PolyData (the cut
    edges from a pv.clip(...)) as a list of (N, 3) np.ndarray
    polylines, sorted by length descending.

    Why this matters: vtkClipPolyData and vtkCutter compute plane
    intersections independently and can produce vertices that are
    spatially identical but not byte-identical. Pulling the cap
    polyline from the clipped piece's OWN open boundary guarantees
    the cap and the clipped nerve share the same vertices at the
    seam, eliminating the 'segment and facet intersect' class of
    TetGen failure."""
    import vtk
    _fe = vtk.vtkFeatureEdges()
    _fe.SetInputData(pd)
    _fe.BoundaryEdgesOn()
    _fe.FeatureEdgesOff()
    _fe.NonManifoldEdgesOff()
    _fe.ManifoldEdgesOff()
    _fe.Update()
    _edges_out = _fe.GetOutput()
    _strip = vtk.vtkStripper()
    _strip.SetInputData(_edges_out)
    _strip.JoinContiguousSegmentsOn()
    _strip.Update()
    _poly = _strip.GetOutput()
    _pts_data = _poly.GetPoints().GetData() if _poly.GetPoints() else None
    if _pts_data is None:
        return []
    _pts = np.asarray(_pts_data)
    _lines = _poly.GetLines()
    _lines.InitTraversal()
    _id = vtk.vtkIdList()
    _loops: list = []
    while _lines.GetNextCell(_id):
        _idx = [_id.GetId(i) for i in range(_id.GetNumberOfIds())]
        if len(_idx) < 3:
            continue
        _pl = _pts[_idx]
        # Drop near-duplicate consecutive points
        _keep = [0]
        for _i in range(1, len(_pl)):
            if np.linalg.norm(_pl[_i] - _pl[_keep[-1]]) > 1.0e-6:
                _keep.append(_i)
        if len(_keep) >= 3:
            _loops.append(_pl[_keep])
    _loops.sort(key=len, reverse=True)
    return _loops


def _fill_small_boundary_holes(pts, tris, max_edges=8, on_line=None):
    """Close tiny boundary holes (<= max_edges) where independently-built cap /
    lateral pieces meet. On curved (off-trunk) cuff placements the cap planes cut
    the nerve obliquely and a single triangle can be missing at a seam; that
    leaves a region non-watertight, so TetGen's region flood leaks and its
    refinement never terminates. Each connected boundary chain/loop is found;
    small ones are fan-triangulated, large ones (intended open ends) untouched.
    Returns (pts, tris, n_added)."""
    from collections import defaultdict
    from scipy.spatial import cKDTree
    ff = np.hstack([np.full((len(tris), 1), 3, np.int64),
                    np.asarray(tris, np.int64)]).ravel()
    pd = pv.PolyData(np.asarray(pts, float), ff).clean(
        tolerance=1e-9, absolute=True)
    P = np.asarray(pd.points, float)
    T = pd.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    be = pd.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False)
    if be.n_cells == 0:
        return P, T, 0
    bmap = cKDTree(P).query(np.asarray(be.points))[1]
    lines = be.lines.reshape(-1, 3)[:, 1:]
    adj = defaultdict(list)
    for a, b in lines:
        ia, ib = int(bmap[a]), int(bmap[b])
        adj[ia].append(ib); adj[ib].append(ia)
    seen = set(); newtris = []
    for start in list(adj):
        if start in seen:
            continue
        stack = [start]; comp = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x); stack.extend(adj[x])
        n_e = sum(len(adj[x]) for x in comp) // 2
        if n_e > max_edges:
            continue                                       # large opening: leave it
        ends = [x for x in comp if len(adj[x]) == 1]
        cur = ends[0] if ends else next(iter(comp))
        order = [cur]; prev = None
        while True:
            nxts = [y for y in adj[cur]
                    if y != prev and not (y == order[0] and len(order) > 1)]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            if cur == order[0] or len(order) > len(comp):
                break
            order.append(cur)
        for k in range(1, len(order) - 1):
            newtris.append([order[0], order[k], order[k + 1]])
    if not newtris:
        return P, T, 0
    if on_line:
        on_line(f"  fill small boundary holes: +{len(newtris)} tris")
    return P, np.vstack([T, np.asarray(newtris, np.int64)]), len(newtris)


def _weld_close_verts(pts, tris, tol):
    """Merge vertices closer than `tol` (union-find over a KD-tree radius
    query), remap triangles to the cluster representative, and drop the
    triangles that collapse to a degenerate (repeated-vertex) result.

    The oblique cuff-window clip of a curved nerve leaves sub-µm micro-
    edges in the clip ring; left in, TetGen's boundary recovery chases
    them with an unbounded Steiner-point cascade (runaway) or flags them
    as self-intersections. pyvista's `clean(point_merging=...)` does NOT
    reliably merge them here (it leaves the tri count unchanged), so this
    does the merge explicitly. Returns (pts, tris, n_dropped)."""
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    if len(pts) == 0 or len(tris) == 0:
        return pts, tris, 0
    parent = np.arange(len(pts))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = cKDTree(pts).query_pairs(r=float(tol), output_type="ndarray")
    for a, b in pairs:
        ra, rb = _find(int(a)), _find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    root = np.array([_find(i) for i in range(len(pts))], dtype=np.int64)
    uniq, inv = np.unique(root, return_inverse=True)
    newpts = pts[uniq]
    newtris = inv[tris]
    good = (
        (newtris[:, 0] != newtris[:, 1])
        & (newtris[:, 1] != newtris[:, 2])
        & (newtris[:, 2] != newtris[:, 0])
    )
    return newpts, newtris[good], int((~good).sum())


def _signed_area(xy: np.ndarray) -> float:
    _x, _y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(
        _x * np.roll(_y, -1) - np.roll(_x, -1) * _y,
    ))


def _orient(xy: np.ndarray, ccw: bool = True) -> np.ndarray:
    return xy if ((_signed_area(xy) > 0) == ccw) else xy[::-1]


def _manual_axial_clip(
    pts: np.ndarray,
    tris: np.ndarray,
    z_lo: float,
    z_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip a triangle mesh against the half-spaces z ≥ z_lo and
    z ≤ z_hi using a pure-numpy plane-clip pass per triangle.
    Returns (new_pts, new_tris) with vertices on the cut planes
    inserted exactly at z = z_lo / z = z_hi.

    Used as a fallback for `_clip_fascicle_to_z_window` when
    pyvista's `surf.clip(...)` (Viskores/VTK) silently returns
    empty on inputs it can't handle — typical for the
    non-manifold fascicle surfaces the µCT segmentation
    pipeline produces.

    Per triangle (classified by how many vertices are inside
    `[z_lo, z_hi]`):
      * 3 in  → kept verbatim.
      * 0 in  → dropped.
      * 1 in  → one sub-triangle anchored at the inside vertex,
                with two new vertices interpolated onto the cut
                plane along the two outgoing edges.
      * 2 in  → two sub-triangles forming the quadrilateral
                between the two inside vertices and the two
                edge-clip points.

    Caching of new edge-clip vertices ensures the resulting mesh
    is watertight along the cut — `_open_boundary_polylines`
    downstream picks up the cut polygon as one or two clean
    loops which the existing earcut path triangulates into
    caps.

    The two clip planes are applied sequentially so a triangle
    that straddles BOTH planes (the fascicle pokes through the
    cuff from both ends) is handled correctly: clip against
    z_hi first → remove or shrink the upper part; then clip
    against z_lo on the result."""
    pts = np.asarray(pts, dtype=np.float64).copy()
    tris = np.asarray(tris, dtype=np.int64)
    if tris.shape[0] == 0:
        return pts, tris

    def _clip_one_plane(
        p: np.ndarray, t: np.ndarray,
        z_cut: float, keep_below: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if t.shape[0] == 0:
            return p, t
        # sign > 0 means "outside" (to be clipped away).
        sign = (p[:, 2] - z_cut) if keep_below else (z_cut - p[:, 2])
        inside = sign <= 0.0
        # New vertices accumulate here; cache keyed by sorted
        # (inside_idx, outside_idx) so the same edge-clip vertex
        # gets reused across adjacent triangles → watertight cut.
        p_list = [p]
        new_verts: list[np.ndarray] = []
        n_old = int(p.shape[0])
        cache: dict[tuple[int, int], int] = {}

        def _edge_vert(i_a: int, i_b: int) -> int:
            key = (min(i_a, i_b), max(i_a, i_b))
            if key in cache:
                return cache[key]
            z_a = float(p[i_a, 2])
            z_b = float(p[i_b, 2])
            denom = z_b - z_a
            if abs(denom) < 1.0e-15:
                # Degenerate edge — shouldn't happen for
                # non-degenerate input, but guard anyway.
                u = 0.5
            else:
                u = (z_cut - z_a) / denom
            u = max(0.0, min(1.0, u))
            new_pt = p[i_a] + u * (p[i_b] - p[i_a])
            new_pt[2] = float(z_cut)   # snap to cut plane exactly
            idx = n_old + len(new_verts)
            new_verts.append(new_pt)
            cache[key] = idx
            return idx

        out_tris: list[list[int]] = []
        for tri in t:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ia, ib, ic = (
                bool(inside[a]), bool(inside[b]), bool(inside[c]),
            )
            n_in = int(ia) + int(ib) + int(ic)
            if n_in == 3:
                out_tris.append([a, b, c])
                continue
            if n_in == 0:
                continue
            # Rotate so the "inside" vertices come first while
            # preserving winding (a → b → c).
            if n_in == 1:
                if ia:
                    v_in, v_o1, v_o2 = a, b, c
                elif ib:
                    v_in, v_o1, v_o2 = b, c, a
                else:
                    v_in, v_o1, v_o2 = c, a, b
                k1 = _edge_vert(v_in, v_o1)
                k2 = _edge_vert(v_in, v_o2)
                # Match original winding: v_in → (where v_o1
                # used to be) → (where v_o2 used to be).
                out_tris.append([v_in, k1, k2])
            else:  # n_in == 2
                if not ia:
                    v_o, v_i1, v_i2 = a, b, c
                elif not ib:
                    v_o, v_i1, v_i2 = b, c, a
                else:
                    v_o, v_i1, v_i2 = c, a, b
                # Edge-clip vertices, one per outgoing edge.
                k1 = _edge_vert(v_i1, v_o)
                k2 = _edge_vert(v_i2, v_o)
                # Quadrilateral (v_i1, v_i2, k2, k1) split into
                # two triangles preserving the original winding.
                out_tris.append([v_i1, v_i2, k2])
                out_tris.append([v_i1, k2, k1])

        if new_verts:
            p_list.append(
                np.asarray(new_verts, dtype=np.float64),
            )
        return (
            np.concatenate(p_list, axis=0),
            (
                np.asarray(out_tris, dtype=np.int64)
                if out_tris
                else np.empty((0, 3), dtype=np.int64)
            ),
        )

    # Plane 1: z ≤ z_hi (keep below).
    pts, tris = _clip_one_plane(pts, tris, z_hi, keep_below=True)
    if tris.shape[0] == 0:
        return pts, np.empty((0, 3), dtype=np.int64)
    # Plane 2: z ≥ z_lo (keep above).
    pts, tris = _clip_one_plane(pts, tris, z_lo, keep_below=False)
    if tris.shape[0] == 0:
        return pts, np.empty((0, 3), dtype=np.int64)
    # Strip unreferenced vertices left behind by the two clip
    # passes. Without this, the result carries every input point
    # even when only a small subset survives — wastes memory in
    # the downstream PolyData and confuses sanity diagnostics on
    # the (xmin, xmax) bbox.
    used = np.unique(tris.flatten())
    remap = -np.ones(pts.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    return pts[used].copy(), remap[tris]


def _count_self_intersections(pts: np.ndarray,
                                 faces: np.ndarray,
                                 ) -> tuple[int, np.ndarray]:
    """Returns (count, bad_indices) for self-intersecting triangles.
    Uses pymeshfix.PyTMesh — most reliable tri-tri intersection
    test in our toolchain. Returns (0, empty) if the detector
    isn't available.

    pymeshfix.PyTMesh.select_intersecting_triangles() returns a
    (N, 3) ndarray where the FIRST column is the triangle index;
    the other two columns hold edge/pair info we don't need.
    Earlier wrappers flattened the whole thing into a 1D array
    and indexed faces with it — which crashed with out-of-bounds
    errors because columns 1-2 contain values in the millions
    (pair handles, not indices). This version extracts column 0
    correctly and bounds-checks against `len(faces)` defensively.
    """
    try:
        from pymeshfix import PyTMesh as _PyTMesh
        m = _PyTMesh()
        m.load_array(
            np.ascontiguousarray(pts, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.int32),
        )
        bad = m.select_intersecting_triangles()
        if bad is None:
            return 0, np.array([], dtype=np.int64)
        bad_arr = np.asarray(bad, dtype=np.int64)
        if bad_arr.size == 0:
            return 0, np.array([], dtype=np.int64)
        n_rows = int(bad_arr.shape[0])
        if bad_arr.ndim == 2 and bad_arr.shape[1] >= 1:
            indices = bad_arr[:, 0]
        else:
            indices = bad_arr.ravel()
        n_faces = int(faces.shape[0])
        in_bounds = (indices >= 0) & (indices < n_faces)
        indices = np.unique(indices[in_bounds])
        # n_rows preserves the "intersecting-pair count" the
        # existing logs use; indices is the deduped + bounds-
        # checked set of bad tri indices suitable for repair.
        return n_rows, indices
    except Exception:
        return 0, np.array([], dtype=np.int64)


def _iterative_si_repair(
    pts: np.ndarray,
    tris: np.ndarray,
    *,
    max_cycles: int = 3,
    on_log=None,
    tag: str = "",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Three-tool cycling SI-removal pass for stubborn surfaces.

    Per cycle:
      1. `MeshFix.repair(joincomp=False, remove_smallest_components=
         False)` — bulk SI removal + small-boundary fill. Catches
         the easy cases (~90% of SI in cap-stitched fascicles).
      2. `PyTMesh.clean(max_iters=20, inner_loops=10)` — iterative
         edge-collapse pass. Aggressively chases pairs MeshFix
         couldn't dislodge.
      3. Surgical drop of any still-flagged tris + small-boundary
         fill, using the bounds-checked indices from
         `_count_self_intersections`. Last resort for the
         remaining 1-3 stubborn pairs.

    Re-counts SI between each step and short-circuits at zero.
    Returns `(pts_out, tris_out, n_si_out)`. Always returns the
    best result seen across cycles (never makes the SI worse).
    `on_log(msg)` is called for per-step progress so the caller
    can route to its `say(...)` channel.
    """
    def _say(msg: str) -> None:
        if on_log is not None:
            try:
                on_log(msg)
            except Exception:                             # noqa: BLE001
                pass
    cur_pts = np.asarray(pts, dtype=np.float64)
    cur_tris = np.asarray(tris, dtype=np.int64)
    best_pts, best_tris = cur_pts, cur_tris
    best_si, _ = _count_self_intersections(cur_pts, cur_tris)
    if best_si == 0:
        return cur_pts, cur_tris, 0

    # M39-B — Catastrophic-shred guard. If a repair step drops
    # the triangle count below this fraction of the input, the
    # tool has "fixed" SI by gutting the surface rather than
    # untangling it. Reject and treat as a failed step so the
    # caller keeps the prior best mesh instead of a 90%-shredded
    # remnant. (User hit this on a fascicle with 3,591 phantom
    # SIs from jagged-edge cap stitching: MeshFix.repair "fixed"
    # it by going 6,589 → 628 tris and the function still
    # returned the gutted result because n_si=0 < 3591.)
    n_tris_in = int(cur_tris.shape[0])
    _MIN_KEEP_FRAC = 0.5

    try:
        import pymeshfix as _pmf
        from pymeshfix import PyTMesh as _PyTMesh
    except ImportError:
        return cur_pts, cur_tris, best_si

    for cycle in range(int(max_cycles)):
        # ---- Step A: MeshFix.repair ----
        try:
            mfx = _pmf.MeshFix(
                np.ascontiguousarray(cur_pts, dtype=np.float64),
                np.ascontiguousarray(cur_tris, dtype=np.int32),
            )
            mfx.repair(
                joincomp=False,
                remove_smallest_components=False,
            )
            new_pts = np.asarray(
                mfx.mesh.points, dtype=np.float64,
            )
            new_tris = (
                np.asarray(mfx.mesh.faces, dtype=np.int64)
                .reshape(-1, 4)[:, 1:]
            )
            if (
                new_tris.shape[0]
                < _MIN_KEEP_FRAC * n_tris_in
            ):
                _say(
                    f"{tag}cycle {cycle + 1} step A "
                    f"(MeshFix.repair): SHRED REJECTED "
                    f"({n_tris_in:,} → {new_tris.shape[0]:,} "
                    "tris, < 50% of input — keeping prior best)"
                )
                # Don't accept this result; cur_* unchanged.
            else:
                cur_pts, cur_tris = new_pts, new_tris
                n_si, _ = _count_self_intersections(
                    cur_pts, cur_tris,
                )
                _say(
                    f"{tag}cycle {cycle + 1} step A "
                    f"(MeshFix.repair): "
                    f"{cur_tris.shape[0]:,} tris, "
                    f"SI={n_si}"
                )
                if n_si < best_si:
                    best_pts, best_tris, best_si = (
                        cur_pts, cur_tris, n_si,
                    )
                if n_si == 0:
                    return best_pts, best_tris, 0
        except Exception as ex:                           # noqa: BLE001
            _say(
                f"{tag}cycle {cycle + 1} step A failed: {ex}"
            )

        # ---- Step B: PyTMesh.clean ----
        try:
            m = _PyTMesh()
            m.load_array(
                np.ascontiguousarray(cur_pts, dtype=np.float64),
                np.ascontiguousarray(cur_tris, dtype=np.int32),
            )
            try:
                m.fill_small_boundaries(refine=True, nbe=100)
            except Exception:                             # noqa: BLE001
                pass
            try:
                m.clean(max_iters=20, inner_loops=10)
            except Exception:                             # noqa: BLE001
                pass
            v, f = m.return_arrays()
            new_pts = np.asarray(v, dtype=np.float64)
            new_tris = np.asarray(f, dtype=np.int64)
            if (
                new_tris.shape[0]
                < _MIN_KEEP_FRAC * n_tris_in
            ):
                _say(
                    f"{tag}cycle {cycle + 1} step B "
                    f"(PyTMesh.clean): SHRED REJECTED "
                    f"({n_tris_in:,} → {new_tris.shape[0]:,} "
                    "tris, < 50% of input — keeping prior best)"
                )
            else:
                cur_pts, cur_tris = new_pts, new_tris
                n_si, _ = _count_self_intersections(
                    cur_pts, cur_tris,
                )
                _say(
                    f"{tag}cycle {cycle + 1} step B "
                    f"(PyTMesh.clean): "
                    f"{cur_tris.shape[0]:,} tris, "
                    f"SI={n_si}"
                )
                if n_si < best_si:
                    best_pts, best_tris, best_si = (
                        cur_pts, cur_tris, n_si,
                    )
                if n_si == 0:
                    return best_pts, best_tris, 0
        except Exception as ex:                           # noqa: BLE001
            _say(
                f"{tag}cycle {cycle + 1} step B failed: {ex}"
            )

        # ---- Step C: surgical drop + refill ----
        try:
            n_si, bad = _count_self_intersections(
                cur_pts, cur_tris,
            )
            if n_si == 0 or bad.size == 0:
                if n_si == 0:
                    return best_pts, best_tris, 0
                _say(
                    f"{tag}cycle {cycle + 1} step C skipped "
                    "(no in-bounds bad indices)"
                )
            else:
                new_pts, new_tris = (
                    _surgical_remove_intersections(
                        cur_pts, cur_tris, bad,
                    )
                )
                if (
                    new_tris.shape[0]
                    < _MIN_KEEP_FRAC * n_tris_in
                ):
                    _say(
                        f"{tag}cycle {cycle + 1} step C "
                        f"(surgical drop+fill): SHRED REJECTED "
                        f"({n_tris_in:,} → "
                        f"{new_tris.shape[0]:,} tris, "
                        "< 50% of input — keeping prior best)"
                    )
                else:
                    cur_pts, cur_tris = new_pts, new_tris
                    n_si, _ = _count_self_intersections(
                        cur_pts, cur_tris,
                    )
                    _say(
                        f"{tag}cycle {cycle + 1} step C "
                        f"(surgical drop+fill): "
                        f"{cur_tris.shape[0]:,} "
                        f"tris, SI={n_si}"
                    )
                    if n_si < best_si:
                        best_pts, best_tris, best_si = (
                            cur_pts, cur_tris, n_si,
                        )
                    if n_si == 0:
                        return best_pts, best_tris, 0
        except Exception as ex:                           # noqa: BLE001
            _say(
                f"{tag}cycle {cycle + 1} step C failed: {ex}"
            )

    return best_pts, best_tris, best_si


def _surgical_remove_intersections(
    pts: np.ndarray,
    faces: np.ndarray,
    bad_indices: np.ndarray,
    *,
    drop_small_components: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Delete intersecting triangles + refill the resulting micro
    holes via pymeshfix's hole-filling. Used as a last-resort
    when MeshFix.repair() / PyTMesh.clean() can't dislodge the
    last 1-3 stubborn pairs.

    `drop_small_components=True` is the legacy behaviour
    (`m.remove_smallest_components()` keeps only the largest
    connected component). That's WRONG for multi-region PLCs
    where the epi, saline, silicone, muscle, and per-fascicle
    surfaces are by construction disconnected — running
    remove_smallest_components on the assembled PLC strips
    everything except the largest single region (the 82% shred
    we kept seeing). Default is now False so the post-assembly
    + per-fascicle paths preserve every region; opt back in
    only when the caller knows the input is a single component.
    """
    from pymeshfix import PyTMesh as _PyTMesh
    # Defensive bounds + dedup — `bad_indices` arrives from
    # _count_self_intersections which now filters and dedups,
    # but other call sites may pass raw arrays.
    bi = np.asarray(bad_indices, dtype=np.int64).ravel()
    bi = bi[(bi >= 0) & (bi < int(faces.shape[0]))]
    bi = np.unique(bi)
    keep = np.ones(len(faces), dtype=bool)
    keep[bi] = False
    faces_kept = faces[keep]
    m = _PyTMesh()
    m.load_array(
        np.ascontiguousarray(pts, dtype=np.float64),
        np.ascontiguousarray(faces_kept, dtype=np.int32),
    )
    try:
        m.fill_small_boundaries(refine=True, nbe=100)
    except Exception:
        pass
    try:
        m.clean(max_iters=20, inner_loops=10)
    except Exception:
        pass
    if drop_small_components:
        m.remove_smallest_components()
    v, f = m.return_arrays()
    return (np.asarray(v, dtype=np.float64),
            np.asarray(f, dtype=np.int64))


def _preprocess_nerve_surface(pts: np.ndarray,
                                tris: np.ndarray,
                                target_tris: int = 50_000,
                                n_passes: int = 2,
                                on_line=None,
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of nerve_studio.py § 5 nerve-surface cleanup. Without
    this, raw .nas surfaces routinely contain sliver triangles + a
    handful of near-collinear edge pairs that make TetGen's boundary
    recovery either skip facets or segfault outright.

    Steps:
      1. Decimate to `target_tris` (volume-preserving) if denser.
      2. `n_passes` × (Taubin smooth → pymeshfix.MeshFix.repair).
      3. Final pv.clean(tolerance=1 µm, absolute=True).

    Optimesh / Lloyd point relaxation is intentionally skipped —
    nerve_studio.py uses it only when the optional `optimesh`
    package is available, and the cleanup converges without it on
    every .nas we've tested.
    """
    import pymeshfix
    say = on_line if on_line is not None else (lambda *_: None)

    def _pv(p, t):
        n = len(t)
        flat = np.empty(n * 4, dtype=np.int64)
        flat[0::4] = 3
        flat[1::4] = t[:, 0]
        flat[2::4] = t[:, 1]
        flat[3::4] = t[:, 2]
        return pv.PolyData(np.asarray(p, dtype=np.float64), flat)

    raw = _pv(pts, tris)
    say(f"  nerve preprocessing: {raw.n_points:,} pts, "
         f"{raw.n_faces:,} tris (target {target_tris:,} tris)")

    # 1. Decimate
    # M48 — SKIP decimation when input is already watertight.
    # pyvista.decimate's edge-collapse algorithm produces
    # degenerate / duplicate / inverted triangles on the
    # regular-grid structure of polygon-extrude meshes (the
    # watertight prisms the histology-bundle + µCT bundle
    # paths produce). User report on a 136 k-tri polygon-
    # extrude epi:
    #
    #   trimesh defensive pass: dropped 12 153 degenerate /
    #     duplicate / inverted tris (37 847 remain)
    #   PyTMesh.clean(50): 17 585 SI → 0 SI but ends with
    #     0 pts 0 tris  →  ValueError downstream.
    #
    # The decimation step gives TetGen marginally fewer
    # triangles to chew on but at the cost of mesh validity.
    # For watertight inputs the polygon-extrude output goes
    # to TetGen directly (it still gets capped by the
    # `decim_target` knob upstream if the user has explicitly
    # tightened it). Legacy MC + STL paths produce non-
    # watertight inputs and need decimation to stay within
    # TetGen's tractable input-size band, so they keep
    # decimating as before.
    _input_watertight = False
    try:
        import trimesh as _trimesh_check_in
        _tm_in = _trimesh_check_in.Trimesh(
            vertices=np.asarray(raw.points, dtype=np.float64),
            faces=(np.asarray(raw.faces).reshape(-1, 4)[:, 1:]
                     .astype(np.int64)),
            process=False,
        )
        _input_watertight = bool(_tm_in.is_watertight)
    except Exception:                                    # noqa: BLE001
        pass
    # M48 fix: the decimation skip must apply ONLY to PRISMATIC (polygon-extrude)
    # watertight meshes — those are the ones pyvista.decimate degrades. Curved /
    # real-3-D nerves (e.g. micro-CT vagus) are ALSO watertight but are NOT
    # prismatic, and they MUST be decimated: keeping the full dense surface
    # (e.g. 424k tris) blows up the multi-region PLC (~900k tris), so the
    # epineurium offset self-intersects and TetGen segfaults (returncode -11).
    _prismatic = False
    if _input_watertight:
        try:
            from golgi.pipeline.mesh import _is_prismatic
            _prismatic = bool(_is_prismatic(pts))
        except Exception:                                # noqa: BLE001
            _prismatic = False
    _skip_decim = _input_watertight and _prismatic
    if (
        raw.n_faces > target_tris
        and not _skip_decim
    ):
        reduction = 1.0 - target_tris / raw.n_faces
        dec = raw.decimate(reduction, volume_preservation=True)
        cur_pts = np.asarray(dec.points, dtype=np.float64)
        cur_tris = (np.asarray(dec.faces).reshape(-1, 4)[:, 1:]
                      .astype(np.int64))
        say(f"  after decimate: {len(cur_pts):,} pts, "
             f"{len(cur_tris):,} tris"
             f"{'' if not _input_watertight else ' (curved/non-prismatic watertight)'}")
    else:
        cur_pts = np.asarray(raw.points, dtype=np.float64)
        cur_tris = (np.asarray(raw.faces).reshape(-1, 4)[:, 1:]
                      .astype(np.int64))
        if raw.n_faces > target_tris:
            say(
                f"  decimate SKIPPED (prismatic watertight input, "
                f"keeping {raw.n_faces:,} tris — "
                f"pyvista.decimate produces degenerates on "
                f"polygon-extrude prisms)"
            )

    # 2. Taubin smooth + MeshFix.repair, repeated.
    # M38 — Sub-step progress logs + joincomp guard.
    # Previously printed only the per-pass summary, which made
    # it impossible to tell whether Taubin or pymeshfix.repair
    # was the slow one when a pass hung. On a fresh-decimated
    # closed epi (post-M31 cap-closing) pymeshfix's `joincomp`
    # routine can spin for many minutes trying to weld phantom
    # disconnected speckles introduced by `pv.decimate`. We now
    # call `remove_smallest_components` FIRST on its own to
    # drop those speckles cheaply, then call `.repair` with
    # joincomp=False — the input is effectively single-component
    # after the prefilter, so the bridging step is a guaranteed
    # no-op anyway.
    import time as _time
    for _pass in range(n_passes):
        _t0 = _time.perf_counter()
        surf = _pv(cur_pts, cur_tris)
        sm = surf.smooth_taubin(
            n_iter=40, pass_band=0.1,
            edge_angle=180.0, feature_angle=180.0,
            boundary_smoothing=True,
            non_manifold_smoothing=True,
        )
        cur_pts = np.asarray(sm.points, dtype=np.float64)
        cur_tris = (np.asarray(sm.faces).reshape(-1, 4)[:, 1:]
                      .astype(np.int64))
        say(f"  pass {_pass + 1}/{n_passes}: Taubin done "
             f"({len(cur_tris):,} tris, "
             f"{_time.perf_counter() - _t0:.2f}s)")
        # Prefilter floating speckles via a fast PyTMesh pass
        # so the subsequent .repair() can use joincomp=False.
        _t1 = _time.perf_counter()
        try:
            from pymeshfix import PyTMesh as _PyTMesh
            _pm = _PyTMesh()
            _pm.load_array(
                np.ascontiguousarray(cur_pts, dtype=np.float64),
                np.ascontiguousarray(cur_tris, dtype=np.int32),
            )
            _pm.remove_smallest_components()
            _v, _f = _pm.return_arrays()
            cur_pts = np.asarray(_v, dtype=np.float64)
            cur_tris = np.asarray(_f, dtype=np.int64)
            say(f"  pass {_pass + 1}/{n_passes}: speckle-drop "
                 f"({len(cur_tris):,} tris, "
                 f"{_time.perf_counter() - _t1:.2f}s)")
        except Exception as ex:                              # noqa: BLE001
            say(f"  pass {_pass + 1}/{n_passes}: speckle-drop "
                 f"skipped ({type(ex).__name__})")
        _t2 = _time.perf_counter()
        _n_pre_repair = int(cur_tris.shape[0])
        # M48 — Check watertightness BEFORE running
        # pymeshfix.MeshFix.repair. The polygon-extrude path
        # produces meshes that are watertight by construction,
        # and when pyvista.decimate trims them to the target tri
        # count the result is STILL near-watertight (small float-
        # noise issues at the cut). pymeshfix.repair on such an
        # input does nothing useful (mesh is already valid) but
        # CAN spin for 5-10 minutes trying to "fix" non-existent
        # problems, eventually returning 0 tris (observed:
        # 50,000 → 0 in 414 s on the user's 50 mm histology
        # bundle epi). Skip when input is already watertight —
        # the rare cases that actually need repair (legacy STL
        # imports with self-intersections) still run it.
        _need_repair = True
        try:
            import trimesh as _trimesh_check
            _tm_chk = _trimesh_check.Trimesh(
                vertices=cur_pts, faces=cur_tris, process=False,
            )
            if bool(_tm_chk.is_watertight):
                _need_repair = False
                say(
                    f"  pass {_pass + 1}/{n_passes}: MeshFix.repair "
                    f"SKIPPED (input already watertight, "
                    f"{_n_pre_repair:,} tris)"
                )
        except Exception:                                # noqa: BLE001
            pass
        if not _need_repair:
            continue
        mf = pymeshfix.MeshFix(
            np.ascontiguousarray(cur_pts, dtype=np.float64),
            np.ascontiguousarray(cur_tris, dtype=np.int32),
        )
        # joincomp=False — the prefilter just dropped every non-
        # largest component, so there's nothing to bridge. This
        # avoids the pathological multi-minute spin on epi meshes
        # with float-noise speckles after decimation.
        mf.repair(joincomp=False, remove_smallest_components=False)
        _new_pts = np.asarray(mf.mesh.points, dtype=np.float64)
        _new_tris = (np.asarray(mf.mesh.faces).reshape(-1, 4)[:, 1:]
                       .astype(np.int64))
        # M48 — shred guard. pymeshfix.MeshFix.repair on a freshly-
        # pyvista-decimated polygon-extrude mesh can spin for many
        # minutes and return ZERO triangles. Defensive guard:
        # reject any result that drops triangle count below 50%
        # of input and keep the prior mesh.
        if _new_tris.shape[0] < 0.5 * max(1, _n_pre_repair):
            say(
                f"  pass {_pass + 1}/{n_passes}: MeshFix.repair "
                f"SHRED REJECTED ({_n_pre_repair:,} → "
                f"{int(_new_tris.shape[0]):,} tris, "
                "< 50% of input — keeping pre-repair mesh "
                f"({_time.perf_counter() - _t2:.2f}s)"
            )
        else:
            cur_pts = _new_pts
            cur_tris = _new_tris
            say(f"  pass {_pass + 1}/{n_passes}: MeshFix.repair "
                 f"({len(cur_tris):,} tris, "
                 f"{_time.perf_counter() - _t2:.2f}s)")

    # 3. Trimesh defensive pass — catches duplicate / zero-area /
    # inverted-normal faces that survive Taubin+MeshFix but trip
    # TetGen's exact predicates with "two facets exactly intersect"
    # (degenerate flat triangles overlap whatever's coplanar with
    # them). This is the step that closes the gap with
    # nerve_studio.py § 5 lines 2115-2165.
    try:
        import trimesh as _trimesh
        tm = _trimesh.Trimesh(
            vertices=cur_pts, faces=cur_tris, process=False,
        )
        n_before = len(tm.faces)
        tm.merge_vertices()
        try:
            uf = tm.unique_faces()
            tm.update_faces(uf)
        except Exception:
            pass
        try:
            nd = tm.nondegenerate_faces()
            tm.update_faces(nd)
        except Exception:
            norms = tm.face_normals
            if norms is not None:
                good = ~np.isnan(norms).any(axis=1)
                if not good.all():
                    tm.update_faces(good)
        tm.remove_unreferenced_vertices()
        try:
            _trimesh.repair.fix_inversion(tm)
            _trimesh.repair.fix_normals(tm)
        except Exception:
            pass
        n_after = len(tm.faces)
        if n_after != n_before:
            say(f"  trimesh defensive pass: dropped "
                 f"{n_before - n_after} degenerate / duplicate / "
                 f"inverted tris ({n_after:,} remain)")
            cur_pts = np.asarray(tm.vertices, dtype=np.float64)
            cur_tris = np.asarray(tm.faces, dtype=np.int64)
        else:
            say(f"  trimesh defensive pass: clean "
                 f"({n_after:,} tris validated)")
    except Exception as ex:
        say(f"  trimesh defensive pass raised {ex}; continuing")

    # 4. Escalating self-intersection eradication. MeshFix.repair()
    # above calls PyTMesh.clean(max_iters=10) internally — for some
    # surfaces that's not enough; try 50 iterations directly, then
    # surgically remove any stragglers and refill the resulting
    # micro-holes.
    n_si, bad = _count_self_intersections(cur_pts, cur_tris)
    if n_si > 0:
        say(f"  ⚠ {n_si} self-intersecting tris after {n_passes}-"
             f"pass cleanup. Escalating PyTMesh.clean(50)…")
        try:
            from pymeshfix import PyTMesh as _PyTMesh
            m_agg = _PyTMesh()
            m_agg.load_array(
                np.ascontiguousarray(cur_pts, dtype=np.float64),
                np.ascontiguousarray(cur_tris, dtype=np.int32),
            )
            m_agg.clean(max_iters=50, inner_loops=10)
            m_agg.remove_smallest_components()
            v, f = m_agg.return_arrays()
            cur_pts = np.asarray(v, dtype=np.float64)
            cur_tris = np.asarray(f, dtype=np.int64)
        except Exception as ex:
            say(f"    PyTMesh.clean(50) raised {ex}; continuing")
        n_si, bad = _count_self_intersections(cur_pts, cur_tris)
        say(f"  after PyTMesh.clean(50): "
             f"self-intersections = {n_si}")
        if n_si > 0:
            say(f"  ⚠ {n_si} stragglers — applying surgical "
                 f"removal (delete bad tris + refill holes)")
            cur_pts, cur_tris = _surgical_remove_intersections(
                cur_pts, cur_tris, bad,
            )
            n_si, _ = _count_self_intersections(cur_pts, cur_tris)
            say(f"  after surgical removal: "
                 f"{len(cur_pts):,} pts, {len(cur_tris):,} tris, "
                 f"self-intersections = {n_si}")
            if n_si > 0:
                # Final fallback: drop the bad faces outright
                # (TetGen will route the boundary around them).
                _, bad2 = _count_self_intersections(
                    cur_pts, cur_tris,
                )
                if len(bad2) > 0 and len(bad2) < len(cur_tris):
                    keep = np.ones(len(cur_tris), dtype=bool)
                    keep[bad2] = False
                    cur_tris = cur_tris[keep]
                    say(f"  dropped {len(bad2)} unrecoverable tris")
    else:
        say(f"  cleanup OK: 0 self-intersecting tris")

    # 5. Final pv.clean (strict, absolute tolerance).
    surf = _pv(cur_pts, cur_tris)
    surf = surf.clean(tolerance=1.0e-6, absolute=True).triangulate()
    out_pts = np.asarray(surf.points, dtype=np.float64)
    out_tris = (np.asarray(surf.faces).reshape(-1, 4)[:, 1:]
                  .astype(np.int64))
    say(f"  preprocessed: {len(out_pts):,} pts, "
         f"{len(out_tris):,} tris")
    return out_pts, out_tris


def _assemble_plc(pieces: list,
                    dedup_tol: float = 1.0e-6,
                    seam_planes_z: tuple = (),
                    seam_tol: float = 1.0e-5,
                    on_line=None,
                    debug_si: bool = False,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (pts, tris) pieces, quantize-and-dedup vertices,
    and drop degenerate triangles. Output is a single (pts, tris)
    pair safe to hand to TetGen.

    Two-pass merge:
      1. Global quantize-and-dedup at `dedup_tol` (default 1 µm) —
         catches verts that are byte-identical or rounding-noise
         apart from each other.
      2. Per-seam-plane snap at `seam_tol` (default 10 µm) — only
         applied to verts whose z is within `dedup_tol` of any
         plane in `seam_planes_z`. The inter-region seams of the
         cuff PLC live at exactly z=z_lo / z=z_hi (saline cap inner
         ↔ epi cap outer; silicone cap inner ↔ saline cap outer;
         muscle cap inner ↔ silicone cap outer; fascicle caps too).
         These are built by independent code paths and can drift a
         few microns apart from float-pipeline differences — the
         global 1 µm pass misses them and TetGen then sees "two
         facets exactly intersect" at the supposed seam. The local
         seam_tol is wider (10 µm = ~10 % of the smallest mesh
         feature) so the snap merges them, but only at the seam
         planes; interior detail is untouched.
    """
    say = on_line if on_line is not None else (lambda *_: None)
    _all_pts = np.vstack([p for p, _ in pieces])
    _offsets = np.cumsum([0] + [len(p) for p, _ in pieces])
    _all_tris = np.vstack([
        t + _offsets[i] for i, (_, t) in enumerate(pieces)
    ])
    n_in = int(_all_pts.shape[0])
    if debug_si:
        _s, _ = _count_self_intersections(_all_pts, _all_tris)
        say(f"  _assemble_plc[DEBUG]: raw concat SI = {_s}")
    _keys = np.round(_all_pts / dedup_tol).astype(np.int64)
    _, _uniq_idx, _inv = np.unique(
        _keys, axis=0, return_index=True, return_inverse=True,
    )
    _pts = _all_pts[_uniq_idx]
    _tris = _inv[_all_tris]
    say(
        f"  _assemble_plc: global dedup @ "
        f"{dedup_tol*1e6:.2f} µm: "
        f"{n_in:,} → {_pts.shape[0]:,} pts "
        f"({n_in - _pts.shape[0]:,} merged)"
    )
    if debug_si:
        _s, _ = _count_self_intersections(_pts, _tris)
        say(f"  _assemble_plc[DEBUG]: post-global-dedup SI = {_s}")

    # ---- Seam-plane snap pass ----
    # For every plane z in `seam_planes_z`, gather verts whose
    # z is within dedup_tol of the plane, KDTree-merge them at
    # `seam_tol`, and rewrite the surviving indices. Iterating
    # plane-by-plane (instead of one big KDTree across the
    # whole PLC) keeps the wider tolerance from accidentally
    # merging unrelated verts in the interior — only the
    # cap-plane seams get the generous merge.
    if seam_planes_z:
        try:
            from scipy.spatial import cKDTree
            for z_plane in seam_planes_z:
                on_plane = (
                    np.abs(_pts[:, 2] - float(z_plane))
                    < float(dedup_tol) * 2.0
                )
                idx_on_plane = np.where(on_plane)[0]
                if idx_on_plane.size < 2:
                    continue
                # KDTree over xy only — verts already share z.
                tree = cKDTree(_pts[idx_on_plane, :2])
                # query_ball_tree against itself finds all pairs
                # within seam_tol. For each cluster of nearby
                # verts, pick a representative (lowest idx) and
                # remap the others to it.
                pairs = tree.query_pairs(
                    r=float(seam_tol), output_type="ndarray",
                )
                if pairs.size == 0:
                    continue
                # Build a parent[] array using union-find via
                # iterative remap so chained clusters collapse.
                local_n = idx_on_plane.size
                parent = np.arange(local_n, dtype=np.int64)
                for a, b in pairs:
                    ra, rb = int(a), int(b)
                    while parent[ra] != ra:
                        ra = int(parent[ra])
                    while parent[rb] != rb:
                        rb = int(parent[rb])
                    if ra == rb:
                        continue
                    parent[max(ra, rb)] = min(ra, rb)
                # Path-compress + map back to global IDs.
                roots = parent.copy()
                for i in range(local_n):
                    r = i
                    while roots[r] != r:
                        r = int(roots[r])
                    roots[i] = r
                # Build a per-global-idx remap. Verts not on this
                # plane stay themselves.
                remap = np.arange(_pts.shape[0], dtype=np.int64)
                for i in range(local_n):
                    remap[idx_on_plane[i]] = (
                        idx_on_plane[int(roots[i])]
                    )
                _tris = remap[_tris]
                n_merged = int(
                    (parent != np.arange(local_n)).sum(),
                )
                say(
                    f"  _assemble_plc: seam snap @ z="
                    f"{z_plane*1e3:+.3f} mm "
                    f"(tol {seam_tol*1e6:.1f} µm): "
                    f"merged {n_merged} vert pair(s) "
                    f"across {idx_on_plane.size} on-plane verts"
                )
            # After all remaps, the pts array still has the
            # orphaned slots (verts that got merged INTO other
            # verts). Compact: keep only verts actually
            # referenced by a tri.
            used = np.zeros(_pts.shape[0], dtype=bool)
            used[_tris.ravel()] = True
            if not used.all():
                used_idx = np.where(used)[0]
                # Build the inverse map: old_idx → new_idx.
                inv_map = -np.ones(_pts.shape[0], dtype=np.int64)
                inv_map[used_idx] = np.arange(used_idx.size)
                _pts = _pts[used_idx]
                _tris = inv_map[_tris]
                say(
                    f"  _assemble_plc: compacted orphans → "
                    f"{_pts.shape[0]:,} pts"
                )
        except Exception as ex:                           # noqa: BLE001
            say(
                f"  _assemble_plc: seam snap skipped "
                f"({type(ex).__name__}: {ex})"
            )

    _bad = (
        (_tris[:, 0] == _tris[:, 1])
        | (_tris[:, 1] == _tris[:, 2])
        | (_tris[:, 0] == _tris[:, 2])
    )
    n_bad = int(_bad.sum())
    _tris = _tris[~_bad]
    if n_bad > 0:
        say(
            f"  _assemble_plc: dropped {n_bad} "
            f"degenerate tri(s) post-merge"
        )
    if debug_si:
        _s, _ = _count_self_intersections(_pts, _tris)
        say(f"  _assemble_plc[DEBUG]: post-seam-snap final SI = {_s}")
    return _pts, _tris


def build_muscle_pieces_for_nerve(
    pre_pts: np.ndarray,
    muscle_radial_pad_m: float,
    muscle_axial_pad_m: float,
    muscle_dx_m: float = 0.0,
    muscle_dy_m: float = 0.0,
    muscle_dz_m: float = 0.0,
    L_cuff_eff: float = 10.0e-3,
    n_circ: int = 96,
    on_line=None,
) -> dict:
    """Auto-fit a cylindrical muscle bbox to `pre_pts` (preprocessed
    nerve in the input frame) and emit the muscle PLC pieces:
    lateral cylinder + low/high disk caps + seed point. All
    geometry is axis-aligned to the input frame's +z. The caller
    can transform the returned pieces to a different frame if
    needed (e.g. canonical-frame muscle → per-design-local frame
    for the F3.2 multi-design build).

    Returns a dict shaped:
      {"lat":     (pts (N, 3), tris (M, 3)),
       "cap_lo":  (pts, tris),
       "cap_hi":  (pts, tris),
       "seed":    [x, y, z],
       "params":  {"R_mus": .., "z_mus_lo": .., "z_mus_hi": ..,
                   "mus_cx": .., "mus_cy": ..}}
    """
    say = on_line if on_line is not None else (lambda *_: None)
    pre_pts = np.asarray(pre_pts, dtype=np.float64)
    pre_pts_xy = pre_pts[:, :2]
    nerve_r_max_global = float(
        np.linalg.norm(pre_pts_xy, axis=1).max()
    )
    nerve_z_min = float(pre_pts[:, 2].min())
    nerve_z_max = float(pre_pts[:, 2].max())
    R_mus = nerve_r_max_global + muscle_radial_pad_m
    z_mus_lo = nerve_z_min - muscle_axial_pad_m + muscle_dz_m
    z_mus_hi = nerve_z_max + muscle_axial_pad_m + muscle_dz_m
    mus_cx = float(muscle_dx_m)
    mus_cy = float(muscle_dy_m)
    L_mus = z_mus_hi - z_mus_lo
    n_axial_mus = min(
        64, max(8, int(round(L_mus / (L_cuff_eff / 8)))),
    )
    say(f"  auto-fit muscle: nerve r_max = "
         f"{nerve_r_max_global*1e3:.2f} mm + "
         f"{muscle_radial_pad_m*1e3:.0f} mm pad → "
         f"R_muscle = {R_mus*1e3:.2f} mm")
    say(f"  muscle z-span: nerve z = "
         f"{nerve_z_min*1e3:+.1f}..{nerve_z_max*1e3:+.1f} mm "
         f"+ {muscle_axial_pad_m*1e3:.0f} mm pad, Δz="
         f"{muscle_dz_m*1e3:+.1f} mm "
         f"→ L_muscle = {L_mus*1e3:.1f} mm")
    mus_lat = _build_cylinder_lateral(
        R_mus, z_mus_lo, z_mus_hi,
        n_circ=n_circ, n_axial=n_axial_mus,
        x_c=mus_cx, y_c=mus_cy,
    )
    r_mus_lo = _orient(mus_lat[0][mus_lat[2], :2], ccw=True)
    r_mus_hi = _orient(mus_lat[0][mus_lat[3], :2], ccw=True)
    # Quality-fill the muscle cap disks (Delaunay + interior Steiner)
    # instead of earcut — earcut fans the disk into long slivers that
    # break TetGen. Boundary ring (r_mus) is preserved → conforms with
    # the muscle lateral cylinder.
    _mlo2d, mus_cap_lo_tris = _quality_triangulate_xy(r_mus_lo, [])
    _mhi2d, mus_cap_hi_tris = _quality_triangulate_xy(r_mus_hi, [])
    mus_cap_lo_pts = np.column_stack(
        [_mlo2d, np.full(len(_mlo2d), z_mus_lo)],
    )
    mus_cap_hi_pts = np.column_stack(
        [_mhi2d, np.full(len(_mhi2d), z_mus_hi)],
    )
    # Muscle seed: midway between cuff outer wall (R_co)
    # equivalent — caller knows R_co; we just give a point on
    # the +x ray that's well inside the muscle annulus.
    _mus_dir = np.array([mus_cx, mus_cy], dtype=np.float64)
    _mus_dir_norm = float(np.linalg.norm(_mus_dir))
    if _mus_dir_norm > 1e-12:
        _ux, _uy = _mus_dir / _mus_dir_norm
    else:
        _ux, _uy = 1.0, 0.0
    # Place seed at radius midway between origin and muscle outer
    # wall (along the muscle dx/dy direction when offset); this
    # is always outside the cuff (cuff R_co < R_mus) and inside
    # the muscle annulus.
    _seed_r = 0.5 * R_mus + 0.5 * _mus_dir_norm
    seed_muscle = [
        mus_cx + _ux * (_seed_r - _mus_dir_norm),
        mus_cy + _uy * (_seed_r - _mus_dir_norm),
        0.5 * (z_mus_lo + z_mus_hi),
    ]
    return {
        "lat": (np.asarray(mus_lat[0], dtype=np.float64),
                 np.asarray(mus_lat[1], dtype=np.int64)),
        "cap_lo": (mus_cap_lo_pts,
                    np.asarray(mus_cap_lo_tris, dtype=np.int64)),
        "cap_hi": (mus_cap_hi_pts,
                    np.asarray(mus_cap_hi_tris, dtype=np.int64)),
        "seed": seed_muscle,
        "params": {
            "R_mus": R_mus,
            "z_mus_lo": z_mus_lo,
            "z_mus_hi": z_mus_hi,
            "mus_cx": mus_cx,
            "mus_cy": mus_cy,
            "L_mus": L_mus,
            "n_axial_mus": n_axial_mus,
        },
    }


def transform_muscle_pieces(
    pieces: dict,
    R: "np.ndarray | None" = None,
    offset: "np.ndarray | None" = None,
) -> dict:
    """Apply a rigid transform `p_out = p_in @ R + offset` (row
    vectors) to every vertex array in a muscle-pieces dict (as
    returned by `build_muscle_pieces_for_nerve`). The seed point
    is transformed too. `R=None` → identity; `offset=None` → zero.

    Used by the F3.2 per-design build to take canonical-frame
    muscle pieces and place them in each design's local frame.
    Triangle index arrays are returned unchanged (point-order
    preserved)."""
    if R is None:
        R = np.eye(3, dtype=np.float64)
    if offset is None:
        offset = np.zeros(3, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    out: dict = {"params": dict(pieces.get("params", {}))}
    for k in ("lat", "cap_lo", "cap_hi"):
        if k in pieces:
            pts, tris = pieces[k]
            out[k] = (pts @ R + offset, np.asarray(tris))
    if "seed" in pieces:
        s = np.asarray(pieces["seed"], dtype=np.float64)
        out["seed"] = (s @ R + offset).tolist()
    return out


def _plc_debug_per_piece(pieces, names, debug_dir, on_line=None) -> None:
    """Instrumentation (gated on `debug_dir`): save each PLC piece as an
    STL and report its WITHIN-piece self-intersection count, so we can
    tell whether SIs are born in a piece's own triangulation (a real
    construction bug) or created later by the assembly dedup / seam-snap
    (which fires only the `_assemble_plc` merge). No-op in normal runs.
    """
    from pathlib import Path as _Path
    say = on_line if on_line is not None else (lambda *_: None)
    dd = _Path(debug_dir)
    dd.mkdir(parents=True, exist_ok=True)
    say(f"  [PLC-DEBUG] per-piece report (within-piece SI) → {dd}")
    total = 0
    for i, (p, t) in enumerate(pieces):
        nm = names[i] if i < len(names) else f"piece_{i:02d}"
        p = np.asarray(p, dtype=np.float64)
        t = np.asarray(t, dtype=np.int64)
        if t.size == 0:
            say(f"    [{i:02d}] {nm}: EMPTY")
            continue
        try:
            si, _ = _count_self_intersections(p, t)
        except Exception:                                  # noqa: BLE001
            si = -1
        total += max(si, 0)
        try:
            ff = np.hstack(
                [np.full((len(t), 1), 3, np.int64), t]
            ).ravel()
            pv.PolyData(p, ff).save(str(dd / f"{i:02d}_{nm}.stl"))
        except Exception:                                  # noqa: BLE001
            pass
        say(f"    [{i:02d}] {nm}: {len(p):,} pts {len(t):,} tris  "
            f"within-SI={si}  z=[{p[:, 2].min()*1e3:+.1f},"
            f"{p[:, 2].max()*1e3:+.1f}]mm")
    say(f"  [PLC-DEBUG] Σ within-piece SI = {total}  (compare to "
        f"post-assembly total: the gap is SIs born in dedup/seam-snap)")


def assemble_bare_nerve_in_bath(nerve_pts_m: np.ndarray,
                                bnd_tris: np.ndarray,
                                bath_radial_pad_m: float,
                                bath_axial_pad_m: float,
                                *,
                                decim_target_tris: int = 50_000,
                                bath_tag: str = "saline",
                                on_line=None,
                                ) -> "tuple[pv.PolyData, dict]":
    """Cuff-free PLC: a nerve surface embedded directly in a single
    homogeneous conductive bath — no cuff, no concentric saline/
    silicone/muscle shells, no cuff-window clip.

    This is the correct domain for an INTRAFASCICULAR electrode (LIFE),
    which has no cuff, and it sidesteps the degeneracy that stalls the
    multi-domain cuff path on idealized axisymmetric geometry: the
    cuff-window clip + concentric coaxial shell caps of a perfect
    cylinder produce coincident, zero-area facets (self-intersections)
    that hang gmsh OCC / make TetGen run away. Here the two closed
    surfaces (nerve + enclosing bath cylinder) are disjoint and share
    no geometry, so it is the canonical "surface inside a box" case
    TetGen/gmsh mesh trivially.

    Returns (plc, seed_pos) with the same contract as
    `assemble_multi_domain_plc`: `seed_pos` carries one region point per
    region present — here `endo` (tag 1, inside the nerve) and the bath
    (`bath_tag`, default `saline` → tag 2, in the gap). Both surfaces
    stay in the input (cuff-local) frame.
    """
    say = on_line if on_line is not None else (lambda *_: None)
    pre_pts, pre_tris = _preprocess_nerve_surface(
        np.asarray(nerve_pts_m, dtype=np.float64),
        np.asarray(bnd_tris, dtype=np.int64),
        target_tris=int(decim_target_tris),
        n_passes=2,
        on_line=say,
    )

    def _faces(tris):
        n = len(tris)
        f = np.empty(n * 4, dtype=np.int64)
        f[0::4] = 3
        f[1::4] = tris[:, 0]
        f[2::4] = tris[:, 1]
        f[3::4] = tris[:, 2]
        return f

    nerve_surf = pv.PolyData(pre_pts, _faces(pre_tris))

    # Enclosing bath cylinder (coaxial with the cuff-local +z axis, but
    # NOT touching the nerve → no coincident caps). Pad radially +
    # axially so the ground BC sits well beyond the nerve.
    r = np.hypot(pre_pts[:, 0], pre_pts[:, 1])
    r_max = float(r.max())
    R_bath = r_max + float(bath_radial_pad_m)
    z_lo = float(pre_pts[:, 2].min()) - float(bath_axial_pad_m)
    z_hi = float(pre_pts[:, 2].max()) + float(bath_axial_pad_m)
    z_c = 0.5 * (z_lo + z_hi)
    bath = pv.Cylinder(
        center=(0.0, 0.0, z_c), direction=(0.0, 0.0, 1.0),
        radius=R_bath, height=(z_hi - z_lo),
        resolution=96, capping=True,
    ).triangulate()

    # Merge the two disjoint closed surfaces into one PLC.
    plc = (nerve_surf + bath).clean()

    endo_seed = pre_pts.mean(axis=0).astype(np.float64)
    bath_seed = np.array(
        [0.5 * (r_max + R_bath), 0.0, z_c], dtype=np.float64,
    )
    seed_pos = {"endo": endo_seed, str(bath_tag): bath_seed}
    say(f"  bare bath: nerve r_max={r_max*1e3:.2f} mm → "
        f"R_bath={R_bath*1e3:.1f} mm, z=[{z_lo*1e3:.1f},"
        f"{z_hi*1e3:.1f}] mm; regions: endo + {bath_tag}")
    return plc, seed_pos


def assemble_multi_domain_plc(nerve_pts_m: np.ndarray,
                                bnd_tris: np.ndarray,
                                L_cuff_m: float,
                                R_ci_m: float,
                                R_co_m: float,
                                muscle_radial_pad_m: float,
                                muscle_axial_pad_m: float,
                                muscle_dx_m: float = 0.0,
                                muscle_dy_m: float = 0.0,
                                muscle_dz_m: float = 0.0,
                                decim_target_tris: int = 50_000,
                                use_epi: bool = False,
                                epi_thickness_m: float = 50.0e-6,
                                scar_thickness_m: float = 0.0,
                                on_line=None,
                                muscle_pieces: "dict | None" = None,
                                inner_surfaces: (
                                    "list[tuple[np.ndarray, "
                                    "np.ndarray]] | None"
                                ) = None,
                                debug_dir=None,
                                ) -> tuple[pv.PolyData, dict]:
    """Mirror of nerve_studio.py § 5 PLC assembly. Produces a single
    closed surface mesh (returned as pv.PolyData) safe to hand to
    TetGen — no self-intersections, no near-duplicate vertices,
    cap polylines shared byte-identically between adjacent pieces.

    The cuff is built AXIS-ALIGNED at the origin in the input frame
    — that's the only configuration TetGen handles reliably across
    real-world branched-VN inputs. `nerve_pts_m` is therefore
    assumed to be in CUFF-LOCAL coordinates (cuff at origin,
    nerve trajectory aligned with +z at the cuff site).

    F3.2 (multi-design) callers: pre-transform the canonical-frame
    nerve to each design's own cuff-local frame BEFORE calling
    this function, and pass the canonical-frame muscle pieces
    transformed to that same design-local frame via
    `muscle_pieces` (defeat the auto-fit). At mesh-restore /
    viewport time, rotate the on-disk mesh back to PCA-translated
    so all designs co-render. See `pipeline/mesh.py` for the
    orchestration.

    Parameters:
      muscle_pieces: optional dict from
        `build_muscle_pieces_for_nerve` (possibly transformed
        via `transform_muscle_pieces`). When provided, the
        muscle isn't auto-fit — these pre-built pieces are
        used directly. When None (legacy single-cuff path), the
        muscle is auto-fit to the input nerve's bbox.

    Pipeline:
      1. Build the full nerve PolyData from (nerve_pts_m, bnd_tris).
      2. Determine z_lo, z_hi (cuff window) and perturb to avoid
         landing on any nerve vertex.
      3. Two paired clips → three open pieces _nb / _nm / _na.
      4. Extract the cap polylines (1+ loops each) from _nb's upper
         boundary and _na's lower boundary using vtkFeatureEdges.
      5. Build saline / silicone lateral cylinders sized to
         (z_lo, z_hi); use `muscle_pieces` for muscle (or
         auto-fit when None).
      6. Triangulate annular caps with mapbox_earcut.
      7. Assemble all pieces, dedup on a 1 µm quantized grid.
    """
    # ---- 1. Nerve surface — preprocess first to kill slivers
    # that would otherwise survive into the PLC and trigger
    # TetGen's "two segments nearly overlapping" / collinear-
    # boundary failure. Stays in the input (cuff-local) frame.
    say = on_line if on_line is not None else (lambda *_: None)
    pre_pts, pre_tris = _preprocess_nerve_surface(
        np.asarray(nerve_pts_m, dtype=np.float64),
        np.asarray(bnd_tris, dtype=np.int64),
        target_tris=int(decim_target_tris),
        n_passes=2,
        on_line=say,
    )
    n = len(pre_tris)
    faces = np.empty(n * 4, dtype=np.int64)
    faces[0::4] = 3
    faces[1::4] = pre_tris[:, 0]
    faces[2::4] = pre_tris[:, 1]
    faces[3::4] = pre_tris[:, 2]
    nerve_surf = pv.PolyData(pre_pts, faces)

    # ---- 2. Cuff window. Perturb the clip planes to avoid
    # landing within `min_clearance` of any nerve vertex —
    # vtkClipPolyData produces near-zero-area slivers when a
    # vertex sits essentially on the cut plane, and TetGen's
    # exact predicates flag those as facet self-intersections
    # even though pymeshfix's SI detector says the input is
    # clean (because the bad triangle is CREATED by the clip,
    # not present beforehand).
    z_lo_nom = -L_cuff_m / 2.0
    z_hi_nom = +L_cuff_m / 2.0
    nerve_z = np.asarray(nerve_surf.points)[:, 2]

    def _safe_clip_z(z_proposed: float,
                       min_clearance: float = 1.0e-6,
                       max_shift: float = 1.0e-3) -> float:
        _z = float(z_proposed)
        for _ in range(50):
            _d = nerve_z - _z
            _imin = int(np.argmin(np.abs(_d)))
            if abs(_d[_imin]) >= min_clearance:
                return _z
            _sign = -1.0 if _d[_imin] > 0 else +1.0
            _z = _z + _sign * (min_clearance + abs(_d[_imin]))
            if abs(_z - float(z_proposed)) > max_shift:
                break
        return _z

    z_lo = _safe_clip_z(z_lo_nom)
    z_hi = _safe_clip_z(z_hi_nom)
    L_cuff_eff = z_hi - z_lo
    say(f"  clip planes: z_lo={z_lo*1e3:+.3f} mm "
         f"(Δ={(z_lo-z_lo_nom)*1e6:+.0f} µm), "
         f"z_hi={z_hi*1e3:+.3f} mm "
         f"(Δ={(z_hi-z_hi_nom)*1e6:+.0f} µm)")

    # ---- 3. Two paired clips → three open pieces ----
    nb, rest = nerve_surf.clip(
        normal=(0, 0, +1.0), origin=(0, 0, z_lo),
        return_clipped=True,
    )
    nm, na = rest.clip(
        normal=(0, 0, +1.0), origin=(0, 0, z_hi),
        return_clipped=True,
    )
    if nm.n_points == 0:
        raise RuntimeError(
            "nerve_middle is empty — the cuff window does not "
            "cross the nerve. Move the cuff onto the nerve "
            "trunk and refit."
        )

    # ---- 4. Cap polylines from clipped pieces' open boundaries ----
    xs_lo_loops = _open_boundary_polylines(nb)
    xs_hi_loops = _open_boundary_polylines(na)
    if not xs_lo_loops or not xs_hi_loops:
        raise RuntimeError(
            "Nerve does not cross at least one cuff cap plane "
            f"(z = {z_lo*1e3:+.2f} or {z_hi*1e3:+.2f} mm). "
            "Shrink L_cuff or move the cuff so both caps land "
            "inside the nerve."
        )
    xs_lo = xs_lo_loops[0]
    xs_hi = xs_hi_loops[0]

    # ---- 5. Lateral cylinders ----
    n_circ = max(96, len(xs_lo), len(xs_hi))
    sal_lat = _build_cylinder_lateral(
        R_ci_m, z_lo, z_hi, n_circ=n_circ, n_axial=8,
    )
    sil_lat = _build_cylinder_lateral(
        R_co_m, z_lo, z_hi, n_circ=n_circ, n_axial=8,
    )
    # F3.2-M3 — optional scar / connective tissue cylinder. When
    # `scar_thickness_m > 0`, builds a third lateral cylinder at
    # R_scar = r_nerve_max + scar_thickness, so the scar layer's
    # radial extent measured outward from the nerve surface IS
    # the user-specified thickness. Bigger thickness ⇒ thicker
    # scar shell (intuitive direction). Clamped to R_ci − ε so
    # the scar never exceeds the cuff inner wall. Saline (tag 2)
    # auto-fills the remaining annular gap from R_scar to R_ci.
    use_scar = bool(scar_thickness_m > 0.0)
    R_scar_m = None
    if use_scar:
        # r_nerve_max from the cap cross-section loops. Cheap
        # and conservative — using the larger of (z_lo, z_hi)
        # cap radii guarantees the scar cylinder always
        # encloses the nerve at the cap planes.
        def _max_loop_radius(loops, cent_xy=None):
            if not loops:
                return 0.0
            c = (cent_xy if cent_xy is not None
                 else loops[0][:, :2].mean(axis=0))
            return max(
                float(np.linalg.norm(
                    loop[:, :2] - c, axis=1,
                ).max())
                for loop in loops
            )
        _r_nerve_cap_lo = _max_loop_radius(xs_lo_loops)
        _r_nerve_cap_hi = _max_loop_radius(xs_hi_loops)
        r_nerve_caps_max = max(_r_nerve_cap_lo, _r_nerve_cap_hi)
        _R_scar_raw = r_nerve_caps_max + float(scar_thickness_m)
        _R_scar_max_safe = float(R_ci_m) - 1.0e-6
        R_scar_m = min(_R_scar_raw, _R_scar_max_safe)
        if R_scar_m <= r_nerve_caps_max:
            say(
                f"  ⚠ scar disabled — requested R_scar "
                f"({_R_scar_raw*1e3:.3f} mm) ≤ nerve max "
                f"({r_nerve_caps_max*1e3:.3f} mm); thickness "
                f"too small for the local cuff clearance"
            )
            use_scar = False
            R_scar_m = None
            scar_lat = None
        else:
            scar_lat = _build_cylinder_lateral(
                R_scar_m, z_lo, z_hi, n_circ=n_circ, n_axial=8,
            )
            say(
                f"  scar cylinder: R_scar={R_scar_m*1e3:.3f} mm "
                f"= r_nerve({r_nerve_caps_max*1e3:.3f}) "
                f"+ thickness({scar_thickness_m*1e3:.3f}) mm "
                f"[R_ci={R_ci_m*1e3:.3f} mm]"
            )
    else:
        scar_lat = None
    # Muscle: use caller-supplied pieces if present (F3.2 multi-
    # design: caller transforms canonical-frame muscle into each
    # design's local frame so the muscle bbox is shape-identical
    # across designs in canonical / viewport space). Otherwise
    # auto-fit to the input nerve in the input frame (legacy
    # single-cuff path).
    if muscle_pieces is None:
        muscle_pieces = build_muscle_pieces_for_nerve(
            pre_pts,
            muscle_radial_pad_m=muscle_radial_pad_m,
            muscle_axial_pad_m=muscle_axial_pad_m,
            muscle_dx_m=muscle_dx_m,
            muscle_dy_m=muscle_dy_m,
            muscle_dz_m=muscle_dz_m,
            L_cuff_eff=L_cuff_eff,
            n_circ=n_circ,
            on_line=say,
        )
    mus_lat_pts = muscle_pieces["lat"][0]
    mus_lat_tris = muscle_pieces["lat"][1]
    mus_cap_lo_pts = muscle_pieces["cap_lo"][0]
    mus_cap_lo_tris = muscle_pieces["cap_lo"][1]
    mus_cap_hi_pts = muscle_pieces["cap_hi"][0]
    mus_cap_hi_tris = muscle_pieces["cap_hi"][1]
    seed_muscle = list(muscle_pieces["seed"])

    # ---- 6. Orient cap rings (CCW) and triangulate annular caps ----
    xs_lo_loops_ccw = [
        _orient(loop[:, :2], ccw=True) for loop in xs_lo_loops
    ]
    xs_hi_loops_ccw = [
        _orient(loop[:, :2], ccw=True) for loop in xs_hi_loops
    ]
    r_ci_lo = _orient(sal_lat[0][sal_lat[2], :2], ccw=True)
    r_ci_hi = _orient(sal_lat[0][sal_lat[3], :2], ccw=True)
    r_co_lo = _orient(sil_lat[0][sil_lat[2], :2], ccw=True)
    r_co_hi = _orient(sil_lat[0][sil_lat[3], :2], ccw=True)
    if use_scar:
        r_scar_lo = _orient(scar_lat[0][scar_lat[2], :2], ccw=True)
        r_scar_hi = _orient(scar_lat[0][scar_lat[3], :2], ccw=True)
    else:
        r_scar_lo = None
        r_scar_hi = None

    if use_scar:
        # Saline cap collapses to a single-loop annulus between
        # R_scar (inner) and R_ci (outer) — the nerve cross-
        # sections are hidden behind the scar cylinder.
        sal_cap_lo_2d, sal_cap_lo_tris = _triangulate_annulus_xy(
            r_ci_lo, r_scar_lo[::-1],
        )
        sal_cap_hi_2d, sal_cap_hi_tris = _triangulate_annulus_xy(
            r_ci_hi, r_scar_hi[::-1],
        )
        # Scar cap: R_scar outer ring, nerve cross-sections as
        # inner holes (multi-loop for branched VN).
        scar_cap_lo_2d, scar_cap_lo_tris = (
            _triangulate_annulus_xy_multi(
                r_scar_lo,
                [loop[::-1] for loop in xs_lo_loops_ccw],
            )
        )
        scar_cap_hi_2d, scar_cap_hi_tris = (
            _triangulate_annulus_xy_multi(
                r_scar_hi,
                [loop[::-1] for loop in xs_hi_loops_ccw],
            )
        )
    else:
        # No scar — saline cap covers the full annulus from R_ci
        # all the way to the nerve cross-sections (legacy path).
        sal_cap_lo_2d, sal_cap_lo_tris = (
            _triangulate_annulus_xy_multi(
                r_ci_lo,
                [loop[::-1] for loop in xs_lo_loops_ccw],
            )
        )
        sal_cap_hi_2d, sal_cap_hi_tris = (
            _triangulate_annulus_xy_multi(
                r_ci_hi,
                [loop[::-1] for loop in xs_hi_loops_ccw],
            )
        )
        scar_cap_lo_2d = None
        scar_cap_lo_tris = None
        scar_cap_hi_2d = None
        scar_cap_hi_tris = None
    # Silicone annular cap (R_ci → R_co)
    sil_cap_lo_2d, sil_cap_lo_tris = _triangulate_annulus_xy(
        r_co_lo, r_ci_lo[::-1],
    )
    sil_cap_hi_2d, sil_cap_hi_tris = _triangulate_annulus_xy(
        r_co_hi, r_ci_hi[::-1],
    )

    # Lift cuff caps to z=z_lo / z_hi
    sal_cap_lo_pts = np.column_stack(
        [sal_cap_lo_2d, np.full(len(sal_cap_lo_2d), z_lo)],
    )
    sal_cap_hi_pts = np.column_stack(
        [sal_cap_hi_2d, np.full(len(sal_cap_hi_2d), z_hi)],
    )
    sil_cap_lo_pts = np.column_stack(
        [sil_cap_lo_2d, np.full(len(sil_cap_lo_2d), z_lo)],
    )
    sil_cap_hi_pts = np.column_stack(
        [sil_cap_hi_2d, np.full(len(sil_cap_hi_2d), z_hi)],
    )
    if use_scar:
        scar_cap_lo_pts = np.column_stack(
            [scar_cap_lo_2d, np.full(len(scar_cap_lo_2d), z_lo)],
        )
        scar_cap_hi_pts = np.column_stack(
            [scar_cap_hi_2d, np.full(len(scar_cap_hi_2d), z_hi)],
        )
    else:
        scar_cap_lo_pts = None
        scar_cap_hi_pts = None

    # ---- 7. Assemble + dedup ----
    def _pv_to_pt(pd: "pv.PolyData") -> tuple[np.ndarray, np.ndarray]:
        _p = np.asarray(pd.points)
        _f = np.asarray(pd.faces).reshape(-1, 4)[:, 1:]
        return _p, _f

    # Diagnostics: SI count on each clipped nerve piece so we can
    # tell whether the clip itself introduced slivers.
    for _name, _pd in (("nb", nb), ("nm", nm), ("na", na)):
        _p, _f = _pv_to_pt(_pd)
        if len(_f) == 0:
            continue
        _n_si, _ = _count_self_intersections(_p, _f)
        say(f"  post-clip SI count [{_name}]: {_n_si} "
             f"({len(_p):,} pts, {len(_f):,} tris)")

    # ---- 6b. Optional epineurium shell — port of nerve_studio.py
    # § 5 lines 2521-2578. Computes an inward-offset of the nerve
    # surface at `epi_thickness_m` and runs pymeshfix twice to
    # heal the self-intersections that inevitably appear in concave
    # regions of branched VN STLs. Adds the offset surface as one
    # PLC piece — TetGen then carves out a thin annular shell
    # (tag 5) between the offset surface and the nerve surface,
    # leaving the rest of the nerve interior as tag 1 (endo).
    epi_surf_pts = None
    epi_surf_normals = None
    off_pts_c = None
    off_faces_c = None
    # V1 — µCT-bundle skip-offset-shell gate. The legacy STL path
    # needs the inward-offset surface because there is no
    # explicit endoneurium boundary in the imported geometry: the
    # offset shell IS the endo/epi interface. A µCT bundle has
    # the inverse situation — each fascicle surface in
    # `inner_surfaces` is already that interface, and the volume
    # between fascicles and the outer nerve hull naturally tags
    # as epi (with the `seed_epi_bundle` we emit below).
    # If we ALSO add the inward-offset shell here, we end up
    # with three nested closed surfaces (outer epi + offset +
    # fascicles) all sitting within ~50 µm of each other. The
    # offset shell almost always pierces one or more fascicles,
    # which is what TetGen flagged as "19 input triangles
    # skipped due to self-intersections" in the user's run. Drop
    # the offset-shell pass entirely when fascicles are present.
    _bundle_skip_offset_shell = use_epi and bool(inner_surfaces)
    if _bundle_skip_offset_shell:
        say(
            "  µCT-bundle: skipping inward-offset epi shell "
            "(fascicles act as inner epi boundary; "
            "use_epi semantics handled by epi seed below)"
        )
    if use_epi and not _bundle_skip_offset_shell:
        try:
            import pymeshfix as _pymeshfix
            say(f"  epineurium: computing inward offset by "
                 f"{epi_thickness_m*1e6:.0f} µm + 2× MeshFix …")
            nerve_n = nerve_surf.compute_normals(
                point_normals=True, cell_normals=False,
                auto_orient_normals=True, consistent_normals=True,
                non_manifold_traversal=False,
            )
            epi_surf_pts = np.asarray(
                nerve_n.points, dtype=np.float64,
            )
            epi_surf_normals = np.asarray(
                nerve_n.point_data["Normals"], dtype=np.float64,
            )
            off_pts = epi_surf_pts - epi_thickness_m * epi_surf_normals
            off_faces = (np.asarray(nerve_n.faces)
                            .reshape(-1, 4)[:, 1:]
                            .astype(np.int64))
            mf1 = _pymeshfix.MeshFix(
                np.ascontiguousarray(off_pts, dtype=np.float64),
                np.ascontiguousarray(off_faces, dtype=np.int32),
            )
            mf1.repair(joincomp=True, remove_smallest_components=True)
            off_pts_c = np.asarray(
                mf1.mesh.points, dtype=np.float64,
            )
            off_faces_c = (np.asarray(mf1.mesh.faces)
                              .reshape(-1, 4)[:, 1:]
                              .astype(np.int64))
            mf2 = _pymeshfix.MeshFix(
                np.ascontiguousarray(off_pts_c, dtype=np.float64),
                np.ascontiguousarray(off_faces_c, dtype=np.int32),
            )
            mf2.repair(joincomp=True,
                        remove_smallest_components=True)
            off_pts_c = np.asarray(
                mf2.mesh.points, dtype=np.float64,
            )
            off_faces_c = (np.asarray(mf2.mesh.faces)
                              .reshape(-1, 4)[:, 1:]
                              .astype(np.int64))
            say(f"  offset surface: {off_pts_c.shape[0]:,} pts, "
                 f"{off_faces_c.shape[0]:,} tris")
        except Exception as ex:
            say(f"  ⚠ epineurium offset failed: {ex}; "
                 f"continuing without epi shell")
            off_pts_c = None
            off_faces_c = None

    pieces = [
        _pv_to_pt(nb), _pv_to_pt(nm), _pv_to_pt(na),
        (sal_lat[0], sal_lat[1]),
        (sal_cap_lo_pts, sal_cap_lo_tris),
        (sal_cap_hi_pts, sal_cap_hi_tris),
        (sil_lat[0], sil_lat[1]),
        (sil_cap_lo_pts, sil_cap_lo_tris),
        (sil_cap_hi_pts, sil_cap_hi_tris),
        (mus_lat_pts, mus_lat_tris),
        (mus_cap_lo_pts, mus_cap_lo_tris),
        (mus_cap_hi_pts, mus_cap_hi_tris),
    ]
    if off_pts_c is not None and off_faces_c is not None:
        pieces.append((off_pts_c, off_faces_c))
    if use_scar:
        pieces.append((scar_lat[0], scar_lat[1]))
        pieces.append((scar_cap_lo_pts, scar_cap_lo_tris))
        pieces.append((scar_cap_hi_pts, scar_cap_hi_tris))
    # V1 — µCT-bundle inner surfaces. Each is a closed fascicle
    # endoneurium boundary that lives entirely INSIDE the
    # nerve_pts_m hull (the bundle's epi.stl), so it doesn't
    # need clipping to the cuff window. Just toss the tris in
    # the PLC alongside the cap/cylinder pieces — TetGen sees
    # one PLC with N+1 closed surfaces and the per-region seed
    # points emitted below tell it which subdomain is which.
    # ---- Inner fascicle surfaces — clip to cuff z window ----
    # ROOT CAUSE the user hit before this block landed: the
    # nerve hull, saline cylinder, silicone cylinder, and
    # muscle bbox are ALL clipped to / built within the cuff z
    # window [z_lo, z_hi] (~10 mm for a typical cuff). The
    # µCT fascicle surfaces, however, were being appended to
    # the PLC AS-IS — and they span the full bundle length
    # (~50 mm for a sheep VN), so every fascicle's lateral
    # wall pokes through both cap planes by ~20 mm at each
    # end. The protruding portion sits in the saline /
    # silicone / muscle regions and TetGen's boundary recovery
    # finds the fascicle-cylinder intersections, reports them
    # as "facets exactly intersect", and aborts with
    # "RuntimeError: The input surface mesh contain self-
    # intersections."
    #
    # Fix: clip each fascicle to [z_lo, z_hi] using the same
    # pv.clip + _open_boundary_polylines + earcut pattern as
    # the nerve hull, lift the cap polylines to the clip
    # planes, and assemble (lateral + low cap + high cap) into
    # a closed prism that fits entirely inside the cuff
    # window.
    def _clip_fascicle_to_z_window(
        f_pts: np.ndarray, f_tris: np.ndarray, fi: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        z_min = float(f_pts[:, 2].min())
        z_max = float(f_pts[:, 2].max())
        say(
            f"  fasc {fi}: input bbox z="
            f"[{z_min*1e3:+.2f}, {z_max*1e3:+.2f}] mm, "
            f"xy=[{float(f_pts[:, 0].min())*1e3:+.2f}, "
            f"{float(f_pts[:, 0].max())*1e3:+.2f}] × "
            f"[{float(f_pts[:, 1].min())*1e3:+.2f}, "
            f"{float(f_pts[:, 1].max())*1e3:+.2f}] mm, "
            f"{len(f_pts):,} pts, {len(f_tris):,} tris"
        )
        # Fascicle entirely outside the cuff window → nothing
        # to keep. Caller skips this fascicle.
        if z_max <= z_lo or z_min >= z_hi:
            say(
                f"  fasc {fi}: entirely outside cuff window "
                f"[{z_lo*1e3:+.2f}, {z_hi*1e3:+.2f}] mm — skip"
            )
            return None
        # Fascicle already fits inside the window — no clip
        # needed, return as-is.
        if z_min >= z_lo and z_max <= z_hi:
            say(
                f"  fasc {fi}: already fits inside cuff window "
                "— passthrough"
            )
            return (
                np.asarray(f_pts, dtype=np.float64),
                np.asarray(f_tris, dtype=np.int64),
            )
        # M39-C — Pre-repair through MeshFix BEFORE the pyvista
        # clip. The µCT-derived fascicle surfaces are watertight
        # after reconstruct3d but can still carry a handful of
        # near-coplanar near-duplicate triangles at the
        # marching-cubes float-grid floor; PyVista's Viskores
        # clip rejects these as non-manifold and falls back to
        # a brute-force manual axial clip that leaves a jagged
        # lateral boundary (hundreds of micro-loops → thousands
        # of phantom self-intersections after cap stitching).
        # Running pymeshfix.repair first with topology-
        # preserving flags (no joincomp, no component drop)
        # almost always turns the input into something
        # Viskores accepts. If the repair raises we silently
        # fall through to the original geometry — the manual-
        # clip fallback still handles that case.
        try:
            import pymeshfix as _pmf_pre
            _mfp = _pmf_pre.MeshFix(
                np.ascontiguousarray(f_pts, dtype=np.float64),
                np.ascontiguousarray(f_tris, dtype=np.int32),
            )
            _mfp.repair(
                joincomp=False,
                remove_smallest_components=False,
            )
            _pr_pts = np.asarray(
                _mfp.mesh.points, dtype=np.float64,
            )
            _pr_tris = (
                np.asarray(_mfp.mesh.faces, dtype=np.int64)
                .reshape(-1, 4)[:, 1:]
            )
            # Same shred guard as M39-B: if pre-repair tossed
            # more than half the surface, the input wasn't
            # actually pathological and the repair over-
            # reacted — keep the original.
            if _pr_tris.shape[0] >= 0.5 * f_tris.shape[0]:
                say(
                    f"  fasc {fi}: pre-clip MeshFix.repair "
                    f"{f_tris.shape[0]:,} → "
                    f"{_pr_tris.shape[0]:,} tris (clean for "
                    "Viskores clip)"
                )
                f_pts = _pr_pts
                f_tris = _pr_tris
            else:
                say(
                    f"  fasc {fi}: pre-clip MeshFix.repair "
                    f"would shred ({f_tris.shape[0]:,} → "
                    f"{_pr_tris.shape[0]:,}); keeping original"
                )
        except Exception as _ex:                          # noqa: BLE001
            say(
                f"  fasc {fi}: pre-clip MeshFix.repair raised "
                f"{type(_ex).__name__}; continuing without"
            )
        n_t = int(f_tris.shape[0])
        faces_flat = np.empty(n_t * 4, dtype=np.int64)
        faces_flat[0::4] = 3
        faces_flat[1::4] = f_tris[:, 0]
        faces_flat[2::4] = f_tris[:, 1]
        faces_flat[3::4] = f_tris[:, 2]
        surf = pv.PolyData(
            np.asarray(f_pts, dtype=np.float64), faces_flat,
        )
        # Clip below z_hi (normal pointing down → keep
        # everything in the -normal direction = below the
        # plane). Then clip above z_lo (normal pointing up).
        if z_max > z_hi:
            surf = surf.clip(
                normal=(0.0, 0.0, -1.0),
                origin=(0.0, 0.0, z_hi),
            )
        if z_min < z_lo:
            surf = surf.clip(
                normal=(0.0, 0.0, +1.0),
                origin=(0.0, 0.0, z_lo),
            )
        # Manual axial-clip fallback. PyVista's clip routes through
        # Viskores (vtkmSlice) which silently returns empty on the
        # non-manifold fascicle surfaces the µCT pipeline can
        # produce — and the documented VTK fallback (vtkClipPolyData)
        # has been observed to inherit the same failure on the
        # same inputs. The pure-numpy plane-clip pass below
        # bypasses both: it operates triangle-by-triangle, snaps
        # new vertices exactly onto z_lo / z_hi, and produces
        # the same kind of open-cap output `_open_boundary_polylines`
        # downstream consumes.
        if surf.n_points == 0 or surf.n_faces == 0:
            say(
                f"  fasc {fi}: pyvista clip returned empty "
                f"(Viskores / VTK rejection on non-manifold "
                f"input) — falling back to manual axial clip",
            )
            m_pts, m_tris = _manual_axial_clip(
                np.asarray(f_pts, dtype=np.float64),
                np.asarray(f_tris, dtype=np.int64),
                float(z_lo), float(z_hi),
            )
            if m_tris.shape[0] == 0:
                say(
                    f"  fasc {fi}: manual clip also produced "
                    "empty mesh — skip"
                )
                return None
            m_faces_flat = np.empty(
                m_tris.shape[0] * 4, dtype=np.int64,
            )
            m_faces_flat[0::4] = 3
            m_faces_flat[1::4] = m_tris[:, 0]
            m_faces_flat[2::4] = m_tris[:, 1]
            m_faces_flat[3::4] = m_tris[:, 2]
            surf = pv.PolyData(m_pts, m_faces_flat)
            say(
                f"  fasc {fi}: manual clip → {surf.n_points:,} "
                f"pts, {surf.n_faces:,} tris"
            )
        # Pull the cap polylines from the clipped piece's OWN
        # open boundary so the cap verts sit exactly on the
        # lateral seam (no near-duplicates for TetGen).
        loops = _open_boundary_polylines(surf)
        if not loops:
            say(
                f"  fasc {fi}: no open boundary after clip — "
                "skip"
            )
            return None
        # Split loops by which cap plane they sit on. Tolerance
        # is generous (10 µm) to absorb numerical jitter at the
        # cut plane.
        _tol = 1.0e-5
        lat_pts = np.asarray(surf.points, dtype=np.float64)
        lat_tris = (
            np.asarray(surf.faces)
              .reshape(-1, 4)[:, 1:]
              .astype(np.int64)
        )
        # Bucket loops by cap plane. Critical for the annular
        # case: a fascicle whose cross-section at z=z_target has
        # an internal void (or where multiple disjoint pieces
        # share the plane) emits MULTIPLE boundary loops on the
        # SAME plane. The legacy code triangulated each loop
        # independently with `_triangulate_polygon_xy`, which
        # then overlapped — that's where the systematic ~4% SI
        # rate we saw was coming from. The fix: group loops by
        # plane, classify outer vs hole via point-in-polygon
        # test, and triangulate the whole group with earcut's
        # multi-ring path (`_triangulate_annulus_xy_multi`).
        loops_by_plane: dict[str, list] = {"lo": [], "hi": []}
        # M39-A — Aggregate the per-loop "not on cap plane"
        # diagnostic into a single summary line. The manual
        # axial clip leaves jagged lateral boundaries on non-
        # manifold input, producing hundreds of micro-loops at
        # random z values; emitting one log line per loop was
        # flooding the busy-lightbox to the point users thought
        # the build was hung.
        ignored_z_mm: list[float] = []
        for li, loop in enumerate(loops):
            z_mean = float(loop[:, 2].mean())
            on_lo = abs(z_mean - z_lo) < _tol
            on_hi = abs(z_mean - z_hi) < _tol
            if not (on_lo or on_hi):
                ignored_z_mm.append(z_mean * 1e3)
                continue
            loops_by_plane["lo" if on_lo else "hi"].append({
                "li": li,
                "xy": loop[:, :2].copy(),
            })
        if ignored_z_mm:
            _arr = np.asarray(ignored_z_mm)
            say(
                f"  fasc {fi}: ignored {len(ignored_z_mm)} "
                f"feature-edge loop(s) not on cap planes "
                f"(z range {_arr.min():+.3f}…{_arr.max():+.3f} mm "
                "— jagged-edge artefacts from manual axial "
                "clip)"
            )

        def _point_in_polygon(
            pt_xy: np.ndarray, poly_xy: np.ndarray,
        ) -> bool:
            """Ray-cast (matplotlib-style) point-in-polygon.
            Used to nest loops: a loop whose representative
            vertex sits inside another loop is a hole."""
            try:
                from matplotlib.path import Path as _P
                return bool(_P(poly_xy).contains_point(pt_xy))
            except Exception:                          # noqa: BLE001
                # Manual ray-cast fallback when matplotlib
                # isn't around.
                x, y = float(pt_xy[0]), float(pt_xy[1])
                n = len(poly_xy)
                inside = False
                j = n - 1
                for i in range(n):
                    xi, yi = poly_xy[i]
                    xj, yj = poly_xy[j]
                    if ((yi > y) != (yj > y)) and (
                        x < (xj - xi) * (y - yi) / (yj - yi + 1e-30)
                        + xi
                    ):
                        inside = not inside
                    j = i
                return inside

        cap_pieces: list[tuple[np.ndarray, np.ndarray]] = []
        for plane_key, plane_loops in loops_by_plane.items():
            if not plane_loops:
                continue
            z_target = z_lo if plane_key == "lo" else z_hi
            on_hi = (plane_key == "hi")
            n_loops = len(plane_loops)
            if n_loops > 1:
                # Multiple loops on this plane — classify each
                # as outer-or-hole via nesting count. A loop is
                # a hole iff it sits inside an ODD number of
                # other loops; outer iff inside an even number
                # (0 = top level, 2 = inside a hole, etc.).
                # Most fascicle cross-sections produce nesting
                # depth 0 (outer) or 1 (hole-in-outer).
                nest_count = [0] * n_loops
                for i, li_a in enumerate(plane_loops):
                    rep = np.asarray(li_a["xy"][0])
                    for j, li_b in enumerate(plane_loops):
                        if i == j:
                            continue
                        if _point_in_polygon(rep, li_b["xy"]):
                            nest_count[i] += 1
                outer_idxs = [
                    i for i, n in enumerate(nest_count) if n % 2 == 0
                ]
                say(
                    f"  fasc {fi}: cap plane '{plane_key}' has "
                    f"{n_loops} loops ({len(outer_idxs)} outer + "
                    f"{n_loops - len(outer_idxs)} hole)"
                )
                # For each outer, find its immediate-child holes
                # (loops with depth exactly +1) and triangulate
                # the (outer + holes) group with the multi-ring
                # earcut path.
                for oi in outer_idxs:
                    outer_xy = _orient(
                        plane_loops[oi]["xy"], ccw=on_hi,
                    )
                    holes_xy: list = []
                    outer_depth = nest_count[oi]
                    rep_outer = outer_xy[0]
                    for hj in range(n_loops):
                        if hj == oi:
                            continue
                        if nest_count[hj] != outer_depth + 1:
                            continue
                        # Confirm this hole is INSIDE this
                        # particular outer (not a sibling
                        # outer's child).
                        rep_h = plane_loops[hj]["xy"][0]
                        if not _point_in_polygon(rep_h, outer_xy):
                            continue
                        # Holes are CW relative to the outer's
                        # CCW orientation.
                        hole_xy = _orient(
                            plane_loops[hj]["xy"], ccw=not on_hi,
                        )
                        holes_xy.append(hole_xy)
                    try:
                        if holes_xy:
                            pts2d, tris2d = (
                                _triangulate_annulus_xy_multi(
                                    outer_xy, holes_xy,
                                )
                            )
                            cap_xy = pts2d
                        else:
                            # quality (Delaunay) fill, not earcut — earcut
                            # fans the cross-section into near-zero-area
                            # slivers that leak the region + make TetGen's
                            # refinement non-terminate off-trunk (curved cut).
                            cap_xy, tris2d = _quality_triangulate_xy(
                                outer_xy, [],
                            )
                    except Exception as ex:           # noqa: BLE001
                        say(
                            f"  fasc {fi}: cap '{plane_key}' "
                            f"outer {oi} triangulation failed "
                            f"({ex}); skipping"
                        )
                        continue
                    cap_pts_3d = np.column_stack([
                        cap_xy,
                        np.full(len(cap_xy), z_target),
                    ])
                    cap_pieces.append((cap_pts_3d, tris2d))
            else:
                # Single loop on this plane — same as legacy
                # path, just renamed for clarity.
                outer_xy = _orient(
                    plane_loops[0]["xy"], ccw=on_hi,
                )
                try:
                    # quality (Delaunay) fill, not earcut (sliver-free caps)
                    outer_xy, tris2d = _quality_triangulate_xy(outer_xy, [])
                except Exception as ex:               # noqa: BLE001
                    say(
                        f"  fasc {fi}: cap '{plane_key}' single "
                        f"loop triangulation failed ({ex}); "
                        "skipping"
                    )
                    continue
                cap_pts_3d = np.column_stack([
                    outer_xy,
                    np.full(len(outer_xy), z_target),
                ])
                cap_pieces.append((cap_pts_3d, tris2d))

        if not cap_pieces:
            say(
                f"  fasc {fi}: no valid cap polygons "
                "produced — skip"
            )
            return None
        # Stitch lateral + caps. Each cap is a separate vertex
        # block (no need to share verts with the lateral — the
        # cap polylines came from the lateral's open boundary
        # so the SAME xy/z coords are already present on the
        # lateral side; sharing requires a kdtree dedup. The
        # `_assemble_plc` call below dedups at 1 µm so the
        # separate blocks fuse into one watertight prism).
        out_pts_list = [lat_pts]
        out_tris_list = [lat_tris]
        v_off = int(lat_pts.shape[0])
        for cap_pts_3d, cap_tris_2d in cap_pieces:
            out_pts_list.append(cap_pts_3d)
            out_tris_list.append(cap_tris_2d + v_off)
            v_off += int(cap_pts_3d.shape[0])
        out_pts = np.concatenate(out_pts_list, axis=0)
        out_tris = np.concatenate(out_tris_list, axis=0)
        say(
            f"  fasc {fi}: clipped → {len(out_pts):,} pts, "
            f"{len(out_tris):,} tris "
            f"(+{len(cap_pieces)} cap polygon"
            f"{'s' if len(cap_pieces) != 1 else ''})"
        )
        try:
            _n_si, _ = _count_self_intersections(
                out_pts, out_tris,
            )
            if _n_si > 0:
                # M19 — per-fascicle iterative SI repair. Cycles
                # MeshFix.repair → PyTMesh.clean → surgical-
                # drop+refill, re-counting SI between each step
                # so we short-circuit when SI hits zero. The
                # three tools attack different SI patterns:
                # MeshFix.repair handles the bulk; PyTMesh.clean
                # is more aggressive about stubborn pairs;
                # surgical drop nukes anything that survives.
                # Running them in isolation (per fascicle, ~10k
                # tris) keeps the bad-tri fraction well below
                # the 30% shred limit the global post-assembly
                # pass trips on. User suggested this cycling
                # approach based on prior success on legacy
                # nerves.
                say(
                    f"  fasc {fi}: post-clip SI count = "
                    f"{_n_si} — iterative repair "
                    "(MeshFix.repair → PyTMesh.clean → surgical)"
                )
                r_pts, r_tris, _n_si_after = _iterative_si_repair(
                    out_pts, out_tris,
                    max_cycles=3,
                    on_log=lambda m: say(f"  fasc {fi}: {m}"),
                    tag="",
                )
                say(
                    f"  fasc {fi}: iterative repair done → "
                    f"{r_pts.shape[0]:,} pts, "
                    f"{r_tris.shape[0]:,} tris, "
                    f"SI={_n_si_after} (was {_n_si})"
                )
                if _n_si_after < _n_si:
                    out_pts, out_tris = r_pts, r_tris
                else:
                    say(
                        f"  fasc {fi}: iterative repair did "
                        "not reduce SI — keeping pre-repair"
                    )
        except Exception:                          # noqa: BLE001
            pass
        return out_pts, out_tris

    if inner_surfaces:
        # GOLGI_FASCICLE_FULL_LENGTH — keep each fascicle at its
        # natural (full) axial extent instead of clipping it to the
        # cuff window. Valid ONLY when the nerve (epi) hull is itself
        # full-length, which it is for the µCT-bundle path: the epi
        # outer surface spans the whole nerve and the cuff-window cap
        # planes (saline / silicone / muscle annuli) live entirely
        # OUTSIDE the epi outer ring. A fascicle sits INSIDE that
        # ring, so passing it through whole means (a) it crosses
        # z_lo/z_hi but hits no cap there (the epi interior is
        # continuous across the planes), and (b) the endo region is
        # bounded by the fascicle's OWN watertight ends inside the
        # muscle block — so no multi-lobe end-cap is ever cut. This
        # is the original "toss the closed fascicle in whole" design
        # (see the block comment above); the window-clip was only
        # needed back when the epi hull was ALSO windowed and the
        # fascicle protruded past it into saline/muscle.
        _full_len = (
            __import__("os").environ.get(
                "GOLGI_FASCICLE_FULL_LENGTH") == "1"
        )
        say(
            f"  inner_surfaces: {len(inner_surfaces)} "
            f"fascicle"
            f"{'s' if len(inner_surfaces) != 1 else ''} → "
            + ("FULL-LENGTH passthrough (no window clip — natural "
               "watertight ends, no multi-lobe caps)"
               if _full_len else
               f"clip to cuff window "
               f"[{z_lo*1e3:+.2f}, {z_hi*1e3:+.2f}] mm")
        )
        _kept = 0
        for fi, (ip, it) in enumerate(inner_surfaces):
            if _full_len:
                res = (
                    np.asarray(ip, dtype=np.float64),
                    np.asarray(it, dtype=np.int64),
                )
            else:
                res = _clip_fascicle_to_z_window(
                    np.asarray(ip, dtype=np.float64),
                    np.asarray(it, dtype=np.int64),
                    fi,
                )
            if res is None:
                continue
            pieces.append(res)
            _kept += 1
        say(
            f"  inner_surfaces: kept {_kept}/"
            f"{len(inner_surfaces)}"
            + (" (full-length)" if _full_len else " after clip")
        )
    # Seam planes are the two cuff-window cap planes (z_lo and
    # z_hi). Every inter-region cap boundary (saline ↔ epi,
    # silicone ↔ saline, muscle ↔ silicone, fascicle ↔ saline)
    # lives at exactly one of these two z values. The seam-snap
    # pass inside _assemble_plc applies a slightly wider
    # tolerance to verts at those planes so float-pipeline
    # drift between independently-built caps doesn't leave
    # "two facets exactly intersect" TetGen warnings at the
    # supposed seams.
    if debug_dir is not None:
        _dbg_names = [
            "nerve_below", "nerve_middle", "nerve_above",
            "saline_lat", "saline_cap_lo", "saline_cap_hi",
            "silicone_lat", "silicone_cap_lo", "silicone_cap_hi",
            "muscle_lat", "muscle_cap_lo", "muscle_cap_hi",
        ]
        _plc_debug_per_piece(pieces, _dbg_names, debug_dir, on_line=say)
    import os as _sos
    # seam-snap tolerance for the cuff-window cap planes. Default 50 µm (tuned for
    # mm-radius nerves); for small nerves (e.g. the ~0.34 mm rabbit vagus) 50 µm is a
    # large fraction of the radius and over-merges nerve verts into degenerate tris →
    # TetGen recoversubfaces failure. Override with GOLGI_PLC_SEAM_UM.
    _seam_um = float(_sos.environ.get("GOLGI_PLC_SEAM_UM", "50"))
    plc_pts, plc_tris = _assemble_plc(
        pieces,
        dedup_tol=1.0e-6,
        seam_planes_z=(float(z_lo), float(z_hi)),
        seam_tol=_seam_um * 1e-6,
        on_line=say,
        debug_si=(debug_dir is not None),
    )
    say(f"  post-assembly: {len(plc_pts):,} pts, "
         f"{len(plc_tris):,} tris")
    import os as _wos
    _stitch = int(_wos.environ.get("GOLGI_PLC_STITCH_HOLES", "8"))
    if _stitch > 0:                                         # close tiny seam holes
        plc_pts, plc_tris, _ns = _fill_small_boundary_holes(
            plc_pts, plc_tris, max_edges=_stitch, on_line=say)
    # Weld sub-µm clip-ring micro-edges. Default-on (8 µm) when the gmsh /
    # CDT cap path is active, because the conforming caps preserve the messy
    # clip-ring verts that otherwise send TetGen's recovery into a runaway
    # off-trunk. lc is 150-300 µm so an 8 µm weld only collapses clip
    # artifacts. Override with GOLGI_PLC_WELD_UM.
    _cdt_on = _wos.environ.get("GOLGI_PLC_CDT", "1") != "0"
    _weld_um = float(_wos.environ.get(
        "GOLGI_PLC_WELD_UM", "8" if _cdt_on else "0"))
    if _weld_um > 0:
        _n0 = len(plc_tris)
        plc_pts, plc_tris, _ndrop = _weld_close_verts(
            plc_pts, plc_tris, _weld_um * 1e-6)
        say(f"  PLC weld {_weld_um:g}µm: {_n0:,} -> {len(plc_tris):,} "
            f"tris ({_ndrop} degenerate dropped)")
    if debug_dir is not None:                              # [sliver-diag] debug-only
        try:
            _v = plc_pts[plc_tris]
            _e0 = _v[:, 1] - _v[:, 0]; _e1 = _v[:, 2] - _v[:, 0]
            _e2 = _v[:, 2] - _v[:, 1]
            _area = 0.5 * np.linalg.norm(np.cross(_e0, _e1), axis=1)
            _lmax = np.maximum.reduce([
                np.linalg.norm(_e0, axis=1), np.linalg.norm(_e1, axis=1),
                np.linalg.norm(_e2, axis=1)])
            _hmin = np.where(_lmax > 0, 2.0 * _area / _lmax, 0.0)
            _thin = _hmin < 2e-5                           # <20µm height (lc is 200-300µm)
            _vthin = _hmin < 5e-6                          # <5µm
            _nt = int(_thin.sum()); _nv = int(_vthin.sum())
            if _nt:
                _zc = _v[_thin].mean(axis=1)[:, 2] * 1e3
                _rc = np.linalg.norm(_v[_thin].mean(axis=1)[:, :2], axis=1) * 1e3
                say(f"  [sliver-diag] thin<20µm={_nt} (<5µm={_nv}); z[{_zc.min():.2f},"
                    f"{_zc.max():.2f}]mm (z_lo={z_lo*1e3:.2f} z_hi={z_hi*1e3:.2f}); "
                    f"r[{_rc.min():.2f},{_rc.max():.2f}]mm; min height {_hmin.min():.2e}")
            else:
                say(f"  [sliver-diag] no thin facets (min height {_hmin.min():.2e})")
        except Exception as _e:                            # noqa: BLE001
            say(f"  [sliver-diag] failed: {_e}")
    _clean_um = float(_wos.environ.get("GOLGI_CLEAN_SLIVERS_UM", "0"))
    if _clean_um > 0:                                       # delete sliver facets + refill
        try:
            _v = plc_pts[plc_tris]
            _a = 0.5 * np.linalg.norm(
                np.cross(_v[:, 1] - _v[:, 0], _v[:, 2] - _v[:, 0]), axis=1)
            _lm = np.maximum.reduce([
                np.linalg.norm(_v[:, 1] - _v[:, 0], axis=1),
                np.linalg.norm(_v[:, 2] - _v[:, 0], axis=1),
                np.linalg.norm(_v[:, 2] - _v[:, 1], axis=1)])
            _hm = np.where(_lm > 0, 2.0 * _a / _lm, 0.0)
            _sl = np.where(_hm < _clean_um * 1e-6)[0].astype(np.int32)
            if _sl.size:
                _rp, _rt = _surgical_remove_intersections(
                    plc_pts, plc_tris, _sl, drop_small_components=False)
                say(f"  sliver-clean {_clean_um:g}µm: removed {_sl.size}; "
                    f"{len(plc_tris):,} -> {len(_rt):,} tris")
                plc_pts, plc_tris = _rp, _rt
        except Exception as _e2:                            # noqa: BLE001
            say(f"  sliver-clean failed: {_e2}")
    if debug_dir is not None:
        from pathlib import Path as _Pth
        try:
            _aff = np.hstack(
                [np.full((len(plc_tris), 1), 3, np.int64), plc_tris]
            ).ravel()
            pv.PolyData(plc_pts, _aff).save(
                str(_Pth(debug_dir) / "99_assembled_plc.stl")
            )
            # float64 faithful dump (STL is float32 → alters SIs)
            np.savez(
                str(_Pth(debug_dir) / "99_assembled_plc.npz"),
                pts=np.asarray(plc_pts, dtype=np.float64),
                tris=np.asarray(plc_tris, dtype=np.int64),
            )
        except Exception:                                  # noqa: BLE001
            pass
    # Diagnostic + targeted repair.
    #
    # pymeshfix's SI detector OVER-COUNTS on a multi-domain PLC
    # (shared cap-polyline / cylinder-seam edges register as
    # intersections even though they're correct), so a naive
    # `m.clean(N)` on the assembled PLC will shred the cuff /
    # muscle surfaces — 73k tris → 27k empirically. But TetGen
    # only needs us to fix the REAL SIs (typically 1-5 stubborn
    # tri-pairs at cap-stitch seams), not the false positives.
    #
    # `_surgical_remove_intersections` already handles this
    # safely (delete bad tris → fill_small_boundaries to refill
    # the resulting micro-holes, then a single clean pass). We
    # gate its use on two thresholds:
    #
    #  * SI fraction < 15% of total tris — anything higher is
    #    almost certainly false-positive shared-seam noise and
    #    surgery would shred the PLC. Leave it alone in that
    #    case; TetGen's boundary recovery sometimes absorbs the
    #    seams via its own tolerance.
    #  * Post-repair shred-rate < 30% — if the repair itself
    #    deletes more than a third of the mesh, abandon and keep
    #    the unrepaired PLC. Guard against pathological hole
    #    cascades.
    n_si, bad_idx = _count_self_intersections(plc_pts, plc_tris)
    say(
        f"  post-assembly SI (pymeshfix, may over-count "
        f"shared seams): {n_si}  (unique bad tris: "
        f"{bad_idx.size})"
    )
    if bad_idx.size > 0:                                    # [diag] locate the SI tris
        try:
            _bc = plc_pts[plc_tris[bad_idx]].mean(axis=1)
            _zr = _bc[:, 2] * 1e3
            _rr = np.linalg.norm(_bc[:, :2], axis=1) * 1e3
            _near_lo = int(np.sum(np.abs(_zr - z_lo * 1e3) < 0.1))
            _near_hi = int(np.sum(np.abs(_zr - z_hi * 1e3) < 0.1))
            say(
                f"  [diag] SI tris z[{_zr.min():.2f},{_zr.max():.2f}]mm "
                f"(z_lo={z_lo*1e3:.2f} z_hi={z_hi*1e3:.2f}; "
                f"@lo={_near_lo} @hi={_near_hi} mid={bad_idx.size-_near_lo-_near_hi}); "
                f"r[{_rr.min():.2f},{_rr.max():.2f}]mm"
            )
        except Exception:                                  # noqa: BLE001
            pass
    # We INTENTIONALLY do not run MeshFix.repair / PyTMesh.clean
    # on the assembled PLC. The PLC is a multi-domain piecewise
    # linear complex with 5+ disconnected closed surfaces (epi,
    # saline/silicone/muscle cylinders, fascicles). pymeshfix's
    # repair pipeline assumes a single connected manifold and
    # fuses/strips disconnected surfaces regardless of the
    # `remove_smallest_components=False` flag — we saw it cut
    # 111k tris → 20k, deleting every region except the
    # endoneurium. The 3k-ish "SI" count at this stage is also
    # almost entirely pymeshfix FALSE positives from shared
    # cap-polyline / cylinder-seam edges (correct geometry that
    # the detector mis-flags). Per-region cleanup runs upstream:
    # nerve-surface preprocessing for the epi, the new iterative
    # repair for each fascicle. TetGen's seam tolerance absorbs
    # the rest, with 3-6 actual bad tris typically tagged as
    # "skipped" in its boundary-recovery pass.
    # M43 — Surgical post-assembly SI repair. Replaces the prior
    # blanket-skip. The big risk on a multi-domain PLC is
    # `pymeshfix.repair` / `PyTMesh.clean` deleting whole regions
    # because they assume single-connected input. The surgical
    # path side-steps that: it only deletes the SPECIFIC bad
    # triangles pymeshfix's SI detector flagged, then refills
    # the resulting micro-holes via fill_small_boundaries, with
    # `drop_small_components=False` so every region survives.
    # Guarded by the 15% / 30% thresholds the comment above
    # outlined. Empirically the 7,000-15,000 bad-tri counts on
    # the user's 50 mm sheep VN come in at ~2-5% of total
    # tris — well within the 15% gate — and the surgical
    # repair drops them with < 10% triangle loss overall.
    if n_si > 0 and bad_idx.size > 0:
        si_frac = float(bad_idx.size) / float(plc_tris.shape[0])
        if si_frac >= 0.15:
            say(
                f"  post-assembly SI repair SKIPPED — bad-tri "
                f"fraction {si_frac:.1%} > 15% guard (likely "
                "shared-seam false positives; TetGen will be "
                "asked to absorb via its own tolerance)"
            )
        else:
            import os as _os
            _shred_guard = float(_os.environ.get("GOLGI_SHRED_GUARD", "0.30"))
            _max_passes = max(1, int(_os.environ.get("GOLGI_SI_PASSES", "1")))
            # Surgical removal deletes the flagged tris then refills the
            # micro-holes; the refill can introduce a few new SIs. One pass
            # leaves a handful (e.g. 14) which make TetGen insert Steiner
            # points endlessly in "Removing exterior tetrahedra". Iterating
            # the surgical pass (each only nibbles the residual) drives SI to
            # 0 so TetGen recovers cleanly. drop_small_components=False keeps
            # every region. Cumulative shred is gated by GOLGI_SHRED_GUARD.
            n_before = int(plc_tris.shape[0])
            cur_pts, cur_tris, cur_bad = plc_pts, plc_tris, bad_idx
            for _ipass in range(_max_passes):
                try:
                    rep_pts, rep_tris = _surgical_remove_intersections(
                        cur_pts, cur_tris, cur_bad,
                        drop_small_components=False,
                    )
                except Exception as _ex:                  # noqa: BLE001
                    say(
                        f"  post-assembly SI repair raised "
                        f"{type(_ex).__name__}: {_ex}; keeping prior PLC"
                    )
                    break
                shred_frac = 1.0 - (rep_tris.shape[0] / float(n_before))
                if shred_frac > _shred_guard:
                    say(
                        f"  post-assembly SI repair ABANDONED (pass "
                        f"{_ipass + 1}) — cum shred {shred_frac:.1%} > "
                        f"{_shred_guard:.0%} guard ({n_before:,} → "
                        f"{rep_tris.shape[0]:,} tris); keeping prior PLC"
                    )
                    break
                n_si_after, bad_after = _count_self_intersections(
                    rep_pts, rep_tris,
                )
                say(
                    f"  post-assembly surgical SI repair pass "
                    f"{_ipass + 1}: {cur_tris.shape[0]:,} → "
                    f"{rep_tris.shape[0]:,} tris (cum shred "
                    f"{shred_frac:.1%}); SI → {n_si_after}"
                )
                cur_pts, cur_tris, cur_bad = rep_pts, rep_tris, bad_after
                plc_pts, plc_tris = rep_pts, rep_tris
                if n_si_after == 0 or bad_after.size == 0:
                    break
                if float(bad_after.size) / float(rep_tris.shape[0]) >= 0.15:
                    break

    # ---- 8. Adaptive region seeds (mirror nerve_studio.py § 5).
    # The simple `(R_ci/2, 0, 0)` seed for saline ASSUMES the nerve
    # is perfectly centred at origin in the cuff frame — which it
    # almost never is in practice (PCA + local-PCA refine put the
    # nerve close to origin but not at it). Slice the nerve at
    # z=0 to get its actual cross-section centroid and use that
    # to position seeds so each one lands UNAMBIGUOUSLY inside the
    # region it's supposed to label.
    try:
        slice0 = nerve_surf.slice(normal=(0.0, 0.0, 1.0),
                                    origin=(0.0, 0.0, 0.0))
        slice_pts = np.asarray(slice0.points, dtype=np.float64)
    except Exception:
        slice_pts = np.zeros((0, 3))
    if slice_pts.shape[0] >= 3:
        cent_xy = slice_pts[:, :2].mean(axis=0)
        r_nerve_max = float(
            np.linalg.norm(slice_pts[:, :2] - cent_xy, axis=1).max()
        )
    else:
        # Fallback if z=0 doesn't intersect the nerve (cuff outside
        # nerve bounds — should already have raised earlier). Use
        # the avg of the two cap centroids as a best guess.
        _c_lo = xs_lo[:, :2].mean(axis=0)
        _c_hi = xs_hi[:, :2].mean(axis=0)
        cent_xy = 0.5 * (_c_lo + _c_hi)
        r_nerve_max = max(
            float(np.linalg.norm(xs_lo[:, :2] - _c_lo,
                                    axis=1).max()),
            float(np.linalg.norm(xs_hi[:, :2] - _c_hi,
                                    axis=1).max()),
        )
    cent_norm = float(np.linalg.norm(cent_xy))
    seed_endo = [float(cent_xy[0]), float(cent_xy[1]), 0.0]
    # V1 — µCT-bundle inner-region seed search. One endo seed
    # per fascicle (its mesh centroid); one epi seed in the
    # nerve-but-not-fascicle region. select_enclosed_points
    # gives a robust "is this point inside that closed surface"
    # test (PyVista wraps vtkSelectEnclosedPoints which uses
    # ray casting + winding counts). If the centroid happens to
    # land outside the fascicle (concave / horseshoe-shaped),
    # we walk a few candidate offsets along its bbox principal
    # axis until one lands inside.
    bundle_endo_extra: list[list[float]] = []
    seed_epi_bundle: list[float] | None = None
    if inner_surfaces:
        def _as_pv_surf(pts_in, tris_in):
            n_t = int(len(tris_in))
            flat = np.empty(n_t * 4, dtype=np.int64)
            flat[0::4] = 3
            flat[1::4] = tris_in[:, 0]
            flat[2::4] = tris_in[:, 1]
            flat[3::4] = tris_in[:, 2]
            return pv.PolyData(np.asarray(
                pts_in, dtype=np.float64,
            ), flat)

        def _point_inside(pt: np.ndarray, surf) -> bool:
            try:
                probe = pv.PolyData(np.atleast_2d(pt))
                sel = probe.select_enclosed_points(
                    surf, check_surface=False, tolerance=1.0e-9,
                )
                return bool(sel["SelectedPoints"][0])
            except Exception:                            # noqa: BLE001
                return False

        def _interior_seed(pts_in: np.ndarray, surf) -> list[float]:
            # The MESHED endo region is this fascicle CLIPPED to the cuff
            # window [z_lo, z_hi] (see `_clip_fascicle_to_z_window`); its
            # caps sit at z_lo / z_hi. The full-fascicle 3-D centroid can
            # land in z OUTSIDE that window when the cuff is off-centre on
            # the nerve (e.g. mid-trunk), so a centroid seed misses the
            # clipped region entirely — TetGen then leaves the endo region
            # untagged and the epi seed (in the same nerve interior) grabs
            # it, scrambling the endo/epi labels. Seed instead from the
            # z=0 cross-section, which is ALWAYS inside the window
            # (z_lo = -L/2 ≤ 0 ≤ +L/2 = z_hi), so the seed lands in the
            # meshed endo region for any cuff offset.
            try:
                _sl = surf.slice(normal=(0.0, 0.0, 1.0),
                                 origin=(0.0, 0.0, 0.0))
                _sp = np.asarray(_sl.points, dtype=np.float64)
            except Exception:                                # noqa: BLE001
                _sp = np.zeros((0, 3))
            if _sp.shape[0] >= 3:
                _c0 = _sp[:, :2].mean(axis=0)
                if _point_inside([_c0[0], _c0[1], 0.0], surf):
                    return [float(_c0[0]), float(_c0[1]), 0.0]
                _rmax = float(
                    np.linalg.norm(_sp[:, :2] - _c0, axis=1).max())
                for _fr in np.linspace(0.0, 0.85, 8):
                    for _ang in np.linspace(0.0, 2.0 * np.pi, 8,
                                            endpoint=False):
                        _p = [float(_c0[0] + _fr * _rmax * np.cos(_ang)),
                              float(_c0[1] + _fr * _rmax * np.sin(_ang)),
                              0.0]
                        if _point_inside(_p, surf):
                            return _p
            # Fallback (no z=0 slice): full-fascicle centroid + bbox walk.
            c = pts_in.mean(axis=0)
            if _point_inside(c, surf):
                return c.tolist()
            bb_lo = pts_in.min(axis=0)
            bb_hi = pts_in.max(axis=0)
            axis = int(np.argmax(bb_hi - bb_lo))
            for t in np.linspace(-0.4, 0.4, 9):
                p = c.copy()
                p[axis] += t * float(bb_hi[axis] - bb_lo[axis])
                if _point_inside(p, surf):
                    return p.tolist()
            # Last-resort fallback: return centroid even if
            # outside — TetGen will still tag SOMETHING for
            # this region; better than crashing the build.
            return c.tolist()

        nerve_surf_for_inside = nerve_surf
        fasc_pvs = []
        endo_seeds_all: list[list[float]] = []
        for ip, it in inner_surfaces:
            ip_arr = np.asarray(ip, dtype=np.float64)
            it_arr = np.asarray(it, dtype=np.int64)
            surf_pv = _as_pv_surf(ip_arr, it_arr)
            fasc_pvs.append(surf_pv)
            endo_seeds_all.append(_interior_seed(ip_arr, surf_pv))
        # First fascicle keeps the canonical "endo" slot; the
        # rest go in "endo_extra" so the mesh.py consumer can
        # emit one [tag=1, seed, vol] entry per fascicle without
        # breaking the legacy single-endo callers.
        seed_endo = endo_seeds_all[0]
        bundle_endo_extra = endo_seeds_all[1:]
        # Inter-fascicle epi seed — inside the nerve hull but
        # outside every fascicle. Sweep candidate points on a
        # polar grid around the cross-section centroid; first
        # survivor wins.
        for r_frac in [0.85, 0.7, 0.55, 0.4, 0.25]:
            if seed_epi_bundle is not None:
                break
            for theta in np.linspace(
                0.0, 2.0 * np.pi, 12, endpoint=False,
            ):
                p = np.array([
                    cent_xy[0]
                    + r_frac * r_nerve_max * np.cos(theta),
                    cent_xy[1]
                    + r_frac * r_nerve_max * np.sin(theta),
                    0.0,
                ])
                if not _point_inside(p, nerve_surf_for_inside):
                    continue
                if any(_point_inside(p, f) for f in fasc_pvs):
                    continue
                seed_epi_bundle = p.tolist()
                break
        if seed_epi_bundle is None:
            # Pathological: every grid candidate sat inside a
            # fascicle. Fall back to a tiny radial nudge from
            # the centroid towards the +x edge — better than
            # leaving the inter-fascicle region untagged.
            seed_epi_bundle = [
                float(cent_xy[0] + 0.95 * r_nerve_max),
                float(cent_xy[1]),
                0.0,
            ]
            say(
                "  ⚠ epi-region seed fallback: no clear "
                "inter-fascicle point found; using rim nudge"
            )
        say(
            f"  bundle seeds: "
            f"{len(inner_surfaces)} fascicle endo + epi"
        )
    seed_silicone = [0.5 * (R_ci_m + R_co_m), 0.0, 0.0]
    if use_scar:
        # F3.2-M3 — saline shifts to the annular gap between
        # R_scar and R_ci; scar takes the inner gap between
        # nerve outer and R_scar. Both seeds offset by the nerve
        # cross-section centroid so we stay on the equator inside
        # the right cylindrical band.
        r_sca = 0.5 * (
            r_nerve_max + (R_scar_m - cent_norm)
        )
        r_sal = 0.5 * (
            (R_scar_m - cent_norm) + (R_ci_m - cent_norm)
        )
        seed_scar = [
            float(cent_xy[0] + r_sca),
            float(cent_xy[1]),
            0.0,
        ]
        seed_saline = [
            float(cent_xy[0] + r_sal),
            float(cent_xy[1]),
            0.0,
        ]
    else:
        # Saline gap from nerve centroid out to the cuff inner
        # wall: half-way between nerve edge and the *closest-
        # side* R_ci.
        r_sal = 0.5 * (r_nerve_max + (R_ci_m - cent_norm))
        seed_saline = [
            float(cent_xy[0] + r_sal),
            float(cent_xy[1]),
            0.0,
        ]
        seed_scar = None
    say(f"  nerve cross-section @ z=0: centroid="
         f"({cent_xy[0]*1e3:+.2f}, {cent_xy[1]*1e3:+.2f}) mm, "
         f"r_max={r_nerve_max*1e3:.2f} mm")
    if use_scar:
        say(
            f"  seeds: endo={seed_endo}, scar={seed_scar}, "
            f"sal={seed_saline}, sil={seed_silicone}, "
            f"mus={seed_muscle}"
        )
    else:
        say(
            f"  seeds: endo={seed_endo}, "
            f"sal={seed_saline}, sil={seed_silicone}, "
            f"mus={seed_muscle}"
        )

    # Pack back into a pv.PolyData and return alongside the seeds
    # so the caller can hand them straight to TetGen.
    n_t = len(plc_tris)
    flat = np.empty(n_t * 4, dtype=np.int64)
    flat[0::4] = 3
    flat[1::4] = plc_tris[:, 0]
    flat[2::4] = plc_tris[:, 1]
    flat[3::4] = plc_tris[:, 2]
    plc_pv = pv.PolyData(plc_pts, flat)
    seed_positions = {
        "endo": seed_endo,
        "saline": seed_saline,
        "silicone": seed_silicone,
        "muscle": seed_muscle,
    }
    if use_scar and seed_scar is not None:
        seed_positions["scar"] = seed_scar
    if (use_epi and off_pts_c is not None
            and epi_surf_pts is not None
            and epi_surf_normals is not None):
        # Epi seed sits BETWEEN the nerve surface and its inward
        # offset, on the equator (z≈0). Half-step of the offset
        # along the inward normal puts the seed solidly inside
        # the thin annular shell that TetGen will carve as tag 5.
        idx_eq = int(np.argmin(np.abs(epi_surf_pts[:, 2])))
        seed_epi = (
            epi_surf_pts[idx_eq]
            - 0.5 * epi_thickness_m * epi_surf_normals[idx_eq]
        ).tolist()
        seed_positions["epi"] = seed_epi
        say(f"  seed: epi={seed_epi}")
    # V1 — µCT-bundle extras. `endo_extra` is the per-fascicle
    # seed list MINUS the primary one already in "endo"; mesh.py
    # iterates over it to emit additional [tag=1, seed, vol]
    # entries. `epi` from the bundle replaces the offset-shell
    # epi seed (both can't coexist — bundle has explicit epi).
    if bundle_endo_extra:
        seed_positions["endo_extra"] = bundle_endo_extra
    if seed_epi_bundle is not None:
        seed_positions["epi"] = seed_epi_bundle
    return plc_pv, seed_positions
