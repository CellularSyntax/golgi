# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""V1 Phase B — 3D nerve reconstruction from labelled µCT slices.

Two entry points, both ending at the same `Mesh` representation
so the binary-STL writer is shared:

  - `extrude_single_slice(...)`  one annotated slice + thickness
    → prismatic STL meshes. Internally runs through the same
    marching-cubes path as the multi-slice case by stacking the
    mask vertically with empty caps; keeps a single geometry
    code path.

  - `reconstruct_stack(...)`  a slice range, with ZOH-filled gaps
    between annotated frames → marching-cubes STL meshes.
    Per-fascicle split uses 3D connected components so a
    fascicle that runs continuously through several slices
    stays one mesh (and gets one .stl).

The split between "build meshes" and "write STLs" is deliberate:
the geometry layer has no disk I/O so it can be unit-tested in
memory, and the action handler decides where the files land
(`<project>/uct/nerve_3d/<timestamp>/`).

Dependencies: numpy (hard), scikit-image (hard — measure.label
+ measure.marching_cubes), scipy (soft — only for the optional
Gaussian smoothing pre-pass on the MC volume).
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------
# Progress / timing instrumentation
# --------------------------------------------------------------


def _emit_progress(
    on_progress: "Optional[Callable[[str, float], None]]",
    stage: str,
    elapsed: float,
    *,
    prefix: str = "[recon]",
) -> None:
    """Print + forward a per-stage timing checkpoint.

    Stdout line goes to the trame-server terminal for the
    developer; `on_progress(stage, elapsed)` (if supplied) lets
    the action layer push the same string into `state.busy_log`
    so the user sees real-time progress in the busy lightbox.
    """
    print(
        f"{prefix} {stage}: {elapsed:.2f} s",
        flush=True,
    )
    if on_progress is not None:
        try:
            on_progress(stage, float(elapsed))
        except Exception:                                 # noqa: BLE001
            pass


# --------------------------------------------------------------
# Mesh container
# --------------------------------------------------------------


@dataclass
class Mesh:
    """Triangle mesh in physical (mm) coordinates.

    `verts` is (N, 3) float32 in (x, y, z) mm — scikit-image's
    marching_cubes returns (Z, Y, X), so the constructor in
    `_mc_to_mesh` permutes to (X, Y, Z) before returning.
    `faces` is (M, 3) int32 — vertex indices per triangle, CCW
    when viewed from outside the solid (same convention as
    scikit-image MC + STL).
    `name` is the per-file slug used for `<name>.stl`.
    """
    verts: np.ndarray
    faces: np.ndarray
    name: str

    @property
    def n_triangles(self) -> int:
        return int(self.faces.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.verts.shape[0])


# --------------------------------------------------------------
# Per-mesh quality diagnostics (M13 Phase 1)
# --------------------------------------------------------------
#
# Goal: when reconstruct_stack / extrude_single_slice produce a
# surface that later trips the PLC / TetGen step, we want enough
# per-surface info to pin down which mesh broke and which quality
# dimension (non-manifold edges, boundary edges, degenerate /
# self-intersecting triangles, near-zero min-angle slivers, etc.)
# is to blame — without having to open the STL in MeshLab.
#
# `mesh_quality_report(mesh)` returns a flat dict of metrics.
# `print_mesh_quality_summary(report)` prints a single-line
# scannable summary to stdout. The action layer can also dump
# the full dicts to JSON for offline inspection.


def mesh_quality_report(mesh: "Mesh") -> dict:
    """Compute a flat dict of per-surface quality metrics.

    Always-populated fields (numpy-only):
      n_triangles, n_vertices, bbox_min_mm, bbox_max_mm,
      bbox_size_mm, edge_{min,median,max,p99}_mm,
      area_{min,median,max}_mm2, n_degenerate_tris.

    Populated when pyvista is available:
      volume_mm3, surface_area_mm2,
      n_boundary_edges, n_non_manifold_edges,
      is_closed, is_manifold,
      tri_quality_radius_ratio_{min,median,max,p99},
      min_angle_deg_{min,median}, n_tris_min_angle_below_5deg,
      n_tris_radius_ratio_above_5.

    Populated when trimesh is available:
      watertight, winding_consistent, volume_signed_mm3.

    Falls back gracefully if pyvista / trimesh missing — the
    numpy-only fields are enough to spot most TetGen-blocking
    failures (degenerate / very-thin triangles).
    """
    report: dict = {
        "name": str(mesh.name),
        "n_triangles": int(mesh.n_triangles),
        "n_vertices": int(mesh.n_vertices),
    }
    pts = np.asarray(mesh.verts, dtype=np.float64)
    tris = np.asarray(mesh.faces, dtype=np.int64)
    if pts.size == 0 or tris.size == 0:
        return report
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    report["bbox_min_mm"] = [float(x) for x in bb_min]
    report["bbox_max_mm"] = [float(x) for x in bb_max]
    report["bbox_size_mm"] = [float(x) for x in (bb_max - bb_min)]

    # Per-edge lengths (3 edges per tri, vectorised).
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    edge_lens = np.concatenate([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ])
    if edge_lens.size:
        report["edge_min_mm"] = float(np.min(edge_lens))
        report["edge_median_mm"] = float(np.median(edge_lens))
        report["edge_max_mm"] = float(np.max(edge_lens))
        report["edge_p99_mm"] = float(np.quantile(edge_lens, 0.99))

    # Per-tri area (0.5 |a × b|).
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if areas.size:
        report["area_min_mm2"] = float(np.min(areas))
        report["area_median_mm2"] = float(np.median(areas))
        report["area_max_mm2"] = float(np.max(areas))
        # `degenerate` = numerically zero-area tris (collinear
        # verts). TetGen treats these as invalid input.
        report["n_degenerate_tris"] = int((areas < 1e-12).sum())

    # PyVista-backed metrics (manifoldness + cell quality).
    try:
        import pyvista as pv
        faces_pv = np.column_stack([
            np.full(tris.shape[0], 3, dtype=np.int64),
            tris.astype(np.int64),
        ]).ravel()
        pdata = pv.PolyData(pts, faces_pv)
        report["volume_mm3"] = float(pdata.volume)
        report["surface_area_mm2"] = float(pdata.area)

        # Edge topology counts via extract_feature_edges:
        #   boundary_edges: edge in only 1 triangle → mesh is
        #     open / has a hole. TetGen needs closed surfaces.
        #   non_manifold_edges: edge in 3+ triangles → mesh is
        #     branching / fused. TetGen segfaults.
        be = pdata.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            non_manifold_edges=False,
            manifold_edges=False,
        )
        nme = pdata.extract_feature_edges(
            boundary_edges=False,
            feature_edges=False,
            non_manifold_edges=True,
            manifold_edges=False,
        )
        report["n_boundary_edges"] = int(be.n_cells)
        report["n_non_manifold_edges"] = int(nme.n_cells)
        report["is_closed"] = bool(be.n_cells == 0)
        report["is_manifold"] = bool(nme.n_cells == 0)

        # PyVista's compute_cell_quality has shipped under two
        # different array-key conventions across versions:
        # older versions returned a "CellQuality" array,
        # newer ones return "Quality" (or sometimes "Cell
        # Quality" with a space). Try them in order and use
        # whichever lands. Also fall back to a numpy-only
        # radius-ratio implementation when pyvista refuses
        # entirely (some VTK builds don't ship the quality
        # filter).
        def _read_quality(
            pd: "pv.PolyData", measure: str,
        ) -> "np.ndarray | None":
            try:
                qpd = pd.compute_cell_quality(
                    quality_measure=measure,
                )
            except Exception:                             # noqa: BLE001
                return None
            for key in ("CellQuality", "Quality", "Cell Quality"):
                if key in qpd.array_names:
                    return np.asarray(
                        qpd[key], dtype=np.float64,
                    )
            return None

        rr = _read_quality(pdata, "radius_ratio")
        if rr is None:
            # Numpy fallback — Heron's formula radius ratio.
            # circumradius = a·b·c / (4·area); inradius = area/s
            # where s = (a+b+c)/2. RR = circumradius / (2·inradius)
            # = a·b·c·s / (8·area²). For an equilateral tri = 1.
            try:
                a = np.linalg.norm(v1 - v0, axis=1)
                b = np.linalg.norm(v2 - v1, axis=1)
                c = np.linalg.norm(v0 - v2, axis=1)
                s = 0.5 * (a + b + c)
                area = areas if "area_median_mm2" in report else (
                    0.5 * np.linalg.norm(
                        np.cross(v1 - v0, v2 - v0), axis=1,
                    )
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    rr = (a * b * c * s) / (8.0 * area * area)
                rr = np.where(np.isfinite(rr), rr, np.inf)
                report["radius_ratio_via_numpy_fallback"] = True
            except Exception as ex:                       # noqa: BLE001
                report["radius_ratio_error"] = str(ex)
                rr = None

        if rr is not None:
            # VTK radius_ratio for triangles: 1.0 = equilateral,
            # large = slivery / degenerate. > 5 → "very bad"
            # (radius ratio 5:1 ≈ 10°-ish min angle).
            rr_finite = rr[np.isfinite(rr)]
            if rr_finite.size:
                report["tri_quality_radius_ratio_min"] = float(
                    rr_finite.min(),
                )
                report["tri_quality_radius_ratio_median"] = float(
                    np.median(rr_finite),
                )
                report["tri_quality_radius_ratio_max"] = float(
                    rr_finite.max(),
                )
                report["tri_quality_radius_ratio_p99"] = float(
                    np.quantile(rr_finite, 0.99),
                )
            report["n_tris_radius_ratio_above_5"] = int(
                (rr > 5.0).sum(),
            )
            report["n_tris_radius_ratio_infinite"] = int(
                (~np.isfinite(rr)).sum(),
            )

        ma = _read_quality(pdata, "min_angle")
        if ma is None:
            # Numpy fallback for min angle (degrees). For each
            # tri compute all three angles via the cosine rule
            # and pick the smallest.
            try:
                a = np.linalg.norm(v1 - v0, axis=1)
                b = np.linalg.norm(v2 - v1, axis=1)
                c = np.linalg.norm(v0 - v2, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cos_a = (b * b + c * c - a * a) / (2.0 * b * c)
                    cos_b = (a * a + c * c - b * b) / (2.0 * a * c)
                    cos_c = (a * a + b * b - c * c) / (2.0 * a * b)
                ang_a = np.degrees(np.arccos(
                    np.clip(cos_a, -1.0, 1.0),
                ))
                ang_b = np.degrees(np.arccos(
                    np.clip(cos_b, -1.0, 1.0),
                ))
                ang_c = np.degrees(np.arccos(
                    np.clip(cos_c, -1.0, 1.0),
                ))
                ma = np.minimum(
                    np.minimum(ang_a, ang_b), ang_c,
                )
                report["min_angle_via_numpy_fallback"] = True
            except Exception as ex:                       # noqa: BLE001
                report["min_angle_error"] = str(ex)
                ma = None

        if ma is not None:
            ma_finite = ma[np.isfinite(ma)]
            if ma_finite.size:
                report["min_angle_deg_min"] = float(ma_finite.min())
                report["min_angle_deg_median"] = float(
                    np.median(ma_finite),
                )
            # Slivers — TetGen really doesn't like these.
            report["n_tris_min_angle_below_5deg"] = int(
                ((ma < 5.0) & np.isfinite(ma)).sum(),
            )
    except Exception as ex:                               # noqa: BLE001
        report["pyvista_error"] = str(ex)

    # trimesh-backed sanity checks. Gated on mesh size — for
    # very large meshes (>250k tris) `is_watertight` and
    # `is_winding_consistent` traverse the entire half-edge
    # structure and can take 30+ seconds, dominating overall
    # reconstruction time. The pyvista-side `is_closed` /
    # `is_manifold` flags above cover the same diagnostic so
    # we don't lose much by skipping the trimesh ones.
    if int(tris.shape[0]) <= 250_000:
        try:
            import trimesh
            tm = trimesh.Trimesh(
                vertices=pts,
                faces=tris.astype(np.int64),
                process=False,
                validate=False,
            )
            try:
                report["watertight"] = bool(tm.is_watertight)
            except Exception:                             # noqa: BLE001
                pass
            try:
                report["winding_consistent"] = bool(
                    tm.is_winding_consistent,
                )
            except Exception:                             # noqa: BLE001
                pass
            try:
                report["volume_signed_mm3"] = float(tm.volume)
            except Exception:                             # noqa: BLE001
                pass
        except Exception as ex:                           # noqa: BLE001
            report["trimesh_error"] = str(ex)
    else:
        report["trimesh_skipped"] = (
            f"mesh has {tris.shape[0]:,} tris > 250k threshold; "
            "trimesh checks would dominate runtime — relying on "
            "pyvista is_closed/is_manifold instead"
        )

    return report


def print_mesh_quality_summary(
    report: dict,
    *,
    prefix: str = "[mesh-quality]",
) -> None:
    """One-line stdout summary of a mesh_quality_report dict.

    Designed to be cheap to read in the terminal while the user
    is running a reconstruction — pulls out the metrics most
    likely to point at a TetGen failure:
      tris / verts / volume / closed? / manifold? / worst
      triangle quality / count of slivers / count of degenerate
      tris.
    """
    name = report.get("name", "<?>")
    n_tri = report.get("n_triangles", "?")
    n_vert = report.get("n_vertices", "?")
    vol = report.get("volume_mm3")
    closed = report.get("is_closed")
    manifold = report.get("is_manifold")
    n_be = report.get("n_boundary_edges")
    n_nme = report.get("n_non_manifold_edges")
    n_deg = report.get("n_degenerate_tris")
    rr_max = report.get("tri_quality_radius_ratio_max")
    n_bad_rr = report.get("n_tris_radius_ratio_above_5")
    a_min = report.get("min_angle_deg_min")
    n_slivers = report.get("n_tris_min_angle_below_5deg")
    watertight = report.get("watertight")

    def _f(v, fmt=".3g"):
        if v is None:
            return "?"
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return str(v)

    print(
        f"{prefix} {name}: "
        f"tris={n_tri} verts={n_vert} "
        f"vol_mm3={_f(vol)} "
        f"closed={closed} manifold={manifold} "
        f"boundary_edges={n_be} non_manifold_edges={n_nme} "
        f"degenerate={n_deg} "
        f"worst_rr={_f(rr_max, '.2f')} bad_rr>5={n_bad_rr} "
        f"min_angle={_f(a_min, '.1f')} slivers<5deg={n_slivers} "
        f"watertight={watertight}",
        flush=True,
    )


def inter_surface_report(
    meshes: list["Mesh"],
    *,
    touch_threshold_mm: float = 0.05,
) -> dict:
    """Pairwise inter-surface diagnostics for PLC / TetGen
    triage. The per-surface mesh_quality_report covers each
    surface in isolation — but TetGen-blocking failures very
    often come from inter-surface relationships the per-surface
    pass can't see:

      * a fascicle whose vertices straddle the epi boundary
        (the epi-clip wasn't a perfect inside check)
      * a fascicle that touches the epi at a single voxel
        (collapses into a non-manifold edge in the PLC)
      * two fascicles touching each other (same problem)
      * a fascicle that's outside the epi entirely (PLC
        nested-domain assembly will misclassify it)

    Two checks, both cheap:

      1. **Fascicle ↔ epi enclosure** — `pyvista.PolyData
         .compute_implicit_distance(epi)` gives a signed
         distance for each fascicle vertex (negative = inside
         the epi's enclosed volume; positive = outside).
         Reports min/max/median signed distance, vertex
         counts on each side of the boundary, and a
         `touches_epi` flag (min absolute distance below
         `touch_threshold_mm`).

      2. **Fascicle ↔ fascicle min vertex distance** — scipy
         cKDTree query of one fascicle's vertex cloud against
         another. Reports min distance per pair + a touches
         flag.

    `touch_threshold_mm` default 0.05 mm (= 50 µm) ≈ a few
    µCT voxels of slack — anything closer than that on a
    µCT-reconstructed surface is almost certainly a real
    contact, not just smoothing wobble.

    Falls back gracefully when pyvista / scipy are missing;
    the returned dict will just be sparser.
    """
    report: dict = {
        "n_surfaces": int(len(meshes)),
        "names": [str(m.name) for m in meshes],
        "touch_threshold_mm": float(touch_threshold_mm),
    }
    if len(meshes) < 2:
        return report

    # Build PolyData lookups once. The implicit-distance pass
    # iterates over every fascicle so building polydata once
    # and reusing matters for big epi meshes.
    try:
        import pyvista as pv
    except Exception as ex:                                # noqa: BLE001
        report["pyvista_error"] = str(ex)
        return report

    pds: dict[str, "pv.PolyData"] = {}
    for m in meshes:
        n_t = int(m.faces.shape[0])
        if n_t == 0:
            continue
        flat = np.empty(n_t * 4, dtype=np.int64)
        flat[0::4] = 3
        flat[1::4] = m.faces[:, 0]
        flat[2::4] = m.faces[:, 1]
        flat[3::4] = m.faces[:, 2]
        pds[str(m.name)] = pv.PolyData(
            np.asarray(m.verts, dtype=np.float64),
            flat,
        )

    epi = pds.get("epi")
    fascicles = {k: v for k, v in pds.items() if k != "epi"}

    if epi is not None and fascicles:
        fasc_vs_epi: dict[str, dict] = {}
        for name, pd in fascicles.items():
            try:
                sampled = pd.compute_implicit_distance(epi)
                d = np.asarray(
                    sampled["implicit_distance"],
                    dtype=np.float64,
                )
            except Exception as ex:                       # noqa: BLE001
                fasc_vs_epi[name] = {
                    "implicit_distance_error": str(ex),
                }
                continue
            n_inside = int((d < 0.0).sum())
            n_outside = int((d > 0.0).sum())
            min_abs = float(np.min(np.abs(d))) if d.size else 0.0
            entry = {
                "n_verts": int(d.size),
                "n_inside_epi": n_inside,
                "n_outside_epi": n_outside,
                "signed_distance_min_mm": float(d.min()),
                "signed_distance_max_mm": float(d.max()),
                "signed_distance_median_mm": float(np.median(d)),
                "min_abs_distance_to_epi_mm": min_abs,
                # Fully-inside = every vertex has signed d ≤ 0,
                # given pyvista's convention for closed surfaces
                # (negative = inside the source's enclosed volume).
                # Note: the sign depends on the source surface's
                # face normals, so a flipped epi would invert
                # this. We also expose the raw vert counts so
                # the user can sanity-check.
                "fully_inside_epi": bool(
                    n_outside == 0 and n_inside > 0,
                ),
                "straddles_epi_boundary": bool(
                    n_inside > 0 and n_outside > 0,
                ),
                "touches_epi": bool(
                    min_abs < float(touch_threshold_mm),
                ),
            }
            fasc_vs_epi[name] = entry
        report["fasc_vs_epi"] = fasc_vs_epi

    if len(fascicles) >= 2:
        try:
            from scipy.spatial import cKDTree
        except Exception as ex:                           # noqa: BLE001
            report["scipy_error"] = str(ex)
            return report
        fasc_pairs: dict[str, dict] = {}
        names = sorted(fascicles.keys())
        for i, n1 in enumerate(names):
            pts1 = np.asarray(fascicles[n1].points)
            for n2 in names[i + 1:]:
                pts2 = np.asarray(fascicles[n2].points)
                try:
                    tree2 = cKDTree(pts2)
                    d, _ = tree2.query(pts1, k=1)
                except Exception as ex:                   # noqa: BLE001
                    fasc_pairs[f"{n1}__vs__{n2}"] = {
                        "error": str(ex),
                    }
                    continue
                min_d = float(d.min()) if d.size else 0.0
                fasc_pairs[f"{n1}__vs__{n2}"] = {
                    "min_vertex_distance_mm": min_d,
                    "touches": bool(
                        min_d < float(touch_threshold_mm),
                    ),
                }
        report["fasc_vs_fasc"] = fasc_pairs

    return report


def print_inter_surface_summary(
    report: dict,
    *,
    prefix: str = "[inter-surface]",
) -> None:
    """Readable stdout summary of inter_surface_report.

    Highlights the diagnostic-relevant facts first: any
    fascicle that straddles the epi boundary or touches the
    epi/another fascicle is the most likely PLC / TetGen
    blocker, so those flags get their own line.
    """
    print(
        f"{prefix} === pairwise checks "
        f"({report.get('n_surfaces', '?')} surfaces) ===",
        flush=True,
    )

    fasc_vs_epi = report.get("fasc_vs_epi", {}) or {}
    for name, data in fasc_vs_epi.items():
        if "implicit_distance_error" in data:
            print(
                f"{prefix} {name} vs epi: ERROR "
                f"{data['implicit_distance_error']}",
                flush=True,
            )
            continue
        sd_min = data.get("signed_distance_min_mm")
        sd_max = data.get("signed_distance_max_mm")
        n_in = data.get("n_inside_epi")
        n_out = data.get("n_outside_epi")
        min_abs = data.get("min_abs_distance_to_epi_mm")
        fully_in = data.get("fully_inside_epi")
        straddles = data.get("straddles_epi_boundary")
        touches = data.get("touches_epi")

        def _f(v, fmt=".4g"):
            return (
                format(float(v), fmt)
                if v is not None else "?"
            )

        print(
            f"{prefix} {name} vs epi: "
            f"signed_d=[{_f(sd_min)}, {_f(sd_max)}] mm  "
            f"inside={n_in}  outside={n_out}  "
            f"min_|d|={_f(min_abs)} mm  "
            f"fully_inside={fully_in}  "
            f"straddles={straddles}  "
            f"touches={touches}",
            flush=True,
        )

    fasc_pairs = report.get("fasc_vs_fasc", {}) or {}
    for pair, data in fasc_pairs.items():
        if "error" in data:
            print(
                f"{prefix} {pair}: ERROR {data['error']}",
                flush=True,
            )
            continue
        min_d = data.get("min_vertex_distance_mm")
        touches = data.get("touches")
        print(
            f"{prefix} {pair}: "
            f"min_d={float(min_d):.4g} mm  touches={touches}",
            flush=True,
        )


def report_mesh_quality_batch(
    meshes: list["Mesh"],
    *,
    prefix: str = "[mesh-quality]",
) -> list[dict]:
    """Compute + print quality reports for every mesh in `meshes`.
    Returns the list of report dicts so the action layer can
    write them to JSON without recomputing."""
    out: list[dict] = []
    print(
        f"{prefix} === per-surface quality reports "
        f"({len(meshes)} mesh{'es' if len(meshes) != 1 else ''}) ===",
        flush=True,
    )
    for m in meshes:
        try:
            rep = mesh_quality_report(m)
        except Exception as ex:                           # noqa: BLE001
            print(
                f"{prefix} {m.name}: quality report failed: "
                f"{ex}",
                flush=True,
            )
            rep = {"name": str(m.name), "report_error": str(ex)}
        out.append(rep)
        try:
            print_mesh_quality_summary(rep, prefix=prefix)
        except Exception:                                 # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------
# ZOH (zero-order hold) fill for sparse per-slice annotations
# --------------------------------------------------------------


def zoh_fill(
    masks_by_slice: dict[int, np.ndarray],
    slice_range: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Zero-order-hold the sparse map `masks_by_slice` across the
    inclusive index range `slice_range`.

    Behaviour:
      * idx ∈ masks_by_slice → use that mask.
      * idx between two annotated slices → hold the PREVIOUS
        annotated mask forward (classical ZOH).
      * idx before the first annotated slice → hold the first
        annotated mask backwards (avoids an empty cap that
        would leave the MC surface open on the first frame).
      * idx after the last annotated slice → hold the last
        annotated mask forward (same logic for the trailing
        cap).

    Returns a dict from every idx in the range to a 2D bool
    mask. An empty input map returns an empty dict — caller
    decides whether that's an error.
    """
    if not masks_by_slice:
        return {}
    s_lo, s_hi = slice_range
    in_range = sorted(
        i for i in masks_by_slice if s_lo <= i <= s_hi
    )
    if not in_range:
        # No annotated frames inside the range; fall back to the
        # globally-nearest annotated slice so the user at least
        # gets SOMETHING back rather than silent emptiness.
        all_idx = sorted(masks_by_slice.keys())
        mid = (s_lo + s_hi) // 2
        closest = min(all_idx, key=lambda i: abs(i - mid))
        nearest = masks_by_slice[closest]
        return {i: nearest for i in range(s_lo, s_hi + 1)}
    out: dict[int, np.ndarray] = {}
    first_idx = in_range[0]
    cur_idx = first_idx
    for i in range(s_lo, s_hi + 1):
        if i in masks_by_slice:
            cur_idx = i
            out[i] = masks_by_slice[i]
        elif i < first_idx:
            # Before the first annotated frame — hold the first
            # mask backwards so the leading cap stays closed.
            out[i] = masks_by_slice[first_idx]
        else:
            out[i] = masks_by_slice[cur_idx]
    return out


# --------------------------------------------------------------
# Marching-cubes helpers
# --------------------------------------------------------------


def _pad_volume(
    volume: np.ndarray, *, pad: int = 1,
) -> np.ndarray:
    """Pad a 3D bool volume with `pad` False slices on every
    side. Closes the surface at the volume's domain boundary —
    without this, MC leaves the boundary triangles open and the
    resulting STL isn't watertight, which most downstream
    re-meshers (TetGen) reject."""
    d, h, w = volume.shape
    out = np.zeros(
        (d + 2 * pad, h + 2 * pad, w + 2 * pad), dtype=bool,
    )
    out[pad:pad + d, pad:pad + h, pad:pad + w] = volume
    return out


def _mc_to_mesh(
    volume: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    name: str,
    smooth_sigma: Optional[float],
    pad: int = 1,
    # M30 — decoupled XY and Z physical-mm Gaussian sigmas.
    # When both are provided they OVERRIDE the legacy
    # `smooth_sigma` scaling (which set sigma_z = sigma_xy ×
    # xy_spacing / z_spacing, producing near-zero Z smoothing
    # on extruded prisms with large z_spacing and leaving
    # marching-cubes axis-aligned step artefacts visible as
    # periodic ripples on the lateral surface). Each is in
    # physical mm so the user picks them independently of
    # voxel spacing. Set either to 0 to disable that axis;
    # set both None to fall back to legacy scaling.
    smooth_sigma_xy_mm: Optional[float] = None,
    smooth_sigma_z_mm: Optional[float] = None,
) -> Optional[Mesh]:
    """Marching cubes on a 3D bool volume → `Mesh` in physical
    units.

    `volume` is (Z, Y, X) bool. `spacing` is (z_mm, y_mm, x_mm)
    — matches scikit-image's expected axis order. When
    `smooth_sigma > 0`, the volume is float-cast and Gaussian-
    smoothed before MC; this rounds off staircase artefacts
    from ZOH-held slices and slice quantisation.

    Returns None when the volume has no True voxels so the
    caller can skip writing an empty STL.
    """
    if not volume.any():
        return None
    try:
        from skimage import measure
    except ImportError as ex:
        raise RuntimeError(
            "3D reconstruction needs scikit-image (skimage). "
            f"Import failed: {ex}",
        ) from ex
    padded = _pad_volume(volume, pad=pad)
    vol_f = padded.astype(np.float32)
    # Decide which smoothing path to use. New decoupled
    # XY/Z mm path wins when either knob is set; otherwise
    # fall back to the legacy isotropic-physical-mm scaling.
    use_decoupled = (
        smooth_sigma_xy_mm is not None
        or smooth_sigma_z_mm is not None
    )
    try:
        from scipy import ndimage as ndi
    except ImportError:
        ndi = None
    # M31 — `mode='constant', cval=0.0` is critical: scipy's
    # default `mode='reflect'` makes the smoothing kernel
    # at the Z boundary REFLECT mask values back from inside
    # the volume, lifting the boundary slab above the 0.5
    # marching-cubes isolevel. MC then doesn't cross 0.5 at
    # the cap and the resulting mesh has open ends (the user
    # saw closed=False, ~3000 boundary edges on every
    # surface). With `mode='constant', cval=0.0` the kernel
    # treats out-of-volume voxels as background, so the
    # boundary slab smooths DOWN through 0.5, MC closes the
    # cap, and the mesh is watertight.
    if use_decoupled and ndi is not None:
        z_mm, y_mm, x_mm = (
            float(spacing[0]),
            float(spacing[1]),
            float(spacing[2]),
        )
        s_xy_mm = float(smooth_sigma_xy_mm or 0.0)
        s_z_mm = float(smooth_sigma_z_mm or 0.0)
        sigma_z = max(0.0, s_z_mm / z_mm) if z_mm > 0 else 0.0
        sigma_y = max(0.0, s_xy_mm / y_mm) if y_mm > 0 else 0.0
        sigma_x = max(0.0, s_xy_mm / x_mm) if x_mm > 0 else 0.0
        if (sigma_z + sigma_y + sigma_x) > 0:
            vol_f = ndi.gaussian_filter(
                vol_f,
                sigma=(sigma_z, sigma_y, sigma_x),
                mode="constant",
                cval=0.0,
            )
    elif (
        smooth_sigma is not None
        and smooth_sigma > 0
        and ndi is not None
    ):
        # Legacy isotropic-physical scaling. Kept so existing
        # callers without the new kwargs see unchanged
        # behaviour. Same boundary-mode fix applies.
        xy_spacing_mm = float(min(spacing[1], spacing[2]))
        sigma_per_axis = tuple(
            float(smooth_sigma) * (xy_spacing_mm / float(s))
            for s in spacing
        )
        vol_f = ndi.gaussian_filter(
            vol_f,
            sigma=sigma_per_axis,
            mode="constant",
            cval=0.0,
        )
    # Defensive guard: when the volume is too small to clear
    # the 0.5 isolevel after smoothing (or — symmetrically —
    # so dense that no voxel sits below 0.5), marching_cubes
    # raises `Surface level must be within volume data range`.
    # Both cases mean "no surface exists here"; return None so
    # the caller can skip writing an empty / degenerate mesh
    # instead of taking down the whole reconstruction.
    if vol_f.max() <= 0.5 or vol_f.min() >= 0.5:
        return None
    try:
        verts, faces, _normals, _vals = measure.marching_cubes(
            vol_f, level=0.5, spacing=spacing,
        )
    except ValueError:
        return None
    # Undo the pad offset (in physical units).
    pz, py, px = spacing
    verts = verts - np.array(
        [pad * pz, pad * py, pad * px], dtype=np.float64,
    )
    # scikit-image returns verts in (Z, Y, X). Convert to
    # (X, Y, Z) — STL convention + matches what trimesh +
    # gmsh + Paraview all expect when loading a file.
    verts = verts[:, [2, 1, 0]].astype(np.float32)
    faces = faces.astype(np.int32)
    return Mesh(verts=verts, faces=faces, name=name)


def _polygon_extrude_component(
    mask: np.ndarray,
    *,
    voxel_xy_mm: float,
    z_lo: float,
    z_hi: float,
    name: str,
    rdp_tol_mm: float,
    # M48 → M48b — tuned default 0.3 mm axial step.
    # Earlier values: 1.0 mm (M45) gave 20:1 lateral aspect
    # ratio, median q_rr ≈ 0.07; 0.1 mm (M48) gave 2:1 aspect
    # ratio, q_rr ≈ 0.58 but 136 k tris for a 50 mm sheep VN
    # epi which pushes TetGen's "Recovering boundaries" stage
    # into 10+ min territory and exceeds the mesh-build's 50 k
    # decimation target. 0.3 mm is the sweet spot: ~46 k tris
    # (skips decimation by being under the 50 k cap), median
    # q_rr ≈ 0.35-0.4, lateral aspect ratio ~6:1. TetGen
    # processes the resulting PLC in 1-2 minutes.
    target_axial_mm: float = 0.3,
    mask_presmooth_sigma_vox: float = 2.5,
) -> Optional[Mesh]:
    """Build a watertight prismatic mesh from a single connected
    2D bool mask by tracing its boundary, simplifying, and
    extruding between `z_lo` and `z_hi`.

    M45 — Replaces marching-cubes for the single-slice extrude
    path. The pixelated mask boundary produces a stair-stepped
    polyline; Ramer-Douglas-Peucker simplification (via
    `skimage.measure.approximate_polygon`) collapses the
    staircase into a smooth polygon with O(perimeter / rdp_tol)
    vertices. Each closed polygon becomes a prism: top cap +
    bottom cap (earcut'd with hole support) + lateral quads
    between consecutive vertices.

    Returns None when the mask is too small to extract a
    sensible boundary (< 3 verts after simplification).
    """
    from skimage import measure
    if not mask.any():
        return None
    # M45b — Gaussian pre-smooth the mask BEFORE tracing.
    # Without this, find_contours follows the 1-voxel staircase
    # pattern of the pixel mask exactly; the resulting polyline
    # has hundreds of micro-segments at right angles to each
    # other. RDP can only collapse near-collinear runs, so on
    # a staircase it produces a faceted-polygon look (visible
    # to the user); turning RDP tolerance down to keep more
    # vertices then preserves the staircase explicitly, and
    # extruding it yields lateral quads with extreme aspect
    # ratios + 0° interior angles at every stair corner →
    # worst_rr in the trillions. A small Gaussian filter
    # (sigma ≈ 1.5 voxel ≈ 15 µm at 10 µm pitch) is enough to
    # blur the staircase below the 0.5 iso-level threshold,
    # producing a smooth contour that RDP can then sample
    # without artefacts.
    if float(mask_presmooth_sigma_vox) > 0:
        from scipy.ndimage import gaussian_filter as _gauss
        mask_f = _gauss(
            mask.astype(np.float32),
            sigma=float(mask_presmooth_sigma_vox),
            mode="constant", cval=0.0,
        )
    else:
        mask_f = mask.astype(np.float32)
    contours_array = measure.find_contours(
        mask_f, level=0.5,
        positive_orientation="high",
    )
    if not contours_array:
        return None
    # Convert (row, col) array coords → (x, y) physical mm.
    contours_xy: list[np.ndarray] = []
    for c in contours_array:
        # Drop the duplicate closing vertex (find_contours
        # returns first==last on closed curves).
        if (
            len(c) >= 2
            and np.allclose(c[0], c[-1], atol=1.0e-9)
        ):
            c = c[:-1]
        xy = np.column_stack([
            c[:, 1] * voxel_xy_mm,
            c[:, 0] * voxel_xy_mm,
        ])
        # Ramer-Douglas-Peucker simplification — kills the
        # 1-voxel staircase. Loop closed by passing the first
        # vertex appended at the end then stripping it again.
        closed = np.vstack([xy, xy[:1]])
        simp = measure.approximate_polygon(
            closed, tolerance=float(rdp_tol_mm),
        )
        if len(simp) >= 2 and np.allclose(
            simp[0], simp[-1], atol=1.0e-9,
        ):
            simp = simp[:-1]
        if len(simp) < 3:
            continue
        contours_xy.append(simp)
    if not contours_xy:
        return None

    def _signed_area(poly: np.ndarray) -> float:
        x = poly[:, 0]; y = poly[:, 1]
        return 0.5 * float(
            np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
        )

    def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
        # Ray-cast.
        x, y = float(pt[0]), float(pt[1])
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-30)
                + xi
            ):
                inside = not inside
            j = i
        return inside

    # Sort by |area| descending. Largest = outer; smaller-and-
    # inside-outer = holes. For typical fascicles (no internal
    # voids) we just get 1 ring.
    contours_sorted = sorted(
        contours_xy,
        key=lambda p: abs(_signed_area(p)),
        reverse=True,
    )
    outer_xy = contours_sorted[0]
    if _signed_area(outer_xy) < 0:
        outer_xy = outer_xy[::-1]  # ensure CCW
    holes_xy: list[np.ndarray] = []
    for cand in contours_sorted[1:]:
        if not _point_in_poly(cand[0], outer_xy):
            # Sibling component — shouldn't happen for a
            # single-CC mask but guard anyway.
            continue
        h = cand
        if _signed_area(h) > 0:
            h = h[::-1]  # holes must be CW
        holes_xy.append(h)

    # Earcut the cap polygon (with hole support).
    import mapbox_earcut as ec
    if holes_xy:
        combined = np.ascontiguousarray(
            np.vstack([outer_xy] + holes_xy), dtype=np.float64,
        )
        rings = [len(outer_xy)]
        for h in holes_xy:
            rings.append(rings[-1] + len(h))
        rings_arr = np.array(rings, dtype=np.uint32)
        cap_tris2d = np.asarray(
            ec.triangulate_float64(combined, rings_arr),
            dtype=np.int64,
        ).reshape(-1, 3)
        cap_xy = combined
    else:
        cap_xy = np.ascontiguousarray(outer_xy, dtype=np.float64)
        rings_arr = np.array([len(outer_xy)], dtype=np.uint32)
        cap_tris2d = np.asarray(
            ec.triangulate_float64(cap_xy, rings_arr),
            dtype=np.int64,
        ).reshape(-1, 3)
    if cap_tris2d.shape[0] == 0:
        return None

    # Axial subdivision: keep lateral triangle aspect ratio
    # reasonable when the extrusion is much longer than the
    # cross-section perimeter. n_axial = ceil(thickness /
    # target_axial_mm), clamped to ≥1.
    thickness = float(z_hi - z_lo)
    n_axial = max(1, int(np.ceil(
        thickness / max(float(target_axial_mm), 1.0e-6),
    )))
    # Build n_axial+1 layers of cap_xy verts at evenly spaced z.
    n_cap = int(cap_xy.shape[0])
    z_levels = np.linspace(
        float(z_lo), float(z_hi), n_axial + 1,
    )
    layer_verts = []
    for z in z_levels:
        layer_verts.append(
            np.column_stack([cap_xy, np.full(n_cap, float(z))]),
        )
    all_verts = np.vstack(layer_verts)
    # Cap indexing helpers — block k's vertex `i` is at global
    # index `k * n_cap + i`.
    bot_off = 0
    top_off = n_axial * n_cap
    # Caps:
    #   bottom: flip winding so normal points -z
    #   top: keep earcut's CCW so normal points +z
    tris_bot = np.column_stack([
        cap_tris2d[:, 0],
        cap_tris2d[:, 2],
        cap_tris2d[:, 1],
    ]).astype(np.int64) + bot_off
    tris_top = (cap_tris2d + top_off).astype(np.int64)
    # Lateral strips: one ring × n_axial slabs × 2 tris per
    # edge. Winding follows the outer-CCW / hole-CW convention
    # which gives outward normals automatically (see
    # _build_cylinder_lateral in plc.py for the same pattern).
    ring_starts = [0]
    for k in range(len(rings_arr) - 1):
        ring_starts.append(int(rings_arr[k]))
    ring_starts.append(int(rings_arr[-1]))
    lateral_tris: list[np.ndarray] = []
    for r_idx in range(len(rings_arr)):
        r0 = ring_starts[r_idx]
        r1 = ring_starts[r_idx + 1]
        m = r1 - r0
        for slab in range(n_axial):
            base_lo = slab * n_cap
            base_hi = (slab + 1) * n_cap
            strip = np.empty((2 * m, 3), dtype=np.int64)
            for i in range(m):
                j = (i + 1) % m
                a_lo = base_lo + r0 + i
                b_lo = base_lo + r0 + j
                a_hi = base_hi + r0 + i
                b_hi = base_hi + r0 + j
                strip[2 * i] = [a_lo, b_lo, b_hi]
                strip[2 * i + 1] = [a_lo, b_hi, a_hi]
            lateral_tris.append(strip)
    all_tris = np.vstack(
        [tris_bot, tris_top] + lateral_tris,
    ).astype(np.int32)
    print(
        f"[polygon-extrude] {name}: outer={len(outer_xy)} vts, "
        f"holes={len(holes_xy)}, n_axial={n_axial} → "
        f"{int(all_verts.shape[0]):,} verts, "
        f"{int(all_tris.shape[0]):,} tris "
        f"(rdp_tol={rdp_tol_mm * 1e3:.1f} µm, "
        f"axial_step={thickness / n_axial * 1e3:.2f} µm)",
        flush=True,
    )
    return Mesh(
        verts=all_verts.astype(np.float32),
        faces=all_tris,
        name=name,
    )


def _split_components_2d(mask: np.ndarray) -> list[np.ndarray]:
    """Connected components of a 2D bool mask, 4-connectivity.

    M40 — flipped from 8- to 4-connectivity. The 8-conn version
    merged neighbouring fascicles that only touched diagonally
    into a single component; after broadcast + Gaussian smooth +
    marching cubes the diagonal pixel-bridge became a thin
    non-manifold pinch-neck surface, which then produced
    thousands of phantom self-intersections after cuff-window
    cap stitching and made TetGen bail with `recoversubfaces`.
    With 4-conn, diagonally-touching fascicles split into
    separate clean prism meshes — one per real fascicle blob.

    M41 — Diagnostic print of component count and sizes. If the
    user expects N fascicles but the labeller finds >N, the
    excess are almost always single-pixel noise speckles from
    SAM2 propagation or annotation jitter — visible immediately
    in the size distribution.

    Returns a list of per-CC bool arrays, one per component.
    """
    from skimage.measure import label
    cc = label(mask, connectivity=1)
    n_cc = int(cc.max())
    components = [cc == i for i in range(1, n_cc + 1)]
    if n_cc > 0:
        sizes = sorted(
            (int(c.sum()) for c in components),
            reverse=True,
        )
        # Show all sizes when ≤ 8 components; otherwise show top
        # 5 + a "+M smaller" tail so a noisy mask is obvious.
        if n_cc <= 8:
            sz_str = ", ".join(str(s) for s in sizes)
        else:
            sz_str = (
                ", ".join(str(s) for s in sizes[:5])
                + f", +{n_cc - 5} smaller "
                + f"(min={sizes[-1]} px)"
            )
        print(
            f"[split2d] {n_cc} component(s); sizes (px): {sz_str}",
            flush=True,
        )
    return components


def _split_components_3d(volume: np.ndarray) -> list[np.ndarray]:
    """26-connectivity 3D CC split. Used to give each fascicle
    that runs continuously through multiple slices its own
    .stl file (rather than one combined fascicles.stl with
    multiple shells inside)."""
    from skimage.measure import label
    cc = label(volume, connectivity=3)
    return [cc == i for i in range(1, int(cc.max()) + 1)]


def cleanup_2d_mask(
    mask: np.ndarray,
    *,
    min_component_px: int = 0,
    min_hole_px: int = 0,
    closing_radius_px: int = 0,
    # M42 — Auto size-gap filter. OFF by default (M43 revert):
    # real sheep VN fascicles routinely span 5-10× in area, so
    # the "smaller-than-30%-of-largest is noise" heuristic
    # mis-drops legitimate small fascicles. Kept as an opt-in
    # knob — a user who knows their data has only similarly-
    # sized fascicles + corner-pixel noise can set this to
    # e.g. 0.3 and get the original M42 behaviour. The right
    # default-on path is to clean noise in the Segment UI via
    # paint-out, where the user can see what they're erasing.
    size_gap_ratio: float = 0.0,
) -> np.ndarray:
    """Remove small connected components (speckle false
    positives), fill small holes (speckle false negatives
    inside the foreground), and optionally seal thin gaps via
    morphological closing in a 2D binary mask.

    Pre-meshing cleanup for segmentation outputs. Typical
    failure modes this fixes:
      * lone fascicle pixels scattered on the background
        layer (SAM2 confidence noise) → drop via
        `min_component_px`
      * lone background pixels speckled INSIDE a fascicle
        (border ambiguity / under-segmentation) → fill via
        `min_hole_px`
      * thin disconnects / 1-2 px gaps in the foreground
        boundary (border jitter) → seal via
        `closing_radius_px`

    All three knobs in PIXELS at the source image resolution.
    Set any to 0 to skip that pass; the function is a no-op
    when all three are 0.

    Order of operations is FIXED:
      1. Drop small foreground components.
      2. Fill small background holes.
      3. Morphological close (disk-shaped structuring element).

    Rationale: closing's dilation step would otherwise pull
    small isolated speckles into the main body before the
    component drop can catch them. Closing LAST also lets the
    user pick a generous radius without worrying about it
    swallowing legitimate small structures that the component
    drop already kept (those are filtered before closing
    runs).

    Implementation: scipy.ndimage.label with default
    8-connectivity, np.bincount + boolean-lookup pass per
    direction, scipy.ndimage.binary_closing with a disk
    kernel. O(N) per slice for the labelling; closing is O(N
    × kernel_size) — generous radii on large slices can show.
    """
    if mask.size == 0:
        return mask
    _gap_active = 0.0 < float(size_gap_ratio) < 1.0
    if (
        int(min_component_px) <= 0
        and int(min_hole_px) <= 0
        and int(closing_radius_px) <= 0
        and not _gap_active
    ):
        return mask
    from scipy.ndimage import (
        binary_closing as _binary_closing,
        label as _label,
    )
    out = np.asarray(mask, dtype=bool).copy()
    if int(min_component_px) > 0:
        labels, n_lab = _label(out)
        if n_lab > 0:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= int(min_component_px)
            # Label 0 is always "background" of the labelled
            # array, i.e. NOT a real component. Force off so
            # the lookup never marks background as kept.
            keep[0] = False
            out = keep[labels]
    # M42 — Auto size-gap filter. Runs AFTER min_component_px
    # so it operates on the surviving "plausibly-real"
    # components only. Sort component sizes descending; walk
    # the list looking for the largest gap where size[i+1] /
    # size[i] < size_gap_ratio. Anything from that index
    # onward is treated as noise and dropped. Never drops the
    # single largest component. No-op when only 0-1
    # components survive or no gap below the threshold exists.
    if _gap_active:
        labels, n_lab = _label(out)
        if n_lab >= 2:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0  # background
            comp_ids = np.arange(1, n_lab + 1)
            comp_sizes = sizes[comp_ids]
            order = np.argsort(comp_sizes)[::-1]
            sorted_ids = comp_ids[order]
            sorted_sizes = comp_sizes[order]
            # Find first gap where next/prev < threshold.
            drop_from = None
            for i in range(len(sorted_sizes) - 1):
                if sorted_sizes[i] <= 0:
                    break
                ratio = (
                    float(sorted_sizes[i + 1])
                    / float(sorted_sizes[i])
                )
                if ratio < float(size_gap_ratio):
                    drop_from = i + 1
                    break
            if drop_from is not None:
                drop_ids = sorted_ids[drop_from:]
                drop_sizes = sorted_sizes[drop_from:]
                keep_mask = np.ones(n_lab + 1, dtype=bool)
                keep_mask[0] = False
                for di in drop_ids:
                    keep_mask[di] = False
                out = keep_mask[labels]
                print(
                    f"[cleanup2d] size-gap drop: kept "
                    f"{int(drop_from)} comp(s) "
                    f"(largest={int(sorted_sizes[0])} px), "
                    f"dropped {len(drop_ids)} below the gap "
                    f"(sizes: "
                    f"{', '.join(str(int(s)) for s in drop_sizes)} "
                    f"px; ratio threshold {size_gap_ratio:.2f})",
                    flush=True,
                )
    if int(min_hole_px) > 0:
        # `~out` swaps foreground / background. Labels in this
        # inverted view are the BACKGROUND components of the
        # original mask, which is what "holes" are. Drop the
        # small ones (fill them in) while leaving the big
        # outer-background component alone.
        labels, n_lab = _label(~out)
        if n_lab > 0:
            sizes = np.bincount(labels.ravel())
            fill = sizes < int(min_hole_px)
            # Label 0 in the inverted labelling corresponds to
            # the ORIGINAL foreground of `out`. Force off so
            # we don't mark already-true pixels for re-fill
            # (it's a no-op but cleaner intent).
            fill[0] = False
            out = out | fill[labels]
    if int(closing_radius_px) > 0:
        # (2r+1)×(2r+1) all-ones (8-connectivity) structuring
        # element. Choosing this over a euclidean disk kernel
        # because at small r the disk `(x² + y²) ≤ r²` is a
        # sparse diamond whose corners get pruned — it can't
        # actually bridge a gap as wide as 2r in practice. The
        # square kernel bridges gaps up to 2r pixels in EVERY
        # direction (L∞ isotropic). The minor boxiness at the
        # blob corners is invisible at typical r ≤ 3, and the
        # marching-cubes / Taubin smooth chain downstream
        # rounds it out anyway.
        r = int(closing_radius_px)
        struct = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        out = _binary_closing(out, structure=struct)
    return out


def cleanup_3d_mask(
    volume: np.ndarray,
    *,
    min_component_vox: int = 0,
    min_hole_vox: int = 0,
    keep_largest_only: bool = False,
) -> np.ndarray:
    """3D analog of `cleanup_2d_mask`. Drops small
    connected-volume speckles and fills small voids inside the
    foreground.

    Per-slice cleanup (`cleanup_2d_mask` applied to each frame)
    handles speckle that lives inside one slice, but cannot
    address Z-direction noise — e.g. a "streak" speckle that
    appears in slices 5-7 and nowhere else, or a column of
    background voxels punched through several slices in the
    middle of a fascicle. Running this after the 3D volume is
    assembled catches those.

    Uses 26-connectivity (face + edge + corner neighbours) for
    component labelling, matching `_split_components_3d`.
    Knobs in VOXELS — voxel count scales differently from 2D
    pixel counts because each 2D speckle now spans potentially
    several Z slices, so threshold scales accordingly (e.g. a
    100 px slice-level speckle that survives 3 slices = 300
    voxels).

    No-op when both knobs are 0."""
    if volume.size == 0:
        return volume
    if (
        int(min_component_vox) <= 0
        and int(min_hole_vox) <= 0
        and not keep_largest_only
    ):
        return volume
    from scipy.ndimage import (
        generate_binary_structure as _gen_struct,
        label as _label,
    )
    struct = _gen_struct(3, 3)   # 26-connectivity
    out = np.asarray(volume, dtype=bool).copy()
    if int(min_component_vox) > 0:
        labels, n_lab = _label(out, structure=struct)
        if n_lab > 0:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= int(min_component_vox)
            keep[0] = False
            out = keep[labels]
    if keep_largest_only:
        # M26 — for the epi specifically: there's anatomically
        # one continuous epineurium per nerve, so any
        # disconnected blob in the segmentation is by definition
        # a phantom (mislabel, distant artefact). Drop everything
        # except the single largest connected component, which
        # catches phantoms regardless of their size — even ones
        # large enough to survive the min_component_vox threshold.
        labels, n_lab = _label(out, structure=struct)
        if n_lab > 1:
            sizes = np.bincount(labels.ravel())
            # Label 0 is background — force its size to 0 so the
            # argmax picks a real foreground component, not bg.
            sizes[0] = 0
            largest = int(np.argmax(sizes))
            out = (labels == largest)
    if int(min_hole_vox) > 0:
        labels, n_lab = _label(~out, structure=struct)
        if n_lab > 0:
            sizes = np.bincount(labels.ravel())
            fill = sizes < int(min_hole_vox)
            fill[0] = False
            out = out | fill[labels]
    return out


# --------------------------------------------------------------
# Mesh refinement (Taubin smooth + pymeshfix repair pipeline)
# --------------------------------------------------------------


def _surface_genus(pts: np.ndarray, tris: np.ndarray) -> int:
    """Topological genus of a 2-manifold triangle surface.

    For a closed triangle mesh:
        genus = (2 − V + E − F) / 2
    where V = vertex count, F = triangle count, and each tri
    has 3 edges with each edge shared by 2 triangles in a
    proper manifold, so E = 3·F/2.

    Returns 0 for sphere-like / simply-connected meshes,
    1 for a torus, N for a sphere with N handles (e.g. the
    µCT epi shell with N fascicle-shaped tunnels).

    Wrapped in try/except — non-manifold meshes can give
    fractional or negative counts. We round to int and
    clip at 0 to be safe.
    """
    try:
        V = int(pts.shape[0])
        F = int(tris.shape[0])
        # E = 3F/2 assumes a clean closed 2-manifold; if it's
        # not, the result is approximate but still useful as
        # a "this is high-genus" indicator.
        E = (3 * F) // 2
        g = (2 - V + E - F) // 2
        return max(0, int(g))
    except Exception:                                    # noqa: BLE001
        return 0


def _maybe_optimesh(
    pts: np.ndarray,
    tris: np.ndarray,
    *,
    max_steps: int = 20,
    tol: float = 1.0e-3,
    genus_guard: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an optimesh CVT (Lloyd) pass on the mesh when the
    `optimesh` package is available. No-op + identity return
    when it's not installed.

    **Genus-guard**: optimesh's edge-flip pass can SEGFAULT
    (not throwable, full process death) on shell meshes with
    multiple handles — the µCT epi shell with N fascicle
    tunnels is exactly this case. We pre-compute the topological
    genus and skip optimesh when it's > 0, since a segfault in
    a C extension can't be caught from Python.

    Warning suppression: optimesh's Delaunay-flip pass emits
    a `UserWarning` line for every facet at the float-epsilon
    noise floor. On a 20-fascicle bundle that's hundreds of
    nearly-identical warnings per cleanup pass. We swallow them
    via `warnings.catch_warnings`.

    Optimesh's public API has shifted across versions (0.x →
    0.10 → 0.11); both signatures are tried in turn.
    """
    if genus_guard:
        g = _surface_genus(pts, tris)
        if g > 0:
            # Shell mesh — optimesh's edge-flip step is known
            # to crash on these. Bail before we go anywhere
            # near the C extension.
            return pts, tris
    try:
        import optimesh as _om
    except ImportError:
        return pts, tris
    import warnings as _w
    pts64 = np.ascontiguousarray(pts, dtype=np.float64)
    tris32 = np.ascontiguousarray(tris, dtype=np.int32)
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            # Modern (≥ 0.10) API.
            new_pts, new_tris = _om.optimize_points_cells(
                pts64, tris32, "CVT (full)",
                tol=float(tol),
                max_num_steps=int(max_steps),
            )
    except (AttributeError, TypeError):
        # Pre-0.10 API.
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                new_pts, new_tris = _om.cvt.full(
                    pts64, tris32,
                    tol=float(tol),
                    max_num_steps=int(max_steps),
                )
        except Exception:                                # noqa: BLE001
            return pts, tris
    except Exception:                                    # noqa: BLE001
        return pts, tris
    return new_pts, new_tris


def _drop_specks(
    pts: np.ndarray,
    tris: np.ndarray,
    *,
    min_triangle_count: int = 50,
    min_surface_area_mm2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop tiny disconnected sub-meshes (specks from MC noise:
    single-triangle islands, 5-tri fragments at the edge of
    the volume, etc.). Both genus and the largest legitimate
    component(s) are preserved — only sub-meshes below the
    `min_triangle_count` and `min_surface_area_mm2` thresholds
    get culled.

    Unlike `pymeshfix.remove_smallest_components` (which keeps
    only the LARGEST component), this lets every "real"
    structure survive even if there are several. For µCT
    extrusions there should usually be just one component per
    `refine_mesh` call, but the speck-drop is cheap insurance
    against MC producing a few isolated boundary triangles at
    the volume edge.

    Returns the cleaned (pts, tris). No-op when trimesh isn't
    installed.
    """
    try:
        import trimesh as _trimesh
    except ImportError:
        return pts, tris
    if tris.shape[0] == 0:
        return pts, tris
    try:
        tm = _trimesh.Trimesh(
            vertices=pts, faces=tris, process=False,
        )
        # trimesh.split returns a list of separate sub-meshes,
        # one per connected component.
        comps = tm.split(only_watertight=False)
        if len(comps) <= 1:
            return pts, tris
        keep = []
        for c in comps:
            if c.faces.shape[0] < int(min_triangle_count):
                continue
            if (
                min_surface_area_mm2 > 0
                and float(c.area) < float(min_surface_area_mm2)
            ):
                continue
            keep.append(c)
        if not keep:
            # Pathological — would mean every component is
            # below threshold. Fall back to the original mesh
            # rather than returning nothing.
            return pts, tris
        merged = _trimesh.util.concatenate(keep)
        return (
            np.asarray(merged.vertices, dtype=np.float64),
            np.asarray(merged.faces, dtype=np.int64),
        )
    except Exception:                                    # noqa: BLE001
        return pts, tris


def _isotropic_remesh(
    pts: np.ndarray,
    tris: np.ndarray,
    *,
    target_edge_len_mm: float,
    subdivide: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """ACVD-based isotropic remesh.

    Tiles the surface with near-uniform triangle edge length
    via the ACVD (Approximated Centroidal Voronoi Diagrams)
    algorithm. Target vertex count is derived from the
    surface area + the requested edge length:

        N = area / edge_len²

    so a small fascicle naturally gets fewer vertices than the
    epi shell — the "single budget for all" problem of plain
    decimation goes away.

    `subdivide` upsamples the input before clustering so the
    output edge length can be smaller than the input mesh's.
    Two passes is enough for most µCT cases; higher values
    blow up the temporary point cloud.

    No-op (returns input) when pyacvd isn't installed.
    """
    try:
        import pyacvd
        import pyvista as pv
    except ImportError:
        return pts, tris
    if tris.shape[0] == 0:
        return pts, tris
    edge_len = float(target_edge_len_mm)
    if edge_len <= 0:
        return pts, tris
    try:
        n_t = int(tris.shape[0])
        flat = np.empty(n_t * 4, dtype=np.int64)
        flat[0::4] = 3
        flat[1::4] = tris[:, 0]
        flat[2::4] = tris[:, 1]
        flat[3::4] = tris[:, 2]
        in_mesh = pv.PolyData(pts, flat)
        # Surface-area-derived target vertex count. Cap at
        # 250 k so a giant epi shell with an unrealistically
        # tiny edge length doesn't OOM the worker.
        area_mm2 = float(in_mesh.area)
        target_n = int(
            max(64, min(250_000, area_mm2 / (edge_len ** 2))),
        )
        clus = pyacvd.Clustering(in_mesh)
        if subdivide > 0:
            clus.subdivide(int(subdivide))
        clus.cluster(target_n)
        out_mesh = clus.create_mesh()
        out_pts = np.asarray(out_mesh.points, dtype=np.float64)
        out_tris = (
            np.asarray(out_mesh.faces, dtype=np.int64)
            .reshape(-1, 4)[:, 1:]
        )
        return out_pts, out_tris
    except Exception:                                    # noqa: BLE001
        return pts, tris


def refine_mesh(
    mesh: Mesh,
    *,
    n_passes: int = 2,
    drop_speck_tris: int = 50,
    use_optimesh: bool = False,
    remesh: bool = False,
    remesh_edge_len_mm: Optional[float] = None,
    target_tris: Optional[int] = None,
    decimate_target_tris: Optional[int] = None,
    # M27 — proportional decimate: keep `decimate_target_fraction`
    # of the input triangle count. Takes precedence over
    # `decimate_target_tris` when both are set. Use this instead
    # of an absolute target when surfaces vary widely in size
    # (e.g. 488 k-tri epi + 20 k-tri fascicle) — a uniform 20 k
    # target crushes the epi but barely touches the fascicle,
    # producing a 100× edge-length disparity. A fraction
    # (e.g. 0.5) applies consistently to both.
    decimate_target_fraction: Optional[float] = None,
    on_progress: "Optional[Callable[[str, float], None]]" = None,
) -> Mesh:
    """µCT-aware mesh-quality pipeline. NO decimation.

    Why no decimation? Marching cubes is already producing
    surfaces at the exact resolution the segmentation defines,
    at the chosen voxel size — there's no "extra detail" to
    throw away. Capping every sub-mesh at the same triangle
    budget (`target_tris`) means the epi shell loses its inner
    walls while the small fascicles get needless artefacts
    added. The histogram-quality lift came from the per-axis
    isotropic Gaussian smoothing + the optimesh CVT pass, not
    from decimation. So we drop it.

    Pipeline:
      1. **Drop specks** — connected-component split via
         trimesh; toss sub-meshes with < `drop_speck_tris`
         triangles. Catches the lone boundary triangles MC
         can emit at the volume edge.
      2. **Taubin smooth** (per pass) — gentle low-pass on
         vertex positions with boundary/non-manifold
         smoothing OFF (preserves inner hole rims).
      3. **pymeshfix repair** (per pass) — fixes self-
         intersections + degenerate triangles with
         `joincomp=False, remove_smallest_components=False`
         so the genus survives.
      4. **Trimesh defensive pass** — drops duplicate /
         zero-area / inverted faces.
      5. **Isotropic remesh** (optional, off by default) —
         pyacvd with target vertex count derived from the
         per-mesh surface area, so small fascicles get few
         vertices and the epi shell gets many, both at
         uniform local resolution. Requires the optional
         `pyacvd` package.
      6. **Optimesh CVT relaxation** — OPT-IN
         (`use_optimesh=True`). The heavy-lifter for triangle-
         shape quality, but optimesh's edge-flip pass has been
         observed to segfault on complex shell meshes (the
         µCT epi exactly fits the failure profile). Auto-
         skipped when the input has non-zero genus via a
         topology check in `_maybe_optimesh`.

    `target_tris` is accepted for backward compatibility with
    earlier call sites but **ignored** — decimation is gone.
    Pass `remesh=True, remesh_edge_len_mm=0.05` to engage the
    new isotropic remesher instead.
    """
    if target_tris is not None:
        # Silent — old callers might still pass this. The new
        # remesh kwargs are the right way to control vertex
        # count; decimation is out.
        pass
    try:
        import pyvista as pv
    except ImportError as ex:
        raise RuntimeError(
            "refine_mesh requires pyvista at minimum. "
            f"Import failed: {ex}",
        ) from ex
    try:
        import pymeshfix
        _HAVE_PYMESHFIX = True
    except ImportError:
        pymeshfix = None
        _HAVE_PYMESHFIX = False

    pts = np.asarray(mesh.verts, dtype=np.float64)
    tris = np.asarray(mesh.faces, dtype=np.int64)
    if tris.shape[0] == 0:
        return mesh

    def _to_pv(p: np.ndarray, t: np.ndarray) -> "pv.PolyData":
        n_t = int(t.shape[0])
        flat = np.empty(n_t * 4, dtype=np.int64)
        flat[0::4] = 3
        flat[1::4] = t[:, 0]
        flat[2::4] = t[:, 1]
        flat[3::4] = t[:, 2]
        return pv.PolyData(p, flat)

    def _from_pv(
        m: "pv.PolyData",
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(m.points, dtype=np.float64)
        t = (
            np.asarray(m.faces, dtype=np.int64)
            .reshape(-1, 4)[:, 1:]
        )
        return p, t

    # ---- 1. Drop specks (run once up front) ----
    _t = time.perf_counter()
    cur_pts, cur_tris = _drop_specks(
        pts, tris, min_triangle_count=int(drop_speck_tris),
    )
    _emit_progress(
        on_progress,
        f"  {mesh.name}: drop_specks "
        f"(in={tris.shape[0]:,} out={cur_tris.shape[0]:,})",
        time.perf_counter() - _t,
        prefix="[refine]",
    )

    for _p in range(int(n_passes)):
        # ---- 2. Taubin smooth ----
        _t = time.perf_counter()
        pv_mesh = _to_pv(cur_pts, cur_tris).smooth_taubin(
            n_iter=20, pass_band=0.1,
            edge_angle=180.0, feature_angle=180.0,
            boundary_smoothing=False,
            non_manifold_smoothing=False,
        )
        cur_pts, cur_tris = _from_pv(pv_mesh)
        _emit_progress(
            on_progress,
            f"  {mesh.name}: Taubin pass {_p + 1}/{int(n_passes)} "
            f"({cur_tris.shape[0]:,} tris)",
            time.perf_counter() - _t,
            prefix="[refine]",
        )

        # ---- 3. pymeshfix repair (genus-preserving) ----
        if _HAVE_PYMESHFIX:
            _t = time.perf_counter()
            mf = pymeshfix.MeshFix(
                np.ascontiguousarray(cur_pts, dtype=np.float64),
                np.ascontiguousarray(cur_tris, dtype=np.int32),
            )
            try:
                mf.repair(
                    verbose=False,
                    joincomp=False,
                    remove_smallest_components=False,
                )
                cur_pts = np.asarray(
                    mf.mesh.points, dtype=np.float64,
                )
                cur_tris = (
                    np.asarray(mf.mesh.faces, dtype=np.int64)
                    .reshape(-1, 4)[:, 1:]
                )
            except Exception:                            # noqa: BLE001
                pass
            _emit_progress(
                on_progress,
                f"  {mesh.name}: pymeshfix pass "
                f"{_p + 1}/{int(n_passes)} "
                f"({cur_tris.shape[0]:,} tris)",
                time.perf_counter() - _t,
                prefix="[refine]",
            )

    # ---- 4. Trimesh defensive pass ----
    _t = time.perf_counter()
    try:
        import trimesh as _trimesh
        tm = _trimesh.Trimesh(
            vertices=cur_pts, faces=cur_tris, process=False,
        )
        tm.merge_vertices()
        try:
            uf = tm.unique_faces()
            tm.update_faces(uf)
        except Exception:                                # noqa: BLE001
            pass
        try:
            nd = tm.nondegenerate_faces()
            tm.update_faces(nd)
        except Exception:                                # noqa: BLE001
            pass
        tm.remove_unreferenced_vertices()
        try:
            _trimesh.repair.fix_inversion(tm)
            _trimesh.repair.fix_normals(tm)
        except Exception:                                # noqa: BLE001
            pass
        cur_pts = np.asarray(tm.vertices, dtype=np.float64)
        cur_tris = np.asarray(tm.faces, dtype=np.int64)
    except ImportError:
        pass
    _emit_progress(
        on_progress,
        f"  {mesh.name}: trimesh defensive pass "
        f"({cur_tris.shape[0]:,} tris)",
        time.perf_counter() - _t,
        prefix="[refine]",
    )

    # ---- 5. Optional isotropic remesh ----
    if remesh and remesh_edge_len_mm is not None:
        _t = time.perf_counter()
        cur_pts, cur_tris = _isotropic_remesh(
            cur_pts, cur_tris,
            target_edge_len_mm=float(remesh_edge_len_mm),
        )
        _emit_progress(
            on_progress,
            f"  {mesh.name}: isotropic remesh "
            f"({cur_tris.shape[0]:,} tris)",
            time.perf_counter() - _t,
            prefix="[refine]",
        )

    # ---- 6. optimesh CVT relaxation (OPT-IN) ----
    # Restored to its pre-M28 position (BEFORE decimation).
    # M28 tried to move this after decimation reasoning that
    # decimation would otherwise undo the polish, but on
    # fascicle-shaped meshes pyvista.decimate introduces near-
    # degenerate edges that optimesh's Delaunay flips then
    # tear into non-manifold topology (user saw 6 of 6
    # fascicles drop to manifold=False, vol=0). Keep optimesh
    # safely upstream of decimation — yes the final mesh loses
    # some of the polish if heavily decimated, but at least
    # the topology survives.
    if use_optimesh:
        _t = time.perf_counter()
        cur_pts, cur_tris = _maybe_optimesh(
            cur_pts, cur_tris,
        )
        _emit_progress(
            on_progress,
            f"  {mesh.name}: optimesh CVT "
            f"({cur_tris.shape[0]:,} tris)",
            time.perf_counter() - _t,
            prefix="[refine]",
        )
        # M34 — post-optimesh repair sweep. Optimesh's edge-
        # flip pass has been observed to introduce non-manifold
        # edges on big shell-like meshes (the user's 2.27 M-tri
        # epi came out with 992 non-manifold edges + ~5k
        # slivers even after the upstream pymeshfix passes).
        # One more MeshFix.repair after optimesh restores
        # manifoldness at a fraction of the optimesh cost.
        # joincomp=False, remove_smallest_components=False
        # preserve the topology / multi-component layout.
        if _HAVE_PYMESHFIX:
            _t = time.perf_counter()
            try:
                mf = pymeshfix.MeshFix(
                    np.ascontiguousarray(
                        cur_pts, dtype=np.float64,
                    ),
                    np.ascontiguousarray(
                        cur_tris, dtype=np.int32,
                    ),
                )
                mf.repair(
                    joincomp=False,
                    remove_smallest_components=False,
                )
                cur_pts = np.asarray(
                    mf.mesh.points, dtype=np.float64,
                )
                cur_tris = (
                    np.asarray(mf.mesh.faces, dtype=np.int64)
                    .reshape(-1, 4)[:, 1:]
                )
                _emit_progress(
                    on_progress,
                    f"  {mesh.name}: post-optimesh pymeshfix "
                    f"({cur_tris.shape[0]:,} tris)",
                    time.perf_counter() - _t,
                    prefix="[refine]",
                )
            except Exception as ex:                       # noqa: BLE001
                _emit_progress(
                    on_progress,
                    f"  {mesh.name}: post-optimesh pymeshfix "
                    f"skipped ({type(ex).__name__})",
                    time.perf_counter() - _t,
                    prefix="[refine]",
                )

    # ---- 7. Optional decimation (OPT-IN, user-driven) ----
    _reduction: Optional[float] = None
    if (
        decimate_target_fraction is not None
        and 0.0 < float(decimate_target_fraction) < 1.0
    ):
        _reduction = max(
            0.0, 1.0 - float(decimate_target_fraction),
        )
    elif (
        decimate_target_tris is not None
        and int(decimate_target_tris) > 0
        and int(cur_tris.shape[0]) > int(decimate_target_tris)
    ):
        _reduction = max(
            0.0,
            1.0 - float(int(decimate_target_tris))
            / float(cur_tris.shape[0]),
        )
    if _reduction is not None and _reduction > 0.0:
        _t = time.perf_counter()
        try:
            n_before = int(cur_tris.shape[0])
            surf = _to_pv(cur_pts, cur_tris)
            dec = surf.decimate(
                float(_reduction),
                volume_preservation=True,
            )
            cur_pts, cur_tris = _from_pv(dec)
            _emit_progress(
                on_progress,
                f"  {mesh.name}: decimate "
                f"({n_before:,} → {cur_tris.shape[0]:,} tris)",
                time.perf_counter() - _t,
                prefix="[refine]",
            )
        except Exception:                                    # noqa: BLE001
            pass

    return Mesh(
        verts=np.asarray(cur_pts, dtype=np.float32),
        faces=np.asarray(cur_tris, dtype=np.int32),
        name=mesh.name,
    )


# --------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------


def extrude_single_slice(
    epi_mask: np.ndarray,
    fasc_mask: np.ndarray,
    *,
    voxel_xy_mm: float,
    thickness_mm: float,
    smooth_sigma: Optional[float] = None,
    smooth_sigma_xy_mm: Optional[float] = None,
    smooth_sigma_z_mm: Optional[float] = None,
    refine: bool = False,
    refine_passes: int = 2,
    refine_use_optimesh: bool = False,
    refine_remesh: bool = False,
    refine_remesh_edge_len_mm: Optional[float] = None,
    # Legacy kwarg — accepted for backwards compatibility,
    # ignored by the new `refine_mesh` (decimation is out).
    refine_target_tris: Optional[int] = None,
    # Opt-in user-driven decimation. Applied AFTER refinement
    # to each output mesh independently when set + the mesh has
    # more triangles than the target. None / 0 = no decimation
    # (legacy default).
    decimate_target_tris: Optional[int] = None,
    # M27 — proportional decimate (0 < x < 1, keep x of input
    # tris). Takes precedence over decimate_target_tris.
    decimate_target_fraction: Optional[float] = None,
    # Opt-in 2D mask cleanup BEFORE marching cubes. Drops small
    # connected components (speckle false positives), fills
    # small holes (speckle false negatives), and seals thin
    # gaps via morphological closing in each input mask at the
    # source image resolution. 0 = pass that direction through
    # unchanged.
    clean_min_component_px: int = 0,
    clean_min_hole_px: int = 0,
    clean_closing_radius_px: int = 0,
    # M24 — Inward-offset each fascicle by N pixels before
    # extrusion so its surface lands cleanly inside the epi
    # after marching cubes. See reconstruct_stack for the
    # rationale. 0 disables.
    fasc_inset_vox: int = 2,
) -> list[Mesh]:
    """Extrude one annotated slice into prismatic meshes.

    Approach (matches what the user asked for):
      1. Treat each 2D mask as a stack of identical Z slabs in
         an isotropic-voxel volume: Z spacing == voxel_xy_mm,
         so a 5 mm-thick prism at 10 µm pixels would ideally
         be 500 slabs. To keep the MC pass tractable for
         large nerves, the slab count is capped at 64 (still
         enough for clean cap rendering — Z spacing then
         tops out at ~8× voxel_xy, which the per-axis sigma
         scaling in `_mc_to_mesh` handles cleanly).
      2. Pad with 1 False slab on every side (closed surface).
      3. Marching cubes at level 0.5 with the per-axis
         spacing.
      4. Optional Gaussian smoothing (per-axis sigma so the
         physical smoothing radius matches across X / Y / Z,
         which avoids the Z-streak / spike artefacts the old
         3-slab MC produced).
      5. Optional refinement (Taubin → pymeshfix → optimesh
         CVT when available — see `refine_mesh`).

    Output ordering: [epi (if non-empty), fascicle_0, …] with
    fascicles split per 2D connected component.
    """
    if epi_mask.shape != fasc_mask.shape:
        raise ValueError(
            "epi and fasc masks must have the same shape "
            f"(got {epi_mask.shape} vs {fasc_mask.shape})",
        )
    # 2D speckle cleanup BEFORE marching cubes. All passes are
    # no-ops when the corresponding knob is 0 (legacy default),
    # so existing callers are unaffected.
    if (
        int(clean_min_component_px) > 0
        or int(clean_min_hole_px) > 0
        or int(clean_closing_radius_px) > 0
    ):
        epi_mask = cleanup_2d_mask(
            epi_mask,
            min_component_px=int(clean_min_component_px),
            min_hole_px=int(clean_min_hole_px),
            closing_radius_px=int(clean_closing_radius_px),
        )
        fasc_mask = cleanup_2d_mask(
            fasc_mask,
            min_component_px=int(clean_min_component_px),
            min_hole_px=int(clean_min_hole_px),
            closing_radius_px=int(clean_closing_radius_px),
        )
    if not epi_mask.any() and not fasc_mask.any():
        return []
    # M24 — Inward-offset fascicles inside the epi by N pixels
    # so the fascicle surface lands cleanly inside the epi
    # after marching cubes. Same logic as reconstruct_stack
    # but in 2D since extrude_single_slice replicates one 2D
    # mask along Z. Skipped when epi was annotated as a shell
    # (negligible fasc/epi overlap) — eroding+intersecting
    # would null the fascicles in that case.
    if (
        int(fasc_inset_vox) > 0
        and epi_mask.any()
        and fasc_mask.any()
    ):
        _overlap = int(
            np.logical_and(epi_mask, fasc_mask).sum(),
        )
        _fasc_total = int(fasc_mask.sum())
        if _overlap >= 0.5 * _fasc_total:
            from scipy.ndimage import binary_erosion as _erode
            _n_before = int(fasc_mask.sum())
            _eroded_epi = _erode(
                epi_mask, iterations=int(fasc_inset_vox),
            )
            fasc_mask = fasc_mask & _eroded_epi
            print(
                f"[extrude] fasc inset {fasc_inset_vox} px "
                f"inside epi (XY): {_n_before:,} → "
                f"{int(fasc_mask.sum()):,} fasc pixels  "
                f"(+ Z inset of {fasc_inset_vox} slab(s) "
                "applied at vol-build time)",
                flush=True,
            )
        else:
            print(
                f"[extrude] fasc inset skipped: only "
                f"{_overlap:,} / {_fasc_total:,} fasc pixels "
                "overlap with epi (epi seems to be a shell)",
                flush=True,
            )
    voxel_xy_mm = float(voxel_xy_mm)
    thickness_mm = float(thickness_mm)
    if voxel_xy_mm <= 0 or thickness_mm <= 0:
        raise ValueError(
            "voxel_xy_mm and thickness_mm must be > 0 "
            f"(got {voxel_xy_mm}, {thickness_mm})",
        )
    # M45 — Polygon-fit + extrude path. Replaces the prior MC
    # volume-based path entirely for single-slice extrusion.
    # Trace the 2D mask boundary as polylines, simplify with
    # Douglas-Peucker, and build the prism by hand: top cap,
    # bottom cap, lateral quad strip per boundary ring.
    # Watertight by construction, ~1000× fewer triangles than
    # MC, near-perfect triangle quality (no staircase slivers).
    #
    # `smooth_sigma_*` and `smooth_sigma` are accepted but no
    # longer used here — the staircase is removed by RDP
    # simplification + a Gaussian pre-smooth of the 2D mask,
    # not by Gaussian volume smoothing. The multi-slice
    # `reconstruct_stack` path keeps MC + smoothing because
    # there each Z slice can differ.
    epi_z_lo = -0.5 * thickness_mm
    epi_z_hi = +0.5 * thickness_mm
    # M36 carry-over: fascicles inset axially by `fasc_inset_vox`
    # XY voxels worth of mm so their cap planes sit STRICTLY
    # inside the epi cap planes (no co-planar caps → no
    # float-drift TetGen mis-detections at the cap planes).
    fasc_z_inset_mm = float(fasc_inset_vox) * voxel_xy_mm
    fasc_z_lo = epi_z_lo + fasc_z_inset_mm
    fasc_z_hi = epi_z_hi - fasc_z_inset_mm
    # RDP tolerance — physical-mm distance under which polyline
    # vertices can be culled. 2 voxels combined with M45c's
    # heavier sigma=2.5 vox pre-smooth: the contour is smooth
    # enough that 2-voxel RDP picks well-distributed verts on
    # a curve, and the larger tolerance guarantees adjacent
    # boundary verts are spaced enough that lateral aspect
    # ratios stay reasonable + cap triangulation doesn't
    # produce near-degenerate slivers at sharp corners.
    rdp_tol_mm = 2.0 * voxel_xy_mm
    meshes: list[Mesh] = []
    if epi_mask.any():
        m = _polygon_extrude_component(
            epi_mask,
            voxel_xy_mm=voxel_xy_mm,
            z_lo=epi_z_lo, z_hi=epi_z_hi,
            name="epi",
            rdp_tol_mm=rdp_tol_mm,
        )
        if m is not None:
            meshes.append(m)
    if fasc_mask.any():
        for i, fcc in enumerate(_split_components_2d(fasc_mask)):
            m = _polygon_extrude_component(
                fcc,
                voxel_xy_mm=voxel_xy_mm,
                z_lo=fasc_z_lo, z_hi=fasc_z_hi,
                name=f"fascicle_{i}",
                rdp_tol_mm=rdp_tol_mm,
            )
            if m is not None:
                meshes.append(m)
    # M45 — Refinement / decimation deliberately SKIPPED on
    # polygon-extruded meshes. The output of
    # `_polygon_extrude_component` is already watertight,
    # manifold, has minimal vertex count for the chosen
    # `rdp_tol_mm`, and contains zero degenerate triangles by
    # construction. Running the legacy refine_mesh pipeline
    # would only degrade it: Taubin smooths flat caps into
    # pillows; pyvista.decimate on an already-minimal mesh
    # collapses edges into near-zero-area slivers (`worst_rr`
    # observed to spike from 22 to 27000 with decimate=0.7 in
    # the user's M45a runs). Control mesh density via
    # `rdp_tol_mm` (cross-section) and `target_axial_mm`
    # (lateral) inside `_polygon_extrude_component` instead.
    del refine, decimate_target_tris, decimate_target_fraction
    # Quality report removed from this entry point — the
    # action layer (do_run_reconstruction) calls
    # report_mesh_quality_batch once after meshes are built, so
    # running it here duplicated the (expensive) pyvista +
    # trimesh checks on the same meshes.
    return meshes


def reconstruct_stack(
    epi_per_slice: dict[int, np.ndarray],
    fasc_per_slice: dict[int, np.ndarray],
    *,
    voxel_xy_mm: float,
    voxel_z_mm: float,
    slice_range: tuple[int, int],
    smooth_sigma: Optional[float] = 1.0,
    smooth_sigma_xy_mm: Optional[float] = None,
    smooth_sigma_z_mm: Optional[float] = None,
    refine: bool = False,
    refine_passes: int = 2,
    refine_use_optimesh: bool = False,
    refine_remesh: bool = False,
    refine_remesh_edge_len_mm: Optional[float] = None,
    refine_target_tris: Optional[int] = None,  # legacy no-op
    # Opt-in user-driven decimation: when > 0, each output mesh
    # is decimated to approximately this many triangles. None /
    # 0 = no decimation.
    decimate_target_tris: Optional[int] = None,
    # M27 — proportional decimate. Same semantics as in
    # refine_mesh: 0 < x < 1, keep `x` of input tris per surface.
    # Takes precedence over decimate_target_tris.
    decimate_target_fraction: Optional[float] = None,
    # Opt-in 2D mask cleanup BEFORE building the 3D volume.
    # Applied per-slice to BOTH the user-annotated frames AND
    # the ZOH-filled intermediates so the volume going into
    # marching cubes has no per-slice speckle artefacts.
    # 0 = pass that direction through unchanged.
    clean_min_component_px: int = 0,
    clean_min_hole_px: int = 0,
    clean_closing_radius_px: int = 0,
    # Opt-in 3D volume cleanup AFTER assembly + per-slice 2D
    # cleanup, BEFORE marching cubes. Catches Z-direction
    # speckles that 2D per-slice can't see (e.g. a streak
    # speckle that lives in slices 5-7 and nowhere else, or a
    # background column punched through several slices in the
    # middle of a fascicle). 0 = pass that direction through
    # unchanged.
    clean_3d_min_component_vox: int = 0,
    clean_3d_min_hole_vox: int = 0,
    min_component_voxels: int = 500,
    # M24 — Inward-offset each fascicle so its surface lands
    # cleanly INSIDE the epi after marching cubes + Gaussian
    # smoothing. Eroding `epi_vol` by N voxels then AND-ing
    # `fasc_vol` against the eroded shape guarantees the
    # fascicle isosurface stays N voxels inside the epi
    # isosurface — enough to absorb the sub-voxel displacement
    # the Gaussian smooth introduces at each boundary. 0 to
    # disable (legacy behaviour: fascicle and epi surfaces can
    # straddle each other, which makes the assembled PLC a
    # mess for TetGen since it can't classify a fascicle whose
    # boundary crosses the parent epi).
    fasc_inset_vox: int = 2,
    # Per-stage progress callback. Receives (stage_name,
    # elapsed_seconds) at every major checkpoint so the action
    # layer can push real-time status into state.busy_log
    # (otherwise the busy lightbox is silent for the full
    # reconstruction). Pure prints go to stdout regardless.
    on_progress: "Optional[Callable[[str, float], None]]" = None,
) -> list[Mesh]:
    """Reconstruct a range of labelled slices into 3D meshes.

    `epi_per_slice` / `fasc_per_slice` are sparse maps from
    slice idx → 2D bool mask. Unannotated frames in
    `slice_range` are filled in via ZOH (see `zoh_fill`).

    Output ordering: epi first (if any), then one mesh per 3D
    fascicle connected component. The 3D split keeps a fascicle
    that's continuous across many frames as a single mesh,
    which is what you want for FEM material assignment later.
    """
    _t0 = time.perf_counter()
    s_lo, s_hi = slice_range
    if s_hi < s_lo:
        raise ValueError(
            f"slice_range start > end: ({s_lo}, {s_hi})",
        )
    voxel_xy_mm = float(voxel_xy_mm)
    voxel_z_mm = float(voxel_z_mm)
    if voxel_xy_mm <= 0 or voxel_z_mm <= 0:
        raise ValueError(
            "voxel sizes must be > 0 "
            f"(got xy={voxel_xy_mm}, z={voxel_z_mm})",
        )
    _t = time.perf_counter()
    epi_filled = zoh_fill(epi_per_slice, slice_range)
    fasc_filled = zoh_fill(fasc_per_slice, slice_range)
    _emit_progress(
        on_progress, "ZOH-fill slices",
        time.perf_counter() - _t,
    )
    # 2D speckle cleanup per slice BEFORE building the 3D
    # volume. Applied to the ZOH-filled view so every slab
    # going into marching cubes is cleaned uniformly (the
    # propagated frames typically inherit the noise of the
    # nearest annotated frame). No-op when all knobs are 0.
    if (
        int(clean_min_component_px) > 0
        or int(clean_min_hole_px) > 0
        or int(clean_closing_radius_px) > 0
    ):
        _t = time.perf_counter()
        epi_filled = {
            k: cleanup_2d_mask(
                v,
                min_component_px=int(clean_min_component_px),
                min_hole_px=int(clean_min_hole_px),
                closing_radius_px=int(
                    clean_closing_radius_px,
                ),
            )
            for k, v in epi_filled.items()
        }
        fasc_filled = {
            k: cleanup_2d_mask(
                v,
                min_component_px=int(clean_min_component_px),
                min_hole_px=int(clean_min_hole_px),
                closing_radius_px=int(
                    clean_closing_radius_px,
                ),
            )
            for k, v in fasc_filled.items()
        }
        _emit_progress(
            on_progress,
            f"2D cleanup ({len(epi_filled) + len(fasc_filled)} slices)",
            time.perf_counter() - _t,
        )
    if not epi_filled and not fasc_filled:
        return []
    sample_mask = (
        next(iter(epi_filled.values())) if epi_filled
        else next(iter(fasc_filled.values()))
    )
    h, w = sample_mask.shape
    n_slices = s_hi - s_lo + 1
    epi_vol = np.zeros((n_slices, h, w), dtype=bool)
    fasc_vol = np.zeros((n_slices, h, w), dtype=bool)
    for z in range(n_slices):
        i = s_lo + z
        if i in epi_filled and epi_filled[i].shape == (h, w):
            epi_vol[z] = epi_filled[i]
        if i in fasc_filled and fasc_filled[i].shape == (h, w):
            fasc_vol[z] = fasc_filled[i]

    # 3D volume cleanup AFTER per-slice 2D cleanup. Catches
    # Z-direction speckles + voids that 2D can't see (a streak
    # speckle living in slices 5-7, or a background column
    # punched through several slices in the middle of a
    # fascicle). 26-connectivity components / holes; the
    # `min_component_voxels` knob below is a SEPARATE built-in
    # filter that's been in place forever for noise-rejection
    # at MC time, and runs AFTER this.
    # M26 — Always run epi cleanup with keep_largest_only=True
    # so a single mislabelled blob far from the main nerve gets
    # dropped regardless of its voxel count. Anatomically there
    # is exactly one continuous epineurium per nerve, so any
    # disconnected component is a phantom. The min_component
    # voxel threshold gets skipped entirely for the epi when
    # keep_largest_only is on (it's a superset filter).
    _do_3d_cleanup = (
        int(clean_3d_min_component_vox) > 0
        or int(clean_3d_min_hole_vox) > 0
    )
    if _do_3d_cleanup or epi_vol.any():
        _t = time.perf_counter()
        if epi_vol.any():
            _n_components_before = 0
            if _do_3d_cleanup or True:
                # Always apply keep_largest_only to the epi.
                from scipy.ndimage import (
                    generate_binary_structure as _gbs,
                    label as _lab,
                )
                _labels_pre, _n_components_before = _lab(
                    epi_vol, structure=_gbs(3, 3),
                )
            epi_vol = cleanup_3d_mask(
                epi_vol,
                min_component_vox=int(
                    clean_3d_min_component_vox,
                ),
                min_hole_vox=int(clean_3d_min_hole_vox),
                keep_largest_only=True,
            )
            if _n_components_before > 1:
                _emit_progress(
                    on_progress,
                    f"Epi keep-largest-component: "
                    f"{_n_components_before} components → 1 "
                    "(phantoms dropped)",
                    0.0,
                )
        if _do_3d_cleanup and fasc_vol.any():
            fasc_vol = cleanup_3d_mask(
                fasc_vol,
                min_component_vox=int(
                    clean_3d_min_component_vox,
                ),
                min_hole_vox=int(clean_3d_min_hole_vox),
            )
        _emit_progress(
            on_progress, "3D volume cleanup",
            time.perf_counter() - _t,
        )

    # M24 — Inward-offset fascicles so their isosurfaces land
    # inside the epi after marching cubes. The inter-surface
    # diagnostic (Phase 2) showed every fascicle straddles the
    # epi by sub-µm to ~3 µm — TetGen then refuses to classify
    # them as nested regions. Eroding `epi_vol` by N voxels and
    # intersecting with `fasc_vol` guarantees a N-voxel margin.
    # Skipped when either volume is empty OR when the
    # fascicle/epi voxel overlap is essentially zero (epi was
    # painted as a shell rather than a filled region, in which
    # case the erosion+intersection would null the fascicles).
    if (
        int(fasc_inset_vox) > 0
        and epi_vol.any()
        and fasc_vol.any()
    ):
        _overlap = int(np.logical_and(epi_vol, fasc_vol).sum())
        _fasc_total = int(fasc_vol.sum())
        if _overlap < 0.5 * _fasc_total:
            _emit_progress(
                on_progress,
                f"Fascicle-inside-epi inset SKIPPED: only "
                f"{_overlap:,} / {_fasc_total:,} fasc voxels "
                "overlap with epi — epi appears to be a shell, "
                "not filled. Annotate epi as a filled region "
                "to enable the inset.",
                0.0,
            )
        else:
            _t = time.perf_counter()
            from scipy.ndimage import binary_erosion as _erode
            _n_before = int(fasc_vol.sum())
            _eroded_epi = _erode(
                epi_vol, iterations=int(fasc_inset_vox),
            )
            fasc_vol = fasc_vol & _eroded_epi
            _n_after = int(fasc_vol.sum())
            _emit_progress(
                on_progress,
                f"Fascicle inset {fasc_inset_vox} vox inside epi: "
                f"{_n_before:,} → {_n_after:,} fasc voxels "
                f"({_n_before - _n_after:,} dropped to enforce "
                "nesting)",
                time.perf_counter() - _t,
            )

    spacing = (voxel_z_mm, voxel_xy_mm, voxel_xy_mm)
    meshes: list[Mesh] = []
    if epi_vol.any():
        _t = time.perf_counter()
        m = _mc_to_mesh(
            epi_vol, spacing=spacing, name="epi",
            smooth_sigma=smooth_sigma,
            smooth_sigma_xy_mm=smooth_sigma_xy_mm,
            smooth_sigma_z_mm=smooth_sigma_z_mm,
        )
        _emit_progress(
            on_progress,
            f"Marching cubes (epi, n_vox={int(epi_vol.sum()):,})",
            time.perf_counter() - _t,
        )
        if m is not None:
            meshes.append(m)
    if fasc_vol.any():
        # Drop noise components below `min_component_voxels`
        # FIRST — SAM2 video propagation can leave 1-10 voxel
        # speckles (single-frame paint artefacts, edge wiggle
        # near a weak fascicle) which vanish below 0.5 after
        # Gaussian smoothing and would trip marching cubes
        # with `Surface level must be within volume data
        # range`. A real fascicle clears 500 voxels by orders
        # of magnitude at 10 µm spacing.
        #
        # Then UNION the surviving components into a single
        # endoneurium mesh — we DON'T split into per-fascicle
        # STLs in multi-slice mode. Reason: in a real µCT
        # stack fascicles split / merge along the nerve length,
        # so 3D connected-component identity is ambiguous (one
        # bridging voxel anywhere fuses two fascicles into one
        # mesh; a thin pinch breaks one into two). For the
        # downstream FEM + axial-extrude fiber model the endo
        # region is one homogeneous conductor anyway, so we
        # serialise it as a single `endoneurium.stl`. The
        # single-slice extrude path (`extrude_single_slice`)
        # still emits per-fascicle STLs — there, each 2D blob
        # IS a distinct fascicle by construction.
        _t = time.perf_counter()
        _all_ccs = _split_components_3d(fasc_vol)
        _emit_progress(
            on_progress,
            f"3D component split (fasc, {len(_all_ccs)} blobs)",
            time.perf_counter() - _t,
        )
        _endo_vol = np.zeros_like(fasc_vol, dtype=bool)
        for fcc in _all_ccs:
            if int(fcc.sum()) >= int(min_component_voxels):
                _endo_vol |= fcc
        if _endo_vol.any():
            _t = time.perf_counter()
            m = _mc_to_mesh(
                _endo_vol, spacing=spacing,
                name="endoneurium",
                smooth_sigma=smooth_sigma,
                smooth_sigma_xy_mm=smooth_sigma_xy_mm,
                smooth_sigma_z_mm=smooth_sigma_z_mm,
            )
            _emit_progress(
                on_progress,
                f"Marching cubes (endo, n_vox={int(_endo_vol.sum()):,})",
                time.perf_counter() - _t,
            )
            if m is not None:
                meshes.append(m)
    if refine and meshes:
        _t = time.perf_counter()
        refined: list[Mesh] = []
        for m in meshes:
            _tm = time.perf_counter()
            r = refine_mesh(
                m,
                n_passes=int(refine_passes),
                use_optimesh=bool(refine_use_optimesh),
                remesh=bool(refine_remesh),
                remesh_edge_len_mm=(
                    refine_remesh_edge_len_mm
                ),
                decimate_target_tris=decimate_target_tris,
                decimate_target_fraction=decimate_target_fraction,
                on_progress=on_progress,
            )
            _emit_progress(
                on_progress,
                f"Refine '{m.name}' "
                f"(in={m.n_triangles:,} out={r.n_triangles:,} tris)",
                time.perf_counter() - _tm,
            )
            refined.append(r)
        meshes = refined
        _emit_progress(
            on_progress,
            f"Refinement (total, {len(meshes)} meshes)",
            time.perf_counter() - _t,
        )
    elif (
        decimate_target_tris is not None
        or decimate_target_fraction is not None
    ) and meshes:
        # Same pattern as extrude_single_slice — allow the
        # user's decimation knob to fire even when refinement
        # is disabled. Forwards to refine_mesh with the other
        # passes zeroed out so the decimate step is the only
        # work performed.
        meshes = [
            refine_mesh(
                m,
                n_passes=0,
                drop_speck_tris=0,
                use_optimesh=False,
                remesh=False,
                decimate_target_tris=decimate_target_tris,
                decimate_target_fraction=decimate_target_fraction,
            )
            for m in meshes
        ]
    # NOTE: per-surface quality reporting (M13 Phase 1) used to
    # run here too — removed because the action layer already
    # calls report_mesh_quality_batch when writing the JSON, and
    # the pyvista + trimesh checks inside each report (feature-
    # edge extraction, is_watertight) are O(n_tri × log n_tri)
    # which doubled total reconstruction time on 100k+ tri
    # meshes. Single call now lives in do_run_reconstruction.
    _emit_progress(
        on_progress,
        f"reconstruct_stack total ({len(meshes)} meshes)",
        time.perf_counter() - _t0,
    )
    return meshes


# --------------------------------------------------------------
# Bundle discovery + loading (the import-wizard side of the
# write_binary_stl pipeline)
# --------------------------------------------------------------


# Magic value the import wizard matches on to distinguish a
# Golgi-generated nerve from an arbitrary directory of STLs.
# Stored in `manifest.json["kind"]`. Bump together with
# manifest "schema" if the bundle layout changes.
BUNDLE_KIND = "golgi-uct-nerve"


def list_bundles(uct_dir: Path | str) -> list[dict]:
    """Scan `<uct_dir>/nerve_3d/` and return a summary list of
    Golgi-µCT nerve bundles found there, sorted newest-first.

    Each entry: ``{"id": <timestamp-subdir-name>, "dir": <abs
    Path>, "manifest": <parsed dict>, "summary": <one-line
    str>}``. Dirs without a `manifest.json`, or whose manifest
    has a different `kind`, are skipped silently — that lets
    the import wizard list only bundles produced by Step-1 of
    the Segment-µCT dialog without misclassifying other STL
    folders the user might drop in.
    """
    import json as _json
    root = Path(uct_dir) / "nerve_3d"
    if not root.is_dir():
        return []
    out: list[dict] = []
    for sub in sorted(root.iterdir(), reverse=True):
        if not sub.is_dir():
            continue
        manifest_path = sub / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            with open(manifest_path) as f:
                manifest = _json.load(f)
        except (OSError, ValueError):
            continue
        if manifest.get("kind") != BUNDLE_KIND:
            continue
        mode = str(manifest.get("mode") or "?")
        n_fasc = int(manifest.get("n_fascicles") or 0)
        # Compact human-readable summary for the picker tile.
        # Single-slice vs multi-slice get a slightly different
        # second clause so the user can tell at a glance.
        if mode == "single":
            slice_clause = (
                f"slice {manifest.get('slice_idx')} · "
                f"{float(manifest.get('thickness_mm') or 0):.2f} "
                "mm extrude"
            )
        else:
            sr = manifest.get("slice_range") or [0, 0]
            slice_clause = (
                f"slices {sr[0]}–{sr[1]} · "
                f"z={float(manifest.get('voxel_z_mm') or 0):.4f} mm"
            )
        # Include the epi in the summary so the picker tile
        # is clear that BOTH outer shell + per-fascicle endo
        # surfaces are part of the bundle. Previously read as
        # "29 fascicles · …" which made it look like the epi
        # was missing.
        summary = (
            f"epi + {n_fasc} fascicle"
            f"{'s' if n_fasc != 1 else ''} · "
            f"{slice_clause}"
        )
        out.append({
            "id": sub.name,
            "dir": sub,
            "manifest": manifest,
            "summary": summary,
        })
    return out


def _read_stl_to_pts_tris(
    path: Path | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL → (verts, faces). Used by `load_bundle`
    to lift each per-class STL back into NumPy arrays the PLC
    pipeline can consume directly.

    STL stores per-triangle vertex coords with no shared-vertex
    table, so we de-duplicate identical positions to build a
    proper indexed mesh — TetGen needs that to detect shared
    edges between fascicle surfaces and the epi hull.
    """
    p = Path(path)
    with open(p, "rb") as f:
        f.read(_STL_HEADER_SIZE)
        n_tri = struct.unpack("<I", f.read(4))[0]
        rec_dtype = np.dtype([
            ("n",   "<f4", (3,)),
            ("v0",  "<f4", (3,)),
            ("v1",  "<f4", (3,)),
            ("v2",  "<f4", (3,)),
            ("att", "<u2"),
        ])
        records = np.frombuffer(
            f.read(n_tri * 50), dtype=rec_dtype, count=n_tri,
        )
    raw_verts = np.empty((n_tri * 3, 3), dtype=np.float64)
    raw_verts[0::3] = records["v0"]
    raw_verts[1::3] = records["v1"]
    raw_verts[2::3] = records["v2"]
    # Quantize to 1 nm to merge near-duplicate verts coming from
    # the marching-cubes float32 round-trip without collapsing
    # adjacent-but-distinct boundary verts. 1 nm is well below
    # any µCT voxel size in practice.
    quant = np.round(raw_verts * 1.0e9).astype(np.int64)
    _, idx_first, inverse = np.unique(
        quant, axis=0, return_index=True, return_inverse=True,
    )
    verts = raw_verts[idx_first]
    # `inverse` maps each raw vert (row-major across triangles)
    # back to its dedup index → reshape into (n_tri, 3) faces.
    faces = inverse.reshape(n_tri, 3).astype(np.int64)
    return verts, faces


def load_bundle(
    bundle_dir: Path | str,
) -> dict:
    """Load a Golgi-µCT bundle into a dict the import pipeline
    can hand to PLC / mesh / fiber stages.

    Returns:
        {
          "kind":         "uct_bundle",
          "manifest":     <parsed manifest.json>,
          "epi":          {"verts": (N, 3), "faces": (M, 3),
                           "stl_path": <Path>},
          "fascicles":    [{"verts": ..., "faces": ...,
                            "stl_path": ...}, ...],
          "voxel_xy_mm":  float,
          "voxel_z_mm":   float | None,
        }

    Raises FileNotFoundError when the manifest is missing,
    ValueError when the manifest has the wrong kind, and a
    chained RuntimeError when any expected .stl is missing.
    """
    import json as _json
    bdir = Path(bundle_dir)
    manifest_path = bdir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json under {bdir}",
        )
    with open(manifest_path) as f:
        manifest = _json.load(f)
    if manifest.get("kind") != BUNDLE_KIND:
        raise ValueError(
            f"{manifest_path} is not a {BUNDLE_KIND!r} bundle "
            f"(kind={manifest.get('kind')!r})",
        )
    epi_path = bdir / "epi.stl"
    if not epi_path.is_file():
        raise RuntimeError(
            f"Bundle {bdir} is missing epi.stl",
        )
    epi_v, epi_f = _read_stl_to_pts_tris(epi_path)
    fasc: list[dict] = []
    # Two valid layouts for the "inside the epi" region:
    #   * Multi-slice marching-cubes recon writes a single
    #     `endoneurium.stl` covering the union of all
    #     fascicles. Downstream code consumes this as one
    #     entry in the fascicles list — the FEM treats endo
    #     as one homogeneous σ region regardless of how it
    #     was generated, and the axial-extrude fiber method
    #     happily samples inside a multi-lobed mesh.
    #   * Single-slice extrude recon writes one STL per
    #     2D-connected-component as `fascicle_<i>.stl`. There
    #     each blob IS a distinct fascicle by construction.
    # Manifest lists files for both layouts; we pick one path
    # per file name to avoid double-counting if both ever
    # coexist in the same bundle (they shouldn't).
    for fn in manifest.get("files", []):
        is_endo_union = (fn == "endoneurium.stl")
        is_per_fasc = fn.startswith("fascicle_")
        if not (is_endo_union or is_per_fasc):
            continue
        p = bdir / fn
        if not p.is_file():
            raise RuntimeError(
                f"Bundle {bdir} references {fn} but the file "
                f"is missing",
            )
        v, fa = _read_stl_to_pts_tris(p)
        fasc.append({
            "verts": v,
            "faces": fa,
            "stl_path": p,
        })
    return {
        "kind": "uct_bundle",
        "manifest": manifest,
        "epi": {
            "verts": epi_v,
            "faces": epi_f,
            "stl_path": epi_path,
        },
        "fascicles": fasc,
        "voxel_xy_mm": float(manifest.get("voxel_xy_mm") or 0.0),
        "voxel_z_mm": (
            float(manifest["voxel_z_mm"])
            if manifest.get("voxel_z_mm") is not None
            else None
        ),
    }


# --------------------------------------------------------------
# Offscreen preview PNG
# --------------------------------------------------------------


def render_preview_png(
    meshes: list[Mesh],
    *,
    window_size: tuple[int, int] = (640, 480),
    epi_opacity: float = 0.35,
    fasc_opacity: float = 0.9,
    epi_color: str = "#4caf50",
    fasc_color: str = "#60a5fa",
    background: str = "#1f2024",
) -> bytes:
    """Render the reconstructed meshes to an offscreen PNG.

    Epi is rendered semi-transparent so the fascicles inside
    are visible through the outer shell — that's the geometric
    insight the user is checking ("does my epi actually contain
    my fascicles?"). Fascicles render solid blue to match the
    Step-2 label colour. Per-fascicle distinct hues are
    deliberately NOT used here so users don't confuse the
    preview palette with the per-proposal hues in Step 2.

    Returns the PNG bytes. Caller wraps in a base64 data URL
    for the dialog's `<img>` widget. Requires PyVista with
    offscreen rendering — raises RuntimeError if it's not
    installed.
    """
    try:
        import pyvista as pv
    except ImportError as ex:
        raise RuntimeError(
            f"render_preview_png needs pyvista: {ex}",
        ) from ex
    if not meshes:
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new(
            "RGB", window_size,
            color=(31, 32, 36),
        ).save(buf, format="PNG")
        return buf.getvalue()
    plotter = pv.Plotter(
        off_screen=True, window_size=list(window_size),
    )
    plotter.set_background(background)
    for m in meshes:
        n_t = int(m.faces.shape[0])
        if n_t == 0:
            continue
        flat = np.empty(n_t * 4, dtype=np.int64)
        flat[0::4] = 3
        flat[1::4] = m.faces[:, 0]
        flat[2::4] = m.faces[:, 1]
        flat[3::4] = m.faces[:, 2]
        pd = pv.PolyData(
            np.asarray(m.verts, dtype=np.float64), flat,
        )
        if m.name == "epi":
            plotter.add_mesh(
                pd, color=epi_color,
                opacity=float(epi_opacity),
                smooth_shading=True, show_edges=False,
            )
        else:
            plotter.add_mesh(
                pd, color=fasc_color,
                opacity=float(fasc_opacity),
                smooth_shading=True, show_edges=False,
            )
    plotter.camera_position = "iso"
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 20
    plotter.reset_camera()
    arr = plotter.screenshot(
        return_img=True, transparent_background=False,
    )
    plotter.close()
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------
# Binary STL writer
# --------------------------------------------------------------


_STL_HEADER_SIZE = 80


def write_binary_stl(mesh: Mesh, path: Path | str) -> None:
    """Write `mesh` to a binary STL file.

    Format (little-endian):
      * 80-byte header (any content; here we write a slug so
        viewers that show it know the provenance).
      * 4-byte uint32 triangle count.
      * Per triangle (50 bytes): 3×float32 face normal, 3×3
        float32 vertex coords, 2-byte attribute count (0).

    Normals are computed from the vertex winding so the STL
    is self-consistent — scikit-image's MC returns CCW-from-
    outside windings, so `(v1 - v0) × (v2 - v0)` points
    outward.
    """
    verts = mesh.verts.astype(np.float32, copy=False)
    faces = mesh.faces.astype(np.int32, copy=False)
    n_tri = int(faces.shape[0])
    if n_tri == 0:
        raise ValueError(
            f"Mesh '{mesh.name}' has no triangles — refusing "
            "to write an empty STL.",
        )
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn = np.where(nn == 0, 1.0, nn)
    n = (n / nn).astype(np.float32)

    # NumPy structured array → tobytes() avoids a Python-level
    # per-triangle loop. Saves ~100× over the naive struct.pack
    # loop on meshes with 50 k+ triangles (which the MC pass
    # easily produces on a 1 k-wide nerve).
    rec_dtype = np.dtype([
        ("n",   "<f4", (3,)),
        ("v0",  "<f4", (3,)),
        ("v1",  "<f4", (3,)),
        ("v2",  "<f4", (3,)),
        ("att", "<u2"),
    ])
    records = np.empty(n_tri, dtype=rec_dtype)
    records["n"] = n
    records["v0"] = v0
    records["v1"] = v1
    records["v2"] = v2
    records["att"] = 0

    header = (
        f"golgi nerve STL: {mesh.name}".encode("ascii")
    )[:_STL_HEADER_SIZE]
    header = header.ljust(_STL_HEADER_SIZE, b"\0")
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n_tri))
        f.write(records.tobytes())
