# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Standalone TetGen runner — receives a TetGenPayload JSON,
tetrahedralises, writes npz.

Spawned as a subprocess by golgi's mesh-build driver
(`pipeline/mesh.py:run_tetgen_subprocess`). Stdout lines prefixed
`[runner]` are streamed to the busy-lightbox log in the UI.

Payload contract: a `TetGenPayload` dataclass serialized to JSON
on sys.argv[1]. See `golgi/jobs/schemas.py` for the field list —
both this script and the driver import the same schema, so a
field rename is a one-PR change that won't silently drift here.

M35 — Retry loop on "input surface mesh contain self-
intersections" errors. TetGen writes the IDs of skipped (=
self-intersecting) input triangles to a `_skipped.face` tmpfile
before bailing. We parse that, drop the listed triangles from
the PLC, and retry. Up to 3 retries; each iteration typically
drops fewer tris than the last (TetGen finds more SI cases as
the surface gets sparser around the original problem area), and
3 attempts is usually enough to converge for the float-precision-
straddle case the user kept hitting on µCT-derived geometries.
"""
import sys
import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv
import tetgen

from golgi.jobs.schemas import TetGenPayload


MAX_RETRIES = 5


def _parse_skipped_face_file(path: Path) -> list[tuple[int, int, int]]:
    """Parse a TetGen `.face` file from a `_skipped` tmpfile.

    Format (line by line):
        <n_faces> <bmark_flag>
        <face_idx> <v0> <v1> <v2> [bmark]
        ...
    Comments start with '#'. We return a list of sorted vertex-
    index triples — sorted so the caller can match against the
    PLC's faces regardless of winding.
    """
    triples: list[tuple[int, int, int]] = []
    if not path.is_file():
        return triples
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return triples
    data_rows: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        data_rows.append(s)
    if len(data_rows) < 2:
        return triples
    # First non-comment line is the header — skip it.
    for row in data_rows[1:]:
        parts = row.split()
        if len(parts) < 4:
            continue
        try:
            v0, v1, v2 = (
                int(parts[1]), int(parts[2]), int(parts[3]),
            )
        except ValueError:
            continue
        triples.append(tuple(sorted((v0, v1, v2))))
    return triples


def _parse_skipped_node_file(path: Path) -> dict[int, tuple[float, float, float]]:
    """Parse a TetGen `.node` file written alongside `_skipped.face`.

    Format (line by line):
        <n_points> <dim> <n_attrs> <bmark_flag>
        <pt_idx> <x> <y> <z> [attrs] [bmark]
        ...

    Returns a dict mapping TetGen's vertex index → (x, y, z).
    Used by `_drop_faces_from_plc` to translate triples from
    TetGen's internal index space into our PLC's index space via
    coordinate matching — robust against silent reindexing
    between retry attempts.
    """
    coords: dict[int, tuple[float, float, float]] = {}
    if not path.is_file():
        return coords
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return coords
    data_rows: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        data_rows.append(s)
    if len(data_rows) < 2:
        return coords
    for row in data_rows[1:]:
        parts = row.split()
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        coords[idx] = (x, y, z)
    return coords


def _find_skipped_node_file(cwd: Path, plc_dir: Path) -> Path | None:
    for d in (cwd, plc_dir, Path.cwd()):
        cand = d / "tetgen-tmpfile_skipped.node"
        if cand.is_file():
            return cand
    return None


def _translate_triples_via_coords(
    triples: list[tuple[int, int, int]],
    tetgen_coords: dict[int, tuple[float, float, float]],
    plc_pts: np.ndarray,
    *,
    tol: float = 1.0e-6,
) -> list[tuple[int, int, int]]:
    """Translate vertex indices in `triples` from TetGen's index
    space into the PLC's index space by KDTree-matching each
    vertex's coordinate against `plc_pts`.

    Triples that reference indices not present in `tetgen_coords`
    (typically Steiner points TetGen inserted internally during
    boundary recovery — not in the input PLC) are dropped here:
    we have no way to identify the corresponding input triangle
    when one of the vertices is internal-only.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(plc_pts)
    out: list[tuple[int, int, int]] = []
    n_steiner = 0
    n_unmatched = 0
    for tri in triples:
        if any(v not in tetgen_coords for v in tri):
            n_steiner += 1
            continue
        new_tri: list[int] = []
        ok = True
        for v in tri:
            d, ni = tree.query(tetgen_coords[v], k=1)
            if d > tol:
                ok = False
                break
            new_tri.append(int(ni))
        if not ok:
            n_unmatched += 1
            continue
        out.append(tuple(sorted(new_tri)))
    if n_steiner or n_unmatched:
        print(
            f"[runner] retry: triple translation dropped "
            f"{n_steiner} Steiner-vertex triple(s) and "
            f"{n_unmatched} unmatched (kept {len(out)}/{len(triples)})",
            flush=True,
        )
    return out


def _find_skipped_face_file(cwd: Path, plc_dir: Path) -> Path | None:
    """TetGen writes `tetgen-tmpfile_skipped.face` somewhere on
    disk. Empirically that's the runner subprocess's working
    directory, but try a few fallback locations defensively."""
    for d in (cwd, plc_dir, Path.cwd()):
        cand = d / "tetgen-tmpfile_skipped.face"
        if cand.is_file():
            return cand
    return None


def _drop_faces_from_plc(
    plc: "pv.PolyData",
    bad_triples: list[tuple[int, int, int]],
) -> "pv.PolyData":
    """Return a new PolyData with all faces matching any triple
    in `bad_triples` removed, then compact orphan vertices.

    M48b — orphan compaction between retries. The original loop
    dropped faces but left orphan vertices in `plc.points`. On the
    next iteration TetGen's internal indexing diverged from our
    face-table indices, and the next `_skipped.face` referenced
    vertex IDs we couldn't match — observed as "dropping 0" on
    retries 2-3 of histology-bundle PLCs with cap-stitch SI
    clusters. After the drop we now run `pv.clean(tolerance=0)`
    which removes unreferenced points and re-emits the face table
    with contiguous indices; the next attempt's `_skipped.face`
    therefore references the SAME compact index space we read
    back here.
    """
    bad_set = set(bad_triples)
    faces_flat = np.asarray(plc.faces)
    if faces_flat.size == 0:
        return plc
    faces = faces_flat.reshape(-1, 4)
    triples = np.sort(faces[:, 1:], axis=1)
    keep_rows = np.array([
        tuple(tri) not in bad_set
        for tri in triples
    ], dtype=bool)
    n_drop = int((~keep_rows).sum())
    print(
        f"[runner] retry: dropping {n_drop} self-intersecting "
        f"tri(s) from PLC ({faces.shape[0]} → "
        f"{int(keep_rows.sum())} after drop)",
        flush=True,
    )
    kept = faces[keep_rows].ravel()
    new = pv.PolyData(plc.points, kept)
    pts_in = int(new.n_points)
    new = new.clean(tolerance=0.0, absolute=True)
    pts_out = int(new.n_points)
    if pts_in != pts_out:
        print(
            f"[runner] retry: compacted "
            f"{pts_in - pts_out} orphan vert(s) "
            f"({pts_in} → {pts_out} pts)",
            flush=True,
        )
    return new


def _build_tetgen(plc: "pv.PolyData", cfg) -> "tetgen.TetGen":
    """Construct a fresh TetGen instance from a PolyData + the
    seed list on the config. Used both for the initial attempt
    and after each retry's PLC mutation."""
    t = tetgen.TetGen(plc)
    for tag, seed, maxv in cfg.seeds:
        t.add_region(int(tag), seed, float(maxv))
        print(
            f"[runner] region {tag}: maxv={maxv:.3e}",
            flush=True,
        )
    return t


def _surgical_drop_fill_by_coords(
    plc: "pv.PolyData",
    skipped_face_path: Path,
    skipped_node_path: "Path | None",
) -> "pv.PolyData | None":
    """Coordinate-based last-resort SI removal for the case the index-match
    drop can't dislodge — TetGen renumbers past our PLC index space during
    boundary recovery (the flagged triangle often involves an internal
    Steiner point), so the `_skipped.face` triple no longer maps to any PLC
    face. We instead locate each flagged triangle by the CENTROID of its
    `_skipped.node` coordinates, delete the NEAREST PLC triangle, and refill
    the resulting micro-hole(s) with pymeshfix's `fill_small_boundaries` so
    the surface stays watertight (plain deletion leaves holes that crash
    TetGen's later `insertpoint` pass). Returns a new PolyData, or None if
    it can't proceed. Driven by TetGen's exact SI flagging (1-2 real tris),
    NOT pymeshfix's false-positive shared-seam count."""
    if skipped_node_path is None or not skipped_node_path.is_file():
        return None
    coords = _parse_skipped_node_file(skipped_node_path)
    triples = _parse_skipped_face_file(skipped_face_path)
    if not coords or not triples:
        return None
    cents = [
        np.mean([coords[v] for v in tri], axis=0)
        for tri in triples
        if all(v in coords for v in tri)
    ]
    if not cents:
        return None
    P = np.asarray(plc.points, dtype=np.float64)
    F = np.asarray(plc.faces).reshape(-1, 4)[:, 1:]
    try:
        from scipy.spatial import cKDTree
        # Delete a small PATCH around each flagged SI (the nearest ~10
        # tris), not just the single nearest one: an SI is a crossing
        # between TWO triangles (here, a conformity mismatch at the
        # nerve cross-section seam where the saline-cap inner ring and
        # the nerve-lateral bottom ring come from different clip pieces),
        # so removing one side and refilling just re-creates the cross.
        # Removing a small patch spanning both sides and refilling it as
        # one conforming patch resolves it.
        _, nidx = cKDTree(P[F].mean(axis=1)).query(
            np.asarray(cents, dtype=np.float64),
            k=min(10, F.shape[0]),
        )
    except Exception:                                      # noqa: BLE001
        return None
    bad = np.unique(np.asarray(nidx, dtype=np.int64).ravel())
    keep = np.ones(F.shape[0], dtype=bool)
    keep[bad] = False
    F2 = np.ascontiguousarray(F[keep])
    print(
        f"[runner] retry: coord-surgery deleted {bad.size} PLC tri(s) "
        f"nearest the flagged SI(s); refilling hole(s)",
        flush=True,
    )
    try:
        from pymeshfix import PyTMesh as _PyTMesh
        m = _PyTMesh()
        m.load_array(
            np.ascontiguousarray(P, dtype=np.float64),
            np.ascontiguousarray(F2, dtype=np.int32),
        )
        m.fill_small_boundaries(nbe=64, refine=True)
        v, f = m.return_arrays()
        ff = np.hstack(
            [np.full((len(f), 1), 3, np.int64), np.asarray(f, np.int64)]
        ).ravel()
        return pv.PolyData(np.asarray(v, dtype=np.float64), ff)
    except Exception as ex:                                # noqa: BLE001
        print(
            f"[runner] retry: coord-surgery refill failed ({ex}); "
            f"keeping delete-only",
            flush=True,
        )
        ff = np.hstack(
            [np.full((F2.shape[0], 1), 3, np.int64), F2]
        ).ravel()
        return pv.PolyData(P, ff)


def main(payload_path: str) -> None:
    cfg = TetGenPayload.deserialize(json.load(open(payload_path, encoding="utf-8")))
    plc = pv.read(str(cfg.plc_path))
    print(
        f"[runner] {plc.n_points:,} pts, {plc.n_faces:,} tris",
        flush=True,
    )
    cwd = Path(os.getcwd())
    plc_dir = Path(cfg.plc_path).parent

    t = _build_tetgen(plc, cfg)
    print("[runner] STARTING tetgen", flush=True)
    r = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = t.tetrahedralize(
                switches=cfg.switches,
                verbose=cfg.verbose,
                epsilon=cfg.epsilon,
                collinear_ang_tol=cfg.collinear_ang_tol,
                facet_separate_ang_tol=cfg.facet_separate_ang_tol,
            )
            break
        except RuntimeError as ex:
            msg = str(ex)
            if "self-intersections" not in msg.lower():
                # Different failure (e.g. degenerate input).
                # Re-raise — the retry loop only handles SI.
                raise
            if attempt >= MAX_RETRIES:
                print(
                    f"[runner] retry exhausted after "
                    f"{attempt} attempt(s); giving up",
                    flush=True,
                )
                raise
            skipped_path = _find_skipped_face_file(cwd, plc_dir)
            if skipped_path is None:
                print(
                    "[runner] retry: TetGen reported SI but "
                    "no _skipped.face tmpfile found — can't "
                    "recover",
                    flush=True,
                )
                raise
            triples = _parse_skipped_face_file(skipped_path)
            if not triples:
                print(
                    "[runner] retry: _skipped.face was empty "
                    "or unparseable — can't recover",
                    flush=True,
                )
                raise
            print(
                f"[runner] retry {attempt + 1}/{MAX_RETRIES}: "
                f"TetGen flagged {len(triples)} input tri(s) "
                f"as self-intersecting (from {skipped_path.name})",
                flush=True,
            )
            # M48c — translate TetGen-internal vertex indices to
            # our PLC's index space via coordinate matching using
            # the `_skipped.node` companion. The .face indices on
            # retry 2+ frequently diverge from our PLC's because
            # TetGen renumbers internally during boundary recovery
            # (and may include Steiner-point indices we never sent
            # in). Coordinate-based translation is robust to both.
            node_path = _find_skipped_node_file(cwd, plc_dir)
            if node_path is not None:
                tetgen_coords = _parse_skipped_node_file(node_path)
                if tetgen_coords:
                    triples = _translate_triples_via_coords(
                        triples,
                        tetgen_coords,
                        np.asarray(plc.points, dtype=np.float64),
                    )
            plc_before = plc
            plc = _drop_faces_from_plc(plc, triples)
            if plc.n_faces == plc_before.n_faces:
                # Index-match drop couldn't dislodge the flagged SI
                # (TetGen renumbered past our PLC index space). Fall back
                # to coordinate-based surgical drop+fill — robust to the
                # renumbering, watertight-preserving, and driven by
                # TetGen's exact flagging.
                print(
                    f"[runner] retry: index drop stalled → coordinate-"
                    f"based surgical drop+fill",
                    flush=True,
                )
                node_p = _find_skipped_node_file(cwd, plc_dir)
                fixed = _surgical_drop_fill_by_coords(
                    plc_before, skipped_path, node_p,
                )
                if fixed is None or fixed.n_faces == 0:
                    print(
                        f"[runner] retry: coord-surgery unavailable; "
                        f"abandoning retry loop",
                        flush=True,
                    )
                    raise
                plc = fixed
            # Clean up the tmpfile so the NEXT iteration's parse
            # doesn't pick up stale content if tetgen succeeds
            # mid-write (defensive — usually it overwrites
            # cleanly, but corner cases happen).
            try:
                skipped_path.unlink()
                node_companion = skipped_path.with_suffix(
                    ".node",
                )
                if node_companion.is_file():
                    node_companion.unlink()
            except Exception:                             # noqa: BLE001
                pass
            t = _build_tetgen(plc, cfg)

    if r is None:
        raise RuntimeError(
            "TetGen retry loop ended without a result — "
            "should not be reachable",
        )
    np.savez(
        str(cfg.out_npz),
        nodes=r[0], elems=r[1], attribs=r[2],
    )
    print(
        f"[runner] DONE: {len(r[0]):,} pts, "
        f"{len(r[1]):,} tets",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1])
