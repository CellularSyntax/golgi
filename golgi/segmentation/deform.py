# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Nerve cross-section deformation (shared by the GUI mesh build and the
Duke batch).

Currently one non-trivial mode: an AREA-PRESERVING round. The nerve's
cross-section PCA gives semi-axes (a, b); the affine

    M = E · diag(r/a, r/b) · Eᵀ ,   r = sqrt(a·b)

maps that ellipse to a circle of the SAME area (det M = 1). The SAME
(M, centre) must be applied to the fascicles so they stay correctly placed
inside the rounded nerve — this is the Duke/ASCENT convention for a clean,
uniform cuff annulus.
"""
from __future__ import annotations

import numpy as np

# selectable modes (kept tiny on purpose — a none/round toggle)
DEFORM_MODES = ("none", "round")


def nerve_round_affine(nerve_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, centre) for the area-preserving round of a nerve outline.

    `nerve_xy`: (N, ≥2) points of the NERVE OUTER cross-section (epineurium
    boundary). Compute M, centre once here and apply with `apply_round_xy`
    to the nerve AND every fascicle.
    """
    xy = np.asarray(nerve_xy, dtype=float)[:, :2]
    c = xy.mean(0)
    ev, evec = np.linalg.eigh(np.cov((xy - c).T))
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    a, b = np.sqrt(max(ev[0], 1e-30)), np.sqrt(max(ev[1], 1e-30))
    r = np.sqrt(a * b)
    M = evec @ np.diag([r / a, r / b]) @ evec.T
    return M, c


def apply_round_xy(pts: np.ndarray, M: np.ndarray, c: np.ndarray,
                   *, keep_centroid: bool = True) -> np.ndarray:
    """Apply the round affine to the xy of `pts` (z untouched).

    keep_centroid=True rounds the section IN PLACE (centroid fixed) — use in
    the GUI so the nerve stays on the cuff axis. keep_centroid=False centres
    the section at the origin — the batch convention (cuff built at origin).
    """
    o = np.asarray(pts, dtype=float).copy()
    o[:, :2] = (o[:, :2] - c) @ M + (c if keep_centroid else 0.0)
    return o


def min_enclosing_circle(xy: np.ndarray, iters: int = 8000):
    """Approximate minimum enclosing circle ``(centre, radius)`` of 2D points.

    Bădoiu–Clarkson: the centre is pulled toward the current farthest point
    with a 1/(i+2) step, converging to the MEC centre; the returned radius is
    the max distance from that centre, which is always ≥ the true MEC radius
    (so the circle never clips the points — safe for sizing a cuff). It is
    fully deterministic, so callers that start from the same outline (the FEM
    mesher and the figure scripts) land in the *identical* frame.

    For the extruded Duke nerves the cross-section is a constant prism, so this
    is the exact tightest circle that the (circular) cuff can hug.
    """
    P = np.ascontiguousarray(np.asarray(xy, float)[:, :2])
    c = P.mean(0)
    for i in range(int(iters)):
        c = c + (P[np.argmax(((P - c) ** 2).sum(1))] - c) / (i + 2.0)
    return c, float(np.sqrt(((P - c) ** 2).sum(1).max()))


def round_mesh_list(meshes, *, keep_centroid: bool = True):
    """Apply the area-preserving round to a list of extruded `Mesh` objects
    (mesh[0] = nerve epi → defines the affine; applied IN PLACE to the nerve
    AND every fascicle so they stay co-registered). Returns `meshes`."""
    if not meshes:
        return meshes
    M, c = nerve_round_affine(np.asarray(meshes[0].verts)[:, :2])
    for m in meshes:
        m.verts = apply_round_xy(
            np.asarray(m.verts, dtype=float), M, c,
            keep_centroid=keep_centroid).astype(
                np.asarray(m.verts).dtype, copy=False)
    return meshes
