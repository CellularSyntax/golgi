# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Per-element shape-quality scalars for surface (triangle) and
volume (tetrahedron) meshes.

Extracted from `golgi/app.py` in step W1.2 of FEATURES.md. Both
functions are pure numpy (no external deps) so they work inside
slim worker processes that don't carry the full viz stack.

Two metrics, both normalised so 1.0 = perfectly regular (equilateral
triangle / regular tetrahedron) and 0.0 = degenerate:

- `surface_quality` (triangle):
      q = 4·√3 · area / (a² + b² + c²)
  Heron's formula for area; degenerates cleanly when the triangle
  has zero area.

- `tet_shape_quality` (tetrahedron):
      q = 6·√2 · V / max(edge)³
  Vol via the signed scalar triple product.

Both are vectorised over the input arrays — cheap enough to run on
a ~1 M-tet mesh without parallelism.
"""
from __future__ import annotations

import numpy as np


def surface_quality(
    pts: np.ndarray,
    tris: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Per-triangle q_radius_ratio in [0, 1]. 1.0 = equilateral,
    → 0 = degenerate sliver. Uses Heron's formula:
        q = 4·√3 · area / (a² + b² + c²)
    No external deps so it works inside the load worker even when
    meshplex isn't available on the system. Returns (q, "Heron")
    — the metric tag is included so consumers can label histograms
    with the formula provenance."""
    a = pts[tris[:, 0]]
    b = pts[tris[:, 1]]
    c = pts[tris[:, 2]]
    e0 = np.linalg.norm(b - c, axis=1)
    e1 = np.linalg.norm(a - c, axis=1)
    e2 = np.linalg.norm(a - b, axis=1)
    s = 0.5 * (e0 + e1 + e2)
    area = np.sqrt(np.maximum(s * (s - e0) * (s - e1) * (s - e2), 0))
    denom = e0 ** 2 + e1 ** 2 + e2 ** 2
    q = np.where(denom > 0, 4.0 * np.sqrt(3.0) * area / denom, 0.0)
    return q.astype(np.float32), "Heron"


def tet_shape_quality(
    pts: np.ndarray,
    tets: np.ndarray,
) -> np.ndarray:
    """Per-tet shape-regularity quality in [0, 1]. Formula:
        q = 6·√2 · V / max(edge)³
    Regular tetrahedron → 1.0; degenerate sliver → 0. Vectorised
    over the input array. Cheap enough to run on a ~1 M-tet mesh
    without parallelism."""
    a = pts[tets[:, 0]]
    b = pts[tets[:, 1]]
    c = pts[tets[:, 2]]
    d = pts[tets[:, 3]]
    # 6·V = | (a-d) · ((b-d) × (c-d)) |
    v_ad = a - d
    v_bd = b - d
    v_cd = c - d
    cross_bd_cd = np.cross(v_bd, v_cd)
    six_v = np.abs(np.einsum("ij,ij->i", v_ad, cross_bd_cd))
    vol = six_v / 6.0
    # max edge per tet
    e = np.stack([
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - a, axis=1),
        np.linalg.norm(d - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(d - b, axis=1),
        np.linalg.norm(d - c, axis=1),
    ], axis=1)
    max_e = e.max(axis=1)
    max_e3 = max_e ** 3
    q = np.where(
        max_e3 > 0.0,
        6.0 * np.sqrt(2.0) * vol / max_e3,
        0.0,
    )
    # Numerical noise can push >1; clip to [0,1].
    return np.clip(q, 0.0, 1.0).astype(np.float32)
