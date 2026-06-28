# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""PCA-based cuff-fitting helpers.

Extracted from `golgi/app.py` in step W1.5 of FEATURES.md. Five
pure-numpy helpers that turn an imported nerve point cloud into a
cuff-aligned coordinate frame:

  1. `global_pca(pts)` — align the largest-variance axis with +z.
  2. `find_cuff_origin_pca(pts_pca, anchor, offset_mm, dx_mm, dy_mm)`
     — pick the cuff origin point in the PCA frame.
  3. `local_pca_refine(pts_pca, cuff_origin, radius_m)` — re-align
     +z to the LOCAL nerve trajectory at the cuff site.
  4. `transform_to_cuff_frame(pts_raw, centroid, R_global,
     cuff_origin_pca, R_local)` — compose the two transforms into
     one mapping that puts the cuff origin at (0,0,0) and the local
     nerve axis at +z.
  5. `autosize_R_ci(pts_cuff, L_cuff_m, clearance_m)` — max nerve
     radius across the cuff axial window + clearance.

All functions are pure numpy (no closure references). Downstream
consumers (pipeline/_frames.py, pipeline/fem.py, watchers/cuff.py)
access them via the H SimpleNamespace bundle built in build_app.
"""
from __future__ import annotations

import numpy as np


def global_pca(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid + rotation matrix aligning the largest-variance
    direction with +z. Returns (centroid, R_global) such that
    pts_pca = (pts - centroid) @ R_global."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(principal, z)
    s = np.linalg.norm(v)
    c = float(np.dot(principal, z))
    if s < 1e-8:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        K = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    return centroid, R


def find_cuff_origin_pca(pts_pca: np.ndarray,
                          anchor: str,
                          offset_mm: float,
                          dx_mm: float = 0.0,
                          dy_mm: float = 0.0) -> np.ndarray:
    """In the PCA frame (+z = nerve axis), find the cuff origin.
    `anchor`: 'trunk' (low-z end), 'branched' (high-z end), or
    'centroid' (PCA centroid, z=0). `offset_mm`: signed offset
    along the trunk axis. `dx_mm`, `dy_mm`: transverse fine-tune
    in the PCA frame (added to the cross-section centroid)."""
    z = pts_pca[:, 2]
    if anchor == "branched":
        z_anchor = z.max()
        z_cuff = z_anchor - offset_mm * 1e-3
    elif anchor == "centroid":
        z_cuff = offset_mm * 1e-3
    else:
        z_anchor = z.min()
        z_cuff = z_anchor + offset_mm * 1e-3
    # xy centroid of points near the cuff plane
    band = np.abs(z - z_cuff) < 2e-3
    if band.sum() < 50:
        # widen
        band = np.abs(z - z_cuff) < 10e-3
    xy = pts_pca[band, :2].mean(axis=0) if band.any() else np.zeros(2)
    return np.array([
        xy[0] + dx_mm * 1e-3,
        xy[1] + dy_mm * 1e-3,
        z_cuff,
    ])


def local_pca_refine(pts_pca: np.ndarray,
                      cuff_origin: np.ndarray,
                      radius_m: float) -> np.ndarray:
    """Local-PCA refinement: align +z to the LOCAL nerve trajectory
    at the cuff site. Returns R_local."""
    d = np.linalg.norm(pts_pca - cuff_origin, axis=1)
    mask = d < radius_m
    if mask.sum() < 100:
        return np.eye(3)
    cov = np.cov((pts_pca[mask] - cuff_origin), rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    if axis[2] < 0:
        axis = -axis
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z); s = np.linalg.norm(v)
    c = float(np.dot(axis, z))
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def transform_to_cuff_frame(pts_raw: np.ndarray,
                             centroid: np.ndarray,
                             R_global: np.ndarray,
                             cuff_origin_pca: np.ndarray,
                             R_local: np.ndarray,
                             rot_z_rad: float = 0.0) -> np.ndarray:
    """Compose the global PCA + local refinement into one transform
    that puts cuff origin at (0,0,0) and local nerve axis at +z.

    F3.2a: `rot_z_rad` adds a rotation around the local cuff axis
    AFTER the local frame is applied — so the nerve sits unchanged
    in the cuff frame but the contact pattern (which is built in
    cuff-frame coordinates with fixed phi) rotates around the
    nerve. Defaults to 0 so legacy callers keep behaviour."""
    pts_pca = (pts_raw - centroid) @ R_global
    pts_cuff = (pts_pca - cuff_origin_pca) @ R_local.T
    if rot_z_rad:
        c = float(np.cos(rot_z_rad))
        s = float(np.sin(rot_z_rad))
        # Rotate the nerve by -rot_z (so contacts, which are at
        # fixed phi in cuff frame, appear rotated BY +rot_z when
        # viewed in the nerve's natural frame). Equivalent to
        # applying Rz(rot_z) to the patch builder, which is what
        # the cuff-fit code consumes downstream.
        Rz_T = np.array(
            [[c, s, 0.0],
             [-s, c, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        pts_cuff = pts_cuff @ Rz_T
    return pts_cuff


def autosize_R_ci(pts_cuff: np.ndarray,
                   L_cuff_m: float,
                   clearance_m: float) -> float:
    """Max nerve radius at cuff axial range + clearance."""
    z_lo, z_hi = -L_cuff_m / 2, +L_cuff_m / 2
    in_cuff = (pts_cuff[:, 2] >= z_lo) & (pts_cuff[:, 2] <= z_hi)
    if in_cuff.sum() < 50:
        # widen
        in_cuff = np.abs(pts_cuff[:, 2]) <= max(L_cuff_m, 4e-3)
    if not in_cuff.any():
        return 1.5e-3
    r = np.linalg.norm(pts_cuff[in_cuff, :2], axis=1).max()
    return float(r) + float(clearance_m)


def refit_design_geometry(
    eid: str,
    *,
    geom,
    state,
) -> bool:
    """F4.1 Phase B — per-design refit lifted out of `build_app()`.

    Computes R_local_elec + R_ci + R_co at the design's current
    (offset, dx, dy) position using a local-PCA pass through the
    nerve, then writes them back into the design dict. Returns
    True on success. Doesn't trigger a render — the caller is
    responsible.

    `geom` is a `scene.geometry.GeometryState`; `state` is
    anything with attribute access (the trame state proxy in the
    GUI path OR the `_HeadlessState` shim used by `golgi.Study`).

    Used by:
      * the per-row "Refit" button (via the refit_design_request
        watcher in `golgi.watchers.cuff`),
      * the design-sweep generator, which has to refit every
        cloned design at its own Z position since the nerve
        isn't a perfect cylinder and a cuff that snugly fits at
        one location can be too tight or too loose a few mm up
        or down the nerve,
      * the headless `Study.run_mesh()` flow — every design that
        lacks a fit gets one on first build.
    """
    target = next(
        (d for d in (state.designs or [])
         if d.get("eid") == eid),
        None,
    )
    if target is None or geom.nerve is None:
        return False
    pts_pca = (
        (geom.nerve["pts_raw"] - geom.centroid) @ geom.R_global
    )
    elec_origin_pca = find_cuff_origin_pca(
        pts_pca, state.cuff_anchor,
        float(target.get("cuff_offset_mm", 0.0)),
        float(target.get("cuff_dx_mm", 0.0)),
        float(target.get("cuff_dy_mm", 0.0)),
    )
    R_local_elec = local_pca_refine(
        pts_pca, elec_origin_pca,
        float(state.local_pca_radius_mm) * 1e-3,
    )
    pts_local = (
        (pts_pca - elec_origin_pca) @ R_local_elec.T
    )
    L_target = float(
        target.get("L_cuff_mm", state.L_cuff_mm),
    ) * 1e-3
    clearance = float(
        target.get(
            "cuff_clearance_mm", state.cuff_clearance_mm,
        ),
    ) * 1e-3
    wall = float(
        target.get("cuff_wall_mm", state.cuff_wall_mm),
    ) * 1e-3
    new_R_ci = autosize_R_ci(pts_local, L_target, clearance)
    new_R_co = float(new_R_ci) + wall

    # LIFE / TIME positioning. Both electrode types live INSIDE
    # the nerve, so auto-fit needs a cross-section centroid in
    # cuff-local frame. The cuff-local xy of the nerve at z=0
    # IS the cuff midplane's nerve centre — same point
    # `find_cuff_origin_pca` already located in PCA frame, just
    # mapped through `R_local_elec` and shifted to local origin.
    # Computed once here so both branches can reuse it.
    _xs_cuff_local_z = pts_local[:, 2]
    _band = np.abs(_xs_cuff_local_z) < 1.0e-3        # ±1 mm
    if int(_band.sum()) < 5:
        _band = np.abs(_xs_cuff_local_z) < 5.0e-3    # widen
    if _band.any():
        _nerve_cx = float(pts_local[_band, 0].mean())
        _nerve_cy = float(pts_local[_band, 1].mean())
    else:
        _nerve_cx = 0.0
        _nerve_cy = 0.0

    def _bundle_fascicle_centroids_local() -> list[tuple[float, float]]:
        """Per-fascicle xy centroid in cuff-local mm at z≈0.
        Empty list for monofascicular / legacy STL imports."""
        if not isinstance(geom.nerve, dict):
            return []
        bundle = geom.nerve.get("bundle")
        if not isinstance(bundle, dict):
            return []
        out: list[tuple[float, float]] = []
        for fasc in (bundle.get("fascicles") or []):
            f_pts = np.asarray(
                fasc.get("verts_m"), dtype=np.float64,
            )
            if f_pts.size == 0:
                continue
            f_pca = (
                (f_pts - geom.centroid) @ geom.R_global
            )
            f_local = (
                (f_pca - elec_origin_pca) @ R_local_elec.T
            )
            _fb = np.abs(f_local[:, 2]) < 1.0e-3
            if int(_fb.sum()) < 5:
                _fb = np.abs(f_local[:, 2]) < 5.0e-3
            if _fb.any():
                out.append((
                    float(f_local[_fb, 0].mean()),
                    float(f_local[_fb, 1].mean()),
                ))
        return out

    designs = list(state.designs or [])
    for d in designs:
        if d.get("eid") != eid:
            continue
        d["R_ci_m"] = float(new_R_ci)
        d["R_co_m"] = float(new_R_co)
        d["R_local_elec"] = [
            float(R_local_elec[i, j])
            for i in range(3) for j in range(3)
        ]
        # Per-type intraneural positioning. Cuff-style electrodes
        # don't need any extra xy because their patches sit on
        # the cuff inner cylinder by construction.
        _etype = str(d.get("electrode_type", "")).lower()
        if _etype.startswith("life ("):
            # LIFE array centre = nerve cross-section centroid
            # at z=0, OR the targeted fascicle's centroid when
            # `life_target_fascicle_idx >= 0` and that index
            # exists in the bundle.
            cx_mm, cy_mm = (
                _nerve_cx * 1e3, _nerve_cy * 1e3,
            )
            try:
                _tgt = int(d.get(
                    "life_target_fascicle_idx", -1,
                ))
            except (TypeError, ValueError):
                _tgt = -1
            if _tgt >= 0:
                _fcs = _bundle_fascicle_centroids_local()
                if 0 <= _tgt < len(_fcs):
                    cx_mm = _fcs[_tgt][0] * 1e3
                    cy_mm = _fcs[_tgt][1] * 1e3
            d["life_x_mm"] = float(cx_mm)
            d["life_y_mm"] = float(cy_mm)
        elif _etype.startswith("time ("):
            # TIME ribbon midpoint = nerve centroid (same as
            # LIFE). Chord orientation = principal direction of
            # the fascicle-centroid xy scatter when ≥ 2 are
            # present; otherwise 0° (ribbon along +x).
            d["time_x_mm"] = float(_nerve_cx * 1e3)
            d["time_y_mm"] = float(_nerve_cy * 1e3)
            phi_deg = 0.0
            _fcs = _bundle_fascicle_centroids_local()
            if len(_fcs) >= 2:
                _f = np.asarray(_fcs, dtype=np.float64)
                _cent = _f.mean(axis=0)
                _cov = np.cov(_f - _cent, rowvar=False)
                if _cov.shape == (2, 2):
                    _eigvals, _eigvecs = np.linalg.eigh(_cov)
                    _principal = _eigvecs[:, -1]
                    phi_deg = float(np.degrees(
                        np.arctan2(_principal[1], _principal[0]),
                    ))
                    # Normalise to [-90, 90] so phi=180 is shown
                    # as phi=0 (chord direction is unsigned).
                    if phi_deg > 90.0:
                        phi_deg -= 180.0
                    elif phi_deg < -90.0:
                        phi_deg += 180.0
            d["time_chord_phi_deg"] = phi_deg
        # If we just refit the FRAME anchor, mirror its rotation
        # + radius into geom's cached fit so the nerve render
        # frame stays consistent (the same bookkeeping the
        # legacy refit watcher did inline).
        if designs[0].get("eid") == eid:
            geom._R_local_cached = R_local_elec
            geom._R_ci_cached = float(new_R_ci)
        break
    state.designs = designs
    return True


def _design_M(design: dict) -> np.ndarray:
    """Build the design's PCA→cuff-local row-vector rotation. A
    PCA-frame row vector `r` maps to cuff-local via
    `r_local = (r - cuff_origin_pca) @ M`, where
    `M = R_local_elec.T @ Rx @ Ry @ Rz` composes the local-PCA
    refinement with the user's intrinsic Euler tilts. Falls back
    to identity when `R_local_elec` isn't set (design never
    refit)."""
    import math
    R_local_flat = design.get("R_local_elec")
    if R_local_flat is not None and len(R_local_flat) == 9:
        R_local_elec = np.asarray(
            R_local_flat, dtype=np.float64,
        ).reshape(3, 3)
    else:
        R_local_elec = np.eye(3, dtype=np.float64)
    M = R_local_elec.T.copy()
    rot_x_rad = math.radians(
        float(design.get("cuff_rot_x_deg", 0.0)),
    )
    rot_y_rad = math.radians(
        float(design.get("cuff_rot_y_deg", 0.0)),
    )
    rot_z_rad = math.radians(
        float(design.get("cuff_rot_z_deg", 0.0)),
    )
    if rot_x_rad:
        c, s = math.cos(rot_x_rad), math.sin(rot_x_rad)
        Rx_col = np.array(
            [[1.0, 0.0, 0.0],
             [0.0,  c,   -s],
             [0.0,  s,    c]],
            dtype=np.float64,
        )
        M = M @ Rx_col
    if rot_y_rad:
        c, s = math.cos(rot_y_rad), math.sin(rot_y_rad)
        Ry_col = np.array(
            [[ c,  0.0,  s],
             [0.0, 1.0, 0.0],
             [-s,  0.0,  c]],
            dtype=np.float64,
        )
        M = M @ Ry_col
    if rot_z_rad:
        c, s = math.cos(rot_z_rad), math.sin(rot_z_rad)
        Rz_col = np.array(
            [[ c,  -s, 0.0],
             [ s,   c, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        M = M @ Rz_col
    return M


def anchor_mesh_frame(
    pts_pca: np.ndarray,
    designs: list,
    cuff_anchor: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Define the SHARED mesh frame in which every design's mesh
    lives. The frame is the ANCHOR design's cuff-local frame —
    cuff at origin, +z = local nerve axis at the cuff. Every
    design's nerve PLC + muscle bbox lives in this frame; only
    the cuff pose differs per design.

    Returns `(anchor_origin_pca, anchor_M)` where:
      anchor_origin_pca (3,) — anchor cuff origin in PCA frame.
      anchor_M (3, 3)        — rotation such that a PCA row
        vector `r` maps to mesh-frame via
        `r_mesh = (r - anchor_origin_pca) @ anchor_M`.

    With this choice, the anchor design's cuff is automatically
    at origin axis-aligned in the mesh frame — i.e. the single-
    design case is bit-exact identical to the pre-F3.2 single-
    cuff path. Other designs sit at a non-trivial (offset, R)
    relative to the mesh frame, computed by `design_cuff_transform`.

    Falls back to identity (centroid at origin, no rotation) when
    no designs exist."""
    if not designs:
        return (
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
        )
    anchor = designs[0]
    anchor_origin_pca = find_cuff_origin_pca(
        pts_pca, cuff_anchor,
        float(anchor.get("cuff_offset_mm", 0.0)),
        float(anchor.get("cuff_dx_mm", 0.0)),
        float(anchor.get("cuff_dy_mm", 0.0)),
    )
    anchor_M = _design_M(anchor)
    return anchor_origin_pca, anchor_M


def design_cuff_transform(
    design: dict,
    pts_pca: np.ndarray,
    anchor_origin_pca: np.ndarray,
    anchor_M: np.ndarray,
    cuff_anchor: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (offset, R) describing where this design's cuff sits
    in the SHARED mesh frame (the anchor's cuff-local frame —
    see `anchor_mesh_frame`).

    Returns:
      offset (3,):   mesh-frame translation of the cuff's local
                     origin. A cuff-local point `p_local` (row) maps
                     to mesh-frame via
                     `p_mesh = p_local @ R + offset`.
      R (3, 3):      rotation that takes cuff-local row vectors to
                     mesh-frame row vectors via right-multiplication.

    The inverse is `p_local = (p_mesh - offset) @ R.T`.

    For the anchor design this returns (0, I) — the anchor cuff is
    at origin axis-aligned in the mesh frame, so the single-design
    case is bit-exact identical to the legacy single-cuff path."""
    offset_mm = float(design.get("cuff_offset_mm", 0.0))
    dx_mm = float(design.get("cuff_dx_mm", 0.0))
    dy_mm = float(design.get("cuff_dy_mm", 0.0))
    cuff_origin_pca = find_cuff_origin_pca(
        pts_pca, cuff_anchor, offset_mm, dx_mm, dy_mm,
    )
    M_D = _design_M(design)
    # Derivation (see F3.2d fix-log note in FEATURES.md):
    #   p_design_local = (p_pca - cuff_D_origin_pca) @ M_D       (D's local)
    #   p_mesh         = (p_pca - anchor_origin_pca) @ anchor_M  (mesh frame)
    # Eliminate p_pca:
    #   p_design_local = (p_mesh @ anchor_M.T + anchor_origin
    #                     - cuff_D_origin_pca) @ M_D
    # Solve for p_mesh:
    #   p_mesh = p_design_local @ R + offset
    # where:
    #   R       = M_D.T @ anchor_M
    #   offset  = (cuff_D_origin_pca - anchor_origin_pca) @ anchor_M
    R = M_D.T @ np.asarray(anchor_M, dtype=np.float64)
    offset = np.asarray(
        (cuff_origin_pca - anchor_origin_pca) @ anchor_M,
        dtype=np.float64,
    )
    return offset, R


def nerve_canonical_pts(
    pts_raw: np.ndarray,
    centroid: np.ndarray,
    R_global: np.ndarray,
    anchor_origin_pca: np.ndarray,
    anchor_M: "np.ndarray | None" = None,
) -> np.ndarray:
    """Express raw-frame nerve points in the SHARED MESH FRAME (the
    anchor's cuff-local frame; see `anchor_mesh_frame`). The
    `anchor_M` argument applies the anchor's PCA→local rotation
    after the PCA-centering. When `anchor_M` is omitted the
    function falls back to the pre-F3.2-fix "PCA minus anchor
    origin" frame (only used by legacy callers; new callers should
    always pass `anchor_M`)."""
    src = np.asarray(pts_raw, dtype=np.float64)
    if src.size == 0:
        return src
    centered_pca = (src - np.asarray(centroid, dtype=np.float64)) \
        @ np.asarray(R_global, dtype=np.float64) \
        - np.asarray(anchor_origin_pca, dtype=np.float64)
    if anchor_M is None:
        return centered_pca
    return centered_pca @ np.asarray(anchor_M, dtype=np.float64)


def anchor_origin_pca_for_designs(
    pts_pca: np.ndarray,
    designs: list,
    cuff_anchor: str,
) -> np.ndarray:
    """Return the anchor design's cuff origin in PCA frame.
    Convenience wrapper around `find_cuff_origin_pca`. Anchor =
    designs[0]; falls back to zero when no designs exist."""
    if not designs:
        return np.zeros(3, dtype=np.float64)
    anchor = designs[0]
    return find_cuff_origin_pca(
        pts_pca,
        cuff_anchor,
        float(anchor.get("cuff_offset_mm", 0.0)),
        float(anchor.get("cuff_dx_mm", 0.0)),
        float(anchor.get("cuff_dy_mm", 0.0)),
    )
