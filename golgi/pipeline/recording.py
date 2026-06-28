# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""R1.3 — cNAP forward sum.

Per fiber:
  I_m(node, t) ≈ (π d² / 4 ρ_a Δx) × ∂²V_m/∂s²   (cable equation,
                                                    discrete 2nd
                                                    difference at
                                                    nodes)
  φ_montage(t) = Σ_nodes I_m(node, t) × (V_e^R+(s_n) − V_e^R−(s_n))

Per population:
  φ_pop(t) = Σ_fibers φ_fiber(t)

Pure numpy. No Trame. No I/O beyond the recording-basis loader at
the bottom. Unit-testable as a black box against analytic cases —
see `tests/test_recording_cable.py`.

Conventions:
  * V_m in mV (as fiber backends emit), V_e^R in Volts (lead field
    per unit A, since solve_nerve.py injects 1 A).
  * φ output in Volts. The figures layer rescales to µV for plots.
  * Arc length in metres internally; node_s_um in µm for the
    callsite contract.
  * Lead field is loaded as V_e per Ampere — already correct
    because solve_nerve.py uses I_TEST = 1.0 A. So multiplying by
    I_m (in Amperes) gives Volts directly with no extra factor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# Axoplasmic resistivity (Ω·m). MRG/CRRSS literature: 70 Ω·cm =
# 0.7 Ω·m. Used by the cable-equation second-difference to convert
# V_m curvature into transmembrane current at each node.
RHO_A_OHM_M = 0.7


# ---------------------------------------------------------------
# Cable equation: V_m → I_m
# ---------------------------------------------------------------


def im_from_vm(
    vm_mV: np.ndarray,
    node_s_um: np.ndarray,
    diameter_um: float,
    rho_a_ohm_m: float = RHO_A_OHM_M,
) -> np.ndarray:
    """Transmembrane current at each node, every timestep.

    Inputs:
      vm_mV       — (n_nodes, n_t) in mV (as emitted by axonml +
                    pyfibers backends).
      node_s_um   — (n_nodes,) arc length along the fiber (µm).
      diameter_um — fiber diameter (µm).
      rho_a_ohm_m — axoplasmic resistivity (Ω·m). Default 0.7
                    (= 70 Ω·cm, standard MRG value).

    Returns I_m in Amperes per node (signed; outward current
    at the firing node is negative in the Plonsey convention,
    inward = positive).

    Formula:
        I_m(n, t) = (π d² / 4 ρ_a Δx) · (V_m(n+1) − 2 V_m(n)
                                          + V_m(n−1))(t)

    Boundary handling: end nodes (n=0 and n=n_nodes-1) are
    clamped to zero. For MRG these are the "silenced" tail nodes
    pyfibers/axonml insert as artificial Dirichlet pins; they
    don't fire APs in practice. For unmyelinated fibers the
    same clamp is a small underestimate of edge contributions
    (negligible because end nodes are typically far from the
    cuff).

    Robustness: `node_s_um` is treated as approximately uniform
    via `np.median(np.diff(s))`. axonml's internodal spacing is
    exactly uniform; pyfibers may have small drift in the tail
    nodes that we deliberately ignore (the median is the right
    typical spacing for the active interior).
    """
    vm_arr = np.asarray(vm_mV, dtype=np.float64)
    if vm_arr.ndim != 2:
        raise ValueError(
            f"vm_mV must be 2-D (n_nodes, n_t); got shape "
            f"{vm_arr.shape}",
        )
    n_nodes, _n_t = vm_arr.shape
    s_m = np.asarray(node_s_um, dtype=np.float64) * 1.0e-6
    if s_m.shape != (n_nodes,):
        raise ValueError(
            f"node_s_um length {s_m.size} doesn't match vm "
            f"n_nodes {n_nodes}",
        )
    if n_nodes < 3:
        return np.zeros_like(vm_arr)
    diffs = np.diff(s_m)
    if diffs.size == 0 or float(np.median(diffs)) <= 0.0:
        return np.zeros_like(vm_arr)
    dx_m = float(np.median(diffs))
    d_m = float(diameter_um) * 1.0e-6
    coef = (np.pi * d_m * d_m) / (4.0 * float(rho_a_ohm_m) * dx_m)

    # mV → V (the coef expects V_m in Volts; output is Amperes).
    vm_V = vm_arr * 1.0e-3
    out = np.zeros_like(vm_arr)
    out[1:-1, :] = coef * (
        vm_V[2:, :] - 2.0 * vm_V[1:-1, :] + vm_V[:-2, :]
    )
    return out


# ---------------------------------------------------------------
# Single-fiber cNAP contribution
# ---------------------------------------------------------------


def _polyline_arc_length_m(poly_xyz_m: np.ndarray) -> np.ndarray:
    """Cumulative arc length (m) at each polyline vertex. The
    first point is at s=0, last point at total length."""
    pts = np.asarray(poly_xyz_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        return np.zeros(pts.shape[0] if pts.ndim == 2 else 0)
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])


def compute_cnap_single(
    *,
    vm_mV: np.ndarray,                 # (n_nodes, n_t) mV
    t_ms: np.ndarray,                  # (n_t,) ms
    node_s_um: np.ndarray,             # (n_nodes,) µm
    diameter_um: float,
    fiber_poly_xyz_m: np.ndarray,      # (n_poly, 3) cuff-frame m
    Ve_rec_plus_poly_V: np.ndarray,    # (n_poly,) V per A injected
    Ve_rec_minus_poly_V: np.ndarray,   # (n_poly,) V per A injected
    rho_a_ohm_m: float = RHO_A_OHM_M,
) -> dict:
    """One fiber's contribution to the cNAP under a bipolar
    montage.

    The lead field is sampled on the fiber's POLYLINE (the same
    points solve_nerve.py wrote to V_e_rec_<id>.npz). We
    interpolate it to the MRG NODE positions, multiply
    elementwise by the cable-equation I_m at each node, and sum.

    Returns:
        {
          "t_ms":  (n_t,) — same time base as the fiber sim,
          "phi_V": (n_t,) — recording trace in Volts.
        }
    """
    im_A = im_from_vm(
        vm_mV=vm_mV,
        node_s_um=node_s_um,
        diameter_um=diameter_um,
        rho_a_ohm_m=rho_a_ohm_m,
    )
    poly_s_m = _polyline_arc_length_m(fiber_poly_xyz_m)
    node_s_m = np.asarray(node_s_um, dtype=np.float64) * 1.0e-6

    # Interpolate lead field along arc length onto node positions.
    # Numpy's interp clamps OOB to the endpoints — a node that
    # falls slightly beyond the polyline (axonml + Dirichlet tail
    # nodes can land at exactly L) gets the end-point value, not
    # NaN. Fine for an integral-style sum.
    plus_at_nodes = np.interp(
        node_s_m, poly_s_m, np.asarray(Ve_rec_plus_poly_V),
    )
    minus_at_nodes = np.interp(
        node_s_m, poly_s_m, np.asarray(Ve_rec_minus_poly_V),
    )
    diff = plus_at_nodes - minus_at_nodes        # (n_nodes,) V/A

    # φ(t) = Σ_n I_m(n, t) * (V_e^R+ − V_e^R−)(n).
    # I_m has units of A (per node); lead-field diff is V (per A).
    # Their product is V — what the recording amplifier sees.
    phi_t = diff @ im_A                          # (n_t,)
    return {
        "t_ms": np.asarray(t_ms, dtype=np.float64).copy(),
        "phi_V": phi_t.astype(np.float64),
    }


# ---------------------------------------------------------------
# Population cNAP — sum across fibers + per-type decomposition
# ---------------------------------------------------------------


def compute_cnap_population(
    *,
    per_fiber_phi_V: dict,             # {fiber_idx: (n_t,) Volts}
    per_fiber_type: Optional[dict] = None,
    # {fiber_idx: type_label}
) -> dict:
    """Sum per-fiber cNAP contributions into a population trace,
    optionally decomposed by fiber-type label.

    Inputs:
      per_fiber_phi_V — dict mapping fiber index → phi(t) Volts.
        ALL fibers must share the same time base; the caller
        should compute_cnap_single all of them against the same
        montage with the same t.
      per_fiber_type  — optional dict fiber_idx → type label. When
        given, the output adds a 'phi_by_type' dict.

    Returns:
        {
          "phi_total_V": (n_t,),
          "phi_by_type": {label: (n_t,)}  # only if types given
        }
    """
    if not per_fiber_phi_V:
        return {"phi_total_V": np.zeros(0, dtype=np.float64)}

    # Determine the shared time-axis length.
    n_t = max(int(np.asarray(p).size) for p in per_fiber_phi_V.values())
    if n_t <= 0:
        return {"phi_total_V": np.zeros(0, dtype=np.float64)}
    total = np.zeros(n_t, dtype=np.float64)
    by_type: dict[str, np.ndarray] = {}
    for idx, phi in per_fiber_phi_V.items():
        phi_arr = np.asarray(phi, dtype=np.float64)
        if phi_arr.size != n_t:
            # Skip mismatched (shouldn't happen if caller is honest).
            continue
        total += phi_arr
        if per_fiber_type is not None:
            label = str(per_fiber_type.get(int(idx), "unknown"))
            if label not in by_type:
                by_type[label] = np.zeros(n_t, dtype=np.float64)
            by_type[label] += phi_arr
    out: dict = {"phi_total_V": total}
    if per_fiber_type is not None:
        out["phi_by_type"] = by_type
    return out


# ---------------------------------------------------------------
# Recording-basis loader
# ---------------------------------------------------------------


def load_recording_basis(
    design_dir: Path,
    montages: list[dict],
) -> Optional[dict]:
    """Load the per-design lead-field arrays for a list of
    recording montages.

    Returns:
        {
          mid: {
            "paths_flat":   (N_pts, 3) m,
            "path_lengths": (n_fibers,) ints,
            "plus_flat":    (N_pts,) V per A,
            "minus_flat":   (N_pts,) V per A,
          },
          ...
        }
    or None if no recording dir is present or every montage's
    files are missing. Per-montage missing-file cases are skipped
    silently (and absent from the returned dict).
    """
    if not montages:
        return None
    rec_dir = Path(design_dir) / "recording"
    if not rec_dir.is_dir():
        return None
    out: dict = {}
    for m in montages:
        try:
            mid = str(m.get("mid", ""))
            plus_id = int(m.get("plus_contact", -1))
            minus_id = int(m.get("minus_contact", -1))
        except (TypeError, ValueError):
            continue
        if not mid or plus_id < 0 or minus_id < 0:
            continue
        plus_path = rec_dir / f"V_e_rec_{plus_id}.npz"
        minus_path = rec_dir / f"V_e_rec_{minus_id}.npz"
        if not plus_path.is_file() or not minus_path.is_file():
            continue
        try:
            pd = np.load(plus_path, allow_pickle=False)
            md = np.load(minus_path, allow_pickle=False)
        except Exception:                                    # noqa: BLE001
            continue
        paths_flat = np.asarray(pd["paths_flat"], dtype=np.float64)
        path_lengths = np.asarray(pd["path_lengths"], dtype=np.int64)
        plus_flat = np.asarray(pd["Ve_flat"], dtype=np.float64)
        minus_flat = np.asarray(md["Ve_flat"], dtype=np.float64)
        # Tolerate a path-length mismatch between + and − (would
        # indicate the recording basis was solved against
        # different fiber sets — should never happen in practice).
        if plus_flat.size != minus_flat.size:
            continue
        if plus_flat.size != paths_flat.shape[0]:
            continue
        out[mid] = {
            "paths_flat": paths_flat,
            "path_lengths": path_lengths,
            "plus_flat": plus_flat,
            "minus_flat": minus_flat,
        }
    return out if out else None


def fiber_polyline_slice(
    paths_flat: np.ndarray,
    path_lengths: np.ndarray,
    fiber_index: int,
) -> tuple[Optional[np.ndarray], Optional[tuple[int, int]]]:
    """Return (poly_xyz, (start, end)) for one fiber inside the
    flat all-fibers polyline array.

    Returns (None, None) if the fiber index is out of range.
    The polyline coordinates are whatever was saved by
    solve_nerve.py — cuff-frame metres."""
    lens = np.asarray(path_lengths, dtype=np.int64)
    if fiber_index < 0 or fiber_index >= lens.size:
        return None, None
    offsets = np.concatenate([[0], np.cumsum(lens)])
    a = int(offsets[fiber_index])
    b = int(offsets[fiber_index + 1])
    return paths_flat[a:b], (a, b)
