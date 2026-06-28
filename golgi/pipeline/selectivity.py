# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""F3.2 — selectivity post-processing on top of F2.1 SweepResults.

Pure functions over numpy arrays; no I/O, no state. The Compare
panel loads one `SweepResult` per config via
`projects.sweep_cache.load_latest_for_config(cid)`, then hands the
arrays to the functions below to produce:

  * Per-branch recruitment curves (fraction activated vs amplitude).
  * Veraart selectivity index per amplitude:
        SI = (R_target − R_offtarget) / (R_target + R_offtarget)
    Range [−1, +1].  +1 ⇒ only target activates;
                       0  ⇒ target and off-target activate equally;
                      −1 ⇒ only off-target activates.
  * Threshold ratio (high = good):
        median(threshold_offtarget) / median(threshold_target)
    > 1 means the target activates at a lower amplitude than the
    off-target population (a desirable design property).
  * Spatial activation min-amplitude per fiber (for the heatmap;
    builders elsewhere render this as a scatter coloured by amp).

The functions accept the raw numpy arrays carried inside a
`SweepResult` so the math module stays free of the dataclass
import and is unit-testable without spinning up the schema
machinery.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


__all__ = [
    "branch_ids_present",
    "compute_branch_recruitment",
    "compute_veraart_si",
    "compute_threshold_ratio",
    "compute_threshold_stats_per_branch",
    "compute_min_activation_per_fiber",
]


def branch_ids_present(branch_idx: np.ndarray) -> list[int]:
    """Sorted unique branch ids actually present in this sweep's
    fiber set. The Compare panel uses this to populate the target-
    branch picker so the user can only pick branches that exist
    in the loaded result(s)."""
    arr = np.asarray(branch_idx, dtype=np.int64)
    if arr.size == 0:
        return []
    return sorted({int(b) for b in np.unique(arr)})


def compute_branch_recruitment(
    activated: np.ndarray,         # (n_fibers, n_amps) bool
    branch_idx: np.ndarray,        # (n_fibers,) int
    branch_ids: Iterable[int] | None = None,
) -> dict[int, np.ndarray]:
    """Per-branch fraction-activated curves. Returns a dict
    {branch_id: (n_amps,) float in [0, 1]}.

    When `branch_ids` is None, every unique branch in
    `branch_idx` gets its own curve. Pass an explicit subset to
    pin the iteration order or to skip branches with no fibers.

    Recruitment-mode sweep results only. `activated` is the
    `SweepResult.activated` array; values must be boolean.
    """
    act = np.asarray(activated, dtype=np.bool_)
    bidx = np.asarray(branch_idx, dtype=np.int64)
    if act.ndim != 2:
        raise ValueError(
            f"activated must be 2D (n_fibers, n_amps), "
            f"got shape {act.shape}"
        )
    if act.shape[0] != bidx.shape[0]:
        raise ValueError(
            f"activated n_fibers ({act.shape[0]}) does not match "
            f"branch_idx length ({bidx.shape[0]})"
        )
    ids = (
        list(branch_ids) if branch_ids is not None
        else branch_ids_present(bidx)
    )
    out: dict[int, np.ndarray] = {}
    for b in ids:
        mask = bidx == int(b)
        if not mask.any():
            out[int(b)] = np.zeros(act.shape[1], dtype=np.float64)
            continue
        out[int(b)] = (
            act[mask].mean(axis=0).astype(np.float64)
        )
    return out


def _coalesce_offtarget_recruitment(
    per_branch: dict[int, np.ndarray],
    target_branch: int,
    offtarget_branches: Iterable[int] | None,
    bidx: np.ndarray | None = None,
    activated: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the off-target recruitment curve as the fraction
    activated across the UNION of off-target branches' fibers
    (not the mean of per-branch fractions — that's wrong when
    branches differ in fiber count).

    When `bidx` + `activated` are provided, compute from the raw
    arrays (correct weighting). Otherwise fall back to averaging
    the per-branch fractions (approximation, used when only the
    pre-aggregated dict is available).
    """
    if target_branch not in per_branch:
        # Target absent → no meaningful SI; return zeros.
        if not per_branch:
            return np.zeros(0, dtype=np.float64)
        return np.zeros_like(
            next(iter(per_branch.values())),
        )
    if offtarget_branches is None:
        off_ids = [
            b for b in per_branch if b != target_branch
        ]
    else:
        off_ids = [
            int(b) for b in offtarget_branches
            if int(b) in per_branch and int(b) != target_branch
        ]
    if not off_ids:
        return np.zeros_like(per_branch[target_branch])
    if bidx is not None and activated is not None:
        act = np.asarray(activated, dtype=np.bool_)
        bi = np.asarray(bidx, dtype=np.int64)
        mask = np.isin(bi, off_ids)
        if not mask.any():
            return np.zeros_like(per_branch[target_branch])
        return act[mask].mean(axis=0).astype(np.float64)
    # Fallback: per-branch-fraction mean (unweighted).
    stacks = np.stack([per_branch[b] for b in off_ids], axis=0)
    return stacks.mean(axis=0).astype(np.float64)


def compute_veraart_si(
    activated: np.ndarray,         # (n_fibers, n_amps) bool
    branch_idx: np.ndarray,        # (n_fibers,) int
    target_branch: int,
    offtarget_branches: Iterable[int] | None = None,
) -> np.ndarray:
    """Veraart selectivity index per amplitude:
        SI = (R_target − R_offtarget) / (R_target + R_offtarget)

    Returns a (n_amps,) array in [−1, +1]. When the denominator
    is zero at some amplitude (no fibers from either group
    activated yet) the SI at that amplitude is 0 — neutral, not
    NaN, so it plots cleanly against amplitude. Higher = better
    target selectivity.
    """
    per_branch = compute_branch_recruitment(
        activated, branch_idx,
        branch_ids=branch_ids_present(branch_idx),
    )
    if target_branch not in per_branch:
        n_amps = (
            int(per_branch[next(iter(per_branch))].shape[0])
            if per_branch else 0
        )
        return np.zeros(n_amps, dtype=np.float64)
    R_t = per_branch[int(target_branch)]
    R_o = _coalesce_offtarget_recruitment(
        per_branch, int(target_branch), offtarget_branches,
        bidx=branch_idx, activated=activated,
    )
    denom = R_t + R_o
    si = np.zeros_like(R_t)
    nz = denom > 0
    si[nz] = (R_t[nz] - R_o[nz]) / denom[nz]
    return si


def compute_threshold_stats_per_branch(
    thresholds_uA: np.ndarray,    # (n_fibers,) NaN where no spike
    branch_idx: np.ndarray,        # (n_fibers,) int
    branch_ids: Iterable[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Per-branch summary statistics over the bisected thresholds.

    Returns {branch_id: {"median": …, "mean": …, "min": …,
    "max": …, "n_activated": …, "n_total": …}}. NaN entries
    (fibers that never activated inside the bisection window) are
    excluded from the stats but counted in `n_total`.
    """
    thr = np.asarray(thresholds_uA, dtype=np.float64)
    bidx = np.asarray(branch_idx, dtype=np.int64)
    if thr.shape[0] != bidx.shape[0]:
        raise ValueError(
            f"thresholds length ({thr.shape[0]}) does not match "
            f"branch_idx length ({bidx.shape[0]})"
        )
    ids = (
        list(branch_ids) if branch_ids is not None
        else branch_ids_present(bidx)
    )
    out: dict[int, dict[str, float]] = {}
    for b in ids:
        mask = bidx == int(b)
        n_total = int(mask.sum())
        if n_total == 0:
            out[int(b)] = {
                "median": float("nan"),
                "mean": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n_activated": 0,
                "n_total": 0,
            }
            continue
        sub = thr[mask]
        good = np.isfinite(sub)
        n_act = int(good.sum())
        if n_act == 0:
            out[int(b)] = {
                "median": float("nan"),
                "mean": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n_activated": 0,
                "n_total": n_total,
            }
            continue
        vals = sub[good]
        out[int(b)] = {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_activated": n_act,
            "n_total": n_total,
        }
    return out


def compute_threshold_ratio(
    thresholds_uA: np.ndarray,
    branch_idx: np.ndarray,
    target_branch: int,
    offtarget_branches: Iterable[int] | None = None,
) -> float:
    """median(off-target threshold) / median(target threshold).

    Returns NaN when the target branch has no activated fibers
    (median undefined) and +inf when the off-target population
    contains no activated fibers (the design perfectly avoids
    them within the bisection window). Higher = better.
    """
    stats = compute_threshold_stats_per_branch(
        thresholds_uA, branch_idx,
    )
    if target_branch not in stats:
        return float("nan")
    t_med = stats[int(target_branch)]["median"]
    if not np.isfinite(t_med) or t_med <= 0:
        return float("nan")
    if offtarget_branches is None:
        off_ids = [
            b for b in stats if b != int(target_branch)
        ]
    else:
        off_ids = [
            int(b) for b in offtarget_branches
            if int(b) in stats and int(b) != int(target_branch)
        ]
    if not off_ids:
        return float("inf")
    # Pool the off-target thresholds across branches before
    # taking the median (matches "single off-target population"
    # semantics rather than averaging per-branch medians).
    thr = np.asarray(thresholds_uA, dtype=np.float64)
    bidx = np.asarray(branch_idx, dtype=np.int64)
    pool_mask = np.isin(bidx, off_ids)
    pool = thr[pool_mask]
    good = np.isfinite(pool)
    if not good.any():
        return float("inf")
    o_med = float(np.median(pool[good]))
    return o_med / t_med


def compute_min_activation_per_fiber(
    activated: np.ndarray,       # (n_fibers, n_amps) bool
    amplitudes_mA: np.ndarray,    # (n_amps,) float
) -> np.ndarray:
    """For each fiber, the lowest amplitude at which it
    activated. NaN where the fiber never activated. Used by the
    spatial activation heatmap to colour each fiber's cross-
    section position by its activation threshold (visualises
    which fascicles activate first as amplitude ramps up).
    """
    act = np.asarray(activated, dtype=np.bool_)
    amps = np.asarray(amplitudes_mA, dtype=np.float64)
    if act.ndim != 2:
        raise ValueError(
            f"activated must be 2D (n_fibers, n_amps), "
            f"got shape {act.shape}"
        )
    if act.shape[1] != amps.shape[0]:
        raise ValueError(
            f"activated n_amps ({act.shape[1]}) does not match "
            f"amplitudes length ({amps.shape[0]})"
        )
    n_fibers = act.shape[0]
    out = np.full(n_fibers, np.nan, dtype=np.float64)
    for f in range(n_fibers):
        idxs = np.where(act[f])[0]
        if idxs.size:
            out[f] = amps[int(idxs.min())]
    return out
