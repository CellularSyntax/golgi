# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Parameter sweep + threshold-finder driver (F2.1).

Two modes, one driver. Both modes are ASYNC (F2.1.c fixup): per-fiber
sims run in `loop.run_in_executor` so the main loop stays responsive
(WS heartbeat, busy-lightbox updates, cancel button). Between
(fiber, amplitude) cells the driver flushes a one-line progress
callback that the action handler routes to state.busy_log + the
terminal.

  * Recruitment-curve mode (`mode="recruitment"`) — sweep a list of
    stim amplitudes; for each (fiber, amplitude) cell, run the
    per-fiber sim and record whether ≥ 1 AP fires.

  * Threshold-finder mode (`mode="threshold"`) — per-fiber bisection
    over amplitude to find the minimum cath_amp_mA that activates
    the fiber. 5-10 sims per fiber typically; tolerance is µA.

Both modes reuse the existing per-fiber sim machinery in
`pipeline/fiber_sim.py` (`_fiber_preflight` + `_do_one_fiber`).
The activating criterion (F2.1 question 2) is the default: ≥ 1 AP
fires (`sim_data["n_real"] > 0`).

Cancellation: drivers poll `ctx.was_cancelled()` between (fiber,
amplitude) cells (recruitment) or between bisection steps
(threshold). Partial results are NOT persisted — a cancelled
sweep returns the cells completed so far but the caller is
responsible for discarding them.

Caching is owned by F2.1.d (`golgi/projects/sweep_cache.py` — not
yet implemented). The driver populates `SweepResult.sha` from the
request shape so the cache key is stable.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from typing import Callable, Optional

import numpy as np

from golgi.jobs.schemas import SweepRequest, SweepResult

from .context import PipelineContext
from .fiber_sim import (
    FiberSimJobRequest,
    _do_one_fiber,
    _fiber_preflight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scaled_pulse_params(
    base: dict, new_cath_mA: float,
) -> dict:
    """Return a copy of `base` with cath_amp_mA = new_cath_mA and
    anod_amp_mA scaled proportionally (so the cathode:anode ratio
    + sign convention is preserved). If the base has zero
    cath_amp_mA we just set the new value as-is and leave anod
    alone — the user is in an unusual config and shouldn't see
    surprise scaling."""
    out = copy.deepcopy(base)
    base_cath = float(base.get("cath_amp_mA", 0.0))
    if abs(base_cath) > 1e-12:
        scale = float(new_cath_mA) / base_cath
        out["cath_amp_mA"] = float(new_cath_mA)
        out["anod_amp_mA"] = float(
            base.get("anod_amp_mA", 0.0)
        ) * scale
    else:
        out["cath_amp_mA"] = float(new_cath_mA)
    return out


def _run_one_amp(
    req_template: FiberSimJobRequest,
    amp_mA: float,
) -> bool:
    """Run the per-fiber sim at the given amplitude. Returns True
    iff the fiber activated (≥ 1 AP fired anywhere). Uses the
    same _do_one_fiber dispatcher as the regular single-fiber sim."""
    pulse = _scaled_pulse_params(req_template.pulse_params, amp_mA)
    one_req = FiberSimJobRequest(
        sel=req_template.sel,
        s_um=req_template.s_um,
        Ve_mV=req_template.Ve_mV,
        diameter_um=req_template.diameter_um,
        length_um=req_template.length_um,
        pulse_params=pulse,
        backend=req_template.backend,
        model_name=req_template.model_name,
        helpers=req_template.helpers,
    )
    # _do_one_fiber returns JobOutputs; we just need n_real.
    out = _do_one_fiber(one_req, on_line=None, cancel=None)
    sim_data = out.outputs["sim_data"]
    return bool(int(sim_data.get("n_real", 0)) > 0)


def _bisect_threshold(
    req_template: FiberSimJobRequest,
    lo_mA: float, hi_mA: float, tol_uA: float,
    cancel_check=None,
) -> tuple[float, int]:
    """Per-fiber bisection. Returns (threshold_uA, iter_count).
    threshold_uA = np.nan if the fiber doesn't activate at
    `hi_mA` (no threshold inside the range).

    Algorithm:
      1. Activation probe at hi_mA — if it doesn't activate, return
         NaN. If it does, we have an upper bound.
      2. Activation probe at lo_mA — if it DOES activate, threshold
         is below `lo_mA`; return `lo_mA` as a conservative upper
         bound on the threshold (caller can widen the range).
      3. Bisect: while (hi - lo) > tol_uA: mid = (lo + hi) / 2;
         activate at mid → hi := mid else lo := mid.
    """
    tol_mA = float(tol_uA) * 1e-3
    n_iter = 0

    # Step 1: probe at hi.
    n_iter += 1
    if not _run_one_amp(req_template, hi_mA):
        return float("nan"), n_iter
    # Step 2: probe at lo.
    n_iter += 1
    if _run_one_amp(req_template, lo_mA):
        # Threshold is below `lo`; report `lo` as conservative
        # upper bound.
        return float(lo_mA * 1e3), n_iter

    # Step 3: bisect.
    while (hi_mA - lo_mA) > tol_mA:
        if cancel_check is not None and cancel_check():
            return float("nan"), n_iter
        mid = 0.5 * (lo_mA + hi_mA)
        n_iter += 1
        if _run_one_amp(req_template, mid):
            hi_mA = mid
        else:
            lo_mA = mid
    return float(hi_mA * 1e3), n_iter  # µA


def _select_fibers(
    geom, req: SweepRequest,
) -> list[int]:
    """Apply the request's fiber filters and return the explicit
    list of fiber indices to sweep. Order: explicit indices >
    branch_filter > fiber_type_filter > all."""
    n_total = int(len(geom.fiber_paths_Ve or []))
    if req.fiber_indices is not None:
        return [int(i) for i in req.fiber_indices
                if 0 <= int(i) < n_total]
    indices = list(range(n_total))
    if req.branch_filter is not None and geom.fiber_branch_idx is not None:
        bidx = np.asarray(geom.fiber_branch_idx)
        indices = [i for i in indices
                   if int(bidx[i]) == int(req.branch_filter)]
    if req.fiber_type_filter is not None and geom.fiber_pop_rows is not None:
        # fiber_pop_rows is parallel to fibers; entries are row-ids
        # (str). The filter compares against row label, which we
        # need to resolve via state.pop_row_meta — keep filter as
        # a row_id string here; caller is responsible for mapping
        # label → row_id before constructing the request.
        rows = list(geom.fiber_pop_rows)
        target = str(req.fiber_type_filter)
        indices = [i for i in indices
                   if i < len(rows) and str(rows[i]) == target]
    return indices


def _build_template(
    ctx: PipelineContext, sel: int, req: SweepRequest,
) -> tuple[Optional[FiberSimJobRequest], Optional[str]]:
    """Resolve (s_um, Ve_mV, length_um) for `sel` and bundle into a
    FiberSimJobRequest carrying the resolved diameter + model +
    backend + pulse params.

    Sources are coupled — model + backend + diameter must come
    from the same place, otherwise you get crashes like "Diameter
    for MRG_INTERPOLATION must be between 2 and 16 µm" when the
    Single-fiber tab's MRG_INTERPOLATION model is paired with a
    Population's per-fiber 0.8 µm C-fiber diameter.

    Per-fiber lookup (req.model_source == "population"):
      - model:    geom.fiber_pop_types[sel] (string from the
                  population row's `model` field)
      - backend:  state.pop_row_meta[fiber_pop_rows[sel]]["backend"]
      - diameter: geom.fiber_pop_diameters_um[sel]
      Falls back per-field to req.model_name / req.backend /
      state.fiber_diameter_um when a fiber has no population
      assignment (uncommon — usually means the user picked fibers
      that weren't in the population sample).

    Single-fiber mode (req.model_source == "single_fiber"):
      All three come from the Single-fiber tab: req.model_name,
      req.backend, state.fiber_diameter_um. Same model + same
      diameter for every swept fiber.

    Returns (template, None) on success or (None, err_msg) on
    failure (out-of-range index, length mismatch, etc.)."""
    geom = ctx.geom
    state = ctx.state
    data, err = _fiber_preflight(geom, sel)
    if err is not None:
        return None, err
    s_um, Ve_mV, length_um = data
    H = ctx.helpers

    # ---- Model + backend + diameter (coupled to model_source) ----
    model_name = str(req.model_name)
    backend = str(req.backend)
    diameter_um = float(state.fiber_diameter_um)

    if req.model_source == "population":
        # Per-fiber model from population assignment, if any.
        if (geom.fiber_pop_types is not None
                and sel < len(geom.fiber_pop_types)):
            per_model = str(geom.fiber_pop_types[sel] or "")
            if per_model:
                model_name = per_model
        # Per-fiber backend from the row's metadata.
        if (geom.fiber_pop_rows is not None
                and sel < len(geom.fiber_pop_rows)):
            row_id = str(geom.fiber_pop_rows[sel] or "")
            if row_id:
                row_meta = (state.pop_row_meta or {}).get(
                    row_id, {},
                )
                per_backend = str(
                    row_meta.get("backend") or "",
                )
                if per_backend:
                    backend = per_backend
        # Per-fiber diameter from population. Stays in sync with
        # the per-fiber model so we never pair MRG_INTERPOLATION
        # with a C-fiber's 0.8 µm sample.
        if (geom.fiber_pop_diameters_um is not None
                and sel < len(geom.fiber_pop_diameters_um)):
            per_diameter = float(
                geom.fiber_pop_diameters_um[sel],
            )
            if per_diameter > 0.0:
                diameter_um = per_diameter
    # else: single_fiber mode — keep all three from the Single-
    #       fiber tab (state.fiber_model + state.fiber_backend +
    #       state.fiber_diameter_um).

    template = FiberSimJobRequest(
        sel=int(sel),
        s_um=np.asarray(s_um, dtype=np.float64),
        Ve_mV=np.asarray(Ve_mV, dtype=np.float64),
        diameter_um=diameter_um,
        length_um=float(length_um),
        pulse_params=dict(req.pulse_params),
        backend=backend,
        model_name=model_name,
        helpers=H,
    )
    return template, None


def _sha_for_request(req: SweepRequest) -> str:
    """Stable hash of the request shape — used as the on-disk
    cache key. SHA-256 of the JSON-serialised request, hex.
    Truncated to 16 chars for readable filenames."""
    blob = json.dumps(
        req.serialize(), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


async def _run_recruitment_sweep(
    ctx: PipelineContext,
    req: SweepRequest,
    on_progress: Optional[Callable[[str], None]] = None,
) -> SweepResult:
    """Recruitment-curve mode (async). For each fiber × each
    amplitude in `req.amplitudes_mA`, run the per-fiber sim in an
    executor and record activation. Honors `ctx.was_cancelled()`
    between cells.

    Progress emission matches single-fiber / population preflight
    style: one header line per fiber with model + diameter, then a
    detail line per amplitude with ✓/· activation, then a per-fiber
    summary. Action handler routes header lines to busy_msg + log,
    detail lines to busy_log only."""
    geom = ctx.geom
    indices = _select_fibers(geom, req)
    n_fibers = len(indices)
    amps = list(req.amplitudes_mA)
    n_amps = len(amps)
    loop = asyncio.get_event_loop()

    t0 = time.perf_counter()
    activated = np.zeros((n_fibers, n_amps), dtype=np.bool_)
    diameters = np.zeros(n_fibers, dtype=np.float64)
    branch_idx_out = np.zeros(n_fibers, dtype=np.int32)
    type_labels: list[str] = []

    bidx_full = (
        np.asarray(geom.fiber_branch_idx)
        if geom.fiber_branch_idx is not None
        else np.zeros(len(geom.fiber_paths_Ve or []), dtype=np.int32)
    )
    rows_full = (
        list(geom.fiber_pop_rows)
        if geom.fiber_pop_rows is not None
        else []
    )

    if on_progress is not None:
        on_progress(
            f"selected {n_fibers} fiber"
            f"{'' if n_fibers == 1 else 's'} × "
            f"{n_amps} amplitude"
            f"{'' if n_amps == 1 else 's'} → "
            f"{n_fibers * n_amps} sims total"
        )

    n_sims_total = 0
    for fi, sel in enumerate(indices):
        if ctx.was_cancelled():
            break
        template, err = _build_template(ctx, sel, req)
        if err is not None:
            # Skip this fiber but keep the row in the result with
            # all-False activation + diameter 0; downstream figures
            # can choose to drop empty rows.
            diameters[fi] = 0.0
            branch_idx_out[fi] = int(bidx_full[sel])
            type_labels.append(
                str(rows_full[sel])
                if sel < len(rows_full) else ""
            )
            if on_progress is not None:
                on_progress(
                    f"[{fi + 1}/{n_fibers}] fiber {sel} — "
                    f"⚠ {err} — skipped"
                )
            continue
        diameters[fi] = template.diameter_um
        branch_idx_out[fi] = int(bidx_full[sel])
        type_labels.append(
            str(rows_full[sel])
            if sel < len(rows_full) else ""
        )
        # Per-fiber header — model + diameter inline so the user
        # knows exactly what is being simulated for each fiber.
        if on_progress is not None:
            on_progress(
                f"[{fi + 1}/{n_fibers}] fiber {sel} · "
                f"{template.model_name} ({template.backend}), "
                f"d = {template.diameter_um:.2f} µm · "
                f"sweeping {n_amps} amplitude"
                f"{'' if n_amps == 1 else 's'}"
            )
        # Per-amplitude sims — run each in the executor so the
        # main loop can flush state pushes + heartbeat WS.
        n_act_so_far = 0
        for ai, amp in enumerate(amps):
            if ctx.was_cancelled():
                break
            fired = await loop.run_in_executor(
                None, _run_one_amp, template, float(amp),
            )
            activated[fi, ai] = fired
            if fired:
                n_act_so_far += 1
            n_sims_total += 1
            if on_progress is not None:
                mark = "✓" if fired else "·"
                amp_uA = float(amp) * 1e3
                on_progress(
                    f"  {mark} {amp_uA:8.1f} µA → "
                    f"{'activated' if fired else 'no AP'} "
                    f"({n_act_so_far}/{ai + 1})"
                )
            # Yield each cell so flushes + WS heartbeats land.
            await asyncio.sleep(0)
        # Per-fiber summary.
        if on_progress is not None:
            on_progress(
                f"  → {n_act_so_far}/{n_amps} activated"
            )

    elapsed = time.perf_counter() - t0
    return SweepResult(
        request=req,
        fiber_indices=np.asarray(indices, dtype=np.int64),
        fiber_diameters_um=diameters,
        fiber_branch_idx=branch_idx_out,
        fiber_type_labels=type_labels,
        activated=activated,
        elapsed_s=float(elapsed),
        sha=_sha_for_request(req),
        n_sims_total=int(n_sims_total),
    )


async def _run_threshold_finder(
    ctx: PipelineContext,
    req: SweepRequest,
    on_progress: Optional[Callable[[str], None]] = None,
) -> SweepResult:
    """Threshold-finder mode (async). Per-fiber bisection between
    `req.bisect_lo_mA` and `req.bisect_hi_mA`. Each probe runs in
    the executor; cancellation polled between probes.

    Progress emission mirrors the recruitment mode + the single-
    fiber preflight format: header line per fiber with model +
    diameter, then a detail line per bisection probe."""
    geom = ctx.geom
    indices = _select_fibers(geom, req)
    n_fibers = len(indices)
    loop = asyncio.get_event_loop()

    t0 = time.perf_counter()
    thresholds = np.full(n_fibers, np.nan, dtype=np.float64)
    iters = np.zeros(n_fibers, dtype=np.int32)
    diameters = np.zeros(n_fibers, dtype=np.float64)
    branch_idx_out = np.zeros(n_fibers, dtype=np.int32)
    type_labels: list[str] = []

    bidx_full = (
        np.asarray(geom.fiber_branch_idx)
        if geom.fiber_branch_idx is not None
        else np.zeros(len(geom.fiber_paths_Ve or []), dtype=np.int32)
    )
    rows_full = (
        list(geom.fiber_pop_rows)
        if geom.fiber_pop_rows is not None
        else []
    )

    if on_progress is not None:
        on_progress(
            f"bisecting {n_fibers} fiber"
            f"{'' if n_fibers == 1 else 's'} "
            f"in [{req.bisect_lo_mA:.3f}, {req.bisect_hi_mA:.3f}] mA"
            f" to ±{req.bisect_tol_uA:.0f} µA"
        )

    n_sims_total = 0
    for fi, sel in enumerate(indices):
        if ctx.was_cancelled():
            break
        template, err = _build_template(ctx, sel, req)
        if err is not None:
            diameters[fi] = 0.0
            branch_idx_out[fi] = int(bidx_full[sel])
            type_labels.append(
                str(rows_full[sel])
                if sel < len(rows_full) else ""
            )
            if on_progress is not None:
                on_progress(
                    f"[{fi + 1}/{n_fibers}] fiber {sel} — "
                    f"⚠ {err} — skipped"
                )
            continue
        diameters[fi] = template.diameter_um
        branch_idx_out[fi] = int(bidx_full[sel])
        type_labels.append(
            str(rows_full[sel])
            if sel < len(rows_full) else ""
        )
        # Per-fiber header (model + diameter, same shape as the
        # recruitment mode for visual consistency).
        if on_progress is not None:
            on_progress(
                f"[{fi + 1}/{n_fibers}] fiber {sel} · "
                f"{template.model_name} ({template.backend}), "
                f"d = {template.diameter_um:.2f} µm · "
                f"bisecting [{req.bisect_lo_mA:.3f}, "
                f"{req.bisect_hi_mA:.3f}] mA"
            )
        # Bisection — each probe is awaited via the executor and
        # emits a detail line. The window shrinks toward the
        # threshold so the user can see convergence live.
        thr_uA, n = await _bisect_threshold_async(
            template,
            float(req.bisect_lo_mA),
            float(req.bisect_hi_mA),
            float(req.bisect_tol_uA),
            cancel_check=ctx.was_cancelled,
            loop=loop,
            on_probe=on_progress,
        )
        thresholds[fi] = thr_uA
        iters[fi] = int(n)
        n_sims_total += int(n)
        if on_progress is not None:
            if np.isfinite(thr_uA):
                on_progress(
                    f"  → threshold = {thr_uA:.1f} µA "
                    f"({n} sims)"
                )
            else:
                on_progress(
                    f"  → no activation in range "
                    f"({n} sims)"
                )
        await asyncio.sleep(0)

    elapsed = time.perf_counter() - t0
    return SweepResult(
        request=req,
        fiber_indices=np.asarray(indices, dtype=np.int64),
        fiber_diameters_um=diameters,
        fiber_branch_idx=branch_idx_out,
        fiber_type_labels=type_labels,
        thresholds_uA=thresholds,
        bisect_iters=iters,
        elapsed_s=float(elapsed),
        sha=_sha_for_request(req),
        n_sims_total=int(n_sims_total),
    )


async def _bisect_threshold_async(
    template: FiberSimJobRequest,
    lo_mA: float, hi_mA: float, tol_uA: float,
    cancel_check=None,
    loop=None,
    on_probe: Optional[Callable[[str], None]] = None,
) -> tuple[float, int]:
    """Async version of _bisect_threshold — each probe runs in the
    executor so the loop stays responsive between iterations. When
    `on_probe` is given, emit a detail line per probe so the user
    sees the bisection window narrow live."""
    if loop is None:
        loop = asyncio.get_event_loop()
    tol_mA = float(tol_uA) * 1e-3
    n_iter = 0

    # Probe at hi.
    n_iter += 1
    hi_activates = await loop.run_in_executor(
        None, _run_one_amp, template, hi_mA,
    )
    if on_probe is not None:
        mark = "✓" if hi_activates else "·"
        on_probe(
            f"  {mark} probe hi = {hi_mA * 1e3:8.1f} µA → "
            f"{'activated' if hi_activates else 'no AP'}"
        )
    if not hi_activates:
        return float("nan"), n_iter
    # Probe at lo.
    n_iter += 1
    lo_activates = await loop.run_in_executor(
        None, _run_one_amp, template, lo_mA,
    )
    if on_probe is not None:
        mark = "✓" if lo_activates else "·"
        on_probe(
            f"  {mark} probe lo = {lo_mA * 1e3:8.1f} µA → "
            f"{'activated' if lo_activates else 'no AP'}"
        )
    if lo_activates:
        return float(lo_mA * 1e3), n_iter

    while (hi_mA - lo_mA) > tol_mA:
        if cancel_check is not None and cancel_check():
            return float("nan"), n_iter
        mid = 0.5 * (lo_mA + hi_mA)
        n_iter += 1
        mid_activates = await loop.run_in_executor(
            None, _run_one_amp, template, mid,
        )
        if mid_activates:
            hi_mA = mid
        else:
            lo_mA = mid
        if on_probe is not None:
            mark = "✓" if mid_activates else "·"
            on_probe(
                f"  {mark} iter {n_iter - 2:2d}: "
                f"{mid * 1e3:8.1f} µA → "
                f"{'activated' if mid_activates else 'no AP'} "
                f"· window [{lo_mA * 1e3:.1f}, "
                f"{hi_mA * 1e3:.1f}] µA"
            )
    return float(hi_mA * 1e3), n_iter


async def run_sweep(
    ctx: PipelineContext,
    req: SweepRequest,
    on_progress: Optional[Callable[[str], None]] = None,
) -> SweepResult:
    """Public entry point (async) — dispatch on `req.mode`. Each
    per-fiber sim runs in `loop.run_in_executor`; the driver awaits
    between cells so state pushes + WS heartbeat keep landing on
    the client during multi-minute sweeps.

    `on_progress(line)` is invoked once per fiber (after its sweep
    or bisection completes); the caller wires it to state.busy_log
    + a terminal print so the user sees live progress."""
    if req.mode == "recruitment":
        return await _run_recruitment_sweep(ctx, req, on_progress)
    if req.mode == "threshold":
        return await _run_threshold_finder(ctx, req, on_progress)
    raise ValueError(
        f"unknown sweep mode {req.mode!r} — expected "
        "'recruitment' or 'threshold'"
    )
