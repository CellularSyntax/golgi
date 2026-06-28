# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Single-fiber simulation pipeline driver.

Different from mesh/fem/fibers in that there's no subprocess:
the per-fiber sim runs in-process via pyfibers/axonml inside
loop.run_in_executor. Uses InProcessRunner so the JobRunner
shape stays uniform — swapping to a subprocess (or remote
service) later is a one-liner.

Flow (do_run_fiber_sim):
  1. Coalesce + validate `state.fiber_sel_indices` (fall back
     to single-pick when empty).
  2. Per-fiber preflight (arclength + Vₑ resampling, length
     sanity check).
  3. Per-fiber sim via InProcessRunner → executor → axonml /
     pyfibers backend dispatcher.
  4. Collect sim_data per fiber on geom.fiber_sim_results +
     per-fiber meta on state.fiber_sim_results_meta.
  5. Pick the first successful fiber as the initial view, push
     its propagation + waterfall figures into the corresponding
     state vars.
  6. _save_fiber_sim_cache so the user gets the plots back on
     project reopen.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from golgi.figures.fiber import (
    _build_fiber_propagation_figure,
    _build_fiber_waterfall_figure,
)
from golgi.figures.recording import build_fiber_cnap_figure
from golgi.jobs import (
    CancelToken, InProcessRunner, JobOutputs, JobRequest,
)
from .context import PipelineContext
from .fem_layout import design_dir as design_dir_fn
from .recording import (
    compute_cnap_single,
    fiber_polyline_slice,
    load_recording_basis,
)


@dataclass
class FiberSimJobRequest(JobRequest):
    """One per fiber. The per-fiber runner reads everything it
    needs off this dataclass and returns a sim_data dict +
    summary string."""
    sel: int
    s_um: np.ndarray
    Ve_mV: np.ndarray
    diameter_um: float
    length_um: float
    pulse_params: dict
    backend: str          # "axonml" | "pyfibers"
    model_name: str       # e.g., "MRG_INTERPOLATION"
    # Helpers bag for the runner. SimpleNamespace from the
    # PipelineContext; carried per-request so the runner stays
    # context-free.
    helpers: Any = None


def _fiber_preflight(geom, sel: int):
    """Resolve a single fiber's (s_um, Ve_mV, length_um) triple
    from geom; returns (data, error_msg). On success `data` is
    the tuple, on failure `error_msg` is a user-facing string
    suitable for the status banner."""
    af_paths = (geom.fiber_paths_for_Ve
                if geom.fiber_paths_for_Ve is not None
                else geom.fiber_paths_raw)
    if sel < 0 or sel >= len(geom.fiber_paths_Ve):
        return None, (
            f"fiber index {sel} out of range "
            f"(0..{len(geom.fiber_paths_Ve) - 1})"
        )
    path_xyz = np.asarray(af_paths[sel], dtype=np.float64)
    if path_xyz.shape[0] < 2:
        return None, (
            f"fiber {sel} has fewer than 2 points"
        )
    ds = np.linalg.norm(np.diff(path_xyz, axis=0), axis=1)
    s_m = np.concatenate([[0.0], np.cumsum(ds)])
    Ve_V = np.asarray(
        geom.fiber_paths_Ve[sel], dtype=np.float64,
    )
    if len(Ve_V) != len(s_m):
        return None, (
            f"length mismatch on fiber {sel}: "
            f"V_e has {len(Ve_V)} samples, path "
            f"has {len(s_m)} — re-run FEM"
        )
    s_um = (s_m - s_m.min()) * 1e6
    Ve_mV = Ve_V * 1e3
    length_um = float(s_um.max())
    return (s_um, Ve_mV, length_um), None


def _run_axonml_branch(req: FiberSimJobRequest):
    """axonml MRG-surrogate path. axonml is GPU-friendly; falls
    back to CPU when CUDA / MPS are missing (with a printed
    warning). Wave is built on the dt grid and handed to
    axonml_run_single via the `wave_mA=` mode."""
    H = req.helpers
    p = req.pulse_params
    dt_ms = 0.005
    n_t = int(round(p["tstop"] / dt_ms)) + 1
    t_grid = np.arange(n_t, dtype=np.float64) * dt_ms
    wave_mA = H.build_pulse_waveform(
        t_grid, p["t0"],
        p["cath_amp_mA"], p["cath_pw_ms"], p["gap_ms"],
        p["anod_amp_mA"], p["anod_pw_ms"], p["anode_first"],
    )
    res = H.axonml_run_single(
        s_um=req.s_um, Ve_mV=req.Ve_mV,
        diameter_um=req.diameter_um,
        amp_uA=0.0, pulse_shape="", pw_us=0.0,
        t0_ms=p["t0"], tstop_ms=p["tstop"], dt_ms=dt_ms,
        wave_mA=wave_mA,
    )
    vm = res["vm"]
    vm_peak = np.nanmax(vm, axis=1)
    real_ap = (res["spike_n"] > 0) & (vm_peak > 0.0)
    spike_t_real = np.where(real_ap, res["spike_t_ms"], np.nan)
    n_real = int(real_ap.sum())
    sim_data = {
        "vm": vm,
        "t": res["time_ms"],
        "node_z_um": res["node_z_um"],
        "spike_t": spike_t_real,
        "vm_peak": vm_peak,
        "n_real": n_real,
        "n_thresh": int((res["spike_n"] > 0).sum()),
        "n_nodes": int(vm.shape[0]),
        "model": "MRG (axonml)",
        "diameter": req.diameter_um,
        "stim_kind": p["kind"],
        "stim_amp_mA": p["cath_amp_mA"],
        "stim_anod_mA": p["anod_amp_mA"],
        "stim_cath_pw_ms": p["cath_pw_ms"],
        "stim_anod_pw_ms": p["anod_pw_ms"],
        "stim_gap_ms": p["gap_ms"],
        "stim_anode_first": p["anode_first"],
        "stim_onset_ms": p["t0"],
        "tstop_ms": p["tstop"],
        "fiber_index": req.sel,
        "source_label": f"trajectory (fiber {req.sel})",
        "wave_t": res["wave_v"],
        "Ve_at_nodes_mV": res["Ve_node_mV"],
    }
    summary = (
        f"axonml (MRG surrogate), d = {req.diameter_um:.2f} µm, "
        f"max V_m peak = {float(vm_peak.max()):+.1f} mV · "
        f"{n_real}/{vm.shape[0]} nodes fired a real AP"
        f"  · fiber {req.sel}"
    )
    return sim_data, summary


def _run_pyfibers_branch(req: FiberSimJobRequest):
    """pyfibers (NEURON) path. Imports lazy so the app doesn't
    pay NEURON's HOC-init cost when nobody runs a sim."""
    H = req.helpers
    p = req.pulse_params
    try:
        import warnings as _w
        _w.filterwarnings("ignore")
        from pyfibers import (
            build_fiber as _build_fiber,
            FiberModel as _FiberModel,
        )
        from pyfibers.stimulation import (
            ScaledStim as _ScaledStim,
        )
        from scipy.interpolate import interp1d as _interp1d
    except ImportError as ex:
        raise RuntimeError(
            f"pyfibers / NEURON not available: {ex}. "
            "Install with: pip install pyfibers neuron"
        ) from ex
    try:
        fiber_model_attr = getattr(_FiberModel, req.model_name)
    except AttributeError as ex:
        raise RuntimeError(
            f"unknown pyfibers FiberModel '{req.model_name}': "
            f"{ex}. Valid names: "
            f"{', '.join(H.MYELINATED_MODELS + H.UNMYELINATED_MODELS)}"
        ) from ex
    fiber = _build_fiber(
        fiber_model_attr,
        diameter=req.diameter_um, length=req.length_um,
    )
    # M46 — Pad the extracellular potential by a small margin
    # at each end (edge-clamped extrapolation). MRG fiber
    # compartment placement puts the first node center at ~half
    # a node + half a paranode (~5 µm) past the fiber's z=0,
    # and the last node typically a similar amount past
    # `length`. With `center=False`, pyfibers' resample
    # validator (pyfibers/fiber.py:661) rejects the fiber when
    # any sample point falls outside `[0, length]` of the
    # supplied potential coordinates, even by a few microns
    # ("Potential coordinates must span the fiber coordinates").
    # 20 µm margin comfortably covers every MRG / SMALL_MRG
    # template; the extrapolation is constant (edge value) so
    # the physically-relevant interior samples are unaffected.
    _pad_um = 20.0
    _s = np.asarray(req.s_um, dtype=np.float64)
    _V = np.asarray(req.Ve_mV, dtype=np.float64)
    _s_pad = np.concatenate([
        [_s[0] - _pad_um], _s, [_s[-1] + _pad_um],
    ])
    _V_pad = np.concatenate([[_V[0]], _V, [_V[-1]]])
    fiber.resample_potentials(
        potentials=_V_pad, potential_coords=_s_pad,
        center=False, inplace=True,
    )
    tp, vp = H.build_pulse_breakpoints(
        p["t0"], p["cath_amp_mA"], p["cath_pw_ms"], p["gap_ms"],
        p["anod_amp_mA"], p["anod_pw_ms"], p["anode_first"],
        p["tstop"],
    )
    wave = _interp1d(
        tp, vp, kind="previous",
        bounds_error=False, fill_value=0.0,
    )
    stim = _ScaledStim(
        waveform=wave, dt=0.005, tstop=p["tstop"],
    )
    fiber.record_vm()
    fiber.apcounts(thresh=-30)
    n_ap, t_last = stim.run_sim(
        1.0, fiber, fail_on_end_excitation=False,
    )
    n_ap = 0 if n_ap is None else int(n_ap)
    vm = np.asarray(fiber.vm)
    tvec = np.asarray(fiber.time)
    node_z = np.linspace(0.0, fiber.length, len(fiber.nodes))
    spike_n = np.array(
        [apc.n for apc in fiber.apc], dtype=float,
    )
    spike_t = np.array(
        [float(apc.time) for apc in fiber.apc],
    )
    spike_t = np.where(spike_n > 0, spike_t, np.nan)
    vm_peak = np.nanmax(vm, axis=1)
    real_ap = (spike_n > 0) & (vm_peak > 0.0)
    spike_t_real = np.where(real_ap, spike_t, np.nan)
    n_real = int(real_ap.sum())
    wave_t = np.asarray([wave(float(tt)) for tt in tvec])
    Ve_at_nodes_mV = np.interp(node_z, req.s_um, req.Ve_mV)
    sim_data = {
        "vm": vm,
        "t": tvec,
        "node_z_um": node_z,
        "spike_t": spike_t_real,
        "vm_peak": vm_peak,
        "n_real": n_real,
        "n_thresh": int((spike_n > 0).sum()),
        "n_nodes": int(vm.shape[0]),
        "model": req.model_name,
        "diameter": req.diameter_um,
        "stim_kind": p["kind"],
        "stim_amp_mA": p["cath_amp_mA"],
        "stim_anod_mA": p["anod_amp_mA"],
        "stim_cath_pw_ms": p["cath_pw_ms"],
        "stim_anod_pw_ms": p["anod_pw_ms"],
        "stim_gap_ms": p["gap_ms"],
        "stim_anode_first": p["anode_first"],
        "stim_onset_ms": p["t0"],
        "tstop_ms": p["tstop"],
        "fiber_index": req.sel,
        "source_label": f"trajectory (fiber {req.sel})",
        "wave_t": wave_t,
        "Ve_at_nodes_mV": Ve_at_nodes_mV,
    }
    summary = (
        f"{req.model_name}, d = {req.diameter_um:.2f} µm, "
        f"max V_m peak = {float(vm_peak.max()):+.1f} mV · "
        f"{n_real}/{vm.shape[0]} nodes fired a real AP"
        f"  · fiber {req.sel}"
    )
    return sim_data, summary


def _do_one_fiber(req: FiberSimJobRequest, on_line, cancel):
    """InProcessRunner.fn — dispatch to the backend, wrap the
    (sim_data, summary) tuple as JobOutputs.outputs."""
    if req.backend == "axonml":
        sim_data, summary = _run_axonml_branch(req)
    else:
        sim_data, summary = _run_pyfibers_branch(req)
    return JobOutputs(
        return_code=0,
        outputs={"sim_data": sim_data, "summary": summary},
    )


# ---------------------------------------------------------------
# R1.4 — cNAP wiring helpers (shared with pipeline/pop_sim.py via
# the lookup helper). Live at module level so both single-fiber
# and population drivers can call them; the population variant
# loops over fibers and folds into compute_cnap_population.
# ---------------------------------------------------------------


def resolve_active_recording_context(ctx) -> tuple:
    """Return (active_design_dir, active_montages, basis_or_None).

    The "active" design is whichever the analysis tab is showing
    (state.active_design_id, set after a FEM solve), with
    state.selected_design_id as a fallback for the pre-FEM state
    where a user is iterating on cuff geometry. Montages come
    from the active config (state.active_config_id) — that's the
    wiring whose FEM outputs the panels are rendering.

    Returns (None, [], None) when any prerequisite is missing
    (no active project, no active design, no recording dir on
    disk). Callers should treat that as "no cNAP to compute".
    """
    state = ctx.state
    H = ctx.helpers
    try:
        out_dir = H.active_project().out_dir
    except Exception:                                        # noqa: BLE001
        return None, [], None
    if not out_dir:
        return None, [], None
    eid = str(
        state.active_design_id or state.selected_design_id or "",
    )
    if not eid:
        return None, [], None
    d_dir = design_dir_fn(out_dir, eid)
    if not d_dir or not d_dir.is_dir():
        return None, [], None
    # Find the active config's montages.
    cid = str(
        state.active_config_id or state.selected_config_id or "",
    )
    montages: list[dict] = []
    if cid:
        for c in (state.configs or []):
            if str(c.get("cid", "")) == cid:
                montages = [
                    dict(m) for m in
                    (c.get("recording_montages") or [])
                ]
                break
    if not montages:
        return d_dir, [], None
    basis = load_recording_basis(d_dir, montages)
    return d_dir, montages, basis


def _compute_and_set_fiber_cnap(
    ctx, sim_data: dict, fiber_idx: int,
) -> None:
    """Compute the single-fiber cNAP contribution at every
    montage attached to the active config and push the figure to
    state.fiber_cnap_figure. Logs each gating decision to stdout
    under the `[cnap]` prefix so we can diagnose why a panel is
    empty without instrumenting the user's browser."""
    import numpy as _np
    state = ctx.state
    geom = ctx.geom
    _d_dir, montages, basis = resolve_active_recording_context(ctx)
    print(
        f"[cnap] single-fiber start: fiber_idx={fiber_idx}, "
        f"design_dir={_d_dir}, n_montages={len(montages)}, "
        f"basis_keys={list(basis.keys()) if basis else None}",
        flush=True,
    )
    if not montages:
        msg = (
            "No recording montages on the active config — "
            "add one in the cuff drawer and re-run the FEM solve."
        )
        state.fiber_cnap_status = msg
        state.fiber_cnap_figure = build_fiber_cnap_figure(
            cnap_by_montage={}, montage_meta=[],
            active_mid="", fiber_label="",
        )
        print(f"[cnap] single-fiber skip: {msg}", flush=True)
        return
    if basis is None or not basis:
        msg = (
            "Recording lead fields missing on disk — re-run "
            "the FEM solve to populate designs/<id>/recording/."
        )
        state.fiber_cnap_status = msg
        state.fiber_cnap_figure = build_fiber_cnap_figure(
            cnap_by_montage={}, montage_meta=montages,
            active_mid="", fiber_label="",
        )
        print(f"[cnap] single-fiber skip: {msg}", flush=True)
        return
    # Fiber polyline (3D cuff-frame metres) for this fiber.
    poly_xyz = None
    if (geom.fiber_paths_for_Ve is not None
            and 0 <= fiber_idx < len(geom.fiber_paths_for_Ve)):
        poly_xyz = _np.asarray(
            geom.fiber_paths_for_Ve[fiber_idx], dtype=_np.float64,
        )
    if poly_xyz is None or poly_xyz.size == 0:
        msg = (
            "Fiber polyline coords missing — re-run the FEM "
            "solve."
        )
        state.fiber_cnap_status = msg
        state.fiber_cnap_figure = build_fiber_cnap_figure(
            cnap_by_montage={}, montage_meta=montages,
            active_mid="", fiber_label="",
        )
        print(f"[cnap] single-fiber skip: {msg}", flush=True)
        return

    vm_mV = _np.asarray(sim_data.get("vm"), dtype=_np.float64)
    t_ms = _np.asarray(sim_data.get("t", []), dtype=_np.float64)
    node_s_um = _np.asarray(
        sim_data.get("node_z_um", []), dtype=_np.float64,
    )
    diameter_um = float(sim_data.get("diameter", 0.0))
    if vm_mV.size == 0 or t_ms.size == 0 or node_s_um.size == 0:
        msg = "Incomplete sim_data — no cNAP."
        state.fiber_cnap_status = msg
        state.fiber_cnap_figure = build_fiber_cnap_figure(
            cnap_by_montage={}, montage_meta=montages,
            active_mid="", fiber_label="",
        )
        print(f"[cnap] single-fiber skip: {msg}", flush=True)
        return

    cnap_by_montage: dict = {}
    for m in montages:
        mid = str(m.get("mid", ""))
        if mid not in basis:
            print(
                f"[cnap] montage {mid!r}: no lead-field files "
                f"on disk — skipping",
                flush=True,
            )
            continue
        b = basis[mid]
        _slice, offsets = fiber_polyline_slice(
            b["paths_flat"], b["path_lengths"], fiber_idx,
        )
        if offsets is None:
            print(
                f"[cnap] montage {mid!r}: fiber_idx="
                f"{fiber_idx} out of range for path_lengths "
                f"(n={b['path_lengths'].size})",
                flush=True,
            )
            continue
        a, b_end = offsets
        plus_poly = b["plus_flat"][a:b_end]
        minus_poly = b["minus_flat"][a:b_end]
        n_nan_plus = int(_np.isnan(plus_poly).sum())
        n_nan_minus = int(_np.isnan(minus_poly).sum())
        if n_nan_plus or n_nan_minus:
            print(
                f"[cnap] montage {mid!r}: lead field has NaN "
                f"(plus={n_nan_plus}, minus={n_nan_minus} of "
                f"{plus_poly.size}) — these came from "
                f"polyline points outside the FEM mesh. "
                f"Replacing with 0 so the trace renders.",
                flush=True,
            )
            plus_poly = _np.nan_to_num(plus_poly, nan=0.0)
            minus_poly = _np.nan_to_num(minus_poly, nan=0.0)
        try:
            out = compute_cnap_single(
                vm_mV=vm_mV,
                t_ms=t_ms,
                node_s_um=node_s_um,
                diameter_um=diameter_um,
                fiber_poly_xyz_m=poly_xyz,
                Ve_rec_plus_poly_V=plus_poly,
                Ve_rec_minus_poly_V=minus_poly,
            )
        except Exception as ex:                              # noqa: BLE001
            print(
                f"[cnap] montage {mid!r}: compute_cnap_single "
                f"raised {type(ex).__name__}: {ex}",
                flush=True,
            )
            continue
        phi_peak_uV = float(
            _np.max(_np.abs(out["phi_V"])) * 1e6,
        )
        print(
            f"[cnap] montage {mid!r}: ok, phi_peak={phi_peak_uV:.3f} µV "
            f"(n_nodes={vm_mV.shape[0]}, "
            f"d={diameter_um:.2f} µm, poly_n={poly_xyz.shape[0]})",
            flush=True,
        )
        cnap_by_montage[mid] = out

    geom.cnap_single = cnap_by_montage

    if not cnap_by_montage:
        msg = (
            "Recording basis on disk but no montage matched "
            "this design — re-run the FEM solve."
        )
        state.fiber_cnap_status = msg
        state.fiber_cnap_figure = build_fiber_cnap_figure(
            cnap_by_montage={}, montage_meta=montages,
            active_mid="", fiber_label="",
        )
        print(f"[cnap] single-fiber skip: {msg}", flush=True)
        return

    # Pick the active montage (preserve prior choice if still
    # present, else first available).
    prev = str(state.active_montage_single or "")
    if prev not in cnap_by_montage:
        state.active_montage_single = next(iter(cnap_by_montage))

    fiber_label = ""
    try:
        H = ctx.helpers
        lab, _color = H.fiber_label_and_color(int(fiber_idx))
        fiber_label = str(lab)
    except Exception:                                        # noqa: BLE001
        pass

    state.fiber_cnap_figure = build_fiber_cnap_figure(
        cnap_by_montage=cnap_by_montage,
        montage_meta=montages,
        active_mid=str(state.active_montage_single or ""),
        fiber_label=fiber_label,
    )
    state.fiber_cnap_status = (
        f"✓ cNAP computed for {len(cnap_by_montage)} montage"
        f"{'s' if len(cnap_by_montage) > 1 else ''}"
    )
    try:
        state.flush()
    except Exception:                                        # noqa: BLE001
        pass
    print(
        f"[cnap] single-fiber done: pushed figure to state "
        f"for {len(cnap_by_montage)} montage(s)",
        flush=True,
    )


async def run_fiber_sim(ctx: PipelineContext) -> None:
    """Full single-fiber sim driver. See module docstring."""
    state = ctx.state
    geom = ctx.geom
    H = ctx.helpers

    # Preconditions.
    if (geom.fiber_paths_raw is None
            or len(geom.fiber_paths_raw) == 0):
        state.fiber_sim_status = (
            "⚠ generate fibers first (Fibers tab → Generate)."
        )
        return
    if geom.fiber_paths_Ve is None:
        state.fiber_sim_status = (
            "⚠ run the FEM solve first — per-fiber V_e is "
            "produced by Solve."
        )
        return
    # Coalesce + validate the multi-pick set; fall back to the
    # single viewed index when nothing's selected.
    raw_sel: list[int] = []
    for v in (state.fiber_sel_indices or []):
        try:
            vi = int(v)
        except (TypeError, ValueError):
            continue
        if (0 <= vi < len(geom.fiber_paths_Ve)
                and vi not in raw_sel):
            raw_sel.append(vi)
    if not raw_sel:
        single = int(state.fiber_sel_idx)
        if 0 <= single < len(geom.fiber_paths_Ve):
            raw_sel = [single]
    if not raw_sel:
        state.fiber_sim_status = (
            "⚠ no fibers selected — pick at least one "
            "trajectory in the combobox."
        )
        return
    backend = str(state.fiber_backend)
    model_name = str(state.fiber_model)
    d_um = float(state.fiber_diameter_um)
    p = H.fiber_pulse_params()

    n_total = len(raw_sel)
    state.fiber_sim_busy = True
    state.fiber_sim_failed = False
    state.fiber_sim_status = (
        f"running {backend} ({model_name}, d = {d_um:.2f} µm) "
        f"on {n_total} fiber{'s' if n_total > 1 else ''} …"
    )
    state.fiber_sim_log = ""
    state.busy = True
    state.busy_msg = (
        f"Simulating {n_total} fiber"
        f"{'s' if n_total > 1 else ''} · {backend} "
        f"({model_name}, d = {d_um:.2f} µm)"
    )
    state.busy_log = ""
    state.flush()
    log_lines: list[str] = []

    def _log(line: str) -> None:
        line = ctx.stamp_user_line(line)
        log_lines.append(line)
        print(f"[fiber] {line}", flush=True)
        state.fiber_sim_log = "\n".join(log_lines[-40:])
        state.busy_log = "\n".join(log_lines[-10:])
        state.flush()

    loop = asyncio.get_event_loop()
    runner = InProcessRunner(_do_one_fiber)
    tok = CancelToken()

    # Per-fiber results: idx → sim_data dict.
    results_by_idx: dict[int, dict] = {}
    results_meta: list[dict] = []
    n_ok = 0
    n_fail = 0
    for ci, sel in enumerate(raw_sel, start=1):
        _log(f"[{ci}/{n_total}] fiber {sel} — preflight")
        state.busy_msg = (
            f"Simulating fiber {sel} "
            f"({ci}/{n_total}) · {backend}"
        )
        state.flush()
        label, color = H.fiber_label_and_color(sel)
        triple, err = _fiber_preflight(geom, sel)
        if err is not None:
            _log(f"  ⚠ skipped: {err}")
            results_meta.append({
                "idx": int(sel), "label": label, "color": color,
                "ok": False, "summary": f"⚠ {err}",
            })
            n_fail += 1
            continue
        s_um, Ve_mV, length_um = triple
        if length_um < 5_000.0:
            msg = (
                f"fiber {sel} too short "
                f"({length_um * 1e-3:.2f} mm)"
            )
            _log(f"  ⚠ skipped: {msg}")
            results_meta.append({
                "idx": int(sel), "label": label, "color": color,
                "ok": False, "summary": f"⚠ {msg}",
            })
            n_fail += 1
            continue
        req = FiberSimJobRequest(
            sel=sel, s_um=s_um, Ve_mV=Ve_mV,
            diameter_um=d_um, length_um=length_um,
            pulse_params=p, backend=backend,
            model_name=model_name, helpers=H,
        )
        try:
            outputs = await loop.run_in_executor(
                None, lambda r=req: runner.run(r, _log, tok),
            )
        except Exception as ex:
            _log(f"  ⚠ {type(ex).__name__}: {ex}")
            results_meta.append({
                "idx": int(sel), "label": label, "color": color,
                "ok": False,
                "summary": f"⚠ {type(ex).__name__}: {ex}",
            })
            n_fail += 1
            continue
        if outputs.return_code != 0:
            _log(f"  ⚠ runner returned rc={outputs.return_code}")
            results_meta.append({
                "idx": int(sel), "label": label, "color": color,
                "ok": False,
                "summary": f"⚠ runner rc={outputs.return_code}",
            })
            n_fail += 1
            continue
        sim_data = outputs.outputs["sim_data"]
        summary = outputs.outputs["summary"]
        results_by_idx[int(sel)] = sim_data
        results_meta.append({
            "idx": int(sel), "label": label, "color": color,
            "ok": True, "summary": summary,
        })
        n_ok += 1
        _log(f"  ✓ {summary}")

    # Persist results + populate the viewed-fiber plots.
    geom.fiber_sim_results = results_by_idx
    state.fiber_sim_results_meta = results_meta
    # Pick the first SUCCESSFUL fiber as the initial view.
    first_ok = next(
        (m["idx"] for m in results_meta if m["ok"]),
        None,
    )
    if first_ok is not None:
        state.fiber_sel_idx = int(first_ok)
        sim_data = results_by_idx[int(first_ok)]
        geom.fiber_sim_data = sim_data
        geom.fiber_sim_summary = next(
            (m["summary"] for m in results_meta
             if m["idx"] == first_ok),
            "",
        )
        state.fiber_sim_summary = geom.fiber_sim_summary
        state.fiber_propagation_figure = (
            _build_fiber_propagation_figure(sim_data)
        )
        state.fiber_waterfall_figure = (
            _build_fiber_waterfall_figure(sim_data)
        )
        # R1.4 — single-fiber cNAP contribution. Pulls the active
        # design's recording basis from disk, computes φ(t) per
        # bipolar montage attached to the active config, stores
        # per-montage on geom.cnap_single, and builds the figure
        # for the panel.
        try:
            _compute_and_set_fiber_cnap(
                ctx, sim_data, int(first_ok),
            )
        except Exception as _cnap_ex:                    # noqa: BLE001
            print(
                f"[fiber] cNAP compute failed: "
                f"{type(_cnap_ex).__name__}: {_cnap_ex}",
                flush=True,
            )
            state.fiber_cnap_status = (
                f"⚠ {type(_cnap_ex).__name__}: {_cnap_ex}"
            )
            state.fiber_cnap_figure = {"data": [], "layout": {}}
        # F3.2-M2.1d — gate the EXPORT navbar tab on at least
        # one nerve-sim being done. has_fiber_sim flips true
        # the first time a single-fiber sim succeeds; the
        # population pipeline sets has_pop_sim on its own
        # success path. State.* (not geom.*) because the JS
        # legend / navbar can only read state vars.
        state.has_fiber_sim = True
    state.fiber_sim_busy = False
    state.busy = False
    state.busy_msg = ""
    state.busy_log = ""
    if n_ok == 0:
        state.fiber_sim_failed = True
        state.fiber_sim_status = (
            f"⚠ all {n_total} fiber sims failed — see log."
        )
    elif n_fail > 0:
        state.fiber_sim_status = (
            f"✓ {n_ok}/{n_total} fiber sims completed "
            f"({n_fail} failed — see log)"
        )
    else:
        state.fiber_sim_status = (
            f"✓ {n_ok}/{n_total} fiber sims completed."
        )
    # Persist the batch to disk so the user gets the same plots
    # back when they reopen the project.
    H.save_fiber_sim_cache()
    state.flush()
