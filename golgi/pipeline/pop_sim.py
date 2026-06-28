# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Population-tab pipeline driver.

Two stages:
  run_pop_generate(ctx) — sample (type, diameter) per fiber from
      the per-branch mixture, write into geom.fiber_pop_*, build
      the KDE + cross-section-at-cuff figures, refresh viewport
      colouring, autosave pop_state. No subprocess; pure CPU.

  run_pop_sim(ctx) — for each fiber assigned a row in
      run_pop_generate, run the single-fiber sim on the
      FEM-derived Vₑ(s). Reuses pipeline.fiber_sim's per-fiber
      machinery (_fiber_preflight, _do_one_fiber,
      FiberSimJobRequest) via InProcessRunner — same pattern as
      run_fiber_sim, but the outer loop iterates fibers selected
      by population assignment instead of by user pick.

Audit event stays at the population level (one row per call to
each function), not per fiber.
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from golgi.figures.fiber import (
    _build_fiber_propagation_figure,
    _build_fiber_waterfall_figure,
)
from golgi.figures.population import (
    _build_pop_kde_figure,
    _build_pop_xsec_at_cuff_figure,
    _build_pop_xsec_figure,
)
from golgi.figures.recording import build_pop_cnap_figure
from golgi.jobs import CancelToken, InProcessRunner
from .context import PipelineContext
from .fiber_sim import (
    FiberSimJobRequest,
    _do_one_fiber,
    _fiber_preflight,
    resolve_active_recording_context,
)
from .recording import (
    compute_cnap_population,
    compute_cnap_single,
    fiber_polyline_slice,
)


# ---------------------------------------------------------------
# R1.4 — population cNAP. Mirrors _compute_and_set_fiber_cnap
# from pipeline/fiber_sim.py but loops over EVERY successful fiber
# per montage, then folds into compute_cnap_population for the
# total trace + per-fiber-type stack.
# ---------------------------------------------------------------


def _compute_and_set_pop_cnap(ctx, results: dict) -> None:
    """Build geom.cnap_pop = {mid: {t_ms, phi_total_V,
    phi_by_type, peak_latencies_ms}} and push the figure to
    state.pop_cnap_figure. `results` is the same per-fiber dict
    pop_sim already populates (idx → sim_data). Logs each gating
    decision under the `[cnap]` prefix so the user can diagnose
    an empty panel."""
    state = ctx.state
    geom = ctx.geom
    _d_dir, montages, basis = resolve_active_recording_context(ctx)
    print(
        f"[cnap] population start: n_fibers="
        f"{len(results) if results else 0}, "
        f"design_dir={_d_dir}, n_montages={len(montages)}, "
        f"basis_keys={list(basis.keys()) if basis else None}",
        flush=True,
    )
    if not montages:
        msg = (
            "No recording montages on the active config — "
            "add one in the cuff drawer and re-run FEM + "
            "population sim."
        )
        state.pop_cnap_status = msg
        state.pop_cnap_figure = build_pop_cnap_figure(
            cnap_by_montage={}, montage_meta=[],
            active_mid="", decompose_by_type=False,
        )
        print(f"[cnap] population skip: {msg}", flush=True)
        return
    if basis is None or not basis:
        msg = (
            "Recording lead fields missing on disk — re-run "
            "the FEM solve."
        )
        state.pop_cnap_status = msg
        state.pop_cnap_figure = build_pop_cnap_figure(
            cnap_by_montage={}, montage_meta=montages,
            active_mid="", decompose_by_type=False,
        )
        print(f"[cnap] population skip: {msg}", flush=True)
        return
    if not results:
        msg = "No successful fiber sims."
        state.pop_cnap_status = msg
        state.pop_cnap_figure = build_pop_cnap_figure(
            cnap_by_montage={}, montage_meta=montages,
            active_mid="", decompose_by_type=False,
        )
        print(f"[cnap] population skip: {msg}", flush=True)
        return

    # Per-fiber type labels (geom.fiber_pop_types is a numpy
    # object array indexed by fiber index — empty string for
    # unassigned fibers).
    pop_types = getattr(geom, "fiber_pop_types", None)
    per_fiber_type: dict[int, str] = {}
    if pop_types is not None:
        for i in range(len(pop_types)):
            label = str(pop_types[i] or "")
            if label:
                per_fiber_type[i] = label

    # Polyline coords per fiber (cuff-frame metres). Indexed by
    # fiber idx; same source as paths_Ve.npz.
    fiber_polys = geom.fiber_paths_for_Ve

    cnap_by_montage: dict = {}
    for m in montages:
        mid = str(m.get("mid", ""))
        if mid not in basis:
            print(
                f"[cnap] montage {mid!r}: no lead-field "
                f"files on disk — skipping",
                flush=True,
            )
            continue
        b = basis[mid]
        plus_flat = b["plus_flat"]
        minus_flat = b["minus_flat"]
        # NaN-clean: lead-field points that fell outside the
        # mesh come back as NaN; let the trace still render.
        n_nan_plus = int(np.isnan(plus_flat).sum())
        n_nan_minus = int(np.isnan(minus_flat).sum())
        if n_nan_plus or n_nan_minus:
            print(
                f"[cnap] montage {mid!r}: lead field has "
                f"NaN (plus={n_nan_plus}, "
                f"minus={n_nan_minus} of {plus_flat.size}) "
                f"— replacing with 0",
                flush=True,
            )
            plus_flat = np.nan_to_num(plus_flat, nan=0.0)
            minus_flat = np.nan_to_num(minus_flat, nan=0.0)
        paths_flat = b["paths_flat"]
        path_lengths = b["path_lengths"]
        per_fiber_phi: dict[int, np.ndarray] = {}
        t_ms_shared: np.ndarray | None = None
        for idx, sim_data in results.items():
            if (fiber_polys is None
                    or idx < 0
                    or idx >= len(fiber_polys)):
                continue
            poly_xyz = np.asarray(
                fiber_polys[idx], dtype=np.float64,
            )
            if poly_xyz.size == 0:
                continue
            _slice, offsets = fiber_polyline_slice(
                paths_flat, path_lengths, idx,
            )
            if offsets is None:
                continue
            a, b_end = offsets
            vm_mV = np.asarray(
                sim_data.get("vm"), dtype=np.float64,
            )
            t_ms = np.asarray(
                sim_data.get("t", []), dtype=np.float64,
            )
            node_s_um = np.asarray(
                sim_data.get("node_z_um", []),
                dtype=np.float64,
            )
            diameter_um = float(sim_data.get("diameter", 0.0))
            if vm_mV.size == 0 or node_s_um.size == 0:
                continue
            if t_ms_shared is None:
                t_ms_shared = t_ms
            try:
                out = compute_cnap_single(
                    vm_mV=vm_mV,
                    t_ms=t_ms,
                    node_s_um=node_s_um,
                    diameter_um=diameter_um,
                    fiber_poly_xyz_m=poly_xyz,
                    Ve_rec_plus_poly_V=plus_flat[a:b_end],
                    Ve_rec_minus_poly_V=minus_flat[a:b_end],
                )
            except Exception:                                # noqa: BLE001
                continue
            per_fiber_phi[int(idx)] = out["phi_V"]
        if not per_fiber_phi:
            continue
        pop_out = compute_cnap_population(
            per_fiber_phi_V=per_fiber_phi,
            per_fiber_type=per_fiber_type or None,
        )
        # Peak latencies per type (absolute-max time).
        peak_latencies: dict[str, float] = {}
        for tlabel, phi in (
            pop_out.get("phi_by_type", {}) or {}
        ).items():
            phi_arr = np.asarray(phi, dtype=np.float64)
            if (phi_arr.size == 0
                    or t_ms_shared is None
                    or t_ms_shared.size != phi_arr.size):
                continue
            i_peak = int(np.argmax(np.abs(phi_arr)))
            peak_latencies[tlabel] = float(t_ms_shared[i_peak])
        cnap_by_montage[mid] = {
            "t_ms": (
                t_ms_shared
                if t_ms_shared is not None
                else np.zeros(0)
            ),
            "phi_total_V": pop_out["phi_total_V"],
            "phi_by_type": pop_out.get("phi_by_type", {}),
            "peak_latencies_ms": peak_latencies,
            "n_fibers": int(len(per_fiber_phi)),
        }

    geom.cnap_pop = cnap_by_montage
    if not cnap_by_montage:
        msg = (
            "Recording basis on disk but no montage matched "
            "this design's fibers."
        )
        state.pop_cnap_status = msg
        state.pop_cnap_figure = build_pop_cnap_figure(
            cnap_by_montage={}, montage_meta=montages,
            active_mid="", decompose_by_type=False,
        )
        print(f"[cnap] population skip: {msg}", flush=True)
        return
    prev = str(state.active_montage_pop or "")
    if prev not in cnap_by_montage:
        state.active_montage_pop = next(iter(cnap_by_montage))
    state.pop_cnap_figure = build_pop_cnap_figure(
        cnap_by_montage=cnap_by_montage,
        montage_meta=montages,
        active_mid=str(state.active_montage_pop or ""),
        decompose_by_type=bool(
            getattr(state, "cnap_decompose_by_type", True),
        ),
    )
    n_fibers_summed = next(iter(cnap_by_montage.values())).get(
        "n_fibers", 0,
    )
    state.pop_cnap_status = (
        f"✓ cNAP from {n_fibers_summed} fiber"
        f"{'s' if n_fibers_summed != 1 else ''} across "
        f"{len(cnap_by_montage)} montage"
        f"{'s' if len(cnap_by_montage) > 1 else ''}"
    )
    try:
        state.flush()
    except Exception:                                        # noqa: BLE001
        pass
    # Peak amplitude per montage for the log.
    summary = ", ".join(
        f"{m}: peak {float(np.max(np.abs(c['phi_total_V'])) * 1e6):.2f} µV "
        f"({c['n_fibers']}f)"
        for m, c in cnap_by_montage.items()
    )
    print(f"[cnap] population done: {summary}", flush=True)


async def run_pop_generate(ctx: PipelineContext) -> None:
    """Sample (type, diameter) for each fiber from the per-branch
    mixture; build KDE + cross-section figures; refresh viewport
    colouring; autosave."""
    state = ctx.state
    geom = ctx.geom
    H = ctx.helpers

    if not state.pop_branches_meta:
        state.pop_status = (
            "⚠ No fibers loaded — generate fibers first "
            "(Fiber trajectories tab)."
        )
        state.pop_generated = False
        return
    bt = dict(state.pop_branch_types or {})
    # Refuse if NOTHING is assigned anywhere.
    if not any(rows for rows in bt.values()):
        state.pop_status = (
            "⚠ Add at least one fiber type to a branch."
        )
        state.pop_generated = False
        return
    # A re-generate invalidates any prior sim results (new
    # assignments may put different rows on different fibers).
    geom.fiber_pop_sim_results = None
    state.pop_sim_done = False
    state.pop_sim_results_meta = []
    state.pop_activated_set = []
    state.pop_xsec_figure = {"data": [], "layout": {}}
    state.pop_propagation_figure = {"data": [], "layout": {}}
    # Note: pop_xsec_cuff_figure (the at-cuff-centre tile) is
    # preserved across sim re-runs because it depends only on
    # the population assignment, not on whether the sim has been
    # run.
    state.pop_waterfall_figure = {"data": [], "layout": {}}
    # Raise the busy lightbox + flush so the client renders it
    # BEFORE we do the heavy KDE build. `await asyncio.sleep(0)`
    # yields control so trame can push the state.
    state.pop_busy = True
    state.busy = True
    state.busy_msg = "Generating fiber population"
    state.busy_log = ""
    state.flush()
    await asyncio.sleep(0)

    bidx_arr = np.asarray(
        geom.fiber_branch_idx, dtype=np.int32,
    )
    n_fibers = int(bidx_arr.size)
    pop_types = np.empty(n_fibers, dtype=object)
    pop_types[:] = ""
    pop_rows = np.empty(n_fibers, dtype=object)
    pop_rows[:] = ""
    pop_diams = np.zeros(n_fibers, dtype=np.float32)
    rng = np.random.default_rng(int(state.pop_seed))
    used_models: set = set()
    empty_branches: list[int] = []
    bad_sum_branches: list[tuple[int, float]] = []
    # Aggregate row metadata across all branches.
    row_meta: dict = {}
    for meta in state.pop_branches_meta:
        b = int(meta["idx"])
        key = str(b)
        rows = bt.get(key, [])
        mask = bidx_arr == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        if not rows:
            empty_branches.append(b)
            continue
        fracs = np.array(
            [max(0.0, float(r.get("frac", 0.0)))
             for r in rows],
            dtype=np.float64,
        )
        raw_sum = float(fracs.sum())
        if raw_sum <= 0:
            fracs = np.ones_like(fracs)
            bad_sum_branches.append((b, raw_sum))
        elif abs(raw_sum - 100.0) > 0.5:
            bad_sum_branches.append((b, raw_sum))
        fracs = fracs / fracs.sum()
        models = [str(r.get("model", "")) for r in rows]
        row_ids = [str(r.get("id", "")) for r in rows]
        row_names = [
            str(r.get("name") or f"Type {i + 1}")
            for i, r in enumerate(rows)
        ]
        row_colors = [
            str(r.get("color")
                or H.TAB10_PALETTE[i % len(H.TAB10_PALETTE)])
            for i, r in enumerate(rows)
        ]
        means = np.array(
            [float(r.get("mean_um", 0.0)) for r in rows],
            dtype=np.float64,
        )
        stds = np.array(
            [max(0.0, float(r.get("std_um", 0.0)))
             for r in rows],
            dtype=np.float64,
        )
        type_choices = rng.choice(
            len(rows), size=n_b, p=fracs,
        )
        chosen_means = means[type_choices]
        chosen_stds = stds[type_choices]
        diam_samples = rng.normal(chosen_means, chosen_stds)
        diam_samples = np.clip(diam_samples, 0.1, 20.0)
        fiber_indices = np.where(mask)[0]
        for j, fi in enumerate(fiber_indices):
            t_idx = int(type_choices[j])
            pop_types[int(fi)] = models[t_idx]
            pop_rows[int(fi)] = row_ids[t_idx]
            pop_diams[int(fi)] = float(diam_samples[j])
            used_models.add(models[t_idx])
        backends = [
            str(r.get("backend") or "pyfibers")
            for r in rows
        ]
        for i, rid in enumerate(row_ids):
            row_meta[rid] = {
                "name": row_names[i],
                "backend": backends[i],
                "model": models[i],
                "color": row_colors[i],
                "branch": b,
                "mean_um": float(means[i]),
                "std_um": float(stds[i]),
            }
    geom.fiber_pop_types = pop_types
    geom.fiber_pop_rows = pop_rows
    geom.fiber_pop_diameters_um = pop_diams
    state.pop_row_colors = {
        rid: m["color"] for rid, m in row_meta.items()
    }
    state.pop_row_meta = row_meta
    state.pop_row_visible = {rid: True for rid in row_meta}
    legacy_colors: dict = {}
    for rid, m in row_meta.items():
        legacy_colors.setdefault(m["model"], m["color"])
    state.pop_type_colors = legacy_colors
    state.pop_generated = True
    n_assigned = int(np.sum(pop_types != ""))
    msg = (
        f"✓ Generated population — {n_assigned}/{n_fibers} "
        f"fibers across {len(state.pop_branches_meta)} "
        f"branches, {len(row_meta)} named subpopulation(s)."
    )
    if empty_branches:
        msg += (
            f"  (no types assigned to "
            f"branch{'es' if len(empty_branches) > 1 else ''} "
            f"{', '.join(str(b) for b in empty_branches)})"
        )
    if bad_sum_branches:
        parts = [
            f"branch {b} sum={s:.0f}%"
            for b, s in bad_sum_branches
        ]
        msg += (
            "  ⚠ Fractions don't add up to 100 % on "
            f"{', '.join(parts)} — proceeded with "
            "auto-normalisation. Adjust the rows and "
            "re-generate if that's not what you wanted."
        )
    state.pop_status = msg
    try:
        _valid_rows = int(
            np.array(
                [bool(r) for r in pop_rows], dtype=bool,
            ).sum()
        )
        print(
            f"[pop_kde] build with "
            f"n_fibers={int(bidx_arr.size)}, "
            f"branches_meta={len(state.pop_branches_meta)}, "
            f"row_meta={len(row_meta)}, "
            f"valid_rows={_valid_rows}, "
            f"diams>0={int((pop_diams > 0).sum())}",
            flush=True,
        )
    except Exception:
        pass
    # R1.4-fix-up #4: tracer prints to localise the segfault that
    # fires during pop-generate. Each numbered step flushes
    # immediately so the LAST print before the crash pin-points
    # which call blew up.
    print("[pop_gen] step1: about to build pop_kde figure", flush=True)
    state.pop_kde_figure = _build_pop_kde_figure(
        bidx_arr, pop_rows, pop_diams,
        state.pop_branches_meta, row_meta,
    )
    print("[pop_gen] step2: pop_kde figure built", flush=True)
    _paths_display_xsec = H.fiber_paths_display()
    print("[pop_gen] step3: about to build xsec_cuff figure", flush=True)
    state.pop_xsec_cuff_figure = (
        _build_pop_xsec_at_cuff_figure(
            paths_display=_paths_display_xsec or [],
            bidx=bidx_arr,
            pop_rows=pop_rows,
            pop_diams=pop_diams,
            row_meta=row_meta,
            nerve_pts_cuff_m=geom.pts_cuff,
        )
    )
    print("[pop_gen] step4: xsec_cuff figure built", flush=True)
    # Re-render scene with population colours.
    print("[pop_gen] step5: about to rebuild scene", flush=True)
    ctx.scene.rebuild_callback()
    print("[pop_gen] step6: scene rebuilt, about to request render",
          flush=True)
    ctx.scene.request_render()
    print("[pop_gen] step7: request_render returned", flush=True)
    # Persist the new population to disk so it survives a project
    # close + reopen.
    H.save_pop_state()
    # Lower the busy lightbox.
    state.pop_busy = False
    state.busy = False
    state.busy_msg = ""
    state.busy_log = ""
    state.flush()


async def run_pop_sim(ctx: PipelineContext) -> None:
    """Run a population batch — for each fiber that was assigned
    a row in run_pop_generate, sample (backend, model) from the
    row's metadata and run the single-fiber sim on the FEM-
    derived Vₑ(s). Results land in `geom.fiber_pop_sim_results`
    (dict idx → sim_data). Per-fiber failures don't abort the
    rest."""
    state = ctx.state
    geom = ctx.geom
    H = ctx.helpers

    # Preconditions.
    if not bool(state.pop_generated):
        state.pop_status = (
            "⚠ Generate the population first."
        )
        return
    if (geom.fiber_paths_raw is None
            or len(geom.fiber_paths_raw) == 0):
        state.pop_status = (
            "⚠ generate fibers first "
            "(Fiber trajectories tab)."
        )
        return
    if geom.fiber_paths_Ve is None:
        state.pop_status = (
            "⚠ run the FEM solve first — per-fiber V_e is "
            "produced by Solve."
        )
        return
    if (geom.fiber_pop_rows is None
            or geom.fiber_pop_diameters_um is None):
        state.pop_status = (
            "⚠ Population state is stale — re-generate."
        )
        return

    row_meta = dict(state.pop_row_meta or {})
    pop_rows_arr = np.asarray(
        geom.fiber_pop_rows, dtype=object,
    )
    pop_diams = np.asarray(
        geom.fiber_pop_diameters_um, dtype=np.float64,
    )
    n_paths = int(len(geom.fiber_paths_raw))
    sim_indices = [
        i for i in range(n_paths)
        if i < len(pop_rows_arr) and pop_rows_arr[i]
    ]
    if not sim_indices:
        state.pop_status = (
            "⚠ Population has no assigned fibers."
        )
        return
    p = H.fiber_pulse_params()
    n_total = len(sim_indices)

    # Raise the busy lightbox.
    state.pop_busy = True
    state.pop_sim_done = False
    state.busy = True
    state.busy_msg = (
        f"Simulating population — 0/{n_total} fibers"
    )
    state.busy_log = ""
    state.flush()
    log_lines: list[str] = []

    def _log(line: str) -> None:
        line = ctx.stamp_user_line(line)
        log_lines.append(line)
        print(f"[pop-sim] {line}", flush=True)
        state.busy_log = "\n".join(log_lines[-10:])
        state.flush()

    loop = asyncio.get_event_loop()
    runner = InProcessRunner(_do_one_fiber)
    tok = CancelToken()

    results: dict[int, dict] = {}
    activated_set: set = set()
    results_meta: list[dict] = []
    n_ok = 0
    n_fail = 0
    for ci, fi in enumerate(sim_indices, start=1):
        rid = str(pop_rows_arr[fi])
        meta = row_meta.get(rid, {})
        backend = str(meta.get("backend") or "pyfibers")
        model_name = str(
            meta.get("model") or H.MYELINATED_MODELS[0]
        )
        d_um = (
            float(pop_diams[fi]) if pop_diams.size else 0.0
        )
        label, color = H.fiber_label_and_color(fi)
        state.busy_msg = (
            f"Simulating {label} ({ci}/{n_total}) · "
            f"{backend}"
        )
        state.flush()
        triple, err = _fiber_preflight(geom, fi)
        if err is not None:
            _log(f"  ⚠ {label} skipped: {err}")
            results_meta.append({
                "idx": int(fi), "label": label, "color": color,
                "ok": False, "activated": False,
                "summary": f"⚠ {err}",
            })
            n_fail += 1
            continue
        s_um, Ve_mV, length_um = triple
        if length_um < 5_000.0:
            msg = (
                f"{label} too short "
                f"({length_um * 1e-3:.2f} mm)"
            )
            _log(f"  ⚠ skipped: {msg}")
            results_meta.append({
                "idx": int(fi), "label": label, "color": color,
                "ok": False, "activated": False,
                "summary": f"⚠ {msg}",
            })
            n_fail += 1
            continue
        if d_um < 0.1:
            _log(
                f"  ⚠ {label} diameter too small "
                f"({d_um:.2f} µm)"
            )
            results_meta.append({
                "idx": int(fi), "label": label, "color": color,
                "ok": False, "activated": False,
                "summary": (
                    f"⚠ diameter too small: {d_um:.2f} µm"
                ),
            })
            n_fail += 1
            continue
        req = FiberSimJobRequest(
            sel=fi, s_um=s_um, Ve_mV=Ve_mV,
            diameter_um=d_um, length_um=length_um,
            pulse_params=p, backend=backend,
            model_name=model_name, helpers=H,
        )
        try:
            outputs = await loop.run_in_executor(
                None, lambda r=req: runner.run(r, _log, tok),
            )
        except Exception as ex:
            _log(
                f"  ⚠ {label} failed: "
                f"{type(ex).__name__}: {ex}"
            )
            results_meta.append({
                "idx": int(fi), "label": label, "color": color,
                "ok": False, "activated": False,
                "summary": f"⚠ {type(ex).__name__}: {ex}",
            })
            n_fail += 1
            continue
        if outputs.return_code != 0:
            _log(
                f"  ⚠ {label} runner rc={outputs.return_code}"
            )
            results_meta.append({
                "idx": int(fi), "label": label, "color": color,
                "ok": False, "activated": False,
                "summary": (
                    f"⚠ runner rc={outputs.return_code}"
                ),
            })
            n_fail += 1
            continue
        sim_data = outputs.outputs["sim_data"]
        summary = outputs.outputs["summary"]
        results[int(fi)] = sim_data
        # Activation criterion: at least one node fired a spike.
        spike_t = sim_data.get("spike_t", [])
        activated = False
        try:
            if isinstance(spike_t, np.ndarray):
                activated = bool(spike_t.size > 0)
            else:
                activated = any(
                    (len(s) if hasattr(s, "__len__")
                     else 0) > 0
                    for s in spike_t
                )
        except (TypeError, ValueError):
            activated = False
        if activated:
            activated_set.add(int(fi))
        results_meta.append({
            "idx": int(fi), "label": label, "color": color,
            "ok": True, "activated": activated,
            "summary": summary,
        })
        n_ok += 1
        _log(
            f"  {'⚡' if activated else '·'} {label} "
            f"({summary})"
        )

    # Persist results.
    geom.fiber_pop_sim_results = results
    state.pop_sim_results_meta = results_meta
    state.pop_activated_set = sorted(activated_set)
    # `state.pop_sim_done` is flipped to True ~40 lines below at
    # line 504, which is the canonical "pop sim succeeded" gate.
    # The EXPORT navbar tab reads `pop_sim_done || has_fiber_sim`
    # for its enable-when-any-sim-done check.

    # Build cross-section figure.
    paths_display = H.fiber_paths_display() or []
    state.pop_xsec_figure = _build_pop_xsec_figure(
        paths_display, pop_rows_arr, pop_diams,
        row_meta, activated_set,
    )

    # Seed heatmap + waterfall with the first activated fiber
    # (or first successful if none fired).
    first_view = next(
        (m["idx"] for m in results_meta
         if m.get("activated")),
        next(
            (m["idx"] for m in results_meta
             if m.get("ok")),
            None,
        ),
    )
    if first_view is not None:
        state.pop_view_idx = int(first_view)
        sim_data = results[int(first_view)]
        state.pop_propagation_figure = (
            _build_fiber_propagation_figure(sim_data)
        )
        state.pop_waterfall_figure = (
            _build_fiber_waterfall_figure(sim_data)
        )
    else:
        state.pop_propagation_figure = {
            "data": [], "layout": {}}
        state.pop_waterfall_figure = {
            "data": [], "layout": {}}

    # R1.4 — population cNAP. Sum compute_cnap_single across all
    # successful fibers per montage; decompose by fiber-type via
    # geom.fiber_pop_types so the figure can stack Aα/Aβ/Aδ/B/C
    # contributions beneath the total trace.
    try:
        _compute_and_set_pop_cnap(ctx, results)
    except Exception as _cnap_ex:                        # noqa: BLE001
        print(
            f"[pop] cNAP compute failed: "
            f"{type(_cnap_ex).__name__}: {_cnap_ex}",
            flush=True,
        )
        state.pop_cnap_status = (
            f"⚠ {type(_cnap_ex).__name__}: {_cnap_ex}"
        )
        state.pop_cnap_figure = {"data": [], "layout": {}}

    state.pop_sim_done = True
    state.pop_busy = False
    state.busy = False
    state.busy_msg = ""
    state.busy_log = ""
    n_activated = len(activated_set)
    if n_ok == 0:
        state.pop_status = (
            f"⚠ all {n_total} fiber sims failed — see log."
        )
    else:
        state.pop_status = (
            f"✓ {n_ok}/{n_total} fiber sims completed, "
            f"{n_activated} activated"
            + (f" ({n_fail} failed — see log)"
               if n_fail > 0 else "")
            + "."
        )
    # Persist the new sim results so they survive a project
    # close + reopen alongside the population assignments.
    H.save_pop_state()
    state.flush()
