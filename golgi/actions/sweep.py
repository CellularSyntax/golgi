# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Sweep-tab action handlers (F2.1.c + fixup).

Three handlers wire the Sweep sub-tab to the F2.1.a/b backend +
figure builders:

  - do_run_amplitude_sweep    one-click "Run amplitude sweep"
  - do_find_thresholds        one-click "Find thresholds"
  - do_toggle_sweep_advanced  flip the Advanced section visibility

Both run handlers:
  1. Collect inputs from state (mode-specific sliders + filters).
  2. Build a SweepRequest. Empty fiber_sel_indices = sweep ALL
     fibers; non-empty = sweep only those.
  3. await pipeline.sweep.run_sweep — async; per-fiber sims run
     in loop.run_in_executor. Honors ctx.was_cancelled() between
     cells.
  4. Wire the on_progress callback to:
       - push the last 12 lines into state.busy_log
       - update state.busy_msg with the latest per-fiber line
       - print to terminal (so the user sees activity)
       - state.flush() so the busy lightbox updates live.
  5. Push the result's three figures into state.sweep_*_figure.

The full busy-lightbox lifecycle (`state.busy + busy_msg + busy_log`)
fires from upfront → finally, so the user always sees the running-job
overlay + the working Cancel button.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable

import numpy as np

from golgi.figures.recruitment import (
    activation_heatmap_to_csv,
    build_activation_heatmap_figure,
    build_recruitment_curve_figure,
    build_threshold_scatter_figure,
    recruitment_to_csv,
    threshold_to_csv,
)
from golgi.jobs.schemas import SweepRequest


def _csv_data_uri(csv_text: str) -> str:
    """Encode a CSV string as a `data:text/csv;base64,…` URI. The
    Download buttons in the Sweep panel use this URI directly as
    the <a href="…">; the browser handles the actual save dialog."""
    b64 = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
    return f"data:text/csv;base64,{b64}"


def _binary_data_uri(blob: bytes, mime: str) -> str:
    """Same shape as `_csv_data_uri` but for arbitrary binary
    content (NPZ cache file). The mime defaults to
    application/octet-stream so the browser doesn't try to
    interpret it."""
    b64 = base64.b64encode(blob).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _empty_figure() -> dict:
    return {"data": [], "layout": {}}


def _expand_amplitudes(
    lo_mA: float, hi_mA: float, n: int, spacing: str,
) -> list[float]:
    """Linspace / logspace expansion of the recruitment-mode axis.
    Logspace defends against lo <= 0 by clamping the lower bound."""
    n = max(2, int(n))
    if spacing == "log":
        lo = max(float(lo_mA), 1e-6)
        hi = max(float(hi_mA), lo * 1.01)
        return np.logspace(
            np.log10(lo), np.log10(hi), n,
        ).tolist()
    return np.linspace(
        float(lo_mA), float(hi_mA), n,
    ).tolist()


def register(
    state,
    *,
    pipeline_ctx,
    pipeline_sweep,
    fiber_pulse_params: Callable[[], dict],
    save_sweep: Callable | None = None,
) -> dict[str, Callable]:
    """Wire the three Sweep-tab handlers."""

    def _push_result_figures(result) -> None:
        """Common tail: build the three figures + flip
        sweep_has_result."""
        if result.activated is not None:
            state.sweep_recruitment_figure = (
                build_recruitment_curve_figure(result)
            )
            state.sweep_heatmap_figure = (
                build_activation_heatmap_figure(result)
            )
        else:
            state.sweep_recruitment_figure = _empty_figure()
            state.sweep_heatmap_figure = _empty_figure()
        if result.thresholds_uA is not None:
            state.sweep_threshold_figure = (
                build_threshold_scatter_figure(result)
            )
        else:
            state.sweep_threshold_figure = _empty_figure()
        n_fibers = int(len(result.fiber_indices))
        if result.activated is not None:
            n_amps = int(result.activated.shape[1])
            state.sweep_result_summary = (
                f"Recruitment sweep · {n_fibers} fibers × "
                f"{n_amps} amplitudes · "
                f"{result.elapsed_s:.1f} s "
                f"({result.n_sims_total} sims)"
            )
        else:
            n_activated = int(np.isfinite(
                np.asarray(result.thresholds_uA),
            ).sum())
            state.sweep_result_summary = (
                f"Threshold finder · {n_fibers} fibers · "
                f"{n_activated} activated · "
                f"{result.elapsed_s:.1f} s "
                f"({result.n_sims_total} sims)"
            )
        state.sweep_has_result = True
        # ---- F2.1.d: cache result to disk + push browser-download
        # data URIs ----
        # save_sweep keeps the disk cache (project reopen restore,
        # F2.2-style study-bundle export). The browser downloads
        # use eagerly-built data URIs computed straight from the
        # in-memory result (no need to read the on-disk files
        # back).
        if save_sweep is not None:
            try:
                paths = save_sweep(result) or {}
            except Exception as ex:                          # noqa: BLE001
                print(
                    f"[sweep] cache write failed: {ex}",
                    flush=True,
                )
                paths = {}
        else:
            paths = {}
        sha = str(result.sha or "")
        # Build the data URIs for whichever payloads this mode
        # populated.
        rec_uri = thr_uri = hm_uri = ""
        rec_name = thr_name = hm_name = ""
        if result.activated is not None:
            rec_uri = _csv_data_uri(recruitment_to_csv(result))
            rec_name = f"sweep_{sha}_recruitment.csv"
            hm_uri = _csv_data_uri(
                activation_heatmap_to_csv(result),
            )
            hm_name = f"sweep_{sha}_activation_heatmap.csv"
        if result.thresholds_uA is not None:
            thr_uri = _csv_data_uri(threshold_to_csv(result))
            thr_name = f"sweep_{sha}_thresholds.csv"
        # NPZ binary URI — read the file save_sweep just wrote.
        # Skip silently when no project is active (no disk path).
        npz_uri = ""
        npz_name = ""
        npz_path = paths.get("npz")
        if npz_path is not None:
            try:
                npz_uri = _binary_data_uri(
                    Path(npz_path).read_bytes(),
                    "application/octet-stream",
                )
                npz_name = f"sweep_{sha}.npz"
            except Exception as ex:                          # noqa: BLE001
                print(
                    f"[sweep] npz data-uri build failed: {ex}",
                    flush=True,
                )
        with state:
            state.sweep_cache_sha = sha
            state.sweep_recruitment_csv_data_uri = rec_uri
            state.sweep_recruitment_csv_filename = rec_name
            state.sweep_threshold_csv_data_uri = thr_uri
            state.sweep_threshold_csv_filename = thr_name
            state.sweep_heatmap_csv_data_uri = hm_uri
            state.sweep_heatmap_csv_filename = hm_name
            state.sweep_npz_data_uri = npz_uri
            state.sweep_npz_filename = npz_name

    def _selected_fiber_indices() -> list[int] | None:
        """Read fiber_sel_indices from state. Empty = None (all
        fibers). Otherwise return as int list."""
        sel = list(state.fiber_sel_indices or [])
        cleaned: list[int] = []
        for v in sel:
            try:
                cleaned.append(int(v))
            except (TypeError, ValueError):
                continue
        return cleaned or None

    def _make_progress_pump(busy_prefix: str) -> Callable[[str], None]:
        """Build an on_progress callback. Each call appends to a
        rolling 18-line buffer + pushes to state.busy_log + prints
        to the terminal + flushes state.

        Header lines (no leading whitespace) ALSO update busy_msg +
        sweep_status — they describe what fiber is currently being
        worked on. Detail lines (leading two spaces, per the
        preflight-log convention in pipeline/fiber_sim.py) only
        flow into busy_log so the top status doesn't flicker on
        every amplitude probe."""
        rolling: list[str] = []
        last_header: list[str] = [""]

        def _pump(line: str) -> None:
            rolling.append(line)
            tail = rolling[-18:]
            is_detail = line.startswith("  ")
            if not is_detail:
                last_header[0] = line
            with state:
                state.busy_log = "\n".join(tail)
                top = last_header[0] or line
                state.busy_msg = f"{busy_prefix} · {top}"
                state.sweep_status = f"{busy_prefix} · {top}"
            print(f"[sweep] {line}", flush=True)
            state.flush()

        return _pump

    async def do_run_amplitude_sweep(*_args) -> None:
        """One-click recruitment-curve sweep."""
        geom = pipeline_ctx.geom
        if (geom.fiber_paths_Ve is None
                or not len(geom.fiber_paths_Ve)):
            state.sweep_status = (
                "⚠ Run a FEM solve first — sweep needs the "
                "cached Vₑ on each fiber path."
            )
            state.sweep_failed = True
            return
        amps = _expand_amplitudes(
            float(state.sweep_amp_min_mA),
            float(state.sweep_amp_max_mA),
            int(state.sweep_amp_n_points),
            str(state.sweep_amp_spacing),
        )
        try:
            pulse = fiber_pulse_params()
        except Exception as ex:                              # noqa: BLE001
            state.sweep_status = (
                f"⚠ Pulse params unavailable: {ex}"
            )
            state.sweep_failed = True
            return
        req = SweepRequest(
            mode="recruitment",
            amplitudes_mA=amps,
            fiber_indices=_selected_fiber_indices(),
            pulse_params=pulse,
            backend=str(state.fiber_backend),
            model_name=str(state.fiber_model),
            model_source=str(state.sweep_model_source),
        )
        # Reset the cancel token. FEM/mesh/fibers clear this via
        # register_subprocess on each subprocess launch; the sweep
        # has no subprocess and so needs an explicit reset
        # — otherwise a previous Cancel click leaves _requested
        # sticky and every new sweep returns on the first
        # was_cancelled() check (0 sims, all-False activation).
        pipeline_ctx.clear_subprocess()
        # Wire the busy lightbox + progress feed. Set state.busy
        # FIRST so the overlay appears immediately.
        with state:
            state.busy = True
            state.busy_msg = "Running amplitude sweep…"
            state.busy_log = "Starting…"
            state.sweep_busy = True
            state.sweep_failed = False
            state.sweep_status = "Running amplitude sweep…"
            state.sweep_has_result = False
        state.flush()
        progress = _make_progress_pump("Amplitude sweep")
        try:
            result = await pipeline_sweep.run_sweep(
                pipeline_ctx, req, on_progress=progress,
            )
        except Exception as ex:                              # noqa: BLE001
            print(
                f"[sweep] amplitude sweep failed: {ex}",
                flush=True,
            )
            with state:
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.sweep_busy = False
                state.sweep_failed = True
                state.sweep_status = (
                    f"⚠ sweep failed: {type(ex).__name__}: {ex}"
                )
            state.flush()
            return
        _push_result_figures(result)
        with state:
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.sweep_busy = False
            state.sweep_status = state.sweep_result_summary
        state.flush()

    async def do_find_thresholds(*_args) -> None:
        """One-click threshold-finder sweep."""
        geom = pipeline_ctx.geom
        if (geom.fiber_paths_Ve is None
                or not len(geom.fiber_paths_Ve)):
            state.sweep_status = (
                "⚠ Run a FEM solve first — threshold finder "
                "needs the cached Vₑ on each fiber path."
            )
            state.sweep_failed = True
            return
        try:
            pulse = fiber_pulse_params()
        except Exception as ex:                              # noqa: BLE001
            state.sweep_status = (
                f"⚠ Pulse params unavailable: {ex}"
            )
            state.sweep_failed = True
            return
        req = SweepRequest(
            mode="threshold",
            bisect_lo_mA=float(state.sweep_bisect_lo_mA),
            bisect_hi_mA=float(state.sweep_bisect_hi_mA),
            bisect_tol_uA=float(state.sweep_bisect_tol_uA),
            fiber_indices=_selected_fiber_indices(),
            pulse_params=pulse,
            backend=str(state.fiber_backend),
            model_name=str(state.fiber_model),
            model_source=str(state.sweep_model_source),
        )
        # Reset the cancel token — see note in do_run_amplitude_sweep.
        pipeline_ctx.clear_subprocess()
        with state:
            state.busy = True
            state.busy_msg = "Finding thresholds…"
            state.busy_log = "Starting bisection…"
            state.sweep_busy = True
            state.sweep_failed = False
            state.sweep_status = "Bisecting per-fiber thresholds…"
            state.sweep_has_result = False
        state.flush()
        progress = _make_progress_pump("Threshold finder")
        try:
            result = await pipeline_sweep.run_sweep(
                pipeline_ctx, req, on_progress=progress,
            )
        except Exception as ex:                              # noqa: BLE001
            print(
                f"[sweep] threshold-finder failed: {ex}",
                flush=True,
            )
            with state:
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.sweep_busy = False
                state.sweep_failed = True
                state.sweep_status = (
                    f"⚠ threshold-finder failed: "
                    f"{type(ex).__name__}: {ex}"
                )
            state.flush()
            return
        _push_result_figures(result)
        with state:
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.sweep_busy = False
            state.sweep_status = state.sweep_result_summary
        state.flush()

    def do_toggle_sweep_advanced(*_args) -> None:
        state.sweep_show_advanced = not bool(
            state.sweep_show_advanced,
        )

    return {
        "do_run_amplitude_sweep": do_run_amplitude_sweep,
        "do_find_thresholds": do_find_thresholds,
        "do_toggle_sweep_advanced": do_toggle_sweep_advanced,
    }
