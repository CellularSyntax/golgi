# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fiber-model backend dispatchers (axonml MRG surrogate).

Extracted from `golgi/app.py` in step W1.6 of FEATURES.md.

Public surface (single entry point used via the H SimpleNamespace
bundle in build_app, then consumed by
`pipeline/fiber_sim.py::_run_axonml_branch`):

  - `axonml_run_single(s_um, Ve_mV, diameter_um, …) -> dict`
    Runs ONE fiber at ONE amplitude on axonml's MRG surrogate.

Internal helpers (underscore-prefixed; only callers are within this
module): `_axonml_n_nodes`, `_axonml_resample_Ve`, `_axonml_device`.

The pyfibers backend dispatcher
(`pipeline/fiber_sim.py::_run_pyfibers_branch`) is already in
golgi/pipeline/ from migration step 4.6; it does not need to call
out here — pyfibers is invoked directly inside that module.
"""
from __future__ import annotations

import numpy as np


def _axonml_n_nodes(length_um: float, diameter_um: float) -> int:
    """MRG-spaced node count fitting in a fiber of `length_um`.
    Internodal spacing ≈ 100 × diameter. Floor 11 nodes so the
    activation function has somewhere to act."""
    internode_um = 100.0 * float(diameter_um)
    return max(11, int(round(length_um / internode_um)) + 1)


def _axonml_resample_Ve(s_um, Ve_mV, n_nodes):
    """Linearly interpolate V_e(s) at the MRG node positions."""
    length = float(np.asarray(s_um).max())
    node_s = np.linspace(0.0, length, n_nodes)
    Ve_node = np.interp(node_s, s_um, Ve_mV)
    return node_s, Ve_node


def _axonml_device():
    """Pick the best PyTorch device for axonml inference and patch
    `axonml.models.core.Axon.device` so the package agrees.

    Priority: CUDA → MPS → CPU. Raises a clear RuntimeError if
    axonml itself can't be imported (so the gated handler can
    catch it + show install instructions). axonml's `Axon.device()`
    is hard-coded to "cuda" if `self.is_cuda` else "cpu", which
    breaks MPS — the class-level patch makes it return our actual
    device string everywhere."""
    try:
        import torch
        from axonml.models.core import Axon
    except ImportError as ex:
        raise RuntimeError(
            f"axonml backend unavailable — `import axonml` failed "
            f"({ex}). Install per https://github.com/wmglab-duke/"
            f"axonml: clone the repo and `pip install -r "
            f"requirements.txt` in a PyTorch env."
        ) from ex
    if torch.cuda.is_available():
        dev = "cuda"
    elif (hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        print("[axonml] no CUDA found; falling back to Apple MPS "
               "— this is untested upstream and may be slow or "
               "hit unsupported ops.", flush=True)
        dev = "mps"
    else:
        print("[axonml] no CUDA/MPS found; falling back to CPU "
               "— this is untested upstream and will be ~10–100× "
               "slower than CUDA.", flush=True)
        dev = "cpu"
    Axon.device = (lambda self, _d=dev: _d)
    return dev


def axonml_run_single(
    s_um, Ve_mV, diameter_um: float,
    pulse_shape: str = "", pw_us: float = 0.0,
    amp_uA: float = 0.0,
    t0_ms: float = 1.0, tstop_ms: float = 8.0,
    dt_ms: float = 0.005, Iref_mA: float = 1.0,
    wave_mA=None,
) -> dict:
    """Run ONE fiber at ONE amplitude on axonml's MRG surrogate.
    Returns a dict shaped like the pyfibers single-amp output —
    {vm, time_ms, node_z_um, spike_t_ms, spike_n, wave_t_ms,
     wave_v, Ve_node_mV, n_nodes}. See nerve_studio §12 for the
    derivation; `wave_mA` (1-D array on the simulator's dt grid)
    is the "custom pulse" mode we use here. Legacy unit-pulse
    mode (`pulse_shape` + `pw_us` + `amp_uA`) is kept for
    population sweeps that the Population tab may use later."""
    device = _axonml_device()
    import torch
    from axonml.models import SMF
    from axonml.models.callbacks import Recorder, Raster

    length_um = float(np.asarray(s_um).max())
    n_nodes = _axonml_n_nodes(length_um, diameter_um)
    node_s_um, Ve_node_mV = _axonml_resample_Ve(
        s_um, Ve_mV, n_nodes,
    )
    n_t = int(round(tstop_ms / dt_ms)) + 1
    t_arr_ms = np.arange(n_t, dtype=np.float64) * dt_ms
    if wave_mA is not None:
        w = np.asarray(wave_mA, dtype=np.float64)
        if len(w) != len(t_arr_ms):
            raise ValueError(
                f"wave_mA length {len(w)} does not match dt-grid "
                f"length {len(t_arr_ms)} (tstop={tstop_ms} ms, "
                f"dt={dt_ms} ms)"
            )
        ve_np = (w[:, None] * Ve_node_mV[None, :])
    else:
        # Unit-pulse legacy path (used by future population sweeps).
        unit_w = np.zeros_like(t_arr_ms)
        pw_ms = pw_us * 1e-3
        if pulse_shape == "monophasic cathodic":
            unit_w[(t_arr_ms >= t0_ms)
                   & (t_arr_ms < t0_ms + pw_ms)] = +1.0
        elif pulse_shape == "monophasic anodic":
            unit_w[(t_arr_ms >= t0_ms)
                   & (t_arr_ms < t0_ms + pw_ms)] = -1.0
        elif pulse_shape == "biphasic cathode-first":
            h = pw_ms / 2
            unit_w[(t_arr_ms >= t0_ms)
                   & (t_arr_ms < t0_ms + h)] = +1.0
            unit_w[(t_arr_ms >= t0_ms + h)
                   & (t_arr_ms < t0_ms + pw_ms)] = -1.0
        else:  # biphasic anode-first
            h = pw_ms / 2
            unit_w[(t_arr_ms >= t0_ms)
                   & (t_arr_ms < t0_ms + h)] = -1.0
            unit_w[(t_arr_ms >= t0_ms + h)
                   & (t_arr_ms < t0_ms + pw_ms)] = +1.0
        scale = (amp_uA * 1e-3) / Iref_mA
        ve_np = (scale * unit_w[:, None] * Ve_node_mV[None, :])
        w = unit_w * scale

    ve_t = (torch.from_numpy(ve_np.astype("float32"))
            .unsqueeze(1).unsqueeze(1).to(device))
    diams = torch.tensor(
        [float(diameter_um)], dtype=torch.float32, device=device,
    )
    model = SMF().to(device).load("MRG")
    model.compile(nodes=n_nodes, axons=1)
    rec = Recorder()
    # Start AP-check just before stim onset (t_start_check is in
    # ms, not a timestep index — see axonml callbacks).
    t_check_ms = max(0.0, t0_ms - 0.01)
    raster = Raster(threshold=-30.0,
                     t_start_check=t_check_ms,
                     node_check=list(range(n_nodes)),
                     dt=dt_ms)
    model.run(ve=ve_t, diameters=diams, dt=dt_ms,
              callbacks=[rec, raster])
    vm_list = [
        r[:, -1, :].detach().cpu().numpy().reshape(-1)
        for r in rec.rec
    ]
    vm_TN = np.array(vm_list, dtype=np.float64)
    T = min(vm_TN.shape[0], t_arr_ms.size)
    t_used_ms = t_arr_ms[:T]
    vm = vm_TN[:T].T  # (N, T) — pyfibers convention.

    # Per-node spike times. Raster records only steps after
    # t_start_check, so offset record[0] back to absolute time.
    raster_t0_ms = int(np.ceil(t_check_ms / dt_ms)) * dt_ms
    spike_t = np.full(n_nodes, np.nan)
    spike_n = np.zeros(n_nodes, dtype=int)
    for ti, step in enumerate(raster.record):
        flag = step.detach().cpu().numpy().reshape(-1)
        for ni in np.flatnonzero(flag):
            spike_n[ni] += 1
            if np.isnan(spike_t[ni]):
                spike_t[ni] = raster_t0_ms + ti * dt_ms

    return {
        "vm": vm,
        "time_ms": t_used_ms,
        "node_z_um": node_s_um,
        "spike_t_ms": spike_t,
        "spike_n": spike_n,
        "wave_t_ms": t_arr_ms,
        "wave_v": w,
        "Ve_node_mV": Ve_node_mV,
        "n_nodes": n_nodes,
    }
