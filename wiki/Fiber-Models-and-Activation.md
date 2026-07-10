# Fiber Models & Activation

Given the extracellular potential **Vₑ** sampled along each fiber and a stimulus pulse, golgi computes
whether the fiber fires and at what amplitude — its **activation threshold** — using full
biophysical cable models. Two interchangeable backends compute the same quantity at different
speed/fidelity trade-offs.

Source: `golgi/pipeline/fiber_sim.py` (driver + dispatch), `golgi/pipeline/fiber_backends.py`
(AxonML), `golgi/state_defaults/fiber.py` (defaults).

---

## Two backends

| Backend | Engine | Use |
|---|---|---|
| **PyFibers / NEURON** *(default)* | full Hodgkin–Huxley cable simulation in NEURON (Marshall et al. 2025; Hines & Carnevale 1997) | reference-grade thresholds; the validated path |
| **AxonML** *(optional)* | GPU surrogate of the MRG model (Torch) | high-throughput sweeps / populations; falls back to CPU without CUDA |

Select the backend per run (`backend="pyfibers"` / `"axonml"`). AxonML is a separately-licensed,
non-commercial Duke package that golgi never bundles — see [License & Citation](License-and-Citation);
the default NEURON/PyFibers path needs nothing extra.

## Fiber models

- **Myelinated** fibers use the **MRG** model (McIntyre–Richardson–Grill double-cable; McIntyre et al.
  2002). golgi offers both a continuous **interpolated** MRG (any diameter in range) and **discrete**
  MRG at the canonical McIntyre diameters.
- **Unmyelinated** (C) fibers use unmyelinated models (e.g. Tigerholm et al. 2014 / Sundt et al. 2015).

Diameter selects the model and its morphology (internodal spacing, node geometry). The
diameter→model mapping and per-model ranges are configured in `FIBER_MODEL_DIAMETER_CONFIG`; the
default fiber diameter is 5.7 µm with the interpolated MRG model. See
[Fiber Populations & Trajectories](Fiber-Populations-and-Trajectories) for how diameters are assigned
across a population.

## Pulses

Stimulus waveforms are built by `build_pulse_waveform` — **monophasic** or **biphasic** (cathodic- or
anodic-first, with an optional interphase gap and charge balancing), parameterized by amplitude (mA),
pulse width (µs), onset, and stop time. Pulse parameters are set in the single-fiber / population /
sweep panels.

## Thresholds & recruitment

An **activation threshold** is the smallest stimulus amplitude that produces a propagating action
potential. golgi finds it by **bisection (titration)** on amplitude — the fiber is simulated at
successive amplitudes, halving the bracket each time until the threshold is bracketed to a tolerance
(default ±10 µA). Because the quasi-static field is linear in current, scaling amplitude simply scales
Vₑ — no FEM re-solve is needed per amplitude (see [Finite-Element Solver](Finite-Element-Solver)).

From thresholds across a population you get **recruitment** (fraction of fibers activated vs amplitude)
and, per branch/type, fiber-type-selective recruitment — the inputs to
[selectivity analysis](Recruitment-Sweeps-and-Selectivity). golgi also reports **conduction velocity**
(myelinated fibers conduct faster; saltatory vs continuous), which is one of the
[validation](Validation) checks against Hursh and McIntyre references.

---

## References

- McIntyre CC, Richardson AG, Grill WM (2002). Modeling the excitability of mammalian nerve fibers: influence of afterpotentials on the recovery cycle. *J Neurophysiol* 87:995. — the MRG myelinated-fiber model.
- Marshall DP, Farah ES, Musselman ED, Pelot NA, Grill WM (2025). PyFibers: an open-source NEURON-Python package to simulate responses of peripheral nerve fibers. *PLOS Comput Biol.* — the default fiber-simulation backend.
- Hines ML, Carnevale NT (1997). The NEURON simulation environment. *Neural Comput* 9:1179. — the simulator PyFibers runs on.
- Tigerholm J, et al. (2014). Modeling activity-dependent changes of axonal spike conduction in primary afferent neurons. *J Neurophysiol.* — unmyelinated C-fiber model.
- Sundt D, Gamper N, Jaffe DB (2015). Spike propagation through the dorsal root ganglia in an unmyelinated sensory neuron: a modeling study. *J Neurophysiol.* — unmyelinated C-fiber model.
- Hursh JB (1939). Conduction velocity and diameter of nerve fibers. *Am J Physiol* 127:131. — conduction-velocity validation target.

See [Validation](Validation#references) for the strength–duration and cross-platform threshold references.

---

### See also
[Fiber Populations & Trajectories](Fiber-Populations-and-Trajectories) ·
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity) ·
[Finite-Element Solver](Finite-Element-Solver) · [Recording & CNAP](Recording-and-CNAP) ·
[Validation](Validation)
