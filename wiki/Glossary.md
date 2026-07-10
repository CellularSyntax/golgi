# Glossary

Domain terms used across golgi and this wiki.

### Anatomy
- **Endoneurium** — the connective tissue inside a fascicle, surrounding the axons. Anisotropic
  (conducts better along the fiber axis). Mesh tag 1.
- **Perineurium** — the thin, highly resistive sheath around each fascicle. Modeled as a
  contact-impedance sheet, not a meshed volume. See [Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties).
- **Epineurium** — the outer connective sheath binding fascicles into the nerve trunk. Mesh tag 5.
- **Fascicle** — a bundle of axons enclosed by perineurium. A nerve has one or many.
- **Branch** — a distal division of the nerve trunk (e.g. cardiac, recurrent laryngeal). golgi traces
  fibers through bifurcations and labels them by branch.

### Electrodes
- **Cuff electrode** — a sleeve placed around the nerve carrying contacts (e.g. LivaNova helical,
  ring arrays). See [Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer).
- **LIFE / TIME** — Longitudinal / Transverse Intrafascicular Electrodes: thin contacts inserted into
  the nerve.
- **Polarity** — a contact's role: `anode`, `cathode`, `ground`, or `off`.
- **Current fraction** — how the stimulus current is split among contacts in a polarity group.
- **Current steering** — shaping the field by re-weighting multi-contact polarities/fractions.
- **Montage** — a specific assignment of polarities (e.g. bipolar, tripolar guard).

### Field & solver
- **Quasi-static / electro-quasistatic** — the resistive regime where ∇·(σ∇Vₑ)=0 governs the field.
- **Vₑ (extracellular potential)** — the potential the cuff produces in tissue; drives the fibers.
- **σ (conductivity)** — tissue electrical conductivity (S/m); anisotropic for endoneurium and muscle.
- **Cole–Cole model** — a frequency-dependent dispersion model for tissue permittivity/conductivity.
- **IT'IS database** — the IT'IS Foundation tissue-property database golgi bundles for σ(f).
- **Lead field** — the per-contact unit-current field; montages and amplitudes are linear combinations
  of lead fields. Also used (by reciprocity) for [recording](Recording-and-CNAP).
- **Activating function** — the second spatial derivative of Vₑ along a fiber; predicts where it's
  excited.
- **Access impedance (Z_access)** — the resistance seen at a contact, from a DC dual-solve.
- **PLC** — Piecewise-Linear Complex: the surface description TetGen tetrahedralizes. See [Meshing](Meshing).
- **Characteristic length (lc)** — target element size for a mesh region.

### Fibers
- **MRG model** — the McIntyre–Richardson–Grill double-cable model of a myelinated axon.
- **PyFibers** — the NEURON-based fiber-simulation library golgi uses by default.
- **AxonML** — an optional GPU surrogate of the MRG model (separately licensed).
- **Threshold** — the minimum stimulus amplitude that elicits a propagating action potential.
- **Recruitment** — the fraction of a fiber population activated at a given amplitude.
- **Selectivity index (Veraart SI)** — `(R_target − R_offtarget)/(R_target + R_offtarget)`, in [−1, 1].
- **Conduction velocity (CV)** — how fast an action potential propagates (scales with diameter).
- **CNAP / ECAP** — compound nerve / evoked compound action potential: the population response a cuff
  records. See [Recording & CNAP](Recording-and-CNAP).

### Platform
- **Study** — the unit of work: a project directory plus the `golgi.Study` API over it.
- **Design** — one electrode geometry on the nerve. **Config** — one polarity/current montage for a design.
- **Study bundle** — a hashed, replayable archive of an entire study. See [Reproducible Study Bundles](Reproducible-Study-Bundles).
- **Replay** — re-hashing (or re-running) a bundle to verify it reproduces.
- **Lead-field basis** — the cached set of per-contact lead fields enabling fast steering/sweeps.

---

### See also
[Pipeline Overview](Pipeline-Overview) · [Finite-Element Solver](Finite-Element-Solver) ·
[Fiber Models & Activation](Fiber-Models-and-Activation) · [Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer)
