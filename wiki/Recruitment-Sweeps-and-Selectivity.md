# Recruitment Sweeps & Selectivity

This is the analysis layer that turns a solved model into the numbers you cite: **recruitment curves**,
**per-fiber thresholds**, **selectivity** between target and off-target branches, and **side-by-side
design comparison**. It runs on the cached fiber field, so amplitude sweeps are fast.

Source: `golgi/pipeline/sweep.py`, `golgi/pipeline/selectivity.py`, `golgi/projects/sweep_cache.py`,
`golgi/figures/recruitment.py` + `selectivity.py`. Schemas in `golgi/jobs/schemas.py`.

---

## Sweeps

A sweep is described by a `SweepRequest` and run via [`Study.run_sweep`](Python-API) (or the GUI Sweep
panel). Two modes:

- **`recruitment`** — simulate every fiber across an **amplitude axis** (`amplitudes_mA`) and record
  an `activated` matrix of shape `(n_fibers, n_amplitudes)`.
- **`threshold`** — find each fiber's activation threshold by **bisection** (`bisect_lo_mA`,
  `bisect_hi_mA`, `bisect_tol_uA`), recording `thresholds_uA` of shape `(n_fibers,)`.

```python
from golgi.jobs.schemas import SweepRequest

# recruitment across amplitude
req = SweepRequest(mode="recruitment",
                   amplitudes_mA=[0.1, 0.25, 0.5, 1.0, 2.0],
                   backend="pyfibers", model_name="MRG_INTERPOLATION")

# or per-fiber thresholds by bisection
req = SweepRequest(mode="threshold",
                   bisect_lo_mA=0.01, bisect_hi_mA=2.0, bisect_tol_uA=10.0)

result = study.run_sweep(req)
```

You can restrict the sweep with `fiber_indices`, `branch_filter`, or `fiber_type_filter`, and set
`pulse_params`. The `SweepResult` carries `activated` / `thresholds_uA` plus per-fiber
`fiber_diameters_um`, `fiber_branch_idx`, and `fiber_type_labels`, so every metric below can be sliced
by branch and type.

### The amplitude shortcut

Because the FEM is **linear in stimulus current**, an `I_stim`/amplitude axis is computed by
**rescaling the cached per-fiber Vₑ** rather than re-solving the field — only the (cheap) fiber
simulation re-runs per amplitude. Axes that touch σ, mesh, electrode geometry, or fiber paths re-enter
the appropriate upstream stage instead. See [Finite-Element Solver](Finite-Element-Solver).

### Caching

Each result is written to `<project>/sweeps/sweep_<sha>.npz` (plus `.json` and CSVs), keyed by a SHA
over the request — **re-running an identical sweep is instant**. Results are also tagged per config so
multiple electrode configs coexist. CSV exports cover recruitment, thresholds, and the activation
heatmap.

---

## Selectivity metrics

`pipeline/selectivity.py` computes, as pure post-processing on sweep results:

- **Branch recruitment** — fraction of fibers in a branch activated at each amplitude.
- **Veraart selectivity index** (Veraart et al. 1993) for a target branch against off-target branches:

  $$\mathrm{SI} = \frac{R_\text{target} - R_\text{off-target}}{R_\text{target} + R_\text{off-target}} \in [-1,\,1]$$

  +1 = only the target is recruited; 0 = equal; −1 = only off-target.
- **Threshold ratio** — `median(T_off-target) / median(T_target)` (>1 means the target activates at a
  lower amplitude — desirable).
- **Spatial activation map** — per cross-section position, the minimum amplitude to activate each
  fiber.

In the GUI you pick a **target branch** (e.g. the cardiac branch); selectivity is computed against the
union of the others (or custom target/off-target sets). These power the selectivity figures
([Figures & Reports](Figures-and-Reports)).

---

## Comparing designs & steering current

A project can hold several electrode **designs** and **configs** on the same nerve. The **Compare
configurations** view puts two designs side by side (axis + slice fields) with a selectivity bar chart
and threshold-ratio table, so you can answer "which cuff is more selective?" directly.

Because the field is a **per-contact lead-field basis**, you can also **steer current** — change which
contacts are anodes/cathodes and their current fractions — and re-evaluate recruitment without
re-solving the FEM. This makes multi-contact montage exploration (the workhorse of selective VNS)
interactive. See [Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer) for setting polarities and
fractions.

---

## References

- Veraart C, Grill WM, Mortimer JT (1993). Selective control of muscle activation with a multipolar nerve cuff electrode. *IEEE Trans Biomed Eng* 40:640. — the selectivity index.
- Musselman ED, Cariello JE, Grill WM, Pelot NA (2021). ASCENT: a pipeline for sample-specific computational modeling of electrical stimulation of peripheral nerves. *PLOS Comput Biol.* — recruitment/threshold methodology.

---

### See also
[Fiber Models & Activation](Fiber-Models-and-Activation) · [Finite-Element Solver](Finite-Element-Solver) ·
[Figures & Reports](Figures-and-Reports) · [Python API](Python-API) · [Validation](Validation)
