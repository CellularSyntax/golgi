# Fiber Populations & Trajectories

golgi traces **curved, fascicle-following fiber trajectories** through the real 3-D nerve — including
through **bifurcations** — and seeds them with biology-realistic **diameter populations**. This is
what enables branch-selective analysis that straight-fiber, cross-section-only tools cannot do.

Source: `golgi/compute/solve_fiber_paths_nerve.py` (trajectory solver),
`golgi/pipeline/fibers.py` (driver), `golgi/state_defaults/pop_presets.py` (populations),
`golgi/figures/population.py` (previews).

---

## How trajectories are generated

Fibers follow the natural current-flow topology of the nerve, obtained from a Laplace solve on the
**endoneurium volume** (a nerve-only tetrahedral mesh, built independently of the cuff so the result
is the same wherever the cuff sits):

1. **Laplace solve.** Solve ∇²φ = 0 on the endoneurium with insulating lateral walls, injecting
   current at the trunk cap and draining it equally across the branch caps (a pinned interior vertex
   fixes the constant).
2. **Cap detection + branch classification.** Caps (trunk and branch ends) are found in the nerve's
   PCA frame: keep facets whose normal is nearly axial, restrict to a z-band near each extremum, then
   cluster by cross-sectional proximity and drop clusters that are too small. Each branch is a cluster.
3. **Streamline integration.** Seed points across the trunk cross-section are integrated along the
   field −∇φ from trunk toward the branches, producing smooth trajectories that respect the nerve's
   geometry and split at bifurcations.

The result is `nerve_paths_fibers.npz` (per-fiber 3-D polylines) plus `nerve_paths_caps.json` and a
branch summary. A simpler straight, **axial** mode is available as an alternative. The two are
selected by `fiber_method` (`"streamlines"` for curved, physics-based paths, or `"axial"` for
straight ones).

### Seeding parameters

Set in the Import wizard's *Fibers* step (GUI) or [`set_fiber_seed`](Python-API):

| Parameter | Meaning |
|---|---|
| `n_fibers` | number of seed points |
| `fiber_method` | `streamlines` (curved) or `axial` (straight) |
| `fiber_max_steps` | max integration steps per streamline |
| `fiber_seed_end` | which cap to seed from |
| `fiber_cluster_eps_mm` | cap-clustering radius (branch separation) |
| `fiber_cap_band_pct` | z-band near each extremum to search for caps |
| `fiber_min_rel_size_pct` | drop cap clusters below this relative size |
| `fiber_axial_normal_thresh` | how "axial" a facet normal must be to count as a cap |
| `fiber_auto_detect_branches` | auto-find branches vs use a fixed count |

See the [Configuration Reference](Configuration-Reference) for defaults.

---

## Fiber populations

A fiber isn't just a path — it has a **diameter** and a **fiber type**, which set its model and its
activation behaviour. golgi ships **biology-realistic population presets** so you don't have to guess
("what diameter for B-fibers again?"), each carrying a literature citation and per-branch composition:

- Per-type rows (**A-α, A-β, A-δ, B, C**) with mean/SD diameter, distribution (gaussian / lognormal),
  and fraction.
- Presets such as **cervical vagus (human / pig / rat)** and **recurrent laryngeal**, plus generic
  myelinated-only and unmyelinated-only sets.
- A live **KDE preview** of the diameter distribution before you commit, and a citation shown beneath
  the preset.

Diameter maps to a fiber model: myelinated types use **MRG** (via NEURON/PyFibers or the AxonML
surrogate); small unmyelinated **C** fibers use unmyelinated models (e.g. Tigerholm/Sundt). If a
preset includes C fibers but the selected model is myelinated-only, golgi flags them rather than
silently dropping them. The mechanics of turning a population into spikes and thresholds are on the
[Fiber Models & Activation](Fiber-Models-and-Activation) page.

> **Branch selectivity.** Because every fiber is labelled with its branch, recruitment and selectivity
> can be computed per branch — e.g. "recruit the cardiac branch while sparing the rest" — see
> [Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity).

---

## Topographic seeding (roadmap)

A topographic-atlas seeding mode (drawing seed locations from per-fascicle functional-group
distributions, e.g. Hammer/Settell vagus atlases) is described in the
[feature roadmap](https://github.com/CellularSyntax/golgi/blob/main/FEATURES.md); the current release
seeds across the cross-section (uniform / cap distribution) with the population presets above.

---

## References

The population presets carry the literature attributions shown in the UI (`pop_presets.py`):

- Soltanpour N, Santer RM (1996). *J Anat.* and Verlinden et al. (2016). *Auton Neurosci.* — human cervical vagus composition.
- Settell ME, et al. (2020). *J Neural Eng.* and Nicolai EN, et al. (2020). *J Neural Eng.* — pig cervical vagus fascicular organisation.
- Pelot NA, et al. (2020). Quantified morphology of the cervical and subdiaphragmatic vagus nerves of human, pig, and rat. *Front Neurosci* 14:1148. — rat cervical vagus + cross-species morphometry.
- Mu L, Sanders I (2009). *Anat Rec.* — recurrent laryngeal nerve.
- Thio BJ, Titus ND, Pelot NA, Grill WM (2024). Reverse-engineered models reveal differential membrane properties of autonomic and cutaneous unmyelinated fibers. *PLOS Comput Biol.* — unmyelinated autonomic fiber properties.

---

### See also
[Fiber Models & Activation](Fiber-Models-and-Activation) · [Meshing](Meshing) ·
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity) ·
[Validation](Validation) · [Configuration Reference](Configuration-Reference)
