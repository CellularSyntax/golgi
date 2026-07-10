# Finite-Element Solver

golgi computes the extracellular potential **Vₑ(x)** produced by a stimulating cuff on a fully open
finite-element stack — **FEniCSx / DOLFINx** with a **PETSc** backend — with **no COMSOL or other
commercial solver** in the loop. This page describes the formulation; see
[Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties) for the material model and
[Meshing](Meshing) for how the domain is discretized.

Solver core: `golgi/compute/solve_nerve.py` (single-field) and `golgi/compute/solve_nerve_ci.py`
(two-field perineurium contact-impedance). The pipeline driver is `golgi/pipeline/fem.py`.

---

## Governing equation

Stimulation is in the **quasi-static (electro-quasistatic) regime**: at the relevant frequencies the
tissue is resistive and the potential satisfies the current-conservation Laplace equation with a
spatially varying, anisotropic conductivity tensor **σ**:

$$\nabla \cdot \big(\boldsymbol{\sigma}\,\nabla V_e\big) = 0 \quad\text{in } \Omega$$

Current is injected through electrode contacts (Neumann flux), one or more contacts are held at a
reference potential (Dirichlet), and the remaining outer boundaries are insulating (homogeneous
Neumann).

| Boundary | Condition | Meaning |
|---|---|---|
| Active contact | Neumann, `−σ∇Vₑ·n = Jₙ` | Injected current density (sign by polarity) |
| Reference contact / outer ground | Dirichlet, `Vₑ = 0` | Potential reference |
| Insulating surfaces | Neumann, `−σ∇Vₑ·n = 0` | Silicone, exterior — no current crosses |

---

## Discretization

- **Library:** DOLFINx (FEniCSx) with PETSc linear algebra.
- **Elements:** continuous Lagrange (CG1) on a tetrahedral mesh.
- **Conductivity field:** piecewise-constant (DG0) rank-2 tensor per cell, so each tissue region —
  and the longitudinal/transverse split for anisotropic regions — is represented exactly per element.
- **Weak form:** `a(u,v) = ∫_Ω σ∇u·∇v dx`, with contact current contributing the linear functional.
- **Solver:** preconditioned Krylov (CG) with **Hypre BoomerAMG** algebraic multigrid.

### Solver presets

The FEM panel exposes three named PETSc configurations (a tolerance/quality trade-off):

| Preset | Krylov tolerance | Use |
|---|---|---|
| **Quick** | loose (≈`1e-4`), capped iterations | fast sanity checks |
| **Balanced** *(default)* | tight (≈`1e-10`) | production runs |
| **HPC** | `1e-8` with BoomerAMG tuned for large/anisotropic problems (stronger coarsening) | 10–20 M+ element meshes |

On multi-rank (MPI) runs, only rank 0 writes outputs, avoiding file corruption.

---

## Multi-contact montages by superposition

Because the quasi-static problem is **linear in injected current**, multi-contact configurations are
assembled by superposition rather than re-meshing. Each contact carries a **polarity**
(`anode` / `cathode` / `ground` / `off`) and an optional **current fraction**; the driver computes
the per-patch injected current as

```
I_patch = I_stim · sign(polarity) · (current_fraction or 1/N_in_group)
```

with charge conservation enforced across the montage (Σ I_anode = −Σ I_cathode = I_stim). This is
what makes **current steering** and arbitrary tri-/quadri-/N-polar configurations possible from a
single solve basis. See [Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer) for how
polarities and fractions are set.

The same linearity underlies the **amplitude-sweep shortcut**: an `I_stim` sweep rescales the cached
per-fiber Vₑ instead of re-solving the FEM (see
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity)).

---

## What the solver writes

For each configuration the solver samples and stores the extracellular field where downstream stages
need it:

- **`paths_Ve.npz`** — Vₑ sampled on every fiber-trajectory node (the input to fiber activation).
- **`axis_line.npz`** — Vₑ along the cuff axis (for the analysis axis plot).
- **`slice_volume.npz`** — Vₑ on a cross-sectional slice (heatmaps).
- **`nerve_surface_Ve.npz`** — Vₑ on the nerve surface (3-D overlays).
- **`Ve.xdmf` / `Ve.h5`, `E.xdmf` / `E.h5`** — full potential and E-field fields for visualization.

---

## Electrode access impedance (optional)

When **Compute electrode access impedance** is enabled (GUI checkbox, or `emit_impedance`), the
solver runs additional **Dirichlet dual-solves** — per contact, and per anode→cathode pair — and
integrates the current to report an access impedance (Z_access), written to `impedance.json`:

```json
{"per_contact": [{"id": 0, "Z_ohm": 1234.5}, ...],
 "per_pair":    [{"id_a": 0, "id_b": 1, "Z_ohm": 1100.2}, ...]}
```

The integration is taken at the outer-muscle ground surface (rather than the contact facet) to avoid
σ-contrast ambiguity at the metal interface. These values feed the impedance figures and can be
overlaid against measured impedance.

---

## Reciprocity / lead fields for recording

For [recording / CNAP](Recording-and-CNAP), the solver can emit **per-contact unit-current lead
fields** (reciprocity solves): inject 1 A at each recording contact with the outer boundary grounded,
sample Vₑ on the fiber nodes, and cache one `V_e_rec_<id>.npz` per contact. Any bipolar/multipolar
recording montage is then a linear combination of these cached lead fields. Enabled via
`GOLGI_EMIT_RECORDING=1`; results are fingerprinted on (mesh, σ, patch geometry, solver preset) so
they are reused until something changes.

---

## Environment flags

| Variable | Purpose |
|---|---|
| `GOLGI_EMIT_IMPEDANCE` | Enable/disable the DC access-impedance dual-solves (default on). |
| `GOLGI_EMIT_RECORDING` | Emit per-contact reciprocity lead fields for CNAP (default off). |
| `GOLGI_RECIP_CI` | Use the two-field perineurium contact-impedance solver (see below). |
| `GOLGI_GROUND_OUTER_FALLBACK` | Ground the full exterior boundary when no muscle bbox is present. |
| `GOLGI_MPIRUN` | MPI launcher binary (`mpirun` / `mpiexec` / `srun`). |
| `GOLGI_FI_PROVIDER` / `FI_PROVIDER` | libfabric provider; pin to `tcp` to avoid an OFI crash on single-rank macOS runs. |
| `SDKROOT` | macOS SDK path for the FFCx JIT form compiler (auto-detected if unset). |

See [Troubleshooting & FAQ](Troubleshooting-and-FAQ) for the macOS/MPI gotchas, and the
[Configuration Reference](Configuration-Reference) for the full list.

---

## Perineurium contact impedance (two-field solve)

The perineurium is a thin, highly resistive sheath whose voltage drop strongly shapes fascicular
recruitment, but meshing it as a µm-scale volume is expensive. golgi instead models it as a
**contact-impedance sheet** with an area-specific sheet resistance `Rs = peri_thickness / σ_peri`,
coupled through a **two-field Robin block** between the endoneurium field and the surrounding field
across the endo↔epi interface Γ. Perineurium thickness scales with fascicle diameter using
species-specific coefficients (rat / pig / human, after Pelot et al. 2019), and σ_peri ≈ 1/1149 S/m.

This path lives in `solve_nerve_ci.py` and is gated on `GOLGI_RECIP_CI=1` together with
`perineurium_ci=True` and a positive perineurium thickness. Turning it on attenuates the raw field
seen by fibers (a tripolar selectivity index in one internal study dropped from ≈0.47 to ≈0.19 with
CI on), so it materially changes thresholds and selectivity. Details and the conductivity model are
on the [Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties) page.

---

## References

- Bossetti CA, Birdno MJ, Grill WM (2008). Analysis of the quasi-static approximation for calculating potentials generated by neural stimulation. *J Neural Eng* 5:44. — justifies the quasi-static formulation.
- Baratta IA, et al. (2023). DOLFINx: the next generation FEniCS problem solving environment. *Zenodo.* — the FEM library golgi solves on.
- Pelot NA, Thio BJ, Grill WM (2019). On the parameters used in finite element modeling of compound peripheral nerves. *J Neural Eng* 16:016007. — FE conductivities + perineurium contact-impedance parameters.
- Musselman ED, Cariello JE, Grill WM, Pelot NA (2021). ASCENT: a pipeline for sample-specific computational modeling of electrical stimulation of peripheral nerves. *PLOS Comput Biol.* — thin-layer perineurium approximation.
- Weerasuriya A, Spangler RA, Rapoport SI, Taylor RE (1984). AC impedance of the perineurium of the frog sciatic nerve. *Biophys J* 46:167. — perineurium conductivity.

See [Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties#references) for the full tissue-conductivity reference list and [Meshing](Meshing#references) for the mesher references.

---

### See also
[Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties) · [Meshing](Meshing) ·
[Fiber Models & Activation](Fiber-Models-and-Activation) ·
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity) ·
[Recording & CNAP](Recording-and-CNAP)
