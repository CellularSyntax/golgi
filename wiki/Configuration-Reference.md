# Configuration Reference

A reference for the parameters and environment variables that control golgi. GUI controls, the
[`Study`](Python-API) setters, and the project's `ui_state.json` all write the same state keys.

---

## Mesh parameters

Set via [`Study.set_mesh(**kwargs)`](Python-API) or the **Mesh** drawer. See [Meshing](Meshing).

| Key | Meaning |
|---|---|
| `use_epi` | generate/include an epineurium shell |
| `epi_thickness_um` | epineurium shell thickness (µm) |
| `lc_endo_um` | endoneurium element size (µm) |
| `lc_epi_um` | epineurium element size (µm) |
| `lc_muscle_um` | muscle / far-field element size (µm) |
| `lc_scar_um` | encapsulation/scar element size (µm) |
| `lc_saline_um` | cuff saline element size (µm) |
| `lc_silicone_um` | cuff silicone element size (µm) |
| `lc_contact_um` | electrode contact element size (µm) |
| `decim_target_k` | imported-surface decimation target (thousands of triangles) |
| `muscle_radial_pad_mm`, `muscle_axial_pad_mm` | far-field bath padding (mm) |
| `muscle_dx_mm`, `muscle_dy_mm`, `muscle_dz_mm` | far-field bath offsets (mm) |

## Conductivities

Anisotropic, frequency-aware tissue properties — full table and the Cole–Cole / IT'IS / perineurium
model are on [Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties). Keys include
`sigma_endo`, `sigma_endo_long`, `sigma_epi`, `sigma_muscle`, `sigma_muscle_long`, `sigma_saline`,
`sigma_silicone`, `sigma_contact`, `sigma_scar`, and the perineurium options `perineurium_ci`,
`sigma_peri`, `peri_thk_m`, `perineurium_species`.

## Electrode / cuff

Set via [`Study.set_electrodes([...])`](Python-API) or the **Designs** drawer; full parameter lists per
electrode type are on [Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer). Common keys:
`electrode_type`, `name`, `eid`, `cuff_anchor` (`trunk`/`branched`/`centroid`), `cuff_offset_mm`,
`L_cuff_mm`, `cuff_clearance_mm`, `cuff_wall_mm`, `contact_polarities`, `contact_current_fractions`.

## Fiber seeding

Set via [`Study.set_fiber_seed(**kwargs)`](Python-API) or the Import wizard's *Fibers* step. See
[Fiber Populations & Trajectories](Fiber-Populations-and-Trajectories).

| Key | Meaning |
|---|---|
| `n_fibers` | number of seed points |
| `fiber_method` | `streamlines` (curved) or `axial` (straight) |
| `fiber_max_steps` | max streamline integration steps |
| `fiber_seed_end` | which cap to seed from |
| `fiber_cluster_eps_mm` | cap-clustering radius (branch separation) |
| `fiber_cap_band_pct` | z-band near each extremum searched for caps |
| `fiber_min_rel_size_pct` | minimum relative cap-cluster size to keep |
| `fiber_axial_normal_thresh` | "axialness" threshold for cap facets |
| `fiber_auto_detect_branches` | auto-find branches vs fixed count |

## FEM / simulation

Set in the **Simulate ▸ Extracellular field (FEM)** panel. See [Finite-Element Solver](Finite-Element-Solver).

| Control | Meaning |
|---|---|
| stimulus current | total injected current (mA) |
| solver preset | `Quick` / `Balanced` / `HPC` (Krylov tolerance + AMG tuning) |
| configs to solve | which polarity configs to solve |
| compute access impedance | run the DC dual-solves for Z_access |

## Sweeps

`SweepRequest` (`golgi/jobs/schemas.py`) — `mode` (`recruitment`/`threshold`), `amplitudes_mA`,
`bisect_lo_mA`/`bisect_hi_mA`/`bisect_tol_uA`, `fiber_indices`, `branch_filter`,
`fiber_type_filter`, `pulse_params`, `backend` (`pyfibers`/`axonml`), `model_name`. See
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity).

---

## Environment variables

| Variable | Purpose |
|---|---|
| `GOLGI_EMIT_IMPEDANCE` | enable DC access-impedance dual-solves (default on) |
| `GOLGI_EMIT_RECORDING` | emit per-contact reciprocity lead fields for [CNAP](Recording-and-CNAP) (default off) |
| `GOLGI_RECIP_CI` | use the two-field perineurium contact-impedance solver |
| `GOLGI_GROUND_OUTER_FALLBACK` | ground the full exterior when no muscle bbox is present |
| `GOLGI_FEM_RUNNER` | `slurm` to dispatch FEM to a cluster (else local) |
| `GOLGI_SLURM_PARTITION` / `_CPUS` / `_MEM_GB` / `_TIME` / `GOLGI_SBATCH` | SLURM runner config / `sbatch` path (or the fake-sbatch shim) |
| `GOLGI_MPIRUN` | MPI launcher (`mpirun` / `mpiexec` / `srun`) |
| `GOLGI_FI_PROVIDER` / `FI_PROVIDER` | libfabric provider — pin to `tcp` on single-rank macOS |
| `SDKROOT` | macOS SDK path for the FFCx JIT (auto-detected if unset) |
| `GOLGI_TEST_NERVE` / `GOLGI_DEMO_STL` | override the test/example nerve geometry |

---

### See also
[Meshing](Meshing) · [Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties) ·
[Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer) · [Finite-Element Solver](Finite-Element-Solver) ·
[Troubleshooting & FAQ](Troubleshooting-and-FAQ)
