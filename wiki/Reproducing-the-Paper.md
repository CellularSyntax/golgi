# Reproducing the Paper

The [`paper_figs/`](https://github.com/CellularSyntax/golgi/tree/main/paper_figs) directory holds the
scripts that regenerate every figure in the golgi papers, plus the tooling that packages the figure
simulations as replay-verified [study bundles](Reproducible-Study-Bundles) for the Zenodo deposit.

> **Where the data lives.** `paper_figs/io_paths.py` defines a shared `ROOT` and output layout
> (`out/figures/{png,pdf,svg}`, `out/tables`, `out/data`). The **git repository ships source only** —
> the multi-gigabyte meshes, FEM results, micro-CT cohort, and rendered figures live outside it. To
> regenerate figures you need the source data and a configured `ROOT` (or, for verification, the
> published study bundles below). `io_paths.save_fig(fig, name)` writes PNG+PDF+SVG for each figure.

---

## Figure → script map

The authoritative mapping is the output filename each script writes (docstring figure numbers in a few
files lag a mid-draft renumbering — trust the filenames) and the curated `KEEP_PAPERFIGS` allowlist in
`make_release_package.py`.

| Figure | Script(s) | Content · species |
|---|---|---|
| **Fig 2** | `fig3_setup.py`, `fig3_render.py` | Modeling setup & FEM solution (cuff, multi-region mesh, cuff-designer, Vₑ & \|E\| cross-sections, along-fiber Vₑ + activating function) · **swine** |
| **Fig 4** | `fig_validation_full.py` (assembles `comsol_validation_fig`, `validate_fig`, `fig_nrv`, `fig_bucksot`) | Integrated validation: analytic/COMSOL, dog-VNS, NRV LIFE, Bucksot |
| **Fig 5** | `fig_species.py` (+ `fig5_population.py`, `fig5_thresholds.py`, `fig06_selectivity.py`) | 7-panel per-species result · **swine** cervical vagus |
| **Fig 6** | `fig_species.py` (human half); `fig06_selectivity.py` | 7-panel per-species result · **human** cervical vagus |
| **Fig 7** | `rabbit_selectivity_fig.py` | Superior-cardiac-branch-selective stim, 4×5 ring cuff + position×config sweep · real-3-D **rabbit** |
| **Fig 8** | `new_human_selectivity_fig.py` | SCB-selective stim, 4×5 ring cuff + position×config sweep · real-3-D **human** |

**Supplements / tables:** `validate_fig.py` (→ `supp_foundations`, the S8 fiber-model foundations),
`validate_mrg.py` (MRG vs McIntyre discrete), `comsol_validation_fig.py` (→ `supp_comsol_validation`,
S15), `fig_supp_cohort.py` (cohort gallery), `cohort_table.py` + `gen_s1_table.py` (the S1 cohort
table). Rendering scripts (`render_*.py`) make the cel-shaded PyVista presentation renders.

## Per-figure pipelines

Each species result is produced by a sequence of pipeline-driver scripts that mesh, solve, seed
fibers, and sweep — for example the human SCB / Fig 8 chain:

```
new_human3d_prep → new_human3d_traj → new_human_mesh → new_human_fem
   → new_human_sweep → new_human_tripole_sweep_build → new_human_tripole_analyze
   → new_human_steer_opt → new_human_xsec_contours
```

with parallel `rabbit_*` (Fig 7) and `human_bundle_*` (Fig 6) chains. Threshold sweeps for the
selectivity panels run through `fig5_thresholds.py` (golgi's exact PyFibers/NEURON bisection path) and
the `run_*_tripole_sweep_thr.sh` drivers. Species covered across the paper: **swine, human, rabbit,
dog** (dog-VNS), and **cat** (NRV LIFE).

## Packaging for Zenodo

- **`make_study_bundles.py`** packages the Fig 4–8 simulations as hashed, **replay-verified** `.golgi`
  bundles (each re-checked byte-for-byte via `golgi.projects.replay`) into
  `paper_figs/out/study_bundles/` with a `BUNDLES.json` + `CHECKSUMS.sha256`. Bundle IDs:
  `fig04a_dogvns_validation`, `fig04c_bucksot_validation`, `fig05_swine_cervical_vagus`,
  `fig06_human_cervical_vagus`, `fig07_rabbit_branching`, `fig08_human_scb_branching`
  (`fig04b` NRV-LIFE ships cached figure data, as the synthetic LIFE geometry isn't re-meshable).

  ```bash
  python paper_figs/make_study_bundles.py            # all figures
  python paper_figs/make_study_bundles.py fig07       # one figure
  ```

- **`make_release_package.py`** assembles the full paper deliverable (both manuscripts, the curated
  code subset, the reproduction bundles + checksums, and a top-level README). The `KEEP_PAPERFIGS`
  list in it (≈lines 53–95) is the definitive index of which script produces which figure.

---

### See also
[Validation](Validation) · [Reproducible Study Bundles](Reproducible-Study-Bundles) ·
[Pipeline Overview](Pipeline-Overview) · [License & Citation](License-and-Citation)
