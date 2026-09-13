# Supplementary analyses for the golgi JNE revision

Scripts and data that reproduce the four supplementary figures added during revision of

> Bhandari et al., *golgi: an open platform for image-to-recruitment modeling of
> peripheral nerve stimulation*, Journal of Neural Engineering (revised).

Each script is standalone, writes its figure into the working directory, and reads
its inputs from `data/` (override with `GOLGI_REV_DATA`). The large cross-solver
inputs live in `data_large/` (override with `GOLGI_REV_DATA_LARGE`) and are
distributed with the Zenodo record rather than this repository, because they total
about 95 MB.

| Script | Figure | What it shows |
| --- | --- | --- |
| `make_S16.py` | S16 | Cross-solver validation against COMSOL, extended from the extracellular field to activation thresholds (M2 idealized cuff) |
| `make_S17.py` | S17 | End-effect taper: the cosine window, and threshold sensitivity to electrode placement along the fiber |
| `make_S18.py` | S18 | Stimulus engine: GUI rectangular pulses, and arbitrary waveforms (quasi-trapezoidal, 10 kHz KHFAC) driven through the scripting API |
| `make_S19.py` | S19 | Cuff-fit sensitivity: clearance swept 0.05–0.50 mm crossed with both nerve section states |

## Running

```bash
python make_S16.py      # needs data_large/ (see below)
python make_S17.py
python make_S18.py
python make_S19.py      # needs golgi importable, for its deform module
```

`make_S19.py` imports `golgi.segmentation.deform` to apply golgi's own
area-preserving nerve-section transform. If golgi is not installed, point it at a
source tree with `GOLGI_REPO=/path/to/golgi`.

Requirements: `numpy`, `scipy`, `matplotlib`, and for `make_S19.py` the `golgi`
package itself.

## Regenerating the S19 sweep from scratch

`data/results/` holds the finished sweep, so `make_S19.py` needs no simulation.
To re-run the sweep itself, `sweep/run_cell.py` drives golgi's headless pipeline
end to end (mesh → fibers → finite-element solve → threshold sweep) for one
configuration:

```bash
python sweep/run_cell.py --deform none  --clearance 0.15 --fibers 200 \
    --min-rel-size-pct 0.3 --lc-scale 2.0 --lc-muscle 3000 --tag none_c015_n200
python sweep/run_cell.py --deform round --clearance 0.15 --fibers 200 \
    --min-rel-size-pct 0.3 --lc-scale 2.0 --lc-muscle 3000 --tag round_c015_n200
```

Eight configurations were run in total, crossing `--deform {none,round}` with
`--clearance {0.05,0.15,0.30,0.50}`. Each takes roughly 25 minutes on a laptop
(mesh ≈ 350 s, fibers ≈ 10 s, field solve ≈ 150 s, threshold sweep the remainder).
`sweep/mesh_probe.py` runs the meshing stage alone, which is useful when tuning
element sizes.

Mesh resolution was set to half the production value (`--lc-scale 2.0`,
`--lc-muscle 3000`) and held identical across all eight configurations, so the
comparison between them is internal. Geometry otherwise follows the production
settings of the reproduction bundles: 20 mm extruded nerve, 5.4 mm cuff, 3 × 4
contact array, 1.0 mm silicone wall, 8 mm radial and 10 mm axial muscle pads.

## Data

`data/`

- `r11_taper.npz` — taper placement sweep (S17)
- `r14_waves.npz` — waveform thresholds and membrane traces (S18)
- `r12_field_diag.npz`, `r12_thresholds.npz` — activating functions and per-fiber thresholds (S16)
- `nerve_xsec.json` — human cervical vagus cuff-plane section: epineurium outline and five fascicle polygons (S19)
- `results/*_n200.json` — the eight sweep configurations: thresholds, bore radius, fiber coordinates, timings
- `logs/*_n200.log` — solver logs. **Load-bearing:** `make_S19.py` parses the realised per-fascicle seed-vertex counts from these, which is how fibers are assigned to fascicles. Assignment comes from golgi's own seeding manifest, not from post-hoc point-in-polygon tests.
- `stl/nerve_{undeformed,rounded}.stl` — extruded nerve surfaces
- `r22_sweep_summary.json` — every statistic quoted in the S19 caption, including the power-law fit

`data_large/` (Zenodo record only)

- `eval_points.csv` — shared evaluation grid for the cross-solver comparison
- `golgi_Ve_VperA.csv` — golgi extracellular potentials at those points
- `M2_Results_from_comsol.txt`, `M2_Results_from_comsol_finer.txt` — COMSOL reference solutions at the standard and finer mesh settings

## Notes on the figures

No smoothing is applied to any potential or threshold. A light Gaussian is applied
before the numerical second derivative in the activating-function panels of S16,
identically to both solvers, and never to the potentials or thresholds.

In S19, fibers whose threshold exceeds the 10 mA bisection ceiling are recorded as
not activated. All ratios and correlations therefore use the matched fibers that
activate at every clearance, which is 47 of 74 traced fibers in the undeformed arm
and 41 of 68 in the rounded arm.

## License

Code is released under AGPL-3.0-or-later, matching golgi. Data files are released
under CC-BY-4.0, matching the accompanying Zenodo datasets.
