# Getting Started

This page runs your **first end-to-end study** two ways — by script and in the GUI — and points you
to the deeper pages for each stage. If you haven't installed golgi yet, do
[Installation](Installation) first.

---

## Option A — your first study in Python (5 minutes)

The fastest way to see the whole pipeline is the bundled example, which runs on a **built-in
synthetic nerve** (no data needed):

```bash
python examples/recruitment_sweep.py
```

It loads a nerve, meshes it, places a bipolar cuff, solves the anisotropic field, generates fiber
trajectories, sweeps stimulus amplitude, writes a recruitment-curve PNG, and exports a reproducible
study bundle. To run on your own surface:

```bash
python examples/recruitment_sweep.py --nerve /path/to/nerve.stl --project ./my_study
```

### What the script does, step by step

```python
import golgi
from golgi.jobs.schemas import SweepRequest

s = golgi.Study.create("vagus_study")          # a project dir on disk

s.import_nerve("nerve.stl")                     # 1. load a surface (STL/NAS/OBJ, mm)
s.set_mesh(use_epi=True, perineurium_ci=True)   # 2a. mesh + material options
s.set_electrodes([{                             # 2b. design a cuff
    "name": "Bipolar @ centre",
    "electrode_type": "bipolar ring-pair",
    "cuff_anchor": "centroid",
}])
s.run_mesh()                                    # 3. multi-region TetGen mesh
s.run_fem()                                     # 4. anisotropic FEM (lead fields)
s.set_fiber_seed(n_fibers=100,
                 fiber_method="streamlines")    # 5a. fiber-generation params
s.run_fibers()                                  # 5b. curved 3-D trajectories
result = s.run_sweep(SweepRequest(              # 6. recruitment across amplitude
    mode="recruitment",
    amplitudes_mA=[0.1, 0.25, 0.5, 1.0, 2.0]))
s.export_bundle("vagus_study.golgi")            # 7. hashed, replayable bundle
s.close()
```

Each numbered step maps to a [pipeline stage](Pipeline-Overview) and a wiki page:

| Step | Method | Learn more |
|---|---|---|
| 1 | `import_nerve` | [Geometry Import & Segmentation](Geometry-Import-and-Segmentation) |
| 2a | `set_mesh` | [Meshing](Meshing) · [Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties) |
| 2b | `set_electrodes` | [Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer) |
| 4 | `run_fem` | [Finite-Element Solver](Finite-Element-Solver) |
| 5 | `run_fibers` | [Fiber Populations & Trajectories](Fiber-Populations-and-Trajectories) |
| 6 | `run_sweep` | [Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity) · [Fiber Models & Activation](Fiber-Models-and-Activation) |
| 7 | `export_bundle` | [Reproducible Study Bundles](Reproducible-Study-Bundles) |

Full method reference: [Python API](Python-API).

### Verify and share

```bash
golgi replay vagus_study.golgi      # byte-for-byte reproducibility check
```

Hand that one file to a colleague; `golgi import` + `golgi replay` reconstitutes and verifies the
whole study. See [Command-Line Interface](Command-Line-Interface).

---

## Option B — your first study in the GUI

```bash
golgi          # open the printed local URL in a browser
```

Then work top to bottom through the navbar:

1. **Sign in / Create project** on the welcome screen.
2. **File ▸ Import Nerve** — a 4-step wizard: *Load nerve* → *Endoneurium* → *Fibers* → *Muscle*.
   (Or **File ▸ Segment µCT slice…** to segment an image first.)
3. **Designs** — add a cuff, pick an electrode type, set per-contact polarities (or open the
   **Cuff Designer** for ASCENT-style presets).
4. **Mesh** — set per-region element sizes and **Build mesh (TetGen)**.
5. **Materials** — set tissue conductivities (with the Cole–Cole evaluator) and **Update**.
6. **Simulate ▸ Extracellular field (FEM)** — pick a solver preset and **Run FEM solve**.
7. **Simulate ▸ Single fiber / Fiber population / Sweep** — compute thresholds and recruitment.
8. **Export ▸ Export figures / Export report**, and **File ▸ Export study** to share.

The complete tour, with every panel and control, is in the [GUI Walkthrough](GUI-Walkthrough).

> Because the GUI and API share the same on-disk `Study` state, you can build a project in the
> browser and continue it from a script — or vice-versa.

---

## Where to go next

- **Understand the model** → [Pipeline Overview](Pipeline-Overview)
- **Design electrodes** → [Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer)
- **Scale up** → [Headless / HPC](Headless-and-HPC)
- **Reproduce the paper** → [Reproducing the Paper](Reproducing-the-Paper)
- **Hit a snag** → [Troubleshooting & FAQ](Troubleshooting-and-FAQ)
