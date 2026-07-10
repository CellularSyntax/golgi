# Installation

golgi runs on **Linux, macOS, and Windows**. Its compiled scientific core — **FEniCSx/DOLFINx**
(FEM) and, for biophysical thresholds, **NEURON** — is installed with **conda/mamba**, because those
are not portably `pip`-installable. **Everything else golgi needs** (the Trame GUI, PyVista/VTK
visualization, meshing helpers, figure export, the auth database, …) is declared as ordinary PyPI
dependencies and is installed **automatically** by `pip install -e .` — you no longer hand-install
those one by one.

> **TL;DR**
> ```bash
> mamba create -n golgi -c conda-forge python=3.12 fenics-dolfinx
> mamba activate golgi
> pip install -e .            # installs golgi + all of its PyPI dependencies
> ```
> This gives a working GUI + FEM install. For NEURON-based activation thresholds, see Step 2.

---

## Prerequisites

- **conda** or (recommended) **[mamba](https://mamba.readthedocs.io/)** / miniforge.
- A C/C++ toolchain (for the FFCx form-compiler JIT). macOS: **Xcode command-line tools**
  (`xcode-select --install`; golgi auto-detects `SDKROOT`). Windows: the **MSVC Build Tools**.
- *(optional)* an **NVIDIA GPU + CUDA** for the AxonML high-throughput backend.

## Step 1 — conda environment with the compiled scientific core

```bash
mamba create -n golgi -c conda-forge python=3.12 \
    fenics-dolfinx gmsh python-gmsh pyvista vtk meshio h5py
mamba activate golgi
```

DOLFINx brings its companions (`basix`, `ffcx`, `ufl`, `petsc4py`, `mpi4py`). Getting
`gmsh`/`pyvista`/`vtk`/`meshio`/`h5py` from **conda-forge** (rather than PyPI) gives the
best-integrated binaries; because golgi's dependencies are unpinned, `pip install -e .` in Step 3
detects these and **skips** them instead of pulling a second copy from PyPI.

## Step 2 — NEURON fiber-biophysics backend *(optional but recommended)*

The GUI and FEM run without NEURON; only **activation-threshold** computation needs it.

```bash
mamba install -c conda-forge neuron        # recommended
# or:  pip install neuron                   # Linux/macOS (and recent Windows wheels)
```

`pyfibers` (the default NEURON/MRG backend) is then pulled by golgi's `neuron` extra in Step 3. The
optional **AxonML** GPU surrogate is *not* bundled and is obtained separately
([see below](#axonml-gpu-backend)).

## Step 3 — install golgi (and its dependencies)

From the repository root:

```bash
pip install -e ".[neuron]"     # golgi + all PyPI deps + pyfibers (NEURON backend)
# or, GUI/FEM only (no thresholds):
pip install -e .
```

This installs the `golgi` package, the top-level `cuff_designer` module, the **`golgi` console
command**, and every remaining PyPI dependency automatically — Trame (`trame`, `trame-vtk`,
`trame-vuetify`), `SQLAlchemy` + `bcrypt` (auth/audit), `aiohttp`, `tetgen`, `trimesh`, `pymeshfix`,
`pyacvd`, `optimesh`, `mapbox_earcut`, `plotly` + `kaleido`, `matplotlib`, `scikit-image`,
`SimpleITK`, `opencv-python`, `tifffile`, and the NumPy/SciPy runtime. Anything already provided by
conda in Step 1 is left untouched.

---

## Verify the install

```bash
# 1) the package imports and the API is reachable
python -c "import golgi; print(golgi.Study)"

# 2) fast, solver-free tests (run anywhere)
pytest tests/test_headless_api.py -m "not integration"
pytest tests/test_recording_cable.py -q

# 3) launch the GUI (then open the printed URL)
golgi

# 4) full end-to-end smoke on a synthetic nerve (needs the solver stack)
python examples/recruitment_sweep.py
```

If step 4 runs to completion and writes `recruitment_curve.png` plus `vagus_study.golgi`, your
install is complete. See [Getting Started](Getting-Started) for what to do next.

---

## Optional extras

### AxonML GPU backend
AxonML is a proprietary, non-commercial Duke University package (not open source) — golgi never
bundles it. If you have separately obtained and accepted its license, install it into the environment
(`pip install -e ".[gpu]"` pulls the supporting `torch`); golgi loads AxonML through a guarded hook
when you select that backend. A CUDA-capable GPU is required for the speedup (it falls back to CPU
otherwise). See [Fiber Models & Activation](Fiber-Models-and-Activation) and
[License & Citation](License-and-Citation).

### Promptable segmentation (MedSAM2 / SAM)
The [segmentation](Geometry-Import-and-Segmentation) panel uses a MedSAM2/SAM-style model when a
checkpoint is available (point it at one via the documented env var / cache path). **Without a model,
golgi falls back to a stub segmenter** so the rest of the app still works — you can also import nerve
surfaces (STL/NAS/OBJ) or pre-built bundles directly and skip segmentation entirely.

---

## Reproducing an exact environment

[`requirements-frozen.txt`](https://github.com/CellularSyntax/golgi/blob/main/requirements-frozen.txt)
is a **pinned version snapshot** of a known-good environment (`name==version` for every package, no
local paths). It is a diagnostic reference, **not** a one-shot installable lock file: the compiled
scientific core (FEniCSx/DOLFINx, PETSc/SLEPc, MPI, NEURON) comes from conda, not PyPI, so those
lines cannot simply be `pip install`-ed. Known-good major versions: DOLFINx 0.10, NEURON 9.0.1,
PyFibers 0.8.5, Gmsh 4.15.2, TetGen 0.8.4, PyVista 0.48, Trame 3.12, NumPy 2.4, SciPy 1.17. For
byte-level reproduction of a *study*, prefer a
**[study bundle](Reproducible-Study-Bundles)** (it carries the version + frozen deps and is
`golgi replay`-verifiable); capture whole environments with `conda env export`.

---

## Platform notes & gotchas

- **macOS FFCx JIT** — if form compilation fails, ensure the Xcode CLT are installed; golgi sets
  `SDKROOT` automatically when it can.
- **macOS / single-rank MPI** — pin the libfabric provider to TCP (`FI_PROVIDER=tcp` /
  `GOLGI_FI_PROVIDER=tcp`) to avoid an OFI finalize crash.
- **Windows / NEURON** — the GUI and FEM install cleanly; NEURON is the one component that is easiest
  on Linux/macOS. If `pip install neuron` has no matching wheel for your Python, use
  `mamba install -c conda-forge neuron` or run threshold computations on Linux/WSL.
- **conda + pip OpenCV** — if a conda `opencv` is already present, pip may still add `opencv-python`
  (different distribution name) as a second copy; harmless in practice, but if `cv2` misbehaves,
  install OpenCV from a single source.
- **MPI launcher** — override the launcher with `GOLGI_MPIRUN` (`mpirun` / `mpiexec` / `srun`) on
  clusters.

More fixes are on the [Troubleshooting & FAQ](Troubleshooting-and-FAQ) page.

---

### See also
[Getting Started](Getting-Started) · [Configuration Reference](Configuration-Reference) ·
[Troubleshooting & FAQ](Troubleshooting-and-FAQ) · [License & Citation](License-and-Citation)
