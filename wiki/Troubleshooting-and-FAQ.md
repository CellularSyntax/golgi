# Troubleshooting & FAQ

Fixes for the issues most likely to bite, plus answers to common questions. See also
[Installation](Installation) and the [Configuration Reference](Configuration-Reference).

---

## Installation & environment

**`import dolfinx` / `import gmsh` fails.** The scientific stack must come from conda, not pip. Recreate
the environment per [Installation](Installation) (`mamba create -n golgi -c conda-forge fenics-dolfinx
python=3.12`) and install golgi into *that* environment.

**`pip install -e .` didn't pull Trame/PyVista/NEURON.** Correct — golgi intentionally does **not**
declare the heavy stack as pip dependencies (FEniCSx/NEURON aren't portably pip-installable, and
pinning them via pip can upgrade NumPy and break DOLFINx). Install the stack first (conda + the pip
extras in [Installation](Installation)), then `pip install -e .`.

**macOS: FFCx form compilation fails / `SDKROOT` errors.** Install the Xcode command-line tools
(`xcode-select --install`). golgi auto-detects and sets `SDKROOT` when it can; set it manually if your
SDK is in a non-standard location.

**macOS / single-rank: an OFI / libfabric crash on solver finalize.** Pin the provider to TCP:
`export FI_PROVIDER=tcp` (or `GOLGI_FI_PROVIDER=tcp`).

**Cluster: wrong MPI launcher.** Override it with `GOLGI_MPIRUN` (`mpirun` / `mpiexec` / `srun`).

## Geometry & meshing

**TetGen fails / "facets intersect" / degenerate cross-section.** golgi assumes a **watertight** input
surface and doesn't do in-app mesh repair. Repair non-manifold/self-intersecting surfaces in MeshLab
or Blender first. A perfectly concentric cylinder can also fail (coincident facets the exact
predicates reject) — real, slightly irregular nerves mesh fine; the bundled synthetic test nerve adds
gentle bumps for exactly this reason. See [Meshing](Meshing).

**The mesh looks under-refined near the contacts.** Lower `lc_contact_um` and `lc_endo_um` (contacts
and endoneurium dominate field accuracy); check the per-region quality histograms in the Mesh drawer.

## Segmentation

**Segmentation only returns crude/placeholder masks.** No MedSAM2/SAM checkpoint is installed, so
golgi is using the **stub segmenter**. Install a model checkpoint (see [Installation](Installation)),
or import a nerve surface (STL/NAS/OBJ) / pre-built bundle directly and skip segmentation.

## Running studies

**`Study.create(...)` raises `FileExistsError`.** It refuses to clobber a non-empty directory. Use a
fresh path, delete the existing one, or attach with `Study.open(...)`.

**An amplitude sweep is slower than expected.** An `I_stim`/amplitude axis should **not** re-solve the
FEM (it rescales the cached field). If it is re-solving, your sweep axis is touching σ, the mesh,
electrode geometry, or fiber paths — those legitimately re-enter the upstream stage. See
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity).

**AxonML backend isn't available / runs on CPU.** AxonML is an optional, separately-licensed Duke
package golgi never bundles; without it (and a CUDA GPU) use the default `pyfibers` backend. See
[Fiber Models & Activation](Fiber-Models-and-Activation).

**`golgi: command not found`.** The console script is created by `pip install -e .`. Until then use
`python -m golgi.app …`. Ensure the conda environment is active.

## Reproducibility

**`golgi replay` reports a stage diverged.** A file's hash doesn't match the manifest — the bundle was
modified, or it was produced by a different environment. The report names the stage and file; see
[Reproducible Study Bundles](Reproducible-Study-Bundles).

**The `paper_figs/` scripts can't find their data.** Those scripts read from a `ROOT` outside the git
repo (the data isn't shipped in the repository). Point `io_paths.ROOT` at your data, or verify results
from the published study bundles instead. See [Reproducing the Paper](Reproducing-the-Paper).

---

## FAQ

**Do I need COMSOL or any commercial software?** No. golgi's solver/mesher stack (FEniCSx, TetGen,
Gmsh) is fully open. The only optional non-open piece is the AxonML GPU backend, which is never
required.

**Does it run on Windows?** The supported platforms are Linux and macOS (the FEniCSx/NEURON stack is
most reliable there). WSL2 is the practical route on Windows.

**Can I use the GUI and scripts on the same project?** Yes — they share the same on-disk `Study`
state. Build in the browser, continue in a script, or vice-versa.

**Where do outputs go?** Everything a study produces lives under the **project directory** — never
`~/.cache` or `/tmp`. That's what makes bundles and figure export self-contained.

**Is there a GPU requirement?** No. A CUDA GPU only accelerates the optional AxonML backend.

---

### See also
[Installation](Installation) · [Configuration Reference](Configuration-Reference) ·
[Glossary](Glossary) · [Getting Started](Getting-Started)
