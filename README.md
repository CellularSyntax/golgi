# golgi

**Anatomically realistic, reproducible image-to-population modeling of
peripheral nerve stimulation.**

golgi builds branch-resolved 3D peripheral-nerve models from images and
computes the stimulated fiber population end to end — image segmentation (or
surface/mask import) → automated multi-region tetrahedral meshing →
anisotropic finite-element solution of the extracellular field with explicit
perineurium contact impedance → realistic fiber populations with straight or
physically curved 3D trajectories → biophysical activation thresholds via
interchangeable NEURON (PyFibers) and GPU surrogate (AxonML) backends. The same
pipeline is exposed through a Trame graphical interface, a scriptable Python
API, and a command-line interface, and every study exports as an
integrity-hashed, byte-verifiable bundle.

## Usage

```python
import golgi
s = golgi.Study.create("/tmp/demo_project")
s.import_nerve("data/sample_nerve.stl")
# ... build mesh, solve field, populate fibers, run thresholds ...
s.export_bundle("/tmp/demo.golgi.zip")
```

Launch the GUI with `python -m golgi.app`. See `FEATURES.md` for the full
feature surface.

## License

golgi is free and open-source software, licensed under the **GNU Affero
General Public License, version 3 or later (AGPL-3.0-or-later)** — see
[LICENSE](LICENSE).

This license is required by golgi's dependency stack: golgi links **Gmsh**
(GPLv2-or-later) and **TetGen** (AGPL-3.0, via the `tetgen` module, which
compiles the TetGen core into its extension) as libraries, and it serves a
**Trame** browser GUI, so the AGPL network-use clause (§13) applies. See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full dependency
license inventory and compatibility notes.

No restriction is imposed beyond the AGPL terms. Note, however, that
redistributing golgi inside a **closed-source or commercial** product
additionally requires separate commercial licenses for TetGen (from WIAS) and
Gmsh, since both are copyleft.

**Data** released alongside golgi — the micro-CT imaging datasets (NIH SPARC)
and the golgi study bundles (Zenodo) — is licensed separately under
**CC-BY-4.0**.

## Citing golgi

If you use golgi, please cite the methods paper (PLOS Computational Biology, in
review) and the companion software paper (SoftwareX, in review). See the
release deposit on Zenodo for the archival DOI.
