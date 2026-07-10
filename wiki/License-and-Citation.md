# License & Citation

## License

golgi is free and open-source software under the **GNU Affero General Public License, version 3 or
later (AGPL-3.0-or-later)** — see [LICENSE](https://github.com/CellularSyntax/golgi/blob/main/LICENSE).

This is **required by the dependency stack**, not an arbitrary choice:

- **TetGen** (the volume mesher core compiled into the `tetgen` wheel) is **AGPL-3.0** (dual-licensed;
  a commercial license is available from WIAS). Linking it makes the combined work AGPL — this is the
  decisive reason golgi is AGPL.
- **Gmsh** (`import gmsh`) is **GPLv2-or-later**, upgradeable to AGPLv3 — compatible.
- golgi serves a **Trame** browser GUI, so the AGPL **§13 "Remote Network Interaction"** clause applies
  to networked use.

Full inventory and verified per-package licenses are in
[THIRD_PARTY_LICENSES.md](https://github.com/CellularSyntax/golgi/blob/main/THIRD_PARTY_LICENSES.md).

### Compatibility flags (confirm with counsel before redistribution)

1. **PyFibers is GPL-2.0-only** (no "or later"), which is not one-way compatible with AGPLv3. golgi
   mitigates this by running the PyFibers→NEURON fiber step and the TetGen mesher in **separate
   subprocesses that exchange data through files** (an argument for mere aggregation, not one linked
   work). PyFibers also carries a non-commercial field restriction.
2. **AxonML is proprietary** (Duke OTC T-008477; non-commercial/academic only) — **not** open source
   and **not** AGPL-compatible. golgi **never bundles or redistributes it**; it is an optional plug-in
   loaded only if the user has separately obtained it and accepted Duke's terms. The default
   NEURON/PyFibers path needs no such backend.

### Commercial / closed-source redistribution

golgi imposes no restriction beyond the AGPL terms. But because both meshers are copyleft, shipping
golgi inside a **closed-source or commercial** product additionally requires separate commercial
licenses for **TetGen (WIAS)** and **Gmsh**, plus resolution of the PyFibers/AxonML field-of-use terms
above.

### Data

Data released alongside golgi — the micro-CT imaging datasets and the golgi study bundles on Zenodo —
is licensed separately under **CC-BY-4.0**.

---

## Citation

If you use golgi in your research, please cite:

- the **methods paper** — *PLOS Computational Biology* (in review);
- the **companion software paper** — *SoftwareX* (in review);
- the archived **software release** on Zenodo (DOI assigned on first release).

A machine-readable `CITATION.cff` is added when the first release is tagged. Until then, please cite
the repository (`https://github.com/CellularSyntax/golgi`) and note the version/commit you used (or,
better, the [study bundle](Reproducible-Study-Bundles), which records the exact version).

Please also acknowledge the upstream tools golgi builds on — FEniCSx/DOLFINx, PETSc, TetGen, Gmsh,
NEURON/PyFibers, PyVista, Trame — and the IT'IS tissue database.

---

### See also
[Reproducible Study Bundles](Reproducible-Study-Bundles) · [Reproducing the Paper](Reproducing-the-Paper) ·
[Contributing](Contributing)
