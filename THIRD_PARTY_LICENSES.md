# Third-party licenses

golgi is licensed under **AGPL-3.0-or-later** (see [LICENSE](LICENSE)). That
choice is *required* by its dependency stack: golgi links two copyleft meshers
as libraries and serves a browser GUI over the network.

- **Gmsh** is imported as a Python library (`import gmsh`) and is **GPLv2+**.
- **TetGen** is linked via the `tetgen` Python module (`import tetgen` →
  `tetgen.TetGen(...).tetrahedralize(...)`), which compiles the TetGen 1.6
  C/C++ core into its compiled extension. The TetGen core is **AGPL-3.0**
  (dual-licensed; a commercial license is available from WIAS).
- golgi serves a **Trame** browser GUI, so the AGPL §13 "Remote Network
  Interaction" clause applies to networked use.

Because an AGPL-3.0 component (TetGen) is linked into golgi, the combined work
is AGPL-3.0-or-later. Gmsh's GPLv2-**or-later** grant is upgradeable to
GPLv3/AGPLv3, so it is compatible.

The versions below are those pinned in the release environment
(`requirements-frozen.txt`). Licenses were read from the installed package
artifacts (dist-info METADATA and bundled LICENSE/NOTICE files), not assumed.

| Component | Version | License (verified) | How golgi uses it | Compatible with AGPL-3.0? |
|---|---|---|---|---|
| **TetGen** (C++ core, bundled in the `tetgen` wheel) | 1.6 (via `tetgen` 0.8.4) | **AGPL-3.0-or-later** (dual; WIAS commercial option) | Linked — compiled into `_tetgen.abi3.so`, imported by golgi | Yes — and it is *why* golgi is AGPL |
| `tetgen` Python wrapper (PyVista) | 0.8.4 | MIT (wrapper glue only; bundles the AGPL TetGen core above) | Linked (`import tetgen`) | Yes (MIT) |
| **Gmsh** | 4.15.2 | **GPLv2-or-later** | Linked (`import gmsh`) | Yes (GPLv2+ upgrades to AGPLv3) |
| PyVista | 0.48.1 | MIT | Linked | Yes |
| Trame (+ `trame_*` packages) | 3.12.0 | Apache-2.0 | Network GUI server | Yes (Apache-2.0 → AGPLv3, one-way) |
| NumPy | 2.4.6 | BSD-3-Clause (and 0BSD/MIT/Zlib/CC0 for vendored parts) | Linked | Yes |
| SciPy | 1.17.1 | BSD-3-Clause | Linked | Yes |
| NEURON | 9.0.1 | BSD-3-Clause* | Linked (through PyFibers) | Yes |
| FEniCSx / DOLFINx | 0.10.0 | LGPL-3.0-or-later | Linked | Yes |
| **PyFibers** | 0.8.5 | **GPL-2.0-only** (Duke dual: GPLv2 non-commercial / commercial) | Linked (`import pyfibers`) | **⚠ See flag below** |
| **AxonML** (optional GPU backend) | not bundled (user-installed) | **Proprietary — Duke University academic / non-commercial license** (OTC T-008477; © 2024 Duke University) | Optional: imported only if the user installs it and enables the GPU backend | **No — not open-source; see flag** |

\* NEURON's bundled LICENSE text is the 3-clause BSD license; note its PyPI
Trove classifier is mislabeled `License :: Other/Proprietary License`. The
governing text is BSD-3-Clause (permissive, compatible).

## Compatibility flags (require human / legal confirmation)

1. **PyFibers (GPL-2.0-only) vs. golgi (AGPL-3.0).** PyFibers 0.8.5 is offered
   under GPL **version 2** (for non-commercial use) with no "or later" grant,
   plus a separate Duke commercial license. GPL-2.0-**only** is *not*
   one-way-compatible with GPLv3/AGPLv3: a single combined work cannot be
   formed from GPL-2.0-only code and AGPL-3.0 code. golgi mitigates this by
   running the fiber-simulation step (PyFibers → NEURON) and the TetGen mesher
   in **separate subprocesses** that exchange data through files, which is a
   strong argument that they are separate programs in mere aggregation rather
   than one linked work. This architectural separation should be confirmed
   with counsel before distributing golgi as a single product, and PyFibers'
   non-commercial field restriction noted for any commercial use.

2. **AxonML is a proprietary, non-commercial Duke license — not open source.**
   AxonML is distributed under a custom Duke University license (OTC File
   T-008477) granting a royalty-free, non-transferable license for
   **non-commercial research and academic testing only**, with a 5-year term,
   a required "Copyright 2024 Duke University. All Rights Reserved" notice, US
   export-control compliance, and acceptance triggered by download/clone/fork.
   This is **not** an open-source license and is **not compatible with
   AGPL-3.0 for combination or redistribution**. golgi therefore must **never
   bundle or redistribute AxonML**: it is an *optional* backend that golgi
   imports only through a guarded hook (`golgi/pipeline/fiber_backends.py`)
   *if* the end user has separately obtained AxonML and accepted Duke's terms.
   Used this way it is a separately-licensed optional plug-in obtained by the
   user, so it does not affect golgi's AGPL licensing, and the default
   NEURON/PyFibers path needs no such backend. Anyone redistributing golgi
   with AxonML bundled or enabled, or using it commercially, must obtain a
   commercial license from Duke (Office for Translation and Commercialization).

## Redistribution note (closed-source / commercial use)

golgi imposes no restriction beyond the AGPL-3.0-or-later terms. However,
because both meshers are copyleft, redistributing golgi as part of a
**closed-source or commercial** product additionally requires separate
**commercial licenses for TetGen (from WIAS, tetgen@wias-berlin.de) and for
Gmsh**, and resolution of the PyFibers/AxonML field-of-use terms above.

Full upstream license texts are available with each package (its `*.dist-info`
or source repository); TetGen's AGPL text is reproduced in golgi's own
[LICENSE](LICENSE).
