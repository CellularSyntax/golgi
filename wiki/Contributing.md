# Contributing

Contributions are welcome. golgi is AGPL-3.0-or-later (see [License & Citation](License-and-Citation));
by contributing you agree your work is licensed under the same terms.

---

## Development setup

Install golgi for development per [Installation](Installation) (conda for the solver stack, then
`pip install -e .`). Work in the conda environment so DOLFINx/NEURON resolve.

## Running the tests

```bash
# fast, solver-free guards — run anywhere
pytest tests/test_headless_api.py -m "not integration"
pytest tests/test_recording_cable.py -q

# full end-to-end pipeline — needs the FEniCSx solver stack (auto-skips otherwise)
pytest tests/test_headless_api.py -m integration

# point the integration test at real geometry
GOLGI_TEST_NERVE=/path/to/nerve.stl pytest tests/test_headless_api.py -m integration

# everything
pytest tests/
```

The `integration` marker gates the heavy end-to-end test; it auto-skips unless `dolfinx`, `gmsh`,
`pyfibers`, and `tetgen`/`wildmeshing` are importable. The headless guards assert that each `Study`
compute method stays wired to its pipeline driver (so a stubbed regression fails loudly). The
[CNAP](Recording-and-CNAP) math is covered by pure-numpy unit tests. A `tests/fake_sbatch.py` shim lets
you exercise the [SLURM runner](Headless-and-HPC) on a workstation.

## Design rules

golgi's features compose because they follow a few invariants — read [Architecture](Architecture) for
the full list. The essentials:

- **Project-local artifacts** — write everything under `<project>/`, never `~/.cache` or `/tmp`.
- **Typed job schemas** — new compute requests are `dataclass`es in `golgi/jobs/schemas.py` with
  `serialize()` / `deserialize()`.
- **Content hashes** — every artifact carries a SHA-256 in a sibling manifest (replay depends on it).
- **Cancellable drivers** — accept a `CancelToken` and check it between sub-units.
- **Register every figure** in `golgi/figures/registry.py` in the same change.
- **No project schema migrations** — new fields are optional; old projects must open cleanly.
- **Keep the three interfaces in sync** — a new capability should be reachable from the GUI *and* the
  headless [`Study`](Python-API) (the API re-implements only the small closures it needs, rather than
  importing the GUI).

## Where things live

See the package map in [Architecture](Architecture). In short: stage logic in `pipeline/`, numerics in
`compute/`, schemas in `jobs/`, GUI in `app.py` + `ui/`, headless API in `api.py`, figures in
`figures/`, bundles in `projects/`.

## Roadmap

The living feature plan — what's done, in flight, and planned, each with goal/depends-on/verify notes
— is [`FEATURES.md`](https://github.com/CellularSyntax/golgi/blob/main/FEATURES.md). Pick work from
there, keep each change small and independently shippable, and verify against the smoke/test items the
plan lists.

## Reporting issues

Open an issue at [github.com/CellularSyntax/golgi/issues](https://github.com/CellularSyntax/golgi/issues).
For reproducible problems, attach a [study bundle](Reproducible-Study-Bundles) — it carries the inputs,
version, and environment needed to reproduce.

---

### See also
[Architecture](Architecture) · [Validation](Validation) · [Headless / HPC](Headless-and-HPC) ·
[License & Citation](License-and-Citation)
