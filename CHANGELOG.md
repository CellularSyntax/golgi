# Changelog

## v1.1.0 — 2026-09 (SoftwareX revision)

### Distribution and reproducibility
- **Docker image**: `Dockerfile` (Ubuntu 24.04 + micromamba, full FEniCSx/Gmsh/TetGen/VTK/NEURON stack,
  PyFibers mechanisms pre-compiled), `docker/docker-compose.yml`, `docker/build_and_test.sh`
  (build → in-image tests → e2e example + benchmark → `docker save` tarball for Zenodo), `docker/README.md`.
- **Pinned conda environment**: `environment.yml` plus explicit lock files
  (`environment.osx-arm64.lock`, `requirements-pip.osx-arm64.txt`; Linux locks are written by the Docker build).
- **Continuous integration** (`.github/workflows/ci.yml`): solver-free tests on Linux and macOS, integration
  tests with the full stack, Docker build with in-container tests, GHCR push on tags.
- `CITATION.cff`, `.zenodo.json`, `golgi.__version__`.

### Tests
- New solver-free tests: `tests/test_bundle_integrity.py` (manifest hashes, determinism, tamper detection,
  import round-trip) and `tests/test_cli.py` (`golgi export|import|replay [--json]`).
- New end-to-end regression test `tests/test_e2e_reference.py` comparing the reference study
  (`examples/recruitment_sweep.py`, 12 fibers) against recorded thresholds in `tests/reference/`.
- `tests/README.md` documents the three tiers.

### Examples
- `examples/benchmark.py`: per-stage wall time and peak memory report (JSON + Markdown); `--profile full|light`
  (the light profile coarsens only the far field so the pipeline fits a 16 GB CI runner).

### Fixes
- Project root is configurable via `GOLGI_PROJECTS_ROOT` (was hard-coded to `~/Documents/Golgi/Projects`
  and created at import time, which broke headless/containerised use).
- `golgi --host/--no-browser` flags (and `GOLGI_HOST`, `GOLGI_NO_BROWSER`) for containers and headless hosts.
- Bundle manifest: pipeline-stage outputs are now resolved inside `designs/<eid>/`, `configs/<cid>/` and
  `sweeps/` (stages previously reported "not present" on real bundles).
- `golgi replay` reports the specific mismatching file(s); `--json` output is clean JSON (informational
  notices moved to stderr); `--full` help text states that re-execution is not implemented.
- `psutil` added as a dependency (benchmark).
