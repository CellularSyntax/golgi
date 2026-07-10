# Reproducible Study Bundles

A **study bundle** is golgi's unit of reproducibility: a single self-contained archive of an entire
study — inputs, every stage's outputs, the software version, a frozen dependency list, and an audit
excerpt — with a **SHA-256 hash of every file and pipeline stage**. Hand it to a colleague or a
reviewer and they can reconstitute the study and verify it **byte-for-byte** with one command.

Source: `golgi/projects/bundle.py` (export/import), `golgi/projects/replay.py` (verification),
`golgi/projects/sweep_cache.py`. CLI in `golgi/cli.py`; API in `golgi.Study`.

---

## What's in a bundle

A bundle is a ZIP (conventionally given the `.golgi` extension) that mirrors the project flat, plus a
manifest:

```text
MANIFEST.json                 # schema, version, exporter, file hashes, pipeline DAG
project.json                  # project metadata
source/<nerve>.stl|nas|obj    # the original imported geometry
mesh_config.json · electrode_config.json · conductivities.json · nerve_paths_seed_config.json
nerve.msh                     # mesh
nerve_paths_fibers.npz · nerve_paths_caps.json   # fibers
axis_line.npz · slice_volume.npz · paths_Ve.npz · nerve_surface_Ve.npz   # FEM
Ve.xdmf/.h5 · E.xdmf/.h5      # full fields
designs/<id>/… · sims/<id>/…  # per-design meshes / FEM / sims
sweep_<sha>.npz · sweep_<sha>.json        # sweep results
env/golgi_version.txt · env/requirements-frozen.txt   # environment
audit/audit_excerpt.json      # project-scoped audit rows
```

### The manifest

`MANIFEST.json` records the **pipeline DAG** and the hashes that make verification possible:

```json
{
  "schema": "golgi.study.bundle.v1",
  "golgi_version": "1.0.0",
  "exported_at": "2026-06-29T15:30:00+00:00",
  "exported_by": "user@example.com",
  "project": { "name": "...", "source_file": "..." },
  "files": [ { "name": "nerve.msh", "sha256": "..." }, ... ],
  "dag": [
    { "stage": "mesh",   "present": true,  "inputs": ["mesh_config.json", "source/*"], "outputs": ["nerve.msh"], "sha256": "..." },
    { "stage": "fem",    "present": true,  "inputs": ["nerve.msh", "electrode_config.json", "conductivities.json"], "outputs": ["paths_Ve.npz", ...], "sha256": "..." },
    { "stage": "fibers", "present": true,  ... },
    { "stage": "fiber_sim", "present": false, ... },
    { "stage": "pop_sim",   "present": false, ... },
    { "stage": "sweep",  "present": true,  "outputs": ["sweep_<sha>.npz", ...], "sha256": "..." }
  ]
}
```

Each file's `sha256` is streamed (64 KB chunks), and each **stage** hash is the SHA-256 of its
outputs' hashes in canonical order — so a single changed byte anywhere is detectable, and you can see
*which stage* diverged.

---

## Export, import, replay

### From the command line

```bash
golgi export ~/projects/vagus_study vagus.golgi   # pack
golgi import vagus.golgi ~/projects/vagus_copy     # unpack into a fresh project
golgi replay vagus.golgi                           # verify (default: re-hash every file)
golgi replay vagus.golgi --json                    # machine-readable report
```

See [Command-Line Interface](Command-Line-Interface) for all flags.

### From Python

```python
study.export_bundle("vagus.golgi")
golgi.Study.import_bundle("vagus.golgi", "/path/to/target")
```

### In the GUI

**File ▸ Export study / Import study**, or the **Export study** button in the Project Details dialog.

---

## How replay verifies

`replay_study` unpacks the bundle to a temp dir and, in the default **check-only** mode, **re-hashes
every file** and compares against the manifest. It returns a structured `ReplayReport`
(`ok`, per-file `matched`, and a per-**stage** breakdown), so a failure pinpoints the diverging stage
and output:

```text
✗ replay FAILED · mode=check_only · 159 / 160 files verified
  stage `fem` diverged:
    paths_Ve.npz: (sha mismatch)
```

Exit code is `0` on a clean verify, `1` on any mismatch. A `--full` re-run-from-inputs mode is part of
the design (currently it falls back to check-only). Import re-registers the project under the importing
user and replays the audit excerpt tagged with the original provenance (see
[Authentication & Audit](Authentication-and-Audit)).

> The paper's figures ship as replay-verified bundles — see [Reproducing the Paper](Reproducing-the-Paper).

---

## Sweep cache

Sweep results live in `<project>/sweeps/` named by a SHA over the request
(`sweep_<sha>.npz` + `.json` + CSVs), with `latest` markers per config. Identical sweeps hit the cache
instantly, and bundles include these results so recruitment/selectivity figures reproduce. See
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity).

---

### See also
[Command-Line Interface](Command-Line-Interface) · [Python API](Python-API) ·
[Reproducing the Paper](Reproducing-the-Paper) · [Authentication & Audit](Authentication-and-Audit)
