# Command-Line Interface

golgi installs a single `golgi` console command (added by `pip install -e .`). With **no
subcommand** it launches the [GUI](GUI-Walkthrough) server; with a subcommand it runs the headless
study-bundle tooling. Every form is also reachable as `python -m golgi.app …` if you prefer not to
rely on the installed script.

```text
golgi [--port 8080]                          # launch the browser GUI
golgi export  <project_dir> [<out.zip>] [--user NAME]
golgi import  <bundle.zip>  [<target_dir>]
golgi replay  <bundle.zip | bundle_dir> [--check-only | --full] [--keep-tmp] [--json]
golgi compute-worker <payload.json>          # remote-side worker (used by the SLURM runner)
```

**Exit codes** are consistent across subcommands: `0` success · `1` user-facing error (file missing,
hash mismatch, replay failure) · `2` internal exception (traceback on stderr).

---

## `golgi` — launch the GUI

```bash
golgi                 # serve on http://localhost:8080
golgi --port 9000     # choose a port
```

Open the printed URL in a browser and work through the pipeline by point-and-click. See the
[GUI Walkthrough](GUI-Walkthrough).

---

## `golgi export` — pack a project into a study bundle

```bash
golgi export ~/projects/vagus_study
golgi export ~/projects/vagus_study vagus.zip --user m.haberbusch
```

Packs a project directory into an integrity-hashed [study bundle](Reproducible-Study-Bundles).
If `out_zip` is omitted, the bundle is written next to the project as
`<project>_study_<timestamp>.zip`. `--user` overrides the `exported_by` field in the manifest
(useful in headless contexts with no logged-in user). Progress is printed to stderr; the final line
reports the written path and size.

---

## `golgi import` — unpack a study bundle

```bash
golgi import vagus.zip
golgi import vagus.zip ~/projects/vagus_imported
```

Unpacks a bundle into a fresh project directory. If `target_dir` is omitted, a non-colliding
`<name>_imported[/_N]` directory is created next to the zip. Defends against path-traversal entries
and streams large payloads to avoid memory spikes. Prints the unpack location, file count, and the
original exporter.

---

## `golgi replay` — verify reproducibility

```bash
golgi replay vagus.zip                 # default: --check-only (re-hash every file)
golgi replay vagus.zip --json          # machine-readable ReplayReport on stdout
golgi replay ./unpacked_bundle_dir     # works on an already-extracted dir, too
```

This is the **reproducibility check**. In the default `--check-only` mode it re-hashes every file in
the bundle and compares against `MANIFEST.json`, detecting any byte-level tampering without
re-running compute. On success it exits `0`; on a mismatch it exits `1` and prints the diverging
stage and the specific output file(s):

```text
✗ replay FAILED · mode=check_only · 159 / 160 files verified
  stage `fem` diverged:
    paths_Ve.npz: (sha mismatch)
```

| Flag | Effect |
|---|---|
| `--check-only` | *(default)* Re-hash and compare to the manifest. |
| `--full` | Re-run each pipeline stage from inputs and hash the outputs. *(Currently falls back to check-only.)* |
| `--keep-tmp` | Keep the extracted temp directory for inspection. |
| `--json` | Emit the full `ReplayReport` as JSON on stdout. |

See [Reproducible Study Bundles](Reproducible-Study-Bundles) for the bundle layout, the manifest
schema, and the DAG-stage hashing model.

---

## `golgi compute-worker` — remote execution entry point

```bash
golgi compute-worker /path/to/payload.json
```

The remote-side worker invoked by the SLURM `sbatch` wrapper that `SlurmJobRunner` generates (and
callable directly for debugging). It reads a typed `JobRequest` payload, dispatches on the
payload's `"kind"` discriminator (`mesh`, `fem`, `fiber_sim`, …), runs the matching pipeline runner,
and writes `outputs.json` next to the payload. See [Headless / HPC](Headless-and-HPC) for the job
schema and the SLURM submission lifecycle.

---

### See also
[Python API](Python-API) · [Reproducible Study Bundles](Reproducible-Study-Bundles) ·
[Headless / HPC](Headless-and-HPC) · [GUI Walkthrough](GUI-Walkthrough)
