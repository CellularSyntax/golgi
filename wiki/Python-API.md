# Python API (`golgi.Study`)

Everything the [GUI Walkthrough](GUI-Walkthrough) does by point-and-click is mirrored by a single
synchronous Python class, **`golgi.Study`**. The API and the GUI act on the *same* on-disk project
state, so a study started in the browser can be continued from a script (and vice-versa), and a
script-built project opens cleanly in the GUI.

Use the API for notebook-based papers, batch parameter studies, CI regression tests, and as the unit
of work that the [Headless / HPC](Headless-and-HPC) runners fan out.

```python
import golgi

s = golgi.Study.create("/tmp/vagus_study")
s.import_nerve("nerve.stl")
s.set_mesh(use_epi=True, epi_thickness_um=50)
s.run_mesh()
s.set_electrodes([{"name": "bipolar @ 5 mm", "cuff_offset_mm": 5.0}])
s.run_fem()
s.run_fibers()
res = s.run_sweep(req)
s.export_bundle("/tmp/vagus_study.golgi")
s.close()
```

> **Import is cheap.** `import golgi` does **not** start Trame/VTK/PyVista — the heavy stack is
> imported lazily the first time you touch a compute method, so `golgi.Study` is fast to import in
> scripts and notebooks.

---

## Lifecycle

| Call | Description |
|---|---|
| `Study.create(project_dir, *, user="headless")` | Create a **new** project. Raises `FileExistsError` if the directory exists and is non-empty (it refuses to clobber an existing study). |
| `Study.open(project_dir, *, user="headless")` | Attach to an **existing** project. Loads the project's `ui_state.json` (if present) so the first compute call sees the same parameters the GUI would. Raises `FileNotFoundError` if the directory is missing. |
| `study.close()` | Release the headless context. Safe to call repeatedly. |
| `with Study.create(...) as s: ...` | Context-manager sugar — `close()` runs on exit. |

Read-only properties: `study.project_dir` (`Path`), `study.user` (`str`).

Every method is **synchronous**. Internally the async pipeline drivers are wrapped in `asyncio.run`,
so you call them from plain functions, scripts, and Jupyter cells without `await`.

---

## The pipeline, method by method

The methods map one-to-one onto the [pipeline stages](Pipeline-Overview). Call them in order; each
stage writes its artifacts under `<project>/` and the next stage reads them back.

### `import_nerve(stl_path, *, scale_factor=1.0e-3) -> dict`
Load a nerve surface (**STL / NAS / OBJ**). Computes the global PCA frame, per-triangle surface
quality, and topology stats. `scale_factor` converts file units to metres (default `1e-3` = mm → m).

Returns a summary: `{n_pts, n_tris, n_components, bbox_mm, watertight, q_median}`.

### `set_mesh(**kwargs) -> None`
Set mesh parameters in bulk (forwarded to project state). Common keys:
`use_epi`, `epi_thickness_um`, `lc_endo_um`, `lc_epi_um`, `lc_muscle_um`, `lc_saline_um`,
`lc_silicone_um`, `lc_contact_um`, `lc_scar_um`, `decim_target_k`, `muscle_radial_pad_mm`,
`muscle_axial_pad_mm`, `muscle_dx_mm`, `muscle_dy_mm`, `muscle_dz_mm`. See
[Meshing](Meshing) and the [Configuration Reference](Configuration-Reference).

### `set_electrodes(designs: list[dict]) -> None`
Replace the project's electrode designs. You may pass **minimal** dicts — just the fields you care
about (e.g. `eid`, `name`, `cuff_offset_mm`, `electrode_type`); every other field is filled from the
factory defaults, contact polarities/fractions are derived from the electrode type, the first design
becomes selected/active, and each design gets a Default FEM config. See
[Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer) for the electrode types and their
parameters.

```python
s.set_electrodes([
    {"name": "Tripolar @ 6 mm",
     "electrode_type": "tripolar (anode-cathode-anode)",
     "cuff_offset_mm": 6.0},
])
```

### `run_mesh() -> dict`
Build a per-design TetGen mesh for **every** design. Outputs land in `<project>/designs/<eid>/`.
Returns `{eid: path_to_nerve.msh}`.

### `run_fem() -> dict`
Solve the anisotropic FEM for **every** config in the project. Outputs land in
`<project>/configs/<cid>/`. Returns `{cid: outputs_dir}`. Requires `run_mesh()` first. See the
[Finite-Element Solver](Finite-Element-Solver) page.

### `run_fibers() -> dict`
Generate curved 3-D fiber trajectories on the loaded nerve (Laplace solve + streamline integration +
cap detection + branch classification). Writes `<project>/nerve_paths_fibers.npz` and friends.
Returns `{n_paths, n_branches, n_pts_total, branch_summary}`. See
[Fiber Populations & Trajectories](Fiber-Populations-and-Trajectories).

### `set_fiber_seed(**kwargs) -> None`
Set fiber-generation parameters in bulk. Common keys: `n_fibers`, `fiber_max_steps`,
`fiber_seed_end`, `fiber_cluster_eps_mm`, `fiber_cap_band_pct`, `fiber_min_rel_size_pct`,
`fiber_axial_normal_thresh`, `fiber_auto_detect_branches`, `fiber_method`.

### `run_sweep(request) -> SweepResult`
Run a parameter sweep — **recruitment** mode (activation across an amplitude axis) or **threshold**
mode (per-fiber activation threshold by bisection). Returns a `SweepResult` and writes it to the
project's sweep cache, tagged with the active config. See
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity) for the request/result
schema.

---

## Reproducible bundles

| Call | Description |
|---|---|
| `study.export_bundle(out_zip) -> Path` | Pack the project into an integrity-hashed study bundle. The `.golgi` extension is conventional; the container is a zip. |
| `Study.import_bundle(zip_path, target_dir) -> Path` *(staticmethod)* | Unpack a bundle into `target_dir`; returns the project directory. |

See [Reproducible Study Bundles](Reproducible-Study-Bundles) for the manifest format and `replay`
verification.

---

## Inspectors

| Call | Description |
|---|---|
| `study.list_designs() -> list[dict]` | Per-cuff designs from the project's `ui_state.json`. |
| `study.list_configs() -> list[dict]` | Per-design FEM configs. |

---

## Authentication in headless mode

Headless runs synthesise a `headless` user (local trust, no password) so the auth-gated pipeline
decorators run unchanged. The [audit](Authentication-and-Audit) writer still fires — every headless
action lands in the same `audit_fallback.jsonl` the GUI writes, so headless runs show up in
`golgi events` / the Activity tab. Pass a different `user=` to `create`/`open` to attribute the run.

---

## End-to-end example

A complete load → mesh → electrodes → FEM → fibers → sweep → bundle script lives in
[`examples/recruitment_sweep.py`](https://github.com/CellularSyntax/golgi/blob/main/examples/recruitment_sweep.py)
and is exercised by `tests/test_headless_api.py`. See [Getting Started](Getting-Started) for a
walk-through.

---

### See also
[Getting Started](Getting-Started) · [Command-Line Interface](Command-Line-Interface) ·
[Headless / HPC](Headless-and-HPC) · [Pipeline Overview](Pipeline-Overview) ·
[Configuration Reference](Configuration-Reference)
