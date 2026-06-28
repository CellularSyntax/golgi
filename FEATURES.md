# golgi — feature roll-out plan

> Companion to `legacy/migration.md`. Migration is done; this document
> sequences the next nine research-grade features.
>
> Every step has the same six fields: **Goal · Depends on · Touches · Do ·
> Verify · Rollback**. Steps land in order; each step ships independently
> and the app must keep passing the smoke test (from `migration.md`) plus
> the per-phase additions defined below.

---

## How to use this document

- Phases F1 → F5 are ordered by impact-per-effort and by inter-feature
  dependency. Do not jump phases unless the **Depends on** row is empty.
- Inside a phase, the steps are *also* dependency-ordered.
- Each step is its own PR / commit. Don't bundle two steps — bisect cost
  on regressions is the whole reason to keep them separate.
- "Verify" is always the existing `migration.md` smoke test **plus** the
  new checks listed under that step. If a baseline smoke item regresses,
  revert before investigating.
- File paths are relative to repo root unless noted.
- When a step says "extend `pipeline/foo.py`" — read it first, don't
  rewrite the existing function shape, just append the new entry point.

---

## Roll-out order rationale

| Phase | Theme | Why now |
|---|---|---|
| W | Architectural debt repayment | `app.py` is still 15.6 k LOC; deferring this means every new feature compounds the monolith. Land next. |
| F1 | Small immediate wins (no new pipeline stages) | Build user trust + ship `figures/export.py` infra that F2.3 consumes. (✅ done) |
| F2 | Comparative-analysis foundation | Sweep + bundle + bulk figure export are the substrate that F3 + F5 stand on. F2.3 also adds per-panel buttons + Generate Report. |
| F3 | Comparative analysis layer | Multi-electrode, multi-contact, selectivity, FEM surrogate, validation — the features users will *cite*. |
| F4 | Infrastructure expansion | Headless API + HPC runner unblock UQ scale and external scripting. |
| F5 | UQ + anatomical realism capstone | UQ + atlases consume everything above; only worth doing once F2.1 + F4.1 are stable. |

| F# | Feature | Effort | Impact | Status |
|---|---|---|---|---|
| W1 | Collapse `app.py` back toward a 100-line composer | M–L | H (compounds every later feature) | ⚠️ partial (-14.8 %, see W1.9 deferred) |
| W1.9 | Build_app closure-helper refactor + remaining do_* extraction | M–L | M (architectural; deferred until needed) | deferred |
| F1.1 | Biology-realistic vagal fiber population presets | S | H | ✅ done |
| F1.2 | Publication-grade figure styling presets | S | H | ✅ done |
| F2.1 | Parameter sweep + threshold-finder + recruitment curves | M | H | ✅ done |
| F2.2 | Reproducible study bundle (export + import + replay) | S–M | H | ✅ done |
| F2.3 | Figure export surface (per-panel + bulk + Generate Report) | M | H | ✅ done |
| F3.1 | Multi-electrode FEM solve | S–M | H | ✅ done |
| M1 | Multi-contact anode/cathode picker (tri-/quadri-/N-polar) | S–M | H | ✅ done |
| F3.2 | Selectivity metrics + side-by-side comparison view | M | H | ✅ done |
| S1 | FEM surrogate — sub-second "what-if" for clinical iteration | M | H (unlocks F2.1, F3.2, F5.1 speed) | pending |
| F3.3 | Validation overlay (measured impedance + recorded thresholds) | S–M | H | pending |
| F4.1 | Headless Python / Jupyter `Study` API | M | H | ✅ done |
| F4.2 | SLURM `JobRunner` + FEM checkpoint-resume | M | H | ⚠️ partial (Phase A: SlurmJobRunner + env dispatch + worker CLI + fake_sbatch shim ✅; Phase B: FEM checkpoint-resume + pipeline wiring) |
| F5.1 | Sensitivity / UQ (Sobol + LHS over σ + fiber distributions) | M–L | H | pending |
| A1 | Topographic-atlas-driven trajectory generation (Hammer / Settell) | M–L | H | pending |
| X1 | Scar / connective tissue shell (per-design geometry + sweep) | M | M | ✅ done (unplanned addition) |
| I1 | Electrode-tissue impedance from FEM (DC; per-contact + per-pair) | S–M | H | ⚠️ partial (Phase A: DC dirichlet dual-solve ✅; Phase B: Cole-Cole frequency sweep + Bode plot) |
| R1 | Recording cuff role + CNAP forward model (reciprocity-based, bipolar montage) | M | H | pending |
| I2 | Electrode-electrolyte interface (Randles cell: R_ct ∥ C_dl + Warburg) post-FEM on Z_access | S | H | pending |
| V1 | µCT slice segmentation + extrusion (MedSAM2; single slice → prismatic nerve) | M | H | pending |

**Roll-out order (sequenced):** W1 → F2.1 → F2.2 → F2.3 → F3.1 → M1 →
F3.2 → S1 → F3.3 → F4.1 → F4.2 → F5.1 → A1. Unplanned-but-landed:
X1 (scar shell), I1 Phase A + A.2 (DC access impedance, sanity-checked
against Newman 1/(4σa)). In flight: V1 (µCT segmentation Phase A).
Queued: R1 (CNAP), I1 Phase B (Cole-Cole sweep), I2 (Randles interface),
F3.3 Phase 2 (measured-vs-simulated overlay).

---

## Smoke test additions (cumulative)

Append the items below to the existing `migration.md` smoke test as each
phase lands. After Fx is in, the smoke test grows by the items under Fx.

- **W1** — `wc -l golgi/app.py` reports < 500 (target). Full migration
  smoke test top-to-bottom still passes; `grep -c "@state.change"
  golgi/app.py` reports 0; `grep -c "def do_" golgi/app.py` reports 0.
- **F1.1** — Population panel → "Load preset → Cervical vagus — human"
  populates per-branch type rows; citation + notes render; preview KDE
  matches the preset's expected shape. ✅ landed.
- **F1.2** — Each figure builder accepts `preset=…`; F2.3 consumes it.
  Smoke is the F2.3 round-trip below. ✅ landed.
- **F2.1** — Analysis tab → Sweep subtab → amplitude sweep 0.1-2.0 mA over
  cached `paths_Ve.npz` produces a recruitment curve in < 5 s (no FEM
  re-solve); threshold-finder converges per fiber inside 10 sims.
- **F2.2** — Project detail dialog → "Export study" → produces `study.zip`;
  fresh `golgi import study.zip` reconstitutes the project + audit row;
  `golgi replay study.zip --headless` exits 0 with all-stage hash match.
- **F2.3** — Three sub-checks:
  - (a) Per-panel: every panel header has an export icon; clicking opens
    the format popover; Export produces a single PDF/SVG/PNG.
  - (b) Bulk tab: "Exports" navbar entry → opens a tab listing every
    figure with live thumbnails; "Export all (PDF, 300 DPI)" writes
    one file per figure plus a `data/` folder of CSVs.
  - (c) Report: "Report" navbar entry → opens Generate Report dialog →
    "Generate" writes one `report_<timestamp>.pdf` under
    `<project>/reports/`. PDF contains cover + ToC + all six sections
    + auto-included reproducibility/conductivity/electrode-config
    tables + bibliography + audit excerpt; 3D viewport variants share
    one camera per category.
- **F3.1** — Cuff drawer → second electrode added → FEM solve produces two
  `fem/<design_id>/` subfolders, axis plot exposes a design selector.
- **M1** — Cuff designer → set contact 1 = Cathode, contacts 2+3 = Anode
  @ 50/50 fractions → FEM solve writes `electrode_config.json` with
  the new polarity vocabulary; legacy `"active"` projects still load
  cleanly.
- **F3.2** — Compare-designs tab → pick two designs → side-by-side axis +
  slice plots + selectivity bar chart render.
- **S1** — FEM panel → "Build contact basis" completes; Live Stimulation
  panel: dragging amplitude slider 0.1 → 2.0 mA updates recruitment +
  propagation tiles at ≥ 5 Hz without re-solving FEM. "Verify against
  full FEM" reports max |ΔVₑ| < 1e-12. Invalidating σ_endo flips the
  panel to `[surrogate stale]`.
- **F3.3** — Validation drawer → paste impedance CSV → Cole-Cole figure
  overlays measured points with R² in the legend.
- **F4.1** — `python -m golgi.api examples/recruitment_sweep.py` runs
  headless and writes a recruitment-curve PNG to the project dir.
- **F4.2** — `GOLGI_FEM_RUNNER=slurm` re-runs item 8 end-to-end against a
  cluster (or fakes via a local sbatch shim). Cancel still works.
- **F5.1** — UQ tab → "Run Sobol-32 over σ" produces tornado plot and
  CI-banded recruitment curve; each sample re-uses cached fiber sims.
- **A1** — Fibers drawer → mode = Atlas → pick "Hammer 2018 sheep VN" →
  Generate → fiber cross-section preview matches the atlas's
  fascicular layout; `fiber_group_labels` array exists in
  `nerve_paths_fibers.npz`; F2.3 Generate Report includes the atlas
  citation in the FEM section caption + bibliography.

---

# Phase W — Architectural debt repayment

The migration plan (`legacy/migration.md`) promised Step 6.1 would land
as: *"`build_app` is now a 100-line composer."* Reality after F1.2:
`golgi/app.py` is **15,604 lines** containing **68 `do_*` handlers**,
**19 `@state.change` watchers**, the 3-term Cole-Cole evaluator, the
IT'IS DB loader, mesh-quality math, PLC assembly, electrode-patch
builders, axonml/pyfibers branch dispatchers, and the cuff-fit PCA
helpers — all inline. The extracted-out modules under `golgi/auth/`,
`golgi/scene/`, `golgi/pipeline/`, etc. are real, but they sit
*alongside* an unreduced monolith rather than replacing it. The
narrative is "the refactor is mostly done"; the reality is "Step 6.1
landed as a file rename and the deep extractions stalled."

W1 finishes the job. It is not a forward feature, but it pays off
every subsequent feature (smaller diffs, clearer ownership, easier
review). Done as a sequence of small commits, each verifying via
`scripts/import_check.py` + the relevant smoke-test items.

## W1 — Collapse `golgi/app.py` back toward a composer

**Goal.** Reduce `golgi/app.py` to a true composer (target: < 500
lines, aspirational 100) by extracting the remaining inline domain
code into the package modules `migration.md` already established.

**Depends on.** Nothing — pure refactor. Lands before F2.x to keep
the substrate clean as new features pile on.

**Touches.** `golgi/app.py` (shrinks) + new files under
`golgi/conductivity/`, `golgi/scene/`, `golgi/pipeline/`,
`golgi/watchers/`, `golgi/actions/`.

**Do.** Eight sub-extractions, each its own PR. Order is bottom-up
(small leaves first; the do_* handler split is last because it has
the most dependencies). Each step ends with a green import-check
+ the corresponding smoke item.

### W1.1 — Extract Cole-Cole + IT'IS DB → `golgi/conductivity/`

- **Targets in app.py.** The 3-term Cole-Cole evaluator
  (`cole_cole_sigma`, the cc_* state defaults, the Cole-Cole dialog
  state plumbing) + the IT'IS DB JSON loader + lookup helpers.
- **New files.**
  - `golgi/conductivity/cole_cole.py` — `cole_cole_sigma(f, eps_inf,
    sigma_ionic, dispersions)` + the 3-term schema (matches user's
    prior preference for 3-term as primary tissue fit).
  - `golgi/conductivity/itis_db.py` — JSON loader + the lookup
    surface (`get_tissue_props(tissue, freq_hz) -> ...`). Keeps the
    Nerve entry path (the IT'IS-specific addition over Gabriel 1996).
  - `golgi/conductivity/__init__.py` — public re-exports.
- **Do.** Cut → paste; replace inline calls in `app.py` with
  `from golgi.conductivity import cole_cole_sigma`, etc. The dialog
  UI (`golgi/ui/dialogs/cole_cole.py`) is already extracted; it just
  imports the evaluator from the new home.
- **Verify.** Migration smoke item 7. Open Cole-Cole dialog; apply a
  preset; numbers match pre-extraction.
- **Rollback.** Revert.

### W1.2 — Extract mesh-quality math → `golgi/pipeline/mesh_quality.py`

- **Targets in app.py.** Functions that compute per-tet shape quality
  (`q_radius_ratio` or equivalent), per-region histograms, mesh stats
  assembly. These currently feed `golgi/figures/mesh_stats.py` via a
  closure passed at render time.
- **Do.** Move the math to `pipeline/mesh_quality.py`. Pass the
  computed arrays into the figure builder instead of injecting the
  function. Simplifies the kwarg-injected `defaults_by_tag` indirection.
- **Verify.** Migration smoke item 6. Tet-quality histogram + per-tag
  table render identically before/after.
- **Rollback.** Revert.

### W1.3 — Extract PLC assembly → `golgi/pipeline/plc.py`

- **Targets in app.py.** Code that assembles the PLC (planar straight
  line graph) for TetGen from the imported nerve + cuff + electrode
  parts. Currently lives just before the `pipeline/mesh.py` driver
  call as ~300 lines of geometric bookkeeping.
- **Do.** Move into `pipeline/plc.py` as `build_plc(geometry, cuff,
  electrodes, options) -> PLCResult`. `pipeline/mesh.py` then calls
  this rather than receiving a pre-built PLC.
- **Verify.** Migration smoke item 6. Mesh builds; tet counts match.
- **Rollback.** Revert.

### W1.4 — Extract electrode-patch builders → `golgi/scene/electrode_patches.py`

- **Targets in app.py.** Functions that turn an `ElectrodeConfig`
  into renderable patch geometry (cuff contact rectangles, helical
  bands). Currently inline + called from both the scene tier and the
  FEM driver.
- **Do.** Move to `scene/electrode_patches.py`. Both call sites
  import from there. Removes the duplication.
- **Verify.** Migration smoke items 5 + 8. Cuff designer dialog + FEM
  solve both render the same patches.
- **Rollback.** Revert.

### W1.5 — Extract cuff-fit PCA helpers → `golgi/scene/cuff_fit.py`

- **Targets in app.py.** PCA-based cuff alignment helpers (extract
  the principal axis of the imported nerve, fit a cuff orientation,
  project contact positions to nerve frame). ~200 lines of `numpy`-
  only math currently inline.
- **Do.** Move into `scene/cuff_fit.py`. Public API:
  `fit_cuff_to_nerve(nerve_pts, cuff_params) -> CuffFitResult`.
- **Verify.** Migration smoke item 5. Cuff fits to imported nerve;
  contact positions match pre-extraction.
- **Rollback.** Revert.

### W1.6 — Extract axonml / pyfibers dispatchers → `golgi/pipeline/fiber_backends.py`

- **Targets in app.py.** The `axonml_run_single` wrapper +
  `_run_axonml_branch` + `_run_pyfibers_branch` dispatchers that
  pick the right backend per fiber sim. Some of this already moved
  in migration Step 4.6 (`pipeline/fiber_sim.py`) but the
  per-backend wrappers stayed behind for ~600 lines.
- **Do.** Move to `pipeline/fiber_backends.py`. `pipeline/fiber_sim.py`
  imports + dispatches. Public functions:
  `run_pyfibers_one(req) -> FiberSimResult`,
  `run_axonml_one(req) -> FiberSimResult`.
- **Verify.** Migration smoke items 10 + 11. Single-fiber sim under
  both backends; population sim. Outputs match pre-extraction
  (deterministic — exact numeric equality).
- **Rollback.** Revert.

### W1.7 — Finish watcher extraction → `golgi/watchers/`

- **Targets in app.py.** The remaining ~19 `@state.change` watchers
  still inline. Migration Step 5.2 extracted some by topic; W1.7
  extracts the rest. Likely groupings:
  - `watchers/electrode_panel.py` — any remaining electrode-row /
    contact-polarity watchers.
  - `watchers/mesh_panel.py` — mesh-stats refresh, tag visibility.
  - `watchers/import_panel.py` — file-picker, import-side state.
  - `watchers/auth.py` — login/profile/upload (already partial).
- **Do.** One PR per group. Each module exports
  `register(state, ctx)` called from `build_app`. The IMPORTANT
  rule from migration.md still applies: `@state.change(*keys)` must
  list every key the handler reads, not just the trigger.
- **Verify.** Migration smoke top-to-bottom; render-toggle changes
  still update the viewport, drawer-exclusion still works.
- **Rollback.** Revert per group.

### W1.8 — Extract `do_*` handlers → `golgi/actions/`

- **Targets in app.py.** The 68 `do_*` user-action handlers
  currently inline in `build_app`. They close over `state`, `geom`,
  `scene`, `_pipeline_ctx`, `_runner`, the `gated`/`log_action`
  decorator factories. Migration step 2.4 lifted the decorators
  out, which makes this extraction possible.
- **Do.** Create `golgi/actions/` with one module per domain:
  - `actions/auth.py` — `do_login`, `do_register`, `do_logout`, etc.
  - `actions/project.py` — `do_new_project`, `do_close_project`,
    `do_open_project`, `do_delete_project`.
  - `actions/import.py` — `do_import_*`, file-picker handlers.
  - `actions/cuff.py` — `do_add_electrode`, `do_remove_electrode`,
    `do_refit_electrode`, `do_rename_electrode`,
    `do_apply_cuff_preset`.
  - `actions/mesh.py` — `do_build_mesh` (thin wrapper over
    `pipeline/mesh.py`).
  - `actions/fem.py` — `do_run_fem` (over `pipeline/fem.py`).
  - `actions/fibers.py` — `do_generate_fibers` (over
    `pipeline/fibers.py`).
  - `actions/fiber_sim.py` — `do_run_fiber_sim`,
    `do_clear_fiber_sim` (over `pipeline/fiber_sim.py`).
  - `actions/pop_sim.py` — `do_pop_*` family
    (over `pipeline/pop_sim.py`). Includes `do_pop_apply_preset`
    that F1.1 added.
  - `actions/conductivity.py` — `do_apply_cc_preset`,
    `do_save_cc_dialog`.
  Each exports `register(ctx) -> dict[str, Callable]` returning a
  name→callable map that `build_app` binds for the UI templates.
- **Do (template binding nuance).** Trame templates capture callables
  by reference at the moment the `with v3.V*` block runs. To preserve
  that, `build_app` does:
  ```python
  actions = {}
  for mod in (auth, project, mesh, fem, fibers, ...):
      actions.update(mod.register(ctx))
  # ...later, in template construction:
  click=actions["do_build_mesh"]
  ```
  Watchers in the UI dialog/drawer files already accept callables
  via their `render(...)` signatures; just thread `actions` through.
- **Verify.** Full migration smoke test top-to-bottom. Every button
  must still fire its handler. Audit log must still attribute every
  action to the logged-in user.
- **Rollback.** Revert per domain (one rollback per action module).

### W1 acceptance criteria (revised after partial completion)

Original goal `wc -l golgi/app.py` < 500 turned out to need the
W1.9 closure-helper refactor below — actual W1.8 stopped at 23/68
handlers because the remaining 46 handlers transitively depend on
~50 inline closures that need to be lifted to module level FIRST.

- [x] `grep -cE '^[[:space:]]+@state\.change\('` reports 0 (all
  watchers extracted in W1.7).
- [x] `scripts/import_check.py` passes after every landed W1.N step.
- [x] W1.1–W1.7 + W1.8a + W1.8b + W1.8c.1 (CancelToken wiring) =
  **-2,308 lines** (-14.8 % from 15,604 baseline → 13,296 lines).
- [ ] **DEFERRED to W1.9 (closure refactor)**:
  - `grep -c "def do_" golgi/app.py` → 0
  - `wc -l golgi/app.py` < 500

## W1.9 — Build_app closure refactor (deferred)

**Status.** Deferred from W1.8 mid-execution after the closure
inventory revealed a denser web than anticipated. W1.8 stopped at
23/68 handlers (W1.8a = 11, W1.8b = 11, W1.8c.1 = 1 token-wiring).

**Why deferred.** The remaining 46 `do_*` handlers transitively
depend on ~50 build_app closures (`_clear_plotter_actors`,
`_restore_*_from_disk`, `_apply_ui_state`, `_extract_region_surfaces_mm`,
`_build_viz_surfaces`, `_refresh_fem_plots`, `_populate_cuff_visible_state`,
`_rebuild_cuff_preview`, `_save_pop_state`, etc.) which themselves
have closure-chain deps. The deeper-refactor scope is ~30 commits
(~50 closures + 5 handler groups) requiring ~1-2 days of focused
work. The marginal architectural benefit doesn't outweigh the time
cost vs shipping F2.x features that are user-visible.

**When to revisit.** When app.py's size starts blocking new feature
work (e.g. F2.3's per-panel buttons + Generate Report — many UI
touchpoints might need do_* references; if app.py is too big to
navigate confidently, that's the trigger to come back here).

**Implementation plan (when revisited).**

### W1.9.1 — Closure inventory + grouping
Document the ~50 build_app closures in one file (e.g. a comment
header at top of build_app, or a sidecar `legacy/closure_inventory.md`).
Group by concern:
- **Scene/actor lifecycle** (~10): `_clear_plotter_actors`,
  `_reset_geom_and_state`, `_request_render`, `_set_*_group`,
  `_update_muscle_preview`, `_remove_muscle_overlay`, `_phong_style`,
  `_polyline_polydata_from_paths`, etc.
- **Project lifecycle** (~9): `_activate_project`,
  `_snapshot_ui_state`, `_apply_ui_state`, `_apply_persisted_sigma`,
  `_format_modified`, `_refresh_projects_list`,
  `_refresh_projects_for_user`, `_capture_thumbnail`, `_autosave`.
- **Restore-from-disk** (~7): `_restore_mesh_from_disk`,
  `_restore_fem_from_disk`, `_restore_fibers_from_disk`,
  `_restore_fiber_sim_cache`, `_restore_pop_state`,
  `_save_fiber_sim_cache`, `_save_pop_state`.
- **Detail dialog** (~2): `_refresh_detail_briefs`, `_persist_labels`.
- **Cuff designer** (~8): `_populate_cuff_visible_state`,
  `_rebuild_cuff_preview`, `_clear_cuff_actors`, `_collect_cuff_overrides`,
  etc.
- **Electrode mgmt** (~7): `_find_electrode`, `_save_selected_to_electrodes`,
  `_load_electrode_to_selected`, etc.
- **Fiber/pop helpers** (~10): `_branch_name`,
  `_refresh_fiber_sel_items`, `_refresh_pop_branches_meta`,
  `_fiber_label_and_color`, `_fiber_paths_display`,
  `_ensure_field_lines_async`, etc.
- **FEM helpers** (~3): `_refresh_fem_plots`, `_override_preset_expr`,
  `_save_electrode_configs`.

### W1.9.2–W1.9.N — Extract one cohesive group per sub-commit
Each sub-step: pick one group, extract closures to a new module
(`golgi/scene/lifecycle.py`, `golgi/projects/restore.py`, etc.),
update callsites, commit, smoke. Order bottom-up: leaf groups
first (groups with no closure-chain deps), then groups that
depend on them.

### W1.9.N+1–W1.9.N+5 — Resume handler extraction
Once closures are module-level, the 46 remaining handlers extract
cleanly (each `register(state, workspace)` call gets a single
context object). Five domain commits as originally planned:
- W1.8c project lifecycle (8)
- W1.8d project_detail (13)
- W1.8e geometry (2)
- W1.8f cuff + electrode (14)
- W1.8g pop_sim + branch (9)

**Acceptance.** Same as original W1: `grep -c "def do_"
golgi/app.py` → 0, `wc -l golgi/app.py` < 500.

---

# Phase F1 — Small immediate wins

Two features that ship user-visible value in days and lay the groundwork
F2.3 will consume. Both are pure additions — no schema migration, no
pipeline stage changes.

## F1.1 — Biology-realistic vagal fiber population presets

**Goal.** Remove the "what diameter for B-fibers again?" footgun and
standardise fiber populations across studies.

**Depends on.** Nothing.

**Touches.**
- `golgi/state_defaults/pop.py` — currently holds per-row defaults
  (`mean_um`, `std_um`, `fraction`). Extend, do not replace.
- New: `golgi/state_defaults/pop_presets.py`.
- `golgi/ui/drawers/fibers.py` — add a preset dropdown above the row table.
- `golgi/figures/population.py:14` (`_build_pop_kde_figure`) — reuse for
  the preview pane; no signature change.

**Do.**
1. Create `pop_presets.py` exporting `POP_PRESETS: dict[str, PopPreset]`
   where:
   ```python
   @dataclass
   class PopRow:
       type: str           # "A-alpha" | "A-beta" | "A-delta" | "B" | "C"
       mean_um: float
       std_um: float
       distribution: str   # "gaussian" | "lognormal"
       fraction: float     # 0..1, must sum to 1 across rows of a branch
       mrg_supported: bool # False for C-fibers under MRG_INTERPOLATION
   @dataclass
   class PopPreset:
       name: str            # "cervical_vagus_human"
       species: str
       nerve: str
       citation: str        # bibtex key or short ref
       per_branch: dict[str, list[PopRow]]   # branch_label → rows
   ```
2. Populate at minimum: `cervical_vagus_human` (Soltanpour 1996),
   `cervical_vagus_pig` (Settell 2020), `cervical_vagus_rat`
   (Pelot 2020), `recurrent_laryngeal`, generic `myelinated_A_only`,
   generic `unmyelinated_C_only`. Each entry must carry citation text.
3. In `ui/drawers/fibers.py`, render a `VSelect` bound to
   `state.pop_preset_choice`. On change, populate `state.pop_rows_<branch>`
   for every detected branch, clipping by `branch_label`. Show citation
   text below the dropdown.
4. If preset includes a row with `mrg_supported=False` AND the user's
   `state.fiber_model` is MRG-only, surface an orange chip "C-fibers
   skipped under MRG model" in the row table. Do *not* silently drop.
5. Live preview: re-render the KDE figure into a small side panel before
   the user commits (button: "Apply preset"). No state mutation until
   apply.

**Verify.**
- Migration smoke items 9 + 11.
- Smoke addition F1.1.
- Save+reload: preset choice persists across project reopen.

**Rollback.** Revert. Existing per-row fields untouched, so no migration.

---

## F1.2 — Publication-grade figure styling presets

**Goal.** Make every existing figure renderable at paper-grade DPI / font /
palette without re-plotting in matplotlib. Pure infra — F2.3 will surface
it project-wide.

**Depends on.** Nothing.

**Touches.**
- New: `golgi/figures/export.py` (`FigureExportPreset`, `apply_preset`,
  `render_publication`).
- `golgi/figures/util.py` — `_fig_to_data_uri` already exists; add
  `_fig_to_file(fig, path, preset)` next to it.
- All `_build_*_figure` / `_render_*_plot` functions in
  `golgi/figures/{fem,fiber,population,cuff,cole_cole,mesh_stats}.py` —
  add an optional `preset: FigureExportPreset | None = None` argument.
  Default `None` = current behaviour. Do **not** restyle anything when
  `preset is None`.

**Do.**
1. Define:
   ```python
   @dataclass(frozen=True)
   class FigureExportPreset:
       name: str                       # "screen" | "paper-300" | "paper-600"
       fmt: str                        # "png" | "pdf" | "svg" | "eps"
       dpi: int                        # used for PNG; ignored for vector
       width_in: float                 # figure width in inches
       height_in: float
       font_family: str                # "DejaVu Sans" | "Times New Roman"
       font_size_pt: float
       palette: str                    # "default" | "viridis-cb" | "ibm-cb" | "gray"
       use_latex: bool = False         # matplotlib mathtext OK; full TeX optional
   SCREEN = FigureExportPreset(name="screen", fmt="png", dpi=120, ...)
   PAPER_300 = FigureExportPreset(name="paper-300", fmt="pdf", dpi=300, ...)
   ```
2. `apply_preset(fig, preset)` mutates a matplotlib `Figure` (set
   `dpi`, `set_size_inches`, walk axes → set tick / label fonts / cmap).
   For Plotly figures: write a sibling `apply_preset_plotly(fig, preset)`
   that updates `fig.layout.font`, `paper_bgcolor`, colorway.
3. `render_publication(fig, path, preset)` — matplotlib: `fig.savefig`
   with format / dpi / bbox_inches='tight'. Plotly: `pio.write_image`
   (requires `kaleido`; pin in `requirements-frozen.txt`).
4. Re-route the three legacy matplotlib renderers (`figures/fem.py:845`,
   `:1074`, `:1259`) through `apply_preset` when `preset is not None`.
   The Plotly tiles in the analysis drawer are untouched at this step —
   F2.3 wires the UI.
5. Add CB-safe palettes: `viridis-cb` (matplotlib `cividis`), `ibm-cb`
   (the IBM 5-colour palette literal). Use `cividis` for the FEM slice
   heatmap so anyone running this preset gets accessible output by
   default.

**Verify.**
- Import-check script: `from golgi.figures.export import PAPER_300`.
- Migration smoke (no figure should regress when `preset is None`).
- Smoke addition F1.2.
- Visual: render `figures/fem.py:_build_fem_axis_figure(..., preset=PAPER_300)`
  to a temp PDF; open in a viewer; fonts + DPI obey the preset.

**Rollback.** Revert. Optional argument is back-compat default-None.

---

# Phase F2 — Comparative-analysis foundation

The three features that turn golgi from a viewer into a study tool.

## F2.1 — Parameter sweep + threshold-finder + recruitment curves

**Goal.** Produce recruitment-curve and per-fiber threshold artefacts
without leaving the app.

**Depends on.** F1.2 (so sweep outputs can be exported paper-grade).

**Touches.**
- New: `golgi/pipeline/sweep.py`, `golgi/jobs/schemas.py` (`SweepRequest`,
  `SweepResult`), `golgi/figures/recruitment.py`.
- `golgi/pipeline/fiber_sim.py:275` — extract the "given Ve_mV + pulse →
  spike" inner loop into a pure function callable per-amplitude without
  re-loading anything.
- `golgi/ui/drawers/analysis.py` — new "Sweep" subtab under the existing
  analysis drawer.
- `golgi/state_defaults/__init__.py` — register sweep defaults.
- New: `golgi/state_defaults/sweep.py`.

**Do.**
1. Add to `golgi/jobs/schemas.py`:
   ```python
   @dataclass
   class SweepAxis:
       param: str           # "I_stim_mA" | "fiber_diameter_um" |
                            # "pulse_width_us" | "sigma_endo" | ...
       kind: str            # "list" | "linspace" | "logspace" | "bisect"
       values: list[float]  # for list / linspace / logspace (resolved upstream)
       target_metric: Optional[str] = None  # for bisect: "spike_count >= 1"
       bisect_lo: Optional[float] = None
       bisect_hi: Optional[float] = None
       bisect_tol: Optional[float] = None
   @dataclass
   class SweepRequest:
       project_dir: Path
       axes: list[SweepAxis]            # cartesian for >1 axis
       fiber_indices: list[int] | None  # None = all
       cache_dir: Path                  # results sharded by hash
       max_workers: int = 4
       seed: int = 0
   @dataclass
   class SweepResult:
       request: SweepRequest
       grid_shape: list[int]
       threshold_uA: Optional[np.ndarray] = None  # shape (n_fibers,) for bisect
       activated: Optional[np.ndarray] = None     # shape (*grid_shape, n_fibers) bool
       extras_path: Path                          # parquet/npz with rich metrics
   ```
2. **Linearity shortcut.** Because the quasi-static FEM is linear in
   stim current, an `I_stim_mA` axis is implemented by **rescaling the
   cached `paths_Ve.npz`** (multiply by `I_stim / I_stim_baseline`) and
   re-running only `pipeline/fiber_sim.py`. Detect this case in
   `sweep.py` and skip the FEM driver. Any axis that touches σ, mesh,
   electrode geometry, or fiber paths re-enters the appropriate
   upstream stage.
3. **Threshold-finder.** Add `find_threshold_amp(fiber_idx, lo, hi, tol)`
   that bisects `I_stim_mA` against "≥ 1 propagating AP" criterion.
   Persist `thresholds.npz` (shape `(n_fibers,)`, units µA) next to
   `paths_Ve.npz`. Brent over bisection if it converges faster; bisect
   guarantees ≤ ⌈log₂((hi-lo)/tol)⌉ sims per fiber (target ≤ 8).
4. **Fan-out.** Use the existing `InProcessRunner` for per-fiber
   evaluation. Honour `CancelToken` between fibers AND between grid
   cells. Stream a progress message per cell to `on_line` so the busy
   lightbox shows real progress.
5. **UI.** New "Sweep" tab in `ui/drawers/analysis.py`. Controls:
   - Axis selector dropdown (param + kind).
   - Range entry: list / linspace start–stop–n / logspace start–stop–n.
   - "Run threshold finder" toggle (replaces explicit amplitude axis).
   - "Estimate cost" button — shows `n_cells × n_fibers × ~ms_per_sim`.
   - Plotly figure: recruitment curve (mean across fibers, ribbons per
     branch / per fiber type), threshold-vs-diameter scatter,
     fiber-by-fiber activation heatmap.
6. Persist `sweep_<sha>.npz` in `<project>/sweeps/`. SHA is over
   `SweepRequest.serialize()` — same request hits cache instantly.

**Verify.**
- Migration smoke 8 + 10 + 11.
- Smoke addition F2.1.
- Cancel mid-sweep: lightbox dismisses cleanly, partial `sweep_<sha>.npz`
  is *not* written (atomic move on success only).
- Hash-cache: re-running same sweep is a no-op (< 100 ms).

**Rollback.** Revert. No changes to existing schemas (`SweepRequest` is
new), so old projects unaffected.

---

## F2.2 — Reproducible study bundle

**Goal.** "Send this zip to a reviewer; they can re-run the central
simulation in one command."

**Depends on.** F2.1 (so sweep results are included in the bundle).

**Touches.**
- New: `golgi/projects/bundle.py`, `golgi/projects/replay.py`.
- New: `golgi/cli.py` (small `argparse` shim) — entry points
  `golgi import` and `golgi replay`. Wire from `golgi.py` `main()`.
- `golgi/ui/dialogs/project_detail.py` — add "Export study…" /
  "Import study…" buttons.
- `golgi/auth/audit.py` — log `study_exported` / `study_imported` events.

**Do.**
1. `bundle.export_study(project_dir, out_zip)` writes a zip containing:
   - `MANIFEST.json` — schema below.
   - `inputs/nerve.stl` (or `.nas` / `.obj`) — the original import.
   - `configs/mesh_config.json`, `electrode_config.json`,
     `nerve_paths_seed_config.json` (all already on disk).
   - `geometry/nerve.msh`, `geometry/nerve_paths_fibers.npz`,
     `geometry/nerve_paths_caps.json`.
   - `fem/axis_line.npz`, `slice_volume.npz`, `paths_Ve.npz`,
     `nerve_surface_Ve.npz`, `Ve.xdmf`+`.h5`, `E.xdmf`+`.h5`.
   - `sims/fiber_sim_cache.json`, `pop_state.json`,
     `sweeps/sweep_<sha>.npz` (F2.1).
   - `env/requirements-frozen.txt`, `env/golgi_version.txt`.
   - `audit/audit_excerpt.json` (project-scoped audit rows).
2. `MANIFEST.json` shape:
   ```json
   {
     "golgi_version": "...",
     "exported_at": "...",
     "exported_by": "<username>",
     "dag": [
       {"stage": "mesh",       "inputs": ["configs/mesh_config.json", "inputs/nerve.stl"],
        "outputs": ["geometry/nerve.msh"],          "sha256": "..."},
       {"stage": "fem",        "inputs": ["geometry/nerve.msh", "configs/electrode_config.json"],
        "outputs": ["fem/axis_line.npz", "fem/paths_Ve.npz"], "sha256": "..."},
       {"stage": "fibers",     ...},
       {"stage": "fiber_sim",  ...},
       {"stage": "pop_sim",    ...},
       {"stage": "sweep",      ...}
     ]
   }
   ```
   `sha256` is over canonicalised inputs+outputs so replay can verify.
3. `bundle.import_study(zip, target_dir, owner_user_id)` inverse op;
   re-registers the project against the auth DB and replays the
   `audit_excerpt.json` events under the importing user with a
   "(imported from <original_user>@<exported_at>)" suffix.
4. `replay.replay_study(zip, headless=True)` walks the DAG in order,
   re-runs each stage (calls `pipeline/*` drivers), and asserts each
   output's sha256 matches `MANIFEST.json`. Exit 1 on first mismatch,
   printing the diverging stage + output. This is the actual
   reproducibility check.
5. `cli.py` adds:
   ```
   golgi export <project_dir> <out.zip>
   golgi import <in.zip>
   golgi replay <in.zip> [--headless] [--check-only]
   ```
6. UI buttons sit in the project detail dialog footer and in the navbar File menu as submenu item. "Export study"
   shows a progress spinner (same that we have in the other busy lightboxes); the resulting zip path is copied to clipboard. 
   There should also be a "Import study" under the navbar File menu as submenu item. It should open up a dialog that lets upload an exported study (shows progress spinner while uploading). Once uploaded, there should be a button in the dialog that lets trigger the reproduction run. This should then show the same progress spinner that we have in other busy lightboxes. Dialog should only close once the study completed end to end. The user should have the ability to cancle at any time by clicking a Cancle button.

**Verify.**
- Migration smoke 12.
- Smoke addition F2.2.
- Round-trip: export → wipe target dir → import → re-open → audit log
  shows the import row + the original events tagged as imported.
- `golgi replay study.zip --check-only` exits 0 on a freshly-exported
  bundle; intentionally mutating one byte in `paths_Ve.npz` makes it
  exit 1 with a clear "stage `fem` output diverges" message.

**Rollback.** Revert. Bundles produced before revert remain readable
because the schema is JSON.

---

## F2.3 — Figure export surface: per-panel buttons + bulk-export tab + Generate Report

**Goal.** Three independent consumers of the F1.2 export infrastructure:
1. **Per-panel export buttons** on every drawer/panel — single-figure
   PDF/SVG/PNG with a small format popover, plus a "Bundle this category"
   button on each category that bundles all panels in that view.
2. **Bulk Figure Export tab** (the original F2.3 plan) — full-pane
   project-wide picker with thumbnails for batch export.
3. **Generate Report** (new) — top-level navbar action that writes a
   single multi-page PDF combining a cover page, reproducibility
   appendix, conductivity table, electrode config table, and every
   selected figure + 3D viewport render, with bibliography + audit
   excerpt + ToC + cancellation warnings as auto-includes.

All three share `figures/registry.py` (single source of truth for what
figures exist) and `figures/export.py` (rendering primitives from F1.2).

**Depends on.** F1.2 (`FigureExportPreset` infra; landed). F2.1 is
NOT a hard prereq but sweep figures only appear in F2.3 once F2.1
ships. F2.2 is NOT a hard prereq but Generate Report can embed F2.2
hashes/manifest if it's available.

**Touches.**
- New: `golgi/figures/registry.py` — single source of truth for the
  "what figures + 3D-renders exist in this project right now?" question.
- New: `golgi/figures/render3d.py` — off-screen pyvista capture for the
  "viewport with X hidden / Y hidden" report variants. Uses a
  caller-supplied `report_camera` so all variants of one category
  share the SAME camera angle (per clarifying question 3).
- New: `golgi/figures/report.py` — `generate_report(project_dir,
  spec, out_pdf)` that walks the report section list and writes a
  multi-page PDF via `matplotlib.backends.backend_pdf.PdfPages`
  (per clarifying question 2 — no new deps).
- New: `golgi/ui/drawers/exports.py` — full-pane bulk Exports tab.
- New: `golgi/ui/dialogs/generate_report.py` — modal: section
  checkboxes + preset selector + "Capture report camera" button +
  output path picker + Generate CTA.
- New: `golgi/ui/components/figure_export_btn.py` — reusable
  per-panel export button. Renders a small Vuetify icon button +
  popover (Format / Preset / Export). Used everywhere a figure
  panel exists.
- New: `golgi/state_defaults/exports.py` — `report_camera_pose`,
  `report_section_choices`, `export_default_preset`,
  `export_default_format`.
- `golgi/ui/navbar.py` — add TWO top-level entries: "Exports"
  (icon: download) and "Report" (icon: file-document). Sit between
  "Analysis" and the user avatar.
- `golgi/figures/export.py` — extend with `bulk_export(project_dir,
  selection, preset, out_dir)` and `export_single(fig_id, ctx,
  preset, out_path)`.
- Per-view UI files — add a `figure_export_btn` next to every
  panel header. Touches at minimum:
  - `golgi/ui/drawers/import_drawer.py` — Import panel.
  - `golgi/ui/drawers/cuff_electrodes.py` — Cuff panel.
  - `golgi/ui/drawers/mesh.py` — Mesh panel.
  - `golgi/ui/drawers/fibers.py` — Fiber trajectories panel.
  - `golgi/ui/drawers/analysis.py` — every FEM / single-fiber /
    population sub-panel inside the analysis drawer. ONE button per
    sub-panel + one "Bundle this view" button per drawer.

**Do.**
1. **Figure registry.** A figure is identified by a stable string ID. In
   `figures/registry.py`:
   ```python
   @dataclass
   class FigureSpec:
       id: str                          # "fem.axis_line" | "pop.kde" | ...
       title: str                       # display name
       category: str                    # "Cuff" | "Mesh" | "FEM" | "Fibers"
                                        # | "Population" | "Sweep" | "Conductivity"
       builder: Callable[[Context], Any]   # returns a fig (mpl or plotly)
       data_dumper: Callable[[Context, Path], None]  # writes CSV
       availability: Callable[[Context], bool]       # are inputs on disk?

   REGISTRY: list[FigureSpec] = [
       FigureSpec("cuff.preview",          "Cuff geometry preview",        "Cuff",         ...),
       FigureSpec("mesh.quality_hist",     "Tet-quality histogram",        "Mesh",         ...),
       FigureSpec("fem.axis_line",         "Vₑ / E_z along centerline",    "FEM",          ...),
       FigureSpec("fem.slice_volume",      "Vₑ slice heatmap (per z)",     "FEM",          ...),
       FigureSpec("fem.activation_fn",     "Activation function ∂²Vₑ/∂s²", "FEM",          ...),
       FigureSpec("fiber.pulse",           "Stimulus pulse waveform",      "Fibers",       ...),
       FigureSpec("fiber.propagation",     "Vm propagation heatmap",       "Fibers",       ...),
       FigureSpec("fiber.waterfall",       "Vm waterfall (sampled nodes)", "Fibers",       ...),
       FigureSpec("pop.kde",               "Diameter KDE per branch/row",  "Population",   ...),
       FigureSpec("pop.xsec_cuff",         "Cross-section @ cuff center",  "Population",   ...),
       FigureSpec("pop.activation",        "Activation map",               "Population",   ...),
       FigureSpec("sweep.recruitment",     "Recruitment curve",            "Sweep",        ...),  # F2.1
       FigureSpec("sweep.threshold_diam",  "Threshold vs diameter",        "Sweep",        ...),  # F2.1
       FigureSpec("cole_cole.sigma_f",     "σ(f) (Cole-Cole)",             "Conductivity", ...),
       # 3D viewport renders for the Generate Report path. Each
       # consumes `report_camera_pose` and writes a PNG/PDF via
       # `figures/render3d.py`.
       FigureSpec("render3d.fem_full",         "FEM result · all visible",       "FEM3D",      ...),
       FigureSpec("render3d.fem_no_muscle",    "FEM result · muscle hidden",     "FEM3D",      ...),
       FigureSpec("render3d.fem_no_epi",       "FEM result · epineurium hidden", "FEM3D",      ...),
       FigureSpec("render3d.fem_no_endo",      "FEM result · endoneurium hidden","FEM3D",      ...),
       FigureSpec("render3d.mesh_quality_all", "Mesh quality · all regions",     "Mesh3D",     ...),
       FigureSpec("render3d.mesh_muscle",      "Mesh quality · muscle only",     "Mesh3D",     ...),
       FigureSpec("render3d.mesh_endo",        "Mesh quality · endoneurium only","Mesh3D",     ...),
       FigureSpec("render3d.mesh_epi",         "Mesh quality · epineurium only", "Mesh3D",     ...),
       FigureSpec("render3d.mesh_cuff",        "Mesh quality · cuff only",       "Mesh3D",     ...),
       FigureSpec("render3d.fibers_epi",       "Fibers · epineurium @ α=0.5",    "Fibers3D",   ...),
       FigureSpec("render3d.electrode_geom",   "Electrode geometry",             "Electrode3D",...),
       FigureSpec("render3d.electrode_polar",  "Electrode · anode/cathode coloured", "Electrode3D",...),
   ]
   ```
   Each `builder` re-uses the existing `_build_*_figure` /
   `_render_*_plot` for matplotlib/Plotly entries; the `render3d.*`
   entries route through `figures/render3d.py` which sets up an
   off-screen pyvista plotter with the supplied `report_camera_pose`
   (so all variants of one category use the same camera —
   clarifying-question answer #3).
2. **Layout (new tab).** `ui/drawers/exports.py` renders a full-pane
   view (not a side drawer) when active:

   ```
   ┌─────────────────────────────────────────────────────────────────┐
   │ Exports                                       [Refresh thumbs] │
   ├─────────────────────────────────────────────────────────────────┤
   │ Preset: [paper-300 ▾]   Format: [PDF ▾]   Palette: [viridis-cb ▾]│
   │ Output dir: <project>/figures_export/<timestamp>/  [pick…]      │
   │ [✓] Dump underlying data as CSV next to each figure              │
   │ [✓] Skip figures whose inputs are stale (no source data on disk) │
   │                                                                  │
   │ ─── Cuff ────────────────────────────────────────────────────── │
   │ [✓] cuff.preview         [thumbnail]   ✅ available             │
   │ ─── Mesh ────────────────────────────────────────────────────── │
   │ [✓] mesh.quality_hist    [thumbnail]   ✅ available             │
   │ ─── FEM ─────────────────────────────────────────────────────── │
   │ [✓] fem.axis_line        [thumbnail]   ✅ available             │
   │ [✓] fem.slice_volume     [thumbnail]   ✅ available             │
   │ [✓] fem.activation_fn    [thumbnail]   ✅ available             │
   │ ─── Fibers ──────────────────────────────────────────────────── │
   │ [ ] fiber.pulse          —             ⚠ no fiber sim run yet   │
   │ ...                                                              │
   │                                                                  │
   │           [Select all] [Select available] [Clear]                │
   │                          [Export N figures ▶]                    │
   └─────────────────────────────────────────────────────────────────┘
   ```

   - **Thumbnails** are generated lazily on tab open by calling each
     available `builder` at low DPI (60) and converting to a base64
     data URI via the existing `_fig_to_data_uri`. Cache thumbnails
     in `<project>/.cache/thumbs/<fig_id>.png`; invalidate when the
     `builder`'s input mtimes change.
   - **Availability** = `spec.availability(ctx)` returns True iff the
     source data (e.g. `paths_Ve.npz` for `fem.axis_line`) exists.
     Unavailable figures show greyed-out and uncheckable.
3. **Bulk export.** `bulk_export(project_dir, [fig_id, ...], preset,
   out_dir)`:
   - For each fig_id: call `builder(ctx)` → `render_publication(fig,
     out_dir/<fig_id>.<fmt>, preset)`.
   - If "dump CSV" is on: call `data_dumper(ctx, out_dir/data/<fig_id>.csv)`.
   - Run in `InProcessRunner` so the busy lightbox shows per-figure
     progress; honour `CancelToken`.
   - At the end, write `out_dir/MANIFEST.json` listing `(fig_id, path,
     preset, source_data_sha256)` so figures are traceable to data.
4. **CSV dumpers.** Each `FigureSpec` ships a `data_dumper`. E.g.:
   - `fem.axis_line` → `s_mm, ve_mean_mV, ve_std_mV, ez_mean_Vpm, ez_std_Vpm`
   - `pop.kde` → `branch, type, diameter_um, density`
   - `sweep.recruitment` → `i_stim_mA, branch, frac_activated, ci_lo, ci_hi`
   Reuse the same arrays the builders already consume (`np.load(...)`).
5. **CLI parity.** Add to `cli.py`:
   ```
   golgi export-figures <project_dir> --preset paper-300 --fmt pdf [--out DIR]
   ```
   So bulk export is also scriptable from F4.1.
6. **Project bundle hook.** When F2.2's bundle is written, optionally
   include `figures_export/latest/` if it exists, so reviewers get the
   rendered figures without re-running anything.

7. **Per-panel export button (`ui/components/figure_export_btn.py`).**
   Tiny reusable Vuetify icon-button (download icon, density=compact)
   that takes a `fig_id` and renders a popover on click:
   ```
   ┌───────────────────────────────┐
   │  Export figure                │
   │  Format:  [PDF ▾]             │
   │  Preset:  [paper-300 ▾]       │
   │  Path:    <project>/exports/  │
   │           fem.axis_line.pdf   │
   │                  [Export ▶]   │
   └───────────────────────────────┘
   ```
   (Clarifying answer #4 — every button opens a format popover.)
   Per-panel placement: next to the panel header on every drawer.
   Per-drawer "Bundle this view" button at the bottom of each
   drawer's panel collection that exports every available figure in
   that drawer to a sibling subfolder
   (`<project>/exports/<timestamp>/<drawer_name>/`).

8. **Generate Report (`ui/dialogs/generate_report.py`).** Modal
   triggered from the navbar "Report" entry:
   ```
   ┌────────────────────────────────────────────────────────────────┐
   │  Generate report                                               │
   ├────────────────────────────────────────────────────────────────┤
   │  Preset:  [paper-300 ▾]      Output:  <project>/reports/        │
   │                                       report_<timestamp>.pdf    │
   │                                                                 │
   │  Report camera:  iso (default)  [Capture current viewport pose] │
   │                                                                 │
   │  Sections to include:                                           │
   │  Auto-included: cover page · reproducibility appendix ·         │
   │                 conductivity table · electrode config table ·   │
   │                 bibliography · audit excerpt · ToC · partial-   │
   │                 run warnings                                    │
   │                                                                 │
   │  [✓] Electrode design                                           │
   │      Includes: 3D geometry + anode/cathode-coloured render      │
   │  [✓] Mesh results                                               │
   │      Includes: 5× region renders + quality histogram + table    │
   │  [✓] Fiber trajectories                                         │
   │      Includes: 3D render (epineurium α=0.5) + branch summary    │
   │  [✓] FEM results                                                │
   │      Includes: 4× viewport variants + axis + slice + AF plots   │
   │  [✓] Single-fiber simulation                                    │
   │      Includes: pulse + propagation + waterfall + sim config     │
   │  [✓] Population simulation                                      │
   │      Includes: KDE + xsec @ cuff + activation map + pop config  │
   │                                                                 │
   │                                       [Cancel]  [Generate ▶]    │
   └────────────────────────────────────────────────────────────────┘
   ```
   Sections whose prereqs are missing are still listed but with a
   greyed checkbox and an inline note ("no FEM run yet — section
   will appear as a placeholder page"). Default policy: include
   anyway, with a placeholder page explaining what's missing.

9. **Report layout (PDF page order).** `figures/report.py` builds in
   this order (auto-included items in bold):
   - **Cover page**: project name, generated by, generation date,
     golgi version, mesh sha256, FEM sha256.
   - **Table of contents** (auto-generated from sections present).
   - **Electrode design** section (if selected):
     * `render3d.electrode_geom` figure (full page)
     * `render3d.electrode_polar` figure (full page)
     * Electrode config table (one row per contact: id, geometry,
       polarity, current).
   - **Mesh results** section: 5× `render3d.mesh_*` figures + the
     existing tet-quality histogram + the per-tag stats table.
   - **Fiber trajectories** section: `render3d.fibers_epi` + branch
     summary table (n_fibers, mean/min/max/std length per branch).
   - **FEM results** section: 4× `render3d.fem_*` figures + every
     Plotly figure from the FEM analysis tab (axis, slice, AF).
   - **Single-fiber simulation** section: pulse + propagation +
     waterfall + a "simulation config" table (backend, model,
     diameter, pulse amplitude/width/onset/gap).
   - **Population simulation** section: KDE + cross-section + activation
     map + a "population config" table (preset name + citation if
     applicable; per-branch row mixture; pop_seed).
   - **Conductivity table** (auto): per tissue, σ at stim frequency
     + Cole-Cole params (ε∞, σ_ionic, dispersions) + IT'IS source.
   - **Reproducibility appendix** (auto): dump of every
     `*_config.json` + per-stage sha256 + `requirements-frozen.txt`
     line for `golgi`. Pairs cleanly with F2.2's `golgi replay`.
   - **Bibliography** (auto): aggregated citations from active
     `pop_preset` + any validation datasets (F3.3) + any
     hand-entered references.
   - **Audit excerpt** (auto): timestamped table of stage runs
     within this project's lifetime.

10. **`figures/render3d.py` mechanics.** Takes
    `(visibility_dict, camera_pose) → png_bytes` via off-screen
    pyvista. The visibility_dict gates which mesh tags / actors are
    added before screenshot. A "report camera" is a `dict(position,
    focal_point, view_up, view_angle, zoom)`; default is computed
    from the bounding box (iso angle: azimuth 35°, elevation 20°);
    user can override via the "Capture current viewport pose"
    button in the Generate Report dialog (clarifying answer #3
    option C).

**Verify.**
- Smoke addition F2.3.
- Open "Exports" with a fresh project (no FEM run): only
  `cuff.preview` + `cole_cole.sigma_f` are available; the rest
  greyed out.
- Run the migration smoke top to bottom; return to Exports — every
  figure is now available; thumbnails populate within a few seconds.
- "Export all" produces N PDFs + N CSVs + MANIFEST.json in
  `<project>/figures_export/<timestamp>/`. Open one PDF: respects
  the selected preset's DPI / font / palette.
- Per-panel button on the Axis-plot panel → popover → Export →
  one-file PDF appears at the indicated path. "Bundle this view"
  on the Analysis drawer → multiple PDFs in a sibling folder.
- Generate Report → produces one `report_<timestamp>.pdf` in
  `<project>/reports/`. Page count = #sections + auto items;
  cover page renders project metadata; FEM section shows four
  3D-render pages all sharing the same camera.
- Cancel mid-bulk-export OR mid-report: lightbox dismisses, partial
  files cleaned up.

**Rollback.** Revert. Removes the new tab + dialog + nav entries;
per-panel buttons + Generate Report disappear. F1.2 export infra
keeps working for headless use.

---

# Phase F3 — Comparative analysis layer

The publishable-research features. All three layer on top of F2.

## F3.1 — Multi-electrode FEM solve

**Goal.** Lift the explicit "multi-electrode not yet supported" limit at
[golgi/pipeline/fem.py:320-326](golgi/pipeline/fem.py#L320-L326). Each
design is a full `ElectrodeConfig`; the mesh is shared.

**Depends on.** Nothing structural; recommended after F2.2 so designs
are bundle-exportable.

**Touches.**
- `golgi/pipeline/fem.py` — accept `list[ElectrodeConfig]` instead of one.
- `golgi/jobs/schemas.py` — no change to `ElectrodeConfig`; just allow
  multiple per solve.
- `golgi/compute/solve_nerve.py` — already deterministic; loop over
  designs, write into `<out>/fem/<design_id>/`.
- `golgi/ui/drawers/cuff_electrodes.py` — already supports multiple
  electrodes; just stop ignoring designs beyond #0 in the FEM trigger.
- `golgi/ui/drawers/analysis.py` — add a design selector dropdown that
  reads from `<out>/fem/*/` subfolders.

**Do.**
1. Change `pipeline/fem.py` driver to iterate over the design list,
   writing each FEM result into `<out>/fem/<design_id>/` (one folder
   per `ElectrodeConfig.name`; collide-safe with `_2`, `_3` suffixes).
2. **Superposition shortcut.** If all designs share a common set of
   contacts and differ only in active/ground role + amplitude, solve
   *once per unique contact* at unit current, then combine in
   post-processing. Detect via a `set(patch.id for patch in patches)`
   comparison across designs. Skip the shortcut when geometry differs.
3. Persist a `<out>/fem/designs.json` index: `[{id, name, n_patches,
   I_stim_mA, sha256}, ...]`.
4. Wire the analysis drawer's selector to switch which design's
   `axis_line.npz` / `slice_volume.npz` / `paths_Ve.npz` is plotted.
   Default to the most recent.
5. Plumb `design_id` through to `pipeline/fiber_sim.py` so per-fiber
   sims are scoped to one design's `paths_Ve.npz`. Sims persist under
   `<out>/sims/<design_id>/`.

**Verify.**
- Migration smoke 8 + 9 + 10 + 11 (all per-design).
- Smoke addition F3.1.
- Two-design solve: `fem/design_A/` and `fem/design_B/` both populated;
  switching designs in the analysis drawer flips every figure.
- Sweep (F2.1) over an axis with two designs in scope produces curves
  in different colours.

**Rollback.** Revert. Existing single-design projects unaffected (the
folder layout falls back to `fem/default/` for them).

---

## M1 — Multi-contact anode/cathode picker (tri- / quadri- / N-polar)

**Goal.** Lift the implicit "one active + one ground" assumption in
the cuff electrode UI. A single cuff design must support arbitrary
N-polar configurations (e.g. tripolar — anode at contact 1, cathode
at contacts 2+3 in parallel — or quadripolar guarded bipoles), with
no fixed cap on how many contacts can be assigned to each role.

**Depends on.** F3.1 (multi-electrode FEM solve) is NOT a prereq —
M1 is single-design but multi-contact. The two compose: a Cuff
design with N polarities, of which there can be N designs.

**Touches.**
- `golgi/jobs/schemas.py` — `ElectrodePatch.role` already accepts
  `"active" | "ground"`; extend to `"anode" | "cathode" | "ground" |
  "off"`. Migration: existing projects with `role == "active"` map to
  `anode`; `role == "ground"` stays. Add a per-patch
  `current_fraction: float | None` (0..1, normalised within polarity
  group) so a user can specify "split 60/40 between contacts 2 and 3".
- `golgi/pipeline/fem.py` — the patch-summing FEM driver already
  sums currents per active patch; extend to handle multiple anodes +
  multiple cathodes with the per-patch `current_fraction` weighting.
  Total injected current per polarity group must be conserved
  (Σ I_anode = −Σ I_cathode = `I_stim`).
- `golgi/scene/electrode_patches.py` (after W1.4) — render anodes in
  red, cathodes in blue, off-contacts in grey. (Matches the report
  electrode_polar render.)
- `golgi/ui/drawers/cuff_electrodes.py` — replace the single
  "active / ground" radio per contact with a per-contact polarity
  dropdown ({Anode, Cathode, Ground, Off}) + an optional current-
  fraction input (visible only when 2+ contacts share a polarity).
  Add a "sum check" chip per polarity group (green when fractions
  sum to 1.0, red otherwise — same idiom as the population sum
  check at app.py:13638).

**Do.**
1. Schema migration is back-compatible: deserialise treats
   `"active"` as `"anode"` for any existing on-disk
   `electrode_config.json`. New writes use the new vocabulary.
2. Compute injected currents per patch in
   `pipeline/fem.py` as:
   `I_patch = I_stim * sign(polarity) * (current_fraction or
   1/N_in_group)`. Sign = +1 for anode, −1 for cathode.
3. UI: replace the radio with a 4-option select; chain a small
   text-field for current_fraction visible when the polarity group
   has ≥ 2 contacts. Live update the patch-render via the existing
   scene group.
4. Add a "Quick presets" dropdown in the electrode picker:
   `Bipolar (2-contact)`, `Tripolar (longitudinal guard)`,
   `Tripolar (transverse guard)`, `Quadripolar guarded bipole`.
   Each preset just sets per-contact polarities for the currently-
   selected cuff geometry.

**Effort.** S–M (1 week). Mostly UI + a small FEM driver tweak.

**Impact.** High — unlocks the most-studied VNS configurations
(tripolar especially is the workhorse for selective stimulation in
the cuff-electrode literature).

**Verify.**
- Migration smoke item 5 (Cuff designer / Add electrode).
- Smoke addition M1: pick tripolar guard preset → FEM solve writes
  electrode_config.json with 1 cathode + 2 anodes, 50/50 fractions.
  Axis plot shows the expected dual-peak guarded field.
- Load an OLD project: polarity field reads `"anode"` even though
  the file still says `"active"`.

**Rollback.** Revert. Old projects continue to read as before.

**Vs. existing tools.** ASCENT supports arbitrary multi-contact
fractionation via `Sim` JSON. Sim4Life supports it via the contact
manager. Putting it in golgi is parity, not innovation; the lack of
it today is the main reason real-world VNS configs can't be modelled.

---

## F3.2 — Selectivity metrics + side-by-side comparison view

**Goal.** Answer "which cuff design is more selective?" inside the app.

**Depends on.** F3.1 (multi-design FEM) and F2.1 (recruitment curves).

**Touches.**
- New: `golgi/figures/selectivity.py`.
- New: `golgi/pipeline/selectivity.py` (pure post-processing on top of
  sweep results — no new compute job).
- `golgi/ui/drawers/analysis.py` — new "Compare designs" subtab.
- `golgi/figures/registry.py` (F2.3) — register `selectivity.bar`,
  `selectivity.threshold_ratio`, `selectivity.spatial_heatmap`.

**Do.**
1. Define metrics in `pipeline/selectivity.py`:
   - **Veraart selectivity index** per branch pair:
     `SI = (R_target − R_offtarget) / (R_target + R_offtarget)` at a
     given amplitude.
   - **Threshold ratio**: `T_offtarget / T_target` (high is good).
   - **Branch recruitment matrix**: `(n_designs, n_branches, n_amplitudes)`
     fraction activated.
   - **Spatial activation heatmap**: per fiber cross-section position,
     min amplitude to activate (colour scale).
   Inputs are the F2.1 `SweepResult` per design.
2. New "Compare designs" subtab in `ui/drawers/analysis.py`. Layout:
   left half — design A panel (axis plot + slice); right half — design
   B panel. Below: selectivity bar chart + threshold-ratio table.
3. The user picks "target branch" (e.g. the cardiac branch) from a
   dropdown; selectivity is computed against the union of all other
   branches as off-target. Allow custom target/off-target sets via a
   chip-input.
4. Register selectivity figures in `figures/registry.py` so they show
   up in F2.3's Exports tab.

**Verify.**
- Smoke addition F3.2.
- Two designs, single sweep over amplitude: selectivity bar shows two
  bars per target branch; threshold ratio table has correct values
  (verify against the underlying `threshold.npz`).
- Export selectivity figures via F2.3 → vector PDFs render correctly.

**Rollback.** Revert. Selectivity is read-only post-processing; nothing
upstream depends on it.

---

## F3.2d — Multi-design meshes share canonical nerve + muscle (fix-log)

**Goal.** When the user has multiple cuff designs on the same nerve,
switching `selected_design_id` should only move the cuff in the
viewport. The nerve and the muscle bbox should be SHAPE-identical
across every design's mesh.

**What didn't work — attempt 1 (shared mesh frame).** First fix
built every design's mesh in the ANCHOR design's cuff-local frame
as a single "shared mesh frame". Non-anchor designs ended up with
a TILTED cuff (rotated by `M_D.T @ anchor_M` in mesh space).
TetGen's `recoversubfaces` consistently failed on the second
design of real branched-VN inputs with cascades of "Two facets
exactly intersect" warnings — the rotated cap planes + cap-
polyline triangulation produced enough near-degenerate facets
that pymeshfix didn't catch but TetGen's exact predicates did.
The anchor design (whose cuff was at `(offset=0, R=I)` in mesh
space) always succeeded because that path collapses to the legacy
single-cuff geometry. Documented in commit `dd60fbf`.

**What works — attempt 2 (per-design cuff-local + shared canonical
muscle).** Each design's mesh is built in **its own cuff-local
frame** (cuff at origin axis-aligned, `+z` = the design's own
local nerve axis at its cuff site). TetGen always sees the
legacy single-cuff geometry — robust across every nerve we've
tested. To keep the nerve and muscle shape-identical across
designs anyway:

- The **nerve** comes from a single canonical-frame point cloud
  (`pts_pca - anchor_origin_pca`) transformed into each design's
  cuff-local via `p_local = (p_canonical - design_offset_canon)
  @ M_D`. Any two designs see the SAME nerve (just rotated /
  translated into different frames); when rendered back in the
  shared PCA-translated viewport frame, the round-trip is
  identity so all designs co-render with byte-equivalent nerve
  geometry (modulo TetGen Steiner-point churn).
- The **muscle bbox** is auto-fit ONCE in canonical (axis-
  aligned to canonical `+z`), then transformed into each
  design's cuff-local via the SAME `(M_D, design_offset_canon)`.
  The transformed muscle is tilted in design-local space —
  TetGen handles arbitrary closed surfaces fine — but the
  underlying shape is identical across designs, so the viewport
  also shows a single shared muscle bbox after the round-trip.

For the anchor design, `M_D = anchor_M` and
`design_offset_canon = 0` — so the rotation step collapses to
the legacy `_design_local_to_anchor_frame` semantics, and the
single-cuff case is bit-exact identical to the pre-F3.2 path.

**Specifically:**

- `golgi/pipeline/plc.py`:
  - `assemble_multi_domain_plc` reverted to the legacy axis-
    aligned-cuff-at-origin contract (no `cuff_offset_m` /
    `cuff_R` parameters — those were attempt 1's API).
  - New `build_muscle_pieces_for_nerve(pre_pts, ...)` factor —
    returns a dict `{lat, cap_lo, cap_hi, seed, params}` of the
    auto-fit muscle pieces in the input frame.
  - New `transform_muscle_pieces(pieces, R, offset)` — applies
    a rigid transform to every vertex array in the muscle
    pieces dict (callers use this to map canonical → design-
    local).
  - `assemble_multi_domain_plc` gains an optional
    `muscle_pieces` parameter; when provided it bypasses the
    auto-fit and uses these pieces directly. When None
    (legacy single-cuff path) muscle is auto-fit to the input
    nerve.
- `golgi/pipeline/mesh.py`:
  - Precomputes canonical nerve + canonical muscle pieces ONCE
    per batch.
  - Per design D: computes `(M_D, design_offset_canon)`, maps
    canonical nerve → design-local, transforms canonical muscle
    pieces → design-local via `transform_muscle_pieces`, calls
    `assemble_multi_domain_plc(nerve_design_local, ...,
    muscle_pieces=muscle_design_local)`.
  - On-disk `<out>/designs/<eid>/nerve.msh` is in D's cuff-
    local frame (what solve_nerve.py reads).
  - Post-build, rotates mesh nodes + per-region surfaces by
    `M_D.T` and translates by `design_offset_canon` before
    handing them to `geom` so the viewport (PCA-translated
    frame) co-renders mesh + cuff actors + fibers.
  - Writes per-design `nerve_surface_pts.npz` at mesh-build
    time from the freshly-built region surface in D's cuff-
    local — no longer depends on which design is currently
    focused (which was the root cause of the
    "nerve_surface_Ve has X pts but region_surfaces[1] has Y
    pts" mismatch).
- `golgi/pipeline/fem.py`:
  - Drops `cuff_offset_m` / `cuff_R_flat` from per-design
    MeshConfig writes — each design's mesh is already in cuff-
    local, so no transform is needed.
  - Writes `nerve_paths_fibers.npz` in EACH parent design's
    own cuff-local frame (`raw → PCA → design-local` via
    that design's `M_D` and cuff_origin), not in the anchor's
    frame. solve_nerve.py samples Vₑ at these points directly.
- `golgi/compute/solve_nerve.py`: unchanged — its existing
  defensive defaults (cuff transform = identity when MeshConfig
  has no `cuff_offset_m` / `cuff_R_flat`) are exactly right for
  the new per-design-local format. Same code also still handles
  attempt-1 projects (with the cuff fields populated) if you
  ever need to load one for diagnostics.
- `golgi/jobs/schemas.py:MeshConfig`: `cuff_offset_m` (3 floats)
  + `cuff_R_flat` (9 floats) optional fields stay on the schema
  for forward/backward compat — new solves leave them None.
- `golgi/scene/cuff_fit.py`: helpers used:
  `anchor_origin_pca_for_designs`, `_design_M`,
  `find_cuff_origin_pca`. `anchor_mesh_frame`,
  `design_cuff_transform`, `nerve_canonical_pts` (attempt-1
  API) are kept around in case other callers need them but
  are no longer used by the mesh / FEM pipeline.
- `golgi/app.py:_restore_mesh_from_disk`: uses the ACTIVE
  design's `M_D` + cuff_origin to rotate the on-disk mesh into
  the viewport frame (was attempt-1's `anchor_M.T` for every
  design — wrong for non-anchor designs after the per-design-
  local switch).

**Migration note.** Projects meshed under the attempt-1 commit
`dd60fbf` have on-disk meshes in the wrong frame. **Re-mesh +
re-solve** to pick up the new format. (The mismatch is silent:
solve_nerve.py with no `cuff_offset_m` in MeshConfig assumes
identity transform, which is wrong for an attempt-1 on-disk
mesh.)

**Unrelated drive-by fixes (still landed):**
- `app.py:_update_muscle_preview` tolerates empty-string Vuetify
  number-input values mid-edit.
- `app.py:do_add_design` no longer double-creates the Default
  config (was showing "Cuff 1 · Default" twice in the Solve
  picker).
- `app.py:_restore_fem_from_disk` calls `_request_render()` so
  Vₑ / E overlays refresh when switching active config in the
  analysis chip.

**Verify.**
- Build meshes for ≥ 2 designs on a real branched-VN. Both
  succeed (no `recoversubfaces` failure on design 2+).
- Toggle `selected_design_id`: only the cuff visibly moves in
  the viewport; the nerve and the muscle bbox stay put.
- Solve FEM on two configs on different parent designs back-
  to-back. Both succeed; no "Could not merge file" error.
- Switch the active config in the FEM analysis chip — the
  Vₑ-on-fibers, Vₑ-on-surface, and E-field overlays all
  refresh to the new design without the "length mismatch on
  tag 1" guard tripping.
- Rotate a non-anchor design by `cuff_rot_x_deg = 15°`. Its
  mesh builds cleanly (the cuff is still axis-aligned in D's
  own local frame; the rotation only enters at the viewport-
  rendering step + at the cuff actor placement step).

**Rollback.** Reverting this commit takes the codebase back to
the attempt-1 (shared-anchor-frame) state, which fails on
multi-design meshes with tilted cuffs. The plain pre-F3.2
single-cuff path remains intact further up the history via the
"M1 polish" commit (`0d8e3ba`).

---

## S1 — FEM surrogate: sub-second "what-if" for clinical iteration

**Goal.** Drop the per-twiddle FEM re-solve to sub-second. A clinician
comparing "anode contact 1 + cathode contact 2 @ 0.8 mA / 250 µs" vs
"anode 1 + cathode 3 @ 1.0 mA / 130 µs" should be able to flip between
them at ~10 Hz rather than wait 30 s — 30 min per change.

**Insight.** The quasi-static FEM is **linear in stimulus current per
contact**. Pre-solve one unit-current FEM per contact (a "contact
transfer-function basis" Φ_k(x)); any later configuration is a single
matrix–vector product over the basis.

**Depends on.** F3.1 (multi-electrode FEM — basis solve loops over
contacts) AND M1 (multi-contact picker — so the live UI can pick
arbitrary fractionations). Builds on existing pipeline; no ML.

**Touches.**
- New: `golgi/surrogate/contact_basis.py` — `build_contact_basis(
  project_dir, mesh_sha, contacts) -> ContactBasis` and
  `eval_basis(basis, currents_per_contact, fiber_paths) ->
  per_fiber_Ve(s)`. The basis is a `(n_contacts, n_fiber_pts)` float
  array cached on disk as `<project>/fem/<design_id>/contact_basis.npz`
  alongside `paths_Ve.npz`.
- `golgi/pipeline/fem.py` — when the user clicks "Build contact basis"
  in the FEM panel, the driver loops over contacts: for each, set
  that contact to 1 A and all others to 0 A → solve → sample on
  fiber paths → store column k. Hash-keyed on
  `(mesh_sha, σ_sha, fibers_sha, contact_geometry_sha)` so the basis
  is reused as long as those don't change.
- New: `golgi/ui/drawers/live_stim.py` — "Live stimulation" panel
  below the Analysis tab. Controls:
  - Per-contact polarity + current-fraction (re-uses M1's picker).
  - Total amplitude slider (mA).
  - Pulse width slider (µs).
  - Backend selector (pyfibers / axonml).
  Outputs update live (target ≤ 100 ms latency):
  - Per-branch recruitment fraction (bar chart).
  - Selected fiber's Vm propagation (uses `InProcessRunner` /
    `pipeline/fiber_sim.py`).
  - Selectivity index (F3.2 metric).
  A prominent badge marks every output `[surrogate]`; switching the
  badge to `[full FEM]` requires explicit "Solve full FEM" click
  (i.e. the basis becomes stale if σ / geometry change).
- New: `golgi/jobs/schemas.py` — `ContactBasisRequest`,
  `ContactBasisResult`. Same `serialize()` / `deserialize()` contract
  as the other Phase-6.2b schemas.
- `golgi/figures/registry.py` — register `surrogate.recruitment_live`
  and `surrogate.selectivity_live` so the panel's outputs can be
  exported via F2.3.

**Do.**
1. **Build the basis.** New button in the FEM Analysis panel:
   "Build contact basis (sub-second what-if)". On click, the driver
   loops `for k in contacts: solve(unit_current_on=k, all_others=0)`
   and stacks the per-contact Vₑ on each fiber-path point into
   `Φ ∈ ℝ^{n_contacts × n_fiber_pts}`. Persist as `contact_basis.npz`
   + a JSON manifest with the hash tuple.
2. **Eval.** `eval_basis(basis, currents_A) -> Ve_per_fiber` is a
   single matrix multiply `Φᵀ @ currents`. For typical sizes
   (16 contacts × 500 fiber-pts) this is < 1 ms.
3. **Live UI.** `ui/drawers/live_stim.py` wires the contact polarity
   picker (M1) + amplitude / PW sliders to `eval_basis` →
   `pipeline/fiber_sim.py` → live recruitment + selectivity figures.
   Throttle to 10 Hz to avoid swamping the InProcessRunner.
4. **Staleness detection.** The basis is keyed on
   `(mesh_sha, σ_sha, fibers_sha, contacts_sha)`. Any of these
   changing invalidates the basis: the panel switches the badge to
   `[surrogate stale — rebuild]` and disables outputs until the user
   re-builds or solves a full FEM.
5. **Falsifiability mode.** Add a debug toggle "Verify against full
   FEM" that picks the current live config, runs a full FEM, and
   reports the max |ΔVₑ| across all fiber points. Should be < 1 e-12
   if linearity holds exactly (any non-trivial delta = bug in the
   basis build).

**Effort.** Medium (2 weeks). Builds on existing FEM driver, no new
solver complexity, no ML.

**Impact.** Very high — clinical / programming iteration cadence
requires sub-second feedback. Also dramatically accelerates F2.1's
amplitude sweep (no per-amplitude re-solve at all for any-contact
configuration), F3.2's selectivity exploration, and F5.1's UQ
sampling for σ-perturbations within the basis's linearity envelope.

**Verify.**
- Smoke addition S1: build basis on a 2-contact bipolar cuff;
  evaluate at I=1 mA via surrogate; max |ΔVₑ| vs full FEM
  < 1e-12 in the "Verify against full FEM" mode.
- Live panel: drag amplitude slider 0.1 → 2.0 mA; recruitment bar
  + propagation tile update at ≥ 5 Hz without re-solving FEM.
- Invalidate σ_endo → live panel shows `[surrogate stale]`; rebuild
  → recovers to `[surrogate]`.

**Rollback.** Revert. The full-FEM path keeps working; basis files
remain on disk but are unused.

**Vs. existing tools.** This is the architecture behind DBS GUIDE
(Boston Scientific GUIDE XT, Medtronic SureTune) — pre-solved
patient-specific field bases combined live for arbitrary contact
fractionation. VNS-world has nothing equivalent. Putting it in
golgi would be best-in-class for peripheral stim.

**Regulatory / clinical-IT angle.** No new regulatory surface
beyond the underlying FEM — surrogate predictions inherit the same
SaMD bucket as the full solve. The IEC 62304 design history file
should document the linearity assumption explicitly; any non-linear
extension (encapsulation tissue evolution over weeks, etc.) is
out-of-scope of the surrogate and must fall back to full FEM.

---

## F3.3 — Validation overlay (measured impedance + recorded thresholds)

**Goal.** Overlay user-supplied measured data on simulated curves.
Bonus: compute simulated electrode impedance via a dual solve.

**Depends on.** Nothing structural; benefits from F3.1 (per-design
impedance).

**Touches.**
- `golgi/compute/solve_nerve.py` — add an optional second solve
  (Dirichlet 1 V on active contact, 0 on ground) → integrate σ∇V·n over
  contact → impedance.
- New: `golgi/validation/`:
  - `validation/datasets.py` — pydantic-ish models for measured datasets
    (impedance, recruitment, threshold-vs-diameter).
  - `validation/loaders.py` — CSV import with a small column-mapping UI.
  - `validation/fit.py` — R², RMSE (dB for impedance, linear for
    recruitment), Bland-Altman points.
- `golgi/figures/cole_cole.py`, `recruitment.py` — accept an optional
  `measured` argument and overlay points + fit metrics in the legend.
- New: `golgi/ui/drawers/validation.py`.

**Do.**
1. In `solve_nerve.py`, gate the impedance solve behind a flag in
   `MeshConfig` (`emit_impedance: bool = True`). Write
   `<out>/fem/<design_id>/impedance.json`:
   ```json
   {"per_contact": [{"id": 0, "Z_ohm": 1234.5}, ...],
    "per_pair":    [{"id_a": 0, "id_b": 1, "Z_ohm": 1100.2}, ...]}
   ```
2. `validation/datasets.py`:
   ```python
   @dataclass
   class MeasuredImpedance:
       frequencies_hz: np.ndarray
       magnitude_ohm: np.ndarray
       phase_deg: Optional[np.ndarray]
       metadata: dict   # species, prep, electrode, citation
   @dataclass
   class MeasuredRecruitment:
       amplitudes_mA: np.ndarray
       fraction_activated: np.ndarray
       branch: Optional[str]
       metadata: dict
   ```
3. New "Validation" drawer: paste CSV (with column mapping wizard) or
   pick file → preview table → "Attach to project". Attached datasets
   live in `<project>/validation/*.json`.
4. Update `figures/cole_cole.py` and `figures/recruitment.py` so they
   accept `measured=...` and overlay points + R² / RMSE in the legend.
5. Register a new figure `validation.bland_altman` showing
   sim-vs-measured residuals.

**Verify.**
- Migration smoke 7 + 8.
- Smoke addition F3.3.
- Import a synthetic CSV (perfect match) → R² = 1.0 in the legend.
- Impedance JSON exists per design; flipping `emit_impedance: False`
  skips the dual solve (verify timing).

**Rollback.** Revert. Validation files in projects remain orphaned;
re-enabling the feature picks them back up.

---

# Phase F4 — Infrastructure expansion

## F4.1 — Headless Python / Jupyter `Study` API

**Goal.** Drive the pipeline without the GUI. Unblocks notebook-based
papers, CI regression tests, and F5.1's UQ orchestration.

**Depends on.** F3.x ideally landed first (so the API can expose
multi-design + selectivity from the start), but **not strictly
required**.

**Touches.**
- New: `golgi/api.py` — the public `Study` class.
- `golgi/scene/` — add a `NullScene` that satisfies the `Scene` API
  with no-ops so the pipeline drivers work without a renderer.
- `golgi/auth/` — add a `bypass_auth=True` headless mode that creates a
  synthetic system user.
- `golgi/app.py` — refactor `_ensure_initialized()` so the headless path
  can call it with `skip_audit_writer=False, skip_assets=True`.
- New: `examples/recruitment_sweep.py`, `examples/selectivity_compare.py`.

**Do.**
1. `Study` surface:
   ```python
   class Study:
       @classmethod
       def open(cls, project_dir: Path, *, user: str = "headless") -> "Study": ...
       @classmethod
       def create(cls, project_dir: Path, *, user: str = "headless") -> "Study": ...
       def import_nerve(self, stl: Path) -> None: ...
       def set_mesh(self, **kwargs) -> None: ...
       def run_mesh(self) -> MeshResult: ...
       def set_electrodes(self, designs: list[ElectrodeConfig]) -> None: ...
       def run_fem(self) -> dict[str, FEMResult]: ...
       def set_fiber_seed(self, **kwargs) -> None: ...
       def run_fibers(self) -> FibersResult: ...
       def run_fiber_sim(self, fiber_idx: int, **pulse) -> FiberSimResult: ...
       def run_population(self, preset: str, n_replicates: int = 1) -> PopResult: ...
       def run_sweep(self, req: SweepRequest) -> SweepResult: ...
       def export_bundle(self, out: Path) -> None: ...    # delegates to F2.2
       def export_figures(self, out: Path, preset: FigureExportPreset) -> None: ...  # F2.3
   ```
2. Internally, `Study` constructs a `PipelineContext` with a `NullScene`,
   an `InProcessRunner` (or whichever `JobRunner` env says), and an
   `ActiveProject` bound to the given dir.
3. `NullScene` implements every `Scene` method as `pass` except mutators
   that drive disk writes (preserve those).
4. `examples/recruitment_sweep.py`:
   ```python
   import golgi
   s = golgi.Study.open("/tmp/demo_project")
   s.import_nerve("data/sample_nerve.stl")
   s.set_mesh(R_cuff_inner=1.5e-3, L_cuff=10e-3)
   s.run_mesh()
   s.set_electrodes([cuff_bipolar(z=5e-3)])
   s.run_fem()
   s.run_fibers()
   res = s.run_sweep(SweepRequest(axes=[SweepAxis("I_stim_mA", "linspace",
                                                  [0.05, 2.0, 40])]))
   s.export_figures("/tmp/demo_figs", preset=PAPER_300)
   ```
5. Document the API surface in `examples/README.md`.

**Verify.**
- Migration smoke (UI path unaffected).
- Smoke addition F4.1.
- The two example scripts exit 0 and produce expected artefacts.
- Running the example in a Jupyter cell prints a recruitment curve.

**Rollback.** Revert. `Study` is additive; deleting `api.py` doesn't
break the GUI.

---

## F4.2 — SLURM `JobRunner` + FEM checkpoint-resume

**Goal.** Implement the deferred Step 6.3 of `migration.md` and let FEM
solves resume from partial state. Two pieces, one feature.

**Depends on.** F4.1 (so headless cluster workers can construct a
`Study` without launching Trame).

**Touches.**
- New: `golgi/jobs/slurm_runner.py` implementing `JobRunner`.
- `golgi/jobs/protocol.py` — no API change; add docstring covering the
  remote contract.
- `golgi/compute/solve_nerve.py` — write `axis_line` / `slice_volume`
  incrementally per z-band; on startup, scan for existing band files
  and resume from the next missing one.
- `golgi/pipeline/fem.py` — compute a `(mesh_sha, sigma_sha,
  electrode_sha)` triple and store as `<out>/fem/<design_id>/.lockhash`;
  treat mismatching hash as "stale, redo".
- `golgi/app.py` — read `GOLGI_FEM_RUNNER` env var (default `local`),
  resolve to runner instance, inject into `PipelineContext`.
- `golgi/cli.py` — add `golgi compute-worker <payload.json>` so the
  remote side has a clean entry point.

**Do.**
1. `slurm_runner.py`:
   ```python
   class SlurmJobRunner(JobRunner):
       def __init__(self, *, partition: str, account: str | None,
                    cpus: int, memory_gb: int, time_limit: str,
                    scratch_root: Path, remote_root: Path | None,
                    sync: str = "rsync"):
           ...
       def run(self, req, on_line, cancel) -> JobOutputs:
           # 1. Serialize req.payload to a project-scoped sbatch dir.
           # 2. Render an sbatch script wrapping
           #    `python -m golgi.cli compute-worker <payload.json>`.
           # 3. `sbatch --parsable` → job id.
           # 4. arm cancel: on cancel.was_requested(), `scancel <job_id>`.
           # 5. Poll `squeue -j <job_id>`; tail the slurm-<jobid>.out
           #    file → on_line.
           # 6. When done: rsync result files back to project_dir.
           # 7. Return JobOutputs(success, return_code, paths).
   ```
   Same shape will later be cloned for PBS / LSF.
2. **Checkpoint-resume in FEM.**
   - Today `solve_nerve.py` writes one big `axis_line.npz`. Change to
     write `axis_line_band_<i>.npz` per z-band (band = chunk of z-samples,
     say 20 per band → 20 files). At the end, concatenate to the legacy
     `axis_line.npz` for back-compat readers.
   - Same scheme for `slice_volume` (per-z slices already chunkable).
   - On startup, hash the inputs; if `<out>/.lockhash` matches and band
     files exist, skip those bands.
   - Cancel mid-solve produces a usable partial checkpoint.
3. `GOLGI_FEM_RUNNER=slurm` requires SLURM env vars. Falls back to
   `local` with a warning if `sbatch` not on PATH.
4. Ship a `tests/fake_sbatch` shim script (a Python file pretending to
   be `sbatch`) so CI can exercise the SLURM path without a real
   cluster. Document in `tests/README.md`.

**Verify.**
- Migration smoke 8 + 13.
- Smoke addition F4.2.
- Local mode: FEM solve cancelled at 50%; re-run picks up the missing
  bands and finishes in ~half the original time.
- SLURM mode (via fake_sbatch): solve completes, outputs land in
  project dir, cancel sends `scancel`.
- Hash invalidation: change one σ value; re-run; entire FEM recomputes.

**Rollback.** Revert. Existing `axis_line.npz` projects still read fine
(the legacy concat is identical bytes when no resume happens).

---

# Phase F5 — UQ capstone

## F5.1 — Sensitivity / uncertainty (Sobol + LHS over σ + fiber dist)

**Goal.** First-order Sobol indices + CI-banded recruitment curves
without leaving the app.

**Depends on.** F2.1 (sweep engine), F4.1 (headless `Study` for fan-out),
F4.2 (SLURM for scale, optional).

**Touches.**
- New: `golgi/pipeline/uq.py`, `golgi/jobs/schemas.py` (`UQSpec`,
  `UQResult`), `golgi/figures/uq.py`.
- `golgi/ui/drawers/analysis.py` — new "UQ" subtab.
- `requirements-frozen.txt` — add `SALib`.

**Do.**
1. Schemas:
   ```python
   @dataclass
   class UQVariable:
       name: str            # "sigma_endo" | "fiber_diameter_mean_um[B]" | ...
       prior: str           # "uniform" | "normal" | "lognormal"
       params: dict         # {"lo": ..., "hi": ...} or {"mean": ..., "std": ...}
   @dataclass
   class UQSpec:
       variables: list[UQVariable]
       sampler: str         # "sobol" | "lhs"
       n_samples: int       # Sobol: actual samples = n*(2k+2) by default
       metrics: list[str]   # ["threshold_uA", "selectivity_target_vs_off"]
       sweep_template: SweepRequest  # axes evaluated *per UQ sample*
       seed: int = 0
   @dataclass
   class UQResult:
       spec: UQSpec
       samples_path: Path        # parquet: row per sample, col per variable
       per_sample_sweeps_dir: Path
       sobol_indices: dict[str, dict[str, float]]  # metric → {var: S1, ST}
       summary_path: Path
   ```
2. Sampler uses SALib (`saltelli.sample`, `latin.sample`). Per sample:
   construct a `Study`, push the perturbed σ / fiber params, run the
   sweep, persist results in `<out>/uq/<sample_idx>/`.
3. Fan-out via the same runner the rest of the pipeline uses (local
   thread pool, or SLURM array job if `GOLGI_FEM_RUNNER=slurm`).
4. Compute Sobol indices via `salib.analyze.sobol.analyze` per metric.
5. UQ subtab visualises:
   - Tornado bar chart of S1 + ST per variable per metric.
   - CI-banded recruitment curve (5/50/95 percentile across samples).
   - Per-variable scatter (metric vs perturbed value).
6. Cache per-sample sweep results (re-uses F2.1's sweep cache by SHA);
   re-running UQ with a superset of samples re-uses prior work.

**Verify.**
- Smoke addition F5.1.
- Sobol-8 (smallest sample) over `sigma_endo` produces non-zero S1
  matching the analytical expectation (σ_endo is the dominant linear
  driver in this problem).
- Cancel partway through: partial samples remain on disk; re-run picks
  them up.

**Rollback.** Revert. No upstream changes; UQ artefacts are
self-contained in `<project>/uq/`.

---

## A1 — Topographic-atlas-driven trajectory generation (Hammer / Settell)

**Goal.** Make fiber trajectories anatomically realistic instead of
purely Laplace-streamline-generated. Add Hammer-style (sheep cervical
VN, Hammer et al. 2018) and Settell-style (pig cervical VN,
Settell et al. 2020) topographic atlases as selectable trajectory
modes in the Fiber-trajectories drawer. Atlas mode samples seeding
locations from per-fascicle prior distributions (cardiac /
pulmonary / recurrent / vagal-subdivision groupings) instead of from
uniform cap-distribution.

**Depends on.** Nothing new in the pipeline. Sits alongside the
existing Laplace+RK4 generator as a "mode" in
`pipeline/fibers.py`.

**Touches.**
- New: `golgi/atlases/` package with one module per atlas:
  - `atlases/__init__.py` — registry + public lookup helpers.
  - `atlases/hammer2018_sheep_vn.py` — per-fascicle xy distributions
    + functional grouping (cardiac / sensory / motor splits).
    Cites Hammer et al. 2018 ("Length of human vagus nerve
    branches…") or the relevant sheep VN histology.
  - `atlases/settell2020_pig_vn.py` — pig cervical VN fascicular
    organisation, Settell et al. 2020 JNE.
  - Future: rat, human (Pelot 2020 etc).
  Each module exports:
  ```python
  @dataclass(frozen=True)
  class Atlas:
      name: str
      species: str
      nerve: str
      citation: str
      notes: str
      # Per-functional-group seeding: name → 2D distribution params
      # in the nerve cross-section frame (mm). Distributions can be
      # Gaussian, gaussian-mixture, or empirical KDE from a points
      # file shipped alongside the module.
      groups: dict[str, AtlasGroup]
  ```
- `golgi/state_defaults/atlases.py` (new) — dropdown items,
  per-atlas seed-fraction state.
- `golgi/pipeline/fibers.py` — add `mode: "uniform" | "atlas"` to
  `FiberSeedConfig`. When `atlas`, the seed sampler reads the named
  atlas and draws seeds from the per-group distributions instead of
  uniformly across the cap.
- `golgi/jobs/schemas.py` — extend `FiberSeedConfig`:
  ```python
  mode: str = "uniform"                # "uniform" | "atlas"
  atlas_name: Optional[str] = None     # registry key
  group_fractions: dict[str, float] = field(default_factory=dict)
  ```
- `golgi/ui/drawers/fibers.py` — add an "Atlas" subsection above
  the existing "Streamlines" controls:
  - Mode selector: Uniform / Atlas.
  - Atlas dropdown (only when mode=atlas) with citation/notes
    sub-text identical pattern to F1.1's pop preset block.
  - Per-functional-group fraction inputs (sum-check chip).
- `golgi/figures/registry.py` — register
  `atlas.cross_section_preview` showing the seeding distribution
  before generation.

**Do.**
1. **Atlas data format.** Each atlas ships its distribution
   parameters as Python literals (small) or a `.npz` of empirical
   points (shipped alongside the module). Coordinate convention:
   nerve cross-section frame, axes aligned with the intrinsic PCA
   used elsewhere (so the atlas + the user's imported nerve align
   without manual registration).
2. **Sampler.** `atlas_sampler(atlas, group_fractions, n_seeds,
   rng) -> np.ndarray (n_seeds, 2)`. Per-group share = fraction ×
   n_seeds; draws within each group from the group's distribution.
3. **Pipeline wiring.** `pipeline/fibers.py` reads `mode` from
   `FiberSeedConfig`; in atlas mode, calls `atlas_sampler` to pick
   xy seed points in the cap plane, then continues with the existing
   RK4 streamline integration.
4. **Provenance.** Every fiber generated in atlas mode persists its
   group label in `nerve_paths_fibers.npz` (new array
   `fiber_group_labels`). Downstream consumers (F3.2 selectivity)
   can target-vs-off-target by functional group instead of by
   geometric branch.
5. **Citation surfacing.** When atlas mode is active, the FEM /
   selectivity figure captions in F2.3 reports include the atlas
   citation in their footers.

**Effort.** M–L (2-3 weeks). Most of the cost is literature work
(extracting reliable per-fascicle distributions from histology
papers and validating against the cited measurements). Code is
straightforward once data is curated.

**Impact.** High — moves golgi from "anatomically plausible" to
"anatomically grounded" for the most-studied VNS preparations.
Reviewers stop asking "where do these fibers go?" because the
answer is "from the cited atlas". Sits naturally with F1.1
(populations) and F5.1 (UQ over atlas group fractions).

**Verify.**
- Migration smoke item 9 (Fiber trajectories) with the new mode
  toggle.
- Smoke addition A1: pick "Hammer 2018 sheep VN" → generate
  trajectories → cross-section preview matches the atlas's
  expected fascicular layout. `fiber_group_labels` array exists in
  `nerve_paths_fibers.npz`.
- F2.3 Generate Report on a project with atlas mode active includes
  the atlas citation in the FEM section caption and the bibliography.

**Rollback.** Revert. Existing uniform-mode projects unaffected
(`mode` defaults to `"uniform"`).

**Vs. existing tools.** ASCENT ships `Sample` configs for cervical
VN that encode rough fascicular organisation but expects the user
to assemble their own histology JSON. Sim4Life is geometry-agnostic
on this dimension. A curated, citable atlas library in golgi is a
genuine first.

---

## Appendix A — Step ledger

Tick as you go. One commit per row.

- [x] W1.1 — Cole-Cole + IT'IS DB → `golgi/conductivity/`
- [x] W1.2 — Mesh-quality math → `golgi/pipeline/mesh_quality.py`
- [x] W1.3 — PLC assembly → `golgi/pipeline/plc.py`
- [x] W1.4 — Electrode-patch builders → `golgi/scene/electrode_patches.py`
- [x] W1.5 — Cuff-fit PCA helpers → `golgi/scene/cuff_fit.py`
- [x] W1.6 — Axonml / pyfibers dispatchers → `golgi/pipeline/fiber_backends.py`
- [x] W1.7 — Finish watcher extraction → `golgi/watchers/`
- [⚠️] W1.8 — `do_*` handlers → `golgi/actions/` (partial: 23/68;
       remaining 46 deferred to W1.9 below)
- [ ] W1.9 — Build_app closure refactor + remaining handler extraction
       (deferred until app.py size blocks new feature work)
- [x] F1.1 — Vagal fiber population presets
- [x] F1.2 — Publication-grade figure styling presets
- [x] F2.1 — Parameter sweep + threshold + recruitment curves
       (`pipeline/sweep.py`, `actions/sweep.py:do_find_thresholds`,
       Sweep subtab gated on `active_analysis === 'sweep'`,
       `figures/recruitment.py`, `projects/sweep_cache.py`)
- [x] F2.2 — Reproducible study bundle (export + import + replay)
       (`projects/bundle.py:export_study/import_study`,
       `projects/replay.py:replay_study`, `cli.py` w/ both
       commands, navbar Export/Import-study + dialogs)
- [x] F2.3a — Per-panel export buttons (everywhere a panel renders)
       (`ui/components/figure_export_btn.py` reused across drawers)
- [x] F2.3b — Bulk Figure Export view (new "Exports" tab)
       (`ui/drawers/exports.py`, `figures/registry.py` 765 LOC,
       `figures/export.py` 518 LOC)
- [x] F2.3c — Generate Report dialog + multi-page PDF writer
       (`ui/dialogs/generate_report.py`, `figures/report.py`)
- [x] F3.1 — Multi-electrode FEM solve (`state.fem_configs`,
       per-design `<out>/designs/<eid>/configs/<cid>/` outputs,
       FEM-driver fans out per config)
- [x] M1 — Multi-contact anode/cathode picker
       (`contact_polarities`, `contact_current_fractions`,
       bipolar / tripolar / N-polar polarity rows + sum-check
       chips in `ui/drawers/cuff_electrodes.py`)
- [x] F3.2 — Selectivity metrics + comparison view
       (`pipeline/selectivity.py` w/ Veraart SI + threshold-
       ratio + per-branch recruitment math; `figures/selectivity.py`
       for the SI bar chart + threshold-ratio HTML table;
       Compare-panel "Selectivity" tile with target / off-target
       branch pickers + amplitude knob; per-cid sweep cache via
       `sweep_cache.load_latest_for_config` so each config's
       sweep is loaded independently; `selectivity.bar`
       registered in figures/registry.py)
- [ ] S1 — FEM surrogate (contact basis + Live Stimulation panel)
- [ ] F3.3 — Validation overlay
- [x] F4.1 — Headless `Study` API
       Phases A–D landed (a7cbebc, 81f1f7f, 08d6683, +
       Phase D). `golgi/api.py` ships the public `Study` class
       (sync API; asyncio.run internally);
       `golgi/scene/null_scene.py` (no-op renderer);
       `golgi/__init__.py` lazy-exposes `golgi.Study`;
       `examples/recruitment_sweep.py` +
       `examples/selectivity_compare.py` + `examples/README.md`.
       Phase B extracted `refit_design_geometry` from
       `build_app()` to `golgi/scene/cuff_fit.py` (the only
       true closure mesh-driver needed). Phase C extended
       `_ensure_ctx()` to 36 helpers covering mesh + fibers +
       fiber-sim + pop-sim + FEM drivers; inlined small
       closures (`_ensure_polarities`, `_cuff_ns_extras`,
       `_save_fiber_sim_cache`, `_save_pop_state`,
       `_fiber_paths_display`, `_contact_count`,
       `_default_polarities`) directly in `_ensure_ctx` rather
       than physically extracting. UI-only refresh closures
       (`_refresh_*`, `_fiber_label_and_color`) replaced with
       no-ops in headless. Phase D was tiny — all FEM helpers
       were already provided in the Phase C forward-compat
       block, so wiring `Study.run_fem()` was a 10-line method.
       Working methods: open, create, close, list_designs,
       list_configs, set_mesh, set_electrodes, set_fiber_seed,
       import_nerve, run_mesh, run_fibers, run_fem, run_sweep,
       export_bundle, import_bundle (classmethod).
       Real-geometry smoke (run examples/*.py against the
       bundled sample STL) intentionally not in CI yet — TetGen
       + FEniCSx + axonml install footprint is heavy. The
       no-runtime-side smoke (PipelineContext build + 36
       helpers reachable + all five compute methods callable
       without NotImplementedError) is wired into the example
       scripts themselves.
- [⚠️] F4.2 — SLURM `JobRunner` + FEM checkpoint-resume
       Phase A (landed): `golgi/jobs/slurm_runner.py` ships the
       `SlurmJobRunner` class (submit via sbatch + poll via
       squeue + tail slurm-%j.out → on_line + cancel via
       scancel + optional rsync-back). `golgi/jobs/__init__.py`
       exposes `resolve_fem_runner(local_runner=…,
       on_warning=…)` that reads `GOLGI_FEM_RUNNER` env
       (default `local`; `slurm` picks SlurmJobRunner built
       from `GOLGI_SLURM_*` env vars; unknown / missing falls
       back to local with a warning). `golgi/cli.py` gets the
       `compute-worker <payload.json>` subcommand the sbatch
       wrapper invokes. `tests/fake_sbatch.py` ships a Python
       shim that pretends to be sbatch / squeue / scancel so
       the runner can be exercised end-to-end without a real
       cluster install (see `tests/README.md`).
       Phase B (pending): FEM checkpoint-resume — per-band
       writes in `golgi/compute/solve_nerve.py`, `.lockhash`
       (mesh_sha + sigma_sha + electrode_sha) in
       `golgi/pipeline/fem.py`, startup scan to skip already-
       done bands, plus the wiring that swaps the inline
       `FEMRunner(_SOLVE_NERVE_PATH)` in
       `pipeline/fem.py:522` for `resolve_fem_runner(…)`.
- [ ] F5.1 — Sensitivity / UQ
- [ ] A1 — Hammer/Settell topographic atlases
- [x] X1 — Scar / connective tissue shell (per-design geometry,
       PLC cylinder + caps, mesh-pipeline wiring, design-drawer
       slider with auto-init, Mesh-drawer `lc_scar_um`,
       Conductivities-drawer `sigma_scar` + the missing
       `sigma_contact` row, scar axis added to the existing
       Design-sweep dialog). Unplanned addition during the
       F3.2-M3 milestone, requested by user; not in the
       original roadmap. The S1 description (FEM surrogate)
       still treats it as out-of-scope of the surrogate path —
       full FEM solves use the scar shell, the surrogate falls
       back to a homogeneous saline pocket. Logged as `X1` to
       flag "not on the original roadmap but shipped".
- [⚠️] I1 — Electrode-tissue impedance from FEM
       Phase A (landed): `golgi/compute/solve_nerve.py` gains
       `_emit_impedance_dc()` — for each contact, builds a
       Dirichlet BC (V=1 on that contact's facets, V=0 on the
       outer muscle bbox facets, located via
       `_outer_muscle_facets()`), re-solves Laplace, integrates
       σ·∇V·n over the contact's facets (both ds + dS for
       interior saline-silicone seam contacts) → Z_i = 1/I.
       Per-pair Z derived from configured anode/cathode pairs
       via a separate dirichlet solve (±0.5 V on the pair).
       Gated by `GOLGI_EMIT_IMPEDANCE` env var, threaded from
       `FEMJobRequest.emit_impedance` → `FEMRunner._build_env`.
       Output: `<config>/impedance.json` with per_contact +
       per_pair arrays. Pipeline loads → `state.fem_impedance`
       keyed by cid. Two new Plotly bar charts in
       `figures/impedance.py` (per-contact log-Z + per-pair
       log-Z); registered as `fem.impedance_bar` +
       `fem.impedance_per_pair` so Bulk Exports + Generate
       Report pick them up. New "Impedance" tile in the
       Compare panel (sibling to Selectivity); toggle "Compute
       electrode impedance on each FEM solve" in the
       Conductivities drawer (default ON, persisted).
       F3.3 ground spec: "outer_muscle" boundary; user-
       configurable strategy deferred to Phase B.
       Phase A.2 (LANDED — root cause + relabel): first real-
       runtime run produced ~100-1000× too-high magnitudes
       (e.g. Z0=390 kΩ, Z_pair=283 kΩ on a bipolar saline
       cuff). Root cause: contact facets sit on the saline ↔
       silicone interior interface; the old code used
       `σ("+")·∇V("+")·n("+") · dS` — taking only ONE side of
       the interior facet, and dolfinx's "+"/"-" labelling is
       a topological choice not a physical one, so per-facet
       it could randomly land on silicone (σ ≈ 1e-15 S/m,
       flux ≈ 0) instead of saline. Fix: for per-contact,
       integrate at the OUTER MUSCLE GROUND (exterior facets,
       no "+/-" ambiguity, single ds integral — Option A in
       the planning). For per-pair (no Dirichlet ground), sum
       BOTH sides of the interior facet jump (Option B); the
       silicone-side contribution drops out naturally because
       σ_si ≈ 0. Post-fix magnitudes match Newman 1/(4σa) ≈
       166 Ω for ~mm² contact in saline. Relabeled UI from
       generic "impedance" → "access impedance (Z_access)"
       everywhere (drawer chips, plot titles, hover text,
       Compare-tile header) with a hover tooltip explaining
       this is tissue spreading only — no electrode-
       electrolyte interface (Helmholtz double-layer + R_ct +
       Warburg) which dominates real DC measurements and is
       I2's scope.
       Phase B (pending — after A.2 lands): Cole-Cole frequency
       sweep — iterate freq list, call
       `golgi.conductivity.cole_cole_sigma` per tissue per
       frequency, re-solve, output Z(f); new Bode plot tile
       per contact; frequency-list picker in the Materials
       drawer. Then F3.3 Phase 2 consumes I1's output for
       measured-vs-simulated overlay.
- [ ] R1 — Recording cuff role + CNAP forward model (reciprocity)
       Scope locked (single cuff, mixed stim+record roles;
       full I_m persistence so montages can be added post-hoc;
       hard-coded bipolar montage in v1; auto-compute transfer
       cache when ≥1 contact is in a recording role; CNAP plot
       as a new tile in the Solve grid). Phase A: contact role
       enum extension (rec_plus / rec_minus / rec_ref / ground)
       + cuff_electrodes drawer + 3D viewport tinting +
       design persistence. Phase B: reciprocity transfer cache
       — piggyback on I1 Phase A's dual-solves, sample g(x) at
       every fiber node, persist as `recording_transfer.npz`.
       Phase C: persist full NEURON I_m(node, t) +
       `compute/cnap.py` doing V_rec = einsum("fnc,fnt->ct",
       g_fiber, I_m) with per-fiber-type breakdown. Phase D:
       bipolar montage + CNAP plot tile (Solve grid restructure
       2×2 → 2×3) + peak-latency / CV readout + figures/cnap.py
       registered. Phase E (stretch): paired stim+record cuffs,
       CV histograms, selective-recording figures-of-merit.
       I1 Phase A.2 unblocked the reciprocity prerequisite —
       basis solves now produce physical Z_access values, so
       g(x, contact) is trustworthy. Refs: Plonsey 1969
       reciprocity; Struijk 1997 cuff CNAP simulation;
       Andreasen & Struijk tripolar; NRV (PLOS Comp Bio 2024).
- [ ] I2 — Electrode-electrolyte interface (Randles cell)
       Scope: Randles equivalent circuit (R_s + (R_ct ∥ C_dl) +
       Warburg) parameterised per contact material, layered
       post-FEM on Z_access from I1. Materials drawer picks the
       interface preset (Au / Pt / Pt-Ir / IrOx / TiN) which
       provides typical R_ct, C_dl, A_Warburg literature
       values. Output: total Z(ω) = Z_access + Z_interface(ω)
       reported per contact + per pair; consumed by F3.3 Phase
       2 for measured-vs-simulated overlay. Small lift after
       I1 Phase A.2; no new FEM solves. Refs: McAdams &
       Jossinet 1995 (tissue impedance decomposition); Cogan
       2008 (Annu Rev Biomed Eng — interface impedance values
       per material); Randles 1947 (original circuit).
- [ ] V1 — µCT slice segmentation + extrusion (single slice)
       Scope locked (MedSAM2 pretrained model; 3-class labels
       background / epi / fascicle — single shared fascicle
       class; re-editable persistence per design; entry-point
       via a new tile in the Import wizard alongside
       Analytical / A1-Atlas; prismatic extrusion of one
       representative slice — no multi-slice lofting in v1).
       Phase A: `golgi/segmentation/` module — tifffile-based
       multi-page TIFF loader (handles 8/16/32-bit), MedSAM2
       wrapper (optional dep, graceful fallback), slice picker
       + ROI crop + contrast windowing, "everything mode"
       candidate mask proposals, click-to-segment refinement,
       label assignment UI. Phase B: marching-squares polygon
       extraction + Ramer-Douglas-Peucker simplification,
       pixel → mm via user-supplied voxel size, PLC layer
       build (epi outer + fascicle prisms) handed to existing
       `pipeline/plc.py` + TetGen. Phase C: classical Otsu +
       morphology + connected-components fallback for the
       MedSAM2-less path (headless/batch). Phase D: Import-
       wizard integration. Why not TotalSegmentator: trained
       on clinical whole-body CT (1228 scans, 117 classes —
       organs/bones/vessels at mm scale); zero fascicle
       awareness. Why MedSAM2: fine-tuned SAM2 on 1.5M+
       medical image-mask pairs spanning CT / MRI /
       microscopy / µCT, generic prompt-based, no per-domain
       training. Refs: Wasserthal 2023 TotalSegmentator
       (Radiology AI); Ma 2024 MedSAM2 (arXiv 2408.03322);
       Pelot 2020 cervical/subdiaphragmatic vagus atlas
       (Front Neurosci); Pena-Ramirez 2024 fascicle seg
       (J Neural Eng).

---

## Appendix B — Cross-cutting design rules

Carry these through every step so the features compose:

- **Every artefact a feature produces lives under `<project>/`** — never
  in `~/.cache`, never in `/tmp`. Bundle export (F2.2) and figure export
  (F2.3) both rely on this.
- **Every new compute job uses a typed `dataclass` request schema in
  `golgi/jobs/schemas.py`** with `serialize()` / `deserialize()`. No
  free-form dicts crossing the boundary. (This is the
  `migration.md` Step 6.2 contract; preserve it.)
- **Every new artefact carries a content sha256** in its sibling
  manifest. The replay check in F2.2 requires this.
- **Every new pipeline driver accepts a `CancelToken`** and checks it
  between sub-units (per fiber, per UQ sample, per FEM z-band). The
  smoke item 13 (cancel) MUST keep passing.
- **Every new figure is registered in `figures/registry.py` (F2.3)** the
  same PR that introduces it. Otherwise bulk export silently misses it.
- **No feature requires schema migration of existing projects.** New
  fields are optional; old projects open cleanly. This keeps the
  rollback story honest.
- **MPI / multiprocessing: rank-0-only writes.** New compute code follows
  the existing `solve_nerve.py:972-1017` pattern. Anything else
  produces silent file corruption on multi-rank runs.

---

## Appendix C — What NOT to build in this plan

Explicit non-goals so they don't sneak in:

- **Closed-loop / real-time stimulation control.** Out of scope; golgi
  is an offline design tool.
- **Custom fiber model authoring (NEURON .mod files).** Reuse
  `pyfibers` / `axonml` model registries.
- **ML surrogate for FEM.** Possible later; needs F4.1 + F4.2 + F5.1 to
  generate training data first. Not in this plan.
- **Multi-tenant cloud SaaS.** Auth + projects remain per-user-local.
- **In-app mesh repair / Boolean editing.** Continue to assume the
  imported nerve surface is watertight; refer users to MeshLab /
  Blender.
