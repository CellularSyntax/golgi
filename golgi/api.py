# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""F4.1 — Headless `Study` API.

Drive the golgi pipeline from a plain Python script or a Jupyter
cell, without the Trame GUI. Unblocks:

* Notebook-based papers (reproducible figures from `Study(...)`).
* CI regression tests against the pipeline.
* The F4.2 SLURM `JobRunner` (which needs a headless surface to
  hand work to a cluster).
* The F5.1 UQ orchestration (which fans out N parameter samples
  in parallel processes, each driven by a `Study`).

## Usage

```python
import golgi

s = golgi.Study.create("/tmp/demo_project")
s.import_nerve("data/sample_nerve.stl")
s.set_mesh(use_epi=True, epi_thickness_um=50, decim_target_k=60)
s.run_mesh()
s.set_electrodes([{"name": "bipolar @ 5 mm", "cuff_offset_mm": 5.0}])
s.run_fem()
s.run_fibers()
res = s.run_sweep(...)
s.export_bundle("/tmp/demo.zip")
s.close()
```

## Status — fully wired (F4.1 Phase D)

All compute methods are implemented and dispatch to the same
pipeline drivers the GUI uses: `import_nerve`, `run_mesh`,
`run_fem`, `run_fibers`, `run_sweep`. The headless
`PipelineContext` is assembled inline in `_ensure_ctx()` (the
closure-extraction that earlier phases deferred is done there),
so no part of `build_app()` is required at run time.

End-to-end usage lives in `examples/recruitment_sweep.py`
(load → mesh → electrodes → FEM → fibers → threshold sweep →
bundle export) and is covered by
`tests/test_headless_api.py::test_end_to_end_pipeline`.

## Synchronous API

Every `Study` method is synchronous. Internally the async
pipeline drivers are wrapped via `asyncio.run`, so you can call
them from plain `def` functions, scripts, and Jupyter cells
without `await`. If you need to run two studies concurrently
in-process you can spawn multiple processes; intra-process
concurrency isn't a v1 use case.

## Auth + audit

Headless mode synthesises a `headless` user (no password, local
trust) so the auth-gated decorators in the pipeline don't have
to be selectively disabled. The audit writer still runs — every
study action lands in the same `audit_fallback.jsonl` the GUI
writes to, so headless runs show up in `golgi events` (CLI) /
the Activity tab (GUI).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


__all__ = ["Study"]


# ---------------------------------------------------------------------------
# Headless state shim — quacks like the trame state proxy for the subset of
# operations the pipeline drivers actually perform.
# ---------------------------------------------------------------------------


class _HeadlessState:
    """Tiny adapter that satisfies the surface trame.state
    exposes to the pipeline drivers:

    * attribute access (`state.foo` / `state.foo = ...`)
    * item access (`state["foo"]` / `state["foo"] = ...`)
    * `flush()` (no-op — there's no client to push to)
    * `change(*keys)` as a no-op decorator (the GUI uses this to
      wire reactive watchers; headless just ignores them — the
      pipeline drivers don't rely on reactive callbacks firing
      during their own execution)

    Backed by a plain dict. Any value the drivers write can be
    inspected post-run via `Study._state_dump()` for debugging.
    """

    def __init__(self, initial: dict[str, Any] | None = None):
        # __setattr__ recurses unless we go through the parent
        # class's __setattr__ for this one bootstrap.
        object.__setattr__(self, "_d", dict(initial or {}))

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup misses, so
        # the `_d` instance attr resolves cleanly without
        # bouncing back here.
        try:
            return self._d[name]
        except KeyError as ex:
            raise AttributeError(name) from ex

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_d":
            object.__setattr__(self, name, value)
        else:
            self._d[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._d[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._d[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._d

    def get(self, name: str, default: Any = None) -> Any:
        return self._d.get(name, default)

    def update(self, mapping: dict[str, Any]) -> None:
        self._d.update(mapping)

    def flush(self) -> None:
        """No-op in headless. The GUI uses this to push pending
        state mutations to the client; headless has no client."""
        return None

    def change(self, *_keys: str):
        """No-op decorator. The GUI wraps watcher callbacks with
        `@state.change("foo")`; headless ignores the wrap so the
        callback definition succeeds without firing anything."""
        def _decorator(fn):
            return fn
        return _decorator

    # Useful for debugging / introspection from Jupyter cells.
    def _asdict(self) -> dict[str, Any]:
        return dict(self._d)


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


class Study:
    """Headless wrapper around the golgi pipeline.

    Construct via `Study.open(project_dir)` to attach to an
    existing project, or `Study.create(project_dir)` to make a
    new one. Use `Study.close()` when done (also fires
    automatically via the context-manager protocol).
    """

    # ----- Construction -------------------------------------------------

    def __init__(
        self,
        project_dir: Path,
        *,
        user: str = "headless",
    ):
        from golgi.app import (  # noqa: E402
            set_active, _ensure_initialized,
            MAX_FIBER_BRANCHES,
        )
        from golgi.scene.null_scene import NullScene  # noqa: E402

        # Idempotent module-side init (auth DB, audit writer,
        # static assets). Safe to call multiple times.
        _ensure_initialized()

        self._project_dir: Path = Path(project_dir).expanduser().resolve()
        self._user: str = str(user)
        self._closed: bool = False

        # Activate the project so all module-level path proxies
        # (`GOLGI_OUT`, `UPLOAD_DIR`, ...) resolve here.
        set_active(self._project_dir)

        # Build the headless context. The state shim, NullScene,
        # and GeometryState all stand in for what build_app
        # would normally construct.
        from golgi.scene.geometry import GeometryState  # noqa: E402
        self._state = _HeadlessState()
        # Seed the same factory defaults build_app() installs, so the
        # pipeline drivers find every state.* key they read.
        self._seed_state_defaults()
        self._geom = GeometryState()
        self._scene = NullScene(
            region_tags=(1, 2, 3, 4, 5, 6, 7),
            max_fiber_branches=int(MAX_FIBER_BRANCHES),
        )

        # `_ctx` is the headless PipelineContext. Built lazily on
        # first compute call via `_ensure_ctx()` and cached here.
        self._ctx = None

    def _seed_state_defaults(self) -> None:
        """Install the same factory state defaults the GUI's
        `build_app()` does, so headless pipeline drivers find every
        `state.*` key they expect.

        Reuses `golgi.state_defaults.<topic>.register()` (the blessed
        source — each seeds its topic's keys from the same dicts the
        GUI uses) for the bulk, then fills the inline A1 multi-electrode
        + status/log keys that `build_app()` seeds outside those
        registries. Extras use a gap-fill (only set when absent) so a
        richer register()-provided value always wins.
        """
        from golgi import state_defaults as _sd
        from golgi.app import (  # noqa: E402
            DEFAULT_CUFF, DEFAULT_ELECTRODE, DEFAULT_MESH, DEFAULT_SIGMA,
            FIBER_MODEL_DIAMETER_CONFIG, _FIBER_MODEL_DIAMETER_DEFAULT,
            MAX_FIBER_BRANCHES, TAB10_PALETTE, list_data_files,
        )
        st = self._state
        _sd.ui_toggles.register(st)
        _sd.fem.register(st)
        _sd.fiber.register(
            st,
            fiber_diameter_config=FIBER_MODEL_DIAMETER_CONFIG,
            fiber_diameter_default=_FIBER_MODEL_DIAMETER_DEFAULT,
            tab10_palette=TAB10_PALETTE,
        )
        _sd.pop.register(st)
        _sd.sweep.register(st)
        _sd.exports.register(st)
        _sd.study_bundle.register(st)
        _sd.import_state.register(st, list_data_files=list_data_files)
        _sd.mesh.register(st)
        _sd.cuff.register(st, default_cuff=DEFAULT_CUFF)
        _sd.electrode.register(st, default_electrode=DEFAULT_ELECTRODE)

        # Mesh-geometry + conductivity factory defaults. build_app()
        # seeds these from DEFAULT_MESH / DEFAULT_SIGMA directly (not
        # via a register()), so do the same here.
        st.update(dict(DEFAULT_MESH))
        st.update(dict(DEFAULT_SIGMA))

        # Inline-seeded build_app() keys not covered by any register():
        # the A1 multi-electrode model + per-run status/log scratch that
        # some drivers read before writing.
        extras = {
            # A1 multi-electrode model
            "designs": [], "selected_design_id": "", "active_design_id": "",
            "next_design_seq": 1,
            "contact_polarities": [], "contact_count": 0,
            "contact_current_fractions": [],
            # per-design FEM configs
            "configs": [], "selected_config_id": "", "active_config_id": "",
            "next_config_seq": 1,
            "fem_configs": [], "solve_config_selection": [],
            "active_montage_single": "", "active_montage_pop": "",
            # status flags
            "has_designer_cuff": False, "has_geometry": False,
            "has_mesh": False, "has_fibers": False, "has_fem": False,
            "has_active_project": False, "designs_mesh_panels": [],
            # busy / log / stats scratch
            "busy": False, "busy_msg": "", "busy_log": "",
            "mesh_log": "", "fem_log": "", "fiber_log": "",
            "mesh_status": "", "fem_status": "", "fiber_status": "",
            "mesh_failed": False, "fem_failed": False, "fiber_failed": False,
            "mesh_stats_html": "", "fiber_stats_html": "",
            "mesh_quality_hist_figure": None,
            "fiber_cnap_figure": None, "fiber_cnap_status": "",
            "pop_cnap_figure": None, "pop_cnap_status": "",
            "fiber_branch_summary": [], "fiber_n_branches": 0,
            "pop_row_meta": [], "fem_impedance": None,
            # mesh decimation target (build_app inline default; not covered by
            # state_defaults.mesh.register()).
            "decim_target_k": 50,
            # fiber-generation / trajectory defaults. build_app() seeds these
            # inline (not via a state_defaults register()), so gap-fill them
            # here too — else headless run_fibers() hits an unset state field.
            "n_fibers": 100, "fiber_max_steps": 10000,
            "fiber_seed_end": "trunk (low z)",
            "fiber_cluster_eps_mm": 2.0, "fiber_cap_band_pct": 15.0,
            "fiber_min_rel_size_pct": 20.0, "fiber_axial_normal_thresh": 0.70,
        }
        # Per-branch display names the GUI seeds inline (build_app), used by the
        # fiber branch-classification step.
        extras.update({
            f"fiber_branch_name_{_i}": "" for _i in range(MAX_FIBER_BRANCHES)
        })
        for _k, _v in extras.items():
            if _k not in st:
                st[_k] = _v

    def _finalize_designs_and_configs(self) -> None:
        """Complete each design the way the GUI's `do_add_design` does:
        merge factory defaults, derive contact polarities/fractions,
        assign a selected/active design, and give every design a
        Default FEM config. Idempotent — fills only what's missing,
        so a fully-specified design passes through untouched.
        """
        from golgi.app import (  # noqa: E402
            DEFAULT_CUFF, DEFAULT_ELECTRODE, DUKE_ELECTRODE_TYPE,
            _CUFF_PRESETS,
        )
        st = self._state
        designs = list(st.get("designs") or [])
        if not designs:
            return

        # Pure copies of the GUI closures (golgi.app._contact_count /
        # _default_polarities) — kept here so set_electrodes doesn't
        # depend on _ensure_ctx() having run yet.
        def _contact_count(elec: dict) -> int:
            kind = str(elec.get("electrode_type", "bipolar ring-pair"))
            if kind == "bipolar ring-pair":
                return 2
            if kind == "tripolar (anode-cathode-anode)":
                return 3
            if kind == "ring-array (NxM)":
                try:
                    return max(0, int(elec.get("array_n_rows", 2))
                               * int(elec.get("array_n_cols", 4)))
                except (TypeError, ValueError):
                    return 8
            if kind == "helical (Livanova-style)":
                return 2
            if kind == "LIFE (longitudinal intrafascicular)":
                try:
                    rows = int(elec.get("life_n_rows", 1))
                    cols = int(elec.get("life_n_cols", 1))
                except (TypeError, ValueError):
                    rows, cols = 1, 1
                return max(0, rows * cols)
            if kind == DUKE_ELECTRODE_TYPE:
                preset = _CUFF_PRESETS.get(str(elec.get("duke_preset") or ""))
                if preset is None:
                    return 0
                return sum(
                    1 for inst in preset.get("instances", [])
                    if inst.get("type") in (
                        "LivaNova_Primitive", "CircleContact_Primitive")
                )
            return 0

        def _default_polarities(elec: dict) -> list:
            n = _contact_count(elec)
            if n <= 0:
                return []
            if n == 1:
                return ["anode"]
            if n == 2:
                return ["anode", "cathode"]
            if n == 3:
                return ["anode", "cathode", "anode"]
            return ["anode" if i % 2 == 0 else "cathode" for i in range(n)]

        # Scaffolding mirroring golgi.app._new_electrode_default.
        base: dict[str, Any] = {}
        base.update({k: DEFAULT_CUFF[k] for k in DEFAULT_CUFF})
        base.update({k: DEFAULT_ELECTRODE[k] for k in DEFAULT_ELECTRODE})
        base.update({
            "R_ci_m": None, "R_co_m": None,
            "duke_preset": "", "duke_overrides": {},
            "vis_master": True, "vis_endo": True, "vis_epi": True,
            "vis_muscle": True, "vis_silicone": True, "vis_saline": True,
            "vis_contacts": True, "vis_scar": True, "vis_mesh": True,
            "vis_mesh_quality": False, "has_mesh": False, "has_fem": False,
        })

        completed: list[dict] = []
        seq = int(st.get("next_design_seq") or 1)
        for d in designs:
            full = dict(base)
            full.update(d)                       # user-supplied fields win
            if not str(full.get("eid") or ""):
                full["eid"] = f"elec_{seq:02d}"
                seq += 1
            full.setdefault("name", str(full["eid"]).replace("elec_", "Cuff "))
            n = _contact_count(full)
            pols = full.get("contact_polarities")
            if not (isinstance(pols, list) and len(pols) == n):
                full["contact_polarities"] = _default_polarities(full)
            fracs = full.get("contact_current_fractions")
            if not (isinstance(fracs, list)
                    and len(fracs) == len(full["contact_polarities"])):
                full["contact_current_fractions"] = (
                    [None] * len(full["contact_polarities"])
                )
            completed.append(full)

        st["designs"] = completed
        st["next_design_seq"] = max(seq, len(completed) + 1)
        if not str(st.get("selected_design_id") or ""):
            st["selected_design_id"] = completed[0]["eid"]
        if not str(st.get("active_design_id") or ""):
            st["active_design_id"] = completed[0]["eid"]

        # Mirror the selected design's params onto the state shim, the
        # way the GUI's _load_design_to_selected does. Several pipeline
        # helpers read cuff params (cuff_anchor, L_cuff_mm, …) off
        # `state`, not the design dict (e.g. cuff_fit.refit_design_geometry
        # reads state.cuff_anchor), so without this the design-level
        # overrides are silently ignored.
        sel = next(
            (d for d in completed
             if d["eid"] == str(st.get("selected_design_id") or "")),
            completed[0],
        )
        st.update({k: v for k, v in sel.items() if k not in ("eid", "name")})
        st["contact_count"] = len(sel.get("contact_polarities") or [])

        # One Default config per design (mirrors _create_config).
        configs = list(st.get("configs") or [])
        have = {str(c.get("design_id", "")) for c in configs}
        cseq = int(st.get("next_config_seq") or 1)
        i_stim = float(st.get("I_stim_mA") or 1.0)
        for d in completed:
            if d["eid"] in have:
                continue
            configs.append({
                "cid": f"cfg_{cseq:02d}",
                "design_id": d["eid"],
                "name": "Default",
                "contact_polarities": list(d.get("contact_polarities") or []),
                "contact_current_fractions":
                    list(d.get("contact_current_fractions") or []),
                "I_stim_mA": i_stim,
                "recording_montages": [],
            })
            cseq += 1
        st["configs"] = configs
        st["next_config_seq"] = cseq
        if configs and not str(st.get("selected_config_id") or ""):
            st["selected_config_id"] = configs[0]["cid"]
        if configs and not str(st.get("active_config_id") or ""):
            st["active_config_id"] = configs[0]["cid"]

    @classmethod
    def open(
        cls,
        project_dir: Path | str,
        *,
        user: str = "headless",
    ) -> "Study":
        """Attach to an existing project on disk. The project's
        `ui_state.json` (if present) is loaded into the state
        shim so the first compute call sees the same parameters
        the GUI would.

        Raises FileNotFoundError if `project_dir` doesn't exist.
        """
        pdir = Path(project_dir).expanduser().resolve()
        if not pdir.is_dir():
            raise FileNotFoundError(
                f"project not found: {pdir}"
            )
        s = cls(pdir, user=user)
        s._load_ui_state_if_present()
        return s

    @classmethod
    def create(
        cls,
        project_dir: Path | str,
        *,
        user: str = "headless",
    ) -> "Study":
        """Create a new empty project. Fails if the directory
        already exists with non-empty content (refuse to clobber
        an existing project; user can pass a fresh path or
        delete the existing one first).
        """
        pdir = Path(project_dir).expanduser().resolve()
        if pdir.exists() and any(pdir.iterdir()):
            raise FileExistsError(
                f"project directory not empty: {pdir} — "
                f"use Study.open(...) to attach instead, or "
                f"delete the directory first."
            )
        pdir.mkdir(parents=True, exist_ok=True)
        return cls(pdir, user=user)

    # ----- Context-manager sugar ----------------------------------------

    def __enter__(self) -> "Study":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    # ----- Lifecycle ----------------------------------------------------

    @property
    def project_dir(self) -> Path:
        return self._project_dir

    @property
    def user(self) -> str:
        return self._user

    def close(self) -> None:
        """Release the headless context. Safe to call multiple
        times. Currently a no-op — kept on the API for future
        cleanup (e.g., explicit FEniCSx mesh teardown)."""
        self._closed = True

    # ----- Read-only inspectors ----------------------------------------

    def list_designs(self) -> list[dict]:
        """Return the project's per-cuff designs (read from the
        active project's persisted `designs` list when present;
        empty when the project hasn't placed any cuffs yet)."""
        meta_path = self._project_dir / "ui_state.json"
        if not meta_path.is_file():
            return []
        try:
            blob = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            return []
        return list(blob.get("designs", []) or [])

    def list_configs(self) -> list[dict]:
        """Return the project's per-design FEM configs (same
        source as `list_designs`).
        """
        meta_path = self._project_dir / "ui_state.json"
        if not meta_path.is_file():
            return []
        try:
            blob = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            return []
        return list(blob.get("configs", []) or [])

    # ----- State setters -----------------------------------------------

    def set_mesh(self, **kwargs) -> None:
        """Set mesh-parameter state vars in bulk. Forwards
        directly to the state shim; no validation beyond what
        the pipeline driver does at run-mesh time.

        Common keys:
            use_epi, epi_thickness_um, lc_endo_um, lc_epi_um,
            lc_muscle_um, lc_saline_um, lc_silicone_um,
            lc_contact_um, lc_scar_um, decim_target_k,
            muscle_radial_pad_mm, muscle_axial_pad_mm,
            muscle_dx_mm, muscle_dy_mm, muscle_dz_mm.
        """
        for k, v in kwargs.items():
            self._state[k] = v

    def set_electrodes(self, designs: list[dict]) -> None:
        """Replace `state.designs` with the given list, completing
        each entry the way the GUI's "+ Add electrode" does.

        Callers may pass *minimal* dicts (e.g. just `eid`, `name`,
        `cuff_offset_mm`, `electrode_type`); every other field is
        filled from `golgi.app.DEFAULT_CUFF` + `DEFAULT_ELECTRODE`,
        contact polarities/fractions are derived from the electrode
        type, the first design becomes the selected/active one, and
        each design gets a Default FEM config — so `run_mesh` /
        `run_fem` have everything they need without the GUI.
        """
        self._state.designs = list(designs)
        self._finalize_designs_and_configs()

    def set_fiber_seed(self, **kwargs) -> None:
        """Set fiber-generation state vars in bulk. Common keys:
        n_fibers, fiber_max_steps, fiber_seed_end,
        fiber_cluster_eps_mm, fiber_cap_band_pct,
        fiber_min_rel_size_pct, fiber_axial_normal_thresh,
        fiber_auto_detect_branches, fiber_method.
        """
        for k, v in kwargs.items():
            self._state[k] = v

    # ----- Compute methods (wired to the production pipeline drivers) --

    def import_nerve(
        self,
        stl_path: Path | str,
        *,
        scale_factor: float = 1.0e-3,
    ) -> dict:
        """Load a nerve surface (STL/NAS/OBJ) into the project.

        Replicates the `_heavy_load` portion of `do_load_geometry`
        from the GUI path: reads the file, computes the global
        PCA frame, computes per-triangle surface quality, gathers
        topology stats. Writes the resulting fields into
        `self._geom` (nerve, centroid, R_global, nerve_q).

        Returns a small summary dict {n_pts, n_tris, n_components,
        bbox_mm, watertight, q_median} so the caller can log /
        assert against expectations.

        `scale_factor` converts source-file units to metres
        (default 1e-3 = mm → m). Same semantics as the GUI's
        scale_factor knob.
        """
        from golgi.app import (  # noqa: E402
            load_nerve_file,
            _surface_quality,
            _topology_stats,
        )
        from golgi.scene.cuff_fit import global_pca  # noqa: E402

        stl = Path(stl_path).expanduser().resolve()
        if not stl.is_file():
            raise FileNotFoundError(
                f"nerve file not found: {stl}"
            )
        nerve = load_nerve_file(
            str(stl), units_factor=float(scale_factor),
        )
        centroid, R_global = global_pca(nerve["pts_raw"])
        q, _ = _surface_quality(
            nerve["pts_raw"], nerve["boundary_raw"],
        )
        topo = _topology_stats(
            nerve["pts_raw"], nerve["boundary_raw"],
        )
        # Write into GeometryState — same fields the GUI path
        # writes; the catalog / FEM / mesh drivers all read
        # these directly.
        self._geom.nerve = nerve
        self._geom.centroid = centroid
        self._geom.R_global = R_global
        self._geom.nerve_q = q
        self._geom.nerve_poly = None
        self._geom._fit_locked = False
        self._geom._R_local_cached = None
        self._geom._R_ci_cached = None
        # Mirror selected_file + scale_factor onto the state
        # shim so any pipeline driver that reads them sees
        # consistent values.
        self._state["selected_file"] = str(stl)
        self._state["scale_factor"] = float(scale_factor)
        self._state["has_geometry"] = True
        bbox = topo.get("bbox_mm", (0.0, 0.0, 0.0))
        return {
            "n_pts": int(topo.get("n_pts", 0)),
            "n_tris": int(topo.get("n_tris", 0)),
            "n_components": int(topo.get("n_components", 0)),
            "bbox_mm": tuple(float(x) for x in bbox),
            "watertight": bool(topo.get("watertight", False)),
            "q_median": (
                float(q.median()) if hasattr(q, "median")
                else float(sorted(q)[len(q) // 2]) if len(q)
                else 0.0
            ),
        }

    def run_mesh(self) -> dict:
        """Build per-design TetGen meshes for every design in
        `state.designs`. Returns {eid: msh_path}.

        Equivalent to clicking "Build mesh (TetGen)" in the GUI
        with every design selected. Each design's outputs land
        in `<project>/designs/<eid>/`; the returned dict points
        at each design's `nerve.msh`.
        """
        import asyncio
        from golgi.pipeline import mesh as _pipeline_mesh
        ctx = self._ensure_ctx()
        # Headless run_mesh builds EVERY design (the GUI builds the
        # selected/multi-selected one); pass all eids explicitly.
        all_eids = [
            str(d.get("eid")) for d in (self._state.get("designs") or [])
            if d.get("eid")
        ]
        asyncio.run(
            _pipeline_mesh.run_mesh_build(ctx, design_eids=all_eids or None)
        )
        out: dict[str, Path] = {}
        designs_dir = self._project_dir / "designs"
        for d in (self._state.get("designs") or []):
            eid = str(d.get("eid") or "")
            if not eid:
                continue
            msh = designs_dir / eid / "nerve.msh"
            if msh.is_file():
                out[eid] = msh
        return out

    def run_fem(self) -> dict:
        """Solve FEM for every config in `state.configs`.
        Returns {cid: fem_outputs_dir}.

        Equivalent to clicking "Simulate → FEM → Solve" in the
        GUI. Iterates the configs the driver picks up from
        state (defaults to every config when none is explicitly
        active). Each config's outputs land in
        `<project>/configs/<cid>/` — the returned dict points
        at each directory.

        Requires `Study.run_mesh()` to have succeeded first
        (the driver short-circuits when no mesh exists).
        """
        import asyncio
        from golgi.pipeline import fem as _pipeline_fem
        ctx = self._ensure_ctx()
        asyncio.run(_pipeline_fem.run_fem_solve(ctx))
        out: dict[str, Path] = {}
        configs_dir = self._project_dir / "configs"
        for c in (self._state.get("configs") or []):
            cid = str(c.get("cid") or "")
            if not cid:
                continue
            d = configs_dir / cid
            if d.is_dir():
                out[cid] = d
        return out

    def run_fibers(self) -> dict:
        """Generate fiber trajectories on the loaded nerve.
        Subprocess that runs the Laplace solve + RK4 integrator
        + cap detection + branch classification; writes
        `<project>/nerve_paths_fibers.npz` and friends.

        Returns a summary dict {n_paths, n_branches,
        n_pts_total, branch_summary}.
        """
        import asyncio
        from golgi.pipeline import fibers as _pipeline_fibers
        ctx = self._ensure_ctx()
        asyncio.run(_pipeline_fibers.run_generate_fibers(ctx))
        n_paths = (
            len(self._geom.fiber_paths_raw)
            if getattr(self._geom, "fiber_paths_raw", None)
            else 0
        )
        n_branches = int(
            getattr(self._geom, "fiber_n_branches", 0) or 0
        )
        n_pts_total = 0
        if n_paths and getattr(
            self._geom, "fiber_paths_raw", None,
        ) is not None:
            n_pts_total = sum(
                int(len(p))
                for p in self._geom.fiber_paths_raw
            )
        return {
            "n_paths": n_paths,
            "n_branches": n_branches,
            "n_pts_total": n_pts_total,
            "branch_summary": list(
                getattr(self._state, "fiber_branch_summary", [])
                or [],
            ),
        }

    def load_cached_geometry(self) -> dict:
        """Hydrate the in-memory ctx.geom from a project's cached
        fibers + per-fiber FEM lead field ON DISK, WITHOUT re-solving.

        `run_sweep` reads `geom.fiber_paths_Ve` (per-fiber extracellular
        potential) and `geom.fiber_paths_raw` from the live context, not
        from disk. For a project that was built in a PREVIOUS session
        (e.g. a staged bundle, or a Duke-pipeline study) those live
        arrays are empty, so a sweep can't run without first re-tracing
        fibers + re-solving the FEM. This loads them straight from the
        cached `nerve_paths_fibers.npz` (per-design) + `paths_Ve.npz`
        (per-config) instead — the same split-by-`path_lengths` layout
        the pipeline writes.

        Returns {n_fibers, n_ve_fibers, fiber_src, ve_src}."""
        import numpy as _np
        from golgi.pipeline import fem_layout as _fl
        geom = self._geom
        pdir = self._project_dir

        def _split(flat, lens):
            out, off = [], 0
            for L in lens:
                n = int(L)
                out.append(_np.asarray(flat[off:off + n]).copy())
                off += n
            return out

        # ---- fibers: designs/<eid>/nerve_paths_fibers.npz, else root ----
        fiber_src = None
        cand = [pdir / "nerve_paths_fibers.npz"]
        dsg = pdir / "designs"
        if dsg.is_dir():
            cand += [
                d / "nerve_paths_fibers.npz"
                for d in sorted(dsg.iterdir()) if d.is_dir()
            ]
        n_fibers = 0
        for fp in cand:
            if not fp.is_file():
                continue
            d = _np.load(fp, allow_pickle=True)
            paths = _split(d["paths_flat"], d["path_lengths"])
            geom.fiber_paths_raw = paths
            geom.fiber_branch_idx = _np.zeros(len(paths), dtype=_np.int32)
            geom.fiber_n_branches = 1
            geom.fibers_in_cuff_frame = bool(
                "frame_is_cuff" in d.files
                and int(d["frame_is_cuff"]) == 1
            )
            n_fibers = len(paths)
            fiber_src = str(fp)
            break

        # ---- per-fiber field: configs/<cid>/paths_Ve.npz ----
        ve_src = None
        cfgs = _fl.enumerate_configs(pdir)
        cfg_ids = [c["id"] for c in cfgs] if cfgs else []
        active = str(getattr(self._state, "active_config_id", "") or "")
        order = ([active] if active in cfg_ids else []) + [
            c for c in cfg_ids if c != active
        ]
        n_ve = 0
        for cid in order:
            pv_path = _fl.config_dir(pdir, cid) / "paths_Ve.npz"
            if not pv_path.is_file():
                continue
            pv = _np.load(pv_path, allow_pickle=True)
            lens = pv["path_lengths"]
            geom.fiber_paths_Ve = _split(pv["Ve_flat"], lens)
            geom.fiber_paths_Ez = (
                _split(pv["Ez_flat"], lens)
                if "Ez_flat" in pv.files else None
            )
            geom.fiber_paths_for_Ve = (
                _split(pv["paths_flat"], lens)
                if "paths_flat" in pv.files else None
            )
            n_ve = len(geom.fiber_paths_Ve)
            ve_src = str(pv_path)
            try:
                self._state.active_config_id = cid
            except Exception:                                # noqa: BLE001
                pass
            break

        return {
            "n_fibers": n_fibers, "n_ve_fibers": n_ve,
            "fiber_src": fiber_src, "ve_src": ve_src,
        }

    def run_sweep(self, request) -> object:
        """Run a parameter sweep (F2.1 `SweepRequest`).
        Returns the `SweepResult`; also writes it into the
        project's sweep cache, tagged with the active config
        via the F3.2 per-config sweep tagging.

        Recruitment-mode and threshold-mode are both supported.
        Cancellation isn't wired in headless — the driver runs
        to completion.
        """
        import asyncio
        from golgi.pipeline import sweep as _pipeline_sweep
        from golgi.projects import sweep_cache as _swc
        ctx = self._ensure_ctx()
        # Fill stimulus pulse parameters from the fiber-stimulus state when the
        # request doesn't carry them. The GUI builds these via
        # H.fiber_pulse_params() before dispatching; headless callers usually
        # leave SweepRequest.pulse_params empty, which would KeyError deep in
        # the per-fiber sim.
        if not getattr(request, "pulse_params", None):
            try:
                pp = ctx.helpers.fiber_pulse_params()
                try:
                    request.pulse_params = pp
                except Exception:                        # frozen dataclass
                    import dataclasses
                    request = dataclasses.replace(request, pulse_params=pp)
            except Exception as ex:                      # noqa: BLE001
                print(f"[headless] pulse-params build failed: {ex}", flush=True)
        result = asyncio.run(
            _pipeline_sweep.run_sweep(ctx, request),
        )
        # Persist to the project's sweep cache, tagged with the
        # currently-active config_id when one is set.
        active_cid = (
            str(getattr(self._state, "active_config_id", "")
                or "")
        )
        try:
            _swc.save_sweep(
                result, self._project_dir,
                write_csvs=True,
                cid=active_cid or None,
            )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[headless] sweep cache write failed: {ex}",
                flush=True,
            )
        return result

    # ----- Bundle operations (already standalone, no closure deps) -----

    def export_bundle(self, out_zip: Path | str) -> Path:
        """Pack the active project into a reproducible study
        zip (F2.2 bundle). Returns the written path.

        Delegates to `golgi.projects.bundle.export_study`.
        """
        from golgi.projects import bundle as _bundle  # noqa: E402

        out = Path(out_zip).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = _bundle.export_study(
            self._project_dir,
            exported_by_user=self._user,
        )
        out.write_bytes(blob)
        return out

    @staticmethod
    def import_bundle(
        zip_path: Path | str,
        target_dir: Path | str,
    ) -> Path:
        """Unpack a study bundle into `target_dir`. Returns the
        project dir of the imported study.

        Delegates to `golgi.projects.bundle.import_study`.
        """
        from golgi.projects import bundle as _bundle  # noqa: E402

        zp = Path(zip_path).expanduser().resolve()
        td = Path(target_dir).expanduser().resolve()
        td.mkdir(parents=True, exist_ok=True)
        manifest = _bundle.import_study(zp, td)
        # `import_study` writes into `td/<original_name>/`; the
        # manifest carries the project name so we can resolve.
        proj_name = (
            manifest.get("project", {}).get("name") or td.name
        )
        return td / proj_name

    # ----- PipelineContext builder (private) ---------------------------

    def _ensure_ctx(self):
        """Build (and cache) the headless `PipelineContext` the
        pipeline drivers expect. Constructed lazily on the first
        compute call so importing `golgi.Study` itself stays
        cheap (no SimpleNamespace populated, no `helpers` deps
        chased) when the user only needs lifecycle / bundle ops.

        The helpers bag mirrors what `build_app()` constructs at
        `app.py:13561` but only includes the entries the headless
        pipeline drivers actually reach. Phase B includes the
        mesh-driver dependencies; Phase C adds the FEM + sweep
        + fibers entries.
        """
        if self._ctx is not None:
            return self._ctx

        from types import SimpleNamespace
        from golgi.pipeline.context import PipelineContext
        from golgi.pipeline.plc import (
            assemble_multi_domain_plc,
            assemble_bare_nerve_in_bath,
        )
        from golgi.pipeline.mesh_quality import (
            tet_shape_quality,
        )
        from golgi.pipeline.fiber_backends import (
            axonml_run_single,
        )
        from golgi.figures.mesh_stats import (
            _build_quality_histogram_figure,
            _compute_mesh_stats_html,
        )
        from golgi.scene.cuff_fit import (
            refit_design_geometry,
        )
        from golgi.scene.electrode_patches import (
            build_electrode_patches_dicts,
        )
        # Helpers already at module level in golgi.app.
        # `_contact_count` and `_default_polarities` are closures
        # inside `build_app()` — we inline equivalents below
        # rather than extract (small, self-contained logic).
        from golgi.app import (
            write_msh22,
            _extract_region_surfaces_mm,
            _build_viz_surfaces,
            _classify_fibers_by_branch,
            _compute_fiber_branch_summary,
            _fiber_effective_anod_pw_ms,
            build_pulse_waveform,
            build_pulse_breakpoints,
            DEFAULTS,
            DEFAULT_ELECTRODE,
            DUKE_ELECTRODE_TYPE,
            MAX_FIBER_BRANCHES as _MAX_FIBER_BRANCHES,
            MYELINATED_MODELS,
            UNMYELINATED_MODELS,
            TAB10_PALETTE,
            _CUFF_PRESETS,
            POLARITY_CHOICES,
            get_active,
        )
        from golgi.watchers.fiber_panel import (
            fiber_pulse_params as _wfiber_pulse_params,
        )
        import cuff_designer  # external pkg

        # Bind `geom` + `state` so headless closures see them.
        _geom = self._geom
        _state = self._state
        _project_dir = self._project_dir

        def _refit_design_geometry_headless(eid: str) -> bool:
            return refit_design_geometry(
                eid, geom=_geom, state=_state,
            )

        # F4.1 Phase C — inline headless equivalents of the
        # `build_app()` closures the pipeline drivers reach for.
        # Each is a small, self-contained translation of the
        # GUI closure that takes only the headless geom + state
        # into account (no UI side effects).

        # `_contact_count` + `_default_polarities` inlined from
        # build_app — they're closures in the GUI path but the
        # logic is self-contained.
        def _contact_count_inline(elec: dict) -> int:
            kind = str(
                elec.get(
                    "electrode_type", "bipolar ring-pair",
                ),
            )
            if kind == "bipolar ring-pair":
                return 2
            if kind == "tripolar (anode-cathode-anode)":
                return 3
            if kind == "ring-array (NxM)":
                try:
                    rows = int(elec.get("array_n_rows", 2))
                    cols = int(elec.get("array_n_cols", 4))
                except (TypeError, ValueError):
                    rows, cols = 2, 4
                return max(0, rows * cols)
            if kind == "helical (Livanova-style)":
                return 2
            if kind == "LIFE (longitudinal intrafascicular)":
                try:
                    rows = int(elec.get("life_n_rows", 1))
                    cols = int(elec.get("life_n_cols", 1))
                except (TypeError, ValueError):
                    rows, cols = 1, 1
                return max(0, rows * cols)
            if kind == DUKE_ELECTRODE_TYPE:
                preset_name = str(
                    elec.get("duke_preset", "") or "",
                )
                preset = _CUFF_PRESETS.get(preset_name)
                if preset is None:
                    return 0
                n = 0
                for inst in preset.get("instances", []):
                    if inst.get("type") in (
                        "LivaNova_Primitive",
                        "CircleContact_Primitive",
                    ):
                        n += 1
                return n
            return 0

        def _default_polarities_inline(elec: dict) -> list:
            n = _contact_count_inline(elec)
            if n <= 0:
                return []
            if n == 1:
                return ["anode"]
            if n == 2:
                return ["anode", "cathode"]
            if n == 3:
                return ["anode", "cathode", "anode"]
            return [
                "anode" if (i % 2 == 0) else "cathode"
                for i in range(n)
            ]

        def _ensure_polarities_headless(elec: dict) -> list:
            """Headless mirror of app.py `_ensure_polarities`.
            Returns a polarity list of the right length for the
            electrode, persisting defaults onto the dict the
            first time. Maps legacy "active" → "anode"."""
            n = _contact_count_inline(elec)
            existing = elec.get("contact_polarities", None)
            if (isinstance(existing, list)
                    and len(existing) == n):
                migrated = [
                    ("anode" if p == "active" else p)
                    for p in existing
                ]
                if all(
                    p in POLARITY_CHOICES for p in migrated
                ):
                    elec["contact_polarities"] = migrated
                    return list(migrated)
            pols = _default_polarities_inline(elec)
            elec["contact_polarities"] = pols
            return list(pols)

        def _cuff_ns_extras_headless(
            r_nerve_m: float | None = None,
        ) -> dict:
            """Headless mirror of app.py `_cuff_ns_extras`."""
            if r_nerve_m is not None and float(r_nerve_m) > 0.0:
                r_nerve = float(r_nerve_m)
            else:
                r_nerve = (
                    float(_geom.R_ci) if _geom.R_ci
                    else 1.5e-3
                )
            return {
                "z_nerve": 0.0,
                "r_nerve": r_nerve,
                "r_n": r_nerve,
            }

        def _save_fiber_sim_cache_headless() -> None:
            """Headless persistence of `geom.fiber_sim_results`
            to `<active_config>/fiber_sim_results.pkl`. Skipped
            silently when there's nothing to save."""
            if not _geom.fiber_sim_results:
                return
            import pickle
            cid = str(
                getattr(_state, "active_config_id", "")
                or "default"
            )
            sim_dir = _project_dir / "configs" / cid / "sims"
            sim_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(
                    sim_dir / "fiber_sim_results.pkl", "wb",
                ) as f:
                    pickle.dump({
                        "version": 1,
                        "results": _geom.fiber_sim_results,
                        "view_idx": int(
                            getattr(
                                _state, "fiber_sel_idx", 0,
                            ),
                        ),
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as ex:                      # noqa: BLE001
                print(
                    f"[headless] fiber sim cache write "
                    f"failed: {ex}",
                    flush=True,
                )

        def _save_pop_state_headless() -> None:
            """Headless persistence of population sim results."""
            if (getattr(_geom, "fiber_pop_types", None)
                    is None):
                return
            import pickle
            cid = str(
                getattr(_state, "active_config_id", "")
                or "default"
            )
            sim_dir = _project_dir / "configs" / cid / "sims"
            sim_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(
                    sim_dir / "pop_state.pkl", "wb",
                ) as f:
                    pickle.dump({
                        "version": 1,
                        "pop_types": _geom.fiber_pop_types,
                        "pop_results": getattr(
                            _geom, "fiber_pop_results", None,
                        ),
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as ex:                      # noqa: BLE001
                print(
                    f"[headless] pop state write failed: "
                    f"{ex}",
                    flush=True,
                )

        def _fiber_paths_display_headless() -> "list | None":
            """Headless mirror of app.py `_fiber_paths_display`
            — returns fiber paths in the viewport's pure-PCA
            frame (metres), None when no fibers exist."""
            import numpy as np
            if _geom.fiber_paths_raw is None:
                return None
            if (getattr(_geom, "fibers_in_cuff_frame", False)
                    and getattr(_geom, "cuff_origin_pca", None)
                    is not None):
                _off = np.asarray(
                    _geom.cuff_origin_pca, dtype=np.float64,
                )
                return [
                    np.asarray(p, dtype=np.float64) + _off
                    for p in _geom.fiber_paths_raw
                ]
            if (_geom.centroid is not None
                    and _geom.R_global is not None):
                _c = np.asarray(
                    _geom.centroid, dtype=np.float64,
                )
                _R = np.asarray(
                    _geom.R_global, dtype=np.float64,
                )
                return [
                    (np.asarray(p, dtype=np.float64) - _c) @ _R
                    for p in _geom.fiber_paths_raw
                ]
            return list(_geom.fiber_paths_raw)

        def _fiber_label_and_color_headless(
            idx: int,
        ) -> tuple[str, str]:
            """Headless: cheap label "Fiber N" + tab10 colour
            cycling. The GUI version pulls branch names from
            state.fiber_branch_name_<i>; headless just uses
            indices."""
            colour = TAB10_PALETTE[
                int(idx) % len(TAB10_PALETTE)
            ]
            return (f"Fiber {int(idx)}", colour)

        def _fiber_pulse_params_headless() -> dict:
            """Headless pulse-params builder. Reuses the same
            module-level function the GUI watcher uses."""
            return _wfiber_pulse_params(
                _state,
                effective_anod_pw_ms=_fiber_effective_anod_pw_ms,
            )

        def _noop_nullary() -> None:
            return None

        helpers = SimpleNamespace(
            # ---- Mesh-driver deps (Phase B) ----
            assemble_multi_domain_plc=assemble_multi_domain_plc,
            assemble_bare_nerve_in_bath=assemble_bare_nerve_in_bath,
            write_msh22=write_msh22,
            tet_shape_quality=tet_shape_quality,
            compute_mesh_stats_html=_compute_mesh_stats_html,
            build_quality_histogram_figure=(
                _build_quality_histogram_figure
            ),
            extract_region_surfaces=_extract_region_surfaces_mm,
            build_viz_surfaces=_build_viz_surfaces,
            defaults_by_tag=DEFAULTS,
            refit_design_geometry=_refit_design_geometry_headless,
            active_project=get_active,
            script_cwd=Path(__file__).parent.parent,
            # ---- Fibers-driver deps (Phase C) ----
            classify_fibers_by_branch=_classify_fibers_by_branch,
            compute_fiber_branch_summary=(
                _compute_fiber_branch_summary
            ),
            MAX_FIBER_BRANCHES=_MAX_FIBER_BRANCHES,
            refresh_fiber_sel_items=_noop_nullary,
            refresh_pop_branches_meta=_noop_nullary,
            # ---- FEM-driver deps (Phase D — partial) ----
            # Provided so a future Study.run_fem() doesn't need
            # additional plumbing. cuff_ns_extras + ensure_polarities
            # are headless inlines above; refresh_fem_plots is a
            # UI no-op. _CUFF_PRESETS / DEFAULT_ELECTRODE / DUKE
            # come from app.py constants.
            transform_to_cuff_frame=None,  # FEM-only; lazy
            build_electrode_patches_dicts=(
                build_electrode_patches_dicts
            ),
            cuff_designer=cuff_designer,
            _CUFF_PRESETS=_CUFF_PRESETS,
            DUKE_ELECTRODE_TYPE=DUKE_ELECTRODE_TYPE,
            DEFAULT_ELECTRODE=DEFAULT_ELECTRODE,
            cuff_ns_extras=_cuff_ns_extras_headless,
            ensure_polarities=_ensure_polarities_headless,
            refresh_fem_plots=_noop_nullary,
            # ---- Fiber-sim-driver deps (Phase C — for sweep) ----
            axonml_run_single=axonml_run_single,
            build_pulse_waveform=build_pulse_waveform,
            build_pulse_breakpoints=build_pulse_breakpoints,
            MYELINATED_MODELS=MYELINATED_MODELS,
            UNMYELINATED_MODELS=UNMYELINATED_MODELS,
            fiber_pulse_params=_fiber_pulse_params_headless,
            fiber_label_and_color=(
                _fiber_label_and_color_headless
            ),
            save_fiber_sim_cache=_save_fiber_sim_cache_headless,
            # ---- Pop-sim-driver deps (Phase C — for sweep) ----
            TAB10_PALETTE=TAB10_PALETTE,
            fiber_paths_display=_fiber_paths_display_headless,
            save_pop_state=_save_pop_state_headless,
        )

        # Headless ctx callbacks: all no-ops. The drivers call
        # `ctx.stamp_user_line(line)` to prefix log lines with
        # `[user@time]`; in headless we leave the line as-is.
        # Cancellation is always False (no cancel UI). The
        # subprocess hooks need to be present but can be no-ops
        # — the TetGen subprocess is registered for the GUI's
        # cancel button; headless ignores it.
        def _stamp_user_line(line: str) -> str:
            return line

        def _noop(*_args, **_kwargs) -> None:
            return None

        def _always_false() -> bool:
            return False

        self._ctx = PipelineContext(
            state=_state,
            geom=_geom,
            scene=self._scene,
            stamp_user_line=_stamp_user_line,
            autosave=_noop,
            safe_update=_noop,
            safe_reset_camera=_noop,
            register_subprocess=_noop,
            clear_subprocess=_noop,
            was_cancelled=_always_false,
            helpers=helpers,
        )
        return self._ctx

    # ----- Debugging hooks (private) ----------------------------------

    def _load_ui_state_if_present(self) -> None:
        """If the project has a persisted ui_state.json, load it
        into the state shim. Called by `Study.open()` so the
        headless state mirrors what the GUI would have on
        attach."""
        path = self._project_dir / "ui_state.json"
        if not path.is_file():
            return
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            return
        for k, v in blob.items():
            self._state[k] = v

    def _state_dump(self) -> dict[str, Any]:
        """Return a copy of every state var the shim is holding.
        Useful for diagnosing what the pipeline saw on a run."""
        return self._state._asdict()
