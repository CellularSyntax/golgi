# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Mesh-build pipeline driver (F3.2a per-design refactor).

Each electrode design owns its own multi-domain mesh under
`<out>/designs/<eid>/nerve.msh`. The nerve + muscle bbox are
SHARED across every design's mesh (built in the canonical PCA-
translated frame with the anchor design at origin); only the
cuff silicone shell + saline gap + contact patches move/rotate
per design. So when the user switches designs in the viewport,
only the cuff visibly changes — the nerve and muscle stay put.

`run_mesh_build(ctx, design_eids=...)` iterates the requested
designs (defaulting to the currently-selected one) and for each:

  1. Refit the design if needed so R_ci_m / R_co_m /
     R_local_elec are populated.
  2. Compute the design's cuff transform (offset_canon, R) in
     the shared canonical frame.
  3. Call `helpers.assemble_multi_domain_plc` with the
     CANONICAL-FRAME nerve points and the design's
     (cuff_offset_m, cuff_R). The PLC assembler places the
     silicone shell + saline gap + cap rings at that pose.
  4. Spawn TetGen via `TetGenRunner`; stream stdout to the
     busy-lightbox + state.mesh_log.
  5. write_msh22 into `<out>/designs/<eid>/nerve.msh`; compute
     per-tet quality + mesh-stats HTML + quality histogram for
     the UI (last design's stats win; if you want per-design
     diagnostics use the per-design dir + designs index).
  6. Pre-extract per-region surfaces via vtkGeometryFilter.

After the loop, the currently-selected design's mesh artefacts
land in `geom.{msh_path, mesh_nodes, mesh_elems, mesh_tags,
mesh_q, region_surfaces}` so the viewport renders the active
mesh; flipping `selected_design_id` swaps which mesh is loaded
(handled by app.py's mesh-restore watcher). All on-disk meshes
are in the canonical frame — no inverse transform needed at
render time.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from golgi.jobs import CancelToken, JobRequest, LocalSubprocessRunner
from golgi.jobs.schemas import TetGenPayload
from ._throttle import FlushThrottle
from .context import PipelineContext
from .fem_layout import design_dir as _design_dir

# Path to the runner script (sibling compute/ directory). Resolved
# at module-load — not call-time — so we don't stat the package
# tree on every build.
_TETGEN_RUNNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "compute" / "tetgen_runner.py"
)


@dataclass
class TetGenJobRequest(JobRequest):
    """Carries the typed `TetGenPayload` schema across the
    runner boundary. `payload_path` is where the JSON serialization
    lands on disk; `out_npz` mirrors `payload.out_npz` and lets
    the runner's _collect_outputs report it without unpacking
    the payload."""
    payload: TetGenPayload
    payload_path: Path
    out_npz: Path


class TetGenRunner(LocalSubprocessRunner):
    """LocalSubprocessRunner specialised for golgi/compute/tetgen_runner.py.
    Overrides the three hooks so the typed payload schema
    round-trips through a flat JSON dict the runner script can
    parse via TetGenPayload.deserialize."""

    def _build_payload_path(self, req):
        return req.payload_path

    def _serialize_payload(self, req):
        # Runner script reads its config via TetGenPayload.
        # deserialize(json.load(...)) — so we hand it the
        # serialized form of just the payload, not the wrapping
        # JobRequest. (Skip the default dataclass-asdict, which
        # would emit the wrapper too.)
        return req.payload.serialize()

    def _collect_outputs(self, req):
        return {"npz": req.out_npz}


def run_tetgen_subprocess(
    plc_path: Path,
    payload: TetGenPayload,
    on_line: "callable",
    proc_sink: "callable | None" = None,
    *,
    payload_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backwards-compatible wrapper around TetGenRunner. Takes
    a typed TetGenPayload instead of the old free-form dict.
    `payload_dir` (kw-only) replaces the implicit GOLGI_OUT
    dependency that the in-golgi.py version had via closure.

    Returns (nodes, elems, cell_attribs). Raises RuntimeError
    on non-zero return code.
    """
    req = TetGenJobRequest(
        payload=payload,
        payload_path=payload_dir / "current_tetgen_payload.json",
        out_npz=payload.out_npz,
    )
    runner = TetGenRunner(_TETGEN_RUNNER_PATH)

    # Adapter: legacy proc_sink → CancelToken arm() so the
    # in-build_app `_register_subprocess` keeps capturing the
    # Popen handle. was_requested()/clear() are no-ops because
    # the legacy flow polls its own _cancellation dict via
    # _was_cancelled; the runner doesn't need to poll here.
    tok = CancelToken()
    if proc_sink is not None:
        _orig_arm = tok.arm

        def _arm_and_forward(proc):
            _orig_arm(proc)
            try:
                proc_sink(proc)
            except Exception:
                pass
        tok.arm = _arm_and_forward  # type: ignore[method-assign]

    out = runner.run(req, on_line, tok)
    if out.return_code != 0:
        raise RuntimeError(
            f"TetGen failed (returncode={out.return_code})"
        )
    r = np.load(out.outputs["npz"])
    return r["nodes"], r["elems"], r["attribs"]


def _is_prismatic(pts, tol: float = 0.10) -> bool:
    """True if the nerve cross-section is ~constant along z (an extruded
    slice) — the regime where the gmsh slice-and-extrude mesher is exact.
    Compares the xy bounding box in the lower vs upper quarter of z."""
    p = np.asarray(pts, dtype=float)
    if p.ndim != 2 or p.shape[0] < 16:
        return False
    z = p[:, 2]
    zr = float(z.max() - z.min())
    if zr <= 1e-9:
        return False
    lo = p[z < z.min() + 0.25 * zr]
    hi = p[z > z.max() - 0.25 * zr]
    if len(lo) < 8 or len(hi) < 8:
        return False

    def _bb(q):
        return np.array([q[:, 0].min(), q[:, 0].max(),
                         q[:, 1].min(), q[:, 1].max()])

    blo, bhi = _bb(lo), _bb(hi)
    scale = max(blo[1] - blo[0], blo[3] - blo[2], 1e-9)
    return float(np.max(np.abs(bhi - blo)) / scale) < tol


async def run_mesh_build(
    ctx: PipelineContext,
    design_eids: list[str] | None = None,
) -> None:
    """Build per-design multi-domain meshes for the designs whose
    eids are in `design_eids` (or just the currently-selected
    design when None). Each design's outputs land in
    `<out>/designs/<eid>/`.

    Preconditions: at least one design exists. Designs that don't
    yet have a refit (R_local_elec absent) are auto-refit first.

    Side effects: writes per-design mesh files; mutates `geom.*`
    + `state.*` to reflect the CURRENTLY-SELECTED design's mesh
    (so the viewport renders the active build). Each meshed
    design's `has_mesh` flag is implicit — its `nerve.msh` either
    exists on disk or it doesn't."""
    state = ctx.state
    geom = ctx.geom
    H = ctx.helpers

    # Resolve which designs to build.
    all_designs = list(state.designs or [])
    if not all_designs:
        state.mesh_log = (
            "No designs to mesh. Add a cuff design first."
        )
        return
    if design_eids:
        wanted = set(design_eids)
        targets = [d for d in all_designs
                   if d.get("eid") in wanted]
    else:
        sel = str(state.selected_design_id or "")
        targets = [d for d in all_designs
                   if d.get("eid") == sel]
    if not targets:
        state.mesh_log = (
            "No matching designs to mesh — pick at least one "
            "in the Mesh tab's multi-select."
        )
        return
    print(
        "[MESH-DRIVER-DEBUG] design_eids arg="
        f"{design_eids!r}  all_designs eids="
        f"{[d.get('eid') for d in all_designs]!r}  "
        f"resolved targets="
        f"{[d.get('eid') for d in targets]!r}",
        flush=True,
    )

    n_total = len(targets)
    state.busy = True
    state.busy_msg = f"Building {n_total} design mesh(es)"
    state.busy_log = "Preparing per-design build…"
    state.mesh_log = state.busy_log
    state.flush()

    loop = asyncio.get_event_loop()
    # Trailing-edge debounce on state.flush() to cap WS traffic
    # during TetGen output bursts (50+ lines/sec → ~7/sec).
    throttle = FlushThrottle(loop=loop, state=state)
    log_lines: list[str] = []

    def _on_line(line: str):
        line = ctx.stamp_user_line(line)
        print(f"[mesh] {line}", flush=True)
        line = line[:220]
        log_lines.append(line)
        tail = "\n".join(log_lines[-14:])
        full = "\n".join(log_lines[-300:])

        def _push():
            state.busy_log = tail
            state.mesh_log = full
            throttle.tick()
        try:
            loop.call_soon_threadsafe(_push)
        except RuntimeError:
            pass

    out_dir = Path(H.active_project().out_dir)
    selected_id = str(state.selected_design_id or "")
    last_built: dict = {}  # per-design outputs for the loaded one
    n_done = 0

    # ---- Shared canonical-frame inputs (one-shot per batch).
    # Each design's mesh is built in ITS OWN cuff-local frame
    # (cuff axis-aligned at origin) — that's what TetGen handles
    # reliably across real-world branched-VN inputs. To keep the
    # NERVE and MUSCLE shape-identical across designs (so only
    # the cuff visibly moves when switching designs), we:
    #
    #   1. Compute the canonical-frame nerve points ONCE (PCA-
    #      translated, anchor design at origin).
    #   2. Pre-build the canonical-frame muscle pieces ONCE (auto-
    #      fit to the canonical-nerve bbox, axis-aligned to
    #      canonical +z).
    #
    # Then for each design D in the loop we compute D's transform
    # canonical → D's cuff-local, apply it to BOTH the nerve points
    # AND the muscle pieces, and hand the design-local nerve +
    # design-local muscle to `assemble_multi_domain_plc`. TetGen
    # always sees an axis-aligned cuff at origin. The on-disk mesh
    # is in D's cuff-local; at viewport time we rotate it back to
    # PCA-translated so all designs co-render and only the cuff
    # appears at a different position/orientation per design.
    from golgi.pipeline.plc import (
        build_muscle_pieces_for_nerve,
        transform_muscle_pieces,
    )
    from golgi.scene.cuff_fit import (
        anchor_origin_pca_for_designs,
        _design_M,
        find_cuff_origin_pca,
    )
    nerve_pts_canon = None
    anchor_origin_pca = None
    pts_pca_all = None
    muscle_canon = None
    if geom.nerve is not None and geom.centroid is not None \
            and geom.R_global is not None:
        pts_pca_all = (
            (geom.nerve["pts_raw"] - geom.centroid)
            @ geom.R_global
        )
        # F3.2-M2.1f — canonical frame == pure PCA frame.
        # Anchor offset is forced to zero so every design's mesh
        # back-transforms to (PCA-centroid)-relative coords —
        # the same frame the nerve actor + fibers + electrodes
        # render in. Without this, design D's mesh sat at
        # (cuff_D_origin - cuff_anchor_origin), while fibers
        # rendered at (cuff_currently_fit - 0), so the two
        # frames only matched when the last-fit cuff was the
        # anchor. The `anchor_origin_pca_for_designs` helper
        # stays in cuff_fit.py but is no longer used by the
        # mesh pipeline — keep it in case external callers
        # (or M2.2 per-design FEM solves) need it.
        anchor_origin_pca = np.zeros(3, dtype=np.float64)
        # Canonical-frame nerve = pure PCA (centroid at origin).
        # No design rotation applied yet — that happens per design
        # inside the loop, when we transform to design-local.
        nerve_pts_canon = pts_pca_all.copy()
        # Precompute the canonical-frame muscle bbox + pieces ONCE.
        # We use the canonical nerve points (un-preprocessed; bbox
        # math is fast even on the raw 275k-pt cloud). The cuff
        # length effective for the n_axial_mus sizing is approx
        # the user's L_cuff_mm — close enough for refinement
        # density. Pass through the user's muscle pad knobs.
        _L_eff = float(state.L_cuff_mm) * 1e-3
        muscle_canon = build_muscle_pieces_for_nerve(
            nerve_pts_canon,
            muscle_radial_pad_m=float(state.muscle_radial_pad_mm) * 1e-3,
            muscle_axial_pad_m=float(state.muscle_axial_pad_mm) * 1e-3,
            muscle_dx_m=float(state.muscle_dx_mm) * 1e-3,
            muscle_dy_m=float(state.muscle_dy_mm) * 1e-3,
            muscle_dz_m=float(state.muscle_dz_mm) * 1e-3,
            L_cuff_eff=_L_eff, n_circ=96,
        )

    try:
        for design in targets:
            eid = str(design.get("eid", ""))
            if not eid:
                continue
            n_done += 1
            display = design.get("name", eid)
            state.busy_msg = (
                f"Mesh [{n_done}/{n_total}]: {display}"
            )
            state.flush()
            _on_line(
                f"=== Design '{eid}' ({display}) "
                f"[{n_done}/{n_total}] ==="
            )
            # Ensure the design has a refit so R_local_elec
            # exists and R_ci / R_co are populated.
            if not design.get("R_local_elec"):
                _on_line(
                    f"  refitting design '{eid}' "
                    f"(no R_local_elec on file)…"
                )
                H.refit_design_geometry(eid)
                # Refit mutates state.designs; pull the fresh
                # dict for this iteration.
                design = next(
                    (d for d in (state.designs or [])
                     if d.get("eid") == eid),
                    design,
                )

            if (nerve_pts_canon is None
                    or anchor_origin_pca is None
                    or muscle_canon is None
                    or pts_pca_all is None):
                _on_line(
                    f"  ⚠ design '{eid}': no nerve loaded — "
                    f"skipping"
                )
                continue

            # Compute design D's transform from canonical to
            # D's cuff-local frame.
            # p_canon = p_pca - anchor_origin_pca
            # p_design_local = (p_canon - design_offset_canon) @ M_D
            #   where:
            #     design_offset_canon = cuff_origin_D_pca - anchor_origin_pca
            #     M_D = D's R_local_elec.T @ Rx @ Ry @ Rz
            M_D = _design_M(design)
            cuff_origin_D_pca = find_cuff_origin_pca(
                pts_pca_all, state.cuff_anchor,
                float(design.get("cuff_offset_mm", 0.0)),
                float(design.get("cuff_dx_mm", 0.0)),
                float(design.get("cuff_dy_mm", 0.0)),
            )
            design_offset_canon = (
                cuff_origin_D_pca - anchor_origin_pca
            )
            # Apply canonical → design-local to both nerve and
            # muscle.
            nerve_design_local = (
                (nerve_pts_canon - design_offset_canon) @ M_D
            )
            muscle_design_local = transform_muscle_pieces(
                muscle_canon, R=M_D,
                offset=-design_offset_canon @ M_D,
            )

            elec_R_ci = (
                float(design.get("R_ci_m") or geom.R_ci or 0.0)
            )
            elec_R_co = (
                float(design.get("R_co_m") or geom.R_co or 0.0)
            )
            elec_L_cuff = float(
                design.get("L_cuff_mm", state.L_cuff_mm),
            ) * 1e-3
            # F3.2-M3 — per-design scar / connective tissue shell.
            # Zero thickness ⇒ disabled (PLC behaves identically
            # to the legacy single-domain saline path).
            elec_use_scar = bool(design.get("use_scar", False))
            elec_scar_thickness_m = (
                float(design.get("scar_thickness_um", 0))
                * 1e-6
                if elec_use_scar else 0.0
            )
            if elec_R_ci <= 0.0 or elec_R_co <= 0.0:
                _on_line(
                    f"  ⚠ design '{eid}': non-positive cuff "
                    f"radii (Rci={elec_R_ci:.4e}, "
                    f"Rco={elec_R_co:.4e}) — refit first"
                )
                continue
            # Each design's outputs live in <out>/designs/<eid>/.
            design_out = _design_dir(out_dir, eid)
            design_out.mkdir(parents=True, exist_ok=True)

            # V1 — µCT bundle: per-fascicle inner surfaces. Apply
            # the SAME raw → canonical → design-local transform
            # used for nerve_pts_canon above. Without that the
            # fascicle verts would land in raw frame while the
            # outer nerve hull is in design-local, and TetGen
            # would either error or carve nonsense regions.
            #
            # For bundles, `geom.nerve["pts_raw"]` /
            # `boundary_raw` are now a COMBINED epi+fascicle
            # buffer (so the workspace viewport shows the
            # whole bundle, not just the outer epi shell).
            # That combined buffer would double-count the
            # fascicles if we fed it to the PLC, so we
            # explicitly pull the epi-only mesh from
            # `bundle["epi"]` here and re-derive its
            # design-local transform from scratch.
            inner_surfaces_design_local: (
                list[tuple[np.ndarray, np.ndarray]]
            ) = []
            _bundle = (
                geom.nerve.get("bundle")
                if isinstance(geom.nerve, dict)
                else None
            )
            plc_nerve_pts_local = nerve_design_local
            plc_nerve_faces = geom.nerve["boundary_raw"]
            if _bundle and _bundle.get("epi"):
                _epi = _bundle["epi"]
                _epi_v = np.asarray(
                    _epi["verts_m"], dtype=np.float64,
                )
                _epi_canon = (
                    (_epi_v - geom.centroid)
                    @ geom.R_global
                )
                plc_nerve_pts_local = (
                    _epi_canon - design_offset_canon
                ) @ M_D
                plc_nerve_faces = np.asarray(
                    _epi["faces"], dtype=np.int64,
                )
            if _bundle and _bundle.get("fascicles"):
                _cent = geom.centroid
                _R = geom.R_global
                for _fasc in _bundle["fascicles"]:
                    _fv = np.asarray(
                        _fasc["verts_m"], dtype=np.float64,
                    )
                    _fv_canon = (_fv - _cent) @ _R
                    _fv_local = (
                        _fv_canon - design_offset_canon
                    ) @ M_D
                    inner_surfaces_design_local.append((
                        _fv_local,
                        np.asarray(
                            _fasc["faces"], dtype=np.int64,
                        ),
                    ))
            # DEBUG (gated): dump the FULL (unclipped) design-local
            # fascicle surfaces so we can test dropping the clip.
            _dbgd = __import__("os").environ.get("GOLGI_PLC_DEBUG_DIR")
            if _dbgd and inner_surfaces_design_local:
                try:
                    import pyvista as _pvd
                    from pathlib import Path as _Pd
                    _Pd(_dbgd).mkdir(parents=True, exist_ok=True)
                    for _fi, (_fp, _ft) in enumerate(
                        inner_surfaces_design_local
                    ):
                        _ff = np.hstack([
                            np.full((len(_ft), 1), 3, np.int64), _ft,
                        ]).ravel()
                        _pvd.PolyData(_fp, _ff).save(
                            str(_Pd(_dbgd)
                                / f"FULLfasc_{_fi:02d}.stl")
                        )
                except Exception:                          # noqa: BLE001
                    pass

            def _heavy(
                plc_nerve_pts_local=plc_nerve_pts_local,
                plc_nerve_faces=plc_nerve_faces,
                muscle_design_local=muscle_design_local,
                elec_R_ci=elec_R_ci,
                elec_R_co=elec_R_co,
                elec_L_cuff=elec_L_cuff,
                design_out=design_out,
                elec_scar_thickness_m=elec_scar_thickness_m,
                inner_surfaces=(
                    inner_surfaces_design_local or None
                ),
            ):
                # NOTE: the nerve cross-section deform (area-preserving round)
                # is applied at IMPORT (histology-bundle / µCT reconstruction)
                # so geom.nerve — and hence rendering, cuff fit, fibers AND the
                # mesh — are all consistent. It is deliberately NOT applied
                # here at mesh-build time.
                # Cuff-free bare-nerve-in-bath path (intrafascicular /
                # LIFE electrodes): no cuff, no concentric shells, no
                # cuff-window clip — sidesteps the axisymmetric-cylinder
                # degeneracy that stalls the multi-domain cuff mesher.
                # Gated on `state.mesh_bare_bath`; helper presence is
                # guarded so GUI helper bags without it never break.
                _bare = (
                    bool(getattr(state, "mesh_bare_bath", False))
                    and hasattr(H, "assemble_bare_nerve_in_bath")
                )
                if _bare:
                    plc, seed_pos = H.assemble_bare_nerve_in_bath(
                        plc_nerve_pts_local,
                        plc_nerve_faces,
                        float(state.muscle_radial_pad_mm) * 1e-3,
                        float(state.muscle_axial_pad_mm) * 1e-3,
                        decim_target_tris=(
                            int(state.decim_target_k) * 1000
                        ),
                        on_line=_on_line,
                    )
                else:
                    plc, seed_pos = H.assemble_multi_domain_plc(
                    plc_nerve_pts_local,
                    plc_nerve_faces,
                    elec_L_cuff, elec_R_ci, elec_R_co,
                    float(state.muscle_radial_pad_mm) * 1e-3,
                    float(state.muscle_axial_pad_mm) * 1e-3,
                    muscle_dx_m=(
                        float(state.muscle_dx_mm) * 1e-3
                    ),
                    muscle_dy_m=(
                        float(state.muscle_dy_mm) * 1e-3
                    ),
                    muscle_dz_m=(
                        float(state.muscle_dz_mm) * 1e-3
                    ),
                    decim_target_tris=(
                        int(state.decim_target_k) * 1000
                    ),
                    use_epi=bool(state.use_epi),
                    epi_thickness_m=(
                        float(state.epi_thickness_um) * 1e-6
                    ),
                    scar_thickness_m=elec_scar_thickness_m,
                    on_line=_on_line,
                    muscle_pieces=muscle_design_local,
                    inner_surfaces=inner_surfaces,
                    debug_dir=(__import__("os").environ.get(
                        "GOLGI_PLC_DEBUG_DIR") or None),
                )
                plc_path = design_out / "current_plc.vtp"
                plc.save(str(plc_path))
                _on_line(
                    f"  PLC: {plc.n_points:,} pts, "
                    f"{plc.n_faces:,} tris"
                )

                # Volume targets per region (regular tet of
                # edge lc): V_reg = lc**3 / (6√2).
                def _vol(lc_um):
                    return (
                        (float(lc_um) * 1e-6) ** 3
                        / (6 * np.sqrt(2))
                    )
                # One region seed per region PRESENT in seed_pos. The
                # full cuff path supplies endo/saline/silicone/muscle;
                # the cuff-free bare-bath path (LIFE) supplies only
                # endo + one bath region — so skip absent keys instead
                # of KeyError-ing or seeding a non-existent region.
                _seed_spec = [
                    ("endo", 1, state.lc_endo_um),
                    ("saline", 2, state.lc_saline_um),
                    ("silicone", 3, state.lc_silicone_um),
                    ("muscle", 4, state.lc_muscle_um),
                ]
                seeds = [
                    [tag, seed_pos[key], _vol(lc)]
                    for (key, tag, lc) in _seed_spec
                    if key in seed_pos
                ]
                # V1 — µCT-bundle: additional per-fascicle
                # endoneurium seeds. All currently share tag 1 +
                # σ_endo + lc_endo, matching the "go easy on
                # conductivity" decision (per-fascicle σ comes
                # later). Each seed becomes its own [tag, pt,
                # vol] entry so TetGen carves the multi-fascicle
                # structure correctly instead of merging all
                # fascicles into one large tet region.
                for extra in seed_pos.get("endo_extra", []):
                    seeds.append(
                        [1, extra, _vol(state.lc_endo_um)],
                    )
                if "epi" in seed_pos:
                    seeds.insert(
                        1,
                        [5, seed_pos["epi"],
                         _vol(state.lc_epi_um)],
                    )
                # F3.2-M3 — scar region tag 7 (per-design). Only
                # added when the design enabled `use_scar` AND
                # the PLC built a scar cylinder + caps + seed.
                if "scar" in seed_pos:
                    seeds.append(
                        [7, seed_pos["scar"],
                         _vol(state.lc_scar_um)],
                    )
                # ---- mesher selection ----
                # gmsh OCC (conformal, robust, quality) is the default for
                # PRISMATIC / extruded-slice nerves — the regime where its
                # slice-and-extrude is exact. For a true-3D nerve (varying
                # cross-section) or if gmsh errors, fall back to PLC+TetGen so
                # 3-D shapes aren't flattened. Toggle: state.use_gmsh_mesher.
                # Env override GOLGI_MESHER={gmsh,tetgen} still wins.
                import os as _os
                _env = str(_os.environ.get("GOLGI_MESHER", "")).strip().lower()
                _want_gmsh = (_env == "gmsh") or (
                    _env != "tetgen"
                    and bool(getattr(state, "use_gmsh_mesher", True))
                )
                # The gmsh path rebuilds the cuff (R_ci/R_co/muscle)
                # parametrically and ignores the assembled PLC — so a
                # cuff-free bare-bath PLC must go through PLC+TetGen,
                # which meshes the surfaces we actually built.
                if _bare:
                    _want_gmsh = False
                _prismatic = _is_prismatic(plc_nerve_pts_local)
                nodes = elems = tags = None
                if _want_gmsh and (_prismatic or _env == "gmsh"):
                    try:
                        from golgi.compute.gmsh_mesher import mesh_nerve_cuff
                        _on_line("  [mesher] gmsh OCC (conformal multi-domain)")
                        nodes, elems, tags = mesh_nerve_cuff(
                            plc_nerve_pts_local, plc_nerve_faces,
                            L_cuff_m=elec_L_cuff, R_ci_m=elec_R_ci,
                            R_co_m=elec_R_co,
                            muscle_radial_pad_m=(
                                float(state.muscle_radial_pad_mm) * 1e-3),
                            muscle_axial_pad_m=(
                                float(state.muscle_axial_pad_mm) * 1e-3),
                            lc_fine_m=min(
                                float(state.lc_endo_um),
                                float(state.lc_saline_um),
                                float(state.lc_silicone_um)) * 1e-6,
                            lc_coarse_m=float(state.lc_muscle_um) * 1e-6,
                            fascicle_surfaces=inner_surfaces,
                            use_epi=bool(state.use_epi),
                            epi_thickness_m=(
                                float(state.epi_thickness_um) * 1e-6),
                            scar_thickness_m=elec_scar_thickness_m,
                            on_line=_on_line,
                        )
                        tags = np.asarray(tags, dtype=int)
                    except Exception as _ex:               # noqa: BLE001
                        _on_line(
                            f"  [mesher] gmsh failed "
                            f"({type(_ex).__name__}: {_ex}); "
                            f"falling back to TetGen")
                        nodes = elems = tags = None
                elif _want_gmsh and not _prismatic:
                    _on_line(
                        "  [mesher] non-prismatic nerve → TetGen "
                        "(gmsh slice-and-extrude assumes an extruded section)")
                if nodes is None:
                    out_npz = design_out / "current_tetgen.npz"
                    import os as _os
                    payload = TetGenPayload(
                        plc_path=plc_path, out_npz=out_npz,
                        switches=_os.environ.get("GOLGI_TETGEN_SWITCHES", "pzAa"),
                        seeds=seeds,
                        epsilon=float(_os.environ.get("GOLGI_TETGEN_EPSILON", "1.0e-6")),
                    )
                    nodes, elems, attribs = run_tetgen_subprocess(
                        plc_path, payload, _on_line,
                        proc_sink=ctx.register_subprocess,
                        payload_dir=design_out,
                    )
                    tags = (
                        attribs[:, 0] if attribs.ndim == 2
                        else attribs
                    ).astype(int)
                msh_path = design_out / "nerve.msh"
                H.write_msh22(msh_path, nodes, elems, tags)
                _on_line("  computing tet quality …")
                nodes_arr = np.asarray(nodes, dtype=np.float64)
                elems_arr = np.asarray(elems, dtype=np.int64)
                tags_arr = np.asarray(tags, dtype=np.int32)
                q_tet = H.tet_shape_quality(nodes_arr, elems_arr)
                stats_html = H.compute_mesh_stats_html(
                    nodes_arr, elems_arr, tags_arr, q_tet,
                    defaults_by_tag=H.defaults_by_tag,
                )
                hist_fig = H.build_quality_histogram_figure(
                    q_tet,
                    x_label=(
                        "tet quality (6√2·V / max_edge³)"
                    ),
                    y_label="# tetrahedra",
                )
                _on_line(
                    f"  q_tet min={q_tet.min():.3f} "
                    f"median={np.median(q_tet):.3f} "
                    f"mean={q_tet.mean():.3f}"
                )
                _on_line(
                    "  extracting per-region surfaces "
                    "(vtkGeometryFilter)…"
                )
                region_surfaces = H.extract_region_surfaces(
                    nodes_arr, elems_arr, tags_arr, q_tet,
                    on_line=_on_line,
                )
                _on_line(
                    f"  {len(region_surfaces)} region "
                    f"surface(s) ready"
                )
                return {
                    "msh_path": msh_path,
                    "nodes": nodes_arr,
                    "elems": elems_arr,
                    "tags": tags_arr,
                    "q_tet": q_tet,
                    "stats_html": stats_html,
                    "hist_fig": hist_fig,
                    "region_surfaces": region_surfaces,
                    "R_ci": elec_R_ci,
                    "R_co": elec_R_co,
                }

            built = await loop.run_in_executor(None, _heavy)
            _on_line(f"  ✓ wrote {built['msh_path']}")
            # F3.2 fix: write this design's nerve-surface sample
            # points (tag 1 region surface) to disk NOW, in this
            # DESIGN's cuff-local frame (same frame solve_nerve.py
            # samples Vₑ in). Doing this here — once per design at
            # mesh-build time — guarantees the count matches that
            # design's own nerve.msh exactly, even when the FEM
            # solve runs on a config whose parent isn't the
            # currently-focused design.
            #
            # Coordinates: `extract_region_surfaces` returns mm;
            # solve_nerve.py expects metres → divide by 1000.
            # Frame: pre-rotation = design's cuff-local (the on-
            # disk nerve.msh frame).
            _rs_mesh = built.get("region_surfaces") or {}
            if 1 in _rs_mesh:
                _surf_mm = np.asarray(
                    _rs_mesh[1].points, dtype=np.float64,
                )
                np.savez(
                    design_out / "nerve_surface_pts.npz",
                    pts=_surf_mm / 1000.0,
                )
            # F3.2 fix: the on-disk mesh is in THIS DESIGN's cuff-
            # local frame (cuff axis-aligned at origin) — that's
            # what solve_nerve.py expects, and it's the only
            # configuration TetGen handles reliably across real-
            # world branched-VN inputs. The viewport renders in
            # the SHARED PCA-translated frame (anchor at origin),
            # so we rotate the freshly-built mesh nodes + per-
            # region surfaces back to that frame BEFORE handing
            # them to geom:
            #   pts_viewport = pts_design_local @ M_D.T + design_offset_canon
            # where design_offset_canon = cuff_origin_D_pca -
            # anchor_origin_pca (already computed above). For
            # the anchor design this collapses to a pure
            # rotation (no translation, since offset = 0). The
            # on-disk nerve.msh is untouched.
            _M_D_T = np.asarray(M_D, dtype=np.float64).T
            _off_canon_m = np.asarray(
                design_offset_canon, dtype=np.float64,
            )
            # F3.2-M2.1f fix — region surfaces come out of
            # `extract_region_surfaces_mm` in MILLIMETRES,
            # but `built["nodes"]` is in METRES (raw TetGen
            # output). `_off_canon_m` is the cuff origin in
            # PCA frame, in METRES. Applying the same metres-
            # scale translation to mm-scale points was a silent
            # 1000× scale error — rotation worked but the
            # translation was effectively zero, so region
            # surfaces ended up at the design-local origin
            # instead of the design's PCA cuff origin. Long-
            # standing latent bug, only surfaces visibly with
            # designs at non-zero cuff offsets.
            _off_canon_mm = _off_canon_m * 1000.0

            def _to_viewport_m(pts: np.ndarray) -> np.ndarray:
                return pts @ _M_D_T + _off_canon_m

            def _to_viewport_mm(pts: np.ndarray) -> np.ndarray:
                return pts @ _M_D_T + _off_canon_mm

            built["nodes"] = _to_viewport_m(built["nodes"])
            _rs_view = {}
            for _tag, _poly in _rs_mesh.items():
                _pts = np.asarray(
                    _poly.points, dtype=np.float64,
                )
                _new = _poly.copy(deep=True)
                _new.points = _to_viewport_mm(_pts)
                _rs_view[_tag] = _new
            built["region_surfaces"] = _rs_view
            # F3.2-M1: stash THIS design's mesh in geom.designs_meshes
            # so the scene pipeline can render every design's
            # geometry simultaneously (each as its own
            # `region_<eid>_<tag>` actor). The single-slot fields
            # (geom.mesh_nodes / region_surfaces / …) are also
            # populated below for back-compat with legacy callers
            # (the FEM driver's bbox calc, project bundle export,
            # etc.) — those mirror the currently-selected design.
            if geom.designs_meshes is None:
                geom.designs_meshes = {}
            _rs_viz = H.build_viz_surfaces(
                built["region_surfaces"],
            )
            geom.designs_meshes[eid] = {
                "mesh_nodes": built["nodes"],
                "mesh_elems": built["elems"],
                "mesh_tags": built["tags"],
                "mesh_q": built["q_tet"],
                "region_surfaces": built["region_surfaces"],
                "region_surfaces_viz": _rs_viz,
                "msh_path": built["msh_path"],
                "R_ci": float(built["R_ci"]),
                "R_co": float(built["R_co"]),
                # F3.2-M2.1e — keep per-design stats + quality
                # histogram alongside the mesh data so the Mesh
                # drawer can render one panel per built design.
                "stats_html": built["stats_html"],
                "hist_fig": built["hist_fig"],
            }
            # M1.1: flip this design's has_mesh flag so the
            # legend exposes its tissue sub-rows. Mutate via a
            # new dict so Vue picks up the row-level reference
            # change.
            _ds_after = []
            for _d in (state.designs or []):
                if _d.get("eid") == eid:
                    _ds_after.append({**_d, "has_mesh": True})
                else:
                    _ds_after.append(_d)
            state.designs = _ds_after
            # Remember the build artefacts for the currently-
            # selected design so we can hydrate geom + state
            # after the loop.
            if (eid == selected_id
                    or not last_built):
                last_built = built

        if last_built:
            geom.msh_path = last_built["msh_path"]
            geom.mesh_nodes = last_built["nodes"]
            geom.mesh_elems = last_built["elems"]
            geom.mesh_tags = last_built["tags"]
            geom.mesh_q = last_built["q_tet"]
            geom.region_surfaces = last_built["region_surfaces"]
            geom.region_surfaces_viz = H.build_viz_surfaces(
                last_built["region_surfaces"],
            )
            # Don't overwrite geom.pts_cuff — `do_fit_cuff` set
            # it in the anchor's PCA-translated frame for the
            # cuff renderer + fiber overlay. Per-design pts_cuff
            # only existed temporarily as TetGen input and is
            # already discarded above.
            geom.R_ci = float(last_built["R_ci"])
            geom.R_co = float(last_built["R_co"])
            state.mesh_stats_html = last_built["stats_html"]
            # F3.2-M2.1e — combined multi-row histogram (one
            # subplot per built design) + per-design stats_html
            # list. The drawer stacks the HTML panels above a
            # single tall plotly figure so the user sees every
            # design's mesh quality at once.
            _hist_panels: list[dict] = []
            _stats_panels: list[dict] = []
            for _d in (state.designs or []):
                _eid_p = str(_d.get("eid", ""))
                if not _eid_p:
                    continue
                _md = (geom.designs_meshes or {}).get(_eid_p)
                if not _md or _md.get("mesh_q") is None:
                    continue
                _name = (
                    _d.get("name")
                    or f"Cuff {_eid_p}"
                )
                _hist_panels.append({
                    "name": _name,
                    "q": _md.get("mesh_q"),
                })
                _stats_panels.append({
                    "eid": _eid_p,
                    "name": _name,
                    "stats_html": _md.get("stats_html", ""),
                })
            from golgi.figures.mesh_stats import (
                _build_combined_quality_histogram_figure as _bc,
            )
            state.mesh_quality_hist_figure = _bc(_hist_panels)
            state.designs_mesh_panels = _stats_panels
            state.has_mesh = True
            # Post-build affordances:
            state.show_mesh_edges = True
            state.show_mesh_quality_color = True
            state.vis_1 = True
            state.vis_4 = True
            state.vis_5 = True
            # F3.2-M2.1a — flip off the pre-mesh previews now that
            # at least one design has a meshed nerve.msh. The
            # per-design Tissues > Epineurium / Muscle rows take
            # over visibility from here. Toggling these via the
            # legend re-mounts the preview (still a valid
            # operation if the user wants to compare).
            state.vis_epi_preview = False
            state.vis_muscle_preview = False
            geom._needs_camera_reset = True
            ctx.scene.request_render()
            # Autosave: at least one mesh built; capture a
            # thumbnail of the active design's render.
            ctx.autosave(stage="mesh", capture_thumb=True)
        else:
            state.mesh_log = (
                "⚠ no design produced a mesh — see log"
            )
    except Exception as ex:
        if ctx.was_cancelled():
            _on_line("⚠ mesh build cancelled by user")
            state.mesh_log = "Mesh build cancelled."
        else:
            _on_line(f"⚠ {type(ex).__name__}: {ex}")
    finally:
        ctx.clear_subprocess()
        state.busy = False
        state.busy_log = ""
        state.flush()
        ctx.safe_update()
        ctx.safe_reset_camera()
