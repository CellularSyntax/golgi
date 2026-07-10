# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fiber-trajectory pipeline driver.

Owns the do_generate_fibers flow:
  1. Write nerve_only_surface.npz (raw-frame nerve surface
     mesh — solve_fiber_paths_nerve.py picks the nerve-only
     branch when this file exists, building its own clean
     single-domain tet mesh + Laplace BVP).
  2. Write nerve_paths_seed_config.json (seed count, max steps,
     cap-detection / clustering knobs).
  3. Spawn solve_fiber_paths_nerve.py via FiberGenRunner;
     heartbeat keeps the trame WS alive through RK4's quiet
     stretches.
  4. Parse nerve_paths_fibers.npz; classify each path into a
     branch via helpers.classify_fibers_by_branch.
  5. Stash paths_raw + branch_idx + n_branches on geom;
     refresh the per-branch UI metadata; autosave.

Output is in raw frame; do_solve_fem (4.4) calls
ensure_fibers_in_cuff_frame at solve time to migrate.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from golgi.jobs import CancelToken, JobRequest, LocalSubprocessRunner
from golgi.jobs.schemas import FiberSeedConfig
from ._throttle import FlushThrottle
from .context import PipelineContext

_SOLVE_FIBER_PATHS_PATH = (
    Path(__file__).resolve().parent.parent
    / "compute" / "solve_fiber_paths_nerve.py"
)
_SOLVE_FIBER_PATHS_BUNDLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "compute" / "solve_fiber_paths_bundle.py"
)


@dataclass
class FiberGenJobRequest(JobRequest):
    """The solver subprocess reads its inputs from files in
    FIBER_OUT_DIR (either `nerve_only_surface.npz` for the
    legacy whole-nerve path or `fascicle_<i>_surface.npz` +
    `fascicle_manifest.json` for the µCT-bundle per-fascicle
    path) plus `nerve_paths_seed_config.json`. It writes
    outputs to the same dir. No payload file; env var + cwd
    only."""
    fiber_out_dir: Path
    cwd: Path


class FiberGenRunner(LocalSubprocessRunner):
    """LocalSubprocessRunner specialised for the fiber-trajectory
    solver scripts. Same env contract for both the legacy
    `solve_fiber_paths_nerve.py` and the µCT-bundle
    `solve_fiber_paths_bundle.py`."""

    def _build_env(self, req: FiberGenJobRequest) -> dict:
        return {"FIBER_OUT_DIR": str(req.fiber_out_dir)}

    def _build_cwd(self, req: FiberGenJobRequest) -> Path:
        return req.cwd


# Fibre seeds are kept at least this far INSIDE the fascicle wall so none land
# on / outside the perineurium border (where the FEM Ve is discontinuous).
SEED_MARGIN_FRAC = 0.12          # fraction of the fascicle effective radius
SEED_MARGIN_MIN_M = 5.0e-6       # ... but at least this many metres (5 µm)


def _xsec_polygon(surf_local, z_mid):
    """Ordered 2D cross-section outline (local xy) of a fascicle prism at z_mid."""
    sl = surf_local.slice(normal="z", origin=(0.0, 0.0, z_mid))
    if sl.n_points == 0:
        return None
    st = sl.strip(join=True)
    lines, loops, i = st.lines, [], 0
    while i < len(lines):
        n = int(lines[i]); loops.append(lines[i + 1:i + 1 + n]); i += 1 + n
    if not loops:
        return None
    return np.asarray(st.points)[max(loops, key=len)][:, :2]


def _dist_to_poly(pts, poly):
    """Min distance from each point to the closed polygon boundary (vectorised)."""
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    ab2 = np.maximum((ab ** 2).sum(1), 1e-30)
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(2) / ab2[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((pts[:, None, :] - proj) ** 2).sum(2)).min(1)


def _compute_axial_paths_for_bundle(
    out_dir: Path,
    bundle: dict,
    seed_cfg,
    on_line,
    seed_end_key: str,
) -> None:
    """Straight-line axial fiber generation for extruded µCT
    bundles. Each fascicle is treated as a prism aligned with
    its surface-point PCA axis:

      1. Principal axis + orthonormal local frame via PCA on
         the fascicle surface vertices.
      2. Project the surface into the local frame → axial
         extent (z_lo, z_hi) and a xy bbox at the mid-plane.
      3. Pick a probe plane at z_mid; reject-sample n_seeds_-
         this_fasc xy points uniformly within the xy bbox,
         keeping only candidates whose 3D position is enclosed
         by the fascicle surface (`pv.select_enclosed_points`).
         This handles non-convex cross-sections correctly.
      4. Extrude each seed along the local +z axis at
         `seed_cfg.step_um` spacing → one straight-line path.
      5. Transform local frame → raw frame
            pt_raw = pt_local @ R.T + centroid
         so the output paths land in the same frame as the
         streamlines solver writes.

    Seed budget split: proportional to the cross-section
    convex-hull area of each fascicle at z_mid — matches the
    streamlines path's "more seeds in fatter fascicles" rule.

    Writes `nerve_paths_fibers.npz` + `nerve_paths_caps.json`
    with the same schema as the streamlines pipeline so the
    caller's load+finalize block consumes them unchanged.
    """
    import pyvista as pv
    from scipy.spatial import ConvexHull

    fascicles = bundle["fascicles"]
    n_total = int(seed_cfg.n_seeds)
    step_m = float(seed_cfg.step_um) * 1.0e-6
    on_line(
        f"# axial mode: {len(fascicles)} fascicle(s), "
        f"step={seed_cfg.step_um:.0f} µm, "
        f"requested seeds={n_total}"
    )

    fasc_meta = []
    for fi, fasc in enumerate(fascicles):
        pts_world = np.asarray(
            fasc["verts_m"], dtype=np.float64,
        )
        tris = np.asarray(fasc["faces"], dtype=np.int64)
        n_t = int(tris.shape[0])
        if pts_world.shape[0] < 4 or n_t < 4:
            fasc_meta.append(None)
            on_line(f"# fasc {fi}: degenerate surface; skip")
            continue
        centroid = pts_world.mean(axis=0)
        cov = np.cov((pts_world - centroid).T)
        _, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, -1]
        if axis[2] < 0:
            axis = -axis
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, axis))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        perp_x = ref - float(np.dot(ref, axis)) * axis
        perp_x = perp_x / np.linalg.norm(perp_x)
        perp_y = np.cross(axis, perp_x)
        # `R` maps local → world: pt_world = pt_local @ R.T
        # + centroid (since R is orthonormal, R.T == R⁻¹).
        R = np.column_stack([perp_x, perp_y, axis])
        local = (pts_world - centroid) @ R
        z_lo = float(local[:, 2].min())
        z_hi = float(local[:, 2].max())
        z_mid = 0.5 * (z_lo + z_hi)
        # Cross-section convex-hull area from a ±5% z-band
        # around z_mid — cheap and stable for tubular shapes.
        z_band_half = 0.05 * max(z_hi - z_lo, 1.0e-9)
        band_mask = np.abs(local[:, 2] - z_mid) < z_band_half
        if int(band_mask.sum()) >= 3:
            xy_band = local[band_mask, :2]
            try:
                hull = ConvexHull(xy_band)
                area = float(hull.volume)  # 2D: .volume = area
                xy_lo = xy_band.min(axis=0)
                xy_hi = xy_band.max(axis=0)
            except Exception:                       # noqa: BLE001
                area = 0.0
                xy_lo = local[:, :2].min(axis=0)
                xy_hi = local[:, :2].max(axis=0)
        else:
            area = 0.0
            xy_lo = local[:, :2].min(axis=0)
            xy_hi = local[:, :2].max(axis=0)
        faces_flat = np.empty(n_t * 4, dtype=np.int64)
        faces_flat[0::4] = 3
        faces_flat[1::4] = tris[:, 0]
        faces_flat[2::4] = tris[:, 1]
        faces_flat[3::4] = tris[:, 2]
        surf_local = pv.PolyData(local, faces_flat)
        fasc_meta.append({
            "centroid": centroid,
            "R": R,
            "z_lo": z_lo,
            "z_hi": z_hi,
            "z_mid": z_mid,
            "area": area,
            "xy_lo": xy_lo,
            "xy_hi": xy_hi,
            "surf_local": surf_local,
        })
        on_line(
            f"# fasc {fi}: axis=("
            f"{axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f}"
            f"), length={(z_hi - z_lo) * 1e3:.2f} mm, "
            f"x-section area={area * 1e6:.3f} mm²"
        )

    areas = np.asarray(
        [m["area"] if m is not None else 0.0 for m in fasc_meta],
        dtype=np.float64,
    )
    total = float(areas.sum())
    if total > 0:
        weights = areas / total
        n_per = [
            max(1, int(round(n_total * float(w))))
            if (m is not None and m["area"] > 0)
            else 0
            for m, w in zip(fasc_meta, weights)
        ]
    else:
        per = max(1, n_total // max(1, len(fasc_meta)))
        n_per = [
            per if m is not None else 0
            for m in fasc_meta
        ]
    on_line(
        f"# proportional seed split: {n_per} "
        f"(total={sum(n_per)})"
    )

    all_paths: list[np.ndarray] = []
    all_branch_idx: list[int] = []
    all_drain_centroids: list[list[float]] = []
    rng = np.random.default_rng(42)

    for fi, (m, n_seeds_this) in enumerate(zip(fasc_meta, n_per)):
        if m is None or n_seeds_this < 1:
            continue
        # Robust STRICT-INSIDE sampling: keep seeds at least `margin` inside the
        # fascicle cross-section outline.  The old select_enclosed_points test
        # had no margin (so ~20-30 % of seeds sat within ~10 µm of the wall) and
        # mis-classified ~3-5 % of points as inside on the prism's open caps (so
        # they landed outside).  A 2D point-in-polygon + distance-to-wall test on
        # the actual cross-section outline fixes both.
        from matplotlib.path import Path as _MplPath
        poly = _xsec_polygon(m["surf_local"], m["z_mid"])
        if poly is None or len(poly) < 3:
            on_line(f"# fasc {fi}: no cross-section outline; skip")
            continue
        path = _MplPath(poly)
        r_eff = float(np.sqrt(max(m["area"], 1.0e-12) / np.pi))
        margin = max(SEED_MARGIN_MIN_M, SEED_MARGIN_FRAC * r_eff)
        plo, phi = poly.min(axis=0), poly.max(axis=0)
        seeds_xy: list[np.ndarray] = []
        for _attempt in range(3):                     # relax margin if too small
            for _trial in range(40):
                batch = rng.uniform(plo, phi, size=(max(128, n_seeds_this * 6), 2))
                ins = path.contains_points(batch)
                if ins.any():
                    cand = batch[ins]
                    for xy in cand[_dist_to_poly(cand, poly) > margin]:
                        seeds_xy.append(xy.copy())
                        if len(seeds_xy) >= n_seeds_this:
                            break
                if len(seeds_xy) >= n_seeds_this:
                    break
            if seeds_xy:
                break
            margin *= 0.4                             # tiny fascicle: relax margin
        if not seeds_xy:                              # last resort: a central seed
            c = poly.mean(axis=0)
            if path.contains_point((float(c[0]), float(c[1]))):
                seeds_xy = [c]
        if not seeds_xy:
            on_line(f"# fasc {fi}: too small for any inside seed; skip")
            continue
        seeds_xy = seeds_xy[:n_seeds_this]
        on_line(
            f"# fasc {fi}: {len(seeds_xy)}/{n_seeds_this} seeds "
            f"(>= {margin * 1e6:.1f} µm inside wall)"
        )
        n_pts_axis = max(
            5,
            int(np.floor(
                (m["z_hi"] - m["z_lo"]) / step_m,
            )) + 1,
        )
        zs = np.linspace(m["z_lo"], m["z_hi"], n_pts_axis)
        for sxy in seeds_xy:
            local_path = np.column_stack([
                np.full(n_pts_axis, sxy[0]),
                np.full(n_pts_axis, sxy[1]),
                zs,
            ])
            world_path = local_path @ m["R"].T + m["centroid"]
            all_paths.append(world_path)
            all_branch_idx.append(fi)
        if seed_end_key == "low":
            drain_local = np.array([0.0, 0.0, m["z_hi"]])
        else:
            drain_local = np.array([0.0, 0.0, m["z_lo"]])
        drain_world = drain_local @ m["R"].T + m["centroid"]
        all_drain_centroids.append(drain_world.tolist())

    if not all_paths:
        raise RuntimeError(
            "axial seeding produced zero paths (no fascicle "
            "yielded valid seed candidates)",
        )

    flat = np.vstack(all_paths)
    lens = np.array(
        [len(p) for p in all_paths], dtype=np.int64,
    )
    branch = np.array(all_branch_idx, dtype=np.int64)
    np.savez(
        out_dir / "nerve_paths_fibers.npz",
        paths_flat=flat,
        path_lengths=lens,
        branch_idx=branch,
        step_m=np.float64(step_m),
        seed_end=np.array([seed_end_key]),
    )
    on_line(
        f"# wrote {len(all_paths)} paths "
        f"({flat.shape[0]:,} total pts) → "
        f"{(out_dir / 'nerve_paths_fibers.npz').name}"
    )
    caps_info = {
        "trunk_end": seed_end_key,
        "branched_end": (
            "high" if seed_end_key == "low" else "low"
        ),
        "n_branch_caps": len(all_drain_centroids),
        "branch_cap_centroids_m": all_drain_centroids,
    }
    (out_dir / "nerve_paths_caps.json").write_text(
        json.dumps(caps_info, indent=2),
     encoding="utf-8")


async def run_generate_fibers(ctx: PipelineContext) -> None:
    """Full fiber-generation driver. See module docstring."""
    state = ctx.state
    geom = ctx.geom
    H = ctx.helpers

    if not state.has_geometry or geom.nerve is None:
        state.fiber_log = "Load a nerve first."
        return
    # Reset failure / stats state before launching.
    state.fiber_failed = False
    state.fiber_stats_html = ""
    state.fiber_branch_summary = []
    state.fiber_log = ""
    state.busy_log = ""
    state.busy = True
    state.busy_msg = "Generating fibers"
    state.flush()

    loop = asyncio.get_event_loop()
    log_lines: list[str] = []
    out_dir = Path(H.active_project().out_dir)
    # Trailing-edge debounce on state.flush() to cap WS traffic
    # during fiber-paths RK4 progress bursts.
    throttle = FlushThrottle(loop=loop, state=state)

    def _on_line(line: str):
        # Mirror every line to the server terminal too — that
        # way if the trame server / browser crashes mid-run,
        # the user still has the full subprocess output in
        # their terminal session and can diagnose what blew up.
        line = ctx.stamp_user_line(line)
        print(f"[fibers] {line}", flush=True)
        # Keep the full log in memory for the eventual sidebar
        # display, but stream only the last ~14 lines to the
        # lightbox so the live tail fits in the box. Trim any
        # over-long line to a reasonable width.
        line = line[:220]
        log_lines.append(line)
        tail = "\n".join(log_lines[-14:])

        def _push():
            state.busy_log = tail
            throttle.tick()
        try:
            loop.call_soon_threadsafe(_push)
        except RuntimeError:
            pass

    def _surface_failure(msg: str) -> None:
        """Push the full log + an error banner to the sidebar
        so the user can read whatever the subprocess printed
        before exiting — instead of it disappearing with the
        lightbox."""
        tail = "\n".join(log_lines[-60:])
        state.fiber_log = tail
        state.fiber_failed = True
        state.fiber_status = msg

    seed_end_key = (
        "low" if state.fiber_seed_end.startswith("trunk") else "high"
    )
    # Cap-detection / clustering knobs from the Fiber-trajectories
    # drawer convert mm → m and % → fraction so the solver script
    # consumes SI-native values.
    seed_cfg = FiberSeedConfig(
        n_seeds=int(state.n_fibers),
        seed_end=seed_end_key,
        step_um=200.0,
        max_steps=int(state.fiber_max_steps),
        cluster_eps_m=float(state.fiber_cluster_eps_mm) * 1.0e-3,
        cap_band_frac=float(state.fiber_cap_band_pct) / 100.0,
        min_rel_size=float(state.fiber_min_rel_size_pct) / 100.0,
        axial_normal_thresh=float(state.fiber_axial_normal_thresh),
    )
    # Write the surface payload the script expects when its
    # nerve-only branch is taken. pts_cuff = pts_raw → the
    # script's internal Procrustes ends up as the identity, so
    # output paths come back in raw frame.
    #
    # For µCT bundles, `geom.nerve["pts_raw"]` /
    # `boundary_raw` are a COMBINED epi+fascicles buffer
    # (so the workspace viewport shows everything at once).
    # The per-fascicle solver builds ONE small tet mesh per
    # fascicle endoneurium (not one big mesh of the epi
    # shell), so we route the bundle path through the new
    # `solve_fiber_paths_bundle.py` subprocess and write one
    # cleaned surface npz per fascicle plus a manifest.
    # The whole-epi shell approach OOM-killed the solver on
    # sheep-scale bundles (~1.8 M tets in one Laplace solve);
    # per-fascicle keeps each solve ≤ ~250 k tets.
    _bundle = (
        geom.nerve.get("bundle")
        if isinstance(geom.nerve, dict)
        else None
    )
    _bundle_mode = bool(
        _bundle and _bundle.get("fascicles"),
    )
    # Method dispatch. "streamlines" is the legacy Laplace +
    # RK4 path; "axial" is the new in-process straight-line
    # extrusion that's only meaningful for µCT-bundle imports
    # (where the geometry IS straight prisms). On a bundle we
    # skip the file write + subprocess spawn entirely and
    # compute paths inline.
    _method = str(
        getattr(state, "fiber_method", "streamlines")
        or "streamlines",
    )
    _is_axial = _bundle_mode and _method == "axial"

    # Stale-output guard: a previous legacy run may have left
    # `nerve_only_surface.npz` or its fascicle-filter sidecar in
    # the project's out_dir, which would cause the bundle solver
    # to pick the wrong file or — worse — the LEGACY solver to
    # run when we want the bundle path. Clear them both so the
    # dispatch downstream sees a clean slate.
    for _stale in (
        "nerve_only_surface.npz",
        "nerve_only_fascicle_surfaces.npz",
        "fascicle_manifest.json",
    ):
        _p = out_dir / _stale
        if _p.exists():
            try:
                _p.unlink()
            except OSError:
                pass
    # Also drop any per-fascicle npzs from a prior run; the new
    # manifest below is the only authoritative listing.
    for _old in out_dir.glob("fascicle_*_surface.npz"):
        try:
            _old.unlink()
        except OSError:
            pass

    if _is_axial:
        # Axial mode: no file write needed — the in-process
        # `_compute_axial_paths_for_bundle` reads the bundle
        # surfaces directly from geom and writes
        # `nerve_paths_fibers.npz` itself.
        _on_line(
            f"# axial mode selected — skipping surface npz "
            f"write + subprocess spawn"
        )
    elif _bundle_mode:
        _fascicles = _bundle["fascicles"]
        _on_line(
            f"# µCT-bundle path: {len(_fascicles)} fascicle(s) "
            "→ per-fascicle surface write + manifest"
        )
        # No SI pre-pass for fascicles. Two reasons:
        # 1. Each fascicle is a small genus-0 tube straight
        #    out of marching cubes + the refine_mesh pipeline
        #    (Taubin + pymeshfix), which already runs the
        #    standard SI repair. TetGen historically swallows
        #    these without complaint.
        # 2. The aggressive `_preprocess_nerve_surface`
        #    (PyTMesh.clean(50)) we used on the whole-epi shell
        #    is a C extension that segfaults on certain mesh
        #    defects — and a segfault in the trame loop's
        #    worker thread kills the whole app, which is exactly
        #    what we just watched happen on fascicle 9. The
        #    per-fascicle solver already wraps each solve in
        #    try/except (skip-on-failure), so if a particular
        #    fascicle really is SI-broken we lose that ONE
        #    fascicle's paths, not the whole run.
        # The legacy fascicle-filter sidecar (`nerve_only_-
        # fascicle_surfaces.npz`) is also intentionally NOT
        # written in bundle mode — the bundle solver seeds
        # inside each fascicle by construction, so the post-hoc
        # filter is obsolete.
        _manifest_entries = []
        for _fi, _fasc in enumerate(_fascicles):
            _fv_m = np.asarray(
                _fasc["verts_m"], dtype=np.float64,
            )
            _ff = np.asarray(
                _fasc["faces"], dtype=np.int64,
            )
            _npz_name = f"fascicle_{_fi}_surface.npz"
            np.savez(
                out_dir / _npz_name,
                pts_raw=_fv_m,
                pts_cuff=_fv_m,    # identity → raw-frame output
                tris=_ff,
            )
            _manifest_entries.append({
                "idx": _fi,
                "npz": _npz_name,
                "n_verts": int(_fv_m.shape[0]),
                "n_tris": int(_ff.shape[0]),
            })
            _on_line(
                f"# fasc {_fi}: wrote {_npz_name} "
                f"({len(_fv_m):,} pts, {int(_ff.shape[0]):,} tris)"
            )

        (out_dir / "fascicle_manifest.json").write_text(
            json.dumps({
                "fascicles": _manifest_entries,
                # 200 µm matches the legacy lc_nerve default;
                # fascicles are small enough that this gives
                # ~30–80 k tets each.
                "lc_target": 2.0e-4,
                "bundle_id": _bundle.get("bundle_id", ""),
            }, indent=2),
         encoding="utf-8")
        _on_line(
            f"# fascicle_manifest.json written "
            f"({len(_manifest_entries)} fascicles)"
        )
    else:
        # Legacy STL / .nas path — one boundary surface, written
        # exactly as before.
        pts_raw = np.asarray(
            geom.nerve["pts_raw"], dtype=np.float64,
        )
        tris = np.asarray(
            geom.nerve["boundary_raw"], dtype=np.int64,
        )
        np.savez(
            out_dir / "nerve_only_surface.npz",
            pts_raw=pts_raw,
            pts_cuff=pts_raw,   # identity → raw-frame output
            tris=tris,
        )
    (out_dir / "nerve_paths_seed_config.json").write_text(
        json.dumps(seed_cfg.serialize(), indent=2),
     encoding="utf-8")

    # Subprocess plumbing is only needed for the streamlines
    # path. In axial mode the compute runs in-process so we
    # never instantiate a runner / token / request.
    runner = None
    tok = None
    req = None
    if not _is_axial:
        _solver_script = (
            _SOLVE_FIBER_PATHS_BUNDLE_PATH if _bundle_mode
            else _SOLVE_FIBER_PATHS_PATH
        )
        runner = FiberGenRunner(_solver_script)
        tok = CancelToken()
        _orig_arm = tok.arm

        def _arm_and_forward(proc):
            _orig_arm(proc)
            try:
                ctx.register_subprocess(proc)
            except Exception:
                pass
        tok.arm = _arm_and_forward  # type: ignore[method-assign]

        req = FiberGenJobRequest(
            fiber_out_dir=out_dir,
            cwd=H.script_cwd,
        )

    async def _heartbeat():
        """Inject a 'still running' line every ~8s during the
        quiet stretches of the subprocess (the RK4 integrator
        only prints progress every 500 steps, so the subprocess
        can go minutes without output). Keeps the trame
        WebSocket warm — browsers drop idle WS at ~30-60s,
        which is why the user was seeing the trame loader
        replace the app mid-run."""
        import time as _time
        t0 = _time.time()
        while True:
            await asyncio.sleep(8)
            elapsed = _time.time() - t0
            _on_line(f"# … still running ({elapsed:.0f}s)")

    try:
        if _is_axial:
            _on_line(
                "# axial extrusion: PCA + xy-sampling + "
                "straight-line extrude (no FEM, no subprocess)"
            )
            state.busy_msg = (
                "Generating fibers — axial extrusion"
            )
            state.flush()
            await loop.run_in_executor(
                None,
                lambda: _compute_axial_paths_for_bundle(
                    out_dir, _bundle, seed_cfg,
                    _on_line, seed_end_key,
                ),
            )
        else:
            _on_line("# launching solve_fiber_paths_nerve.py "
                      "(nerve-only path)")
            _on_line(f"# FIBER_OUT_DIR = {out_dir}")
            _on_line(f"# seed cfg: {seed_cfg}")
            hb = asyncio.create_task(_heartbeat())
            try:
                outputs = await loop.run_in_executor(
                    None, lambda: runner.run(req, _on_line, tok),
                )
            finally:
                hb.cancel()
                try:
                    await hb
                except asyncio.CancelledError:
                    pass
            rc = outputs.return_code
            if ctx.was_cancelled():
                _on_line(
                    "⚠ fiber generation cancelled by user",
                )
                state.fiber_status = (
                    "Fiber generation cancelled."
                )
                _surface_failure(
                    "⚠ fiber generation cancelled — see log",
                )
                return
            if rc != 0:
                _on_line(
                    f"⚠ subprocess exited with code {rc}",
                )
                _surface_failure(
                    f"⚠ fiber generation failed (rc={rc}) "
                    "— see log",
                )
                return
        npz_path = out_dir / "nerve_paths_fibers.npz"
        if not npz_path.exists():
            _on_line(f"⚠ expected {npz_path.name} not found")
            _surface_failure(
                "⚠ fiber generation produced no output — see log",
            )
            return
        d = np.load(npz_path, allow_pickle=True)
        flat = np.asarray(d["paths_flat"])
        lens = np.asarray(d["path_lengths"])
        paths_raw: list[np.ndarray] = []
        off = 0
        for L in lens:
            paths_raw.append(flat[off:off + int(L)].copy())
            off += int(L)
        _on_line(f"# {len(paths_raw)} fiber paths, "
                  f"{flat.shape[0]:,} total pts")
        # F3.2-M3: branch classification is gated on the import-
        # stepper "Auto-detect branches" checkbox. When off, every
        # path goes into branch 0 — the legend shows a single
        # bundle and the per-branch toggles collapse to one row.
        # (Cap detection for seed placement still runs inside the
        # subprocess; we just don't post-process the result into
        # branches.)
        caps_json = out_dir / "nerve_paths_caps.json"
        # µCT-bundle path: the per-fascicle solver writes
        # `branch_idx` directly into the npz, with each entry
        # equal to the fascicle index that path was seeded in.
        # That assignment is authoritative — skip the cap-kNN
        # reclassification (which would have to round-trip
        # through endpoint geometry and lose information when
        # two fascicles happen to share a drain region).
        _has_branch_idx_from_solver = bool(
            _bundle_mode and "branch_idx" in d.files
        )
        if _has_branch_idx_from_solver:
            branch_idx = np.asarray(
                d["branch_idx"], dtype=np.int64,
            )
            n_branches = int(branch_idx.max()) + 1 if (
                branch_idx.size > 0
            ) else 0
            _on_line(
                f"# bundle: using per-path branch_idx from "
                f"solver ({n_branches} fascicle(s))"
            )
        elif bool(getattr(state, "fiber_auto_detect_branches", True)):
            branch_idx, n_branches = (
                H.classify_fibers_by_branch(
                    paths_raw, caps_json, seed_end_key,
                )
            )
            _on_line(
                f"# classified into {n_branches} branch(es)"
            )
        else:
            branch_idx = np.zeros(
                len(paths_raw), dtype=np.int64,
            )
            n_branches = 1
            _on_line(
                "# auto-detect branches OFF — single bundle"
            )
        geom.fiber_paths_raw = paths_raw
        geom.fiber_branch_idx = branch_idx
        geom.fiber_n_branches = n_branches
        # Freshly-generated fibers are in raw frame
        # (solve_fiber_paths_nerve.py with pts_cuff=pts_raw
        # has identity Procrustes). They'll get rewritten in
        # cuff frame by ensure_fibers_in_cuff_frame just
        # before the first FEM solve.
        geom.fibers_in_cuff_frame = bool(
            "frame_is_cuff" in d.files
            and int(d["frame_is_cuff"]) == 1
        )
        state.fiber_n_branches = n_branches
        state.has_fibers = True
        H.refresh_fiber_sel_items()
        H.refresh_pop_branches_meta()
        ctx.scene.request_render()
        state.fiber_branch_summary = (
            H.compute_fiber_branch_summary(
                paths_raw, caps_json, seed_end_key,
                branch_labels={
                    _i: str(
                        state[f"fiber_branch_name_{_i}"] or ""
                    )
                    for _i in range(H.MAX_FIBER_BRANCHES)
                },
            )
        )
        state.fiber_stats_html = ""
        state.fiber_status = (
            f"✓ {len(paths_raw)} trajectories generated"
        )
        # Autosave: fibers on disk; thumbnail reflects the new
        # coloured polylines mounted over the nerve.
        ctx.autosave(stage="fibers", capture_thumb=True)
    except Exception as ex:
        if ctx.was_cancelled():
            _on_line("⚠ fiber generation cancelled by user")
            state.fiber_status = "Fiber generation cancelled."
        else:
            _on_line(f"⚠ {type(ex).__name__}: {ex}")
            _surface_failure(
                f"⚠ {type(ex).__name__}: {ex} — see log",
            )
    finally:
        ctx.clear_subprocess()
        state.busy = False
        state.busy_log = ""
        state.flush()
        ctx.safe_update()
