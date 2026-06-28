# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Compute-pipeline trigger handlers — four thin async wrappers that
forward to the extracted pipeline drivers.

W1.8a (step 1/5 of the do_* handler extraction).

Each handler is decorated with @gated(<topic>) and immediately
delegates to the matching pipeline module's `run_*` coroutine.
The wrappers exist so:
1. The UI templates have a stable name to bind to (click=callable).
2. The auth gate runs before the long-running pipeline body.
3. The pipeline body itself stays out of build_app's closure (the
   real work is in golgi/pipeline/<topic>.py, migrated in steps
   4.3b–4.6).
"""
from __future__ import annotations

from typing import Callable


def register(
    *,
    gated: Callable,
    pipeline_ctx,
    pipeline_mesh,
    pipeline_fem,
    pipeline_fibers,
    pipeline_fiber_sim,
) -> dict[str, Callable]:
    """Wire the four compute triggers. All deps are eager-bound at
    register time — caller must invoke this AFTER pipeline_ctx is
    fully constructed."""

    @gated("mesh_build")
    async def do_build_mesh():
        """Mesh the designs the user picked in the Mesh-tab
        multi-select (`state.mesh_design_selection`). Empty
        selection falls back to the currently-active design so
        the legacy single-cuff workflow still works."""
        state = pipeline_ctx.state
        picked = list(state.mesh_design_selection or [])
        # Multi-select carries a list of eids; filter out any
        # stale entries that point to deleted designs so the
        # driver sees a clean set.
        known = {
            str(d.get("eid", ""))
            for d in (state.designs or [])
        }
        eids = [str(p) for p in picked if str(p) in known]
        print(
            "[MESH-PICK-DEBUG] "
            f"raw mesh_design_selection={picked!r} "
            f"designs eids={sorted(known)!r} "
            f"filtered eids={eids!r}",
            flush=True,
        )
        await pipeline_mesh.run_mesh_build(
            pipeline_ctx,
            design_eids=eids or None,
        )

    @gated("fem_solve")
    async def do_solve_fem():
        await pipeline_fem.run_fem_solve(pipeline_ctx)

    @gated("fiber_generate")
    async def do_generate_fibers():
        await pipeline_fibers.run_generate_fibers(pipeline_ctx)

    @gated("fiber_sim_run")
    async def do_run_fiber_sim():
        await pipeline_fiber_sim.run_fiber_sim(pipeline_ctx)

    return {
        "do_build_mesh": do_build_mesh,
        "do_solve_fem": do_solve_fem,
        "do_generate_fibers": do_generate_fibers,
        "do_run_fiber_sim": do_run_fiber_sim,
    }
