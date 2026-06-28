# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cuff + electrode reactivity watchers.

Split into reactivity groups:
  POSITION_KEYS  — pure translation. Re-render the cuff moved
                   to new (offset, dx, dy) but keep R_local +
                   R_ci from cache — the cuff slides through the
                   nerve without resizing or re-orienting.
  GEOMETRY_KEYS  — cuff shape / orientation / autosize inputs.
                   Force a full re-fit so R_local + R_ci update;
                   subsequent position adjustments translate the
                   new rigid cuff.
  ELECTRODE_KEYS — electrode-mode + per-mode contact params.
                   Re-fit at the current position.
  RENDER_TOGGLE_KEYS — visibility toggles; no geometric refit.

Plus single-key watchers for contact_polarities, selected_design_id,
remove_design_request, refit_design_request,
rename_design_request, and the _VIS_MIRROR_KEYS visibility-
mirror group.
"""
from __future__ import annotations

import asyncio
from typing import Callable


def register(
    state,
    *,
    geom,
    elec_sync_guard: dict,
    default_electrode: dict,
    save_selected_to_designs: Callable,
    do_fit_cuff: Callable,
    apply_electrode_visibility: Callable,
    safe_update: Callable,
    load_design_to_selected: Callable,
    do_remove_design: Callable,
    do_delete_mesh: Callable | None = None,
    find_design: Callable,
    find_cuff_origin_pca: Callable,
    local_pca_refine: Callable,
    autosize_R_ci: Callable,
    compute_polarity_sums: Callable | None = None,
    refit_design_geometry: Callable | None = None,
) -> None:
    POSITION_KEYS = (
        "cuff_offset_mm", "cuff_dx_mm", "cuff_dy_mm",
        # F3.2a: intrinsic Euler tilt + twist of the cuff.
        # Doesn't change R_local_elec or R_ci, so the translate-
        # only refit path is enough — but the cuff transform has
        # to be rebuilt so the rotated cuff (and contact patches)
        # actually move in the viewport.
        "cuff_rot_x_deg", "cuff_rot_y_deg", "cuff_rot_z_deg",
    )
    GEOMETRY_KEYS = (
        "cuff_anchor", "local_pca_radius_mm",
        "L_cuff_mm", "cuff_clearance_mm", "cuff_wall_mm",
    )
    ELECTRODE_KEYS = tuple(default_electrode.keys())
    # `show_saline` is a pure rendering toggle — handled by its
    # own watcher so it doesn't force a full refit. F3.2-M3 —
    # `use_scar` and `scar_thickness_um` join the same family:
    # toggling / sliding them changes only the per-design pre-
    # mesh preview cylinder, no R_local_elec / R_ci recompute.
    RENDER_TOGGLE_KEYS = (
        "show_saline", "use_scar", "scar_thickness_um",
    )
    _VIS_MIRROR_KEYS = (
        "vis_master", "vis_silicone",
        "vis_saline", "vis_contacts",
        # F3.2-M3 — per-design scar legend toggle.
        "vis_scar",
    )

    # W1.7b: show_cuff drawer-open watcher. When the cuff drawer
    # opens, immediately fit the cuff so the user sees the silicone
    # shell + contacts the moment they click the menu. When it
    # closes, re-render so the selection halo (gated on show_cuff)
    # is stripped from the viewport. Both cases route through a
    # translate-only fit — no geometry recompute needed.
    @state.change("show_cuff")
    def _on_show_cuff_change(**_kwargs):
        if geom.nerve is not None and getattr(
            geom, "_fit_locked", False,
        ):
            asyncio.create_task(do_fit_cuff(refit=False))

    @state.change(*POSITION_KEYS)
    def _on_cuff_position_change(**_kwargs):
        if elec_sync_guard["loading"]:
            return
        save_selected_to_designs()
        # Translate-only update (refit=False).
        if geom.nerve is not None:
            asyncio.create_task(do_fit_cuff(refit=False))

    @state.change(*GEOMETRY_KEYS)
    def _on_cuff_geometry_change(**_kwargs):
        if elec_sync_guard["loading"]:
            return
        save_selected_to_designs()
        # Full re-fit: rebuild R_local + R_ci at the current
        # cuff position, then cache them.
        if geom.nerve is not None:
            asyncio.create_task(do_fit_cuff(refit=True))

    @state.change(*ELECTRODE_KEYS)
    def _on_electrode_change(**_kwargs):
        if elec_sync_guard["loading"]:
            return
        save_selected_to_designs()
        # Electrode-mode + per-mode contact params: position and
        # cuff size don't change, so this is a translate-only
        # update (do_fit_cuff still rebuilds the contact patches
        # each call).
        if geom.nerve is not None:
            asyncio.create_task(do_fit_cuff(refit=False))

    @state.change(*RENDER_TOGGLE_KEYS)
    def _on_render_toggle_change(**_kwargs):
        if elec_sync_guard["loading"]:
            return
        save_selected_to_designs()
        # Visibility-only toggles — re-run render path but no
        # geometric refit.
        if geom.nerve is not None and geom._fit_locked:
            asyncio.create_task(do_fit_cuff(refit=False))

    # F3.2-M3 — when the user FIRST enables scar (False→True),
    # auto-seed `scar_thickness_um` to ~60% of the current cuff
    # clearance gap so the default scar layer is a "reasonable
    # thick" fraction of the saline pocket (leaves ~40% saline).
    # The user can still adjust the slider afterwards. Toggling
    # scar off and on again resets to this auto-default — that's
    # by design so the auto-init behaviour stays predictable.
    _scar_init_guard = {"loading": False}

    @state.change("use_scar")
    def _on_use_scar_change(**_kwargs):
        if elec_sync_guard["loading"]:
            return
        if _scar_init_guard["loading"]:
            return
        if not bool(state.use_scar):
            return
        try:
            clearance_um = (
                float(state.cuff_clearance_mm or 0.2) * 1000.0
            )
        except (TypeError, ValueError):
            clearance_um = 200.0
        desired = max(50.0, 0.6 * clearance_um)
        _scar_init_guard["loading"] = True
        try:
            state.scar_thickness_um = int(round(desired))
        finally:
            _scar_init_guard["loading"] = False

    @state.change("contact_polarities", "contact_current_fractions")
    def _on_contact_polarities_change(**_kwargs):
        # Polarity / current-fraction edits are visualization-only
        # (no geometry recompute) but they DO need to persist to
        # the selected electrode's dict + repaint the contact
        # tints. M1 also recomputes the per-polarity sum-check
        # chip data so the drawer updates in real time.
        if elec_sync_guard["loading"]:
            return
        save_selected_to_designs()
        # Recompute the polarity-sum chips off the LATEST state
        # so the drawer's v-for sees the new values immediately.
        # `compute_polarity_sums` is injected from build_app so
        # this watcher stays free of an app.py import.
        if compute_polarity_sums is not None:
            try:
                state.contact_polarity_sums = (
                    compute_polarity_sums(
                        list(state.contact_polarities or []),
                        list(state.contact_current_fractions or []),
                    )
                )
            except Exception:                            # noqa: BLE001
                pass
        if geom.nerve is not None and geom._fit_locked:
            asyncio.create_task(do_fit_cuff(refit=False))

    @state.change(*_VIS_MIRROR_KEYS)
    def _on_electrode_visibility_change(**_kwargs):
        # Pure visibility flips — no geometry rebuild. Save the
        # mirror state to the electrode dict, walk actors and
        # apply the flags, then poke a view-update.
        if elec_sync_guard["loading"]:
            return
        save_selected_to_designs()
        apply_electrode_visibility()
        safe_update()

    # Selection bridge — Vue clicks set state.selected_design_id,
    # this watcher saves the previous and loads the new.
    @state.change("selected_design_id")
    def _on_selected_electrode_change(**_kwargs):
        # Don't re-save on the load itself (selection guard wraps
        # the load call separately in do_select_design).
        if elec_sync_guard["loading"]:
            return
        eid = str(state.selected_design_id or "")
        if not eid:
            return
        load_design_to_selected(eid)
        # Selection drives the 3D halo — fire a translate-only
        # re-render so the red glow follows whichever cuff the
        # user just clicked.
        if geom.nerve is not None and geom._fit_locked:
            asyncio.create_task(do_fit_cuff(refit=False))

    # Remove bridge — Vue click on the ✕ icon writes the target
    # eid into remove_design_request, this watcher dispatches
    # do_remove_design + clears.
    @state.change("remove_design_request")
    def _on_remove_electrode_request(**_kwargs):
        eid = str(state.remove_design_request or "").strip()
        if not eid:
            return
        state.remove_design_request = ""
        do_remove_design(eid)

    # Mesh-delete bridge — confirm_delete_mesh dialog writes the
    # target eid into `remove_mesh_request` on confirm; this
    # watcher routes it to `do_delete_mesh` (injected from
    # app.py) and clears the bridge var so the next click fires
    # fresh.
    @state.change("remove_mesh_request")
    def _on_remove_mesh_request(**_kwargs):
        eid = str(
            getattr(state, "remove_mesh_request", "") or "",
        ).strip()
        if not eid:
            return
        state.remove_mesh_request = ""
        if do_delete_mesh is None:
            return
        try:
            do_delete_mesh(eid)
        except Exception as _ex:                          # noqa: BLE001
            print(
                f"[mesh-delete] do_delete_mesh({eid!r}) "
                f"raised {type(_ex).__name__}: {_ex}",
                flush=True,
            )

    # Refit bridge — per-row Refit click writes the target eid
    # here. The watcher recomputes JUST that electrode's local
    # nerve axis (R_local) AND its R_ci so the cuff both rotates
    # to follow the trunk's local trajectory AND tightens to the
    # local cross-section, without touching the global cuff
    # frame's nerve transform. The heavy math lives in
    # `refit_design_geometry` (injected from app.py) so the same
    # function is reusable from the design-sweep generator.
    @state.change("refit_design_request")
    def _on_refit_electrode_request(**_kwargs):
        eid = str(state.refit_design_request or "").strip()
        if not eid:
            return
        state.refit_design_request = ""
        if find_design(eid) is None or geom.nerve is None:
            return
        # Commit pending slider edits before the helper reads
        # them off the dict.
        save_selected_to_designs()
        if refit_design_geometry is None:
            return
        if not refit_design_geometry(eid):
            return
        # Translate-only re-render — the new R_local_elec is
        # picked up by the per-electrode transform in the render
        # loop, not by do_fit_cuff's frame fit.
        if geom.nerve is not None:
            asyncio.create_task(do_fit_cuff(refit=False))

    # Rename bridge — Vue handlers write {eid, name} into
    # rename_design_request, this watcher applies + clears.
    @state.change("rename_design_request")
    def _on_rename_electrode_request(**_kwargs):
        edit = state.rename_design_request
        if not edit:
            return
        state.rename_design_request = None
        try:
            eid = str(edit.get("eid", ""))
            new_name = str(edit.get("name", "")).strip()[:48]
        except Exception:
            return
        if not eid or not new_name:
            return
        electrodes = list(state.designs or [])
        for e in electrodes:
            if e.get("eid") == eid:
                e["name"] = new_name
                break
        state.designs = electrodes
