# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Overlay-toggle + per-region / per-fiber-branch visibility
watchers + mesh-panel show/edges/quality toggles."""
from __future__ import annotations

from typing import Callable, Iterable, Optional


def register(
    state,
    *,
    geom,
    elec_sync_guard: dict,
    request_render,
    ensure_field_lines_async,
    region_tags: "Iterable[int] | None" = None,
    max_fiber_branches: int = 0,
    # W1.7b — mesh-panel deps. All optional so callers extracted
    # before W1.7b (none today, but the kwargs become live as the
    # extraction lands) keep working with their old call.
    palette_edges_key: Optional[str] = None,
    update_muscle_preview: Optional[Callable[[], None]] = None,
    remove_muscle_overlay: Optional[Callable[[], None]] = None,
    muscle_pad_keys: "Iterable[str] | None" = None,
    # F3.2-M2.1a — epi-shell preview (mirrors muscle preview).
    update_epi_preview: Optional[Callable[[], None]] = None,
    remove_epi_overlay: Optional[Callable[[], None]] = None,
) -> None:
    """Wire show_ve_surface / show_ve_fibers / show_field_lines
    overlay toggles, per-region vis_<tag>, vis_fibers master,
    per-branch vis_fiber_branch_<i>, plus the mesh-panel
    show_mesh / show_mesh_edges / show_*quality_color watchers and
    the muscle-padding multi-key preview rebuild.

    `region_tags`: list of region-tag ints (vis_<tag> keys).
    `max_fiber_branches`: # of vis_fiber_branch_<i> keys.
    `palette_edges_key`: the dynamic key
        f"{pl._id_name}_edge_visibility" the PyVista trame toolbar
        binds to, mirrored ↔ show_mesh_edges.
    `update_muscle_preview`: closure that re-renders the muscle
        preview from current state (pl + safe_update bound inside).
    `remove_muscle_overlay`: closure that removes the muscle_overlay
        actor (called when the Mesh drawer closes).
    `muscle_pad_keys`: state-keys that should re-render the muscle
        preview when changed (defaults to the five muscle_*_mm
        controls).
    """

    @state.change("show_ve_surface")
    def _on_show_ve_surface_change(**_kwargs):
        """Re-render the built mesh when the Vₑ-on-surface toggle
        flips. Re-uses geom.region_surfaces[1] + the cached
        nerve_surface_Ve scalar — no recompute."""
        if elec_sync_guard["loading"]:
            return
        if not state.has_mesh or geom.msh_path is None:
            return
        # Folding the Vₑ scalar onto endo+epi happens in
        # _set_region_groups; we just request a render.
        request_render()

    @state.change("show_ve_fibers")
    def _on_show_ve_fibers_change(**_kwargs):
        """Switch fiber mode (palette ↔ Vₑ-tube). Scene folder
        decides the right actor namespace from show_ve_fibers."""
        if elec_sync_guard["loading"]:
            return
        if geom.fiber_paths_raw is None:
            return
        request_render()

    @state.change("show_field_lines")
    def _on_show_field_lines_change(**_kwargs):
        """Toggle the 3D E-field streamlines. The first mount
        after a solve schedules an executor compute; cached
        polydata is reused on subsequent toggles."""
        if elec_sync_guard["loading"]:
            return
        if not state.has_fem:
            return
        ensure_field_lines_async()

    # W1.7a — per-region vis_<tag> + master vis_fibers + per-fiber-
    # branch vis_fiber_branch_<i>. All three families do the same
    # thing — flip the relevant `_visible` flag on the scene group
    # by re-rendering; `_apply_*` lives inside the scene folder
    # itself and is keyed off the state value at render time.
    for _tag in list(region_tags or []):
        _state_key = f"vis_{_tag}"
        def _make(tag=_tag, key=_state_key):
            @state.change(key)
            def _cb(**_kwargs):
                request_render()
            return _cb
        _make()

    @state.change("vis_fibers")
    def _on_vis_fibers_master(**_kwargs):
        request_render()

    for _i in range(int(max_fiber_branches)):
        _vis_key = f"vis_fiber_branch_{_i}"
        def _make_branch_cb(key=_vis_key):
            @state.change(key)
            def _cb(**_kwargs):
                request_render()
            return _cb
        _make_branch_cb()

    # ---- W1.7b: mesh-panel show / edges / quality / muscle pad ----

    @state.change("show_mesh_quality_color")
    def _on_mesh_quality_color_change(**_kwargs):
        """Flip per-tet quality colouring on the region actors.
        The scene folder reads `state.show_mesh_quality_color` and
        swaps the region styles; we just request a render."""
        if geom.msh_path is None or not state.has_mesh:
            return
        request_render()

    @state.change("show_quality_color")
    def _on_quality_color_change(**_kwargs):
        """Flip per-triangle quality colouring on the raw nerve
        actor. Pre-mesh only — `_set_nerve_group` suppresses the
        nerve actor entirely once the FEM mesh exists."""
        if geom.nerve is None or geom.nerve_q is None:
            return
        if state.has_mesh:
            # Toggle is a no-op once regions own the nerve volume.
            return
        request_render()

    @state.change("show_mesh")
    def _on_show_mesh_change(**_kwargs):
        """Entering the Mesh drawer primes edges on. The muscle
        preview used to mount/unmount on this watcher; as of
        F3.2-M2.1a the muscle preview is persistent (mounted
        whenever the nerve is loaded), so we no longer toggle it
        here."""
        if state.show_mesh:
            # Auto-activate the wireframe overlay (flag stays where
            # the user puts it after that — closing the drawer does
            # NOT auto-disable).
            if not state.show_mesh_edges:
                state.show_mesh_edges = True

    # PyVista's trame plotter_ui mounts a small palette toolbar
    # whose edge-visibility toggle is bound to a state var named
    # `{pl._id_name}_edge_visibility`. We mirror show_mesh_edges
    # ↔ that key both ways so server-side flips update the
    # toolbar visual + user clicks on the toolbar update our flag.
    # The lock prevents either watcher from re-triggering the other.
    _edge_sync_lock = {"locked": False}

    @state.change("show_mesh_edges")
    def _on_show_mesh_edges_change(**_kwargs):
        """Re-style the region actors so the new `show_edges` kwarg
        lands on the next render. Cheap — `_set_region_groups`
        only mutates the style dict; the mesh payload is unchanged."""
        # Sync the PyVista trame toolbar toggle FIRST so its visual
        # state matches even when no mesh has been built yet.
        if (palette_edges_key is not None
                and not _edge_sync_lock["locked"]):
            _edge_sync_lock["locked"] = True
            try:
                _target = bool(state.show_mesh_edges)
                if bool(state[palette_edges_key]) != _target:
                    state[palette_edges_key] = _target
            except Exception:
                pass
            finally:
                _edge_sync_lock["locked"] = False
        if geom.msh_path is None or not state.has_mesh:
            return
        request_render()

    if palette_edges_key is not None:
        @state.change(palette_edges_key)
        def _on_palette_edges_change(**_kwargs):
            """Mirror palette-toolbar clicks back into
            show_mesh_edges so the legend stays in sync."""
            if _edge_sync_lock["locked"]:
                return
            _edge_sync_lock["locked"] = True
            try:
                _target = bool(state[palette_edges_key])
                if bool(state.show_mesh_edges) != _target:
                    state.show_mesh_edges = _target
            except Exception:
                pass
            finally:
                _edge_sync_lock["locked"] = False

    # Muscle-padding multi-key — re-render the preview as the user
    # drags any of the five muscle_* sliders, but only while the
    # Mesh drawer is open (no point rebuilding a hidden actor).
    _mp_keys = tuple(
        muscle_pad_keys
        if muscle_pad_keys is not None
        else (
            "muscle_radial_pad_mm", "muscle_axial_pad_mm",
            "muscle_dx_mm", "muscle_dy_mm", "muscle_dz_mm",
        )
    )

    @state.change(*_mp_keys)
    def _on_muscle_pad_change(**_kwargs):
        # F3.2-M2.1a: persistent preview, not gated on show_mesh
        # OR on `geom.pts_cuff` (the function itself falls back
        # to raw nerve points pre-fit). Just needs `geom.nerve`
        # to be loaded.
        if (geom.nerve is not None
                and bool(state.vis_muscle_preview)
                and update_muscle_preview is not None):
            update_muscle_preview()

    # F3.2-M2.1a — pre-mesh epi + muscle preview lifecycle.
    # `has_geometry` mounts both previews the moment the nerve
    # is loaded; `vis_*_preview` toggle hides without unmount;
    # epi-specific params (use_epi, epi_thickness_um) trigger
    # recompute. The "any design has a mesh" cutover that hands
    # visibility to the per-design Tissues rows is handled by
    # `_update_epi_preview` / `_update_muscle_preview` checking
    # internally, plus the post-build hook in pipeline/mesh.py
    # that calls remove_*_overlay once the first mesh lands.
    @state.change("has_geometry")
    def _on_has_geometry_for_previews(**_kwargs):
        if not bool(state.has_geometry):
            # Geometry was removed (project closed / nerve cleared)
            # — strip both previews so they don't linger over an
            # empty viewport.
            if remove_muscle_overlay is not None:
                remove_muscle_overlay()
            if remove_epi_overlay is not None:
                remove_epi_overlay()
            return
        if (bool(state.vis_muscle_preview)
                and update_muscle_preview is not None
                and geom.nerve is not None):
            update_muscle_preview()
        if (bool(state.vis_epi_preview)
                and update_epi_preview is not None):
            update_epi_preview()

    @state.change("vis_muscle_preview")
    def _on_vis_muscle_preview(**_kwargs):
        if bool(state.vis_muscle_preview):
            if (update_muscle_preview is not None
                    and geom.nerve is not None):
                update_muscle_preview()
        else:
            if remove_muscle_overlay is not None:
                remove_muscle_overlay()

    # F3.2-M3 — one-way unlock for the muscle bbox preview. The
    # stepper flips `muscle_preview_unlocked` to True the first
    # time the user reaches Step 4; that's our cue to actually
    # mount the cylinder. `_update_muscle_preview` short-circuits
    # before this flag flips, so calling it here is what makes
    # the bbox first appear.
    @state.change("muscle_preview_unlocked")
    def _on_muscle_preview_unlocked(**_kwargs):
        if not bool(state.muscle_preview_unlocked):
            return
        if (update_muscle_preview is not None
                and geom.nerve is not None
                and bool(state.vis_muscle_preview)):
            update_muscle_preview()

    # Watcher on the stepper step itself: flips
    # `muscle_preview_unlocked` True when the user first reaches
    # Step 4 (via the Continue button OR by clicking the step
    # pill directly). One-way — once True, never resets.
    @state.change("import_stepper_step")
    def _on_import_stepper_step_for_muscle_unlock(**_kwargs):
        if (str(state.import_stepper_step or "") == "4"
                and not bool(state.muscle_preview_unlocked)):
            state.muscle_preview_unlocked = True

    @state.change("vis_epi_preview", "use_epi", "epi_thickness_um")
    def _on_epi_preview_param_change(**_kwargs):
        # `_update_epi_preview` already short-circuits on
        # use_epi=False / no nerve / any design meshed, so the
        # only thing we need to do here is re-invoke it (or
        # explicitly remove when the visibility flag flips off).
        if not bool(state.vis_epi_preview):
            if remove_epi_overlay is not None:
                remove_epi_overlay()
            return
        if update_epi_preview is not None:
            update_epi_preview()
