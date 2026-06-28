# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Drawer flags, viewport mode, overlay toggles, legend collapse
flags, busy status. The "global UI chrome" defaults."""
from __future__ import annotations


def register(state) -> None:
    # Drawer open/close flags
    state.show_import = False
    # F3.2-M2.1 — Import nerve wizard (replaces show_import drawer).
    # Active step pill is a "1".."4" string because Vuetify's
    # VStepper v-model carries the step value as a string.
    state.show_import_stepper = False
    state.import_stepper_step = "1"
    state.show_cuff = False
    state.show_mesh = False
    state.show_sigma = False
    state.show_fibers = False
    # Analysis tabs (Phase A skeleton — populated in Phases B/C/D)
    state.show_solve = False
    state.show_fiber = False
    state.show_pop = False
    state.show_sweep = False         # F2.1.c
    state.show_compare = False       # F3.2e — Compare analysis tab
    state.show_exports = False       # F2.3.b — bulk Exports drawer
    state.show_export_menu = False   # F2.3 umbrella menu (figures/report)
    # Viewport mode: "3d" → full-screen pyvista plotter,
    # "analysis" → 3D shrinks to a top-right thumbnail and the
    # main area shows analysis plots. Derived reactively from
    # show_solve/show_fiber/show_pop/show_sweep/show_compare.
    state.viewport_mode = "3d"
    state.active_analysis = ""       # "solve"|"fiber"|"population"|"sweep"|"compare"

    # F3.2-M2.1a — top-level Nerve section in the legend.
    # vis_nerve_raw toggles the raw endoneurium boundary actor
    # (the one rendered from the imported STL/NAS before any
    # design has a mesh). Once any design has a meshed
    # nerve.msh, _set_nerve_group auto-suppresses the raw
    # actor — at that point the per-design Tissues > Endoneurium
    # row takes over visibility control.
    state.vis_nerve_raw = True
    # F3.2-M2.1a — pre-mesh previews of the epineurium shell
    # (inward-offset of the nerve boundary by epi_thickness_um)
    # and the muscle bbox cylinder. Both mount as translucent
    # actors as soon as `has_geometry` is true; they auto-hide
    # once any design has a meshed nerve.msh (per-design Tissues
    # rows take over). Visible by default so the legend immediately
    # shows what the stepper's Step 2 / Step 4 params are
    # producing.
    state.vis_epi_preview = True
    state.vis_muscle_preview = True
    # F3.2-M3 — gate the muscle bbox preview on the user having
    # reached Step 4 of the import stepper at least once. Without
    # this, the cylinder pops in immediately after the nerve
    # loads, before the user has any chance to read what it is.
    # One-way flag (False → True only) — once unlocked, the row
    # stays available even after re-loading the project, so the
    # legend toggle behaves predictably across sessions.
    state.muscle_preview_unlocked = False

    # Vₑ overlay toggles
    state.show_ve_fibers = False     # legend toggle for Ve on fibers
    state.show_ve_surface = False    # legend toggle for Ve on endo surface
    # Legend toggle for the 3D E-field streamlines view. Mounts a
    # polyline / tube actor coloured by |E| through the full
    # domain (muscle → cuff → epi → endo → saline) so the user
    # can see how current actually flows from the contacts.
    state.show_field_lines = False

    # Floating visibility-legend panel — hidden by default,
    # toggled by an eye-icon FAB anchored in the top-right of
    # the viewport. User opted for "Legend hidden by default" so
    # the workspace starts uncluttered.
    state.legend_visible = False
    # Per-section collapse flags for the floating legend so the
    # user can fold Tissues / Electrodes / Fibers / Overlays /
    # Fiber types independently and save vertical space. All
    # default open. Ephemeral — viewport prefs not worth
    # persisting; the legend opens fresh each session.
    state.legend_tissues_open = True
    state.legend_electrodes_open = True
    state.legend_fibers_open = True
    state.legend_overlays_open = True
    state.legend_fiber_types_open = True
    # Phase 6b — supersection collapse flags for the new
    # restructured legend (Nerve / Fibers / Muscle / Cuff
    # Electrode / Mesh / Overlays). All default-open so the
    # user sees everything available at this stage on first
    # reveal; chevron click on each header collapses.
    state.legend_nerve_open = True
    state.legend_muscle_open = True
    state.legend_cuff_open = True
    state.legend_mesh_open = True

    # Import-stepper Advanced-section toggles (Step 3 fibers,
    # Step 4 muscle bbox translation). Closed by default so the
    # minimal-knob view is what the user sees first.
    state.stepper_fiber_advanced_open = False
    state.stepper_muscle_advanced_open = False
    # Design drawer — Advanced toggle inside "Cuff frame &
    # placement". Hides the transverse Δx/Δy, pitch/yaw/twist,
    # and local PCA radius knobs (used for off-axis / non-
    # coaxial experiments) so the default placement panel only
    # shows the anchor + axial offset.
    state.cuff_placement_advanced_open = False
    # Step 3 — fiber-generation method + auto-branch-detect flag.
    # Method is the algorithm picker (only "streamlines" works
    # today; "algorithmic_1" / "algorithmic_2" are placeholders).
    # Auto-detect off ⇒ skip cap detection, generate one bundle
    # of fibers, hide all four branch params even from Advanced.
    state.fiber_method = "streamlines"
    state.fiber_auto_detect_branches = True

    # Activity flag (shows overlay progress)
    state.busy = False
    state.busy_msg = ""
    state.busy_log = ""           # last few lines streamed into the lightbox
