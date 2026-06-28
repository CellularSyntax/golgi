# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Geometry / mesh status state defaults."""
from __future__ import annotations


def register(state) -> None:
    # Loaded mesh summary
    state.geom_summary = "no geometry loaded"
    state.quality_hist_figure = {"data": [], "layout": {}}
    state.has_geometry = False
    state.show_quality_color = False
    # Mesher: gmsh OCC (conformal, robust) by default for prismatic/extruded
    # nerves, with automatic PLC+TetGen fallback for true-3D geometry or on
    # gmsh failure. Toggle in the Mesh panel.
    state.use_gmsh_mesher = True
    # Nerve cross-section deformation, applied AT IMPORT (histology-bundle /
    # µCT reconstruction) so geometry, rendering, cuff fit, fibers and mesh
    # are all consistent:
    #   "round" — area-preserving circularization (nerve + fascicles, Duke-
    #             style; clean cuff annulus); "none" — keep the real shape.
    state.nerve_deform = "round"
