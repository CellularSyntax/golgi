# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cuff-designer preview renderer — offscreen pyvista screenshot."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyvista as pv

import cuff_designer

if TYPE_CHECKING:                                  # pragma: no cover
    from .export import FigureExportPreset


def _render_cuff_design_preview(
    parts: list,
    r_nerve_m: float = 0.0,
    *,
    preset: "FigureExportPreset | None" = None,
) -> str:
    """Render a list of (instance_label, sublabel, mesh, role) tuples
    from cuff_designer.render_design into a small offscreen PNG with
    pyvista. The returned string is a base64-encoded data URI ready
    to drop into an html.Img src. r_nerve_m, if non-zero, adds a
    semi-transparent nerve cylinder for context."""
    import base64
    import io
    # F1.2: preset controls the screenshot resolution. width_in *
    # dpi rounds up to an integer pixel count; height likewise.
    # Default 640×480 reproduces the pre-F1.2 screen behaviour.
    if preset is not None:
        win_w = max(64, int(round(preset.width_in * preset.dpi)))
        win_h = max(64, int(round(preset.height_in * preset.dpi)))
    else:
        win_w, win_h = 640, 480
    plotter = pv.Plotter(off_screen=True, window_size=(win_w, win_h))
    plotter.set_background("#fafafa")
    if not parts:
        plotter.add_text(
            "No design to preview", color="#888a90", font_size=14,
        )
    else:
        # Nerve cylinder for context — sized to the cuff axial
        # extent so the preview is self-framing.
        if r_nerve_m > 0:
            z_min = min(m.bounds[4] for _, _, m, _ in parts)
            z_max = max(m.bounds[5] for _, _, m, _ in parts)
            pad = 0.25 * (z_max - z_min)
            nerve_cyl = pv.Cylinder(
                center=(0.0, 0.0, 0.5 * (z_min + z_max)),
                direction=(0.0, 0.0, 1.0),
                radius=r_nerve_m,
                height=(z_max - z_min) + 2.0 * pad,
                resolution=48, capping=True,
            )
            plotter.add_mesh(
                nerve_cyl, color="#1f1240", opacity=0.30,
                show_edges=False, smooth_shading=True,
            )
        for inst_label, sub_label, mesh, role in parts:
            color = cuff_designer.ROLE_COLORS.get(
                role, (0.7, 0.7, 0.7),
            )
            opacity = cuff_designer.ROLE_OPACITIES.get(role, 1.0)
            plotter.add_mesh(
                mesh, color=color, opacity=opacity,
                show_edges=False, smooth_shading=True,
                specular=0.4, specular_power=15.0,
            )
    # Frame the camera looking down the +y axis so the cuff axis is
    # horizontal in the preview — easier to compare with the design
    # drawings users have in their head.
    plotter.camera_position = "yz"
    plotter.camera.elevation = 20
    plotter.camera.azimuth = -25
    plotter.reset_camera()
    img = plotter.screenshot(return_img=True,
                                transparent_background=False)
    plotter.close()
    # Encode as PNG via matplotlib's writer (already a project dep)
    # to avoid pulling in PIL just for this.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, img, format="png")
    return (
        "data:image/png;base64,"
        + base64.b64encode(buf.getvalue()).decode("ascii")
    )
