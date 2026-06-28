# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Tiny shared helpers used by the figure builders in this package."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                  # pragma: no cover
    from .export import FigureExportPreset


# Seaborn `flare` colormap, sampled at 7 stops. Sequential
# light-orange → dark-purple-red, colourblind-friendly and high-
# contrast on white. Used as the heatmap colorscale on the V_m
# propagation tile so the AP wavefront shows up as a dark band
# against the resting-membrane baseline.
_FLARE_COLORSCALE = [
    [0.00, "#edb482"],
    [0.16, "#e29372"],
    [0.33, "#d2716a"],
    [0.50, "#bd536d"],
    [0.66, "#a13a72"],
    [0.83, "#7c2773"],
    [1.00, "#54186c"],
]

# Shared axis-title font for the fiber-tab + population-tab
# Plotly tiles. Bumped vs the Plotly default so axis labels are
# immediately readable inside the small 320 px tiles.
_FIBER_AXIS_TITLE_FONT = {"size": 12, "color": "#1f2024"}
_FIBER_AXIS_TICK_FONT = {"size": 10, "color": "#4a4a52"}


def _fig_to_data_uri(
    fig, preset: "FigureExportPreset | None" = None,
) -> str:
    """matplotlib Figure → base64 PNG data URI. Shared by the
    quality histogram + the FEM Ve/E plots.

    When `preset` is given (F1.2), the figure is restyled in-place
    via `figures.export.apply_preset_to_mpl_fig` BEFORE the PNG is
    written, and the PNG is rendered at the preset's DPI. The
    return type stays a base64 data URI — for vector formats use
    `_fig_to_file` instead."""
    import base64
    import io
    import matplotlib.pyplot as plt
    save_kwargs = {"format": "png", "bbox_inches": "tight",
                    "facecolor": "white"}
    if preset is not None:
        from .export import apply_preset_to_mpl_fig
        apply_preset_to_mpl_fig(fig, preset)
        save_kwargs["dpi"] = preset.dpi
    buf = io.BytesIO()
    fig.savefig(buf, **save_kwargs)
    plt.close(fig)
    return ("data:image/png;base64,"
             + base64.b64encode(buf.getvalue()).decode("ascii"))


def _fig_to_file(
    fig, out_path: "Path | str",
    preset: "FigureExportPreset | None" = None,
) -> Path:
    """matplotlib Figure → on-disk file at `out_path`. When `preset`
    is None, writes PNG at screen DPI (120) for ad-hoc snapshots.
    When `preset` is given, delegates to `render_publication_mpl`
    which applies the preset and writes in the preset's format
    (PDF / SVG / EPS / PNG)."""
    from .export import (
        FigureExportPreset, render_publication_mpl, SCREEN,
    )
    p = Path(out_path)
    if preset is None:
        # Use SCREEN preset but override the format from the path
        # suffix so an explicit `out_path='foo.svg'` still does the
        # right thing without a preset.
        suffix = p.suffix.lstrip(".").lower() or "png"
        preset = FigureExportPreset(
            name="adhoc", fmt=suffix,
            dpi=SCREEN.dpi, width_in=SCREEN.width_in,
            height_in=SCREEN.height_in,
            font_family=SCREEN.font_family,
            font_size_pt=SCREEN.font_size_pt,
            palette=SCREEN.palette,
        )
    return render_publication_mpl(fig, p, preset)


def _plotly_placeholder(msg: str) -> dict:
    """Tiny Plotly figure used when an inputs-missing branch fires —
    same dict shape as the real figures so the trame widget can render
    it without any conditional logic on the caller side."""
    return {
        "data": [],
        "layout": {
            "annotations": [{
                "text": msg,
                "x": 0.5, "y": 0.5,
                "xref": "paper", "yref": "paper",
                "showarrow": False,
                "font": {"size": 13, "color": "#888a90"},
            }],
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "margin": {"l": 30, "r": 20, "t": 30, "b": 30},
            "paper_bgcolor": "rgba(255,255,255,0)",
            "plot_bgcolor": "rgba(255,255,255,0)",
        },
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """#rrggbb → rgba(r,g,b,alpha). Used for KDE fill colour so
    the line stays opaque while the area below it goes
    translucent. Defensive: unparseable input → neutral grey."""
    h = hex_color.lstrip("#")
    try:
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        r, g, b = 102, 102, 102
    return f"rgba({r},{g},{b},{alpha:.2f})"
