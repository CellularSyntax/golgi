# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Publication-grade figure export presets (F1.2).

Every figure builder in `golgi/figures/` accepts an optional
`preset: FigureExportPreset | None` kwarg. When `preset is None`
(the default) the builder behaves exactly as before — screen-DPI
PNG data URI for the matplotlib renderers, on-screen Plotly dict
for the interactive tiles. When a preset is passed, the builder
applies its DPI / font / palette overrides before returning.

This module is the single source of truth for those overrides + a
pair of helpers (`render_publication_mpl`, `render_publication_plotly`)
that F2.3 (Bulk Figure Export view) will call to write each figure
to a vector file on disk.

Intentionally zero coupling to the Trame UI or to any pipeline
stage — F2.3 wires the buttons; here we only ship the presets and
the apply/render helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Palettes — colour-blind safe primaries
# ---------------------------------------------------------------------------
# Each entry maps to:
#   - `mpl_cmap`: name of a matplotlib continuous cmap (heatmaps, fields)
#   - `qualitative`: ordered list of hex colours for categorical traces
# When a preset selects a palette, the apply functions push these onto
# matplotlib axes (rcParams override for cmap; axes.prop_cycle for
# qualitative) and onto Plotly layout (colorway).

PALETTES: dict[str, dict[str, Any]] = {
    # Default = matplotlib's tab10 + viridis. Same as today's screen
    # behaviour — the SCREEN preset uses this so applying SCREEN is
    # functionally a no-op restyle.
    "default": {
        "mpl_cmap": "viridis",
        "qualitative": [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ],
    },
    # Cividis — Nuñez et al. 2018; perceptually uniform AND
    # colour-blind safe (designed for deuteranomalous viewers).
    # First choice for the FEM slice heatmap in paper figures.
    "viridis-cb": {
        "mpl_cmap": "cividis",
        "qualitative": [
            "#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
            "#CC6677", "#AA4499", "#882255",
        ],
    },
    # IBM design-library 5-colour palette — high-contrast, distinct
    # under all common CVD types (deutan, protan, tritan).
    "ibm-cb": {
        "mpl_cmap": "cividis",
        "qualitative": [
            "#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000",
        ],
    },
    # Print-grayscale fallback — useful when a journal still rejects
    # colour figures (some legacy journals) or for thumbnailing.
    "gray": {
        "mpl_cmap": "gray",
        "qualitative": [
            "#1a1a1a", "#404040", "#707070", "#9a9a9a", "#c0c0c0",
        ],
    },
}


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureExportPreset:
    """Describes how a figure should look + how it should be written
    to disk. `fmt` is only consulted by the `render_publication_*`
    helpers — `apply_preset_to_*` ignore it (they only restyle).

    `width_in` / `height_in` are inches because that's the unit
    matplotlib's `set_size_inches` already accepts and the unit most
    journals use in their figure spec sheets. For Plotly the apply
    function converts to the equivalent px at the preset's DPI.

    `layout_mode` controls how aggressively the apply functions
    rewrite the input figure:
      * "match_ui"  → preserve the builder's width/height/fonts;
                      the exporter passes `scale=…` to kaleido to
                      raise resolution without changing layout. The
                      best choice when the user wants the export to
                      look like what they see on screen, just at a
                      higher DPI.
      * "override"  → mutate canvas size, fonts, palette, bg to the
                      preset values. The classical "journal spec"
                      path — small 3-column-width PDF with 8 pt text.

    For matplotlib figures both modes apply font + palette overrides
    (matplotlib has no equivalent of kaleido scale and re-renders
    fully each export), but only `override` resizes the canvas."""
    name: str
    fmt: str             # "png" | "pdf" | "svg" | "eps"
    dpi: int             # used for PNG / EPS; PDF/SVG are vector
    width_in: float
    height_in: float
    font_family: str
    font_size_pt: float
    palette: str         # key into PALETTES
    use_latex: bool = False
    # Optional per-axis tick label size; falls back to font_size_pt - 1
    # when None. Useful because journal figure sheets often spec the
    # tick size separately from the label size.
    tick_size_pt: float | None = None
    # "match_ui" | "override" — see class docstring.
    layout_mode: str = "override"


# Stock presets.
#
# MATCH_UI is the default: it preserves the on-screen layout +
# fonts that the figure builders set (so the export looks like the
# UI), and asks kaleido to render at 2× scale for crisp PDFs / PNGs.
# This solved the first-user feedback that paper-300's 8-pt fonts on
# a 3.4-inch canvas felt out of step with the UI figure.
MATCH_UI = FigureExportPreset(
    name="match-ui", fmt="pdf", dpi=192,   # 96 px screen DPI × 2
    width_in=0.0, height_in=0.0,            # ignored in match_ui
    font_family="DejaVu Sans", font_size_pt=12.0,
    palette="default", use_latex=False,
    layout_mode="match_ui",
)
MATCH_UI_PNG = FigureExportPreset(
    name="match-ui-png", fmt="png", dpi=192,
    width_in=0.0, height_in=0.0,
    font_family="DejaVu Sans", font_size_pt=12.0,
    palette="default", use_latex=False,
    layout_mode="match_ui",
)
SCREEN = FigureExportPreset(
    name="screen", fmt="png", dpi=120,
    width_in=6.0, height_in=4.0,
    font_family="DejaVu Sans", font_size_pt=10.0,
    palette="default", use_latex=False,
)
PAPER_300 = FigureExportPreset(
    name="paper-300", fmt="pdf", dpi=300,
    width_in=3.4, height_in=2.6,
    font_family="DejaVu Sans", font_size_pt=8.0,
    palette="viridis-cb", use_latex=False,
)
PAPER_600 = FigureExportPreset(
    name="paper-600", fmt="png", dpi=600,
    width_in=3.4, height_in=2.6,
    font_family="DejaVu Sans", font_size_pt=8.0,
    palette="viridis-cb", use_latex=False,
)
PAPER_SVG = FigureExportPreset(
    name="paper-svg", fmt="svg", dpi=300,
    width_in=3.4, height_in=2.6,
    font_family="DejaVu Sans", font_size_pt=8.0,
    palette="viridis-cb", use_latex=False,
)

# Registry for the F2.3 dropdown. Order matches the UI dropdown
# top-to-bottom: defaults + match-ui first, journal specs after.
PRESETS: dict[str, FigureExportPreset] = {
    p.name: p for p in (
        MATCH_UI, MATCH_UI_PNG, SCREEN,
        PAPER_300, PAPER_600, PAPER_SVG,
    )
}


# ---------------------------------------------------------------------------
# Apply — matplotlib
# ---------------------------------------------------------------------------


def _palette_for(preset: FigureExportPreset) -> dict[str, Any]:
    """Resolve the preset's palette key to a palette spec. Unknown
    keys fall back to 'default' so callers don't crash on a typo."""
    return PALETTES.get(preset.palette, PALETTES["default"])


def apply_preset_to_mpl_fig(fig, preset: FigureExportPreset) -> None:
    """In-place restyle of a matplotlib Figure to match the preset.
    Idempotent — safe to call multiple times.

    Sets: figure size, DPI, font family + size on every text
    artefact, tick sizes, qualitative colour cycle on every axes,
    continuous cmap on every imshow/pcolormesh that uses the default
    cmap. Does NOT touch artist-level overrides (a per-line
    `color='red'` survives; intentional — builders that hand-pick
    semantic colours like the Cole-Cole red curve should keep them)."""
    pal = _palette_for(preset)
    fig.set_size_inches(preset.width_in, preset.height_in)
    fig.set_dpi(preset.dpi)
    tick_pt = (
        preset.tick_size_pt
        if preset.tick_size_pt is not None
        else max(1.0, preset.font_size_pt - 1.0)
    )
    # Font on the figure suptitle if any.
    if fig._suptitle is not None:
        fig._suptitle.set_fontfamily(preset.font_family)
        fig._suptitle.set_fontsize(preset.font_size_pt + 1.0)
    import matplotlib as _mpl  # local import — keep module slim
    cmap = _mpl.colormaps.get(pal["mpl_cmap"]) if "mpl_cmap" in pal else None
    qual = list(pal.get("qualitative", []))
    for ax in fig.get_axes():
        # Title / axis labels.
        if ax.get_title():
            ax.title.set_fontfamily(preset.font_family)
            ax.title.set_fontsize(preset.font_size_pt + 1.0)
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            lbl.set_fontfamily(preset.font_family)
            lbl.set_fontsize(preset.font_size_pt)
        # Tick labels.
        ax.tick_params(
            axis="both", which="major", labelsize=tick_pt,
        )
        for txt in ax.get_xticklabels() + ax.get_yticklabels():
            txt.set_fontfamily(preset.font_family)
        # Legend.
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontfamily(preset.font_family)
                txt.set_fontsize(preset.font_size_pt)
        # Qualitative prop cycle — only affects traces that have NOT
        # had an explicit colour set. Existing line/scatter colours
        # are NOT mutated.
        if qual:
            ax.set_prop_cycle(color=qual)
        # Continuous cmap — applied to any AxesImage / QuadMesh that
        # currently uses matplotlib's default cmap. We do NOT clobber
        # a builder's deliberate choice (e.g. RdYlGn for the mesh
        # quality histogram is semantic — green = good, red = bad —
        # and switching it to viridis would lose the meaning). We
        # detect "default" by name match to the rcParams default.
        if cmap is not None:
            default_cmap_name = _mpl.rcParams.get("image.cmap", "viridis")
            for art in ax.get_images():
                try:
                    if art.get_cmap().name == default_cmap_name:
                        art.set_cmap(cmap)
                except Exception:
                    pass
    # Tight layout — only safe if there is at least one axes;
    # bare figures (rare) skip this.
    if fig.get_axes():
        try:
            fig.tight_layout()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Apply — Plotly
# ---------------------------------------------------------------------------


def _pt_to_px(pt: float, dpi: int) -> float:
    """Plotly font sizes are pixels, NOT points. To render an N-pt
    font on an image rasterized at `dpi`, we need N × (dpi / 72) px.
    Forgetting this conversion produces the symptom that motivated
    F2.3.a's first bug report: an 8-pt journal title on a 300-DPI
    1020×780 PNG becomes literally 8 pixels of text — invisible.
    The correct value is 8 × 300/72 ≈ 33 px, which prints at exactly
    8 pt on paper."""
    return float(pt) * float(dpi) / 72.0


def apply_preset_to_plotly_layout(
    layout: dict, preset: FigureExportPreset,
) -> dict:
    """In-place merge of preset font/palette/size into a Plotly
    layout dict. Returns the same dict (for chaining). Unlike the
    matplotlib path, we do NOT mutate trace data — Plotly trace
    colours are set per-trace by the builders and treated as
    semantic.

    In `match_ui` mode this is a near no-op: we only set palette
    (colorway) + background colours; the builder's width / height /
    fonts survive untouched and kaleido's `scale` parameter handles
    resolution at render time. That keeps the export visually
    identical to the on-screen figure.

    In `override` mode (paper-300 / paper-600 / paper-svg /
    screen): width/height are forced to `width_in × dpi`, fonts to
    `font_size_pt` converted through pt → px (Plotly's font.size is
    pixels), palette + background applied."""
    pal = _palette_for(preset)
    # Always-on bits — both modes apply these because they don't
    # affect layout, just colour. paper_bgcolor + colorway turn a
    # default-themed builder figure into one that matches the
    # selected palette.
    if pal.get("qualitative"):
        layout["colorway"] = list(pal["qualitative"])
    layout["paper_bgcolor"] = "white"
    layout["plot_bgcolor"] = "white"

    if preset.layout_mode == "match_ui":
        # Done — keep builder's width/height/fonts so the export
        # matches what the user sees on-screen.
        return layout

    # ---- "override" mode below ----
    body_px = _pt_to_px(preset.font_size_pt, preset.dpi)
    tick_pt = (
        preset.tick_size_pt
        if preset.tick_size_pt is not None
        else max(1.0, preset.font_size_pt - 1.0)
    )
    tick_px = _pt_to_px(tick_pt, preset.dpi)
    title_px = _pt_to_px(preset.font_size_pt + 1.0, preset.dpi)

    font_dict = layout.setdefault("font", {})
    font_dict["family"] = preset.font_family
    font_dict["size"] = body_px
    # Width / height — px at the preset DPI.
    layout["width"] = int(round(preset.width_in * preset.dpi))
    layout["height"] = int(round(preset.height_in * preset.dpi))
    # Walk axes — set tick + title fonts on every x/y axis present
    # (multi-subplot figures have xaxis, xaxis2, … so we match the
    # prefix rather than the exact key).
    for ax_key in list(layout.keys()):
        if not (ax_key.startswith("xaxis")
                 or ax_key.startswith("yaxis")):
            continue
        ax = layout[ax_key]
        if not isinstance(ax, dict):
            continue
        ax_title = ax.get("title")
        if isinstance(ax_title, dict):
            ax_title_font = ax_title.setdefault("font", {})
            ax_title_font["family"] = preset.font_family
            ax_title_font["size"] = body_px
        tickfont = ax.setdefault("tickfont", {})
        tickfont["family"] = preset.font_family
        tickfont["size"] = tick_px
    # Title (chart-level) — match label size + 1 for visual weight.
    title = layout.get("title")
    if isinstance(title, dict):
        title_font = title.setdefault("font", {})
        title_font["family"] = preset.font_family
        title_font["size"] = title_px
    # Legend — Plotly defaults to a smaller font that looks weirdly
    # cramped after the body bump. Pin to body size.
    legend = layout.get("legend")
    if isinstance(legend, dict):
        legend_font = legend.setdefault("font", {})
        legend_font["family"] = preset.font_family
        legend_font["size"] = body_px
    # Annotations — colour-bar tick labels live here on some
    # builders. Scale them too so e.g. the Cole-Cole "1.234 S/m"
    # readout doesn't become a 6-px speck on a 300-DPI export.
    annotations = layout.get("annotations")
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            ann_font = ann.setdefault("font", {})
            ann_font["family"] = preset.font_family
            ann_font.setdefault("size", body_px)
            # If the builder set a smaller per-annotation size,
            # rescale by the same pt→px ratio so the visual
            # hierarchy survives (titles big, callouts small).
            current = ann_font.get("size")
            if (current is not None
                    and current != body_px
                    and current < 32):
                # Treat the existing value as pt-ish (most builders
                # set sizes in the 9-12 range thinking pt) and
                # convert.
                ann_font["size"] = _pt_to_px(
                    float(current), preset.dpi,
                )
    return layout


def apply_preset_to_plotly_fig(
    fig_dict: dict, preset: FigureExportPreset,
) -> dict:
    """Same as `apply_preset_to_plotly_layout` but takes a full
    `{data, layout}` dict — what builders return. Returns the same
    dict for chaining."""
    layout = fig_dict.get("layout")
    if isinstance(layout, dict):
        apply_preset_to_plotly_layout(layout, preset)
    return fig_dict


# ---------------------------------------------------------------------------
# Render to file
# ---------------------------------------------------------------------------


def render_publication_mpl(
    fig, out_path: Path, preset: FigureExportPreset,
) -> Path:
    """Apply the preset and save to out_path. Closes the figure
    afterwards — caller does not need to. Returns out_path for
    chaining (so callers can collect a list of written paths)."""
    apply_preset_to_mpl_fig(fig, preset)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    # PDF / SVG / EPS are vector — DPI only matters for embedded
    # raster artists (e.g. an `imshow`); the rest scales freely.
    save_kwargs: dict[str, Any] = {
        "format": preset.fmt,
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    if preset.fmt in {"png", "jpg", "jpeg"}:
        save_kwargs["dpi"] = preset.dpi
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


_RASTER_FMTS = frozenset({"png", "jpg", "jpeg", "webp"})


def _plotly_render_kwargs(
    preset: FigureExportPreset, fmt: str | None = None,
) -> dict[str, Any]:
    """Build the width / height / scale kwargs for
    plotly.io.write_image (or to_image) according to the preset's
    layout_mode.

    * match_ui  → width=None, height=None (let kaleido use the
                  layout's width/height that the builder set). For
                  RASTER fmts (png/jpg) we pass scale = dpi / 96 so
                  the PNG comes out 2× the on-screen pixel size with
                  the same visual proportions. For VECTOR fmts
                  (pdf/svg) scale stays at 1 — vector means
                  resolution-independent, and scale>1 would just
                  multiply the page dimensions (a 600 × 400 layout
                  + scale=2 = a 16.67-inch-wide PDF, which is
                  hilariously oversized for a journal column).
    * override  → width × dpi pixels, scale=1 (the apply function
                  has already set the font sizes in pt-converted
                  pixels)."""
    fmt = (fmt or preset.fmt).lower()
    if preset.layout_mode == "match_ui":
        scale = max(1.0, preset.dpi / 96.0) if fmt in _RASTER_FMTS else 1.0
        return {
            "format": fmt,
            "scale": scale,
        }
    return {
        "format": fmt,
        "width": int(round(preset.width_in * preset.dpi)),
        "height": int(round(preset.height_in * preset.dpi)),
        "scale": 1,
    }


def render_publication_plotly(
    fig_dict: dict, out_path: Path, preset: FigureExportPreset,
) -> Path:
    """Apply preset, then write via plotly.io.write_image (kaleido).
    Returns out_path. Raises a clear error if kaleido is missing —
    it is pinned in requirements-frozen.txt but ships separately on
    some platforms."""
    apply_preset_to_plotly_fig(fig_dict, preset)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.io as pio
    except ImportError as ex:                       # pragma: no cover
        raise RuntimeError(
            "Plotly is required for figure export "
            "(missing `plotly`).",
        ) from ex
    try:
        import kaleido  # noqa: F401
    except ImportError as ex:                       # pragma: no cover
        raise RuntimeError(
            "Plotly vector export requires `kaleido`. "
            "Install with `pip install kaleido`.",
        ) from ex
    # plotly.io.write_image accepts a dict; format inferred from
    # path suffix unless `format` is passed explicitly.
    pio.write_image(
        fig_dict, str(out_path),
        **_plotly_render_kwargs(preset, fmt=preset.fmt),
    )
    return out_path


def render_publication(
    fig_or_dict, out_path: Path, preset: FigureExportPreset,
) -> Path:
    """Dispatch by type — matplotlib Figure → mpl path,
    `{data, layout}` dict → Plotly path. Most callers will know
    which they have and call the specific function directly; this
    exists for the F2.3 registry where the builder type is opaque
    until invocation."""
    # Avoid importing matplotlib at module load time; do it now.
    import matplotlib.figure as _mpl_fig
    if isinstance(fig_or_dict, _mpl_fig.Figure):
        return render_publication_mpl(fig_or_dict, out_path, preset)
    if isinstance(fig_or_dict, dict):
        return render_publication_plotly(
            fig_or_dict, out_path, preset,
        )
    raise TypeError(
        f"Unsupported figure type for export: "
        f"{type(fig_or_dict).__name__}",
    )
