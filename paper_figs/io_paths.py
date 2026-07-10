# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Shared output layout for paper figures/tables/data.

out/
  figures/{png,pdf,svg}/   one dir per format, all rendered figures
  tables/                  csv tabular deliverables
  data/                    json + npz (summaries, matrices, probes)
  _intermediate/           render scratch (auto-regenerated)
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "paper_figs/out"
FIG_FORMATS = ("png", "pdf", "svg")
FIG_DIRS = {e: OUT / "figures" / e for e in FIG_FORMATS}
TABLES = OUT / "tables"
DATA = OUT / "data"
TMP = OUT / "_intermediate"

for _d in (*FIG_DIRS.values(), TABLES, DATA, TMP):
    _d.mkdir(parents=True, exist_ok=True)


def transparent_render(png_path, white_thresh=250):
    """RGBA view of a white-background render PNG with near-white pixels made
    transparent, so it composites cleanly as a plot inset (the plot lines show
    through the background instead of being covered by a white box)."""
    from PIL import Image
    import numpy as np
    im = np.asarray(Image.open(png_path).convert("RGB")).astype(np.uint8)
    alpha = np.where(im.min(axis=2) > white_thresh, 0, 255).astype(np.uint8)
    return np.dstack([im, alpha])


def scale_sidecar(png_path):
    """Path of the px/mm sidecar JSON written next to a render PNG."""
    p = Path(png_path)
    return p.with_name(p.stem + ".scale.json")


def write_ppmm(png_path, px_per_mm):
    import json
    scale_sidecar(png_path).write_text(json.dumps({"px_per_mm": float(px_per_mm)}))


def load_ppmm(png_path):
    """pixels-per-mm for a render PNG (from its sidecar), or None if absent."""
    import json
    s = scale_sidecar(png_path)
    if s.exists():
        try:
            return float(json.loads(s.read_text())["px_per_mm"])
        except Exception:
            return None
    return None


def px_per_mm(plotter, center, L=1.0):
    """pixels-per-mm in `plotter`'s current view at world point `center` (mm),
    measured along the screen-horizontal axis. Call after the camera is set; the
    plotter window size must equal the screenshot size. Geometry must be in mm."""
    import numpy as np
    plotter.render()
    cam = plotter.camera
    vd = np.asarray(cam.focal_point, float) - np.asarray(cam.position, float)
    vd /= np.linalg.norm(vd)
    right = np.cross(vd, np.asarray(cam.up, float)); right /= np.linalg.norm(right)
    ren = plotter.renderer

    def w2d(pt):
        ren.SetWorldPoint(float(pt[0]), float(pt[1]), float(pt[2]), 1.0); ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint(); return np.array([x, y])
    c = np.asarray(center, float)
    return float(np.linalg.norm(w2d(c + right * L) - w2d(c)) / L)


def draw_scalebar(ax, w_px, h_px, ppmm, frac=0.22, pad_frac=0.05,
                  color="#1b1d22", lw=3.2, loc="lower right"):
    """Horizontal scale bar on an imshow'd render axis (data coords = pixels). The
    bar length is auto-rounded to a nice value near `frac` of the image width. A
    white halo keeps it legible over any background."""
    import numpy as np
    import matplotlib.patheffects as pe
    if not ppmm or ppmm <= 0:
        return
    target = frac * w_px / ppmm
    nice = np.array([0.2, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100])
    mm = float(nice[np.argmin(np.abs(nice - target))])
    bar = mm * ppmm
    padx, pady = pad_frac * w_px, pad_frac * h_px
    x1 = (w_px - padx) if "right" in loc else (padx + bar)
    x0 = x1 - bar
    y = (h_px - pady) if "lower" in loc else pady
    halo = [pe.withStroke(linewidth=lw + 2.6, foreground="white")]
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=12, path_effects=halo)
    t = ax.text((x0 + x1) / 2, y - 0.018 * h_px, f"{mm:g} mm", ha="center", va="bottom",
                fontsize=9.5, color=color, zorder=12, clip_on=False)
    t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])


def render_legend(ax, entries, y=-0.03, fontsize=9.5, ncol=None):
    """Horizontal domain legend below an imshow'd render axis. `entries` is a list
    of (color, label); a coloured square is drawn for each."""
    from matplotlib.patches import Patch
    h = [Patch(facecolor=c, edgecolor="#33333366", lw=0.5, label=lbl) for c, lbl in entries]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol or len(entries), frameon=False, fontsize=fontsize,
              handlelength=1.2, columnspacing=1.5, handletextpad=0.5, borderaxespad=0.0)


def save_fig(fig, name, keep_titles=False, **kw):
    """Save a matplotlib figure as png+pdf+svg into figures/<fmt>/<name>.<fmt>.

    Panel (axes) titles are stripped by default so each figure carries only its
    panel letters; final titles are added later in slides. A leading panel-letter
    baked into a title (e.g. "a   Study-bundle ...") is preserved as just "a", so
    labels survive whether they are separate text or part of the title. Pass
    keep_titles=True to retain full titles."""
    kw.setdefault("bbox_inches", "tight")
    if not keep_titles:
        for ax in fig.axes:
            for loc in ("left", "center", "right"):
                t = ax.get_title(loc=loc)
                if t:
                    m = re.match(r"^\s*([A-Ha-h])\s{2,}", t)
                    ax.set_title(m.group(1) if m else "", loc=loc)
    for e, d in FIG_DIRS.items():
        fig.savefig(d / f"{name}.{e}", **kw)
    return [str(FIG_DIRS[e] / f"{name}.{e}") for e in FIG_FORMATS]
