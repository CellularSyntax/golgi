# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Bucksot 2019 multi-fascicle recruitment reproduction (golgi), matching ASCENT's
figure (Musselman 2021 reproducing Bucksot et al. 2019 PLoS ONE 14:e0215191).

Columns = the two electrode orientations on the same 5-fascicle rabbit-sciatic nerve:
"Circumferential" (270-deg contact, gap up) vs "Inverted" (gap rotated 180 deg).
Rows:
  a/b — cel-shaded setup render (fascicles coloured per their recruitment curve).
  c/d — per-fascicle recruitment: % activated vs current; golgi (solid) overlaid on
        the digitized Bucksot reference (dashed), paired by threshold order.
  e/f — whole-nerve recruitment, each fiber coloured by MRG diameter.

golgi reproduces the published behaviour: thresholds in the ~0.1-1.5 mA band, large
fibers recruited first, and a near-orientation-independent aggregate (circumferential
uniformity). Reads thr_{circ,inverted}.npz + bucksot_reference.json + bucksot_*.png.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig, DATA, transparent_render   # noqa: E402

WORK = ROOT / "paper_figs/out/_intermediate/bucksot"
PNGDIR = ROOT / "paper_figs/out/renders"
FCOLORS = ["#1f77b4", "#17becf", "#9467bd", "#2ca02c", "#d62728"]   # per-fascicle palette
REFKEY = {"circ": "circumferential", "inverted": "inverted"}        # slug -> reference json key
DMIN, DMAX, XMAX, INK = 2.0, 16.0, 1.5, "#1f2a37"


def load(orient):
    d = np.load(WORK / f"thr_{orient}.npz")
    return d["thr_mA"], d["diam_um"], d["fascicle"]


def recruit_curve(thr, amps):
    t = thr[np.isfinite(thr)]
    return np.array([(t <= a).mean() * 100 for a in amps])


def half_current(x, y):
    """current at 50% recruitment (x in mA, y in %)."""
    x, y = np.asarray(x), np.asarray(y)
    o = np.argsort(x)
    return float(np.interp(50.0, np.clip(y[o], 0, 100), x[o]))


def ref_pairing(thr, fasc, ref_group):
    """Map each digitized reference fascicle (name) to a golgi fascicle id by 50%-current
    order, so paired curves share a colour."""
    gids = sorted(int(g) for g in np.unique(fasc))
    amps = np.linspace(0, XMAX, 300)
    g50 = {g: half_current(amps, recruit_curve(thr[(fasc == g) & np.isfinite(thr)], amps)) for g in gids}
    gorder = sorted(gids, key=lambda g: g50[g])
    names = list(ref_group.keys())
    r50 = {n: half_current(ref_group[n]["thr_mA"], ref_group[n]["pct"]) for n in names}
    rorder = sorted(names, key=lambda n: r50[n])
    return {rorder[k]: gorder[k] for k in range(min(len(gorder), len(rorder)))}   # name -> golgi id


def draw_panel(ax, slug, inset_pos=(0.40, 0.03, 0.60, 0.56), show_legend=True, legend_fs=8.0):
    """Draw one Bucksot orientation on `ax`: per-fascicle recruitment (golgi solid +
    Bucksot dashed, fascicle colours matching the colour-coded render inset, which is
    the fascicle key). Reused by fig_validation."""
    amps = np.linspace(0, XMAX, 300)
    ref = json.loads((DATA / "bucksot_reference.json").read_text())
    thr, diam, fasc = load(slug)
    ok = np.isfinite(thr)
    pair = ref_pairing(thr, fasc, ref[REFKEY[slug]])                  # ref name -> golgi id
    for g in sorted(int(x) for x in np.unique(fasc)):
        m = (fasc == g) & ok
        ax.plot(amps, recruit_curve(thr[m], amps), "-", color=FCOLORS[g % len(FCOLORS)], lw=2.0)
    for name, gid in pair.items():
        r = ref[REFKEY[slug]][name]
        ax.plot(r["thr_mA"], r["pct"], "--", color=FCOLORS[gid % len(FCOLORS)], lw=1.3, alpha=0.85)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 115)            # headroom so the top-right legend clears the plateau
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("% fibers activated"); ax.set_xlabel("activation current (mA)")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=3)
    if show_legend:
        ax.legend(handles=[Line2D([], [], color=INK, lw=2.0, ls="-", label="golgi"),
                           Line2D([], [], color=INK, lw=1.3, ls="--", label="Bucksot 2019")],
                  frameon=False, fontsize=legend_fs, loc="upper right", handlelength=1.8, borderaxespad=0.4)
    png = PNGDIR / f"bucksot_{slug}.png"
    if png.exists() and inset_pos:
        ins = ax.inset_axes(list(inset_pos)); ins.set_zorder(1)
        ins.imshow(transparent_render(png)); ins.axis("off")


def main():
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.9, wspace=0.18)
    for j, (slug, title) in enumerate([("circ", "Circumferential"), ("inverted", "Inverted")]):
        ax = axes[j]
        draw_panel(ax, slug)
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.945, title, ha="center", fontsize=13, fontweight="bold")
        ax.text(-0.12, 1.05, ["a", "b"][j], transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top")
    save_fig(fig, "fig_bucksot", dpi=200, facecolor="white")
    for slug in ("circ", "inverted"):
        thr, _, _ = load(slug); t = thr[np.isfinite(thr)]
        print(f"  {slug}: median {np.median(t):.3f} mA, 90% at {np.percentile(t,90):.3f} mA")
    print("wrote fig_bucksot")


if __name__ == "__main__":
    main()
