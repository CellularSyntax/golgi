# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""NRV LIFE validation figure.
  a: cel-shaded render of the setup (monofascicular nerve + intrafascicular LIFE).
  b: recruitment vs stimulus current at 20 & 50 µs — golgi (solid) overlaid on the
     digitized NRV-mean curves (dashed, Couppey 2024) and in-vivo NH-1991 points
     (dots); golgi lands in the NH/NRV band, strength-duration rate ratio 2.00 vs
     in-vivo 2.1 / NRV 2.4.

Reads validate_nrv_thr.npz + validate_nrv.json + nrv_reference.json + nrv_setup.png.
Writes figures/*/fig_nrv.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig, DATA, transparent_render   # noqa: E402

WARM, COOL = "#c0392b", "#2f8fb3"      # 50 µs (warm), 20 µs (cool)
INK = "#1f2a37"
RENDER = ROOT / "paper_figs/out/renders/nrv_setup.png"


def recruit(thr_col, amps):
    t = thr_col[np.isfinite(thr_col)]
    return np.array([(t <= a).mean() for a in amps]) if t.size else np.zeros_like(amps)


def draw(axb, inset_pos=(0.43, 0.02, 0.57, 0.56), legend_fs=8.0):
    """Draw the NRV LIFE intensity-sweep panel on `axb` (golgi solid vs NRV dashed vs
    NH-1991 dots), with the LIFE setup render as an inset. Reused by fig_validation."""
    z = np.load(DATA / "validate_nrv_thr.npz")
    thr, pws = z["thr"], z["pws"]                       # thr (n_fibers, n_pw) µA
    ref = json.loads((DATA / "nrv_reference.json").read_text())["panel_a"]
    i20, i50 = list(pws).index(0.02), list(pws).index(0.05)
    amps = np.linspace(0, 180, 360)
    g50, g20 = recruit(thr[:, i50], amps), recruit(thr[:, i20], amps)
    axb.plot(amps, g50, "-", color=WARM, lw=2.6, zorder=5)
    axb.plot(amps, g20, "-", color=COOL, lw=2.6, zorder=5)
    axb.plot(ref["NRV_50us"]["x"], ref["NRV_50us"]["y"], "--", color=WARM, lw=1.5, alpha=0.8, zorder=3)
    axb.plot(ref["NRV_20us"]["x"], ref["NRV_20us"]["y"], "--", color=COOL, lw=1.5, alpha=0.8, zorder=3)
    axb.plot(ref["NH1991_50us"]["x"], ref["NH1991_50us"]["y"], "o", color=WARM, ms=5.5, mec="white", mew=0.7, zorder=4)
    axb.plot(ref["NH1991_20us"]["x"], ref["NH1991_20us"]["y"], "o", color=COOL, ms=5.5, mec="white", mew=0.7, zorder=4)
    axb.set_xlabel("stimulus current (µA)"); axb.set_ylabel("recruitment (norm.)")
    axb.set_xlim(0, 180); axb.set_ylim(-0.02, 1.04)
    axb.spines[["top", "right"]].set_visible(False); axb.tick_params(length=3)
    handles = [Line2D([], [], color=WARM, lw=2.6, ls="-", label="golgi · 50 µs"),
               Line2D([], [], color=COOL, lw=2.6, ls="-", label="golgi · 20 µs"),
               Line2D([], [], color=INK, lw=1.5, ls="--", alpha=0.8, label="NRV (Couppey 2024)"),
               Line2D([], [], color=INK, marker="o", ls="none", ms=5.5, label="in-vivo (NH 1991)")]
    axb.legend(handles=handles, frameon=False, fontsize=legend_fs, loc="upper left",
               handlelength=1.8, borderaxespad=0.6)
    if RENDER.exists() and inset_pos:
        ins = axb.inset_axes(list(inset_pos)); ins.set_zorder(1)
        ins.imshow(transparent_render(RENDER)); ins.axis("off")


def main():
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axb = plt.subplots(figsize=(6.6, 4.6))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.13, top=0.96)
    draw(axb)
    save_fig(fig, "fig_nrv", dpi=200, facecolor="white")
    print("wrote fig_nrv")


if __name__ == "__main__":
    main()
