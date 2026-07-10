# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Final integrated validation figure — golgi vs commercial solver, experiment, models.

Solver fidelity first (numerical correctness is logically prior to recruitment), then
physiology:
  a  M1 monopole in saline: golgi = analytic = COMSOL (numerical accuracy)
  b  M3 real swine cervical vagus: golgi vs COMSOL lead fields, point-for-point
  c  in-vivo dog cervical VNS (Yoo 2013 / ASCENT) — cuff, A/B/C fibers
  d  NRV LIFE recruitment (Couppey 2024 / Nannini-Horch 1991) — intrafascicular wire
  e  Bucksot 2019 multifascicular cuff — circumferential
  f  Bucksot 2019 multifascicular cuff — inverted (270-deg contact rotated 180 deg)

Panels a/b reuse comsol_validation_fig (full M1/M2/M3 detail stays in supp_comsol).
Panels c-f reuse the standalone figures' draw() so they stay in sync.
(CV-vs-diameter and strength-duration — the MRG fiber-model checks — live in the
separate fig_validation foundations supplement.)
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig   # noqa: E402
import validate_fig as vf       # noqa: E402  (panel_dog)
import comsol_validation_fig as cvf  # noqa: E402  (panel_m1, panel_scatter, COMSOL data)
import fig_nrv                  # noqa: E402  (draw)
import fig_bucksot             # noqa: E402  (draw_panel)


def _lab(ax, letter):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", ha="right")


def panel_comsol_scatter(ax):
    """M3 swine: golgi vs COMSOL lead fields, with the cross-model per-contact summary."""
    md, tag = "M3_swine_sub-4_sam-3", "M3"
    G, C = cvf.golgi_ve(md, "golgi_Ve_VperA.csv"), cvf.comsol_ve(tag)
    cvf.panel_scatter(ax, G, C, "M3 · swine vagus vs COMSOL (40 fasc.)")
    m2 = cvf.per_contact_pct(cvf.golgi_ve("M2_idealized_cuff", "golgi_Ve_VperA.csv"),
                             cvf.comsol_ve("M2")).mean()
    m3 = cvf.per_contact_pct(G, C).mean()
    ax.text(0.96, 0.05, f"mean per-contact $|\\Delta|$ vs COMSOL:\nM2 cuff {m2:.1f}%   "
            f"M3 swine {m3:.1f}%", transform=ax.transAxes, fontsize=6.8, va="bottom",
            ha="right", color="0.35")


def main():
    plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(15.0, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30, top=0.95, bottom=0.08,
                          left=0.055, right=0.975)

    # row 1 — solver fidelity (a,b) + first physiology panel (c)
    axa = fig.add_subplot(gs[0, 0]); cvf.panel_m1(axa); _lab(axa, "a")
    axb = fig.add_subplot(gs[0, 1]); panel_comsol_scatter(axb); _lab(axb, "b")
    axc = fig.add_subplot(gs[0, 2]); vf.panel_dog(axc, inset_pos=(0.50, 0.02, 0.50, 0.44), legend_fs=7.0); _lab(axc, "c")
    # row 2 — physiology (d,e,f)
    axd = fig.add_subplot(gs[1, 0]); fig_nrv.draw(axd, inset_pos=(0.45, 0.03, 0.55, 0.52), legend_fs=7.0); _lab(axd, "d")
    axe = fig.add_subplot(gs[1, 1]); fig_bucksot.draw_panel(axe, "circ", inset_pos=(0.40, 0.03, 0.60, 0.58), legend_fs=7.0); _lab(axe, "e")
    axf = fig.add_subplot(gs[1, 2]); fig_bucksot.draw_panel(axf, "inverted", inset_pos=(0.40, 0.03, 0.60, 0.58), legend_fs=7.0); _lab(axf, "f")

    save_fig(fig, "fig4_validation", dpi=200, facecolor="white")
    print("wrote fig04_validation")


if __name__ == "__main__":
    main()
