# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Per-nerve integrated results for the extruded swine + human cervical vagus.
fig5 = swine, fig6 = human — unified 7-panel layout (parallel to the rabbit, fig7):
  a render (full width)
  b population by class | c recruitment by class | d targeting cross-section
  e steering window     | f selectivity barplot  | g evoked AP propagation
Reuses population panels from fig5_population.py and selectivity panels from
fig06_selectivity.py; the selectivity barplot (f) is added here.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig, DATA   # noqa: E402
import fig5_population as fp           # noqa: E402
import fig06_selectivity as fs        # noqa: E402
from fig5_population import UNMYEL, classify, ORDER, CLASS_COL   # noqa: E402

SPECIES = [
    dict(name="Swine", out="fig5_swine", slug="swine",
         recruit=DATA / "thr_swine4_recruit.npz", sel=DATA / "thr_swine4.npz",
         nd="sub-4_sam-3", apkey="swine"),
    dict(name="Human", out="fig6_human", slug="human",
         recruit=DATA / "thr_human50_recruit.npz", sel=DATA / "thr_human50.npz",
         nd="human_sub-50_sam-2", apkey="human",
         # Panels f/g population re-derived on the reseed set (contact 10): the
         # published human_sub-50_sam-2 generation is gone; the reseed is the same
         # base population + a densified target fascicle (branch 9: 35->235 fibers)
         # so per-class selectivity has enough fibers. Non-target counts unchanged.
         pop=DATA / "thr_pop_human_reseed1200.npz"),
]


def selectivity_bar(ax, T):
    """On- vs off-target recruitment at the selective operating point (panel f)."""
    on, off = T["on_at_op"] * 100, T["off_at_op"] * 100
    ax.bar([0, 1], [on, off], width=0.62, color=[fs.C_ON_ACT, fs.C_OFF_ACT], edgecolor="none")
    for x, v in [(0, on), (1, off)]:
        ax.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["on-target", "off-target"])
    ax.set_ylabel("% recruited at operating point"); ax.set_ylim(0, 108)
    ax.set_xlim(-0.6, 1.6)
    ax.text(0.97, 0.95, f"selectivity index {T['si']:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="0.2")


def build(sp, AP):
    nm = sp["name"]
    d = np.load(sp["recruit"], allow_pickle=True)
    diam = d["diameter_um"]; model = d["model"].astype(str)
    typ = classify(diam, np.isin(model, list(UNMYEL))); thr1 = d["thr_uA"][:, 0]
    D = fs.load(sp["sel"], sp["nd"]); S = fs.selectivity(D); T = fs.select_target(D, S)
    pct = (S["best_si"] > 0.5).mean() * 100

    # ---- re-seeded population at the best contact (panels f, g) ----
    # The target fascicle was densified (reseed_target_fascicle.py) so the per-class
    # selectivity bar has enough fibers per class; threshold under the best contact.
    pop = np.load(sp.get("pop", DATA / f"thr_pop_{sp['slug']}.npz"), allow_pickle=True)
    pthr = pop["thr_uA"][:, 0]
    on01 = (pop["branch_idx"].astype(int) == T["kt"]).astype(int)
    ptyp = classify(pop["diameter_um"], np.isin(pop["model"].astype(str), list(UNMYEL)))
    pdiam = pop["diameter_um"]
    _fin = pthr[np.isfinite(pthr)]
    _amps = np.logspace(np.log10(max(_fin.min(), 1.0)), np.log10(_fin.max()), 200)
    _on = fs._rfrac_pct(pthr[on01 == 1], _amps); _off = fs._rfrac_pct(pthr[on01 == 0], _amps)
    _iop = int(np.argmax(_on - _off)); op_amp_mA = float(_amps[_iop] / 1e3)
    all_si = float((_on[_iop] - _off[_iop]) / 100.0)
    # Panels d (cross-section) + e (window) keep the clean uniform-10um geometric
    # targeting (D, S, T); panels f (per-class) + g (threshold-vs-diameter) use the
    # realistic population computed above.

    plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 12.5, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False})
    # full-width render on top (like the rabbit), then 2x3 data panels below;
    # render row tall enough that the ~3.5:1 wide render spans the full figure width
    fig = plt.figure(figsize=(14, 11.6))
    gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.32, height_ratios=[1.32, 1.0, 1.0],
                  top=0.985, bottom=0.05, left=0.05, right=0.975)
    fp.render_panel(fig.add_subplot(gs[0, :]), sp["slug"], f"{nm} — image-derived 3D model + cuff", "a")
    fp.diam_panel(fig.add_subplot(gs[1, 0]), diam, typ, f"{nm} — fiber population", "b")
    fp.recruit_panel(fig.add_subplot(gs[1, 1]), thr1, diam, typ, nm, "c",
                     legend_anchor=((1.19 if nm == "Human" else 1.16), 0.0))  # pushed into the inter-panel gap so it clears the curves
    axd = fig.add_subplot(gs[1, 2]); fs.panel_xsec(axd, D, S, T, nm); fp._lab(axd, "d", "")
    # on/off-target x activated/silent legend below the cross-section (parallels fig7/8 panel d)
    _xh = [Line2D([], [], marker="o", ls="", mfc=cc, mec="none", ms=7, label=ll) for cc, ll in
           [(fs.C_ON_ACT, "on-target act."), (fs.C_ON_SIL, "on-target silent"),
            (fs.C_OFF_ACT, "off-target act."), (fs.C_OFF_SIL, "off-target silent")]]
    axd.legend(handles=_xh, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
               fontsize=9.5, frameon=False, handletextpad=0.3, columnspacing=1.2)
    axe = fig.add_subplot(gs[2, 0]); fs.panel_window(axe, S, T); fp._lab(axe, "e", "")
    axf = fig.add_subplot(gs[2, 1])
    fs.panel_class(axf, pthr, on01, ptyp, all_si, ORDER, CLASS_COL,
                   ylabel="selectivity index (class vs off-target)"); fp._lab(axf, "f", "")
    axg = fig.add_subplot(gs[2, 2])
    fs.panel_thrdiam(axg, pdiam, pthr, on01 == 1, op_amp_mA,
                     on_label="on-target (fascicle)", off_label="off-target"); fp._lab(axg, "g", "")

    save_fig(fig, sp["out"], dpi=200, facecolor="white")
    print(f"wrote {sp['out']} ({nm}: target fasc {T['kt']} c{T['cid']}, op {T['amp_op']:.0f}uA "
          f"on {T['on_at_op']*100:.0f}%/off {T['off_at_op']*100:.0f}%, {pct:.0f}% steerable)")


def main():
    apf = DATA / "fig5_ap.npz"
    AP = dict(np.load(apf, allow_pickle=True)) if apf.exists() else {}
    want = {a.lower() for a in sys.argv[1:]}   # e.g. "human" / "swine" -> only that species
    for sp in SPECIES:
        if want and sp["slug"] not in want and sp["name"].lower() not in want:
            continue
        build(sp, AP)


if __name__ == "__main__":
    main()
