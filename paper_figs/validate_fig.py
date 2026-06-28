# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fiber-model foundations figure (supplementary) — golgi vs literature.
  a  conduction velocity vs diameter (golgi vs Hursh 6 m/s/um + McIntyre-2002 MRG)
  b  strength-duration: threshold vs pulse width + Weiss fit (rheobase, chronaxie)

These are the two MRG fiber-model checks. The numerical solver checks now live with
the field solver: the analytic monopole + COMSOL cross-validation are in the main
integrated figure (fig_validation_full.py, panels a/b) and supp_comsol_validation.py.
panel_fem / panel_dog below are retained for reuse by other figures.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig, DATA, transparent_render   # noqa: E402
GOLGI = "#1a7fb5"
REF = "#444444"


def _lab(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", ha="right")


def panel_cv(ax):
    d = json.loads((DATA / "validate_fiber.json").read_text())
    cv = [c for c in d["cv_diameter"] if c["cv"]]
    diam = np.array([c["diam"] for c in cv]); g = np.array([c["cv"] for c in cv])
    mrg = np.array([c["mrg_ref"] for c in cv if c["mrg_ref"]])
    dm = np.array([c["diam"] for c in cv if c["mrg_ref"]])
    xs = np.linspace(diam.min(), diam.max(), 50)
    ax.plot(xs, 6.0 * xs, color=REF, ls="--", lw=1.6, label="Hursh bound (6 m s$^{-1}$ µm$^{-1}$)")
    ax.plot(dm, mrg, "s", color="0.55", ms=6.5, label="McIntyre 2002 (discrete MRG)")
    ax.plot(diam, g, "o-", color=GOLGI, lw=2.0, ms=6, label="golgi (interpolated MRG)")
    ax.text(0.04, 0.92, f"slope {d['hursh_slope_fit']:.1f} m s$^{{-1}}$ µm$^{{-1}}$",
            transform=ax.transAxes, fontsize=9, color=GOLGI)
    ax.set_xlabel("fiber diameter (µm)"); ax.set_ylabel("conduction velocity (m s$^{-1}$)")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title("conduction velocity vs diameter", fontsize=11, loc="left")


def panel_sd(ax):
    d = json.loads((DATA / "validate_fiber.json").read_text())["strength_duration"]
    pw = np.array([p["pw_ms"] for p in d["points"] if p["th_mA"]])
    it = np.array([p["th_mA"] for p in d["points"] if p["th_mA"]])
    rheo, chron = d["rheobase_mA"], d["chronaxie_ms"]
    xf = np.linspace(pw.min(), pw.max(), 200)
    ax.plot(xf, rheo * (1 + chron / xf), color=GOLGI, lw=2.0, label="Weiss--Lapicque fit")
    ax.plot(pw, it, "o", color=GOLGI, ms=6)
    ax.axhline(rheo, color=REF, ls=":", lw=1.2)
    ax.axvline(chron, color=REF, ls=":", lw=1.2)
    ax.text(0.98, 0.83, f"rheobase {rheo*1e3:.0f} µA\nchronaxie {chron*1e3:.0f} µs",
            transform=ax.transAxes, fontsize=9, color="0.2", ha="right", va="top")
    ax.set_xlabel("pulse width (ms)"); ax.set_ylabel("threshold (mA)")
    ax.set_ylim(0, max(it) * 1.1)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_title("strength--duration (10 µm myelinated)", fontsize=11, loc="left")


def panel_fem(ax):
    f = json.loads((DATA / "validate_fem_analytic.json").read_text())
    r = np.array(f["r_mm"]); vf = np.array(f["V_fem"]) * 1e3; ve = np.array(f["V_exact"]) * 1e3
    ax.plot(r, ve, color=REF, lw=2.0, label=r"analytic $I/4\pi\sigma r$")
    ax.plot(r, vf, "o", color=GOLGI, ms=6, label="golgi FEM (FEniCSx)")
    ax.text(0.98, 0.78, f"mean error {f['mean_rel_err']*100:.1f}%\n({f['n_nodes']:,} nodes)",
            transform=ax.transAxes, fontsize=9, color="0.2", ha="right", va="top")
    ax.set_xlabel("radial distance (mm)"); ax.set_ylabel("potential (mV)")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_title("FEM solver vs analytic point source", fontsize=11, loc="left")


def panel_dog(ax, inset_pos=(0.50, 0.02, 0.50, 0.40), legend_fs=7.5):
    # in-vivo dog cervical VNS (Yoo et al. 2013, J Neural Eng 10:026003; mean ± SD,
    # n=5 dogs, C-fibers n=4) and ASCENT's modeled thresholds (digitized from their
    # Yoo-benchmark Fig 5B). golgi = this work (ASCENT masks, LivaNova separated cuffs).
    INVIVO = {"A": (0.37, 0.18), "fast B": (1.6, 0.35), "slow B": (3.8, 0.84), "C": (17.0, 7.6)}
    ASC = {"A": 0.533, "fast B": 1.841, "slow B": 3.149, "C": 20.736}
    p = DATA / "validate_dogvns_ascent.json"
    if not p.exists():
        ax.text(0.5, 0.5, "dog VNS\n(run validate_dogvns_ascent.py)", ha="center", va="center",
                transform=ax.transAxes, color="0.5"); ax.set_title("in-vivo dog VNS", fontsize=11, loc="left")
        return
    th = json.loads(p.read_text())["thresholds"]
    g = {t["type"].split(" (")[0]: t["thr_mA"] for t in th}
    order = ["A", "fast B", "slow B", "C"]
    labels = ["A\n7.8 µm", "fast-B\n3.6 µm", "slow-B\n2.1 µm", "C\n1.0 µm"]
    x = np.arange(len(order))
    ax.errorbar(x, [INVIVO[k][0] for k in order], yerr=[INVIVO[k][1] for k in order],
                fmt="none", ecolor="#888888", capsize=5, lw=1.6, zorder=2)
    ax.plot(x, [INVIVO[k][0] for k in order], "_", color="#222222", ms=16, mew=2.2, zorder=3,
            label="Yoo 2013 in vivo (mean ± SD)")
    ax.plot(x, [ASC[k] for k in order], "D", color="#2ca02c", ms=8, zorder=5, label="ASCENT")
    ax.plot(x, [g.get(k) for k in order], "o", color="#1a7fb5", ms=9, zorder=6, mec="white", mew=0.6,
            label="golgi (this work)")
    ax.set_yscale("log"); ax.set_ylim(0.12, 45); ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("activation threshold (mA)")
    ax.legend(fontsize=legend_fs, loc="upper left", frameon=False)
    ax.set_title("in-vivo dog VNS: experiment vs ASCENT vs golgi (300 µs)", fontsize=10, loc="left")
    # cel-shaded dog-VNS device render as an inset, bottom-right (empty region, below
    # the rising data band: slow-B golgi sits at y-fraction ~0.44, so keep the top < 0.40)
    img = ROOT / "paper_figs/out/renders/dogvns_render.png"
    if img.exists() and inset_pos:
        ins = ax.inset_axes(list(inset_pos)); ins.set_zorder(1)
        ins.imshow(transparent_render(img)); ins.axis("off")


def panel_render(ax):
    # cel-shaded device illustration of the dog-VNS model (render_dogvns.py)
    img = ROOT / "paper_figs/out/renders/dogvns_render.png"
    ax.axis("off")
    if img.exists():
        ax.imshow(plt.imread(str(img)))
    else:
        ax.text(0.5, 0.5, "dog-VNS render\n(run render_dogvns.py)", ha="center",
                va="center", transform=ax.transAxes, color="0.5")
    ax.set_title("dog-VNS model: ASCENT nerve + LivaNova separated cuff",
                 fontsize=10, loc="left")


def main():
    # Foundations figure (supplementary): the two MRG fiber-model checks only.
    # The numerical solver checks (analytic monopole + COMSOL) now live with the field
    # solver — main integrated figure (fig_validation_full.py) + supp_comsol_validation.py.
    # The system-level recruitment benchmarks (dog VNS, NRV LIFE, Bucksot) are in the
    # main-text integrated figure, which imports panel_dog above.
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(9, 4.3))
    gs = fig.add_gridspec(1, 2, wspace=0.26, top=0.92, bottom=0.16, left=0.08, right=0.97)
    for cell, fn, lt in [(gs[0], panel_cv, "a"), (gs[1], panel_sd, "b")]:
        ax = fig.add_subplot(cell); fn(ax); _lab(ax, lt)
    save_fig(fig, "supp_foundations", dpi=200, facecolor="white")
    print("wrote fig_validation (foundations: CV-vs-diameter, strength-duration)")


if __name__ == "__main__":
    main()
