# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 4 — fiber populations and their recruitment (Nature-minimalist).
a-d  diameter distributions by fiber class (2 cuff-simulated nerves + 2 real-3D,
     the latter illustrative); e-f recruitment-by-class (Aalpha->...->C order,
     contact c05); g-h threshold vs diameter.
Fibers in EVERY panel/nerve are classified uniformly by diameter + myelination
into the standard Aalpha/Abeta/Adelta/B/C classes for consistency.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig, DATA   # noqa: E402
UNMYEL = {"SUNDT", "TIGERHOLM", "RATTAY", "SCHILD94", "SCHILD97"}
CLIP = {"MRG_INTERPOLATION": (2.0, 16.0), "SMALL_MRG_INTERPOLATION": (1.5, 5.0)}
HI_UA = 80_000.0   # matches the 1 ms / 80 mA recruit ceiling so C-fibers are shown
ORDER = ["Aα", "Aβ", "Aδ", "B", "C"]              # Aalpha Abeta Adelta B C
CLASS_COL = {c: col for c, col in zip(ORDER, plt.cm.viridis(np.linspace(0.05, 0.92, 5)))}
SIM = [("Swine", DATA / "thr_swine4_recruit.npz"),
       ("Human", DATA / "thr_human50_recruit.npz")]
REAL3D = [("Real-3D human", "cervical_vagus_human", 235),
          ("Real-3D rabbit", "cervical_vagus_pig", 468)]


def fiber_class(diam, unmyel):
    """Uniform diameter+myelination classification into standard fiber classes."""
    if unmyel:
        return "C"
    if diam >= 10.0:
        return "Aα"
    if diam >= 7.0:
        return "Aβ"
    if diam >= 4.0:
        return "Aδ"
    return "B"


def classify(diam, unmyel):
    return np.array([fiber_class(d, u) for d, u in zip(diam, unmyel)], object)


def present(typ):
    return [t for t in ORDER if (typ == t).any()]


def sample_preset(name, n_total, n_c=40, seed=1):
    from golgi.state_defaults.pop_presets import POP_PRESETS
    rows = POP_PRESETS[name].templates[0].rows
    myel = [r for r in rows if r.model not in UNMYEL]
    crow = [r for r in rows if r.model in UNMYEL][0]
    rng = np.random.default_rng(seed); n_c = min(n_c, n_total); n_my = n_total - n_c
    w = np.array([r.frac for r in myel], float); w /= w.sum()
    diam, unmyel = [], []
    for k in rng.choice(len(myel), size=n_my, p=w):
        r = myel[k]; lo, hi = CLIP.get(r.model, (1, 16))
        diam.append(float(np.clip(rng.normal(r.mean_um, r.std_um), lo, hi))); unmyel.append(False)
    for _ in range(n_c):
        diam.append(float(np.clip(rng.normal(crow.mean_um, crow.std_um), 0.25, 2))); unmyel.append(True)
    diam = np.array(diam); unmyel = np.array(unmyel)
    return diam, classify(diam, unmyel)


def diam_panel(ax, diam, typ, label, letter, illus=False, legend_loc="upper left"):
    bins = np.logspace(np.log10(0.3), np.log10(18), 34); bottom = np.zeros(len(bins) - 1)
    for t in present(typ):
        h, _ = np.histogram(diam[typ == t], bins=bins)
        ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
               color=CLASS_COL[t], label=f"{t} (n={(typ == t).sum()})", edgecolor="none")
        bottom += h
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
    ax.set_xlabel("diameter (µm)"); ax.set_ylabel("count")
    ax.legend(fontsize=8, loc=legend_loc, frameon=False)
    _lab(ax, letter, label)


def recruit_panel(ax, thr1, diam, typ, label, letter, legend_loc="lower right",
                  legend_anchor=(1.0, 0.0)):
    amps = np.logspace(np.log10(20), np.log10(HI_UA), 80)
    for t in present(typ):
        m = typ == t; n = int(m.sum()); tm = thr1[m]
        frac = np.array([100 * (np.isfinite(tm) & (tm <= a)).sum() / max(n, 1) for a in amps])
        ax.semilogx(amps / 1e3, frac, color=CLASS_COL[t], lw=2.2, label=f"{t} (n={n})")
    ax.set_xlabel("stimulus amplitude (mA)"); ax.set_ylabel("% recruited")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc=legend_loc, bbox_to_anchor=legend_anchor,
              borderaxespad=0.0, frameon=False)
    _lab(ax, letter, f"{label} — recruitment by class")


def thrdiam_panel(ax, thr1, diam, typ, label, letter):
    rec = np.isfinite(thr1)
    for t in present(typ):
        m = (typ == t) & rec
        if m.any():
            ax.scatter(diam[m], thr1[m] / 1e3, s=12, color=CLASS_COL[t], alpha=0.6,
                       label=f"{t} (n={(typ == t).sum()})", edgecolors="none")
    if (~rec).any():
        ax.scatter(diam[~rec], np.full((~rec).sum(), HI_UA / 1e3 * 1.3), s=12, marker="^",
                   color="0.6", alpha=0.5, label="not recruited")
    ax.axhline(HI_UA / 1e3, color="0.7", ls=":", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("diameter (µm)"); ax.set_ylabel("threshold (mA)")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.92),
              borderaxespad=0.0, frameon=False)
    _lab(ax, letter, f"{label} — threshold vs diameter")


def _lab(ax, letter, title):
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="right")


POP_LEGEND = [("#e3a7c6", "Epineurium"), ("#5a2d96", "Fascicle (endoneurium)"),
              ("#d63a3a", "Axons"), ("#4f93cf", "Cuff (silicone)"),
              ("#f1c62f", "Contact"), ("#8fd0ee", "Saline")]


def render_panel(ax, slug, label, letter):
    img = ROOT / "paper_figs/out/renders/popnerve" / f"{slug}.png"
    ax.axis("off")
    if img.exists():
        from io_paths import load_ppmm, draw_scalebar, render_legend
        im = plt.imread(str(img)); ax.imshow(im)
        draw_scalebar(ax, im.shape[1], im.shape[0], load_ppmm(img))
        render_legend(ax, POP_LEGEND)
    else:
        ax.text(0.5, 0.5, f"{slug} render\n(run render_popnerve.py)", ha="center",
                va="center", transform=ax.transAxes, color="0.5")
    _lab(ax, letter, label)


def main():
    if any(not p.exists() for _, p in SIM):
        print("waiting for recruit matrices:", [str(p) for _, p in SIM]); return
    simdata = []
    for nm, p in SIM:
        d = np.load(p, allow_pickle=True)
        diam = d["diameter_um"]; model = d["model"].astype(str)
        unmyel = np.isin(model, list(UNMYEL))
        simdata.append((nm, diam, classify(diam, unmyel), d["thr_uA"][:, 0]))

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    # two per-nerve rows: cel-shaded 3D model | population by class | recruitment
    fig = plt.figure(figsize=(13.6, 8.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.27, width_ratios=[1.28, 1.0, 1.0],
                  top=0.92, bottom=0.09, left=0.04, right=0.985)
    slugs = ["swine", "human"]
    lets = [("a", "b", "c"), ("d", "e", "f")]
    for row, (nm, dia, typ, thr1) in enumerate(simdata):
        render_panel(fig.add_subplot(gs[row, 0]), slugs[row],
                     f"{nm} — image-derived 3D model + cuff", lets[row][0])
        diam_panel(fig.add_subplot(gs[row, 1]), dia, typ, f"{nm} — fiber population", lets[row][1])
        recruit_panel(fig.add_subplot(gs[row, 2]), thr1, dia, typ, nm, lets[row][2])

    save_fig(fig, "fig5_populations", dpi=200, facecolor="white")
    print("wrote fig04_populations (2x3: render | diameter | recruitment per nerve)")
    for nm, dia, typ, thr1 in simdata:
        print(f"  {nm}: classes {dict((t, int((typ == t).sum())) for t in present(typ))}")


if __name__ == "__main__":
    main()
