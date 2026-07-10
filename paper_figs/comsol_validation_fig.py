# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.
"""S15: cross-validation of golgi's open FEniCSx solver against COMSOL.

Four models from comsol_handover/ (the student ran COMSOL independently,
re-extruding and re-meshing each geometry at the finest mesh setting):
  M1  monopole in a saline cylinder  -> golgi & COMSOL Ve(r) vs analytic
  M2  idealized cuff, 7 fascicles, full physics (anisotropy + perineurium CI)
  M3  real swine cervical vagus (sub-4/sam-3)
  M4  real human cervical vagus (sub-47/sam-2) -- COMSOL omitted the perineurium
      contact impedance on the re-extruded geometry, so golgi is matched
      (no perineurium) here: M4 isolates geometry + solver agreement on a
      complex human nerve (the perineurium-CI physics is validated by M2).
For M2-M4 the per-contact lead fields are compared point-for-point at the
identical golgi sample positions (12 contacts, ~1e5 points each).

Output: paper_figs/out/figures/png/supp_comsol_validation.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__import__("os").environ.get("GOLGI_PAPER_ROOT") or Path(__file__).resolve().parents[1])
HAND = ROOT / "comsol_handover"
OUT = ROOT / "paper_figs/out/figures/png/supp_comsol_validation.png"

# ---- palette (consistent, Nature-style) -----------------------------------
C_GOLGI, C_COMSOL, C_ANA = "#2c6fb0", "#d1563f", "#222222"
C_VE_G, C_VE_C = "#2c6fb0", "#86b4d8"          # Ve: golgi / COMSOL (dark / light blue)
C_AF_G, C_AF_C = "#d1563f", "#f0a98f"          # activating fn: golgi / COMSOL (dark / light orange)
C_MODEL = {"M2": "#4c72b0", "M3": "#55a868", "M4": "#c44e52"}
HEX = "#d9e3ef"

# (tag, model dir, title, golgi csv, note)
MODELS = [
    ("M2", "M2_idealized_cuff", "Idealized cuff · full physics", "golgi_Ve_VperA.csv", ""),
    ("M3", "M3_swine_sub-4_sam-3", "Swine cervical vagus (sub-4, 40 fasc.)", "golgi_Ve_VperA.csv", ""),
]


def comsol_ve(tag):
    a = np.genfromtxt(HAND / f"results/{tag}/{tag}_Results_from_comsol.txt", comments="%")
    return a[:, 3:15]


def golgi_ve(md, csv):
    r = np.genfromtxt(HAND / "models" / md / csv, delimiter=",", names=True)
    cols = sorted(c for c in r.dtype.names if c.startswith("Ve_c"))
    return np.column_stack([r[c] for c in cols])


def eval_pts(md):
    r = np.genfromtxt(HAND / "models" / md / "eval_points.csv", delimiter=",", names=True)
    return r["fiber_id"].astype(int), r["z_m"]


def per_contact_pct(G, C):
    out = []
    for k in range(G.shape[1]):
        g, c = G[:, k], C[:, k]
        m = np.isfinite(g) & np.isfinite(c)
        out.append(np.mean(100 * np.abs(c[m] - g[m]) / np.max(np.abs(g[m]))))
    return np.array(out)


def resample_ve_af(zc_mm, ve):
    from scipy.ndimage import gaussian_filter1d
    zu = np.linspace(zc_mm.min(), zc_mm.max(), 400)
    vu = np.interp(zu, zc_mm, ve)
    dz = zu[1] - zu[0]
    vs = gaussian_filter1d(vu, 0.7 / dz, mode="nearest")
    d2 = np.gradient(np.gradient(vs, zu * 1e-3), zu * 1e-3)
    return zu, vu, d2


def _letter(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="bottom", ha="left")


def panel_m1(ax):
    SIG, I, R_CYL = 1.76, 1.0, 12.0e-3
    com = np.genfromtxt(HAND / "results/M1/M1_data_from_comsol.txt", comments="%")
    rc = com[:, 0]
    g = np.load(ROOT / "paper_figs/out/data/m1_golgi_Ve.npz")
    rg, vg = g["r_mm"], g["V_fem"]
    rd = np.linspace(0.25, 7.5, 300); rd_m = rd * 1e-3
    ax.plot(rd, I / (4 * np.pi * SIG * rd_m), ":", color="0.6", lw=1.5,
            label=r"analytic, infinite  $I/4\pi\sigma r$")
    ax.plot(rd, I / (4 * np.pi * SIG) * (1 / rd_m - 1 / R_CYL), "-", color=C_ANA, lw=2.0,
            label="analytic, finite domain")
    ax.plot(rg, vg, "-", color=C_GOLGI, lw=2.6, label="golgi (FEniCSx)", zorder=5)
    ax.plot(rc, com[:, 5], "o", color=C_COMSOL, ms=3.2, mfc="none", mew=0.9,
            label="COMSOL", zorder=4)

    def nf(r_mm, v):
        m = r_mm <= 2.0
        va = I / (4 * np.pi * SIG) * (1 / (r_mm[m] * 1e-3) - 1 / R_CYL)
        return np.mean(100 * np.abs(v[m] - va) / va)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.22, 7.5)
    ax.set_xlabel("radial distance $r$ (mm)"); ax.set_ylabel("$V_e$ (V A$^{-1}$)")
    ax.legend(fontsize=10.5, frameon=False, loc="lower left", handlelength=1.8)
    ax.text(0.97, 0.95, f"near-field ($r\\leq$2 mm) vs analytic:\n"
            f"golgi {nf(rg, vg):.1f}%   COMSOL {nf(rc, com[:, 5]):.1f}%",
            transform=ax.transAxes, fontsize=11, va="top", ha="right")


def panel_scatter(ax, G, C, title, note=""):
    g, c = G.ravel(), C.ravel()
    m = np.isfinite(g) & np.isfinite(c); g, c = g[m], c[m]
    lim = [min(g.min(), c.min()), max(g.max(), c.max())]
    ax.hexbin(g, c, gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0)
    ax.plot(lim, lim, "-", color="0.4", lw=1.1, zorder=3)
    slope = np.polyfit(g, c, 1)[0]
    r2 = 1 - np.sum((c - g) ** 2) / np.sum((g - g.mean()) ** 2)
    rr = np.corrcoef(g, c)[0, 1]
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("golgi $V_e$ (V A$^{-1}$)"); ax.set_ylabel("COMSOL $V_e$ (V A$^{-1}$)")
    ax.text(0.05, 0.95, f"slope {slope:.2f}\n$R^2$ {r2:.3f}\n$r$ {rr:.3f}",
            transform=ax.transAxes, fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    if note:
        ax.text(0.96, 0.05, note, transform=ax.transAxes, fontsize=10, va="bottom",
                ha="right", color="0.4", style="italic")


def panel_footprint(ax, G, C, md, note=""):
    fid, z = eval_pts(md)
    n = min(len(G), len(fid)); G, C, fid, z = G[:n], C[:n], fid[:n], z[:n]
    vals, counts = np.unique(fid, return_counts=True)
    sel = fid == vals[np.argmax(counts)]
    o = np.argsort(z[sel]); zc = z[sel][o] * 1e3
    kc = G.shape[1] // 2
    zu, gvu, gaf = resample_ve_af(zc, G[sel, kc][o])
    _, cvu, caf = resample_ve_af(zc, C[sel, kc][o])
    l1, = ax.plot(zu, gvu, "-", color=C_VE_G, lw=2.4, label="golgi  $V_e$")
    l2, = ax.plot(zu, cvu, "--", color=C_VE_C, lw=2.2, label="COMSOL  $V_e$")
    ax.set_xlabel("axial position $z$ (mm)"); ax.set_ylabel("$V_e$ (V A$^{-1}$)", color=C_VE_G)
    ax.tick_params(axis="y", labelcolor=C_VE_G)
    ax.set_ylim(0, max(gvu.max(), cvu.max()) * 1.34)          # headroom for the legend
    ax2 = ax.twinx()
    l3, = ax2.plot(zu, gaf, "-", color=C_AF_G, lw=2.0, label="golgi  act. fn.")
    l4, = ax2.plot(zu, caf, "--", color=C_AF_C, lw=1.9, label="COMSOL  act. fn.")
    ax2.set_ylabel(r"activating fn. $d^2V_e/dz^2$ (V A$^{-1}$ m$^{-2}$)", color=C_AF_G, fontsize=8.5)
    ax2.tick_params(axis="y", labelcolor=C_AF_G)
    afmax = max(np.abs(gaf).max(), np.abs(caf).max())
    ax2.set_ylim(-afmax * 1.25, afmax * 1.55)
    ax.legend(handles=[l1, l2, l3, l4], fontsize=9, frameon=False,
              loc="upper center", ncol=2, handlelength=2.2, columnspacing=1.3)


def panel_summary(ax):
    data = {tag: per_contact_pct(golgi_ve(md, csv), comsol_ve(tag))
            for (tag, md, _t, csv, _n) in MODELS}
    x = np.arange(12); w = 0.38
    for i, (tag, *_rest) in enumerate(MODELS):
        pct = data[tag]
        ax.bar(x + (i - 0.5) * w, pct, w, color=C_MODEL[tag], zorder=3,
               label=f"{tag}  (mean {pct.mean():.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels([f"{k}" for k in range(12)], fontsize=9.5)
    ax.set_xlabel("contact"); ax.set_ylabel("mean |%diff| vs COMSOL")
    ax.set_ylim(0, max(v.max() for v in data.values()) * 1.24)
    ax.legend(fontsize=10, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=1.4)
    ax.yaxis.grid(True, color="0.9", lw=0.6, zorder=0); ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.9, "xtick.major.width": 0.9,
                         "ytick.major.width": 0.9})
    fig = plt.figure(figsize=(11.6, 12.4))
    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.40,
                  top=0.965, bottom=0.05, left=0.075, right=0.905)
    a = fig.add_subplot(gs[0, 0]); panel_m1(a); _letter(a, "a")
    b = fig.add_subplot(gs[0, 1]); panel_summary(b); _letter(b, "b")
    letters = iter("cdef")
    for row, (tag, md, title, csv, note) in enumerate(MODELS, start=1):
        G, C = golgi_ve(md, csv), comsol_ve(tag)
        axs = fig.add_subplot(gs[row, 0]); panel_scatter(axs, G, C, f"{tag} · {title}", note)
        _letter(axs, next(letters))
        axf = fig.add_subplot(gs[row, 1]); panel_footprint(axf, G, C, md, note)
        _letter(axf, next(letters))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, facecolor="white")
    print(f"wrote {OUT}")
    for (tag, md, _t, csv, _n) in MODELS:
        p = per_contact_pct(golgi_ve(md, csv), comsol_ve(tag))
        print(f"  {tag}: per-contact mean {p.mean():.2f}%  worst {p.max():.2f}%")


if __name__ == "__main__":
    main()
