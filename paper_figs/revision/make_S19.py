"""Regenerates S19 Fig of Bhandari et al., "golgi: an open platform for
image-to-recruitment modeling of peripheral nerve stimulation" (J. Neural Eng., revised).

Cuff-fit sensitivity: clearance swept 0.05-0.50 mm crossed with golgi's two nerve
section states (undeformed, area-preserving round) on the human cervical vagus.

Usage:  python make_S19.py
Data is read from ./data (override with GOLGI_REV_DATA); the large cross-solver
inputs are read from ./data_large (override with GOLGI_REV_DATA_LARGE) and are
distributed with the Zenodo record rather than the git repository.
"""
import os

def _d(name):
    return os.path.join(os.environ.get("GOLGI_REV_DATA", "data"), name)

def _L(name):
    return os.path.join(os.environ.get("GOLGI_REV_DATA_LARGE", "data_large"), name)

import os
import sys
import json
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Polygon, Patch
from matplotlib.lines import Line2D

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


# golgi's own area-preserving nerve-section deform (the Duke/ASCENT convention).
# Imported from the installed package; set GOLGI_REPO to load it from a source tree.
_repo = os.environ.get("GOLGI_REPO")
if _repo:
    spec = importlib.util.spec_from_file_location(
        "golgi_deform", os.path.join(_repo, "golgi", "segmentation", "deform.py"))
    gd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gd)
else:
    from golgi.segmentation import deform as gd


j = json.load(open(_d("nerve_xsec.json")))
out0 = np.array(j["nerve_outline_xy_um"], float) / 1e3
fas0 = [np.array(f["polygon_xy_um"], float) / 1e3 for f in j["fascicles"]]
M, c = gd.nerve_round_affine(out0)


def rnd(p):
    return gd.apply_round_xy(np.column_stack([p, np.zeros(len(p))]), M, c)[:, :2]


GEO = {"undeformed": (out0, fas0), "rounded": (rnd(out0), [rnd(f) for f in fas0])}


def polyarea(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


CLEAR = [0.05, 0.15, 0.30, 0.50]
cells = {}
for mode, (o, fs) in GEO.items():
    r_max = gd.min_enclosing_circle(o)[1]
    A = polyarea(o)
    for cl in CLEAR:
        R = r_max + cl
        cells[(mode, cl)] = dict(R=R, gap=(1 - A / (np.pi * R**2)) * 100, r_max=r_max)

SUF = {0.05: "c005", 0.15: "c015", 0.30: "c030", 0.50: "c050"}
D = {}
for m in ("none", "round"):
    for cl in CLEAR:
        D[(m, cl)] = json.load(open(_d(f"results/{m}_{SUF[cl]}_n200.json")))

import re


def seed_counts(tag):
    txt = open(_d(f"logs/{tag}.log")).read()
    return [(int(m2.group(1)), int(m2.group(2)))
            for m2 in re.finditer(r"\[fasc (\d+)\] (\d+) seed vertices", txt)]


LAB = {}
TT = {}
for m in ("none", "round"):
    sc = seed_counts(f"{m}_c015_n200")
    lab = np.concatenate([np.full(n, k, int) for k, n in sc])
    LAB[m] = lab
    TT[m] = np.array([np.array(D[(m, cl)]["thresholds_uA"], float) for cl in CLEAR])

MATCH = {}
for m in ("none", "round"):
    T = TT[m]
    ok = np.all(T > 0, axis=0)
    MATCH[m] = ok

from scipy.stats import spearmanr

Tn2 = TT["none"][:, MATCH["none"]]
Tr2 = TT["round"][:, MATCH["round"]]
rho_n2 = spearmanr(Tn2[1], Tn2[3]).statistic
rho_r2 = spearmanr(Tr2[1], Tr2[3]).statistic

FASC2 = {}
for m in ("none", "round"):
    T = TT[m]
    ok = MATCH[m]
    lab = LAB[m]
    Tm = T[:, ok]
    lm = lab[ok]
    FASC2[m] = {}
    for k in range(5):
        sel = lm == k
        n_m = int(sel.sum())
        if n_m == 0:
            continue
        med = np.array([np.median(Tm[i][sel]) for i in range(4)])
        FASC2[m][k] = (n_m, med)

Rmax = max(v["R"] for v in cells.values())
C_SAL = "#cfe9f4"
C_EPI = "#ece0c8"
C_FAS = "#e7a6bb"
C_PERI = "#9c5f74"
C_BORE = "#3d6b86"
CN = "#1f6fb2"
CR = "#c0392b"
F5 = {0: "#1f6fb2", 1: "#c0392b", 2: "#8a5a2c", 3: "#2c8a4a", 4: "#5b3a86"}

handles = [Patch(facecolor=C_SAL, label="saline gap"),
           Line2D([0], [0], color=C_BORE, ls=(0, (5, 3)), lw=1.4, label="cuff bore"),
           Patch(facecolor=C_EPI, label="epineurium"),
           Patch(facecolor=C_FAS, edgecolor=C_PERI, label="fascicle")]

apply_figure_style()
fig = plt.figure(figsize=(13.6, 10.6))
outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.05], hspace=0.24)
gtop = outer[0].subgridspec(2, 4, hspace=0.28, wspace=0.10)
gbot = outer[1].subgridspec(1, 3, wspace=0.30)

for i, (mode, o, fsl) in enumerate((("undeformed", *GEO["undeformed"]), ("rounded", *GEO["rounded"]))):
    cx, cy = o.mean(0)
    for jx, cl in enumerate(CLEAR):
        a = fig.add_subplot(gtop[i, jx])
        v = cells[(mode, cl)]
        Rb = D[("none" if mode == "undeformed" else "round", cl)]["R_bore_mm"]
        a.add_patch(Circle((cx, cy), Rb, facecolor=C_SAL, edgecolor="none", zorder=1))
        a.add_patch(Polygon(o, closed=True, facecolor=C_EPI, edgecolor="#b8a577", lw=1.0, zorder=2))
        for f in fsl:
            a.add_patch(Polygon(f, closed=True, facecolor=C_FAS, edgecolor=C_PERI, lw=0.6, zorder=3))
        a.add_patch(Circle((cx, cy), Rb, facecolor="none", edgecolor=C_BORE, lw=1.2, ls=(0, (5, 3)), zorder=4))
        lim = Rmax * 1.10
        a.set_xlim(cx - lim, cx + lim)
        a.set_ylim(cy - lim * 1.26, cy + lim)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_frame_on(False)
        if i == 0:
            a.set_title(f"clearance {cl:.2f} mm", fontsize=10, fontweight="bold", pad=4)
        if jx == 0:
            a.set_ylabel("undeformed\n(as in paper)" if i == 0 else "area-preserving\nround",
                         fontsize=9.5, fontweight="bold")
            a.text(-0.30, 1.00, "AB"[i], transform=a.transAxes, ha="left", va="top",
                   fontsize=13, fontweight="bold")
        a.text(0.5, 0.03, f"$R_{{bore}}$ {Rb:.3f} mm · saline {v['gap']:.0f}%",
               transform=a.transAxes, ha="center", va="bottom", fontsize=8.0, color="0.3")

a = fig.add_subplot(gbot[0])
for T, cc, lab in ((Tn2, CN, "undeformed"), (Tr2, CR, "rounded")):
    a.plot(CLEAR, np.median(T, axis=1) / 1000, "-o", color=cc, ms=4.5, lw=1.8, label=lab)
    a.fill_between(CLEAR, np.percentile(T, 25, axis=1) / 1000, np.percentile(T, 75, axis=1) / 1000,
                   color=cc, alpha=0.15, lw=0)
a.set_xlabel("cuff clearance (mm)")
a.set_ylabel("threshold (mA)")
a.set_title("Thresholds rise with the gap", fontsize=9.5)
a.legend(frameon=False, fontsize=8.2)
a.margins(0.07)
a.text(-0.19, 1.04, "C", transform=a.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")

a = fig.add_subplot(gbot[1])
a.plot([0.3, 11], [0.3, 11], color="0.7", lw=1.0, zorder=1)
for T, cc in ((Tn2, CN), (Tr2, CR)):
    a.scatter(T[1] / 1000, T[3] / 1000, s=15, color=cc, edgecolor="k", lw=0.25, alpha=0.85, zorder=3)
a.set_xscale("log")
a.set_yscale("log")
a.set_xlabel("threshold @ 0.15 mm (mA)")
a.set_ylabel("threshold @ 0.50 mm (mA)")
a.set_title("Ordering preserved across gap", fontsize=9.5)
a.text(0.97, 0.05, f"Spearman $\\rho$ = {rho_n2:.3f} / {rho_r2:.3f}", transform=a.transAxes,
       va="bottom", ha="right", fontsize=8.0, color="0.3")
a.text(-0.19, 1.04, "D", transform=a.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")

a = fig.add_subplot(gbot[2])
for k in sorted(FASC2["none"]):
    n_, med = FASC2["none"][k]
    a.plot(CLEAR, med / 1000, "-o", color=F5[k], ms=4.2, lw=1.7, label=f"fasc {k} (n={n_})")
a.set_xlabel("cuff clearance (mm)")
a.set_ylabel("threshold (mA)")
a.set_title("All five fascicles, undeformed", fontsize=9.5)
a.legend(frameon=False, fontsize=7.2, ncol=2, columnspacing=0.9)
a.margins(0.07)
a.text(-0.19, 1.04, "E", transform=a.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")

fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.8,
           bbox_to_anchor=(0.5, -0.012))
fig.suptitle("Cuff-fit sensitivity (human cervical vagus, 5 fascicles): the saline gap rescales "
             "thresholds without reordering them", fontweight="bold", fontsize=11.5, y=0.975)
fig.savefig("supp_cuffgap_revision.png", dpi=200, bbox_inches="tight")