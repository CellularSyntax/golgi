"""Regenerates S17 Fig of Bhandari et al., "golgi: an open platform for
image-to-recruitment modeling of peripheral nerve stimulation" (J. Neural Eng., revised).

End-effect taper: characterization of the cosine window and the sensitivity of
activation threshold to electrode placement along the fiber.

Usage:  python make_S17.py
Data is read from ./data (override with GOLGI_REV_DATA); the large cross-solver
inputs are read from ./data_large (override with GOLGI_REV_DATA_LARGE) and are
distributed with the Zenodo record rather than the git repository.
"""
import os

def _d(name):
    return os.path.join(os.environ.get("GOLGI_REV_DATA", "data"), name)

def _L(name):
    return os.path.join(os.environ.get("GOLGI_REV_DATA_LARGE", "data_large"), name)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


tp = np.load(_d("r11_taper.npz"))
CG = "#1f6fb2"; CT = "#c0392b"; CW = "#555555"

def taper(s, frac=0.2):
    s = np.asarray(s, float); lo0 = s.min(); L = s.max() - lo0; a = frac * L; w = np.ones_like(s)
    lo = s < lo0 + a; hi = s > lo0 + L - a
    w[lo] = 0.5 * (1 - np.cos(np.pi * (s[lo] - lo0) / a))
    w[hi] = 0.5 * (1 - np.cos(np.pi * (lo0 + L - s[hi]) / a))
    return w

L = 20.0; s = np.linspace(0, L, 301); sig = 0.2; d = 1e-3; s_m = s * 1e-3; w = taper(s)

def mono(z0):
    v = 1.0 / (4 * np.pi * sig * np.sqrt((s_m - z0)**2 + d**2))
    return v / v.max()

ve_c = mono(0.5 * L * 1e-3); ve_e = mono(0.10 * L * 1e-3)

apply_figure_style()
fig, ax = plt.subplots(1, 2, figsize=(12.2, 5.0))
a = ax[0]
a.axvspan(0, 0.2 * L, color="0.90", zorder=0); a.axvspan(0.8 * L, L, color="0.90", zorder=0)
a.plot(s, w, color=CW, lw=2.4, label="taper window $w(s)$")
a.plot(s, ve_c, color=CG, lw=2.0, label="$V_e$ (mid-fiber source)")
a.plot(s, ve_c * w, color=CG, lw=1.6, ls="--", label="$V_e\\times w$ (applied)")
a.plot(s, ve_e, color=CT, lw=2.0, label="$V_e$ (near-terminal source)")
a.plot(s, ve_e * w, color=CT, lw=1.6, ls="--", label="$V_e\\times w$ (suppressed)")
a.set_ylim(-0.03, 1.62); a.set_xlim(0, L)
a.set_xlabel("position along fiber $s$ (mm)", fontsize=11.5); a.set_ylabel("normalized amplitude", fontsize=11.5)
a.set_title("A  Cosine taper (frac = 0.2) and applied field", loc="left", fontweight="bold", fontsize=12.5)
a.tick_params(labelsize=10.5)
a.legend(frameon=False, fontsize=9.2, loc="upper center", ncol=2, handlelength=1.8, columnspacing=1.3, borderaxespad=0.2)
for xc in (0.1 * L, 0.9 * L): a.text(xc, 1.1, "taper\nzone", ha="center", va="center", fontsize=8.3, color="0.45")
a.text(0.5 * L, 1.1, "central 60% (untouched)", ha="center", va="center", fontsize=8.6, color="0.45")
b = ax[1]
x = tp["frac"] * 100.0
b.axvspan(0, 20, color="0.90", zorder=0)
b.plot(x, tp["thr_notaper"], "-o", color="0.55", lw=1.8, ms=5, label="no taper")
b.plot(x, tp["thr_taper"], "-s", color=CG, lw=1.9, ms=5, label="with taper (golgi)")
b.axvline(20, color="0.5", ls=":", lw=1.1)
b.set_xlim(52, 2)
ymax = max(tp["thr_notaper"].max(), tp["thr_taper"].max())
b.set_ylim(0.10, ymax * 1.16)
b.set_xlabel("source axial position (% of fiber length from end)", fontsize=11.5)
b.set_ylabel("activation threshold (mA)", fontsize=11.5)
b.set_title("B  Threshold vs electrode placement", loc="left", fontweight="bold", fontsize=12.5)
b.tick_params(labelsize=10.5)
b.text(11, ymax * 1.11, "taper zone\n(outer 20%)", ha="center", va="top", fontsize=8.8, color="0.45")
b.text(37, 0.160, "central region: taper effect $\\leq$10%\n($\\leq$2% at mid-fiber)", ha="center", va="center", fontsize=8.8, color="0.4")
b.annotate("near-terminal:\ndrive suppressed\n(+23–40%)", xy=(7.5, tp["thr_taper"][x < 12].max() * 0.98),
           xytext=(27, ymax * 1.05), fontsize=8.8, color=CT, ha="center",
           arrowprops=dict(arrowstyle="->", color=CT, lw=1.1))
b.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.suptitle("End-effect taper: negligible for centered cuffs, only affects near-terminal placements",
             fontweight="bold", fontsize=12.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("supp_endeffect_revision.png", dpi=200, bbox_inches="tight")