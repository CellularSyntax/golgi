"""Regenerates S16 Fig of Bhandari et al., "golgi: an open platform for
image-to-recruitment modeling of peripheral nerve stimulation" (J. Neural Eng., revised).

Cross-solver validation of golgi against COMSOL, extended from the extracellular
field to activation thresholds (M2 idealized cuff).

Usage:  python make_S16.py
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
from scipy.ndimage import gaussian_filter1d

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


# Load data
ep = np.genfromtxt(_L("eval_points.csv"), delimiter=",", names=True)
gr = np.genfromtxt(_L("golgi_Ve_VperA.csv"), delimiter=",", names=True)
G = np.column_stack([gr[c] for c in sorted(gr.dtype.names) if c.startswith("Ve_c")])
C = np.genfromtxt(_L("M2_Results_from_comsol.txt"), comments="%")[:, 3:15]
Cf = np.genfromtxt(_L("M2_Results_from_comsol_finer.txt"), comments="%")[:, 3:15]

fid = ep["fiber_id"].astype(int)
z = ep["z_m"]
kc = 6
vals, counts = np.unique(fid, return_counts=True)
foot = vals[np.argmax(counts)]
s = fid == foot
o = np.argsort(z[s])
zc = z[s][o] * 1e3
gVr = G[s, kc][o]
cVr = C[s, kc][o]
fVr = Cf[s, kc][o]
Lh = 2.7
CG, CC = "#1f6fb2", "#d1602a"

fd = np.load(_d("r12_field_diag.npz"))
th = np.load(_d("r12_thresholds.npz"))


def afpk(zz, ve):
    zu = np.linspace(zz.min(), zz.max(), 400)
    vu = np.interp(zu, zz, ve)
    dz = zu[1] - zu[0]
    vs = gaussian_filter1d(vu, 0.7 / dz, mode="nearest")
    return np.abs(np.gradient(np.gradient(vs, zu * 1e-3), zu * 1e-3)).max()


afr = np.array([afpk(z[fid == fv][np.argsort(z[fid == fv])] * 1e3, C[fid == fv, kc][np.argsort(z[fid == fv])]) /
                afpk(z[fid == fv][np.argsort(z[fid == fv])] * 1e3, G[fid == fv, kc][np.argsort(z[fid == fv])]) for fv in th["fv"].astype(int)])
tpdiff = 100 * np.abs(th["thr_c"] - th["thr_g"]) / th["thr_g"]

apply_figure_style()
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12.5,
                     "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 10.5})
fig, ax = plt.subplots(2, 2, figsize=(11.2, 9.2))
a = ax[0, 0]
a.axvspan(-Lh, Lh, color="0.93", zorder=0)
for e in (-Lh, Lh):
    a.axvline(e, color="0.55", ls=":", lw=1.2, zorder=1)
a.plot(zc, gVr, "-o", color=CG, ms=3.6, lw=1.5, label="golgi (raw)", zorder=4)
a.plot(zc, cVr, "-s", color=CC, ms=3.6, lw=1.4, label="COMSOL (raw)", zorder=3)
a.plot(zc, fVr, ":D", color="0.45", ms=3.0, lw=1.1, label="COMSOL (finer)", zorder=2)
a.set_xlabel("axial position $z$ (mm)")
a.set_ylabel(r"$V_e$ (V A$^{-1}$)")
a.set_title("A  Center-contact potential (raw samples)", loc="left", fontweight="bold")
a.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.3)
a.set_ylim(0, 990)
a.text(0.0, 150, "insulating cuff", ha="center", fontsize=9.8, color="0.5")
a.text(0.97, 0.965, "peak $V_e$ diff 7% (golgi lower);\nno filtering applied", transform=a.transAxes,
       va="top", ha="right", fontsize=9.8, color="0.35")
a.annotate("cuff edge:\nCOMSOL step,\ngolgi smooth", xy=(Lh, 545), xytext=(6.1, 600), fontsize=9.5, color="0.3",
           ha="center", arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
b = ax[0, 1]
zu = fd["zu"]
gA = fd["gAF"] / 1e6
cA = fd["cAF"] / 1e6
fA = fd["fAF"] / 1e6
b.plot(zu, gA, "-", color=CG, lw=2.4, label="golgi")
b.plot(zu, cA, "--", color=CC, lw=2.0, label="COMSOL")
b.plot(zu, fA, ":", color="0.45", lw=1.9, label="COMSOL (finer)")
b.set_xlabel("axial position $z$ (mm)")
b.set_ylabel(r"$d^2V_e/dz^2$ (MV A$^{-1}$ m$^{-2}$)")
b.set_title("B  Activating function", loc="left", fontweight="bold")
b.set_ylim(cA.min() * 1.14, cA.max() * 1.30)
b.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.015, 0.015), handlelength=1.8)
b.text(0.02, 0.03, "COMSOL std vs finer:\n0.6% (mesh-converged)", transform=b.transAxes, va="bottom", ha="left", fontsize=9.5, color="0.35")
c = ax[1, 0]
sc = c.scatter(th["thr_g"], th["thr_c"], s=34, c=th["dist"], cmap="viridis", edgecolor="k", lw=0.3, zorder=3)
lim = [min(th["thr_g"].min(), th["thr_c"].min()) * 0.97, max(th["thr_g"].max(), th["thr_c"].max()) * 1.03]
c.plot(lim, lim, "-", color="0.5", lw=1.2, zorder=1)
c.set_xlim(lim)
c.set_ylim(lim)
c.set_aspect("equal")
c.set_xlabel("golgi threshold (mA)")
c.set_ylabel("COMSOL threshold (mA)")
c.set_title("C  Activation thresholds", loc="left", fontweight="bold")
c.text(0.05, 0.95, "$r$ = 0.997\nratio 0.91 ± 0.01\n$n$ = 45 fibers\n(raw fields)", transform=c.transAxes, va="top",
       fontsize=10.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8"))
cb = fig.colorbar(sc, ax=c, fraction=0.046, pad=0.04)
cb.set_label("dist. to contact (mm)", fontsize=10.5)
cb.ax.tick_params(labelsize=10)
d = ax[1, 1]
d.scatter(afr, tpdiff, s=34, color="#5b3a86", edgecolor="k", lw=0.3, zorder=3)
d.axhline(np.median(tpdiff), color=CC, ls="--", lw=1.4, label=f"median {np.median(tpdiff):.0f}% threshold diff")
d.set_xlabel("activating-function peak ratio (COMSOL/golgi)")
d.set_ylabel("|threshold difference| (%)")
d.set_title("D  Field difference decouples from threshold", loc="left", fontweight="bold")
d.set_ylim(0, max(20, tpdiff.max() * 1.35))
d.legend(frameon=False, loc="upper left")
fig.suptitle("golgi vs COMSOL cross-solver validation — M2 idealized cuff", fontweight="bold", fontsize=15, y=1.002)
fig.tight_layout(w_pad=2.4, h_pad=2.6)
fig.savefig("supp_comsol_revision.png", dpi=200, bbox_inches="tight")