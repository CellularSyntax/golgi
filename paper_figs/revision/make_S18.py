"""Regenerates S18 Fig of Bhandari et al., "golgi: an open platform for
image-to-recruitment modeling of peripheral nerve stimulation" (J. Neural Eng., revised).

Stimulus engine: rectangular pulses from the GUI pulse designer and arbitrary
user-defined waveforms driven through the scripting API.

Usage:  python make_S18.py
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
import matplotlib as mpl

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


dt = 0.005

def grid(tstop):
    return np.arange(0, tstop + dt, dt)

def w_mono(t, t0=0.5, pw=0.1):
    a = np.zeros_like(t); a[(t >= t0) & (t < t0 + pw)] = -1.0; return a

def w_bsym(t, t0=0.5, pw=0.1):
    a = np.zeros_like(t); a[(t >= t0) & (t < t0 + pw)] = -1.0; a[(t >= t0 + pw) & (t < t0 + 2 * pw)] = 1.0; return a

def w_basym(t, t0=0.5, pw=0.1):
    a = np.zeros_like(t); a[(t >= t0) & (t < t0 + pw)] = -1.0; a[(t >= t0 + pw) & (t < t0 + 4 * pw)] = 1 / 3; return a

def w_qt(t, t0=0.5, pw=0.1, ramp=0.4):
    a = np.zeros_like(t); a[(t >= t0) & (t < t0 + pw)] = -1.0
    m = (t >= t0 + pw) & (t < t0 + pw + ramp); a[m] = 0.25 * (1 - (t[m] - (t0 + pw)) / ramp); return a

def w_khz(t, t0=0.5, f=10.0, burst=2.0):
    a = np.zeros_like(t); m = (t >= t0) & (t < t0 + burst); a[m] = np.sin(2 * np.pi * f * (t[m] - t0)); return a

w = np.load(_d("r14_waves.npz"))
vm = w["vm_k"]; tv = w["tv_k"]; ctr = vm.shape[0] // 2

apply_figure_style()
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9))
t = grid(1.4); tk = grid(2.8); CN = "#1f6fb2"; CGN = "#2c8a4a"; CBR = "#8a5a2c"; CE = "#c0392b"; CEND = "#33648f"
a = ax[0]
a.plot(t, w_mono(t), color=CN, lw=1.8); a.plot(t, w_bsym(t) - 2.6, color=CGN, lw=1.8); a.plot(t, w_basym(t) - 5.2, color=CBR, lw=1.8)
a.set_yticks([]); a.set_xlabel("time (ms)"); a.set_ylabel("stimulus (a.u.)"); a.set_ylim(-6.5, 1.7); a.set_xlim(0, 1.4)
a.set_title("A  golgi pulse designer (in GUI)", loc="left", fontweight="bold", fontsize=10.5)
a.text(1.38, 0.28, "monophasic", color=CN, fontsize=8.6, ha="right", va="bottom", fontweight="bold")
a.text(1.38, -2.32, "biphasic sym.", color=CGN, fontsize=8.6, ha="right", va="bottom", fontweight="bold")
a.text(1.38, -4.92, "biphasic asym.", color=CBR, fontsize=8.6, ha="right", va="bottom", fontweight="bold")
b = ax[1]
b.plot(tk, w_qt(tk), color="#5b3a86", lw=1.9, label="quasi-trapezoidal")
b.plot(tk, w_khz(tk, f=10, burst=2.0) - 2.6, color=CE, lw=1.9, label="10 kHz (KHFAC)")
b.set_yticks([]); b.set_xlabel("time (ms)"); b.set_ylim(-4.0, 1.7); b.set_xlim(0, 2.8)
b.set_title("B  Arbitrary waveforms (backend)", loc="left", fontweight="bold", fontsize=10.5)
b.legend(frameon=False, fontsize=8.5, loc="upper right")
c = ax[2]
c.plot(tv, vm[0], color=CEND, lw=1.3, label="fiber end (at rest)", zorder=3)
c.plot(tv, vm[ctr], color=CE, lw=0.8, label="under electrode", zorder=2)
c.set_xlabel("time (ms)"); c.set_ylabel("$V_m$ (mV)"); c.set_ylim(-165, 205); c.set_xlim(tv.min(), tv.max())
c.set_title("C  Fiber response to 10 kHz drive", loc="left", fontweight="bold", fontsize=10.5)
c.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle("golgi stimulus engine: rectangular pulses (GUI) and arbitrary user-defined waveforms (API backend)",
             fontweight="bold", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("supp_waveforms_revision.png", dpi=200, bbox_inches="tight")
fig.savefig("supp_waveforms_revision.pdf", bbox_inches="tight")
print("panel C clean: two-node legend, note removed")