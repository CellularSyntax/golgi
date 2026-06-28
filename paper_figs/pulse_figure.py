# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Elegant, minimal figure of golgi's stimulation pulse types (the pulse designer).

Faithful to golgi's pulse model (app.build_pulse_waveform): a cathodic phase (+)
optionally followed by a gap and an anodic phase (-), repeated as a train. Warm fill
= cathodic (depolarizing) phase, cool fill = anodic (recovery) phase."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
WARM, COOL, LINE, BASE = "#e8643c", "#2f8fb3", "#1a1a1a", "#cfcfcf"


def train(t, cath_pw, gap, anod_amp, anod_pw, period, n, t0=0.2, cath_amp=1.0):
    """A pulse train sampled on t (ms). Cathodic (+cath_amp) then optional anodic
    (-anod_amp), repeated every `period` ms, `n` times."""
    w = np.zeros_like(t)
    for k in range(n):
        s = t0 + k * period
        w[(t >= s) & (t < s + cath_pw)] = cath_amp
        a = s + cath_pw + gap
        if anod_amp > 0 and anod_pw > 0:
            w[(t >= a) & (t < a + anod_pw)] = -anod_amp
    return w


def main():
    # single stimuli (trains are not yet simulated) — one pulse per design
    t = np.linspace(0, 1.2, 4000)
    specs = [
        ("Monophasic",                     train(t, 0.20, 0.00, 0.0, 0.0, 1.0, 1, t0=0.30)),
        ("Biphasic — symmetric",           train(t, 0.20, 0.00, 1.0, 0.20, 1.0, 1, t0=0.30)),
        ("Biphasic — uneven pulse widths", train(t, 0.20, 0.02, 0.34, 0.59, 1.0, 1, t0=0.30)),
        ("Biphasic — interphase gap",      train(t, 0.20, 0.12, 1.0, 0.20, 1.0, 1, t0=0.30)),
    ]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.8))
    for ax, (title, w) in zip(axes.ravel(), specs):
        ax.axhline(0, color=BASE, lw=1.0, zorder=0)
        ax.fill_between(t, 0, w, where=w > 0, color=WARM, alpha=0.92, lw=0, interpolate=True)
        ax.fill_between(t, 0, w, where=w < 0, color=COOL, alpha=0.92, lw=0, interpolate=True)
        ax.plot(t, w, color=LINE, lw=1.7, solid_joinstyle="round")
        ax.set_ylim(-1.45, 1.45); ax.set_xlim(0, 1.2)
        ax.set_title(title, fontsize=13, color="#1a1a1a", pad=9, loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
        ax.spines["bottom"].set_color("#999")
        ax.set_xticks([0, 0.5, 1.0]); ax.tick_params(colors="#777", labelsize=9.5)
    for ax in axes[1]:
        ax.set_xlabel("time (ms)", fontsize=11, color="#444")
    fig.legend(handles=[Patch(facecolor=WARM, label="cathodic phase"),
                        Patch(facecolor=COOL, label="anodic phase")],
               loc="upper right", bbox_to_anchor=(0.995, 1.0), frameon=False,
               fontsize=10.5, handlelength=1.1, ncol=2)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.12,
                        wspace=0.10, hspace=0.55)
    out = ROOT / "paper_figs/out/figures/png/pulse_types.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
