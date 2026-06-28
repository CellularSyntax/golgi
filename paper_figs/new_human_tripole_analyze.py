# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Analyze NEURON thresholds for the SCB-targeting multipolar configs: per-pattern
recruitment (ROC) curve + max-margin selectivity (max over amplitude of %SCB-%trunk).
Compares mono / longitudinal-tripole / transverse / full-guard against each other and the
no-selectivity diagonal.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
THR = ROOT / "paper_figs/out/data/new_human_tripole_thr.npz"
SRC = ROOT / "paper_figs/out/data/new_human_tripole/paths_Ve.npz"
OUTP = ROOT / "paper_figs/out/figures/png/new_human_tripole_selectivity.png"

d = np.load(THR, allow_pickle=True)
thr = d["thr_uA"]; b = d["branch_idx"]
names = list(np.load(SRC, allow_pickle=True)["pattern_names"])
n_scb = int((b == 1).sum()); n_tr = int((b == 0).sum())
COL = {"mono": "#1f77b4", "long_tripole": "#e6550d", "trans_tripole": "#2ca02c",
       "full_guard": "#9467bd"}

fig, ax = plt.subplots(figsize=(6, 5.6))
print(f"{'pattern':14s} {'recruit%':>8} {'max-margin':>10} {'op: %SCB @ %trunk':>20} {'SCBthr/trunkthr':>16}")
for j, nm in enumerate(names):
    t = thr[:, j]
    rec = np.isfinite(t).mean()
    amps = np.unique(t[np.isfinite(t)])
    best = (-1, 0, 0, 0)
    for I in amps:
        fs = np.mean(t[b == 1] <= I); ft = np.mean(t[b == 0] <= I)
        if fs - ft > best[0]:
            best = (fs - ft, fs, ft, I)
    ts = t[b == 1]; tt = t[b == 0]
    med_ratio = np.nanmedian(ts) / np.nanmedian(tt)
    print(f"{nm:14s} {rec*100:7.0f}% {best[0]:10.3f}   SCB {best[1]*100:3.0f}% @ trunk {best[2]*100:3.0f}%   "
          f"{med_ratio:6.2f}")
    aa = np.linspace(0, np.nanmax(t[np.isfinite(t)]), 400)
    fsc = [np.mean(t[b == 1] <= I) for I in aa]; ftr = [np.mean(t[b == 0] <= I) for I in aa]
    ax.plot(np.array(ftr) * 100, np.array(fsc) * 100, lw=2.4, color=COL.get(nm, "k"),
            label=f"{nm} (margin {best[0]:.2f})")
ax.plot([0, 100], [0, 100], "--", c="gray", lw=0.8, label="no selectivity")
ax.set_xlabel("% trunk fibers recruited"); ax.set_ylabel("% SCB fibers recruited")
ax.set_title("SCB selectivity: multipolar cuff configs (real NEURON)\n(4x5 cuff, SCB-side column, ~20 mm from bifurcation)")
ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal"); fig.tight_layout()
OUTP.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUTP, dpi=200)
print(f"\nwrote {OUTP}")
