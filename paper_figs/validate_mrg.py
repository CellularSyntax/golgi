# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Validation — MRG fiber-model morphometry (ASCENT's Bucksot 2019 check).

golgi simulates myelinated axons through pyfibers (its main backend) using the
MRG_INTERPOLATION model: continuous piecewise-polynomial fits that give every
geometric parameter of the double-cable MRG axon (internode length, paranodal
FLUT length, axon & node diameter, number of myelin lamellae) as a function of
fiber diameter, extending McIntyre-Richardson-Grill 2002's discrete measurements
to arbitrary diameters. This is the interpolation scheme introduced with ASCENT
(Musselman 2021) and validated there against Bucksot et al. 2019 (their Fig. H).

We reproduce that validation directly from golgi's backend: the interpolation
golgi uses vs the canonical McIntyre 2002 discrete morphometry, across 2-16 um.
The fiber geometry underlies every threshold/CV golgi computes, so this anchors
the fiber model before any field is applied.

Writes paper_figs/out/data/validate_mrg.json + figures/*/fig_mrg.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig, DATA   # noqa: E402

INK = "#1f2a37"
DOT = "#c1432e"          # McIntyre discrete (measured)
CURVE = "#2f6db0"        # golgi/pyfibers interpolation

# parameter -> (label, unit, discrete-key, interp callable name)
PARAMS = [
    ("Internode length", "μm", "delta_z"),
    ("Paranode (FLUT) length", "μm", "paranodal_length_2"),
    ("Axon diameter", "μm", "axon_diam"),
    ("Node diameter", "μm", "node_diam"),
    ("Myelin lamellae", "count", "nl"),
]


def load_params():
    from pyfibers.models.mrg import fiber_parameters_all as P
    return P["MRG_DISCRETE"], P["MRG_INTERPOLATION"]


def main():
    disc, interp = load_params()
    dd = np.array(disc["diameters"], float)
    # MRG_INTERPOLATION is calibrated for >= 2 um (SMALL handles smaller); compare there
    sel = dd >= 2.0
    dgrid = np.linspace(2.0, 16.0, 400)

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13, 7.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    letters = "abcdef"
    errs = {}

    for i, (label, unit, key) in enumerate(PARAMS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ycurve = np.array([interp[key](float(d)) for d in dgrid])
        ydisc = np.array(disc[key], float)
        ax.plot(dgrid, ycurve, "-", color=CURVE, lw=2.2, zorder=2,
                label="golgi (pyfibers MRG interp.)")
        ax.plot(dd[sel], ydisc[sel], "o", color=DOT, ms=6.5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8,
                label="McIntyre 2002 (measured)")
        # relative error of interpolation at each measured diameter (>=2 um)
        yint_at = np.array([interp[key](float(d)) for d in dd[sel]])
        e = np.abs(yint_at - ydisc[sel]) / np.maximum(np.abs(ydisc[sel]), 1e-9) * 100
        errs[key] = float(np.mean(e))
        ax.set_ylabel(f"{label}\n({unit})")
        ax.set_xlabel("fiber diameter (μm)")
        ax.set_xlim(1.5, 16.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
        ax.text(-0.16, 1.04, letters[i], transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top", ha="right")
        ax.text(0.97, 0.06, f"mean |err| {errs[key]:.1f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="0.35")
        if i == 0:
            ax.legend(frameon=False, fontsize=7.6, loc="upper left",
                      handlelength=1.4, bbox_to_anchor=(-0.02, 1.02))

    # f: summary bar of mean |error| per parameter
    axf = fig.add_subplot(gs[1, 2])
    keys = [p[2] for p in PARAMS]
    names = ["internode", "FLUT", "axon ⌀", "node ⌀", "lamellae"]
    vals = [errs[k] for k in keys]
    bars = axf.bar(names, vals, color=CURVE, alpha=0.85, width=0.66)
    for b, v in zip(bars, vals):
        axf.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.1f}", ha="center",
                 fontsize=8, fontweight="bold")
    axf.set_ylabel("mean |interp. − measured| (%)")
    axf.set_ylim(0, max(vals) * 1.35 + 0.3)
    axf.spines[["top", "right"]].set_visible(False)
    axf.tick_params(length=3, axis="x", rotation=30)
    axf.text(-0.16, 1.04, letters[5], transform=axf.transAxes, fontsize=15,
             fontweight="bold", va="top", ha="right")

    save_fig(fig, "fig_mrg", dpi=200, facecolor="white")
    overall = float(np.mean(vals))
    (DATA / "validate_mrg.json").write_text(json.dumps(
        {"per_param_mean_abs_pct_err": errs, "overall_mean_abs_pct_err": overall,
         "discrete_diameters_um": disc["diameters"], "model": "MRG_INTERPOLATION (pyfibers)"},
        indent=2))
    print(f"wrote fig_mrg + validate_mrg.json; overall mean |err| = {overall:.2f}% "
          f"across {sel.sum()} measured diameters")
    for k, v in errs.items():
        print(f"  {k:22s} {v:5.2f}%")


if __name__ == "__main__":
    main()
