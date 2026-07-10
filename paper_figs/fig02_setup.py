# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 2 — modeling setup & FEM solution (multifascicular swine nerve, sub-4_sam-3).
16:9 landscape, 2x3:
  a  nerve in 12-contact cuff (golgi's authentic 3-D render)
  b  multi-region FEM mesh — longitudinal section (+ material legend)
  c  cuff designer + PCA autofit (GUI screenshot placeholder)
  d  FEM lead-field V_e on a cuff cross-section (contact c05, z = 0)
  e  FEM E-field |E| + in-plane streamlines on the same cross-section
  f  V_e and activating function (d2V_e/ds2) sampled along the fibers
The field cross-sections (d, e) are the 2-D source of the along-fiber traces (f):
V_e sampled on each fiber -> d2V_e/ds2 sets where the fiber is driven.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Polygon as MplPoly
from matplotlib.colors import LogNorm
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig   # noqa: E402
from fig02_render import render_mesh, MAT, CONTACT_COL   # noqa: E402
ND = ROOT / "results_golgi/duke_meshes/sub-4_sam-3"
CONTACT = 5                                            # central contact, z = 0


def crop(png):
    im = Image.open(png).convert("RGBA"); bb = im.getbbox()
    return np.asarray(im.crop(bb) if bb else im)


def _lab(ax, letter, title):
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="right")


def load_xsec(contact_id=CONTACT):
    """Central-plane (z=0) cross-section grid of V_e and E for one contact (mm)."""
    d = np.load(ND / "xsec_Ve.npz", allow_pickle=True)
    nx = int(d["nx"]); xy = d["xy"]; zp = d["z_planes"]; cids = list(d["contact_ids"])
    p = int(np.argmin(np.abs(zp))); c = cids.index(contact_id)
    sl = slice(p * nx * nx, (p + 1) * nx * nx)
    X = xy[:, 0].reshape(nx, nx) * 1e3; Y = xy[:, 1].reshape(nx, nx) * 1e3
    Ve = d["Ve_xsec"][sl, c].reshape(nx, nx)
    E = d["E_xsec"][sl, c].reshape(nx, nx, 3)
    Emag = np.linalg.norm(E, axis=2)
    F = dict(X=X, Y=Y, x1=X[0, :], y1=Y[:, 0], Ve=Ve, Emag=Emag,
             Ex=E[:, :, 0], Ey=E[:, :, 1], outline=None, polys=[])
    try:
        x = json.loads((ND / "nerve_xsec.json").read_text())
        F["outline"] = np.array(x["nerve_outline_xy_um"]) / 1e3
        F["polys"] = [np.array(f["polygon_xy_um"]) / 1e3 for f in x["fascicles"]]
    except Exception:
        pass
    # contact location = grid point of max V_e
    j = np.unravel_index(np.argmax(Ve), Ve.shape)
    F["contact_xy"] = (X[j], Y[j])
    return F


def _overlay(ax, F, lim=2.0):
    if F["outline"] is not None:
        ax.add_patch(MplPoly(F["outline"], closed=True, fill=False, ec="w", lw=1.6))
    for poly in F["polys"]:
        ax.add_patch(MplPoly(poly, closed=True, fill=False, ec="w", lw=0.5, alpha=0.6))
    ax.plot(*F["contact_xy"], "s", mfc="w", mec="k", ms=11, mew=1.2)
    ax.annotate(f"contact {CONTACT}", F["contact_xy"], textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")


def panel_ve(ax, F):
    cf = ax.contourf(F["X"], F["Y"], F["Ve"], levels=np.geomspace(
        max(F["Ve"].min(), 20), F["Ve"].max(), 16), norm=LogNorm(), cmap="viridis", extend="both")
    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.02, label="$V_e$ (V A$^{-1}$)")
    ticks = [t for t in (100, 200, 400, 800) if F["Ve"].min() <= t <= F["Ve"].max()]
    cb.set_ticks(ticks); cb.ax.set_yticklabels([str(t) for t in ticks]); cb.minorticks_off()
    _overlay(ax, F)


def panel_efield(ax, F):
    im = ax.pcolormesh(F["X"], F["Y"], F["Emag"], cmap="magma",
                       norm=LogNorm(vmin=max(F["Emag"].min(), 5e2), vmax=F["Emag"].max()),
                       shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="$|E|$ (V m$^{-1}$ A$^{-1}$)")
    ax.streamplot(F["x1"], F["y1"], F["Ex"], F["Ey"], color="w", density=0.7,
                  linewidth=0.6, arrowsize=0.7)
    _overlay(ax, F)


def activating(ax):
    """Mean +/- s.d. of the smoothed V_e and activating function (d2V_e/ds2)
    across fibers, for the central contact."""
    from scipy.signal import savgol_filter
    from matplotlib.ticker import MaxNLocator
    d = np.load(ND / "paths_Ve.npz", allow_pickle=True)
    Ve, lens, flat = d["Ve_mat"], d["path_lengths"], d["paths_flat"]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    su = np.linspace(-6, 6, 601); ds = su[1] - su[0]; win, poly = 61, 3
    rng = np.random.default_rng(0)
    sub = rng.choice(len(lens), size=min(400, len(lens)), replace=False)
    Vs, AFs = [], []
    for i in sub:
        xyz = flat[off[i]:off[i + 1]]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]) * 1e3
        s -= s[s.size // 2]
        Vu = np.interp(su, s, Ve[off[i]:off[i + 1], CONTACT])
        Vs.append(savgol_filter(Vu, win, poly))
        AFs.append(savgol_filter(Vu, win, poly, deriv=2, delta=ds))
    V, AF = np.array(Vs), np.array(AFs)
    Vm, Vsd, AFm, AFsd = V.mean(0), V.std(0), AF.mean(0), AF.std(0)
    bl, rd = "#1f77b4", "#d62728"
    ax.plot(su, Vm, color=bl, lw=2.4)
    ax.fill_between(su, Vm - Vsd, Vm + Vsd, color=bl, alpha=0.20, lw=0)
    ax.set_xlabel("axial position (mm)"); ax.set_ylabel("$V_e$ (V A$^{-1}$)", color=bl)
    ax.set_xlim(-6, 6); ax.set_xticks([-6, -3, 0, 3, 6])
    ax.tick_params(axis="y", colors=bl); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax2 = ax.twinx()
    ax2.plot(su, AFm, color=rd, lw=2.4)
    ax2.fill_between(su, AFm - AFsd, AFm + AFsd, color=rd, alpha=0.16, lw=0)
    ax2.set_ylabel("activating function (V A$^{-1}$ mm$^{-2}$)", color=rd)
    ax2.axhline(0, color="0.7", lw=0.6)
    ax2.tick_params(axis="y", colors=rd); ax2.yaxis.set_major_locator(MaxNLocator(5))
    ax2.spines["top"].set_visible(False)


def main():
    print("rendering fig2 mesh + field cross-sections ...")
    mesh_png = render_mesh()
    F = load_xsec()

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 9, "axes.spines.top": False,
                         "axes.linewidth": 1.0})
    fig = plt.figure(figsize=(16.5, 9))
    # wider wspace so panel e's right colorbar label clears panel f's left y-axis
    # label (the e/f collision in the bottom row).
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.46,
                  height_ratios=[1, 1])

    axa = fig.add_subplot(gs[0, 0]); axa.axis("off")
    _lab(axa, "a", "Cuff designer + PCA autofit")
    axa.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.8, transform=axa.transAxes,
                  fill=False, ls="--", ec="0.6"))
    axa.text(0.5, 0.5, "GUI screenshot:\ninteractive cuff designer\n+ autofit to local "
             "nerve axis\n(to be inserted)", transform=axa.transAxes,
             ha="center", va="center", fontsize=10, color="0.45")

    axb = fig.add_subplot(gs[0, 1]); axb.imshow(crop(ND / "render_components.png")); axb.axis("off")
    _lab(axb, "b", "Nerve in 12-contact cuff")

    axc = fig.add_subplot(gs[0, 2]); axc.imshow(crop(mesh_png)); axc.axis("off")
    _lab(axc, "c", "Multi-region FEM mesh (longitudinal)")
    leg = [Patch(facecolor=c, edgecolor="0.5", label=n) for _, (n, c) in MAT.items()]
    leg.append(Patch(facecolor=CONTACT_COL, edgecolor="0.5", label="platinum contact"))
    axc.legend(handles=leg, fontsize=8.5, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    axd = fig.add_subplot(gs[1, 0]); panel_ve(axd, F)
    _lab(axd, "d", "FEM lead-field $V_e$ — cross-section (z = 0)")

    axe = fig.add_subplot(gs[1, 1]); panel_efield(axe, F)
    _lab(axe, "e", "FEM $E$-field + current streamlines")

    axf = fig.add_subplot(gs[1, 2]); activating(axf)
    _lab(axf, "f", "$V_e$ + activating function along fibers")

    save_fig(fig, "fig02_modeling_setup", dpi=200, facecolor="white")
    print("wrote figures/*/fig02_modeling_setup")


if __name__ == "__main__":
    main()
