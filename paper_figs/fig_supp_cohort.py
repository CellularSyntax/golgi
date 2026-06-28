# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Supplementary cohort gallery: every duke_meshes sample as a multi-panel row
  cross-section (saline + silicone cuff + contacts + epineurium + fascicles, with
  a 1 mm scale bar) | FEM Ve map | FEM |E| map (shared colorbars),
drawn from each sample's nerve_xsec.json + xsec_Ve.npz + electrode_config.json.
Grouped by species, 6 samples per supplementary figure.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Circle, Wedge
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig   # noqa: E402
DUKE = ROOT / "results_golgi/duke_meshes"
ROWS_PER_FIG = 6
VE_NORM = LogNorm(vmin=30, vmax=900)            # V A^-1, fixed across samples
E_NORM = LogNorm(vmin=5e2, vmax=1e6)            # V m^-1 A^-1
HDRS = ["cross-section", r"FEM $V_e$ (z=0)", r"FEM $|E|$ (z=0)",
        r"$V_e$ along fibers", "activating function"]
C_SAL, C_CUFF, C_CON, C_EPI, C_ENDO = "#a6cee3", "#d9d9d9", "#444444", "#fdd870", "#fb9a4b"
C_VE, C_AF = "#1f77b4", "#d62728"          # along-fiber Ve / activating function
AF_C = 5                                    # central cathode contact (matches the maps)


def outline_polys(nd):
    x = json.loads((nd / "nerve_xsec.json").read_text())
    out = np.array(x["nerve_outline_xy_um"]) / 1e3
    polys = [np.array(f["polygon_xy_um"]) / 1e3 for f in x["fascicles"]]
    return out, polys


def cuff_geom(nd):
    R, contacts = 1.3, []
    try:
        ec = json.loads((nd / "electrode_config.json").read_text())
        for p in ec.get("patches", []):
            R = float(p["R"]) * 1e3
            if abs(float(p.get("z", 0.0))) < 1e-4:
                contacts.append((float(p["phi"]), float(p.get("dphi", 0.6))))
    except Exception:
        pass
    return R, contacts


def load_field(nd, cid=5):
    d = np.load(nd / "xsec_Ve.npz", allow_pickle=True)
    nx = int(d["nx"]); xy = d["xy"]; zp = d["z_planes"]; cids = list(d["contact_ids"])
    p = int(np.argmin(np.abs(zp))); c = cids.index(cid) if cid in cids else len(cids) // 2
    sl = slice(p * nx * nx, (p + 1) * nx * nx)
    X = xy[:, 0].reshape(nx, nx) * 1e3; Y = xy[:, 1].reshape(nx, nx) * 1e3
    Ve = d["Ve_xsec"][sl, c].reshape(nx, nx)
    Emag = np.linalg.norm(d["E_xsec"][sl, c].reshape(nx, nx, 3), axis=2)
    return X, Y, Ve, Emag


def draw_xsec(ax, nd, out, polys):
    R, contacts = cuff_geom(nd)
    ax.add_patch(Circle((0, 0), R, fc=C_SAL, ec="none", zorder=0))                       # saline
    ax.add_patch(Wedge((0, 0), R + 0.32, 0, 360, width=0.32, fc=C_CUFF, ec="0.55", lw=0.4, zorder=1))  # silicone
    for phi, dphi in contacts:                                                            # contacts
        ax.add_patch(Wedge((0, 0), R + 0.05, np.rad2deg(phi - dphi / 2),
                     np.rad2deg(phi + dphi / 2), width=0.22, fc=C_CON, ec="none", zorder=3))
    ax.add_patch(MplPoly(out, closed=True, fc=C_EPI, ec="0.35", lw=0.7, zorder=2))        # epineurium
    for p in polys:
        ax.add_patch(MplPoly(p, closed=True, fc=C_ENDO, ec="#9c5a18", lw=0.3, zorder=2.4))  # fascicle endoneurium
    m = R + 0.45
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([m - 1.25, m - 0.25], [-m + 0.16, -m + 0.16], "k-", lw=2.2, solid_capstyle="butt")
    ax.text(m - 0.75, -m + 0.26, "1 mm", ha="center", va="bottom", fontsize=6.5)


def draw_field(ax, kind, nd, out, polys):
    X, Y, Ve, Emag = load_field(nd); R, contacts = cuff_geom(nd)
    if kind == "ve":
        ax.contourf(X, Y, Ve, levels=np.geomspace(30, 900, 14), norm=VE_NORM, cmap="viridis", extend="both")
    else:
        ax.pcolormesh(X, Y, Emag, cmap="magma", norm=E_NORM, shading="auto", rasterized=True)
    ax.add_patch(MplPoly(out, closed=True, fill=False, ec="w", lw=1.0))
    for p in polys:
        ax.add_patch(MplPoly(p, closed=True, fill=False, ec="w", lw=0.3, alpha=0.55))
    for phi, dphi in contacts:
        ax.plot(R * np.cos(phi), R * np.sin(phi), "s", mfc="w", mec="k", ms=3, mew=0.5)
    m = R + 0.22
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_aspect("equal"); ax.axis("off")


def n_fasc(nd):
    try:
        return json.loads((nd / "fascicles.json").read_text()).get("n_fascicles", "?")
    except Exception:
        return "?"


def along_fiber(nd, cid=AF_C, n_use=300):
    """Per-fiber Ve + activating function (d2Ve/ds2) over a cuff-centered +/-6 mm
    axial window, central cathode contact. Returns (su, V[fibers], AF[fibers])."""
    from scipy.signal import savgol_filter
    d = np.load(nd / "paths_Ve.npz", allow_pickle=True)
    Ve, lens, flat = d["Ve_mat"], d["path_lengths"], d["paths_flat"]
    cids = list(d["contact_ids"]); c = cids.index(cid) if cid in cids else len(cids) // 2
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    su = np.linspace(-6, 6, 401); ds = su[1] - su[0]; win, poly = 41, 3
    rng = np.random.default_rng(0)
    sub = rng.choice(len(lens), size=min(n_use, len(lens)), replace=False)
    Vs, AFs = [], []
    for i in sub:
        xyz = flat[off[i]:off[i + 1]]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]) * 1e3
        s -= s[s.size // 2]
        Vu = np.interp(su, s, Ve[off[i]:off[i + 1], c])
        Vs.append(savgol_filter(Vu, win, poly))
        AFs.append(savgol_filter(Vu, win, poly, deriv=2, delta=ds))
    return su, np.array(Vs), np.array(AFs)


def draw_line(ax, su, indiv, col, n_show=70):
    """Mean (bold) + per-fiber silhouettes (faint) of a quantity along the fiber."""
    idx = np.linspace(0, len(indiv) - 1, min(n_show, len(indiv))).astype(int)
    for k in idx:
        ax.plot(su, indiv[k], color=col, alpha=0.06, lw=0.5)
    ax.plot(su, indiv.mean(0), color=col, lw=1.8)
    ax.axhline(0, color="0.75", lw=0.5)
    ax.set_xlim(-6, 6); ax.set_xticks([-5, 0, 5])
    ax.tick_params(labelsize=6, length=2, pad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def build(samples, tag):
    nfig = (len(samples) + ROWS_PER_FIG - 1) // ROWS_PER_FIG
    for fi in range(nfig):
        chunk = samples[fi * ROWS_PER_FIG:(fi + 1) * ROWS_PER_FIG]; nr = len(chunk)
        fig = plt.figure(figsize=(12.8, 1.85 * nr + 1.0))
        gs = fig.add_gridspec(nr, 6, width_ratios=[0.40, 1, 1, 1, 0.95, 0.95],
                              hspace=0.12, wspace=0.20,
                              top=1 - 0.55 / (1.85 * nr + 1.0), bottom=0.95 / (1.85 * nr + 1.0))
        ax_ve = ax_ef = None
        for r, s in enumerate(chunk):
            nd = DUKE / s; out, polys = outline_polys(nd)
            axl = fig.add_subplot(gs[r, 0]); axl.axis("off")
            sp = "Human" if s.startswith("human") else "Swine"
            axl.text(0.5, 0.5, f"{sp}\n{s.replace('human_', '').replace('_', ' ')}\n({n_fasc(nd)} fasc)",
                     ha="center", va="center", fontsize=8.5, fontweight="bold", transform=axl.transAxes)
            for c, kind in enumerate(["xsec", "ve", "ef"]):
                ax = fig.add_subplot(gs[r, c + 1])
                try:
                    draw_xsec(ax, nd, out, polys) if kind == "xsec" else draw_field(ax, kind, nd, out, polys)
                except Exception as e:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes,
                            fontsize=8, color="0.6"); ax.axis("off"); print(f"  {s}/{kind}: {e}")
                if kind == "ve":
                    ax_ve = ax
                if kind == "ef":
                    ax_ef = ax
                if r == 0:
                    ax.set_title(HDRS[c], fontsize=10.5, fontweight="bold")
            # two along-fiber columns: Ve(s) and activating function (mean + silhouettes)
            try:
                su, V, AF = along_fiber(nd)
                axv = fig.add_subplot(gs[r, 4]); draw_line(axv, su, V, C_VE)
                axa = fig.add_subplot(gs[r, 5]); draw_line(axa, su, AF, C_AF)
                if r == 0:
                    axv.set_title(HDRS[3], fontsize=10.5, fontweight="bold")
                    axa.set_title(HDRS[4], fontsize=10.5, fontweight="bold")
                if r == nr - 1:
                    axv.set_xlabel("axial (mm)", fontsize=7); axa.set_xlabel("axial (mm)", fontsize=7)
            except Exception as e:
                print(f"  {s}/along-fiber: {e}")
        # shared field colorbars under the Ve & |E| map columns (positions from the axes)
        if ax_ve is not None and ax_ef is not None:
            pve, pef = ax_ve.get_position(fig), ax_ef.get_position(fig)
            cve = fig.add_axes([pve.x0, 0.045, pve.width, 0.011])
            cef = fig.add_axes([pef.x0, 0.045, pef.width, 0.011])
            fig.colorbar(ScalarMappable(norm=VE_NORM, cmap="viridis"), cax=cve, orientation="horizontal").set_label(
                r"$V_e$ (V A$^{-1}$)", fontsize=8)
            fig.colorbar(ScalarMappable(norm=E_NORM, cmap="magma"), cax=cef, orientation="horizontal").set_label(
                r"$|E|$ (V m$^{-1}$ A$^{-1}$)", fontsize=8)
            for cax in (cve, cef):
                cax.tick_params(labelsize=6.5)
        fig.suptitle(f"Cohort gallery — {tag} samples ({fi + 1}/{nfig})",
                     fontsize=13, fontweight="bold", x=0.5, y=0.997)
        name = f"supp_cohort_{tag.lower()}_{fi + 1}"
        save_fig(fig, name, dpi=150, facecolor="white")
        print(f"wrote {name} ({nr} samples)")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    samples = sorted(d.name for d in DUKE.iterdir() if d.is_dir())
    build([s for s in samples if not s.startswith("human")], "Swine")
    build([s for s in samples if s.startswith("human")], "Human")


if __name__ == "__main__":
    main()
