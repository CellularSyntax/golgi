# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 5 — fascicular targeting & selective stimulation (swine | human).
A peripheral TARGET fascicle is chosen; current steering finds the contact +
amplitude that recruits it while sparing the rest.

Rows (2 cols = swine | human):
  a/b  targeting cross-section — fascicle boundaries, target shaded (on-target)
       vs off-target, fibers styled by {on/off target} x {activated/silent},
       optimal contact marked on the cuff.
  c/d  selective recruitment window — on-target vs off-target recruitment vs
       amplitude, operating point marked.
  e/f  evoked action-potential propagation along an activated on-target fiber
       (Vm in space x time; conduction velocity annotated).

Uniform-10um, 12-contact lead-field matrices isolate the GEOMETRIC selectivity
(spatial selectivity is ~diameter-independent). AP traces from fig5_ap_data.py.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPoly, Patch, Circle, Wedge
from matplotlib.lines import Line2D

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import save_fig, DATA   # noqa: E402
HI_UA = 10_000.0
NERVES = [("Swine", DATA / "thr_swine4.npz", "sub-4_sam-3"),
          ("Human", DATA / "thr_human50.npz", "human_sub-50_sam-2")]
# colors — match fig7/fig8 (on-target orange-red, off-target blue) for cross-figure consistency
C_ON_ACT, C_ON_SIL = "#e6550d", "#fdbe85"      # on-target activated / silent (orange-red)
C_OFF_ACT, C_OFF_SIL = "#1f77b4", "#a9c8e6"    # off-target activated / silent (blue)
TARGET_BG, OFF_BG = "#fdebd9", "#f6f6f6"        # target fascicle = light orange tint


def load(npz, nerve_dir):
    d = np.load(npz, allow_pickle=True)
    D = dict(thr=d["thr_uA"], branch=d["branch_idx"].astype(int), xy=d["xy_cuff_mm"],
             cids=d["contact_ids"].astype(int),
             fidx=(d["fiber_idx"].astype(int) if "fiber_idx" in d.files else None),
             cangle={}, polys={}, outline=None, cuffR=None, zcontacts=[])
    nd = ROOT / "results_golgi/duke_meshes" / nerve_dir
    try:
        ec = json.loads((nd / "electrode_config.json").read_text())
        for p in ec.get("patches", []):
            D["cangle"][int(p["id"])] = float(np.rad2deg(p["phi"]))
            if "R" in p:
                D["cuffR"] = float(p["R"]) * 1e3                       # cuff inner radius (mm)
            if abs(float(p.get("z", 1.0))) < 1e-4:                     # z=0 contact row
                D["zcontacts"].append((float(p["phi"]), float(p.get("dphi", 0.75))))
    except Exception:
        pass
    try:
        x = json.loads((nd / "nerve_xsec.json").read_text())
        D["outline"] = np.array(x["nerve_outline_xy_um"]) / 1e3
        for f in x["fascicles"]:
            D["polys"][int(f["id"])] = np.array(f["polygon_xy_um"]) / 1e3
    except Exception:
        pass
    return D


def rfrac(t, amps):
    t = t[np.isfinite(t)]
    return np.array([(t <= a).mean() if t.size else 0.0 for a in amps])


def selectivity(D):
    thr, branch = D["thr"], D["branch"]
    fasc = np.unique(branch); ncon = thr.shape[1]
    amps = np.logspace(np.log10(max(np.nanmin(thr), 1)), np.log10(HI_UA), 90)
    R = np.zeros((ncon, len(fasc), len(amps)))
    for c in range(ncon):
        for ki, k in enumerate(fasc):
            R[c, ki] = rfrac(thr[branch == k][:, c], amps)
    Roff = (R.sum(1, keepdims=True) - R) / max(len(fasc) - 1, 1)
    SI = np.divide(R - Roff, R + Roff, out=np.zeros_like(R), where=(R + Roff) > 1e-9)
    best_si = SI.max(axis=(0, 2))
    best_idx = [np.unravel_index(np.argmax(SI[:, k, :]), SI[:, k, :].shape)
                for k in range(len(fasc))]
    return dict(fasc=fasc, amps=amps, R=R, SI=SI, best_si=best_si, best_idx=best_idx)


def select_target(D, S, min_fib=6):
    """Pick a PERIPHERAL, well-steered target fascicle + its optimal contact and
    operating amplitude (max on-target minus off-target recruitment)."""
    fasc, thr, branch, xy = S["fasc"], D["thr"], D["branch"], D["xy"]
    cen = np.array([xy[branch == k].mean(0) for k in fasc])
    dist = np.linalg.norm(cen - xy.mean(0), axis=1)
    nfib = np.array([(branch == k).sum() for k in fasc])
    elig = (dist >= np.percentile(dist, 55)) & (nfib >= min_fib)
    score = np.where(elig, S["best_si"], -1)
    kt_i = int(np.argmax(score)); kt = int(fasc[kt_i])
    c = int(S["best_idx"][kt_i][0])
    on = branch == kt; off = ~on
    amps = S["amps"]
    onf = np.array([(np.isfinite(thr[on, c]) & (thr[on, c] <= a)).mean() for a in amps])
    offf = np.array([(np.isfinite(thr[off, c]) & (thr[off, c] <= a)).mean() for a in amps])
    iop = int(np.argmax(onf - offf)); amp_op = amps[iop]
    # lowest-threshold on-target fiber = best-coupled -> cleanest central AP init
    on_rec = np.where(on & np.isfinite(thr[:, c]) & (thr[:, c] <= amp_op))[0]
    ex = int(on_rec[np.argmin(thr[on_rec, c])]) if on_rec.size else -1
    return dict(kt=kt, kt_i=kt_i, c=c, cid=int(D["cids"][c]), amp_op=float(amp_op),
                onf=onf, offf=offf, iop=iop, ex=ex, si=float(S["best_si"][kt_i]),
                on_at_op=float(onf[iop]), off_at_op=float(offf[iop]))


def panel_xsec(ax, D, S, T, nm):
    thr, branch, xy = D["thr"], D["branch"], D["xy"]
    rec = np.isfinite(thr[:, T["c"]]) & (thr[:, T["c"]] <= T["amp_op"])
    on = branch == T["kt"]; R = D.get("cuffR")
    # saline + silicone cuff + contacts (optimal highlighted), behind the nerve
    if R:
        ax.add_patch(Circle((0, 0), R, fc="#d6e9f8", ec="none", zorder=0))                       # saline
        ax.add_patch(Wedge((0, 0), R + 0.32, 0, 360, width=0.32, fc="#d9d9d9", ec="0.55",
                     lw=0.8, zorder=0.5))                                                         # silicone cuff
        cid_phi = np.deg2rad(D["cangle"].get(T["cid"], 1e9))
        for phi, dphi in D.get("zcontacts", []):
            opt = abs(((phi - cid_phi + np.pi) % (2 * np.pi)) - np.pi) < 0.15
            ax.add_patch(Wedge((0, 0), R + 0.05, np.rad2deg(phi - dphi / 2),
                         np.rad2deg(phi + dphi / 2), width=0.22,
                         fc=("#d62728" if opt else "#555"), ec="none", zorder=3))
    # nerve epineurium fill (so saline doesn't bleed through), then fascicles
    if D["outline"] is not None:
        ax.add_patch(MplPoly(D["outline"], closed=True, fc="#fbfaf5", ec="0.35", lw=1.4, zorder=1))
    for k, poly in D["polys"].items():
        is_t = (k == T["kt"])
        ax.add_patch(MplPoly(poly, closed=True, fc=(TARGET_BG if is_t else OFF_BG),
                     ec=(C_ON_ACT if is_t else "0.7"), lw=(2.0 if is_t else 0.6), zorder=2))
    # fibers by 4 categories
    cats = [(on & rec, C_ON_ACT, 16), (on & ~rec, C_ON_SIL, 13),
            (~on & rec, C_OFF_ACT, 13), (~on & ~rec, C_OFF_SIL, 5)]
    for m, col, s in cats:
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=s, c=col, marker="o", lw=0,
                       zorder=4 if col != C_OFF_SIL else 3.5)
    lim = (R + 0.5) if R else 1.12 * np.abs(D["outline"]).max()
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([lim - 1.25, lim - 0.25], [-lim + 0.18, -lim + 0.18], "k-", lw=2.2, solid_capstyle="butt")
    ax.text(lim - 0.75, -lim + 0.30, "1 mm", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"{nm} — target fascicle {T['kt']} (peripheral) · SI {T['si']:.2f}",
                 fontsize=11, loc="left")


def panel_window(ax, S, T):
    amps = S["amps"] / 1e3
    ax.fill_between(amps, 0, T["onf"] * 100, color=C_ON_ACT, alpha=0.18, lw=0)
    ax.semilogx(amps, T["onf"] * 100, color=C_ON_ACT, lw=2.6, label="on-target")
    ax.semilogx(amps, T["offf"] * 100, color=C_OFF_ACT, lw=2.4, label="off-target")
    ao = T["amp_op"] / 1e3
    ax.axvline(ao, color="0.35", ls="--", lw=1.2)
    # focus the x-range on the informative rising region (stop just past off-target
    # saturation) rather than running out to the 80 mA sweep ceiling
    off, on = T["offf"], T["onf"]
    ref = off if off.max() >= 0.9 else on
    isat = int(np.argmax(ref >= 0.99 * ref.max())) if ref.max() > 0 else len(amps) - 1
    hi = min(float(amps.max()), float(amps[max(isat, 1)]) * 1.4)
    ax.set_xlim(amps.min(), hi)
    ax.text(0.03, 0.97, f"operating point\non {T['on_at_op']*100:.0f}% / off {T['off_at_op']*100:.0f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="0.15",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
    ax.set_xlabel("stimulus amplitude (mA)"); ax.set_ylabel("% recruited")
    ax.set_ylim(0, 100); ax.legend(fontsize=9, loc="lower right", frameon=False)


def panel_ap(ax, ap, nm, stim_z=0.0):
    if ap is None:
        ax.text(0.5, 0.5, "AP propagation\n(run fig5_ap_data.py)", ha="center",
                va="center", transform=ax.transAxes, color="0.5", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); return
    vm, t, z = ap["vm"], ap["t"], ap["z"]          # [node,time], ms, mm(centered)
    im = ax.pcolormesh(t, z, vm, cmap="inferno", vmin=-90, vmax=40,
                       shading="gouraud", rasterized=True)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="$V_m$ (mV)")
    ax.set_xlim(0, min(float(t.max()), 1.3))
    ax.set_xlabel("time (ms)"); ax.set_ylabel("axial position (mm)")
    # white arrow: where (stim_z = the contact / AP-initiation site) + when (pulse onset)
    import matplotlib.patheffects as pe
    stroke = [pe.withStroke(linewidth=2.5, foreground="black")]
    ax.annotate("stim", xy=(0.15, stim_z), xytext=(0.22, stim_z),
                color="white", fontsize=9, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8, path_effects=stroke),
                path_effects=stroke)
    ax.set_title(f"{nm} — evoked AP propagation", fontsize=11, loc="left")


def _rfrac_pct(t, amps):
    t = t[np.isfinite(t)]
    return np.array([(t <= a).mean() * 100 if t.size else 0.0 for a in amps])


def class_si_rows(thr_col, on01, typ, ORDER):
    """Per-class selectivity index — on-target (on01==1) class-c recruitment minus
    pooled off-target (on01==0) recruitment, maximised over amplitude (Veraart).
    Same definition as fig7/8 panel f, with on/off = target fascicle vs rest."""
    on = on01 == 1; off = on01 == 0
    fin = thr_col[np.isfinite(thr_col)]
    if fin.size == 0:
        return []
    amps = np.logspace(np.log10(max(fin.min(), 1.0)), np.log10(fin.max()), 200)
    offall = _rfrac_pct(thr_col[off], amps)
    rows = []
    for c in ORDER:
        m = on & (typ == c)
        if not m.any():
            continue
        onc = _rfrac_pct(thr_col[m], amps); diff = onc - offall; i = int(np.argmax(diff))
        rows.append((c, diff[i] / 100.0, onc[i], offall[i], amps[i] / 1e3, int(m.sum())))
    return rows


def panel_class(ax, thr_col, on01, typ, all_si, ORDER, CLASS_COL,
                ylabel="selectivity index (class vs off-target)"):
    """Per-class selectivity bar (target fascicle vs rest) — matches fig7/8 panel f."""
    rows = class_si_rows(thr_col, on01, typ, ORDER)
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], width=0.66,
           color=[CLASS_COL[r[0]] for r in rows], edgecolor="none")
    for i, r in enumerate(rows):
        ax.text(i, max(r[1], 0) + 0.02, f"{r[1]:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="0.2")
    ax.axhline(all_si, ls="--", lw=1.2, color="0.45")
    ax.text(len(rows) - 0.5, all_si + 0.015, f"all {all_si:.2f}", ha="right", va="bottom",
            fontsize=8, color="0.45")
    ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\n(n={r[5]})" for r in rows])
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05)


def panel_thrdiam(ax, diam, thr_uA, on_mask, op_amp_mA, on_label="on-target",
                  off_label="off-target", hi_mA=None):
    """Threshold vs fiber diameter — fibers coloured on/off-target, the selective
    operating amplitude marked. Directly shows which fiber SIZES are recruited
    on- vs off-target at the operating point (a point below the line is recruited):
    selective stimulation = on-target points fall below the line while off-target
    points stay above it. Shared by fig5--8 (panel g) for cross-figure consistency."""
    t = np.asarray(thr_uA, float) / 1e3                       # mA
    d = np.asarray(diam, float)
    on = np.asarray(on_mask, bool); off = ~on
    fin = np.isfinite(t) & (t > 0)
    # shaded "recruited" band below the operating amplitude
    ax.axhspan(0, op_amp_mA, color="#eef3f7", lw=0, zorder=0)
    ax.scatter(d[off & fin], t[off & fin], s=13, c=C_OFF_ACT, marker="o", lw=0,
               alpha=0.55, label=off_label, zorder=2)
    ax.scatter(d[on & fin], t[on & fin], s=18, c=C_ON_ACT, marker="o", lw=0,
               alpha=0.85, label=on_label, zorder=3)
    ax.axhline(op_amp_mA, ls="--", lw=1.3, color="0.35", zorder=2.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("fiber diameter (µm)"); ax.set_ylabel("threshold (mA)")
    ax.text(0.02, op_amp_mA, "operating\npoint", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=7.6, color="0.35")
    ax.legend(fontsize=8.5, loc="upper right", frameon=False, handletextpad=0.2)


def _lab(ax, letter):
    ax.text(-0.06, 1.12, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="right")


def main():
    apf = DATA / "fig5_ap.npz"
    AP = dict(np.load(apf, allow_pickle=True)) if apf.exists() else {}
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12.5, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.26,
                  height_ratios=[1.25, 0.85, 0.9], top=0.965, bottom=0.05)
    steer = []
    for col, (nm, npz, nd) in enumerate(NERVES):
        D = load(npz, nd); S = selectivity(D); T = select_target(D, S)
        pct = (S["best_si"] > 0.5).mean() * 100; steer.append(pct)
        axa = fig.add_subplot(gs[0, col]); panel_xsec(axa, D, S, T, nm); _lab(axa, "ab"[col])
        axb = fig.add_subplot(gs[1, col]); panel_window(axb, S, T); _lab(axb, "cd"[col])
        axb.set_title(f"selective window · {pct:.0f}% of fascicles steerable (SI>0.5)",
                      fontsize=11, loc="left")
        ap = None
        key = nm.lower()
        if f"{key}_vm" in AP:
            ap = dict(vm=AP[f"{key}_vm"], t=AP[f"{key}_t"], z=AP[f"{key}_z"],
                      cv=float(AP[f"{key}_cv"]))
        axc = fig.add_subplot(gs[2, col]); panel_ap(axc, ap, nm); _lab(axc, "ef"[col])
        print(f"{nm}: target fasc {T['kt']} (c{T['cid']}), op {T['amp_op']:.0f}uA "
              f"on {T['on_at_op']*100:.0f}% off {T['off_at_op']*100:.0f}% | {pct:.0f}% steerable")
    # shared legend (4 categories) under row 1
    handles = [Line2D([], [], marker="o", ls="", mfc=c, mec="none", ms=9, label=l)
               for c, l in [(C_ON_ACT, "on-target activated"), (C_ON_SIL, "on-target silent"),
                            (C_OFF_ACT, "off-target activated"), (C_OFF_SIL, "off-target silent")]]
    y1b = gs[0, 0].get_position(fig).y0          # bottom of the a/b row
    fig.legend(handles=handles, ncol=4, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, y1b - 0.004), fontsize=9.5)
    save_fig(fig, "fig6_selectivity", dpi=200, facecolor="white")
    print(f"wrote figures/*/fig05_selectivity | steerable swine/human = {steer[0]:.0f}/{steer[1]:.0f}%")


if __name__ == "__main__":
    main()
