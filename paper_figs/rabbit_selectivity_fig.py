# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 7 (replacement) — superior-cardiac-branch-selective stimulation of the real-3D RABBIT
cervical vagus with the 4x5 ring cuff. Same house style + layout as fig8 (new-human): full-
width render on top, then realistic-population data panels, with a supplementary cuff-POSITION
x multipolar-CONFIG sweep. RABBIT analogue of new_human_selectivity_fig.py.

  MAIN (best result, realistic rabbit population):
    a render (swept cuff positions, best highlighted) | b pop by class | c recruitment by class
    d cuff cross-section (nerve boundary + fibers SCB/trunk x act/silent) | e selective window
    f SCB-to-trunk selectivity index per class        | g evoked AP along a recruited SCB fiber
  SUPP (controlled 10 um sweep):
    a SI vs cuff position | b operating amplitude vs position | c config comparison @ best
    d pulse-width comparison (100 vs 300 us)

Selectivity index = (SCB% - trunk%)/100 at the operating point (= argmax SCB%-trunk%). Panels
b-g use the realistic rabbit population (C-dominated) at the best position; the supp uses the
fixed-10um controlled-spatial-selectivity sweep across the common-trunk positions (3-6 mm).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Wedge, Polygon as MplPoly
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from fig5_population import classify, UNMYEL, diam_panel, recruit_panel, ORDER, CLASS_COL   # noqa: E402
from fig06_selectivity import panel_ap, panel_thrdiam                     # noqa: E402
from io_paths import load_ppmm, draw_scalebar                            # noqa: E402

SW = ROOT / "paper_figs/out/data/rabbit_tripole_sweep"
RENDER = ROOT / "paper_figs/out/renders/rabbit_setup.png"
APZ = ROOT / "paper_figs/out/data/rabbit_ap.npz"
OUTP = ROOT / "paper_figs/out/figures/png/fig7_rabbit_selectivity.png"
OUTSUPP = ROOT / "paper_figs/out/figures/png/supp_rabbit_scb_sweep.png"

TARGET = 1                                        # SCB = branch 1
C_ON_ACT, C_ON_SIL = "#e6550d", "#fdbe85"         # SCB activated / silent (orange)
C_OFF_ACT, C_OFF_SIL = "#1f77b4", "#a9c8e6"       # trunk activated / silent (blue)
C_CATH, C_ANODE = "#2ca02c", "#7b3294"            # cathode green / guard anodes purple
NERVE_FC, ENDO_FC, ENDO_EC = "#e7dcc1", "#f7f0dc", "#caa96b"
CFG_COL = {"mono": "#9e9e9e", "long_tripole": "#e6550d", "trans_tripole": "#2ca02c"}
CFG_LAB = {"mono": "monopole", "long_tripole": "longitudinal tripole",
           "trans_tripole": "transverse tripole"}
PHI_DEG = 30.0
BEST_TAG, BEST_CFG = "off3_4x5", "long_tripole"   # sweep result: SI best farthest from branch
# (off3=0.73 > off4=0.69 > off5=0.48 > off6), where SCB/trunk fascicles are most angularly separated
# all swept offsets (3-6 mm) sit on the COMMON TRUNK, upstream of the z~8 mm bifurcation;
# dist-from-branch = 8 - offset.
POSITIONS = [("off3_4x5", 5), ("off4_4x5", 4), ("off5_4x5", 3), ("off6_4x5", 2)]


def rfrac(col, mask, amps):
    t = col[mask]
    return np.array([100 * np.mean(np.isfinite(t) & (t <= a)) for a in amps])


def analyse(thr, branch, names):
    on = branch == TARGET; off = ~on
    fin = thr[np.isfinite(thr)]
    amps = np.logspace(np.log10(max(fin.min(), 50)), np.log10(fin.max()), 160)
    out = {"amps": amps, "on": on, "off": off, "names": names}
    for j, nm in enumerate(names):
        Ron = rfrac(thr[:, j], on, amps); Roff = rfrac(thr[:, j], off, amps)
        iop = int(np.argmax(Ron - Roff))
        out[nm] = dict(Ron=Ron, Roff=Roff, iop=iop, amp=amps[iop], col=j,
                       son=Ron[iop], soff=Roff[iop], si=(Ron[iop] - Roff[iop]) / 100.0)
    return out


def load_thr(tag, fname):
    d = np.load(SW / tag / fname, allow_pickle=True)
    names = list(np.load(SW / tag / "paths_Ve.npz", allow_pickle=True)["pattern_names"])
    return d, names


def fiber_xy(tag, col):
    d = np.load(SW / tag / "paths_Ve.npz", allow_pickle=True)
    flat = d["paths_flat"] * 1e3; lens = d["path_lengths"]; Ve = np.abs(d["Ve_mat"][:, col])
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    pk = np.array([flat[off[i]:off[i + 1]][np.argmax(Ve[off[i]:off[i + 1]]), 2] for i in range(len(lens))])
    cz = float(np.median(pk))
    xy = np.array([flat[off[i]:off[i + 1]][np.argmin(np.abs(flat[off[i]:off[i + 1], 2] - cz))][:2]
                   for i in range(len(lens))])
    return xy - flat[:, :2].mean(0)


def load_contours(tag):
    f = SW / tag / "xsec_contours.npz"
    if not f.exists():
        return None, []
    d = np.load(f, allow_pickle=True)
    n = int(d["n_endo"])
    return np.asarray(d["epi"]), [np.asarray(d[f"endo{i}"]) for i in range(n)]


# ---------------------------------------------------------------- panels
def _letter(ax, L):
    ax.text(-0.02, 1.14, L, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", ha="right")


def ap_stim_z(ap):
    if ap is None:
        return 0.0
    vm, t, z = ap["vm"], ap["t"], ap["z"]
    cross = np.array([t[np.where(vm[i] > 0)[0][0]] if (vm[i] > 0).any() else np.inf
                      for i in range(vm.shape[0])])
    return 0.0 if not np.isfinite(cross).any() else float(z[int(np.argmin(cross))])


def panel_render(ax):
    ax.axis("off")
    if RENDER.exists():
        im = plt.imread(str(RENDER)); ax.imshow(im)
        try:
            draw_scalebar(ax, im.shape[1], im.shape[0], load_ppmm(RENDER), loc="lower left")
        except Exception:
            pass
    else:
        ax.text(0.5, 0.5, "render missing (run render_rabbit_sweep.py)", ha="center",
                va="center", transform=ax.transAxes, color="0.5")
    _letter(ax, "a")


def panel_thumb(ax, tag, dist, is_best):
    """A tiny nerve+cuff render for one swept position (small-multiples row under panel a)."""
    f = ROOT / f"paper_figs/out/renders/rabbit_setup_thumb_{tag}.png"
    if f.exists():
        ax.imshow(plt.imread(str(f)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(is_best)
        if is_best:
            s.set_edgecolor(C_CATH); s.set_linewidth(2.2)
    ax.set_title(f"{dist:g} mm" + ("  ·  best" if is_best else ""), fontsize=9, pad=2,
                 color=(C_CATH if is_best else "0.35"), fontweight=("bold" if is_best else "normal"))


def panel_xsec(ax, tag, S, best, thr, branch, fidx=None):
    xy = fiber_xy(tag, S[best]["col"])
    if fidx is not None:                      # realistic pop is subsampled -> match thr/branch fibers
        xy = xy[fidx]
    on = branch == TARGET; off = ~on
    rec = np.isfinite(thr[:, S[best]["col"]]) & (thr[:, S[best]["col"]] <= S[best]["amp"])
    epi, endo = load_contours(tag)
    meta = json.loads((SW / tag / "meta.json").read_text())
    rad = np.linalg.norm(xy, axis=1)
    R = (np.abs(epi).max() if epi is not None and len(epi) else np.percentile(rad, 98)) + 0.18
    ax.add_patch(Circle((0, 0), R, fc="#d6e9f8", ec="none", zorder=0))                      # saline
    ax.add_patch(Wedge((0, 0), R * 1.14, 0, 360, width=R * 0.14, fc="#cfd4da", ec="0.5",
                       lw=1.0, zorder=0.5))                                                  # cuff
    if epi is not None and len(epi):
        ax.add_patch(MplPoly(epi, closed=True, fc=NERVE_FC, ec="0.4", lw=1.5, zorder=1))
    for e in endo:
        ax.add_patch(MplPoly(e, closed=True, fc=ENDO_FC, ec=ENDO_EC, lw=1.0, zorder=1.3))
    base = meta["contact_angle_deg"][str(meta["cathode"])]
    for k in range(5):
        ph = base + k * 360.0 / 5
        ax.add_patch(Wedge((0, 0), R * 1.13, ph - PHI_DEG / 2, ph + PHI_DEG / 2,
                           width=R * 0.12, fc=(C_CATH if k == 0 else "#5a5f66"), ec="none",
                           zorder=3.2 if k == 0 else 3))
    for mk, col, s, z in [(on & rec, C_ON_ACT, 22, 4.4), (on & ~rec, C_ON_SIL, 15, 3.8),
                          (off & rec, C_OFF_ACT, 15, 4.2), (off & ~rec, C_OFF_SIL, 8, 3.6)]:
        if mk.any():
            ax.scatter(xy[mk, 0], xy[mk, 1], s=s, c=col, marker="o", lw=0, zorder=z)
    lim = R * 1.28
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    bar = 0.5                                                                # 0.5 mm (rabbit scale)
    ax.plot([lim - bar - 0.06 * R, lim - 0.06 * R], [-lim + 0.07 * R] * 2, "k-", lw=2.2,
            solid_capstyle="butt")
    ax.text(lim - bar / 2 - 0.06 * R, -lim + 0.10 * R, f"{bar:g} mm", ha="center", va="bottom",
            fontsize=7.5)
    handles = [Line2D([], [], marker="o", ls="", mfc=cc, mec="none", ms=7, label=ll) for cc, ll in
               [(C_ON_ACT, "SCB act."), (C_ON_SIL, "SCB silent"),
                (C_OFF_ACT, "trunk act."), (C_OFF_SIL, "trunk silent")]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4,
              fontsize=7.4, frameon=False, handletextpad=0.3, columnspacing=1.0)
    _letter(ax, "d")


def panel_window(ax, S, best, letter="e"):
    amps = S["amps"] / 1e3; on = S[best]["Ron"]; off = S[best]["Roff"]
    ax.fill_between(amps, 0, on, color=C_ON_ACT, alpha=0.18, lw=0)
    ax.semilogx(amps, on, color=C_ON_ACT, lw=2.6, label="on-target (SCB)")
    ax.semilogx(amps, off, color=C_OFF_ACT, lw=2.4, label="off-target (trunk)")
    ax.axvline(S[best]["amp"] / 1e3, color="0.35", ls="--", lw=1.2)
    ref = off if off.max() >= 90 else on
    isat = int(np.argmax(ref >= 0.99 * ref.max())) if ref.max() > 0 else len(amps) - 1
    ax.set_xlim(amps.min(), min(float(amps.max()), float(amps[max(isat, 1)]) * 1.4))
    ax.set_xlabel("stimulus amplitude (mA)"); ax.set_ylabel("% recruited"); ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.text(0.03, 0.74, f"operating point\non {S[best]['son']:.0f}% / off {S[best]['soff']:.0f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="0.15",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))
    _letter(ax, letter)


def class_si(thr, branch, typ, col):
    # amplitude grid spans the ACTIVE column's threshold range (an all-column grid
    # undersamples the margin maximum); converged, matches controls_lib (S12/S13).
    trunk = branch == 0; scb = branch == 1
    fin = thr[:, col][np.isfinite(thr[:, col])]
    amps = np.logspace(np.log10(max(fin.min(), 50)), np.log10(fin.max()), 200)
    offall = rfrac(thr[:, col], trunk, amps)
    out = []
    for c in ORDER:
        onm = scb & (typ == c)
        if not onm.any():
            continue
        on = rfrac(thr[:, col], onm, amps); m = on - offall; i = int(np.argmax(m))
        out.append((c, m[i] / 100, on[i], offall[i], amps[i] / 1e3, int(onm.sum())))
    return out


def panel_class(ax, thr, branch, typ, col, all_si, letter="f"):
    rows = class_si(thr, branch, typ, col)
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], width=0.66,
           color=[CLASS_COL[r[0]] for r in rows], edgecolor="none")
    for i, r in enumerate(rows):
        ax.text(i, max(r[1], 0) + 0.02, f"{r[1]:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="0.2")
    ax.axhline(all_si, ls="--", lw=1.2, color="0.45")
    ax.text(len(rows) - 0.5, all_si + 0.015, f"all SCB {all_si:.2f}", ha="right", va="bottom",
            fontsize=8, color="0.45")
    ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\n(n={r[5]})" for r in rows])
    ax.set_ylabel("selectivity index (SCB class vs trunk)"); ax.set_ylim(0, 1.05)
    _letter(ax, letter)


def panel_pw(ax, letter="d"):
    names = list(np.load(SW / BEST_TAG / "paths_Ve.npz", allow_pickle=True)["pattern_names"])
    series = [("100 µs", "thr_pop.npz", "#9e9e9e"), ("300 µs", "thr_pop_pw300.npz", "#e6550d")]
    bars, amps = [], []
    for lab, f, col in series:
        p = SW / BEST_TAG / f
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        typ = classify(d["diameter_um"], np.isin(d["model"].astype(str), list(UNMYEL)))
        rows = {r[0]: r[1] for r in class_si(d["thr_uA"], d["branch_idx"].astype(int), typ, 1)}
        A1 = analyse(d["thr_uA"], d["branch_idx"].astype(int), names)
        bars.append((lab, col, rows)); amps.append((lab, A1[BEST_CFG]["amp"] / 1e3))
    classes = [c for c in ORDER if any(c in b[2] for b in bars)]
    x = np.arange(len(classes)); w = 0.38
    for i, (lab, col, rows) in enumerate(bars):
        ax.bar(x + (i - 0.5) * w, [rows.get(c, 0) for c in classes], w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("selectivity index (SCB class vs trunk)"); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False, title="pulse width")
    ax.text(0.02, 0.98, "op amplitude (all SCB): " + " → ".join(f"{a:.1f} mA" for _, a in amps),
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.2")
    _letter(ax, letter)


def panel_sweep(ax, A, order, letter="a"):
    for nm in ["mono", "long_tripole", "trans_tripole"]:
        d = np.array([(A[t]["dist"], A[t][nm]["si"]) for t in order])
        ax.plot(d[:, 0], d[:, 1], "o-", color=CFG_COL[nm], lw=2.3, ms=7, label=CFG_LAB[nm])
    ax.set_xlabel("cuff distance from bifurcation (mm)\n← nearer the branch")
    ax.set_ylabel("selectivity index"); ax.set_ylim(0, 1.04); ax.invert_xaxis()
    ax.legend(fontsize=8.2, loc="upper left", frameon=False)
    _letter(ax, letter)


def panel_amp(ax, A, order, letter="b"):
    for nm in ["mono", "long_tripole", "trans_tripole"]:
        d = np.array([(A[t]["dist"], A[t][nm]["amp"] / 1e3) for t in order])
        ax.plot(d[:, 0], d[:, 1], "o-", color=CFG_COL[nm], lw=2.3, ms=7, label=CFG_LAB[nm])
    ax.set_xlabel("cuff distance from bifurcation (mm)\n← nearer the branch")
    ax.set_ylabel("operating amplitude (mA)"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.axhspan(0, 3, color="#cfe8cf", alpha=0.5, lw=0, zorder=0)
    ax.text(0.98, 3, "≤3 mA (clinical)", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=9, color="#3a7a3a")
    ax.legend(fontsize=8.2, loc="upper left", frameon=False)
    _letter(ax, letter)


def panel_configcmp(ax, A, tag, letter="c"):
    S = A[tag]; cfgs = ["mono", "long_tripole", "trans_tripole"]
    x = np.arange(3); w = 0.38
    on = [S[c]["son"] for c in cfgs]; off = [S[c]["soff"] for c in cfgs]
    ax.bar(x - w / 2, on, w, color=C_ON_ACT, label="SCB (on)")
    ax.bar(x + w / 2, off, w, color=C_OFF_ACT, label="trunk (off)")
    for i, c in enumerate(cfgs):
        ax.text(i, max(on[i], off[i]) + 4, f"SI {S[c]['si']:.2f}\n{S[c]['amp']/1e3:.1f} mA",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="0.2")
    ax.set_xticks(x); ax.set_xticklabels(["mono", "long.\ntripole", "trans.\ntripole"])
    ax.set_ylabel("% recruited at op."); ax.set_ylim(0, 132)
    ax.legend(fontsize=8.2, loc="upper right", frameon=False)
    _letter(ax, letter)


def main():
    # ---- controlled 10um sweep (supp a-c) ----
    A = {}
    for tag, dist in POSITIONS:
        if not (SW / tag / "thr.npz").exists():
            continue
        d, names = load_thr(tag, "thr.npz")
        A[tag] = analyse(d["thr_uA"], d["branch_idx"].astype(int), names); A[tag]["dist"] = dist
    order = [t for t, _ in POSITIONS if t in A]

    # ---- realistic rabbit population at best position (b-g); prefer clinical 300 us ----
    pf = next((c for c in ["thr_pop_pw300.npz", "thr_pop.npz", "thr.npz"]
               if (SW / BEST_TAG / c).exists()), "thr.npz")
    have_pop = pf.startswith("thr_pop")
    dpop, names = load_thr(BEST_TAG, pf)
    print(f"main population file: {pf}")
    P = analyse(dpop["thr_uA"], dpop["branch_idx"].astype(int), names)
    diam = dpop["diameter_um"] if "diameter_um" in dpop.files else np.full(len(dpop["thr_uA"]), 10.0)
    model = dpop["model"].astype(str) if "model" in dpop.files else np.array(["MRG_INTERPOLATION"] * len(diam))
    typ = classify(diam, np.isin(model, list(UNMYEL)))
    thr_best = dpop["thr_uA"]; branch_best = dpop["branch_idx"].astype(int)

    # controlled 10um data for panels d (cross-section) + e (window) -- the clean
    # geometric targeting demo; panels f (per-class) + g (thr-vs-diameter) stay on the
    # realistic population, matching fig5/6.
    dctrl, _namesc = load_thr(BEST_TAG, "thr.npz")
    Pc = A.get(BEST_TAG) or analyse(dctrl["thr_uA"], dctrl["branch_idx"].astype(int), _namesc)
    thr_ctrl = dctrl["thr_uA"]; branch_ctrl = dctrl["branch_idx"].astype(int)
    fidx_ctrl = dctrl["fiber_idx"] if "fiber_idx" in dctrl.files else None

    print(f"population: {'realistic ' + str({t: int((typ==t).sum()) for t in np.unique(typ)}) if have_pop else 'fixed 10um (pop pending)'}")
    print(f"BEST {CFG_LAB[BEST_CFG]} @ {BEST_TAG}: SI {P[BEST_CFG]['si']:.2f} "
          f"(SCB {P[BEST_CFG]['son']:.0f}% / trunk {P[BEST_CFG]['soff']:.0f}% @ {P[BEST_CFG]['amp']/1e3:.1f} mA)")

    ap = dict(np.load(APZ, allow_pickle=True)) if APZ.exists() else None

    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False})

    # ---------- MAIN figure: best result (render + realistic pop, fig5-8 style) ----------
    fig = plt.figure(figsize=(14, 14.5))
    gs = GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.30, height_ratios=[1.62, 1.0, 1.0],
                  top=0.975, bottom=0.05, left=0.06, right=0.975)
    # panel a = big best-cuff render on top + a small-multiples row of the swept positions below
    gsa = gs[0, :].subgridspec(2, 4, height_ratios=[3.0, 1.0], hspace=0.24, wspace=0.07)
    panel_render(fig.add_subplot(gsa[0, :]))
    for j, (tag, dist) in enumerate(POSITIONS):
        panel_thumb(fig.add_subplot(gsa[1, j]), tag, dist, tag == BEST_TAG)
    axb = fig.add_subplot(gs[1, 0]); diam_panel(axb, diam, typ, "", "b", legend_loc="upper right")
    axb.set_title("", loc="left")
    axc = fig.add_subplot(gs[1, 1])
    recruit_panel(axc, thr_best[:, P[BEST_CFG]["col"]], diam, typ, "", "c",
                  legend_loc="upper left", legend_anchor=(0.0, 1.0)); axc.set_title("", loc="left")
    fidx = dpop["fiber_idx"] if "fiber_idx" in dpop.files else None
    panel_xsec(fig.add_subplot(gs[1, 2]), BEST_TAG, Pc, BEST_CFG, thr_ctrl, branch_ctrl, fidx_ctrl)
    panel_window(fig.add_subplot(gs[2, 0]), Pc, BEST_CFG, "e")
    panel_class(fig.add_subplot(gs[2, 1]), thr_best, branch_best, typ, P[BEST_CFG]["col"],
                P[BEST_CFG]["si"], "f")
    axg = fig.add_subplot(gs[2, 2])
    panel_thrdiam(axg, diam, thr_best[:, P[BEST_CFG]["col"]], branch_best == 1,
                  P[BEST_CFG]["amp"] / 1e3, on_label="on-target (SCB)",
                  off_label="off-target (trunk)")
    _letter(axg, "g")
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTP, dpi=200, facecolor="white"); plt.close(fig)
    print(f"wrote {OUTP}")

    # ---------- SUPPLEMENTARY: controlled 10 um sweep (a-c) + pulse-width (d) ----------
    figs = plt.figure(figsize=(12.5, 9))
    gss = GridSpec(2, 2, figure=figs, wspace=0.28, hspace=0.46, top=0.90, bottom=0.09,
                   left=0.07, right=0.975)
    panel_sweep(figs.add_subplot(gss[0, 0]), A, order, "a")
    panel_amp(figs.add_subplot(gss[0, 1]), A, order, "b")
    panel_configcmp(figs.add_subplot(gss[1, 0]), A, BEST_TAG, "c")
    panel_pw(figs.add_subplot(gss[1, 1]), "d")
    figs.suptitle("Supplementary (rabbit) — controlled 10 µm cuff-position × config sweep (a–c) "
                  "and pulse-width comparison (d)", fontsize=12, fontweight="bold", y=0.965)
    figs.savefig(OUTSUPP, dpi=200, facecolor="white"); plt.close(figs)
    print(f"wrote {OUTSUPP}")


if __name__ == "__main__":
    main()
