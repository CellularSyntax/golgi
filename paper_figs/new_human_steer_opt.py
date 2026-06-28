# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Optimized multi-contact current STEERING on the 20-contact (4x5) cuff at the best
position (off15 ~20 mm from the bifurcation). Tests the one lever not yet pulled: can a
weighted combination of all 20 contacts separate the superior cardiac branch (SCB) from
the trunk better than any single contact?

Activation model: a fiber is recruited in order of its peak depolarizing ACTIVATING
FUNCTION (AF = d2Ve/ds2 along the fiber). The lead fields superpose linearly, so for a
current pattern w (one weight per contact), AF_total = sum_j w_j AF_j and the per-fiber
recruitment ease = max_s AF_total. Selectivity = max-margin = max over amplitude of
(%SCB recruited - %trunk recruited)  [a one-sided KS separation of the SCB vs trunk
peak-AF distributions; 0 = no selectivity, 1 = perfect].

w is charge-balanced (sum w = 0) and the metric is scale-invariant, so we optimize the
pattern DIRECTION (smooth log-sum-exp surrogate, multi-restart L-BFGS-B) then score the
true max-margin. Compares: best single contact vs optimized steering. Writes a recruitment
(ROC-like) plot + the optimized current pattern.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
TAG = "off15_4x5"   # best position from the sweep (~20 mm from branch)
D = ROOT / f"paper_figs/out/data/new_human_branch_{TAG}/paths_Ve.npz"
OUTP = ROOT / "paper_figs/out/figures/png/new_human_steering.png"
SMOOTH_PTS = 4.0    # Gaussian smoothing (in 60 um samples) before the 2nd derivative


def load_af():
    d = np.load(D, allow_pickle=True)
    Ve = np.asarray(d["Ve_mat"], float)          # (N_pts, 20)  V/A
    flat = np.asarray(d["paths_flat"], float)    # (N_pts, 3) m
    lens = np.asarray(d["path_lengths"], int)
    bidx = np.asarray(d["branch_idx"], int)      # 0=trunk, 1=SCB
    cids = np.asarray(d["contact_ids"], int)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    AF = np.zeros_like(Ve)
    for i in range(len(lens)):
        sl = slice(off[i], off[i + 1])
        p = flat[sl]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        if len(s) < 5:
            continue
        ve = gaussian_filter1d(Ve[sl], SMOOTH_PTS, axis=0, mode="nearest")
        d1 = np.gradient(ve, s, axis=0)
        AF[sl] = np.gradient(d1, s, axis=0)
    # normalize AF scale for conditioning (max-margin is invariant to this)
    AF /= (np.std(AF) + 1e-30)
    return AF, off, bidx, cids, flat


def main():
    AF, off, bidx, cids, flat = load_af()
    nf = len(off) - 1
    n_scb = int((bidx == 1).sum()); n_tr = int((bidx == 0).sum())
    starts = off[:-1]
    print(f"loaded {TAG}: {nf} fibers ({n_tr} trunk / {n_scb} SCB), 20 contacts, AF ready", flush=True)

    def peak_af(w):
        af = AF @ w
        return np.maximum.reduceat(af, starts)        # per-fiber max depolarizing AF

    def maxmargin(pk):
        o = np.argsort(-pk); pos = pk[o] > 0
        fs = np.cumsum((bidx[o] == 1) & pos) / n_scb
        ft = np.cumsum((bidx[o] == 0) & pos) / n_tr
        m = fs - ft
        k = int(np.argmax(m))
        return float(m[k]), float(fs[k]), float(ft[k])    # margin, %SCB, %trunk at op

    # ---- baseline: best single contact (monopolar +/-) ----
    best_s = (-1, None, None)
    for j in range(20):
        for sgn in (+1.0, -1.0):
            w = np.zeros(20); w[j] = sgn
            m = maxmargin(peak_af(w))
            if m[0] > best_s[0]:
                best_s = (m[0], (j, sgn), m)
    print(f"best single contact: c{best_s[1][0]} ({'cathode' if best_s[1][1]<0 else 'anode'}) "
          f"margin {best_s[0]:.3f}  (SCB {best_s[2][1]*100:.0f}% @ trunk {best_s[2][2]*100:.0f}%)", flush=True)

    # ---- optimized steering: smooth surrogate, multi-restart ----
    def proj(w):
        w = w - w.mean(); n = np.linalg.norm(w)
        return w / n if n > 1e-12 else w

    BETA = 8.0

    def seg_lse(af):                                    # per-fiber log-sum-exp (smooth max)
        m = np.maximum.reduceat(af, starts)
        e = np.exp(BETA * (af - np.repeat(m, np.diff(off))))
        s = np.add.reduceat(e, starts)
        return m + np.log(s) / BETA

    def neg_obj(w):
        wp = proj(w); lse = seg_lse(AF @ wp)
        return -(lse[bidx == 1].mean() - lse[bidx == 0].mean())

    rng = np.random.default_rng(0)
    best_w, best_m = None, (-1, None, None)
    for r in range(40):
        w0 = rng.standard_normal(20) if r else (best_s_dir := np.zeros(20))
        if r == 0:                                     # seed with the best single contact
            w0 = np.zeros(20); w0[best_s[1][0]] = best_s[1][1]
        res = minimize(neg_obj, w0, method="L-BFGS-B",
                       options=dict(maxiter=300, ftol=1e-9))
        wp = proj(res.x); m = maxmargin(peak_af(wp))
        if m[0] > best_m[0]:
            best_m, best_w = m, wp
    print(f"OPTIMIZED steering:  margin {best_m[0]:.3f}  (SCB {best_m[1]*100:.0f}% @ trunk {best_m[2]*100:.0f}%)", flush=True)
    print(f"  -> selectivity gain: {best_m[0]/max(best_s[0],1e-9):.2f}x the best single contact", flush=True)

    # contact angle (from each contact's lead-field-peak point) to interpret the pattern
    Ve = np.load(D, allow_pickle=True)["Ve_mat"]
    ctr = flat[:, :2].mean(0)
    ang = np.array([np.degrees(np.arctan2(flat[np.argmax(np.abs(Ve[:, c])), 1] - ctr[1],
                                          flat[np.argmax(np.abs(Ve[:, c])), 0] - ctr[0])) % 360
                    for c in range(20)])
    print("\noptimized current pattern (contact: weight @ angle):")
    o = np.argsort(-best_w)
    for c in o[:4]:
        print(f"  ANODE  c{c:2d}: {best_w[c]:+.2f} @ {ang[c]:3.0f} deg")
    for c in o[-4:]:
        print(f"  CATHODE c{c:2d}: {best_w[c]:+.2f} @ {ang[c]:3.0f} deg")

    # ---- recruitment (ROC-like) curves ----
    def curve(w):
        pk = peak_af(w); o = np.argsort(-pk); pos = pk[o] > 0
        fs = np.cumsum((bidx[o] == 1) & pos) / n_scb
        ft = np.cumsum((bidx[o] == 0) & pos) / n_tr
        return ft, fs
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ws = np.zeros(20); ws[best_s[1][0]] = best_s[1][1]
    ftx, fsx = curve(ws); ax.plot(ftx * 100, fsx * 100, lw=2, color="#1f77b4",
                                  label=f"best single contact (margin {best_s[0]:.2f})")
    fto, fso = curve(best_w); ax.plot(fto * 100, fso * 100, lw=2.5, color="#e6550d",
                                      label=f"optimized 20-contact steering (margin {best_m[0]:.2f})")
    ax.plot([0, 100], [0, 100], "--", c="gray", lw=0.8, label="no selectivity")
    ax.set_xlabel("% trunk fibers recruited"); ax.set_ylabel("% SCB fibers recruited")
    ax.set_title("Current steering vs single contact\n(4x5 cuff, ~20 mm from bifurcation, real-3D fibers)")
    ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal"); fig.tight_layout()
    OUTP.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUTP, dpi=200)
    np.savez(ROOT / "paper_figs/out/data/new_human_steering.npz",
             w_opt=best_w, margin_opt=best_m[0], margin_single=best_s[0], angles=ang)
    print(f"\nwrote {OUTP}")


if __name__ == "__main__":
    main()
