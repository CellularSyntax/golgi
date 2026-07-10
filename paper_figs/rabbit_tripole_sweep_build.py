# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Build CLEAN multipolar stimulation patterns on the 4x5 ring cuff for EVERY rabbit cuff
position in the sweep, then write them as columns of a per-position paths_Ve.npz so
fig5_thresholds.py can run real NEURON thresholds on each. RABBIT analogue of
new_human_tripole_sweep_build.py (identical topology + cathode-selection logic).

The 4x5 ring array is laid out ROW-MAJOR by golgi (electrode_patches.py): contact id
c -> row = c//5 (axial, z=[-1.5,-0.5,0.5,1.5] mm), col = c%5 (angular, ~72 deg by DESIGN).
  column k  = contacts {k, k+5, k+10, k+15}   (4 axial rows at the same design angle)
  row r     = contacts {5r .. 5r+4}           (5 angular columns, design-adjacent = +/-72 deg)

Adjacency is taken from the cuff DESIGN; the cathode / SCB column is chosen by actual FIELD
COUPLING to the SCB fibers (robust to the cuff-fit angle distortion on the non-circular nerve).

Patterns (charge-balanced; a column is the per-unit CATHODIC lead field, cathode +Ve, anode -Ve):
  mono           : SCB-column cathode only
  long_tripole   : cathode + 2 AXIAL anodes  (same column, rows +/-1)
  trans_tripole  : cathode + 2 ANGULAR anodes (same row, design cols +/-72deg)
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).parent.parent
DD = ROOT / "paper_figs/out/data"
OUT = DD / "rabbit_tripole_sweep"

# cuff position = offset (mm) along the nerve centerline from the proximal (trunk, z~0) end.
# The rabbit vagus branches "hard" from z ~ 8 mm (rabbit_cuff_positions_sanity.py: SCB-trunk
# separation grows past z~8); all swept offsets sit on the COMMON TRUNK (3-6 mm) so the cuff
# is upstream of the bifurcation. distance-from-branch = BRANCH_Z - offset.
BRANCH_Z = 8.0
POSITIONS = [
    ("off3_4x5", 3.0),
    ("off4_4x5", 4.0),
    ("off5_4x5", 5.0),
    ("off6_4x5", 6.0),
]
N_ROW, N_COL = 4, 5


def col_of(c):
    return c % N_COL


def row_of(c):
    return c // N_COL


def build_one(tag, offset):
    d = np.load(DD / f"rabbit_branch_{tag}/paths_Ve.npz", allow_pickle=True)
    Ve = np.asarray(d["Ve_mat"], float)            # (N_pts, 20)
    flat = np.asarray(d["paths_flat"], float)
    lens = np.asarray(d["path_lengths"], int)
    bidx = np.asarray(d["branch_idx"], int)         # 0 = trunk, 1 = SCB
    off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    nf = len(lens)
    ctr = flat[:, :2].mean(0)

    # per-fiber peak |Ve| for every contact (the recruitment driver ~ scales with this)
    peak = np.zeros((nf, 20))
    for i in range(nf):
        peak[i] = np.abs(Ve[off[i]:off[i + 1]]).max(0)
    scb = bidx == 1
    trunk = bidx == 0
    # SCB-coupling advantage per contact: how much more it drives SCB than the trunk
    adv = peak[scb].mean(0) / np.maximum(peak[trunk].mean(0), 1e-30)

    # contact apparent (z, angle) -- for plotting/interpretation only (NOT for adjacency)
    cz = np.zeros(20); cang = np.zeros(20)
    for c in range(20):
        k = int(np.argmax(np.abs(Ve[:, c])))
        cz[c] = flat[k, 2] * 1e3
        cang[c] = np.degrees(np.arctan2(flat[k, 1] - ctr[1], flat[k, 0] - ctr[0])) % 360

    # cathode = best SCB-coupling contact among the INNER rows (1,2) so a symmetric
    # longitudinal tripole (a row above AND below) exists.
    inner = [c for c in range(20) if row_of(c) in (1, 2)]
    cathode = int(max(inner, key=lambda c: adv[c]))
    scol, crow = col_of(cathode), row_of(cathode)
    # longitudinal (axial) anodes: same column, the rows immediately above & below
    a_lo = (crow - 1) * N_COL + scol
    a_hi = (crow + 1) * N_COL + scol
    # transverse (angular) anodes: same row, the two DESIGN-adjacent columns (+/-72 deg)
    a_l = crow * N_COL + (scol - 1) % N_COL
    a_r = crow * N_COL + (scol + 1) % N_COL

    pats = {
        "mono":          {cathode: +1.0},
        "long_tripole":  {cathode: +1.0, a_lo: -0.5, a_hi: -0.5},
        "trans_tripole": {cathode: +1.0, a_l: -0.5, a_r: -0.5},
    }

    def field(weights):
        v = np.zeros(len(Ve))
        for c, w in weights.items():
            v += w * Ve[:, c]
        return v

    Vp = np.column_stack([field(w) for w in pats.values()])

    odir = OUT / tag
    odir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(odir / "paths_Ve.npz",
                        paths_flat=flat, Ve_mat=Vp, path_lengths=lens, branch_idx=bidx,
                        contact_ids=np.arange(len(pats)),
                        pattern_names=np.array(list(pats.keys())))
    # circular-mean SCB / trunk angle in the cuff band (for the cross-section panel)
    zc = float(np.median(cz))
    band = np.abs(flat[:, 2] * 1e3 - zc) < 3.0
    fa = np.full(nf, np.nan)
    for i in range(nf):
        sl = slice(off[i], off[i + 1]); m = band[sl]
        if m.any():
            p = flat[sl][m][:, :2].mean(0)
            fa[i] = np.degrees(np.arctan2(p[1] - ctr[1], p[0] - ctr[0])) % 360

    def circmean(a):
        a = np.radians(a[~np.isnan(a)])
        return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)

    dist = BRANCH_Z - offset
    meta = dict(
        tag=tag, offset_mm=offset, dist_from_branch_mm=dist, n_fibers=int(nf),
        n_scb=int(scb.sum()), n_trunk=int(trunk.sum()),
        cathode=cathode, scb_col=int(scol), cath_row=int(crow),
        long_anodes=[int(a_lo), int(a_hi)], trans_anodes=[int(a_l), int(a_r)],
        cath_adv=float(adv[cathode]),
        contact_angle_deg={int(c): float(cang[c]) for c in range(20)},
        contact_z_mm={int(c): float(cz[c]) for c in range(20)},
        scb_angle_deg=circmean(fa[scb]), trunk_angle_deg=circmean(fa[trunk]),
        pattern_names=list(pats.keys()),
    )
    (odir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[{tag}] offset {offset:.0f}mm ({dist:.1f} mm from branch)  cathode c{cathode} "
          f"(col {scol}, row {crow}, SCB-adv {adv[cathode]:.2f}x, ~{cang[cathode]:.0f}deg)")
    print(f"   long  anodes c{a_lo},c{a_hi}  (~{cang[a_lo]:.0f},{cang[a_hi]:.0f}deg, axial)")
    print(f"   trans anodes c{a_l},c{a_r}  (~{cang[a_l]:.0f},{cang[a_r]:.0f}deg, +/-72deg design)")
    print(f"   SCB fibers ~{meta['scb_angle_deg']:.0f}deg ; trunk ~{meta['trunk_angle_deg']:.0f}deg")
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metas = []
    for tag, offset in POSITIONS:
        if not (DD / f"rabbit_branch_{tag}/paths_Ve.npz").exists():
            print(f"[{tag}] MISSING FEM -- skip"); continue
        metas.append(build_one(tag, offset))
    (OUT / "sweep_meta.json").write_text(json.dumps(metas, indent=2))
    print(f"\nwrote {len(metas)} positions -> {OUT}")


if __name__ == "__main__":
    main()
