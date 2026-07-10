# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 6 — rabbit real-3D branch-selective stimulation, FEM stage.

The user MESHED the rabbit in the golgi GUI (curved real-3D nerve, 3x4 ring-array
cuff on the trunk) and exported a study bundle. The GUI does NOT solve the FEM,
so we do it here: reconstruct the headless Study geometry from the bundle, then
run golgi's NATIVE per-contact reciprocity lead-field solve (solve_nerve.py,
GOLGI_EMIT_RECORDING via recording_montages) — one unit-current Laplace solve per
contact, sampled along the fibers. By reciprocity the recording lead field equals
the stimulation lead field, so this gives the 12-contact Ve lead-field matrix the
current-steering selectivity analysis needs.

The make-or-break is the cuff FRAME: our headless refit must reproduce the GUI's
R_local_elec so the electrode patches match the on-disk mesh facets. `--check`
verifies that (and the fiber frame) WITHOUT running any FEM.

Usage:
    python paper_figs/rabbit_fem.py --check     # cheap: reconstruct + verify frame
    python paper_figs/rabbit_fem.py --run       # full: native reciprocity FEM
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
PROJ = ROOT / "paper_figs/out/_intermediate/rabbit_gui_study"
SRC = PROJ / "source/Reduced_Reduced_Smoothed_Wrapped_mask_from_object 3_wrapped.nas"
EID = "elec_02"
SCALE = 1e-4


def _recon(verbose=True):
    """Reconstruct the headless Study geom + designs/configs from the bundle.
    Returns (study, design, gui_elec)."""
    import golgi
    pj = json.loads((PROJ / "project.json").read_text())
    ui = pj.get("ui_state", {})
    gui_elec = json.loads((PROJ / "electrodes" / f"{EID}.json").read_text())

    s = golgi.Study.open(PROJ)
    # The bundle ships project.json (ui_state nested), NOT ui_state.json, so
    # Study.open loaded nothing — push the GUI params into the state shim by hand.
    for k, v in ui.items():
        s._state[k] = v
    if verbose:
        print(f"[recon] pushed {len(ui)} ui_state keys; "
              f"designs in ui_state: {[d.get('eid') for d in ui.get('designs', [])]}")

    # geometry: load the source nerve exactly as the GUI did (deterministic PCA)
    info = s.import_nerve(SRC, scale_factor=SCALE)
    if verbose:
        print(f"[recon] import_nerve: {info}")

    # designs + Default config (with per-contact recording montages)
    s.set_electrodes(list(ui.get("designs", [])))
    design = next(d for d in s._state.designs if d.get("eid") == EID)

    # refit the cuff frame at its trunk offset — must match the GUI's mesh frame
    from golgi.scene.cuff_fit import refit_design_geometry
    ok = refit_design_geometry(EID, geom=s._geom, state=s._state)
    if verbose:
        print(f"[recon] refit -> {ok}")
    return s, design, gui_elec


def _frame_check(design, gui_elec):
    gm = gui_elec["_golgi_meta"]
    R_gui = np.asarray(gm["R_local_elec"], float).reshape(3, 3)
    R_mine = np.asarray(design.get("R_local_elec"), float).reshape(3, 3)
    dR = float(np.abs(R_gui - R_mine).max())
    rci_gui, rci_mine = float(gm["R_ci_m"]), float(design.get("R_ci_m"))
    rco_gui, rco_mine = float(gm["R_co_m"]), float(design.get("R_co_m"))
    print("\n=== FRAME CHECK (mine vs GUI) ===")
    print(f"  R_local_elec  max|Δ| = {dR:.3e}   (want ~0)")
    print(f"  R_ci_m        mine={rci_mine*1e3:.4f} mm  gui={rci_gui*1e3:.4f} mm  "
          f"Δ={abs(rci_mine-rci_gui)*1e6:.2f} µm")
    print(f"  R_co_m        mine={rco_mine*1e3:.4f} mm  gui={rco_gui*1e3:.4f} mm")
    ok = dR < 1e-3 and abs(rci_mine - rci_gui) < 5e-6
    print(f"  => frame {'MATCHES' if ok else 'MISMATCH — will force GUI frame'}")
    return ok


def _fiber_frame_check(study):
    fz = np.load(PROJ / "nerve_paths_fibers.npz", allow_pickle=True)
    flat = np.asarray(fz["paths_flat"], float)
    lens = np.asarray(fz["path_lengths"], np.int64)
    raw = study._geom.nerve["pts_raw"]
    print("\n=== FIBER FRAME CHECK ===")
    print(f"  fibers: {len(lens)} paths, {len(flat)} pts, seed_end={fz['seed_end']}")
    print(f"  fiber  bbox (m): x[{flat[:,0].min():.4f},{flat[:,0].max():.4f}] "
          f"y[{flat[:,1].min():.4f},{flat[:,1].max():.4f}] "
          f"z[{flat[:,2].min():.4f},{flat[:,2].max():.4f}]")
    print(f"  raw nerve bbox (m): x[{raw[:,0].min():.4f},{raw[:,0].max():.4f}] "
          f"y[{raw[:,1].min():.4f},{raw[:,1].max():.4f}] "
          f"z[{raw[:,2].min():.4f},{raw[:,2].max():.4f}]")
    # are the fibers inside the raw bbox (=> raw frame) ?
    inside = ((flat[:, 0] >= raw[:, 0].min() - 1e-3) & (flat[:, 0] <= raw[:, 0].max() + 1e-3)
              & (flat[:, 2] >= raw[:, 2].min() - 1e-3) & (flat[:, 2] <= raw[:, 2].max() + 1e-3))
    print(f"  fibers within raw nerve bbox: {100*inside.mean():.0f}%  "
          f"=> {'RAW frame' if inside.mean() > 0.9 else 'NOT raw frame (already transformed?)'}")


OUT_NERVE = ROOT / "paper_figs/out/data/rabbit_branch"   # holds paths_Ve.npz for the engine
RECDIR = PROJ / "designs" / EID / "recording"


def _load_raw_fibers():
    fz = np.load(PROJ / "nerve_paths_fibers.npz", allow_pickle=True)
    flat = np.asarray(fz["paths_flat"], float)
    lens = np.asarray(fz["path_lengths"], np.int64)
    out, off = [], 0
    for L in lens:
        out.append(flat[off:off + int(L)]); off += int(L)
    return out, lens


def _run(s, design):
    """Populate geom fibers + recording montages for all 12 contacts, run the
    native golgi reciprocity FEM (one unit-current lead-field solve per contact,
    sampled along the fibers → designs/<eid>/recording/V_e_rec_<id>.npz)."""
    raw_fibers, _ = _load_raw_fibers()
    s._geom.fiber_paths_raw = raw_fibers
    # The mesh was built in the GUI (not via run_mesh), so the in-memory mesh
    # flags the FEM driver guards on (fem.py:171) were never set — set them to
    # point at the on-disk per-design mesh.
    s._geom.msh_path = str(PROJ / "designs" / EID / "nerve.msh")
    s._state.has_mesh = True
    # Skip the per-contact + per-pair impedance dual-solves — pure overhead for
    # the lead fields (the default 12-contact wiring makes ~36 per-pair solves).
    # We only want the reciprocity (recording) lead fields.
    s._state.emit_impedance = False
    if not getattr(s._geom, "R_ci", None):
        s._geom._R_ci_cached = float(design.get("R_ci_m"))
        s._geom._R_co_cached = float(design.get("R_co_m"))
    n_contacts = int(design.get("array_n_rows", 3)) * int(design.get("array_n_cols", 4))
    montages = [{"mid": f"rec{i}", "label": f"rec{i}", "kind": "bipolar",
                 "plus_contact": 2 * i, "minus_contact": 2 * i + 1}
                for i in range(n_contacts // 2)]            # covers ids 0..11
    cfgs = list(s._state.configs)
    for c in cfgs:
        if c.get("design_id") == EID:
            c["recording_montages"] = montages
    s._state.configs = cfgs
    print(f"[run] {len(raw_fibers)} fibers into geom; recording montages cover "
          f"contacts 0..{n_contacts - 1}; launching native run_fem ...", flush=True)
    res = s.run_fem()
    print(f"[run] run_fem -> {res}", flush=True)


def _assemble():
    """Stack the per-contact reciprocity lead fields into Ve_mat + add branch
    labels (nearest distal cap), in the engine's paths_Ve.npz format."""
    cols, flat, lens = {}, None, None
    for f in sorted(RECDIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if "Ve_flat" not in d.files:
            continue
        cid = int(d["contact_id"])
        cols[cid] = np.asarray(d["Ve_flat"], float)
        flat = np.asarray(d["paths_flat"], float)         # cuff-local frame
        lens = np.asarray(d["path_lengths"], np.int64)
    if not cols:
        raise SystemExit(f"no V_e_rec_*.npz in {RECDIR} — did run_fem record?")
    cids = sorted(cols)
    # solve_nerve.py's reciprocity stores the lead field with the opposite sign
    # convention to solve_nerve_ci.py (the duke Fig 4/5 solver): the CI solver
    # gives Ve>0 for +current injection, the reciprocity gives Ve<0. Flip so the
    # rabbit uses the SAME (+current -> +V) convention as every other nerve.
    Ve_mat = -np.column_stack([cols[c] for c in cids])     # [N_pts, n_contacts]
    print(f"[assemble] {len(cids)} contacts {cids}; Ve_mat {Ve_mat.shape}; "
          f"NaN frac {np.mean(~np.isfinite(Ve_mat)):.3f}; "
          f"range [{np.nanmin(Ve_mat):.3g},{np.nanmax(Ve_mat):.3g}] (flipped to +convention)")

    # branch labels: each fiber's DISTAL endpoint (high-z, seed_end=low) → nearest
    # branch cap centroid (RAW frame). Computed from the RAW fibers (same order).
    raw_fibers, raw_lens = _load_raw_fibers()
    caps = json.loads((PROJ / "nerve_paths_caps.json").read_text())
    cap_c = np.asarray(caps["branch_cap_centroids_m"], float)    # (n_branch, 3) raw
    cap_area = np.asarray(caps.get("branch_cap_areas_m2", []), float)
    bidx = np.empty(len(raw_fibers), np.int64)
    for i, p in enumerate(raw_fibers):
        tip = p[np.argmax(p[:, 2])]                        # high-z distal tip
        bidx[i] = int(np.argmin(np.linalg.norm(cap_c - tip, axis=1)))
    print(f"[assemble] branch split: " +
          ", ".join(f"branch{b}({cap_area[b]*1e6:.1f}mm²,n={int((bidx==b).sum())})"
                    for b in range(len(cap_c))))

    OUT_NERVE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NERVE / "paths_Ve.npz", paths_flat=flat, path_lengths=lens,
                        branch_idx=bidx, Ve_mat=Ve_mat,
                        contact_ids=np.asarray(cids, np.int64),
                        units="V_per_A", inject_A=1.0)
    print(f"[assemble] wrote {OUT_NERVE/'paths_Ve.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    if not (a.check or a.run or a.assemble):
        a.check = True

    if a.assemble and not a.run:
        _assemble(); return

    s, design, gui_elec = _recon()
    ok = _frame_check(design, gui_elec)
    _fiber_frame_check(s)
    print("\n=== CONFIGS ===")
    for c in s._state.configs:
        print(f"  cid={c.get('cid')} design={c.get('design_id')} "
              f"polarities={len(c.get('contact_polarities') or [])} "
              f"rec_montages={len(c.get('recording_montages') or [])}")

    if a.run:
        if not ok:
            raise SystemExit("frame mismatch — aborting FEM")
        _run(s, design)
        _assemble()


if __name__ == "__main__":
    main()
