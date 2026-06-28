# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""gmsh-based multi-domain mesher for the nerve + cuff + muscle FEM.

Drop-in alternative to the hand-rolled PLC assembly + TetGen subprocess
(`pipeline/plc.assemble_multi_domain_plc` + `compute/tetgen_runner`). Same
output contract: returns `(nodes, elems, tags)` —

  nodes : (N, 3) float64, METRES, in the cuff-local frame (cuff axis = +z,
          cuff centred at z = 0), exactly like the TetGen path.
  elems : (M, 4) int64, 0-indexed tetra vertex indices.
  tags  : (M,)  int64, per-tet region id —
          1=endo, 2=saline, 3=silicone, 4=muscle, 5=epi, 7=scar.

so `golgi.app.write_msh22(path, nodes, elems, tags)` and the per-region
surface extraction downstream work unchanged.

Why gmsh: the PLC+TetGen path hand-builds cylinders (n_circ inherited from
the dense nerve loop → extreme-aspect slivers) and earcut caps, then fights
TetGen's exact-predicate boundary recovery (`recoversubfaces` / spurious
self-intersections at the coincident multi-domain seams). gmsh instead
builds the regions as CAD solids, makes them conformal with a boolean
`fragment`, and meshes the whole thing with a quality 3D algorithm and a
uniform-ish size field — no PLC, no caps, no dedup, no SI fighting. On the
real Duke nerve this meshes in one shot with mean tet quality ≈ 0.85 and
correct conformal region interfaces.

Approach (single-slice / prismatic nerve — the µCT-extrusion case):
  1. Slice the nerve surface at the cuff centre to get its cross-section
     contour; smooth + resample to uniform spacing (kills boundary slivers).
  2. Build the nerve SOLID by extruding that contour over the nerve's
     z-extent. Build saline / silicone / muscle as OCC cylinders.
  3. `fragment` everything → one conformal multi-domain assembly.
  4. Mesh with a Box size field (fine in/near the cuff, coarse muscle).
  5. Tag each tet by its centroid geometry (the fragment guarantees a tet
     lies wholly in one region, so the centroid classifies it cleanly).
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np


@contextlib.contextmanager
def _suppress_c_fd():
    """Redirect C-level stdout+stderr (fd 1/2) to /dev/null for the duration.

    gmsh's Netgen optimiser writes "BFGS update error…" etc. straight to the
    C stdout, bypassing General.Terminal=0 — only an fd-level redirect quiets
    it. Python-level logging is flushed first so nothing is lost."""
    sys.stdout.flush()
    sys.stderr.flush()
    save1, save2 = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(save1, 1)
        os.dup2(save2, 2)
        os.close(devnull)
        os.close(save1)
        os.close(save2)


# region ids — must match write_msh22's PhysicalNames
TAG_ENDO = 1
TAG_SALINE = 2
TAG_SILICONE = 3
TAG_MUSCLE = 4
TAG_EPI = 5
TAG_CATHODE = 6     # (legacy bipolar) stimulating contact (Pt)
TAG_SCAR = 7
TAG_ANODE = 8       # (legacy bipolar) return contact (Pt)
# multi-contact: contact k → tag TAG_CONTACT_BASE + k (all platinum σ)
TAG_CONTACT_BASE = 100

# region id → physical name (used by the .msh writer + paraview colouring)
TAG_NAMES = {
    TAG_ENDO: "endo",
    TAG_SALINE: "saline",
    TAG_SILICONE: "silicone",
    TAG_MUSCLE: "muscle",
    TAG_EPI: "epi",
    TAG_CATHODE: "cathode",
    TAG_SCAR: "scar",
    TAG_ANODE: "anode",
}


def tag_name(t: int) -> str:
    """Region name for a tag, incl. multi-contact pads (tag ≥ 100)."""
    t = int(t)
    if t >= TAG_CONTACT_BASE:
        return f"contact_{t - TAG_CONTACT_BASE:02d}"
    return TAG_NAMES.get(t, f"region{t}")


def _nerve_cross_section(
    nerve_pts_m: np.ndarray,
    bnd_tris: np.ndarray,
    z_slice: float,
    *,
    resample_h_m: float,
    smooth_iters: int = 3,
) -> np.ndarray:
    """Ordered, smoothed, uniformly-resampled (x, y) loop of the nerve
    cross-section at z=`z_slice`. Assumes a roughly star-convex section
    (true for µCT-extruded nerves), ordering boundary points by angle
    about their centroid."""
    import pyvista as pv

    faces = np.hstack(
        [np.full((len(bnd_tris), 1), 3, np.int64),
         np.asarray(bnd_tris, np.int64)]
    ).ravel()
    pd = pv.PolyData(np.asarray(nerve_pts_m, float), faces)
    sl = pd.slice(normal=(0, 0, 1), origin=(0, 0, float(z_slice)))
    if sl.n_points < 8:
        raise RuntimeError(
            f"nerve cross-section slice at z={z_slice:.4g} gave "
            f"{sl.n_points} pts — nerve doesn't cross the cuff centre"
        )
    xy = np.asarray(sl.points)[:, :2]
    c = xy.mean(0)
    ring = xy[np.argsort(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))]
    # drop near-duplicate consecutive points
    keep = [0]
    for i in range(1, len(ring)):
        if np.linalg.norm(ring[i] - ring[keep[-1]]) > 1.0e-6:
            keep.append(i)
    ring = ring[keep]
    # light periodic smoothing
    for _ in range(int(smooth_iters)):
        ring = (np.roll(ring, 1, 0) + 2 * ring + np.roll(ring, -1, 0)) / 4.0
    # uniform arc-length resample
    seg = np.linalg.norm(
        np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1,
    )
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    n = max(24, int(round(total / max(resample_h_m, 1.0e-9))))
    s = np.linspace(0.0, total, n, endpoint=False)
    fx = np.concatenate([ring[:, 0], ring[:1, 0]])
    fy = np.concatenate([ring[:, 1], ring[:1, 1]])
    return np.column_stack([np.interp(s, cum, fx), np.interp(s, cum, fy)])


def mesh_nerve_cuff(
    nerve_pts_m: np.ndarray,
    bnd_tris: np.ndarray,
    *,
    L_cuff_m: float,
    R_ci_m: float,
    R_co_m: float,
    muscle_radial_pad_m: float,
    muscle_axial_pad_m: float,
    lc_fine_m: float,
    lc_coarse_m: float,
    fascicle_surfaces=None,
    contacts=None,
    use_epi: bool = False,
    epi_thickness_m: float = 50.0e-6,
    scar_thickness_m: float = 0.0,
    separated_collar_m: float = 0.0,
    on_line=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh the nerve + cuff + muscle multi-domain FEM with gmsh. Returns
    (nodes_m, elems_0idx, tags). `nerve_pts_m` is in the cuff-local frame
    (cuff at origin, +z along the cuff axis); single-slice / prismatic
    nerve (and fascicles).

    Anatomy / region tags:
      * `fascicle_surfaces` (list of (verts_m, faces) prisms) are the
        ENDONEURIUM bundles → tag 1 (endo). The nerve bulk *between* the
        fascicles (inside the nerve outer boundary) is the EPINEURIUM →
        tag 5 (epi). When no fascicles are given, the whole nerve interior
        is tagged endo (single-domain nerve, legacy behaviour).
      * saline → 2, silicone → 3, muscle → 4.
      * `contacts` (list of dicts) are metal electrode pads carved out of
        the inner silicone wall (radius [R_ci, R_ci+thickness_m]), exposed
        to the saline lumen. Each dict:
            {"id": int, "z_m": float, "dz_m": float,
             "phi_rad": float, "dphi_rad": float, "thickness_m": float}
        — axial extent [z_m ± dz_m/2], azimuth [phi_rad ± dphi_rad/2]
        (full ring if dphi_rad ≥ 2π). Contact k → tag 100+id, each a
        distinct paraview region (all platinum σ). Roles (anode/cathode)
        are an FEM/montage concept, kept in electrode_config.json, not the
        mesh — per-contact lead-field solves treat every contact alike.
    `use_epi` / `epi_thickness_m` / the old offset-shell path are
    superseded by the fascicle-based epi/endo split and are ignored when
    fascicles are present.
    """
    import gmsh
    from matplotlib.path import Path as _Path

    say = on_line if on_line is not None else (lambda *_: None)
    P = np.asarray(nerve_pts_m, dtype=np.float64)
    z_min, z_max = float(P[:, 2].min()), float(P[:, 2].max())
    z_lo, z_hi = -L_cuff_m / 2.0, L_cuff_m / 2.0

    # nerve outer (epineurium boundary) cross-section contour at the cuff centre
    ring = _nerve_cross_section(
        P, bnd_tris, 0.0, resample_h_m=lc_fine_m,
    )
    # fascicle (endoneurium) cross-section contours, same z-slice
    fasc_rings = []
    for fv, ff in (fascicle_surfaces or []):
        try:
            fr = _nerve_cross_section(
                np.asarray(fv, float), np.asarray(ff, np.int64), 0.0,
                resample_h_m=lc_fine_m, smooth_iters=2,
            )
            if len(fr) >= 6:
                fasc_rings.append(fr)
        except Exception:                                  # noqa: BLE001
            pass  # fascicle doesn't cross the slice plane — skip
    say(f"  [gmsh] {len(fasc_rings)} fascicle contours "
        f"(of {len(fascicle_surfaces or [])} supplied)")
    r_max = float(np.max(np.hypot(ring[:, 0], ring[:, 1])))
    R_mus = max(float(R_co_m) + 0.5e-3,
                r_max + float(muscle_radial_pad_m))
    z_mus_lo = z_min - float(muscle_axial_pad_m)
    z_mus_hi = z_max + float(muscle_axial_pad_m)
    say(f"  [gmsh] nerve r_max={r_max*1e3:.3f} mm, R_ci={R_ci_m*1e3:.2f}, "
        f"R_co={R_co_m*1e3:.2f}, R_mus={R_mus*1e3:.2f} mm; "
        f"nerve z=[{z_min*1e3:.1f},{z_max*1e3:.1f}], cuff ±{L_cuff_m/2*1e3:.1f}")

    # gmsh.initialize() installs a SIGINT handler, which raises
    # "signal only works in main thread of the main interpreter" when we
    # are called from run_mesh_build's asyncio executor thread. Neutralise
    # signal.signal (no-op) around initialize so the handler install is
    # skipped safely; restore it immediately after.
    import signal as _sig
    _orig_sig = _sig.signal
    _sig.signal = lambda *a, **k: None
    try:
        gmsh.initialize()
    finally:
        _sig.signal = _orig_sig
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("nerve_cuff")
        occ = gmsh.model.occ

        def _extrude_contour(xy, z0, dz):
            pts = [occ.addPoint(float(x), float(y), float(z0)) for x, y in xy]
            lns = [occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                   for i in range(len(pts))]
            surf = occ.addPlaneSurface([occ.addCurveLoop(lns)])
            ex = occ.extrude([(2, surf)], 0.0, 0.0, float(dz))
            return [e[1] for e in ex if e[0] == 3][0]

        nerve_solid = _extrude_contour(ring, z_min, z_max - z_min)
        tools = []
        # optional concentric offset shells around the nerve (epi / scar)
        # built as scaled contours so they hug the irregular section.
        cscale = ring.mean(0)
        def _scaled_ring(offset_m):
            v = ring - cscale
            rr = np.hypot(v[:, 0], v[:, 1])
            f = (rr + offset_m) / np.maximum(rr, 1e-9)
            return cscale + v * f[:, None]

        scar_solid = None
        if scar_thickness_m and scar_thickness_m > 0.0:
            scar_solid = _extrude_contour(
                _scaled_ring(float(scar_thickness_m)), z_min, z_max - z_min,
            )
            tools.append((3, scar_solid))
        # fascicle (endoneurium) solids INSIDE the nerve — the fragment
        # carves them out of the epineurium bulk, giving conformal
        # fascicle/epi interfaces.
        for fr in fasc_rings:
            tools.append((3, _extrude_contour(fr, z_min, z_max - z_min)))

        saline = occ.addCylinder(0, 0, z_lo, 0, 0, L_cuff_m, float(R_ci_m))
        silic = occ.addCylinder(0, 0, z_lo, 0, 0, L_cuff_m, float(R_co_m))
        muscle = occ.addCylinder(
            0, 0, z_mus_lo, 0, 0, z_mus_hi - z_mus_lo, float(R_mus),
        )
        # electrode contacts are NOT separate OCC solids — for many small
        # pads that is slow/fragile. They are carved out of the silicone
        # wall by centroid tagging below (a sub-region of the silicone, so
        # the wall mesh already resolves them when lc_fine is small enough).
        clist = list(contacts or [])
        tools += [(3, nerve_solid), (3, saline), (3, silic)]
        occ.fragment([(3, muscle)], tools)
        occ.synchronize()
        nvol = len(gmsh.model.getEntities(3))
        say(f"  [gmsh] fragment → {nvol} conformal sub-volumes")

        # size field: fine inside/around the cuff, coarse in the muscle bulk
        f_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f_box, "VIn", float(lc_fine_m))
        gmsh.model.mesh.field.setNumber(f_box, "VOut", float(lc_coarse_m))
        pad = 1.0e-3
        gmsh.model.mesh.field.setNumber(f_box, "XMin", -R_co_m - pad)
        gmsh.model.mesh.field.setNumber(f_box, "XMax", R_co_m + pad)
        gmsh.model.mesh.field.setNumber(f_box, "YMin", -R_co_m - pad)
        gmsh.model.mesh.field.setNumber(f_box, "YMax", R_co_m + pad)
        gmsh.model.mesh.field.setNumber(f_box, "ZMin", z_lo - pad)
        gmsh.model.mesh.field.setNumber(f_box, "ZMax", z_hi + pad)
        gmsh.model.mesh.field.setNumber(f_box, "Thickness", 2.0e-3)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_box)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        # Netgen's optimiser spams "BFGS update error2" to the C stdout
        # (harmless line-search hiccups). Quiet it at the fd level so it
        # doesn't flood the GUI/console; the mesh + quality are unaffected.
        with _suppress_c_fd():
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.optimize("Netgen")

        # extract nodes
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        ntags = np.asarray(ntags, np.int64)
        ncoords = np.asarray(ncoords, float).reshape(-1, 3)
        idmap = -np.ones(int(ntags.max()) + 1, np.int64)
        idmap[ntags] = np.arange(len(ntags))
        nodes = ncoords

        # extract tets (type 4)
        etypes, etags_l, enodes_l = gmsh.model.mesh.getElements(3)
        tets = None
        for et, en in zip(etypes, enodes_l):
            if et == 4:
                tets = idmap[np.asarray(en, np.int64).reshape(-1, 4)]
        if tets is None or len(tets) == 0:
            raise RuntimeError("gmsh produced no tetrahedra")
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()

    # ---- region tagging by tet centroid ----
    # The nerve / epi / scar solids only span the nerve's z-extent
    # [z_min, z_max]; the muscle fills the rest of the bbox. A tet whose
    # (x,y) is inside the nerve contour but whose z is OUTSIDE the nerve
    # extent belongs to the muscle (it sits in the muscle column above/
    # below the nerve), so every nerve-shaped region must be gated on z
    # as well as the in-plane contour test — otherwise the nerve gets
    # mis-tagged through the whole muscle column.
    cent = nodes[tets].mean(axis=1)
    cr = np.hypot(cent[:, 0], cent[:, 1])
    cz = cent[:, 2]
    in_cuff = np.abs(cz) <= (L_cuff_m / 2.0 + 1.0e-9)
    if separated_collar_m and separated_collar_m > 0.0 and contacts:
        # Separated electrode cuffs (e.g. a LivaNova bipolar pair): the silicone +
        # saline fill only a short collar around each contact, and the inter-contact
        # gap is bare nerve in muscle. Restrict the cuff (silicone/saline) region to
        # those collars; everything else inside the bbox defaults to muscle.
        _collar = np.zeros(len(tets), dtype=bool)
        for _c in contacts:
            _collar |= np.abs(cz - float(_c["z_m"])) <= (
                float(_c["dz_m"]) / 2.0 + float(separated_collar_m) + 1.0e-9)
        in_cuff &= _collar
    in_nerve_z = (cz >= z_min - 1.0e-9) & (cz <= z_max + 1.0e-9)
    nerve_path = _Path(ring)
    in_nerve = nerve_path.contains_points(cent[:, :2]) & in_nerve_z
    in_fasc = np.zeros(len(tets), dtype=bool)
    for fr in fasc_rings:
        in_fasc |= _Path(fr).contains_points(cent[:, :2])
    in_fasc &= in_nerve_z
    cphi = np.arctan2(cent[:, 1], cent[:, 0])    # tet-centroid azimuth
    tags = np.full(len(tets), TAG_MUSCLE, np.int64)
    # outermost-first so inner assignments win
    tags[in_cuff & (cr <= R_co_m)] = TAG_SILICONE
    # electrode contact metal pads carved out of the silicone wall:
    # r∈(R_ci, R_ci+thk], axial |z−z_c|≤dz/2, azimuth within ±dφ/2 of φ_c
    # (full ring if dphi≥2π). Each contact k → tag TAG_CONTACT_BASE+id.
    for c in (contacts or []):
        zc = float(c["z_m"]); dzc = float(c["dz_m"])
        tcon = float(c["thickness_m"])
        dphi = float(c.get("dphi_rad", 2.0 * np.pi))
        phic = float(c.get("phi_rad", 0.0))
        ctag = TAG_CONTACT_BASE + int(c.get("id", 0))
        band = (np.abs(cz - zc) <= dzc / 2.0 + 1.0e-9) \
            & (cr > R_ci_m + 1.0e-12) & (cr <= R_ci_m + tcon + 1.0e-9)
        if dphi < 2.0 * np.pi - 1.0e-6:
            dd = np.abs((cphi - phic + np.pi) % (2.0 * np.pi) - np.pi)
            band &= dd <= dphi / 2.0
        tags[band] = ctag
    tags[in_cuff & (cr <= R_ci_m)] = TAG_SALINE
    if scar_thickness_m and scar_thickness_m > 0.0:
        scar_path = _Path(_scaled_ring(float(scar_thickness_m)))
        tags[scar_path.contains_points(cent[:, :2])
             & ~in_nerve & in_nerve_z] = TAG_SCAR
    if fasc_rings:
        # nerve bulk between fascicles = epineurium; fascicles = endoneurium
        tags[in_nerve] = TAG_EPI
        tags[in_fasc] = TAG_ENDO
    else:
        # no fascicles → whole nerve interior is endoneurium (single domain)
        tags[in_nerve] = TAG_ENDO

    counts = {int(t): int((tags == t).sum()) for t in np.unique(tags)}
    say(f"  [gmsh] MESHED: {len(nodes):,} nodes, {len(tets):,} tets; "
        f"region tet counts {counts}")
    return nodes, tets.astype(np.int64), tags
