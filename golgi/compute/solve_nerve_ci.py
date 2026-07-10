# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Per-contact lead-field contact-impedance FEM solve for the batch meshes.

Quasi-static EQS (-div(σ∇V)=0) with:
  * the perineurium as a CONTACT-IMPEDANCE sheet at the endo↔epi interface
    (Rs = peri_thk/σ_peri) — a two-field (endo submesh + rest submesh) block
    system with a Robin coupling at Γ. See project_dolfinx_contact_impedance.
  * anisotropic σ (endo/muscle longitudinal along +z); contacts (tag ≥ 100)
    are platinum.
  * a MULTI-CONTACT cuff: the block matrix is factorised ONCE (MUMPS) and one
    unit-current RHS is solved per contact (contact k = source, muscle-bbox =
    ground) → a per-contact LEAD FIELD in V/A. Any montage = a weighted sum.

Outputs (in the sample dir):
  * fiber_potentials.h5  — pyFibers/NEURON: per fiber, Ve[n_pts × n_contacts]
                           (V/A) + s_m, xyz_m, fascicle; contact metadata.
  * paths_Ve.npz         — same, flat (Ve_mat [N_pts × n_contacts]).
  * fibers.vtp           — re-written with per-point Ve of a representative
                           (centre) contact, for paraview colouring.
  * Ve_rest.xdmf / Ve_endo.xdmf / E_rest.xdmf — volume fields of the centre
                           contact's unit-injection solve (paraview).
  * ci_solve_info.json   — diagnostics.

Standalone:  python -m golgi.compute.solve_nerve_ci <sample_dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector, create_vector, set_bc, assign,
)
import dolfinx.la.petsc as _lap
from mpi4py import MPI
from petsc4py import PETSc

from golgi.conductivity.materials import (
    MATERIAL_SIGMA, sigma_transverse as _st, sigma_longitudinal as _sl,
)
from golgi.compute.gmsh_mesher import TAG_CONTACT_BASE

T_ENDO, T_SALINE, T_SILICONE, T_MUSCLE, T_EPI, T_SCAR = 1, 2, 3, 4, 5, 7
_PT = _st("platinum")
SIG_T = {T_ENDO: _st("endoneurium"), T_SALINE: _st("saline"),
         T_SILICONE: _st("silicone"), T_MUSCLE: _st("muscle"),
         T_EPI: _st("epineurium"), T_SCAR: _st("encapsulation")}
SIG_L = dict(SIG_T); SIG_L[T_ENDO] = _sl("endoneurium"); SIG_L[T_MUSCLE] = _sl("muscle")
SIG_PERI = MATERIAL_SIGMA["perineurium"]


def _sig_t(t):
    return _PT if int(t) >= TAG_CONTACT_BASE else SIG_T.get(int(t), 1.0)


def _sig_l(t):
    return _PT if int(t) >= TAG_CONTACT_BASE else SIG_L.get(int(t), _sig_t(t))


def _tag_by_cell(domain, cell_tags):
    tdim = domain.topology.dim
    n = (domain.topology.index_map(tdim).size_local
         + domain.topology.index_map(tdim).num_ghosts)
    out = np.zeros(n, np.int32)
    out[cell_tags.indices] = cell_tags.values
    return out


def _sigma_tensor(submesh, parent_cells, tbc):
    tdim = submesh.topology.dim
    T = fem.functionspace(submesh, ("DG", 0, (3, 3)))
    f = fem.Function(T, name="sigma")
    ncell = (submesh.topology.index_map(tdim).size_local
             + submesh.topology.index_map(tdim).num_ghosts)
    ptags = tbc[np.asarray(parent_cells)[:ncell]]
    st = np.array([_sig_t(t) for t in ptags])
    sl = np.array([_sig_l(t) for t in ptags])
    a = f.x.array.reshape(-1, 9)
    a[:] = 0.0
    a[:, 0] = st
    a[:, 4] = st
    a[:, 8] = sl
    return f


def _interface_entities(domain, cell_tags, tag_a, tag_b, first):
    """(c0,lf0,c1,lf1) for facets between tag_a & tag_b cells, `first` as '+'."""
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)
    f2c = domain.topology.connectivity(fdim, tdim)
    c2f = domain.topology.connectivity(tdim, fdim)
    tbc = _tag_by_cell(domain, cell_tags)
    fa, fo = f2c.array, f2c.offsets
    two = np.where(np.diff(fo) == 2)[0]
    c0, c1 = fa[fo[two]], fa[fo[two] + 1]
    t0, t1 = tbc[c0], tbc[c1]
    sel = ((t0 == tag_a) & (t1 == tag_b)) | ((t0 == tag_b) & (t1 == tag_a))
    fac, cc0, cc1, tt1 = two[sel], c0[sel], c1[sel], t1[sel]
    swap = tt1 == first
    cc0c, cc1c = cc0.copy(), cc1.copy()
    cc0[swap], cc1[swap] = cc1c[swap], cc0c[swap]
    ca, co = c2f.array, c2f.offsets

    def _lf(cells, facets):
        out = np.empty(len(cells), np.int32)
        for k in range(len(cells)):
            c = cells[k]
            out[k] = np.where(ca[co[c]:co[c + 1]] == facets[k])[0][0]
        return out

    ents = np.empty(len(fac) * 4, np.int32)
    ents[0::4], ents[1::4] = cc0, _lf(cc0, fac)
    ents[2::4], ents[3::4] = cc1, _lf(cc1, fac)
    return ents


def _sample(func, pts, submesh):
    from dolfinx import geometry
    pts = np.ascontiguousarray(pts, np.float64)
    tree = geometry.bb_tree(submesh, submesh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(submesh, cand, pts)
    cells = np.full(len(pts), -1, np.int64)
    for i in range(len(pts)):
        lc = coll.links(i)
        if len(lc):
            cells[i] = lc[0]
    ok = cells >= 0
    vals = np.full(len(pts), np.nan)
    if ok.any():
        vals[ok] = func.eval(pts[ok], cells[ok]).reshape(-1)
    return vals, ok


def _locate_cells(pts, submesh):
    """Containing cell per point (-1 = off-mesh). Computed once so a fixed point
    set can be re-evaluated for many functions (each contact's V and E)."""
    from dolfinx import geometry
    pts = np.ascontiguousarray(pts, np.float64)
    tree = geometry.bb_tree(submesh, submesh.topology.dim)
    coll = geometry.compute_colliding_cells(
        submesh, geometry.compute_collisions_points(tree, pts), pts)
    cells = np.full(len(pts), -1, np.int64)
    for i in range(len(pts)):
        lc = coll.links(i)
        if len(lc):
            cells[i] = lc[0]
    return cells, cells >= 0


def _eval_at(func, pts, cells, ok, ncomp=1):
    """Evaluate `func` at pre-located points; off-mesh -> NaN."""
    out = np.full((len(pts), ncomp), np.nan)
    if ok.any():
        out[ok] = func.eval(np.ascontiguousarray(pts[ok], np.float64),
                            cells[ok]).reshape(-1, ncomp)
    return out if ncomp > 1 else out[:, 0]


def _efield(submesh, u):
    """E = -grad(u) projected onto a CG1 vector space (sample-able per point)."""
    W = fem.functionspace(submesh, ("Lagrange", 1, (3,)))
    E = fem.Function(W)
    E.interpolate(fem.Expression(-ufl.grad(u), W.element.interpolation_points))
    return E


def _fill_nan_per_fiber(Ve, lens):
    out = np.asarray(Ve, float).copy()
    off = 0
    for L in lens:
        L = int(L)
        seg = out[off:off + L]
        m = np.isfinite(seg)
        if m.any() and not m.all():
            idx = np.arange(L)
            seg[~m] = np.interp(idx[~m], idx[m], seg[m])
            out[off:off + L] = seg
        off += L
    return out


def run_ci_solve(sample_dir, inject_A: float = 1.0) -> dict:
    comm = MPI.COMM_WORLD
    d = Path(sample_dir)
    mc = json.loads((d / "mesh_config.json").read_text(encoding="utf-8"))
    ec = json.loads((d / "electrode_config.json").read_text(encoding="utf-8"))
    sp = float(mc.get("sigma_peri") or SIG_PERI)
    Rs = (float(mc["peri_thk_m"]) / sp
          if mc.get("perineurium_ci") and mc.get("peri_thk_m") else None)

    md = io.gmsh.read_from_msh(str(d / "nerve.msh"), comm, gdim=3)
    domain, ct = md.mesh, md.cell_tags
    tdim = domain.topology.dim
    fdim = tdim - 1
    tbc = _tag_by_cell(domain, ct)
    utags = [int(t) for t in np.unique(ct.values)]
    contact_tags = sorted(t for t in utags if t >= TAG_CONTACT_BASE)
    rest_bulk_tags = [t for t in (T_SALINE, T_SILICONE, T_MUSCLE, T_EPI, T_SCAR)
                      if t in utags] + contact_tags

    all_cells = np.arange(domain.topology.index_map(tdim).size_local,
                          dtype=np.int32)
    endo_cells = ct.find(T_ENDO).astype(np.int32)
    rest_cells = np.setdiff1d(all_cells, endo_cells).astype(np.int32)
    sub_e, e_map, _, _ = mesh.create_submesh(domain, tdim, endo_cells)
    sub_r, r_map, _, _ = mesh.create_submesh(domain, tdim, rest_cells)

    Ee = fem.functionspace(sub_e, ("Lagrange", 1))
    Er = fem.functionspace(sub_r, ("Lagrange", 1))
    ue, ve = ufl.TrialFunction(Ee), ufl.TestFunction(Ee)
    ur, vr = ufl.TrialFunction(Er), ufl.TestFunction(Er)
    sig_e = _sigma_tensor(sub_e, endo_cells, tbc)
    sig_r = _sigma_tensor(sub_r, rest_cells, tbc)

    # one dS for Γ (id 1) + each contact's saline interface (id 2+i)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    gam = _interface_entities(domain, ct, T_ENDO, T_EPI, first=T_ENDO)
    sub_data = [(1, gam)]
    cid_of = {}
    for i, t in enumerate(contact_tags):
        sub_data.append((2 + i, _interface_entities(domain, ct, T_SALINE, t,
                                                     first=T_SALINE)))
        cid_of[t] = 2 + i
    dS = ufl.Measure("dS", domain=domain, subdomain_data=sub_data)
    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    A_of = {t: comm.allreduce(fem.assemble_scalar(
        fem.form(one('+') * dS(cid_of[t]))), op=MPI.SUM) for t in contact_tags}

    g = fem.Constant(domain, PETSc.ScalarType((1.0 / Rs) if Rs else 0.0))
    a00 = ufl.inner(ufl.dot(sig_e, ufl.grad(ue)), ufl.grad(ve)) * dx(T_ENDO) \
        + g * ue('+') * ve('+') * dS(1)
    a11 = sum(ufl.inner(ufl.dot(sig_r, ufl.grad(ur)), ufl.grad(vr)) * dx(t)
              for t in rest_bulk_tags) + g * ur('-') * vr('-') * dS(1)
    a01 = -g * ur('-') * ve('+') * dS(1)
    a10 = -g * ue('+') * vr('-') * dS(1)
    a_form = fem.form([[a00, a01], [a10, a11]], entity_maps=[e_map, r_map])

    # ground V=0 on the muscle outer wall + axial caps of the rest submesh
    sub_r.topology.create_connectivity(fdim, tdim)
    xr = sub_r.geometry.x
    rmax = float(np.hypot(xr[:, 0], xr[:, 1]).max())
    zlo, zhi = float(xr[:, 2].min()), float(xr[:, 2].max())
    gnd = mesh.locate_entities_boundary(
        sub_r, fdim, lambda x: (np.hypot(x[0], x[1]) > 0.985 * rmax)
        | (x[2] < zlo + 1e-6) | (x[2] > zhi - 1e-6))
    bc = fem.dirichletbc(PETSc.ScalarType(0.0),
                         fem.locate_dofs_topological(Er, fdim, gnd), Er)

    # assemble A ONCE; MUMPS factorisation reused across all contact RHS
    A = assemble_matrix(a_form, bcs=[bc], kind="mpi")
    A.assemble()
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    x = create_vector([Ee, Er], kind="mpi")
    ueh, urh = fem.Function(Ee), fem.Function(Er)
    L0 = fem.Constant(domain, PETSc.ScalarType(0.0)) * ve * dx(T_ENDO)

    fpath = d / "nerve_paths_fibers.npz"
    fz = np.load(fpath, allow_pickle=True) if fpath.exists() else None
    flat = np.asarray(fz["paths_flat"], float) if fz is not None else None
    lens = np.asarray(fz["path_lengths"], np.int64) if fz is not None else None
    bidx = (np.asarray(fz["branch_idx"], np.int64) if fz is not None
            and "branch_idx" in fz else
            (np.zeros(len(lens), np.int64) if lens is not None else None))

    centre = contact_tags[len(contact_tags) // 2] if contact_tags else None
    # also sample each contact's lead field on the epineurium + perineurium
    # surfaces (for surface-coloured renders).  surf_*.vtp are in metres, in the
    # same cuff-local frame as the mesh; epi <- rest solution, peri (endo
    # boundary) <- endo solution.
    import pyvista as _pv
    sp_epi, sp_endo, sp_sal = (d / "surf_epi.vtp", d / "surf_endo.vtp",
                               d / "surf_saline.vtp")
    epi_pts = (np.asarray(_pv.read(str(sp_epi)).points, float)
               if sp_epi.exists() else None)
    peri_pts = (np.asarray(_pv.read(str(sp_endo)).points, float)
                if sp_endo.exists() else None)
    sal_pts = (np.asarray(_pv.read(str(sp_sal)).points, float)
               if sp_sal.exists() else None)
    epi_loc = _locate_cells(epi_pts, sub_r) if epi_pts is not None else None
    peri_loc = _locate_cells(peri_pts, sub_e) if peri_pts is not None else None
    sal_loc = _locate_cells(sal_pts, sub_r) if sal_pts is not None else None
    Ve_cols, on_mesh = [], None
    Ve_epi_cols, E_epi_cols, Ve_peri_cols, E_peri_cols = [], [], [], []
    Ve_sal_cols, E_sal_cols = [], []
    # cross-section grids at each electrode-row z-plane (for slice plots): full
    # cuff extent; field = endo solution inside fascicles, else rest solution.
    zrows = sorted(set(round(float(p["z"]), 7) for p in ec.get("patches", [])))
    Rxs = max((float(p.get("R", 0.0)) for p in ec.get("patches", [])), default=0.0)
    if Rxs <= 0.0:                       # headless configs lack the GUI per-patch 'R'
        Rxs = float(mc.get("R_cuff_inner") or 2.0e-3)
    Rxs += 1.1e-3
    NGX, xsec_all = 161, None
    if zrows and Rxs > 0:
        _gx = np.linspace(-Rxs, Rxs, NGX)
        _GX, _GY = np.meshgrid(_gx, _gx, indexing="xy")
        xsec_xy = np.column_stack([_GX.ravel(), _GY.ravel()])
        xsec_all = np.vstack([np.column_stack(
            [xsec_xy, np.full(len(xsec_xy), zc)]) for zc in zrows])
        xsec_loc_e = _locate_cells(xsec_all, sub_e)
        xsec_loc_r = _locate_cells(xsec_all, sub_r)
        Ve_xsec_cols, E_xsec_cols = [], []
    for t in contact_tags:
        Jt = PETSc.ScalarType(inject_A / A_of[t]) if A_of[t] > 0 else PETSc.ScalarType(0.0)
        L1 = fem.Constant(domain, Jt) * vr('+') * dS(cid_of[t])
        L_form = fem.form([L0, L1], entity_maps=[e_map, r_map])
        b = assemble_vector(L_form, kind="mpi")
        _lap._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(b, [[], [bc]])
        ksp.solve(b, x)
        _lap._ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        assign(x, [ueh, urh])
        b.destroy()
        if flat is not None:
            ve_k, ok = _sample(ueh, flat, sub_e)
            Ve_cols.append(_fill_nan_per_fiber(ve_k, lens))
            on_mesh = ok if on_mesh is None else on_mesh
        need_rest_E = epi_pts is not None or sal_pts is not None or xsec_all is not None
        need_endo_E = peri_pts is not None or xsec_all is not None
        E_rest = _efield(sub_r, urh) if need_rest_E else None
        E_endo = _efield(sub_e, ueh) if need_endo_E else None
        if epi_pts is not None:                       # epi <- rest solution + E
            ec_, eo_ = epi_loc
            Ve_epi_cols.append(_eval_at(urh, epi_pts, ec_, eo_))
            E_epi_cols.append(_eval_at(E_rest, epi_pts, ec_, eo_, 3))
        if sal_pts is not None:                       # saline <- rest solution + E
            sc_, so_ = sal_loc
            Ve_sal_cols.append(_eval_at(urh, sal_pts, sc_, so_))
            E_sal_cols.append(_eval_at(E_rest, sal_pts, sc_, so_, 3))
        if peri_pts is not None:                      # peri <- endo solution + E
            pc_, po_ = peri_loc
            Ve_peri_cols.append(_eval_at(ueh, peri_pts, pc_, po_))
            E_peri_cols.append(_eval_at(E_endo, peri_pts, pc_, po_, 3))
        if xsec_all is not None:                      # cross-section grids
            xe, xeo = xsec_loc_e; xr, xro = xsec_loc_r
            ve_e = _eval_at(ueh, xsec_all, xe, xeo)
            ve_r = _eval_at(urh, xsec_all, xr, xro)
            Ve_xsec_cols.append(np.where(np.isfinite(ve_e), ve_e, ve_r))
            Ee = _eval_at(E_endo, xsec_all, xe, xeo, 3)
            Er = _eval_at(E_rest, xsec_all, xr, xro, 3)
            E_xsec_cols.append(np.where(np.isfinite(Ee[:, :1]), Ee, Er))
        if t == centre:
            _write_volume(comm, d, sub_e, sub_r, ueh, urh)

    info = {"n_contacts": len(contact_tags), "contact_tags": contact_tags,
            "Rs_peri": Rs, "inject_A": float(inject_A),
            "I_stim_configured_A": float(ec.get("I_stim", 1e-3)),
            "centre_contact_tag": centre,
            "contact_area_m2": {int(t): float(A_of[t]) for t in contact_tags}}

    if flat is not None and Ve_cols:
        Ve_mat = np.column_stack(Ve_cols)               # [N_pts, n_contacts]
        # A few boundary fibres are seeded 1-20 µm OUTSIDE the meshed endoneurium
        # (seeding outline vs tet endo mismatch), so their whole trace is off-mesh
        # → NaN, which _fill_nan_per_fiber can't interpolate. Fill each remaining
        # NaN point from its nearest on-mesh point so NEURON never sees NaN.
        bad = ~np.isfinite(Ve_mat).all(1)
        if bad.any() and (~bad).any():
            from scipy.spatial import cKDTree
            nn = cKDTree(flat[~bad]).query(flat[bad])[1]
            Ve_mat[bad] = Ve_mat[~bad][nn]
            info["fiber_pts_nan_filled"] = int(bad.sum())
        cmeta = _contact_meta(ec, contact_tags)
        np.savez(d / "paths_Ve.npz", paths_flat=flat, path_lengths=lens,
                 branch_idx=bidx, Ve_mat=Ve_mat,
                 contact_ids=np.array([c["id"] for c in cmeta]),
                 units="V_per_A", inject_A=inject_A)
        _write_fiber_h5(d / "fiber_potentials.h5", flat, lens, bidx, Ve_mat,
                        cmeta, inject_A, float(ec.get("I_stim", 1e-3)),
                        mc.get("perineurium_species"))
        ccol = contact_tags.index(centre)
        _color_fibers_vtp(d / "fibers.vtp", Ve_mat[:, ccol], centre - TAG_CONTACT_BASE)
        info["fiber_Ve_on_mesh_frac"] = float(np.mean(on_mesh))
    if epi_pts is not None and Ve_epi_cols:
        cid_order = np.array([t - TAG_CONTACT_BASE for t in contact_tags])
        f32 = lambda a: np.asarray(a, np.float32)
        save = dict(epi_pts=f32(epi_pts), contact_ids=cid_order,
                    units="V_per_A", inject_A=inject_A,
                    Ve_epi=f32(np.column_stack(Ve_epi_cols)),       # (n_epi, 12)
                    E_epi=f32(np.stack(E_epi_cols, axis=1)))        # (n_epi, 12, 3)
        if peri_pts is not None and Ve_peri_cols:
            save.update(peri_pts=f32(peri_pts),
                        Ve_peri=f32(np.column_stack(Ve_peri_cols)),
                        E_peri=f32(np.stack(E_peri_cols, axis=1)))
        if sal_pts is not None and Ve_sal_cols:
            save.update(sal_pts=f32(sal_pts),
                        Ve_sal=f32(np.column_stack(Ve_sal_cols)),
                        E_sal=f32(np.stack(E_sal_cols, axis=1)))
        np.savez(d / "surf_Ve.npz", **save)
        info["surf_Ve_saved"] = True
    if xsec_all is not None and Ve_xsec_cols:
        np.savez(d / "xsec_Ve.npz",
                 xy=np.asarray(xsec_xy, np.float32),
                 z_planes=np.asarray(zrows, float), nx=NGX, R=float(Rxs),
                 Ve_xsec=np.asarray(np.stack(Ve_xsec_cols, 1), np.float32),
                 E_xsec=np.asarray(np.stack(E_xsec_cols, 1), np.float32),
                 contact_ids=np.array([t - TAG_CONTACT_BASE for t in contact_tags]))
        info["xsec_Ve_saved"] = True
    (d / "ci_solve_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _contact_meta(ec, contact_tags):
    """Per-contact id/role/z/phi in the SAME column order as Ve_mat."""
    by_id = {int(p["id"]): p for p in ec.get("patches", [])}
    out = []
    for t in contact_tags:
        cid = t - TAG_CONTACT_BASE
        p = by_id.get(cid, {})
        out.append({"id": cid, "tag": t, "role": p.get("role", ""),
                    "z_m": p.get("z"), "phi_rad": p.get("phi")})
    return out


def _write_volume(comm, d, sub_e, sub_r, ueh, urh):
    with io.XDMFFile(comm, str(d / "Ve_endo.xdmf"), "w") as xf:
        xf.write_mesh(sub_e); xf.write_function(ueh)
    with io.XDMFFile(comm, str(d / "Ve_rest.xdmf"), "w") as xf:
        xf.write_mesh(sub_r); xf.write_function(urh)
    Wr = fem.functionspace(sub_r, ("Lagrange", 1, (3,)))
    Efn = fem.Function(Wr, name="E")
    Efn.interpolate(fem.Expression(-ufl.grad(urh),
                                   Wr.element.interpolation_points))
    with io.XDMFFile(comm, str(d / "E_rest.xdmf"), "w") as xf:
        xf.write_mesh(sub_r); xf.write_function(Efn)


def _write_fiber_h5(path, flat, lens, bidx, Ve_mat, cmeta, inject_A, I_cfg, sp):
    import h5py
    off = 0
    with h5py.File(path, "w") as h:
        h.attrs["units"] = "V_per_A"
        h.attrs["inject_A"] = inject_A
        h.attrs["I_stim_configured_A"] = I_cfg
        h.attrs["frame"] = "cuff_local_m"
        h.attrs["n_fibers"] = len(lens)
        h.attrs["n_contacts"] = Ve_mat.shape[1]
        if sp:
            h.attrs["perineurium_species"] = str(sp)
        # contact metadata, column-aligned with Ve
        h.create_dataset("contact_id", data=np.array([c["id"] for c in cmeta]))
        h.create_dataset("contact_role",
                         data=np.array([str(c["role"]) for c in cmeta],
                                       dtype=h5py.string_dtype()))
        h.create_dataset("contact_z_m", data=np.array(
            [np.nan if c["z_m"] is None else c["z_m"] for c in cmeta]))
        h.create_dataset("contact_phi_rad", data=np.array(
            [np.nan if c["phi_rad"] is None else c["phi_rad"] for c in cmeta]))
        for i, L in enumerate(lens):
            L = int(L)
            xyz = flat[off:off + L]
            s = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
            grp = h.create_group(f"fiber_{i:04d}")
            grp.create_dataset("s_m", data=s)
            grp.create_dataset("xyz_m", data=xyz)
            grp.create_dataset("Ve_VperA", data=Ve_mat[off:off + L, :])
            grp.attrs["fascicle"] = int(bidx[i])
            off += L


def _color_fibers_vtp(path, ve_centre, centre_id):
    if not path.exists():
        return
    import pyvista as pv
    poly = pv.read(path)
    if poly.n_points == len(ve_centre):
        poly.point_data[f"Ve_contact{centre_id:02d}_VperA"] = ve_centre
        poly.save(path)


def main():
    sd = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SOLVE_DIR")
    if not sd:
        raise SystemExit("usage: python -m golgi.compute.solve_nerve_ci <dir>")
    info = run_ci_solve(sd)
    if MPI.COMM_WORLD.rank == 0:
        print("CI solve:", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
