# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Validation V3 — FEM solver vs analytic point/sphere source.

golgi's FEM core is FEniCSx/dolfinx solving the quasi-static Laplace equation
div(sigma grad V)=0 with a current (Neumann) source on the contact and a distant
ground (Dirichlet). Here we exercise that SAME stack on the one geometry with a
closed-form solution: a small spherical electrode (radius a, injecting current I)
concentric with a grounded outer sphere (radius R) in a homogeneous medium
(conductivity sigma). The exact solution is

    V(r) = I / (4 pi sigma) * (1/r - 1/R),   a <= r <= R.

We build the mesh with gmsh (as golgi does), solve with dolfinx + CG (as
solve_nerve.py does), sample V(r) radially, and report the error vs analytic.

Writes paper_figs/out/data/validate_fem_analytic.json (+ .npz with V(r) curves).
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT / "paper_figs"))
from io_paths import DATA   # noqa: E402

A_ELEC_M = 0.5e-3      # electrode (inner sphere) radius
R_OUT_M = 14.0e-3      # grounded outer sphere radius (>> a, so near/mid field is ~point source)
SIGMA = 0.2            # S/m homogeneous medium
I_INJ = 1.0e-3        # A injected at the electrode


def build_mesh(path: Path):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("spheres")
    occ = gmsh.model.occ
    outer = occ.addSphere(0, 0, 0, R_OUT_M)
    inner = occ.addSphere(0, 0, 0, A_ELEC_M)
    # medium = outer minus inner (keep the inner surface as the electrode wall)
    cut, _ = occ.cut([(3, outer)], [(3, inner)], removeTool=True)
    occ.synchronize()
    vol = [t for (d, t) in cut if d == 3]
    gmsh.model.addPhysicalGroup(3, vol, 1)                     # medium
    # classify boundary surfaces by bounding-box extent (both sphere surfaces have
    # their COM at the origin, so COM radius cannot distinguish them): inner sphere
    # bbox ~ a, outer sphere bbox ~ R.
    elec_s, gnd_s = [], []
    for (d, t) in gmsh.model.getBoundary([(3, v) for v in vol], oriented=False):
        bb = gmsh.model.getBoundingBox(2, t)         # xmin,ymin,zmin,xmax,ymax,zmax
        rr = max(abs(c) for c in bb)
        (elec_s if rr < 0.5 * (A_ELEC_M + R_OUT_M) else gnd_s).append(t)
    gmsh.model.addPhysicalGroup(2, elec_s, 10)                 # electrode wall
    gmsh.model.addPhysicalGroup(2, gnd_s, 20)                  # ground wall
    # graded sizing: moderately fine at the electrode, coarse at the outer boundary
    # (mid-field comparison doesn't need a sub-mm mesh everywhere)
    gmsh.option.setNumber("Mesh.MeshSizeMin", A_ELEC_M / 4.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", R_OUT_M / 6.0)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", elec_s)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", A_ELEC_M / 4.0)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", R_OUT_M / 6.0)
    gmsh.model.mesh.field.setNumber(2, "DistMin", A_ELEC_M)
    gmsh.model.mesh.field.setNumber(2, "DistMax", R_OUT_M / 4.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(path))
    n_nodes = gmsh.model.mesh.getNodes()[0].size
    gmsh.finalize()
    return int(n_nodes)


def solve(path: Path):
    import ufl
    from dolfinx import fem, io, mesh as dmesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    md = io.gmsh.read_from_msh(str(path), MPI.COMM_WORLD, gdim=3)
    domain, ft = md.mesh, md.facet_tags
    V = fem.functionspace(domain, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    # electrode surface area (tag 10) -> uniform Neumann flux J = I / A
    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    A_elec = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(one * ds(10))), op=MPI.SUM)
    J = fem.Constant(domain, PETSc.ScalarType(I_INJ / A_elec))
    a = SIGMA * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = J * v * ds(10)
    gnd = fem.locate_dofs_topological(V, 2, ft.find(20))
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), gnd, V)
    prob = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="val_",
                         petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                                        "pc_hypre_type": "boomeramg", "ksp_rtol": 1e-10})
    uh = prob.solve()
    # sample V at the mesh vertices, bin by radius
    x = domain.geometry.x
    r = np.linalg.norm(x, axis=1)
    vals = uh.x.array.real
    return r, vals, float(A_elec)


def main():
    msh = ROOT / "paper_figs/out/_intermediate/validate_spheres.msh"
    msh.parent.mkdir(parents=True, exist_ok=True)
    print("[validate_fem] meshing concentric spheres ...", flush=True)
    nn = build_mesh(msh)
    print(f"[validate_fem] mesh nodes = {nn:,}; solving ...", flush=True)
    r, vals, A_elec = solve(msh)
    # analytic
    def Vexact(rr):
        return I_INJ / (4 * np.pi * SIGMA) * (1.0 / rr - 1.0 / R_OUT_M)
    # compare in the mid-field band a*2 .. R/2 (away from the discretised electrode
    # singularity and the outer ground), binned radially
    band = (r > 2 * A_ELEC_M) & (r < 0.5 * R_OUT_M)
    redges = np.linspace((2 * A_ELEC_M), (0.5 * R_OUT_M), 24)
    rc = 0.5 * (redges[:-1] + redges[1:])
    vfem, vex = [], []
    for lo, hi in zip(redges[:-1], redges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum():
            vfem.append(float(np.mean(vals[m]))); vex.append(float(Vexact(0.5 * (lo + hi))))
        else:
            vfem.append(np.nan); vex.append(float(Vexact(0.5 * (lo + hi))))
    vfem = np.array(vfem); vex = np.array(vex)
    ok = np.isfinite(vfem)
    rel_err = np.abs(vfem[ok] - vex[ok]) / np.abs(vex[ok])
    res = dict(n_nodes=nn, A_elec_m2=A_elec, A_elec_analytic=4 * np.pi * A_ELEC_M ** 2,
               sigma=SIGMA, I_inj=I_INJ, a_m=A_ELEC_M, R_m=R_OUT_M,
               max_rel_err=float(rel_err.max()), mean_rel_err=float(rel_err.mean()),
               r_mm=(rc[ok] * 1e3).tolist(), V_fem=vfem[ok].tolist(), V_exact=vex[ok].tolist())
    (DATA / "validate_fem_analytic.json").write_text(json.dumps(res, indent=2))
    np.savez(DATA / "validate_fem_analytic.npz", r_mm=rc[ok] * 1e3,
             V_fem=vfem[ok], V_exact=vex[ok])
    print(f"[validate_fem] mid-field V(r): max rel err = {rel_err.max()*100:.2f}%, "
          f"mean = {rel_err.mean()*100:.2f}%  (A_elec FEM/analytic = "
          f"{A_elec/(4*np.pi*A_ELEC_M**2):.3f}); wrote validate_fem_analytic.json", flush=True)


if __name__ == "__main__":
    main()
