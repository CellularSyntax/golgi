# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.
"""golgi (FEniCSx) solve of the M1 monopole-in-saline-cylinder, matching the
COMSOL M1 build exactly (saline cylinder R=12 mm, L=40 mm; spherical electrode
a=0.25 mm at the centre injecting I=1 A; grounded outer walls + caps; σ=1.76 S/m).

Same dolfinx stack as solve_nerve.py / validate_fem_analytic.py (P1 Lagrange,
Neumann current on the electrode, Dirichlet ground, CG+hypre). Samples Ve along
+x at the COMSOL radial points so golgi can be overlaid on the M1 panel.

Output: paper_figs/out/data/m1_golgi_Ve.npz  (r_mm, V_fem, V_analytic)
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
HAND = ROOT / "comsol_handover"
OUT = ROOT / "paper_figs/out/data/m1_golgi_Ve.npz"

A_ELEC_M = 0.25e-3
R_CYL_M = 12.0e-3
L_CYL_M = 40.0e-3
SIGMA = 1.76
I_INJ = 1.0


def build_mesh(path: Path):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("m1cyl")
    occ = gmsh.model.occ
    cyl = occ.addCylinder(0, 0, -L_CYL_M / 2, 0, 0, L_CYL_M, R_CYL_M)
    elec = occ.addSphere(0, 0, 0, A_ELEC_M)
    cut, _ = occ.cut([(3, cyl)], [(3, elec)], removeTool=True)
    occ.synchronize()
    vol = [t for (d, t) in cut if d == 3]
    gmsh.model.addPhysicalGroup(3, vol, 1)
    elec_s, gnd_s = [], []
    for (d, t) in gmsh.model.getBoundary([(3, v) for v in vol], oriented=False):
        bb = gmsh.model.getBoundingBox(2, t)
        ext = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
        (elec_s if ext < 4 * A_ELEC_M else gnd_s).append(t)
    gmsh.model.addPhysicalGroup(2, elec_s, 10)        # electrode wall
    gmsh.model.addPhysicalGroup(2, gnd_s, 20)         # ground (cylinder walls + caps)
    gmsh.option.setNumber("Mesh.MeshSizeMin", A_ELEC_M / 8.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", R_CYL_M / 12.0)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", elec_s)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", A_ELEC_M / 8.0)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", R_CYL_M / 12.0)
    gmsh.model.mesh.field.setNumber(2, "DistMin", A_ELEC_M)
    gmsh.model.mesh.field.setNumber(2, "DistMax", R_CYL_M / 2.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(path))
    nn = gmsh.model.mesh.getNodes()[0].size
    gmsh.finalize()
    return int(nn)


def solve_and_sample(path: Path, r_line_m):
    import ufl
    from dolfinx import fem, io, geometry
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    md = io.gmsh.read_from_msh(str(path), MPI.COMM_WORLD, gdim=3)
    domain, ft = md.mesh, md.facet_tags
    V = fem.functionspace(domain, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    A_elec = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(one * ds(10))), op=MPI.SUM)
    J = fem.Constant(domain, PETSc.ScalarType(I_INJ / A_elec))
    a = SIGMA * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = J * v * ds(10)
    gnd = fem.locate_dofs_topological(V, 2, ft.find(20))
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), gnd, V)
    prob = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="m1_",
                         petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                                        "pc_hypre_type": "boomeramg", "ksp_rtol": 1e-10})
    uh = prob.solve()
    # point-evaluate Ve along +x at z=0 at the requested radii
    pts = np.column_stack([r_line_m, np.zeros_like(r_line_m), np.zeros_like(r_line_m)])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    vals = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        cells = coll.links(i)
        if len(cells):
            vals[i] = uh.eval(pts[i], cells[:1])[0]
    return vals, float(A_elec), int(uh.x.array.size)


def main():
    msh = ROOT / "paper_figs/out/_intermediate/m1_cylinder.msh"
    msh.parent.mkdir(parents=True, exist_ok=True)
    com = np.genfromtxt(HAND / "results/M1/M1_data_from_comsol.txt", comments="%")
    r_mm = com[:, 0]
    r_m = r_mm * 1e-3
    sel = (r_m > A_ELEC_M * 1.02) & (r_m < 0.62 * R_CYL_M)   # above electrode wall, inside domain
    print(f"[m1] meshing cylinder R={R_CYL_M*1e3:.0f} L={L_CYL_M*1e3:.0f} mm, a={A_ELEC_M*1e3:.2f} mm ...", flush=True)
    nn = build_mesh(msh)
    print(f"[m1] mesh nodes = {nn:,}; solving ...", flush=True)
    vfem, A_elec, ndof = solve_and_sample(msh, r_m[sel])
    Vana = I_INJ / (4 * np.pi * SIGMA) * (1.0 / r_m[sel] - 1.0 / R_CYL_M)
    ok = np.isfinite(vfem)
    near = ok & (r_mm[sel] <= 2.0)
    err = 100 * np.abs(vfem[near] - Vana[near]) / Vana[near]
    np.savez(OUT, r_mm=r_mm[sel][ok], V_fem=vfem[ok], V_analytic=Vana[ok])
    print(f"[m1] A_elec FEM/analytic = {A_elec/(4*np.pi*A_ELEC_M**2):.3f}; "
          f"golgi vs analytic (r<=2mm) mean = {err.mean():.2f}%, max = {err.max():.2f}%", flush=True)
    print(f"[m1] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
