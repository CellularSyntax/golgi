# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Quasi-static EQS on a nerve cylinder with configurable surface electrodes.

Loads electrode geometry from `results/electrode_config.json` if present;
otherwise falls back to the original Ch. 6.2.4 textbook setup
(two 1 x 1 mm point patches, 7 mm apart, same azimuthal side).

Equation and BCs:
    -div(sigma * grad(Ve)) = 0            in Omega
    -sigma * grad(Ve) . n  = J_n          on each active patch (Neumann current)
                Ve         = 0            on each ground patch (Dirichlet)
    -sigma * grad(Ve) . n  = 0            elsewhere on dOmega   (insulating)

Each patch in the config is described by:
    {"id": int, "type": "axial" | "helical", "role": "active"|"ground",
     ... patch-shape parameters ... }

Patch types:
  axial:   {"z": z_center, "dz": axial_extent,
            "phi": phi_center, "dphi": angular_extent}
  helical: {"z_start": ..., "z_end": ...,
            "phi0": phi at z_start, "pitch": axial advance per 2π,
            "dphi": angular width along the helix}

Sign convention: J_n is inward current density (positive = current flowing INTO
the domain). Cathodic stim (current leaving the domain) → J_n < 0.

Step 7.1a refactor — invocation + presets + MPI:
    mpirun -n N python -u solve_nerve.py [--preset Quick|Balanced|HPC] \\
                                          [--cores N] [<PETSc args>]

CLI/config logic is separated from the mathematical solver body
(`main()` parses args, `run_solve(cfg, comm)` does the math). Unknown
CLI args (e.g. `-log_view`) are left in sys.argv where petsc4py picks
them up at import time, so you can profile the solver with
`mpirun -n 8 python -u solve_nerve.py --preset HPC -log_view`.

All file I/O and console printing is gated on `comm.rank == 0`. The
sampling helper `sample_function` reduces per-rank results across the
communicator so the .npz outputs are complete even when N > 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import (
    bb_tree,
    compute_colliding_cells,
    compute_collisions_points,
)
from mpi4py import MPI
from petsc4py import PETSc

# macOS JIT fix: FFCx/CFFI compiles new variational forms at run time with
# clang, which needs the SDK sysroot to find system headers (assert.h, …).
# On recent Command-Line-Tools the default sysroot isn't applied unless
# SDKROOT is set, so a *fresh* form compile fails with
# "'assert.h' file not found" even though cached forms load fine. Set it
# once at import (only on darwin, only if unset) so any form — incl. the
# two-field contact-impedance block — compiles regardless of how the
# solver subprocess was launched.
if sys.platform == "darwin" and not os.environ.get("SDKROOT"):
    _sdk_cands = []
    # full-path xcrun (PATH may be sanitised under mpirun, so don't rely on
    # a bare `xcrun` being found) …
    try:
        import subprocess as _sp
        _o = _sp.run(
            ["/usr/bin/xcrun", "--show-sdk-path"],
            capture_output=True, text=True, timeout=15,
        )
        if _o.returncode == 0:
            _sdk_cands.append(_o.stdout.strip())
    except Exception:                                        # noqa: BLE001
        pass
    # … then the canonical Command-Line-Tools / Xcode SDK locations.
    _sdk_cands += [
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
        "/Applications/Xcode.app/Contents/Developer/Platforms/"
        "MacOSX.platform/Developer/SDKs/MacOSX.sdk",
    ]
    for _c in _sdk_cands:
        if _c and Path(_c).is_dir():
            os.environ["SDKROOT"] = _c
            break

# Typed boundary schemas shared with the pipeline driver
# (golgi/pipeline/fem.py). Renaming a field here breaks loudly
# at deserialize on the boundary; see migration.md Step 6.2.
from golgi.jobs.schemas import (
    ElectrodeConfig as _ElectrodeConfig,
    MeshConfig as _MeshConfig,
)
from golgi.conductivity.materials import (
    MATERIAL_SIGMA as _MAT_SIGMA,
    sigma_longitudinal as _mat_long,
    sigma_transverse as _mat_trans,
)


# ===========================================================================
# Solver presets
# ===========================================================================
# Three named PETSc-options bundles. "Quick" relaxes tolerance + caps
# iteration count for fast geometry sanity checks. "Balanced" preserves
# the pre-refactor hard-coded options so existing callers behave
# identically. "HPC" tunes BoomerAMG for 20M+ element meshes.
#
# All three set `ksp_view` so the solver hierarchy prints at solve time
# (useful when profiling with `-log_view`).

PRESETS: dict[str, dict[str, Any]] = {
    "Quick": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1.0e-4,
        "ksp_max_it": 200,
        "ksp_view": None,
    },
    "Balanced": {
        # Same options as the pre-7.1 hard-coded block — preserves
        # prior behaviour for callers that don't pick explicitly.
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1.0e-10,
        "ksp_view": None,
    },
    "HPC": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1.0e-8,
        # BoomerAMG knobs tuned for 20M+ element meshes:
        #   strong_threshold 0.7   — denser coarse hierarchy on
        #                            anisotropic problems
        #   agg_nl 4               — 4 levels of aggressive
        #                            coarsening to limit memory
        #   coarsen_type HMIS      — parallel-friendly coarsener
        "pc_hypre_boomeramg_strong_threshold": "0.7",
        "pc_hypre_boomeramg_agg_nl": "4",
        "pc_hypre_boomeramg_coarsen_type": "HMIS",
        "ksp_view": None,
    },
}
DEFAULT_PRESET = "Balanced"


# ===========================================================================
# Constants
# ===========================================================================
# Defaults used when MeshConfig is missing / partial. Originally
# hard-coded; preserved here so the pre-refactor behaviour reproduces.
R_DEFAULT = 1.0e-3        # nerve radius [m]
L_DEFAULT = 30e-3         # nerve length [m]
SIGMA_DEFAULT = 1.0       # isotropic conductivity [S/m]
I_STIM_DEFAULT = 1.0e-3   # total cathodic stim current [A] (1 mA)

N_AXIS = 401              # samples along the central axis
N_SLICES = 41             # z stations in the slice volume
SLICE_N = 60              # samples per axis on each slice grid

# Default σ values (S/m) per cell tag — sourced from the canonical
# materials table (golgi.conductivity.materials). Overridable via
# MeshConfig.sigma_*. For anisotropic tissues this is the TRANSVERSE
# (radial) value; the longitudinal component lives in SIGMA_LONG_BY_TAG.
DEFAULT_SIGMA_BY_TAG = {
    1: _mat_trans("endoneurium"),   # endoneurium  (1/6 S/m transverse)
    2: _mat_trans("saline"),        # saline       (1.76 S/m)
    3: _mat_trans("silicone"),      # silicone cuff body (insulator)
    4: _mat_trans("muscle"),        # muscle       (0.086 S/m transverse)
    5: _mat_trans("epineurium"),    # epineurium   (1/6.3 S/m)
    6: _mat_trans("platinum"),      # cathode contact metal (Pt bulk)
    7: _mat_trans("encapsulation"), # scar / encapsulation (1/6.3 S/m)
    8: _mat_trans("platinum"),      # anode contact metal (Pt bulk)
}

# Longitudinal (along +z, the nerve / muscle-fibre axis) σ for the
# anisotropic tissues. Tags absent here are treated as isotropic. Built
# diag(σ_T, σ_T, σ_L) per cell in the tensor σ field. Overridable via
# MeshConfig.sigma_endo_long / sigma_muscle_long.
DEFAULT_SIGMA_LONG_BY_TAG = {
    1: _mat_long("endoneurium"),    # 1/1.75 S/m
    4: _mat_long("muscle"),         # 0.35 S/m
}

# Perineurium bulk σ for the contact-impedance sheet at the endo↔epi
# interface (Rs = peri_thk / σ_peri). Overridable via MeshConfig.
DEFAULT_SIGMA_PERINEURIUM = _MAT_SIGMA["perineurium"]  # 1/1149 S/m

HERE = Path(__file__).parent


# ===========================================================================
# Resolved per-run config
# ===========================================================================

@dataclass
class SolverConfig:
    """Resolved per-run knobs. Built by `resolve_config` from CLI
    + on-disk MeshConfig + env vars."""
    preset_name: str                         # "Quick" | "Balanced" | "HPC"
    petsc_options: dict[str, Any]            # PRESETS[preset_name]
    out_dir: Path                            # SOLVE_OUT_DIR
    shared_dir: Path                         # SOLVE_SHARED_DIR (mesh inputs)
    cores_requested: int | None              # for sanity-check vs comm.size


# ===========================================================================
# CLI
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "FENiCSx EQS solver for nerve cuff geometry. "
            "Unknown args (e.g. `-log_view`) are forwarded to PETSc."
        ),
        # Don't try to match abbreviated long options — keeps the
        # PETSc args (which often start with `-foo_bar_baz`)
        # unambiguous against argparse's own flags.
        allow_abbrev=False,
    )
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help=(
            "Solver preset to use. Overrides "
            "MeshConfig.solver_preset (mesh_config.json). "
            f"Default: whatever the MeshConfig carries, or "
            f"`{DEFAULT_PRESET}` if neither is set."
        ),
    )
    p.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "Number of MPI ranks the caller requested. "
            "Informational — the actual rank count is set by "
            "`mpirun -n N`. If given and != COMM_WORLD.size, "
            "we print a warning."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for .npz / .xdmf files. "
            "Defaults to $SOLVE_OUT_DIR, then to "
            "`{this script's dir}/results`."
        ),
    )
    return p


def resolve_config(
    args: argparse.Namespace,
    mesh_cfg: _MeshConfig,
    comm: MPI.Comm,
) -> SolverConfig:
    """Pick the active preset: CLI overrides MeshConfig.solver_preset
    overrides DEFAULT_PRESET. Resolves out_dir from CLI/env/HERE."""
    cli_preset = getattr(args, "preset", None)
    cfg_preset = getattr(mesh_cfg, "solver_preset", None)
    preset_name = cli_preset or cfg_preset or DEFAULT_PRESET
    if preset_name not in PRESETS:
        if comm.rank == 0:
            print(
                f"WARN: unknown preset {preset_name!r} — "
                f"falling back to {DEFAULT_PRESET!r}",
                flush=True,
            )
        preset_name = DEFAULT_PRESET

    out_dir = (
        args.out_dir
        or Path(os.environ.get("SOLVE_OUT_DIR", str(HERE / "results")))
    )
    # SOLVE_SHARED_DIR is the F3.2c-renamed mesh-input dir. Under
    # F3.2c the mesh + cuff-frame fibers + nerve-surface points
    # live in `<project>/designs/<eid>/` while the FEM OUTPUTS
    # land in `<project>/configs/<cid>/` (one solve per polarity
    # wiring). When the env var is unset, mesh inputs and outputs
    # share `out_dir` — that's the F3.2a per-design-only path.
    shared_env = os.environ.get("SOLVE_SHARED_DIR", "")
    shared_dir = Path(shared_env) if shared_env else out_dir

    return SolverConfig(
        preset_name=preset_name,
        petsc_options=dict(PRESETS[preset_name]),
        out_dir=out_dir,
        shared_dir=shared_dir,
        cores_requested=getattr(args, "cores", None),
    )


# ===========================================================================
# Config file loaders (typed schemas at the boundary, dicts internally)
# ===========================================================================

def load_mesh_config(out_dir: Path) -> _MeshConfig:
    """Read mesh_config.json into a typed MeshConfig. Returns an
    empty MeshConfig (all defaults) when the file is missing or
    unparseable — preserves prior behaviour."""
    p = out_dir / "mesh_config.json"
    if not p.exists():
        return _MeshConfig()
    try:
        with open(p, encoding="utf-8") as f:
            return _MeshConfig.deserialize(json.load(f))
    except Exception:
        return _MeshConfig()


def load_electrode_config(
    out_dir: Path,
    R: float,
    comm: MPI.Comm,
) -> tuple[str, list[dict], float, list[dict]]:
    """Read electrode_config.json into
    (name, patches, I_stim, recording_montages).

    `patches` and `recording_montages` come back as lists of
    plain dicts (not dataclass instances) so the existing facet-
    mask code that does `p["type"] == "axial"` can stay
    unchanged. The schema check still happens at deserialize on
    the boundary. Legacy configs without `recording_montages`
    deserialise the list as empty."""
    p = out_dir / "electrode_config.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            ec = _ElectrodeConfig.deserialize(json.load(f))
        if comm.rank == 0:
            print(
                f"loaded electrode config from {p}: {ec.name}",
                flush=True,
            )
        return (
            ec.name,
            [pp.serialize() for pp in ec.patches],
            float(ec.I_stim),
            [m.serialize() for m in ec.recording_montages],
        )

    if comm.rank == 0:
        print(
            "no electrode_config.json — using default point-pair",
            flush=True,
        )
    default_patches = [
        {"id": 0, "type": "axial", "role": "active",
         "z": -3.5e-3, "dz": 1.0e-3, "phi": 0.0,
         "dphi": 1.0e-3 / R},
        {"id": 1, "type": "axial", "role": "ground",
         "z": +3.5e-3, "dz": 1.0e-3, "phi": 0.0,
         "dphi": 1.0e-3 / R},
    ]
    return (
        "point-pair (textbook)",
        default_patches,
        I_STIM_DEFAULT,
        [],
    )


# ===========================================================================
# MPI-aware sampling helper
# ===========================================================================

def sample_function(
    func: fem.Function,
    points: np.ndarray,
    *,
    comm: MPI.Comm,
    tree,
    domain,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `func` at an (N, 3) array of points; return
    (values, mask).

    Under MPI > 1, each rank evaluates only at points its
    partition owns, then values + ownership counts are reduced
    across the communicator (Allreduce-SUM) so every rank ends
    up with the complete result. Off-mesh points come back as
    NaN and `mask[i] == False`.

    This is the post-7.1 replacement for the old single-rank
    `sample()` — previously each rank wrote its local slice to
    .npz, which under MPI > 1 was silently incomplete output."""
    pts = np.asarray(points, dtype=np.float64)
    vs = func.function_space.value_size
    if pts.size == 0:
        return (
            np.empty((0, vs), dtype=np.float64),
            np.empty((0,), dtype=bool),
        )

    cand = compute_collisions_points(tree, pts)
    coll = compute_colliding_cells(domain, cand, pts)
    cells = np.full(len(pts), -1, dtype=np.int32)
    for i in range(len(pts)):
        if len(coll.links(i)) > 0:
            cells[i] = coll.links(i)[0]
    local_mask = cells >= 0
    local_vals = np.zeros((len(pts), vs), dtype=np.float64)
    if local_mask.any():
        local_vals[local_mask] = func.eval(
            pts[local_mask], cells[local_mask],
        )

    # Allreduce-SUM: dolfinx's partitioning means each point is
    # owned by ≤ 1 rank, so SUM == "value from owning rank, or 0
    # if unowned". `global_owned` counts contributors per point
    # (0 = off-mesh, 1 = single owner, >1 = on a partition
    # interface — averaged below).
    global_vals = np.empty_like(local_vals)
    comm.Allreduce(local_vals, global_vals, op=MPI.SUM)
    global_owned = np.zeros(len(pts), dtype=np.int32)
    comm.Allreduce(
        local_mask.astype(np.int32),
        global_owned,
        op=MPI.SUM,
    )

    # Average shared-ownership points (rare; partition-interface
    # facets that more than one rank's bb_tree claimed).
    shared = global_owned > 1
    if shared.any():
        global_vals[shared] = (
            global_vals[shared] / global_owned[shared, None]
        )

    out = global_vals
    out[global_owned == 0] = np.nan
    return out, global_owned > 0


# ===========================================================================
# I1 Phase A — DC impedance via Dirichlet dual-solve
# ===========================================================================

def _outer_muscle_facets(
    domain, cell_tags, fdim, tdim,
) -> "np.ndarray":
    """Find the exterior facets adjacent to a muscle (tag-4)
    cell — i.e., the outer surface of the muscle bbox. Used as
    the V=0 ground for the impedance dual-solve.

    Returns an int32 array of facet indices. Empty when no
    muscle tag exists in the mesh (e.g., legacy single-domain
    runs); the caller falls back to the original ground BC.
    """
    if cell_tags is None or len(cell_tags.values) == 0:
        return np.array([], dtype=np.int32)
    _ext = mesh.exterior_facet_indices(domain.topology)
    domain.topology.create_connectivity(fdim, tdim)
    _f2c = domain.topology.connectivity(fdim, tdim)
    _cv = cell_tags.values
    out: list[int] = []
    for _f in _ext:
        _cells = _f2c.links(int(_f))
        if len(_cells) == 1 and int(_cv[int(_cells[0])]) == 4:
            out.append(int(_f))
    # A properly-formed muscle far-field box IS the domain's entire outer boundary, so the
    # muscle-adjacent exterior facets (`out`) should be ~all of `_ext`. On very small nerves
    # (e.g. the ~0.34 mm rabbit vagus) the muscle region cannot be sealed as a distinct PLC
    # region: it either merges into the surrounding saline (out == []) or survives only as a
    # degenerate handful of tets at a box corner (out == a few facets that touch the boundary).
    # BOTH cases give no usable far-field ground from tag-4 — and a 3-facet corner patch is a
    # silently-wrong localized ground, worse than an honest fallback. Since the saline has
    # ballooned out to where the muscle box would be, the domain's OUTER boundary still
    # coincides with the intended ground surface (the nerve is fully embedded axially), so fall
    # back to the full exterior facet set whenever tag-4 is absent OR covers < half the
    # exterior. Geometrically the same surface; only the σ in the cuff→box annulus differs
    # (saline vs muscle), which is second-order for the field near the nerve. Gate off with
    # GOLGI_GROUND_OUTER_FALLBACK=0 to restore the strict tag-4-only behaviour.
    if len(out) >= 0.5 * max(len(_ext), 1):
        return np.array(out, dtype=np.int32)
    if os.environ.get("GOLGI_GROUND_OUTER_FALLBACK", "1") != "0" and len(_ext):
        try:
            if domain.comm.rank == 0:
                print(
                    f"  ground: muscle far-field absent/degenerate on small nerve "
                    f"({len(out)} tag-4 facets, {len(_ext)} exterior) → using full "
                    f"exterior boundary as ground",
                    flush=True,
                )
        except Exception:                                        # noqa: BLE001
            pass
        return np.asarray(_ext, dtype=np.int32)
    return np.array(out, dtype=np.int32)


def _emit_impedance_dc(
    *,
    domain, cell_tags, V, sigma_fn, patches,
    indices_list, patch_areas, ft, ds, dS,
    fdim, tdim, comm, OUT, config_name, petsc_options,
) -> None:
    """Compute per-contact DC impedance via N Dirichlet dual-
    solves, plus per-pair impedance from the configured
    anode/cathode pairs in `patches`. Writes impedance.json.

    Algorithm:
        For each contact i:
          1. Build a BC stack: V=1 on patches[i]'s facets,
             V=0 on outer muscle bbox facets.
          2. Solve the Laplace system (same `a` form as the
             main solve; zero RHS for Dirichlet-only problem).
          3. Integrate I_i = ∮_patch_i σ·∇V·n dA via UFL +
             assemble_scalar.
          4. Z_i = V_drive / I_i = 1 / I_i (V_drive = 1).

        Per-pair (anode_id, cathode_id) when both are present:
          1. Build BC: V=+0.5 on anode facets, V=-0.5 on
             cathode facets. No ground BC needed (the system
             is well-posed because both ends are fixed).
          2. Solve, integrate I through the anode facets
             (== −I through cathode by KCL).
          3. Z_pair = V_drive / I = 1 / I (V_drive = 1).

    Single-frequency = DC; the Cole-Cole frequency sweep
    extension is I1 Phase B.
    """
    import json as _json
    patch_tag_offset = 1
    n_patches = len(patches)
    if n_patches == 0:
        return

    # Identify outer muscle facets for the per-contact ground.
    ground_facets = _outer_muscle_facets(
        domain, cell_tags, fdim, tdim,
    )
    if len(ground_facets) == 0:
        if comm.rank == 0:
            print(
                "  impedance: no outer-muscle facets found "
                "(non-multi-domain mesh?); skipping per-contact "
                "solves",
                flush=True,
            )
        return
    ground_dofs = fem.locate_dofs_topological(
        V, fdim, ground_facets,
    )

    # I1 Phase A.2 fix — integrate the current AT THE GROUND
    # boundary, not at the contact. The contact facets sit on
    # the saline ↔ silicone interior interface (see the candidate
    # facet construction in solve_nerve.py ~L856-866); on those
    # internal facets dolfinx's "+"/"-" orientation is an
    # arbitrary topological choice, not a physical "saline vs
    # silicone" choice. The previous implementation took only the
    # "+" side, so per facet it could randomly land on silicone
    # (σ ≈ 1e-15 S/m → flux ≈ 0) or saline (σ ≈ 1.4 S/m → real
    # flux). On real bipolar cuffs this gave Z values ~1000× too
    # high. By current conservation the current leaving the
    # contact = the current arriving at ground, and ground facets
    # are EXTERIOR — single ds integral, no "+/-" ambiguity, no
    # contrast-dependent gotchas. Works identically for interior
    # cuff contacts (saline-silicone seam) and exterior contacts
    # (TIME-style protrusions) without code branching.
    GROUND_TAG = 9999
    _g_indices = np.asarray(ground_facets, dtype=np.int32)
    _g_indices = np.sort(_g_indices)
    _g_values = np.full(_g_indices.shape, GROUND_TAG, dtype=np.int32)
    ft_ground = mesh.meshtags(
        domain, fdim, _g_indices, _g_values,
    )
    ds_ground = ufl.Measure(
        "ds", domain=domain, subdomain_data=ft_ground,
    )

    # Variational forms — `a` matches the main solve so the
    # PETSc preconditioner can reuse its tuning.
    u_imp = ufl.TrialFunction(V)
    v_imp = ufl.TestFunction(V)
    a_imp = (
        ufl.inner(ufl.dot(sigma_fn, ufl.grad(u_imp)), ufl.grad(v_imp))
        * ufl.dx
    )
    # Zero RHS: Dirichlet BCs do all the work.
    zero_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    Lf_imp = zero_const * v_imp * ufl.dx
    n_normal = ufl.FacetNormal(domain)

    per_contact: list[dict] = []
    if comm.rank == 0:
        print(
            f"\n[impedance] computing per-contact Z via "
            f"{n_patches} Dirichlet dual-solves "
            f"(ground = outer muscle bbox; {len(ground_facets):,} "
            f"facets; current integrated at the ground boundary)",
            flush=True,
        )

    # Per-contact basis solves.
    for i, p in enumerate(patches):
        if len(indices_list[i]) == 0:
            per_contact.append({
                "id": int(p.get("id", i)),
                "role": str(p.get("role", "")),
                "Z_ohm": float("nan"),
                "V_drive_V": 1.0,
                "I_inj_A": 0.0,
                "area_m2": float(patch_areas[i]),
                "note": "no facets",
            })
            continue
        # BC: V=1 on patch i's facets.
        patch_dofs = fem.locate_dofs_topological(
            V, fdim, indices_list[i].astype(np.int32),
        )
        bc_active = fem.dirichletbc(
            PETSc.ScalarType(1.0), patch_dofs, V,
        )
        bc_ground = fem.dirichletbc(
            PETSc.ScalarType(0.0), ground_dofs, V,
        )
        problem_imp = LinearProblem(
            a_imp, Lf_imp, bcs=[bc_active, bc_ground],
            petsc_options_prefix=f"imp_{i}_",
            petsc_options=petsc_options,
        )
        try:
            Ve_imp = problem_imp.solve()
        except Exception as ex:                          # noqa: BLE001
            per_contact.append({
                "id": int(p.get("id", i)),
                "role": str(p.get("role", "")),
                "Z_ohm": float("nan"),
                "V_drive_V": 1.0, "I_inj_A": 0.0,
                "area_m2": float(patch_areas[i]),
                "note": f"solve failed: {ex}",
            })
            continue
        # I1 Phase A.2 — integrate at the ground (single ds
        # integral over the outer-muscle bbox facets) instead of
        # at the contact (mixed interior dS + "+/-" ambiguity).
        # By current conservation: I_leaving_contact = I_arriving_ground.
        # Sign convention: J = -σ∇V; outward flux at ground
        # ∮σ∇V·n is NEGATIVE (current flows INTO the V=0 ground
        # from the V=1 contact). Z = V_drive / |I| = 1 V / |I|.
        flux_ground = ufl.dot(
            ufl.dot(sigma_fn, ufl.grad(Ve_imp)), n_normal,
        ) * ds_ground(GROUND_TAG)
        try:
            I_ground = comm.allreduce(
                fem.assemble_scalar(fem.form(flux_ground)),
                op=MPI.SUM,
            )
        except Exception:                                # noqa: BLE001
            I_ground = 0.0
        I_total = float(abs(I_ground))
        Z_i = float("inf") if I_total <= 0 else (1.0 / I_total)
        per_contact.append({
            "id": int(p.get("id", i)),
            "role": str(p.get("role", "")),
            "Z_ohm": Z_i,
            "V_drive_V": 1.0,
            "I_inj_A": I_total,
            "area_m2": float(patch_areas[i]),
        })
        if comm.rank == 0:
            print(
                f"  contact id={p.get('id', i):>2}: "
                f"I={I_total * 1e6:.3f} µA  →  "
                f"Z={Z_i:.1f} Ω",
                flush=True,
            )

    # Per-pair impedance from the configured polarities. We
    # pair every "anode" with every "cathode" and run a single
    # Dirichlet solve per pair (V=+0.5 anode, V=-0.5 cathode,
    # no extra ground — system is well-posed). The pair's
    # impedance is V_pair / I_pair = 1 / |I_anode|.
    anode_ids = [
        i for i, p in enumerate(patches)
        if str(p.get("role", "")).lower() in ("anode", "active")
    ]
    cathode_ids = [
        i for i, p in enumerate(patches)
        if str(p.get("role", "")).lower() == "cathode"
    ]
    per_pair: list[dict] = []
    for a in anode_ids:
        for c in cathode_ids:
            if (len(indices_list[a]) == 0
                    or len(indices_list[c]) == 0):
                continue
            anode_dofs = fem.locate_dofs_topological(
                V, fdim, indices_list[a].astype(np.int32),
            )
            cathode_dofs = fem.locate_dofs_topological(
                V, fdim, indices_list[c].astype(np.int32),
            )
            bc_a = fem.dirichletbc(
                PETSc.ScalarType(+0.5), anode_dofs, V,
            )
            bc_c = fem.dirichletbc(
                PETSc.ScalarType(-0.5), cathode_dofs, V,
            )
            problem_p = LinearProblem(
                a_imp, Lf_imp, bcs=[bc_a, bc_c],
                petsc_options_prefix=f"imp_pair_{a}_{c}_",
                petsc_options=petsc_options,
            )
            try:
                Ve_p = problem_p.solve()
            except Exception as ex:                      # noqa: BLE001
                per_pair.append({
                    "anode": int(patches[a].get("id", a)),
                    "cathode": int(patches[c].get("id", c)),
                    "Z_pair_ohm": float("nan"),
                    "note": f"solve failed: {ex}",
                })
                continue
            # I1 Phase A.2 — per-pair has NO Dirichlet ground
            # (just ±0.5 V on anode/cathode), so the
            # "integrate-at-ground" trick from per-contact
            # doesn't apply (the muscle bbox carries only
            # leakage, not the pair current). Use Option B
            # instead: sum BOTH sides of the interior facet.
            # n("-") = -n("+"), so summing σ·∇V·n on each side
            # gives the JUMP of σ∇V·n across the facet — which
            # equals the current injected by the Dirichlet BC
            # at that facet. Because σ_silicone ≈ 1e-15 S/m,
            # the silicone side contributes ~0 and the saline
            # side carries the real flux, regardless of which
            # side dolfinx labels "+" vs "−".
            tag_a = patch_tag_offset + int(patches[a]["id"])
            flux_ext = ufl.dot(
                ufl.dot(sigma_fn, ufl.grad(Ve_p)), n_normal,
            ) * ds(tag_a)
            flux_int = (
                ufl.dot(
                    ufl.dot(sigma_fn("+"), ufl.grad(Ve_p)("+")),
                    n_normal("+"),
                )
                + ufl.dot(
                    ufl.dot(sigma_fn("-"), ufl.grad(Ve_p)("-")),
                    n_normal("-"),
                )
            ) * dS(tag_a)
            try:
                I_ext = comm.allreduce(
                    fem.assemble_scalar(fem.form(flux_ext)),
                    op=MPI.SUM,
                )
            except Exception:                            # noqa: BLE001
                I_ext = 0.0
            try:
                I_int = comm.allreduce(
                    fem.assemble_scalar(fem.form(flux_int)),
                    op=MPI.SUM,
                )
            except Exception:                            # noqa: BLE001
                I_int = 0.0
            I_total = float(abs(I_ext + I_int))
            Z_pair = (
                float("inf") if I_total <= 0
                else (1.0 / I_total)
            )
            per_pair.append({
                "anode": int(patches[a].get("id", a)),
                "cathode": int(patches[c].get("id", c)),
                "Z_pair_ohm": Z_pair,
                "V_drive_V": 1.0,
                "I_pair_A": I_total,
            })
            if comm.rank == 0:
                print(
                    f"  pair anode={patches[a].get('id', a):>2} "
                    f"cathode={patches[c].get('id', c):>2}: "
                    f"I={I_total * 1e6:.3f} µA  →  "
                    f"Z={Z_pair:.1f} Ω",
                    flush=True,
                )

    if comm.rank == 0:
        out = {
            "schema": "v1",
            "frequency_hz": 0.0,
            "ground_strategy": "outer_muscle",
            "electrode_config_name": str(config_name),
            "per_contact": per_contact,
            "per_pair": per_pair,
        }
        with open(OUT / "impedance.json", "w", encoding="utf-8") as f:
            _json.dump(out, f, indent=2)
        print(
            f"saved impedance.json ({len(per_contact)} contacts, "
            f"{len(per_pair)} pairs)",
            flush=True,
        )


# ===========================================================================
# R1.2 — Per-contact reciprocity solves (cNAP lead fields)
# ===========================================================================


def _recording_fingerprint(
    *,
    msh_path: Path,
    sigma_by_tag: dict,
    patch: dict,
    solver_preset: str,
) -> str:
    """SHA-256 fingerprint used as the per-contact cache key.
    Folds in:
      * mesh file mtime + size (cheap proxy for content)
      * σ values per tag, sorted
      * this contact's geometry fields (z, dz, phi, … as
        applicable to the patch type)
      * solver preset name (Quick / Balanced / HPC). A solve
        run at Quick rtol=1e-4 should NOT be reused as a
        Balanced rtol=1e-10 result — the user is explicitly
        asking for a different convergence target. Including
        the preset in the hash invalidates the cache on
        preset change, which is the behaviour you want.

    Stim wiring (polarity / current_fraction) is deliberately
    EXCLUDED — the lead field is independent of how the user
    wires the stim, so flipping polarities does not invalidate
    the recording basis.

    Convergence quality (I_check) is also EXCLUDED from the
    hash — that's checked separately at cache-lookup time
    against a tolerance, because a fingerprint-match with a
    bad prior convergence should still re-solve."""
    import hashlib as _hashlib
    parts: dict = {}
    try:
        st = msh_path.stat()
        parts["msh_mtime"] = float(st.st_mtime)
        parts["msh_size"] = int(st.st_size)
    except Exception:                                        # noqa: BLE001
        parts["msh_mtime"] = 0.0
        parts["msh_size"] = 0
    parts["sigma"] = [
        [int(t), float(s)]
        for t, s in sorted(sigma_by_tag.items())
    ]
    # Geometry fields that actually affect facet selection.
    geom_keys = (
        "type", "z", "dz", "phi", "dphi",
        "z_start", "z_end", "phi0", "pitch", "R",
        "x", "y", "R_wire",
    )
    parts["geom"] = {
        k: patch[k]
        for k in geom_keys
        if patch.get(k) is not None
    }
    parts["geom"]["id"] = int(patch.get("id", -1))
    parts["solver_preset"] = str(solver_preset or "")
    return _hashlib.sha256(
        json.dumps(parts, sort_keys=True).encode("utf-8"),
    ).hexdigest()


# Per-contact current-conservation tolerance for cache reuse.
# If a cached solve's stored I_inj_check_A was > this many percent
# off the unit injection current, treat the cache as STALE and
# re-solve. 1 % is a forgiving threshold — well-converged solves
# typically hit < 0.01 %, while the user's pre-fix run showed 4 %
# which is exactly what this guard catches.
_RECORDING_CACHE_I_CHECK_TOL_PCT = 1.0


def _emit_reciprocity_solves(
    *,
    domain, cell_tags, V, sigma_fn, patches,
    indices_list, patch_areas, ds, dS,
    fdim, tdim, comm, SHARED, config_name,
    petsc_options, recording_montages, sigma_by_tag,
    sample_function_,    # closure: (func, pts) → (vals, mask)
    solver_preset: str,
    peri_rs: float | None = None,
) -> None:
    """R1.2 — per-contact reciprocity (Helmholtz) solves.

    For each contact id referenced by `recording_montages`,
    inject `I_TEST` Amperes at the contact (Neumann current BC),
    Dirichlet V=0 on the outer-muscle bbox, all other contacts
    insulating (default natural Neumann). Solve the Laplace
    system, sample V_e^R at the fiber-trajectory points
    (`<SHARED>/nerve_paths_fibers.npz`), and write
    `<SHARED>/recording/V_e_rec_<contact_id>.npz` plus a
    `manifest.json` carrying the cache fingerprint per contact.

    Per-contact cache: on a subsequent solve, contacts whose
    fingerprint (mesh mtime+size, σ-per-tag, patch geometry)
    still matches are skipped — the existing .npz survives. Stim
    polarity / current fraction are excluded from the fingerprint
    by design, so flipping the stim wiring does not retrigger
    the recording basis.

    Current-conservation sanity check prints `I_solved` vs
    `I_TEST` per contact; the FEM is well-posed by construction
    so the only deviation is solver tolerance.

    Limitations (R1.2):
      * Reads contact geometry from the patches list, so any
        contact referenced by a montage MUST be present in
        electrode_config.json. Parametric cuffs emit all
        contacts (incl. OFF) so this is fine; DUKE-designer
        cuffs filter OFF contacts before writing the config,
        so recording on DUKE cuffs is deferred.
      * Sampling is currently fiber-paths-only (the cNAP forward
        sum only needs node values). Adding axis/slice sampling
        for visualisation is a small follow-up edit.
    """
    if not recording_montages:
        return

    # ---- contact-impedance (perineurium) two-field path ----
    # Opt-in via GOLGI_RECIP_CI=1 AND a perineurium sheet resistance Rs
    # (perineurium_ci=True) AND a mesh that actually resolves the
    # endo(tag 1)↔epi(tag 5) interface. When all three hold, the
    # perineurium is modelled as an ASCENT-style thin-layer contact
    # impedance (V discontinuous across Γ, J_n=(V_endo−V_rest)/Rs) via a
    # two-field block solve, and we return before the single-field path.
    # Default-off so every other model (incl. the monofascicular rabbit,
    # which has no perineurium) keeps the plain reciprocity untouched.
    if (
        os.environ.get("GOLGI_RECIP_CI", "0") == "1"
        and peri_rs is not None
        and float(peri_rs) > 0.0
        and cell_tags is not None
    ):
        _utags = {int(t) for t in np.unique(cell_tags.values)}
        if 1 in _utags and 5 in _utags:
            if comm.rank == 0:
                print(
                    f"[reciprocity] CONTACT-IMPEDANCE mode "
                    f"(Rs={float(peri_rs):.4g} Ω·m²): two-field "
                    f"endo↔epi block solve",
                    flush=True,
                )
            _emit_reciprocity_solves_ci(
                domain=domain, cell_tags=cell_tags, patches=patches,
                indices_list=indices_list, patch_areas=patch_areas,
                ds=ds, dS=dS, fdim=fdim, tdim=tdim, comm=comm,
                SHARED=SHARED, recording_montages=recording_montages,
                peri_rs=float(peri_rs),
            )
            return
        elif comm.rank == 0:
            print(
                "[reciprocity] GOLGI_RECIP_CI set but mesh lacks "
                "endo(1)+epi(5) regions — using plain reciprocity",
                flush=True,
            )

    import json as _json
    import hashlib as _hashlib  # noqa: F401  (left for symmetry)

    I_TEST = 1.0   # unit injection → output is V_e per A (ohms)

    # Unique contact ids referenced as + or − across all montages.
    rec_ids: set[int] = set()
    for m in recording_montages:
        try:
            rec_ids.add(int(m["plus_contact"]))
            rec_ids.add(int(m["minus_contact"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not rec_ids:
        return
    rec_ids = sorted(rec_ids)

    # Resolve patches by id for cache fingerprint + facet lookup.
    id_to_idx = {
        int(p.get("id", i)): i for i, p in enumerate(patches)
    }

    rec_dir = SHARED / "recording"
    if comm.rank == 0:
        rec_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    manifest_path = rec_dir / "manifest.json"

    # Load existing manifest on rank 0, broadcast to all ranks
    # so the cache decision is identical everywhere.
    manifest: dict = {
        "schema": "v1",
        "I_test": float(I_TEST),
        "ground_strategy": "outer_muscle",
        "contacts": {},
    }
    if comm.rank == 0 and manifest_path.is_file():
        try:
            manifest = _json.loads(
                manifest_path.read_text(encoding="utf-8"),
            )
        except Exception:                                    # noqa: BLE001
            pass
    if comm.size > 1:
        manifest = comm.bcast(manifest, root=0)
    contacts_meta: dict = dict(manifest.get("contacts", {}) or {})

    # Ground BC = outer muscle bbox facets (same as impedance).
    ground_facets = _outer_muscle_facets(
        domain, cell_tags, fdim, tdim,
    )
    if len(ground_facets) == 0:
        if comm.rank == 0:
            print(
                "  reciprocity: no outer-muscle facets found "
                "(non-multi-domain mesh?); skipping",
                flush=True,
            )
        return
    ground_dofs = fem.locate_dofs_topological(
        V, fdim, ground_facets,
    )
    # Build a meshtag for the ground so we can use ds(GROUND_TAG)
    # for the current-conservation check.
    GROUND_TAG = 9998
    _g_indices = np.sort(
        np.asarray(ground_facets, dtype=np.int32),
    )
    _g_values = np.full(_g_indices.shape, GROUND_TAG, dtype=np.int32)
    ft_ground = mesh.meshtags(
        domain, fdim, _g_indices, _g_values,
    )
    ds_ground = ufl.Measure(
        "ds", domain=domain, subdomain_data=ft_ground,
    )

    # Variational forms (LHS shared across contacts).
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_rec = (
        ufl.inner(ufl.dot(sigma_fn, ufl.grad(u)), ufl.grad(v)) * ufl.dx
    )
    n_normal = ufl.FacetNormal(domain)
    patch_tag_offset = 1     # matches the main-solve convention

    # Fiber-path sampling input. The cNAP forward model wants the
    # lead field at each fiber node — that's what we save. If the
    # design has no fibers, skip silently; the user can't compute
    # a cNAP without them anyway.
    _paths_npz = SHARED / "nerve_paths_fibers.npz"
    paths_flat = None
    path_lengths = None
    if _paths_npz.is_file():
        try:
            _pd = np.load(_paths_npz, allow_pickle=True)
            paths_flat = np.asarray(
                _pd["paths_flat"], dtype=np.float64,
            )
            path_lengths = np.asarray(
                _pd["path_lengths"], dtype=np.int64,
            )
        except Exception as ex:                              # noqa: BLE001
            if comm.rank == 0:
                print(
                    f"  reciprocity: failed to read "
                    f"nerve_paths_fibers.npz: {ex}",
                    flush=True,
                )
    if paths_flat is None or paths_flat.shape[0] == 0:
        if comm.rank == 0:
            print(
                "  reciprocity: no fiber paths in SHARED dir — "
                "skipping (cNAP requires fibers; mesh + "
                "trajectories first)",
                flush=True,
            )
        return

    if comm.rank == 0:
        print(
            f"\n[reciprocity] computing lead field for "
            f"{len(rec_ids)} recording contact(s); "
            f"output → {rec_dir}",
            flush=True,
        )

    msh_path = SHARED / "nerve.msh"
    for contact_id in rec_ids:
        idx = id_to_idx.get(int(contact_id))
        if idx is None:
            if comm.rank == 0:
                print(
                    f"  ⚠ contact id={contact_id} not in "
                    f"electrode_config patches — skipping "
                    f"(DUKE-designer cuffs filter OFF contacts "
                    f"before writing the config; R1.2 supports "
                    f"parametric cuffs only)",
                    flush=True,
                )
            continue
        p = patches[idx]
        if len(indices_list[idx]) == 0:
            if comm.rank == 0:
                print(
                    f"  ⚠ contact id={contact_id}: zero matching "
                    f"facets — skipping",
                    flush=True,
                )
            continue
        A_i = float(patch_areas[idx])
        if A_i <= 0.0:
            if comm.rank == 0:
                print(
                    f"  ⚠ contact id={contact_id}: zero area — "
                    f"skipping",
                    flush=True,
                )
            continue

        # Cache fingerprint.
        fp = _recording_fingerprint(
            msh_path=msh_path,
            sigma_by_tag=sigma_by_tag,
            patch=p,
            solver_preset=solver_preset,
        )
        prev_entry = (
            contacts_meta.get(str(contact_id), {}) or {}
        )
        prev_fp = prev_entry.get("fingerprint")
        # R1.4 fix-up #7: also gate the cache on the prior run's
        # current-conservation quality. A fingerprint match with
        # a bad I_check (e.g. the 4 % off solve from the user's
        # first run) should NOT be reused — that lead field is
        # numerically unreliable. Re-solve in that case.
        try:
            prev_I_check = float(
                prev_entry.get("I_inj_check_A", 0.0),
            )
        except (TypeError, ValueError):
            prev_I_check = 0.0
        prev_err_pct = (
            abs(prev_I_check - float(I_TEST))
            / float(I_TEST) * 100.0
            if float(I_TEST) > 0 else 100.0
        )
        npz_path = rec_dir / f"V_e_rec_{int(contact_id)}.npz"
        cache_ok = (
            prev_fp == fp
            and npz_path.is_file()
            and prev_err_pct <= _RECORDING_CACHE_I_CHECK_TOL_PCT
        )
        if cache_ok:
            if comm.rank == 0:
                print(
                    f"  contact id={contact_id}: cache hit "
                    f"({fp[:12]}…, prior I_check "
                    f"Δ={prev_err_pct:.3f}%) — skipping solve",
                    flush=True,
                )
            continue
        if prev_fp == fp and npz_path.is_file() and comm.rank == 0:
            # Same fingerprint but bad convergence — be loud
            # about WHY we're re-solving so the user sees it.
            print(
                f"  contact id={contact_id}: fingerprint match "
                f"but prior I_check Δ={prev_err_pct:.3f}% "
                f"exceeds {_RECORDING_CACHE_I_CHECK_TOL_PCT}% "
                f"tolerance — re-solving",
                flush=True,
            )

        # Neumann RHS on contact's facets (exterior + interior).
        tag = patch_tag_offset + int(p["id"])
        J_n = -float(I_TEST) / A_i
        Lf_rec = (
            J_n * v * ds(tag)
            + J_n * v("+") * dS(tag)
        )
        bc_ground = fem.dirichletbc(
            PETSc.ScalarType(0.0), ground_dofs, V,
        )
        problem_rec = LinearProblem(
            a_rec, Lf_rec, bcs=[bc_ground],
            petsc_options_prefix=f"rec_{int(contact_id)}_",
            petsc_options=petsc_options,
        )
        try:
            Ve_rec = problem_rec.solve()
        except Exception as ex:                              # noqa: BLE001
            if comm.rank == 0:
                print(
                    f"  ⚠ contact id={contact_id}: solve failed: "
                    f"{ex}",
                    flush=True,
                )
            continue
        Ve_rec.name = f"Ve_rec_{int(contact_id)}"

        # Current-conservation sanity: flux through the ground
        # facets should equal -I_TEST (current flows INTO ground
        # from the injecting contact). Same integrate-at-ground
        # trick as impedance — no "+/−" ambiguity on the exterior
        # facets, no σ-contrast issues.
        flux_ground = ufl.dot(
            ufl.dot(sigma_fn, ufl.grad(Ve_rec)), n_normal,
        ) * ds_ground(GROUND_TAG)
        try:
            I_at_ground = comm.allreduce(
                fem.assemble_scalar(fem.form(flux_ground)),
                op=MPI.SUM,
            )
        except Exception:                                    # noqa: BLE001
            I_at_ground = 0.0
        I_check = float(abs(I_at_ground))

        # Sample at fiber nodes.
        ve_at_paths, _mask = sample_function_(Ve_rec, paths_flat)

        if comm.rank == 0:
            np.savez(
                npz_path,
                contact_id=np.int64(int(contact_id)),
                I_test=np.float64(I_TEST),
                paths_flat=paths_flat,
                path_lengths=path_lengths,
                Ve_flat=ve_at_paths[:, 0].astype(np.float64),
            )
            err_pct = (
                abs(I_check - I_TEST) / I_TEST * 100.0
                if I_TEST > 0 else 0.0
            )
            print(
                f"  contact id={contact_id}: solved + sampled "
                f"({fp[:12]}…)  "
                f"I_check={I_check:.6f} A vs I_test=1.0 A "
                f"(Δ={err_pct:.3f}%)",
                flush=True,
            )
        contacts_meta[str(int(contact_id))] = {
            "fingerprint": fp,
            "patch_role": str(p.get("role", "")),
            "area_m2": float(A_i),
            "n_facets": int(len(indices_list[idx])),
            "I_inj_check_A": float(I_check),
        }

    # Persist manifest (rank 0 only).
    if comm.rank == 0:
        manifest["contacts"] = contacts_meta
        manifest["electrode_config_name"] = str(config_name)
        manifest["I_test"] = float(I_TEST)
        manifest["ground_strategy"] = "outer_muscle"
        manifest["schema"] = "v1"
        with open(manifest_path, "w", encoding="utf-8") as f:
            _json.dump(manifest, f, indent=2)
        print(
            f"saved recording/manifest.json "
            f"({len(contacts_meta)} contact(s) cached)",
            flush=True,
        )


def _emit_reciprocity_solves_ci(
    *,
    domain, cell_tags, patches, indices_list, patch_areas,
    ds, dS, fdim, tdim, comm, SHARED, recording_montages, peri_rs,
) -> None:
    """Two-field (endoneurium + rest) CONTACT-IMPEDANCE reciprocity.

    Drop-in replacement for `_emit_reciprocity_solves` when the
    perineurium is an ASCENT-style thin-layer contact-impedance sheet at
    the endo(tag 1)↔epi(tag 5) interface Γ: V jumps across Γ with normal
    current density J_n = (V_endo − V_rest)/Rs. The endoneurium carries
    its own CG1 field on a submesh, the rest of the domain another,
    coupled by the symmetric Robin term ∫_Γ (1/Rs)(u_e−u_r)(v_e−v_r) dΓ.

    Contacts are the SAME FEM-time facet patches the single-field solver
    uses (no contact volumes required), applied as a Neumann current on
    the rest field; ground = outer-muscle facets of the rest submesh.
    The block matrix is factorised once (MUMPS) and re-used across all
    recording contacts. Writes the IDENTICAL per-contact output as the
    single-field path — ``recording/V_e_rec_<id>.npz`` with ``Ve_flat``
    in the same (reciprocity) sign convention — so downstream assembly is
    unchanged. Reuses the validated block machinery of solve_nerve_ci.
    """
    import json as _json
    from dolfinx.fem.petsc import (
        assemble_matrix, assemble_vector, create_vector, set_bc, assign,
    )
    import dolfinx.la.petsc as _lap
    from golgi.compute.solve_nerve_ci import (
        _interface_entities, _sigma_tensor, _tag_by_cell,
        _locate_cells, _eval_at, _fill_nan_per_fiber,
    )

    T_ENDO, T_MUSCLE, T_EPI = 1, 4, 5
    I_TEST = 1.0
    g_val = 1.0 / float(peri_rs)

    rec_ids: set[int] = set()
    for m in recording_montages:
        try:
            rec_ids.add(int(m["plus_contact"]))
            rec_ids.add(int(m["minus_contact"]))
        except (KeyError, TypeError, ValueError):
            continue
    rec_ids = sorted(rec_ids)
    if not rec_ids:
        return
    id_to_idx = {int(p.get("id", i)): i for i, p in enumerate(patches)}

    _paths_npz = SHARED / "nerve_paths_fibers.npz"
    if not _paths_npz.is_file():
        if comm.rank == 0:
            print("  reciprocity(CI): no fiber paths — skipping", flush=True)
        return
    _pd = np.load(_paths_npz, allow_pickle=True)
    paths_flat = np.asarray(_pd["paths_flat"], np.float64)
    path_lengths = np.asarray(_pd["path_lengths"], np.int64)
    if paths_flat.shape[0] == 0:
        return

    # ---- endo (in) + rest (everything else) submeshes ----
    n_local = domain.topology.index_map(tdim).size_local
    all_cells = np.arange(n_local, dtype=np.int32)
    endo_cells = cell_tags.find(T_ENDO).astype(np.int32)
    endo_cells = endo_cells[endo_cells < n_local]
    rest_cells = np.setdiff1d(all_cells, endo_cells).astype(np.int32)
    sub_e, e_map, _, _ = mesh.create_submesh(domain, tdim, endo_cells)
    sub_r, r_map, _, _ = mesh.create_submesh(domain, tdim, rest_cells)
    tbc = _tag_by_cell(domain, cell_tags)

    Ee = fem.functionspace(sub_e, ("Lagrange", 1))
    Er = fem.functionspace(sub_r, ("Lagrange", 1))
    ue, ve = ufl.TrialFunction(Ee), ufl.TestFunction(Ee)
    ur, vr = ufl.TrialFunction(Er), ufl.TestFunction(Er)
    sig_e = _sigma_tensor(sub_e, endo_cells, tbc)
    sig_r = _sigma_tensor(sub_r, rest_cells, tbc)
    rest_bulk_tags = [int(t) for t in np.unique(cell_tags.values)
                      if int(t) != T_ENDO]

    # ---- block bilinear form with the Robin coupling at Γ ----
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    gam = _interface_entities(domain, cell_tags, T_ENDO, T_EPI, first=T_ENDO)
    dS_g = ufl.Measure("dS", domain=domain, subdomain_data=[(1, gam)])
    g = fem.Constant(domain, PETSc.ScalarType(g_val))
    a00 = ufl.inner(ufl.dot(sig_e, ufl.grad(ue)), ufl.grad(ve)) * dx(T_ENDO) \
        + g * ue('+') * ve('+') * dS_g(1)
    a11 = sum(ufl.inner(ufl.dot(sig_r, ufl.grad(ur)), ufl.grad(vr)) * dx(t)
              for t in rest_bulk_tags) + g * ur('-') * vr('-') * dS_g(1)
    a01 = -g * ur('-') * ve('+') * dS_g(1)
    a10 = -g * ue('+') * vr('-') * dS_g(1)
    a_form = fem.form([[a00, a01], [a10, a11]], entity_maps=[e_map, r_map])

    # ---- ground: outer-muscle facets of the rest submesh (matches the
    # single-field _outer_muscle_facets convention, ported to the submesh) ----
    sub_r.topology.create_connectivity(fdim, tdim)
    rest_tbc = tbc[rest_cells]
    ext_r = mesh.exterior_facet_indices(sub_r.topology)
    f2c_r = sub_r.topology.connectivity(fdim, tdim)
    gnd = np.array([f for f in ext_r
                    if len(f2c_r.links(f)) and rest_tbc[f2c_r.links(f)[0]] == T_MUSCLE],
                   dtype=np.int32)
    if len(gnd) == 0:                       # no muscle far-field → all exterior
        gnd = ext_r.astype(np.int32)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0),
                         fem.locate_dofs_topological(Er, fdim, gnd), Er)

    # ---- assemble A once; reuse the MUMPS factorisation per contact ----
    A = assemble_matrix(a_form, bcs=[bc], kind="mpi")
    A.assemble()
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    xvec = create_vector([Ee, Er], kind="mpi")
    ueh, urh = fem.Function(Ee), fem.Function(Er)
    L0 = fem.Constant(domain, PETSc.ScalarType(0.0)) * ve * dx(T_ENDO)

    rec_dir = SHARED / "recording"
    if comm.rank == 0:
        rec_dir.mkdir(parents=True, exist_ok=True)
        for _stale in rec_dir.glob("V_e_rec_*.npz"):
            _stale.unlink()                 # drop any prior (CI-off) lead fields
    comm.Barrier()
    if comm.rank == 0:
        print(f"[reciprocity] CI two-field solve: {len(rec_ids)} contact(s), "
              f"endo {len(endo_cells):,} cells / rest {len(rest_cells):,} cells, "
              f"ground {len(gnd):,} facets "
              f"[SDKROOT={os.environ.get('SDKROOT', '<unset>')}]", flush=True)

    # Pre-locate the (fixed) fiber points on BOTH submeshes once. A fiber
    # node inside a fascicle is sampled from the endoneurium field (behind
    # the perineurium); a node that falls in the epineurium/rest (boundary
    # nodes, fiber caps) from the rest field — so every node gets a real,
    # physical value and we avoid filling across fibers (which would inject
    # spurious second-differences into the activating function).
    ce_cells, ce_ok = _locate_cells(paths_flat, sub_e)
    cr_cells, cr_ok = _locate_cells(paths_flat, sub_r)

    def _combine(v):
        """MPI-reduce a per-rank sampled column (NaN off the local partition)."""
        fin = np.isfinite(v)
        vs = comm.allreduce(np.where(fin, v, 0.0), op=MPI.SUM)
        cc = comm.allreduce(fin.astype(np.float64), op=MPI.SUM)
        return np.where(cc > 0, vs / np.maximum(cc, 1.0), np.nan)

    n_done = 0
    for contact_id in rec_ids:
        idx = id_to_idx.get(int(contact_id))
        if idx is None or len(indices_list[idx]) == 0:
            continue
        p = patches[idx]
        A_i = float(patch_areas[idx])
        if A_i <= 0.0:
            continue
        tag = 1 + int(p["id"])              # patch_tag_offset = 1 (main convention)
        # Inject −I_TEST (same sign convention as the single-field
        # reciprocity, so the downstream −flip yields +current→+V).
        J_n = fem.Constant(domain, PETSc.ScalarType(-I_TEST / A_i))
        L1 = J_n * vr('+') * dS(tag) + J_n * vr * ds(tag)
        L_form = fem.form([L0, L1], entity_maps=[e_map, r_map])
        b = assemble_vector(L_form, kind="mpi")
        _lap._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(b, [[], [bc]])
        ksp.solve(b, xvec)
        _lap._ghost_update(xvec, PETSc.InsertMode.INSERT,
                           PETSc.ScatterMode.FORWARD)
        assign(xvec, [ueh, urh])
        b.destroy()

        # A fascicular fiber lives in the endoneurium, so its lead field is
        # the ENDO solution. Sampling the rest field for the ~16% of nodes
        # that fall just outside the endo submesh (boundary nodes, fiber
        # caps) would inject the perineurium voltage JUMP into the trace and
        # corrupt the activating function (d²Ve). Instead, sample endo-only
        # and interpolate each fiber's off-endo gaps ALONG THAT FIBER
        # (smooth, no jump). Only a fiber with NO endo node at all (a genuine
        # epineurial fiber) falls back to the rest field for its whole trace.
        ve_e = _combine(_eval_at(ueh, paths_flat, ce_cells, ce_ok))
        ve_r = _combine(_eval_at(urh, paths_flat, cr_cells, cr_ok))
        ve_e_fill = _fill_nan_per_fiber(ve_e, path_lengths)   # per-fiber interp of endo gaps
        ve_r_fill = _fill_nan_per_fiber(ve_r, path_lengths)
        ve_k = np.where(np.isfinite(ve_e_fill), ve_e_fill, ve_r_fill)
        if comm.rank == 0:
            bad = ~np.isfinite(ve_k)
            if bad.any() and (~bad).any():
                from scipy.spatial import cKDTree
                nn = cKDTree(paths_flat[~bad]).query(paths_flat[bad])[1]
                ve_k[bad] = ve_k[~bad][nn]
            np.savez(
                rec_dir / f"V_e_rec_{int(contact_id)}.npz",
                contact_id=np.int64(int(contact_id)),
                I_test=np.float64(I_TEST),
                paths_flat=paths_flat,
                path_lengths=path_lengths,
                Ve_flat=ve_k.astype(np.float64),
            )
            print(f"  contact id={contact_id}: CI solved + sampled "
                  f"(endo {float(np.mean(np.isfinite(ve_e))):.3f}, "
                  f"cover {float(np.mean(np.isfinite(ve_e) | np.isfinite(ve_r))):.3f}, "
                  f"|Ve| max {np.nanmax(np.abs(ve_k)):.3g})", flush=True)
        n_done += 1

    if comm.rank == 0:
        (rec_dir / "manifest.json").write_text(_json.dumps({
            "schema": "v1-ci", "I_test": float(I_TEST),
            "ground_strategy": "outer_muscle_submesh",
            "perineurium_contact_impedance": True,
            "Rs_peri_ohm_m2": float(peri_rs),
            "n_contacts": int(n_done),
        }, indent=2), encoding="utf-8")
        print(f"[reciprocity] CI two-field done: {n_done} contact(s) → "
              f"{rec_dir}", flush=True)


# ===========================================================================
# Solver body
# ===========================================================================

def run_solve(cfg: SolverConfig, comm: MPI.Comm) -> None:
    """The mathematical solver: load mesh + configs, build the
    PETSc system, solve, sample, write outputs.

    All file I/O and console printing is gated on
    `comm.rank == 0`. Sampling uses `sample_function` which
    reduces across ranks so the .npz files are complete."""

    OUT = cfg.out_dir
    SHARED = cfg.shared_dir
    OUT.mkdir(parents=True, exist_ok=True)

    # Clear stale slice files from previous runs (legacy per-slice
    # format). Done by rank 0 only to avoid races.
    if comm.rank == 0:
        for f in OUT.glob("slice_z_*.npz"):
            f.unlink()
    comm.Barrier()

    # ---- mesh ----
    # F3.2c: each design owns its multi-domain mesh under
    # `<project>/designs/<eid>/`. The FEM outputs (this solve)
    # land in `<project>/configs/<cid>/` (or another caller-
    # chosen dir). SHARED points at the mesh-input dir;
    # OUT at the output dir. They're equal when the caller
    # doesn't split them (F3.2a per-design-only flow).
    #
    # Each `solve_nerve.py` invocation is a fresh subprocess, so
    # gmsh state cannot leak across solves under the current
    # FEMRunner. `dolfinx.io.gmsh.read_from_msh` handles
    # `gmsh.initialize()` + `gmsh.merge()` internally — no
    # wrapper needed.
    mesh_data = io.gmsh.read_from_msh(
        str(SHARED / "nerve.msh"), comm, gdim=3,
    )
    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # ---- piecewise σ from cell tags ----
    # Defaults; overridable via MeshConfig.
    R = R_DEFAULT
    L = L_DEFAULT
    SIGMA = SIGMA_DEFAULT

    mesh_cfg = load_mesh_config(OUT)
    SIGMA_BY_TAG = dict(DEFAULT_SIGMA_BY_TAG)
    MESH_MODE = mesh_cfg.mode
    if mesh_cfg.sigma_endo is not None:
        SIGMA_BY_TAG[1] = mesh_cfg.sigma_endo
    if mesh_cfg.sigma_saline is not None:
        SIGMA_BY_TAG[2] = mesh_cfg.sigma_saline
    if mesh_cfg.sigma_silicone is not None:
        SIGMA_BY_TAG[3] = mesh_cfg.sigma_silicone
    if mesh_cfg.sigma_muscle is not None:
        SIGMA_BY_TAG[4] = mesh_cfg.sigma_muscle
    if mesh_cfg.sigma_epi is not None:
        SIGMA_BY_TAG[5] = mesh_cfg.sigma_epi
    if mesh_cfg.sigma_contact is not None:
        SIGMA_BY_TAG[6] = mesh_cfg.sigma_contact
    if mesh_cfg.sigma_scar is not None:
        SIGMA_BY_TAG[7] = mesh_cfg.sigma_scar

    # Longitudinal (+z, fibre-axis) σ for the anisotropic tissues —
    # endo (tag 1) and muscle (tag 4). Tags absent here are isotropic.
    SIGMA_LONG_BY_TAG = dict(DEFAULT_SIGMA_LONG_BY_TAG)
    if mesh_cfg.sigma_endo_long is not None:
        SIGMA_LONG_BY_TAG[1] = mesh_cfg.sigma_endo_long
    if mesh_cfg.sigma_muscle_long is not None:
        SIGMA_LONG_BY_TAG[4] = mesh_cfg.sigma_muscle_long

    # Perineurium contact-impedance sheet at the endo↔epi interface.
    # Active iff requested; Rs = peri_thk / σ_peri  [Ω·m²].
    PERI_CI = bool(getattr(mesh_cfg, "perineurium_ci", False))
    SIGMA_PERI = (
        mesh_cfg.sigma_peri if mesh_cfg.sigma_peri is not None
        else DEFAULT_SIGMA_PERINEURIUM
    )
    PERI_THK_M = mesh_cfg.peri_thk_m
    PERI_RS = (
        (PERI_THK_M / SIGMA_PERI)
        if (PERI_CI and PERI_THK_M and PERI_THK_M > 0.0 and SIGMA_PERI > 0.0)
        else None
    )

    # In imported mode, override the cylinder defaults R, L with
    # values from the cuff frame. Used by axis-line +
    # slice-volume sampling, which assume a cylindrical region
    # around the +z axis at the cuff origin.
    if MESH_MODE == "imported":
        if mesh_cfg.R_cuff_inner is not None:
            R = mesh_cfg.R_cuff_inner
        if mesh_cfg.L_cuff is not None:
            L = max(30e-3, 5.0 * mesh_cfg.L_cuff)

    # Optional explicit z-span for axis sampling.
    AXIS_Z_LO = None
    AXIS_Z_HI = None
    if (mesh_cfg.axis_z_lo_m is not None
            and mesh_cfg.axis_z_hi_m is not None):
        AXIS_Z_LO = mesh_cfg.axis_z_lo_m
        AXIS_Z_HI = mesh_cfg.axis_z_hi_m
        L = AXIS_Z_HI - AXIS_Z_LO  # for slice volume too

    # Optional explicit xy half-extent for slice-volume sampling.
    SLICE_XY_HALF = mesh_cfg.slice_xy_half_m

    # F3.2 — cuff transform that maps cuff-local row vectors to
    # canonical (mesh) frame:   p_canon = p_local @ R + offset.
    # Inverse:   p_local = (p_canon - offset) @ R.T.
    # The mesh lives in the canonical frame; electrode patches'
    # (z, phi) are in cuff-local; axis-line + slice-volume sampling
    # is also defined in cuff-local. We apply the transform at
    # every boundary between the two frames.
    if (mesh_cfg.cuff_offset_m is not None
            and mesh_cfg.cuff_R_flat is not None
            and len(mesh_cfg.cuff_R_flat) == 9):
        CUFF_OFFSET = np.asarray(
            mesh_cfg.cuff_offset_m, dtype=np.float64,
        ).reshape(3)
        CUFF_R = np.asarray(
            mesh_cfg.cuff_R_flat, dtype=np.float64,
        ).reshape(3, 3)
    else:
        # Legacy single-design path: mesh IS in cuff-local frame.
        CUFF_OFFSET = np.zeros(3, dtype=np.float64)
        CUFF_R = np.eye(3, dtype=np.float64)

    def _local_from_canon(pts_canon: np.ndarray) -> np.ndarray:
        return (pts_canon - CUFF_OFFSET) @ CUFF_R

    def _canon_from_local(pts_local: np.ndarray) -> np.ndarray:
        return pts_local @ CUFF_R.T + CUFF_OFFSET

    # Build a DG0 (piecewise-constant) ANISOTROPIC σ field as a rank-2
    # tensor: per cell σ = diag(σ_T, σ_T, σ_L), where σ_L (longitudinal,
    # along +z = the nerve / muscle-fibre axis) defaults to σ_T
    # (isotropic) unless the tag is in SIGMA_LONG_BY_TAG. Storing σ as a
    # 3×3 tensor lets the weak form use the full J = σ·∇V with correct
    # endoneurium/muscle anisotropy. Single-domain legacy meshes get
    # isotropic σ = SIGMA everywhere. `sigma_fn` is a tensor Function
    # from here on; all forms use ufl.dot(sigma_fn, grad(·)).
    DG0t = fem.functionspace(domain, ("DG", 0, (3, 3)))
    sigma_fn = fem.Function(DG0t, name="sigma")
    _ncells = (
        domain.topology.index_map(tdim).size_local
        + domain.topology.index_map(tdim).num_ghosts
    )
    _sig_t = np.full(_ncells, SIGMA)   # transverse / isotropic component
    _sig_l = np.full(_ncells, SIGMA)   # longitudinal (z) component
    if cell_tags is not None and len(cell_tags.values) > 0:
        unique_tags = np.unique(cell_tags.values)
        for tag in unique_tags:
            cells = cell_tags.find(int(tag))
            s_t = SIGMA_BY_TAG.get(int(tag), SIGMA)
            _sig_t[cells] = s_t
            _sig_l[cells] = SIGMA_LONG_BY_TAG.get(int(tag), s_t)
        if comm.rank == 0:
            print("piecewise σ from cell tags:", flush=True)
            for tag in unique_tags:
                n_cells = int(np.sum(cell_tags.values == tag))
                s_t = SIGMA_BY_TAG.get(int(tag), SIGMA)
                s_l = SIGMA_LONG_BY_TAG.get(int(tag), s_t)
                _an = "" if s_l == s_t else f", σ_L={s_l:g}"
                print(
                    f"  tag={int(tag)}  σ_T={s_t:g} S/m{_an}  "
                    f"({n_cells} cells)",
                    flush=True,
                )
    else:
        if comm.rank == 0:
            print(
                f"single-domain mesh: σ = {SIGMA} S/m everywhere",
                flush=True,
            )
    # Pack the per-cell diagonal into the 3×3 DG0 tensor dofs
    # (row-major flatten: index 0=xx, 4=yy, 8=zz; off-diagonals 0).
    _tarr = sigma_fn.x.array.reshape(-1, 9)
    _tarr[:] = 0.0
    _tarr[:, 0] = _sig_t
    _tarr[:, 4] = _sig_t
    _tarr[:, 8] = _sig_l

    # ---- electrode config ----
    (
        config_name, patches, I_STIM, recording_montages,
    ) = load_electrode_config(OUT, R, comm)

    # ---- facet tagging ----
    # Build the PATCH-CANDIDATE FACET SET. In single-domain meshes
    # this is just the exterior facets. In multi-domain meshes the
    # actual cuff inner wall is the saline-silicone interface
    # (interior facets between tags 2/3) — we must include those.
    ext_facets = mesh.exterior_facet_indices(domain.topology)
    n_ext_facets = len(ext_facets)
    interior_patch_facets = np.array([], dtype=np.int32)
    if cell_tags is not None and len(cell_tags.values) > 0:
        _utags = set(int(t) for t in np.unique(cell_tags.values))
        # The cuff inner wall is the saline(2) interface with the wall
        # material on the other side: silicone(3) or — where a contact is
        # embedded — the contact metal (cathode 6 / anode 8). Treat all of
        # these as patch-candidate facets so a contact's Neumann BC lands on
        # its own saline-facing metal surface, not just bare silicone.
        _WALL = {3, 6, 8}
        if 2 in _utags and (_utags & _WALL):
            domain.topology.create_connectivity(fdim, tdim)
            _f2c = domain.topology.connectivity(fdim, tdim)
            _ct_vals = cell_tags.values
            _n_facets = (
                domain.topology.index_map(fdim).size_local
                + domain.topology.index_map(fdim).num_ghosts
            )
            _interior_list = []
            for _f in range(_n_facets):
                _cells = _f2c.links(_f)
                if len(_cells) == 2:
                    _t1 = int(_ct_vals[_cells[0]])
                    _t2 = int(_ct_vals[_cells[1]])
                    if ((_t1 == 2 and _t2 in _WALL)
                            or (_t2 == 2 and _t1 in _WALL)):
                        _interior_list.append(_f)
            interior_patch_facets = np.array(
                _interior_list, dtype=np.int32,
            )
            if comm.rank == 0:
                print(
                    f"  found {len(interior_patch_facets):,} "
                    f"interior facets at the saline-wall interface "
                    f"(silicone/contact) — the cuff inner wall, where "
                    f"contacts sit in a multi-domain mesh",
                    flush=True,
                )

    candidate_facets = np.concatenate(
        [ext_facets.astype(np.int32), interior_patch_facets]
    )
    candidate_is_interior = np.zeros(
        len(candidate_facets), dtype=bool,
    )
    candidate_is_interior[n_ext_facets:] = True

    # F3.2 — facet midpoints come back in canonical (mesh) frame.
    # Transform to cuff-local before computing (r, z, phi) so the
    # patch (z, phi) match the cuff's local axis regardless of how
    # the cuff is rotated/translated in canonical space. When the
    # mesh IS in cuff-local (legacy single-design path), this is a
    # no-op (offset = 0, R = I).
    mids_canon = mesh.compute_midpoints(domain, fdim, candidate_facets)
    mids = _local_from_canon(mids_canon)
    mx, my, mz = mids[:, 0], mids[:, 1], mids[:, 2]
    mr = np.sqrt(mx * mx + my * my)
    mphi = np.arctan2(my, mx)

    # Lazy kdtree on facet midpoints for LIFE / TIME nearest-
    # facet lookup. Built once on first intrafascicular patch
    # so the cuff-electrode-only path pays nothing for it.
    _intrafasc_kdtree: list = [None]

    def _ensure_intrafasc_kdtree():
        if _intrafasc_kdtree[0] is None:
            from scipy.spatial import cKDTree
            _intrafasc_kdtree[0] = cKDTree(mids)
        return _intrafasc_kdtree[0]

    def patch_facet_mask(p: dict) -> np.ndarray:
        R_p = float(p.get("R", R))
        _multi_domain = (
            cell_tags is not None
            and len(np.unique(cell_tags.values)) > 1
        )
        if MESH_MODE == "imported" and not _multi_domain:
            on_surface = np.ones_like(mr, dtype=bool)
        else:
            on_surface = np.abs(mr - R_p) <= 0.05 * R_p
        if p["type"] == "axial":
            dz_ok = np.abs(mz - p["z"]) <= p["dz"] / 2.0
            full_ring = p["dphi"] >= 2.0 * np.pi - 1e-9
            if full_ring:
                phi_ok = np.ones_like(dz_ok)
            else:
                d = (
                    (mphi - p["phi"] + np.pi)
                    % (2.0 * np.pi) - np.pi
                )
                phi_ok = np.abs(d) <= p["dphi"] / 2.0
            return on_surface & dz_ok & phi_ok
        if p["type"] == "helical":
            z_lo, z_hi = p["z_start"], p["z_end"]
            z_ok = (mz >= z_lo) & (mz <= z_hi)
            phi_helix = (
                p["phi0"]
                + 2.0 * np.pi * (mz - z_lo) / p["pitch"]
            )
            d = (mphi - phi_helix + np.pi) % (2.0 * np.pi) - np.pi
            phi_ok = np.abs(d) <= p["dphi"] / 2.0
            return on_surface & z_ok & phi_ok
        if p["type"] in ("life_band", "time_rect"):
            # Intrafascicular contacts (LIFE / TIME) sit inside
            # the endoneurium volume — the wire/ribbon body is
            # zero-volume in this FEM (thin-conductor convention),
            # so we apply the Neumann current to the K endo-
            # boundary facets nearest the contact's centroid.
            # K is set so the selected patch area scales with the
            # contact's nominal physical area; the FEM is robust
            # to the exact value (a few facets each works fine for
            # selectivity-style studies).
            tree = _ensure_intrafasc_kdtree()
            cx, cy, cz = (
                float(p["x"]), float(p["y"]), float(p["z"]),
            )
            # Pick K facets: 3 for LIFE bands (∼ 0.13 mm²
            # nominal site), 1 for TIME rects (∼ 0.02 mm²).
            # Cuff-style patches already use full-ring or
            # phi-sector selection, so this small-K rule only
            # fires for the intraneural kinds.
            K = 3 if p["type"] == "life_band" else 1
            _dist, _idx = tree.query(
                np.array([[cx, cy, cz]]), k=K,
            )
            sel = np.asarray(_idx).reshape(-1)
            mask = np.zeros(len(mr), dtype=bool)
            mask[sel] = True
            return mask
        raise ValueError(f"unknown patch type: {p['type']}")

    # Build per-patch facet lists.
    indices_list = []
    values_list = []
    ext_indices_per_patch = []
    int_indices_per_patch = []
    patch_facet_counts = []
    patch_areas: list[float] = []
    patch_tag_offset = 1

    for p in patches:
        mask = patch_facet_mask(p)
        facets = candidate_facets[mask]
        ext_subset = candidate_facets[mask & ~candidate_is_interior]
        int_subset = candidate_facets[mask & candidate_is_interior]
        patch_facet_counts.append(int(len(facets)))
        indices_list.append(facets)
        ext_indices_per_patch.append(ext_subset.astype(np.int32))
        int_indices_per_patch.append(int_subset.astype(np.int32))
        tag = patch_tag_offset + int(p["id"])
        values_list.append(np.full(len(facets), tag, dtype=np.int32))

    if not indices_list or all(len(a) == 0 for a in indices_list):
        try:
            _p0 = dict(patches[0]); _Rp = float(_p0.get("R", R))
            _ons = np.abs(mr - _Rp) <= 0.05 * _Rp
            print(f"  [patch-diag] {len(patches)} patches; facets r[{mr.min()*1e3:.3f},"
                  f"{mr.max()*1e3:.3f}]mm med {np.median(mr)*1e3:.3f}; z[{mz.min()*1e3:.2f},"
                  f"{mz.max()*1e3:.2f}]mm; p0 R={_Rp*1e3:.3f}mm z={float(_p0.get('z',0))*1e3:.2f} "
                  f"dz={float(_p0.get('dz',0))*1e3:.2f} type={_p0.get('type')}; "
                  f"on_surface={int(_ons.sum())}/{len(mr)}", flush=True)
        except Exception as _e:                                # noqa: BLE001
            print(f"  [patch-diag] failed: {_e}", flush=True)
        raise RuntimeError(
            "No facets matched any patch — "
            "check electrode_config.json",
        )

    indices = np.concatenate(indices_list).astype(np.int32)
    values = np.concatenate(values_list)
    order = np.argsort(indices)
    ft = mesh.meshtags(domain, fdim, indices[order], values[order])

    if comm.rank == 0:
        # XDMF write is collective inside dolfinx, so all ranks
        # must enter the with-block — but only rank 0 needs to
        # see the path-existence side effects.
        pass
    with io.XDMFFile(comm, OUT / "facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(ft, domain.geometry)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)

    # Per-patch area = ds (exterior) + dS (interior).
    for p in patches:
        tag = patch_tag_offset + int(p["id"])
        A_ext = comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * ds(tag))),
            op=MPI.SUM,
        )
        try:
            A_int = comm.allreduce(
                fem.assemble_scalar(fem.form(1.0 * dS(tag))),
                op=MPI.SUM,
            )
        except Exception:
            A_int = 0.0
        patch_areas.append(float(A_ext) + float(A_int))

    if comm.rank == 0:
        print(f"\n{len(patches)} electrode patches:", flush=True)
        for p, A, n in zip(
            patches, patch_areas, patch_facet_counts,
        ):
            print(
                f"  id={p['id']:2d} {p['role']:8s} "
                f"{p['type']:8s} area={A * 1e6:7.4f} mm² "
                f"({n} facets)",
                flush=True,
            )

    # ---- variational problem ----
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.dot(sigma_fn, ufl.grad(u)), ufl.grad(v)) * ufl.dx

    # M1: extended role vocab — "active" and "cathode" are
    # aliased (both = Neumann current source); "ground" and
    # "anode" are aliased (both = Dirichlet 0 reference).
    # "off" patches don't appear in this list (the FEM driver
    # filters them out before writing the config). Legacy
    # configs with only "active"/"ground" still solve.
    CATHODE_ROLES = ("active", "cathode")
    GROUND_ROLES = ("ground", "anode")
    active_idx = [
        i for i, p in enumerate(patches)
        if p["role"] in CATHODE_ROLES
    ]
    ground_idx = [
        i for i, p in enumerate(patches)
        if p["role"] in GROUND_ROLES
    ]
    if not active_idx:
        raise RuntimeError(
            "No active (Neumann current) patches in config",
        )
    if not ground_idx:
        raise RuntimeError(
            "No ground (Dirichlet) patches in config",
        )

    # M1: per-patch current_fraction weighting. When all
    # cathode patches set the field, each gets its specified
    # share of I_STIM. When none are set, fall back to equal
    # split. Mixed (some set, some unset) → unset patches
    # share the remainder.
    fractions_explicit: list[float | None] = [
        (
            float(patches[i].get("current_fraction"))
            if patches[i].get("current_fraction") is not None
            else None
        )
        for i in active_idx
    ]
    n_explicit = sum(
        1 for f in fractions_explicit if f is not None
    )
    n_implicit = len(fractions_explicit) - n_explicit
    explicit_sum = sum(
        f for f in fractions_explicit if f is not None
    )
    if n_implicit > 0:
        # Unset patches share whatever's left of the 1.0 total.
        remainder = max(0.0, 1.0 - explicit_sum)
        implicit_share = remainder / n_implicit
        fractions = [
            (f if f is not None else implicit_share)
            for f in fractions_explicit
        ]
    else:
        # All explicit — normalise in case the user didn't
        # quite sum to 1 (UI sum-check chip warns them but
        # we still want a clean solve).
        s = explicit_sum or 1.0
        fractions = [f / s for f in fractions_explicit]
    Lf = None
    for k, i in enumerate(active_idx):
        p = patches[i]
        A = patch_areas[i]
        if A <= 0:
            if comm.rank == 0:
                print(
                    f"  WARN: active patch id={p['id']} has "
                    f"zero area — skipped",
                    flush=True,
                )
            continue
        tag = patch_tag_offset + int(p["id"])
        I_this = I_STIM * fractions[k]
        J_n = -I_this / A
        term_ext = J_n * v * ds(tag)
        term_int = J_n * v("+") * dS(tag)
        Lf = (
            (term_ext + term_int) if Lf is None
            else (Lf + term_ext + term_int)
        )

    ground_facets = np.concatenate([
        indices_list[i] for i in ground_idx
        if len(indices_list[i]) > 0
    ]).astype(np.int32)
    if len(ground_facets) == 0:
        raise RuntimeError(
            "All ground patches have zero matching facets",
        )
    ground_dofs = fem.locate_dofs_topological(
        V, fdim, ground_facets,
    )
    bc = fem.dirichletbc(
        PETSc.ScalarType(0.0), ground_dofs, V,
    )

    if comm.rank == 0:
        print(
            f"\nsolving with PETSc preset {cfg.preset_name!r}; "
            f"options: {cfg.petsc_options}",
            flush=True,
        )

    problem = LinearProblem(
        a, Lf, bcs=[bc],
        petsc_options_prefix="nerve_",
        petsc_options=cfg.petsc_options,
    )
    Ve = problem.solve()
    Ve.name = "Ve"

    # E = -grad Ve, projected onto CG1 vector space.
    W = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    E = fem.Function(W, name="E")
    E_expr = fem.Expression(
        -ufl.grad(Ve), W.element.interpolation_points,
    )
    E.interpolate(E_expr)

    with io.XDMFFile(comm, OUT / "Ve.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(Ve)
    with io.XDMFFile(comm, OUT / "E.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(E)

    # ---- sampling helpers ----
    tree = bb_tree(domain, tdim)

    def _samp(func, pts):
        return sample_function(
            func, pts, comm=comm, tree=tree, domain=domain,
        )

    # 1D axis-line sample. Built in CUFF-LOCAL ((0,0,z) along the
    # cuff's own axis), then transformed to canonical for mesh
    # sampling. Axis_line.npz keeps the cuff-local z axis so the
    # plot legend matches the patch (z, phi) the user designed.
    if AXIS_Z_LO is not None and AXIS_Z_HI is not None:
        z_axis = np.linspace(
            AXIS_Z_LO + 1e-5, AXIS_Z_HI - 1e-5, N_AXIS,
        )
    else:
        z_axis = np.linspace(
            -L / 2 + 1e-5, L / 2 - 1e-5, N_AXIS,
        )
    axis_pts_local = np.column_stack(
        [np.zeros(N_AXIS), np.zeros(N_AXIS), z_axis],
    )
    axis_pts = _canon_from_local(axis_pts_local)
    Ve_axis, _ = _samp(Ve, axis_pts)
    E_axis_canon, _ = _samp(E, axis_pts)
    # Rotate E vector from canonical to cuff-local so axis_line
    # "Ez" stays the AXIAL component along the cuff (what the AF
    # plot expects). Ve is a scalar — no rotation needed.
    E_axis = E_axis_canon @ CUFF_R

    # Per-patch axial summary (for the 1D plot legend).
    patch_z_summary = []
    patch_dz_summary = []
    patch_labels = []
    for p in patches:
        if p["type"] == "axial":
            patch_z_summary.append(float(p["z"]))
            patch_dz_summary.append(float(p["dz"]))
        elif p["type"] == "helical":
            patch_z_summary.append(
                0.5 * (p["z_start"] + p["z_end"]),
            )
            patch_dz_summary.append(
                float(p["z_end"] - p["z_start"]),
            )
        elif p["type"] == "life_band":
            # LIFE: contact band centred on (x,y,z) with
            # axial extent dz. Summary uses z + dz so the
            # 1D legend reports the band's axial span.
            patch_z_summary.append(float(p["z"]))
            patch_dz_summary.append(float(p["dz"]))
        elif p["type"] == "time_rect":
            # TIME: contact patch at (x,y,z) with axial
            # extent dz (transverse extent dl shown along
            # ribbon chord but irrelevant for the axial
            # plot).
            patch_z_summary.append(float(p["z"]))
            patch_dz_summary.append(float(p["dz"]))
        else:
            patch_z_summary.append(0.0)
            patch_dz_summary.append(0.0)
        patch_labels.append(
            f"id={p['id']} ({p['role']})",
        )
    elec_z_arr = np.array(patch_z_summary, dtype=float)
    elec_labels_arr = np.array(patch_labels, dtype=object)
    elec_dz_default = (
        float(np.mean(patch_dz_summary))
        if patch_dz_summary else 0.0
    )

    if comm.rank == 0:
        np.savez(
            OUT / "axis_line.npz",
            z=z_axis,
            Ve=Ve_axis[:, 0],
            Ex=E_axis[:, 0], Ey=E_axis[:, 1], Ez=E_axis[:, 2],
            elec_z=elec_z_arr,
            elec_labels=elec_labels_arr,
            elec_dz=elec_dz_default,
            electrode_config_name=str(config_name),
        )
        print(f"saved axis line ({N_AXIS} points)", flush=True)

    # ---- dense cross-section slice volume ----
    # Vectorised batch: build ALL (N_SLICES × SLICE_N²) points
    # in one array, sample once, reshape back. Pre-7.1 did one
    # Python-level loop iteration per slice with its own bb_tree
    # query — the rebuild dominated for moderate-size meshes.
    if SLICE_XY_HALF is not None:
        _xy_half = SLICE_XY_HALF
    else:
        _xy_half = R * 1.05
    xx = np.linspace(-_xy_half, _xy_half, SLICE_N)
    yy = np.linspace(-_xy_half, _xy_half, SLICE_N)
    X, Y = np.meshgrid(xx, yy, indexing="xy")
    if AXIS_Z_LO is not None and AXIS_Z_HI is not None:
        zz = np.linspace(
            AXIS_Z_LO + 1e-5, AXIS_Z_HI - 1e-5, N_SLICES,
        )
    else:
        zz = np.linspace(
            -L / 2 + 1e-5, L / 2 - 1e-5, N_SLICES,
        )

    # Broadcast-build the full (N_SLICES, SLICE_N, SLICE_N, 3)
    # point grid in one go; flatten to (N, 3) for sampling. The
    # grid is in CUFF-LOCAL (centered on cuff axis, perpendicular
    # slices at each z); transform to canonical for sampling so
    # the slice planes intersect the mesh on the correct planes
    # regardless of how the cuff is rotated in canonical space.
    pts4 = np.empty(
        (N_SLICES, SLICE_N, SLICE_N, 3), dtype=np.float64,
    )
    pts4[..., 0] = X[None, :, :]
    pts4[..., 1] = Y[None, :, :]
    pts4[..., 2] = zz[:, None, None]
    slice_pts_local = pts4.reshape(-1, 3)
    slice_pts = _canon_from_local(slice_pts_local)

    Ve_slice_flat, _ = _samp(Ve, slice_pts)
    E_slice_flat_canon, _ = _samp(E, slice_pts)
    # Rotate E components from canonical to cuff-local so the
    # slice heatmap's Ez plot stays the AXIAL component along
    # the cuff, even when the cuff is rotated in canonical.
    E_slice_flat = E_slice_flat_canon @ CUFF_R

    grid_shape = (N_SLICES, SLICE_N, SLICE_N)
    Ve_vol = Ve_slice_flat[:, 0].reshape(grid_shape)
    Ex_vol = E_slice_flat[:, 0].reshape(grid_shape)
    Ey_vol = E_slice_flat[:, 1].reshape(grid_shape)
    Ez_vol = E_slice_flat[:, 2].reshape(grid_shape)

    if comm.rank == 0:
        np.savez(
            OUT / "slice_volume.npz",
            x=xx, y=yy, z=zz,
            Ve=Ve_vol, Ex=Ex_vol, Ey=Ey_vol, Ez=Ez_vol,
            R=R, L=L,
            elec_z=elec_z_arr,
            elec_dz=elec_dz_default,
            electrode_config_name=str(config_name),
        )
        print(
            f"saved slice volume "
            f"({N_SLICES} z stations × {SLICE_N}² grid)",
            flush=True,
        )

    # ---- per-fiber Ve sampling ----
    # If `nerve_paths_fibers.npz` is present, sample Ve at every
    # trajectory point straight from the FEM function.
    _paths_npz = SHARED / "nerve_paths_fibers.npz"
    if _paths_npz.exists():
        try:
            _pd = np.load(_paths_npz, allow_pickle=True)
            _paths_flat = np.asarray(
                _pd["paths_flat"], dtype=np.float64,
            )
            _path_lens = np.asarray(
                _pd["path_lengths"], dtype=np.int64,
            )
            if comm.rank == 0:
                print(
                    f"sampling Ve directly at "
                    f"{_paths_flat.shape[0]:,} trajectory points "
                    f"across {len(_path_lens)} fibers …",
                    flush=True,
                )
            _Ve_flat, _mask_flat = _samp(Ve, _paths_flat)
            _E_flat_canon, _ = _samp(E, _paths_flat)
            # F3.2 — Ez along a fiber is interpreted downstream as
            # the AXIAL component (along the cuff axis), e.g. by
            # the activation-function plot. Rotate from canonical
            # to cuff-local so that semantic survives cuff
            # rotation. Vₑ is scalar and frame-independent.
            _E_flat = _E_flat_canon @ CUFF_R
            if comm.rank == 0:
                _n_off = int(np.sum(~_mask_flat))
                if _n_off > 0:
                    print(
                        f"  warning: {_n_off:,} / "
                        f"{_paths_flat.shape[0]:,} trajectory "
                        f"points fell outside the FEM mesh "
                        f"(NaN). Often these are seeded just "
                        f"outside the nerve volume — should be "
                        f"a tiny fraction.",
                        flush=True,
                    )
                np.savez(
                    OUT / "paths_Ve.npz",
                    paths_flat=_paths_flat,
                    path_lengths=_path_lens,
                    Ve_flat=_Ve_flat[:, 0].astype(np.float64),
                    Ex_flat=_E_flat[:, 0].astype(np.float64),
                    Ey_flat=_E_flat[:, 1].astype(np.float64),
                    Ez_flat=_E_flat[:, 2].astype(np.float64),
                )
                print(
                    "  saved paths_Ve.npz "
                    "(per-point Ve sampled directly from FEM function)",
                    flush=True,
                )
        except Exception as _ex:
            if comm.rank == 0:
                print(
                    f"  could not sample Ve along trajectories: "
                    f"{_ex}",
                    flush=True,
                )

    # ---- nerve surface Ve sampling ----
    _ns_pts_path = SHARED / "nerve_surface_pts.npz"
    if _ns_pts_path.exists():
        try:
            _nsd = np.load(_ns_pts_path, allow_pickle=True)
            _ns_pts = np.asarray(_nsd["pts"], dtype=np.float64)
            if comm.rank == 0:
                print(
                    f"sampling Ve at {_ns_pts.shape[0]:,} nerve "
                    f"surface vertices …",
                    flush=True,
                )
            _ns_Ve, _ns_mask = _samp(Ve, _ns_pts)
            _ns_E_canon, _ = _samp(E, _ns_pts)
            # F3.2 — same cuff-local-axis rotation as the per-
            # fiber sampling above, so the surface "Ez" stays the
            # cuff-axial component under rotation.
            _ns_E = _ns_E_canon @ CUFF_R
            if comm.rank == 0:
                _n_off = int(np.sum(~_ns_mask))
                if _n_off > 0:
                    print(
                        f"  warning: {_n_off:,} / "
                        f"{_ns_pts.shape[0]:,} surface points "
                        f"fell outside the FEM mesh (NaN). Often "
                        f"vertices that round just outside the "
                        f"discretised boundary.",
                        flush=True,
                    )
                np.savez(
                    OUT / "nerve_surface_Ve.npz",
                    Ve=_ns_Ve[:, 0].astype(np.float64),
                    Ex=_ns_E[:, 0].astype(np.float64),
                    Ey=_ns_E[:, 1].astype(np.float64),
                    Ez=_ns_E[:, 2].astype(np.float64),
                    n_pts=int(_ns_pts.shape[0]),
                )
                print(
                    f"  saved nerve_surface_Ve.npz "
                    f"({_ns_pts.shape[0]} pts)",
                    flush=True,
                )
        except Exception as _ex:
            if comm.rank == 0:
                print(
                    f"  surface Ve sampling failed: {_ex}",
                    flush=True,
                )

    # ---- I1 Phase A — DC impedance via Dirichlet dual-solve ----
    # Gated on the GOLGI_EMIT_IMPEDANCE env var (set by
    # pipeline/fem.py from FEMJobRequest.emit_impedance). For each
    # contact: build a BC stack [V=1 on that contact's facets,
    # V=0 on the outer muscle bbox facets], re-solve Laplace,
    # integrate σ·∇V·n over the contact's surface → I → Z = 1/I.
    # Per-pair impedance is derived from the user's configured
    # anode/cathode pairs (or from contact_polarities when the
    # config is loaded). Output: `OUT/impedance.json`.
    if os.environ.get("GOLGI_EMIT_IMPEDANCE", "0") == "1":
        try:
            _emit_impedance_dc(
                domain=domain,
                cell_tags=cell_tags,
                V=V,
                sigma_fn=sigma_fn,
                patches=patches,
                indices_list=indices_list,
                patch_areas=patch_areas,
                ft=ft,
                ds=ds, dS=dS,
                fdim=fdim, tdim=tdim,
                comm=comm,
                OUT=OUT,
                config_name=config_name,
                petsc_options=cfg.petsc_options,
            )
        except Exception as _ex:                         # noqa: BLE001
            if comm.rank == 0:
                print(
                    f"WARN: impedance solve failed: "
                    f"{type(_ex).__name__}: {_ex}",
                    flush=True,
                )

    # ---- R1.2 — per-contact reciprocity (cNAP lead fields) ----
    # Gated on the GOLGI_EMIT_RECORDING env var (set by
    # pipeline/fem.py from FEMJobRequest.emit_recording). For each
    # contact referenced by electrode_config.json's
    # `recording_montages`, run one Laplace solve with 1 A
    # injected at that contact (Neumann) + V=0 on the outer-muscle
    # bbox (Dirichlet), all other contacts insulating. Output
    # lives in <SHARED>/recording/ (per-design, shared across all
    # configs bound to this design) — the lead field depends only
    # on geometry + σ, NOT on stim wiring.
    if (
        os.environ.get("GOLGI_EMIT_RECORDING", "0") == "1"
        and recording_montages
    ):
        try:
            _emit_reciprocity_solves(
                domain=domain,
                cell_tags=cell_tags,
                V=V,
                sigma_fn=sigma_fn,
                patches=patches,
                indices_list=indices_list,
                patch_areas=patch_areas,
                ds=ds, dS=dS,
                fdim=fdim, tdim=tdim,
                comm=comm,
                SHARED=SHARED,
                config_name=config_name,
                petsc_options=cfg.petsc_options,
                recording_montages=recording_montages,
                sigma_by_tag=SIGMA_BY_TAG,
                sample_function_=_samp,
                solver_preset=cfg.preset_name,
                peri_rs=PERI_RS,
            )
        except Exception as _ex:                         # noqa: BLE001
            if comm.rank == 0:
                print(
                    f"WARN: reciprocity solves failed: "
                    f"{type(_ex).__name__}: {_ex}",
                    flush=True,
                )

    if comm.rank == 0:
        print(
            "\nDone. To visualize:  python plot_results.py  or  "
            "marimo edit notebook.py",
            flush=True,
        )
        print(
            "Or open results/Ve.xdmf and results/E.xdmf in ParaView.",
            flush=True,
        )


# ===========================================================================
# Entry point
# ===========================================================================

def main(argv: list[str] | None = None) -> None:
    """CLI entry point. `argv` defaults to sys.argv[1:]; pass a
    list (e.g. from tests) to override."""
    comm = MPI.COMM_WORLD

    # Parse our own flags; leave unknown args for PETSc. petsc4py
    # auto-reads sys.argv at import time, so flags like
    # `-log_view` and `-ksp_monitor` work transparently — we just
    # need to not error on them here.
    parser = build_arg_parser()
    args, _petsc_extras = parser.parse_known_args(argv)

    if comm.rank == 0:
        print(
            f"Running FENiCSx solver on {comm.size} MPI processes.",
            flush=True,
        )

    # Sanity-check --cores vs the actual MPI world size.
    if (args.cores is not None
            and int(args.cores) != int(comm.size)):
        if comm.rank == 0:
            print(
                f"WARN: --cores={args.cores} doesn't match "
                f"MPI.COMM_WORLD.size={comm.size}. The actual "
                f"rank count is controlled by `mpirun -n N`.",
                flush=True,
            )

    # Resolve preset: CLI overrides MeshConfig overrides default.
    # Need to read the on-disk MeshConfig now to honour
    # solver_preset baked in by the pipeline driver.
    out_dir_for_lookup = (
        args.out_dir
        or Path(
            os.environ.get("SOLVE_OUT_DIR", str(HERE / "results"))
        )
    )
    mesh_cfg_for_preset = load_mesh_config(out_dir_for_lookup)
    cfg = resolve_config(args, mesh_cfg_for_preset, comm)

    if comm.rank == 0 and _petsc_extras:
        print(
            f"forwarding {len(_petsc_extras)} extra arg(s) to "
            f"PETSc via sys.argv: {_petsc_extras}",
            flush=True,
        )

    run_solve(cfg, comm)


if __name__ == "__main__":
    main()
