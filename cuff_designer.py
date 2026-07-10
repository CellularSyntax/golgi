# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""ASCENT-style cuff electrode designer.

Pure-Python re-implementation of a subset of the ASCENT (Duke / wmglab)
COMSOL "part primitive" library, together with a small expression
evaluator that understands the COMSOL-flavoured syntax used in the
bundled `DUKE_cuffs/*.json` presets (e.g. `1.7145 [mm]`, `rev_BD_insul_LN`,
`sqrt(...)`).

Supported primitives (covers 100% of the bundled presets):
  - CuffFill_Primitive       : solid cylinder filling the cuff bore
  - TubeCuff_Primitive       : hollow cylinder with optional angular gap
  - CircleContact_Primitive  : curved disc contact on the inner cuff wall
  - LivaNova_Primitive       : helical insulator + helical conductor

The bundled presets are:
  - LivaNova 2000/3000        (LivaNova + CuffFill)
  - MultiContact 2000/3000    (TubeCuff + CuffFill + N×CircleContact)

Geometry is built as pyvista PolyData. Returned designs are lists of
(label, mesh, role, material) tuples; `role` ∈ {"insulator", "conductor",
"recess", "fill"}.

All geometry is produced in the cuff's LOCAL frame (z = cuff axis,
centred on origin) in SI units (metres). Callers are expected to
transform into a nerve-fitted cuff frame via the existing PCA flow.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# COMSOL expression evaluator
#
# The bundled JSONs use COMSOL's scripting syntax:
#   * Numeric literals with unit brackets: `0.673 [mm]`, `90 [deg]`
#   * Math functions: sqrt, sin, cos, max, min, pi
#   * Cross-parameter references: `rev_PD_insul_LN/2`
#
# We strip unit brackets to "* (factor)" and eval against a namespace
# of {param_name: float}. Topo-resolution is iterative — params are
# evaluated in passes until every one resolves (or stops progressing).
# ---------------------------------------------------------------------------

_UNIT_FACTORS = {
    # length → metres
    "m": 1.0,
    "cm": 1.0e-2,
    "mm": 1.0e-3,
    "um": 1.0e-6,
    "nm": 1.0e-9,
    "inch": 0.0254,
    # angle → radians
    "deg": math.pi / 180.0,
    "rad": 1.0,
    # time → seconds (defensive; presets don't use these but keep symmetry)
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
}

_SAFE_NAMES: dict = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "max": max,
    "min": min,
    "abs": abs,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _strip_units(expr: str) -> str:
    """Replace `… [unit]` suffixes with `… * (factor)` so the result
    is a plain numeric expression in SI units, and convert COMSOL's
    `^` power operator to Python's `**`. (Python's `^` is XOR, which
    silently produced wrong results before this fix.)"""
    def _replace(m):
        unit = m.group(1).strip()
        factor = _UNIT_FACTORS.get(unit, 1.0)
        return f"*({factor})"
    out = re.sub(r"\s*\[\s*([a-zA-Z]+)\s*\]", _replace, expr)
    out = out.replace("^", "**")
    return out


def eval_expr(expr, namespace: dict) -> float:
    """Evaluate one expression in `namespace`. Plain numbers pass
    through; strings get unit-stripped + eval'd against the namespace
    plus our whitelisted math globals. Raises ValueError on parse
    failure or unresolved references."""
    if expr is None:
        return 0.0
    if isinstance(expr, (int, float)):
        return float(expr)
    s = str(expr).strip()
    if not s:
        return 0.0
    s = _strip_units(s)
    env = dict(_SAFE_NAMES)
    env.update(namespace)
    try:
        return float(eval(s, {"__builtins__": {}}, env))
    except NameError:
        # Re-raise so the caller's topo loop can retry on the next pass.
        raise
    except Exception as ex:
        raise ValueError(
            f"Could not evaluate expression {expr!r}: {ex}"
        )


def resolve_params(params: list,
                      extras: dict | None = None,
                      override_names: set | None = None,
                      ) -> dict:
    """Resolve a list of {name, expression, description?} dicts into
    a flat {name: float} namespace.

    `extras` seeds the namespace with externally-supplied values
    (z_nerve, r_nerve, etc.) that the preset references but doesn't
    declare.

    `override_names`, if given, lists param names whose values are
    SUPPLIED VIA `extras` AND MUST NOT BE RE-EVALUATED FROM THEIR
    EXPRESSION. Used by the designer UI: when the user types a
    custom radius, we want that value to propagate through downstream
    expressions instead of getting overwritten on the next pass."""
    ns: dict[str, float] = dict(extras or {})
    fixed: set = set(override_names or [])
    # Skip overridden params entirely — their value is already in ns.
    remaining = [p for p in params if p.get("name") not in fixed]
    # Iterate until no progress — handles forward refs without a
    # full topo sort. ASCENT presets are well-ordered so this
    # usually converges in 1-2 passes.
    last_count = len(remaining) + 1
    while remaining and len(remaining) < last_count:
        last_count = len(remaining)
        still: list = []
        for p in remaining:
            try:
                ns[p["name"]] = eval_expr(p.get("expression"), ns)
            except (NameError, ValueError):
                still.append(p)
        remaining = still
    # Anything still unresolved → set to 0.0 so the renderer can
    # at least produce a partial result (and we don't crash on a
    # single bad cell).
    for p in remaining:
        ns.setdefault(p["name"], 0.0)
    return ns


def resolve_instance_def(
    instance: dict, ns: dict,
) -> dict:
    """Resolve every key in `instance["def"]` against the global
    namespace, returning a flat {key: float} dict."""
    out: dict[str, float] = {}
    for k, v in (instance.get("def") or {}).items():
        try:
            out[k] = eval_expr(v, ns)
        except Exception:
            out[k] = 0.0
    return out


# ---------------------------------------------------------------------------
# Preset loader
# ---------------------------------------------------------------------------

def load_cuff_presets(preset_dir) -> dict:
    """Scan a directory for `*.json` cuff presets. Returns
    {filename_stem: preset_dict}. Empty if the dir doesn't exist."""
    out: dict = {}
    d = Path(preset_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as ex:
            print(
                f"[cuff_designer] failed to load {p.name}: {ex}",
                flush=True,
            )
    return out


# ---------------------------------------------------------------------------
# Primitive renderers
#
# Each renderer:
#   * Takes the per-instance parameter dict (already resolved to SI
#     units) merged onto the primitive's local defaults.
#   * Returns a list of (label, mesh, role) tuples — one per material
#     in the primitive (insulator + conductor + optional recess + …).
# ---------------------------------------------------------------------------


def _polygon_revolve(
    cross_xz: np.ndarray,
    theta_start: float,
    theta_end: float,
    n_az: int = 64,
) -> pv.PolyData:
    """Revolve a closed 2-D polygon (in the xz plane, x ≥ 0) around
    the +z axis from `theta_start` to `theta_end`. Returns a closed
    PolyData with side walls + start/end caps (if partial revolve).
    Cross-section coords are (x = radius, z = axial)."""
    nx = cross_xz.shape[0]
    angles = np.linspace(theta_start, theta_end, n_az)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # Vertices: nx polygon points × n_az angular slices.
    pts = np.empty((n_az * nx, 3), dtype=np.float64)
    for i, (ca, sa) in enumerate(zip(cos_a, sin_a)):
        base = i * nx
        pts[base:base + nx, 0] = cross_xz[:, 0] * ca
        pts[base:base + nx, 1] = cross_xz[:, 0] * sa
        pts[base:base + nx, 2] = cross_xz[:, 1]
    # Side faces: connect consecutive angular slices, looping around
    # the polygon's edges (nx vertices, nx edges since closed).
    faces: list = []
    for i in range(n_az - 1):
        base0 = i * nx
        base1 = (i + 1) * nx
        for j in range(nx):
            j_next = (j + 1) % nx
            faces.extend([4,
                           base0 + j, base0 + j_next,
                           base1 + j_next, base1 + j])
    # End caps for a partial revolve — close the open ends with
    # triangle fans from the polygon centroid.
    full_circle = abs(
        (theta_end - theta_start) - 2.0 * math.pi
    ) < 1.0e-9
    if not full_circle:
        # Triangulate the start-angle polygon (i = 0)
        # and end-angle polygon (i = n_az - 1).
        # Use ear-clipping equivalent via a simple fan from centroid.
        for slice_idx, normal_sign in ((0, -1), (n_az - 1, +1)):
            base = slice_idx * nx
            cx = float(np.mean(pts[base:base + nx, 0]))
            cy = float(np.mean(pts[base:base + nx, 1]))
            cz = float(np.mean(pts[base:base + nx, 2]))
            centre_idx = pts.shape[0]
            pts = np.vstack([pts, [cx, cy, cz]])
            for j in range(nx):
                j_next = (j + 1) % nx
                if normal_sign > 0:
                    faces.extend([3, centre_idx,
                                   base + j_next, base + j])
                else:
                    faces.extend([3, centre_idx,
                                   base + j, base + j_next])
    return pv.PolyData(pts, np.asarray(faces, dtype=np.int64))


def render_cuff_fill(p: dict) -> list:
    """CuffFill_Primitive — solid cylinder filling between the cuff
    inner wall and the nerve. Used for the "saline" region of a
    bipolar / multipolar cuff.

    Params:
        Center      z-centre (m)
        Radius      outer radius (m)
        Thk         radial thickness of the fill (m)
        L           axial length (m)
        x_shift     transverse shift (m)
        y_shift     transverse shift (m)
        r_n         nerve outer radius (m) [3D variants only]
    """
    center = float(p.get("Center", 0.0))
    radius = float(p.get("Radius", 1.5e-3))
    L = float(p.get("L", 4.0e-3))
    x_shift = float(p.get("x_shift", 0.0))
    y_shift = float(p.get("y_shift", 0.0))
    # The 3D variants describe the fill as the annulus between r_n
    # (the nerve) and radius. Without an r_n we render as a solid
    # cylinder, which is fine for visual preview.
    r_inner = float(p.get("r_n", 0.0))
    if r_inner <= 0 or r_inner >= radius:
        mesh = pv.Cylinder(
            center=(x_shift, y_shift, center),
            direction=(0.0, 0.0, 1.0),
            radius=radius,
            height=L,
            resolution=64,
            capping=True,
        )
    else:
        # Annular fill: revolve a rectangle (r_inner, -L/2) → (radius, -L/2)
        # → (radius, +L/2) → (r_inner, +L/2) around z.
        cross = np.array([
            [r_inner, -L / 2],
            [radius, -L / 2],
            [radius, +L / 2],
            [r_inner, +L / 2],
        ])
        mesh = _polygon_revolve(cross, 0.0, 2.0 * math.pi, n_az=64)
        mesh = mesh.translate(
            (x_shift, y_shift, center), inplace=False,
        )
    return [("CuffFill", mesh, "fill")]


def render_tube_cuff(p: dict) -> list:
    """TubeCuff_Primitive — hollow cylinder R_in→R_out × Tube_L.
    Optional angular gap when `Tube_theta` < 2π; the bundled holes
    feature (N_holes) is skipped here (visual approximation —
    rendering them needs CSG which doesn't fit the live-preview
    budget). The hole knobs in the JSON still resolve to numbers so
    the parameter table reads cleanly.

    Params:
        R_in, R_out     inner / outer radius (m)
        Tube_L          axial length (m)
        Center          z-centre (m)
        Tube_theta      angular sweep (rad; full circle = 2π)
        Rot_def         rotation of the gap (rad)
    """
    R_in = float(p.get("R_in", 1.0e-3))
    R_out = float(p.get("R_out", 1.5e-3))
    L = float(p.get("Tube_L", 5.0e-3))
    center = float(p.get("Center", 0.0))
    theta = float(p.get("Tube_theta", 2.0 * math.pi))
    rot = float(p.get("Rot_def", 0.0))
    # Clamp to (0, 2π] — 359° in the JSON is meant as full-circle.
    if theta >= 2.0 * math.pi - 1.0e-3:
        theta = 2.0 * math.pi
    theta = max(theta, 1.0e-3)
    cross = np.array([
        [R_in, -L / 2],
        [R_out, -L / 2],
        [R_out, +L / 2],
        [R_in, +L / 2],
    ])
    mesh = _polygon_revolve(
        cross, rot, rot + theta, n_az=max(16, int(theta / (2 * math.pi) * 96)),
    )
    mesh = mesh.translate((0.0, 0.0, center), inplace=False)
    return [("TubeCuff", mesh, "insulator")]


def render_circle_contact(p: dict) -> list:
    """CircleContact_Primitive — curved disc-shaped contact sitting
    on the inner cuff wall, optionally recessed into the cuff.
    Rendered as a thin radially-thick patch covering a small
    angular + axial region centred on `Rotation_angle` × `Center`.

    Params:
        R_in            cuff inner radius (m)
        Circle_recess   recess depth (m)
        Circle_thk      contact radial thickness (m)
        Circle_diam     contact in-plane diameter (m) — interpreted
                        as both arc-length AND axial extent
        Rotation_angle  azimuthal position (rad)
        Center          z-centre (m)
        Overshoot       padding for partition cuts (m)
    """
    R_in = float(p.get("R_in", 1.5e-3))
    recess = float(p.get("Circle_recess", 0.0))
    thk = float(p.get("Circle_thk", 50.0e-6))
    diam = float(p.get("Circle_diam", 1.0e-3))
    center = float(p.get("Center", 0.0))
    phi0 = float(p.get("Rotation_angle", 0.0))
    # The arc-length and axial extents are both `diam` (a circle
    # projected onto a cylinder — the contact looks elliptical on
    # the curved surface but ASCENT keeps the param naming simple).
    r_in_contact = R_in + recess
    r_out_contact = r_in_contact + thk
    half_dphi = (diam / 2.0) / max(r_in_contact, 1.0e-9)
    half_dz = diam / 2.0
    out: list = []
    # 1) The contact volume itself — a thin radial slab over the
    #    angular + axial patch.
    cross = np.array([
        [r_in_contact, center - half_dz],
        [r_out_contact, center - half_dz],
        [r_out_contact, center + half_dz],
        [r_in_contact, center + half_dz],
    ])
    contact_mesh = _polygon_revolve(
        cross, phi0 - half_dphi, phi0 + half_dphi,
        n_az=max(8, int(2.0 * half_dphi / (2.0 * math.pi) * 64)),
    )
    out.append(("Contact", contact_mesh, "conductor"))
    # 2) Optional recess pocket — the void between the cuff inner
    #    wall (R_in) and where the contact starts (R_in + recess).
    #    Renders as a thin shell; useful for visual debugging.
    if recess > 0:
        cross_r = np.array([
            [R_in, center - half_dz],
            [r_in_contact, center - half_dz],
            [r_in_contact, center + half_dz],
            [R_in, center + half_dz],
        ])
        recess_mesh = _polygon_revolve(
            cross_r, phi0 - half_dphi, phi0 + half_dphi,
            n_az=max(8, int(2.0 * half_dphi / (2.0 * math.pi) * 64)),
        )
        out.append(("Recess", recess_mesh, "recess"))
    return out


def _orthonormal_frame_along_helix(
    s_vals: np.ndarray, R_in: float, pitch: float, center_z: float,
    L_cuff: float, n_rev: float,
) -> tuple:
    """For each s ∈ s_vals, compute (point, tangent, radial, normal)
    on the helix x(s) = R cos(2πs), y(s) = R sin(2πs),
    z(s) = center_z + L_cuff·(s/n_rev) - L_cuff/2. Returns 4 (N, 3)
    arrays."""
    two_pi = 2.0 * math.pi
    cos_t = np.cos(two_pi * s_vals)
    sin_t = np.sin(two_pi * s_vals)
    z_vals = center_z + L_cuff * (s_vals / n_rev) - L_cuff / 2
    pts = np.column_stack([R_in * cos_t, R_in * sin_t, z_vals])
    # Tangent: d/ds of the parametric curve.
    dx_ds = -two_pi * R_in * sin_t
    dy_ds = +two_pi * R_in * cos_t
    dz_ds = (L_cuff / n_rev) * np.ones_like(s_vals)
    tangent = np.column_stack([dx_ds, dy_ds, dz_ds])
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    # Radial (outward from z axis), in the xy plane.
    radial = np.column_stack([cos_t, sin_t, np.zeros_like(s_vals)])
    # Third leg of the frame — perpendicular to both, forms a stable
    # rotation-minimising-ish frame for the sweep.
    normal = np.cross(tangent, radial)
    normal_n = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(normal_n, 1.0e-12)
    return pts, tangent, radial, normal


def _sweep_rectangle_along_helix(
    s_vals: np.ndarray,
    R_centre: float, pitch: float, center_z: float, L_cuff: float,
    n_rev: float, radial_thk: float, axial_width: float,
    radial_offset: float = 0.0,
) -> pv.PolyData:
    """Sweep an axis-aligned rectangle of size
    (radial_thk × axial_width) along the helix x²+y²=R_centre², with
    `radial_offset` shifting it radially outward from the helix
    spine. Closed mesh with start/end caps."""
    R_spine = R_centre + radial_offset
    pts_c, T, R_hat, N_hat = _orthonormal_frame_along_helix(
        s_vals, R_spine, pitch, center_z, L_cuff, n_rev,
    )
    half_r = radial_thk / 2.0
    half_w = axial_width / 2.0
    # Four corners of the rectangle in 3D — radial direction along
    # R_hat (outward), "width" direction along N_hat.
    c1 = pts_c + (-half_r) * R_hat + (-half_w) * N_hat
    c2 = pts_c + (+half_r) * R_hat + (-half_w) * N_hat
    c3 = pts_c + (+half_r) * R_hat + (+half_w) * N_hat
    c4 = pts_c + (-half_r) * R_hat + (+half_w) * N_hat
    n_s = s_vals.shape[0]
    # Flatten into a single point array of order [c1, c2, c3, c4] per
    # parameter step. Total 4·n_s points.
    pts = np.empty((n_s * 4, 3), dtype=np.float64)
    pts[0::4] = c1
    pts[1::4] = c2
    pts[2::4] = c3
    pts[3::4] = c4
    faces: list = []
    for i in range(n_s - 1):
        a0, a1, a2, a3 = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        b0, b1, b2, b3 = a0 + 4, a1 + 4, a2 + 4, a3 + 4
        # Four side quads connecting cross-section i → i+1.
        faces.extend([4, a0, a1, b1, b0])  # outer-radial face
        faces.extend([4, a1, a2, b2, b1])  # top face
        faces.extend([4, a2, a3, b3, b2])  # inner-radial face
        faces.extend([4, a3, a0, b0, b3])  # bottom face
    # End caps: quads on the start (i=0) and end (i=n_s-1).
    s0 = 0
    e0 = 4 * (n_s - 1)
    faces.extend([4, s0 + 0, s0 + 3, s0 + 2, s0 + 1])
    faces.extend([4, e0 + 0, e0 + 1, e0 + 2, e0 + 3])
    return pv.PolyData(pts, np.asarray(faces, dtype=np.int64))


def render_livanova(p: dict) -> list:
    """LivaNova_Primitive — bipolar helical electrode. Three
    segments along the helix: insulator-only, insulator + conductor
    (the active contact region), insulator-only.

    Params (all in SI units, defaults from Part.java):
        Center      z-centre (m)
        Thk_cuff    radial thickness of the insulator (m)
        W_cuff      axial width of the insulator (m)
        R_in        cuff inner radius (m)
        L_cuff      total axial length (m)
        Rev_insul   number of insulator revolutions (turns)
        Rev_cond    number of conductor revolutions (turns)
        Recess      radial recess of the conductor (m)
        Thk_elec    radial thickness of the conductor (m)
        W_elec      axial width of the conductor (m)
    """
    center = float(p.get("Center", 0.0))
    Thk_cuff = float(p.get("Thk_cuff", 610.0e-6))
    W_cuff = float(p.get("W_cuff", 1410.0e-6))
    R_in = float(p.get("R_in", 1109.4e-6))
    L_cuff = float(p.get("L_cuff", 3852.6e-6))
    Rev_insul = float(p.get("Rev_insul", 2.2471))
    Rev_cond = float(p.get("Rev_cond", 0.84514))
    Recess = float(p.get("Recess", 0.0))
    Thk_elec = float(p.get("Thk_elec", 50.0e-6))
    W_elec = float(p.get("W_elec", 775.0e-6))
    # Pitch implied by L_cuff / Rev_insul (m per revolution).
    pitch = L_cuff / max(Rev_insul, 1.0e-9)
    # Sampling density — roughly 60 points per turn keeps the helix
    # visually smooth without bloating the polydata.
    samples_per_turn = 60
    n_total = int(max(2, samples_per_turn * Rev_insul))
    s_full = np.linspace(0.0, Rev_insul, n_total)
    # Conductor region centred on Rev_insul/2.
    s_cond_lo = Rev_insul / 2.0 - Rev_cond / 2.0
    s_cond_hi = Rev_insul / 2.0 + Rev_cond / 2.0
    # Insulator sweep — full helix. Cross-section radial axis goes
    # outward from R_in (placement matches Part.java: rectangle pos
    # = R_in + 0.5*Thk_cuff in the radial direction).
    insul_mesh = _sweep_rectangle_along_helix(
        s_full, R_in, pitch, center, L_cuff, Rev_insul,
        radial_thk=Thk_cuff, axial_width=W_cuff,
        radial_offset=Thk_cuff / 2.0,
    )
    out = [("LivaNova_Insulator", insul_mesh, "insulator")]
    # Conductor sweep — only over the central segment.
    s_cond = np.linspace(
        s_cond_lo, s_cond_hi,
        int(max(2, samples_per_turn * Rev_cond)),
    )
    cond_mesh = _sweep_rectangle_along_helix(
        s_cond, R_in, pitch, center, L_cuff, Rev_insul,
        radial_thk=Thk_elec, axial_width=W_elec,
        # Radial offset = Recess + Thk_elec/2 keeps the conductor's
        # inner face at R_in + Recess (the recess depth pushes the
        # contact further out from the nerve when Recess > 0).
        radial_offset=Recess + Thk_elec / 2.0,
    )
    out.append(("LivaNova_Conductor", cond_mesh, "conductor"))
    return out


# ---------------------------------------------------------------------------
# Dispatch + master render
# ---------------------------------------------------------------------------

_PRIMITIVES = {
    "CuffFill_Primitive": render_cuff_fill,
    "TubeCuff_Primitive": render_tube_cuff,
    "CircleContact_Primitive": render_circle_contact,
    "LivaNova_Primitive": render_livanova,
}


# Some ASCENT presets (notably LivaNova_v2) carry the primitive's
# local parameters as top-level preset params with a code suffix
# (e.g. `r_cuff_in_LN` → primitive's `R_in`), and leave the
# instance.def with only `Center` + `Corr`. COMSOL resolves these
# at the model level via global parameter scope; in our Python
# pipeline we make the resolution explicit by mapping
# {primitive_local_name: namespace_key} per primitive type.
#
# Keys here are SECONDARY fallbacks — if the instance.def already
# supplies a value for a given primitive-local, that wins. The
# mapping only fills in the gaps.
_PRIMITIVE_NS_FALLBACK_MAP = {
    "LivaNova_Primitive": {
        # Use post-deformation values (PD) since those describe the
        # cuff as it sits around the nerve, not its rest shape.
        "R_in": "r_cuff_in_LN",
        "Thk_cuff": "thk_cuff_LN",
        "W_cuff": "w_cuff_LN",
        "L_cuff": "L_cuff_LN_PD",
        "Rev_insul": "rev_PD_insul_LN",
        "Rev_cond": "rev_PD_cond_LN",
        "Recess": "recess_LN",
        "Thk_elec": "thk_elec_LN",
        "W_elec": "w_elec_LN",
    },
}


# Default colours per material role — picked up by golgi for actor
# styling. Insulator = silicone grey, conductor = gold, fill =
# translucent saline blue, recess = darker grey overlay.
ROLE_COLORS = {
    "insulator": (0.94, 0.94, 0.94),
    "conductor": (0.95, 0.78, 0.18),
    "fill":      (0.37, 0.77, 0.94),
    "recess":    (0.55, 0.55, 0.60),
}

ROLE_OPACITIES = {
    "insulator": 0.85,
    "conductor": 1.00,
    "fill":      0.30,
    "recess":    0.55,
}


def supported_primitive_types() -> set:
    return set(_PRIMITIVES.keys())


# ---------------------------------------------------------------------------
# Designer UI metadata — which preset params to expose as sliders.
#
# Keyed by preset `code` (the JSON's top-level `code` field: "LN",
# "MCT", ...). Each entry is a list of {name, label, unit, min, max,
# step} dicts. Only these params get a slider/numeric row in the
# designer dialog — the rest stay implicit (computed from these via
# the preset's expression graph).
#
# Unit factors below match cuff_designer._UNIT_FACTORS so display
# values can round-trip cleanly between the UI and the SI namespace.
# ---------------------------------------------------------------------------
DESIGNER_VISIBLE_PARAMS: dict = {
    "LN": [
        {"name": "r_cuff_in_pre_LN",
         "label": "cuff inner radius",
         "unit": "mm", "min": 0.30, "max": 5.00, "step": 0.05},
        {"name": "thk_cuff_LN",
         "label": "cuff thickness",
         "unit": "mm", "min": 0.10, "max": 1.50, "step": 0.01},
        {"name": "w_cuff_LN",
         "label": "cuff width (axial)",
         "unit": "mm", "min": 0.30, "max": 4.00, "step": 0.05},
        {"name": "rev_BD_insul_LN",
         "label": "insulator revolutions",
         "unit": "turns", "min": 0.50, "max": 5.00, "step": 0.05},
        {"name": "rev_BD_cond_LN",
         "label": "conductor revolutions",
         "unit": "turns", "min": 0.10, "max": 3.00, "step": 0.05},
        {"name": "helix_pitch_LN",
         "label": "helix pitch",
         "unit": "mm", "min": 0.30, "max": 5.00, "step": 0.05},
        {"name": "w_elec_LN",
         "label": "contact width",
         "unit": "mm", "min": 0.10, "max": 2.50, "step": 0.05},
        {"name": "thk_elec_LN",
         "label": "contact thickness",
         "unit": "um", "min": 5.0, "max": 500.0, "step": 5.0},
        {"name": "recess_LN",
         "label": "contact recess",
         "unit": "mm", "min": 0.00, "max": 0.50, "step": 0.01},
        {"name": "sep_elec_LN",
         "label": "bipolar separation",
         "unit": "mm", "min": 1.0, "max": 20.0, "step": 0.2},
    ],
    "MCT": [
        {"name": "R_in_MCT",
         "label": "cuff inner radius",
         "unit": "mm", "min": 0.50, "max": 5.00, "step": 0.05},
        {"name": "R_out_MCT",
         "label": "cuff outer radius",
         "unit": "mm", "min": 1.00, "max": 6.00, "step": 0.05},
        {"name": "L_MCT",
         "label": "cuff length",
         "unit": "mm", "min": 2.0, "max": 20.0, "step": 0.2},
        {"name": "Theta_MCT",
         "label": "cuff angular sweep",
         "unit": "deg", "min": 180.0, "max": 360.0, "step": 5.0},
        {"name": "L_elec_MCT",
         "label": "contact diameter",
         "unit": "mm", "min": 0.20, "max": 2.50, "step": 0.05},
        {"name": "Thk_elec_MCT",
         "label": "contact thickness",
         "unit": "um", "min": 5.0, "max": 500.0, "step": 5.0},
        {"name": "Recess_MCT",
         "label": "contact recess",
         "unit": "mm", "min": 0.00, "max": 0.50, "step": 0.01},
        {"name": "ang_contactcenter_MCT",
         "label": "contact angular pitch",
         "unit": "deg", "min": 5.0, "max": 90.0, "step": 1.0},
    ],
}


# Display-unit → SI conversion factor. Mirrors _UNIT_FACTORS but
# spelled out for clarity in the UI code. Sliders/inputs are in
# DISPLAY units; the resolver expects SI everywhere.
DISPLAY_UNIT_TO_SI: dict = {
    "mm": 1.0e-3,
    "um": 1.0e-6,
    "m": 1.0,
    "deg": math.pi / 180.0,
    "rad": 1.0,
    "turns": 1.0,
    "": 1.0,
}


def render_design(preset: dict,
                    param_overrides: dict | None = None,
                    ns_extras: dict | None = None,
                    ) -> list:
    """Render every instance in a preset into a list of
    (instance_label, primitive_label, mesh, role) tuples. Skips
    unsupported primitive types with a printed warning.

    `param_overrides` is a `{name: float_value_SI}` dict layered on
    top of the preset's `params` (matched by name). Overridden
    params are treated as user-fixed: their expressions are NOT
    re-evaluated and downstream params that depend on them pick up
    the new value.

    `ns_extras` seeds the namespace with values the preset
    references but doesn't declare (z_nerve, r_nerve, ...). Both
    default to empty."""
    overrides = {
        k: float(v) for k, v in (param_overrides or {}).items()
    }
    extras = dict(ns_extras or {})
    extras.update(overrides)   # overrides seed the namespace
    # Build the global namespace from preset.params + extras +
    # explicit overrides.
    base_params = list(preset.get("params", []))
    # Local params (3D variants) get folded into the same list —
    # they're identical in shape (name + expression).
    base_params.extend(preset.get("local_params", []))
    ns = resolve_params(
        base_params, extras, override_names=set(overrides.keys()),
    )
    out: list = []
    for inst in preset.get("instances", []):
        ptype = inst.get("type", "")
        renderer = _PRIMITIVES.get(ptype)
        if renderer is None:
            print(
                f"[cuff_designer] skipping unsupported primitive "
                f"type {ptype!r}",
                flush=True,
            )
            continue
        local_p = resolve_instance_def(inst, ns)
        # Fall back to namespace mapping for primitive locals the
        # def didn't explicitly bind. Keeps LivaNova-style presets
        # (which only pin Center+Corr) working with the user-edited
        # _LN-suffixed sliders.
        fallback_map = _PRIMITIVE_NS_FALLBACK_MAP.get(ptype, {})
        for prim_key, ns_key in fallback_map.items():
            if prim_key in local_p:
                continue
            if ns_key in ns:
                local_p[prim_key] = float(ns[ns_key])
        try:
            meshes = renderer(local_p)
        except Exception as ex:
            print(
                f"[cuff_designer] primitive {ptype} for instance "
                f"{inst.get('label', '?')!r} failed: {ex}",
                flush=True,
            )
            continue
        for sublabel, mesh, role in meshes:
            out.append((
                inst.get("label", ptype),
                sublabel, mesh, role,
            ))
    return out
