# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Electrode-contact patch geometry on a cylindrical cuff.

Extracted from `golgi/app.py` in step W1.4 of FEATURES.md. Direct
port of nerve_studio.py § 3 patch dispatcher.

Two builders, two consumers:

- `axial_patch_polydata(z_c, dz, phi_c, dphi, R, n_phi)` and
  `helical_patch_polydata(z_start, z_end, phi0, pitch, dphi, R, ...)`
  produce pv.PolyData surfaces on the cylinder r = R for the
  rendered cuff (scene-tier consumer).

- `build_electrode_patches_dicts(L_cuff_m, R_ci_m, kind, cfg)`
  produces a JSON-friendly list of dicts in the format
  `electrode_config.json` expects, consumed by the FEM driver
  (`golgi/pipeline/fem.py`) which hands them to
  `compute/solve_nerve.py` as Neumann current patches.

- `build_electrode_patches(L_cuff_m, R_ci_m, kind, cfg)` is the
  PolyData equivalent for the viewport.

Both dispatchers handle four electrode kinds:
  - "bipolar ring-pair"
  - "tripolar (anode-cathode-anode)"
  - "ring-array (NxM)"
  - "helical (Livanova-style)"

Unknown `kind` returns an empty list — cuff shell renders alone.
"""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv


def axial_patch_polydata(z_c: float, dz: float,
                          phi_c: float, dphi: float,
                          R: float, n_phi: int = 36) -> pv.PolyData:
    """A rectangular band on the cylinder r = R, centred on z = z_c,
    phi = phi_c. Returns a thin shell (just the inner surface)."""
    phi_lo, phi_hi = phi_c - dphi / 2, phi_c + dphi / 2
    phis = np.linspace(phi_lo, phi_hi, n_phi)
    zs = np.linspace(z_c - dz / 2, z_c + dz / 2, 8)
    P, Z = np.meshgrid(phis, zs, indexing="xy")
    X = R * np.cos(P); Y = R * np.sin(P)
    sg = pv.StructuredGrid(X, Y, Z)
    return sg.extract_surface(algorithm="dataset_surface")


def cylinder_band_polydata(x_c: float, y_c: float, z_c: float,
                             dz: float, R_wire: float,
                             n_phi: int = 24,
                             n_z: int = 4) -> pv.PolyData:
    """A short cylindrical band (open shell) of radius `R_wire`,
    centred at (x_c, y_c, z_c) with full axial height `dz` along
    local +z. Used to render a LIFE contact band on the wire's
    own surface.

    The same geometry shape is used by both the viewport renderer
    (one polydata per contact band) and conceptually drives the
    FEM-side `life_band` patch dict — the solver applies the
    Neumann current to the nearest endo-mesh facets to this
    band's surface."""
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=True)
    zs = np.linspace(z_c - dz / 2.0, z_c + dz / 2.0, n_z)
    P, Z = np.meshgrid(phis, zs, indexing="xy")
    X = x_c + R_wire * np.cos(P)
    Y = y_c + R_wire * np.sin(P)
    sg = pv.StructuredGrid(X, Y, Z)
    return sg.extract_surface(algorithm="dataset_surface")


def rect_patch_polydata(x_c: float, y_c: float, z_c: float,
                          dl: float, dz: float, phi: float,
                          n_l: int = 8, n_z: int = 4,
                          ) -> pv.PolyData:
    """A flat axis-aligned rectangular patch at (x_c, y_c, z_c)
    with chord-direction extent `dl` (rotated by `phi` in the xy
    plane) and axial extent `dz` along local +z. Sits on the
    front face of the TIME ribbon — i.e. the side facing the
    direction `phi` rotates the +x axis to.

    Used both for the viewport renderer (one polydata per
    contact rectangle) and the FEM `time_rect` patch dict."""
    # Local rectangle frame: chord axis = ξ (length dl),
    # axial = z (length dz). Rotate ξ by phi into world xy.
    cphi = float(np.cos(phi))
    sphi = float(np.sin(phi))
    xis = np.linspace(-dl / 2.0, dl / 2.0, n_l)
    zs = np.linspace(z_c - dz / 2.0, z_c + dz / 2.0, n_z)
    XI, Z = np.meshgrid(xis, zs, indexing="xy")
    X = x_c + XI * cphi
    Y = y_c + XI * sphi
    sg = pv.StructuredGrid(X, Y, Z)
    return sg.extract_surface(algorithm="dataset_surface")


def helical_patch_polydata(z_start: float, z_end: float,
                            phi0: float, pitch: float, dphi: float,
                            R: float,
                            n_long: int = 40,
                            n_lat: int = 8) -> pv.PolyData:
    """A helical strip on the cylinder r = R going from z_start to
    z_end, starting at angle phi0, with axial pitch (m per 2π rev)
    and angular width dphi (radians)."""
    if pitch <= 0:
        pitch = 1e-6
    s = np.linspace(0.0, 1.0, n_long)
    t = np.linspace(-0.5, 0.5, n_lat)
    S, T = np.meshgrid(s, t, indexing="ij")
    Z = z_start + S * (z_end - z_start)
    PHI_C = phi0 + 2.0 * np.pi * (Z - z_start) / pitch
    PHI = PHI_C + T * dphi
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    sg = pv.StructuredGrid(X, Y, Z)
    return sg.extract_surface(algorithm="dataset_surface")


def build_electrode_patches_dicts(L_cuff_m: float, R_ci_m: float,
                                     kind: str,
                                     cfg: dict) -> list[dict]:
    """Per-electrode-type patches as DICTs in the format
    solve_nerve.py expects in `electrode_config.json`. Mirrors
    nerve_studio.py § 3 dispatcher exactly.

    Axial patches:
      {id, type='axial',   role, z, dz, phi, dphi}
    Helical patches:
      {id, type='helical', role, z_start, z_end, phi0, pitch, dphi}
    """
    # M1 vocab: stimulating contact = "cathode", return =
    # "anode". solve_nerve.py aliases "active" ↔ "cathode" and
    # "ground" ↔ "anode" so legacy configs still solve, but
    # new writes use the explicit vocab.
    if kind == "bipolar ring-pair":
        sep = float(cfg["bipolar_axial_sep_mm"]) * 1e-3
        w = float(cfg["bipolar_ring_width_mm"]) * 1e-3
        return [
            {"id": 0, "type": "axial", "role": "cathode",
             "z": -sep / 2, "dz": w,
             "phi": 0.0, "dphi": 2 * math.pi},
            {"id": 1, "type": "axial", "role": "anode",
             "z": +sep / 2, "dz": w,
             "phi": 0.0, "dphi": 2 * math.pi},
        ]
    if kind == "tripolar (anode-cathode-anode)":
        sep = float(cfg["tripolar_axial_sep_mm"]) * 1e-3
        w = float(cfg["tripolar_ring_width_mm"]) * 1e-3
        return [
            {"id": 0, "type": "axial", "role": "anode",
             "z": -sep, "dz": w,
             "phi": 0.0, "dphi": 2 * math.pi},
            {"id": 1, "type": "axial", "role": "cathode",
             "z": 0.0, "dz": w,
             "phi": 0.0, "dphi": 2 * math.pi},
            {"id": 2, "type": "axial", "role": "anode",
             "z": +sep, "dz": w,
             "phi": 0.0, "dphi": 2 * math.pi},
        ]
    if kind == "ring-array (NxM)":
        nrows = int(cfg["array_n_rows"])
        ncols = int(cfg["array_n_cols"])
        w = float(cfg["array_contact_w_mm"]) * 1e-3
        dphi = math.radians(float(cfg["array_contact_phi_deg"]))
        row_sep = float(cfg["array_row_sep_mm"]) * 1e-3
        zs = (np.arange(nrows) - (nrows - 1) / 2.0) * row_sep
        phis = np.linspace(0.0, 2 * np.pi, ncols, endpoint=False)
        out: list[dict] = []
        pid = 0
        for zi, z in enumerate(zs):
            for pi, phi in enumerate(phis):
                role = ("cathode" if (zi + pi) % 2 == 0
                         else "anode")
                out.append({
                    "id": pid, "type": "axial", "role": role,
                    "z": float(z), "dz": w,
                    "phi": float(phi), "dphi": dphi,
                })
                pid += 1
        return out
    if kind == "helical (Livanova-style)":
        nbands = int(cfg["helix_n_bands"])
        pitch = float(cfg["helix_pitch_mm"]) * 1e-3
        dphi = math.radians(float(cfg["helix_dphi_deg"]))
        # Band-to-band axial spacing. FREE parameter `helix_band_sep_mm` when
        # given, so the contact separation is decoupled from the cuff length —
        # a real LivaNova has ~8 mm-separated contacts in a ~10 mm cuff. Falls
        # back to the legacy L_cuff-derived spacing (0.3*L_cuff for 2 bands).
        _sep = cfg.get("helix_band_sep_mm")
        step = (float(_sep) * 1e-3 if _sep not in (None, 0, 0.0)
                else L_cuff_m * 0.6 / max(nbands, 1))
        z0s = (np.arange(nbands) - (nbands - 1) / 2.0) * step
        out = []
        for i, z0 in enumerate(z0s):
            role = "cathode" if i % 2 == 0 else "anode"
            out.append({
                "id": i, "type": "helical", "role": role,
                "z_start": float(z0 - pitch * 0.25),
                "z_end":   float(z0 + pitch * 0.25),
                "phi0": float(i * math.pi / 4),
                "pitch": pitch, "dphi": dphi,
            })
        return out
    if kind == "LIFE (longitudinal intrafascicular)":
        # N axial contacts per wire × M parallel wires laid out
        # along a chord at angle phi in the cuff transverse
        # plane. Each contact is a short cylindrical band on
        # the wire's surface — patch dict carries the wire's
        # (x, y), the band's z + axial extent, and the wire
        # radius so the FEM-side nearest-facet lookup can
        # apply the Neumann BC at the right interior point.
        nrows = int(cfg["life_n_rows"])
        ncols = int(cfg["life_n_cols"])
        row_sep = float(cfg["life_row_sep_mm"]) * 1e-3
        col_sep = float(cfg["life_col_sep_mm"]) * 1e-3
        dz = float(cfg["life_contact_length_mm"]) * 1e-3
        R_wire = float(cfg["life_diameter_um"]) * 0.5e-6
        phi = math.radians(float(cfg["life_chord_phi_deg"]))
        cx = float(cfg["life_x_mm"]) * 1e-3
        cy = float(cfg["life_y_mm"]) * 1e-3
        zs = (np.arange(nrows) - (nrows - 1) / 2.0) * row_sep
        # Wires sit on a transverse chord centred on
        # (cx, cy), spaced by col_sep along the chord.
        xis = (np.arange(ncols) - (ncols - 1) / 2.0) * col_sep
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        out = []
        pid = 0
        for pi, xi in enumerate(xis):
            wx = cx + xi * cphi
            wy = cy + xi * sphi
            for zi, z in enumerate(zs):
                role = ("cathode" if (zi + pi) % 2 == 0
                         else "anode")
                out.append({
                    "id": pid, "type": "life_band",
                    "role": role,
                    "x": float(wx), "y": float(wy),
                    "z": float(z), "dz": dz,
                    "R_wire": R_wire,
                })
                pid += 1
        return out
    if kind == "TIME (transverse intrafascicular)":
        # N axial rows × M transverse columns of contacts on
        # the ribbon's front face. Ribbon midpoint at
        # (time_x, time_y, 0); chord axis = +x rotated by
        # `time_chord_phi_deg` (so phi=0 places the ribbon
        # along +x).
        nrows = int(cfg["time_n_rows"])
        ncols = int(cfg["time_n_cols"])
        row_sep = float(cfg["time_row_sep_mm"]) * 1e-3
        col_sep = float(cfg["time_col_sep_mm"]) * 1e-3
        dz = float(cfg["time_contact_w_mm"]) * 1e-3
        phi = math.radians(float(cfg["time_chord_phi_deg"]))
        cx = float(cfg["time_x_mm"]) * 1e-3
        cy = float(cfg["time_y_mm"]) * 1e-3
        zs = (np.arange(nrows) - (nrows - 1) / 2.0) * row_sep
        xis = (np.arange(ncols) - (ncols - 1) / 2.0) * col_sep
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        out = []
        pid = 0
        for zi, z in enumerate(zs):
            for pi, xi in enumerate(xis):
                wx = cx + xi * cphi
                wy = cy + xi * sphi
                role = ("cathode" if (zi + pi) % 2 == 0
                         else "anode")
                out.append({
                    "id": pid, "type": "time_rect",
                    "role": role,
                    "x": float(wx), "y": float(wy),
                    "z": float(z),
                    # `dl` = chord-direction extent of the
                    # contact patch. Implied by `col_sep` for
                    # uniform packing; the FEM-side nearest-
                    # facet integration is not sensitive to
                    # the patch's exact extent.
                    "dl": float(col_sep * 0.8),
                    "dz": dz,
                    "phi": float(phi),
                })
                pid += 1
        return out
    return []


def build_electrode_patches(L_cuff_m: float, R_ci_m: float,
                              kind: str,
                              cfg: dict) -> list[pv.PolyData]:
    """Build the contact-patch list for the chosen electrode type.
    Returns surface PolyData per contact on the cylinder r = R_ci.

    Mirrors the dispatcher in nerve_studio.py §3 (line 1040-1127).
    """
    L = float(L_cuff_m)
    R = float(R_ci_m)

    if kind == "bipolar ring-pair":
        sep = float(cfg["bipolar_axial_sep_mm"]) * 1e-3
        w = float(cfg["bipolar_ring_width_mm"]) * 1e-3
        return [
            axial_patch_polydata(-sep/2, w, 0.0, 2*math.pi, R, n_phi=72),
            axial_patch_polydata(+sep/2, w, 0.0, 2*math.pi, R, n_phi=72),
        ]

    if kind == "tripolar (anode-cathode-anode)":
        sep = float(cfg["tripolar_axial_sep_mm"]) * 1e-3
        w = float(cfg["tripolar_ring_width_mm"]) * 1e-3
        return [
            axial_patch_polydata(-sep, w, 0.0, 2*math.pi, R, n_phi=72),
            axial_patch_polydata( 0.0, w, 0.0, 2*math.pi, R, n_phi=72),
            axial_patch_polydata(+sep, w, 0.0, 2*math.pi, R, n_phi=72),
        ]

    if kind == "ring-array (NxM)":
        nrows = int(cfg["array_n_rows"])
        ncols = int(cfg["array_n_cols"])
        w = float(cfg["array_contact_w_mm"]) * 1e-3
        dphi = math.radians(float(cfg["array_contact_phi_deg"]))
        row_sep = float(cfg["array_row_sep_mm"]) * 1e-3
        # Rows centred about the cuff midplane, spaced by user-set
        # `array_row_sep_mm`. For nrows=1 the single row sits at z=0.
        zs = (np.arange(nrows) - (nrows - 1) / 2.0) * row_sep
        phis = np.linspace(0.0, 2*np.pi, ncols, endpoint=False)
        out: list[pv.PolyData] = []
        for z in zs:
            for phi in phis:
                out.append(axial_patch_polydata(
                    float(z), w, float(phi), dphi, R, n_phi=24,
                ))
        return out

    if kind == "helical (Livanova-style)":
        nbands = int(cfg["helix_n_bands"])
        pitch = float(cfg["helix_pitch_mm"]) * 1e-3
        dphi = math.radians(float(cfg["helix_dphi_deg"]))
        z0s = (np.arange(nbands) - (nbands - 1) / 2.0) * (
            L * 0.6 / max(nbands, 1)
        )
        out = []
        for i, z0 in enumerate(z0s):
            out.append(helical_patch_polydata(
                float(z0 - pitch * 0.25),
                float(z0 + pitch * 0.25),
                phi0=float(i * math.pi / 4),
                pitch=pitch, dphi=dphi, R=R,
            ))
        return out

    if kind == "LIFE (longitudinal intrafascicular)":
        nrows = int(cfg["life_n_rows"])
        ncols = int(cfg["life_n_cols"])
        row_sep = float(cfg["life_row_sep_mm"]) * 1e-3
        col_sep = float(cfg["life_col_sep_mm"]) * 1e-3
        dz = float(cfg["life_contact_length_mm"]) * 1e-3
        R_wire = float(cfg["life_diameter_um"]) * 0.5e-6
        phi = math.radians(float(cfg["life_chord_phi_deg"]))
        cx = float(cfg["life_x_mm"]) * 1e-3
        cy = float(cfg["life_y_mm"]) * 1e-3
        zs = (np.arange(nrows) - (nrows - 1) / 2.0) * row_sep
        xis = (np.arange(ncols) - (ncols - 1) / 2.0) * col_sep
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        out: list[pv.PolyData] = []
        for xi in xis:
            wx = cx + xi * cphi
            wy = cy + xi * sphi
            for z in zs:
                out.append(cylinder_band_polydata(
                    x_c=float(wx), y_c=float(wy),
                    z_c=float(z), dz=dz, R_wire=R_wire,
                    n_phi=16,
                ))
        return out

    if kind == "TIME (transverse intrafascicular)":
        nrows = int(cfg["time_n_rows"])
        ncols = int(cfg["time_n_cols"])
        row_sep = float(cfg["time_row_sep_mm"]) * 1e-3
        col_sep = float(cfg["time_col_sep_mm"]) * 1e-3
        dz = float(cfg["time_contact_w_mm"]) * 1e-3
        phi = math.radians(float(cfg["time_chord_phi_deg"]))
        cx = float(cfg["time_x_mm"]) * 1e-3
        cy = float(cfg["time_y_mm"]) * 1e-3
        zs = (np.arange(nrows) - (nrows - 1) / 2.0) * row_sep
        xis = (np.arange(ncols) - (ncols - 1) / 2.0) * col_sep
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        # Each contact's chord-direction extent is 80 % of
        # the column spacing so adjacent contacts don't touch.
        dl = col_sep * 0.8
        out: list[pv.PolyData] = []
        for z in zs:
            for xi in xis:
                wx = cx + xi * cphi
                wy = cy + xi * sphi
                out.append(rect_patch_polydata(
                    x_c=float(wx), y_c=float(wy),
                    z_c=float(z), dl=float(dl),
                    dz=float(dz), phi=float(phi),
                ))
        return out

    # Unknown kind → empty list (renders cuff shell only).
    return []


def build_intrafascicular_body_polydata(
    L_cuff_m: float,
    kind: str,
    cfg: dict,
) -> "pv.PolyData | None":
    """Build the *insulator body* polydata (the wire shafts for
    LIFE, the flat ribbon for TIME) so the viewport renders the
    intrafascicular electrode's structural body alongside the
    individual contact patches. Cuff-style electrode types
    (bipolar / tripolar / ring-array / helical) have their body
    rendered separately as the silicone+saline cuff shell, so
    this returns None for them.

    The body is rendered as a thin opaque shape; the contacts on
    top of it (from `build_electrode_patches`) are highlighted
    by colour so the user can see active sites vs insulation."""
    if kind == "LIFE (longitudinal intrafascicular)":
        ncols = int(cfg["life_n_cols"])
        col_sep = float(cfg["life_col_sep_mm"]) * 1e-3
        R_wire = float(cfg["life_diameter_um"]) * 0.5e-6
        phi = math.radians(float(cfg["life_chord_phi_deg"]))
        cx = float(cfg["life_x_mm"]) * 1e-3
        cy = float(cfg["life_y_mm"]) * 1e-3
        # Each wire runs the full cuff length so the user sees
        # it threading through the cuff window; the wire that
        # appears in the segmentation is the insertion stub,
        # extending the rendering past the cuff caps is just
        # visualisation (the FEM patches are inside L_cuff).
        xis = (np.arange(ncols) - (ncols - 1) / 2.0) * col_sep
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        meshes: list[pv.PolyData] = []
        for xi in xis:
            wx = cx + xi * cphi
            wy = cy + xi * sphi
            meshes.append(cylinder_band_polydata(
                x_c=float(wx), y_c=float(wy),
                z_c=0.0, dz=float(L_cuff_m),
                R_wire=R_wire, n_phi=16, n_z=8,
            ))
        if not meshes:
            return None
        if len(meshes) == 1:
            return meshes[0]
        body = meshes[0]
        for m in meshes[1:]:
            body = body.merge(m)
        return body
    if kind == "TIME (transverse intrafascicular)":
        W = float(cfg["time_ribbon_width_mm"]) * 1e-3
        phi = math.radians(float(cfg["time_chord_phi_deg"]))
        cx = float(cfg["time_x_mm"]) * 1e-3
        cy = float(cfg["time_y_mm"]) * 1e-3
        # Ribbon spans the user-set chord length, full cuff
        # length axially. Thickness is rendered as zero in the
        # viewport (single-face polydata) — the user can see
        # through it onto the fascicles inside.
        return rect_patch_polydata(
            x_c=float(cx), y_c=float(cy), z_c=0.0,
            dl=float(W), dz=float(L_cuff_m), phi=float(phi),
            n_l=16, n_z=24,
        )
    return None
