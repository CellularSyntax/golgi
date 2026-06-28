# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Perineurium morphology → electrical thin-layer parameters.

The perineurium is the connective-tissue sheath around each fascicle. It is
thin (a few µm to a few tens of µm) yet highly resistive (σ ≈ 1/1149 S/m), so
in the FEM it is represented as a *contact-impedance sheet* at the
endoneurium↔epineurium interface rather than as a meshed volume — ASCENT's
"thin-layer approximation", which avoids the µm-scale local refinement an
explicit shell would force. Its area-specific sheet resistance is

    Rs = peri_thk / σ_peri          [Ω·m²]

where the thickness is a species-specific linear function of the fascicle
diameter (Pelot et al. 2019, J Neural Eng 16(1):016007; morphology refined in
Pelot et al. 2020, Front Neurosci 14:1148):

    species   peri_thk(d_fasc) [µm]
    rat       0.01292·d + 1.367
    pig       0.02547·d + 3.440      (alias: swine / porcine)
    human     0.03702·d + 10.50

d_fasc is the equivalent-circle fascicle diameter in µm. σ_peri = 1/1149 S/m
(Weerasuriya et al. 1984, Biophys J 46:167; adopted by Pelot et al. 2019).
"""
from __future__ import annotations

import math
from typing import Optional

# species → (slope [µm of perineurium per µm of fascicle diameter],
#            intercept [µm]).  Pelot et al. 2019, Table — coefficients.
PERINEURIUM_COEFFS: dict[str, tuple[float, float]] = {
    "rat": (0.01292, 1.367),
    "pig": (0.02547, 3.440),
    "human": (0.03702, 10.50),
}

# Common common-name → canonical-key aliases so callers can pass "swine".
_SPECIES_ALIASES: dict[str, str] = {
    "swine": "pig",
    "porcine": "pig",
    "sus": "pig",
    "sus scrofa": "pig",
    "rattus": "rat",
    "homo": "human",
    "homo sapiens": "human",
}

# Perineurium bulk conductivity [S/m] (Weerasuriya 1984; Pelot 2019).
SIGMA_PERINEURIUM: float = 1.0 / 1149.0


def _canon_species(species: str) -> str:
    s = str(species).strip().lower()
    return _SPECIES_ALIASES.get(s, s)


def perineurium_thickness_um(
    species: str,
    dfasc_um: float,
    *,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
) -> float:
    """Perineurium thickness [µm] from fascicle diameter [µm].

    Parameters
    ----------
    species
        ``'rat'`` | ``'pig'`` (alias ``'swine'``/``'porcine'``) | ``'human'``
        | ``'custom'``. For ``'custom'`` you must pass ``slope`` and
        ``intercept`` (the linear coefficients in µm).
    dfasc_um
        Equivalent-circle fascicle diameter in µm
        (see :func:`fascicle_diameter_um`).
    slope, intercept
        Only used (and required) when ``species == 'custom'``.
    """
    d = float(dfasc_um)
    if d < 0.0:
        raise ValueError("dfasc_um must be non-negative")
    sp = _canon_species(species)
    if sp == "custom":
        if slope is None or intercept is None:
            raise ValueError(
                "custom species requires explicit slope and intercept (µm)"
            )
        m, b = float(slope), float(intercept)
    else:
        try:
            m, b = PERINEURIUM_COEFFS[sp]
        except KeyError:
            raise ValueError(
                f"unknown species {species!r}; choose one of "
                f"{sorted(PERINEURIUM_COEFFS)} or 'custom'"
            ) from None
    return m * d + b


def fascicle_diameter_um(
    area_um2: Optional[float] = None,
    *,
    area_m2: Optional[float] = None,
) -> float:
    """Equivalent-circle diameter [µm] from a fascicle cross-sectional area.

    Pass either ``area_um2`` or ``area_m2``. The equivalent diameter is
    ``2·sqrt(A/π)`` — the diameter of a circle with the same area, the
    convention Pelot et al. use to feed the thickness formula.
    """
    if area_m2 is not None:
        area_um2 = float(area_m2) * 1.0e12
    if area_um2 is None:
        raise ValueError("provide area_um2 or area_m2")
    if area_um2 < 0.0:
        raise ValueError("area must be non-negative")
    return 2.0 * math.sqrt(float(area_um2) / math.pi)


def perineurium_sheet_resistance(
    species: str,
    dfasc_um: float,
    *,
    sigma_peri: float = SIGMA_PERINEURIUM,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
) -> float:
    """Area-specific perineurium sheet resistance ``Rs = thk/σ_peri`` [Ω·m²].

    This is the contact-impedance parameter applied at the endo↔epi
    interface in the FEM. Multiply a local current density [A/m²] by ``Rs``
    to get the trans-perineurium voltage drop.
    """
    thk_m = (
        perineurium_thickness_um(
            species, dfasc_um, slope=slope, intercept=intercept
        )
        * 1.0e-6
    )
    return thk_m / float(sigma_peri)


def aggregate_perineurium(
    fascicle_areas_um2,
    species: str,
    *,
    sigma_peri: float = SIGMA_PERINEURIUM,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
) -> dict:
    """Per-fascicle + representative perineurium parameters for a nerve.

    Given each fascicle's cross-sectional area (µm²), returns a JSON-able
    dict with the per-fascicle equivalent diameter, thickness, and sheet
    resistance, plus an area-weighted representative thickness / Rs for the
    single-sheet contact-impedance model (all fascicles share the endo tag,
    so the FEM applies one representative Rs at the endo↔epi interface).
    """
    areas = [float(a) for a in fascicle_areas_um2 if float(a) > 0.0]
    sp = _canon_species(species)
    diam = [fascicle_diameter_um(a) for a in areas]
    thk = [
        perineurium_thickness_um(sp, d, slope=slope, intercept=intercept)
        for d in diam
    ]
    rs = [(t * 1.0e-6) / float(sigma_peri) for t in thk]
    total_a = sum(areas)
    # area-weighted representative thickness (typical perineurium across
    # the nerve's fascicles, weighted by fascicle size)
    mean_thk = (
        sum(a * t for a, t in zip(areas, thk)) / total_a
        if total_a > 0.0 else 0.0
    )
    return {
        "species": sp,
        "n_fascicles": len(areas),
        "sigma_peri_S_per_m": float(sigma_peri),
        "fascicle_area_um2": areas,
        "fascicle_diam_um": diam,
        "fascicle_peri_thk_um": thk,
        "fascicle_sheet_resistance_ohm_m2": rs,
        "mean_peri_thk_um": mean_thk,
        "peri_thk_m": mean_thk * 1.0e-6,
        "sheet_resistance_ohm_m2": (
            (mean_thk * 1.0e-6) / float(sigma_peri)
            if sigma_peri > 0.0 else None
        ),
    }
