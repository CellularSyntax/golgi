# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Canonical FEM material conductivities (single source of truth).

The values below are the project's adopted defaults for the peripheral-nerve
volume-conductor FEM. Anisotropic tissues carry ``(transverse, longitudinal)``
in S/m, where *longitudinal* is along the nerve / muscle-fibre axis (the
mesh +z axis for the extruded Duke nerves); isotropic materials carry a
single float.

References (as supplied):
  [1]  Weerasuriya et al. 1984, Biophys J 46:167 — perineurium AC impedance.
  [2]  Pelot et al. 2019, J Neural Eng 16:016007 — FE parameters of nerves.
  [3]  de Podesta 1996 — platinum bulk conductivity.
  [4]  Ranck & BeMent 1965, Exp Neurol 11:451 — endoneurium anisotropy.
  [5]  Pelot et al. 2019 (see [2]).
  [6]  Stolinski 1995, J Anat 186:123 — nerve sheath structure (epineurium).
  [7]  Grill & Mortimer 1994, Ann Biomed Eng 22:23 — encapsulation tissue.
  [9]  Pelot et al. 2017, J Neural Eng 14:046022 — muscle anisotropy use.
  [10] Gielen et al. 1984, Med Biol Eng Comput 22:569 — fat / muscle σ.
  [11] Geddes & Baker 1967, Med Biol Eng 5:271 — saline reference.

Anisotropic entries are stored as ``(σ_transverse, σ_longitudinal)``.
"""
from __future__ import annotations

from typing import Union

from .perineurium import SIGMA_PERINEURIUM  # noqa: F401  (re-export)

Sigma = Union[float, tuple[float, float]]

# Canonical material → σ [S/m]. Tuples are (transverse, longitudinal).
MATERIAL_SIGMA: dict[str, Sigma] = {
    "silicone": 1.0e-12,                 # [2]  cuff body (near-insulator)
    "platinum": 9.43e6,                  # [3]  electrode contact metal
    "endoneurium": (1.0 / 6.0, 1.0 / 1.75),   # [4,5] {1/6, 1/6, 1/1.75}
    "epineurium": 1.0 / 6.3,             # [6,7,8]
    "muscle": (0.086, 0.35),             # [9]  {0.086, 0.086, 0.35}
    "fat": 1.0 / 30.0,                   # [10]
    "encapsulation": 1.0 / 6.3,          # [7]  (scar / fibrotic capsule)
    "saline": 1.76,                      # [11]
    "perineurium": SIGMA_PERINEURIUM,    # [1,5]  1/1149 S/m (contact-impedance)
}


def sigma_transverse(material: str) -> float:
    """Transverse (or isotropic) σ [S/m] for a canonical material."""
    v = MATERIAL_SIGMA[material]
    return float(v[0]) if isinstance(v, tuple) else float(v)


def sigma_longitudinal(material: str) -> float:
    """Longitudinal σ [S/m]; equals the transverse value if isotropic."""
    v = MATERIAL_SIGMA[material]
    return float(v[1]) if isinstance(v, tuple) else float(v)


def is_anisotropic(material: str) -> bool:
    return isinstance(MATERIAL_SIGMA[material], tuple)
