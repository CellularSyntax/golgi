# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cole-Cole conductivity math + IT'IS Material Database loader.

Extracted from `golgi/app.py` in step W1.1. Re-exports the public
surface so callers can `from golgi.conductivity import cole_cole_sigma,
COLE_COLE_PRESETS` instead of reaching deeper.

Two sub-modules:
  - `cole_cole`: pure math, no side effects (no DB read, no I/O).
  - `itis_db`: IT'IS V4.2 SQLite loader + derived preset dicts.
    The DB read happens lazily on first access to any of the public
    dicts — that preserves the pre-W1.1 eager-at-module-import
    semantics for callers, while keeping import-time cost low for
    anyone who only wants `cole_cole_sigma` (e.g. headless scripts).
"""
from __future__ import annotations

from .cole_cole import (  # noqa: F401
    EPS_0,
    cole_cole_3term_sigma,
    cole_cole_sigma,
)
from .itis_db import (  # noqa: F401
    COLE_COLE_PRESET_ITEMS,
    COLE_COLE_PRESETS,
    ITIS_COLE_COLE,
    ITIS_CURATED_30,
    ITIS_DB_PATH,
    ITIS_PRESET_FREQS,
    fmt_freq_label,
    itis_preset_rows,
    itis_sigma_at,
    load_itis_cole_cole_db,
)
from .perineurium import (  # noqa: F401
    PERINEURIUM_COEFFS,
    SIGMA_PERINEURIUM,
    aggregate_perineurium,
    fascicle_diameter_um,
    perineurium_sheet_resistance,
    perineurium_thickness_um,
)
from .materials import (  # noqa: F401
    MATERIAL_SIGMA,
    is_anisotropic,
    sigma_longitudinal,
    sigma_transverse,
)
