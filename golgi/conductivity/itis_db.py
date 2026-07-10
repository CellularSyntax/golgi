# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""IT'IS Material Database (V4.2) loader + derived Cole-Cole presets.

Source: `<repo>/resources/tissue_db/IT'IS_Material_database_V4.2.db`,
a free SQLite database you download from IT'IS — it is *not*
redistributed with golgi (see resources/tissue_db/README.md).
941 of the 1300 materials carry
the `gabriel_parameters` vector that this module unpacks into
4-term Cole-Cole dispersions consumable by
`golgi.conductivity.cole_cole.cole_cole_sigma`.

The IT'IS DB is the authoritative dielectric reference; it adds a
Nerve entry that Gabriel 1996 lacks (see project memory).

Module attributes:
- `ITIS_DB_PATH`        — default path to the shipped SQLite DB.
- `ITIS_COLE_COLE`      — dict[tissue_name → {eps_inf, sigma_ionic,
                          dispersions: [(Δε, τ_s, α), ×4]}].
- `COLE_COLE_PRESETS`   — {"Custom": …, "IT'IS · <tissue>": …}.
- `COLE_COLE_PRESET_ITEMS` — Vue-ready VSelect items list.
- `ITIS_CURATED_30`     — hand-picked subset (filtered to those
                          actually present in the loaded DB).
- `ITIS_PRESET_FREQS`   — frequencies surfaced as σ presets per IT'IS
                          tissue (DC-ish to upper-bound of neurostim).

The DB load is *eager* the first time any of the above attributes is
referenced from outside this module. If the .db file is missing, the
load degrades to an empty dict (printing a one-time hint pointing to
the download) and the Cole-Cole dialog still works with the Custom
preset only.
"""
from __future__ import annotations

import struct
from pathlib import Path

from .cole_cole import cole_cole_sigma

# ---------------------------------------------------------------------------
# Default DB path resolution
# ---------------------------------------------------------------------------
# golgi/conductivity/itis_db.py → parents[2] = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
ITIS_DB_PATH: Path = (
    _REPO_ROOT / "resources" / "tissue_db"
    / "IT'IS_Material_database_V4.2.db"
)


def load_itis_cole_cole_db(
    db_path: Path = ITIS_DB_PATH,
) -> dict[str, dict]:
    """Read every material in the IT'IS V4.2 SQLite DB that carries
    a `gabriel_parameters` vector. Returns a dict keyed by tissue
    name → {eps_inf, sigma_ionic, dispersions: [(Δε, τ_s, α), ×4]}.

    Vector layout (14 doubles, verified against published Gabriel
    1996 nerve + muscle values):
        [ ε∞,
          Δε₁, τ₁(ps),  α₁,
          Δε₂, τ₂(ns),  α₂,
          Δε₃, τ₃(μs),  α₃,
          Δε₄, τ₄(ms),  α₄,
          σ_ionic ]
    τ units are per-dispersion (ps/ns/μs/ms) and get normalised to
    SI seconds here. Empty dict if the DB isn't on disk."""
    out: dict[str, dict] = {}
    if not db_path.exists():
        print(
            f"[golgi] IT'IS tissue database not found at {db_path} — using the "
            "Custom Cole-Cole preset only. Download the free IT'IS material "
            "database from https://itis.swiss/virtual-population/tissue-properties/ "
            "and place IT'IS_Material_database_V4.2.db in resources/tissue_db/ to "
            "enable the built-in tissue presets (see resources/tissue_db/README.md).",
            flush=True,
        )
        return out
    # Per-dispersion τ unit factors → seconds. Order matches the
    # vector layout (D1, D2, D3, D4).
    tau_factors = (1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3)
    try:
        import sqlite3
        with sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True,
        ) as db:
            cur = db.cursor()
            rows = cur.execute("""
                SELECT m.name, v.vals
                FROM vectors v
                JOIN materials m ON v.mat_id = m.mat_id
                JOIN properties p ON v.prop_id = p.prop_id
                WHERE p.variable = 'gabriel_parameters'
            """).fetchall()
            for name, blob in rows:
                if blob is None or len(blob) != 14 * 8:
                    continue
                vals = struct.unpack("<14d", blob)
                eps_inf = vals[0]
                disps = []
                for i in range(4):
                    de = vals[1 + 3 * i]
                    tau_raw = vals[2 + 3 * i]
                    alpha = vals[3 + 3 * i]
                    tau_s = tau_raw * tau_factors[i]
                    disps.append((float(de), float(tau_s),
                                   float(alpha)))
                sigma_ionic = vals[13]
                out[str(name)] = {
                    "eps_inf": float(eps_inf),
                    "sigma_ionic": float(sigma_ionic),
                    "dispersions": disps,
                }
    except Exception as ex:                              # noqa: BLE001
        print(f"[golgi] IT'IS DB load failed: {ex}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Eager load at module import — matches pre-W1.1 semantics.
# ---------------------------------------------------------------------------
ITIS_COLE_COLE: dict[str, dict] = load_itis_cole_cole_db()


# ---------------------------------------------------------------------------
# Derived dicts surfaced to the Cole-Cole dialog + per-domain σ
# preset dropdowns.
# ---------------------------------------------------------------------------
COLE_COLE_PRESETS: dict[str, dict] = {
    "Custom": {
        "eps_inf": 4.0,
        "sigma_ionic": 0.20,
        "dispersions": [
            (50.0, 7.234e-12, 0.10),
            (7000.0, 353.678e-9, 0.10),
            (1.2e6, 318.310e-6, 0.10),
            (0.0, 1.0e-3, 0.00),     # 4th left disabled in Custom
        ],
    },
}
for _name in sorted(ITIS_COLE_COLE.keys()):
    COLE_COLE_PRESETS[f"IT'IS · {_name}"] = ITIS_COLE_COLE[_name]

# Vue-ready items list for the dialog's preset dropdown — all 941
# IT'IS tissues plus "Custom".
COLE_COLE_PRESET_ITEMS = [
    {"title": k, "value": k} for k in COLE_COLE_PRESETS
]


# ---------------------------------------------------------------------------
# Curated 30-tissue subset surfaced as σ presets in the per-domain
# Conductivities-drawer dropdowns. Hand-picked for VN / nerve-cuff
# work: nerve, muscle, fat, fluids, electrode-adjacent tissues. The
# full 941 stay available in the Cole-Cole dialog.
# ---------------------------------------------------------------------------
_ITIS_CURATED_30_FULL = [
    # Nervous-system tissues (endoneurium / nerve domain)
    "Nerve", "Spinal Cord", "Spinal Cord (Grey Matter)",
    "Spinal Cord (White Matter)",
    "Brain (Grey Matter)", "Brain (White Matter)",
    "Cerebellum", "Cerebrospinal Fluid", "Dura",
    # Muscle / muscular tissues (muscle domain)
    "Muscle", "Heart Muscle", "Tongue", "Esophagus",
    "Stomach (Wall)", "Trachea",
    # Connective / electrode-adjacent (epi domain)
    "Connective Tissue", "Tendon (Ligaments)", "Cartilage",
    "Skin (Dry)", "Skin (Wet)",
    # Fat / fatty tissues
    "SAT (Subcutaneous Fat)", "Fat",
    # Fluids (saline domain)
    "Blood", "Bile",
    # Bone (saturation electrode region)
    "Bone (Cortical)", "Bone (Cancellous)",
    # Visceral tissues that may surround the nerve cuff
    "Larynx", "Thyroid Gland", "Lung (Inflated)",
    "Vagus Nerve",
]
# Filter to what actually loaded — silently drops anything missing
# from the shipped DB without breaking the σ-preset dropdowns.
ITIS_CURATED_30 = [
    t for t in _ITIS_CURATED_30_FULL if t in ITIS_COLE_COLE
]


def itis_sigma_at(tissue: str, freq_hz: float) -> float | None:
    """Compute σ(f) for an IT'IS tissue from its stored Cole-Cole
    parameters. Returns None if the tissue isn't in the loaded DB."""
    cfg = ITIS_COLE_COLE.get(tissue)
    if cfg is None:
        return None
    return cole_cole_sigma(
        freq_hz,
        cfg["eps_inf"],
        cfg["sigma_ionic"],
        cfg["dispersions"],
    )


def fmt_freq_label(f_hz: float) -> str:
    """Human-readable frequency tag for σ preset labels:
    '100 Hz', '1 kHz', '10 kHz', '100 kHz', '1 MHz'."""
    if f_hz >= 1e6:
        return f"{f_hz/1e6:g} MHz"
    if f_hz >= 1e3:
        return f"{f_hz/1e3:g} kHz"
    return f"{f_hz:g} Hz"


# Standard frequencies surfaced as σ presets per IT'IS tissue.
# Spans DC-ish to upper-bound of neurostim — captures the α / β
# dispersion regime.
ITIS_PRESET_FREQS = (100.0, 1.0e3, 10.0e3, 100.0e3, 1.0e6)


def itis_preset_rows(
    tissue: str,
) -> list[tuple[str, float, str]]:
    """Build the (label, σ, source) tuples for one IT'IS tissue at
    the standard preset frequencies. Empty list if not in DB."""
    if tissue not in ITIS_COLE_COLE:
        return []
    rows = []
    for f in ITIS_PRESET_FREQS:
        sigma = itis_sigma_at(tissue, f)
        if sigma is None or sigma <= 0:
            continue
        rows.append((
            f"IT'IS {tissue} @ {fmt_freq_label(f)}",
            float(sigma),
            "IT'IS DB v4.2",
        ))
    return rows
