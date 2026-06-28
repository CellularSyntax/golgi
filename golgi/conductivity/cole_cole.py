# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cole-Cole conductivity model — pure math, no I/O.

Complex permittivity:
    ε*(ω) = ε∞ + Σᵢ Δεᵢ / [1 + (jωτᵢ)^(1-αᵢ)] + σ_i / (jωε₀)

Total real conductivity:
    σ(ω) = σ_i + ωε₀ · ε''(ω)

4-term is the format the IT'IS DB stores (Gabriel 1996 γβ₁β₂α
dispersions); 3-term breast-tissue fits also work — just leave the
4th dispersion at Δε=0 to disable it. The user's prior breast-
tissue work uses 3-term as the primary fit (see project memory).
"""
from __future__ import annotations

import math

# Vacuum permittivity (F/m). Pinned here so the module has zero
# external dependency beyond stdlib `math`.
EPS_0 = 8.8541878128e-12


def cole_cole_sigma(
    f_hz: float,
    eps_inf: float,
    sigma_ionic: float,
    dispersions: list[tuple[float, float, float]],
) -> float:
    """Evaluate the Cole-Cole real conductivity σ(f) [S/m] for an
    arbitrary number of dispersions (use 3 or 4 per Gabriel). Each
    dispersion is (Δε, τ_seconds, α). Dispersions with Δε ≤ 0 or
    τ ≤ 0 are skipped so the user can disable one by zeroing it."""
    if f_hz <= 0:
        return float(sigma_ionic)
    omega = 2.0 * math.pi * f_hz
    eps_disp = complex(eps_inf, 0.0)
    for de, tau, alpha in dispersions:
        if de <= 0.0 or tau <= 0.0:
            continue
        # (jωτ)^(1-α) — Python's ** on complex uses the principal
        # branch, which is correct for the Cole-Cole exponent.
        x = (1j * omega * tau) ** (1.0 - alpha)
        eps_disp += de / (1.0 + x)
    sigma_star = sigma_ionic + 1j * omega * EPS_0 * eps_disp
    return float(sigma_star.real)


def cole_cole_3term_sigma(f_hz, eps_inf, sigma_ionic, dispersions):
    """Back-compat shim — the 3-term name is still referenced in test
    scripts / notebooks. New code should use `cole_cole_sigma`
    directly (it handles any number of dispersions)."""
    return cole_cole_sigma(f_hz, eps_inf, sigma_ionic, dispersions)
