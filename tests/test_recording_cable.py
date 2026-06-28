# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""R1.3 — unit tests for the cable-equation I_m + cNAP forward
sum.

The tests don't require FEniCSx; they exercise the pure-numpy
math in `golgi/pipeline/recording.py` against analytic / hand-
checked cases. Run with:

    pytest tests/test_recording_cable.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from golgi.pipeline.recording import (
    compute_cnap_population,
    compute_cnap_single,
    im_from_vm,
)


def test_im_zero_for_uniform_vm():
    """Constant V_m everywhere → zero second difference → zero
    transmembrane current. Smoke test that the formula doesn't
    invent current out of nothing."""
    n_nodes, n_t = 21, 50
    vm = np.full((n_nodes, n_t), -75.0)         # mV (rest)
    s_um = np.linspace(0, 20_000, n_nodes)       # 1 mm internodes
    im = im_from_vm(vm, s_um, diameter_um=10.0)
    assert np.allclose(im, 0.0)


def test_im_quadratic_vm_constant_second_diff():
    """V_m(s) = α·s² → ∂²V/∂s² = 2α (uniform). The discrete
    second difference at any interior node returns the same
    value; multiply by coef → I_m is uniform on the interior."""
    n_nodes = 21
    n_t = 1
    s_um = np.linspace(0.0, 20_000.0, n_nodes)   # uniform spacing
    s_m = s_um * 1e-6
    alpha_per_m2 = 1e6      # V/m² — gives ∂²V/∂s² = 2e6 V/m²
    # V_m in mV.
    vm = (alpha_per_m2 * (s_m ** 2)).reshape(-1, 1) * 1e3
    d_um = 10.0
    rho_a = 0.7
    im = im_from_vm(vm, s_um, diameter_um=d_um, rho_a_ohm_m=rho_a)
    dx_m = float(np.median(np.diff(s_m)))
    d_m = d_um * 1e-6
    coef = (np.pi * d_m * d_m) / (4.0 * rho_a * dx_m)
    # Analytic: I_m,interior = coef * (V_{n+1} - 2 V_n + V_{n-1})
    # where the bracket = 2 α dx² (in Volts; alpha_per_m2 * dx²).
    expected_node_A = coef * 2.0 * alpha_per_m2 * dx_m * dx_m
    assert im[0, 0] == 0.0
    assert im[-1, 0] == 0.0
    assert np.allclose(im[1:-1, 0], expected_node_A, rtol=1e-10)


def test_im_total_current_conserved():
    """Σ_n I_m(n) = 0 when the discrete cable's two ends are
    insulated (V_0 = V_1 AND V_{N-1} = V_{N-2}) — KCL by
    construction. Algebraically the second-difference sum
    telescopes to (V_0 − V_1) + (V_{N-1} − V_{N-2}); both
    bracketed terms are zero, so the result is exact to machine
    precision."""
    rng = np.random.default_rng(42)
    n_nodes, n_t = 17, 5
    s_um = np.linspace(0, 16_000, n_nodes)
    vm = rng.standard_normal((n_nodes, n_t)) * 50.0
    # Both end pairs equal → insulating BC at both ends.
    vm[0, :] = vm[1, :]
    vm[-1, :] = vm[-2, :]
    im = im_from_vm(vm, s_um, diameter_um=10.0)
    # Sum over nodes vanishes to floating-point precision.
    assert np.allclose(im.sum(axis=0), 0.0, atol=1e-18)


def test_cnap_single_zero_for_uniform_lead_field():
    """If V_e^R+ = V_e^R−, the differential lead field is zero
    everywhere → φ(t) ≡ 0 regardless of V_m. Tests that the
    montage subtraction wiring is correct."""
    n_nodes, n_t, n_poly = 21, 50, 21
    node_s_um = np.linspace(0, 20_000, n_nodes)
    poly = np.column_stack([
        np.zeros(n_poly), np.zeros(n_poly),
        np.linspace(0, 20_000e-6, n_poly),
    ])
    vm = np.zeros((n_nodes, n_t))
    # Inject one AP at the middle node, t=10.
    vm[n_nodes // 2, 10] = 80.0
    plus = np.linspace(1.0, 2.0, n_poly)         # arbitrary, non-flat
    minus = np.linspace(1.0, 2.0, n_poly)        # IDENTICAL to plus
    out = compute_cnap_single(
        vm_mV=vm, t_ms=np.arange(n_t) * 0.01,
        node_s_um=node_s_um, diameter_um=10.0,
        fiber_poly_xyz_m=poly,
        Ve_rec_plus_poly_V=plus,
        Ve_rec_minus_poly_V=minus,
    )
    assert np.allclose(out["phi_V"], 0.0)


def test_cnap_single_signs_with_canonical_tripole():
    """Single-node activation (a "monopole" in I_m) at the middle
    node, with a step-shaped lead field +1 on the left and -1 on
    the right of the activation point. The integral picks up
    only the activation node's I_m × (lead_left − lead_right) →
    phi sign equals -sign(I_m at the firing node) when
    plus_at_node = +1 and minus_at_node = -1.

    More importantly: phi must be NON-ZERO and finite. This is
    the "differential lead field with non-trivial wiring works"
    sanity check."""
    n_nodes, n_t = 21, 10
    n_poly = 41
    node_s_um = np.linspace(0, 20_000, n_nodes)
    poly_z_um = np.linspace(0, 20_000, n_poly)
    poly_xyz = np.column_stack([
        np.zeros(n_poly), np.zeros(n_poly), poly_z_um * 1e-6,
    ])
    mid_node = n_nodes // 2
    vm = np.full((n_nodes, n_t), -75.0)
    vm[mid_node - 1, 5] = -75.0      # neighbours at rest
    vm[mid_node, 5] = +25.0          # firing
    vm[mid_node + 1, 5] = -75.0
    # Lead field: step at the firing node's arc-length, +1 below,
    # -1 above.
    fire_s_um = node_s_um[mid_node]
    plus = np.where(poly_z_um < fire_s_um, 1.0, 0.0)
    minus = np.where(poly_z_um < fire_s_um, 0.0, 1.0)
    out = compute_cnap_single(
        vm_mV=vm, t_ms=np.arange(n_t) * 0.01,
        node_s_um=node_s_um, diameter_um=10.0,
        fiber_poly_xyz_m=poly_xyz,
        Ve_rec_plus_poly_V=plus,
        Ve_rec_minus_poly_V=minus,
    )
    phi = out["phi_V"]
    assert np.isfinite(phi).all()
    # Only timestep 5 has any AP activity.
    assert np.allclose(phi[:5], 0.0)
    assert np.allclose(phi[6:], 0.0)
    assert phi[5] != 0.0


def test_cnap_population_sums_and_decomposes():
    """Two fibers → population trace is the sum; the
    decomposition splits the sum correctly by type label."""
    n_t = 10
    f1 = np.full(n_t, 0.5)
    f2 = np.full(n_t, -0.2)
    out = compute_cnap_population(
        per_fiber_phi_V={0: f1, 1: f2},
        per_fiber_type={0: "A", 1: "B"},
    )
    assert np.allclose(out["phi_total_V"], 0.3)
    assert np.allclose(out["phi_by_type"]["A"], 0.5)
    assert np.allclose(out["phi_by_type"]["B"], -0.2)


def test_cnap_single_rejects_bad_shapes():
    """Length-mismatch between vm and node_s_um should raise."""
    with pytest.raises(ValueError):
        im_from_vm(
            vm_mV=np.zeros((10, 5)),
            node_s_um=np.zeros(11),         # one off
            diameter_um=10.0,
        )
