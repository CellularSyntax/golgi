# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cole-Cole evaluator watchers (Conductivities dialog)."""
from __future__ import annotations

from typing import Callable

from golgi.figures.cole_cole import _build_cole_cole_figure


def register(
    state,
    *,
    cole_cole_sigma: Callable,
    cole_cole_presets: dict,
) -> None:
    """Wire the Cole-Cole evaluator dialog's watchers:
      _on_cc_params       — recompute σ + live plot
      _on_cc_preset       — load preset values
      _on_cc_dialog_open  — render the σ(f) plot on dialog open
    """
    _CC_PARAM_KEYS = (
        "cc_freq_hz", "cc_eps_inf", "cc_sigma_ionic",
        "cc_d1_de", "cc_d1_tau", "cc_d1_alpha",
        "cc_d2_de", "cc_d2_tau", "cc_d2_alpha",
        "cc_d3_de", "cc_d3_tau", "cc_d3_alpha",
        "cc_d4_de", "cc_d4_tau", "cc_d4_alpha",
    )

    def _cc_current_dispersions():
        return [
            (float(state.cc_d1_de), float(state.cc_d1_tau),
             float(state.cc_d1_alpha)),
            (float(state.cc_d2_de), float(state.cc_d2_tau),
             float(state.cc_d2_alpha)),
            (float(state.cc_d3_de), float(state.cc_d3_tau),
             float(state.cc_d3_alpha)),
            (float(state.cc_d4_de), float(state.cc_d4_tau),
             float(state.cc_d4_alpha)),
        ]

    @state.change(*_CC_PARAM_KEYS)
    def _on_cc_params(**_kwargs):
        """Recompute σ(f) live + re-render the σ(f) plot whenever
        any Cole-Cole param changes. Plot render is moderately
        expensive (~50 ms with matplotlib) so we only do it
        while the dialog is open."""
        try:
            disps = _cc_current_dispersions()
            sigma = cole_cole_sigma(
                float(state.cc_freq_hz),
                float(state.cc_eps_inf),
                float(state.cc_sigma_ionic),
                disps,
            )
        except Exception:
            sigma = 0.0
            disps = []
        state.cc_sigma_result = float(sigma)
        # Pick a format that reads cleanly across the σ range
        # we expect (1e-6 to 10 S/m).
        if sigma == 0.0:
            s = "0 S/m"
        elif abs(sigma) < 1e-4 or abs(sigma) > 100:
            s = f"{sigma:.4g} S/m"
        else:
            s = f"{sigma:.4f} S/m"
        state.cc_sigma_result_str = s
        if state.show_cole_cole_dialog and disps:
            try:
                state.cc_plot_figure = _build_cole_cole_figure(
                    float(state.cc_eps_inf),
                    float(state.cc_sigma_ionic),
                    disps,
                    float(state.cc_freq_hz),
                    sigma_fn=cole_cole_sigma,
                )
            except Exception as ex:
                print(
                    f"[cole-cole] plot build failed: {ex}",
                    flush=True,
                )

    @state.change("cc_preset")
    def _on_cc_preset(**_kwargs):
        """Fill all Cole-Cole inputs from a tissue preset. New
        layout: cfg["dispersions"] is a list of up to 4
        (Δε, τ_s, α) tuples. We pad to 4 with zero-Δε rows so
        the UI rows always have something bound."""
        key = str(state.cc_preset)
        cfg = cole_cole_presets.get(key)
        if cfg is None:
            return
        disps = list(cfg.get("dispersions", []))
        while len(disps) < 4:
            disps.append((0.0, 1.0e-3, 0.0))
        with state:
            state.cc_eps_inf = float(cfg["eps_inf"])
            state.cc_sigma_ionic = float(cfg["sigma_ionic"])
            for _i, _d in enumerate(disps[:4], start=1):
                state[f"cc_d{_i}_de"] = float(_d[0])
                state[f"cc_d{_i}_tau"] = float(_d[1])
                state[f"cc_d{_i}_alpha"] = float(_d[2])

    @state.change("show_cole_cole_dialog")
    def _on_cc_dialog_open(**_kwargs):
        """Render the σ(f) plot when the dialog opens so the
        first view is populated. Subsequent param edits re-render
        via _on_cc_params."""
        if not state.show_cole_cole_dialog:
            return
        try:
            state.cc_plot_figure = _build_cole_cole_figure(
                float(state.cc_eps_inf),
                float(state.cc_sigma_ionic),
                _cc_current_dispersions(),
                float(state.cc_freq_hz),
                sigma_fn=cole_cole_sigma,
            )
        except Exception as ex:
            print(
                f"[cole-cole] initial plot failed: {ex}",
                flush=True,
            )
