# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cole-Cole evaluator dialog (Conductivities tab → "Evaluate σ").

Computes σ(f) from a 3-4 term Cole-Cole fit and applies it to
whichever tissue opened the dialog."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

try:                                                # pragma: no cover
    from trame.widgets import plotly as twp
except ImportError:                                 # pragma: no cover
    twp = None


def render(
    *,
    do_cole_cole_cancel: Callable,
    do_cole_cole_apply: Callable,
    export_btn: Callable | None = None,
) -> None:
    """Render the Cole-Cole evaluator dialog. `export_btn(fig_id)` —
    when supplied — drops a per-figure export button (F2.3.a) into
    the σ(f) plot tile."""
    with v3.VDialog(
        v_model=("show_cole_cole_dialog",),
        max_width=620,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Cole-Cole evaluator "
                    "— {{ cole_cole_target_label }}",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "σ(ω) = σ_ionic + ωε₀·ε''(ω), with ε* = ε∞ + "
                    "Σᵢ Δεᵢ / [1 + (jωτᵢ)^(1-αᵢ)]. "
                    "Pick a preset to seed the fields, set the "
                    "frequency, then Apply to write the "
                    "computed σ into the σ field for "
                    "{{ cole_cole_target_label }}.",
                    classes="golgi-dialog-body mb-3",
                )
                # Preset + frequency row
                with html.Div(
                    classes="d-flex align-center",
                    style="gap: 12px; margin-bottom: 12px;",
                ):
                    v3.VAutocomplete(
                        v_model=("cc_preset",),
                        items=("cc_preset_items",),
                        item_title="title",
                        item_value="value",
                        label="tissue preset",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        style="flex: 1 1 auto;",
                    )
                    v3.VTextField(
                        v_model_number=("cc_freq_hz",),
                        label="frequency",
                        suffix="Hz",
                        type="number",
                        step="any",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        style=("flex: 0 0 150px; "
                                "font-family: monospace;"),
                    )
                # Base params row: ε∞ + σ_ionic
                with html.Div(
                    classes="d-flex align-center",
                    style="gap: 12px; margin-bottom: 12px;",
                ):
                    v3.VTextField(
                        v_model_number=("cc_eps_inf",),
                        label="ε∞",
                        type="number", step="any",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        style=("flex: 1 1 0; "
                                "font-family: monospace;"),
                    )
                    v3.VTextField(
                        v_model_number=("cc_sigma_ionic",),
                        label="σ_ionic",
                        suffix="S/m",
                        type="number", step="any",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        style=("flex: 1 1 0; "
                                "font-family: monospace;"),
                    )
                # Dispersion rows. 4 rows for full Gabriel
                # 4-term layout (γ + β₁ + β₂ + α). τ is in
                # SECONDS — IT'IS DB stores per-dispersion units
                # (ps/ns/μs/ms) but we normalise on load, and the
                # formula expects SI seconds throughout. Leave a
                # row's Δε at 0 to disable that dispersion (e.g.
                # for a 3-term fit).
                for _i in range(1, 5):
                    with html.Div(
                        classes="d-flex align-center",
                        style="gap: 12px; margin-bottom: 8px;",
                    ):
                        html.Div(
                            f"#{_i}",
                            style=("font-weight: 600; "
                                    "color: #888a90; "
                                    "width: 28px; "
                                    "font-size: 12px;"),
                        )
                        v3.VTextField(
                            v_model_number=(f"cc_d{_i}_de",),
                            label="Δε",
                            type="number", step="any",
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            style=("flex: 1 1 0; "
                                    "font-family: monospace;"),
                        )
                        v3.VTextField(
                            v_model_number=(f"cc_d{_i}_tau",),
                            label="τ",
                            suffix="s",
                            type="number", step="any",
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            style=("flex: 1 1 0; "
                                    "font-family: monospace;"),
                        )
                        v3.VTextField(
                            v_model_number=(f"cc_d{_i}_alpha",),
                            label="α",
                            type="number", step="any",
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            style=("flex: 0 0 80px; "
                                    "font-family: monospace;"),
                        )
                # σ(f) plot — interactive Plotly log-log curve
                # over 1 Hz-1 GHz with a marker dot + dashed line
                # at the chosen frequency. Re-renders whenever any
                # cc_* param changes (see watchers.cole_cole).
                with html.Div(
                    style=("width: 100%; height: 280px; "
                            "border: 1px solid #e6e6e8; "
                            "border-radius: 6px; "
                            "background: white; "
                            "margin-top: 16px; "
                            "position: relative;"),
                ):
                    if export_btn is not None:
                        export_btn("conductivity.sigma_f")
                    if twp is not None:
                        twp.Figure(
                            state_variable_name="cc_plot_figure",
                            display_logo=False,
                            display_mode_bar=True,
                        )
                # Computed σ readout — big, monospaced.
                with html.Div(
                    style=("display: flex; "
                            "align-items: baseline; "
                            "justify-content: center; "
                            "gap: 12px; "
                            "background: #fff5f5; "
                            "border: 1px solid #fad4d4; "
                            "border-radius: 8px; "
                            "padding: 12px; "
                            "margin-top: 14px;"),
                ):
                    html.Span(
                        "σ at {{ cc_freq_hz }} Hz =",
                        style=("font-size: 13px; "
                                "color: #4c4c50;"),
                    )
                    html.Span(
                        "{{ cc_sigma_result_str }}",
                        style=("font-size: 20px; "
                                "font-weight: 600; "
                                "color: #e24b4a; "
                                "font-family: monospace;"),
                    )
                    html.Span(
                        "· {{ cc_n_presets }} tissues loaded "
                        "from IT'IS DB v4.2",
                        v_show=("cc_n_presets > 1",),
                        style=("font-size: 11px; "
                                "color: #888a90; "
                                "font-style: italic; "
                                "margin-left: 8px;"),
                    )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=do_cole_cole_cancel,
                )
                html.Button(
                    "Apply to {{ cole_cole_target_label }}",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    click=do_cole_cole_apply,
                )
