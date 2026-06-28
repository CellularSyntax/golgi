# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Manual contact-pair sweep dialog (F3.2b).

Lets the user assemble a list of (cathode, anode) pairs by hand,
then generates one bipolar config per row when 'Generate' is
clicked. State lives in `state.sweep_manual_pairs` (list of
{cathode_idx, anode_idx, name} dicts); the dialog adds rows via
the do_sweep_manual_add_row trigger and submits via
do_sweep_manual_run.
"""
from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render() -> None:
    with v3.VDialog(
        v_model=("show_sweep_manual_dialog",),
        max_width=620,
        persistent=False,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Manual contact-pair sweep",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Pick (cathode, anode) pairs by hand. Each "
                    "row becomes one bipolar config on the "
                    "selected design — contact indices are "
                    "1-based to match the polarity picker.",
                    classes="golgi-dialog-body mb-3",
                )
                # Existing pair rows.
                html.Div(
                    "Pairs queued: {{ "
                    "(sweep_manual_pairs || []).length "
                    "}}",
                    style=(
                        "font-size: 11px; color: #555; "
                        "margin-bottom: 4px;"
                    ),
                )
                with html.Div(
                    v_show=(
                        "(sweep_manual_pairs || []).length > 0",
                    ),
                    style=(
                        "display: flex; "
                        "flex-direction: column; "
                        "gap: 4px; "
                        "max-height: 240px; "
                        "overflow-y: auto; "
                        "margin-bottom: 10px;"
                    ),
                ):
                    with html.Div(
                        v_for=(
                            "(p, i) in "
                            "(sweep_manual_pairs || [])"
                        ),
                        key="i",
                        style=(
                            "padding: 4px 8px; "
                            "background: #f7f7f9; "
                            "border-radius: 4px; "
                            "display: flex; "
                            "align-items: center; "
                            "gap: 8px;"
                        ),
                    ):
                        html.Span(
                            "C{{ p.cathode_idx + 1 }}↓  "
                            "C{{ p.anode_idx + 1 }}↑",
                            style=(
                                "font-family: ui-monospace,"
                                "Menlo,Consolas,monospace; "
                                "font-size: 11px; "
                                "color: #1f2024; "
                                "flex: 0 0 auto;"
                            ),
                        )
                        html.Span(
                            "{{ p.name "
                            "  || '(unnamed)' }}",
                            style=(
                                "flex: 1 1 auto; "
                                "font-size: 11px; "
                                "color: #555; "
                                "overflow: hidden; "
                                "text-overflow: ellipsis; "
                                "white-space: nowrap;"
                            ),
                        )
                        html.Button(
                            "✕",
                            type="button",
                            title="Remove",
                            classes=(
                                "golgi-btn-secondary "
                                "golgi-btn-round "
                                "golgi-btn-sm"
                            ),
                            click=(
                                "trigger("
                                "  'do_sweep_manual_remove_row',"
                                "  [i])"
                            ),
                        )
                # Add-row form.
                html.Div(
                    "Add a pair:",
                    style=(
                        "font-size: 11px; color: #555; "
                        "margin-bottom: 4px;"
                    ),
                )
                with html.Div(
                    style=(
                        "display: grid; "
                        "grid-template-columns: "
                        "  1fr 1fr 2fr auto; "
                        "gap: 6px; align-items: center; "
                        "margin-bottom: 8px;"
                    ),
                ):
                    v3.VTextField(
                        label="Cathode (1..N)",
                        v_model_number=(
                            "sweep_manual_new_cathode",
                        ),
                        type="number",
                        min=1, step=1,
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                    v3.VTextField(
                        label="Anode (1..N)",
                        v_model_number=(
                            "sweep_manual_new_anode",
                        ),
                        type="number",
                        min=1, step=1,
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                    v3.VTextField(
                        label="Name (optional)",
                        v_model=("sweep_manual_new_name",),
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                    # We store indices 0-based on the server; the
                    # UI shows 1-based, so subtract 1 before
                    # writing.
                    html.Button(
                        "+ Add",
                        type="button",
                        classes=(
                            "golgi-btn-primary golgi-btn-sm"
                        ),
                        click=(
                            "sweep_manual_new_cathode = "
                            "  Number(sweep_manual_new_cathode) - 1; "
                            "sweep_manual_new_anode = "
                            "  Number(sweep_manual_new_anode) - 1; "
                            "trigger("
                            "  'do_sweep_manual_add_row', []); "
                            "sweep_manual_new_cathode = "
                            "  Number(sweep_manual_new_cathode) + 1; "
                            "sweep_manual_new_anode = "
                            "  Number(sweep_manual_new_anode) + 1"
                        ),
                    )
                html.Div(
                    "Hint: cathode ≠ anode, both in "
                    "1..{{ contact_count }}.",
                    style=(
                        "font-size: 10px; color: #888; "
                        "margin-bottom: 4px;"
                    ),
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=(
                        "show_sweep_manual_dialog = false; "
                        "sweep_manual_pairs = []"
                    ),
                )
                html.Button(
                    "Generate",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    disabled=(
                        "(sweep_manual_pairs || []).length === 0",
                    ),
                    click="trigger('do_sweep_manual_run', [])",
                )
