# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Conductivities (σ) drawer — Tissues group (endoneurium /
epineurium / muscle) + Electrode materials group (silicone /
saline). Each row: label + Cole-Cole evaluator button + preset
picker + numeric value. Footer: Update CTA + transient status."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    sigma_tag_map: dict[str, int],
    sigma_label_map: dict[str, str],
    do_reset_sigma: Callable,
    do_update_sigma: Callable,
) -> None:
    with v3.VNavigationDrawer(
        v_model=("show_sigma",),
        location="right", width=400,
        elevation=8,
    ):
        with v3.VContainer(classes="pa-4"):
            with html.Div(
                classes="d-flex align-center justify-space-between mb-3",
            ):
                html.H3("Conductivities (σ)",
                         classes="text-h6 mb-0")
                with html.Div(
                    classes="d-flex align-center",
                    style="gap: 6px;",
                ):
                    html.Button(
                        "Reset",
                        type="button",
                        classes=(
                            "golgi-btn-secondary golgi-btn-sm"
                        ),
                        click=do_reset_sigma,
                    )
                    v3.VBtn(
                        icon="mdi-close", size="small",
                        variant="text",
                        click="show_sigma = false",
                    )
            html.Div(
                "Quasi-static conductivities used by "
                "solve_nerve.py for the FEM solve. Tag refers "
                "to the domain label in the built TetGen mesh.",
                style=("font-size: 11px; color: #666; "
                        "margin-bottom: 12px; line-height: 1.4;"),
            )

            # Inline helper: render one σ row (heading +
            # Cole-Cole button + preset picker + numeric value).
            def _sigma_row(_key: str) -> None:
                _tag = sigma_tag_map[_key]
                _lbl = sigma_label_map[_key]
                _preset_key = f"{_key}_preset"
                _items_key = f"{_key}_preset_items"
                with html.Div(classes="mb-3"):
                    # Row 1: label + Cole-Cole evaluator button.
                    with html.Div(
                        classes=(
                            "d-flex align-center "
                            "justify-space-between"
                        ),
                        style="margin-bottom: 4px;",
                    ):
                        html.Div(
                            f"{_lbl} (tag {_tag})",
                            style=("font-size: 12px; "
                                    "font-weight: 500; "
                                    "color: #1f2024;"),
                        )
                        v3.VBtn(
                            "Cole-Cole",
                            prepend_icon="mdi-sine-wave",
                            size="x-small",
                            variant="text",
                            style=("text-transform: none; "
                                    "color: #e24b4a; "
                                    "padding: 0 6px;"),
                            click=(
                                f"cole_cole_target = "
                                f"  '{_key}'; "
                                f"cole_cole_target_label = "
                                f"  '{_lbl}'; "
                                f"show_cole_cole_dialog = true"
                            ),
                        )
                    # Row 2: preset picker + numeric value.
                    with html.Div(
                        classes="d-flex align-center",
                        style="gap: 8px;",
                    ):
                        v3.VAutocomplete(
                            v_model=(_preset_key,),
                            items=(_items_key,),
                            item_title="title",
                            item_value="value",
                            label="preset",
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            menu_props=(
                                "{maxHeight: 320}",
                            ),
                            style=(
                                "flex: 1 1 auto; "
                                "min-width: 0;"
                            ),
                        )
                        v3.VTextField(
                            v_model_number=(_key,),
                            type="number",
                            step="any",
                            suffix="S/m",
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            style=(
                                "flex: 0 0 122px; "
                                "font-family: monospace;"
                            ),
                        )

            # ---- Tissue group ----
            html.H4(
                "Tissues",
                classes="text-subtitle-2 mt-1 mb-2",
                style=(
                    "color: #888a90; "
                    "letter-spacing: 0.04em; "
                    "text-transform: uppercase; "
                    "font-size: 10px;"
                ),
            )
            for _key in (
                "sigma_endo",
                "sigma_epi",
                # F3.2-M3 — scar / connective tissue (tag 7).
                # Sits between epi and muscle in the radial
                # stack inside the cuff; UI order mirrors that.
                "sigma_scar",
                "sigma_muscle",
            ):
                _sigma_row(_key)

            # ---- Electrode-material group ----
            html.H4(
                "Electrode materials",
                classes="text-subtitle-2 mt-4 mb-2",
                style=(
                    "color: #888a90; "
                    "letter-spacing: 0.04em; "
                    "text-transform: uppercase; "
                    "font-size: 10px;"
                ),
            )
            for _key in (
                "sigma_silicone",
                "sigma_saline",
                # F3.2-M3 — contact metal σ (tag 6). The 9-entry
                # preset list (Au / Pt / Pt-Ir 90:10 + 80:20 /
                # SS316LVM / Ti / TiN / IrOx / perfect conductor)
                # was already defined in SIGMA_PRESETS but the
                # row was never added here. Default 4.10e7 S/m
                # (bulk gold) matches the gold-coloured contact
                # actor.
                "sigma_contact",
            ):
                _sigma_row(_key)
            # ---- Update CTA ----
            html.Button(
                "▶ Update conductivities",
                type="button",
                classes=(
                    "golgi-btn-primary golgi-btn-block "
                    "mt-4 mb-2"
                ),
                title=(
                    "Write current σ values to "
                    "conductivities.json and log an "
                    "activity event"
                ),
                click=do_update_sigma,
            )
            # Transient confirmation — self-clears on the next
            # σ slider tick (the per-key watcher resets the
            # message so stale "updated" chips don't linger).
            html.Div(
                "{{ sigma_update_status }}",
                v_show=("sigma_update_status",),
                style=("font-size: 11px; "
                        "color: #146e3a; "
                        "background: #eaf6ee; "
                        "border-radius: 4px; "
                        "padding: 6px 10px; "
                        "margin-top: 4px; "
                        "font-weight: 500;"),
            )
            html.Div(
                "Auto-saved to results_golgi/"
                "conductivities.json on edit. The Update "
                "button commits the current values and "
                "logs an activity entry.",
                style=("font-size: 10px; color: #999; "
                        "margin-top: 8px; "
                        "font-style: italic;"),
            )
