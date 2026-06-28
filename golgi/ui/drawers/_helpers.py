# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Shared widget helpers used by multiple drawer modules."""
from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def slider_row(
    model: str, label: str, mn, mx, step,
    fmt: str = "toFixed(1)",
    info: str | None = None,
) -> None:
    """A label + (optional info-icon tooltip) + slider + numeric
    field row. Used by the Cuff & electrodes and Fibers drawers."""
    with html.Div(classes="mt-2 mb-1"):
        if info:
            with html.Div(
                classes="d-flex align-center",
                style="gap: 2px; margin-bottom: 2px;",
            ):
                html.Span(
                    label,
                    style=(
                        "font-size: 11px; color: #555;"
                    ),
                )
                with v3.VTooltip(
                    location="top", max_width=320,
                ):
                    with v3.Template(
                        v_slot_activator=(
                            "{ props }",
                        ),
                    ):
                        with v3.VBtn(
                            v_bind="props",
                            icon=True,
                            size="x-small",
                            variant="text",
                            density="compact",
                        ):
                            v3.VIcon(
                                "mdi-information-outline",
                                size="14",
                                color="grey-darken-1",
                            )
                    html.Span(info)
        else:
            html.Div(
                label,
                style=("font-size: 11px; color: #555; "
                        "margin-bottom: 2px;"),
            )
        with html.Div(classes="d-flex align-center"):
            v3.VSlider(
                v_model=(model,),
                min=mn, max=mx, step=step,
                density="compact", hide_details=True,
                thumb_label=False, color="primary",
                classes="flex-grow-1",
            )
            v3.VTextField(
                v_model_number=(model,),
                type="number", step=step,
                density="compact", hide_details=True,
                variant="outlined",
                style=("max-width: 92px; "
                        "margin-left: 8px;"),
            )


def param_row_with_info(
    model: str, label: str, info: str, suffix: str,
    step, mn=None, mx=None,
) -> None:
    """One "label + info-icon (tooltip) + numeric field" row,
    used by the Mesh and Fibers drawers."""
    with html.Div(
        classes="d-flex align-center mt-2 mb-1",
        style="gap: 8px;",
    ):
        with html.Div(
            classes="d-flex align-center",
            style=(
                "flex: 1 1 auto; min-width: 0; "
                "gap: 2px;"
            ),
        ):
            html.Span(
                label,
                style=(
                    "font-size: 12px; "
                    "color: #555; "
                    "font-weight: 500;"
                ),
            )
            with v3.VTooltip(
                location="top", max_width=320,
            ):
                with v3.Template(
                    v_slot_activator=("{ props }",),
                ):
                    with v3.VBtn(
                        v_bind="props",
                        icon=True,
                        size="x-small",
                        variant="text",
                        density="compact",
                    ):
                        v3.VIcon(
                            "mdi-information-outline",
                            size="16",
                            color="grey-darken-1",
                        )
                html.Span(info)
        _kw = dict(
            v_model_number=(model,),
            type="number", step=step,
            suffix=suffix,
            density="compact", hide_details=True,
            variant="outlined",
            style="flex: 0 0 116px;",
        )
        if mn is not None:
            _kw["min"] = mn
        if mx is not None:
            _kw["max"] = mx
        v3.VTextField(**_kw)
