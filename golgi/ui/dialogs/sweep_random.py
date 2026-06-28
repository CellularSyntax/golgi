# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Random polarity-sweep dialog (F3.2b).

Asks for N_draws + K cathodes + L anodes + (rest off/ground)
+ optional seed, then generates N_draws random polarity
assignments on the currently-selected design via the
do_sweep_random_run server trigger.
"""
from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render() -> None:
    with v3.VDialog(
        v_model=("show_sweep_random_dialog",),
        max_width=520,
        persistent=False,
    ):
        with v3.VCard():
            with v3.VCardText(classes="pa-6"):
                html.Div(
                    "Random polarity draws",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "Generates N random configs on the selected "
                    "design. Each config picks K cathodes + L "
                    "anodes uniformly at random from the "
                    "{{ contact_count }} contacts; the rest are "
                    "either off or wired to ground.",
                    classes="golgi-dialog-body mb-3",
                )
                # Numeric inputs in a 3-column grid.
                with html.Div(
                    style=(
                        "display: grid; "
                        "grid-template-columns: 1fr 1fr 1fr; "
                        "gap: 8px; margin-bottom: 10px;"
                    ),
                ):
                    v3.VTextField(
                        label="Number of draws (N)",
                        v_model_number=("sweep_random_n_draws",),
                        type="number",
                        min=1, max=500, step=1,
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                    v3.VTextField(
                        label="Cathodes (K)",
                        v_model_number=(
                            "sweep_random_k_cathodes",
                        ),
                        type="number",
                        min=1, step=1,
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                    v3.VTextField(
                        label="Anodes (L)",
                        v_model_number=(
                            "sweep_random_l_anodes",
                        ),
                        type="number",
                        min=0, step=1,
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                    )
                # K + L must not exceed contact_count — show a
                # warning inline so the user can't accidentally
                # submit invalid params.
                html.Div(
                    "⚠ K + L > "
                    "{{ contact_count }} — reduce one of them.",
                    v_show=(
                        "(sweep_random_k_cathodes "
                        "+ sweep_random_l_anodes) "
                        "> contact_count",
                    ),
                    style=(
                        "color: #b8336a; font-size: 11px; "
                        "margin-bottom: 8px;"
                    ),
                )
                # Rest = off / ground.
                html.Div(
                    "Remaining contacts:",
                    style=(
                        "font-size: 11px; color: #555; "
                        "margin-bottom: 4px;"
                    ),
                )
                with v3.VRadioGroup(
                    v_model=("sweep_random_rest",),
                    inline=True,
                    density="compact",
                    hide_details=True,
                    classes="mb-2",
                ):
                    v3.VRadio(label="Off", value="off")
                    v3.VRadio(label="Ground", value="ground")
                # Optional seed.
                v3.VTextField(
                    label="Seed (optional)",
                    v_model=("sweep_random_seed",),
                    placeholder="leave blank for unseeded",
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click="show_sweep_random_dialog = false",
                )
                html.Button(
                    "Generate",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    disabled=(
                        "(sweep_random_k_cathodes "
                        "+ sweep_random_l_anodes) "
                        "> contact_count "
                        "|| sweep_random_n_draws < 1",
                    ),
                    click="trigger('do_sweep_random_run', [])",
                )
