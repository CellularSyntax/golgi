# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Analysis drawers — only the Solve (FEM) drawer is left.

The Single-fiber and Population analysis tabs no longer render
side drawers; their controls now live directly inside their
respective `golgi-fiber-panel` / `golgi-pop-panel` analysis
panels (the `show_fiber` / `show_pop` state vars still drive
the tab-active watcher that flips `active_analysis`)."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from ._helpers import slider_row


def render(*, do_solve_fem: Callable) -> None:
    with v3.VNavigationDrawer(
        v_model=("show_solve",),
        location="right", width=420,
        elevation=8,
    ):
        with v3.VContainer(classes="pa-4"):
            # Title row: H3 + info-icon tooltip + close button.
            with html.Div(
                classes=(
                    "d-flex align-center "
                    "justify-space-between mb-3"
                ),
            ):
                with html.Div(
                    classes="d-flex align-center",
                ):
                    html.H3(
                        "Extracellular field (FEM)",
                        classes="text-h6 mb-0",
                    )
                    with v3.VTooltip(
                        location="bottom",
                        max_width=360,
                    ):
                        with v3.Template(
                            v_slot_activator=(
                                "{ props }",
                            ),
                        ):
                            with v3.VBtn(
                                v_bind="props",
                                icon=True,
                                size="small",
                                variant="text",
                                density="compact",
                                classes="ml-1",
                            ):
                                v3.VIcon(
                                    "mdi-information-outline",
                                    size="18",
                                    color="grey-darken-1",
                                )
                        html.Span(
                            "FEM extracellular-potential "
                            "solve on the built nerve.msh. "
                            "Uses the σ values from the "
                            "Conductivities tab and the "
                            "electrode configuration from "
                            "the Electrodes tab."
                        )
                v3.VBtn(
                    icon="mdi-close", size="small",
                    variant="text",
                    click="show_solve = false",
                )

            html.H4("Stimulus",
                     classes="text-subtitle-2 mt-2 mb-1")
            slider_row(
                "I_stim_mA",
                "Cathodic stimulation current (mA)",
                0.001, 50.0, 0.001, "toFixed(3)",
            )

            # Solver preset (Step 7.1b). Drives the --preset CLI
            # flag passed through to solve_nerve.py. "Quick" =
            # loose tolerance for sanity checks; "Balanced" =
            # current production; "HPC" = BoomerAMG tuned for
            # 20M+ element meshes.
            html.H4(
                "Solver preset",
                classes="text-subtitle-2 mt-3 mb-1",
            )
            v3.VSelect(
                v_model=("fem_preset",),
                items=("fem_preset_options",),
                density="compact",
                hide_details=True,
                variant="outlined",
                classes="mb-1",
            )
            html.Div(
                "Quick = ksp_rtol 1e-4, 200 iter cap · "
                "Balanced = current production · "
                "HPC = BoomerAMG tuned for 20 M+ elements. "
                "Core count comes from $GOLGI_FEM_CORES.",
                style=(
                    "font-size: 10px; color: #888a90; "
                    "margin-top: 2px; margin-bottom: 6px; "
                    "line-height: 1.4;"
                ),
            )

            # F3.2c: which contact configs to solve. One mesh per
            # design, one solve per config. Empty selection falls
            # back to the currently-active config (the one
            # highlighted in the Designs drawer's configs panel).
            html.H4(
                "Configs to solve",
                classes="text-subtitle-2 mt-3 mb-1",
            )
            html.Div(
                "Pick which polarity configurations to run "
                "solve_nerve.py on. Outputs land in "
                "<out>/configs/<cid>/. Each solve reuses the "
                "parent design's mesh — no remeshing.",
                style=(
                    "font-size: 10px; color: #888a90; "
                    "margin-bottom: 6px; line-height: 1.4;"
                ),
            )
            # Multi-select reads from the precomputed
            # `solve_config_items` list — see app.py's
            # _on_config_items_rebuild watcher. The inline
            # `.map()` form we had before dropped multi-picks
            # because Vue saw a new array reference every render.
            with html.Div(
                classes="d-flex align-center",
                style="gap: 6px;",
            ):
                v3.VSelect(
                    v_model=("solve_config_selection",),
                    items=("solve_config_items",),
                    item_title="title",
                    item_value="value",
                    multiple=True,
                    chips=True,
                    closable_chips=True,
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    style="flex: 1 1 auto;",
                )
                html.Button(
                    "All",
                    type="button",
                    title=(
                        "Tick every config in this project"
                    ),
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=(
                        "solve_config_selection = "
                        "  (solve_config_items || [])"
                        "    .map(c => c.value)"
                    ),
                )
                html.Button(
                    "None",
                    type="button",
                    title="Clear the selection",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    click=(
                        "solve_config_selection = []"
                    ),
                )
            html.Div(
                "Will solve "
                "{{ (solve_config_selection || []).length "
                "    || 1 }} config(s). "
                "Empty list = the active config only.",
                style=(
                    "font-size: 10px; color: #888; "
                    "margin-bottom: 8px; margin-top: 4px;"
                ),
            )

            # ---- I1 Phase A — Outputs section. Toggles for
            # auxiliary post-solve computations. Lives here (and
            # not in the Conductivities drawer) because it gates
            # what gets WRITTEN OUT, not what σ values feed the
            # primary solve. When ON (default), every FEM solve
            # runs N+M Dirichlet dual-solves (one per contact +
            # one per anode/cathode pair) and writes
            # `<config>/impedance.json`; results render as the
            # chip summary below + the Compare-panel bar tile.
            html.H4(
                "Outputs",
                classes="text-subtitle-2 mt-3 mb-1",
            )
            v3.VCheckbox(
                v_model=("emit_impedance",),
                label=(
                    "Compute electrode access impedance "
                    "(Z_access) on each solve"
                ),
                density="compact", hide_details=True,
                color="primary", classes="mb-0",
            )
            html.Div(
                "Adds N+M dirichlet dual-solves per FEM run "
                "(N contacts, M pairs). This is the pure tissue "
                "spreading / access resistance from FEM — it does "
                "NOT include the electrode-electrolyte interface "
                "(double-layer capacitance, charge-transfer "
                "resistance), which dominates real DC "
                "measurements. Summary chips appear below the "
                "status banner; full bars in the Compare panel.",
                style=(
                    "font-size: 10px; color: #888; "
                    "margin-bottom: 8px; "
                    "line-height: 1.4;"
                ),
            )

            # Fancy CTA-styled Run button — same animated
            # conic-gradient wrapper as the "Create new project"
            # button on the welcome screen, scoped via
            # `.golgi-cta-wrapper` so the rotation + dark-red
            # fill + leading icon all come along.
            with html.Button(
                type="button",
                classes=(
                    "golgi-cta-wrapper golgi-cta-wrapper-block "
                    "mt-2 mb-3"
                ),
                disabled=(
                    "!(configs && configs.length > 0)",
                ),
                click=do_solve_fem,
            ):
                html.Span(classes="golgi-cta-spinner")
                with html.Span(classes="golgi-cta-inner"):
                    html.Span("▶ Run FEM solve")

            # Status / error display.
            with html.Div(v_show=("has_fem && !fem_failed",)):
                html.Div(
                    "{{ fem_status }}",
                    style=("font-size: 11px; color: #146e3a; "
                            "font-weight: 500; "
                            "background:#eaf6ee; "
                            "border-radius:4px; "
                            "padding:8px 10px;"),
                )
                # I1 Phase A — Impedance summary chips for the
                # ACTIVE config. Hidden when no impedance was
                # computed (toggle off in the Conductivities
                # drawer, or older solve before I1 landed). Full
                # bar charts live in the Compare panel.
                with html.Div(
                    v_show=(
                        "(fem_impedance_chips_contact && "
                        " fem_impedance_chips_contact.length) "
                        "|| (fem_impedance_chips_pair && "
                        "    fem_impedance_chips_pair.length)",
                    ),
                    style=("margin-top: 10px;"),
                ):
                    with html.Div(
                        classes=(
                            "d-flex align-center "
                            "justify-space-between"
                        ),
                        style="margin-bottom: 4px;",
                    ):
                        # Section label — explicit "Access" to
                        # distinguish from total measured impedance
                        # (which includes electrode-electrolyte
                        # interface terms we don't model). Tooltip
                        # spells out the caveat for newcomers.
                        with v3.VTooltip(
                            location="bottom",
                            max_width=320,
                        ):
                            with v3.Template(
                                v_slot_activator=(
                                    "{ props }",
                                ),
                            ):
                                html.Div(
                                    "Access impedance (Z_access)",
                                    v_bind="props",
                                    style=("font-size: 11px; "
                                            "font-weight: 600; "
                                            "color: #1f2024; "
                                            "text-transform: "
                                            "uppercase; "
                                            "letter-spacing: "
                                            "0.04em; "
                                            "cursor: help; "
                                            "border-bottom: "
                                            "1px dotted #aaa;"),
                                )
                            html.Span(
                                "Pure tissue spreading resistance "
                                "from FEM (saline + tissues). "
                                "Real DC measurements also include "
                                "an electrode-electrolyte "
                                "interface impedance (double-layer "
                                "capacitance + charge-transfer "
                                "resistance) that is NOT modeled "
                                "here — typical interface adds "
                                "0.5-50 kΩ at DC depending on "
                                "contact material."
                            )
                        html.Div(
                            "{{ fem_impedance_chips_meta }}",
                            style=("font-size: 10px; "
                                    "color: #888a90; "
                                    "font-family: ui-monospace,"
                                    "Menlo,Consolas,monospace;"),
                        )
                    # Per-contact chips. Role-tinted (cathode =
                    # pink, anode = blue, none = grey) so the
                    # polarity reads at a glance.
                    with html.Div(
                        v_show=(
                            "fem_impedance_chips_contact && "
                            "fem_impedance_chips_contact.length",
                        ),
                        classes="d-flex flex-wrap",
                        style=("gap: 4px; margin-bottom: 6px;"),
                    ):
                        # Note trailing comma in `style=(...)`:
                        # tuple → `:style="..."` so c.bg actually
                        # binds (vs static `style="..."` which
                        # Vue would never re-interpolate). See
                        # cuff_electrodes.py L142-148 for the
                        # same pattern + reasoning.
                        html.Div(
                            (
                                "Z_acc({{ c.id }}, "
                                "{{ c.role || '—' }}) = "
                                "{{ c.z_fmt }}"
                            ),
                            v_for=(
                                "c in "
                                "fem_impedance_chips_contact"
                            ),
                            key=("c.id",),
                            style=(
                                "'font-size: 11px; "
                                "padding: 3px 8px; "
                                "border-radius: 10px; "
                                "color: #1f2024; "
                                "font-family: ui-monospace,"
                                "Menlo,Consolas,monospace; "
                                "background: ' + c.bg",
                            ),
                        )
                    # Per-pair chips (anode → cathode).
                    with html.Div(
                        v_show=(
                            "fem_impedance_chips_pair && "
                            "fem_impedance_chips_pair.length",
                        ),
                        classes="d-flex flex-wrap",
                        style="gap: 4px;",
                    ):
                        html.Div(
                            (
                                "Z_acc({{ p.anode }} → "
                                "{{ p.cathode }}) = {{ p.z_fmt }}"
                            ),
                            v_for=(
                                "p in fem_impedance_chips_pair"
                            ),
                            key=("p.anode + '_' + p.cathode",),
                            style=(
                                "font-size: 11px; "
                                "padding: 3px 8px; "
                                "border-radius: 10px; "
                                "background: #f4ecde; "
                                "color: #1f2024; "
                                "font-family: ui-monospace,"
                                "Menlo,Consolas,monospace;"
                            ),
                        )
            with html.Div(v_show=("fem_failed",)):
                html.Div(
                    "{{ fem_status }}",
                    style=("color: #b8336a; font-size: 12px; "
                            "font-weight: 600; "
                            "margin-bottom: 6px;"),
                )
                html.Pre(
                    "{{ fem_log }}",
                    style=("font-family: ui-monospace,"
                            "Menlo,Consolas,monospace; "
                            "font-size: 10px; color:#222; "
                            "background:#f4f4f4; padding:8px; "
                            "max-height:300px; overflow:auto;"
                            "white-space: pre-wrap;"),
                )
