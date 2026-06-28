# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Drawer mutual-exclusion + viewport-mode swap.

Only one drawer/analysis tab may be open at a time. Clicking a
second one closes the first. The analysis-subset
(Solve / Fiber / Population) also flips the viewport between 3D
and "analysis" mode.

Returns a dict with `do_close_all_tabs` because the navbar logo
click handler needs to bind to it.
"""
from __future__ import annotations


def register(state) -> dict:
    """Install the drawer-exclusion watcher + the close-all-tabs
    controller action. Returns {"do_close_all_tabs": fn}."""
    _analysis_keys = (
        "show_solve", "show_fiber", "show_pop", "show_sweep",
        "show_compare",
    )
    _analysis_labels = {
        "show_solve": "solve",
        "show_fiber": "fiber",
        "show_pop": "population",
        "show_sweep": "sweep",
        "show_compare": "compare",
    }
    _drawer_keys = (
        "show_import", "show_fibers", "show_cuff", "show_mesh",
        "show_sigma",
        *_analysis_keys,
    )
    _last_drawer = {"key": None}      # mutable for closure

    @state.change(*_drawer_keys)
    def _on_drawer_change(**_kwargs):
        # Detect which key was just turned on (compared to the
        # cached last-opened) and close every other open drawer.
        # Without this guard a click on a second menu item would
        # stack two drawers v_show-true at once.
        opened: list = []
        for k in _drawer_keys:
            if bool(state[k]) and k != _last_drawer["key"]:
                opened.append(k)
        if opened:
            new = opened[-1]
            with state:
                for k in _drawer_keys:
                    if k != new and state[k]:
                        state[k] = False
            _last_drawer["key"] = new
        # Derived state: viewport mode + active analysis. The
        # viewport panel-swap only applies to the analysis subset
        # (Solve / Fiber / Population); the property-editor
        # drawers leave the 3D scene as the central view.
        analysis_open = next(
            (k for k in _analysis_keys if state[k]),
            None,
        )
        if analysis_open is not None:
            state.viewport_mode = "analysis"
            state.active_analysis = _analysis_labels[analysis_open]
            _last_drawer["key"] = analysis_open
        else:
            state.viewport_mode = "3d"
            state.active_analysis = ""
            if not any(bool(state[k]) for k in _drawer_keys):
                _last_drawer["key"] = None

    def do_close_all_tabs(*_args) -> None:
        """Close every open drawer / analysis tab so the viewport
        falls back to its full-screen 3-D render. Bound to the
        navbar logo so a click on it acts like a global 'back
        to workspace' affordance. The `_on_drawer_change`
        watcher picks up the cascade and resets viewport_mode +
        active_analysis automatically."""
        with state:
            for k in _drawer_keys:
                if state[k]:
                    state[k] = False

    return {"do_close_all_tabs": do_close_all_tabs}
