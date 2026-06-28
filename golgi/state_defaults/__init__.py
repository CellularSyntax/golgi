# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Per-topic state-default initialisers, one module per concern.

Each module exports `register(state)` that seeds the relevant
state.* keys to their factory defaults. build_app() calls them
all once early in startup (step 5.1 of migration.md), replacing
the ~250-line inline block of `state.X = Y` assignments.

Usage:
    from golgi.state_defaults import (
        ui_toggles, fem, fiber, pop, cuff, electrode,
        import_state, mesh,
    )
    ui_toggles.register(state)
    fem.register(state)
    ...
"""
from . import (  # noqa: F401
    cuff, electrode, exports, fem, fiber, import_state, mesh, pop,
    pop_presets, study_bundle, sweep, ui_toggles,
)
