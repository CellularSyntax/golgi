# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""@state.change watchers extracted from build_app, grouped by topic.

Each group exposes a `register(state, ...)` function that decorates
its handlers with `@state.change` against the trame state proxy.
Most groups also need access to build_app closures (autosave,
scene helpers, etc.); those are passed as additional kwargs.
Some `register()` functions return a dict of controller actions
(do_*) that build_app then binds to UI elements.

Step 5.2 of migration.md: 53 watchers move out, one topic at a
time.
"""
from . import (  # noqa: F401
    auth_upload,
    cole_cole,
    cuff,
    cuff_designer,
    drawer_exclusion,
    fem_panel,
    fiber_panel,
    import_panel,
    project_detail,
    render_toggles,
    sigma,
)
