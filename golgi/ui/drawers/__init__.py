# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Right-side VNavigationDrawer panels — one per workflow step.

Each module exports `render(...)` that builds its drawer's
widget tree inside the current VAppLayout context. Call from
build_app right where the inline `with v3.VNavigationDrawer(...)`
block used to be."""
from . import (  # noqa: F401
    analysis,
    conductivities,
    cuff_electrodes,
    exports,
    fibers,
    import_drawer,
    mesh,
)
