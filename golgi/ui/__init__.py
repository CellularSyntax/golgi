# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""UI tier — Trame layout split per dialog and drawer.

Phase 5.3 of migration.md extracts the 12 dialogs from the
giant `with VAppLayout` block in build_app; Phase 5.4 does the
same for the drawers + navbar + welcome view + busy lightbox.

Each submodule exports a `render(...)` function that builds its
piece of the Trame widget tree. The functions must be called
from inside the VAppLayout context manager — the trame widgets
use a thread-local stack to attach to the current parent.
"""
from . import (  # noqa: F401
    busy_lightbox,
    components,
    dialogs,
    drawers,
    navbar,
    welcome,
)
