# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Dialog sub-modules — one per VDialog in build_app.

Each module exports `render(...)` that builds the dialog's
widget tree inside the current VAppLayout context. Call from
build_app right where the inline `with v3.VDialog(...)` block
used to be.
"""
from . import (  # noqa: F401
    auth,
    bundle_import,
    cancel_busy,
    close_project,
    cole_cole,
    confirm_delete_electrode,
    confirm_delete_mesh,
    confirm_delete_project,
    confirm_remove_geometry,
    cuff_designer,
    export_study,
    generate_report,
    import_stepper,
    import_study,
    logout,
    new_project,
    profile,
    project_detail,
    segment_uct,
    sweep_designs,
    sweep_manual,
    sweep_random,
)
