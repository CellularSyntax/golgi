# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""User-action handlers (do_* callables) extracted from build_app,
grouped by topic.

Each group exposes a `register(state, ...)` function that returns a
dict[str, Callable] of {handler_name: callable} that build_app then
rebinds to local names so existing UI template references continue
to resolve.

Step W1.8 of FEATURES.md: 68 do_* handlers move out of build_app's
closure across five sub-commits (W1.8a through W1.8e), one domain
group per commit.
"""
from . import (  # noqa: F401
    auth,
    bundle_import,
    cancel_busy,
    compute,
    conductivity,
    figure_export,
    segment_uct,
    study_bundle,
    sweep,
)
