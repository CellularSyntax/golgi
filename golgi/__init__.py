# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""golgi — vagal-nerve cuff-electrode FEM design + analysis.

Top-level public surface (F4.1):

    import golgi
    s = golgi.Study.create("/tmp/demo_project")
    s.import_nerve("data/sample_nerve.stl")
    ...
    s.export_bundle("/tmp/demo.zip")

The GUI app is started via the original entrypoint (`python -m
golgi.app`) — this module-level import does NOT spin up trame or
build the workspace; it only makes the headless `Study` API
reachable for scripts and notebooks.
"""

__all__ = ["Study"]


def __getattr__(name: str):
    # Lazy import so `import golgi` stays fast and doesn't pull
    # in the trame / vtk / pyvista stack until the user actually
    # asks for the headless API.
    if name == "Study":
        from golgi.api import Study
        return Study
    raise AttributeError(name)
