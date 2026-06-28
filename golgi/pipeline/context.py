# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""PipelineContext — the bundle of per-server context that every
pipeline driver (mesh, fem, fibers, fiber_sim, pop_sim) needs.

Built once in build_app() after every closure / helper is defined.
Passed to each pipeline runner so they don't have to close over
build_app's huge local namespace.

Fields fall into three buckets:

* primary refs:   state, geom, scene
* closure hooks:  stamp_user_line, autosave, safe_update,
                  safe_reset_camera, register_subprocess,
                  clear_subprocess, was_cancelled
* helpers:        SimpleNamespace bag of module-level callables /
                  constants that the pipeline drivers need but that
                  still live in golgi.py (assemble_multi_domain_plc,
                  write_msh22, _tet_shape_quality, DEFAULTS, ...).
                  When those eventually move into proper modules
                  the helpers bag goes away.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable


@dataclass
class PipelineContext:
    """Per-server pipeline context. One instance per build_app()."""
    state: object              # trame state proxy
    geom: object               # GeometryState
    scene: object              # Scene

    stamp_user_line: Callable[[str], str]
    autosave: Callable
    safe_update: Callable[[], None]
    safe_reset_camera: Callable[[], None]
    register_subprocess: Callable
    clear_subprocess: Callable
    was_cancelled: Callable[[], bool]

    helpers: SimpleNamespace
