# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Job-runner package — viz/compute boundary.

Public API:
    from golgi.jobs import (
        JobRequest, JobOutputs, JobRunner,
        CancelToken,
        LocalSubprocessRunner, InProcessRunner,
        SlurmJobRunner, is_slurm_available,
        resolve_fem_runner,
    )
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .cancel import CancelToken
from .in_process import InProcessRunner
from .local_subprocess import LocalSubprocessRunner
from .protocol import JobOutputs, JobRequest, JobRunner
from .slurm_runner import SlurmJobRunner, is_slurm_available

__all__ = [
    "CancelToken",
    "InProcessRunner",
    "JobOutputs",
    "JobRequest",
    "JobRunner",
    "LocalSubprocessRunner",
    "SlurmJobRunner",
    "is_slurm_available",
    "resolve_fem_runner",
]


def resolve_fem_runner(
    *,
    local_runner: Any,
    on_warning: Optional[Any] = None,
) -> Any:
    """F4.2 — pick the runner for the FEM pipeline based on the
    `GOLGI_FEM_RUNNER` env var.

    Values:
      ``local``  (default) — return `local_runner` unchanged.
                  This is the LocalSubprocessRunner-based path
                  that has shipped since F3.1.
      ``slurm``  — return a SlurmJobRunner built from the
                  `GOLGI_SLURM_*` env vars. Falls back to local
                  with a warning when `sbatch` isn't on PATH
                  (and no `GOLGI_SBATCH` override is set).
      anything else — warn + fall back to local.

    Cluster-tunable env vars (read when mode=slurm):
      GOLGI_SLURM_PARTITION  (required)   — sbatch partition
      GOLGI_SLURM_ACCOUNT    (optional)   — sbatch account
      GOLGI_SLURM_CPUS       (default 4)
      GOLGI_SLURM_MEM_GB     (default 16)
      GOLGI_SLURM_TIME       (default 02:00:00)
      GOLGI_SLURM_SCRATCH    (optional)   — scratch_root dir
      GOLGI_SLURM_MODULES    (optional)   — colon-separated
                                            `module load` names

    `on_warning(msg)` is called for the fallback paths; the
    pipeline driver wires it to its log channel so the user
    sees the reason in the busy lightbox.

    F4.2 Phase A: this helper is the env-var dispatch surface.
    The actual FEM-pipeline wiring (replacing the inline
    `FEMRunner(_SOLVE_NERVE_PATH)` in pipeline/fem.py:522 with
    `resolve_fem_runner(local_runner=FEMRunner(...))`) lands in
    Phase B alongside the checkpoint-resume work, because the
    SLURM path also needs the per-band output-collection logic
    Phase B introduces.
    """
    mode = os.environ.get("GOLGI_FEM_RUNNER", "local").lower()

    def _warn(msg: str) -> None:
        if on_warning is not None:
            try:
                on_warning(msg)
            except Exception:                            # noqa: BLE001
                pass
        else:
            import sys
            print(f"[fem-runner] {msg}", file=sys.stderr)

    if mode in ("", "local"):
        return local_runner
    if mode == "slurm":
        if not is_slurm_available():
            _warn(
                "GOLGI_FEM_RUNNER=slurm but `sbatch` not on "
                "PATH (and GOLGI_SBATCH not set) — falling "
                "back to local runner"
            )
            return local_runner
        partition = os.environ.get(
            "GOLGI_SLURM_PARTITION", "",
        ).strip()
        if not partition:
            _warn(
                "GOLGI_FEM_RUNNER=slurm but "
                "GOLGI_SLURM_PARTITION not set — falling "
                "back to local runner"
            )
            return local_runner
        try:
            cpus = int(os.environ.get("GOLGI_SLURM_CPUS", "4"))
            mem_gb = int(
                os.environ.get("GOLGI_SLURM_MEM_GB", "16"),
            )
        except (TypeError, ValueError):
            cpus, mem_gb = 4, 16
        return SlurmJobRunner(
            partition=partition,
            account=os.environ.get("GOLGI_SLURM_ACCOUNT")
                    or None,
            cpus=cpus,
            memory_gb=mem_gb,
            time_limit=os.environ.get(
                "GOLGI_SLURM_TIME", "02:00:00",
            ),
            scratch_root=(
                Path(os.environ["GOLGI_SLURM_SCRATCH"])
                if os.environ.get("GOLGI_SLURM_SCRATCH")
                else None
            ),
            extra_modules=[
                m for m in os.environ.get(
                    "GOLGI_SLURM_MODULES", "",
                ).split(":")
                if m
            ],
        )
    _warn(
        f"GOLGI_FEM_RUNNER={mode!r} unrecognised — falling "
        f"back to local runner (valid: local, slurm)"
    )
    return local_runner
