# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""LocalSubprocessRunner — spawn a python script, stream stdout, return outputs.

Today's default JobRunner. Wraps the existing pattern used by
run_tetgen_subprocess / solve_nerve.py-driver /
solve_fiber_paths_nerve.py-driver in golgi.py:

  1. write the request payload to a JSON file (or env vars)
  2. spawn `python -u <script_path>` with the path as argv[1]
  3. stream stdout line-by-line into `on_line`
  4. poll cancel between batches; the actual SIGTERM happens via
     the cancel token's terminate() (cancel.arm(proc) wires it)
  5. on exit, return JobOutputs with return_code + a paths dict

Subclasses override _build_payload, _build_argv, _build_env, and
_collect_outputs to wire each pipeline's specific contract. The
loop body (spawn / stream / wait / wrap-up) is shared.

Step 4.1 of migration.md: skeleton class only — actual pipeline
integrations come in 4.2-4.5.
"""
from __future__ import annotations

import json
import subprocess
import sys
import dataclasses
from pathlib import Path
from typing import Callable

from .cancel import CancelToken
from .protocol import JobOutputs, JobRequest


class LocalSubprocessRunner:
    """One instance per pipeline (mesh, fem, fibers, ...). Holds
    the script path; .run() is callable repeatedly.

    Subclass hooks (override per pipeline):
      _build_payload_path(req)  → Path or None — the JSON file
                                  passed as argv[1]
      _build_env(req)           → dict[str, str] env overrides
      _collect_outputs(req)     → dict[str, Path] expected on
                                  return-code 0

    The default implementation:
      * dataclass-asdict's the JobRequest into JSON at the
        payload path returned by _build_payload_path (skips the
        write when the hook returns None).
      * spawns `python -u <script_path> [payload_path]`.
      * streams stdout to on_line.
      * arms the cancel token before stream-read so the UI's
        cancel.request() can terminate the process immediately.
    """

    def __init__(self, script_path: Path):
        self.script_path = Path(script_path)

    # Hooks ------------------------------------------------------

    def _build_payload_path(self, req: JobRequest) -> "Path | None":
        """Override to specify where the JSON payload lands.
        Default: no payload file (env-var only)."""
        return None

    def _build_env(self, req: JobRequest) -> dict:
        """Override to add env-var overrides for the subprocess.
        Default: empty (subprocess inherits parent env)."""
        return {}

    def _build_cwd(self, req: JobRequest) -> "Path | None":
        """Override to set the subprocess working directory.
        Default: None (subprocess inherits parent cwd)."""
        return None

    def _collect_outputs(self, req: JobRequest) -> dict:
        """Override to declare expected output files.
        Default: no outputs (caller checks return_code only)."""
        return {}

    def _build_argv(self, req: JobRequest) -> list[str]:
        """Override to control the executable + arg list. Default
        is `python -u <script_path>`. Subclasses can prepend
        wrappers (e.g. FEMRunner prepends `mpirun -n N` when
        MPI parallelism is requested) or append extra script-side
        CLI flags."""
        return [sys.executable, "-u", str(self.script_path)]

    def _serialize_payload(self, req: JobRequest) -> dict:
        """Override to produce the JSON-serializable payload that
        gets written to _build_payload_path. Default: dataclass-
        asdict of the request itself, which works when the request
        IS the payload schema. Subclasses can return a precomputed
        dict (e.g., when the request wraps a legacy payload)."""
        if dataclasses.is_dataclass(req):
            return dataclasses.asdict(req)
        return dict(req.__dict__)

    # Body -------------------------------------------------------

    def run(
        self,
        req: JobRequest,
        on_line: Callable[[str], None],
        cancel: CancelToken,
    ) -> JobOutputs:
        # Write the payload JSON if the subclass wants one.
        payload_path = self._build_payload_path(req)
        if payload_path is not None:
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._serialize_payload(req)
            payload_path.write_text(json.dumps(payload, default=str), encoding="utf-8")

        argv = list(self._build_argv(req))
        if payload_path is not None:
            argv.append(str(payload_path))

        env_overrides = self._build_env(req)
        import os
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in env_overrides.items()})
        cwd = self._build_cwd(req)

        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
            cwd=str(cwd) if cwd is not None else None,
        )
        cancel.arm(proc)
        try:
            for line in proc.stdout:
                on_line(line.rstrip())
            rc = proc.wait()
        finally:
            cancel.clear()

        return JobOutputs(
            return_code=int(rc),
            outputs=(self._collect_outputs(req) if rc == 0 else {}),
        )
